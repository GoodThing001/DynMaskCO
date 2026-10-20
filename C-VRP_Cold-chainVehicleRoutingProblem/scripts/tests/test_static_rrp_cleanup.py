"""Administrative guard tests; protected files never become cleanup targets."""
import importlib.util
from pathlib import Path
import pytest
import hashlib
import json
import zipfile

EXT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cleanup', EXT / 'tools/static_rrp_cleanup.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


def test_guard_rejects_outside_workspace():
    with pytest.raises(ValueError, match='outside'):
        cleanup.guarded(EXT.parent.parent / 'unrelated')


def test_legacy_roots_keep_static_upstream_and_user_results():
    roots = cleanup.legacy_roots()
    for root in roots:
        assert not root.is_relative_to(EXT.parent / 'MASKCO_code')
        assert not root.is_relative_to(EXT.parent / 'CC_Compare')
        assert not root.is_relative_to(EXT.parent / 'results')
        assert not root.is_relative_to(EXT / 'scripts/static_delivery')
        assert not root.is_relative_to(EXT / 'data/public_sources_20261020')


def test_private_and_original_mapping_are_protected_without_reading():
    assert cleanup.excluded(EXT / 'research/private/unknown')
    assert cleanup.excluded(EXT / 'research/a/synthetic_mapping.bin')
    assert cleanup.excluded(EXT.parent / '.vscode/sftp.json')


def fixture_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, 'ROOT', tmp_path)
    monkeypatch.setattr(cleanup, 'EXT', tmp_path / 'extension')
    monkeypatch.setattr(cleanup, 'OUT', tmp_path)
    monkeypatch.setattr(cleanup, 'MANIFEST', tmp_path / 'manifest.json')
    monkeypatch.setattr(cleanup, 'ARCHIVE', tmp_path / 'archive.zip')
    (tmp_path / 'research').mkdir()
    files = [tmp_path / 'research/a', tmp_path / 'research/b']
    for file in files:
        file.write_bytes(b'original-' + file.name.encode())
    rows = []
    with zipfile.ZipFile(cleanup.ARCHIVE, 'w') as bundle:
        for file in files:
            sha = hashlib.sha256(file.read_bytes()).hexdigest()
            bundle.write(file, 'objects/' + sha)
            rows.append({'path': file.relative_to(tmp_path).as_posix(), 'sha256': sha, 'bytes': file.stat().st_size})
    data = {'files': rows, 'applied': False, 'directories': [], 'protected': [],
            'archive_sha256': hashlib.sha256(cleanup.ARCHIVE.read_bytes()).hexdigest()}
    cleanup.MANIFEST.write_text(json.dumps(data))
    return files


def test_source_changed_after_precheck_must_not_be_deleted(tmp_path, monkeypatch):
    files = fixture_plan(tmp_path, monkeypatch)
    original = cleanup.verify_objects
    def mutate(objects):
        original(objects)
        files[0].write_bytes(b'new unarchived user edit')
    monkeypatch.setattr(cleanup, 'verify_objects', mutate)
    with pytest.raises(ValueError, match='SHA'):
        cleanup.apply()
    assert files[0].read_bytes() == b'new unarchived user edit'


def test_locked_second_file_records_progress_and_can_retry(tmp_path, monkeypatch):
    files = fixture_plan(tmp_path, monkeypatch)
    with files[1].open('rb'):
        with pytest.raises(PermissionError):
            cleanup.apply()
    assert not files[0].exists()
    cleanup.apply()
    assert not files[1].exists()
    assert json.loads(cleanup.MANIFEST.read_text())['applied']


def test_apply_revalidates_preserved_ranges(tmp_path, monkeypatch):
    files = fixture_plan(tmp_path, monkeypatch)
    forbidden = tmp_path / 'MASKCO_code'
    forbidden.mkdir()
    target = forbidden / 'source.py'
    target.write_bytes(files[0].read_bytes())
    data = json.loads(cleanup.MANIFEST.read_text())
    data['files'][0]['path'] = 'MASKCO_code/source.py'
    cleanup.MANIFEST.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='preserved|allow'):
        cleanup.apply()
    assert target.is_file()
