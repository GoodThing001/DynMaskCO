"""Freeze delivery engineering QA only, without a numerical research pilot."""
from pathlib import Path
import ast
import hashlib
import json
import platform
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['scripts/delivery/delivery_contract.py', 'scripts/delivery/delivery_evaluator.py',
         'scripts/delivery/delivery_baselines.py', 'scripts/tests/test_delivery_contract.py',
         'scripts/coldchain/coldchain_contract.py', 'scripts/coldchain/coldchain_state.py']


def main():
    out = ROOT / 'results/delivery_engineering_1008_v1'
    out.mkdir(exist_ok=True)
    manifest = {}
    archive = ROOT / 'tools_remote/delivery_engineering_1008_v1.zip'
    if archive.exists():
        raise FileExistsError('Preserve an existing release; use a new release version')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for relative in FILES:
            path = ROOT / relative
            ast.parse(path.read_text(encoding='utf-8'))
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            z.write(path, relative)
        z.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
    (out/'LOCAL_RELEASE.json').write_text(json.dumps({'source':manifest,
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'purpose':'delivery_contract_and_real_baseline_engineering_only',
        'formal_pilot_started':False},indent=2))
    disk = shutil.disk_usage(ROOT)
    resources = {'platform':platform.platform(), 'disk_free_gb':disk.free/2**30,
        'policy':'server CPU-only engineering; no GPU allocation',
        'skill_detector_note':'detect_resources.py failed on Windows available_gb; native fallback'}
    try:
        import psutil
        resources.update(cpu_count=psutil.cpu_count(), memory_available_gb=psutil.virtual_memory().available/2**30)
    except ImportError:
        resources['psutil']='unavailable_locally'
    (out/'local_resources.json').write_text(json.dumps(resources,indent=2))
    print(json.dumps({'archive':str(archive),'files':len(manifest),'local_resources':resources}))


if __name__=='__main__':main()
