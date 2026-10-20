"""Public-data command contracts; no synthetic customers or reference lookup in solve."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'static_delivery'))
import run_rrp


def test_selected_case_rejects_paths_and_unknown_instances():
    assert run_rrp.selected_case('D1-n4-np33').nodes == 4
    with pytest.raises(ValueError, match='public'):
        run_rrp.selected_case('../private')


def test_teacher_profile_mismatch_is_retained_as_a_rejection():
    cases, rejected = run_rrp.teachers(['D1-n4-np33'], 'with_wage', 'paper')
    assert len(cases) == 1 and not rejected


def test_solve_cli_needs_no_teacher_and_records_full_common_objective(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Reference labels must not be read in inference')
    monkeypatch.setattr(run_rrp, 'load_reference', forbidden)
    result = run_rrp.solve_case(case='D1-n4-np33', out=tmp_path / 'solve',
                               methods=['local_search'], iterations=1, seconds=2,
                               objective='with_wage', profile='paper', seed=888,
                               checkpoint=None, model_profile='smoke_cpu', keeping_rate=.2)
    assert result['methods'][0]['actual_evaluations'] >= 1
    assert result['methods'][0]['report']['terminal_load_kg'] == 0
    assert (tmp_path / 'solve/result.json').is_file()


def test_fleet_cli_uses_same_public_orders_without_loading_labels(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No label lookup in fleet inference')
    monkeypatch.setattr(run_rrp, 'load_reference', forbidden)
    result = run_rrp.solve_case(case='D1-n4-np33', out=tmp_path / 'fleet',
                               methods=['local_search'], iterations=1, seconds=2,
                               objective='with_wage', profile=run_rrp.PRIMARY_PROFILE, seed=888,
                               checkpoint=None, model_profile='smoke_cpu', keeping_rate=.2,
                               max_vehicles=2)
    assert result['fleet_extension']['max_vehicles'] == 2
    assert result['methods'][0]['report']['feasible']
    assert sum(len(r) - 1 for r in result['methods'][0]['routes']) == 3


def test_archive_audit_uses_saved_clock_diagnostic_waits_only(tmp_path, monkeypatch):
    path = run_rrp.SOURCE / 'Datasets/D2/D2-n11-np30.json'
    monkeypatch.setattr(run_rrp, 'public_instances', lambda: [path])
    run_rrp.audit(tmp_path / 'audit')
    row = json.loads((tmp_path / 'audit/reference_audit.json').read_text(encoding='utf-8'))['rows'][0]
    primary = row['profiles'][run_rrp.PRIMARY_PROFILE]['evaluation']
    archive = row['profiles']['archive_compatibility']['evaluation']
    assert primary['waits_s'] == [0, 0, 327, 837, 0, 1851, 0, 0, 0, 45, 0]
    assert archive['waits_s'] == [0, 327, 837, 0, 1851, 0, 0, 0, 45, 0, 0]
    assert max(abs(v) for v in row['profiles']['archive_compatibility']['arrival_residual_s']) < 1


def test_initial_model_only_records_actual_config_and_digest(tmp_path):
    result = run_rrp.solve_case(case='D1-n4-np33', out=tmp_path / 'initial',
                               methods=['maskco_initial'], iterations=0, seconds=1,
                               objective='with_wage', profile=run_rrp.PRIMARY_PROFILE, seed=888,
                               checkpoint=None, model_profile='smoke_cpu', keeping_rate=.2)
    identity = result['models']['maskco_initial']
    assert identity['config']['profile'] == 'smoke_cpu'
    assert len(identity['parameter_sha256']) == 64
    assert result['methods'][0]['model_identity'] == identity
