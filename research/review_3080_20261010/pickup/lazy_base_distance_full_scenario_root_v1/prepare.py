"""Prospective registration of the pickup full-scenario paired lazy-reuse component."""
import ast
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent / 'independent_rolling_incremental_engineering_root_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if (HERE / 'REGISTRATION.json').exists():
        raise SystemExit('REGISTRATION.json already exists')
    sources = [HERE / n for n in ('lazy_base_distance.py', 'run.py', 'audit.py', 'prepare.py', 'supervise.py')]
    sources += [ROOT / 'tools' / 'engineering_stage_closure_v1.py',
                ROOT / 'tools' / 'qualify_pickup_joint_policy_transaction_root_v3.py',
                ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/scenario_saa.py',
                HERE.parent / 'shadow_energy_correction_root_v1/energy_complete_saa.py',
                BASE / 'SIMULATION_CONTEXT.json',
                HERE.parent / 'shared_history_qualification_root_v1/COMMON_HISTORY.pkl']
    for p in sources:
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'))
        if not p.exists():
            raise SystemExit('missing registered source %s' % p)
    reg = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={str(p): sha(p) for p in sources},
        stage='pickup_full_scenario_paired_lazy_base_reuse',
        goal=('Extend the registered loop-scoped lazy base-distance reuse from the first four orders to the whole '
              'reveal-sorted order list of the fixed archived scenario index 8, and measure the removed repeated '
              'base evaluations against the unmodified original insertion on the same fixed branch.'),
        mechanism='Only the repeated same-loop base route distance of the unchanged original _insert_u is reused',
        scenario_index=8,
        order_selection=('the full original reveal-sorted order list of the fixed archived scenario index 8 under '
                         'common K10 seed202610130901; the list is fixed by the archived scenario and is not chosen '
                         'by cost, label or outcome'),
        arms=dict(REF='unmodified original EnergyCompleteSAA._insert_u',
                  CAND='registered LazyBaseDistanceSAA same-loop base reuse'),
        design=('one untimed warm-up per arm, then interleaved REF,CAND repeated; only the insertion, local-search '
                'and completion section is timed; both arms run on the same fixed sealed inputs'),
        exact_gates=('identical insertion choices, identical feasibility calls and values, identical scored '
                     'candidates, identical C0 physical transitions, identical terminal energy and distance, and the '
                     'candidate distance log must be the reference log with only repeated same-loop base calls '
                     'deleted in order; every distance value recomputed by an independent ordered double sum'),
        timing_policy=('descriptive only; no speed threshold is registered, because a reduced call count is not a '
                       'speedup and the timed section is small relative to process noise'),
        timed_repeats=2,
        K=10, seed=202610130901, incr_eval=False, local_passes=4,
        supervision_cap_seconds=600,
        resource=dict(python='3.12.7', numpy='1.26.4', cpu_affinity=[7], no_model_min_available_ram_gib=0.5),
        full_request_budget_qualified=False, formal_strong_dynamic_qualified=False,
        scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
        stable_online_gain=False,
        papers=[
            dict(title='PyVRP: a high-performance VRP solver package', doi='10.1287/ijoc.2023.0055',
                 url='https://arxiv.org/html/2403.13795v2',
                 read_scope='sections 4.3, 4.3.2 and 5.3 read in the previous cycle',
                 use='route-evaluation reuse inspiration only; no PyVRP code is installed or executed'),
        ],
    )
    with (HERE / 'REGISTRATION.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(reg, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(registration_sha256=sha(HERE / 'REGISTRATION.json'))))


if __name__ == '__main__':
    main()
