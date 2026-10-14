"""Additive correction record; old registrations and scientific results are immutable."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
OUT=EXT/'results/research_20261010_continuation_v2/review_followup_v3'


def pin(p):
    return {'path':p.relative_to(ROOT).as_posix(), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    p=EXT/'results/research_20261010_continuation_v2/pickup'
    d={
        'version':'THREE-LINE-REVIEW-CONTINUATION-v3',
        'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'kind':'additive_scope_correction_and_engineering_continuation',
        'authority':'Latest human request: audit all new work, correct it, then continue toward stable gain; both servers permitted',
        'pins':[pin(p/n) for n in ['P_LIVE_DEV_PANEL_V1_REGISTRATION.json',
                  'P_LIVE_DEV_PANEL_V2_REGISTRATION.json','P_LIVE_DEV_PANEL_V2_RUN_REGISTRATION.json']],
        'parent_program':pin(EXT/'results/research_20261010_continuation_v2/PROGRAM.json'),
        'protected_final_seeds':[20260927,20260928],
        'old_bytes_must_remain_unchanged':True,
        'prospective_pickup_constraints':{
            'hidden_future':'Actor sees only public packet and prespecified historical scenarios; terminal realized J is an evaluation sink, never an acceptance signal',
            'units':'Use make_c0_contract_v2 with explicit km_per_unit20 and speed_kmh30, simulator speed1.5; reject any contract/env mismatch before generation',
            'energy_budget_kwh':710.6548152249378,
            'threshold_p_c':{'0':5,'1':10,'2':15},
            'K':10,'fleet_size':15,'capacity':50,
            'budget_unit':'10 seconds per newly revealed admission request, matching the existing per-request SAA definition. Shared original start time includes sampling, preparation, inference, search, certification and submission; reconstruction gets no additional10s. Multiple-request event is not a single10s event.',
            'full_day_budget_is_not_10seconds':True,
            'all_arms_same_budget_and_explicit_history_pool':True,
            'history_pool':'Pin identical dataset/number of historical days and hist argument for every relevant arm before formal generation; bare hist=None in rolling arm is not acceptable',
            'main_threshold':'Prospective candidate−strong_rolling mean>=20 utility/day with registered paired CI lower>0, as v2; additionally candidate−own_unmodified_base mean>0 and CI lower>0',
            'attribution':'candidate−reconstruction_only mean>0 and CI lower>0, matched initialization, mask schedule, actual calls and computation; random same architecture remains an attribution control',
            'reported_arms':['cond_hist','uncond_hist','own_unmodified_base','discard_control'],
            'scope':'Joint revealed unserved accepted orders can move between legal mutable suffixes; executed/committed path and onboard cargo remain fixed',
            'qualification_not_yet_complete':True,
            'new_online_data_generation_enabled':False,
            'reason':'Complete actor integration, resource/accounting and all-arm semantic regressions required; engineering QA is not the five-arm online panel'
        },
        'seed_integrity_correction':{
            'literal_search_cannot_prove_freshness':'A registration itself contains its reserved seed. A fresh-run check must distinguish reservation from actual consumed run seeds and dataset hashes.',
            'observed_stage_a_seeds':[2026120201,2026120231,2026120232,2026120233],
            'same_old_four_days_allowed_for_diagnostic_only':True,
            'new_research_seed_assignment':'Not yet assigned: requires consumed-run ledger and prespecified splits before new research data; no false fresh claim'
        },
        'delivery_constraints':{
            'old_head_only_training':'Closed negative diagnostic, not full encoder/decoder masked task training; retain all original rows and weights',
            'immediate_min_delta':'Cannot be beaten for same request/feasible positions by definition; use complete candidate cost and future legal recourse diagnostic for mechanism',
            'new_full_model_stage':'Synthetic engineering only: full parameter updates, shared schedule, final save/reload; real checkpoint projection/teacher/data/strong-baseline still required',
            'research_data_generation_enabled':False,
            'registered_teacher_requirement':'Complete routes, manifest, cost decomposition, all hard/time outcomes including failed candidates, source/dataset/contract/weights start/end identities',
            'future_confirmatory_gate':'Only after development selection and frozen candidate: three disjoint40 complete days, >=1% against each registered strong/matched attribution control with multiplicity-adjusted paired CI lower>0; no snapshot claim as online'
        },
        'reproduction_constraints':{
            'negative_183_package':'Immutable; original R1/C1 judgments remain negative',
            'numeric_vs_qualification':'Existence or filename of route file is inventory, not semantic replay or source identity',
            'blocked_claims':['stable_maskco_core_online_gain','pickup_impossibility','equivalence','full_from_zero_reproduction'],
            'required_guard':'Reject heldout seeds, empty named trajectories, unsupported exhaustive claims; explicitly mark finite condition subset'
        },
        'resource_amendment':{
            'GPU_permission':'Human permits3080 and5090; old CPU-only constraint prospectively relaxed for new registered training only',
            'actual_probe':'2026-10-10 19:00: 3080 idle; both5090 GPUs busy92/95%, other-user jobs preserved',
            'training_can_wait':'Use isolated CPU engineering now; recheck GPU before actual training',
            'evaluation_resource':'Must be frozen and identical per paired arm; changing machine or GPU is not silently interchangeable wall-clock budget',
            'long_linux_process':'Registered finite tmux pipeline with exclusive directory, failure/exit and actual process tree'
        },
        'scientific_gain_established':False,
        'old_stage_b_stop_rule_triggered':False,
        'old_stage_b_disposition':'Do not execute defective v2 implementation. This is a new audited qualification barrier, not a retroactive trigger of old numeric stopping rules.',
        'shared_MD_upload_authorized':False,'public_release_authorized':False,
        'next_engineering_work':['Causal joint pickup actor integration and baseline/resource equivalence',
           'Real-size pretrained full masked training projection and teacher audit',
           'Fail-closed support gate corrections and independent raw-identity replay']
    }
    with (OUT/'CONTINUATION_V3_REGISTRATION.json').open('x',encoding='utf-8') as f:
        json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'path':str(OUT),'registration_sha256':hashlib.sha256((OUT/'CONTINUATION_V3_REGISTRATION.json').read_bytes()).hexdigest(),
                      'old_files_modified':0,'formal_online_run_enabled':False}))


if __name__=='__main__':main()
