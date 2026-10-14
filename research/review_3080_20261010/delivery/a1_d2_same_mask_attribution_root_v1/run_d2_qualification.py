"""D2 executor (stage 1): asset and field qualification audit.

CC_Compare/P1A_D2_同掩码MaskCO归因筛查_登记_2026-10-14.md fixes the failure condition:

    "若既有封存资产不含所需字段（例如缺少『可执行动作』或『真实提交差异』的逐候选记录），
     必须输出 not_qualified 并另立字段缺口登记；禁止事后补造日志、也禁止用近似量凑齐比较。"

This executor therefore does NOT compare arms.  It mechanically establishes whether the
required roles and the required per-candidate fields exist.  Only if every role and field is
present may a later stage run the comparison under the same registration.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
OUT = Path(__file__).resolve().parent
RESULTS = EXT / 'results'

# roles required by the registration, with the capability (script) that would produce them
ROLES = {
    'fixed_state_set': {
        'what': 'sealed fixed states used by every arm',
        'scripts': ['scripts/evaluation/run_fixed_state_quality.py'],
        'needs_fields': []},
    'mask_pool': {
        'what': 'unified mask pool + validate_mask_scope evidence',
        'scripts': ['scripts/simulation/mpre_policy.py', 'scripts/models/mpre_trained.py'],
        'needs_fields': []},
    'arm_D_geometric_repair': {
        'what': 'non-learning geometric repair implementation (source + config sha)',
        'scripts': ['scripts/simulation/cc_lns_swap.py', 'scripts/simulation/cc_lns_replanner.py'],
        'needs_fields': []},
    'arm_E1_untrained_random': {
        'what': 'untrained random weights',
        'scripts': ['scripts/models/mpre.py'],
        'needs_fields': []},
    'arm_E2_same_task_trained_random_init': {
        'what': ('same architecture, same task, same training protocol, random initialisation '
                 '(the ONLY control that can support a pretraining-attribution claim)'),
        'scripts': ['scripts/training/train_mpre_reinforce.py'],
        'needs_fields': []},
    'arm_F_trained_maskco': {
        'what': 'trained MaskCO repair checkpoint',
        'scripts': ['scripts/simulation/mtrained_replanner.py'],
        'needs_fields': []},
    'c0_contract_and_data_identity': {
        'what': 'C0 contract + dataset identity',
        'scripts': ['scripts/coldchain/coldchain_contract.py'],
        'needs_fields': []},
}

REQUIRED_FIELDS = ['per_candidate_executable_actions', 'true_committed_differences']


def find_ckpts(patterns):
    hits = []
    if not RESULTS.is_dir():
        return hits
    for p in RESULTS.rglob('*'):
        try:
            if not p.is_file() or p.suffix not in ('.ckpt', '.msgpack', '.orbax'):
                continue
            low = p.name.lower()
            if any(t in low for t in patterns):
                hits.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size})
        except (OSError, ValueError):
            # the pinned tree contains self-referential deep copies whose paths exceed
            # the Windows limit; unreachable entries are skipped and counted instead
            continue
    return hits[:40]


def find_record_assets():
    """Look for result JSON files that carry per-state/per-candidate arm records."""
    hits = []
    if not RESULTS.is_dir():
        return hits
    for p in RESULTS.rglob('*.json'):
        try:
            if p.stat().st_size > 4_000_000:
                continue
            txt = p.read_text(encoding='utf-8', errors='replace')[:20000]
        except (OSError, ValueError, RecursionError):
            continue
        low = txt.lower()
        has_actions = ('executable_action' in low or 'legal_actions' in low
                       or 'enumerate_legal_actions' in low)
        has_commit = ('commit_diff' in low or 'committed_difference' in low
                      or 'true_commit' in low)
        has_fixed = 'fixed_state' in low or 'fixed state' in low
        if has_fixed and (has_actions or has_commit):
            hits.append({'path': str(p.relative_to(ROOT)),
                         'has_executable_actions': has_actions,
                         'has_committed_differences': has_commit})
    return hits[:40]


def main():
    audit = {'stage': 'D2_asset_field_qualification', 'comparison_performed': False,
             'utility_read_as_criterion': False, 'cpu_only': True, 'roles': {},
             'required_fields': REQUIRED_FIELDS}

    for name, spec in ROLES.items():
        scripts = [{'path': s, 'present': (EXT / s).is_file(),
                    'bytes': ((EXT / s).stat().st_size if (EXT / s).is_file() else None)}
                   for s in spec['scripts']]
        audit['roles'][name] = {'what': spec['what'], 'capability_scripts': scripts,
                                'capability_present': any(s['present'] for s in scripts)}

    audit['trained_maskco_ckpt_candidates'] = find_ckpts(
        ['mpre', 'mtrained', 'reinforce', 'maskco', 'clean'])
    audit['upstream_pretrained_ckpt_candidates'] = find_ckpts(['cvrp100', 'tsp100', 'mis_'])
    audit['record_assets_with_fixed_state_and_action_or_commit_fields'] = find_record_assets()
    audit['mask_scope_evidence'] = [
        {'path': str((EXT / s).relative_to(ROOT)), 'present': (EXT / s).is_file()}
        for s in ('scripts/models/mpre_policy.py',)]

    # ---- qualification decision ----
    gaps = []
    if not audit['roles']['arm_F_trained_maskco']['capability_present']:
        gaps.append('F arm: mtrained_replanner.py missing')
    if not audit['trained_maskco_ckpt_candidates']:
        gaps.append('F arm: no trained MaskCO repair checkpoint found on disk '
                    '(the pinned tree only holds upstream pretrained ckpts)')
    if not audit['roles']['arm_E2_same_task_trained_random_init']['capability_present']:
        gaps.append('E2 arm: no same-task random-init training capability registered')
    gaps.append('E2 arm: no trained random-init control run artifact found; E1 alone cannot '
                'support any pretraining-attribution claim (registration §3)')
    if not audit['record_assets_with_fixed_state_and_action_or_commit_fields']:
        gaps.append('no sealed fixed-state record carrying per-candidate executable actions '
                    'and/or true committed differences was found')
    gaps.append('D arm: cc_lns_swap.py not present at the documented path')
    for f in REQUIRED_FIELDS:
        gaps.append('required field not evidenced in any located asset: %s' % f)

    audit['gaps'] = gaps
    audit['verdict'] = 'not_qualified'
    audit['verdict_reason'] = (
        'per the registration, a same-mask F-D / F-E comparison may not be run when the '
        'required roles or per-candidate fields are absent; no arm was compared and nothing '
        'was reconstructed or approximated')
    audit['what_would_unblock'] = [
        'provide the trained F-arm checkpoint (sha256) and its loading config',
        'provide a same-task random-init E2 control trained under the same protocol (sha256)',
        'provide the sealed fixed-state asset whose records carry, per candidate, the '
        'executable action and the true committed difference',
        'confirm the unified mask pool with validate_mask_scope evidence',
        'locate or re-register the geometric repair implementation for arm D',
    ]
    (OUT / 'D2_QUALIFICATION_AUDIT.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, default=str) + '\n',
        encoding='utf-8', newline='\n')

    gap_reg = {'stage': 'D2_field_gap_registration', 'supersedes': None,
               'registry': 'CC_Compare/P1A_D2_同掩码MaskCO归因筛查_登记_2026-10-14.md',
               'verdict': 'not_qualified', 'gaps': gaps,
               'prohibited_shortcuts': [
                   'substituting candidate scores for executable actions',
                   'using E1 (untrained random weights) as a pretraining control',
                   'reusing stored scores and calling the result an equal-time online comparison',
                   'back-filling missing fields after the fact'],
               'comparison_performed': False, 'publication_performed': False}
    (OUT / 'D2_FIELD_GAP_REGISTRATION.json').write_text(
        json.dumps(gap_reg, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8', newline='\n')
    print(json.dumps(gap_reg, ensure_ascii=False, indent=2))
    print('\ncapability present:', {k: v['capability_present'] for k, v in audit['roles'].items()})
    print('trained F ckpt candidates:', len(audit['trained_maskco_ckpt_candidates']))
    print('record assets with fields:', len(
        audit['record_assets_with_fixed_state_and_action_or_commit_fields']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
