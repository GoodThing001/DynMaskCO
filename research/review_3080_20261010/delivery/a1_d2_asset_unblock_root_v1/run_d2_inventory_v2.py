"""D2 unblock, stage 2: asset inventory v2 (READ-ONLY).  No F-D / F-E comparison, no training.

Fixes the three search biases the user identified in the v1 auditor
(`run_d2_qualification.py`):
  1. checkpoints are located by an authorised-directory walk, NOT by a filename keyword
     filter (the historical trainer writes the generic name `model.ckpt`);
  2. mask-pool evidence is located at the REAL module
     `scripts/simulation/dynmaskco_cc_context.py` (v1 pointed at `scripts/models/mpre_policy.py`,
     a path that does not exist);
  3. sealed assets are checked against their ACTUAL schema (probed from the sealed files
     themselves), not by substring counts over a truncated read.

Scope honesty: the server walk is bounded and the exact directories searched are recorded.
"not found in the searched scope" is never reported as "absent from the server".
"""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
OUT = Path(__file__).resolve().parent

F_DOCUMENTED_PATHS = [
    'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt',
]
F_DOCUMENTED_SHA = 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5'

# ---- server facts, taken verbatim from the preserved raw probe outputs ----------
SERVER_ENV = {'host': 'user', 'date': '2026-10-14T15:47:54+08:00',
              'python': 'Python 3.10.19',
              'cwd': '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'}
SERVER_SCOPE = {
    'checkpoint_walk': "find results -maxdepth 6, pruned at '*/_bverify_pinned'",
    'dir_walk': "find results -maxdepth 4 for *mpre*/*mtrained*/*reinforce*/*clean* dirs",
    'file_walk': "find results -maxdepth 6 -type f -name '*.json' -size -512k",
    'not_covered': ['trees outside the extension root', 'B_REPRO_CLEAN', 'any path the '
                    'login user cannot read', 'archives not mounted on the host'],
}

SERVER_F = {
    'path': 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt',
    'exists': True, 'bytes': 89592616, 'mtime': 'Sep 22 11:11',
    'sha256': 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5',
    'matches_documented_sha': True,
    'identity_record': 'results/m0_scale/mpre_reinforce_s42_clean/training_identity.json',
    'also_named_by': ('results/correction_validation_1009_v2/SOURCE_START.json and '
                      'SOURCE_END.json, under the B_REPRO_CLEAN mirror path, same sha256'),
}
SERVER_SIBLINGS = {
    'mpre_reinforce_s43_clean/model.ckpt': 'e536407535d5cd99215055f1fcc826185fdb55588b16c6615418df445ae9113b',
    'mpre_reinforce_s44_clean/model.ckpt': 'f0b276e1dc7e80d6c0957c286439f919b1182523dfa21179071ce5b945412066',
    'mpre_reinforce_s42_defective/model.ckpt': '670159037f54b7ab8f049a3c93f0d82a6fa2156c6a66c996db26ff705b9a3a8f',
    'mpre_reinforce_s42_corrected/model.ckpt': '60f6360e8b4256e9ca9eea78435afc00537a92f12755041aaf42f41c7145ab17',
    'repair_s43/model.ckpt': '(present; sha not requested in this probe)',
}
SERVER_SRC = {
    'scripts/simulation/cc_lns_replanner.py': 'ae634f5323595aafda85aaf8e99eb4e240173106d60b2523bb033ae603e22f25',
    'scripts/simulation/cc_lns_swap.py': None,
    'scripts/evaluation/run_cc_lns_swap_fixed_state.py': None,
    'scripts/simulation/dynmaskco_cc_context.py': '4eebb2a51a024a9e881ce7100bbe568219e1afa5ac8c6d33c707924a36ff1685',
    'scripts/evaluation/run_fixed_state_quality.py': 'fd38cb3c2b1084b899e4ec78e2d2f93592e3287ed1f1847ed7657a65300b6544',
}
# F's frozen training identity (from training_identity.json) vs the CURRENT server source
F_TRAINING_SRC = {
    'scripts/training/train_mpre_reinforce.py': '33c617a3e24bfc4c5d68c2ca4e72f41561999906dcea4795493cc8a2aad1b5fd',
    'scripts/simulation/dynmaskco_cc_context.py': '9de8ceb54d088b8afa087948d6469f8cbd6f503678715bbf39e9971f5a91be54',
    'scripts/simulation/cc_lns_replanner.py': '7ea14c07e13943ea41bf9123f8c4f47d40546970b09d0e70777f8d4f8d9437c3',
    'scripts/simulation/mtrained_replanner.py': '3b414e4a9f64953ec759ee2d05ea529abbb8710bc94bbb909bd267e071830b48',
    'scripts/models/mpre.py': '585f0296e31fc98060f393cad023a6c028c70bbbdb8bcc0c6d27f46f03b3e57c',
    'scripts/models/mpre_trained.py': 'e88c7d0d0d21c73b6335880a20111ee4d02b0a6693071872c74d217d34beb300',
    'scripts/coldchain/coldchain_contract.py': '901d3957bc77543e55c2483328b1ab750e4e46d727b2b531ac09eb90bcf2c412',
}
MASK_CALL_SITES = [
    'scripts/evaluation/run_sgbs_fixed_state.py:32',
    'scripts/evaluation/run_fixed_state_quality.py:44-45,101,107',
    'scripts/evaluation/run_headroom_census.py:50,100,106',
    'scripts/simulation/mtrained_replanner.py:17,206,235',      # F arm
    'scripts/simulation/cc_lns_replanner.py:26-27,114,151',     # D arm
    'scripts/training/train_mpre_reinforce.py:38,159,165',      # F training
]
MASK_DEFINITION = {'file': 'scripts/simulation/dynmaskco_cc_context.py',
                   'mask_candidate_pool': 'line 124', 'validate_mask_scope': 'line 155'}
# execution-pass evidence, read from the sealed per_state.json's own schema
MASK_EXEC_EVIDENCE = {
    'asset': 'results/correction_validation_1009_v3/fixed_state/per_state.json',
    'bytes': 6973, 'n_records': 2,
    'fields': ['mask_audit.per_customer[o].in_decision_pool', '.is_committed_next',
               '.in_non_mutable_tail', '.in_protected_tail', '.in_mutable_vehicle',
               'mask_audit.n_out_of_pool', '.decision_pool_size', '.mask_size'],
    'summary_counterpart': 'results/correction_validation_1009_v3/fixed_state/summary.json',
    'recorded': {'n_states_with_out_of_pool': 0, 'n_out_of_pool_customers': 0,
                 'reason_counts': {}},
}
FIELD_SCHEMA = {
    'asset': 'results/correction_validation_1009_v3/fixed_state/per_state.json',
    'per_arm_record_keys': ['J', 'D', 'Q', 'E', 'fail', 'fail_reason', 'time', 'hash',
                            'cert_ok', 'cert{dup,missing,extra,order_changed,'
                            'protected_vehicle_changed,vehicle_changed}', 'diag'],
    'hash_is': 'plan hash only (e.g. 5c28be5a88095753) -- not an action identity',
}
SCALAR_ONLY_ASSET = {
    'asset': 'results/m0_scale/sgbs_fixed_cal/per_state.json',
    'bytes': 10795, 'n_records': 32,
    'keys': ['inst', 'event', 'J0', 'R', 'Mpre_greedy', 'Mtrained_greedy', 'Mpre_sgbs',
             'Mtrained_sgbs', 'dist_sgbs', 'Mtrained_greedy7'],
    'note': 'scalar J per arm per state only: no plan, no mask, no action, no commit',
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def local_presence(rel):
    p = EXT / rel
    return {'path': rel, 'present': p.is_file(),
            'bytes': (p.stat().st_size if p.is_file() else None)}


def main():
    env = {'local': {'platform': platform.platform(), 'python': platform.python_version(),
                     'cwd': str(ROOT)},
           'server': SERVER_ENV,
           'searched_scope': {'local': ['C-VRP_Cold-chainVehicleRoutingProblem/results',
                                        'CC_Compare', 'tools',
                                        'research/review_3080_20261010'],
                              'server': SERVER_SCOPE},
           'raw_evidence': {}}
    for name in ('d2_server_inventory.txt', 'd2_schema_probe.txt', 'd2_schema_probe.sh'):
        p = OUT / 'raw' / name
        env['raw_evidence'][name] = {'present': p.is_file(),
                                     'sha256': sha(p) if p.is_file() else None}

    # ---------------- [1] F ----------------
    f = {'role': 'F: trained MaskCO repair',
         'documented_path': F_DOCUMENTED_PATHS[0], 'documented_sha256': F_DOCUMENTED_SHA,
         'server': SERVER_F, 'local': local_presence(F_DOCUMENTED_PATHS[0]),
         'sibling_runs': SERVER_SIBLINGS,
         'verdict': 'ready',
         'verdict_basis': ('the checkpoint exists at the documented path with the documented '
                           'sha256, and is independently named by the frozen '
                           'SOURCE_START/END identity pair'),
         'caveats': [
             'the frozen identity pair names it under a B_REPRO_CLEAN mirror path (same '
             'sha256); the extension-tree copy is the one verified here',
             'the sibling directory mpre_reinforce_s42_defective must never be used as F',
             'F may only be registered if its load config, training protocol and architecture '
             'all match the task contract in force (registration S1)']}

    # ---------------- [2] D ----------------
    d = {'role': 'D: non-learning geometric repair',
         'server_source': SERVER_SRC,
         'geometric_baseline_present': {
             'scripts/simulation/cc_lns_replanner.py': True},
         'swap_variant_present': {'scripts/simulation/cc_lns_swap.py': False,
                                  'scripts/evaluation/run_cc_lns_swap_fixed_state.py': False},
         'verdict': 'incompatible',
         'verdict_basis': ('the geometric baseline EXISTS (regret-2 reconstruction in '
                           'cc_lns_replanner.py), so the v1 statement "D arm missing" was '
                           'wrong; but two blockers remain: (a) the registration does not fix '
                           'which D semantics it means, and the documented swap variant does '
                           'not exist in either tree; (b) cc_lns_replanner.py has DRIFTED from '
                           'the version F was trained against'),
         'source_drift_vs_F_training_identity': [
             {'file': 'scripts/simulation/cc_lns_replanner.py',
              'at_F_training': F_TRAINING_SRC['scripts/simulation/cc_lns_replanner.py'],
              'current_server': SERVER_SRC['scripts/simulation/cc_lns_replanner.py'],
              'match': False},
             {'file': 'scripts/simulation/dynmaskco_cc_context.py',
              'at_F_training': F_TRAINING_SRC['scripts/simulation/dynmaskco_cc_context.py'],
              'current_server': SERVER_SRC['scripts/simulation/dynmaskco_cc_context.py'],
              'match': False,
              'why_it_matters': 'this module DEFINES the unified mask pool that every arm must '
                                'share; it changed after F was trained'},
             {'file': 'scripts/models/mpre.py',
              'at_F_training': F_TRAINING_SRC['scripts/models/mpre.py'],
              'current_server': '585f0296e31fc98060f393cad023a6c028c70bbbdb8bcc0c6d27f46f03b3e57c',
              'match': True},
         ],
         'required_decision': ('declare which D semantics the registration means, and either '
                               'recover/freeze the training-era sources or demonstrate that '
                               'the current mask-pool and D implementations are '
                               'behaviour-equivalent on the fixed states')}

    # ---------------- [3] mask pool ----------------
    m = {'role': 'unified mask pool + scope evidence',
         'definition': MASK_DEFINITION,
         'call_sites': MASK_CALL_SITES,
         'v1_error_corrected': ('v1 pointed at scripts/models/mpre_policy.py, which does not '
                                'exist; the real definition module is '
                                'scripts/simulation/dynmaskco_cc_context.py'),
         'execution_pass_evidence': MASK_EXEC_EVIDENCE,
         'verdict': 'ready',
         'verdict_basis': ('source definition + call paths exist in the F arm, the D arm, the '
                           'trainer and the fixed-state evaluators; and a sealed asset records '
                           'the per-customer mask-scope audit with 0 out-of-pool customers'),
         'scope_limit': ('the execution-pass evidence covers THAT sealed batch only; it does '
                         'not license the claim that every historical batch passed')}

    # ---------------- [4] E2 ----------------
    e2 = {'role': 'E2: same-task, same-protocol, random-initialisation control',
          'searched': ['results/m0_scale/*mpre*/*mtrained*/*reinforce*/*random*',
                       'results/**/manifest.json', 'server training dirs (depth<=4)'],
          'artifacts_found': [],
          'random_init_control_exists': False,
          'why_seed_change_is_not_E2': (
              'train_mpre_reinforce.py loads the CVRP pretrained checkpoint, so changing --seed '
              're-initialises only the added trainable layers while still sharing the same '
              'pretrained backbone; only a control that randomises the backbone (with the same '
              'architecture, task data, optimiser, step count and model-selection rule) counts '
              'as E2'),
          'source_evidence': {
              'file': 'scripts/training/train_mpre_reinforce.py',
              'verified': [
                  'line 43-44: imports load_cvrp_model and load_mpre_trained/partition',
                  'line 213: backbone, cfg, _step = load_cvrp_model(args.cvrp_ckpt)  <- the '
                  'pretrained backbone is loaded from the checkpoint',
                  'line 214: load_mpre_trained(args.cvrp_ckpt, F_NODE, F_ACTION_EXPLICIT, '
                  'seed=args.seed)  <- --seed only reaches the model builder',
                  'line 259: opt_state = tx.init(train_state)  <- only train_state is optimised, '
                  'so the frozen backbone is not trained',
              ],
              'conclusion': ('--seed changes the initialisation of the added trainable layers '
                             'only; the pretrained backbone is shared with F, so a seeded rerun '
                             'is NOT a random-initialisation control'),
          },
          'verdict': 'missing',
          'verdict_basis': 'no E2 training artifact or manifest was found in the searched scope',
          'consequence': ('E2 cannot be unblocked by asset inventory; it requires a separately '
                          'authorised training pre-registration, and until then no '
                          'pretraining-attribution claim is admissible'),
          'not_started': True}

    # ---------------- [5] field coverage ----------------
    fields = {'role': 'per-candidate fields required by the registration',
              'required': ['per_candidate_executable_actions', 'true_committed_differences'],
              'resolved_by_schema': {
                  'richest_fixed_state_asset': FIELD_SCHEMA,
                  'scalar_only_asset': SCALAR_ONLY_ASSET,
                  'source_level': {
                      'file': 'scripts/simulation/mtrained_replanner.py',
                      'finding': 'no installed / write_back / committed / P_after persistence '
                                 'exists in the F-arm online path (source-level check, not a '
                                 'substring probe of outputs)'}},
              'coverage': {
                  'per_candidate_executable_actions': {
                      'present': False,
                      'what_exists_instead': 'a plan hash per candidate (e.g. 5c28be5a88095753) '
                                             'plus the C0 certificate reconciliation lists',
                      'why_not_substitutable': 'a plan hash identifies a plan, not the '
                                               'executable action identity the registration '
                                               'requires'},
                  'true_committed_differences': {
                      'present': False,
                      'what_exists_instead': 'pre-commit candidate evaluation (which candidate '
                                             'scored best), not what the online caller '
                                             'actually installed',
                      'why_not_substitutable': ('model scores and final routes are candidates; '
                                                'only a recorded post-install plan or '
                                                'accept/reject decision is commit evidence')}},
              'verdict': 'not_verifiable',
              'consequence': ('the two fields are non-traceable from the existing assets; they '
                              'must not be reconstructed or approximated, and a prospective '
                              'collection would be a separate pre-registered study')}

    unblock = {'items': {'F': f['verdict'], 'D': d['verdict'], 'mask_pool': m['verdict'],
                         'E2': e2['verdict'], 'fields': fields['verdict']},
               'blocking_items': [k for k, v in
                                  {'F': f['verdict'], 'D': d['verdict'],
                                   'mask_pool': m['verdict'], 'E2': e2['verdict'],
                                   'fields': fields['verdict']}.items()
                                  if v != 'ready'],
               'registration_preconditions_satisfied': False,
               'overall': 'not_qualified',
               'overall_basis': ('three of five items are ready (F, D-with-semantic-gap, mask '
                                 'pool); E2 has no asset and the two required per-candidate '
                                 'fields are absent from every located asset. The original '
                                 'not_qualified verdict therefore STANDS, and no F-D / F-E '
                                 'comparison was performed'),
               'comparison_performed': False, 'training_started': False,
               'original_audit_untouched': True,
               'new_blocker_found': ('F was trained against a different '
                                     'dynmaskco_cc_context.py (the module that defines the '
                                     'shared mask pool) and a different cc_lns_replanner.py; the '
                                     'same-mask prerequisite is therefore not satisfied against '
                                     'the current sources')}
    for name, payload in (('D2_ASSET_INVENTORY.json', {'env': env, 'F': f, 'D': d}),
                          ('D2_FIELD_COVERAGE.json', fields),
                          ('D2_MASK_SCOPE_AUDIT.json', m),
                          ('D2_UNBLOCK_VERDICT.json', unblock)):
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str)
                                + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(unblock, ensure_ascii=False, indent=2))
    print('\nE2:', e2['verdict'], '| fields:', fields['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
