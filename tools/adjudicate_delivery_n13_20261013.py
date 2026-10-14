"""Adjudicate delivery n13: preserved audit FAIL, measured mechanism, derived criterion."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / 'research/review_3080_20261010/delivery/action_matching_generalization_root_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


R = json.loads((H / 'RESULT.json').read_text(encoding='utf-8'))
A = json.loads((H / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
RUN = json.loads((H / 'RUN_EXIT.json').read_text(encoding='utf-8'))
AEXIT = json.loads((H / 'AUDIT_RUN_EXIT.json').read_text(encoding='utf-8'))
PA = R['primary_arm']
before = R['arms'][PA]['before']

adj = {
    'stage': 'delivery_n13_action_matching_generalization',
    'registration_sha256': sha(H / 'REGISTRATION.json'),
    'producer': dict(checks=R['checks_count'], failed=R['failed'], exit_code=RUN['exit_code']),
    'audit': dict(checks=A['checks_count'], failed=A['failed'], exit_code=AEXIT['exit_code'],
                  failure_names=A['failure_names']),
    'preserved_failures': {
        'sgd_trajectory_reproduced:sgd_matched': {
            'param_max_abs_diff': A['audit_arms']['sgd_matched']['param_max_abs_diff'],
            'trajectory_max_abs_diff': A['audit_arms']['sgd_matched']['trajectory_max_abs_diff'],
            'registered_bound': R['arms'] and 1e-4},
        'sgd_trajectory_reproduced:sgd_conservative': {
            'param_max_abs_diff': A['audit_arms']['sgd_conservative']['param_max_abs_diff'],
            'trajectory_max_abs_diff': A['audit_arms']['sgd_conservative']['trajectory_max_abs_diff'],
            'registered_bound': 1e-4},
        'note': ('Both SGD arms fail the pre-registered 1e-4 trajectory/parameter bound. The registered claim that '
                 'plain SGD is "reproducible by construction" was too strong and is NOT rewritten. The Adam control '
                 'diverges far more (0.0632 parameter) and was pre-registered as reported-not-asserted.'),
    },
    'what_did_reproduce': {
        'functional_outcome_after_ce_max_abs_diff': {k: v['after_ce_max_abs_diff'] for k, v in A['audit_arms'].items()},
        'functional_outcome_after_argmax_matches': {k: v['after_argmax_matches'] for k, v in A['audit_arms'].items()},
        'before_metrics_match': True,
        'note': ('Every arm\'s trained functional outcome (train and held-out mean CE and argmax) reproduces to '
                 '2-5e-8 even though the parameter vectors differ, so the delivered head and its measured behaviour '
                 'are reproducible; only the parameter path is not.'),
    },
    'mechanism_confirmed': {
        'gradient_float32_vs_float64_max_abs': A['mechanism']['gradient_float32_vs_float64_max_abs'],
        'gradient_sign_agreement_on_nonzero_union': A['mechanism']['gradient_sign_agreement_on_nonzero_union'],
        'adam_t1_step_max_abs_diff_f32_vs_f64': A['mechanism']['adam_t1_step_max_abs_diff_f32_vs_f64'],
        'sgd_step_max_abs_diff_f32_vs_f64': A['mechanism']['sgd_step_max_abs_diff_f32_vs_f64'],
        'train_step0_zero_gradient_fraction': R['mechanism']['train_step0_zero_gradient_fraction'],
        'adam_t1_saturated_fraction': R['mechanism']['adam_t1_saturated_fraction'],
        'explanation': ('At the float32 rounding floor the gradient sign agrees between float32 and float64 only '
                        '%.4f of the time, i.e. at chance. A normalised (Adam) update is a function of that sign, so '
                        'a sign flip moves a coordinate by a full +/-lr (measured %.3g across arithmetics). An SGD '
                        'update is proportional to |g|, so the same flip moves it by lr*|g| (measured %.3g), but the '
                        'divergence still accumulates in proportion to lr, the number of steps and the gradient '
                        'resolution, and an unstable arm (lr=0.086, oscillating train CE) amplifies it to %.3g.'
                        % (A['mechanism']['gradient_sign_agreement_on_nonzero_union'],
                           A['mechanism']['adam_t1_step_max_abs_diff_f32_vs_f64'],
                           A['mechanism']['sgd_step_max_abs_diff_f32_vs_f64'],
                           A['audit_arms']['sgd_matched']['param_max_abs_diff'])),
    },
    'generalisation_finding': {
        'split': dict(train=R['split']['train_records'], held_out=R['split']['held_out_records']),
        'before': dict(train=before['train'], held_out=before['held_out']),
        'after': {k: dict(train=v['after']['train'], held_out=v['after']['held_out']) for k, v in R['arms'].items()},
        'reading': ('Training CE falls substantially for the two arms that actually optimise (sgd_conservative '
                    '1.2509->0.2172, adam 1.2509->0.2344) while held-out CE RISES (1.3735->1.8231 and 1.3735->1.5245). '
                    'Held-out argmax rises only from 2/9 to 4-5/9 on nine states, which is not evidence of transfer. '
                    'The head-only action-matching update over frozen latents therefore memorises the 20 training '
                    'states rather than learning a transferable scoring function, and its held-out calibration gets '
                    'worse. The primary sgd_matched arm lowers both CEs but its train trajectory oscillates '
                    '(0.989->1.135) and never converges, so its held-out change is not attributable to successful '
                    'optimisation.'),
        'full_candidate_metric_not_usable': ('full_candidate_argmax_correct is 0 everywhere because the grid holds 25 '
                                             'customers x 30 positions and the global argmax need not fall on the '
                                             'designated customer; it must not be read as an action-match statistic.'),
        'consequence': ('The route "train a head on already sealed latents of the pretrained_initial arm" is closed as '
                        'a path to online gain. Any delivery learning claim needs the backbone and real training data '
                        '(scientific family 3 after all registered barriers), not more head-only work on these 30 '
                        'sealed states.'),
    },
    'derived_criterion_for_future_stages': ('Reproducibility of an optimizer trajectory must be stated relative to '
                                            'the float32 arithmetic floor (lr, step count and gradient resolution), '
                                            'not as bit-equality; functional-outcome equality is the criterion this '
                                            'stage actually supports.'),
    'no_rerun_to_chase_pass': True,
    'v2_created': False,
    'flags': dict(formal_trained_control_qualified=False, formal_controls_qualified=False,
                  full_request_budget_qualified=False, scientific_data_generated=False,
                  new_science_family=False, scientific_family3_started=False, learned_output_qualified=False,
                  stable_online_gain=False),
    'saved_sha256': {n: sha(H / n) for n in ('RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json',
                                            'AUDIT_RUN_EXIT.json', 'HEAD_AFTER_sgd_matched.npz',
                                            'HEAD_AFTER_sgd_conservative.npz', 'HEAD_AFTER_adam_control.npz')},
}
out = H / 'STAGE_ADJUDICATION.json'
assert not out.exists()
out.write_text(json.dumps(adj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'written': str(out.relative_to(ROOT)), 'audit_failed': A['failed'],
                  'sign_agreement': A['mechanism']['gradient_sign_agreement_on_nonzero_union']}))
