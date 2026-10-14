"""Explicit evidence prerequisites. Engineering checks cannot grant research eligibility."""


REQUIRED = {
    'pickup_new_method': ('new_protocol_registered','source_identity_complete',
        'model_lineage_complete','information_isolation_pass','incremental_trace_replay_pass',
        'matched_controls_implemented','budget_contract_pass','finite_data_complete'),
    'delivery_dynamic': ('new_protocol_registered','source_identity_complete',
        'event_contract_pass','quality_energy_trace_pass','physical_parameter_scope_declared',
        'strong_rolling_controls_implemented','model_activation_pass','matched_iteration_controls',
        'finite_data_complete','budget_contract_pass'),
    'negative_full_reproduction': ('original_training_source_hash_match',
        'all_historical_dependencies_identified','original_weights_identified',
        'original_event_logs_present','environment_identified','isolated_replay_pass'),
}


def assess(direction, evidence):
    if direction not in REQUIRED:
        raise ValueError('Unknown qualification scope')
    missing=[key for key in REQUIRED[direction] if evidence.get(key) is not True]
    return {'scope':direction,'technical_prerequisites_met':not missing,
            'missing':missing,'scientific_execution_authorized':False,
            'online_gain_established':False}
