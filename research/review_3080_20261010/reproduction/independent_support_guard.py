"""Independent partial qualification guard; never overwrites the actual support auditor.

Manifest and JSON field bindings establish provenance only. This tool has no raw
execution replay engine and therefore cannot issue scientific qualification.
"""
import hashlib
import json
import math
import re
from pathlib import Path


class GuardError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise GuardError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise GuardError('nonfinite JSON token: ' + value)
    out = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                     parse_constant=invalid)
    def finite_tree(value):
        if type(value) is float:
            require(math.isfinite(value), 'nonfinite JSON number')
        elif type(value) is dict:
            for item in value.values(): finite_tree(item)
        elif type(value) is list:
            for item in value: finite_tree(item)
    finite_tree(out)
    return out


def safe(root, relative):
    require(type(relative) is str and relative, 'relative path required')
    root = Path(root).resolve()
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'unsafe relative path')
    p = (root / p).resolve()
    require(p.is_relative_to(root), 'path escaped root')
    return p


def bind_fields(root, references):
    require(type(references) is list and references, 'evidence references required')
    checked = []
    for ref in references:
        require(type(ref) is dict, 'reference must be object')
        p = safe(root, ref.get('path'))
        require(sha(p) == ref.get('sha256'), 'actual evidence SHA mismatch')
        current = load(p)
        field = ref.get('field')
        require(type(field) is list and field and all(type(x) is str for x in field),
                'field must be nonempty explicit JSON key list')
        for key in field:
            require(type(current) is dict and key in current, 'evidence field missing')
            current = current[key]
        require('expected' in ref and type(current) is type(ref['expected']) and
                current == ref['expected'], 'evidence value/type mismatch')
        checked.append({'path': ref['path'], 'sha256': ref['sha256'], 'field': field})
    return {'bindings_verified': checked, 'reference_binding_pass': True,
            'scientific_qualification_established': False}


def design_subset(protocol, actual_program_sha):
    require(type(protocol) is dict, 'protocol must be object')
    reasons = []
    if protocol.get('program_sha256') != actual_program_sha:
        reasons.append('actual_PROGRAM_binding_missing')
    if protocol.get('unit') != 'complete_day' or protocol.get('kind') != 'dynamic_online_confirmation':
        reasons.append('not_registered_dynamic_complete_days')
    cohorts = protocol.get('confirmation_cohorts')
    if type(cohorts) is not list or len(cohorts) != 3:
        reasons.append('not_three_cohorts'); cohorts = []
    seeds = set()
    for cohort in cohorts:
        if type(cohort) is not dict:
            reasons.append('malformed_cohort'); continue
        seed = cohort.get('data_seed')
        if seed in (20260927, 20260928):
            reasons.append('protected_pickup_final_seed_forbidden')
        if type(seed) is not int or seed in seeds:
            reasons.append('missing_or_reused_data_seed')
        else:
            seeds.add(seed)
        days = cohort.get('day_ids')
        if (type(days) is not list or len(days) != 40 or any(type(day) is not int for day in days)
                or len(set(days)) != 40):
            reasons.append('not_40_unique_complete_days')
    j, k = protocol.get('confirmation_family_index'), protocol.get('registered_contrasts')
    alpha = None
    if type(j) is not int or j < 1 or type(k) is not int or k < 1:
        reasons.append('family_or_contrast_count_missing')
    else:
        alpha = .05 / (j * (j + 1)) / 2 / (3 * k)
    supplied = protocol.get('contrast_cohort_two_sided_alpha')
    if (alpha is None or type(supplied) not in (int, float) or not math.isfinite(supplied)
            or abs(supplied - alpha) > 1e-14):
        reasons.append('allocated_alpha_missing_or_wrong')
    if alpha is None or protocol.get('bootstrap_quantiles') != [alpha / 2, 1 - alpha / 2]:
        reasons.append('bootstrap_not_family_adjusted')
    reps = protocol.get('bootstrap_repetitions')
    if type(reps) is not int or alpha is None or reps * alpha / 2 < 25:
        reasons.append('tail_MC_precision_insufficient')
    return {'checked_design_subset_pass': not reasons, 'missing_conditions': reasons,
            'all_design_qualification_established': False,
            'stable_online_gain_supported': False, 'MaskCO_attribution_established': False,
            'unchecked': ['immutable pre-data registration time and selection lineage',
                          'actual hard execution, complete actual identities and paired outcomes',
                          'qualified same-contract strong rolling comparator',
                          'independent semantic replay and all main/attribution tests']}


def raw_presence(root, manifest):
    """A filename or empty JSON is never a trajectory or identity certificate."""
    if type(manifest) is not dict:
        return {'provenance_manifest_pass': False, 'raw_qualification_pass': False,
                'reason': 'explicit source/trajectory identity manifests absent'}
    for field in ('source_files', 'trajectory_files'):
        items = manifest.get(field)
        require(type(items) is list and items, 'nonempty ' + field + ' required')
        seen = set()
        for ref in items:
            require(type(ref) is dict, 'manifest entry required')
            p = safe(root, ref.get('path'))
            require(ref['path'] not in seen, 'duplicate manifest path'); seen.add(ref['path'])
            require(sha(p) == ref.get('sha256'), 'manifest identity mismatch')
            if field == 'trajectory_files':
                payload = load(p)
                require(type(payload) is dict and type(payload.get('events')) is list
                        and payload['events'] and type(payload.get('day_id')) is int,
                        'empty/malformed event trajectory')
    return {'provenance_manifest_pass': True, 'raw_qualification_pass': False,
            'reason': 'Manifest binding and minimal schema only; semantic replay not implemented.'}


def prose_review(text):
    """Human-review contexts, never a positive certificate based on missing keywords."""
    context, cues = 'current', []
    for line in text.splitlines():
        if line.startswith('#'):
            context = 'historical' if re.search('历史|归档|旧稿', line) else 'current'
        if not re.search('不可能|唯一未测|学习.*关闭|等价', line):
            continue
        negative = bool(re.search('不能证明|未证明|未建立|不应|不是|不等于|不得|撤回|不支持', line))
        cues.append({'context': context if context == 'historical' else
                     ('negation_or_correction' if negative else 'needs_human_review'),
                     'statement': line})
    return {'cues': cues, 'automatic_scientific_adjudication': False}
