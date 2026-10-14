"""Separate four support dimensions per headline claim, read-only.

Why v2 exists. v1 collapsed every headline claim into one boolean and one list of
excesses. That made four very different failures look identical: a number that does
not reproduce from the artifact it cites, a panel whose raw per-instance
trajectories were never archived, a descriptive interval that is not a registered
confirmation, and prose that generalises past what was measured. v2 reports the
four separately, each with the evidence path it rests on:

  numeric_consistency   do the reported numbers reproduce from the archived
                        artifacts/JSON they cite?
  eligibility           bounded file inventory only; content, coverage, exact
                        identity and independent raw replay remain unverified.
                        A route filename alone never confers qualification.
  statistical_support   is the claim backed by a paired instance-level interval
                        excluding zero under the registered multiplicity/alpha
                        discipline? An interval spanning zero is reported as not
                        established, never as equivalence.
  generalization_scope  is the claim scoped to the configurations/cohorts actually
                        measured, and does the text avoid universal wording
                        (impossibility, equivalence, "never", "proved", "唯一")?

Exit-code contract (unchanged): 2 when any claim excess, unsupported claim,
fail-closed missing input or headline inconsistency is present; 0 only when the
audited document set is clean. A non-zero exit is the correct behaviour here, not a
bug. The v1 detection categories are retained, with current/history/negation context; every new finding is reported
in `additional_findings` without promoting lexical absence into scientific evidence.

Historical evidence and scripts are read as data; their code is never executed.
Nothing is written inside the authoritative input tree unless the caller passes
--allow-tool-output, which permits writes only under this tool's own reports/.

Usage:
    python -B program_support_audit.py --project <extension root> --output <report.json>
    python -B program_support_audit.py --selftest
"""
import argparse
import ast
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Standalone review copy: explicit project supplies unchanged support dependencies.
# Installation in scripts/research_claim_gate keeps ordinary same-directory imports.
if not (Path(__file__).resolve().parent / 'evidence_gate.py').is_file():
    _probe = argparse.ArgumentParser(add_help=False)
    _probe.add_argument('--project')
    _known, _unknown = _probe.parse_known_args()
    if _known.project:
        sys.path.insert(0, str(Path(_known.project).resolve() / 'scripts/research_claim_gate'))

from evidence_gate import EvidenceError, require, finite, load_json, file_hash
from verify_negative_archive import inside, PROGRAM_SHA


SCHEMA = 'program-support-audit-v3'
BASE_SOURCE_SHA256 = '2871e5b9971c79b3703252b33eb7ec567269786c0f0c1374accec7de9776cfc1'
TOOL_REPORT_DIR = Path(__file__).resolve().parent / 'reports'

# --------------------------------------------------------------------------
# Dimension vocabulary. Every verdict is either a pass or a failure for its
# dimension; `input_missing` is always a failure so a missing input can never
# be read as a clean result.
# --------------------------------------------------------------------------
DIMENSION_DEFINITIONS = {
    'numeric_consistency': {
        'question': 'do the numbers the claim reports reproduce from the archived '
                    'artifacts/JSON those numbers cite?',
        'pass_verdicts': ['reproduced_from_cited_artifacts', 'no_cited_numeric_claim'],
        'verdicts': ['reproduced_from_cited_artifacts', 'partially_reproduced',
                     'not_reproduced', 'no_cited_numeric_claim', 'input_missing'],
    },
    'eligibility': {
        'question': 'which bounded filename candidates are present? This is not identity or replay qualification.',
        'pass_verdicts': [],
        'verdicts': ['file_inventory_present_identity_and_replay_unverified',
                     'not_eligible_per_instance_trajectories_absent',
                     'not_eligible_source_identity_absent',
                     'not_eligible_no_raw_records',
                     'input_missing'],
    },
    'statistical_support': {
        'question': 'is the claim backed by a paired instance-level interval excluding '
                    'zero under the registered multiplicity/alpha discipline?',
        'pass_verdicts': ['established_under_registered_discipline'],
        'verdicts': ['established_under_registered_discipline',
                     'not_established_interval_spans_zero',
                     'not_established_descriptive_interval_only',
                     'not_established_insufficient_preregistered_cohorts',
                     'not_established_no_positive_paired_interval',
                     'not_established_no_interval_supports_a_universal_claim',
                     'input_missing'],
    },
    'generalization_scope': {
        'question': 'is the claim scoped to the configurations/cohorts actually '
                    'measured, and does the text avoid universal wording '
                    '(impossibility, equivalence, "never", "proved", "唯一")?',
        'pass_verdicts': ['scoped_to_measured_configurations'],
        'verdicts': ['scoped_to_measured_configurations',
                     'asserted_flag_exceeds_measured_scope',
                     'universal_wording_exceeds_measured_scope',
                     'input_missing'],
    },
}

OVERALL_VERDICTS = ('claim_excess_upheld', 'supported_within_measured_scope',
                    'not_asserted_in_audited_documents', 'indeterminate_input_missing')
OVERALL_NON_BLOCKING = ('supported_within_measured_scope', 'not_asserted_in_audited_documents')

# Document that records where the v1 excesses live. Required input: its absence is
# a failure, because the disposition of the five excesses would then be unknown.
EXCESS_LOCATIONS = 'results/research_20261010_continuation_v2/reproduction/CLAIM_EXCESS_LOCATIONS.md'

REQUIRED_DOCUMENTS = (
    'results/research_20261010/PROGRAM.json',
    'results/research_20261010/PROGRAM_CLOSEOUT.md',
    'results/research_20261010/ROUND_20261010_GUIDE_DOSE.md',
    'results/research_20261010/WORK_STATUS.json',
    'scripts/research_20261010/program_evidence_audit.py',
    'scripts/research_20261010/learning_attribution_adjudication.py',
    'results/research_20261010/delivery_any50_confirm_v1/result.json',
    'results/research_20261010/pickup_tail_census_v1/result.json',
    'results/research_20261010/pickup_clairvoyant_gap_v1/result.json',
    'results/research_20261010/delivery_dynamic_suffix_v1/result.json',
    'scripts/research_20261010/delivery_dynamic_suffix.py',
    EXCESS_LOCATIONS,
)

# Documents scanned for cross-cohort percentage comparisons and for universal
# wording. They are the narrative/status documents of the audited cycle.
NARRATIVE_DOCUMENTS = (
    'results/research_20261010/PROGRAM_CLOSEOUT.md',
    'results/research_20261010/ROUND_20261010_GUIDE_DOSE.md',
    'results/research_20261010/WORK_STATUS.json',
)

# --------------------------------------------------------------------------
# v1 detections, preserved verbatim. (reason, trigger, documents)
# `trigger` is None for the artifact-flag detection.
# --------------------------------------------------------------------------
V1_DETECTIONS = (
    ('ANY50_stable_flag_exceeds_snapshot_95pct_merge_gate_and_PROGRAM', None, ('artifact',)),
    ('pickup_impossibility_not_proven_by_finite_policy_diagnostics', '不可能', ('closeout',)),
    ('learning_side_closed_exceeds_tested_checkpoint_adapter_roles', 'learning_side_closed', ('closeout',)),
    ('untested_path_list_not_exhaustive_no_global_exclusion_proof', '唯一未测', ('closeout', 'rounddoc')),
    ('CI_crossing_zero_does_not_prove_runtime_mechanism_absent', '工时机制**也不存在', ('rounddoc',)),
    ('LNS_blanket_random_removal_statement_unsupported', 'LNS 文献从不使用随机', ('rounddoc',)),
    ('one_instance_distance_share_times_cohort_gain_is_not_total_utility_upper_bound', '~0.095%', ('rounddoc',)),
)

# --------------------------------------------------------------------------
# New-check tables
# --------------------------------------------------------------------------
# Nodes whose "pruned" status is a claim in its own right: pruning without the
# cited evidence on disk would be an undocumented pass.
PRUNED_NODE_EVIDENCE = (
    ('pickup', 'n3', ('results/research_20261010/pickup_tail_census_v1/result.json',)),
    ('pickup', 'n6', ('results/research_20261010/pickup_residual_census_v1/result.json',)),
    ('pickup', 'n7', ('results/research_20261010/pickup_rejection_attribution_v1/result.json',)),
    ('delivery', 'n4', ('results/research_20261010/delivery_dose_v1/result.json',)),
)

# Universal wording the generalization dimension looks for. A line carrying an
# explicit negation or an explicit supersession note is not counted: the program
# does record several of these phrases in order to withdraw them.
UNIVERSAL_WORDING = (
    ('impossibility', re.compile('不可能')),
    ('unique_path', re.compile('唯一')),
    ('never', re.compile('从不')),
    ('proved', re.compile(r'已证|\bproved\b|\bproven\b')),
    ('does_not_exist', re.compile('也不存在|不存在')),
    ('equivalence', re.compile('等价|等效')),
)
NEGATED_WORDING = re.compile('不被声称|不声称|未被证明|没有被证明|未证明|不能证明|不足以证明|未建立|不应|不得|不支持|并不|不构成|不能因此|不表示|不等于|不是')
SUPERSEDED_WORDING = re.compile('更正|覆盖|旧解释|旧版|历史|superseded|historical|撤回|WITHDRAWN|retract')
# Universal wording only counts against a claim when the line also asserts exhaustiveness
# or impossibility. "the only remaining control" is a factual statement about a registered
# control; "the only untested path" is a closure claim.
EXHAUSTIVENESS_WORDING = re.compile(r'路径|path\b|不可|不存在|关闭|穷尽|exhaust|never|唯一可能|唯一可行')

COHORT_LABELS = (
    ('dev', re.compile(r'\bdev\b|开发队列')),
    ('confirmation', re.compile(r'\bconfirmation\b|确认队列|确认批')),
    ('third_cohort', re.compile(r'第三队列|third cohort', re.I)),
)
MULTI_COHORT_PHRASE = re.compile(r'\bcohorts?\b|队列')
CROSS_COHORT_CUE = re.compile(r'\bvs\.?\b|versus|across|trend|falls|rises|than\b|dominates|'
                              r'下降|上升|趋势|跨|相比|对比|一致|比较|同向|better|worse', re.I)
CROSS_COHORT_EXCLUSION = re.compile(r'错因|撤回|WITHDRAWN|retract|not run|未运行|同一队列|同一个队列|'
                                    r'skipped|skip|更正|覆盖|事后|post-hoc|posthoc|不构成')

DATA_SUFFIXES = ('.json', '.jsonl', '.npz', '.csv', '.pkl', '.npy')
TRAJECTORY_TOKENS = ('route', 'traj', 'event', 'suffix', 'episode', 'reorder')
PER_INSTANCE_TOKENS = ('per_state', 'per_instance', 'progress', 'rows', 'analysis', 'summary', 'record')


# --------------------------------------------------------------------------
# v1 helpers, unchanged (the module's own test suite imports these by name)
# --------------------------------------------------------------------------
def constants(path, names):
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    result = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    require(set(result) == set(names), 'literal audit reference table missing')
    return result


def dig(obj, path):
    parts = path.split('.'); current = obj; i = 0
    while i < len(parts):
        if isinstance(current, dict):
            for j in range(len(parts), i, -1):
                key = '.'.join(parts[i:j])
                if key in current:
                    current = current[key]; i = j; break
            else: raise EvidenceError('missing reference ' + path)
        elif isinstance(current, list) and parts[i].isdigit():
            current = current[int(parts[i])]; i += 1
        else: raise EvidenceError('bad reference ' + path)
    return current


def interval_support(mean, ci, direction):
    mean = finite(mean, 'mean')
    require(isinstance(ci, list) and len(ci) == 2, 'CI endpoints required')
    lo, hi = [finite(v, 'CI') for v in ci]
    require(lo <= hi, 'reversed CI')
    worse_point = mean > 0 if direction == 'harmful_positive' else mean < 0
    worse_ci = lo > 0 if direction == 'harmful_positive' else hi < 0
    return {'mean': mean, 'descriptive_95_CI': ci,
            'point_estimate_worse': worse_point,
            'descriptive_interval_entirely_worse': worse_ci,
            'zero_inside_descriptive_interval': lo <= 0 <= hi,
            'equivalence_established': False, 'family_adjusted_discovery': False,
            'global_upper_bound_or_learning_exclusion': False}


def program_confirmation_conditions(protocol):
    """Fail closed, including if supplied positive flags are forged.

Necessary design checks only. No raw replay engine is provided here, so neither
passing conditions nor the caller's stable flag can confer scientific qualification.
"""
    reasons = []
    if protocol.get('program_sha256') != PROGRAM_SHA: reasons.append('no_exact_PROGRAM_binding')
    if protocol.get('unit') != 'complete_day' or protocol.get('kind') != 'dynamic_online_confirmation':
        reasons.append('not_three_dynamic_complete_day_cohorts')
    cohorts = protocol.get('confirmation_cohorts')
    if not isinstance(cohorts, list) or len(cohorts) != 3:
        reasons.append('confirmation_cohort_count_not_three'); cohorts = []
    identifiers, data_seeds = set(), set()
    for cohort in cohorts:
        seed = cohort.get('data_seed') if isinstance(cohort, dict) else None
        if type(seed) is int and seed in (20260927, 20260928):
            reasons.append('protected_pickup_final_seed_forbidden')
        days = cohort.get('day_ids') if isinstance(cohort, dict) else None
        if not isinstance(days, list) or len(days) != 40 or any(type(x) is not int for x in days) or len(set(days)) != 40:
            reasons.append('cohort_not_40_unique_registered_days'); continue
        seed = cohort.get('data_seed')
        if type(seed) is not int: reasons.append('reused_or_missing_cohort_seed')
        elif seed in data_seeds: reasons.append('reused_or_missing_cohort_seed')
        else: data_seeds.add(seed)
        if type(seed) is int:
            keys = {(seed, day) for day in days}
            if keys & identifiers: reasons.append('cohort_ids_reused')
            identifiers.update(keys)
    j, k = protocol.get('confirmation_family_index'), protocol.get('registered_contrasts')
    if type(j) is not int or j < 1 or type(k) is not int or k < 1:
        reasons.append('missing_valid_family_index_or_contrast_count'); expected_alpha = None
    else: expected_alpha = .05 / (j * (j + 1)) / 2 / (k * 3)
    alpha = protocol.get('contrast_cohort_two_sided_alpha')
    endpoints, reps = protocol.get('bootstrap_quantiles'), protocol.get('bootstrap_repetitions')
    if expected_alpha is None or type(alpha) not in (float, int) or not math.isfinite(alpha) or abs(alpha - expected_alpha) > 1e-14:
        reasons.append('family_alpha_spending_missing_or_wrong')
    if expected_alpha is None or endpoints != [expected_alpha / 2, 1 - expected_alpha / 2]:
        reasons.append('bootstrap_endpoints_not_family_adjusted')
    if type(reps) is not int or expected_alpha is None or reps * expected_alpha / 2 < 25:
        reasons.append('bootstrap_tail_MC_precision_under_25_expected_draws')
    for field in ('candidate_frozen_before_confirmation', 'source_and_weight_ids_preregistered',
                  'complete_raw_trajectory_replay_required', 'qualified_strong_rolling_control_required',
                  'all_main_and_attribution_gates_required'):
        if protocol.get(field) is not True: reasons.append('missing_literal_required_design_' + field)
    return {'checked_design_subset_pass': not reasons,
            'necessary_design_conditions_pass': False,
            'deprecated_alias_note': 'not an all-design qualification; use checked_design_subset_pass',
            'all_design_qualification_established': False,
            'unchecked_requirements': ['immutable pre-data registration timing', 'fresh cohorts independent of development', 'actual data/source/weight/budget identity', 'complete trajectory semantic replay', 'main and attribution paired gates'],
            'missing_conditions': reasons,
            'allocated_contrast_cohort_alpha': expected_alpha,
            'independent_semantic_raw_replay_implemented_here': False,
            'stable_online_gain_supported': False, 'stable_maskco_attribution_supported': False}


# --------------------------------------------------------------------------
# v2 input plumbing
# --------------------------------------------------------------------------
def read_text_input(project, relative, hashes):
    """Read a text input, recording presence/unusability instead of raising."""
    record = {'path': relative, 'exists': False, 'usable': False}
    try:
        path = inside(project, relative)
    except EvidenceError as exc:
        record['error'] = str(exc)
        return record, None
    if not path.is_file():
        record['error'] = 'missing input'
        return record, None
    record['exists'] = True
    try:
        record['sha256'] = file_hash(path)
        hashes[relative] = record['sha256']
        record['usable'] = True
        return record, path.read_text(encoding='utf-8')
    except OSError as exc:
        record['error'] = '%s: %s' % (type(exc).__name__, exc)
        return record, None


def read_json_input(project, relative, hashes):
    record, text = read_text_input(project, relative, hashes)
    if text is None:
        return record, None
    try:
        return record, load_json(inside(project, relative))
    except (EvidenceError, ValueError, TypeError) as exc:
        record['usable'] = False
        record['error'] = '%s: %s' % (type(exc).__name__, exc)
        return record, None


def read_repo_json(repo_root, relative, hashes):
    """Read a repository-level artifact outside the extension project boundary.

    The only widening is `repo_root`, which is the project's parent (or the project
    itself) and is reported in the audit output.
    """
    record = {'path': relative, 'scope': 'repo_root', 'exists': False, 'usable': False}
    try:
        path = inside(repo_root, relative)
    except EvidenceError as exc:
        record['error'] = str(exc)
        return record, None
    if not path.is_file():
        record['error'] = 'missing input'
        return record, None
    record['exists'] = True
    try:
        record['sha256'] = file_hash(path)
        hashes['repo:' + relative] = record['sha256']
        record['usable'] = True
        return record, load_json(path)
    except (EvidenceError, OSError, ValueError, TypeError) as exc:
        record['error'] = '%s: %s' % (type(exc).__name__, exc)
        return record, None


def infer_repo_root(project):
    for candidate in (project, project.parent):
        if (candidate / 'research').is_dir():
            return candidate
    return project


def dimension(dimension_name, verdict, reason, evidence_paths, **extra):
    definition = DIMENSION_DEFINITIONS[dimension_name]
    require(verdict in definition['verdicts'], 'unknown `%s` verdict %r' % (dimension_name, verdict))
    record = {'verdict': verdict, 'pass': verdict in definition['pass_verdicts'],
              'reason': reason, 'evidence_paths': list(evidence_paths)}
    record.update(extra)
    return record


def overall_verdict(asserted, indeterminate, dimensions):
    if indeterminate:
        return 'indeterminate_input_missing'
    if not asserted:
        return 'not_asserted_in_audited_documents'
    if all(d['pass'] for d in dimensions.values()):
        return 'supported_within_measured_scope'
    return 'claim_excess_upheld'


# --------------------------------------------------------------------------
# v2 scanners
# --------------------------------------------------------------------------
def current_assertion_lines(text):
    """Lexical current assertions only. History and negation are not positive claims.

    This is a context filter, never a scientific validity or registration certificate.
    Mixed negated and positive clauses separated by punctuation are checked separately.
    """
    historical_depth = None
    for number, line in enumerate(text.splitlines(), 1):
        heading = re.match(r'^\s*(#{1,6})\s+(.+)', line)
        if heading:
            depth, title = len(heading.group(1)), heading.group(2)
            if historical_depth is not None and depth <= historical_depth:
                historical_depth = None
            if re.search('历史|封存|旧版|归档|historical|archived', title, re.I):
                historical_depth = depth
        if historical_depth is not None:
            continue
        clauses = [clause for clause in re.split(r'[。；;]', line)
                   if not NEGATED_WORDING.search(clause) and not SUPERSEDED_WORDING.search(clause)
                   and not re.search(r'\bFalse\b|不产生任何正向结论', clause, re.I)]
        surface = ';'.join(clauses)
        if surface.strip():
            yield number, surface


def asserted_literal_lines(text, literal):
    return [number for number, surface in current_assertion_lines(text) if literal in surface]


def find_literal(text, literal):
    return [number for number, line in enumerate(text.splitlines(), 1) if literal in line]


def scan_directory(root, relative, max_entries=400, max_depth=3):
    """Enumerate what a panel directory actually contains. Never assumes a file."""
    record = {'directory': relative, 'exists': False, 'usable': False, 'files': [],
              'truncated': False, 'trajectory_artifacts': [],
              'per_instance_record_artifacts': []}
    try:
        path = inside(root, relative)
    except EvidenceError as exc:
        record['error'] = str(exc)
        return record
    if not path.is_dir():
        record['error'] = 'missing directory'
        return record
    record['exists'] = True; record['usable'] = True
    seen = 0
    for entry in sorted(path.rglob('*')):
        if not entry.is_file():
            continue
        if len(entry.relative_to(path).parts) > max_depth:
            continue
        seen += 1
        if seen > max_entries:
            record['truncated'] = True
            break
        name = entry.name.lower()
        rel = entry.relative_to(root).as_posix()
        record['files'].append(rel)
        # Only machine-readable data files can be per-instance raw artifacts; a log or a
        # prose file whose name merely contains "trace" is not a trajectory.
        if not name.endswith(DATA_SUFFIXES):
            continue
        if any(token in name for token in TRAJECTORY_TOKENS):
            record['trajectory_artifacts'].append(rel)
        elif name.endswith('.jsonl') or any(token in name for token in PER_INSTANCE_TOKENS):
            record['per_instance_record_artifacts'].append(rel)
    return record


def scan_universal_wording(relative, text, patterns=None):
    hits = []
    for number, line in current_assertion_lines(text):
        matched = [name for name, pattern in UNIVERSAL_WORDING
                   if pattern.search(line) and (patterns is None or name in patterns)]
        if matched:
            hits.append({'document': relative, 'line': number, 'patterns': matched,
                         'asserts_exhaustiveness': bool(EXHAUSTIVENESS_WORDING.search(line)),
                         'excerpt': line.strip()[:240]})
    return hits


def wording_hits_for(relative, text, literals=(), patterns=None):
    """Wording hits on the lines that carry this claim's own triggers.

    Restricting the scan to the claim's own lines keeps the per-claim evidence precise;
    the corpus-wide scan is reported separately.
    """
    body = dict(current_assertion_lines(text))
    wanted = set()
    for literal in literals:
        wanted.update(asserted_literal_lines(text, literal))
    if patterns:
        wanted.update(hit['line'] for hit in scan_universal_wording(relative, text, patterns))
    hits = []
    for number in sorted(wanted):
        line = body.get(number, '')
        matched = [name for name, pattern in UNIVERSAL_WORDING if pattern.search(line)]
        hits.append({'document': relative, 'line': number,
                     'patterns': matched or ['claim_trigger_string'],
                     'asserts_exhaustiveness': bool(EXHAUSTIVENESS_WORDING.search(line)),
                     'excerpt': line.strip()[:240]})
    return hits


def scan_cross_cohort(lines_by_document):
    """Flag percentage differences compared across cohorts.

    Rule (the programme's own, `rule_within_cohort_comparison` in WORK_STATUS n15):
    percentage differences may be compared ONLY within the same cohort. A line is
    flagged when it carries at least two percentage values, a comparison cue, and
    either two distinct named cohorts or a plural-cohort phrase. Lines that are
    themselves corrections, withdrawals or "confirmation not run" notes are
    excluded: those are the programme recording the rule, not violating it.
    """
    flags = []
    for relative, text in lines_by_document.items():
        for number, line in enumerate(text.splitlines(), 1):
            percentages = re.findall(r'[-+]?\d+(?:\.\d+)?\s*%', line)
            if len(percentages) < 2 or not CROSS_COHORT_CUE.search(line):
                continue
            if CROSS_COHORT_EXCLUSION.search(line):
                continue
            named = sorted({name for name, pattern in COHORT_LABELS if pattern.search(line)})
            plural = bool(MULTI_COHORT_PHRASE.search(line))
            if len(named) >= 2 or (plural and len(named) >= 1):
                flags.append({'document': relative, 'line': number,
                              'named_cohorts': named, 'plural_cohort_phrase': plural,
                              'percentages': percentages,
                              'excerpt': line.strip()[:280]})
    return flags


def document_number(label, document, text, literal, artifact, field, value,
                    scale=1.0, tolerance=5e-4):
    """Check that a number quoted in a document reproduces from the cited artifact."""
    record = {'label': label, 'kind': 'quoted_number_vs_artifact',
              'document': document, 'literal': literal,
              'artifact_path': artifact, 'field': field, 'artifact_value': value,
              'document_implied_value': value * scale, 'tolerance': tolerance,
              'literal_present_in_document': literal in text}
    record['pass'] = bool(record['literal_present_in_document'] and
                          abs(value * scale - float(literal.replace('+', ''))) <= tolerance)
    record['reason'] = '' if record['pass'] else (
        'literal %r in %s does not reproduce from %s:%s' % (literal, document, artifact, field))
    return record


def recomputation_check(label, inputs, weights, recomputed, document, text, literals,
                        tolerance=1e-3):
    """Document-internal arithmetic. Explicitly NOT artifact-backed."""
    record = {'label': label, 'kind': 'document_internal_arithmetic_not_artifact_backed',
              'document': document, 'inputs': inputs, 'weights': weights,
              'recomputed': recomputed, 'literals': list(literals), 'tolerance': tolerance}
    present = [literal in text for literal in literals]
    record['literals_present_in_document'] = present
    record['pass'] = all(present) and all(abs(rec - float(lit)) <= tolerance
                                          for rec, lit in zip(recomputed, literals))
    record['reason'] = '' if record['pass'] else 'document-internal arithmetic does not reproduce'
    return record


def numeric_dimension(checks, cited_inputs):
    if any(check.get('input_missing') for check in checks):
        return dimension('numeric_consistency', 'input_missing',
                         'a cited numeric input could not be read', cited_inputs)
    if not checks:
        return dimension('numeric_consistency', 'no_cited_numeric_claim',
                         'the assertion cites no number that can be re-read from an artifact',
                         cited_inputs)
    not_artifact_backed = [c['label'] for c in checks
                           if c['kind'] != 'quoted_number_vs_artifact' or not c['pass']]
    if all(c['pass'] for c in checks):
        if any(c['kind'] != 'quoted_number_vs_artifact' for c in checks):
            return dimension('numeric_consistency', 'partially_reproduced',
                             'cited numbers reproduce, but part of the numeric basis is '
                             'document-internal rather than artifact-backed',
                             cited_inputs, checks=checks,
                             not_artifact_backed=not_artifact_backed)
        return dimension('numeric_consistency', 'reproduced_from_cited_artifacts',
                         'every checked number reproduces from the artifact it cites',
                         cited_inputs, checks=checks)
    return dimension('numeric_consistency', 'not_reproduced',
                     'a checked number does not reproduce from the cited artifact',
                     cited_inputs, checks=checks, failing=not_artifact_backed)


def eligibility_dimension(directories, source_records, cited_inputs):
    missing = [r['path'] for r in source_records if not r['usable']]
    unusable_dirs = [d['directory'] for d in directories if not d['usable']]
    evidence = cited_inputs + [r['path'] for r in source_records] + \
        [d['directory'] for d in directories]
    if missing or unusable_dirs:
        return dimension('eligibility', 'input_missing',
                         'declared source/raw-data input missing or unreadable; a missing '
                         'input is a failure, never a pass',
                         evidence, missing_inputs=missing + unusable_dirs)
    trajectories = [f for d in directories for f in d['trajectory_artifacts']]
    records = [f for d in directories for f in d['per_instance_record_artifacts']]
    if trajectories:
        return dimension('eligibility', 'file_inventory_present_identity_and_replay_unverified',
                         'bounded filename inventory contains trajectory candidates; content, coverage, registered identity and semantic replay are unverified',
                         evidence, inventory_candidates_present=True, raw_trajectory_qualification_established=False,
                         source_identity_qualification_established=False,
                         trajectory_artifacts=trajectories[:40],
                         per_instance_record_artifacts=records[:40])
    if records:
        return dimension('eligibility', 'not_eligible_per_instance_trajectories_absent',
                         'per-instance record filenames are found but no trajectory filename was found in the bounded scan; '
                         'this is not proof of deletion or global absence; this checker cannot establish the claim as '
                         'replayable from raw data',
                         evidence, trajectory_artifacts=[],
                         per_instance_record_artifacts=records[:40])
    return dimension('eligibility', 'not_eligible_no_raw_records',
                     'no trajectory or record candidate filename found in the bounded scan; this does not prove global absence',
                     evidence, trajectory_artifacts=[], per_instance_record_artifacts=[])


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------
def audit(project, repo_root=None):
    project = Path(project).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else infer_repo_root(project)
    hashes = {}
    input_records = {}
    additional_findings = []

    def read(relative):
        record, value = read_json_input(project, relative, hashes)
        input_records[relative] = record
        if value is None:
            additional_findings.append('required_input_missing_or_unreadable:' + relative)
        return value

    def text(relative):
        record, value = read_text_input(project, relative, hashes)
        input_records[relative] = record
        if value is None:
            additional_findings.append('required_input_missing_or_unreadable:' + relative)
        return value

    # ---- presence of every required document, fail closed -----------------
    required_status = {}
    for relative in REQUIRED_DOCUMENTS:
        if relative not in input_records:
            if relative.endswith('.json'):
                read(relative)
            else:
                text(relative)
        record = input_records[relative]
        required_status[relative] = {'exists': record['exists'], 'usable': record['usable'],
                                     'sha256': record.get('sha256'), 'error': record.get('error')}
    missing_required = sorted(r for r, s in required_status.items() if not s['usable'])

    read('results/research_20261010/PROGRAM.json')
    require(hashes['results/research_20261010/PROGRAM.json'] == PROGRAM_SHA, 'PROGRAM identity mismatch')

    refs_path = inside(project, 'scripts/research_20261010/program_evidence_audit.py')
    hashes[refs_path.relative_to(project).as_posix()] = file_hash(refs_path)
    refs = constants(refs_path, ['CLAIMS', 'EXPECTED_FLAGS', 'SCOPE_FLAGS'])

    trees = {line: read('results/research_20261010/' + line + '/.arbor/tree.json')
             for line in ('pickup', 'delivery', 'reproduction')}

    closeout_rel = 'results/research_20261010/PROGRAM_CLOSEOUT.md'
    rounddoc_rel = 'results/research_20261010/ROUND_20261010_GUIDE_DOSE.md'
    workdoc_rel = 'results/research_20261010/WORK_STATUS.json'
    # A searched document that is missing is already recorded as a failure above.
    # Substituting the empty string here keeps the audit running so that the failure is
    # reported explicitly instead of crashing; absence is never turned into a pass.
    closeout = text(closeout_rel) or ''
    rounddoc = text(rounddoc_rel) or ''
    workdoc = text(workdoc_rel) or ''
    documents = {'closeout': closeout, 'rounddoc': rounddoc, 'workdoc': workdoc}

    # ---- v1 headline consistency and v1 interval records ------------------
    engineering = []
    for line, node, rel, field, label, sign in refs['CLAIMS']:
        value = finite(dig(read('results/' + rel), field), label)
        score = finite(trees[line]['nodes'][node]['metadata']['dev_score'], 'tree score')
        engineering.append({'kind': 'headline_number', 'line': line, 'node': node, 'field': field,
                            'pass': abs(score - value * 100 * sign) < .02})
    for category in ('EXPECTED_FLAGS', 'SCOPE_FLAGS'):
        for reference in refs[category]:
            line, node, rel, field, expected = reference[:5]
            value = dig(read('results/' + rel), field)
            engineering.append({'kind': category, 'line': line, 'node': node, 'field': field,
                                'pass': type(value) is type(expected) and value == expected})
    lpath = inside(project, 'scripts/research_20261010/learning_attribution_adjudication.py')
    hashes[lpath.relative_to(project).as_posix()] = file_hash(lpath)
    learned = []
    for label, rel, field, claim, direction in constants(lpath, ['EVIDENCE'])['EVIDENCE']:
        data = read('results/' + rel); value = dig(data, field)
        parent = dig(data, field.rsplit('.', 1)[0])
        supporting = interval_support(value, parent['descriptive_95_CI'], direction)
        if direction == 'harmful':
            consistent = supporting['mean'] < 0
        elif direction == 'harmful_positive':
            consistent = supporting['mean'] > 0
        else:
            consistent = supporting['mean'] < 0.01
        supporting.update(label=label, original_declared_direction=direction, source=rel,
                          direction_consistent=bool(consistent))
        learned.append(supporting)

    protocol_support = {}
    scripts = project / 'scripts/research_20261010'
    for path in sorted(scripts.glob('*protocol*.json')):
        rel = path.relative_to(project).as_posix(); protocol = read(rel)
        protocol_support[path.name] = program_confirmation_conditions(protocol)
    confirm = read('results/research_20261010/delivery_any50_confirm_v1/result.json')
    protocol_support['actual_ANY50_result_protocol'] = program_confirmation_conditions(confirm['protocol'])
    tail = read('results/research_20261010/pickup_tail_census_v1/result.json')
    clair = read('results/research_20261010/pickup_clairvoyant_gap_v1/result.json')
    dynamic_source = text('scripts/research_20261010/delivery_dynamic_suffix.py') or ''
    dynamic = read('results/research_20261010/delivery_dynamic_suffix_v1/result.json')
    if isinstance(dynamic.get('protocol'), dict):
        protocol_support['actual_dynamic_suffix_screen_protocol'] = program_confirmation_conditions(dynamic['protocol'])
    audited = {(e['line'], e['node']) for e in engineering}
    unaudited = [(line, node) for line, tree in trees.items() for node, entry in tree['nodes'].items()
                 if entry.get('status') in ('executed', 'merged') and (line, node) not in audited]

    # ---- the five v1 claim-excess detections, verbatim --------------------
    v1_excesses = []
    for reason, trigger, where in V1_DETECTIONS:
        fired = False
        if where == ('artifact',):
            fired = confirm.get('stable_gain_established') is True
        elif trigger == '~0.095%':
            fired = bool(asserted_literal_lines(rounddoc, '~0.095%') or asserted_literal_lines(closeout, '0.095%'))
        else:
            fired = any(asserted_literal_lines(documents[doc], trigger) for doc in where)
        if fired:
            v1_excesses.append(reason)

    # ---- documented disposition of those excesses -------------------------
    locations_record, locations_text = read_text_input(project, EXCESS_LOCATIONS, hashes)
    input_records[EXCESS_LOCATIONS] = locations_record
    documented_triggers = set()
    if locations_text:
        for cell in re.findall(r'\|\s*`([^`]+)`[^|]*\|', locations_text):
            documented_triggers.update(part.strip() for part in cell.split(';'))
    else:
        additional_findings.append('claim_excess_locations_document_unavailable')
    trigger_of = {reason: trigger for reason, trigger, _ in V1_DETECTIONS}
    # the disposition table quotes "0.095%" while the detector's literal is "~0.095%"
    normalised_documented = {trigger.strip('~') for trigger in documented_triggers}
    undocumented_excesses = sorted(
        reason for reason, trigger, _ in V1_DETECTIONS
        if reason in v1_excesses and trigger and trigger.strip('~') not in normalised_documented)

    # ---- per-claim four-dimension records ---------------------------------
    claim_excess_by_id = {
        'stable_online_gain_supported': (
            'ANY50_stable_flag_exceeds_snapshot_95pct_merge_gate_and_PROGRAM',),
        'MaskCO_attributable_gain_established': (
            'LNS_blanket_random_removal_statement_unsupported',),
        'learning_side_closed': (
            'learning_side_closed_exceeds_tested_checkpoint_adapter_roles',
            'untested_path_list_not_exhaustive_no_global_exclusion_proof'),
        'pickup_impossibility_proven': (
            'one_instance_distance_share_times_cohort_gain_is_not_total_utility_upper_bound',),
        'equivalence_established': (
            'CI_crossing_zero_does_not_prove_runtime_mechanism_absent',),
    }

    def assertion_block(claim_id):
        """Was the claim actually asserted by the audited document set?"""
        artifact_notes = {'ANY50_stable_flag_exceeds_snapshot_95pct_merge_gate_and_PROGRAM':
                          'results/research_20261010/delivery_any50_confirm_v1/result.json :: '
                          'stable_gain_established == true'}
        evidence = []
        reasons = []
        triggers = []
        for reason in claim_excess_by_id[claim_id]:
            if reason not in v1_excesses:
                continue
            reasons.append(reason)
            trigger = trigger_of.get(reason)
            if trigger is None:
                evidence.append({'path': 'results/research_20261010/delivery_any50_confirm_v1/result.json',
                                 'locator': 'stable_gain_established', 'value': True,
                                 'detection_reason': reason})
                continue
            triggers.append(trigger)
            for doc in ('closeout', 'rounddoc'):
                body = documents[doc]
                for number in asserted_literal_lines(body, trigger):
                    evidence.append({'path': {'closeout': closeout_rel, 'rounddoc': rounddoc_rel}[doc],
                                     'locator': 'line %d' % number, 'trigger': trigger,
                                     'detection_reason': reason,
                                     'excerpt': body.splitlines()[number - 1].strip()[:240]})
        return {'asserted_in_audited_documents': bool(evidence),
                'trigger_strings': triggers,
                'detection_reasons': reasons,
                'assertion_evidence': evidence,
                'note': artifact_notes.get(reasons[0] if reasons else '', '')}

    # ---------------- claim C1: stable online gain -------------------------
    confirm_protocol = confirm['protocol']
    primary = confirm['primary_ANY50_over_LNS']
    c1_numeric = []
    tree_n7 = trees['delivery']['nodes']['n7']['metadata'].get('dev_score')
    if tree_n7 is not None:
        expected = float(primary['mean_relative_gain']) * 100.0
        c1_numeric.append({'label': 'delivery tree n7 dev_score vs ANY50 confirmation artifact',
                           'kind': 'quoted_number_vs_artifact',
                           'document': 'results/research_20261010/delivery/.arbor/tree.json',
                           'literal': str(tree_n7),
                           'artifact_path': 'results/research_20261010/delivery_any50_confirm_v1/result.json',
                           'field': 'primary_ANY50_over_LNS.mean_relative_gain',
                           'artifact_value': primary['mean_relative_gain'],
                           'document_implied_value': expected, 'tolerance': 0.02,
                           'literal_present_in_document': True,
                           'pass': abs(float(tree_n7) - expected) < 0.02,
                           'reason': ''})
    gate_ok = (primary['mean_relative_gain'] >= 0.01 and primary['descriptive_95_CI'][0] > 0)
    c1_numeric.append({'label': 'artifact gate_passed agrees with its own registered gate rule',
                       'kind': 'quoted_number_vs_artifact',
                       'document': 'results/research_20261010/delivery_any50_confirm_v1/result.json',
                       'literal': str(confirm.get('gate_passed')),
                       'artifact_path': 'results/research_20261010/delivery_any50_confirm_v1/result.json',
                       'field': 'gate_passed', 'artifact_value': confirm.get('gate_passed'),
                       'document_implied_value': gate_ok, 'tolerance': 0.0,
                       'literal_present_in_document': True,
                       'pass': bool(gate_ok == confirm.get('gate_passed')),
                       'reason': ''})

    # ---------------- claim C2: MaskCO attribution -------------------------
    evidence_paths_c2 = sorted({l['source'] for l in learned})
    c2_numeric_pass = bool(learned) and all(l['direction_consistent'] for l in learned)
    c2_numeric = [{'label': 'direction consistency: ' + l['label'][:70],
                   'kind': 'artifact_interval_direction',
                   'document': 'scripts/research_20261010/learning_attribution_adjudication.py',
                   'artifact_path': 'results/' + l['source'], 'field': l['label'],
                   'artifact_value': l['mean'], 'document_implied_value': None,
                   'tolerance': 0.0, 'literal_present_in_document': True,
                   'pass': bool(l['direction_consistent']),
                   'reason': '' if l['direction_consistent'] else 'declared direction not reproduced'}
                  for l in learned]

    # ---------------- shared eligibility inputs ----------------------------
    def directories_for(paths):
        return [scan_directory(project, path) for path in paths]

    def sources_for(paths, repo_paths=()):
        records = []
        for path in paths:
            reader = read_json_input if path.endswith('.json') else read_text_input
            record, _ = reader(project, path, hashes)
            input_records[path] = record
            records.append(record)
        for path in repo_paths:
            record, _ = read_repo_json(repo_root, path, hashes)
            input_records['repo:' + path] = record
            records.append(record)
        return records

    c1_dirs = directories_for(('results/research_20261010/delivery_any50_confirm_v1',))
    c1_sources = sources_for(('results/research_20261010/delivery_any50_confirm_v1/PROTOCOL.json',
                              'results/research_20261010/delivery_any50_confirm_v1/identity_start.json',
                              'scripts/research_20261010/delivery_anytime.py',
                              'scripts/research_20261010/delivery_any50_confirm.py'))
    c1_eligibility = eligibility_dimension(
        c1_dirs, c1_sources,
        ['results/research_20261010/delivery_any50_confirm_v1/result.json'])

    learn_dirs = directories_for(('results/m0_scale/_bverify_pinned',
                                  'negative_result_paper/results/09_unified_v1',
                                  'negative_result_paper/results/10_confirm_c1',
                                  'results/m0_scale/fsq_clean_train'))
    learn_sources = sources_for(('results/three_line_corrections_20261009/'
                                 'FINAL_PROTECTED_ACCEPTANCE_20261010_v2.json',
                                 '03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/RECOVERY_MANIFEST.json'))
    c2_eligibility = eligibility_dimension(learn_dirs, learn_sources, evidence_paths_c2)

    pickup_dirs = directories_for(('results/research_20261010/pickup_tail_census_v1',
                                   'results/research_20261010/pickup_budget_headroom_v1'))
    pickup_sources = sources_for(('results/research_20261010/pickup_rejection_attribution_v1/result.json',
                                  'results/research_20261010/pickup_residual_census_v1/result.json',
                                  'results/research_20261010/pickup_clairvoyant_gap_v1/result.json'),
                                 ('research/pickup_20round_audit/known_batch_v1/BATCH_SUMMARY.json',))
    witness = input_records.get('repo:research/pickup_20round_audit/known_batch_v1/BATCH_SUMMARY.json')
    c4_eligibility = eligibility_dimension(
        pickup_dirs, pickup_sources, ['results/research_20261010/pickup_tail_census_v1/result.json'])
    c4_eligibility['counter_witness_present'] = bool(witness and witness['usable'])
    c4_eligibility['counter_witness_path'] = \
        'research/pickup_20round_audit/known_batch_v1/BATCH_SUMMARY.json (resolved under repo_root)'
    c4_eligibility['counter_witness_note'] = (
        'the round-20 closeout cites this 16-state exact-utility witness as contradicting the '
        'utility-ceiling reading; it is present on disk' if witness and witness['usable'] else
        'the cited 16-state exact-utility witness could not be resolved; that is a missing input, '
        'not a pass')

    decomp_dirs = directories_for(('results/research_20261010/delivery_decomp_v1',
                                   'results/research_20261010/delivery_recheck_v1'))
    decomp_sources = sources_for(('results/research_20261010/delivery_decomp_v1/result.json',
                                  'results/research_20261010/delivery_recheck_v1/result.json'))
    c5_eligibility = eligibility_dimension(
        decomp_dirs, decomp_sources, ['results/research_20261010/delivery_decomp_v1/result.json'])

    # ---------------- cross-cohort percentage comparisons ------------------
    cross_cohort = scan_cross_cohort({rel: documents[key] for key, rel in
                                      (('closeout', closeout_rel), ('rounddoc', rounddoc_rel),
                                       ('workdoc', workdoc_rel))})
    for flag in cross_cohort:
        additional_findings.append('cross_cohort_percentage_comparison:%s:%d'
                                   % (flag['document'], flag['line']))

    # ---------------- confirmation-family pre-registration ------------------
    anytime_protocol = read('results/research_20261010/delivery_anytime_v1/PROTOCOL.json')
    posthoc = read('results/research_20261010/delivery_anytime_v1/posthoc_any50_comparison.json')
    posthoc_source = text('scripts/research_20261010/delivery_anytime_posthoc.py')
    posthoc_marks_non_preregistered = (isinstance(posthoc, dict) and
                                       posthoc.get('not_a_preregistered_test') is True)
    cohorts = []
    if isinstance(anytime_protocol, dict):
        cohorts.append({'cohort': 'D-ANYTIME-v1 dev',
                        'dataset_seed': anytime_protocol.get('dataset_seed_dev'),
                        'instance_ids': anytime_protocol.get('dev_instance_ids'),
                        'declared_preregistration_marker': not posthoc_marks_non_preregistered,
                        'evidence_path': 'results/research_20261010/delivery_anytime_v1/PROTOCOL.json'})
        cohorts.append({'cohort': 'D-ANYTIME-v1 confirmation',
                        'dataset_seed': anytime_protocol.get('dataset_seed_confirmation'),
                        'instance_ids': anytime_protocol.get('confirmation_instance_ids'),
                        'declared_preregistration_marker': not posthoc_marks_non_preregistered,
                        'evidence_path': 'results/research_20261010/delivery_anytime_v1/PROTOCOL.json'})
    if isinstance(confirm_protocol, dict):
        cohorts.append({'cohort': 'D-ANY50-CONFIRM-v1',
                        'dataset_seed': confirm_protocol.get('dataset_seed'),
                        'instance_ids': confirm_protocol.get('instance_ids'),
                        'declared_preregistration_marker': confirm_protocol.get('preregistered_primary') is True,
                        'evidence_path':
                            'results/research_20261010/delivery_any50_confirm_v1/result.json'})
    preregistered_required = 3
    preregistered_found = sum(1 for c in cohorts if c['declared_preregistration_marker'])
    cohort_shortfall = preregistered_found < preregistered_required
    if cohort_shortfall:
        additional_findings.append('preregistered_confirmation_cohorts_found_%d_of_%d'
                                   % (preregistered_found, preregistered_required))
    confirmation_check = {
        'verdict': 'insufficient_preregistered_cohorts' if cohort_shortfall
                   else 'three_declared_preregistration_markers_found_not_time_verified',
        'blocks_clean_exit': bool(cohort_shortfall),
        'cohorts_implied_by_documents': len(cohorts),
        'cohorts_declared_preregistration_marker': preregistered_found,
        'registration_timing_independently_verified': False,
        'cohorts_required_by_the_registered_program': preregistered_required,
        'cohorts': cohorts,
        'non_preregistration_marker_present': bool(posthoc_marks_non_preregistered),
        'marker_source': {
            'artifact': 'results/research_20261010/delivery_anytime_v1/posthoc_any50_comparison.json',
            'field': 'not_a_preregistered_test',
            'value': posthoc.get('not_a_preregistered_test') if isinstance(posthoc, dict) else None},
        'evidence_paths': [
            'results/research_20261010/delivery_anytime_v1/PROTOCOL.json',
            'results/research_20261010/delivery_anytime_v1/posthoc_any50_comparison.json',
            'results/research_20261010/delivery_any50_confirm_v1/result.json',
            'scripts/research_20261010/delivery_anytime_posthoc.py'],
        'note': ('the two D-ANYTIME-v1 cohorts are marked not_a_preregistered_test=true by the '
                 'panel that produced the comparison; only the third cohort declares '
                 'preregistered_primary=true. A preregistration marker is a stored literal, not '
                 'proof of registration time.'),
        'posthoc_source_declares_non_preregistration': bool(
            posthoc_source and 'not_a_preregistered_test' in posthoc_source),
    }

    # ---------------- pruned / clean status with missing inputs -------------
    pruned_records = []
    pruned_failures = []
    for line, node, references in PRUNED_NODE_EVIDENCE:
        tree = trees.get(line) or {}
        entry = (tree.get('nodes') or {}).get(node) or {}
        metadata = entry.get('metadata') or {}
        status = entry.get('status')
        reason_present = bool(metadata.get('prune_reason')) or bool((metadata.get('result') or '').strip())
        refs = []
        for relative in references:
            record, value = read_json_input(project, relative, hashes)
            input_records[relative] = record
            refs.append({'path': relative, 'exists': record['exists'], 'usable': record['usable'],
                         'sha256': record.get('sha256'), 'error': record.get('error')})
            if not record['usable']:
                pruned_failures.append('pruned_node_evidence_missing:%s/%s:%s' % (line, node, relative))
        if status != 'pruned':
            pruned_failures.append('expected_pruned_status_absent:%s/%s' % (line, node))
        if not reason_present:
            pruned_failures.append('pruned_status_without_recorded_reason:%s/%s' % (line, node))
        pruned_records.append({'tree': line, 'node': node, 'status': status,
                               'branch_ref': metadata.get('branch_ref'),
                               'recorded_reason_present': reason_present,
                               'referenced_evidence': refs})
    missing_searched = sorted(rel for rel in NARRATIVE_DOCUMENTS
                              if not input_records.get(rel, {}).get('usable', False))
    for rel in missing_searched:
        pruned_failures.append('searched_document_missing:' + rel)
    pruned_check = {
        'verdict': 'fail_closed_missing_input' if pruned_failures else 'all_pruned_statuses_have_evidence_on_disk',
        'blocks_clean_exit': bool(pruned_failures),
        'definition': ('a pruned/clean status passes only when the documents searched for claim '
                       'excesses are present AND every artifact the pruning cites is present and '
                       'readable; a missing input is a failure, never a pass'),
        'pruned_nodes': pruned_records,
        'searched_documents': {rel: bool(input_records.get(rel, {}).get('usable', False))
                               for rel in NARRATIVE_DOCUMENTS},
        'failures': pruned_failures,
        'evidence_paths': sorted({r['path'] for record in pruned_records
                                  for r in record['referenced_evidence']} |
                                 set(NARRATIVE_DOCUMENTS)),
    }
    additional_findings.extend(pruned_failures)

    # ---------------- claim records ----------------------------------------
    claims = []

    def claim_record(claim_id, statement, assertions, numeric, eligible, statistical, scope):
        dims = {'numeric_consistency': numeric, 'eligibility': eligible,
                'statistical_support': statistical, 'generalization_scope': scope}
        indeterminate = any(v['verdict'] == 'input_missing' for v in dims.values())
        asserted = assertions['asserted_in_audited_documents']
        overall = overall_verdict(asserted, indeterminate, dims)
        evidence_paths = sorted({p for v in dims.values() for p in v['evidence_paths']} |
                                {e['path'] for e in assertions['assertion_evidence']})
        return {'claim_id': claim_id, 'statement': statement,
                'asserted_in_audited_documents': asserted,
                'assertion': assertions,
                'dimensions': dims,
                'dimension_verdicts': {k: v['verdict'] for k, v in dims.items()},
                'failing_dimensions': [k for k, v in dims.items() if not v['pass']],
                'evidence_paths': evidence_paths,
                'overall_verdict': overall,
                'overall_reason': ('the documents assert the claim and at least one support '
                                   'dimension fails' if overall == 'claim_excess_upheld' else
                                   'no audited document asserts the claim' if overall ==
                                   'not_asserted_in_audited_documents' else
                                   'a required input is missing, so support is indeterminate' if
                                   overall == 'indeterminate_input_missing' else
                                   'all four dimensions pass for the scope asserted')}

    # C1
    c1_assertion = assertion_block('stable_online_gain_supported')
    c1_assertion['claim_as_asserted'] = ('the ANY50 confirmation panel records a literal '
                                         'stable_gain_established=true although the PROGRAM '
                                         'requires three fresh 40-complete-day preregistered cohorts')
    c1 = claim_record(
        'stable_online_gain_supported', 'a stable online gain is established',
        c1_assertion,
        numeric_dimension(c1_numeric, ['results/research_20261010/delivery/.arbor/tree.json',
                                       'results/research_20261010/delivery_any50_confirm_v1/result.json']),
        c1_eligibility,
        dimension('statistical_support', 'not_established_descriptive_interval_only',
                  'the interval is a descriptive 95% CI over a 32-instance snapshot cohort; the '
                  'registered confirmation family (three fresh 40-complete-day preregistered '
                  'cohorts with allocated alpha) was not executed',
                  ['results/research_20261010/delivery_any50_confirm_v1/result.json'],
                  interval_excludes_zero=bool(primary['descriptive_95_CI'][0] > 0),
                  descriptive_ci=primary['descriptive_95_CI'],
                  mean_relative_gain=primary['mean_relative_gain'],
                  registered_family_alpha_applied=False,
                  preregistered_cohorts_found=preregistered_found,
                  preregistered_cohorts_required=preregistered_required,
                  equivalence_established=False),
        dimension('generalization_scope', 'asserted_flag_exceeds_measured_scope',
                  'the asserted flag is worded as a stable gain while the measured unit is a '
                  '32-instance deterministic snapshot on one cohort',
                  ['results/research_20261010/delivery_any50_confirm_v1/result.json',
                   'results/research_20261010/WORK_STATUS.json'],
                  universal_wording_hits=[],
                  scoped_to='32-instance snapshot cohort, seed 2026101141, instance ids 2000-2031'))
    claims.append(c1)

    # C2
    c2_assertion = assertion_block('MaskCO_attributable_gain_established')
    c2_assertion['claim_as_asserted'] = ('the round document states that the LNS literature '
                                         'never uses random removal, which is used to bound what '
                                         'a learned mask selector could find')
    c2 = claim_record(
        'MaskCO_attributable_gain_established', 'a MaskCO-attributable gain is established',
        c2_assertion,
        dimension('numeric_consistency',
                  'reproduced_from_cited_artifacts' if c2_numeric_pass else 'not_reproduced',
                  'each cited learning-reference interval was re-read from its archived artifact '
                  'and its declared direction re-checked',
                  evidence_paths_c2, checks=c2_numeric),
        c2_eligibility,
        dimension('statistical_support', 'not_established_no_positive_paired_interval',
                  'the cited paired intervals are harmful or fail a positive threshold; none '
                  'excludes zero in favour of a learned gain',
                  evidence_paths_c2,
                  n_cited_intervals=len(learned),
                  n_harmful=sum(1 for l in learned if l['original_declared_direction'] == 'harmful'),
                  n_no_signal=sum(1 for l in learned if l['original_declared_direction'] == 'no_signal')),
        dimension('generalization_scope', 'universal_wording_exceeds_measured_scope',
                  'the text generalises over the literature rather than over the measured operator',
                  [rounddoc_rel],
                  universal_wording_hits=wording_hits_for(
                      rounddoc_rel, rounddoc, literals=('LNS 文献从不使用随机',),
                      patterns=('never',)),
                  scoped_to='uniform random destruction vs one single-seed Euclidean distance '
                            'clustering removal on the delivery dev cohort'))
    claims.append(c2)

    # C3
    c3_assertion = assertion_block('learning_side_closed')
    c3_assertion['claim_as_asserted'] = ('the round document records the adjudicator verdict '
                                         'learning_side_closed and calls the remaining training '
                                         'route the unique untested path')
    c3 = claim_record(
        'learning_side_closed', 'the learning side of the programme is closed',
        c3_assertion,
        numeric_dimension([], ['scripts/research_20261010/learning_attribution_adjudication.py']),
        c2_eligibility,
        dimension('statistical_support', 'not_established_no_interval_supports_a_universal_claim',
                  'no paired interval can establish a universal closure; the cited intervals '
                  'constrain five measured channels only',
                  evidence_paths_c2, channels_measured=5),
        dimension('generalization_scope', 'universal_wording_exceeds_measured_scope',
                  'a limited negative result is written as a closed side and an exhaustive path list',
                  [rounddoc_rel, closeout_rel],
                  universal_wording_hits=(
                      wording_hits_for(rounddoc_rel, rounddoc,
                                       literals=('learning_side_closed', '唯一未测')) +
                      wording_hits_for(closeout_rel, closeout,
                                       literals=('learning_side_closed', '唯一未测'))),
                  scoped_to='zero-shot pretrained CVRP weights, current adapter, measured operators'))
    claims.append(c3)

    # C4
    c4_assertion = assertion_block('pickup_impossibility_proven')
    c4_assertion['claim_as_asserted'] = ('the round document converts a single-instance distance '
                                         'share into a utility ceiling of about 0.095 percent')
    c4_numeric = []
    if c4_assertion['asserted_in_audited_documents']:
        c4_numeric.append(document_number(
            'pickup tail distance headroom quoted in the round document',
            rounddoc_rel, rounddoc, '0.997',
            'results/research_20261010/pickup_tail_census_v1/result.json',
            'distance_headroom.mean_relative_gain', tail['distance_headroom']['mean_relative_gain'],
            scale=100.0, tolerance=5e-4))
        # The quoted components are read from the round document's own prose, not from an
        # archived artifact, so this check is labelled document-internal on purpose.
        distance_km, energy_kwh, quality_loss = 23.4229, 2213.5795, 2.2939
        composite = 1.0 * distance_km + 0.1 * energy_kwh + 1.0 * quality_loss
        share_fraction = distance_km / composite
        headroom_fraction = tail['distance_headroom']['mean_relative_gain']
        c4_numeric.append(recomputation_check(
            'single-instance distance share and the 0.095 percent ceiling product',
            {'distance_km': distance_km, 'energy_kwh': energy_kwh, 'quality_loss': quality_loss,
             'composite_cost_quoted': 247.0747,
             'distance_headroom_from_artifact': headroom_fraction},
            {'distance': 1.0, 'energy': 0.1, 'quality': 1.0},
            [share_fraction * 100.0, share_fraction * headroom_fraction * 100.0, composite],
            rounddoc_rel, rounddoc, ['9.48', '0.095', '247.0747'], tolerance=1e-3))
    c4 = claim_record(
        'pickup_impossibility_proven', 'pickup is proven to have no improvable strategy',
        c4_assertion,
        numeric_dimension(c4_numeric, ['results/research_20261010/pickup_tail_census_v1/result.json',
                                       rounddoc_rel]),
        c4_eligibility,
        dimension('statistical_support', 'not_established_no_interval_supports_a_universal_claim',
                  'a distribution-free impossibility reading is not a paired interval; the '
                  'distance census is a distance-only measurement on a different operating point',
                  ['results/research_20261010/pickup_tail_census_v1/result.json'],
                  measured_scope='distance only; protocol.not_a_verdict_on = %r'
                                 % tail['protocol'].get('not_a_verdict_on')),
        dimension('generalization_scope', 'universal_wording_exceeds_measured_scope',
                  'a distance share measured on one instance is multiplied by a cohort-average '
                  'distance headroom and then read as a total-utility ceiling',
                  [rounddoc_rel, 'results/research_20261010/pickup_tail_census_v1/result.json'],
                  universal_wording_hits=wording_hits_for(
                      rounddoc_rel, rounddoc, literals=('0.095%',), patterns=('impossibility',)),
                  scoped_to='same-vehicle tail reordering under the old composite cost on the '
                            'pickup dev-check cohort'))
    claims.append(c4)

    # C5
    c5_assertion = assertion_block('equivalence_established')
    c5_assertion['claim_as_asserted'] = ('the round document reads a duty-time interval that '
                                         'crosses zero as evidence that the runtime mechanism '
                                         'does not exist')
    c5_numeric = []
    decomp = read('results/research_20261010/delivery_decomp_v1/result.json')
    duty = decomp['summary']['duty_time_difference_h']
    if c5_assertion['asserted_in_audited_documents']:
        for literal, value, tolerance in (('0.0435', duty['mean'], 5e-5),
                                          ('-0.081', duty['descriptive_95_CI'][0], 5e-4),
                                          ('+0.147', duty['descriptive_95_CI'][1], 5e-4)):
            c5_numeric.append(document_number(
                'duty-time difference and CI quoted in WORK_STATUS n15_result',
                workdoc_rel, workdoc, literal,
                'results/research_20261010/delivery_decomp_v1/result.json',
                'summary.duty_time_difference_h', value, scale=1.0, tolerance=tolerance))
    c5 = claim_record(
        'equivalence_established', 'equivalence of the duty-time mechanism is established',
        c5_assertion,
        numeric_dimension(c5_numeric, ['results/research_20261010/delivery_decomp_v1/result.json',
                                       workdoc_rel]),
        c5_eligibility,
        dimension('statistical_support', 'not_established_interval_spans_zero',
                  'the duty-time paired interval spans zero, which reports an unestablished '
                  'difference and never equivalence',
                  ['results/research_20261010/delivery_decomp_v1/result.json'],
                  duty_time_mean=duty['mean'], duty_time_descriptive_95_CI=duty['descriptive_95_CI'],
                  zero_inside_interval=bool(duty['descriptive_95_CI'][0] <= 0 <= duty['descriptive_95_CI'][1]),
                  equivalence_established=False),
        dimension('generalization_scope', 'universal_wording_exceeds_measured_scope',
                  'an interval that spans zero is written as the mechanism not existing',
                  [rounddoc_rel],
                  universal_wording_hits=wording_hits_for(
                      rounddoc_rel, rounddoc, literals=('工时机制**也不存在',),
                      patterns=('does_not_exist',)),
                  scoped_to='12-instance delivery dev subset, duty time transcribed'))
    claims.append(c5)

    # ---------------- additional universal wording -------------------------
    wording_hits = scan_universal_wording(rounddoc_rel, rounddoc) + \
        scan_universal_wording(closeout_rel, closeout)
    fired_triggers = {trigger_of[reason] for reason in v1_excesses if trigger_of.get(reason)}
    extra_wording = [hit for hit in wording_hits
                     if hit['asserts_exhaustiveness']
                     and not any(trigger in hit['excerpt'] for trigger in fired_triggers)]
    for hit in extra_wording:
        additional_findings.append('universal_wording_without_measured_scope:%s:%d:%s'
                                   % (hit['document'], hit['line'], ','.join(hit['patterns'])))

    # ---------------- assemble ---------------------------------------------
    scientific_support = {
        'stable_online_gain_supported': next(c['overall_verdict'] == 'supported_within_measured_scope'
                                             for c in claims if c['claim_id'] == 'stable_online_gain_supported'),
        'MaskCO_attributable_gain_established': next(c['overall_verdict'] == 'supported_within_measured_scope'
                                                     for c in claims
                                                     if c['claim_id'] == 'MaskCO_attributable_gain_established'),
        'pickup_impossibility_proven': next(c['overall_verdict'] == 'supported_within_measured_scope'
                                            for c in claims if c['claim_id'] == 'pickup_impossibility_proven'),
        'learning_side_closed': next(c['overall_verdict'] == 'supported_within_measured_scope'
                                     for c in claims if c['claim_id'] == 'learning_side_closed'),
        'equivalence_established': next(c['overall_verdict'] == 'supported_within_measured_scope'
                                        for c in claims if c['claim_id'] == 'equivalence_established'),
        'reported_findings_level': 'bounded_descriptive_snapshot_and_policy_diagnostics',
        'claim_excesses': v1_excesses,
        'claim_excesses_note': ('the v1 detection categories in v1 order, now filtered by assertion context; new '
                                'findings are reported in additional_findings so this list stays '
                                'comparable with earlier archived audits'),
        'claim_overall_verdicts': {c['claim_id']: c['overall_verdict'] for c in claims},
        'dimension_failure_counts': {
            name: sum(1 for c in claims if not c['dimensions'][name]['pass'])
            for name in DIMENSION_DEFINITIONS},
        'pickup_tail_mean_distance_gain': tail['distance_headroom']['mean_relative_gain'],
        'pickup_tail_objective_scope': tail['protocol']['not_a_verdict_on'],
        'clairvoyant_reference': clair['protocol']['full_information_reference'],
        'dynamic_suffix_screen': {'n_instances': dynamic.get('n_instances'),
            'reported_mean_relative_distance_gain': dynamic.get('mean_relative_distance_gain'),
            'raw_execution_replayed_here': False,
            'source_contains_shared_NN_initialization': "engine.plan(vid, build_suffix(cargo, obs, 'reorder', node) + [0])" in dynamic_source,
            'limited_to': 'same loaded cargo, same ownership, deterministic nearest-neighbour suffix recomputation',
            'does_not_exclude': ['other suffix searches or MaskCO reconstructions', 'reassignment/admission/dispatch policies', 'different release and loading patterns'],
            'logical_note': 'A deterministic nearest-neighbour sequence retains the same successor choices after its executed prefix is removed, when no input geometry/cargo changes; its zero recourse effect is not an action-set upper bound.'},
        'why_no_impossibility': 'A feasible greedy full-information policy is a lower bound on an optimum; a null or worse result cannot close policies or action sets.',
        'why_no_learning_bound': 'Finite nonlearned masking heuristics do not upper-bound learned policies; worse downstream objective is not a measured representation correlation.',
        'open_paths_nonexhaustive': ['task-specific training', 'objective-aligned guide/critic and decoding',
            'learned or adaptive mask with equal decoder overhead control', 'joint reassignment/admission and recourse',
            'true dynamic delivery lifecycle and qualified strong rolling control'],
    }

    result = {
        'schema': SCHEMA,
        'project_root': str(project),
        'repo_root': str(repo_root),
        'dimension_definitions': DIMENSION_DEFINITIONS,
        'overall_verdict_definitions': {
            'claim_excess_upheld': 'the documents assert the claim and at least one dimension fails',
            'supported_within_measured_scope': 'asserted and all four dimensions pass',
            'not_asserted_in_audited_documents': 'no audited document asserts the claim',
            'indeterminate_input_missing': 'a required input is missing, so support is indeterminate',
        },
        'claims': claims,
        'checks': {
            'confirmation_family_preregistered_cohorts': confirmation_check,
            'pruned_and_clean_status_missing_input': pruned_check,
            'cross_cohort_percentage_comparisons': {
                'verdict': 'flagged' if cross_cohort else 'none_found',
                'blocks_clean_exit': bool(cross_cohort),
                'definition': ("the programme's own adopted rule (WORK_STATUS delivery.n15 "
                               "rule_within_cohort_comparison): percentage differences may be "
                               "compared ONLY within the same cohort. A line is flagged when it "
                               "carries >=2 percentage values, a comparison cue, and either two "
                               "distinct named cohorts or a plural-cohort phrase; correction, "
                               "withdrawal and 'confirmation not run' lines are excluded"),
                'n_flags': len(cross_cohort),
                'flags': cross_cohort,
                'documents_scanned': list(NARRATIVE_DOCUMENTS),
                'evidence_paths': list(NARRATIVE_DOCUMENTS) + [f['document'] for f in cross_cohort],
                'caveat': ('this is a lexical scan over narrative documents; every flag is shown '
                           'with its line so it can be judged, and a flag is a rule violation '
                           'under the adopted within-cohort rule rather than a new scientific '
                           'claim excess'),
            },
            'documented_excess_locations': {
                'verdict': 'available' if locations_record['usable'] else 'missing_input',
                'blocks_clean_exit': not locations_record['usable'],
                'path': EXCESS_LOCATIONS,
                'documented_trigger_strings': sorted(t for t in documented_triggers if t),
                'detected_not_documented': undocumented_excesses,
                'all_detected_excesses_documented': not undocumented_excesses,
                'evidence_paths': [EXCESS_LOCATIONS],
                'note': ('presence of the disposition document is fail-closed; the specific '
                         'trigger-to-reason mapping it records is advisory because the file is '
                         'prose and its table is read lexically'),
            },
        },
        'additional_findings': sorted(set(additional_findings)),
        'required_documents': required_status,
        'missing_required_documents': missing_required,
        'input_records': {rel: {'exists': r['exists'], 'usable': r['usable'],
                                'error': r.get('error')}
                          for rel, r in sorted(input_records.items())},
        # ---- v1 payload retained ----
        'engineering_headline_consistency': {'n_checks': len(engineering),
            'n_failures': sum(e['pass'] is not True for e in engineering), 'checks': engineering,
            'unaudited_executed_nodes': unaudited, 'supports_scientific_sufficiency': False},
        'learning_reference_intervals': learned,
        'program_confirmation_protocols': protocol_support,
        'scientific_support': scientific_support,
        'read_only_input_sha256': hashes,
        'limitations': [
            'No new numerical experiment or original audit script executed.',
            'Headline values and CIs were read, not reconstructed here from raw trajectories.',
            'Literal registered_before_data flags do not prove registration time or immutable pre-data closure.',
            'Eligibility is a bounded filename inventory only; content, coverage, exact identity and replay remain unverified.',
            'The cross-cohort and universal-wording scans are lexical over narrative documents and '
            'show every hit with its line for judgement.',
            'Even four passing dimensions cannot substitute for independent executed-cohort replay '
            'and main/attribution tests.',
            'A missing input is always reported as a failure and never as a clean result.',
        ],
    }
    return result


def exit_code_for(result):
    if result.get('audit_error'):
        return 2
    blocking = [c['claim_id'] for c in result['claims']
                if c['overall_verdict'] not in OVERALL_NON_BLOCKING]
    blocking_checks = [name for name, check in result['checks'].items()
                       if check.get('blocks_clean_exit')]
    return 2 if (blocking or result['scientific_support']['claim_excesses'] or
                 result['additional_findings'] or
                 result['engineering_headline_consistency']['n_failures'] or
                 blocking_checks or result['missing_required_documents']) else 0


# --------------------------------------------------------------------------
# Self-test fixtures
# --------------------------------------------------------------------------
def fixture_files(authentic_program_bytes):
    """A synthetic project that exercises every new check end to end."""
    ids_dev = list(range(32))
    ids_conf = list(range(1000, 1032))
    ids_third = list(range(2000, 2032))
    files = {
        'results/research_20261010/PROGRAM.json': authentic_program_bytes.decode('utf-8'),
        'results/research_20261010/PROGRAM_CLOSEOUT.md': (
            '# 合成干净结题（自检夹具，非研究结果）\n\n'
            '本文件是 support audit v2 自检使用的合成文档。四项支持维度分别为数值一致性、\n'
            '原始数据可得性、统计支持与泛化范围。本轮不主张任何顶层正向结论，驱动脚本以\n'
            '机器可读标志记录其范围。\n'),
        'results/research_20261010/ROUND_20261010_GUIDE_DOSE.md': (
            '# 合成轮次记录（自检夹具，非研究结果）\n\n'
            '本文件仅用于 support audit v2 的自检。它记录一个已注册对照在单一开发队列上的\n'
            '描述性区间，并且不把它外推。\n'),
        'results/research_20261010/WORK_STATUS.json': json.dumps({
            'schema': 'synthetic-selftest-fixture-v1',
            'current_authoritative': {
                'stable_online_gain_established': False,
                'MaskCO_attribution_established': False,
                'pickup_impossibility_proven': False,
                'learning_side_closed': False,
                'equivalence_established': False}}, ensure_ascii=False, indent=2),
        'results/research_20261010/pickup/.arbor/tree.json': json.dumps({'nodes': {
            'n0': {'status': 'root', 'metadata': {}},
            'n3': {'status': 'pruned', 'metadata': {
                'branch_ref': 'pruned_by_census_synthetic_gate',
                'dev_score': None, 'test_score': None,
                'result': 'SYNTHETIC prune record: pruned on the strength of its own screening gate.'}},
            'n6': {'status': 'pruned', 'metadata': {
                'branch_ref': 'synthetic_residual_census', 'dev_score': -18.1, 'test_score': None,
                'result': 'SYNTHETIC prune record: the residual census refuted this node.'}},
            'n7': {'status': 'pruned', 'metadata': {
                'branch_ref': 'synthetic_rejection_attribution', 'dev_score': 92.71, 'test_score': None,
                'result': 'SYNTHETIC prune record: the attribution panel refuted this node.'}},
            'n4': {'status': 'executed', 'metadata': {'dev_score': 0.997}}}}, ensure_ascii=False, indent=2),
        'results/research_20261010/delivery/.arbor/tree.json': json.dumps({'nodes': {
            'n0': {'status': 'root', 'metadata': {}},
            'n4': {'status': 'pruned', 'metadata': {
                'branch_ref': 'synthetic_dose_gate', 'prune_reason': 'SYNTHETIC prune reason',
                'dev_score': -1.874, 'test_score': None, 'result': 'SYNTHETIC pruned, no confirmation run.'}},
            'n7': {'status': 'merged', 'metadata': {
                'branch_ref': 'synthetic_confirm', 'dev_score': 2.09, 'test_score': 2.09,
                'result': 'SYNTHETIC merge gate record.'}}}}, ensure_ascii=False, indent=2),
        'results/research_20261010/reproduction/.arbor/tree.json': json.dumps(
            {'nodes': {'n0': {'status': 'root', 'metadata': {}}}}, ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_any50_confirm_v1/result.json': json.dumps({
            'protocol': {'version': 'SYNTH-ANY50-CONFIRM-v1', 'dataset_seed': 2026101141,
                         'instance_ids': ids_third, 'preregistered_primary': True,
                         'registered_before_data': True,
                         'cohort_disjoint_from': ['SYNTH-dev', 'SYNTH-confirmation']},
            'primary_ANY50_over_LNS': {'n_instances': 32, 'control': 'LNS', 'other': 'ANY50',
                                       'mean_relative_gain': 0.020854778158967097,
                                       'descriptive_95_CI': [0.0019622408950310693,
                                                             0.03912516272808401],
                                       'instances_other_better': 21, 'instances_other_worse': 11,
                                       'instances_tied': 0},
            'gate_passed': True,
            'gate_rule': 'PASS only if mean relative gain>=1% and paired 95% CI lower>0',
            'online_gain_established': False, 'maskco_attributable_gain_established': False,
            'stable_gain_established': False}, ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_any50_confirm_v1/PROTOCOL.json': json.dumps(
            {'version': 'SYNTH-ANY50-CONFIRM-v1', 'dataset_seed': 2026101141,
             'instance_ids': ids_third, 'preregistered_primary': True,
             'registered_before_data': True}, ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_any50_confirm_v1/identity_start.json': json.dumps(
            {'synthetic': 'source/weight identity snapshot'}, indent=2),
        'results/research_20261010/delivery_any50_confirm_v1/progress.jsonl':
            '\n'.join(json.dumps({'instance_id': i, 'initial_cost': 300.0,
                                  'arms': {'ANY50': {'final_cost': 70.0 + i * 0.01}}})
                      for i in ids_third) + '\n',
        'results/research_20261010/delivery_anytime_v1/PROTOCOL.json': json.dumps(
            {'version': 'SYNTH-ANYTIME-v1', 'dataset_seed_dev': 2026101002,
             'dev_instance_ids': ids_dev, 'dataset_seed_confirmation': 2026101013,
             'confirmation_instance_ids': ids_conf, 'registered_before_data': True},
            ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_anytime_v1/posthoc_any50_comparison.json': json.dumps(
            {'analysis': 'synthetic', 'not_a_preregistered_test': False,
             'preregistered_primary': True, 'registered_before_data': True},
            ensure_ascii=False, indent=2),
        'results/research_20261010/pickup_tail_census_v1/result.json': json.dumps({
            'protocol': {'version': 'SYNTH-P-CENSUS-v1',
                         'not_a_verdict_on': 'A-v1 utility headroom (energy and quality not re-simulated)'},
            'n_instances': 16,
            'distance_headroom': {'mean_relative_gain': 0.009970540505734128,
                                  'descriptive_95_CI': [0.007141753489970731, 0.013383910612499793]},
            'online_gain_established': False, 'stable_gain_established': False},
            ensure_ascii=False, indent=2),
        'results/research_20261010/pickup_clairvoyant_gap_v1/result.json': json.dumps({
            'protocol': {'full_information_reference': 'synthetic greedy clairvoyant'},
            'served_count_comparison': {'mean_relative_gain': -0.0817,
                                        'descriptive_95_CI': [-0.1055, -0.0566]}},
            ensure_ascii=False, indent=2),
        'results/research_20261010/pickup_rejection_attribution_v1/result.json': json.dumps(
            {'shares': {'threshold': 0.9271, 'no_feasible_route': 0.0,
                        'budget_certify_failed': 0.0729}}, ensure_ascii=False, indent=2),
        'results/research_20261010/pickup_residual_census_v1/result.json': json.dumps(
            {'verdict': 'residual_is_strategic_not_indivisible'}, ensure_ascii=False, indent=2),
        'results/research_20261010/pickup_budget_headroom_v1/result.json': json.dumps(
            {'ceiling_if_per_order_cost_held': {'relative_service_gain': 0.1027}},
            ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_decomp_v1/result.json': json.dumps({
            'summary': {'duty_time_difference_h': {'mean': 0.043541592340333946,
                                                   'descriptive_95_CI': [-0.08094848768633144,
                                                                         0.14719624624745034]}},
            'n_instances': 12}, ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_dose_v1/result.json': json.dumps(
            {'dev_selected': None, 'online_gain_established': False}, ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_recheck_v1/result.json': json.dumps(
            {'relative_distance_advantage_ANY50_over_LNS': 0.0113}, ensure_ascii=False, indent=2),
        'results/research_20261010/delivery_dynamic_suffix_v1/result.json': json.dumps(
            {'n_instances': 8, 'mean_relative_distance_gain': 0.0,
             'protocol': {'version': 'SYNTH-D-DYN-v1'}}, ensure_ascii=False, indent=2),
        'results/research_20261010/INDEPENDENT_EQ8_PROGRESS_ADJUDICATION.json': json.dumps(
            {'comparisons': {'P_over_H': {'mean_relative_gain': -0.09,
                                          'descriptive_95_CI': [-0.15, -0.03]}}},
            ensure_ascii=False, indent=2),
        'scripts/research_20261010/program_evidence_audit.py': (
            'CLAIMS = [\n'
            "    ('delivery', 'n7', 'research_20261010/delivery_any50_confirm_v1/result.json',\n"
            "     'primary_ANY50_over_LNS.mean_relative_gain', 'synthetic ANY50 over LNS', 1),\n"
            ']\n'
            'EXPECTED_FLAGS = [\n'
            "    ('delivery', 'n7', 'research_20261010/delivery_any50_confirm_v1/result.json',\n"
            "     'online_gain_established', False, 'synthetic scope flag'),\n"
            ']\n'
            'SCOPE_FLAGS = [\n'
            "    ('delivery', 'n7', 'research_20261010/delivery_any50_confirm_v1/result.json',\n"
            "     'online_gain_established', False),\n"
            ']\n'),
        'scripts/research_20261010/learning_attribution_adjudication.py': (
            'EVIDENCE = [\n'
            "    ('SYNTH learned reference',\n"
            "     'research_20261010/INDEPENDENT_EQ8_PROGRESS_ADJUDICATION.json',\n"
            "     'comparisons.P_over_H.mean_relative_gain',\n"
            "     'synthetic harmful direction', 'harmful'),\n"
            ']\n'),
        'scripts/research_20261010/delivery_dynamic_suffix.py': (
            "def build_suffix(cargo, obs, mode, node):\n"
            "    return engine.plan(vid, build_suffix(cargo, obs, 'reorder', node) + [0])\n"),
        'scripts/research_20261010/delivery_anytime.py': '# synthetic fixture source\n',
        'scripts/research_20261010/delivery_any50_confirm.py': '# synthetic fixture source\n',
        'scripts/research_20261010/delivery_anytime_posthoc.py': (
            "RESULT = {\n"
            "    'analysis': 'posthoc_paired_comparison_of_preregistered_arms',\n"
            "    'not_a_preregistered_test': True,\n"
            "}\n"),
        EXCESS_LOCATIONS: (
            '# 合成 excess 定位表（自检夹具）\n\n'
            '| # | 检查串 | 位置 | 原表述的问题 | 当前正文是否已覆盖 |\n'
            '|---|---|---|---|---|\n'),
        'results/m0_scale/_bverify_pinned/per_state.json': json.dumps({'synthetic': 'per-state rows'}, indent=2),
        'results/m0_scale/fsq_clean_train/per_state.json': json.dumps({'synthetic': 'per-state rows'}, indent=2),
        'negative_result_paper/results/09_unified_v1/step5_heldout_s42.summary.json': json.dumps(
            {'n': 16, 'rows': [{'inst': i} for i in range(16)]}, indent=2),
        'negative_result_paper/results/10_confirm_c1/c1_step5_s42.summary.json': json.dumps(
            {'n': 16, 'rows': [{'inst': i} for i in range(16)]}, indent=2),
        'results/three_line_corrections_20261009/FINAL_PROTECTED_ACCEPTANCE_20261010_v2.json':
            json.dumps({'negative_static_checks_pass': True, 'static_qualification_pass': True},
                       indent=2),
        '03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/RECOVERY_MANIFEST.json': json.dumps(
            {'synthetic': 'recovered dependency identities'}, indent=2),
    }
    return files


def materialize(root, files):
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # bytes, so no platform newline translation can perturb a pinned hash
        path.write_bytes(body.encode('utf-8') if isinstance(body, str) else body)


def build_fixture(base, name, authentic_program_bytes, mutate=None):
    root = Path(base) / name
    project = root / 'proj'
    files = fixture_files(authentic_program_bytes)
    if mutate:
        files = mutate(dict(files))
    materialize(project, files)
    # the round-20 exact-utility witness lives at repository level, outside the project
    witness = root / 'research' / 'pickup_20round_audit' / 'known_batch_v1' / 'BATCH_SUMMARY.json'
    witness.parent.mkdir(parents=True, exist_ok=True)
    witness.write_text(json.dumps({'mean_delta_J': 0.4715, 'n_states': 16}, indent=2),
                       encoding='utf-8')
    return project


def positive_mutation(files):
    """Inject one unsupported universal claim; the auditor must exit 2 and name it."""
    files['results/research_20261010/PROGRAM_CLOSEOUT.md'] = (
        '# 合成正例结题（自检夹具）\n\n'
        '本夹具故意写入一个不被证据支持的顶层主张，用于证明审计器不是空操作：\n'
        '训练一个交付专属模型是唯一未测路径，因此学习侧已经关闭。\n')
    return files


def missing_input_mutation(files):
    del files['results/research_20261010/ROUND_20261010_GUIDE_DOSE.md']
    return files


def run_audit(project, output, repo_root=None, current_docs=()):
    try:
        result = audit(project, repo_root)
        reviews = []
        for relative in current_docs:
            record, body = read_text_input(Path(project).resolve(), relative, result['read_only_input_sha256'])
            hits = scan_universal_wording(relative, body) if body is not None else []
            reviews.append({'path': relative, 'usable': record['usable'], 'current_assertion_hits': hits, 'scope_only_lexical_review': True, 'scientific_qualification_established': False})
            if body is None:
                result['additional_findings'].append('current_document_missing_or_unreadable:' + relative)
            elif hits:
                result['additional_findings'].append('current_document_assertion_requires_review:' + relative)
        result['current_documents_context_review'] = reviews
        result['review_base_source_sha256'] = BASE_SOURCE_SHA256
    except (EvidenceError, KeyError, TypeError, ValueError, OSError) as error:
        result = {'schema': SCHEMA, 'audit_error': '%s: %s' % (type(error).__name__, error),
                  'claims': [], 'checks': {}, 'additional_findings': ['audit_error'],
                  'engineering_headline_consistency': {'n_checks': 0, 'n_failures': 1},
                  'scientific_support': {'claim_excesses': []},
                  'missing_required_documents': []}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result, exit_code_for(result)


def run_selftest(fixtures_dir=None, authentic_project=None):
    extension_root = Path(authentic_project).resolve() if authentic_project else Path(__file__).resolve().parents[2]
    authentic = (extension_root / 'results/research_20261010/PROGRAM.json').read_bytes()
    keep = fixtures_dir is not None
    base = Path(fixtures_dir) if keep else tempfile.mkdtemp(prefix='support_audit_selftest_')
    Path(base).mkdir(parents=True, exist_ok=True)
    failures = []
    summary = {}
    try:
        cases = (
            ('clean', None,
             'exit 0 and every claim not asserted'),
            ('unsupported_claim', positive_mutation,
             'exit 2 and the injected claim is named'),
            ('missing_input', missing_input_mutation,
             'exit 2 because a searched document is missing'),
        )
        for name, mutate, expectation in cases:
            project = build_fixture(base, name, authentic, mutate)
            output = Path(base) / (name + '.report.json')
            result, code = run_audit(project, output)
            record = {'fixture': name, 'expectation': expectation,
                      'exit_code': code,
                      'claim_overall_verdicts': {c['claim_id']: c['overall_verdict']
                                                 for c in result.get('claims', [])},
                      'claim_excesses': result.get('scientific_support', {}).get('claim_excesses'),
                      'additional_findings': result.get('additional_findings'),
                      'report': str(output)}
            if name == 'clean':
                ok = code == 0 and all(v == 'not_asserted_in_audited_documents'
                                       for v in record['claim_overall_verdicts'].values())
                ok = ok and bool(record['claim_overall_verdicts'])
            elif name == 'unsupported_claim':
                ok = (code == 2 and
                      'untested_path_list_not_exhaustive_no_global_exclusion_proof'
                      in (record['claim_excesses'] or []) and
                      record['claim_overall_verdicts'].get('learning_side_closed') ==
                      'claim_excess_upheld')
            else:
                ok = (code == 2 and
                      any('required_input_missing_or_unreadable:'
                          'results/research_20261010/ROUND_20261010_GUIDE_DOSE.md' == finding
                          or finding.startswith('searched_document_missing:')
                          for finding in (record['additional_findings'] or [])))
            record['pass'] = bool(ok)
            if not ok:
                failures.append(name)
            summary[name] = record
            print('%-18s exit=%d %s' % (name, code, 'PASS' if ok else 'FAIL'))
    finally:
        if not keep:
            shutil.rmtree(base, ignore_errors=True)
    print(json.dumps({'selftest_pass': not failures, 'fixtures': summary},
                     ensure_ascii=False, indent=2))
    return 0 if not failures else 1


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project')
    parser.add_argument('--output')
    parser.add_argument('--current-doc', action='append', default=[], help='additional project-relative current document; reviewed separately from historical artifacts')
    parser.add_argument('--selftest', action='store_true',
                        help='run the synthetic positive/clean/missing-input fixtures')
    parser.add_argument('--fixtures-dir',
                        help='keep the self-test fixtures in this directory instead of a temp dir')
    parser.add_argument('--repo-root',
                        help='repository root used to resolve repository-level artifacts such as '
                             'research/pickup_20round_audit; defaults to the project or its parent')
    parser.add_argument('--allow-tool-output', action='store_true',
                        help="permit --output under this tool's own reports/ directory; the "
                             'default guard (output must be outside the audited project) is '
                             'unchanged without this flag')
    args = parser.parse_args()

    if args.selftest:
        return run_selftest(args.fixtures_dir, args.project)

    require(args.project, '--project is required unless --selftest is used')
    require(args.output, '--output is required unless --selftest is used')
    project = Path(args.project).resolve()
    output = Path(args.output).resolve()
    if not args.allow_tool_output:
        require(not output.is_relative_to(project),
                'output must be outside authoritative input')
    else:
        require(output.is_relative_to(TOOL_REPORT_DIR),
                '--allow-tool-output only permits writes under ' + str(TOOL_REPORT_DIR))
    result, code = run_audit(project, output, args.repo_root, args.current_doc)
    print(json.dumps({
        'schema': result.get('schema'),
        'headline_checks': result.get('engineering_headline_consistency', {}).get('n_checks'),
        'headline_failures': result.get('engineering_headline_consistency', {}).get('n_failures'),
        'claim_overall_verdicts': result.get('scientific_support', {}).get('claim_overall_verdicts'),
        'dimension_failure_counts': result.get('scientific_support', {}).get('dimension_failure_counts'),
        'claim_excesses': result.get('scientific_support', {}).get('claim_excesses'),
        'additional_findings': result.get('additional_findings'),
        'checks': {name: check.get('verdict') for name, check in (result.get('checks') or {}).items()},
        'report': str(output),
    }, ensure_ascii=False, indent=2))
    # Nonzero makes inadequate scientific sufficiency visible, independently of numerical consistency.
    return code


if __name__ == '__main__':
    raise SystemExit(main())
