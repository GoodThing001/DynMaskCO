"""Start a new bounded research batch without changing the twenty-round record."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
OLD = E / 'results/research_20261010'
NEW = E / 'results/research_20261010_continuation_v2'
TREE = Path('C:/Users/mmc/.agents/skills/arbor/scripts/tree.py')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    resuming_archive_only = NEW.exists()
    if resuming_archive_only and (NEW / 'PROGRAM.json').exists():
        raise RuntimeError('Registration already complete; refuse repetition')
    NEW.mkdir(exist_ok=resuming_archive_only)
    archive = NEW / 'pre_correction_archive'
    archive.mkdir(exist_ok=resuming_archive_only)
    inventory = {}
    for p in (OLD / 'PROGRAM_CLOSEOUT.md', OLD / 'ROUND_20261010_GUIDE_DOSE.md',
              OLD / 'WORK_STATUS.json', OLD / 'PROGRAM.json'):
        dest = archive / p.name
        if dest.exists():
            if dest.read_bytes() != p.read_bytes():
                raise RuntimeError('Archived source changed during registration')
        else:
            with dest.open('xb') as f:
                f.write(p.read_bytes())
        inventory[p.relative_to(E).as_posix()] = sha(p)
    for line in ('pickup', 'delivery', 'reproduction'):
        for name in ('tree.json', 'run.json'):
            p = OLD / line / '.arbor' / name
            dest = archive / (line + '_' + name)
            if dest.exists():
                if dest.read_bytes() != p.read_bytes():
                    raise RuntimeError('Archived tree changed during registration')
            else:
                with dest.open('xb') as f:
                    f.write(p.read_bytes())
            inventory[p.relative_to(E).as_posix()] = sha(p)
    if not (NEW / 'PRIOR_RECORD_SHA256.json').exists():
        save(NEW / 'PRIOR_RECORD_SHA256.json', inventory)
    else:
        if json.loads((NEW / 'PRIOR_RECORD_SHA256.json').read_text(encoding='utf-8')) != inventory:
            raise RuntimeError('Archive inventory mismatch')
        save(NEW / 'REGISTRATION_RECOVERY.json', {'reason': 'Local default GBK could not read UTF8 parent PROGRAM; no new data or program written before correction.', 'scientific_parameters_changed': False})
    spec = {
        'version': 'three-line-continuation-20261010-v2',
        'registered_at': datetime.datetime.now().astimezone().isoformat(),
        'authorization': 'User requested audit/correction of the additional twenty rounds and continued three-line improvement.',
        'parent_program_sha256': sha(OLD / 'PROGRAM.json'),
        'old_rounds_retained': 20,
        'new_bounded_cycles_per_line': 12,
        'objective': 'Stable attributable MaskCO online gain in pickup and delivery; truthful independently reproducible reporting in the negative-result line.',
        'validation': json.loads((OLD / 'PROGRAM.json').read_text(encoding='utf-8'))['validation'],
        'correction_priors': [
            'ANY50 descriptive snapshot advantage is a nonlearned baseline clue; stable online and learning attribution remain false.',
            'No formal impossibility or total utility upper bound follows from distance-only census, baseline rejection reasons or greedy clairvoyant counts.',
            'Same wall-time ceilings are a valid deployment comparison; matching opportunities and consumption answers a separate attribution question.',
            'Headline agreement and valid tree bookkeeping do not establish complete source/trajectory identity or scientific claim validity.',
            'Each new attempt must persist full instance, routes, trajectory, protocol and actual source/weight identity; old missing raw artifacts cannot be regenerated with an anytime rerun and called original.'
        ],
        'first_hypotheses': {
            'delivery': 'At fixed eight destroy-repair opportunities and matched actual decoder calls, pretrained destroy guidance beats pretrained-decode-discard uniform masks and random-model guidance.',
            'pickup': 'Exact known-order small-subproblem replay can identify a feasible same-budget total-utility improvement that the prior distance-only or greedy screens could not exclude.',
            'reproduction': 'A support gate can distinguish numeric consistency from unsupported impossibility, equivalence, attribution and stable-online claims.'
        },
        'execution': {
            'never_modify': ['MASKCO_code', 'original comparison repositories', 'old scientific results/protocols/weights', 'team processes'],
            'old_pickup_final_seeds_forbidden': [20260927, 20260928],
            'no_success_conditioned_resampling': True,
            'server_shared_md_upload': 'still not authorized; do not retry',
            'publication': False,
            'new_scientific_protocol_before_data': True,
            'no_stable_positive_guarantee': True,
            'CPU_only_until_fresh_GPU_availability_check': True
        }
    }
    save(NEW / 'PROGRAM.json', spec)
    (NEW / 'PROGRAM.sha256').write_text(sha(NEW / 'PROGRAM.json') + '\n', encoding='ascii')
    for line in ('pickup', 'delivery', 'reproduction'):
        run = NEW / line
        run.mkdir()
        objective = spec['first_hypotheses'][line]
        cmd = ['python', str(TREE), '--run-dir', str(run), 'init', '--objective', objective,
               '--dev-eval', 'registered full-artifact development replay; no stable claim',
               '--test-eval', 'frozen candidate, fresh independent cohorts and parent PROGRAM qualification',
               '--material', str(E), '--metric-direction', 'max', '--branching', '3',
               '--max-depth', '3', '--budget', '12']
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(['python', str(TREE), '--run-dir', str(run), 'cycle'], check=True, capture_output=True)
        p = subprocess.run(['python', str(TREE), '--run-dir', str(run), 'add-node', '--parent', 'n0',
                            '--hypothesis', objective], check=True, capture_output=True, text=True)
        save(run / 'INITIAL_NODE_REGISTRATION.json', {'stdout': p.stdout, 'hypothesis': objective})
    print(json.dumps({'directory': str(NEW), 'program_sha256': sha(NEW / 'PROGRAM.json'),
                      'historical_records_archived': len(inventory), 'old_budget_not_reset': True}))

if __name__ == '__main__':
    main()
