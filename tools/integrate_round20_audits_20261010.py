"""Copy only new, separately reviewed audit artifacts; preserve old research bytes."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
WORK = Path('C:/Users/mmc/.codex/worktrees/ba63/MASKCO-Main/.arbor')
NEW = E / 'results/research_20261010_continuation_v2'

def copy(source, target, inventory):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != source.read_bytes():
            raise RuntimeError('Refuse overwrite: ' + str(target))
    else:
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    inventory.append({'path': str(target.relative_to(ROOT)),
                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})

def main():
    inventory=[]
    source = WORK / 'pickup/research/pickup_20round_audit'
    for p in source.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:
            copy(p, ROOT / 'research/pickup_20round_audit' / p.relative_to(source), inventory)
    base=WORK / 'delivery/C-VRP_Cold-chainVehicleRoutingProblem'
    for name in ('DELIVERY_20_ROUND_AUDIT.md', 'audit_delivery_20_rounds.py'):
        copy(base/'scripts/research_20261010'/name, E/'scripts/research_20261010'/name, inventory)
    p=base/'results/delivery_20_round_audit_20261010_v2/RECOMPUTATION.json'
    copy(p,E/'results/delivery_20_round_audit_20261010_v2/RECOMPUTATION.json',inventory)
    # Journal revision is an explicit future artifact, never the current frozen runner.
    for name in ('delivery_destroy_eq8.py','test_delivery_destroy_eq8.py'):
        copy(base/'scripts/research_20261010'/name, NEW/'delivery_future_journal_revision'/name,inventory)
    out=NEW/'INTEGRATED_AUDIT_ARTIFACTS.json'
    if out.exists(): raise RuntimeError('Already integrated')
    out.write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps({'new_artifacts':len(inventory),'old_experiment_overwritten':False}))

if __name__=='__main__':main()
