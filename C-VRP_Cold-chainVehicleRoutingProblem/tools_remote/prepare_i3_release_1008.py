"""Stage only new I3 code; never upload local shared SAA over server experiments."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'scripts/training/i3_pool_lineage.py', 'scripts/training/train_i3_scenario.py',
    'scripts/training/collect_i3_teachers.py', 'scripts/training/merge_i3_teachers.py',
    'scripts/training/i3_training_engine.py', 'scripts/training/i3_optimizer.py',
    'scripts/training/i3_decision_objective.py', 'scripts/training/i3_actor_draws.py',
    'scripts/training/i3_counterfactual_teacher.py', 'scripts/training/i3_teacher_archive.py',
    'scripts/evaluation/run_i3_online_compare.py', 'scripts/evaluation/run_a1_step3_compare.py',
    'scripts/evaluation/diag_i3_online_adjudicate.py',
    'scripts/tests/test_i3_pool_lineage.py', 'scripts/tests/test_i3_collection.py',
    'scripts/tests/test_i3_online_adapter.py', 'scripts/tests/test_train_i3_runner.py',
    'scripts/tests/test_i3_counterfactual_teacher.py',
    'scripts/tests/test_i3_online_qualification.py',
    'scripts/tests/validate_i3_registered_release.py',
]


if __name__ == '__main__':
    out = ROOT / 'tools_remote/i3_release_1008_v1'
    out.mkdir(exist_ok=False)
    manifest = {}
    for rel in FILES:
        source, target = ROOT / rel, out / rel
        if not target.resolve().is_relative_to(out.resolve()):
            raise ValueError('Staging target outside release directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / 'release.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('STAGED', len(FILES))
