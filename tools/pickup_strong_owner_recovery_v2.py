"""Load the exact previously qualified SAA source before owner dependencies.

This engineering-only dependency overlay does not alter the shared module.
"""
import hashlib
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'research/review_3080_20261010/goal_round_20261013_r9_strong_owner_resource/frozen_saa/scenario_saa.py'
EXPECTED = '4e47b99f14be03351506ccccf92f1d414d12b063020805aca07ba6ef56874188'

def load():
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('frozen scenario_saa identity mismatch')
    if 'scenario_saa' in sys.modules:
        raise ValueError('scenario_saa imported before frozen identity gate')
    sys.path.insert(0, str(FROZEN.parent))
    mod = importlib.import_module('scenario_saa')
    if Path(mod.__file__).resolve() != FROZEN.resolve():
        raise ValueError('scenario_saa resolved outside frozen input')
    return mod
