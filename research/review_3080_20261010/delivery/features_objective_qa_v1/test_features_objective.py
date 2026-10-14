import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_objective import extend_projection, pair_label


class ObjectiveTests(unittest.TestCase):
    def test_zero_extension_preserves_original(self):
        k = np.arange(1536, dtype=np.float32).reshape(3, 512) / 1000
        x = np.arange(40, dtype=np.float32).reshape(4, 10)
        expanded = extend_projection(k)
        self.assertTrue(np.array_equal(x @ expanded, x[:, :3] @ k))
        self.assertEqual(expanded.shape, (10, 512))

    def test_wrong_actual_projection_rejected(self):
        with self.assertRaises(ValueError):
            extend_projection(np.zeros((3, 16)))

    def test_pair_both_signs_and_tie(self):
        a = {'hard': True, 'complete': True, 'cost': 10.}
        b = dict(a, cost=8.)
        self.assertEqual(pair_label(a, b), 1)
        self.assertEqual(pair_label(b, a), -1)
        self.assertEqual(pair_label(a, a), 0)

    def test_failure_not_positive_label(self):
        a = {'hard': True, 'complete': True, 'cost': 10.}
        for value in (False, 1):
            with self.assertRaises(ValueError):
                pair_label(a, dict(a, hard=value))


if __name__ == '__main__':
    unittest.main()
