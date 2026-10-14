import unittest
import numpy as np
from training_engineering import adjacency, retained_adjacency, validate_routes, validate_splits


class EngineeringTests(unittest.TestCase):
    def test_immediate_delta_is_not_completed_route_objective(self):
        from min_delta_counterexample import counterexample
        fixture = counterexample()
        self.assertLess(fixture['better_completed']['final_distance'], fixture['greedy_immediate']['final_distance'])

    def test_complete_routes(self):
        validate_routes([[1, 2], [3]], [0, 1, 1, 1], 2)
        for routes in ([[1, 1], [3]], [[1, 2]], [[True, 2], [3]], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                validate_routes(routes, [0, 1, 1, 1], 2)

    def test_single_customer_depot_multiplicity(self):
        adj = adjacency([[1]], 2)
        self.assertEqual(adj[0, 1], 2)
        self.assertTrue(np.array_equal(adj, adj.T))
        self.assertTrue(np.all(np.diag(adj) == 0))

    def test_observation_preserves_subset(self):
        routes = [[1, 2], [3]]
        target = adjacency(routes, 4)
        for keep in (0., .5, 1.):
            obs = retained_adjacency(routes, 4, np.random.default_rng(3), keep)
            self.assertTrue(np.all(obs <= target))
            self.assertTrue(np.array_equal(obs, obs.T))
        self.assertTrue(np.array_equal(retained_adjacency(routes, 4, np.random.default_rng(3), 1.), target))

    def test_splits_block_identity_and_content_overlap(self):
        names = ['teacher_train', 'teacher_diagnostic', 'development', 'confirmation']
        splits = {name: [{'instance_id': i, 'data_sha256': ('%064x' % i)}] for i, name in enumerate(names)}
        validate_splits(splits)
        splits['confirmation'][0]['data_sha256'] = splits['development'][0]['data_sha256']
        with self.assertRaises(ValueError):
            validate_splits(splits)
        splits['confirmation'][0]['data_sha256'] = 'a' * 64
        splits['confirmation'][0]['instance_id'] = 0
        with self.assertRaises(ValueError):
            validate_splits(splits)


if __name__ == '__main__':
    unittest.main()
