import unittest
from actual_full_step_qa import strict


class RegistrationTypes(unittest.TestCase):
    def test_actual_int_is_required(self):
        self.assertTrue(strict(1, 1))
        self.assertFalse(strict(True, 1))
        self.assertFalse(strict(1.0, 1))

    def test_scientific_scope_cannot_be_changed(self):
        self.assertTrue(strict(False, False))
        self.assertFalse(strict(0, False))
        self.assertFalse(strict(True, False))

    def test_optimizer_identity_is_exact(self):
        self.assertTrue(strict('sgd_without_momentum', 'sgd_without_momentum'))
        self.assertFalse(strict('adam', 'sgd_without_momentum'))


if __name__ == '__main__':
    unittest.main()
