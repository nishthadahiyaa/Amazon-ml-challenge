"""Checks for competition metric edge cases, independent of pipeline execution."""
import unittest
from analyze import entity_f

class MetricTests(unittest.TestCase):
    def test_singletons(self):
        self.assertEqual(entity_f(set(), set()), 1)
        self.assertEqual(entity_f(set(), {'wrong'}), 0)

    def test_missing_all_matches(self):
        self.assertEqual(entity_f({'a'}, set()), 0)

    def test_false_positive_penalty(self):
        self.assertAlmostEqual(entity_f({'a', 'b'}, {'a', 'b', 'c'}), 5/7)
        self.assertLess(entity_f({'a', 'b'}, {'a', 'b', 'c'}), entity_f({'a', 'b'}, {'a'}))

if __name__ == '__main__':
    unittest.main()
