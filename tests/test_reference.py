import itertools
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from reference import count_bruteforce, count_forward, normalize


class ReferenceTests(unittest.TestCase):
    def assert_counts(self, edges, expected, vertices=()):
        self.assertEqual(count_bruteforce(edges, vertices), expected)
        self.assertEqual(count_forward(edges, vertices), expected)

    def test_known_graphs(self):
        self.assert_counts([], 0, range(4))
        self.assert_counts([(0, 1)], 0)
        self.assert_counts([(0, 1), (1, 2)], 0)
        self.assert_counts(list(itertools.combinations(range(3), 2)), 1)
        self.assert_counts(list(itertools.combinations(range(5), 2)), 10)
        self.assert_counts([(0, i) for i in range(1, 9)], 0)
        self.assert_counts([(u, v) for u in range(4) for v in range(4, 9)], 0)

    def test_disconnected(self):
        edges = list(itertools.combinations(range(3), 2))
        edges += list(itertools.combinations(range(3, 7), 2))
        self.assert_counts(edges, 5, range(9))

    def test_cleaning(self):
        edges = [(10, 20), (20, 10), (10, 20), (20, 30), (30, 10), (30, 30)]
        self.assert_counts(edges, 1, [99])
        self.assertEqual(normalize(edges, [99])[0], [10, 20, 30, 99])

    def test_random_and_relabeling(self):
        for seed in range(30):
            rng = random.Random(seed)
            edges = [e for e in itertools.combinations(range(20), 2)
                     if rng.random() < (0.05, 0.2, 0.6)[seed % 3]]
            expected = count_bruteforce(edges, range(20))
            self.assertEqual(count_forward(edges, range(20)), expected)
            labels = rng.sample(range(100, 1000), 20)
            remapped = [(labels[u], labels[v]) for u, v in edges]
            rng.shuffle(remapped)
            self.assertEqual(count_forward(remapped, labels), expected)


if __name__ == '__main__':
    unittest.main()
