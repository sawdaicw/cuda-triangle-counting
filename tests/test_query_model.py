"""Independent oracle check of the exploratory CPU query mapping, not CUDA."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'query_model', Path(__file__).resolve().parents[1] / 'scripts/explore_queries.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class QueryMappingChecks(unittest.TestCase):
    def test_graphs_against_independent_triplet_oracle(self):
        result = model.verify()
        self.assertEqual(result['graphs'], 204)
        self.assertEqual(result['batch_configurations'], 816)


if __name__ == '__main__':
    unittest.main()
