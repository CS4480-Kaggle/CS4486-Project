from pathlib import Path
import sys
import unittest

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "ensemble"
sys.path.insert(0, str(SRC_DIR))

from blend_predictions import blend_probabilities, search_blend_weight


class TestBlendPredictions(unittest.TestCase):
    def test_blend_probabilities_uses_weight_for_first_model(self):
        first = np.array([0.2, 0.8])
        second = np.array([0.4, 0.6])
        actual = blend_probabilities(first, second, first_weight=0.25)
        np.testing.assert_allclose(actual, np.array([0.35, 0.65]))

    def test_weight_search_can_select_endpoint(self):
        targets = np.array([0, 1])
        first = np.array([0.01, 0.99])
        second = np.array([0.4, 0.6])
        result = search_blend_weight(targets, first, second, step=0.1)
        self.assertEqual(result.iloc[0]["first_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
