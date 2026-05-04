import unittest

import numpy as np

from meteora import MeteoraSelector, statistical_elbow


class StaticEncoder:
    def __init__(self):
        self.vectors = {
            "assignment rationale": [1.0, 0.0],
            "termination rationale": [0.0, 1.0],
            "assignment chunk": [1.0, 0.0],
            "billing chunk": [0.2, 0.0],
            "termination chunk": [0.0, 1.0],
        }

    def encode(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


class SelectorTest(unittest.TestCase):
    def test_selector_runs_pair_pool_expand(self):
        selector = MeteoraSelector(StaticEncoder(), expansion_window=1)

        result = selector.select(
            chunks=["assignment chunk", "billing chunk", "termination chunk"],
            rationales=["assignment rationale", "termination rationale"],
        )

        self.assertEqual(result.details.pairing_indices, [0, 2])
        self.assertIn(1, result.details.expansion_indices)
        self.assertEqual(result.selected_indices, [0, 1, 2])

    def test_statistical_elbow_handles_short_lists(self):
        self.assertIsNone(statistical_elbow([]))
        self.assertEqual(statistical_elbow([0.9]), 0)
        self.assertEqual(statistical_elbow([0.9, 0.8]), 1)

    def test_empty_rationales_returns_empty_selection(self):
        selector = MeteoraSelector(StaticEncoder(), expansion_window=1)

        result = selector.select(chunks=["assignment chunk"], rationales=[])

        self.assertEqual(result.selected_indices, [])
        self.assertEqual(result.to_dict()["details"]["all_stages"], [])

    def test_pairing_allows_rationale_convergence(self):
        selector = MeteoraSelector(StaticEncoder(), expansion_window=0, enable_pooling=False)

        result = selector.select(
            chunks=["assignment chunk", "billing chunk"],
            rationales=["assignment rationale", "assignment rationale"],
        )

        self.assertEqual(result.details.pairing_indices, [0])
        self.assertEqual(result.details.rationale_contributions[0], [1, 2])


if __name__ == "__main__":
    unittest.main()
