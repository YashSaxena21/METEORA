import unittest

from meteora import precision_recall_f1


class EvaluationTest(unittest.TestCase):
    def test_precision_recall_f1(self):
        metrics = precision_recall_f1([1, 2, 3], [2, 4])

        self.assertAlmostEqual(metrics.precision, 1 / 3)
        self.assertAlmostEqual(metrics.recall, 1 / 2)
        self.assertAlmostEqual(metrics.f1, 0.4)
        self.assertTrue(metrics.correct_chunk_found)


if __name__ == "__main__":
    unittest.main()
