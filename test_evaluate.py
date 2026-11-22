import unittest

import numpy as np
from scipy.stats import binomtest

from evaluate import paired_auc_interval, wilson_interval


class EvaluationIntervalTests(unittest.TestCase):
    def test_wilson_matches_scipy_at_boundaries_and_observed_counts(self):
        for successes, trials in ((0, 40), (40, 40), (25, 40), (158, 160)):
            with self.subTest(successes=successes, trials=trials):
                expected = binomtest(successes, trials).proportion_ci(method="wilson")
                np.testing.assert_allclose(wilson_interval(successes, trials),
                                           [expected.low, expected.high], atol=1e-14)

    def test_wilson_rejects_invalid_counts(self):
        for successes, trials in ((0, 0), (-1, 10), (11, 10), (1.5, 10)):
            with self.subTest(successes=successes, trials=trials):
                with self.assertRaises(ValueError):
                    wilson_interval(successes, trials)

    def test_identical_models_have_zero_difference_in_every_paired_resample(self):
        # Independent draws for the two models would produce a nonzero interval.
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.6, 0.9, 0.3, 0.4, 0.8]
        result = paired_auc_interval(labels, scores, scores, n_resamples=100)
        self.assertEqual(result["difference"], 0)
        self.assertEqual(result["confidence_interval"], [0, 0])

    def test_reversing_models_reverses_the_paired_interval(self):
        labels = [0, 0, 1, 1]
        a, b = [0.1, 0.9, 0.2, 0.8], [0.1, 0.4, 0.6, 0.9]
        forward = paired_auc_interval(labels, a, b, n_resamples=100)
        backward = paired_auc_interval(labels, b, a, n_resamples=100)
        self.assertEqual(forward["difference"], -backward["difference"])
        np.testing.assert_allclose(forward["confidence_interval"],
                                   -np.array(backward["confidence_interval"])[::-1])

    def test_bootstrap_is_reproducible_and_leaves_global_rng_unchanged(self):
        labels, a, b = [0, 0, 1, 1], [0.1, 0.9, 0.2, 0.8], [0.1, 0.4, 0.6, 0.9]
        before = np.random.get_state()
        first = paired_auc_interval(labels, a, b, n_resamples=100)
        self.assertEqual(first, paired_auc_interval(labels, a, b, n_resamples=100))
        after = np.random.get_state()
        np.testing.assert_equal(before, after)

    def test_stratification_retains_rare_class_in_small_samples(self):
        result = paired_auc_interval([0, 0, 0, 1], [0.1, 0.9, 0.2, 0.8],
                                     [0.1, 0.2, 0.3, 0.9], n_resamples=100)
        self.assertTrue(np.isfinite(result["confidence_interval"]).all())

    def test_bootstrap_rejects_missing_classes_bad_shapes_and_nonfinite_scores(self):
        cases = [([0, 0], [0.1, 0.2], [0.2, 0.3]),
                 ([0, 1], [0.1], [0.2, 0.3]),
                 ([0, 1], [0.1, float("nan")], [0.2, 0.3])]
        for labels, a, b in cases:
            with self.subTest(labels=labels, a=a):
                with self.assertRaises(ValueError):
                    paired_auc_interval(labels, a, b)
        with self.assertRaises(ValueError):
            paired_auc_interval([0, 1], [0, 1], [0, 1], n_resamples=0)


if __name__ == "__main__":
    unittest.main()
