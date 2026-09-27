import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("truncate_indices.py")
SPEC = importlib.util.spec_from_file_location("truncate_indices", MODULE_PATH)
truncate_indices_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(truncate_indices_module)


class TruncateIndicesTest(unittest.TestCase):
    def test_shortest_unique_prefix(self):
        indices = {
            "0": ["<a_1>", "<b_1>", "<c_1>", "<d_1>"],
            "1": ["<a_1>", "<b_1>", "<c_1>", "<d_2>"],
            "2": ["<a_2>", "<b_2>", "<c_2>", "<d_2>"],
        }

        truncated, lengths = truncate_indices_module.truncate_indices(
            indices, min_length=1, max_length=4
        )

        # "2" is unique at length 1: ["<a_2>"]
        self.assertEqual(truncated["2"], ["<a_2>"])
        self.assertEqual(lengths["2"], 1)

        # "0" and "1" share prefix up to length 3, distinct at length 4
        self.assertEqual(truncated["0"], ["<a_1>", "<b_1>", "<c_1>", "<d_1>"])
        self.assertEqual(lengths["0"], 4)
        self.assertEqual(truncated["1"], ["<a_1>", "<b_1>", "<c_1>", "<d_2>"])
        self.assertEqual(lengths["1"], 4)

        self.assertEqual(len({tuple(tokens) for tokens in truncated.values()}), 3)

    def test_rejects_unresolvable_collisions_when_strict(self):
        indices = {
            "0": ["<a_1>", "<b_1>"],
            "1": ["<a_1>", "<b_1>"],
        }

        with self.assertRaisesRegex(ValueError, "cannot produce unique"):
            truncate_indices_module.truncate_indices(
                indices, min_length=1, max_length=2, allow_base_collisions=False
            )

    def test_allows_unresolvable_base_collisions_by_default(self):
        indices = {
            "0": ["<a_1>", "<b_1>"],
            "1": ["<a_1>", "<b_1>"],
            "2": ["<a_1>", "<b_2>"],
        }

        truncated, lengths = truncate_indices_module.truncate_indices(
            indices, min_length=1, max_length=2
        )
        self.assertEqual(truncated["0"], ["<a_1>", "<b_1>"])
        self.assertEqual(truncated["1"], ["<a_1>", "<b_1>"])
        self.assertEqual(truncated["2"], ["<a_1>", "<b_2>"])
        self.assertEqual(lengths, {"0": 2, "1": 2, "2": 2})

    def test_longer_indices_truncation(self):
        indices = {
            "0": ["<a_1>", "<b_1>", "<c_1>", "<d_1>", "<e_1>"],
            "1": ["<a_1>", "<b_1>", "<c_1>", "<d_1>", "<e_2>"],
            "2": ["<a_1>", "<b_2>", "<c_1>", "<d_1>", "<e_1>"],
        }
        truncated, lengths = truncate_indices_module.truncate_indices(
            indices, min_length=1, max_length=5
        )
        self.assertEqual(truncated["0"], ["<a_1>", "<b_1>", "<c_1>", "<d_1>", "<e_1>"])
        self.assertEqual(lengths["0"], 5)
        self.assertEqual(truncated["1"], ["<a_1>", "<b_1>", "<c_1>", "<d_1>", "<e_2>"])
        self.assertEqual(lengths["1"], 5)
        self.assertEqual(truncated["2"], ["<a_1>", "<b_2>"])
        self.assertEqual(lengths["2"], 2)

    def test_popularity_strategy_distribution(self):
        # 4 items with distinct prefixes across layers 1 to 4
        # Range 1 to 4 -> 4 tiers:
        # Tier 0 (lowest freq): min_len = 4
        # Tier 1: min_len = 3
        # Tier 2: min_len = 2
        # Tier 3 (highest freq): min_len = 1
        indices = {
            "0": ["<a_0>", "<b_0>", "<c_0>", "<d_0>"],  # High freq
            "1": ["<a_1>", "<b_1>", "<c_1>", "<d_1>"],  # Mid-high freq
            "2": ["<a_2>", "<b_2>", "<c_2>", "<d_2>"],  # Mid-low freq
            "3": ["<a_3>", "<b_3>", "<c_3>", "<d_3>"],  # Rare / tail
        }
        freqs = {
            "0": 1000,
            "1": 100,
            "2": 10,
            "3": 1,
        }

        truncated, lengths = truncate_indices_module.truncate_indices(
            indices,
            min_length=1,
            max_length=4,
            strategy="popularity",
            item_frequencies=freqs,
        )

        # "0" is top tier (rank 3/4 -> tier 3): min_length=1 -> truncated to 1
        self.assertEqual(lengths["0"], 1)
        self.assertEqual(truncated["0"], ["<a_0>"])

        # "1" is tier 2 (rank 2/4): min_length=2 -> truncated to 2
        self.assertEqual(lengths["1"], 2)
        self.assertEqual(truncated["1"], ["<a_1>", "<b_1>"])

        # "2" is tier 1 (rank 1/4): min_length=3 -> truncated to 3
        self.assertEqual(lengths["2"], 3)
        self.assertEqual(truncated["2"], ["<a_2>", "<b_2>", "<c_2>"])

        # "3" is bottom tier (rank 0/4): min_length=4 -> stays at 4
        self.assertEqual(lengths["3"], 4)
        self.assertEqual(truncated["3"], ["<a_3>", "<b_3>", "<c_3>", "<d_3>"])

        # All IDs are unique
        self.assertEqual(len({tuple(tokens) for tokens in truncated.values()}), 4)

    def test_popularity_strategy_requires_frequencies(self):
        indices = {"0": ["<a_0>", "<b_0>"]}
        with self.assertRaisesRegex(ValueError, "requires item_frequencies"):
            truncate_indices_module.truncate_indices(
                indices, min_length=1, max_length=2, strategy="popularity"
            )

    def test_residual_strategy(self):
        # 3 items, each prefix unique at depth 1
        indices = {
            "0": ["<a_0>", "<b_0>", "<c_0>", "<d_0>"],
            "1": ["<a_1>", "<b_1>", "<c_1>", "<d_1>"],
            "2": ["<a_2>", "<b_2>", "<c_2>", "<d_2>"],
        }
        residuals = {
            # "0" has low error already at depth 1 (0.10 <= 0.20) -> should truncate to 1
            "0": [0.10, 0.05, 0.02, 0.01],
            # "1" has high error at depth 1 (0.40), but acceptable at depth 2 (0.15 <= 0.20) -> should truncate to 2
            "1": [0.40, 0.15, 0.08, 0.02],
            # "2" has high error at depth 1, 2, and 3 (> 0.20) -> must stay at max_length (4)
            "2": [0.60, 0.45, 0.25, 0.05],
        }

        truncated, lengths = truncate_indices_module.truncate_indices(
            indices,
            min_length=1,
            max_length=4,
            strategy="residual",
            residuals=residuals,
            residual_threshold=0.20,
        )

        self.assertEqual(lengths["0"], 1)
        self.assertEqual(truncated["0"], ["<a_0>"])

        self.assertEqual(lengths["1"], 2)
        self.assertEqual(truncated["1"], ["<a_1>", "<b_1>"])

        self.assertEqual(lengths["2"], 4)
        self.assertEqual(truncated["2"], ["<a_2>", "<b_2>", "<c_2>", "<d_2>"])

    def test_residual_strategy_requires_residuals(self):
        indices = {"0": ["<a_0>", "<b_0>"]}
        with self.assertRaisesRegex(ValueError, "requires residuals"):
            truncate_indices_module.truncate_indices(
                indices, min_length=1, max_length=2, strategy="residual"
            )

    def test_unknown_strategy_raises_error(self):
        indices = {"0": ["<a_0>", "<b_0>"]}
        with self.assertRaisesRegex(ValueError, "Unknown strategy"):
            truncate_indices_module.truncate_indices(
                indices, min_length=1, max_length=2, strategy="invalid_strategy"
            )

    def test_collaborative_user_entropy_distinguishes_power_users(self):
        # Item "0" has 10 interactions from 10 distinct users -> high entropy
        # Item "1" has 10 interactions from 1 single user -> low entropy (0.0)
        # Item "2" has 1 interaction -> entropy 0.0
        inter_data = {
            f"user_{i}": ["0"] for i in range(10)
        }
        inter_data["power_user"] = ["1"] * 10
        inter_data["solo_user"] = ["2"]

        scores, freqs = truncate_indices_module.compute_interaction_signals(
            inter_data, signal="user_entropy"
        )
        self.assertEqual(freqs["0"], 10)
        self.assertEqual(freqs["1"], 10)
        self.assertGreater(scores["0"], 3.0)  # log2(10) ~ 3.32
        self.assertAlmostEqual(scores["1"], 0.0)

        # Now test truncation tiering with user_entropy
        indices = {
            "0": ["<a_0>", "<b_0>", "<c_0>"],
            "1": ["<a_1>", "<b_1>", "<c_1>"],
            "2": ["<a_2>", "<b_2>", "<c_2>"],
        }
        truncated, lengths = truncate_indices_module.truncate_indices(
            indices,
            min_length=1,
            max_length=3,
            strategy="collaborative",
            item_scores=scores,
        )
        # "0" should be ranked highest (shortest length: 1)
        self.assertEqual(lengths["0"], 1)
        # "1" has low entropy, so it is ranked lower than "0"
        self.assertGreater(lengths["1"], lengths["0"])

    def test_collaborative_target_frequency(self):
        inter_data = {
            "u1": ["0", "1", "2"],  # "2" is target
            "u2": ["0", "2"],       # "2" is target
            "u3": ["1", "0"],       # "0" is target
        }
        scores, freqs = truncate_indices_module.compute_interaction_signals(
            inter_data, signal="target"
        )
        # "2" is target 2 times, "0" is target 1 time, "1" is target 0 times
        self.assertEqual(scores["2"], 2.0)
        self.assertEqual(scores["0"], 1.0)
        self.assertEqual(scores["1"], 0.0)

        indices = {
            "0": ["<a_0>", "<b_0>", "<c_0>"],
            "1": ["<a_1>", "<b_1>", "<c_1>"],
            "2": ["<a_2>", "<b_2>", "<c_2>"],
        }
        truncated, lengths = truncate_indices_module.truncate_indices(
            indices,
            min_length=1,
            max_length=3,
            strategy="collaborative",
            item_scores=scores,
        )
        self.assertEqual(lengths["2"], 1)
        self.assertEqual(lengths["1"], 3)

    def test_collaborative_pagerank(self):
        # 0 -> 1, 2 -> 1, 3 -> 1: Item 1 is a major transition sink/hub
        inter_data = {
            "u1": ["0", "1"],
            "u2": ["2", "1"],
            "u3": ["3", "1"],
        }
        scores, _ = truncate_indices_module.compute_interaction_signals(
            inter_data, signal="pagerank"
        )
        self.assertGreater(scores["1"], scores["0"])
        self.assertGreater(scores["1"], scores["2"])
        self.assertGreater(scores["1"], scores["3"])

        indices = {
            "0": ["<a_0>", "<b_0>", "<c_0>"],
            "1": ["<a_1>", "<b_1>", "<c_1>"],
            "2": ["<a_2>", "<b_2>", "<c_2>"],
            "3": ["<a_3>", "<b_3>", "<c_3>"],
        }
        _, lengths = truncate_indices_module.truncate_indices(
            indices,
            min_length=1,
            max_length=3,
            strategy="collaborative",
            item_scores=scores,
        )
        # Item 1 should have the shortest length
        self.assertEqual(lengths["1"], 1)

    def test_collaborative_composite_signal(self):
        inter_data = {
            f"user_{i}": ["0"] for i in range(10)
        }
        inter_data["solo"] = ["1"]
        scores, _ = truncate_indices_module.compute_interaction_signals(
            inter_data, signal="composite"
        )
        self.assertGreater(scores["0"], scores["1"])


if __name__ == "__main__":
    unittest.main()
