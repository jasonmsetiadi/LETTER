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



if __name__ == "__main__":
    unittest.main()
