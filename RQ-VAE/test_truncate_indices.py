import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("truncate_indices.py")
SPEC = importlib.util.spec_from_file_location("truncate_indices", MODULE_PATH)
truncate_indices_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(truncate_indices_module)


class TruncateIndicesTest(unittest.TestCase):
    def test_extends_colliding_prefixes_until_unique(self):
        indices = {
            "0": ["<a_1>", "<b_1>", "<c_1>", "<d_1>"],
            "1": ["<a_1>", "<b_1>", "<c_1>", "<d_2>"],
            "2": ["<a_2>", "<b_2>", "<c_2>", "<d_2>"],
        }

        truncated, lengths = truncate_indices_module.truncate_indices(
            indices, min_length=1, max_length=4, seed=42
        )

        self.assertEqual(len({tuple(tokens) for tokens in truncated.values()}), 3)
        self.assertTrue(all(1 <= length <= 4 for length in lengths.values()))

    def test_rejects_unresolvable_collisions(self):
        indices = {
            "0": ["<a_1>", "<b_1>"],
            "1": ["<a_1>", "<b_1>"],
        }

        with self.assertRaisesRegex(ValueError, "cannot produce unique"):
            truncate_indices_module.truncate_indices(
                indices, min_length=1, max_length=2, seed=42
            )


if __name__ == "__main__":
    unittest.main()
