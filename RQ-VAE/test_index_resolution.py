import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

# Mock dependencies if not available in testing environment
if "torch" not in sys.modules:
    torch_mock = MagicMock()
    nn_mock = MagicMock()
    nn_mock.Module = object
    nn_mock.ModuleList = list
    torch_mock.nn = nn_mock
    torch_mock.utils.data.Dataset = object
    sys.modules["torch"] = torch_mock
    sys.modules["torch.nn"] = nn_mock
    sys.modules["torch.nn.functional"] = MagicMock()
    init_mock = MagicMock()
    sys.modules["torch.nn.init"] = init_mock
    nn_mock.init = init_mock
    sys.modules["torch.utils"] = torch_mock.utils
    sys.modules["torch.utils.data"] = torch_mock.utils.data
    sys.modules["torch.distributed"] = MagicMock()

if "tqdm" not in sys.modules:
    sys.modules["tqdm"] = MagicMock()
if "transformers" not in sys.modules:
    sys.modules["transformers"] = MagicMock()
if "numpy" not in sys.modules:
    sys.modules["numpy"] = MagicMock()

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestIndexResolution(unittest.TestCase):
    def test_tiger_index_resolution(self):
        tiger_dir = os.path.join(REPO_ROOT, "LETTER-TIGER")
        if tiger_dir not in sys.path:
            sys.path.insert(0, tiger_dir)
        import data as tiger_data

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = "Instruments"
            data_dir = os.path.join(tmpdir, dataset)
            tok_dir = os.path.join(data_dir, "letter")
            os.makedirs(tok_dir, exist_ok=True)

            # Subdirectory path
            tok_file = os.path.join(tok_dir, f"{dataset}.index.fixed.json")
            with open(tok_file, "w") as f:
                f.write("{}")

            args = SimpleNamespace(
                dataset=dataset,
                data_path=tmpdir,
                max_his_len=10,
                his_sep=",",
                index_file=f"letter/{dataset}.index.fixed.json",
                add_prefix=False,
            )
            ds = tiger_data.BaseDataset(args)
            self.assertEqual(ds._resolve_index_file(), tok_file)

            # Legacy suffix
            legacy_file = os.path.join(data_dir, f"{dataset}.index.json")
            with open(legacy_file, "w") as f:
                f.write("{}")
            args.index_file = ".index.json"
            ds = tiger_data.BaseDataset(args)
            self.assertEqual(ds._resolve_index_file(), legacy_file)

    def test_lcrec_index_resolution(self):
        lcrec_dir = os.path.join(REPO_ROOT, "LETTER-LC-Rec")
        if "data" in sys.modules:
            del sys.modules["data"]
        if lcrec_dir not in sys.path:
            sys.path.insert(0, lcrec_dir)
        import data as lcrec_data

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = "Instruments"
            data_dir = os.path.join(tmpdir, dataset)
            tok_dir = os.path.join(data_dir, "rqvae")
            os.makedirs(tok_dir, exist_ok=True)

            # Subdirectory path
            tok_file = os.path.join(tok_dir, f"{dataset}.index.fixed.json")
            with open(tok_file, "w") as f:
                f.write("{}")

            args = SimpleNamespace(
                dataset=dataset,
                data_path=tmpdir,
                max_his_len=10,
                his_sep=",",
                index_file=f"rqvae/{dataset}.index.fixed.json",
                add_prefix=False,
            )
            ds = lcrec_data.BaseDataset(args)
            self.assertEqual(ds._resolve_index_file(), tok_file)

            # Legacy suffix
            legacy_file = os.path.join(data_dir, f"{dataset}.index.json")
            with open(legacy_file, "w") as f:
                f.write("{}")
            args.index_file = ".index.json"
            ds = lcrec_data.BaseDataset(args)
            self.assertEqual(ds._resolve_index_file(), legacy_file)


if __name__ == "__main__":
    unittest.main()
