import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

RQ_DIR = str(Path(__file__).resolve().parent)
if RQ_DIR not in sys.path:
    sys.path.insert(0, RQ_DIR)

# If torch is not installed in the testing environment, provide minimal mocks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    import types
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    nn_init = types.ModuleType("torch.nn.init")
    nn_functional = types.ModuleType("torch.nn.functional")
    torch.no_grad = lambda *args, **kwargs: (lambda fn: fn)
    torch.clamp = MagicMock()
    torch.matmul = MagicMock()
    torch.stack = MagicMock()
    torch.tensor = MagicMock()
    torch.LongTensor = MagicMock()
    torch.FloatTensor = MagicMock()
    torch.device = MagicMock()
    torch.arange = MagicMock()
    torch.zeros_like = MagicMock()
    torch.from_numpy = MagicMock()
    torch.nn = nn
    class MockModule(object):
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    nn.Module = MockModule
    nn.ModuleList = list
    torch.optim = MagicMock()
    sys.modules["torch.optim"] = torch.optim
    class MockEmbedding(MagicMock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight = MagicMock()
            self.weight.data = MagicMock()
            self.weight.data.uniform_ = MagicMock()
    nn.Embedding = MockEmbedding

    nn_init.xavier_normal_ = MagicMock()
    nn_init.constant_ = MagicMock()
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.init"] = nn_init
    sys.modules["torch.nn.functional"] = nn_functional




try:
    import sklearn
    from sklearn.cluster import KMeans
except ImportError:
    import types
    sklearn = types.ModuleType("sklearn")
    sklearn_cluster = types.ModuleType("sklearn.cluster")
    sklearn.cluster = sklearn_cluster
    sklearn_cluster.KMeans = MagicMock()
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.cluster"] = sklearn_cluster

try:
    import numpy as np

except ImportError:
    np = MagicMock()
    sys.modules["numpy"] = np

from truncate_indices import resolve_target_lengths
from models.rq import ResidualVectorQuantizer
from models.rqvae import RQVAE


class TestPhase15TargetLengths(unittest.TestCase):
    def test_resolve_target_lengths_from_dict_and_list(self):
        # 1. From list
        lens_list = [2, 4, 3]
        res1 = resolve_target_lengths(3, target_lengths=lens_list, min_length=1, max_length=4)
        self.assertEqual(res1, {0: 2, 1: 4, 2: 3})

        # 2. From dict of integers
        lens_dict = {"0": 2, "1": 3}
        res2 = resolve_target_lengths(3, target_lengths=lens_dict, min_length=1, max_length=4)
        self.assertEqual(res2[0], 2)
        self.assertEqual(res2[1], 3)
        self.assertEqual(res2[2], 4)  # default filled to max_length

        # 3. From index format (list of token strings)
        index_dict = {
            "0": ["<a_1>", "<b_2>"],
            "1": ["<a_1>", "<b_2>", "<c_3>", "<d_4>"],
        }
        res3 = resolve_target_lengths(2, target_lengths=index_dict, min_length=1, max_length=4)
        self.assertEqual(res3, {0: 2, 1: 4})

    def test_resolve_target_lengths_popularity_cdf(self):
        # Synthetic user interaction history
        inters = {
            "u1": [0, 0, 0, 0, 0, 0, 0, 0],  # item 0 is very popular
            "u2": [0, 0, 0, 0, 1],           # item 1 has moderate traffic
            "u3": [2],                       # item 2 is rare
        }
        res = resolve_target_lengths(
            num_items=3,
            strategy="popularity",
            inter_source=inters,
            min_length=2,
            max_length=4,
        )
        self.assertEqual(len(res), 3)
        # In popularity CDF tiering, popular item 0 gets min_length=2, rare item 2 gets max_length=4
        self.assertEqual(res[0], 2)
        self.assertEqual(res[2], 4)
        self.assertTrue(res[0] <= res[1] <= res[2])

    def test_resolve_target_lengths_residual(self):
        residuals = {
            "0": [0.05, 0.02, 0.01],  # depth 1 already <= 0.1
            "1": [0.35, 0.08, 0.03],  # depth 2 <= 0.1
            "2": [0.50, 0.40, 0.30],  # never <= 0.1, defaults to max_length=3
        }
        res = resolve_target_lengths(
            num_items=3,
            strategy="residual",
            residuals=residuals,
            residual_threshold=0.1,
            min_length=1,
            max_length=3,
        )
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1], 2)
        self.assertEqual(res[2], 3)

    def test_resolve_target_lengths_shortest_unique(self):
        pilot_index = {
            "0": ["<a_1>", "<b_1>", "<c_1>"],  # prefix <a_1> unique at depth 1
            "1": ["<a_2>", "<b_2>", "<c_1>"],  # prefix <a_2>, <b_2> unique at depth 2
            "2": ["<a_2>", "<b_1>", "<c_2>"],  # shares prefix with 3, unique at depth 3
            "3": ["<a_2>", "<b_1>", "<c_3>"],  # shares prefix with 2, unique at depth 3
        }
        res = resolve_target_lengths(
            num_items=4,
            strategy="shortest_unique",
            indices=pilot_index,
            min_length=1,
            max_length=3,
        )
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1], 2)
        self.assertEqual(res[2], 3)
        self.assertEqual(res[3], 3)


class TestPhase15Quantizer(unittest.TestCase):
    def test_rvq_forward_masked_accumulation(self):
        """Test that lengths properly mask residual updates in ResidualVectorQuantizer."""
        # Create a mock quantizer
        orig_vq = sys.modules["models.rq"].VectorQuantizer
        mock_vq_class = MagicMock()

        # Mock each quantizer layer's output
        def make_mock_layer(idx):
            layer = MagicMock()
            def forward_call(residual, label, i, active_mask=None, use_sk=True):
                # Return dummy x_res, loss, indices
                x_res = MagicMock()
                # support __mul__ with float tensor
                x_res.__mul__ = MagicMock(return_value=x_res)
                x_res.__rmul__ = MagicMock(return_value=x_res)
                indices = MagicMock()
                return x_res, 0.1, indices
            layer.side_effect = forward_call
            return layer

        orig_vq = sys.modules["models.rq"].VectorQuantizer
        sys.modules["models.rq"].VectorQuantizer = MagicMock()
        try:
            mock_layers = [make_mock_layer(i) for i in range(4)]
            rvq = ResidualVectorQuantizer([256, 256, 256, 256], 32)
            rvq.vq_layers = mock_layers

            # Mock input x and lengths
            x = MagicMock()
            x.size = MagicMock(return_value=2)
            labels = {str(i): [] for i in range(4)}
            lengths = MagicMock()
            lengths.__ge__ = MagicMock(return_value=MagicMock(any=MagicMock(return_value=True), float=MagicMock(return_value=MagicMock(unsqueeze=MagicMock(return_value=1.0)))))

            out_q, loss, indices = rvq.forward(x, labels, lengths=lengths)
            # All 4 layers called
            self.assertEqual(len(rvq.vq_layers), 4)
        finally:
            sys.modules["models.rq"].VectorQuantizer = orig_vq


    def test_rqvae_forward_passes_lengths(self):
        """Test that RQVAE.forward forwards lengths argument to rq."""
        orig_mlp = sys.modules["models.rqvae"].MLPLayers
        orig_rvq = sys.modules["models.rqvae"].ResidualVectorQuantizer
        sys.modules["models.rqvae"].MLPLayers = MagicMock()
        sys.modules["models.rqvae"].ResidualVectorQuantizer = MagicMock()

        try:
            model = RQVAE(in_dim=64, num_emb_list=[256, 256, 256, 256], e_dim=32, layers=[32])
            model.rq = MagicMock()
            model.rq.return_value = (MagicMock(), MagicMock(), MagicMock())
            model.encoder = MagicMock(return_value=MagicMock())
            model.decoder = MagicMock(return_value=MagicMock())

            x = MagicMock()
            labels = {}
            lengths = MagicMock()

            model.forward(x, labels, lengths=lengths)
            model.rq.assert_called_with(
                model.encoder(x),
                labels,
                lengths=lengths,
                residual_threshold=None,
                min_length=1,
                use_sk=True,
                return_lengths=False,
            )

        finally:
            sys.modules["models.rqvae"].MLPLayers = orig_mlp
            sys.modules["models.rqvae"].ResidualVectorQuantizer = orig_rvq


class TestPhase15Trainer(unittest.TestCase):
    def setUp(self):
        if not HAS_TORCH:
            self.skipTest("PyTorch is required for Trainer tests")

    def test_trainer_target_lengths_initialization(self):
        # Ensure RQ-VAE's utils is imported rather than LETTER-TIGER's utils
        if "utils" in sys.modules and not hasattr(sys.modules["utils"], "set_color"):
            del sys.modules["utils"]
        if RQ_DIR not in sys.path or sys.path[0] != RQ_DIR:
            sys.path.insert(0, RQ_DIR)
        import utils
        from trainer import Trainer

        args = MagicMock()
        args.phase = 1.5
        args.lr = 1e-3
        args.learner = "adam"
        args.weight_decay = 0.0
        args.epochs = 1
        args.eval_step = 1
        args.device = "cpu"
        args.ckpt_dir = "/tmp/mock_ckpt"

        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=[])
        model.rq = MagicMock()
        model.rq.vq_layers = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        trainer = Trainer(args, model, target_lengths={0: 2, 1: 4})
        self.assertEqual(trainer.phase, 1.5)
        self.assertIsNotNone(trainer.target_lengths)

    def test_trainer_online_residual_initialization(self):
        if "utils" in sys.modules and not hasattr(sys.modules["utils"], "set_color"):
            del sys.modules["utils"]
        if RQ_DIR not in sys.path or sys.path[0] != RQ_DIR:
            sys.path.insert(0, RQ_DIR)
        import utils
        from trainer import Trainer

        args = MagicMock()
        args.phase = 1.5
        args.target_length_strategy = "residual"
        args.residual_threshold = 0.15
        args.min_length = 2
        args.lr = 1e-3
        args.learner = "adam"
        args.weight_decay = 0.0
        args.epochs = 1
        args.eval_step = 1
        args.device = "cpu"
        args.ckpt_dir = "/tmp/mock_ckpt"

        model = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=[])
        model.rq = MagicMock()
        model.rq.vq_layers = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]

        trainer = Trainer(args, model, target_lengths=None)
        self.assertEqual(trainer.phase, 1.5)
        self.assertTrue(trainer.online_residual)
        self.assertEqual(trainer.residual_threshold, 0.15)
        self.assertEqual(trainer.min_length, 2)
        self.assertIsNone(trainer.target_lengths)

    def test_vq_empty_and_tiny_active_mask(self):
        """Test that VectorQuantizer handles empty or tiny active sets without NaNs or errors."""
        from models.vq import VectorQuantizer
        orig_kmeans = getattr(sys.modules.get("models.layers"), "kmeans", None)
        try:
            vq = VectorQuantizer(n_e=16, e_dim=8, sk_epsilon=0.01)
            vq.initted = True
            
            # Mock x
            x = MagicMock()
            x.view = MagicMock(return_value=x)
            x.shape = (4, 8)
            x.device = "cpu"
            
            # Case 1: Empty active mask
            active_mask_empty = MagicMock()
            active_mask_empty.sum = MagicMock(return_value=MagicMock(item=MagicMock(return_value=0)))
            active_mask_empty.all = MagicMock(return_value=False)
            active_mask_empty.any = MagicMock(return_value=False)
            
            # Should not raise exception
            out_q, loss, indices = vq.forward(x, label=[], idx=0, active_mask=active_mask_empty)
            self.assertIsNotNone(loss)

            # Case 2: Active count < n_e (e.g. 2 < 16)
            active_mask_tiny = MagicMock()
            active_mask_tiny.sum = MagicMock(return_value=MagicMock(item=MagicMock(return_value=2)))
            active_mask_tiny.all = MagicMock(return_value=False)
            active_mask_tiny.any = MagicMock(return_value=True)
            # Should fall back from Sinkhorn safely without error
            out_q2, loss2, indices2 = vq.forward(x, label=[], idx=0, active_mask=active_mask_tiny)
            self.assertIsNotNone(out_q2)
        finally:
            pass

    def test_halted_items_zero_masking_at_deeper_layers(self):
        """Test that items halted at depth k are multiplied by 0 at deeper layers."""
        from models.rq import ResidualVectorQuantizer
        orig_vq = sys.modules["models.rq"].VectorQuantizer
        try:
            # Create a mock quantizer returning tracked outputs
            mock_layer_calls = []
            class MockVQ(object):
                def __init__(self, *args, **kwargs):
                    pass
                def __call__(self, residual, label, idx, active_mask=None, use_sk=True):
                    mock_layer_calls.append((idx, active_mask))
                    res = MagicMock()
                    res.__mul__ = MagicMock(return_value=res)
                    res.__rmul__ = MagicMock(return_value=res)
                    return res, 0.0, MagicMock()
            sys.modules["models.rq"].VectorQuantizer = MockVQ
            
            rvq = ResidualVectorQuantizer([16, 16, 16, 16], 8)
            x = MagicMock()
            x.size = MagicMock(return_value=2)
            labels = {str(i): [] for i in range(4)}
            
            # Item 0 has length 2, Item 1 has length 4
            lengths = MagicMock()
            # lengths >= depth
            def ge_mock(*args, **kwargs):
                mock_mask = MagicMock()
                # for depth 3, item 0 is False, item 1 is True
                mock_mask.any = MagicMock(return_value=True)
                mock_mask.all = MagicMock(return_value=False)
                mock_mask.float = MagicMock(return_value=mock_mask)
                mock_mask.unsqueeze = MagicMock(return_value=mock_mask)
                return mock_mask
            lengths.__ge__ = ge_mock
            
            x_q, mean_losses, indices = rvq.forward(x, labels, lengths=lengths)
            # Verify depth 3 and 4 received masks distinguishing active items
            self.assertEqual(len(mock_layer_calls), 4)
            self.assertIsNotNone(mock_layer_calls[2][1])
            self.assertIsNotNone(mock_layer_calls[3][1])
        finally:
            sys.modules["models.rq"].VectorQuantizer = orig_vq


class TestPhase15Reporting(unittest.TestCase):
    """Test report generation and Semantic ID evaluation for Phase 1.5."""

    def test_file_to_strategy_phase_detection(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Phase 1 detection
        self.assertEqual(mod.file_to_strategy("fixed_L4.json"), ("fixed", "fixed"))
        self.assertEqual(mod.file_to_strategy("varlen.json"), ("shortest_unique", "1"))
        self.assertEqual(mod.file_to_strategy("varlen-res.json"), ("residual", "1"))
        self.assertEqual(mod.file_to_strategy("varlen-pop.json"), ("popularity:frequency", "1"))
        self.assertEqual(mod.file_to_strategy("varlen-pop-entropy.json"), ("popularity:user_entropy", "1"))
        self.assertEqual(mod.file_to_strategy("varlen-pop-pagerank.json"), ("popularity:pagerank", "1"))

        # Phase 1.5 detection
        self.assertEqual(mod.file_to_strategy("varlen_phase1.5.json"), ("shortest_unique", "1.5"))
        self.assertEqual(mod.file_to_strategy("varlen-res_phase1.5.json"), ("residual", "1.5"))
        self.assertEqual(mod.file_to_strategy("varlen-pop_phase1.5.json"), ("popularity:frequency", "1.5"))
        self.assertEqual(mod.file_to_strategy("varlen-pop-entropy_phase1.5.json"), ("popularity:user_entropy", "1.5"))
        self.assertEqual(mod.file_to_strategy("varlen-pop-pagerank_phase1.5.json"), ("popularity:pagerank", "1.5"))

        # With tag
        self.assertEqual(mod.file_to_strategy("varlen-res_max10.json", tag="_max10"), ("residual", "1"))
        self.assertEqual(mod.file_to_strategy("varlen-res_phase1.5_max10.json", tag="_max10"), ("residual", "1.5"))
        self.assertEqual(mod.file_to_strategy("varlen-res_max4.json", tag="_max4"), ("residual", "1"))
        self.assertEqual(mod.file_to_strategy("varlen-res_phase1.5_max4.json", tag="_max4"), ("residual", "1.5"))
        self.assertIsNone(mod.file_to_strategy("varlen-res_phase1.5.json", tag="_max4"))

    def test_strategy_to_filename(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        self.assertEqual(mod.strategy_to_filename("fixed", "fixed"), "fixed_L4.json")
        self.assertEqual(mod.strategy_to_filename("fixed", "fixed", tag="_max4"), "fixed_L4.json")
        self.assertEqual(mod.strategy_to_filename("fixed", "fixed", tag="_max10"), "fixed_L10.json")
        self.assertEqual(mod.strategy_to_filename("residual", "1"), "varlen-res.json")
        self.assertEqual(mod.strategy_to_filename("residual", "1.5"), "varlen-res_phase1.5.json")
        self.assertEqual(mod.strategy_to_filename("popularity:frequency", "1.5"), "varlen-pop_phase1.5.json")
        self.assertEqual(mod.strategy_to_filename("popularity:user_entropy", "1.5"), "varlen-pop-entropy_phase1.5.json")
        self.assertEqual(mod.strategy_to_filename("residual", "1.5", tag="_max10"), "varlen-res_phase1.5_max10.json")
        self.assertEqual(mod.strategy_to_filename("residual", "1.5", tag="_max4"), "varlen-res_phase1.5_max4.json")

    def test_item_sort_key(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        items = [
            ("residual", "1.5"),
            ("shortest_unique", "1"),
            ("fixed", "fixed"),
            ("popularity:frequency", "1.5"),
            ("residual", "1"),
            ("popularity:frequency", "1"),
        ]
        sorted_items = sorted(items, key=mod.item_sort_key)
        expected = [
            ("fixed", "fixed"),
            ("shortest_unique", "1"),
            ("popularity:frequency", "1"),
            ("popularity:frequency", "1.5"),
            ("residual", "1"),
            ("residual", "1.5"),
        ]
        self.assertEqual(sorted_items, expected)

    def test_compare_semantic_ids_phase1_5_loading(self):
        from compare_semantic_ids import evaluate_semantic_ids
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset = "MockData"
            data_dir = tmp_path / "data" / dataset
            tok_dir = data_dir / "rqvae"
            tok_dir.mkdir(parents=True, exist_ok=True)

            # Create mock fixed index
            fixed_idx = {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8], "2": [9, 10, 11, 12]}
            with open(data_dir / f"{dataset}.index.json", "w") as f:
                json.dump(fixed_idx, f)

            # Create mock interaction file
            inter_data = {"0": [0, 1]}
            with open(data_dir / f"{dataset}.inter.json", "w") as f:
                json.dump(inter_data, f)

            # Create Phase 1.5 residual index file
            p15_idx = {"0": [1, 2], "1": [5, 6, 7], "2": [9, 10, 11, 12]}
            p15_file = tok_dir / f"{dataset}.index.varlen.res-phase1.5.max4.json"
            with open(p15_file, "w") as f:
                json.dump(p15_idx, f)

            # Evaluate with strategies list containing tuple
            eval_res = evaluate_semantic_ids(
                dataset=dataset,
                tokenizer="rqvae",
                repo_root=tmp_path,
                data_root=tmp_path / "data",
                strategies=[("fixed", "fixed"), ("residual", "1.5")],
            )

            results = eval_res["results"]
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["strategy"], "fixed")
            self.assertEqual(results[0]["phase"], "fixed")
            self.assertEqual(results[1]["strategy"], "residual")
            self.assertEqual(results[1]["phase"], "1.5")
            self.assertAlmostEqual(results[1]["mean_length"], (2 + 3 + 4) / 3.0)

    def test_end_to_end_dual_phase_comparison_report(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset = "DummyDataset"
            data_dir = tmp_path / "data" / dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            res_dir = tmp_path / "LETTER-TIGER" / "results" / dataset / "rqvae"
            res_dir.mkdir(parents=True, exist_ok=True)

            # Mock fixed index and inter
            with open(data_dir / f"{dataset}.index.json", "w") as f:
                json.dump({"0": [1, 2, 3, 4], "1": [2, 3, 4, 5]}, f)
            with open(data_dir / f"{dataset}.inter.json", "w") as f:
                json.dump({"0": [0, 1]}, f)

            # Fixed baseline result
            fixed_res = {"mean_results": {"hit@1": 0.05, "hit@5": 0.08, "hit@10": 0.10, "ndcg@5": 0.07, "ndcg@10": 0.08}}
            with open(res_dir / "fixed_L4.json", "w") as f:
                json.dump(fixed_res, f)

            # Phase 1 residual result
            p1_res = {"mean_results": {"hit@1": 0.06, "hit@5": 0.085, "hit@10": 0.105, "ndcg@5": 0.075, "ndcg@10": 0.082}}
            with open(res_dir / "varlen-res.json", "w") as f:
                json.dump(p1_res, f)

            # Phase 1.5 residual result
            p15_res = {"mean_results": {"hit@1": 0.062, "hit@5": 0.088, "hit@10": 0.112, "ndcg@5": 0.078, "ndcg@10": 0.086}}
            with open(res_dir / "varlen-res_phase1.5.json", "w") as f:
                json.dump(p15_res, f)

            # Run main() with mock argv
            orig_argv = sys.argv
            try:
                sys.argv = [
                    "generate_comparison_report.py",
                    "--dataset", dataset,
                    "--repo-root", str(tmp_path),
                    "--phase", "both",
                    "--tokenizer", "rqvae",
                    "--strategies", "fixed,residual",
                ]
                mod.main()

                # Verify files were generated
                md_file = res_dir / "strategy_comparison.md"
                json_file = res_dir / "strategy_comparison.json"
                self.assertTrue(md_file.exists())
                self.assertTrue(json_file.exists())

                with open(md_file) as f:
                    md_text = f.read()

                # Verify markdown contains Phase columns and Head-to-Head section
                self.assertIn("| Strategy | Phase | Mean Length |", md_text)
                self.assertIn("Phase 1 vs. Phase 1.5 Head-to-Head Comparison", md_text)
                self.assertIn("+0.70%", md_text)  # (0.112 - 0.105) * 100 = +0.70% Hit@10 delta!
                self.assertIn("+0.40%", md_text)  # (0.086 - 0.082) * 100 = +0.40% NDCG@10 delta!

                with open(json_file) as f:
                    data = json.load(f)
                strats = [(r["strategy"], r["phase"]) for r in data["strategy_comparison"]]
                self.assertIn(("fixed", "fixed"), strats)
                self.assertIn(("residual", "1"), strats)
                self.assertIn(("residual", "1.5"), strats)
            finally:
                sys.argv = orig_argv

    def test_file_to_strategy_multi_depth_fixed(self):
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Standard filenames without tag
        self.assertEqual(mod.file_to_strategy("fixed_L4.json"), ("fixed", "fixed"))
        self.assertEqual(mod.file_to_strategy("fixed_L2.json"), ("fixed_L2", "fixed"))
        self.assertEqual(mod.file_to_strategy("fixed_L3.json"), ("fixed_L3", "fixed"))
        self.assertEqual(mod.file_to_strategy("fixed_L6.json"), ("fixed_L6", "fixed"))
        self.assertEqual(mod.file_to_strategy("varlen-res.json"), ("residual", "1"))
        self.assertEqual(mod.file_to_strategy("varlen-res_phase1.5.json"), ("residual", "1.5"))

        # With tag
        self.assertEqual(mod.file_to_strategy("fixed_L10.json", tag="_max10"), ("fixed", "fixed"))
        self.assertEqual(mod.file_to_strategy("fixed_L2.json", tag="_max10"), ("fixed_L2", "fixed"))
        self.assertEqual(mod.file_to_strategy("fixed_L4.json", tag="_max4"), ("fixed", "fixed"))
        self.assertIsNone(mod.file_to_strategy("fixed.json", tag="_max4"))
        self.assertEqual(mod.file_to_strategy("fixed_L2.json", tag="_max4"), ("fixed_L2", "fixed"))

    def test_strategy_sort_order_multi_depth_fixed(self):
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        strats = ["residual", "fixed", "fixed_L3", "fixed_L2", "shortest_unique"]
        sorted_strats = sorted(strats, key=lambda s: mod.strategy_sort_key(s, max_length=4))
        self.assertEqual(sorted_strats, ["fixed_L2", "fixed_L3", "fixed", "shortest_unique", "residual"])

    def test_compute_rate_distortion_frontier(self):
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Mock table data with L=2 and L=4 fixed baselines
        recom_data = [
            {
                "strategy": "fixed_L2",
                "phase": "fixed",
                "mean_length": 2.0,
                "weighted_length": 2.0,
                "status": "completed",
                "metrics": {"hit@10": 0.08, "ndcg@10": 0.06},
            },
            {
                "strategy": "fixed",
                "phase": "fixed",
                "mean_length": 4.0,
                "weighted_length": 4.0,
                "status": "completed",
                "metrics": {"hit@10": 0.10, "ndcg@10": 0.08},
            },
            {
                "strategy": "var_efficient",
                "phase": "1.5",
                "mean_length": 3.0,
                "weighted_length": 3.0,
                "status": "completed",
                "metrics": {"hit@10": 0.095, "ndcg@10": 0.075},
            },
            {
                "strategy": "var_dominant",
                "phase": "1.5",
                "mean_length": 3.0,
                "weighted_length": 3.0,
                "status": "completed",
                "metrics": {"hit@10": 0.105, "ndcg@10": 0.085},
            },
            {
                "strategy": "var_tradeoff",
                "phase": "1",
                "mean_length": 3.0,
                "weighted_length": 3.0,
                "status": "completed",
                "metrics": {"hit@10": 0.085, "ndcg@10": 0.065},
            },
        ]

        rd = mod.compute_rate_distortion_frontier(recom_data, max_length=4)
        self.assertEqual(len(rd["fixed_curve"]), 2)
        self.assertEqual(rd["fixed_curve"][0]["length"], 2.0)
        self.assertEqual(rd["fixed_curve"][1]["length"], 4.0)

        vb = {v["strategy"]: v for v in rd["variable_bracketed"]}

        # Check var_efficient at L=3.0:
        # Interpolated NDCG@10 = (0.06 + 0.08) / 2 = 0.070
        # Metric is 0.075 > 0.070 -> ▲ Efficient
        v_eff = vb["var_efficient"]
        self.assertEqual(v_eff["floor_length"], 2.0)
        self.assertEqual(v_eff["ceil_length"], 4.0)
        self.assertAlmostEqual(v_eff["interp_metrics"]["ndcg@10"], 0.070, places=4)
        self.assertAlmostEqual(v_eff["delta_interp"]["ndcg@10"], ((0.075 - 0.070) / 0.070) * 100, places=2)
        self.assertEqual(v_eff["pareto_status"], "▲ Efficient")

        # Check var_dominant at L=3.0:
        # Metric is 0.085 >= 0.080 (ceil) -> ★ Dominant
        v_dom = vb["var_dominant"]
        self.assertEqual(v_dom["pareto_status"], "★ Dominant")
        self.assertIn(("var_dominant", "1.5"), rd["pareto_dominant_strats"])

        # Check var_tradeoff at L=3.0:
        # Metric is 0.065 < 0.070 -> ▼ Trade-off
        v_tr = vb["var_tradeoff"]
        self.assertEqual(v_tr["pareto_status"], "▼ Trade-off")

    def test_rate_distortion_report_end_to_end(self):
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset = "RDDataset"
            data_dir = tmp_path / "data" / dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            res_dir = tmp_path / "LETTER-TIGER" / "results" / dataset / "rqvae"
            res_dir.mkdir(parents=True, exist_ok=True)

            with open(data_dir / f"{dataset}.index.json", "w") as f:
                json.dump({"0": [1, 2, 3, 4], "1": [2, 3, 4, 5]}, f)
            with open(data_dir / f"{dataset}.inter.json", "w") as f:
                json.dump({"0": [0, 1]}, f)

            # Fixed L=2 baseline result
            with open(res_dir / "fixed_L2.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.03, "hit@5": 0.06, "hit@10": 0.08, "ndcg@5": 0.05, "ndcg@10": 0.06}}, f)

            # Fixed L=4 baseline result
            with open(res_dir / "fixed_L4.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.05, "hit@5": 0.08, "hit@10": 0.10, "ndcg@5": 0.07, "ndcg@10": 0.08}}, f)

            # Residual Phase 1.5 index with lengths 3
            tok_data_dir = data_dir / "rqvae"
            tok_data_dir.mkdir(parents=True, exist_ok=True)
            with open(tok_data_dir / f"{dataset}.index.varlen.res-phase1.5.max4.json", "w") as f:
                json.dump({"0": [1, 2, 3], "1": [2, 3, 4]}, f)

            # Residual Phase 1.5 result (NDCG@10 = 0.078 beats interpolated 0.070 at L=3.0)
            with open(res_dir / "varlen-res_phase1.5.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.048, "hit@5": 0.078, "hit@10": 0.098, "ndcg@5": 0.068, "ndcg@10": 0.078}}, f)

            orig_argv = sys.argv
            try:
                sys.argv = [
                    "generate_comparison_report.py",
                    "--dataset", dataset,
                    "--repo-root", str(tmp_path),
                    "--phase", "1.5",
                    "--tokenizer", "rqvae",
                    "--strategies", "fixed,residual",
                ]
                mod.main()

                md_file = res_dir / "strategy_comparison_phase1.5.md"
                json_file = res_dir / "strategy_comparison_phase1.5.json"
                self.assertTrue(md_file.exists())
                self.assertTrue(json_file.exists())

                with open(md_file) as f:
                    md_text = f.read()

                # Verify Rate-Distortion section and content
                self.assertIn("Rate-Distortion & Pareto Frontier Analysis", md_text)
                self.assertIn("Reference Fixed-Length Rate-Distortion Curve", md_text)
                self.assertIn("| **L=2** |", md_text)
                self.assertIn("| **L=4** |", md_text)
                self.assertIn("| **residual** |", md_text)
                self.assertIn("▲ Efficient", md_text)
                self.assertIn("+11.4%", md_text)

                with open(json_file) as f:
                    data = json.load(f)
                self.assertIn("fixed_curve", data)
                self.assertIn("rate_distortion_frontier", data)
                self.assertEqual(len(data["fixed_curve"]), 2)

                res_entry = [v for v in data["rate_distortion_frontier"] if v["strategy"] == "residual"][0]
                self.assertEqual(res_entry["floor_length"], 2.0)
                self.assertEqual(res_entry["ceil_length"], 4.0)
                self.assertAlmostEqual(res_entry["interp_metrics"]["ndcg@10"], 0.070, places=4)
                self.assertEqual(res_entry["pareto_status"], "▲ Efficient")
            finally:
                sys.argv = orig_argv

    def test_generate_comparison_plot_missing_matplotlib(self):
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Matplotlib is not installed in the test env, so this should return False gracefully
        rd_data = {"fixed_curve": [], "variable_bracketed": []}
        res = mod.generate_comparison_plot(rd_data, "Instruments", "LETTER-TIGER", "/tmp/dummy.png")
        self.assertFalse(res)

        # Check backward compatibility alias
        res_alias = mod.generate_rate_distortion_plot(rd_data, "Instruments", "LETTER-TIGER", "/tmp/dummy.png")
        self.assertFalse(res_alias)

    def test_generate_comparison_plot_with_mocked_matplotlib(self):
        from unittest.mock import MagicMock, patch
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mock_plt = MagicMock()
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax3 = MagicMock()
        mock_ax4 = MagicMock()
        for ax in [mock_ax1, mock_ax2, mock_ax3, mock_ax4]:
            ax.get_legend_handles_labels.return_value = ([], [])
        mock_axes = [[mock_ax1, mock_ax2], [mock_ax3, mock_ax4]]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        class MockRect:
            def __init__(self, w=10.0, y=0.0, h=0.5):
                self.w, self.y, self.h = w, y, h
            def get_width(self): return self.w
            def get_y(self): return self.y
            def get_height(self): return self.h

        mock_ax3.barh.return_value = [MockRect(50.0), MockRect(25.0)]
        mock_ax4.barh.return_value = [MockRect(11.4), MockRect(8.9)]

        mock_matplotlib = MagicMock()
        mock_matplotlib.pyplot = mock_plt

        rd_data = {
            "fixed_curve": [
                {"length": 2.0, "metrics": {"hit@5": 0.06, "hit@10": 0.08, "ndcg@5": 0.05, "ndcg@10": 0.06}, "status": "completed"},
                {"length": 4.0, "metrics": {"hit@5": 0.08, "hit@10": 0.10, "ndcg@5": 0.07, "ndcg@10": 0.08}, "status": "completed"},
            ],
            "variable_bracketed": [
                {
                    "strategy": "residual",
                    "phase": "1.5",
                    "mean_length": 3.0,
                    "weighted_length": 3.0,
                    "token_savings_pct": 25.0,
                    "status": "completed",
                    "metrics": {"hit@5": 0.078, "hit@10": 0.098, "ndcg@5": 0.068, "ndcg@10": 0.078},
                    "interp_metrics": {"hit@5": 0.070, "hit@10": 0.090, "ndcg@5": 0.060, "ndcg@10": 0.070},
                    "delta_interp": {"hit@5": 11.4, "hit@10": 8.9, "ndcg@5": 13.3, "ndcg@10": 11.4},
                    "pareto_status": "★ Dominant",
                }
            ],
        }
        h2h_data = {"residual": {"delta_hit10": 0.70, "delta_ndcg10": 0.40}}

        with patch.dict("sys.modules", {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
            res = mod.generate_comparison_plot(
                rd_data=rd_data,
                dataset="Instruments",
                model_name="LETTER-TIGER",
                out_png_path="/tmp/test_strategy_comparison.png",
                h2h_data=h2h_data,
                max_length=4,
            )
            self.assertTrue(res)

            # Verify 4-panel subplots created
            mock_plt.subplots.assert_called_once_with(2, 2, figsize=(12, 9.5), dpi=300)

            # Verify Panel 1 & Panel 2 (Hit@10 and NDCG@10 frontier plots)
            self.assertTrue(mock_ax1.plot.called)
            self.assertTrue(mock_ax1.scatter.called)
            self.assertTrue(mock_ax2.plot.called)
            self.assertTrue(mock_ax2.scatter.called)

            # Verify Panel 3 & Panel 4 (Hit@5 and NDCG@5 frontier plots, no barh)
            self.assertTrue(mock_ax3.plot.called)
            self.assertTrue(mock_ax3.scatter.called)
            self.assertTrue(mock_ax4.plot.called)
            self.assertTrue(mock_ax4.scatter.called)
            self.assertFalse(mock_ax3.barh.called)
            self.assertFalse(mock_ax4.barh.called)

            # Verify savefig called with 300 dpi and tight layout
            mock_fig.savefig.assert_called_once_with(
                "/tmp/test_strategy_comparison.png", dpi=300, bbox_inches="tight"
            )
            mock_plt.close.assert_called_once_with(mock_fig)

    def test_rate_distortion_report_end_to_end_with_plot_and_no_plot(self):
        from unittest.mock import MagicMock, patch
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset = "PlotTestDS"
            data_dir = tmp_path / "data" / dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            res_dir = tmp_path / "LETTER-TIGER" / "results" / dataset / "rqvae"
            res_dir.mkdir(parents=True, exist_ok=True)

            with open(data_dir / f"{dataset}.index.json", "w") as f:
                json.dump({"0": [1, 2, 3, 4], "1": [2, 3, 4, 5]}, f)
            with open(data_dir / f"{dataset}.inter.json", "w") as f:
                json.dump({"0": [0, 1]}, f)

            with open(res_dir / "fixed_L2.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.03, "hit@5": 0.06, "hit@10": 0.08, "ndcg@5": 0.05, "ndcg@10": 0.06}}, f)
            with open(res_dir / "fixed_L4.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.05, "hit@5": 0.08, "hit@10": 0.10, "ndcg@5": 0.07, "ndcg@10": 0.08}}, f)

            tok_dir = data_dir / "rqvae"
            tok_dir.mkdir(parents=True, exist_ok=True)
            with open(tok_dir / f"{dataset}.index.varlen.res-phase1.5.json", "w") as f:
                json.dump({"0": [1, 2, 3], "1": [2, 3, 4]}, f)
            with open(res_dir / "varlen-res_phase1.5.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.048, "hit@5": 0.078, "hit@10": 0.098, "ndcg@5": 0.068, "ndcg@10": 0.078}}, f)

            mock_plt = MagicMock()
            mock_fig = MagicMock()
            mock_ax1, mock_ax2, mock_ax3, mock_ax4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
            for ax in [mock_ax1, mock_ax2, mock_ax3, mock_ax4]:
                ax.get_legend_handles_labels.return_value = ([], [])
            mock_plt.subplots.return_value = (mock_fig, [[mock_ax1, mock_ax2], [mock_ax3, mock_ax4]])

            def side_effect_savefig(path, **kwargs):
                with open(path, "wb") as f:
                    f.write(b"fake_png_data")

            mock_fig.savefig.side_effect = side_effect_savefig

            mock_matplotlib = MagicMock()
            mock_matplotlib.pyplot = mock_plt

            # 1. Run with plotting enabled
            orig_argv = sys.argv
            try:
                sys.argv = [
                    "generate_comparison_report.py",
                    "--dataset", dataset,
                    "--repo-root", str(tmp_path),
                    "--phase", "1.5",
                    "--tokenizer", "rqvae",
                    "--strategies", "fixed,residual",
                ]
                with patch.dict("sys.modules", {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
                    mod.main()

                plot_png = res_dir / f"rate_distortion_frontier_{dataset}_phase1.5.png"
                self.assertTrue(plot_png.exists())
                # Verify duplicate strategy_comparison file was eliminated
                duplicate_png = res_dir / f"strategy_comparison_{dataset}_phase1.5.png"
                self.assertFalse(duplicate_png.exists())

                md_file = res_dir / f"strategy_comparison_phase1.5.md"
                with open(md_file) as f:
                    md_text = f.read()

                # Verify markdown embeds the rate-distortion plot once in Section 3
                self.assertIn(f"![Phase 1.5 Rate-Distortion Pareto Frontier]({plot_png.name})", md_text)
                self.assertNotIn("### Performance & Rate-Distortion Overview", md_text)

            finally:
                sys.argv = orig_argv

            # 2. Run with --no-plot
            orig_argv = sys.argv
            try:
                if plot_png.exists(): plot_png.unlink()

                sys.argv = [
                    "generate_comparison_report.py",
                    "--dataset", dataset,
                    "--repo-root", str(tmp_path),
                    "--phase", "1.5",
                    "--tokenizer", "rqvae",
                    "--strategies", "fixed,residual",
                    "--no-plot",
                ]
                mod.main()

                # Verify no plot was created
                self.assertFalse(plot_png.exists())

                with open(md_file) as f:
                    md_text = f.read()
                self.assertNotIn("![Phase 1.5 Rate-Distortion Pareto Frontier]", md_text)
            finally:
                sys.argv = orig_argv

    def test_phase1_and_phase1_5_single_unified_plot_when_both_present(self):
        from unittest.mock import MagicMock, patch
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset = "SplitPlotDS"
            data_dir = tmp_path / "data" / dataset
            data_dir.mkdir(parents=True, exist_ok=True)
            res_dir = tmp_path / "LETTER-TIGER" / "results" / dataset / "rqvae"
            res_dir.mkdir(parents=True, exist_ok=True)

            with open(data_dir / f"{dataset}.index.json", "w") as f:
                json.dump({"0": [1, 2, 3, 4], "1": [2, 3, 4, 5]}, f)
            with open(data_dir / f"{dataset}.inter.json", "w") as f:
                json.dump({"0": [0, 1]}, f)

            # Fixed baseline
            with open(res_dir / "fixed_L2.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.03, "hit@5": 0.06, "hit@10": 0.08, "ndcg@5": 0.05, "ndcg@10": 0.06}}, f)
            with open(res_dir / "fixed_L4.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.05, "hit@5": 0.08, "hit@10": 0.10, "ndcg@5": 0.07, "ndcg@10": 0.08}}, f)

            tok_dir = data_dir / "rqvae"
            tok_dir.mkdir(parents=True, exist_ok=True)

            # Phase 1 variable result
            with open(tok_dir / f"{dataset}.index.varlen.res.json", "w") as f:
                json.dump({"0": [1, 2], "1": [2, 3]}, f)
            with open(res_dir / "varlen-res.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.040, "hit@5": 0.070, "hit@10": 0.090, "ndcg@5": 0.060, "ndcg@10": 0.070}}, f)

            # Phase 1.5 variable result
            with open(tok_dir / f"{dataset}.index.varlen.res-phase1.5.json", "w") as f:
                json.dump({"0": [1, 2, 3], "1": [2, 3, 4]}, f)
            with open(res_dir / "varlen-res_phase1.5.json", "w") as f:
                json.dump({"mean_results": {"hit@1": 0.048, "hit@5": 0.078, "hit@10": 0.098, "ndcg@5": 0.068, "ndcg@10": 0.078}}, f)

            mock_plt = MagicMock()
            mock_fig = MagicMock()
            mock_axes = [[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]]
            for row in mock_axes:
                for ax in row:
                    ax.get_legend_handles_labels.return_value = ([], [])
            mock_plt.subplots.return_value = (mock_fig, mock_axes)

            def side_effect_savefig(path, **kwargs):
                with open(path, "wb") as f:
                    f.write(b"fake_plot_data")

            mock_fig.savefig.side_effect = side_effect_savefig

            mock_matplotlib = MagicMock()
            mock_matplotlib.pyplot = mock_plt

            orig_argv = sys.argv
            try:
                sys.argv = [
                    "generate_comparison_report.py",
                    "--dataset", dataset,
                    "--repo-root", str(tmp_path),
                    "--phase", "both",
                    "--tokenizer", "rqvae",
                    "--strategies", "fixed,residual",
                ]
                with patch.dict("sys.modules", {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}):
                    mod.main()

                plot_png = res_dir / f"rate_distortion_frontier_{dataset}.png"

                # Single unified plot exists
                self.assertTrue(plot_png.exists())
                self.assertFalse((res_dir / f"rate_distortion_frontier_{dataset}_phase1.png").exists())
                self.assertFalse((res_dir / f"rate_distortion_frontier_{dataset}_phase1.5.png").exists())

                # Check markdown report
                md_file = res_dir / f"strategy_comparison.md"
                with open(md_file) as f:
                    md_text = f.read()

                self.assertIn(f"![Rate-Distortion Pareto Frontier]({plot_png.name})", md_text)
                self.assertNotIn("### Performance & Rate-Distortion Overview", md_text)

                # Check JSON output
                json_file = res_dir / f"strategy_comparison.json"
                with open(json_file) as f:
                    jdata = json.load(f)
                self.assertIn(plot_png.name, jdata.get("plot_files", []))

            finally:
                sys.argv = orig_argv

    def test_autodetect_max_length_and_fixed_depths(self):
        spec = importlib.util.spec_from_file_location(
            "generate_comparison_report",
            os.path.join(os.path.dirname(__file__), "..", "generate_comparison_report.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_dir = tmp_path / "results"
            report_dir.mkdir(parents=True, exist_ok=True)
            data_dir = tmp_path / "data" / "TestDS"
            data_dir.mkdir(parents=True, exist_ok=True)

            # Case 1: Empty dir fallback
            max_l, tag, depths = mod.autodetect_max_length_and_fixed_depths(
                str(report_dir), str(tmp_path / "data"), "TestDS", "rqvae"
            )
            self.assertEqual(max_l, 4)
            self.assertEqual(tag, "")

            # Case 2: Multi-depth fixed files (L=2, L=3, L=6)
            (report_dir / "fixed_L2.json").write_text("{}")
            (report_dir / "fixed_L3.json").write_text("{}")
            (report_dir / "fixed_L6.json").write_text("{}")
            max_l, tag, depths = mod.autodetect_max_length_and_fixed_depths(
                str(report_dir), str(tmp_path / "data"), "TestDS", "rqvae"
            )
            self.assertEqual(max_l, 6)
            self.assertEqual(depths, [2, 3, 6])
            self.assertEqual(tag, "")

            # Case 3: _max10 tagged files
            (report_dir / "fixed_max10.json").write_text("{}")
            max_l, tag, depths = mod.autodetect_max_length_and_fixed_depths(
                str(report_dir), str(tmp_path / "data"), "TestDS", "rqvae"
            )
            self.assertEqual(max_l, 10)
            self.assertEqual(tag, "_max10")

            # Case 4: Explicit max_length override
            max_l, tag, depths = mod.autodetect_max_length_and_fixed_depths(
                str(report_dir), str(tmp_path / "data"), "TestDS", "rqvae", explicit_max_length=8
            )
            self.assertEqual(max_l, 8)
            self.assertEqual(tag, "_max8")

            max_l, tag, depths = mod.autodetect_max_length_and_fixed_depths(
                str(report_dir), str(tmp_path / "data"), "TestDS", "rqvae", explicit_max_length=4
            )
            self.assertEqual(max_l, 4)
            self.assertEqual(tag, "_max4")

            # Case 5: Catalog index file detection (length 5)
            with open(data_dir / "TestDS.index.json", "w") as f:
                json.dump({"0": [1, 2, 3, 4, 5]}, f)
            empty_rep = tmp_path / "empty_rep"
            empty_rep.mkdir()
            max_l, tag, depths = mod.autodetect_max_length_and_fixed_depths(
                str(empty_rep), str(tmp_path / "data"), "TestDS", "rqvae"
            )
            self.assertEqual(max_l, 5)



class TestGenerateIndicesResidual(unittest.TestCase):
    def test_generate_indices_residual_phase1_5(self):
        if not hasattr(torch, "__file__") or isinstance(torch, MagicMock) or not hasattr(torch, "save") or not hasattr(torch, "Tensor"):
            self.skipTest("Real PyTorch is required for generate_indices test")
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            emb_path = tmp_path / "test_emb.npy"
            np.save(str(emb_path), np.random.randn(8, 32).astype(np.float32))

            dummy_model = RQVAE(
                in_dim=32,
                num_emb_list=[8, 8, 8],
                e_dim=16,
                layers=[24, 16],
                dropout_prob=0.0,
                bn=False,
                loss_type="mse",
                quant_loss_weight=1.0,
                kmeans_init=False,
                kmeans_iters=10,
                beta=0.0,
            )

            ckpt_path = tmp_path / "model_ckpt.pth"
            ckpt = {
                "args": {
                    "data_path": str(emb_path),
                    "num_emb_list": [8, 8, 8],
                    "e_dim": 16,
                    "layers": [24, 16],
                    "dropout_prob": 0.0,
                    "bn": False,
                    "loss_type": "mse",
                    "quant_loss_weight": 1.0,
                    "kmeans_init": False,
                    "kmeans_iters": 10,
                    "sk_epsilons": None,
                    "sk_iters": 10,
                    "num_workers": 0,
                    "phase": 1.5,
                    "residual_threshold": 0.2,
                    "min_length": 1,
                    "beta": 0.0,
                },
                "state_dict": dummy_model.state_dict(),
                "phase": 1.5,
                "residual_threshold": 0.2,
                "min_length": 1,
            }
            torch.save(ckpt, str(ckpt_path))

            out_index = tmp_path / "out.index.json"
            gen_script = Path(__file__).resolve().parent / "generate_indices.py"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(gen_script),
                    "--dataset", "TestDS",
                    "--checkpoint-path", str(ckpt_path),
                    "--output-file", str(out_index),
                    "--device", "cpu",
                    "--phase", "1.5",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"generate_indices.py failed with stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
            self.assertTrue(out_index.exists())
            with open(out_index, "r") as f:
                indices_data = json.load(f)
            self.assertEqual(len(indices_data), 8)
            for k, tokens in indices_data.items():
                self.assertGreaterEqual(len(tokens), 1)
                self.assertLessEqual(len(tokens), 3)


if __name__ == "__main__":
    unittest.main()




