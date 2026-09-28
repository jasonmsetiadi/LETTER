import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

RQ_DIR = str(Path(__file__).resolve().parent)
if RQ_DIR not in sys.path:
    sys.path.insert(0, RQ_DIR)

# If torch is not installed in the testing environment, provide a minimal mock
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = MagicMock()
    nn = MagicMock()
    nn.Module = object
    nn.ModuleList = list
    torch.nn = nn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = MagicMock()
    init_mock = MagicMock()
    sys.modules["torch.nn.init"] = init_mock
    nn.init = init_mock

try:
    import sklearn
except ImportError:
    sklearn = MagicMock()
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.cluster"] = MagicMock()

try:
    import numpy as np
except ImportError:
    np = MagicMock()
    sys.modules["numpy"] = np

from models.rq import ResidualVectorQuantizer
from models.rqvae import RQVAE


class TestResidualVectorQuantizerInit(unittest.TestCase):
    def test_default_sk_epsilons_none(self):
        """ResidualVectorQuantizer should handle sk_epsilons=None gracefully."""
        # Mock VectorQuantizer
        orig_vq = sys.modules["models.rq"].VectorQuantizer
        sys.modules["models.rq"].VectorQuantizer = MagicMock()
        try:
            rvq = ResidualVectorQuantizer([256, 256, 256, 256], 64, sk_epsilons=None)
            self.assertEqual(rvq.sk_epsilons, [0.0, 0.0, 0.0, 0.003])

            # Also without passing sk_epsilons arg
            rvq2 = ResidualVectorQuantizer([256, 256], 64)
            self.assertEqual(rvq2.sk_epsilons, [0.0, 0.003])
        finally:
            sys.modules["models.rq"].VectorQuantizer = orig_vq

    def test_scalar_and_mismatched_sk_epsilons(self):
        orig_vq = sys.modules["models.rq"].VectorQuantizer
        sys.modules["models.rq"].VectorQuantizer = MagicMock()
        try:
            # Scalar
            rvq = ResidualVectorQuantizer([256, 256], 64, sk_epsilons=0.01)
            self.assertEqual(rvq.sk_epsilons, [0.01, 0.01])

            # Too short list
            rvq2 = ResidualVectorQuantizer([256, 256, 256], 64, sk_epsilons=[0.01])
            self.assertEqual(rvq2.sk_epsilons, [0.01, 0.0, 0.0])
        finally:
            sys.modules["models.rq"].VectorQuantizer = orig_vq

    def test_rqvae_init_default_sk_epsilons(self):
        orig_vq = sys.modules["models.rq"].VectorQuantizer
        orig_mlp = sys.modules["models.rqvae"].MLPLayers
        sys.modules["models.rq"].VectorQuantizer = MagicMock()
        sys.modules["models.rqvae"].MLPLayers = MagicMock()
        try:
            model = RQVAE(
                in_dim=64,
                num_emb_list=[256, 256, 256, 256],
                e_dim=64,
                layers=[64],
            )
            self.assertEqual(model.sk_epsilons, [0.0, 0.0, 0.0, 0.003])
            self.assertEqual(model.rq.sk_epsilons, [0.0, 0.0, 0.0, 0.003])
        finally:
            sys.modules["models.rq"].VectorQuantizer = orig_vq
            sys.modules["models.rqvae"].MLPLayers = orig_mlp

    def test_checkpoint_args_compatibility(self):
        """Simulate compute_residuals model initialization with different ckpt_args."""
        import argparse

        orig_vq = sys.modules["models.rq"].VectorQuantizer
        orig_mlp = sys.modules["models.rqvae"].MLPLayers
        sys.modules["models.rq"].VectorQuantizer = MagicMock()
        sys.modules["models.rqvae"].MLPLayers = MagicMock()

        try:
            # Case 1: Namespace without sk_epsilons
            ns_args = argparse.Namespace(
                num_emb_list=[256, 256, 256],
                e_dim=64,
                layers=[64],
                dropout_prob=0.0,
                bn=False,
                loss_type="mse",
                quant_loss_weight=1.0,
            )
            def get_arg_ns(name, default=None):
                return getattr(ns_args, name, default)

            m1 = RQVAE(
                in_dim=64,
                num_emb_list=get_arg_ns("num_emb_list"),
                e_dim=get_arg_ns("e_dim"),
                layers=get_arg_ns("layers"),
                dropout_prob=get_arg_ns("dropout_prob", 0.0),
                bn=get_arg_ns("bn", False),
                loss_type=get_arg_ns("loss_type", "mse"),
                quant_loss_weight=get_arg_ns("quant_loss_weight", 1.0),
                kmeans_init=get_arg_ns("kmeans_init", False),
                kmeans_iters=get_arg_ns("kmeans_iters", 100),
                sk_epsilons=get_arg_ns("sk_epsilons", None),
                sk_iters=get_arg_ns("sk_iters", 100),
            )
            self.assertEqual(m1.rq.sk_epsilons, [0.0, 0.0, 0.003])

            # Case 2: Dict with explicit sk_epsilons
            dict_args = {
                "num_emb_list": [256, 256],
                "e_dim": 64,
                "layers": [64],
                "sk_epsilons": [0.001, 0.002],
            }
            def get_arg_dict(name, default=None):
                return dict_args.get(name, default)

            m2 = RQVAE(
                in_dim=64,
                num_emb_list=get_arg_dict("num_emb_list"),
                e_dim=get_arg_dict("e_dim"),
                layers=get_arg_dict("layers"),
                dropout_prob=get_arg_dict("dropout_prob", 0.0),
                bn=get_arg_dict("bn", False),
                loss_type=get_arg_dict("loss_type", "mse"),
                quant_loss_weight=get_arg_dict("quant_loss_weight", 1.0),
                kmeans_init=get_arg_dict("kmeans_init", False),
                kmeans_iters=get_arg_dict("kmeans_iters", 100),
                sk_epsilons=get_arg_dict("sk_epsilons", None),
                sk_iters=get_arg_dict("sk_iters", 100),
            )
            self.assertEqual(m2.rq.sk_epsilons, [0.001, 0.002])
        finally:
            sys.modules["models.rq"].VectorQuantizer = orig_vq
            sys.modules["models.rqvae"].MLPLayers = orig_mlp


if __name__ == "__main__":
    unittest.main()
