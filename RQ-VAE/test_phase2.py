import unittest
import os
import sys
import json
from pathlib import Path

RQ_DIR = str(Path(__file__).resolve().parent)
if RQ_DIR not in sys.path:
    sys.path.insert(0, RQ_DIR)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestPhase2DiscreteMath(unittest.TestCase):
    """Pure-python verification of Phase 2 mathematical logic (survival, rate, entropy)."""

    def test_survival_probability_math(self):
        # Suppose stop probabilities over lengths [1, 2, 3, 4] are:
        # q(L) = [0.1, 0.4, 0.3, 0.2]
        q = [0.1, 0.4, 0.3, 0.2]
        # P(L >= l) = sum_{j=l}^L q_j
        # P(L >= 1) = 0.1 + 0.4 + 0.3 + 0.2 = 1.0
        # P(L >= 2) = 0.4 + 0.3 + 0.2 = 0.9
        # P(L >= 3) = 0.3 + 0.2 = 0.5
        # P(L >= 4) = 0.2
        expected_survival = [1.0, 0.9, 0.5, 0.2]

        # Compute survival using reverse cumsum (same algorithm as PyTorch tensor)
        rev_q = list(reversed(q))
        cum_rev = []
        running = 0.0
        for val in rev_q:
            running += val
            cum_rev.append(running)
        computed_survival = list(reversed(cum_rev))

        for c, e in zip(computed_survival, expected_survival):
            self.assertAlmostEqual(c, e, places=5)

    def test_expected_length_and_entropy_math(self):
        import math
        q = [0.1, 0.4, 0.3, 0.2]
        # E[L] = 1*0.1 + 2*0.4 + 3*0.3 + 4*0.2 = 0.1 + 0.8 + 0.9 + 0.8 = 2.6
        exp_len = sum(idx * prob for idx, prob in enumerate(q, start=1))
        self.assertAlmostEqual(exp_len, 2.6, places=5)

        # Entropy H(q) = - sum q_i log(q_i)
        entropy = -sum(p * math.log(p) for p in q)
        self.assertGreater(entropy, 0.0)

        # Rate loss = length_cost * E[L] - length_entropy_weight * H(q)
        gamma = 0.05
        lam = 0.01
        rate_loss = gamma * exp_len - lam * entropy
        self.assertAlmostEqual(rate_loss, 0.05 * 2.6 - 0.01 * entropy, places=5)


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "PyTorch and NumPy required for Phase 2 neural tests")
class TestPhase2StopPredictor(unittest.TestCase):
    def test_stop_predictor_forward_and_properties(self):
        from models.layers import StopPredictor
        torch.manual_seed(42)
        bsz = 16
        in_dim = 32
        max_depth = 4
        predictor = StopPredictor(in_dim=in_dim, max_depth=max_depth, hidden_dim=64)

        x = torch.randn(bsz, in_dim)
        logits, probs = predictor(x)

        # Check shapes
        self.assertEqual(logits.shape, (bsz, max_depth))
        self.assertEqual(probs.shape, (bsz, max_depth))

        # Check probability axioms
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(bsz), atol=1e-5))

    def test_stop_predictor_gradient_flow(self):
        from models.layers import StopPredictor
        torch.manual_seed(42)
        in_dim = 16
        max_depth = 4
        predictor = StopPredictor(in_dim=in_dim, max_depth=max_depth)
        x = torch.randn(8, in_dim)

        logits, probs = predictor(x)
        loss = (probs * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum()
        loss.backward()

        self.assertIsNotNone(predictor.net[0].weight.grad)
        self.assertGreater(predictor.net[0].weight.grad.abs().sum().item(), 0.0)


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "PyTorch and NumPy required for Phase 2 neural tests")
class TestPhase2QuantizerPrefixes(unittest.TestCase):
    def test_return_prefixes(self):
        from models.rq import ResidualVectorQuantizer
        torch.manual_seed(42)
        bsz = 8
        e_dim = 16
        n_e_list = [32, 32, 32, 32]
        rq = ResidualVectorQuantizer(n_e_list=n_e_list, e_dim=e_dim, sk_epsilons=[0.0] * 4)

        x = torch.randn(bsz, e_dim)
        labels = {str(i): [] for i in range(len(n_e_list))}

        stacked_xq, all_losses, all_indices = rq(x, labels, return_prefixes=True, use_sk=False)

        self.assertEqual(stacked_xq.shape, (bsz, 4, e_dim))
        self.assertEqual(len(all_losses), 4)
        self.assertEqual(all_indices.shape, (bsz, 4))

        x_q, mean_loss, indices = rq(x, labels, return_prefixes=False, use_sk=False)
        self.assertEqual(x_q.shape, (bsz, e_dim))
        self.assertEqual(indices.shape, (bsz, 4))


@unittest.skipIf(not (HAS_TORCH and HAS_NUMPY), "PyTorch and NumPy required for Phase 2 neural tests")
class TestPhase2RQVAE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_dim = 64
        self.num_emb_list = [16, 16, 16, 16]
        self.e_dim = 16
        self.layers = [32]
        self.bsz = 10

    def test_model_initialization_and_forward(self):
        from models.rqvae import RQVAE
        model = RQVAE(
            in_dim=self.in_dim,
            num_emb_list=self.num_emb_list,
            e_dim=self.e_dim,
            layers=self.layers,
            sk_epsilons=[0.0] * 4,
            phase=2.0,
            length_cost=0.05,
            length_entropy_weight=0.01,
        )
        self.assertIsNotNone(model.stop_predictor)

        x = torch.randn(self.bsz, self.in_dim)
        labels = {str(i): [] for i in range(4)}

        out, rq_losses, indices, dense_out, stop_probs, survival_probs, pred_lengths = model(
            x, labels, use_sk=False, return_lengths=True
        )

        self.assertEqual(out.shape, (self.bsz, 4, self.in_dim))
        self.assertEqual(dense_out.shape, (self.bsz, 4, self.e_dim))
        self.assertEqual(stop_probs.shape, (self.bsz, 4))
        self.assertEqual(survival_probs.shape, (self.bsz, 4))
        self.assertEqual(pred_lengths.shape, (self.bsz,))

        self.assertTrue(torch.allclose(survival_probs[:, 0], torch.ones(self.bsz), atol=1e-5))
        self.assertTrue(torch.allclose(survival_probs[:, -1], stop_probs[:, -1], atol=1e-5))
        for l in range(3):
            self.assertTrue(torch.all(survival_probs[:, l] >= survival_probs[:, l + 1] - 1e-6))

        expected_pred = torch.argmax(stop_probs, dim=-1) + 1
        self.assertTrue(torch.equal(pred_lengths, expected_pred))

    def test_phase2_loss_and_gradient_flow_vanilla(self):
        from models.rqvae import RQVAE
        model = RQVAE(
            in_dim=self.in_dim,
            num_emb_list=self.num_emb_list,
            e_dim=self.e_dim,
            layers=self.layers,
            sk_epsilons=[0.0] * 4,
            phase=2.0,
            alpha=0.0,
            length_cost=0.1,
            length_entropy_weight=0.01,
        )

        x = torch.randn(self.bsz, self.in_dim)
        labels = {str(i): [] for i in range(4)}
        emb_idx = torch.arange(self.bsz)

        out, rq_losses, indices, dense_out, stop_probs, survival_probs = model(
            x, labels, use_sk=False, return_lengths=False
        )

        total_loss, cf_loss, recon_loss, quant_loss, exp_len, entropy = model.compute_loss(
            out,
            rq_losses,
            emb_idx,
            dense_out,
            xs=x,
            stop_probs=stop_probs,
            survival_probs=survival_probs,
        )

        self.assertEqual(cf_loss.item(), 0.0)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertGreater(recon_loss.item(), 0.0)
        self.assertTrue(1.0 <= exp_len.item() <= 4.0)

        total_loss.backward()

        stop_pred_weight_grad = model.stop_predictor.net[0].weight.grad
        self.assertIsNotNone(stop_pred_weight_grad)
        self.assertGreater(stop_pred_weight_grad.abs().sum().item(), 0.0)

    def test_phase2_loss_and_gradient_flow_letter(self):
        from models.rqvae import RQVAE
        num_items = 20
        cf_embs = np.random.randn(num_items, self.e_dim).astype(np.float32)

        model = RQVAE(
            in_dim=self.in_dim,
            num_emb_list=self.num_emb_list,
            e_dim=self.e_dim,
            layers=self.layers,
            sk_epsilons=[0.0] * 4,
            phase=2.0,
            alpha=0.5,
            length_cost=0.05,
            length_entropy_weight=0.01,
            cf_embedding=cf_embs,
        )

        x = torch.randn(self.bsz, self.in_dim)
        labels = {str(i): [] for i in range(4)}
        emb_idx = torch.arange(self.bsz)

        out, rq_losses, indices, dense_out, stop_probs, survival_probs = model(
            x, labels, use_sk=False, return_lengths=False
        )

        total_loss, cf_loss, recon_loss, quant_loss, exp_len, entropy = model.compute_loss(
            out,
            rq_losses,
            emb_idx,
            dense_out,
            xs=x,
            stop_probs=stop_probs,
            survival_probs=survival_probs,
        )

        self.assertGreater(cf_loss.item(), 0.0)
        self.assertGreater(total_loss.item(), 0.0)

        total_loss.backward()

        stop_pred_weight_grad = model.stop_predictor.net[0].weight.grad
        self.assertIsNotNone(stop_pred_weight_grad)
        self.assertGreater(stop_pred_weight_grad.abs().sum().item(), 0.0)

    def test_phase2_get_indices(self):
        from models.rqvae import RQVAE
        model = RQVAE(
            in_dim=self.in_dim,
            num_emb_list=self.num_emb_list,
            e_dim=self.e_dim,
            layers=self.layers,
            sk_epsilons=[0.0] * 4,
            phase=2.0,
        )
        model.eval()

        x = torch.randn(self.bsz, self.in_dim)
        labels = {str(i): [] for i in range(4)}

        indices, pred_lens = model.get_indices(x, labels, use_sk=False, return_lengths=True)

        self.assertEqual(indices.shape, (self.bsz, 4))
        self.assertEqual(pred_lens.shape, (self.bsz,))
        self.assertTrue(torch.all(pred_lens >= 1))
        self.assertTrue(torch.all(pred_lens <= 4))


if __name__ == '__main__':
    unittest.main()

