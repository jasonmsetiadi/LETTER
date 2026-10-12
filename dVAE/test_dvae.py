import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DVAE_DIR = str(Path(__file__).resolve().parent)
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
RQ_DIR = os.path.join(REPO_ROOT, "RQ-VAE")

for p in [DVAE_DIR, REPO_ROOT, RQ_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from models.gpt import GPT, norm
from models.layers import Encoder, Decoder, InferenceEncoder, InferenceDecoder
from models.game import Game, VarlenGame
from datasets import EmbDataset
from utils import disambiguate_indices


class TestDVAEComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.embed_dim = 32
        self.hidden_size = 64
        self.vocab_size = 128
        self.maxlen = 4
        self.batch_size = 8

    def test_gpt_forward(self):
        gpt = GPT(
            num_layers=2,
            num_heads=2,
            model_dim=self.hidden_size,
            max_seq_len=self.maxlen,
            causal=True,
        )
        x = torch.randn(self.batch_size, self.maxlen, self.hidden_size)
        out = gpt(x)
        self.assertEqual(out.shape, (self.batch_size, self.maxlen, self.hidden_size))

    def test_fixed_encoder_and_decoder(self):
        encoder = Encoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            maxlen=self.maxlen,
            varlen=False,
        )
        decoder = Decoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            maxlen=self.maxlen,
        )
        x = torch.randn(self.batch_size, self.embed_dim)
        logits, gs_probs = encoder(x, tau=1.0)

        self.assertEqual(logits.shape, (self.batch_size, self.maxlen, self.vocab_size))
        self.assertEqual(gs_probs.shape, (self.batch_size, self.maxlen, self.vocab_size))

        recon = decoder(gs_probs)
        self.assertEqual(recon.shape, (self.batch_size, self.maxlen, self.embed_dim))

        # Test inference
        enc_inf = encoder.inference(mode="argmax")
        dec_inf = decoder.inference()
        codes = enc_inf(x)
        self.assertEqual(codes.shape, (self.batch_size, self.maxlen))
        recon_inf = dec_inf(codes)
        self.assertEqual(recon_inf.shape, (self.batch_size, self.maxlen, self.embed_dim))

    def test_varlen_encoder_and_survival(self):
        encoder = Encoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            maxlen=self.maxlen,
            varlen=True,
        )
        x = torch.randn(self.batch_size, self.embed_dim)
        logits, gs_probs, length_logits, survival_logits = encoder(x, tau=1.0)

        self.assertEqual(length_logits.shape, (self.batch_size, self.maxlen))
        self.assertEqual(survival_logits.shape, (self.batch_size, self.maxlen))

        # Verify length distribution axioms
        length_probs = length_logits.softmax(dim=-1)
        self.assertTrue(torch.allclose(length_probs.sum(dim=-1), torch.ones(self.batch_size), atol=1e-5))

        # Test all three inference length modes: argmax, mean, median
        for mode in ["argmax", "mean", "median"]:
            enc_inf = encoder.inference(mode=mode)
            codes, lengths = enc_inf(x)
            self.assertEqual(codes.shape, (self.batch_size, self.maxlen))
            self.assertEqual(lengths.shape, (self.batch_size,))
            self.assertTrue(torch.all(lengths >= 1))
            self.assertTrue(torch.all(lengths <= self.maxlen))

    def test_game_loss_and_gradients(self):
        encoder = Encoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            maxlen=self.maxlen,
            varlen=False,
        )
        decoder = Decoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            maxlen=self.maxlen,
        )
        game = Game(encoder, decoder)
        x = torch.randn(self.batch_size, self.embed_dim)
        loss, metrics = game(x, tau=1.0, beta=torch.tensor(0.01))

        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        # Verify gradients reached codebook
        self.assertIsNotNone(encoder.codebook.grad)
        self.assertFalse(torch.isnan(encoder.codebook.grad).any())

    def test_varlen_game_loss_and_gradients(self):
        encoder = Encoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            maxlen=self.maxlen,
            varlen=True,
        )
        decoder = Decoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            maxlen=self.maxlen,
        )
        game = VarlenGame(encoder, decoder, alpha=0.1)
        x = torch.randn(self.batch_size, self.embed_dim)
        cf_emb = torch.randn(self.batch_size, self.embed_dim)

        loss, metrics = game(
            x,
            tau=1.0,
            length_cost=torch.tensor(0.05),
            beta=torch.tensor(0.01),
            cf_emb=cf_emb,
        )

        self.assertGreater(loss.item(), 0.0)
        self.assertIn("expected_len", metrics)
        self.assertIn("recon_loss", metrics)
        self.assertIn("cf_loss", metrics)

        loss.backward()
        # Verify gradients reached stop predictor and codebook
        self.assertIsNotNone(encoder.stop_predictor.weight.grad)
        self.assertIsNotNone(encoder.codebook.grad)
        self.assertFalse(torch.isnan(encoder.stop_predictor.weight.grad).any())

    def test_dataset_and_disambiguation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            arr = np.random.randn(20, self.embed_dim).astype(np.float32)
            emb_file = os.path.join(tmpdir, "test.emb.npy")
            np.save(emb_file, arr)

            ds = EmbDataset(emb_file)
            self.assertEqual(len(ds), 20)
            self.assertEqual(ds.dim, self.embed_dim)

            # Test disambiguation
            sample_indices = {
                "0": ["<a_1>", "<b_2>"],
                "1": ["<a_1>", "<b_2>"],  # collision with 0
                "2": ["<a_3>", "<b_4>"],  # unique
            }
            resolved = disambiguate_indices(sample_indices)
            self.assertEqual(resolved["0"], ["<a_1>", "<b_2>", "<dis_1>"])
            self.assertEqual(resolved["1"], ["<a_1>", "<b_2>", "<dis_2>"])
            self.assertEqual(resolved["2"], ["<a_3>", "<b_4>", "<dis_0>"])


if __name__ == "__main__":
    unittest.main()

