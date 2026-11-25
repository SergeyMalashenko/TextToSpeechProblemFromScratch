"""Small regression examples for the Transformer teaching code."""
import unittest

import torch

from tts_tacotron_losses import guided_attention_loss, masked_l1_loss
from tts_transformer_config import TransformerTacotronConfig
from tts_transformer_train import compute_gate_metrics, compute_attention_path_metrics, diagnostic_inference
from tts_dataset import LengthBucketBatchSampler


class TransformerToolsTests(unittest.TestCase):
    def test_guided_attention_fp16_matches_fp32(self):
        attention = torch.full((1, 400, 200), 1 / 200)
        lengths = torch.tensor([200]), torch.tensor([400])
        expected = guided_attention_loss(attention, *lengths, sigma=0.45)
        half = attention.half().requires_grad_()
        actual = guided_attention_loss(half, *lengths, sigma=0.45)
        self.assertGreater(actual.item(), 0)
        torch.testing.assert_close(actual, expected, rtol=0.001, atol=1e-7)
        actual.backward()
        self.assertTrue(torch.isfinite(half.grad).all())
        self.assertGreater(half.grad.abs().sum().item(), 0)

    def test_mel_l1_fp16_does_not_overflow(self):
        pred = torch.ones(2, 500, 80, dtype=torch.float16, requires_grad=True)
        loss = masked_l1_loss(pred, torch.zeros_like(pred), torch.tensor([500, 500]))
        self.assertEqual(loss.item(), 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_gate_metrics_ignore_padding(self):
        lengths = torch.tensor([3])
        expected = compute_gate_metrics(torch.tensor([[-10., -10., -10.]]),
                                        torch.tensor([[0., 0., 1.]]), lengths, threshold=0.7)
        actual = compute_gate_metrics(torch.tensor([[-10., -10., -10., 10., 10.]]),
                                      torch.tensor([[0., 0., 1., 1., 1.]]), lengths, threshold=0.7)
        self.assertEqual(actual, expected)

    def test_gate_threshold_is_explicit(self):
        logits = torch.tensor([[0.4]])  # sigmoid is about 0.60
        target, lengths = torch.ones(1, 1), torch.tensor([1])
        self.assertEqual(compute_gate_metrics(logits, target, lengths, 0.5)["recall"], 1)
        self.assertEqual(compute_gate_metrics(logits, target, lengths, 0.7)["recall"], 0)

    def test_attention_skip_is_not_full_coverage(self):
        attention = torch.zeros(2, 10)
        attention[0, 0] = attention[1, 9] = 1
        metrics = compute_attention_path_metrics(attention)
        self.assertEqual(metrics["ar_text_coverage"], 0.2)
        self.assertEqual(metrics["ar_final_position_ratio"], 1)
        self.assertEqual(metrics["ar_forward_skip_rate"], 1)

    def test_attention_single_frame_has_no_jumps(self):
        metrics = compute_attention_path_metrics(torch.ones(1, 1))
        self.assertEqual(metrics["ar_text_coverage"], 1)
        self.assertEqual(metrics["ar_backward_jump_rate"], 0)
        self.assertEqual(metrics["ar_forward_skip_rate"], 0)

    def test_diagnostic_seed_preserves_training_rng(self):
        class RandomModel:
            def inference(self, **kwargs):
                return torch.rand(5)
        state = torch.random.get_rng_state().clone()
        first = diagnostic_inference(RandomModel(), None, None, torch.device("cpu"))
        second = diagnostic_inference(RandomModel(), None, None, torch.device("cpu"))
        torch.testing.assert_close(first, second)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_bucket_order_changes_with_epoch(self):
        sampler = LengthBucketBatchSampler([10] * 20, batch_size=4, seed=42)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        self.assertNotEqual(first, second)
        sampler.set_epoch(0)
        self.assertEqual(first, list(sampler))

    def test_configuration_checks_rope_dimensions(self):
        config = TransformerTacotronConfig(transformer_d_model=12, transformer_nhead=4,
                                           transformer_positional_encoding="rope")
        with self.assertRaisesRegex(ValueError, "even head"):
            config.validate()
        config.transformer_positional_encoding = "sinusoidal"
        config.validate()


if __name__ == "__main__":
    unittest.main()
