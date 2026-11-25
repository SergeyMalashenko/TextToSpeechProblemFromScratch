import io
import unittest

import torch
from torch import nn

from tts_mel2mag_model import MelToMagModel, ConvBlock
from tts_mel2mag_losses import MelToMagLoss


class MelToMagTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = dict(n_mels=4, n_freq=9, hidden=12, kernel=3,
                           dilations=[1, 2, 4], dropout=0.1, output_activation="sigmoid")

    def test_padding_does_not_change_valid_predictions_or_gradients(self):
        model = MelToMagModel(self.config).eval()
        mel = torch.rand(1, 11, 4)
        lengths = torch.tensor([11])
        reference = model(mel, lengths)
        padded = torch.cat([mel, torch.rand(1, 9, 4)], dim=1)
        output = model(padded, lengths)
        torch.testing.assert_close(reference, output[:, :11])
        self.assertEqual(output[:, 11:].count_nonzero().item(), 0)
        self.assertTrue(((reference > 0) & (reference < 1)).all())
        reference.sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_round_trip_and_legacy_loading(self):
        model = MelToMagModel(self.config).eval()
        mel = torch.rand(2, 7, 4)
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), model_config=model.config), stream)
        stream.seek(0)
        restored = MelToMagModel.from_checkpoint(torch.load(stream, weights_only=True)).eval()
        torch.testing.assert_close(model(mel), restored(mel))
        legacy = nn.Module()
        legacy.convs = nn.ModuleList([ConvBlock(4, 12, 5), ConvBlock(12, 9, 5, False)])
        restored = MelToMagModel.from_checkpoint(dict(model=legacy.state_dict())).eval()
        expected = mel.transpose(1, 2)
        legacy.eval()
        for layer in legacy.convs:
            expected = layer(expected)
        torch.testing.assert_close(restored(mel), expected.transpose(1, 2))

    def test_losses_ignore_padding_and_have_finite_gradients(self):
        basis = torch.rand(4, 9) * 0.01
        criterion = MelToMagLoss(mel_basis=basis, spectral_weight=0.1)
        pred = torch.rand(2, 8, 9, requires_grad=True)
        target = torch.rand_like(pred)
        mel = torch.rand(2, 8, 4)
        lengths = torch.tensor([8, 3])
        first = criterion(pred, target, mel, lengths)
        changed_target, changed_mel = target.clone(), mel.clone()
        changed_target[1, 3:] = 50
        changed_mel[1, 3:] = -50
        second = criterion(pred, changed_target, changed_mel, lengths)
        for key in first:
            torch.testing.assert_close(first[key], second[key])
        first["loss"].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[1, 3:].count_nonzero().item(), 0)

    def test_exact_spectrum_has_zero_reconstruction_losses(self):
        criterion = MelToMagLoss(mel_weight=0, spectral_weight=0.1)
        target = torch.rand(2, 1, 9)
        losses = criterion(target, target, torch.rand(2, 1, 4), torch.tensor([1, 1]))
        self.assertEqual(losses["loss"].item(), 0)

    def test_short_training_run_reduces_loss(self):
        config = dict(self.config, dropout=0)
        model = MelToMagModel(config)
        criterion = MelToMagLoss(mel_weight=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        mel, target = torch.rand(2, 8, 4), torch.full((2, 8, 9), 0.7)
        lengths = torch.tensor([8, 5])
        initial = criterion(model(mel, lengths), target, mel, lengths)["loss"].item()
        for _ in range(25):
            optimizer.zero_grad()
            loss = criterion(model(mel, lengths), target, mel, lengths)["loss"]
            loss.backward()
            optimizer.step()
        final = criterion(model(mel, lengths), target, mel, lengths)["loss"].item()
        self.assertLess(final, initial * 0.5)


if __name__ == "__main__":
    unittest.main()
