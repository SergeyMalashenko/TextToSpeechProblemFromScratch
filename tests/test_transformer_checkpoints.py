"""Checkpoint selection and compatibility examples; no audio is generated."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import hyperparams_transformer as hp
from tts_transformer_config import build_transformer_tacotron_config, check_checkpoint_config
from tts_transformer_synthesis import load_transformer_model, resolve_transformer_checkpoint, resolve_mel2mag_checkpoint
import hyperparams_mel2mag as hp_mel2mag
from tts_transformer_train import maybe_resume_from_checkpoint


class TransformerCheckpointTests(unittest.TestCase):
    def test_decoding_settings_can_change_but_architecture_cannot(self):
        saved = asdict(build_transformer_tacotron_config())
        saved["gate_threshold"] = 0.1
        saved["max_decoder_steps"] += 10
        check_checkpoint_config(saved)
        saved["transformer_d_model"] += 16
        with self.assertRaisesRegex(ValueError, "architecture"):
            check_checkpoint_config(saved)
        check_checkpoint_config(None)  # legacy checkpoints

    def test_select_epoch_from_flat_directory_and_run_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = "checkpoint_transformer_tacotron2_epoch_0012.pth.tar"
            with patch.object(hp, "checkpoint_path", directory), patch.object(hp, "restore_transformer_epoch", 12):
                flat = root / filename
                flat.touch()
                self.assertEqual(resolve_transformer_checkpoint(None), flat)
                flat.unlink()
                run = root / "transformer_run"
                run.mkdir()
                nested = run / filename
                nested.touch()
                self.assertEqual(resolve_transformer_checkpoint(None), nested)
                self.assertEqual(resolve_transformer_checkpoint("explicit.pth.tar"), Path("explicit.pth.tar"))

    def test_missing_and_ambiguous_epochs_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(hp, "checkpoint_path", directory), patch.object(hp, "restore_transformer_epoch", 12):
                with self.assertRaises(FileNotFoundError):
                    resolve_transformer_checkpoint(None)
                for name in ("first", "second"):
                    run = root / name
                    run.mkdir()
                    (run / "checkpoint_transformer_tacotron2_epoch_0012.pth.tar").touch()
                with self.assertRaisesRegex(ValueError, "Several experiments"):
                    resolve_transformer_checkpoint(None)

    def test_select_mel2mag_epoch_and_preserve_explicit_and_legacy_paths(self):
        with patch.object(hp, "mel2mag_checkpoint_path", "/tmp/checkpoints"), \
             patch.object(hp_mel2mag, "restore_mel2mag_epoch", 12):
            self.assertEqual(resolve_mel2mag_checkpoint(None), Path("/tmp/checkpoints/checkpoint_mel2mag_epoch_0012.pth.tar"))
            self.assertEqual(resolve_mel2mag_checkpoint("old_weights.pth.tar"), Path("old_weights.pth.tar"))
        with patch.object(hp_mel2mag, "restore_mel2mag_epoch", None), \
             patch.object(hp, "restore_step2", 2000, create=True):
            self.assertEqual(resolve_mel2mag_checkpoint(None).name, "checkpoint_mel2mag_2000.pth.tar")

    def test_load_prefixed_and_unprefixed_weights(self):
        reference = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth.tar"
            for prefix in ("", "module."):
                state = {prefix + key: value.clone() for key, value in reference.state_dict().items()}
                for wrapped in (False, True):
                    torch.save({"model": state} if wrapped else state, path)
                    with patch("tts_transformer_synthesis.TransformerTacotron2", side_effect=lambda: nn.Linear(2, 2)):
                        loaded = load_transformer_model(path, torch.device("cpu"))
                    for key, value in reference.state_dict().items():
                        torch.testing.assert_close(loaded.state_dict()[key], value)
                    self.assertFalse(loaded.training)

    def test_strict_loading_rejects_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth.tar"
            torch.save({"model": {"weight": torch.ones(2, 2)}}, path)
            with patch("tts_transformer_synthesis.TransformerTacotron2", side_effect=lambda: nn.Linear(2, 2)):
                with self.assertRaises(RuntimeError):
                    load_transformer_model(path, torch.device("cpu"))
                with self.assertWarnsRegex(UserWarning, "Partial Transformer load"):
                    load_transformer_model(path, torch.device("cpu"), strict=False)

    def test_resume_prefixed_weights_with_changed_decoding_settings(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        config = asdict(build_transformer_tacotron_config())
        config["gate_threshold"] = 0.1
        config["max_decoder_steps"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth.tar"
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            torch.save({"model": {"module." + key: value for key, value in expected.items()},
                        "model_config": config, "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": 3, "global_step": 20}, path)
            with torch.no_grad():
                model.weight.zero_()
            position = maybe_resume_from_checkpoint(model, optimizer, scaler, path, torch.device("cpu"))
            self.assertEqual(position, (3, 20))
            for key, value in expected.items():
                torch.testing.assert_close(model.state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
