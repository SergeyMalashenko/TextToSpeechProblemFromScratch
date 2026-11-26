"""Epoch checkpointing, including continuation from older mid-epoch saves."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import Dataset

import hyperparams_mel2mag as hp
from tts_mel2mag_model import MelToMagModel
import tts_mel2mag_train as trainer


class TinyPairs(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        return {"mel": rng.random((4, 4), dtype=np.float32),
                "mag": rng.random((4, 9), dtype=np.float32), "mel_length": 4}


class MelToMagEpochTests(unittest.TestCase):
    config = dict(n_mels=4, n_freq=9, hidden=8, kernel=3, dilations=[1],
                  dropout=0.0, output_activation="sigmoid")

    def test_epoch_interval_final_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = Mock()
            with patch.multiple(hp, epochs=3, batch_size=2, num_workers=0, val_ratio=0.2,
                                mel2mag_save_every_epoch=2, mel2mag_sample_every_epoch=0,
                                mel2mag_mel_consistency_weight=0.0, checkpoint_path=directory,
                                resume_mel2mag_checkpoint=None), \
                 patch.object(trainer, "get_mel2mag_dataset", TinyPairs), \
                 patch.object(trainer, "MelToMagModel", side_effect=lambda: MelToMagModel(self.config)), \
                 patch.object(trainer, "SummaryWriter", return_value=writer), \
                 patch.object(trainer, "get_device", return_value=torch.device("cpu")), \
                 patch.object(trainer, "get_amp_device_type", return_value="cpu"):
                trainer.main()
                root = Path(directory)
                numbered = sorted(p.name for p in root.glob("checkpoint_mel2mag_*.pth.tar"))
                self.assertEqual(numbered, ["checkpoint_mel2mag_epoch_0002.pth.tar",
                                            "checkpoint_mel2mag_epoch_0003.pth.tar"])
                last = root / "mel2mag_last.pth.tar"
                saved = torch.load(last, map_location="cpu")
                self.assertEqual((saved["epoch"], saved["batch_in_epoch"], saved["global_step"]), (3, 0, 6))
                self.assertTrue((root / "mel2mag_best.pth.tar").exists())
                validation_epochs = [call.args[2] for call in writer.add_scalar.call_args_list
                                     if call.args[0] == "val/loss"]
                self.assertEqual(validation_epochs, [1, 2, 3])
                with patch.object(hp, "epochs", 4), patch.object(hp, "resume_mel2mag_checkpoint", str(last)):
                    trainer.main()
                resumed = torch.load(last, map_location="cpu")
                self.assertEqual((resumed["epoch"], resumed["global_step"]), (4, 8))
                self.assertTrue((root / "checkpoint_mel2mag_epoch_0004.pth.tar").exists())

    def test_cleanup_preserves_step_checkpoints_and_named_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in (1, 2, 3):
                path = root / f"checkpoint_mel2mag_epoch_{epoch:04d}.pth.tar"
                path.touch()
                os.utime(path, (epoch, epoch))
            for name in ("checkpoint_mel2mag_2000.pth.tar", "mel2mag_best.pth.tar", "mel2mag_last.pth.tar"):
                (root / name).touch()
            trainer.cleanup_old_checkpoints(root, 0)
            self.assertEqual(len(list(root.iterdir())), 6)
            trainer.cleanup_old_checkpoints(root, 1)
            self.assertEqual(sorted(p.name for p in root.iterdir()), [
                "checkpoint_mel2mag_2000.pth.tar", "checkpoint_mel2mag_epoch_0003.pth.tar",
                "mel2mag_best.pth.tar", "mel2mag_last.pth.tar"])

    def test_old_mid_epoch_checkpoint_finishes_remaining_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MelToMagModel(self.config)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
            scaler = torch.amp.GradScaler("cpu")
            legacy = Path(directory) / "checkpoint_mel2mag_1.pth.tar"
            trainer.save_checkpoint(model, optimizer, scaler, 0, 1, legacy, batch_in_epoch=1)
            with patch.multiple(hp, epochs=1, batch_size=2, num_workers=0, val_ratio=0.2,
                                mel2mag_save_every_epoch=2, mel2mag_sample_every_epoch=0,
                                mel2mag_mel_consistency_weight=0.0, checkpoint_path=directory,
                                resume_mel2mag_checkpoint=str(legacy)), \
                 patch.object(trainer, "get_mel2mag_dataset", TinyPairs), \
                 patch.object(trainer, "MelToMagModel", side_effect=lambda: MelToMagModel(self.config)), \
                 patch.object(trainer, "SummaryWriter"), \
                 patch.object(trainer, "get_device", return_value=torch.device("cpu")), \
                 patch.object(trainer, "get_amp_device_type", return_value="cpu"):
                trainer.main()
            saved = torch.load(Path(directory) / "checkpoint_mel2mag_epoch_0001.pth.tar", map_location="cpu")
            self.assertEqual((saved["epoch"], saved["batch_in_epoch"], saved["global_step"]), (1, 0, 2))
            self.assertTrue(legacy.exists())

    def test_invalid_epoch_interval_is_reported(self):
        with patch.object(hp, "mel2mag_save_every_epoch", 0):
            with self.assertRaises(ValueError):
                trainer.get_checkpoint_every_epoch()


if __name__ == "__main__":
    unittest.main()
