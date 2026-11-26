"""The real trainer on tiny synthetic data: save, resume and new-stage loading."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import Dataset

import hyperparams_transformer as hp
import tts_transformer_train as trainer
from tts_transformer_sampling import TeacherForcingScheduler
import test_transformer_sampling as sampling_tests


class TinyTacotronPairs(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        mel = rng.random((6 + index % 3, 4), dtype=np.float32)
        return {"text": np.array([2, 3, 4], dtype=np.int32), "text_length": 3,
                "mel": mel, "mel_input": np.concatenate([np.zeros((1, 4), dtype=np.float32), mel[:-1]]),
                "mel_length": len(mel)}


# Reuse only the small-model setup, not all unit-test methods.
class SamplingTrainerIntegrationTests(unittest.TestCase):
    setUp = sampling_tests.SmallTransformerTest.setUp

    def run_stage(self, root, epochs, checkpoint=None, weights_only=False, unit="epoch"):
        sampling = {"enabled": True, "validation_enabled": True,
                    "scheduler": {"type": "linear", "unit": unit,
                                  "start": 1.0, "end": 0.0, "duration": 1 if unit == "epoch" else 4}}
        writer = Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(
                hp, n_mels=4, n_symbols=32, symbols_embedding_dim=16, transformer_d_model=16,
                transformer_nhead=4, transformer_encoder_layers=1, transformer_decoder_layers=2,
                transformer_ffn_dim=32, transformer_dropout=0.0, transformer_layerscale_init=0.1,
                prenet_dims=[8], postnet_embedding_dim=8, postnet_kernel_size=3, postnet_n_convolutions=2,
                max_decoder_steps=4, max_text_positions=32, epochs=epochs, batch_size=2,
                num_workers=0, val_ratio=0.34, val_every_epoch=1, image_every_epoch=100,
                sample_every_epoch=100, checkpoint_every_epoch=2, autoregressive_val_samples=1,
                scheduled_sampling=sampling, freeze={}, lr_multipliers={"encoder": 0.1},
                resume_transformer_tacotron_path=str(checkpoint) if checkpoint else None,
                resume_weights_only=weights_only, checkpoint_path=str(root / "checkpoints"),
                transformer_log_dir=str(root / "logs"), sample_path=str(root / "samples"),
            ))
            stack.enter_context(patch.object(trainer, "get_tacotron_dataset", TinyTacotronPairs))
            stack.enter_context(patch.object(trainer, "SummaryWriter", return_value=writer))
            stack.enter_context(patch.object(trainer, "get_device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(trainer, "get_amp_device_type", return_value="cpu"))
            trainer.main()
        last = max((root / "checkpoints").glob("*/transformer_last.pth.tar"), key=lambda p: p.stat().st_mtime_ns)
        return last, torch.load(last, map_location="cpu"), writer

    def test_epoch_schedule_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, continuous, _ = self.run_stage(root / "continuous", 2)
            checkpoint, first, _ = self.run_stage(root / "resumed", 1)
            self.assertEqual(first["epoch"], 1)
            _, resumed, writer = self.run_stage(root / "resumed", 2, checkpoint)
            self.assertEqual((resumed["epoch"], resumed["global_step"]), (2, 4))
            self.assertEqual(continuous["training_state"]["sampling_scheduler"],
                             resumed["training_state"]["sampling_scheduler"])
            self.assertEqual(continuous["training_state"]["split"], resumed["training_state"]["split"])
            for key, expected in continuous["model"].items():
                torch.testing.assert_close(resumed["model"][key], expected, atol=0, rtol=0)
            logged = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
            self.assertEqual(logged["train/teacher_forcing_ratio"], 0)
            self.assertEqual(logged["train/replaced_frames_ratio"], 1)
            self.assertGreater(logged["train/grad_norm_encoder"], 0)
            self.assertAlmostEqual(logged["train/lr_encoder"] / logged["train/lr_decoder"], 0.1)
            self.assertIn("val/teacher_forced_loss", logged)
            self.assertIn("val/scheduled_sampling_loss", logged)

    def test_step_schedule_resumes_and_weights_only_starts_new_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path, first, _ = self.run_stage(root, 1, unit="step")
            restored_scheduler = TeacherForcingScheduler(first["training_state"]["sampling_scheduler"]["config"])
            restored_scheduler.load_state_dict(first["training_state"]["sampling_scheduler"])
            self.assertEqual(restored_scheduler.get_ratio(first["epoch"], first["global_step"]), 0.5)
            _, resumed, _ = self.run_stage(root, 2, first_path, unit="step")
            self.assertEqual((resumed["epoch"], resumed["global_step"]), (2, 4))
            # New stage can change schedule unit and still load exactly the same model weights.
            _, fresh, writer = self.run_stage(root / "new_stage", 1, first_path, weights_only=True, unit="epoch")
            self.assertEqual((fresh["epoch"], fresh["global_step"]), (1, 2))
            logged = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
            self.assertEqual(logged["train/teacher_forcing_ratio"], 1)
