"""Numerical invariants of two-pass scheduled sampling (small CPU models)."""
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import hyperparams_transformer as hp
import tts_transformer_model as architecture
import tts_transformer_train as trainer
from tts_tacotron_losses import Tacotron2Loss
from tts_transformer_sampling import (
    TeacherForcingScheduler, mix_decoder_inputs, build_optimizer_groups,
    component_modules, set_frozen_eval, optimizer_group_layout,
)
from tts_transformer_experiment import configure_experiment


class SmallTransformerTest(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        constants = dict(N_MELS=4, N_SYMBOLS=32, SYMBOL_EMBED_DIM=16,
                         TRANSFORMER_D_MODEL=16, TRANSFORMER_NHEAD=4,
                         TRANSFORMER_ENCODER_LAYERS=1, TRANSFORMER_DECODER_LAYERS=2,
                         TRANSFORMER_FFN_DIM=32, TRANSFORMER_DROPOUT=0.0,
                         TRANSFORMER_LAYERSCALE_INIT=0.1, PRENET_DIMS=[8],
                         POSTNET_EMBED_DIM=8, POSTNET_KERNEL_SIZE=3, POSTNET_N_CONVS=2,
                         MAX_DECODER_STEPS=4, MAX_TEXT_POSITIONS=32)
        self.context.enter_context(patch.multiple(architecture, **constants))
        torch.manual_seed(123)
        self.model = architecture.TransformerTacotron2()
        target = torch.rand(2, 6, 4)
        mel_input = torch.cat([torch.zeros(2, 1, 4), target[:, :-1]], dim=1)
        mel_input[1, 4:] = 0
        gate = torch.zeros(2, 6)
        gate[0, 5:] = 1
        gate[1, 3:] = 1
        self.batch = dict(text=torch.tensor([[2, 3, 4, 5, 6], [2, 3, 4, 0, 0]]),
                          text_lengths=torch.tensor([5, 3]), mel_input=mel_input,
                          mel_target=target, output_lengths=torch.tensor([6, 4]), gate_target=gate)

    def forward(self, ratio=1.0, **kwargs):
        b = self.batch
        return self.model(b["text"], b["text_lengths"], b["mel_input"], b["output_lengths"],
                          teacher_forcing_ratio=ratio, **kwargs)

    def test_ratio_one_matches_original_teacher_forcing(self):
        torch.manual_seed(8)
        b = self.batch
        memory = self.model.encoder(self.model.embedding(b["text"]), b["text_lengths"])
        mel, gate, alignments = self.model.decoder(memory, b["mel_input"], b["text_lengths"], b["output_lengths"])
        expected = dict(mel_before=mel, mel_after=mel + self.model.postnet(mel), gate=gate, alignments=alignments)
        calls = []
        handle = self.model.decoder.register_forward_hook(lambda *args: calls.append(1))
        self.addCleanup(handle.remove)
        torch.manual_seed(8)
        actual = self.forward(1.0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)

    def test_two_passes_share_memory_detach_first_and_train_all_components(self):
        calls, decoder_inputs, memories = [], [], []
        encoder_calls, postnet_calls = [], []
        def capture_inputs(module, args, kwargs):
            decoder_inputs.append(kwargs["mel_input"].detach().clone())
            memories.append(kwargs["memory"])
        handles = [
            self.model.encoder.register_forward_hook(lambda *args: encoder_calls.append(1)),
            self.model.postnet.register_forward_hook(lambda *args: postnet_calls.append(1)),
            self.model.decoder.register_forward_pre_hook(capture_inputs, with_kwargs=True),
            self.model.decoder.register_forward_hook(lambda module, args, out: calls.append(out)),
        ]
        for handle in handles:
            self.addCleanup(handle.remove)
        before = [m.num_batches_tracked.clone() for m in self.model.postnet.modules()
                  if isinstance(m, torch.nn.BatchNorm1d)]
        outputs = self.forward(0.0, return_sampling_stats=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(encoder_calls), 1)
        self.assertEqual(len(postnet_calls), 1)
        self.assertIs(memories[0], memories[1])
        self.assertTrue(memories[1].requires_grad)
        self.assertFalse(calls[0].requires_grad)
        self.assertIsNone(calls[0].grad_fn)
        self.assertTrue(calls[1][0].requires_grad)
        torch.testing.assert_close(outputs["alignments"], calls[1][2])
        first, mixed = calls[0], decoder_inputs[1]
        torch.testing.assert_close(mixed[:, 0], self.batch["mel_input"][:, 0])
        torch.testing.assert_close(mixed[0, 1:], first[0, :-1])
        torch.testing.assert_close(mixed[1, 1:4], first[1, :3])
        torch.testing.assert_close(mixed[1, 4:], self.batch["mel_input"][1, 4:])
        self.assertEqual(outputs["sampling_teacher_frames"].sum().item(), 0)
        self.assertEqual(outputs["sampling_valid_frames"].sum().item(), 8)
        loss = Tacotron2Loss()(outputs, self.batch)["loss"]
        loss.backward()
        for name, modules in component_modules(self.model).items():
            gradients = [p.grad for module in modules for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0, name)
        after = [m.num_batches_tracked for m in self.model.postnet.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        for old, new in zip(before, after):
            self.assertEqual((new - old).item(), 1)

    def test_sampling_mask_is_per_frame_and_preserves_go_and_padding(self):
        mel_input = self.batch["mel_input"]
        first = (torch.arange(mel_input.numel()).reshape_as(mel_input).float() + 100).requires_grad_()
        torch.manual_seed(4)
        mixed, teachers, valid = mix_decoder_inputs(mel_input, first, self.batch["output_lengths"], 0.5)
        kept = mixed == mel_input
        self.assertTrue((kept == kept[..., :1]).all())
        self.assertTrue(kept[:, 0].all())
        self.assertTrue(kept[1, 4:].all())
        self.assertEqual(valid.tolist(), [5, 3])
        self.assertEqual(teachers.sum().item(), kept[0, 1:, 0].sum().item() + kept[1, 1:4, 0].sum().item())
        self.assertFalse(mixed.requires_grad)

    def test_both_passes_are_causal_for_rope_and_sinusoidal(self):
        for mode in ("rope", "sinusoidal"):
            with self.subTest(mode=mode), patch.object(architecture, "TRANSFORMER_POSITIONAL_ENCODING", mode):
                self.model = architecture.TransformerTacotron2().eval()
                torch.manual_seed(11)
                expected = self.forward(0.4, apply_scheduled_sampling=True)["mel_before"]
                original = self.batch["mel_input"].clone()
                self.batch["mel_input"][:, 3:] += 10
                torch.manual_seed(11)
                actual = self.forward(0.4, apply_scheduled_sampling=True)["mel_before"]
                torch.testing.assert_close(actual[:, :3], expected[:, :3], atol=1e-6, rtol=1e-6)
                self.batch["mel_input"] = original

    def test_eval_remains_teacher_forced_unless_explicitly_requested(self):
        self.model.eval()
        calls = []
        handle = self.model.decoder.register_forward_hook(lambda *args: calls.append(1))
        self.addCleanup(handle.remove)
        self.forward(0.0)
        self.assertEqual(len(calls), 1)
        self.forward(0.0, apply_scheduled_sampling=True)
        self.assertEqual(len(calls), 3)

    def test_inference_uses_existing_autoregressive_decoder_only(self):
        self.model.eval()
        before_keys = set(self.model.state_dict())
        decoder_calls = []
        handle = self.model.decoder.register_forward_hook(lambda *args: decoder_calls.append(1))
        self.addCleanup(handle.remove)
        out = self.model.inference(self.batch["text"][:1], self.batch["text_lengths"][:1])
        self.assertFalse(decoder_calls)  # inference calls decode directly, never the two-pass forward
        self.assertEqual(set(out), {"mel_before", "mel_after", "gate", "alignments"})
        self.assertLessEqual(out["mel_before"].size(1), 4)
        self.assertEqual(before_keys, set(self.model.state_dict()))
        clone = architecture.TransformerTacotron2()
        clone.load_state_dict(self.model.state_dict(), strict=True)

    def test_frozen_components_do_not_update_and_optimizer_covers_every_parameter_once(self):
        for name in component_modules(self.model):
            with self.subTest(component=name):
                model = architecture.TransformerTacotron2()
                groups = build_optimizer_groups(model, 0.001, {name: True}, {"encoder": 0.1})
                params = [p for group in groups for p in group["params"]]
                self.assertEqual(len(params), len({id(p) for p in params}))
                self.assertEqual({id(p) for p in params}, {id(p) for p in model.parameters() if p.requires_grad})
                frozen_modules = component_modules(model)[name]
                old = [{key: value.clone() for key, value in module.state_dict().items()} for module in frozen_modules]
                model.train()
                set_frozen_eval(model, {name: True})
                self.assertTrue(all(not module.training for module in frozen_modules))
                optimizer = torch.optim.AdamW(groups)
                original = self.model
                self.model = model
                loss = Tacotron2Loss()(self.forward(0.5), self.batch)["loss"]
                self.model = original
                loss.backward()
                optimizer.step()
                for module, expected in zip(frozen_modules, old):
                    self.assertTrue(all(p.grad is None for p in module.parameters()))
                    for key, value in expected.items():
                        torch.testing.assert_close(module.state_dict()[key], value, atol=0, rtol=0)
                if name == "encoder":
                    self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)
                if name == "decoder_prenet":
                    x = torch.rand(2, 4, 4)
                    torch.testing.assert_close(model.decoder.prenet(x), model.decoder.prenet(x), atol=0, rtol=0)

    def test_lr_schedule_preserves_component_multipliers(self):
        groups = build_optimizer_groups(self.model, 0.001, lr_multipliers={"encoder": 0.1})
        optimizer = torch.optim.AdamW(groups)
        trainer.set_optimizer_lr(optimizer, 0.0002)
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], 0.0002 * group["lr_multiplier"])
        self.assertEqual(len(optimizer_group_layout(optimizer)), 7)

    def test_cpu_amp_two_pass_loss_and_gradients_are_finite(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = self.forward(0.4)
            loss = Tacotron2Loss()(output, self.batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_amp_two_pass(self):
        self.model.cuda()
        self.batch = {key: value.cuda() for key, value in self.batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = Tacotron2Loss()(self.forward(0.4), self.batch)["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_validation_keeps_tf_and_sampling_losses_separate(self):
        with patch.object(hp, "autoregressive_val_samples", 0):
            torch.manual_seed(31)
            reference = trainer.validate(self.model, Tacotron2Loss(), [self.batch], torch.device("cpu"), "cpu", 1.0)
            state_after_tf = torch.random.get_rng_state().clone()
            torch.manual_seed(31)
            sampled = trainer.validate(self.model, Tacotron2Loss(), [self.batch], torch.device("cpu"), "cpu", 1.0,
                                       teacher_forcing_ratio=0.4, sampling_validation=True)
            self.assertEqual(reference["loss"], sampled["teacher_forced_loss"])
            self.assertTrue(torch.equal(state_after_tf, torch.random.get_rng_state()))
            self.assertIn("scheduled_sampling_loss", sampled)
            torch.manual_seed(75)
            repeat = trainer.validate(self.model, Tacotron2Loss(), [self.batch], torch.device("cpu"), "cpu", 1.0,
                                      teacher_forcing_ratio=0.4, sampling_validation=True)
            self.assertEqual(sampled["scheduled_sampling_loss"], repeat["scheduled_sampling_loss"])
            equal = trainer.validate(self.model, Tacotron2Loss(), [self.batch], torch.device("cpu"), "cpu", 1.0,
                                     teacher_forcing_ratio=1.0, sampling_validation=True)
            self.assertEqual(equal["teacher_forced_loss"], equal["scheduled_sampling_loss"])

    def test_old_optimizer_requires_explicit_weights_only(self):
        old_optimizer = torch.optim.AdamW(self.model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pth.tar"
            torch.save({"model": self.model.state_dict(), "optimizer": old_optimizer.state_dict(),
                        "epoch": 20, "global_step": 80}, path)
            optimizer = torch.optim.AdamW(build_optimizer_groups(self.model, 0.001))
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            with self.assertRaisesRegex(ValueError, "weights-only"):
                trainer.maybe_resume_from_checkpoint(self.model, optimizer, scaler, path, torch.device("cpu"))
            self.assertEqual(trainer.maybe_resume_from_checkpoint(self.model, optimizer, scaler, path,
                             torch.device("cpu"), weights_only=True), (0, 0))


class SchedulerTests(unittest.TestCase):
    def test_supported_schedules(self):
        schedules = [
            ({"type": "constant", "value": 0.7}, [0.7, 0.7, 0.7]),
            ({"type": "linear", "start": 1, "end": 0, "duration": 10}, [1, 0.5, 0]),
            ({"type": "piecewise_linear", "points": [[0, 1], [5, 0.5], [10, 0]]}, [1, 0.5, 0]),
        ]
        for config, expected in schedules:
            scheduler = TeacherForcingScheduler({"enabled": True, "scheduler": config})
            self.assertEqual([scheduler.get_ratio(epoch, 0) for epoch in [0, 5, 10]], expected)
        for kind in ("exponential", "inverse_sigmoid"):
            scheduler = TeacherForcingScheduler({"enabled": True, "scheduler": {"type": kind}})
            self.assertEqual(scheduler.get_ratio(0, 0), 1)
            values = [scheduler.get_ratio(epoch, 0) for epoch in [1, 10, 100, 10000]]
            self.assertEqual(values, sorted(values, reverse=True))
            self.assertTrue(all(0.2 <= value <= 1 for value in values))

    def test_checkpoint_restores_relative_epoch_and_step_origin(self):
        for unit in ("epoch", "step"):
            config = {"enabled": True, "scheduler": {"type": "linear", "unit": unit,
                                                     "start": 1, "end": 0, "duration": 10}}
            original = TeacherForcingScheduler(config, origin_epoch=20, origin_step=100)
            restored = TeacherForcingScheduler(config)
            restored.load_state_dict(original.state_dict())
            self.assertEqual(restored.get_ratio(25, 105), 0.5)
            config["scheduler"]["end"] = 0.2
            with self.assertRaisesRegex(ValueError, "weights-only"):
                TeacherForcingScheduler(config).load_state_dict(original.state_dict())
        self.assertEqual(TeacherForcingScheduler({"enabled": False}).get_ratio(100, 10000), 1)

    def test_invalid_configuration_is_rejected(self):
        for schedule in ({"type": "unknown"}, {"type": "constant", "value": 2},
                         {"type": "piecewise_linear", "points": [[0, 1], [0, 0]]},
                         {"type": "linear", "duration": 0}):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                TeacherForcingScheduler({"enabled": True, "scheduler": schedule})
        with self.assertRaises(ValueError):
            TeacherForcingScheduler({"detach_predictions": False})

    def test_ablation_presets_keep_dataset_seed_and_budget(self):
        original = SimpleNamespace(freeze={key: False for key in ("encoder", "text_embedding", "decoder")},
                                   lr_multipliers={key: 1.0 for key in ("encoder", "text_embedding", "decoder")},
                                   scheduled_sampling={"enabled": False}, lr=0.001, epochs=60, seed=42,
                                   data_path="same-data", experiment_name="")
        for preset in ("baseline", "ss1", "ss2", "ss3"):
            module = copy.deepcopy(original)
            configure_experiment(module, {}, preset)
            self.assertEqual((module.data_path, module.seed, module.epochs), ("same-data", 42, 60))
            self.assertEqual(module.scheduled_sampling["enabled"], preset != "baseline")
            self.assertEqual(module.lr_multipliers["encoder"], 0.1 if preset == "ss2" else 1)
            self.assertEqual(module.freeze["encoder"], preset == "ss3")


if __name__ == "__main__":
    unittest.main()
