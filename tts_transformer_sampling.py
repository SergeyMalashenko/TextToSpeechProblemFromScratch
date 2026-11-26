"""Small, independent helpers for two-pass scheduled-sampling experiments."""
from __future__ import annotations

import copy
import math

import torch


COMPONENTS = ("text_embedding", "encoder", "decoder_prenet", "decoder",
              "mel_projection", "gate_projection", "postnet")


def mix_decoder_inputs(mel_input, first_mel, lengths, teacher_forcing_ratio):
    """Replace whole shifted frames, preserving the original GO and padding.

    Input at t uses prediction t-1, never prediction t. The mask is (B, T, 1).
    Counts are per utterance so DataParallel can gather them without averaging
    ratios from replicas with different numbers of valid frames.
    """
    if mel_input.shape != first_mel.shape:
        raise ValueError("Preliminary predictions must match mel_input shape")
    if lengths is None:
        raise ValueError("Scheduled sampling requires output_lengths")
    if not 0 <= teacher_forcing_ratio <= 1:
        raise ValueError("teacher_forcing_ratio must be in [0, 1]")
    positions = torch.arange(mel_input.size(1), device=mel_input.device)[None, :]
    eligible = (positions > 0) & (positions < lengths[:, None])
    predicted_input = torch.cat((mel_input[:, :1], first_mel.detach()[:, :-1]), dim=1)
    keep_teacher = torch.rand((*mel_input.shape[:2], 1), device=mel_input.device) < teacher_forcing_ratio
    keep_teacher = keep_teacher | ~eligible.unsqueeze(-1)
    mixed = torch.where(keep_teacher, mel_input, predicted_input)
    teacher_frames = (keep_teacher.squeeze(-1) & eligible).sum(dim=1)
    return mixed, teacher_frames, eligible.sum(dim=1)


class TeacherForcingScheduler:
    """Epochs/steps are relative to the start of this training stage, zero based."""
    def __init__(self, config=None, origin_epoch=0, origin_step=0):
        self.config = copy.deepcopy(config or {"enabled": False})
        self.origin_epoch, self.origin_step = int(origin_epoch), int(origin_step)
        supported = {"method": "two_pass", "detach_predictions": True,
                     "sampling_granularity": "frame", "prediction_source": "mel_before_postnet"}
        for key, expected in supported.items():
            if self.config.get(key, expected) != expected:
                raise ValueError(f"Only {key}={expected!r} is supported")
        schedule = self.config.get("scheduler", {"type": "constant", "value": 1.0})
        self.schedule = copy.deepcopy(schedule)
        if schedule.get("unit", "epoch") not in {"epoch", "step"}:
            raise ValueError("TF scheduler unit must be epoch or step")
        kind = schedule.get("type", "constant")
        if kind not in {"constant", "linear", "exponential", "inverse_sigmoid", "piecewise_linear"}:
            raise ValueError(f"Unknown TF scheduler: {kind}")
        for key in ("value", "start", "end"):
            if key in schedule and not 0 <= float(schedule[key]) <= 1:
                raise ValueError(f"TF scheduler {key} must be in [0, 1]")
        if kind == "piecewise_linear":
            points = schedule.get("points", [])
            if not points or float(points[0][0]) != 0:
                raise ValueError("Piecewise schedule must start at position 0")
            previous = -1.0
            for position, ratio in points:
                if not math.isfinite(float(position)) or position <= previous or not 0 <= ratio <= 1:
                    raise ValueError("Schedule points need increasing positions and ratios in [0, 1]")
                previous = position
        if kind == "linear" and float(schedule.get("duration", 60)) <= 0:
            raise ValueError("Linear schedule duration must be positive")
        if kind == "exponential" and not 0 < float(schedule.get("decay_rate", 0.95)) <= 1:
            raise ValueError("Exponential decay_rate must be in (0, 1]")
        if kind == "inverse_sigmoid" and float(schedule.get("k", 10)) <= 0:
            raise ValueError("Inverse sigmoid k must be positive")

    def get_ratio(self, epoch, global_step):
        if not self.config.get("enabled", False):
            return 1.0
        s = self.schedule
        x = max(0, epoch - self.origin_epoch) if s.get("unit", "epoch") == "epoch" else max(0, global_step - self.origin_step)
        kind = s.get("type", "constant")
        start, end = float(s.get("start", 1.0)), float(s.get("end", 0.2))
        if kind == "constant":
            value = float(s.get("value", 1.0))
        elif kind == "linear":
            progress = min(1.0, x / float(s.get("duration", 60)))
            value = start + (end - start) * progress
        elif kind == "exponential":
            value = end + (start - end) * float(s.get("decay_rate", 0.95)) ** x
        elif kind == "inverse_sigmoid":
            k = float(s.get("k", 10))
            value = end if x / k > 700 else end + (start - end) * (k + 1) / (k + math.exp(x / k))
        else:
            points = s["points"]
            value = float(points[-1][1])
            for (left, a), (right, b) in zip(points, points[1:]):
                if x <= right:
                    value = a + (b - a) * (x - left) / (right - left)
                    break
        return float(value)

    def state_dict(self):
        return {"config": copy.deepcopy(self.config), "origin_epoch": self.origin_epoch,
                "origin_step": self.origin_step}

    def load_state_dict(self, state):
        if state["config"] != self.config:
            raise ValueError("TF schedule differs from checkpoint; use weights-only for a new stage")
        self.origin_epoch = int(state["origin_epoch"])
        self.origin_step = int(state["origin_step"])


def component_modules(model):
    """Decoder means its core; the prenet and output heads have separate controls."""
    return {
        "text_embedding": [model.embedding],
        "encoder": [model.encoder],
        "decoder_prenet": [model.decoder.prenet, model.decoder.prenet_proj],
        "decoder": [model.decoder.positional_encoding, model.decoder.layers],
        "mel_projection": [model.decoder.mel_projection],
        "gate_projection": [model.decoder.gate_projection],
        "postnet": [model.postnet],
    }


def set_frozen_eval(model, freeze):
    """Call after model.train(); frozen blocks must not update BN or use dropout."""
    for name, modules in component_modules(model).items():
        if freeze.get(name, False):
            for module in modules:
                module.eval()
    # Tacotron prenet intentionally uses dropout in eval; frozen prenet is an exception.
    model.decoder.prenet.frozen = bool(freeze.get("decoder_prenet", False))


def build_optimizer_groups(model, base_lr, freeze=None, lr_multipliers=None):
    freeze, multipliers = freeze or {}, lr_multipliers or {}
    unknown = (set(freeze) | set(multipliers)) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown components: {sorted(unknown)}")
    groups, seen = [], set()
    parameter_names = {id(p): name for name, p in model.named_parameters()}
    for name, modules in component_modules(model).items():
        multiplier = float(multipliers.get(name, 1.0))
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"LR multiplier for {name} must be finite and positive")
        parameters = []
        for module in modules:
            for p in module.parameters():
                if id(p) in seen:
                    raise ValueError("A parameter belongs to more than one component")
                seen.add(id(p))
                p.requires_grad_(not freeze.get(name, False))
                p.grad = None
                if p.requires_grad:
                    parameters.append(p)
        if parameters:
            groups.append({"params": parameters, "name": name, "lr_multiplier": multiplier,
                           "lr": base_lr * multiplier,
                           "parameter_names": [parameter_names[id(p)] for p in parameters]})
    if seen != {id(p) for p in model.parameters()}:
        raise ValueError("Some model parameters do not belong to a component")
    if not groups:
        raise ValueError("At least one component must be trainable")
    set_frozen_eval(model, freeze)
    return groups


def optimizer_group_layout(optimizer):
    return [{"name": group.get("name"), "lr_multiplier": group.get("lr_multiplier", 1.0),
             "parameter_names": group.get("parameter_names")} for group in optimizer.param_groups]


def component_gradient_norms(model):
    norms = {}
    for name, modules in component_modules(model).items():
        values = [p.grad.detach().float().square().sum() for module in modules
                  for p in module.parameters() if p.grad is not None]
        norms[f"grad_norm_{name}"] = float(torch.stack(values).sum().sqrt().item()) if values else 0.0
    return norms


def capture_rng_state():
    """Store only tensors and Python primitives, compatible with torch.load defaults."""
    import random
    import numpy as np
    numpy = np.random.get_state()
    return {"torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": (numpy[0], numpy[1].tolist(), numpy[2], numpy[3], numpy[4])}


def restore_rng_state(state):
    import random
    import numpy as np
    torch.random.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint; use weights-only")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
