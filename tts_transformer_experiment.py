"""Configure hyperparameters BEFORE importing model/train modules.

JSON is used instead of YAML to keep the teaching project dependency-free.
The existing `python tts_transformer_train.py` entrypoint is still the baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def configure_experiment(hp, values, experiment=None):
    for key, value in values.items():
        if key == "optimizer":
            if value.get("name", "adamw").lower() != "adamw":
                raise ValueError("This trainer supports AdamW")
            unknown = set(value) - {"name", "learning_rate", "lr_multipliers"}
            if unknown:
                raise ValueError(f"Unknown optimizer settings: {sorted(unknown)}")
            hp.lr = float(value.get("learning_rate", hp.lr))
            hp.lr_multipliers = copy.deepcopy(value.get("lr_multipliers", hp.lr_multipliers))
        elif hasattr(hp, key):
            setattr(hp, key, copy.deepcopy(value))
        else:
            raise ValueError(f"Unknown hyperparameter: {key}")
    if experiment is not None:
        if experiment not in {"baseline", "ss1", "ss2", "ss3"}:
            raise ValueError("Unknown experiment")
        # Presets change only SS, freezing and component multipliers; shared
        # dataset, seed, stage LR and training budget stay identical.
        hp.freeze = {name: False for name in hp.freeze}
        hp.lr_multipliers = {name: 1.0 for name in hp.lr_multipliers}
        hp.scheduled_sampling = copy.deepcopy(hp.scheduled_sampling)
        hp.scheduled_sampling["enabled"] = experiment != "baseline"
        if experiment == "ss2":
            hp.lr_multipliers.update(text_embedding=0.1, encoder=0.1)
        elif experiment == "ss3":
            hp.freeze["encoder"] = True
        hp.experiment_name = experiment


def main():
    parser = argparse.ArgumentParser(description="Transformer TF / two-pass scheduled-sampling experiments")
    parser.add_argument("--config", type=Path, help="JSON hyperparameters")
    parser.add_argument("--experiment", choices=["baseline", "ss1", "ss2", "ss3"])
    parser.add_argument("--checkpoint", type=Path, help="Starting or resumed checkpoint")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--weights-only", action="store_true", help="New stage (default for --checkpoint)")
    mode.add_argument("--resume", action="store_true", help="Restore optimizer, TF scheduler, split and counters")
    parser.add_argument("--epochs", type=int, help="Total epochs in this stage, including completed ones on resume")
    args = parser.parse_args()
    if (args.resume or args.weights_only) and args.checkpoint is None:
        parser.error("--resume / --weights-only require --checkpoint")
    import hyperparams_transformer as hp
    values = json.loads(args.config.read_text()) if args.config else {}
    configure_experiment(hp, values, args.experiment)
    if args.checkpoint is not None:
        hp.resume_transformer_tacotron_path = str(args.checkpoint)
        hp.resume_weights_only = not args.resume
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        hp.epochs = args.epochs
    # Important: this import must follow the configuration above.
    from tts_transformer_train import main as train
    train()


if __name__ == "__main__":
    main()
