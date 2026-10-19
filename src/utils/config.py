"""YAML config loading + validation.

Every hyperparameter the paper leaves unspecified (Section 2.3 / Section 12
of IMPLEMENTATION_HANDOFF.md) lives here as an explicit, named default rather
than being hardcoded inside model/training code. `REQUIRED_KEYS` enumerates
the minimal structure every task config must have; `ASSUMPTIONS` documents
which fields are paper-confirmed vs. best-guess defaults so that never gets
silently lost when configs are copied/edited.
"""
import argparse
import os
import sys

import yaml

REQUIRED_TOP_KEYS = ["task", "model", "loss", "optimizer", "scheduler", "training", "decode", "data"]

REQUIRED_MODEL_KEYS = [
    "variant", "d_model", "d_feat", "num_encoder_layers", "num_decoder_layers",
    "num_heads", "ff_size", "dropout", "lid_split_layer", "use_lid_branch",
]

REQUIRED_LOSS_KEYS = ["lambda_lid", "lambda_txt", "lambda_attn", "label_smoothing"]

REQUIRED_TRAINING_KEYS = [
    "batch_size", "seed", "grad_clip_norm", "early_stopping_patience",
    "checkpoint_every_epochs", "max_epochs",
]

REQUIRED_DATA_KEYS = ["dataset", "tokenizer", "feature_cache_dir", "splits_dir"]

# Fields whose values are Claude-chosen defaults, not values recovered from
# the paper or the reference repo (Section 12). Surfaced by --check so a
# config can never be mistaken for a fully paper-confirmed spec.
ASSUMPTION_FIELDS = {
    "model.dropout": "paper unspecified; Transformer-default 0.1 (Section 12 item 4)",
    "optimizer.betas": "paper unspecified; reference repo (signjoey/builders.py) default (0.9, 0.999)",
    "scheduler.name": "paper unspecified; reference repo's real default is ReduceLROnPlateau on dev BLEU, adopted here (Section 12 item 3, corrected after inspecting Claire874/Gloss-free-MLSLT)",
    "loss.label_smoothing": "paper/repo default off (0.0); repo supports it but does not enable it by default (Section 12 item 6, confirmed via signjoey/loss.py)",
    "data.tokenizer.type": "paper implied SentencePiece; reference repo (signjoey/vocabulary.py) actually uses word-level Counter-based vocab. Word-level is now the default — see Section 12 item 5 correction in reference_inspection_notes.md",
    "decode.ctc_weight": "paper unspecified; ESPnet-style default 0.3 (Section 12 item 7)",
    "decode.beam_size": "paper unspecified; default 5 (Section 12 item 7)",
    "lid_self_conditioning.gate_init": "proposed novelty; sigmoid(-2.0)=0.12 initial gate chosen so training starts close to the paper's model (PLAN_LID_SELF_CONDITIONING.md)",
    "model.d_feat": "paper unspecified exact value; must be set to the real SlowFastSign checkpoint's output dim before real training (Section 12 item 1)",
}


class ConfigError(ValueError):
    pass


def _require(d: dict, keys, path: str):
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"Missing required key(s) under '{path}': {missing}")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        if k == "extends":
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_raw(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    extends = cfg.get("extends")
    if extends:
        base_path = os.path.join(os.path.dirname(path), extends)
        base_cfg = _load_raw(base_path)
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def load_config(path: str) -> dict:
    cfg = _load_raw(path)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError("Top-level config must be a mapping")
    _require(cfg, REQUIRED_TOP_KEYS, "<root>")
    _require(cfg["model"], REQUIRED_MODEL_KEYS, "model")
    _require(cfg["loss"], REQUIRED_LOSS_KEYS, "loss")
    _require(cfg["training"], REQUIRED_TRAINING_KEYS, "training")
    _require(cfg["data"], REQUIRED_DATA_KEYS, "data")

    if cfg["task"] not in ("one_to_one", "many_to_one", "many_to_many"):
        raise ConfigError(f"Unknown task '{cfg['task']}'")

    if cfg["model"]["variant"] not in ("baseline", "sign2_lid_text"):
        raise ConfigError(f"Unknown model.variant '{cfg['model']['variant']}'")

    if cfg["model"]["variant"] == "baseline" and cfg["model"]["use_lid_branch"]:
        raise ConfigError(
            "model.variant=baseline has no CTC branch at all (Section 2.3, "
            "Architecture A) — use_lid_branch must be false regardless of task."
        )
    if cfg["model"]["variant"] == "sign2_lid_text":
        if cfg["task"] == "one_to_one" and cfg["model"]["use_lid_branch"]:
            raise ConfigError(
                "one_to_one task must set model.use_lid_branch=false — "
                "paper: 'Since Sign2LID is not utilized in this task, it is a "
                "purely gloss-free SLT setting' (Section 2.3)."
            )
        if cfg["task"] != "one_to_one" and not cfg["model"]["use_lid_branch"]:
            raise ConfigError(
                f"task '{cfg['task']}' requires model.use_lid_branch=true (lambda1=1)."
            )

    if cfg.get("lid_self_conditioning", {}).get("enabled", False):
        if cfg["model"]["variant"] != "sign2_lid_text" or not cfg["model"]["use_lid_branch"]:
            raise ConfigError(
                "lid_self_conditioning conditions on the LID-CTC posterior — it "
                "requires model.variant=sign2_lid_text and model.use_lid_branch=true."
            )

    if not cfg["model"]["use_lid_branch"] and cfg["loss"]["lambda_lid"] != 0:
        raise ConfigError("lambda_lid must be 0 when use_lid_branch is false.")


def list_assumptions() -> None:
    print("Fields in this config schema that are assumptions, not paper-confirmed values:")
    for field, why in ASSUMPTION_FIELDS.items():
        print(f"  - {field}: {why}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", help="path to a YAML config")
    parser.add_argument("--check", action="store_true", help="validate the given config")
    parser.add_argument("--assumptions", action="store_true", help="list assumption fields and exit")
    args = parser.parse_args()

    if args.assumptions:
        list_assumptions()
        return

    if not args.config:
        parser.error("config path required unless --assumptions is given")

    cfg = load_config(args.config)
    if args.check:
        print(f"OK: {args.config} is valid.")
        used_assumptions = [f for f in ASSUMPTION_FIELDS if _get_path(cfg, f) is not None]
        if used_assumptions:
            print(f"({len(used_assumptions)} fields in this config are Section-12 assumptions; run --assumptions for details)")


def _get_path(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


if __name__ == "__main__":
    try:
        main()
    except ConfigError as e:
        print(f"INVALID CONFIG: {e}", file=sys.stderr)
        sys.exit(1)
