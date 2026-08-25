"""Construct a model + optional family router from a validated config
(Section 7: V1 modules are only ever composed in here, never edited into
baseline_transformer.py / sign2_lid_text.py directly)."""
from typing import Optional

from src.models.baseline_transformer import BaselineTransformer
from src.models.family_router import FamilyRouter
from src.models.sign2_lid_text import Sign2LIDText


def build_model(
    config: dict,
    txt_vocab_size: int,
    txt_pad_index: int,
    lid_vocab_size: Optional[int] = None,
):
    m = config["model"]
    family_router = None
    fr_cfg = config.get("family_routing", {})
    if fr_cfg.get("enabled", False):
        if m["variant"] != "sign2_lid_text":
            raise ValueError("family_routing is only composed into sign2_lid_text (Section 8)")
        family_router = FamilyRouter(
            d_model=m["d_model"],
            language_to_family=fr_cfg["language_to_family"],
            bottleneck_dim=fr_cfg.get("bottleneck_dim", 64),
        )

    if m["variant"] == "baseline":
        model = BaselineTransformer(
            d_feat=m["d_feat"],
            d_model=m["d_model"],
            num_encoder_layers=m["num_encoder_layers"],
            num_decoder_layers=m["num_decoder_layers"],
            num_heads=m["num_heads"],
            ff_size=m["ff_size"],
            dropout=m["dropout"],
            txt_vocab_size=txt_vocab_size,
            txt_pad_index=txt_pad_index,
        )
    elif m["variant"] == "sign2_lid_text":
        model = Sign2LIDText(
            d_feat=m["d_feat"],
            d_model=m["d_model"],
            num_encoder_layers=m["num_encoder_layers"],
            num_decoder_layers=m["num_decoder_layers"],
            num_heads=m["num_heads"],
            ff_size=m["ff_size"],
            dropout=m["dropout"],
            lid_split_layer=m["lid_split_layer"],
            use_lid_branch=m["use_lid_branch"],
            txt_vocab_size=txt_vocab_size,
            txt_pad_index=txt_pad_index,
            lid_vocab_size=lid_vocab_size,
            family_router=family_router,
        )
    else:
        raise ValueError(f"Unknown model.variant '{m['variant']}'")

    return model, family_router
