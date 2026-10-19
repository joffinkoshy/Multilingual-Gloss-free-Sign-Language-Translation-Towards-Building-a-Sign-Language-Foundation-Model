"""Joint CTC/Attention decoding (Section 2.4): attention decoder leads beam
search; Txt-CTC log-probs provide auxiliary scoring, weighted by
`ctc_weight` (a decode-time hyperparameter, separate from training-time
lambda2 — Section 12 item 7).

IMPLEMENTATION CHOICE (paper doesn't specify the exact per-step joint-scoring
algorithm, only that it follows the Yan et al. 2023 convention): rather than
per-step CTC prefix scoring during beam expansion, this does attention beam
search to produce beam_size candidates, then rescores each *complete*
candidate with its exact CTC sequence log-probability (reusing nn.CTCLoss on
a single candidate, which computes exactly that via the forward algorithm)
and picks the best combined score. This is a valid, simpler variant of joint
decoding — flag it as such if compared line-by-line against a from-scratch
ESPnet-style implementation.
"""
from typing import List, Optional

import torch

from src.models.baseline_transformer import BaselineTransformer
from src.models.layers import generate_causal_mask, key_padding_mask_from_lengths


@torch.no_grad()
def greedy_decode(model, features, feature_lengths, bos_id: int, eos_id: int, max_len: int) -> List[List[int]]:
    model.eval()
    device = features.device
    bsz = features.size(0)
    src_key_padding_mask = key_padding_mask_from_lengths(feature_lengths, features.size(1))

    if isinstance(model, BaselineTransformer):
        memory = model.encode(features, src_key_padding_mask)
    else:
        memory, _ = model.encode(features, src_key_padding_mask)

    ys = torch.full((bsz, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    for _ in range(max_len):
        tgt_mask = generate_causal_mask(ys.size(1), device=device)
        log_probs = model.decode(ys, memory, tgt_mask, None, src_key_padding_mask)
        next_ids = log_probs[:, -1, :].argmax(dim=-1)
        next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
        ys = torch.cat([ys, next_ids.unsqueeze(1)], dim=1)
        finished = finished | (next_ids == eos_id)
        if bool(finished.all()):
            break

    return [row[1:].tolist() for row in ys]  # drop leading <bos>


@torch.no_grad()
def beam_search_decode(
    model,
    features: torch.Tensor,
    feature_lengths: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    beam_size: int = 5,
    ctc_weight: float = 0.0,
) -> List[List[int]]:
    model.eval()
    device = features.device
    bsz = features.size(0)
    src_key_padding_mask = key_padding_mask_from_lengths(feature_lengths, features.size(1))

    is_baseline = isinstance(model, BaselineTransformer)
    if is_baseline:
        memory = model.encode(features, src_key_padding_mask)
        txt_ctc_log_probs = None
    else:
        memory, _ = model.encode(features, src_key_padding_mask)
        txt_ctc_log_probs = model.txt_ctc_head(memory) if ctc_weight > 0 else None

    results = []
    for b in range(bsz):
        beams = [(torch.tensor([bos_id], device=device), 0.0, False)]  # (seq, attn_score, finished)
        mem_b = memory[b : b + 1]
        src_mask_b = src_key_padding_mask[b : b + 1]

        for _ in range(max_len):
            candidates = []
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, finished))
                    continue
                tgt_mask = generate_causal_mask(seq.size(0), device=device)
                log_probs = model.decode(seq.unsqueeze(0), mem_b, tgt_mask, None, src_mask_b)
                step_log_probs = log_probs[0, -1, :]
                topk = torch.topk(step_log_probs, k=min(beam_size, step_log_probs.size(-1)))
                for score_i, id_i in zip(topk.values.tolist(), topk.indices.tolist()):
                    new_seq = torch.cat([seq, torch.tensor([id_i], device=device)])
                    candidates.append((new_seq, score + score_i, id_i == eos_id))
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = candidates[:beam_size]
            if all(f for _, _, f in beams):
                break

        if ctc_weight > 0 and txt_ctc_log_probs is not None:
            ctc_criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction="sum")
            log_probs_tbv = txt_ctc_log_probs[b : b + 1].transpose(0, 1)
            in_len = torch.tensor([log_probs_tbv.size(0)], device=device)
            rescored = []
            for seq, attn_score, _ in beams:
                target = seq[1:].unsqueeze(0)  # drop <bos>
                target = target[target != eos_id].unsqueeze(0) if target.numel() else target
                if target.numel() == 0:
                    rescored.append((seq, attn_score))
                    continue
                tgt_len = torch.tensor([target.size(1)], device=device)
                ctc_score = -ctc_criterion(log_probs_tbv, target, in_len, tgt_len)
                combined = (1 - ctc_weight) * attn_score + ctc_weight * float(ctc_score)
                rescored.append((seq, combined))
            rescored.sort(key=lambda c: c[1], reverse=True)
            best_seq = rescored[0][0]
        else:
            best_seq = beams[0][0]

        results.append(best_seq[1:].tolist())
    return results


@torch.no_grad()
def decode_loader(model, loader, tokenizer, config: dict, device: str, family_router=None):
    """Joint CTC/attention beam search over a whole DataLoader. Returns
    (hypotheses, references, src_langs, video_ids) as parallel lists."""
    hyps, refs, src_langs, video_ids = [], [], [], []
    for batch in loader:
        if family_router is not None:
            family_router.set_batch_languages(batch["src_langs"])
        hyp_ids = beam_search_decode(
            model, batch["features"].to(device), batch["feature_lengths"].to(device),
            bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
            max_len=batch["text_ids"].shape[1] + 10,
            beam_size=config["decode"]["beam_size"], ctc_weight=config["decode"]["ctc_weight"],
        )
        hyps.extend(tokenizer.decode(ids) for ids in hyp_ids)
        refs.extend(batch["texts"])
        src_langs.extend(batch["src_langs"])
        video_ids.extend(batch["video_ids"])
    return hyps, refs, src_langs, video_ids
