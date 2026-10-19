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
def beam_search_decode_reference(
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
def _encode(model, features, feature_lengths):
    src_mask = key_padding_mask_from_lengths(feature_lengths, features.size(1))
    if isinstance(model, BaselineTransformer):
        return model.encode(features, src_mask), src_mask
    memory, _ = model.encode(features, src_mask)
    return memory, src_mask


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
    """Batched joint CTC/attention beam search: same algorithm as
    beam_search_decode_reference (attention beam search, then rescoring of the
    final beams with the exact TxtCTC sequence log-prob), vectorised over
    batch x beams instead of looping per example and per beam.

    Two deliberate corrections vs. the reference: CTC rescoring uses each
    example's true frame count (the reference used the padded length), and
    infeasible CTC alignments score -inf (the reference's zero_infinity made
    them score 0, i.e. the best possible). Both only matter for padded
    batches / candidates longer than the input.
    """
    model.eval()
    device = features.device
    bsz = features.size(0)
    K = beam_size
    memory, src_mask = _encode(model, features, feature_lengths)
    mem = memory.repeat_interleave(K, dim=0)
    mem_mask = src_mask.repeat_interleave(K, dim=0)

    neg_inf = float("-inf")
    seqs = torch.full((bsz, K, 1), bos_id, dtype=torch.long, device=device)
    scores = torch.full((bsz, K), neg_inf, device=device)
    scores[:, 0] = 0.0  # a single live beam at the start, like the reference
    finished = torch.zeros(bsz, K, dtype=torch.bool, device=device)

    for _ in range(max_len):
        if bool(finished[scores > neg_inf].all()):
            break
        flat = seqs.view(bsz * K, -1)
        tgt_mask = generate_causal_mask(flat.size(1), device=device)
        step_lp = model.decode(flat, mem, tgt_mask, None, mem_mask)[:, -1, :].view(bsz, K, -1)
        k = min(K, step_lp.size(-1))
        top_v, top_i = torch.topk(step_lp, k=k, dim=-1)               # [B, K, k]
        cand_scores = scores.unsqueeze(-1) + top_v                      # [B, K, k]
        cand_ids = top_i
        # finished beams carry over unchanged as a single candidate
        cand_scores = torch.where(finished.unsqueeze(-1), torch.full_like(cand_scores, neg_inf), cand_scores)
        cand_scores[:, :, 0] = torch.where(finished, scores, cand_scores[:, :, 0])
        cand_ids = cand_ids.clone()
        cand_ids[:, :, 0] = torch.where(finished, torch.full_like(cand_ids[:, :, 0], eos_id), cand_ids[:, :, 0])

        best_scores, best_flat = torch.topk(cand_scores.view(bsz, K * k), k=K, dim=-1)
        beam_idx = torch.div(best_flat, k, rounding_mode="floor")        # parent beam
        new_ids = cand_ids.view(bsz, K * k).gather(1, best_flat)
        seqs = torch.cat([seqs.gather(1, beam_idx.unsqueeze(-1).expand(-1, -1, seqs.size(-1))), new_ids.unsqueeze(-1)], dim=-1)
        finished = finished.gather(1, beam_idx) | (new_ids == eos_id)
        scores = best_scores

    best = scores.argmax(dim=-1)  # attention-only choice (beams are sorted, so index 0)
    if ctc_weight > 0 and not isinstance(model, BaselineTransformer):
        ctc_lp = model.txt_ctc_head(memory)                                # [B, T, V]
        body = seqs[:, :, 1:]                                              # drop <bos>
        keep = body != eos_id
        tgt_len = keep.sum(-1)                                             # [B, K]
        order = torch.argsort((~keep).int(), dim=-1, stable=True)          # non-eos tokens first, order kept
        targets = body.gather(-1, order).view(bsz * K, -1)
        ctc = torch.nn.CTCLoss(blank=0, zero_infinity=False, reduction="none")
        loss = ctc(
            ctc_lp.repeat_interleave(K, dim=0).transpose(0, 1), targets,
            feature_lengths.repeat_interleave(K), tgt_len.view(-1),
        ).view(bsz, K)
        ctc_score = -loss                                                  # -inf when infeasible
        combined = (1 - ctc_weight) * scores + ctc_weight * ctc_score
        combined = torch.where(tgt_len == 0, scores, combined)             # reference skips empty targets
        combined = torch.where(scores > neg_inf, combined, torch.full_like(combined, neg_inf))
        best = combined.argmax(dim=-1)

    out = seqs[torch.arange(bsz, device=device), best][:, 1:]
    results = []
    for row in out.tolist():  # finished beams were padded with <eos>: trim after the first one
        results.append(row[: row.index(eos_id) + 1] if eos_id in row else row)
    return results


def decode_loader(model, loader, tokenizer, config: dict, device: str, family_router=None, strategy: str = "beam"):
    """Decode a whole DataLoader. strategy="beam": joint CTC/attention beam
    search (final dev/test evaluation); "greedy": batched attention-only
    greedy decoding (per-epoch checkpoint selection — the reference repo's
    default, eval_translation_beam_size=1). Returns (hypotheses, references,
    src_langs, video_ids) as parallel lists."""
    if strategy not in ("beam", "greedy"):
        raise ValueError(f"Unknown decoding strategy '{strategy}'")
    hyps, refs, src_langs, video_ids = [], [], [], []
    for batch in loader:
        if family_router is not None:
            family_router.set_batch_languages(batch["src_langs"])
        features, lengths = batch["features"].to(device), batch["feature_lengths"].to(device)
        max_len = batch["text_ids"].shape[1] + 10
        if strategy == "greedy":
            hyp_ids = greedy_decode(model, features, lengths, tokenizer.bos_id, tokenizer.eos_id, max_len)
        else:
            hyp_ids = beam_search_decode(
                model, features, lengths, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, max_len=max_len,
                beam_size=config["decode"]["beam_size"], ctc_weight=config["decode"]["ctc_weight"],
            )
        hyps.extend(tokenizer.decode(ids) for ids in hyp_ids)
        refs.extend(batch["texts"])
        src_langs.extend(batch["src_langs"])
        video_ids.extend(batch["video_ids"])
    return hyps, refs, src_langs, video_ids
