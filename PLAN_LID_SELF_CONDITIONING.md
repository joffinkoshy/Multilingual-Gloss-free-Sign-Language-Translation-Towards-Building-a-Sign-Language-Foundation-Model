# Plan: LID Self-Conditioning (single project novelty)

Decided 2026-10-20. This supersedes the three-improvement "V1 ladder"
(family routing / augmentation / pretraining) **for the course deliverable**.
Those modules stay in the repo but are not run, reported, or claimed.

Paper: Tan, Miyazaki, Nakadai — *Multilingual Gloss-free Sign Language
Translation: Towards Building a Sign Language Foundation Model*, ACL 2025
(short). Reference code: https://github.com/Claire874/Gloss-free-MLSLT

---

## 1. The novelty in one paragraph

Sign2(LID+Text) predicts a token-level sign-language ID (LID_tok) from the
first encoder layer and trains it with a CTC loss (Eq. 2) — but the
prediction is **only a loss target and is discarded**: layers 2..N never see
it. We feed the LID-CTC posterior back into the residual stream before the
Sign2Text layers (self-conditioning, Nozaki & Komatsu, Interspeech 2021):

    Z      = softmax(W_lid · h_int)               # [B, |F|, V_lid], already computed by the paper
    h_int' = h_int + g · W_c · Z,   g = sigmoid(a)  # NEW: W_c ∈ R^{d×V_lid}, a scalar gate
    h_fin  = Enc_{2..N}(h_int')                   # unchanged

Everything else (losses, λ=(1,5,3), decoder, decoding) is identical to the
paper. Added parameters: V_lid·d + 1 ≈ 12×256 + 1 ≈ 3k (≈0.008% of 39.5M), so
any gain cannot be explained by extra capacity.

How it differs from the paper:

| | Paper (baseline) | Proposed |
|---|---|---|
| LID_tok prediction | auxiliary loss only | loss **and** input to layers 2..N |
| Gradient into LID head | only from L_LID | from L_LID + L_Txt + L_Attn |
| Effect of LID at few languages | fixed λ1=1, can hurt (Table 9: 7.48→6.22 at 2→2) | learnable gate g lets the model down-weight it |
| Test-time language info | none used | predicted (not oracle) LID posterior |

## 2. Theoretical justification (report Section 3 skeleton)

1. **Relaxing CTC conditional independence.** P_CTC(T|h) = Σ_π Π_t p(π_t|h)
   factorises per frame. Conditioning h_fin on Z makes every frame's output
   a function of the whole-sequence LID hypothesis, i.e. p(π_t | F, Ẑ) — the
   same argument by which self-conditioned CTC improves ASR.
2. **Language conflict as interference in shared weights.** With a shared
   encoder, per-language gradients g_i, g_j on shared θ conflict when
   g_i·g_j < 0 — the source of the paper's −1.50 BLEU universal-training drop
   (Table 8). Adding e = W_c Z gives the first Sign2Text layer
   W(h + e) = Wh + W e: a *token-wise, language-dependent* shift of Q/K/V and
   FFN inputs, i.e. a soft language-specific sub-network selected by the
   model's own LID belief (cf. language tokens in Johnson et al. 2017, but
   token-level and inferred rather than oracle).
3. **Making the auxiliary task cooperative.** In the paper ∂L_Txt/∂W_lid = 0,
   so LID can compete with translation for h_int's capacity (consistent with
   Table 9 at 2–3 languages). With conditioning, ∂(L_Txt+L_Attn)/∂W_lid ≠ 0:
   the LID head is trained to be *useful for translation*, not just
   correct; the gate g ∈ (0,1) bounds the injected signal, and g→0 recovers the
   paper exactly (so the proposed model ⊇ baseline).

Prediction to test: gains concentrated on languages the baseline confuses
(low LID accuracy / largest Table-8 drops: csl, ise, swl), and a learned g
clearly above 0.

## 3. Features (the actual blocker) — decision

Retraining SlowFastSign on spoken text is out of scope (no public checkpoint,
4 GB VRAM). Both baseline and proposed use the **same frozen, gloss-free
features**, so the comparison is fair; absolute BLEU is not expected to match
the paper and the report says so.

- **Primary: DINOv2 ViT-S/14 per-frame CLS embeddings** (self-supervised,
  no gloss/text supervision), d_feat = 384, frames resized to 224, every 2nd
  frame, fp16 inference — fits easily in 4 GB.
- Fallback if DINOv2 is too weak: MediaPipe Holistic keypoints (pose upper
  body + both hands, normalised to shoulder centre/width).
- Cache: `data/features/sp10/<video_id>.pt` → float16 tensor [T, d_feat].

## 4. Implementation tasks

| # | Change | File(s) |
|---|---|---|
| 1 | `LIDSelfConditioner(lid_vocab_size, d_model, gate_init=-2.0)` | new `src/models/lid_self_conditioning.py` |
| 2 | Optional `lid_conditioner=` arg, applied to h_int after `lid_head` | `src/models/sign2_lid_text.py` (`encode`) |
| 3 | Compose from config `lid_self_conditioning.enabled` | `src/models/build.py`, `src/utils/config.py` schema |
| 4 | Proposed config `extends: sp10_many_to_one.yaml`, `d_feat: 384` both configs | `configs/sp10_m2o_dinov2.yaml`, `configs/proposed_lid_selfcond.yaml` |
| 5 | Feature extraction script (DINOv2) | new `scripts/extract_features_dinov2.py` |
| 6 | `CachedFeatureReader`; **hard error** if a cache file is missing (no silent stub fallback outside tests) | `src/datasets/feature_cache.py`, `scripts/train.py` |
| 7 | Test-set evaluation: BLEU/ROUGE overall + per source language, LID frame accuracy, gate value; writes `metrics.json` + hypotheses | new `scripts/evaluate.py` |
| 8 | Tests: shape; gate→0 ≡ baseline output; L_Attn gradient reaches `lid_head` only when enabled | `tests/test_lid_self_conditioning.py` |
| 9 | Runner for all main runs | `scripts/run_main_experiments.sh` |

## 5. Minimal experiments (SP-10 many-to-one, all 10 SLs → English)

Identical data, features, hyper-parameters (paper App. B: d=256, ffn 2048,
6+6 layers, Adam 1e-3, λ=(1,5,3)), batch size set by 4 GB VRAM (expect 16,
use grad accumulation to 32 if needed).

| Run | Model | Purpose | Seeds |
|---|---|---|---|
| E1 | Sign2(LID+Text) — paper method | **baseline** | 3 |
| E2 | E1 + LID self-conditioning | **proposed** | 3 |
| E0 (optional) | vanilla Transformer, universal | shows the language conflict exists with our features | 1 |

Report: dev/test BLEU-4 (SacreBLEU) and ROUGE-L, mean ± std over seeds;
paired bootstrap significance (`sacrebleu --paired-bs`) E2 vs E1; per-language
test BLEU (Table-4 style); learned g; LID frame accuracy; parameter count and
time/epoch as the trade-off. If time allows: 2→2 many-to-many E1 vs E2 to test
the Table-9 claim directly.

Go/no-go checkpoints:
- After features: E1 dev BLEU must be clearly > 0 (≥ ~1). If not, switch to
  the keypoint fallback before running anything else.
- After 1 seed each: if E2 ≈ E1, still finish 3 seeds and report honestly
  (a null result with correct analysis beats an unsupported claim).

## 6. Notes from implementation (2026-10-20)

- **Bug fixed:** `scripts/train.py` previously never filtered by target
  language, so "many-to-one" trained on all 15,770 SP-10 records (incl.
  csl→zh, gsg→de, …). `src/training/setup.py::language_filter` now keeps the
  8,300 `→en` records (tested against the real index).
- Features are keyed per video (`410_zh_to_en` → `410_zh.pt`); a missing
  file is a hard error, checked for the whole split before training starts.
  `--allow-stub-features` exists only for smoke tests.
- Our model has 17.47M trainable params vs. the paper's reported 39.48M with
  the same d=256/ffn=2048/6+6 layers — the paper's count likely includes
  parts we don't train (e.g. a feature-extractor head). Report our own
  count; don't claim to match theirs.
- The self-conditioner adds 3,585 params (14×256 + 1).
- CTC blank is index 0 (`<unk>`) in both vocabularies — inherited convention,
  harmless since train targets never contain `<unk>`.

## 7. Deliverables checklist

- [ ] Features extracted for SP-10 train/dev/test (needs finished video download)
- [x] Tasks 1–9 implemented, all tests green (34/34, 2026-10-20); end-to-end
      smoke run of `scripts/run_main_experiments.sh` on stub features passes
      (train → dev/test eval → comparison tables)
- [ ] E1 ×3, E2 ×3 trained; checkpoints + logs uploaded (Google Drive / HF Hub, public link)
- [ ] `README.md` covering the 7 required sections
- [ ] One-page report in the provided LaTeX template (need template ZIP + team details)
- [ ] `TeamXX_26_DLCV_Project_Report.pdf`
