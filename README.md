# LID Self-Conditioning for Multilingual Gloss-free Sign Language Translation

Course project for **DA 6021 Deep Learning for Computer Vision**.

We reproduce the Sign2(LID+Text) model from the paper below and test one
change to it, **LID self-conditioning**. Both models are trained on SP-10
many-to-one (10 sign languages → English) using the same frozen DINOv2 frame
features.

---

## 1. Project overview

### The paper

> Sihan Tan, Taro Miyazaki, Kazuhiro Nakadai.
> *Multilingual Gloss-free Sign Language Translation: Towards Building a Sign
> Language Foundation Model.* ACL 2025 (short paper).
> Original code: <https://github.com/Claire874/Gloss-free-MLSLT>

The paper trains one Transformer that translates many sign languages without
gloss annotations. Its Sign2(LID+Text) model adds a **token-level
sign-language-ID (LID) head** on the output of the first encoder layer
(`h_int`), trained with CTC (λ₁). The remaining encoder layers feed a text CTC
head (λ₂) and an attention decoder (λ₃), with λ = (1, 5, 3).

### Our novelty: LID self-conditioning

In the paper the LID-CTC prediction is **only a loss target and is then
discarded**: encoder layers 2..N never see it. We feed the LID-CTC posterior
back into the residual stream before the Sign2Text layers. This is the
self-conditioned CTC idea from Nozaki & Komatsu (Interspeech 2021):

```
Z      = softmax(W_lid · h_int)                 # [B, |F|, V_lid]; the paper already computes this
h_int' = h_int + g · W_c · Z,   g = sigmoid(a)   # NEW: W_c ∈ R^{d×V_lid}, scalar gate a
h_fin  = Enc_{2..N}(h_int')                     # unchanged
```

Everything else is identical to the paper: losses, λ, decoder and decoding.
The conditioner adds **3,585 parameters** (14 × 256 + 1), about 0.02% of our
18.56M-parameter baseline. That is too small for any gain to come from extra
capacity.

| | Paper (baseline, E1) | Proposed (E2) |
|---|---|---|
| LID_tok prediction | auxiliary loss only | loss **and** input to layers 2..N |
| Gradient into the LID head | only from L_LID | from L_LID + L_Txt + L_Attn |
| Effect of LID with few languages | fixed λ₁ = 1, can hurt (paper Table 9) | learnable gate g lets the model down-weight it |
| Test-time language information | none | predicted (not oracle) LID posterior |

Why it should help (full argument in
[PLAN_LID_SELF_CONDITIONING.md](PLAN_LID_SELF_CONDITIONING.md) §2):

These are hypotheses, not proven mechanisms (see §6 for what the results do
and do not support):

1. **It makes language identity an explicit input.** Each frame's later layers
   see the model's own language hypothesis, in the spirit of self-conditioned
   CTC.
2. **It gives soft language-specific sub-networks.** Adding `W_c Z` shifts the
   Q/K/V and FFN inputs of the shared layers per token and per language. This
   reduces interference between languages in the shared weights, which is the
   cause of the paper's universal-training drop (Table 8).
3. **It makes the auxiliary task cooperate with translation.** In the paper,
   ∂L_Txt/∂W_lid = 0. With conditioning, the translation losses also train the
   LID head. As g → 0 the model reduces exactly to the paper's model, so the
   proposed model contains the baseline.

The gate starts at `g = sigmoid(-2) ≈ 0.12`.

### Features (deviation from the paper)

The paper uses SlowFastSign features retrained with spoken-text supervision.
That checkpoint is not public, and retraining it is out of reach on an 8 GB
Apple M1.
Instead, **both** models use the same frozen, gloss-free **DINOv2 ViT-S/14
per-frame CLS embeddings** (d_feat = 384, every 2nd frame, 224×224). The
comparison between baseline and proposed is therefore fair, but absolute BLEU
is **not** expected to match the paper.

---

## 2. Environment and dependencies

| | Used for the reported runs |
|---|---|
| Machine | MacBook, **Apple M1**, **8 GB** unified memory |
| OS | macOS |
| Accelerator | PyTorch **MPS** backend (Apple GPU), no CUDA. Everything also runs on CPU |
| Python | 3.12 (venv) |
| PyTorch | standard macOS wheel (`pip install torch`), MPS is built in |
| Other | `ffmpeg` (needed by SP-10's `video_converter.py`), from Homebrew |

Python packages ([requirements.txt](requirements.txt), minimum versions):
`torch>=2.1`, `pyyaml`, `sacrebleu`, `rouge-score`, `numpy`, `pytest`,
`opencv-python`. The exact versions used for every reported run are pinned in
[requirements.lock.txt](requirements.lock.txt) (torch 2.13.0, numpy 2.5.2,
sacrebleu 2.6.0, Python 3.12). To match them exactly, use
`pip install -r requirements.lock.txt` instead of `requirements.txt`. There is
no CUDA requirement: all reported runs used the Apple MPS backend.

```bash
git clone https://github.com/joffinkoshy/Multilingual-Gloss-free-Sign-Language-Translation-Towards-Building-a-Sign-Language-Foundation-Model.git mlslt-repro
cd mlslt-repro
brew install ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt      # exact versions of the reported runs (or: pip install torch && pip install -r requirements.txt)
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # should print True
python -m pytest tests -q
```

The test suite has 45 tests and runs on CPU in a few seconds. It covers
shapes, CTC loss, decoding, gradient flow, a tiny overfit run, and the
self-conditioning checks: output shape, g → 0 matching the baseline, and the
L_Attn gradient reaching `lid_head` only when the conditioner is enabled.

**Using MPS:** pass `--device mps` to every training, evaluation and
extraction command, or set `DEVICE=mps` for the runner script. PyTorch's
`nn.CTCLoss` has no MPS kernel. `scripts/train.py` and `scripts/evaluate.py`
therefore set `PYTORCH_ENABLE_MPS_FALLBACK=1` themselves, and the two CTC
losses run on the CPU automatically while everything else stays on the GPU.

> **Verified:** all reported runs (seven trainings, 11,741 videos of feature
> extraction) were done on this setup. Feature extraction took about 3.5 hours,
> each training run about 1.5–2 hours (about 2–2.5 minutes per epoch).

---

## 3. Dataset preparation (SP-10)

SP-10 (<https://github.com/MLSLT/SP-10>) is public and needs no access
request. It has 830 / 142 / 214 train / dev / test groups. Each group contains
the same sentence signed in up to 10 sign languages, each with its spoken-
language text.

### 3a. Index files (already in the repo)

The upstream annotation files `dataset/{train,dev,test}.json` are committed as
`references/sp10_{train,dev,test}_raw.json`. They are converted into this
repo's index format, `data/splits/sp10/{train,dev,test}.json`, which is also
committed. To rebuild:

```bash
python scripts/convert_sp10_index.py --raw-dir references --out-dir data/splits/sp10
```

Each group produces two kinds of record:

- **same-language** pairs `{gid}_{lang}` (sign video → its own spoken text),
  8,300 / 1,420 / 2,021 records (3 csl test groups have no video upstream and are skipped);
- **cross-to-English** pairs `{gid}_{lang}_to_en` (each non-English video →
  the group's English text).

Many-to-one training keeps only records whose target is English. On train
that is 8,300 records (`src/training/setup.py::language_filter`). Both kinds
of record point to the same video file `{gid}_{lang}`.

### 3b. Download and convert the videos

Use SP-10's own scripts. The downloader pulls clips from
`media.spreadthesign.com`, and the converter re-encodes clips that are not
25 fps with ffmpeg:

```bash
git clone https://github.com/MLSLT/SP-10 references/SP-10-video
cd references/SP-10-video
python video_downloader.py --save_path ../../data/raw/sp10
python video_converter.py --save_path ../../data/raw/sp10
cd ../..
```

Skip SP-10's own `requirements.txt`. It pins an old `numpy==1.21.5`, and the
scripts only need `opencv-python`, which our `requirements.txt` already
installs. The downloader can be resumed because it skips files that already
exist. On Windows, the upstream converter needs the fix in
[scripts/patches/sp10_video_converter_windows.py](scripts/patches/sp10_video_converter_windows.py).

The resulting layout (`data/raw/` is git-ignored) is:

```
data/raw/sp10/<split>/<group_id>/<lang>.mp4      e.g. data/raw/sp10/train/410/zh.mp4
```

Check that it is complete (expect about 830 / 142 / 214 group folders):

```bash
for s in train dev test; do echo "$s: $(ls data/raw/sp10/$s | wc -l)"; done
```

### 3c. Feature extraction (frozen DINOv2)

```bash
python scripts/extract_features_dinov2.py --video-dir data/raw/sp10 --output-dir data/features/sp10_dinov2 --device mps
```

- Loads `facebookresearch/dinov2` / `dinov2_vits14` through `torch.hub`. The
  first run needs internet access.
- Keeps every 2nd frame (`--frame-stride 2`, 25 → 12.5 fps) and resizes the
  whole frame to 224×224 (`--image-size`). The default `--batch-size` is 64.
  Without CUDA the default device is CPU, so pass `--device mps`. On MPS the
  encoder runs in fp32 (fp16 autocast is CUDA-only), and the features are
  still saved as fp16. If memory runs short, use `--batch-size 32`.
- Writes `data/features/sp10_dinov2/<group_id>_<lang>.pt`, a float16 tensor
  of shape `[T, 384]`, plus `extraction_summary.json` (frame statistics and
  any unreadable videos).
- Can be resumed because it skips existing `.pt` files.

Training refuses to start if any feature file for a split is missing
(`data.require_real_features: true`). `--allow-stub-features` exists only for
smoke tests.

---

## 4. Training

| Run | Config | Model |
|---|---|---|
| **E1 baseline** | [configs/sp10_m2o_dinov2.yaml](configs/sp10_m2o_dinov2.yaml) | paper's Sign2(LID+Text) |
| **E2 proposed** | [configs/proposed_lid_selfcond.yaml](configs/proposed_lid_selfcond.yaml) | E1 + `lid_self_conditioning.enabled: true` |
| E0 (optional) | [configs/sp10_m2o_dinov2_universal.yaml](configs/sp10_m2o_dinov2_universal.yaml) | vanilla Transformer, no LID branch |

The configs inherit through `extends:`: `base_reproduction.yaml` →
`sp10_many_to_one.yaml` → `sp10_m2o_dinov2.yaml` →
`proposed_lid_selfcond.yaml`. The only difference between E1 and E2 is the
self-conditioning flag.

**Key hyper-parameters (shared by E1 and E2):**

| | Value | Source |
|---|---|---|
| d_model / FFN / heads | 256 / 2048 / 8 | paper App. B (heads: assumption) |
| Encoder + decoder layers | 6 + 6, LID head after encoder layer 1 | paper |
| d_feat | 384 (DINOv2 ViT-S/14) | ours |
| Loss weights λ = (LID, Txt-CTC, Attn) | (1, 5, 3) | paper Eq. 5 |
| Optimiser | Adam, **lr 1e-4**, β = (0.9, 0.999), eps 1e-8 | **deviation:** the paper's 1e-3 stalled in our code (§6) |
| LR schedule | ReduceLROnPlateau on dev BLEU, patience 10, factor 0.1 | reference code |
| Batch size | 16 × 2 gradient-accumulation steps = 32 effective | small-memory GPU (paper: 64) |
| Dropout / grad clip | 0.1 / 5.0 | assumption |
| Early stopping | 10 epochs without dev-BLEU gain, max 100 epochs | |
| Decoding | greedy for dev checks and final scores (beam 5 + CTC 0.3 selectable) | chosen on dev (§5) |
| Self-conditioning gate | init logit −2.0 (g ≈ 0.12), learnable | ours |
| Seeds | 1, 2, 3 | |

`python -m src.utils.config --assumptions` lists every config field that is an
assumption rather than a value from the paper.

### Single runs

```bash
# E1 baseline
python scripts/train.py --config configs/sp10_m2o_dinov2.yaml --model-dir experiments/baseline/seed1 --seed 1 --device mps
# E2 proposed
python scripts/train.py --config configs/proposed_lid_selfcond.yaml --model-dir experiments/proposed/seed1 --seed 1 --device mps
```

`--device` defaults to `cpu`, so pass `--device mps`. Resume an interrupted
run with `--resume experiments/<run>/seed1/latest.ckpt`. Each `--model-dir`
receives:

- `best.ckpt` (best dev BLEU) and `latest.ckpt`;
- `train.log`, with per-epoch loss, dev BLEU and the gate value `g`;
- `config.resolved.yaml` and `txt_vocab.txt`.

The per-epoch dev BLEU used for checkpoint selection and early stopping comes
from greedy decoding, run on the CPU (`decode.dev_strategy`, `decode.dev_device`).
The reported scores use the same greedy decoder (§5).

### All main experiments

[scripts/run_main_experiments.sh](scripts/run_main_experiments.sh) trains E1
and E2 for each seed, evaluates each on dev and test, and writes the
comparison tables. Runs that already have a `best.ckpt` or `metrics.json` are
skipped, so it is safe to re-run.

```bash
SEEDS="1 2 3" DEVICE=mps bash scripts/run_main_experiments.sh
WITH_UNIVERSAL=1 SEEDS="1 2 3" DEVICE=mps bash scripts/run_main_experiments.sh   # also trains E0 (first seed only)
EXTRA_ARGS="--allow-stub-features --max-epochs 1" SEEDS=1 DEVICE=mps bash scripts/run_main_experiments.sh   # pipeline smoke test, no data needed
```

Other variables: `PYTHON` (interpreter, default `python`) and `EXP_DIR`
(output root, default `experiments`).

---

## 5. Inference and evaluation

### Evaluate one checkpoint

```bash
python scripts/evaluate.py --config configs/proposed_lid_selfcond.yaml --model-dir experiments/proposed/seed1 --split test --device mps
```

| | Path |
|---|---|
| Input | `<model-dir>/best.ckpt` (or `--checkpoint <path>`), `<model-dir>/txt_vocab.txt`, `data/splits/sp10/<split>.json`, `data/features/sp10_dinov2/` |
| Output | `<model-dir>/eval_<split>/` (or `--output-dir <dir>`) |
| `metrics.json` | SacreBLEU BLEU-4 and ROUGE-L, overall and per source sign language; LID frame and utterance accuracy; learned gate `g` (proposed only); parameter count |
| `hypotheses.tsv` | `video_id, src_lang, reference, hypothesis` |

Decoding is greedy (`decode.final_strategy: greedy` in `sp10_m2o_dinov2.yaml`).
This was chosen on the **dev** set and applies to both models. On the baseline
seed 1, dev BLEU was greedy 4.99, beam 4.73, beam + CTC rescoring (weight 0.3)
4.06, and the proposed seed 1 shows the same ordering (4.79 / 4.63 / 3.86), so our
CTC rescoring lowers BLEU. Beam search stays available:

```bash
python scripts/evaluate.py --config configs/sp10_m2o_dinov2.yaml --model-dir experiments/baseline/seed1 --split test --strategy beam --ctc-weight 0.3 --device mps
```

Use the same `--config` the model was trained with. The `metrics.json` records
the decoding used. The beam + CTC results for seed 1 are kept in
`experiments/*/seed1/eval_*_beam5ctc0.3/`.

### Reproduce the reported numbers from the released checkpoints

The quickest way to check the results table in §6, without retraining. From a
fresh clone (after §2):

1. Download `mlslt_checkpoints.zip` from the Google Drive folder in §6 and unzip
   it into `experiments/` (it contains `baseline/`, `proposed/`, `universal/`,
   each with `seed*/best.ckpt`, `txt_vocab.txt`, `config.resolved.yaml`,
   `train.log`):
   ```bash
   unzip -o mlslt_checkpoints.zip -d experiments/
   ```
2. Get the DINOv2 features. Either download `sp10_dinov2_features.zip` from the
   same folder and unzip it into `data/features/`
   (`unzip -o sp10_dinov2_features.zip -d data/features/`), or rebuild them from the
   videos with §3 (about 3.5 hours on an M1).
3. Run the evaluation of all six checkpoints and the comparison:
   ```bash
   DEVICE=cpu bash scripts/reproduce_reported_metrics.sh      # about 8 minutes on an M1 CPU; DEVICE=mps also works
   ```
   This writes `experiments/results/reproduced_test.md`, which should match
   [experiments/results/comparison_test.md](experiments/results/comparison_test.md)
   (BLEU 3.04 ± 0.77 for E1 and 3.62 ± 0.28 for E2), because the checkpoints and
   the greedy decoder are the same. Small differences in the last digit can come
   from the bootstrap seed or the device.

### Compare baseline and proposed

```bash
python scripts/compare_runs.py --split test --baseline experiments/baseline/seed1 experiments/baseline/seed2 experiments/baseline/seed3 --proposed experiments/proposed/seed1 experiments/proposed/seed2 experiments/proposed/seed3 --output experiments/results/comparison_test
```

This reads `<dir>/eval_<split>/{metrics.json,hypotheses.tsv}` for each seed.
List the seeds in the same order for both groups. It writes:

- `experiments/results/comparison_test.json`
- `experiments/results/comparison_test.md`

Both contain BLEU and ROUGE-L as mean ± std over seeds, per-language BLEU,
and a paired bootstrap test (Koehn 2004, `--bootstrap-samples 1000`) on
corpus BLEU for each matched seed pair. The test reports Δ, a 95% CI and p.

---

## 6. Checkpoints and results

All numbers below were produced by the commands in §4–§5 on an Apple M1 (8 GB),
and are copied from `experiments/results/comparison_{dev,test}.md`. Decoding is
greedy (see §5 for why). Per-run `train.log`, `metrics.json`, `hypotheses.tsv`
and `config.resolved.yaml` are committed under `experiments/`. The checkpoints
are hosted outside git. We upload slim copies (model weights only, 85 MB each, 519 MB zipped as `mlslt_checkpoints.zip`; the full training files with optimizer state are 233 MB each).

### Checkpoints

| Run | Seeds | Checkpoints (`best.ckpt` + `txt_vocab.txt`) |
|---|---|---|
| E1 baseline | 1, 2, 3 | [Google Drive folder](https://drive.google.com/drive/folders/1y_Jx64GrD9BbjK2w7Uplpc_t7g8_l9p3?usp=drive_link) (`baseline/seed*/`) |
| E2 proposed | 1, 2, 3 | [Google Drive folder](https://drive.google.com/drive/folders/1y_Jx64GrD9BbjK2w7Uplpc_t7g8_l9p3?usp=drive_link) (`proposed/seed*/`) |
| E0 vanilla Transformer | 1 | [Google Drive folder](https://drive.google.com/drive/folders/1y_Jx64GrD9BbjK2w7Uplpc_t7g8_l9p3?usp=drive_link) (`universal/seed*/`) |
| DINOv2 features (`data/features/sp10_dinov2`, 11,741 files, 441 MB zipped as `sp10_dinov2_features.zip`) | n/a | same Drive folder; or rebuild with §3c |

The same checkpoints are also attached to the GitHub release
[v1.0](https://github.com/joffinkoshy/Multilingual-Gloss-free-Sign-Language-Translation-Towards-Building-a-Sign-Language-Foundation-Model/releases/tag/v1.0) (`mlslt_checkpoints.zip`).

To evaluate a downloaded checkpoint, put `best.ckpt` and `txt_vocab.txt` in
`experiments/<run>/seed<k>/` and run the §5 commands.

### Main result: SP-10 many-to-one (→ English), test set

SacreBLEU-4 and ROUGE-L, mean ± std over seeds 1–3 (E0: one seed).

| Model | BLEU-4 | ROUGE-L | LID utt. acc. (%) | Learned g | Params |
|---|---|---|---|---|---|
| E0 vanilla Transformer, universal (1 seed) | 3.56 | 17.54 | n/a | n/a | 18,188,425 |
| E1 Sign2(LID+Text) (baseline) | 3.04 ± 0.77 | 16.16 ± 1.71 | 91.11 ± 6.84 | n/a | 18,556,192 |
| **E2 + LID self-conditioning (ours)** | **3.62 ± 0.28** | **17.89 ± 0.15** | **98.33 ± 0.92** | 0.120 | 18,559,777 |
| Δ (E2 − E1) | **+0.58** | **+1.73** | +7.2 | | +3,585 |

Dev set (the same checkpoints, greedy decoding):

| Model | BLEU-4 | ROUGE-L | LID utt. acc. (%) |
|---|---|---|---|
| E0 (1 seed) | 5.18 | 18.61 | n/a |
| E1 baseline | 3.64 ± 1.18 | 16.07 ± 2.34 | 90.54 ± 7.19 |
| E2 ours | 4.95 ± 0.31 | 19.36 ± 0.63 | 98.54 ± 0.95 |

### Per seed (test BLEU) and paired bootstrap

| Seed | E1 | E2 | Δ | 95% CI (bootstrap over test examples) | p |
|---|---|---|---|---|---|
| 1 | 3.93 | 3.58 | -0.34 | [-0.83, +0.13] | 0.918 |
| 2 | 2.70 | 3.36 | +0.68 | [+0.16, +1.20] | 0.005 |
| 3 | 2.50 | 3.93 | +1.44 | [+0.99, +1.93] | 0.000 |

### Per-language test BLEU (mean over 3 seeds)

| | csl | ukl | rsl | icl | gsg | ise | bqn | swl | lls | bfi |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 3.05 | 3.25 | 3.13 | 3.20 | 2.59 | 2.45 | 2.20 | 3.46 | 3.09 | 3.18 |
| E2 | 3.64 | 3.47 | 3.25 | 3.58 | 3.74 | 3.48 | 2.29 | 3.92 | 4.15 | 4.00 |

### How to read these results

- **What is supported:** E2 has a higher mean test BLEU (+0.58, +19%) and
  ROUGE-L (+1.73), is better in **all 10 languages**, and varies less between
  seeds (std 0.28 vs 0.77). The baseline's LID head is less reliable in its weak
  seeds (88.3% and 86.1% utterance accuracy, against 98.9% in its best seed);
  E2's stays at 97–99% in every seed.
- **What is not supported:** with 3 seeds the gain is **not statistically
  conclusive**. The per-seed differences are −0.35, +0.66 and +1.43 (std 0.89),
  and the bootstrap above only measures test-set sampling noise, not variation
  between training runs. We did not run an ablation (for example a stop-gradient
  version) to show *why* it helps. The learned gate barely moved from its
  initial 0.119 (final 0.119–0.121), so any benefit comes from the extra
  gradient path into the LID head rather than from a strong conditioning signal.
- **E0:** the plain Transformer without LID/CTC (one seed) scores 3.56
  on test, inside E1's seed range (2.50–3.93). With our features, the paper's
  dual-CTC model is not clearly better than a plain one. We did not train
  per-language ("individual") models.
- **Early stopping** ended E1 runs at 54, 39 and 35 epochs and E2 runs at 40,
  43 and 66, which could favour E2 slightly.
- **Comparison with the paper:** E1's test BLEU (3.04) is close to the paper's universal
  baseline (3.10, Table 4) but below its full method (5.18). Absolute scores are
  not comparable, because our features are frozen DINOv2 rather than retrained
  SlowFastSign (see §1), and we report BLEU pooled over all 10 languages while
  the paper averages per language.
- **Deviations from the paper** (applied equally to E1 and E2): Adam learning rate
  1e-4 instead of 1e-3 (the paper's value stalled in our implementation: dev BLEU
  0.01 and early stop at epoch 13, log kept in
  `experiments/baseline/seed1_lr1e-3_failed/`); greedy decoding instead of beam
  search with CTC rescoring; DINOv2 features. Each run took about 1.5–2 hours.

---

## 7. Code organisation

```
configs/                    YAML configs (extends: chains, see §4)
data/splits/sp10/           index files (committed); data/raw/, data/features/ are git-ignored
references/                 paper's reference code (Gloss-free-MLSLT) and SP-10 raw annotations
scripts/                    CLI entry points (train, evaluate, compare, data prep)
src/datasets/               SP-10 / PHOENIX14T loaders, tokenizer, feature cache
src/models/                 model code (see below)
src/losses/                 CTC and attention losses, joint λ-weighted loss
src/training/               trainer, checkpointing, LR schedulers, run setup
src/evaluation/             beam search decoding, BLEU / ROUGE metrics
tests/                      pytest suite (45 tests)
```

### Reproduction of the paper

| File | Contents |
|---|---|
| `src/models/sign2_lid_text.py` | Sign2(LID+Text): split encoder, LID-CTC head after layer 1, text-CTC head, attention decoder |
| `src/models/baseline_transformer.py` | vanilla Transformer (E0) |
| `src/models/layers.py` | Transformer building blocks, masks |
| `src/losses/` | LID-CTC, text-CTC, attention loss, λ-weighted joint loss |
| `src/evaluation/decode.py`, `metrics.py` | greedy and batched beam search with CTC rescoring (the per-example reference version is kept for tests), SacreBLEU / ROUGE-L |
| `src/training/`, `scripts/train.py` | training loop, plateau scheduler, early stopping on dev BLEU |
| `scripts/convert_sp10_index.py`, `src/datasets/sp10.py` | SP-10 data |
| `configs/base_reproduction.yaml`, `sp10_many_to_one.yaml`, `sp10_m2o_dinov2.yaml` | paper hyper-parameters (assumptions are marked inline) |

### The proposed change (LID self-conditioning)

| File | Contents |
|---|---|
| **`src/models/lid_self_conditioning.py`** | `LIDSelfConditioner`: `h_int + sigmoid(a) · W_c · exp(lid_log_probs)` |
| **`src/models/sign2_lid_text.py`** | the only hook: optional `lid_conditioner=` constructor argument, applied to `h_int` in `encode()` after the LID head. The default `None` gives the paper's model unchanged |
| **`src/models/build.py`** | builds the conditioner from `config.lid_self_conditioning` (`enabled`, `gate_init`, `learn_gate`) |
| `configs/proposed_lid_selfcond.yaml` | the E2 config |
| `tests/test_lid_self_conditioning.py` | shape, g → 0 equivalence with the baseline, and gradient-path tests |

### Shared pipeline for this project

- `scripts/extract_features_dinov2.py`: frozen DINOv2 features.
- `src/datasets/feature_cache.py`: cached-feature reader that fails hard on
  missing files.
- `scripts/evaluate.py`, `scripts/compare_runs.py`,
  `scripts/run_main_experiments.sh`: evaluation and comparison.
- `scripts/patches/sp10_video_converter_windows.py`: fix for SP-10's
  converter on Windows (not needed on macOS).

### Not part of the reported project

These are kept in the repo from an earlier plan, but are **not run, reported
or claimed**:

- the "V1" modules `src/models/family_router.py`, `src/datasets/augment.py`
  and `src/models/pretrained_encoder.py`;
- their configs `configs/v1_*.yaml`, plus `scripts/run_ablation.sh` and
  `tests/test_v1_improvements.py`;
- the SlowFastSign path (`scripts/extract_features.py`,
  `src/datasets/feature_extraction.py`) and the PHOENIX14T, CSL-Daily and
  one-to-one configs and runners.

The planning and hand-off notes
([PLAN_LID_SELF_CONDITIONING.md](PLAN_LID_SELF_CONDITIONING.md),
[docs/archive/REPRODUCTION_STATUS.md](docs/archive/REPRODUCTION_STATUS.md),
`docs/archive/HANDOFF_*.md`) record how the project developed. **This README is the
reference for the submission.**

---

### Citation

```bibtex
@inproceedings{tan2025multilingual,
  title     = {Multilingual Gloss-free Sign Language Translation: Towards Building a Sign Language Foundation Model},
  author    = {Tan, Sihan and Miyazaki, Taro and Nakadai, Kazuhiro},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  year      = {2025}
}
```
