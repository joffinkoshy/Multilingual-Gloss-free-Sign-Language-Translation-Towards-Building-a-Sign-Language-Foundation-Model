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
17.47M-parameter baseline. That is too small for any gain to come from extra
capacity.

| | Paper (baseline, E1) | Proposed (E2) |
|---|---|---|
| LID_tok prediction | auxiliary loss only | loss **and** input to layers 2..N |
| Gradient into the LID head | only from L_LID | from L_LID + L_Txt + L_Attn |
| Effect of LID with few languages | fixed λ₁ = 1, can hurt (paper Table 9) | learnable gate g lets the model down-weight it |
| Test-time language information | none | predicted (not oracle) LID posterior |

Why it should help (full argument in
[PLAN_LID_SELF_CONDITIONING.md](PLAN_LID_SELF_CONDITIONING.md) §2):

1. **It relaxes CTC's conditional independence.** Each frame's output now
   depends on the LID hypothesis for the whole sequence.
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
That checkpoint is not public, and retraining it does not fit in 4 GB of VRAM.
Instead, **both** models use the same frozen, gloss-free **DINOv2 ViT-S/14
per-frame CLS embeddings** (d_feat = 384, every 2nd frame, 224×224). The
comparison between baseline and proposed is therefore fair, but absolute BLEU
is **not** expected to match the paper.

---

## 2. Environment and dependencies

| | Used for the reported runs |
|---|---|
| OS | Windows 11 (development and tests also run on macOS) |
| GPU | NVIDIA RTX 3050, **4 GB** VRAM (driver 592.82, CUDA 13.1 max) |
| Python | 3.11 (Windows) / 3.12 (macOS) |
| PyTorch | CUDA 12.6 wheel (`cu126`) |
| Other | `ffmpeg` on `PATH` (needed by SP-10's `video_converter.py`) |

Python packages ([requirements.txt](requirements.txt)): `torch>=2.1`,
`pyyaml`, `sacrebleu`, `rouge-score`, `numpy`, `pytest`, `opencv-python`.

> **Install CUDA PyTorch *before* `requirements.txt`.** If
> `requirements.txt` runs first, its plain `torch>=2.1` pulls in a CPU-only
> wheel. pip then treats torch as already installed, so a later `cu126`
> install silently does nothing (see
> [HANDOFF_WINDOWS_SESSION2.md](HANDOFF_WINDOWS_SESSION2.md) §1a). `cu121`
> no longer has wheels for current torch releases. Choose a tag at or below
> the CUDA version that `nvidia-smi` reports.

**Windows (PowerShell):**

```powershell
git clone https://github.com/joffinkoshy/Multilingual-Gloss-free-Sign-Language-Translation-Towards-Building-a-Sign-Language-Foundation-Model.git mlslt-repro
cd mlslt-repro
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
pip install -r requirements.txt
python -m pytest tests -q
```

If torch was already installed as a CPU wheel, fix it with:

```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**Linux / macOS (bash):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # Linux+CUDA; on macOS just `pip install torch`
pip install -r requirements.txt
python -m pytest tests -q
```

The test suite has 34 tests and runs on CPU in a few seconds. It covers
shapes, CTC loss, decoding, gradient flow, a tiny overfit run, and the
self-conditioning checks: output shape, g → 0 matching the baseline, and the
L_Attn gradient reaching `lid_head` only when the conditioner is enabled.

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
  8,300 / 1,420 / 2,024 records;
- **cross-to-English** pairs `{gid}_{lang}_to_en` (each non-English video →
  the group's English text).

Many-to-one training keeps only records whose target is English. On train
that is 8,300 records (`src/training/setup.py::language_filter`). Both kinds
of record point to the same video file `{gid}_{lang}`.

### 3b. Download and convert the videos

Clone SP-10's own scripts into `references/SP-10-video/`. On Windows, three
fixes are needed:

1. Skip SP-10's `requirements.txt`: it pins `numpy==1.21.5`, which does not
   install on Python 3.11. Install only `opencv-python` (already in our
   `requirements.txt`).
2. Run with **`python -X utf8`**. The downloader opens the JSON without an
   encoding, so Windows' cp1252 default fails on the CJK and Cyrillic text.
3. Replace `video_converter.py` with our patched copy,
   [scripts/patches/sp10_video_converter_windows.py](scripts/patches/sp10_video_converter_windows.py).
   Upstream hard-codes `/tmp/sign.mp4` and single-quotes the ffmpeg paths,
   and both break on Windows.

```powershell
git clone https://github.com/MLSLT/SP-10 references\SP-10-video
Copy-Item scripts\patches\sp10_video_converter_windows.py references\SP-10-video\video_converter.py
cd references\SP-10-video
python -X utf8 video_downloader.py --save_path ../../data/raw/sp10
python -X utf8 video_converter.py --save_path ../../data/raw/sp10   # re-encodes non-25-fps clips; needs ffmpeg
cd ..\..
```

On Linux or macOS, run the same two commands without the patch. The
downloader can be resumed because it skips files that already exist. A full
download took about 4.5–5 hours.

The resulting layout (`data/raw/` is git-ignored) is:

```
data/raw/sp10/<split>/<group_id>/<lang>.mp4      e.g. data/raw/sp10/train/410/zh.mp4
```

Check that it is complete (expect about 830 / 142 / 214 group folders):

```powershell
Get-ChildItem data\raw\sp10\train | Measure-Object
Get-ChildItem data\raw\sp10\dev   | Measure-Object
Get-ChildItem data\raw\sp10\test  | Measure-Object
```

### 3c. Feature extraction (frozen DINOv2)

```bash
python scripts/extract_features_dinov2.py --video-dir data/raw/sp10 --output-dir data/features/sp10_dinov2 --device cuda
```

- Loads `facebookresearch/dinov2` / `dinov2_vits14` through `torch.hub`. The
  first run needs internet access.
- Keeps every 2nd frame (`--frame-stride 2`, 25 → 12.5 fps), resizes the
  whole frame to 224×224 (`--image-size`), and runs inference in fp16 with
  `--batch-size 64`.
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
| Optimiser | Adam, lr 1e-3, β = (0.9, 0.999), eps 1e-8 | paper / reference code |
| LR schedule | ReduceLROnPlateau on dev BLEU, patience 10, factor 0.1 | reference code |
| Batch size | 16 × 2 gradient-accumulation steps = 32 effective | 4 GB VRAM |
| Dropout / grad clip | 0.1 / 5.0 | assumption |
| Early stopping | 10 epochs without dev-BLEU gain, max 100 epochs | |
| Decoding | beam 5, CTC weight 0.3 | assumption |
| Self-conditioning gate | init logit −2.0 (g ≈ 0.12), learnable | ours |
| Seeds | 1, 2, 3 | |

`python -m src.utils.config --assumptions` lists every config field that is an
assumption rather than a value from the paper.

### Single runs

```bash
# E1 baseline
python scripts/train.py --config configs/sp10_m2o_dinov2.yaml --model-dir experiments/baseline/seed1 --seed 1 --device cuda
# E2 proposed
python scripts/train.py --config configs/proposed_lid_selfcond.yaml --model-dir experiments/proposed/seed1 --seed 1 --device cuda
```

`--device` defaults to `cpu`, so pass `--device cuda`. Resume an interrupted
run with `--resume experiments/<run>/seed1/latest.ckpt`. Each `--model-dir`
receives `best.ckpt` (best dev BLEU), `latest.ckpt`, `train.log` (per-epoch
loss, dev BLEU and gate value `g`), `config.resolved.yaml` and
`txt_vocab.txt`.

### All main experiments

[scripts/run_main_experiments.sh](scripts/run_main_experiments.sh) trains E1
and E2 for each seed, evaluates each on dev and test, and writes the
comparison tables. Runs that already have a `best.ckpt` or `metrics.json` are
skipped, so it is safe to re-run.

```bash
SEEDS="1 2 3" DEVICE=cuda bash scripts/run_main_experiments.sh
WITH_UNIVERSAL=1 SEEDS="1 2 3" DEVICE=cuda bash scripts/run_main_experiments.sh   # also trains E0 (first seed only)
EXTRA_ARGS="--allow-stub-features --max-epochs 1" SEEDS=1 DEVICE=cpu bash scripts/run_main_experiments.sh   # pipeline smoke test, no data needed
```

On Windows, the script runs as-is from Git Bash with the venv activated.
This is the equivalent in PowerShell:

```powershell
$seeds = 1, 2, 3
foreach ($s in $seeds) {
  foreach ($run in @(@{name="baseline"; cfg="configs/sp10_m2o_dinov2.yaml"},
                     @{name="proposed"; cfg="configs/proposed_lid_selfcond.yaml"})) {
    $dir = "experiments/$($run.name)/seed$s"
    if (-not (Test-Path "$dir/best.ckpt")) {
      python scripts/train.py --config $run.cfg --model-dir $dir --seed $s --device cuda
    }
    foreach ($split in "dev", "test") {
      python scripts/evaluate.py --config $run.cfg --model-dir $dir --split $split --device cuda
    }
  }
}
foreach ($split in "dev", "test") {
  python scripts/compare_runs.py --split $split `
    --baseline experiments/baseline/seed1 experiments/baseline/seed2 experiments/baseline/seed3 `
    --proposed experiments/proposed/seed1 experiments/proposed/seed2 experiments/proposed/seed3 `
    --output experiments/results/comparison_$split
}
```

---

## 5. Inference and evaluation

### Evaluate one checkpoint

```bash
python scripts/evaluate.py --config configs/proposed_lid_selfcond.yaml --model-dir experiments/proposed/seed1 --split test --device cuda
```

| | Path |
|---|---|
| Input | `<model-dir>/best.ckpt` (or `--checkpoint <path>`), `<model-dir>/txt_vocab.txt`, `data/splits/sp10/<split>.json`, `data/features/sp10_dinov2/` |
| Output | `<model-dir>/eval_<split>/` (or `--output-dir <dir>`) |
| `metrics.json` | SacreBLEU BLEU-4 and ROUGE-L, overall and per source sign language; LID frame and utterance accuracy; learned gate `g` (proposed only); parameter count |
| `hypotheses.tsv` | `video_id, src_lang, reference, hypothesis` |

Decoding is joint CTC/attention beam search (beam 5, CTC weight 0.3, from the
config). Use the same `--config` the model was trained with.

### Compare baseline and proposed

```bash
python scripts/compare_runs.py --split test --baseline experiments/baseline/seed1 experiments/baseline/seed2 experiments/baseline/seed3 --proposed experiments/proposed/seed1 experiments/proposed/seed2 experiments/proposed/seed3 --output experiments/results/comparison_test
```

This reads `<dir>/eval_<split>/{metrics.json,hypotheses.tsv}` for each seed.
List the seeds in the same order for both groups. It writes:

- `experiments/results/comparison_test.json`: mean ± std over seeds and
  per-language BLEU.
- `experiments/results/comparison_test.md`: the same as a Markdown table,
  plus a paired bootstrap significance test (Koehn 2004,
  `--bootstrap-samples 1000`) on corpus BLEU for each matched seed pair.

---

## 6. Checkpoints and results

> **TODO — no real results yet.** SP-10 videos are still being downloaded
> and features have not been extracted. The numbers below will be filled in
> from `experiments/results/comparison_{dev,test}.md` once E1 and E2 have
> finished. Nothing in this section is a measured result.

### Checkpoints and logs

| Run | Seeds | Checkpoints (`best.ckpt`) + `train.log` + `eval_*/` |
|---|---|---|
| E1 baseline | 1, 2, 3 | **TODO: add public link (Google Drive / HF Hub)** |
| E2 proposed | 1, 2, 3 | **TODO: add public link (Google Drive / HF Hub)** |
| DINOv2 features (`data/features/sp10_dinov2`) | n/a | **TODO: link, or rebuild with §3c** |

To evaluate a downloaded checkpoint, put `best.ckpt` and `txt_vocab.txt` in
`experiments/<run>/seed<k>/` and run the §5 commands.

### Results: SP-10 many-to-one (→ English), test set, mean ± std over 3 seeds

| Model | BLEU-4 | ROUGE-L | LID utt. acc. | Learned g | Params |
|---|---|---|---|---|---|
| E1 Sign2(LID+Text) (baseline) | TODO | TODO | TODO | n/a | TODO |
| E2 + LID self-conditioning (ours) | TODO | TODO | TODO | TODO | TODO |
| Paired bootstrap p (E2 vs E1) | TODO | | | | |

### Per-language test BLEU

| | csl | ukl | rsl | icl | gsg | ise | bqn | swl | lls | bfi |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| E2 | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

Absolute scores are not comparable to the paper's Table 4, because our
features are frozen DINOv2 rather than retrained SlowFastSign (see §1).

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
tests/                      pytest suite (34 tests)
```

### Reproduction of the paper

| File | Contents |
|---|---|
| `src/models/sign2_lid_text.py` | Sign2(LID+Text): split encoder, LID-CTC head after layer 1, text-CTC head, attention decoder |
| `src/models/baseline_transformer.py` | vanilla Transformer (E0) |
| `src/models/layers.py` | Transformer building blocks, masks |
| `src/losses/` | LID-CTC, text-CTC, attention loss, λ-weighted joint loss |
| `src/evaluation/decode.py`, `metrics.py` | joint CTC/attention beam search, SacreBLEU / ROUGE-L |
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
- `scripts/patches/sp10_video_converter_windows.py`: the Windows fix from §3b.

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
[REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md),
`HANDOFF_*.md`) record how the project developed. **This README is the
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
