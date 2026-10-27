# Handoff: Sign2(LID+Text) reproduction — Mac session → RTX 3050 machine

**Read this file first, then `IMPLEMENTATION_HANDOFF.md`, then
`REPRODUCTION_STATUS.md`.** This document exists because all the work below
was done in a separate Claude Code session (on a Mac, no GPU) and needs to
transfer cleanly to a fresh Claude Code instance on this machine, which has
no memory of any of it. Don't re-derive decisions already made here — treat
this as ground truth for "what happened and why," and `IMPLEMENTATION_HANDOFF.md`
as ground truth for "what the paper says and what we assumed where it doesn't."

**Hardware correction (found 2026-08-27, this machine):** `IMPLEMENTATION_HANDOFF.md`
assumed an 8GB RTX 3050. `nvidia-smi` on this actual machine shows **4096MiB
(4GB) VRAM** — the RTX 3050 ships in both 4GB and 8GB variants, and this is
the 4GB one. Treat every batch-size/model-size assumption in the handoff as
needing to be re-checked downward once real training starts; nothing to act
on yet, just don't be surprised by earlier-than-expected OOMs.

---

## 1. What this project is

Reproducing **Sign2(LID+Text)** (Tan, Miyazaki, Nakadai — ACL 2025,
*"Multilingual Gloss-free Sign Language Translation: Towards Building a Sign
Language Foundation Model"*), then implementing exactly three researched
improvements (family-aware routing, cost-reduced augmentation, cost-reduced
multilingual pretraining) as a sequential ablation ladder. Full spec,
target numbers, architecture diagrams, and every explicitly-flagged
assumption live in **`IMPLEMENTATION_HANDOFF.md`** — that file is the
project's single source of truth and was produced by a prior research phase
(paper analysis + reproduction planning + cost-minimization decisions for a
"local desktop, 8GB VRAM, no cloud burst" hardware constraint, which is this
RTX 3050 machine). Do not redo that research or substitute new improvement
ideas — Section 9 of that file is authoritative on what "improvements" means
here.

**This handoff document covers everything that happened in the follow-up
implementation session** (Mac, CPU-only, no GPU, no dataset video access) —
i.e. everything built, tested, and discovered *after* that research phase,
up to the point of switching to this machine.

## 2. What already exists in this repo (built + verified on the Mac session)

The full codebase for Stages 1–7 (`IMPLEMENTATION_HANDOFF.md` Section 5) is
implemented and passes **22/22 tests** (`python -m pytest tests/ -q`):

- Both architectures: `src/models/baseline_transformer.py` (vanilla
  Transformer, Section 2.3 Architecture A) and `src/models/sign2_lid_text.py`
  (the hierarchical LID+Txt CTC + attention model, Architecture B) — sharing
  encoder/decoder building blocks in `src/models/layers.py`.
- Losses (`src/losses/`): CTC wrapper, attention CE with optional label
  smoothing, joint weighted loss (Eq. 5).
- Full training loop with checkpoint save/resume (`src/training/`),
  early stopping on dev BLEU, config-driven optimizer/scheduler.
- Decoding + eval (`src/evaluation/`): joint CTC/attention beam search
  (implemented as attention-beam-then-CTC-rescore — a documented
  simplification, see that file's docstring), SacreBLEU/ROUGE wrappers,
  `compare_to_paper.py` diffing against every target table in Section 4 with
  PASS(<1%)/WARN(1–3%)/FAIL(>3%) tolerance.
- All three V1 improvements, built in isolation and composed only via config
  flags (never edited into the reproduction files — this separation rule is
  load-bearing, see Section 7 of the handoff):
  `src/models/family_router.py`, `src/datasets/augment.py`,
  `src/models/pretrained_encoder.py`.
- A config system (`configs/*.yaml`) with `extends:` inheritance and an
  explicit `python -m src.utils.config --assumptions` command that lists
  every field that's a Section-12 assumption rather than a paper-confirmed
  value — check this before trusting any given config's hyperparameters.
- CLI entry points: `scripts/train.py` (single run, now supports
  `--language-pair SRC:TGT` for isolating one SP-10 pair out of its shared
  index file), `scripts/run_one_to_one.sh` (all 11 one-to-one models),
  `scripts/run_ablation.sh` (the 4-run V1 ladder).

**What Stages 1–7 verify**: config validation, seed determinism, dataset
batch shapes (`|F| > |T|`, no NaNs), per-component shape + gradient-flow
checks (including the one-to-one "LID branch not instantiated at all" case),
CTC/attention loss correctness against hand-built references, the
tiny-dataset overfitting gate, and the resume-reproduces-loss-trajectory
gate. All of this ran on CPU with a `StubFeatureGenerator` (deterministic
random features) standing in for real SlowFastSign features — this proves
the *pipeline* is correct, not that any BLEU number is real yet.

## 3. Reference-repo inspection: what we learned by actually reading the code

`git clone` is blocked inside the Mac session's Claude Code sandbox (writes
into any `.git/` are denied — a safety restriction of that sandbox, not a
real limitation; git works fine once a human runs it outside the sandbox,
which is how this repo's own git history exists at all). So the two
reference repos were fetched via GitHub's REST API instead and saved under
`references/`. Full findings are in
`experiments/results/reference_inspection_notes.md` — read that file, but
the headline corrections to `IMPLEMENTATION_HANDOFF.md` Section 12's
assumptions are:

- **Tokenizer (item 5): CORRECTED.** Not SentencePiece — the real reference
  implementation (`references/Gloss-free-MLSLT/signjoey/vocabulary.py`) uses
  a plain word-level Counter vocabulary. `src/datasets/tokenizer.py` matches
  this. Independently confirmed twice: real SP-10 English vocab = 1,417
  words (paper's estimate ~1.1k) and real PHOENIX14T German vocab = 2,891
  words (paper's Table 6 value ~2.9k, almost exact).
- **LR scheduler (item 3): CORRECTED.** The reference repo's real default is
  `ReduceLROnPlateau` on dev BLEU, not a constant LR. Adopted as the new
  default (`configs/base_reproduction.yaml`'s `scheduler.name: plateau`).
- **`d_feat` (item 1): UPGRADED from guess to grounded value.**
  `slr_network_multi.SLRModel`'s literal constructor default is
  `hidden_size=1024` — this is now the config default, though it's the
  *gloss*-CSLR checkpoint's dimension; confirm against whatever checkpoint
  you actually train in Section 5 below.
- **Precomputed features (item 10): RESOLVED, negative result.** No public
  precomputed SlowFastSign features exist for this paper's actual setup.
  `kaistmm/SlowFastSign`'s public checkpoints are gloss-supervised (trained
  for WER, a recognition metric) on PHOENIX2014/2014-T/CSL-Daily only — no
  SP-10 support, and no spoken-text-supervised variant. **You will have to
  retrain/fine-tune SlowFastSign yourself.** This is the single biggest
  remaining piece of work — see Section 5 below.
- **Label smoothing (item 6): CONFIRMED, not newly assumed.** Off by
  default in the real repo too, exactly as originally guessed.

## 4. Real dataset status (as of this handoff)

| Dataset | Text/label data | Video | Notes |
|---|---|---|---|
| SP-10 | **Real, converted.** `data/splits/sp10/{train,dev,test}.json` — 15,770/2,698/3,834 records (8,300/1,420/2,024 same-language pairs matching Section 2.1 almost exactly, plus cross-to-English pairs for many-to-one). Built by `scripts/convert_sp10_index.py` from `references/sp10_{split}_raw.json`, which came from the fully-open [MLSLT/SP-10](https://github.com/MLSLT/SP-10) repo — **no access request needed.** | Not downloaded yet. | `MLSLT/SP-10`'s own `video_downloader.py` pulls clips straight from `media.spreadthesign.com`; needs ffmpeg. |
| PHOENIX14T | **Real, converted.** `data/splits/phoenix14t/{train,dev,test}.json` — 7,096/519/642 records, exact match to Section 2.1. Built by `scripts/convert_phoenix14t_index.py` from small annotation CSVs mirrored at `huggingface.co/datasets/quagrain/RWTH-Phoenix-2014T` (separate from the 39GB video archive). Uses only the `translation` column, never `orth` (gloss) — the reproduction is gloss-free by design. | Not downloaded yet. | Official 39GB tarball: `https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz`. No access form found. |
| CSL-Daily | **Deliberately deferred** (decision made 2026-08-26). | N/A | Access needs a signed release agreement from a full-time staff member (not a student) emailed to `ustc_vslrg@126.com`, CC'ing the PI at `zhwg@ustc.edu.cn`. The user is a student without a readily available signatory. Per the handoff's own Section 9 ("least central to the multilingual story"), work proceeds on SP-10 + PHOENIX14T first; pursue this agreement in parallel, not as a blocker. A derived pose/keypoint mirror exists on Hugging Face (`ZechengLi19/Uni-Sign`) but (a) its authorization chain wasn't verified and (b) it's skeleton data, not the RGB video this paper's SlowFastSign feature extractor needs — not a clean substitute. |

`SLTExample` (in `src/datasets/common.py`) now carries an optional
`video_url` field populated by both converters — that's where the actual
per-clip download source lives once you're ready to pull video.

## 5. The actual bottleneck: SlowFastSign feature extraction

This is the one piece of real, non-mechanical engineering work standing
between "pipeline verified" and "real numbers." No shortcut exists:

1. Clone `kaistmm/SlowFastSign`, get its own repo running per its README
   (PyTorch 1.13, `ctcdecode`, a PySlowFast pretrained backbone checkpoint —
   `SLOWFAST_64x2_R101_50_50.pkl` from `dl.fbaipublicfiles.com`).
2. Its training script expects **gloss** targets by default
   (`configs/baseline.yaml` -> `model: slr_network_multi.SLRModel`,
   CTC-over-gloss-vocab loss). This paper explicitly needs **spoken-text**
   targets instead (Section 2.2: "trained with spoken-text supervision, not
   gloss supervision"). Swapping this is real modification work on their
   `dataset/dataloader_video.py` and loss setup — not a config flag.
3. Once retrained on real data (start with PHOENIX14T — smaller, single
   language, easiest to sanity-check quality before trusting it across
   SP-10's 10 languages), run it over each dataset's video to produce
   per-frame features, one `.pt` tensor per video at
   `data/features/<dataset>/<video_id>.pt`.
4. Wire the real forward pass into `src/datasets/feature_extraction.py`'s
   `SlowFastSignExtractor` — it currently raises `NotImplementedError` by
   design; that's the exact integration point.
5. Read the real `d_feat` off that checkpoint and set
   `configs/base_reproduction.yaml`'s `model.d_feat` accordingly (see
   Section 3 above — 1024 is a grounded starting guess, not confirmed for a
   retrained checkpoint).

## 6. Ordered next steps for this machine

```bash
# 1. Environment
git clone <this-repo-url> mlslt-repro   # or: you're already in it
cd mlslt-repro
python -m venv .venv && source .venv/bin/activate
# Install CUDA torch FIRST, then requirements.txt — the other order lets
# requirements.txt's plain `torch>=2.1` install a CPU-only wheel first,
# and pip won't reinstall over an already-satisfied package even from a
# different index, so the CUDA install silently no-ops. Confirmed the hard
# way (2026-08-27): find your driver's max CUDA version with `nvidia-smi`
# first (top-right of its output), then check which +cuXXX tags actually
# exist for the current torch release before picking one — cu121 is no
# longer built for recent torch versions:
#   curl -s https://download.pytorch.org/whl/torch/ | grep -o 'torch-2\.[0-9.]*+cu[0-9]*-cp3[0-9]*-cp3[0-9]*-win_amd64[^"]*\.whl' | sort -u
pip install torch --index-url https://download.pytorch.org/whl/cu126   # pick the tag one MINOR version at or below nvidia-smi's reported CUDA version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"   # must print True before continuing
pip install -r requirements.txt
python -m pytest tests/ -q   # must be 22/22 passing before anything else

# 2. SP-10 video (fastest, no approval needed)
git clone https://github.com/MLSLT/SP-10 references/SP-10-video
cd references/SP-10-video && pip install -r requirements.txt   # + install ffmpeg via your OS package manager
python video_downloader.py --save_path ../../data/raw/sp10
python video_converter.py --save_path ../../data/raw/sp10
cd ../..

# 3. PHOENIX14T video (text side already done, see Section 4)
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz -P data/raw/
tar -xzf data/raw/phoenix-2014-T.v3.tar.gz -C data/raw/

# 4. SlowFastSign setup (see Section 5 — this is the real work, not copy-paste)
git clone https://github.com/kaistmm/SlowFastSign references/SlowFastSign
cd references/SlowFastSign && pip install -r requirements.txt
mkdir ckpt && cd ckpt
wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_64x2_R101_50_50.pkl
cd ../../..

# 5. (after step 4 produces a real checkpoint) wire it into
#    src/datasets/feature_extraction.py::SlowFastSignExtractor, then:
python scripts/extract_features.py --checkpoint <your-checkpoint> \
  --video-dir data/raw/<dataset> --output-dir data/features/<dataset>

# 6. Only once real features exist for at least one dataset:
bash scripts/run_one_to_one.sh    # Table 2/3 — 10 SP-10 pairs + PHOENIX14T
bash scripts/run_ablation.sh      # baseline -> +routing -> +augmentation -> +pretraining
python -m src.evaluation.compare_to_paper experiments/results/<run>/metrics.json --table all
```

Steps 1–3 are mechanical and can be run immediately. Step 4 is genuinely
open-ended engineering work — expect to iterate on it with Claude Code
rather than run it as a single command. Steps 5–6 only make sense after 4
produces something real.

## 7. Hard constraints — do not violate these regardless of what seems convenient

Carried directly from `IMPLEMENTATION_HANDOFF.md` Section 11 — repeating
here because they're the easiest things for a fresh session to accidentally
undo:

- **No gloss supervision anywhere**, in the baseline, Sign2(LID+Text), or
  any V1 improvement — this is the entire point of the paper. This is why
  Section 5 above is real work: SlowFastSign's default training path *is*
  gloss-supervised and must be changed, not used as-is.
- **Baseline vs. V1-improvement code must stay physically separate**
  (`src/models/baseline_transformer.py` / `sign2_lid_text.py` are the
  reproduction; `family_router.py` / `augment.py` / `pretrained_encoder.py`
  are V1, composed only via config flags). Don't merge them for convenience.
- **Augmentation is video/feature-space transforms, not a trained SignBT
  generator.** Deliberate compute decision, already made — don't revert it.
- **Pretraining is a frozen-backbone contrastive head, not full pretraining
  from scratch.** Same reasoning.
- **The V1 ablation ladder is sequential/cumulative (4 runs: baseline,
  +routing, +routing+augmentation, +routing+augmentation+pretraining), not
  the full 8-run factorial.** `scripts/run_ablation.sh` already encodes this
  — don't "complete" it into a factorial sweep.
- **Unspecified paper hyperparameters must stay config-driven and flagged as
  assumptions**, never silently hardcoded as fact. Run
  `python -m src.utils.config --assumptions` any time you're unsure which
  fields in a config are paper-confirmed vs. best-guess.
- **This machine's `git` is presumably NOT sandboxed** the way the Mac
  session's was — verify with a trivial `git status` early, since if Claude
  Code here does have the same `.git/`-write restriction, commits will need
  to happen from your own terminal instead, exactly like on the Mac.

## 8. Where everything lives (file map)

- `IMPLEMENTATION_HANDOFF.md` — the original research/plan, single source of truth for what the paper says
- `REPRODUCTION_STATUS.md` — Definition-of-Done tracking against Section 14, updated as work progresses
- `experiments/results/reference_inspection_notes.md` — full detail on Section 3
- `configs/` — one YAML per task/experiment, `extends:`-based inheritance, `--assumptions` flag
- `src/` — the actual implementation (see Section 2 above)
- `tests/` — 22 passing tests covering Stages 1–7
- `scripts/` — CLI entry points (`train.py`, `run_one_to_one.sh`, `run_ablation.sh`, `convert_sp10_index.py`, `convert_phoenix14t_index.py`, `download_data.py`, `extract_features.py`)
- `references/` — fetched copies of the paper's reference repo (`Gloss-free-MLSLT`) and `kaistmm/SlowFastSign`'s README/model code, plus the raw SP-10/PHOENIX14T annotation files the converters consumed
