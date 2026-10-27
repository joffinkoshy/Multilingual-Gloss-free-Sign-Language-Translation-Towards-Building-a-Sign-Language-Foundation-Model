# Handoff: continuing on the same Windows/RTX 3050 machine (session 2)

**Read in this order:** this file → `HANDOFF_TO_RTX3050.md` → `IMPLEMENTATION_HANDOFF.md`
→ `REPRODUCTION_STATUS.md`. This file only covers what happened *after*
`HANDOFF_TO_RTX3050.md` was written and handed to this same Windows machine —
it does not repeat that document's content, it picks up exactly where it
left off. Do not re-derive anything already decided in those files.

---

## 0. The chain of documents, and why there are several

1. `IMPLEMENTATION_HANDOFF.md` — the original research/plan (paper analysis,
   architecture spec, target numbers, every explicitly-flagged assumption).
   Single source of truth for "what the paper says."
2. `REPRODUCTION_STATUS.md` — running Definition-of-Done tracker against
   that plan's Section 14, updated as work progresses.
3. `experiments/results/reference_inspection_notes.md` — findings from
   directly reading the paper's reference repo and SlowFastSign's repo,
   which corrected/resolved several of Section 12's open assumptions.
4. `HANDOFF_TO_RTX3050.md` — written when work moved from a Mac (no GPU,
   code-only) to this Windows/RTX 3050 machine. Covers: the full codebase
   (Stages 1–7, 22 passing tests), real SP-10 + PHOENIX14T text/label data
   already converted, CSL-Daily deliberately deferred, and the ordered
   next-steps plan.
5. **This file** — what actually happened once work started running *on*
   this Windows machine (environment issues, fixes, current live progress).

## 1. What happened on this machine since `HANDOFF_TO_RTX3050.md`

### 1a. PyTorch was CPU-only, now fixed
`torch.cuda.is_available()` returned `False` after initial setup. Root
cause: `pip install -r requirements.txt` ran before the CUDA-specific torch
install, so a plain CPU wheel got installed first — pip won't reinstall an
already-satisfied package even from a different `--index-url`, so the
subsequent CUDA install silently no-opped.

**Fix applied:**
```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu126
```
Note: `cu121` (originally suggested) no longer has wheels for current torch
releases — verify available tags before picking one:
```powershell
curl -s https://download.pytorch.org/whl/torch/ | grep -o 'torch-2\.[0-9.]*+cu[0-9]*-cp3[0-9]*-cp3[0-9]*-win_amd64[^"]*\.whl' | sort -u
```
`nvidia-smi` on this machine reports driver 592.82, CUDA 13.1 max — `cu126`
was chosen as the safest, most battle-tested tag at/below that. `HANDOFF_TO_RTX3050.md`
Section 6 has been updated to install CUDA torch *before* `requirements.txt`
to prevent this from recurring.

**Hardware correction, also already logged in `HANDOFF_TO_RTX3050.md`:**
this machine's RTX 3050 has **4GB VRAM** (`nvidia-smi` shows 4096MiB), not
the 8GB `IMPLEMENTATION_HANDOFF.md` assumed. Nothing to act on yet, but
expect to need smaller batch sizes / model dims than that document's
planning assumed once real training starts.

### 1b. `python -m pytest tests/ -q` — confirm this was run and passed
(22/22 expected). If this hasn't been re-verified on this machine yet, do
it now before anything else — it's the cheapest possible check that nothing
about this environment (Windows path handling, Python 3.11 vs. the Mac
session's 3.12, etc.) broke anything.

### 1c. SP-10 video download — in progress, three bugs fixed along the way

Cloned `https://github.com/MLSLT/SP-10` into `references/SP-10-video/` and
ran its `video_downloader.py` / `video_converter.py` to get the actual
video files behind the SP-10 text data already converted in
`data/splits/sp10/`. Hit three Windows-specific bugs in that upstream repo
(all specific to Windows — the repo was only ever tested on Linux/Mac):

1. **`requirements.txt` pins `numpy==1.21.5`**, incompatible with Python
   3.11. Fix: ignore the pinned file, just `pip install opencv-python`
   directly — numpy comes along as its dependency at a compatible version.
   Neither script imports numpy directly.
2. **`video_downloader.py` crashes reading the dataset JSON** with
   `UnicodeDecodeError` — its `open(..., "r")` calls have no encoding
   specified, so Windows defaults to `cp1252`, which can't decode the
   Chinese/Cyrillic/etc. text in the JSON. Fix: run with `python -X utf8`
   to force UTF-8 mode for the whole process, no code edit needed.
3. **`video_converter.py` hardcodes `/tmp/sign.mp4`** (doesn't exist on
   Windows) and wraps ffmpeg path args in single quotes (Windows' cmd.exe
   doesn't treat those as quote characters, so they get passed to ffmpeg
   literally). Fix: a patched, Windows-safe drop-in replacement is tracked
   at `scripts/patches/sp10_video_converter_windows.py` — copy it over the
   broken one:
   ```powershell
   cd references\SP-10-video
   Copy-Item ..\..\scripts\patches\sp10_video_converter_windows.py video_converter.py
   ```

**Working commands (run from `references/SP-10-video/`):**
```powershell
python -X utf8 video_downloader.py --save_path ../../data/raw/sp10
python -X utf8 video_converter.py --save_path ../../data/raw/sp10
```

**Live status as of the last observed log** (train set download running,
~15s per 10-language group, processing group IDs like 410, 420, 620, ...,
2798 — group IDs are arbitrary, not a progress counter): the downloader was
partway through the 830-group train set. Estimated total time for
train+dev+test video download: **~4.5–5 hours** at the observed pace.

**First thing to do in this session: check whether the download (and the
converter run after it) actually finished.**
```powershell
Get-ChildItem data\raw\sp10\train | Measure-Object   # expect close to 830
Get-ChildItem data\raw\sp10\dev | Measure-Object     # expect close to 142
Get-ChildItem data\raw\sp10\test | Measure-Object    # expect close to 214
```
If incomplete, the downloader is safely resumable — it skips files that
already exist (`if file_path.exists(): ... continue`) — just re-run the same
command. If `video_converter.py` hasn't been run yet (it should run *after*
the downloader completes), run it now.

### 1d. Git sync — same sandbox caveat may or may not apply here

On the Mac session, Claude Code's sandbox blocked all writes into `.git/`
(a safety restriction), so every commit had to be run by the human directly
in a normal terminal, not through Claude Code. **Check whether this
Windows Claude Code session has the same restriction** with a trivial:
```powershell
git status
```
If that works fine, commits can probably be run directly through this
session; if it errors on `.git/index.lock` or similar, the same
manual-commit-only workflow applies here too. Whichever it is, the current
git state as of the Mac session's last push:
```
6d65ac5 Add real PHOENIX14T text data and per-language-pair one_to_one runner
a54b151 Initial Sign2(LID+Text) reproduction scaffold
```
plus `HANDOFF_TO_RTX3050.md` and `scripts/patches/` were committed and
pushed after that (commit message: "Add RTX3050 handoff doc and Windows fix
for SP-10's video_converter.py"). Run `git log --oneline -5` to confirm this
session sees the same history before doing anything else git-related.

## 2. What's left — the implementation plan from here

In order, matching `IMPLEMENTATION_HANDOFF.md` Section 10's task list and
`HANDOFF_TO_RTX3050.md` Section 6:

1. **Finish SP-10 video download + conversion** (in progress, see 1c above).
2. **Download PHOENIX14T video** — text/labels already done
   (`data/splits/phoenix14t/`), only the actual video is missing:
   ```powershell
   wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz -P data/raw/
   tar -xzf data/raw/phoenix-2014-T.v3.tar.gz -C data/raw/
   ```
   (39GB — budget real time/disk for this too. If `wget` isn't available on
   Windows, use `Invoke-WebRequest` or a download manager instead.)
3. **CSL-Daily remains deliberately deferred** — no readily available
   full-time-staff signatory for its access agreement. Don't pursue this
   unless that changes; it's explicitly the least central dataset per the
   handoff's own Section 9 prioritization.
4. **The real bottleneck: get SlowFastSign running and retrain it with
   spoken-text targets instead of gloss targets.** No public checkpoint
   matches what this paper needs (confirmed by direct inspection — see
   `experiments/results/reference_inspection_notes.md` item 10). This is
   genuine engineering work, not a config change:
   ```powershell
   git clone https://github.com/kaistmm/SlowFastSign references/SlowFastSign
   cd references/SlowFastSign
   pip install -r requirements.txt   # PyTorch 1.13, ctcdecode — may itself need Windows-specific fixes, same pattern as SP-10's scripts
   mkdir ckpt; cd ckpt
   Invoke-WebRequest https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/pretrain/SLOWFAST_64x2_R101_50_50.pkl -OutFile SLOWFAST_64x2_R101_50_50.pkl
   cd ../..
   ```
   Then modify its `dataset/dataloader_video.py` and loss setup to train on
   the `translation` text (already in our converted split files) instead of
   gloss labels. Expect this to take real iteration — treat it as its own
   debugging session, not a one-shot command. Start with PHOENIX14T (single
   language, easiest to validate) before trusting a retrained checkpoint
   across SP-10's 10 languages.
5. **Extract real features** once a checkpoint exists — wire the actual
   forward pass into `src/datasets/feature_extraction.py::SlowFastSignExtractor`
   (currently a documented `NotImplementedError` stub, exactly at this
   integration point), then:
   ```powershell
   python scripts/extract_features.py --checkpoint <your-checkpoint> --video-dir data/raw/<dataset> --output-dir data/features/<dataset>
   ```
   Read the real `d_feat` off that checkpoint and set it in
   `configs/base_reproduction.yaml` (currently `1024`, grounded in
   SlowFastSign's own `hidden_size` default but not confirmed for a
   retrained checkpoint — see reference_inspection_notes.md item 1).
6. **Only once real features exist for at least one dataset — run actual
   training:**
   ```powershell
   bash scripts/run_one_to_one.sh    # 10 SP-10 pairs + PHOENIX14T (Table 2/3)
   bash scripts/run_ablation.sh      # baseline -> +routing -> +augmentation -> +pretraining (V1 ladder, Section 9)
   ```
   (On Windows, run these via Git Bash / WSL, or translate the loop into
   PowerShell — they're just sequential `python scripts/train.py` calls.)
7. **Compare against the paper:**
   ```powershell
   python -m src.evaluation.compare_to_paper experiments/results/<run>/metrics.json --table all
   ```
8. **Write the failure-map summary** (task 29) — per-language, per-length,
   per-task error analysis of the V1-combined model. Only possible once 6-7
   produce real results.

## 3. Hard constraints — repeated because they're easy to accidentally break

Same list as `HANDOFF_TO_RTX3050.md` Section 7 — repeating because a fresh
session is the most likely place for these to get silently violated:

- **No gloss supervision anywhere.** This is why step 4 above is real work:
  SlowFastSign's default training path *is* gloss-supervised and must be
  changed, not used as-is or worked around with a flag.
- **Baseline vs. V1-improvement code stays physically separate**
  (`src/models/baseline_transformer.py` / `sign2_lid_text.py` are the
  reproduction; `family_router.py` / `augment.py` / `pretrained_encoder.py`
  are V1, composed only via config flags — never merge them for convenience).
- **Augmentation is video/feature-space transforms, not a trained SignBT
  generator.** Already decided, don't revert.
- **Pretraining is a frozen-backbone contrastive head, not full pretraining
  from scratch.** Same reasoning.
- **The V1 ablation ladder is sequential/cumulative (4 runs), not the full
  8-run factorial.** `scripts/run_ablation.sh` already encodes this.
- **Unspecified hyperparameters stay config-driven and flagged as
  assumptions** — run `python -m src.utils.config --assumptions` whenever
  unsure which fields in a config are paper-confirmed vs. best-guess.

## 4. Quick status snapshot (fill in / verify at the start of this session)

| Item | Status as of last Mac-session check-in |
|---|---|
| Codebase (Stages 1–7) | ✅ Done, 22/22 tests passing (re-verify on this machine) |
| `torch.cuda.is_available()` | ✅ Fixed via cu126 wheel — re-confirm still `True` |
| SP-10 text/labels | ✅ Real, converted |
| SP-10 video | 🔄 In progress — check completion first (Section 1c) |
| PHOENIX14T text/labels | ✅ Real, converted |
| PHOENIX14T video | ❌ Not started |
| CSL-Daily | ❌ Deliberately deferred |
| SlowFastSign retraining | ❌ Not started — the actual bottleneck |
| Real feature extraction | ❌ Blocked on above |
| Stage 8 (real training) | ❌ Blocked on above |
| Stage 9 (paper comparison) | ❌ Blocked on above |
