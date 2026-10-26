# Reproduction status

Tracks progress against IMPLEMENTATION_HANDOFF.md Section 14's Definition of
Done. Written from this session's actual sandboxed-environment constraints:
no GPU, no `pip`-blocked-then-unblocked network subtleties aside, no access
to raw SP-10/PHOENIX14T/CSL-Daily video, no SlowFastSign checkpoint. Full
real-data training (Stage 8) and the resulting paper comparison (Stage 9)
must run on the user's own RTX 3050 machine — this repo is built and tested
to make that a matter of running scripts, not writing code.

## Done and verified in this session

- **Stages 1-7 (Section 5), all verification checks passing.** 22/22 tests
  green (`pytest tests/ -q`): config load/validate, seed determinism,
  tokenizer round-trip, dataset batch shapes (`|F| > |T|`, no NaNs),
  component shape checks, gradient-flow checks (including the one_to_one
  "LID branch not instantiated at all" case), CTC-loss-matches-reference and
  attention-CE-matches-`F.nll_loss` checks, the tiny-set overfitting gate,
  the resume-reproduces-loss-trajectory gate, decode well-formedness, and
  BLEU=100 on identical hyp/ref.
- Both architecture variants (Section 2.3): `src/models/baseline_transformer.py`
  (A) and `src/models/sign2_lid_text.py` (B, hierarchical LID+Txt CTC +
  attention decoder), sharing encoder/decoder building blocks
  (`src/models/layers.py`).
- Full CLI training loop (`scripts/train.py`) smoke-tested end-to-end against
  synthetic data — trains, evaluates dev BLEU each epoch, checkpoints,
  resumes.
- All three V1 improvements implemented **in isolation** from the
  reproduction files (Section 7's separation rule) and unit-tested:
  `family_router.py`, `augment.py` (feature-space half; video-level half is a
  documented stub, see below), `pretrained_encoder.py`.
- Config system with inheritance (`extends:`), schema validation, and an
  explicit `--assumptions` flag listing every Section-12 assumption baked
  into a given config — nothing is silently hardcoded as fact.
- Reference repos (Claire874/Gloss-free-MLSLT, kaistmm/SlowFastSign) fetched
  and read directly (via GitHub's API, since `git clone` is blocked by this
  session's sandbox — see below); findings in
  `experiments/results/reference_inspection_notes.md` **corrected** one
  Section-12 assumption (tokenizer: word-level, not SentencePiece) and
  **resolved** another with a negative result (no public precomputed
  features exist for this paper's actual setup).

## Update: real SP-10 text data is now wired in

SP-10 turned out to require **no access request at all** —
[MLSLT/SP-10](https://github.com/MLSLT/SP-10) is a public repo with a
`video_downloader.py` script pulling clips straight from
`media.spreadthesign.com`, plus `dataset/{train,dev,test}.json` annotations.
Those annotation files were fetched and converted
(`scripts/convert_sp10_index.py`) into real `data/splits/sp10/{train,dev,test}.json`:
830/142/214 groups x up to 10 languages = **8,300/1,420/2,021 records** (earlier 2,024 included 3 csl test groups with no video; skipped now),
matching Section 2.1's stated 8,300/1,420 almost exactly (test is 2,024 vs.
the paper's 2,021 — a handful of groups are missing one language's video).
Loading this through the real tokenizer gives an English target vocab of
1,417 words, close to the handoff's ~1.1k estimate — a good independent
sanity check that the many-to-one construction (Section 2.1) is right.

**Only the actual video files are still missing** — `video_downloader.py`
needs to actually run (ffmpeg, real bandwidth, thousands of small files) on
your machine, then real SlowFastSign feature extraction still has the
blocker described below.

**PHOENIX14T's text data is now real too.** Its annotation CSVs are
separately mirrored on Hugging Face
(`quagrain/RWTH-Phoenix-2014T`, small pipe-delimited files, independent of
the ~39GB video archive) and converted via `scripts/convert_phoenix14t_index.py`
into `data/splits/phoenix14t/{train,dev,test}.json` — **7,096/519/642
records, an exact match** to Section 2.1's stated splits. Loading it through
the real word-level tokenizer gives a German vocab of **2,891 words**,
matching the paper's Table 6 figure of ~2.9k almost exactly — strong
independent confirmation that the tokenizer correction (item 5 above) is
right. Only `translation` (spoken text) was used, never `orth` (gloss),
respecting the gloss-free constraint. Actual video frames still need the
official RWTH release or an equivalent mirror.

**CSL-Daily is deliberately deferred** (decision made 2026-08-26): its
access agreement requires a full-time-staff signature the user doesn't have
readily available as a student. Per the handoff's own dataset-priority
ordering (Section 9: "least central to the multilingual story"), work
proceeds on SP-10 + PHOENIX14T first; CSL-Daily's agreement can be pursued
in parallel without blocking anything.

## Explicitly still open (Section 14's own bar: documented, not silently guessed past)

1. **`d_feat` (Section 12 item 1)** — no confirmed value recoverable from the
   reference repos available; kept at a documented placeholder (1024).
2. **Precomputed SlowFastSign features (Section 12 item 10)** — confirmed
   none exist publicly for this paper's spoken-text-supervised setup. Real
   feature extraction requires retraining SlowFastSign yourself (see
   `scripts/extract_features.py`'s docstring for the concrete steps) — this
   is real work on your RTX 3050, not a config change.
3. **Zhang et al. 2025's 6.61 BLEU figure (Section 12 item 8)** — not
   independently re-verified in this pass; still don't cite it as a target.
4. **Batch size, weight decay beyond Adam's default, exact epoch counts** —
   no shipped config exists in the reference repo to recover these from;
   unchanged from the handoff's original assumptions.
5. **Stage 8 (full real-data training) and Stage 9 (paper comparison)** —
   blocked on items 1-2 plus real dataset access
   (`scripts/download_data.py`'s docstring has the three datasets' actual
   access processes) and GPU time. Nothing in this repo needs to change to
   run them once those exist — `scripts/run_ablation.sh` and
   `python -m src.evaluation.compare_to_paper` are ready.
6. **Task 29 (failure-map summary)** — depends on Stage 8/9 real results;
   not producible yet.

## A note on this session's tooling

`git clone`/`git init` are blocked inside this sandbox (writes into any
`.git/` directory are denied — a safety restriction, not a bug worth routing
around). Reference-repo inspection was done via GitHub's REST API and raw
file URLs instead, saved under `references/`. This has no effect on the
reproduction itself; it only means `references/*/` are plain file trees, not
git checkouts, and won't track upstream changes.

`pip` initially failed with `SSLCertVerificationError` (macOS Keychain access
denied inside the sandbox, unrelated to network reachability — plain
`curl`/`urllib` worked throughout). Fixed with
`--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org`.
A CPU-only `.venv` (Python 3.12) was used for all verification above — real
training needs your actual CUDA-enabled PyTorch install on the RTX 3050.

## How to pick this up on the RTX 3050 machine

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121  # match your CUDA version

python -m pytest tests/ -q   # re-verify Stages 1-7 on your machine first

# then, once data/splits/*/{train,dev,test}.json and data/features/ are real:
python scripts/train.py --config configs/sp10_many_to_one.yaml \
  --model-dir experiments/baseline/sp10_many_to_one --device cuda
```
