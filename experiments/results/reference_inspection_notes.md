# Reference-repo inspection notes

Resolves IMPLEMENTATION_HANDOFF.md Section 12's open items by directly
reading `references/Gloss-free-MLSLT/signjoey/*.py` (the paper's official
repo, fetched via GitHub's API/raw content since `git clone` is blocked by
this session's sandbox — writes into any `.git/` directory are denied) and
`references/SlowFastSign-README.md` (kaistmm/SlowFastSign, linked from
Gloss-free-MLSLT's own readme).

## Item 5 — Tokenization scheme: CORRECTED, not confirmed as assumed

`deep-research-report.md` guessed SentencePiece subwords. Reading
`signjoey/vocabulary.py::build_vocab` shows the real implementation is a
**plain word-level vocabulary**: whitespace-split tokens, a `Counter`,
cut by `(min_freq, max_size)`, sorted by frequency then alphabetically —
classic JoeyNMT/SignJoey style, no learned subword model anywhere in the
codebase. `src/datasets/tokenizer.py::WordTokenizer` reproduces this exactly
instead of SentencePiece. Config default: `data.tokenizer.type: word`.

## Item 6 — Label smoothing: CONFIRMED default off, but present in the repo

`signjoey/loss_labelsmoothing.py` does exist (as `deep-research-report.md`
speculated), and it also contains an experimental `SmoothCTCLoss` (KL-uniform
smoothed CTC, weight=0.01) that isn't in the handoff's original description.
However, `signjoey/training.py:225` shows it commented out
(`#SmoothCTCLoss()`) — the active path uses plain `torch.nn.CTCLoss`. The
attention-side `XentLoss` defaults to `smoothing=0.0` (off), config-toggleable
via `train_config.get("label_smoothing", 0.0)`. Adopted default:
`loss.label_smoothing: 0.0`, matching the handoff's original guess — now
confirmed rather than assumed.

## Item 3 — LR schedule: CORRECTED after inspection

Handoff's original default was "constant LR, no schedule." The actual
reference implementation's `build_scheduler` (`signjoey/builders.py`)
defaults its "plateau" option to `torch.optim.lr_scheduler.ReduceLROnPlateau`
tracking the eval metric (BLEU) — `scheduler_mode="max"`, `patience`,
`decrease_factor` all config-driven. This composes naturally with the
handoff's own early-stopping-on-dev-BLEU strategy (Section 2.3). Per
Section 3's explicit instruction to "reproduce its actual training config
... once inspected," `configs/base_reproduction.yaml` now defaults to
`scheduler.name: plateau`, patience=10, decrease_factor=0.1 (the repo's own
literal defaults). `scheduler.name: constant` remains available as the
paper-literal fallback reading.

## Item 4 — Dropout: still unconfirmed, default unchanged

`signjoey`'s own code defaults dropout to `0.0` if unset in its (non-existent
in this repo) YAML config — no shipped config file exists in
Claire874/Gloss-free-MLSLT to recover the actual trained value from. Kept the
handoff's Transformer-standard default of `0.1`, still flagged as an
assumption (not newly confirmed).

## Item 1 — `d_feat`: UPGRADED from placeholder to a grounded default

Fetched `slr_network_multi.py` directly (not just the YAML configs, which
don't expose this). `SLRModel.__init__` has `hidden_size=1024` as its literal
default, and every classifier/head in the class (`self.classifier`,
`self.conv1d.fc`) is built off that `hidden_size` — this is the actual
per-frame feature dimensionality this architecture produces before its
gloss-classification head. `d_feat: 1024` in `configs/base_reproduction.yaml`
is now a real default recovered from the reference model code, not just a
lineage-convention guess. Caveat still open: this is the *gloss-CSLR*
checkpoint's dimension — if the MLSLT authors changed `hidden_size` when
retraining for spoken-text supervision (item 10), the true value could
differ. Confirm against whatever checkpoint you actually end up training.

## Item 10 — Precomputed features: RESOLVED, none exist

Confirmed by reading `Gloss-free-MLSLT/readme.md` directly: it points at
`https://github.com/kaistmm/SlowFastSign` for feature extraction and states
explicitly the paper does **not** use gloss supervision for that extractor,
only spoken text. But `kaistmm/SlowFastSign`'s own README shows its public
checkpoints are trained with **gloss supervision** for CSLR (reports WER, a
recognition metric) on PHOENIX2014/2014-T/CSL-Daily only — no SP-10 support,
no spoken-text-supervised variant, no precomputed feature release at all.

**Conclusion: there is no publicly available precomputed feature set usable
for this reproduction as-is.** Real numbers require either (a) retraining
SlowFastSign with spoken-text targets per-dataset, or (b) contacting the
paper's authors for their actual checkpoint. This is the single largest
remaining blocker to Stage 8 (Section 11 already flagged this as the highest-
leverage open item) — see `scripts/extract_features.py` for the concrete next
steps to run on the target RTX 3050 machine.

## Item 8 — Zhang et al. 2025's 6.61 BLEU figure: NOT independently verified

Not re-checked in this pass (would require fetching and reading that paper
directly, out of scope for the code-focused work done here). Still flagged
per the handoff as unconfirmed — do not cite it as a target number.

## Bonus finding not in the original Section 12 list

`signjoey/vocabulary.py`'s `GlossVocabulary` hardcodes the silence/blank
token at index 0 (`assert self.stoi[SIL_TOKEN] == 0`) so CTC decoding can
treat it as the blank label. `src/losses/ctc_losses.py::CTCLossWrapper`
adopts the same convention (`blank=0` default) for both the LID-CTC and
Txt-CTC heads — every CTC vocab built for this reproduction must reserve
index 0 for blank to match.

## What was NOT resolved by this inspection

- Batch size, weight decay beyond Adam's own default (0.0, confirmed
  unchanged), gradient clipping norm, exact epoch counts — no shipped config
  file exists in Claire874/Gloss-free-MLSLT to recover these from; they
  remain the handoff's original assumptions (Section 12 items 2/4 and
  Section 2.3's other unspecified fields).
- The exact `d_feat` value and any precomputed SlowFastSign checkpoint or
  cache — genuinely blocked on external resources not available in this
  session (no GPU, no dataset/checkpoint hosting reachable here).
