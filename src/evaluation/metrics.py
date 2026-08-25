"""BLEU-4 (SacreBLEU) + ROUGE (Lin 2004) — via the libraries directly, no
hand-rolled BLEU (Section 2.5)."""
from typing import List

import sacrebleu
from rouge_score import rouge_scorer


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge(hypotheses: List[str], references: List[str]) -> float:
    scores = [_scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hypotheses, references)]
    return 100.0 * sum(scores) / max(len(scores), 1)


def decode_ids_to_text(ids_batch, tokenizer) -> List[str]:
    return [tokenizer.decode(ids) for ids in ids_batch]
