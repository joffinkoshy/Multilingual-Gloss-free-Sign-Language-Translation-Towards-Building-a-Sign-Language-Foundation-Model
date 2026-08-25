"""Word-level tokenizer/vocabulary.

CORRECTED per Section 12 item 5 of IMPLEMENTATION_HANDOFF.md: the handoff's
default assumption was SentencePiece subwords (inferred from vocab sizes
looking like learned-subword counts). Reading the actual reference repo
(references/Gloss-free-MLSLT/signjoey/vocabulary.py::build_vocab) shows this
is wrong — it's a plain word-level vocabulary built from whitespace-split
tokens with a Counter, cut by (min_freq, max_size), exactly like classic
JoeyNMT/SignJoey. No subword model is trained anywhere in that codebase.
This module reproduces that behavior instead of SentencePiece.
"""
from collections import Counter
from typing import List, Optional

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPECIALS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


class WordTokenizer:
    def __init__(self, min_freq: int = 1, max_size: int = 50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos: List[str] = list(SPECIALS)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.unk_id = self.stoi[UNK_TOKEN]
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

    @staticmethod
    def pre_tokenize(text: str) -> List[str]:
        return text.strip().split()

    def build(self, texts: List[str]) -> "WordTokenizer":
        counter = Counter()
        for t in texts:
            counter.update(self.pre_tokenize(t))
        if self.min_freq > 1:
            counter = Counter({w: c for w, c in counter.items() if c >= self.min_freq})
        # sort by frequency desc, then alphabetically for determinism (matches
        # signjoey.vocabulary.sort_and_cut)
        ranked = sorted(counter.items(), key=lambda kv: kv[0])
        ranked.sort(key=lambda kv: kv[1], reverse=True)
        for word, _ in ranked[: self.max_size]:
            if word not in self.stoi:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
        return self

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(w, self.unk_id) for w in self.pre_tokenize(text)]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], cut_at_eos: bool = True) -> str:
        words = []
        for i in ids:
            if i == self.pad_id or i == self.bos_id:
                continue
            if cut_at_eos and i == self.eos_id:
                break
            words.append(self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN)
        return " ".join(words)

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for tok in self.itos:
                f.write(tok + "\n")

    @classmethod
    def from_file(cls, path: str, min_freq: int = 1, max_size: int = 50000) -> "WordTokenizer":
        tok = cls(min_freq=min_freq, max_size=max_size)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.rstrip("\n")
                if w and w not in tok.stoi:
                    tok.stoi[w] = len(tok.itos)
                    tok.itos.append(w)
        return tok


class LIDTokenizer:
    """Token-level language-ID vocabulary (Section 2.3: I_tok is a sequence of
    language-ID tokens the same length as the text target)."""

    def __init__(self, languages: List[str]):
        self.itos = list(SPECIALS) + [f"<{lang}>" for lang in languages]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def token_for(self, language: str) -> int:
        return self.stoi[f"<{language}>"]

    def encode_repeated(self, language: str, length: int) -> List[int]:
        """I_tok: same LID token repeated for every position in the text target."""
        return [self.token_for(language)] * length
