"""Karakter düzeyinde belirteçleyici (character-level tokenizer).

Rapor Bölüm III.A: Sözlük V, külliyattaki ayrık karakterler kümesidir.
Karakter düzeyi seçildiği için BPE gibi alt-sözcük algoritmalarına gerek
yoktur; her karakter bir belirteç (token) x_t ∈ {0, ..., |V|-1} olur.
"""
from __future__ import annotations


class CharTokenizer:
    """Metin <-> tamsayı dizisi dönüşümü."""

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def label(self, i: int) -> str:
        """Grafik eksenlerinde okunur etiket ('\\n' -> '\\n', ' ' -> '␣')."""
        ch = self.itos[int(i)]
        return {"\n": "\\n", " ": "␣", "\t": "\\t"}.get(ch, ch)
