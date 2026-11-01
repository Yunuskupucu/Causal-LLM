"""İstatistiksel n-gram dil modeli (add-alpha / Lidstone düzgünleştirme).

Rapor Bölüm II (Kronoloji) ve Denk. (20):
    p(x_t | x_{t-n+1:t-1}) = (c(ctx, x_t) + alpha) / (c(ctx) + alpha |V|)
Amaç: "istatistiksel -> nöral" geçişin gerekçesini (veri seyrekliği,
boyutluluk laneti; Bengio vd., 2003) aynı külliyat üzerinde nicel olarak
göstermek.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class NGramLM:
    def __init__(self, n: int, vocab_size: int, alpha: float = 0.1) -> None:
        self.n, self.V, self.alpha = n, vocab_size, alpha
        self.counts: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(vocab_size, dtype=np.int64))

    def fit(self, ids: np.ndarray) -> "NGramLM":
        n = self.n
        for t in range(n - 1, len(ids)):
            ctx = tuple(ids[t - n + 1:t]) if n > 1 else ()
            self.counts[ctx][ids[t]] += 1
        return self

    def log_prob(self, ctx: tuple, x: int) -> float:
        c = self.counts.get(ctx)
        if c is None:                                 # görülmemiş bağlam
            return -math.log(self.V)                  # tekdüze
        return math.log((c[x] + self.alpha) / (c.sum() + self.alpha * self.V))

    def cross_entropy(self, ids: np.ndarray, alpha: float | None = None):
        """Ortalama negatif log-olabilirlik (nat / karakter)."""
        if alpha is not None:
            self.alpha = alpha
        n, total = self.n, 0.0
        for t in range(n - 1, len(ids)):
            ctx = tuple(ids[t - n + 1:t]) if n > 1 else ()
            total += self.log_prob(ctx, int(ids[t]))
        return -total / (len(ids) - n + 1)

    def n_contexts(self) -> int:
        return len(self.counts)
