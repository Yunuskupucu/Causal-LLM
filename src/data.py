"""Külliyat yükleme ve mini-yığın (mini-batch) üretimi.

Rapor Bölüm IV.A: Tiny Shakespeare (1.1M karakter, |V| = 65) külliyatı
%90 eğitim / %10 doğrulama olarak zamansal sırayla bölünür (karıştırma yok;
böylece doğrulama kümesi eğitimde görülmemiş metindir).
"""
from __future__ import annotations

import torch

from .tokenizer import CharTokenizer
from .utils import DATA_DIR


def load_corpus(name: str = "tinyshakespeare.txt") -> str:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return f.read()


def make_splits(text: str, tokenizer: CharTokenizer, train_frac: float = 0.9):
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(ids))
    return ids[:n], ids[n:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              generator: torch.Generator):
    """Rastgele pencereler: x = x_{t:t+T}, y = x_{t+1:t+T+1}.

    Hedef dizisi y, girdinin bir sağa kaydırılmış hâlidir; i. konumdaki
    çıktı, (i+1). karakteri tahmin etmelidir -> Rapor Denk. (1)-(2),
    "sıradaki belirteci tahmin etme" (next-token prediction).
    """
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,),
                       generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
