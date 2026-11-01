"""Çıkış örnekleme mekanizmaları: Sıcaklık, Top-k, Top-p (Nucleus).

Tüm fonksiyonlar son boyutu sözlük (|V|) olan logit tensörleri üzerinde
çalışır. Uygulama sırası, Hugging Face `transformers` ile aynıdır:
    sıcaklık -> top-k -> top-p   (sonra multinomial örnekleme).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def softmax_temperature(logits: torch.Tensor, temperature: float):
    """Rapor Denk. (12): p_i(T) = exp(z_i / T) / sum_j exp(z_j / T).

    T -> 0  : argmax (greedy, tek-sıcak dağılım)
    T = 1   : modelin ham dağılımı
    T -> inf: tekdüze dağılım (uniform)
    """
    if temperature <= 1e-8:                      # greedy sınır durumu
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return F.softmax(logits / temperature, dim=-1)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Rapor Denk. (17): en yüksek k logit dışındakiler -inf yapılır.

    Softmax yeniden normalize eder: p~_i = p_i / sum_{j in V_k} p_j.
    """
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]  # k. büyük değer
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Rapor Denk. (18): Çekirdek (nucleus) örnekleme (Holtzman vd., 2020).

    Olasılıklar azalan sıraya dizilir; kümülatif kütlesi p'yi aşan EN KÜÇÜK
    küme V_p tutulur (eşiği geçen belirteç dâhil), kalanlar -inf olur.
    Kümenin boyutu bağlama göre değişir: kararlı dağılımda 1-2, belirsiz
    dağılımda onlarca belirteç.
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # "Bu belirteçten ÖNCEKİ kütle zaten p'yi aştı" -> bu belirteci at.
    remove = (cum - sorted_probs) > p
    remove[..., 0] = False                       # en az bir belirteç kalır
    remove_orig = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
    return logits.masked_fill(remove_orig, float("-inf"))


def sampling_pipeline(logits: torch.Tensor, temperature: float = 1.0,
                      top_k: int | None = None,
                      top_p: float | None = None) -> torch.Tensor:
    """Sıcaklık -> top-k -> top-p zinciri; olasılık vektörü döndürür."""
    if temperature <= 1e-8:
        return softmax_temperature(logits, temperature)
    z = logits / temperature                     # Denk. (12)
    if top_k is not None and top_k > 0:
        z = top_k_filter(z, top_k)               # Denk. (17)
    if top_p is not None and top_p < 1.0:
        z = top_p_filter(z, top_p)               # Denk. (18)
    return F.softmax(z, dim=-1)


def sample_from_probs(probs: torch.Tensor,
                      generator: torch.Generator | None = None):
    """Kategorik dağılımdan bir belirteç çeker (torch.multinomial)."""
    return torch.multinomial(probs, num_samples=1, generator=generator)


def entropy(probs: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """Rapor Denk. (15): H(p) = -sum_i p_i log p_i  (varsayılan: bit)."""
    p = probs.clamp_min(1e-12)
    return -(probs * torch.log(p)).sum(-1) / math.log(base)


def support_size(probs: torch.Tensor) -> torch.Tensor:
    """Sıfırdan büyük olasılık taşıyan belirteç sayısı (|V_k| veya |V_p|)."""
    return (probs > 0).sum(-1)


def softmax_jacobian_analytic(logits: torch.Tensor, temperature: float):
    """Rapor Denk. (13): d p_i / d z_j = (1/T) p_i (delta_ij - p_j).

    1-boyutlu logit vektörü için (V x V) Jakobiyen matrisi döndürür.
    """
    p = softmax_temperature(logits, temperature)
    return (torch.diag(p) - torch.outer(p, p)) / temperature


def entropy_temperature_derivative_analytic(logits: torch.Tensor,
                                            temperature: float):
    """Rapor Denk. (16): dH/dT = Var_{p_T}(z) / T^3  (nat cinsinden).

    Logitlerin p_T altındaki varyansı daima >= 0 olduğundan entropi,
    sıcaklıkla monoton artar.
    """
    p = softmax_temperature(logits, temperature)
    mean = (p * logits).sum()
    var = (p * (logits - mean) ** 2).sum()
    return var / temperature ** 3
