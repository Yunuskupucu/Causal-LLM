"""Nedensel (causal) öz-dikkat tabanlı mini GPT - "kütüphane soyutlaması yok".

Her fonksiyonun başındaki "Rapor Denk. (k)" etiketi, bildirideki numaralı
denkleme karşılık gelir. nn.MultiheadAttention veya
F.scaled_dot_product_attention KULLANILMAZ; Q, K, V projeksiyonları,
QK^T/sqrt(d_k) çarpımı, alt-üçgen maske ve Softmax açıkça yazılır.

Tensör boyutları (B: yığın, T: dizi uzunluğu, d: model boyutu,
h: baş sayısı, d_k = d/h):
    x        : (B, T, d)
    Q, K, V  : (B, h, T, d_k)
    S = QK^T : (B, h, T, T)
    M        : (T, T)   alt-üçgen maske, M_ij = 0 (j<=i), -inf (j>i)
    A        : (B, h, T, T) satır toplamı 1 olan dikkat ağırlıkları
    logits   : (B, T, |V|)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import sampling


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Rapor Denk. (7): M_ij = 0 eğer j <= i, aksi hâlde -inf.

    Üst üçgen (gelecek konumlar) -inf alır; Softmax'ta exp(-inf) = 0
    olduğu için i. konum yalnızca j <= i konumlarına ağırlık verebilir.
    """
    mask = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)  # köşegen ve altı 0 kalır


def log_softmax_stable(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sayısal kararlı log-softmax (Rapor Bölüm V: logsumexp hilesi).

    log p_i = z_i - log sum_j exp(z_j) = (z_i - m) - log sum_j exp(z_j - m),
    m = max_j z_j. Büyük logit değerlerinde exp taşmasını (overflow) önler.
    """
    m = z.max(dim=dim, keepdim=True).values
    zs = z - m
    return zs - torch.log(torch.exp(zs).sum(dim=dim, keepdim=True))


def sequence_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Rapor Denk. (2): L = -(1/N) sum_t log p_theta(x_{t+1} | x_{<=t}).

    logits: (B, T, V), targets: (B, T). Her konumda yalnızca doğru
    belirtecin log-olasılığı toplanır (çapraz entropi, one-hot hedef).
    """
    logp = log_softmax_stable(logits, dim=-1)             # (B, T, V)
    picked = logp.gather(-1, targets.unsqueeze(-1))        # (B, T, 1)
    return -picked.mean()


class CausalSelfAttention(nn.Module):
    """Rapor Denk. (5)-(9): çok başlı, maskeli, ölçekli nokta-çarpım dikkat."""

    def __init__(self, d_model: int, n_head: int, block_size: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        # Denk. (5): Q = X W_Q, K = X W_K, V = X W_V  (her biri d x d)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # Denk. (9): başların birleştirilmesi sonrası çıkış projeksiyonu W_O
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # Maske sabittir, öğrenilmez: tampon (buffer) olarak saklanır.
        self.register_buffer("mask", causal_mask(block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor, use_mask: bool = True,
                return_attn: bool = False):
        B, T, d = x.shape
        h, d_k = self.n_head, self.d_k
        # (B, T, d) -> (B, T, h, d_k) -> (B, h, T, d_k)
        q = self.w_q(x).view(B, T, h, d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, h, d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, h, d_k).transpose(1, 2)

        # Denk. (6): S = Q K^T / sqrt(d_k)  -> (B, h, T, T)
        scores_raw = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        # Denk. (7)-(8): S_masked = S + M  (use_mask=False -> sızıntı deneyi)
        scores = scores_raw + self.mask[:T, :T] if use_mask else scores_raw
        attn = F.softmax(scores, dim=-1)       # Denk. (8), satır toplamı = 1
        attn = self.attn_drop(attn)
        out = attn @ v                          # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, d)  # başları birleştir
        out = self.resid_drop(self.w_o(out))    # Denk. (9)
        if return_attn:
            return out, {"scores_raw": scores_raw, "scores_masked": scores,
                         "attn": attn}
        return out


class MLP(nn.Module):
    """Konum-bazlı ileri besleme: GELU(x W_1) W_2, iç boyut 4d."""

    def __init__(self, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Rapor Denk. (10): Ön-LayerNorm'lu artık (residual) blok."""

    def __init__(self, d_model: int, n_head: int, block_size: int,
                 dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x, use_mask=True, return_attn=False):
        a = self.attn(self.ln1(x), use_mask=use_mask, return_attn=return_attn)
        if return_attn:
            a, info = a
        x = x + a                       # artık bağlantı 1
        x = x + self.mlp(self.ln2(x))   # artık bağlantı 2
        return (x, info) if return_attn else x


class MiniGPT(nn.Module):
    """Karakter düzeyinde otoregresif dil modeli (decoder-only Transformer)."""

    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4,
                 n_head: int = 4, d_model: int = 128, dropout: float = 0.1,
                 tie_weights: bool = True) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        # Denk. (4): h^0 = E[x_t] + P[t]
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, block_size, dropout)
             for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        # Denk. (11): z = h^L W_E^T  (ağırlık bağlama: Press & Wolf, 2017)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                use_mask: bool = True, return_attn: bool = False):
        B, T = idx.shape
        assert T <= self.block_size, "dizi, bağlam penceresinden uzun"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # Denk. (4)
        attn_infos = []
        for block in self.blocks:
            if return_attn:
                x, info = block(x, use_mask, return_attn=True)
                attn_infos.append(info)
            else:
                x = block(x, use_mask)
        logits = self.lm_head(self.ln_f(x))                    # Denk. (11)
        loss = sequence_nll(logits, targets) if targets is not None else None
        if return_attn:
            return logits, loss, attn_infos
        return logits, loss

    def forward_from_embeddings(self, x: torch.Tensor, use_mask: bool = True):
        """Gömme (embedding) girdisinden ileri geçiş: Jakobiyen sızıntı testi.

        Rapor Denk. (19): d z_i / d x_j = 0  (j > i) olmalı.
        """
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        h = self.drop(x + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h, use_mask)
        return self.lm_head(self.ln_f(h))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Otoregresif üretim döngüsü: x_{t+1} ~ p(. | x_{<=t}).

        Her adımda yalnızca SON konumun logit vektörü kullanılır; sıcaklık,
        top-k ve top-p filtreleri sampling.py'deki sırayla uygulanır
        (Rapor Denk. (12), (17), (18)).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            z = logits[:, -1, :]                         # (B, V)
            probs = sampling.sampling_pipeline(z, temperature, top_k, top_p)
            nxt = sampling.sample_from_probs(probs, generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
