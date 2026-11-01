"""Deney 05 - Zincir kuralı ve otoregresif log-olabilirlik açılımı.

Kılavuz Konu 18 (Savunma Çıktısı): "otoregresif kayıp fonksiyonunun zincir
kuralı cinsinden log-olabilirlik açılımı". Rapor Denk. (1)-(3).

  A) log p(x_{1:T}) = sum_t log p(x_t | x_{<t}): tek bir paralel (teacher
     forcing) ileri geçişle hesaplanan toplam, T ayrı "yalnızca önek" ileri
     geçişiyle elde edilen toplama eşittir -> eşitliği sağlayan şey nedensel
     maskedir (maske kaldırılınca eşitlik bozulur).
  B) Konum bazlı kayıp: bağlam uzadıkça koşullu entropi düşer; t = 0'daki
     kayıp (yalnızca bir karakter bağlam) bigram taban çizgisine yakındır.
  C) Perplexity = exp(L), bit/karakter = L / ln 2 ilişkileri.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C                                              # noqa: E402
from experiments.common import load_everything                  # noqa: E402
from src.data import get_batch                                  # noqa: E402
from src.utils import (COLORS, banner, load_json, save_fig,     # noqa: E402
                       save_json, set_seed, setup_matplotlib)

LN2 = math.log(2.0)


@torch.no_grad()
def parallel_logprobs(model, x, y, use_mask=True):
    """Tek ileri geçiş: her konumun log p(y_t | x_{<=t}) değeri (B, T)."""
    logits, _ = model(x, use_mask=use_mask)
    return torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1))[..., 0]


@torch.no_grad()
def sequential_logprobs(model, x, y, use_mask=True):
    """T ayrı ileri geçiş: yalnızca x_{<=t} verilir, son konum okunur."""
    B, T = x.shape
    out = torch.zeros(B, T)
    for t in range(T):
        logits, _ = model(x[:, :t + 1], use_mask=use_mask)
        out[:, t] = torch.log_softmax(logits[:, -1], -1).gather(
            -1, y[:, t:t + 1])[:, 0]
    return out


def part_a(model, val_ids, results):
    banner("Deney 05-A: paralel (teacher forcing) == sıralı (önek) açılım")
    g = torch.Generator().manual_seed(C.SEED + 51)
    x, y = get_batch(val_ids, 64, 8, g)
    par = parallel_logprobs(model, x, y)
    seq = sequential_logprobs(model, x, y)
    par_nm = parallel_logprobs(model, x, y, use_mask=False)
    seq_nm = sequential_logprobs(model, x, y, use_mask=False)
    d_mask = (par.sum(1) - seq.sum(1)).abs().max().item()
    d_nomask = (par_nm.sum(1) - seq_nm.sum(1)).abs().max().item()
    print(f"8 dizi x 64 karakter: maskeli |paralel - sıralı| maks = "
          f"{d_mask:.2e} | maskesiz = {d_nomask:.2e}")
    print("dizi 0: log p(x_1:T) paralel = %.4f, sıralı = %.4f"
          % (par[0].sum(), seq[0].sum()))
    results["part_a"] = {
        "max_abs_diff_masked": d_mask, "max_abs_diff_unmasked": d_nomask,
        "seq0_parallel_logprob": par[0].sum().item(),
        "seq0_sequential_logprob": seq[0].sum().item(),
        "seq0_cum_parallel": torch.cumsum(par[0], 0).tolist(),
        "seq0_cum_sequential": torch.cumsum(seq[0], 0).tolist()}
    return par[0], seq[0]


@torch.no_grad()
def part_b(model, val_ids, results, n_windows=1000):
    banner("Deney 05-B: konum bazlı kayıp (bağlam uzunluğu etkisi)")
    T = model.block_size
    g = torch.Generator().manual_seed(C.SEED + 52)
    acc = torch.zeros(T)
    for _ in range(n_windows // 25):
        x, y = get_batch(val_ids, T, 25, g)
        acc += -parallel_logprobs(model, x, y).mean(0)
    per_pos = (acc / (n_windows // 25)).numpy()
    ng = load_json("ngram_baselines")
    bigram = [m for m in ng["models"] if m["n"] == 2][0]["val_nat"]
    trigram = [m for m in ng["models"] if m["n"] == 3][0]["val_nat"]
    print(f"t=0: {per_pos[0]:.3f} nat (bigram taban çizgisi {bigram:.3f}), "
          f"t=1: {per_pos[1]:.3f} (trigram {trigram:.3f}), t=8: "
          f"{per_pos[8]:.3f}, t=32: {per_pos[32]:.3f}, t=127: "
          f"{per_pos[-1]:.3f} nat")
    mean_loss = float(per_pos.mean())
    results["part_b"] = {"per_position_nat": per_pos.tolist(),
                         "bigram_nat": bigram, "trigram_nat": trigram,
                         "mean_nat": mean_loss, "mean_bpc": mean_loss / LN2,
                         "ppl": math.exp(mean_loss)}
    print(f"ortalama L = {mean_loss:.4f} nat = {mean_loss/LN2:.4f} bit/kr. "
          f"-> PPL = exp(L) = {math.exp(mean_loss):.3f}")
    return per_pos, bigram, trigram


def plot(par0, seq0, per_pos, bigram, trigram):
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.6))
    cp = torch.cumsum(par0, 0).numpy()
    cs = torch.cumsum(seq0, 0).numpy()
    t = np.arange(1, len(cp) + 1)
    ax1.plot(t, cp, color=COLORS["blue"],
             label="paralel: tek ileri geçiş, maskeli")
    ax1.plot(t[::4], cs[::4], "o", ms=3.5, mfc="white", color=COLORS["orange"],
             label="sıralı: $t$ ayrı önek geçişi")
    ax1.set_xlabel("$t$")
    ax1.set_ylabel("$\\sum_{s \\leq t} \\log p(x_s \\mid x_{<s})$")
    ax1.set_title("(a) Zincir kuralı açılımı (Denk. 1)", fontsize=8.5)
    ax1.legend(fontsize=6.8)
    ax2.plot(np.arange(1, len(per_pos) + 1), per_pos / LN2,
             color=COLORS["blue"], label="MiniGPT, konum $t$")
    ax2.set_xscale("log", base=2)
    ax2.axhline(bigram / LN2, color=COLORS["aqua"], lw=0.9,
                label=f"bigram ({bigram/LN2:.2f})")
    ax2.axhline(trigram / LN2, color=COLORS["violet"], lw=0.9,
                label=f"trigram ({trigram/LN2:.2f})")
    ax2.set_xlabel("Bağlamdaki karakter sayısı $t$ (log$_2$)")
    ax2.set_ylabel("Ortalama kayıp (bit / karakter)")
    ax2.set_title("(b) Bağlam uzadıkça koşullu entropi düşer", fontsize=8.5)
    ax2.legend(fontsize=6.8)
    fig.tight_layout()
    save_fig(fig, "fig05_chain_rule")


def main():
    set_seed(C.SEED)
    _, tok, _, val_ids, model = load_everything()
    results = {}
    par0, seq0 = part_a(model, val_ids, results)
    per_pos, bigram, trigram = part_b(model, val_ids, results)
    plot(par0, seq0, per_pos, bigram, trigram)
    save_json(results, "exp05_chain_rule")


if __name__ == "__main__":
    main()
