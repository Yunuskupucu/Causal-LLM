"""Deney 02 - Eğitim: Next-token prediction ile çapraz entropi minimizasyonu.

Rapor Bölüm IV.B-C, Şekil: kayıp eğrisi ve n-gram taban çizgileri.
Çıktılar:
    results/model.pt            : eğitilmiş model (diğer deneyler kullanır)
    results/train_log.json      : iterasyon bazlı kayıp/öğrenme oranı
    results/ngram_baselines.json: istatistiksel LM taban çizgileri
    figures/fig_loss_curve.*    : eğitim/doğrulama kaybı
    figures/fig_ngram_vs_neural.*: n-gram vs nöral (bit/karakter)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C                                              # noqa: E402
from src.data import get_batch, load_corpus, make_splits        # noqa: E402
from src.model import MiniGPT                                   # noqa: E402
from src.ngram import NGramLM                                   # noqa: E402
from src.tokenizer import CharTokenizer                         # noqa: E402
from src.utils import (COLORS, RES_DIR, banner, get_device,     # noqa: E402
                       save_fig, save_json, set_seed,
                       setup_matplotlib)

LN2 = math.log(2.0)


def lr_at(it: int) -> float:
    """Doğrusal ısınma + kosinüs sönümleme (Loshchilov & Hutter, 2017)."""
    if it < C.WARMUP_ITERS:
        return C.LEARNING_RATE * (it + 1) / C.WARMUP_ITERS
    frac = (it - C.WARMUP_ITERS) / max(1, C.MAX_ITERS - C.WARMUP_ITERS)
    return 0.1 * C.LEARNING_RATE + 0.9 * C.LEARNING_RATE * 0.5 * (
        1 + math.cos(math.pi * frac))


@torch.no_grad()
def estimate_loss(model, data, device, seed):
    """Sabit tohumlu, sabit mini-yığınlarla kayıp kestirimi (gürültüsüz)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(C.EVAL_BATCHES):
        x, y = get_batch(data, C.BLOCK_SIZE, C.BATCH_SIZE, g)
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train():
    banner("Deney 02a - Nöral modelin eğitimi (next-token prediction)")
    set_seed(C.SEED)
    device = get_device(C.DEVICE)
    text = load_corpus()
    tok = CharTokenizer(text)
    train_ids, val_ids = make_splits(text, tok)
    print(f"külliyat: {len(text):,} karakter, |V| = {tok.vocab_size}, "
          f"eğitim {len(train_ids):,} / doğrulama {len(val_ids):,}")

    model = MiniGPT(tok.vocab_size, C.BLOCK_SIZE, C.N_LAYER, C.N_HEAD,
                    C.D_MODEL, C.DROPOUT).to(device)
    print(f"model: {model.n_params():,} parametre, cihaz = {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE,
                            betas=(0.9, 0.95), weight_decay=C.WEIGHT_DECAY)
    g_train = torch.Generator().manual_seed(C.SEED)
    log = {"iter": [], "train_loss": [], "val_loss": [], "lr": []}
    t0 = time.time()
    for it in range(C.MAX_ITERS + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(it)
        if it % C.EVAL_INTERVAL == 0 or it == C.MAX_ITERS:
            tr = estimate_loss(model, train_ids, device, C.SEED + 1)
            va = estimate_loss(model, val_ids, device, C.SEED + 2)
            log["iter"].append(it)
            log["train_loss"].append(tr)
            log["val_loss"].append(va)
            log["lr"].append(lr_at(it))
            print(f"iter {it:5d} | eğitim {tr:.4f} nat ({tr/LN2:.3f} bpc) | "
                  f"doğrulama {va:.4f} nat ({va/LN2:.3f} bpc) | "
                  f"lr {lr_at(it):.2e} | {time.time()-t0:6.1f}s")
        if it == C.MAX_ITERS:
            break
        x, y = get_batch(train_ids, C.BLOCK_SIZE, C.BATCH_SIZE, g_train)
        _, loss = model(x.to(device), y.to(device))   # Denk. (2)
        opt.zero_grad(set_to_none=True)
        loss.backward()                                # dL/d(theta)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
        opt.step()

    RES_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "chars": "".join(tok.itos[i] for i in range(tok.vocab_size)),
                "config": {k: getattr(C, k) for k in
                           ("BLOCK_SIZE", "N_LAYER", "N_HEAD", "D_MODEL",
                            "DROPOUT", "SEED")}},
               RES_DIR / "model.pt")
    # Süre, bit-eşitlik kontrolünü bozmamak için ayrı dosyaya yazılır.
    save_json({"train_time_sec": time.time() - t0}, "train_timing")
    log["n_params"] = model.n_params()
    log["final_val_nat"] = log["val_loss"][-1]
    log["final_val_bpc"] = log["val_loss"][-1] / LN2
    log["final_val_ppl"] = math.exp(log["val_loss"][-1])
    save_json(log, "train_log")
    return log, tok, train_ids, val_ids


def ngram_baselines(tok, train_ids, val_ids):
    banner("Deney 02b - İstatistiksel n-gram taban çizgileri (Denk. 20)")
    tr = train_ids.numpy()
    va = val_ids.numpy()
    dev_cut = int(0.95 * len(tr))
    alphas = [1e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
    out = {"uniform_nat": math.log(tok.vocab_size),
           "uniform_bpc": math.log2(tok.vocab_size), "models": []}
    for n in (1, 2, 3, 4, 5, 6, 7):
        t0 = time.time()
        # alpha, eğitim kümesinin son %5'lik "geliştirme" dilimiyle seçilir.
        tuner = NGramLM(n, tok.vocab_size).fit(tr[:dev_cut])
        dev_scores = {a: tuner.cross_entropy(tr[dev_cut:], alpha=a)
                      for a in alphas}
        best_a = min(dev_scores, key=dev_scores.get)
        lm = NGramLM(n, tok.vocab_size, best_a).fit(tr)
        ce = lm.cross_entropy(va)
        rec = {"n": n, "alpha": best_a, "val_nat": ce, "val_bpc": ce / LN2,
               "val_ppl": math.exp(ce), "n_contexts": lm.n_contexts(),
               "possible_contexts": tok.vocab_size ** (n - 1)}
        print(f"{n}-gram | alpha={best_a:<6} | doğrulama {ce:.4f} nat "
              f"({ce/LN2:.3f} bpc) | görülen bağlam {lm.n_contexts():,} / "
              f"olası {tok.vocab_size**(n-1):,} [{time.time()-t0:.1f}s]")
        out["models"].append(rec)
    save_json(out, "ngram_baselines")
    return out


def plot(log, ng):
    setup_matplotlib()
    # Şekil: kayıp eğrisi
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(log["iter"], log["train_loss"], color=COLORS["blue"],
            label="Eğitim")
    ax.plot(log["iter"], log["val_loss"], color=COLORS["orange"],
            label="Doğrulama")
    ax.axhline(math.log(65), color=COLORS["gray"], lw=0.8,
               label="Tekdüze: ln 65 = 4.17")
    best_bigram = [m for m in ng["models"] if m["n"] == 2][0]["val_nat"]
    ax.axhline(best_bigram, color=COLORS["aqua"], lw=0.8,
               label=f"Bigram: {best_bigram:.2f}")
    ax.set_xlabel("İterasyon")
    ax.set_ylabel("Çapraz entropi (nat / karakter)")
    ax.set_ylim(0.8, 4.8)
    ax.legend(loc="upper right", fontsize=7)
    fv = log["val_loss"][-1]
    ax.annotate(f"son doğrulama: {fv:.3f} nat = {fv/LN2:.2f} bit",
                xy=(log["iter"][-1], fv), xytext=(0.97, 0.52),
                textcoords="axes fraction", ha="right", va="center",
                fontsize=7.5, color=COLORS["ink2"],
                arrowprops=dict(arrowstyle="-", color=COLORS["gray"], lw=0.6))
    save_fig(fig, "fig_loss_curve")

    # Şekil: n-gram vs nöral (bit/karakter)
    labels = ["Tekdüze"] + [str(m["n"]) for m in ng["models"]] + \
             ["MiniGPT"]
    vals = [ng["uniform_bpc"]] + [m["val_bpc"] for m in ng["models"]] + \
           [log["final_val_bpc"]]
    cols = [COLORS["gray"]] + [COLORS["blue"]] * len(ng["models"]) + \
           [COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    bars = ax.bar(labels, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=COLORS["ink2"])
    ax.set_ylabel("Doğrulama CE (bit / kr.)")
    ax.set_ylim(0, 7.2)
    ax.set_xlabel("n-gram derecesi $n$ (add-$\\alpha$) ve nöral model")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=7.5)
    save_fig(fig, "fig_ngram_vs_neural")


if __name__ == "__main__":
    log, tok, train_ids, val_ids = train()
    ng = ngram_baselines(tok, train_ids, val_ids)
    plot(log, ng)
