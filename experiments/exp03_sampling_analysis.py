"""Deney 03 - Çıkış örnekleme mekanizmaları: Sıcaklık, Top-k, Top-p.

Kılavuz Konu 18: "logit dağılımına farklı Temperature (T ∈ {0.2, 0.7, 1.5}),
Top-k ve Top-p filtreleri uygulandığında kümülatif olasılık kütlesinin ve
entropinin nasıl dönüştüğünün adım adım olasılık çubuk grafikleriyle
gösterilmesi".  Rapor Denk. (12), (15), (17), (18); Şekil 4-6, Tablo II-III.
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
from experiments.common import load_everything, next_token_logits  # noqa
from src import sampling as S                                   # noqa: E402
from src.utils import (COLORS, SERIES, banner, save_fig,        # noqa: E402
                       save_json, set_seed, setup_matplotlib)

TOPN = 12          # çubuk grafiklerde gösterilen belirteç sayısı


@torch.no_grad()
def scan_contexts(model, val_ids, n_windows=40):
    """Doğrulama kümesinde konum bazlı entropi taraması (T=1).

    Üç temsilî bağlam seçilir: en düşük entropili ('kararlı'), en yüksek
    entropili ('belirsiz') ve medyana en yakın ('tipik').
    """
    T = model.block_size
    ents, locs = [], []
    for w in range(n_windows):
        start = w * T
        x = val_ids[start:start + T + 1]
        logits, _ = model(x[:-1].unsqueeze(0))
        p = torch.softmax(logits[0], -1)
        h = S.entropy(p)                                 # (T,)
        for t in range(32, T):                           # yeterli bağlam
            ents.append(h[t].item())
            locs.append(start + t)
    ents, locs = np.array(ents), np.array(locs)
    order = np.argsort(ents)
    picks = {"kararlı": locs[order[0]],
             "tipik": locs[order[len(order) // 2]],
             "belirsiz": locs[order[-1]]}
    return picks, ents


def context_at(val_ids, loc, tok, model):
    """loc konumuna kadar olan bağlam (dâhil) ve gerçek sıradaki karakter."""
    ctx = val_ids[max(0, loc + 1 - model.block_size):loc + 1]
    nxt = int(val_ids[loc + 1])
    return ctx, nxt, tok.decode(ctx[-28:].tolist())


def nucleus_size(probs, p=0.9):
    """|V_p|: kütlesi p'yi aşan en küçük küme (Denk. 18)."""
    ps = torch.sort(probs, descending=True).values
    cum = torch.cumsum(ps, 0)
    return int(((cum - ps) <= p).sum())


def bar_panel(ax, probs, tok, true_next=None, title="", color=COLORS["blue"],
              topn=TOPN, ylim=1.0, stat="support"):
    """Tek bir olasılık dağılımının ilk `topn` belirteci için çubuk grafik."""
    p = probs.numpy()
    idx = np.argsort(-p)[:topn]
    labels = [tok.label(i) for i in idx]
    cols = [color] * topn
    edge = ["none"] * topn
    if true_next is not None:
        for r, i in enumerate(idx):
            if i == true_next:
                cols[r] = COLORS["orange"]
    ax.bar(range(topn), p[idx], color=cols, edgecolor=edge, width=0.7)
    ax.set_xticks(range(topn), labels, fontsize=6.5, family="monospace")
    ax.set_ylim(0, ylim)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="y", labelsize=6.5)
    h = max(S.entropy(probs).item(), 0.0)
    if stat == "support":
        txt = f"H = {h:.2f} bit\n|supp| = {int(S.support_size(probs))}"
    else:
        txt = f"H = {h:.2f} bit\n$|V_{{0.9}}|$ = {nucleus_size(probs)}"
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color=COLORS["ink2"])
    ax.set_title(title, fontsize=7.5)


def fig_temperature(model, tok, ctx, nxt, ctx_str, results):
    """Şekil: aynı logit vektörüne 4 farklı sıcaklık (Denk. 12)."""
    z = next_token_logits(model, ctx)
    setup_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
    rec = []
    for ax, T in zip(axes, C.TEMPERATURES_EXTRA):
        p = S.softmax_temperature(z, T)
        bar_panel(ax, p, tok, nxt, title=f"T = {T}", stat="nucleus")
        rec.append({"T": T, "entropy_bits": S.entropy(p).item(),
                    "max_prob": p.max().item(),
                    "support_gt_1e-3": int((p > 1e-3).sum())})
    axes[0].set_ylabel("$p(x_{t+1} \\mid x_{\\leq t})$", fontsize=8)
    fig.suptitle("Bağlam (son 28 kr.): \"" + ctx_str.replace("\n", "\\n")
                 + f"\"  →  gerçek sıradaki: '{tok.label(nxt)}' (turuncu)",
                 fontsize=7.5, y=1.08, family="monospace")
    save_fig(fig, "fig03_temperature_bars")
    results["temperature_single_context"] = rec
    return z


def fig_cumulative_and_entropy(model, tok, ctxs, results):
    """Şekil: (a) kümülatif olasılık kütlesi, (b) H(T) eğrileri."""
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    z = next_token_logits(model, ctxs["tipik"][0])
    nuc = {}
    for c, T in zip(SERIES, C.TEMPERATURES_EXTRA):
        p = S.softmax_temperature(z, T)
        ps = torch.sort(p, descending=True).values
        cum = torch.cumsum(ps, 0).numpy()
        ranks = np.arange(1, len(cum) + 1)
        ax1.plot(ranks, cum, color=c, label=f"T = {T}", drawstyle="steps-post")
        k90 = int((cum - ps.numpy() <= 0.9).sum())      # |V_p|, p = 0.9
        nuc[T] = k90
        ax1.plot([k90], [cum[k90 - 1]], "o", ms=4, color=c)
    ax1.axhline(0.9, color=COLORS["gray"], lw=0.8)
    ax1.text(40, 0.905, "p = 0.9", fontsize=7, color=COLORS["ink2"])
    ax1.set_xscale("log")
    ax1.set_xlabel("Sıralanmış belirteç sayısı (log)")
    ax1.set_ylabel("Kümülatif olasılık kütlesi")
    ax1.set_title("(a) 'tipik' bağlam; nokta = çekirdek boyutu $|V_{0.9}|$",
                  fontsize=8)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_ylim(0, 1.02)

    Ts = np.linspace(0.05, 3.0, 120)
    for c, (name, (ctx, _, _)) in zip(SERIES, ctxs.items()):
        z = next_token_logits(model, ctx)
        H = [S.entropy(S.softmax_temperature(z, t)).item() for t in Ts]
        ax2.plot(Ts, H, color=c, label=f"{name} bağlam")
    ax2.axhline(math.log2(tok.vocab_size), color=COLORS["gray"], lw=0.8)
    ax2.text(2.0, math.log2(tok.vocab_size) + 0.12,
             "$\\log_2 |V| = 6.02$ bit", fontsize=7, color=COLORS["ink2"])
    for T in C.TEMPERATURES:
        ax2.axvline(T, color=COLORS["grid"], lw=0.8)
    ax2.set_xlabel("Sıcaklık $T$")
    ax2.set_ylabel("Entropi $H(p_T)$ (bit)")
    ax2.set_title("(b) Entropi sıcaklıkla monoton artar (Denk. 16)",
                  fontsize=8)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.set_ylim(0, 6.5)
    save_fig(fig, "fig03_cumulative_entropy")
    results["nucleus_size_p0.9_typical_context"] = nuc


def fig_topk_topp(model, tok, ctxs, results):
    """Şekil: kararlı vs belirsiz bağlamda top-k ve top-p (Denk. 17-18)."""
    setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 2.9), sharey=True)
    rec = {}
    for row, name in enumerate(("kararlı", "belirsiz")):
        ctx, nxt, s = ctxs[name]
        z = next_token_logits(model, ctx)
        p_full = S.softmax_temperature(z, 1.0)
        configs = [("tam dağılım (T=1)", p_full),
                   ("top-k, k = 5", S.sampling_pipeline(z, 1.0, top_k=5)),
                   ("top-p, p = 0.9", S.sampling_pipeline(z, 1.0, top_p=0.9))]
        rec[name] = {"context_tail": s, "true_next": tok.label(nxt)}
        for col, (title, p) in enumerate(configs):
            ax = axes[row, col]
            bar_panel(ax, p, tok, nxt, title=f"{name}: {title}")
            kept = int(S.support_size(p))
            removed = float(1 - p_full[p > 0].sum()) if col else 0.0
            rec[name][title] = {"entropy_bits": S.entropy(p).item(),
                                "support": kept, "removed_mass": removed,
                                "max_prob": p.max().item()}
            if col:
                ax.text(0.97, 0.56, f"atılan kütle\n= {removed:.3f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6.5, color=COLORS["orange"])
    axes[0, 0].set_ylabel("olasılık", fontsize=8)
    axes[1, 0].set_ylabel("olasılık", fontsize=8)
    fig.suptitle("Üst: '" + ctxs["kararlı"][2].replace("\n", "\\n") +
                 "'   Alt: '" + ctxs["belirsiz"][2].replace("\n", "\\n") +
                 "'", fontsize=7, y=1.0, family="monospace")
    fig.tight_layout()
    save_fig(fig, "fig03_topk_topp")
    results["topk_topp_two_contexts"] = rec


def fig_stepwise(model, tok, val_ids, results):
    """Şekil: ardışık 6 konumda, 3 sıcaklıkta adım adım dağılımlar."""
    start = 20 * model.block_size + 50
    # Okunur bir başlangıç: bir satır başından itibaren
    while tok.itos[int(val_ids[start])] != "\n":
        start += 1
    start += 1
    n_steps = 5
    setup_matplotlib()
    fig, axes = plt.subplots(len(C.TEMPERATURES), n_steps,
                             figsize=(7.2, 2.8), sharey=True)
    rec = []
    for step in range(n_steps):
        loc = start + 24 + step
        ctx = val_ids[start:loc + 1]
        nxt = int(val_ids[loc + 1])
        z = next_token_logits(model, ctx)
        for r, T in enumerate(C.TEMPERATURES):
            p = S.softmax_temperature(z, T)
            ax = axes[r, step]
            bar_panel(ax, p, tok, nxt, topn=8, stat="nucleus",
                      title=(f"…{tok.decode(ctx[-10:].tolist())!s}"
                             .replace("\n", "\\n")) if r == 0 else "")
            if step == 0:
                ax.set_ylabel(f"T = {T}", fontsize=8)
            rec.append({"step": step, "T": T, "true_next": tok.label(nxt),
                        "entropy_bits": S.entropy(p).item(),
                        "p_true": p[nxt].item()})
    fig.suptitle("Ardışık konumlarda sıradaki-karakter dağılımı; turuncu = "
                 "gerçek karakter", fontsize=8, y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig03_stepwise_bars")
    results["stepwise"] = rec


@torch.no_grad()
def aggregate_statistics(model, tok, val_ids, results, n_positions=4000):
    """Tablo: binlerce doğrulama konumu üzerinde ortalama istatistikler."""
    banner("Deney 03 - toplu istatistikler (Tablo II-III)")
    T_blk = model.block_size
    zs = []
    n_windows = n_positions // (T_blk - 32)
    for w in range(n_windows):
        x = val_ids[w * T_blk:(w + 1) * T_blk + 1]
        logits, _ = model(x[:-1].unsqueeze(0))
        zs.append(logits[0, 32:])
    Z = torch.cat(zs)[:n_positions]                   # (N, V)
    out = {"n_positions": int(Z.shape[0]), "temperature": [], "top_k": [],
           "top_p": []}
    for T in [0.2, 0.5, 0.7, 1.0, 1.5, 2.0]:
        P = S.softmax_temperature(Z, T)
        ps = torch.sort(P, descending=True, dim=-1).values
        cum = torch.cumsum(ps, -1)
        nuc90 = ((cum - ps) <= 0.9).sum(-1).float()
        nuc95 = ((cum - ps) <= 0.95).sum(-1).float()
        out["temperature"].append({
            "T": T, "mean_entropy_bits": S.entropy(P).mean().item(),
            "mean_max_prob": P.max(-1).values.mean().item(),
            "mean_nucleus_0.9": nuc90.mean().item(),
            "mean_nucleus_0.95": nuc95.mean().item()})
        print(f"T={T:<4} | H̄ = {S.entropy(P).mean():.3f} bit | max p̄ = "
              f"{P.max(-1).values.mean():.3f} | |V_0.9|¯ = {nuc90.mean():.2f}"
              f" | |V_0.95|¯ = {nuc95.mean():.2f}")
    P1 = S.softmax_temperature(Z, 1.0)
    for k in C.TOP_KS:
        Pk = S.sampling_pipeline(Z, 1.0, top_k=k)
        removed = 1 - (P1 * (Pk > 0)).sum(-1)
        out["top_k"].append({"k": k, "mean_entropy_bits":
                             S.entropy(Pk).mean().item(),
                             "mean_removed_mass": removed.mean().item(),
                             "max_removed_mass": removed.max().item()})
        print(f"top-k k={k:<3}| H̄ = {S.entropy(Pk).mean():.3f} bit | atılan"
              f" kütle ort. {removed.mean():.3f}, maks {removed.max():.3f}")
    for p in C.TOP_PS:
        Pp = S.sampling_pipeline(Z, 1.0, top_p=p)
        sup = S.support_size(Pp).float()
        removed = 1 - (P1 * (Pp > 0)).sum(-1)
        out["top_p"].append({"p": p, "mean_entropy_bits":
                             S.entropy(Pp).mean().item(),
                             "mean_support": sup.mean().item(),
                             "min_support": int(sup.min()),
                             "max_support": int(sup.max()),
                             "mean_removed_mass": removed.mean().item(),
                             "max_removed_mass": removed.max().item()})
        print(f"top-p p={p:<4}| H̄ = {S.entropy(Pp).mean():.3f} bit | |V_p| "
              f"ort. {sup.mean():.2f} (min {int(sup.min())}, maks "
              f"{int(sup.max())}) | atılan kütle ort. {removed.mean():.3f},"
              f" maks {removed.max():.3f}")
    results["aggregate"] = out


def main():
    banner("Deney 03 - Sıcaklık / Top-k / Top-p mekanizma analizi")
    set_seed(C.SEED)
    _, tok, _, val_ids, model = load_everything()
    picks, ents = scan_contexts(model, val_ids)
    results = {"entropy_scan": {"n": int(len(ents)),
                                "mean_bits": float(ents.mean()),
                                "min_bits": float(ents.min()),
                                "max_bits": float(ents.max())}}
    ctxs = {name: context_at(val_ids, loc, tok, model)
            for name, loc in picks.items()}
    for name, (ctx, nxt, s) in ctxs.items():
        print(f"{name:>8} bağlam: '{s!r}' -> gerçek '{tok.label(nxt)}'")
    results["contexts"] = {n: {"tail": s, "true_next": tok.label(nxt)}
                           for n, (ctx, nxt, s) in ctxs.items()}
    fig_temperature(model, tok, *ctxs["tipik"], results)
    fig_cumulative_and_entropy(model, tok, ctxs, results)
    fig_topk_topp(model, tok, ctxs, results)
    fig_stepwise(model, tok, val_ids, results)
    aggregate_statistics(model, tok, val_ids, results)
    save_json(results, "exp03_sampling_analysis")


if __name__ == "__main__":
    main()
