"""Deney 01 - Nedensel maskenin doğrulanması (Mekanizma İspatı).

Kılavuz Konu 18: "alt-üçgen maske matrisinin (M_ij = -inf, j > i) dikkat
skorlarına uygulanmasının doğrulanması".

Dört bağımsız kanıt üretilir:
  A) Tek bir baş için S, S+M ve A = softmax(S+M) matrislerinin ısı haritası
     (Rapor Denk. (6)-(8), Şekil 2).
  B) Eğitilmiş modelin TÜM katman/başlarında A'nın üst üçgeninin tam 0 ve
     satır toplamlarının 1 olduğunun sayısal kontrolü.
  C) Bozma (perturbation) testi: x_j değiştirildiğinde yalnızca i >= j
     konumlarının logitleri değişir -> alt-üçgen etki matrisi.
  D) Jakobiyen testi (Denk. (19)): ||d z_i / d e_j||_F = 0 (j > i), otomatik
     türevle; maske kaldırıldığında matris tam dolu hâle gelir.
  E) Sızıntı eğitimi: maskesiz eğitilen model teacher-forcing kaybını ~0'a
     indirir ("hile"), ancak dürüst (yalnızca geçmiş) değerlendirmede çöker.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C                                              # noqa: E402
from src.data import get_batch, load_corpus, make_splits        # noqa: E402
from src.model import MiniGPT, causal_mask                      # noqa: E402
from src.tokenizer import CharTokenizer                         # noqa: E402
from src.utils import (COLORS, DIV_BLUE_RED, RES_DIR, SEQ_BLUE,  # noqa: E402
                       banner, save_fig, save_json, set_seed,
                       setup_matplotlib)

CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("div", DIV_BLUE_RED)


def load_trained(tok):
    ck = torch.load(RES_DIR / "model.pt", map_location="cpu")
    cfg = ck["config"]
    model = MiniGPT(tok.vocab_size, cfg["BLOCK_SIZE"], cfg["N_LAYER"],
                    cfg["N_HEAD"], cfg["D_MODEL"], cfg["DROPOUT"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def part_a_heatmaps(model, tok, results):
    banner("Deney 01-A: S, S+M ve A matrisleri (tek baş)")
    text = "ROMEO: I"
    idx = torch.tensor([tok.encode(text)])
    with torch.no_grad():
        _, _, infos = model(idx, return_attn=True)
    layer, head = 0, 0
    S = infos[layer]["scores_raw"][0, head].numpy()
    SM = infos[layer]["scores_masked"][0, head].numpy()
    A = infos[layer]["attn"][0, head].numpy()
    T = len(text)
    labels = [tok.label(i) for i in idx[0].tolist()]

    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),
                             constrained_layout=True)
    vmax = np.abs(S).max()
    panels = [(S, "(a) Ham skor $S = QK^T/\\sqrt{d_k}$", CMAP_DIV, -vmax, vmax),
              (SM, "(b) Maskeli skor $S + M$", CMAP_DIV, -vmax, vmax),
              (A, "(c) $A = \\mathrm{softmax}(S+M)$", CMAP_SEQ, 0, 1)]
    for ax, (mat, title, cmap, lo, hi) in zip(axes, panels):
        show = np.where(np.isinf(mat), np.nan, mat)
        im = ax.imshow(show, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(range(T), labels, fontsize=7)
        ax.set_yticks(range(T), labels, fontsize=7)
        ax.set_xlabel("anahtar konumu $j$", fontsize=7.5)
        ax.grid(False)
        for i in range(T):
            for j in range(T):
                v = mat[i, j]
                dec = 2 if cmap is CMAP_SEQ else 1
                txt = "$-\\infty$" if np.isinf(v) else f"{v:.{dec}f}"
                col = "white" if (cmap is CMAP_SEQ and v > 0.55) else \
                    COLORS["ink"]
                if np.isinf(v):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                               color="#f0efec"))
                    col = COLORS["ink2"]
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.4,
                        color=col)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("sorgu konumu $i$", fontsize=7.5)
    fig.suptitle(f"Katman {layer+1}, Baş {head+1}; girdi: \"{text}\" "
                 f"(T = {T})", fontsize=8.5)
    save_fig(fig, "fig01_mask_heatmaps")
    results["part_a"] = {"text": text, "layer": layer, "head": head,
                         "upper_triangle_max_A": float(
                             np.triu(A, 1).max()),
                         "row_sums": A.sum(1).tolist()}


def part_b_numeric(model, tok, val_ids, results):
    banner("Deney 01-B: tüm katman/başlarda üst üçgen == 0 kontrolü")
    g = torch.Generator().manual_seed(C.SEED + 11)
    x, _ = get_batch(val_ids, C.BLOCK_SIZE, 8, g)
    with torch.no_grad():
        _, _, infos = model(x, return_attn=True)
    T = x.shape[1]
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    worst_upper, worst_rowsum, n_checked = 0.0, 0.0, 0
    for li, info in enumerate(infos):
        A = info["attn"]                        # (B, h, T, T)
        worst_upper = max(worst_upper, A[..., upper].abs().max().item())
        worst_rowsum = max(worst_rowsum,
                           (A.sum(-1) - 1).abs().max().item())
        n_checked += A.shape[0] * A.shape[1]
    print(f"{n_checked} dikkat matrisi (T={T}) kontrol edildi:")
    print(f"  max |A_ij| (j>i)      = {worst_upper:.3e}  (beklenen: 0)")
    print(f"  max |sum_j A_ij - 1|  = {worst_rowsum:.3e}  (beklenen: ~1e-7)")
    results["part_b"] = {"n_matrices": n_checked, "T": T,
                         "max_upper_triangle": worst_upper,
                         "max_rowsum_error": worst_rowsum}


def perturbation_matrix(model, idx, use_mask, g):
    """D[i, j] = ||z_i(x') - z_i(x)||_2, x' = x ile j. belirteç değiştirilmiş."""
    T = idx.shape[1]
    with torch.no_grad():
        base, _ = model(idx, use_mask=use_mask)
    D = np.zeros((T, T))
    for j in range(T):
        alt = idx.clone()
        new = (idx[0, j] + 1 + torch.randint(0, model.vocab_size - 1, (1,),
                                             generator=g)) % model.vocab_size
        alt[0, j] = new
        with torch.no_grad():
            z, _ = model(alt, use_mask=use_mask)
        D[:, j] = (z[0] - base[0]).norm(dim=-1).numpy()
    return D


def jacobian_matrix(model, idx, use_mask):
    """J[i, j] = ||d z_i / d e_j||_F  (Denk. (19)), e_j: j. konum gömmesi."""
    e = model.tok_emb(idx).detach()            # (1, T, d)

    def f(emb):
        return model.forward_from_embeddings(emb.unsqueeze(0), use_mask)[0]

    J = torch.autograd.functional.jacobian(f, e[0])   # (T, V, T, d)
    return J.flatten(2).norm(dim=2).norm(dim=1).numpy() if False else \
        torch.sqrt((J ** 2).sum(dim=(1, 3))).numpy()


def part_cd_leak_tests(model, tok, results):
    banner("Deney 01-C/D: bozma ve Jakobiyen sızıntı testleri")
    text = "First Citizen:\nBefore we proceed"
    text = text[:16]
    idx = torch.tensor([tok.encode(text)])
    T = idx.shape[1]
    g = torch.Generator().manual_seed(C.SEED + 12)
    D_mask = perturbation_matrix(model, idx, True, g)
    D_nomask = perturbation_matrix(model, idx, False, g)
    J_mask = jacobian_matrix(model, idx, True)
    J_nomask = jacobian_matrix(model, idx, False)
    upper = np.triu(np.ones((T, T), dtype=bool), 1)
    lower = ~upper
    results["part_c_perturbation"] = {
        "T": T, "text": text,
        "masked_max_upper": float(D_mask[upper].max()),
        "masked_min_lower": float(D_mask[lower].min()),
        "unmasked_max_upper": float(D_nomask[upper].max())}
    results["part_d_jacobian"] = {
        "masked_max_upper": float(J_mask[upper].max()),
        "masked_min_lower": float(J_mask[lower].min()),
        "unmasked_max_upper": float(J_nomask[upper].max())}
    for name, r in (("bozma", results["part_c_perturbation"]),
                    ("Jakobiyen", results["part_d_jacobian"])):
        print(f"{name:>9} | maskeli: max üst-üçgen = {r['masked_max_upper']:.2e}"
              f", min alt-üçgen = {r['masked_min_lower']:.2e} | maskesiz: "
              f"max üst-üçgen = {r['unmasked_max_upper']:.2e}")

    setup_matplotlib()
    labels = [tok.label(i) for i in idx[0].tolist()]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2),
                             constrained_layout=True)
    mats = [(D_mask, "(a) Bozma, maskeli"), (D_nomask, "(b) Bozma, maskesiz"),
            (J_mask, "(c) Jakobiyen, maskeli"),
            (J_nomask, "(d) Jakobiyen, maskesiz")]
    for ax, (m, title) in zip(axes, mats):
        im = ax.imshow(m, cmap=CMAP_SEQ, vmin=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(T), labels, fontsize=5.5)
        ax.set_yticks(range(T), labels, fontsize=5.5)
        ax.grid(False)
        ax.set_xlabel("değiştirilen konum $j$", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axes[0].set_ylabel("etkilenen çıktı konumu $i$", fontsize=7)
    save_fig(fig, "fig01_leak_tests")


def train_small(use_mask: bool, train_ids, val_ids, tok, iters=1200):
    """Kısa eğitim: maskeli vs maskesiz (2 katman, d=64, T=64)."""
    set_seed(C.SEED)
    model = MiniGPT(tok.vocab_size, 64, n_layer=2, n_head=4, d_model=64,
                    dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(C.SEED + 13)
    losses = []
    for it in range(iters):
        x, y = get_batch(train_ids, 64, 32, g)
        _, loss = model(x, y, use_mask=use_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    # Teacher-forcing (paralel) kaybı: tüm dizi aynı anda verilir.
    g_eval = torch.Generator().manual_seed(C.SEED + 14)
    xv, yv = get_batch(val_ids, 64, 16, g_eval)
    with torch.no_grad():
        _, tf_loss = model(xv, yv, use_mask=use_mask)
        # Dürüst (yalnızca geçmiş) kayıp: her t için yalnızca x_{<=t} verilir
        # ve son konumun logiti kullanılır -> Denk. (1)'in sıralı hâli.
        total, count = 0.0, 0
        for t in range(1, 33):
            z, _ = model(xv[:, :t], use_mask=use_mask)
            logp = torch.log_softmax(z[:, -1], -1)
            total += -logp.gather(-1, yv[:, t - 1:t]).sum().item()
            count += xv.shape[0]
    return losses, tf_loss.item(), total / count


def part_e_leak_training(tok, train_ids, val_ids, results):
    banner("Deney 01-E: maskesiz eğitim = bilgi sızıntısı ('hile')")
    t0 = time.time()
    lm, tf_m, hon_m = train_small(True, train_ids, val_ids, tok)
    ln, tf_n, hon_n = train_small(False, train_ids, val_ids, tok)
    print(f"maskeli : eğitim son {np.mean(lm[-20:]):.3f} | teacher-forcing "
          f"{tf_m:.3f} | dürüst {hon_m:.3f} nat")
    print(f"maskesiz: eğitim son {np.mean(ln[-20:]):.3f} | teacher-forcing "
          f"{tf_n:.3f} | dürüst {hon_n:.3f} nat  (tekdüze = "
          f"{math.log(tok.vocab_size):.3f})  [{time.time()-t0:.0f}s]")
    results["part_e"] = {"masked": {"train_last20": float(np.mean(lm[-20:])),
                                    "teacher_forced": tf_m, "honest": hon_m},
                         "unmasked": {"train_last20": float(np.mean(ln[-20:])),
                                      "teacher_forced": tf_n,
                                      "honest": hon_n},
                         "uniform": math.log(tok.vocab_size)}
    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.plot(lm, color=COLORS["blue"], label="Maskeli (nedensel)")
    ax1.plot(ln, color=COLORS["orange"], label="Maskesiz (çift yönlü)")
    ax1.axhline(math.log(tok.vocab_size), color=COLORS["gray"], lw=0.8,
                label="Tekdüze ln|V|")
    ax1.set_xlabel("İterasyon")
    ax1.set_ylabel("Eğitim kaybı (nat / karakter)")
    ax1.legend()
    ax1.set_title("(a) Teacher-forcing eğitim kaybı (2 katman, d=64)",
                  fontsize=8.5)
    ax1.set_ylim(-0.1, 4.6)
    xs = np.arange(2)
    w = 0.36
    ax2.bar(xs - w / 2, [tf_m, tf_n], w, color=COLORS["blue"],
            label="Teacher-forcing (paralel)")
    ax2.bar(xs + w / 2, [hon_m, hon_n], w, color=COLORS["orange"],
            label="Dürüst (yalnızca geçmiş)")
    ax2.axhline(math.log(tok.vocab_size), color=COLORS["gray"], lw=0.8)
    ax2.set_xticks(xs, ["Maskeli", "Maskesiz"])
    ax2.set_ylabel("Doğrulama kaybı (nat)")
    ax2.set_title("(b) Değerlendirme kipi", fontsize=8.5)
    ax2.legend(fontsize=6.5, loc="upper left")
    ax2.set_ylim(0, 9.8)
    ax2.grid(axis="x", visible=False)
    for x_, v in zip(list(xs - w / 2) + list(xs + w / 2),
                     [tf_m, tf_n, hon_m, hon_n]):
        ax2.text(x_, v + 0.05, f"{v:.2f}", ha="center", fontsize=6.5,
                 color=COLORS["ink2"])
    save_fig(fig, "fig01_leak_training")


def main():
    set_seed(C.SEED)
    text = load_corpus()
    tok = CharTokenizer(text)
    train_ids, val_ids = make_splits(text, tok)
    results = {"mask_T6": causal_mask(6).tolist()}
    print("M (T=6):\n", causal_mask(6))
    model = load_trained(tok)
    part_a_heatmaps(model, tok, results)
    part_b_numeric(model, tok, val_ids, results)
    part_cd_leak_tests(model, tok, results)
    part_e_leak_training(tok, train_ids, val_ids, results)
    save_json(results, "exp01_mask_verification")


if __name__ == "__main__":
    main()
