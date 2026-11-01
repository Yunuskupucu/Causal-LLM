"""Deney 04 - Sıcaklığın Softmax türevi üzerindeki analitik etkisi.

Kılavuz Konu 18 (Savunma Çıktısı): "sıcaklık parametresinin Softmax türevi
üzerindeki analitik etkisi". Rapor Denk. (12)-(16).

  A) Jakobiyen: d p_i / d z_j = (1/T) p_i (δ_ij - p_j); otomatik türev ve
     sonlu farklarla doğrulama.
  B) Kayıp gradyanı: dL/dz = (1/T)(p_T - y); normu T ile 1/T gibi ölçeklenir.
  C) Entropi türevi: dH/dT = Var_{p_T}(z) / T^3 >= 0 (monotonluk).
  D) Sınır durumları: T -> 0 (argmax), T -> inf (tekdüze).
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


@torch.no_grad()
def pick_context(model, val_ids, lo=1.5, hi=2.5):
    """Entropisi [lo, hi] bit aralığında olan ilk konum (deterministik)."""
    T = model.block_size
    for w in range(50):
        x = val_ids[w * T:(w + 1) * T + 1]
        logits, _ = model(x[:-1].unsqueeze(0))
        h = S.entropy(torch.softmax(logits[0], -1))
        for t in range(32, T):
            if lo <= h[t] <= hi:
                return w * T + t
    raise RuntimeError("uygun bağlam bulunamadı")


def part_a_jacobian(z, results):
    banner("Deney 04-A: Softmax Jakobiyeni (Denk. 13)")
    rows = []
    for T in C.TEMPERATURES_EXTRA:
        J_an = S.softmax_jacobian_analytic(z, T)
        J_ad = torch.autograd.functional.jacobian(
            lambda v: torch.softmax(v / T, -1), z)
        eps = 1e-3
        J_fd = torch.zeros_like(J_an)
        for j in range(len(z)):
            e = torch.zeros_like(z)
            e[j] = eps
            J_fd[:, j] = (torch.softmax((z + e) / T, -1) -
                          torch.softmax((z - e) / T, -1)) / (2 * eps)
        rows.append({"T": T,
                     "max_err_autograd": (J_an - J_ad).abs().max().item(),
                     "max_err_finite_diff": (J_an - J_fd).abs().max().item(),
                     "frobenius_norm": J_an.norm().item(),
                     "trace": torch.trace(J_an).item()})
        print(f"T={T:<4}| max|J_an - J_autograd| = "
              f"{rows[-1]['max_err_autograd']:.1e} | max|J_an - J_fd| = "
              f"{rows[-1]['max_err_finite_diff']:.1e} | ||J||_F = "
              f"{rows[-1]['frobenius_norm']:.4f}")
    results["jacobian"] = rows


def part_b_loss_gradient(z, results):
    """İki hedef: y = argmax (doğru, kendinden emin) ve y = 2. en olası
    (yanlış). (p_T - y) ilkinde T->0 iken üstel hızla sıfıra gider, ikincisinde
    sabit kalır; 1/T çarpanı yalnızca ikincisinde patlamaya yol açar.
    """
    banner("Deney 04-B: dL/dz = (1/T)(p_T - y) (Denk. 14)")
    Ts = np.geomspace(0.05, 5.0, 60)
    order = torch.argsort(z, descending=True)
    targets = {"doğru (y = argmax)": int(order[0]),
               "yanlış (y = 2. aday)": int(order[1])}
    out, max_err = {}, 0.0
    for name, y_idx in targets.items():
        y = torch.zeros_like(z)
        y[y_idx] = 1.0
        norms, norms_ad = [], []
        for T in Ts:
            p = S.softmax_temperature(z, float(T))
            g_an = (p - y) / T
            zz = z.clone().requires_grad_(True)
            loss = -torch.log_softmax(zz / T, -1)[y_idx]
            loss.backward()
            norms.append(g_an.norm().item())
            norms_ad.append(zz.grad.norm().item())
        max_err = max(max_err, float(np.max(np.abs(
            np.array(norms) - np.array(norms_ad)))))
        out[name] = norms
    print(f"analitik vs autograd gradyan normu, maks fark = {max_err:.1e}")
    print(f"T=0.05: doğru hedef ||g|| = {out['doğru (y = argmax)'][0]:.2e}, "
          f"yanlış hedef ||g|| = {out['yanlış (y = 2. aday)'][0]:.2e} "
          f"(sqrt(2)/T = {math.sqrt(2)/Ts[0]:.2e})")
    results["loss_gradient"] = {"T": Ts.tolist(), "grad_norm": out,
                                "max_err_vs_autograd": max_err}
    return Ts, out


def part_c_entropy_derivative(z, results):
    banner("Deney 04-C: dH/dT = Var_p(z)/T^3 (Denk. 16)")
    Ts = np.linspace(0.1, 3.0, 59)
    H, dH_an, dH_fd = [], [], []
    eps = 1e-3
    for T in Ts:
        p = S.softmax_temperature(z, float(T))
        H.append(S.entropy(p, base=math.e).item())          # nat
        dH_an.append(S.entropy_temperature_derivative_analytic(
            z, float(T)).item())
        hp = S.entropy(S.softmax_temperature(z, float(T) + eps), math.e)
        hm = S.entropy(S.softmax_temperature(z, float(T) - eps), math.e)
        dH_fd.append(((hp - hm) / (2 * eps)).item())
    err = float(np.max(np.abs(np.array(dH_an) - np.array(dH_fd))))
    print(f"analitik vs sonlu-fark dH/dT, maks fark = {err:.1e}; "
          f"min dH/dT = {min(dH_an):.3e} (>= 0 beklenir)")
    results["entropy_derivative"] = {"T": Ts.tolist(), "H_nat": H,
                                     "dHdT_analytic": dH_an,
                                     "dHdT_finite_diff": dH_fd,
                                     "max_err": err, "min_dHdT": min(dH_an)}
    return Ts, H, dH_an, dH_fd


def part_d_limits(z, results):
    banner("Deney 04-D: sınır durumları")
    rows = []
    for T in (1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4):
        p = S.softmax_temperature(z, T)
        rows.append({"T": T, "max_prob": p.max().item(),
                     "entropy_bits": S.entropy(p).item()})
        print(f"T={T:<8g}| max p = {p.max():.4f} | H = {S.entropy(p):.4f} bit")
    print(f"tekdüze sınır: log2|V| = {math.log2(len(z)):.4f} bit")
    results["limits"] = rows


def plot(Ts_b, norms, Ts_c, H, dH_an, dH_fd, z):
    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.8),
                             constrained_layout=True)
    ax = axes[0]
    ax.plot(Ts_c, H, color=SERIES[0], label="$H(p_T)$ (nat)")
    ax.plot(Ts_c, dH_an, color=SERIES[1],
            label="$dH/dT$ analitik: $\\mathrm{Var}_{p_T}(z)/T^3$")
    ax.plot(Ts_c[::4], dH_fd[::4], "o", ms=3.5, color=SERIES[1], mfc="white",
            label="$dH/dT$ sonlu fark")
    ax.set_xlabel("Sıcaklık $T$")
    ax.set_title("(a) Entropi ve türevi", fontsize=8.5)
    ax.set_ylim(top=5.6)   # gösterge eğrilerin üstündeki boşlukta kalsın
    ax.legend(fontsize=6.2, loc="upper center")
    ax = axes[1]
    for c, (name, nrm) in zip(SERIES, norms.items()):
        ax.plot(Ts_b, nrm, color=c, label=name)
    ax.plot(Ts_b, math.sqrt(2) / Ts_b, color=COLORS["gray"], lw=0.9,
            label="$\\sqrt{2}/T$ (üst sınır)")
    ax.set_ylim(1e-6, 1e2)
    ax.set_ylabel("$\\|(p_T - y)/T\\|_2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sıcaklık $T$ (log)")
    ax.set_title("(b) Kayıp gradyanının normu", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="lower right")
    ax = axes[2]
    Ts = np.geomspace(1e-2, 1e2, 200)
    mx = [S.softmax_temperature(z, float(t)).max().item() for t in Ts]
    ax.plot(Ts, mx, color=SERIES[0])
    ax.axhline(1 / len(z), color=COLORS["gray"], lw=0.8)
    ax.text(0.012, 1 / len(z) + 0.03, "$1/|V|$ (tekdüze)", fontsize=6.5,
            color=COLORS["ink2"])
    ax.set_xscale("log")
    ax.set_xlabel("Sıcaklık $T$ (log)")
    ax.set_ylabel("$\\max_i p_i(T)$")
    ax.set_title("(c) argmax → tekdüze geçişi", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    save_fig(fig, "fig04_temperature_derivatives")


def main():
    set_seed(C.SEED)
    _, tok, _, val_ids, model = load_everything()
    # Orta belirsizlikte (1.5-2.5 bit) ilk doğrulama konumu seçilir.
    loc = pick_context(model, val_ids)
    ctx = val_ids[max(0, loc + 1 - model.block_size):loc + 1]
    y_idx = int(val_ids[loc + 1])
    z = next_token_logits(model, ctx).detach()
    print(f"bağlam sonu: {tok.decode(ctx[-30:].tolist())!r} -> gerçek "
          f"'{tok.label(y_idx)}'; logit aralığı [{z.min():.2f}, "
          f"{z.max():.2f}]")
    results = {"context_tail": tok.decode(ctx[-30:].tolist()),
               "true_next": tok.label(y_idx)}
    part_a_jacobian(z, results)
    Ts_b, norms = part_b_loss_gradient(z, results)
    Ts_c, H, dH_an, dH_fd = part_c_entropy_derivative(z, results)
    part_d_limits(z, results)
    plot(Ts_b, norms, Ts_c, H, dH_an, dH_fd, z)
    save_json(results, "exp04_softmax_temperature")


if __name__ == "__main__":
    main()
