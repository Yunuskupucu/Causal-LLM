"""Deney 07 - Endüstriyel kütüphane karşılaştırması ve sayısal kararlılık.

Rapor Bölüm V (Tartışma): kütüphanelerin arka planda yaptığı örtük işler.
  A) Açık dikkat (bizim) vs torch.nn.functional.scaled_dot_product_attention
     (is_causal=True): sayısal eşitlik ve T'ye bağlı süre/bellek (O(T^2)).
  B) Manuel log-softmax (logsumexp hilesi) vs F.cross_entropy; saf
     softmax->log uygulamasının büyük logitlerde çökmesi.
  C) Maske değeri: -inf vs torch.finfo(dtype).min; tamamen maskelenmiş
     satırın NaN üretmesi (Hugging Face'in sonlu değer tercihinin nedeni).
  D) Top-p uygulamamız ile Hugging Face TopPLogitsWarper algoritmasının
     (artan sıralama + cumsum <= 1-p) kümelerinin eşitliği.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C                                              # noqa: E402
from src.model import (CausalSelfAttention, causal_mask,         # noqa: E402
                       log_softmax_stable, sequence_nll)
from src.sampling import top_p_filter                           # noqa: E402
from src.utils import (COLORS, banner, save_fig, save_json,     # noqa: E402
                       set_seed, setup_matplotlib)


def explicit_attention(q, k, v):
    T = q.shape[-2]
    S = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    A = F.softmax(S + causal_mask(T), dim=-1)
    return A @ v


def part_a(results, timing):
    banner("Deney 07-A: açık dikkat vs F.scaled_dot_product_attention")
    set_seed(C.SEED)                      # q, k, v rastgele: tohum sabit
    rows = []
    for T in (16, 128, 512, 1024, 2048):
        q, k, v = (torch.randn(1, C.N_HEAD, T, C.D_MODEL // C.N_HEAD)
                   for _ in range(3))
        ours = explicit_attention(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        diff = (ours - ref).abs().max().item()
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            explicit_attention(q, k, v)
        t_ours = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        t_ref = (time.perf_counter() - t0) / reps
        mem_mb = C.N_HEAD * T * T * 4 / 2 ** 20
        rows.append({"T": T, "max_abs_diff": diff, "attn_matrix_MB": mem_mb})
        # Süreler donanıma bağlıdır; bit-eşitlik kontrolü dışında tutulur.
        timing.append({"T": T, "t_explicit_ms": t_ours * 1e3,
                       "t_sdpa_ms": t_ref * 1e3})
        print(f"T={T:5d} | max|fark| = {diff:.1e} | açık {t_ours*1e3:7.2f} ms"
              f" | SDPA {t_ref*1e3:7.2f} ms | A matrisi {mem_mb:7.2f} MB")
    results["part_a"] = rows
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    Ts = [r["T"] for r in rows]
    ax.plot(Ts, [r["t_explicit_ms"] for r in timing], "o-", ms=4,
            color=COLORS["blue"], label="Açık (bu çalışma, $T\\times T$)")
    ax.plot(Ts, [r["t_sdpa_ms"] for r in timing], "s-", ms=4,
            color=COLORS["orange"], label="torch SDPA (is_causal=True)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Dizi uzunluğu $T$")
    ax.set_ylabel("Süre (ms, CPU)")
    ax.legend(fontsize=7)
    save_fig(fig, "fig07_attention_timing")


def part_b(results):
    banner("Deney 07-B: log-softmax kararlılığı")
    set_seed(C.SEED)
    logits = torch.randn(4, 16, 65) * 3
    targets = torch.randint(0, 65, (4, 16))
    ours = sequence_nll(logits, targets).item()
    ref = F.cross_entropy(logits.view(-1, 65), targets.view(-1)).item()
    big = logits * 200                         # aşırı büyük logitler
    naive = torch.log(torch.softmax(big, -1)).gather(
        -1, targets.unsqueeze(-1))
    stable = log_softmax_stable(big).gather(-1, targets.unsqueeze(-1))
    naive_nonfinite = int((~torch.isfinite(naive)).sum())
    print(f"manuel NLL {ours:.6f} vs F.cross_entropy {ref:.6f} "
          f"(fark {abs(ours-ref):.1e})")
    print(f"x200 logit: saf softmax->log'da sonlu olmayan değer sayısı = "
          f"{naive_nonfinite}/{naive.numel()}; logsumexp sürümü tamamı "
          f"sonlu = {bool(torch.isfinite(stable).all())}")
    results["part_b"] = {"manual_nll": ours, "torch_cross_entropy": ref,
                         "naive_nonfinite_count": naive_nonfinite,
                         "stable_all_finite": bool(
                             torch.isfinite(stable).all())}


def part_c(results):
    banner("Deney 07-C: -inf vs finfo.min maske değeri")
    s = torch.randn(4)
    row_inf = torch.softmax(s + torch.full((4,), float("-inf")), -1)
    row_min = torch.softmax(s + torch.full((4,), torch.finfo(
        torch.float32).min), -1)
    half = torch.float16
    s16 = s.to(half)
    row16 = torch.softmax(s16 + torch.finfo(half).min, -1)
    print("tamamen maskeli satır, -inf      ->", row_inf.tolist())
    print("tamamen maskeli satır, finfo.min ->", row_min.tolist())
    print("fp16, finfo(fp16).min            ->", row16.tolist())
    results["part_c"] = {"all_minus_inf_row": row_inf.tolist(),
                         "all_finfo_min_row": row_min.tolist(),
                         "fp16_finfo_min_row": row16.float().tolist()}


def hf_top_p(logits, p, min_tokens_to_keep=1):
    """Hugging Face TopPLogitsWarper (artan sıralama) birebir kopyası."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def part_d(results):
    banner("Deney 07-D: top-p uygulaması vs Hugging Face algoritması")
    set_seed(C.SEED)
    mism, trials = 0, 0
    for p in (0.5, 0.8, 0.9, 0.95):
        for _ in range(500):
            z = torch.randn(65) * torch.rand(1) * 4
            a = torch.isinf(top_p_filter(z, p))
            b = torch.isinf(hf_top_p(z, p))
            mism += int(not torch.equal(a, b))
            trials += 1
    print(f"{trials} rastgele dağılımda tutulan küme farkı: {mism}")
    results["part_d"] = {"trials": trials, "mismatches": mism}


if __name__ == "__main__":
    results, timing = {}, []
    part_a(results, timing)
    save_json({"part_a_timing": timing}, "exp07_timing")
    part_b(results)
    part_c(results)
    part_d(results)
    save_json(results, "exp07_library_comparison")
