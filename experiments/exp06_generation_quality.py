"""Deney 06 - Örnekleme ayarlarının üretilen metne etkisi (nicel).

Rapor Bölüm IV.D: Holtzman vd. (2020) "olabilirlik tuzağı" ve "güvenilmez
kuyruk" gözlemlerinin karakter düzeyinde yeniden üretilmesi.

Her ayar için N örnek üretilir ve ölçülür:
  * öz-NLL   : üretilen metnin modelin kendisi altındaki ortalama NLL'si
  * tekrar   : tekrar eden karakter 4-gramlarının oranı (1 - distinct-4)
  * geçerli  : eğitim külliyatı sözlüğünde bulunan sözcük oranı
  * ayrık-2  : sözcük bigramlarının ayrıklık oranı (çeşitlilik)
Doğrulama metni (insan yazısı) referans noktası olarak aynı ölçütlerle
raporlanır.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C                                              # noqa: E402
from experiments.common import load_everything                  # noqa: E402
from src.utils import (COLORS, RES_DIR, banner, save_fig,       # noqa: E402
                       save_json, set_seed, setup_matplotlib)

CONFIGS = [
    ("greedy (T→0)", dict(temperature=0.0)),
    ("T = 0.2", dict(temperature=0.2)),
    ("T = 0.7", dict(temperature=0.7)),
    ("T = 1.0", dict(temperature=1.0)),
    ("T = 1.5", dict(temperature=1.5)),
    ("top-k = 5", dict(temperature=1.0, top_k=5)),
    ("top-k = 20", dict(temperature=1.0, top_k=20)),
    ("top-p = 0.9", dict(temperature=1.0, top_p=0.9)),
    ("top-p = 0.95", dict(temperature=1.0, top_p=0.95)),
    ("T=0.7 + p=0.9", dict(temperature=0.7, top_p=0.9)),
]
N_SAMPLES, LEN = 10, 400
WORD_RE = re.compile(r"[A-Za-z']+")


def text_metrics(text: str, vocab: set[str]):
    words = WORD_RE.findall(text)
    valid = np.mean([w.lower() in vocab for w in words]) if words else 0.0
    grams4 = [text[i:i + 4] for i in range(len(text) - 3)]
    rep4 = 1 - len(set(grams4)) / max(1, len(grams4))
    bigrams = list(zip(words, words[1:]))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))
    return {"valid_word_ratio": float(valid), "repeat_4gram": float(rep4),
            "distinct_2": float(distinct2), "n_words": len(words)}


@torch.no_grad()
def self_nll(model, tok, text: str) -> float:
    ids = torch.tensor(tok.encode(text))
    T = model.block_size
    total, count = 0.0, 0
    for s in range(0, len(ids) - 1, T):
        x = ids[s:s + T].unsqueeze(0)
        y = ids[s + 1:s + 1 + x.shape[1]].unsqueeze(0)
        x = x[:, :y.shape[1]]
        logits, _ = model(x)
        lp = torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1))
        total += -lp.sum().item()
        count += y.numel()
    return total / count


def main():
    banner("Deney 06 - Örnekleme ayarları ve üretim kalitesi")
    set_seed(C.SEED)
    text, tok, train_ids, val_ids, model = load_everything()
    train_text = tok.decode(train_ids.tolist())
    vocab = {w.lower() for w in WORD_RE.findall(train_text)}
    prompt = torch.tensor([tok.encode("ROMEO:\n")]).repeat(N_SAMPLES, 1)
    results, samples_md = {"configs": []}, ["# Üretilen örnekler\n"]
    for name, kw in CONFIGS:
        g = torch.Generator().manual_seed(C.SEED + 61)
        out = model.generate(prompt, LEN, generator=g, **kw)
        texts = [tok.decode(o[prompt.shape[1]:].tolist()) for o in out]
        ms = [text_metrics(t, vocab) for t in texts]
        nll = float(np.mean([self_nll(model, tok, t) for t in texts]))
        rec = {"name": name, **kw, "self_nll_nat": nll,
               **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}}
        results["configs"].append(rec)
        print(f"{name:<16}| öz-NLL {nll:.3f} | tekrar-4g {rec['repeat_4gram']:.3f}"
              f" | geçerli sözcük {rec['valid_word_ratio']:.3f} | ayrık-2 "
              f"{rec['distinct_2']:.3f}")
        samples_md.append(f"\n## {name}  ({kw})\n\n```\n{texts[0][:300]}\n```\n")
    # İnsan referansı: doğrulama metninden aynı uzunlukta parçalar
    g = torch.Generator().manual_seed(C.SEED + 62)
    starts = torch.randint(0, len(val_ids) - LEN, (N_SAMPLES,), generator=g)
    ref_texts = [tok.decode(val_ids[s:s + LEN].tolist()) for s in starts]
    ms = [text_metrics(t, vocab) for t in ref_texts]
    ref = {"name": "insan (doğrulama)", "self_nll_nat": float(np.mean(
        [self_nll(model, tok, t) for t in ref_texts])),
        **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}}
    results["reference"] = ref
    print(f"{'insan (doğrulama)':<16}| öz-NLL {ref['self_nll_nat']:.3f} | "
          f"tekrar-4g {ref['repeat_4gram']:.3f} | geçerli sözcük "
          f"{ref['valid_word_ratio']:.3f} | ayrık-2 {ref['distinct_2']:.3f}")
    with open(RES_DIR / "exp06_samples.md", "w", encoding="utf-8") as f:
        f.write("\n".join(samples_md))
    save_json(results, "exp06_generation_quality")

    setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 5.0))
    cf = results["configs"]
    xs = [c["self_nll_nat"] for c in cf]
    ys = [c["valid_word_ratio"] for c in cf]
    ax1.scatter(xs, ys, s=22, color=COLORS["blue"], zorder=3)
    for i, (x_, y_) in enumerate(zip(xs, ys), start=1):
        ax1.annotate(str(i), (x_, y_), fontsize=6.5, xytext=(3, 2),
                     textcoords="offset points", color=COLORS["ink2"])
    ax1.scatter([ref["self_nll_nat"]], [ref["valid_word_ratio"]], s=40,
                marker="*", color=COLORS["orange"], zorder=4,
                label="insan (doğrulama)")
    ax1.set_xlabel("Üretilen metnin öz-NLL'si (nat / karakter)")
    ax1.set_ylabel("Geçerli sözcük oranı")
    ax1.set_title("(a) Olabilirlik–tutarlılık dengesi (numaralar: b)",
                  fontsize=8.5)
    ax1.legend(fontsize=7, loc="lower left")
    names = [f"{i}: {c['name']}" for i, c in enumerate(cf, start=1)] + \
            [ref["name"]]
    rep = [c["repeat_4gram"] for c in cf] + [ref["repeat_4gram"]]
    cols = [COLORS["blue"]] * len(cf) + [COLORS["orange"]]
    ax2.barh(range(len(names)), rep, color=cols, height=0.65)
    ax2.set_yticks(range(len(names)), names, fontsize=6.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Tekrarlanan karakter 4-gram oranı")
    ax2.set_title("(b) Tekrar (dejenerasyon)", fontsize=8.5)
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, "fig06_generation_quality")


if __name__ == "__main__":
    main()
