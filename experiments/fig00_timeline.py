"""Şekil 00 - Dört izli zaman çizelgesi (rapor Böl. II).

Olaylar kaynakca/EVRIM_MATRISI.md'deki dört izden, dönemler sunumun 8.
slaytından alınmıştır. Her olayın yanındaki anahtar refs.bib'deki doğrulanmış
kaynağın anahtarıdır; şekil yalnızca raporda atıf alan kaynakları gösterir.
Veri ya da model dosyası gerektirmez.

Yatay eksen dönem başına eşit genişliktedir (1948-2000 sıkıştırılır,
2017 sonrası açılır); dönem içinde yıllar doğrusal yerleşir. Olayların
yılları alttaki eksenden okunur; tam yıllar rapor metnindedir.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import COLORS, banner, save_fig, setup_matplotlib  # noqa: E402

REFS = Path(__file__).resolve().parents[2] / "rapor" / "refs.bib"

# (ad, yıl aralığı, eksendeki başlangıç ve bitiş yılı)
ERAS = [("Saymak", "1948–2000", 1945.0, 2001.0),
        ("Benzerlik", "2003–2016", 2001.0, 2016.5),
        ("Dikkat", "2017–2019", 2016.5, 2019.5),
        ("Ölçek", "2020–2025", 2019.5, 2026.0)]

LANES = [("Model ailesi", COLORS["blue"]),
         ("Hedef ve maske", COLORS["orange"]),
         ("Örnekleme", COLORS["aqua"]),
         ("Tahmin birimi", COLORS["violet"])]

# (iz, yıl, etiket, refs.bib anahtarları, rakip mi)
EVENTS = [
    (0, 1948, "Shannon", ["shannon1948"], False),
    (0, 1995, "Kneser–Ney", ["kneser1995"], False),
    (0, 2003, "nöral LM", ["bengio2003nplm"], False),
    (0, 2010, "RNN LM", ["mikolov2010rnnlm"], False),
    (0, 2017, "Transformer", ["vaswani2017attention"], False),
    (0, 2018, "GPT", ["radford2018gpt"], False),
    (0, 2019, "BERT", ["devlin2019bert"], True),
    (0, 2020, "GPT-3", ["brown2020gpt3"], False),
    (0, 2022, "Chinchilla", ["hoffmann2022chinchilla"], False),
    (1, 1989, "teacher forcing", ["williams1989learning"], False),
    (1, 2015, "MADE", ["germain2015made"], False),
    (1, 2016, "PixelCNN", ["oord2016pixelrnn"], False),
    (1, 2017, "−∞ maske", ["vaswani2017attention"], False),
    (1, 2022, "FlashAttention", ["dao2022flashattention"], False),
    (1, 2024, "çoklu\nbelirteç", ["gloeckle2024multitoken"], False),
    (2, 1985, "Boltzmann T", ["ackley1985boltzmann"], False),
    (2, 2015, "damıtma T", ["hinton2015distilling"], False),
    (2, 2018, "top-k", ["fan2018hierarchical"], False),
    (2, 2020, "top-p", ["holtzman2020nucleus"], False),
    (2, 2023, "typical", ["meister2023typical"], False),
    (2, 2025, "min-p", ["nguyen2025minp"], False),
    (3, 1951, "harf", ["shannon1951"], False),
    (3, 2011, "karakter", ["sutskever2011text"], False),
    (3, 2016, "alt-sözcük", ["sennrich2016bpe"], False),
    (3, 2019, "bayt BPE", ["radford2019gpt2"], False),
    (3, 2025, "bayt · vektör",
     ["hayase2025bytesampler", "shao2025calm"], False),
]

# Varsayılan: önce çizginin üstü, sonra altı. İki satırlık etiket
# komşu ize taşmasın diye FlashAttention alta alınır.
TIER_ORDER = {"FlashAttention": (-1, +1)}

W, H = 3.45, 1.6             # inç; IEEE tek sütun genişliği
X0, X1 = 0.05, W - 0.03      # çizim alanının yatay sınırları
HEAD = 0.27                  # dönem başlıkları için üst şerit
FOOT = 0.13                  # yıl ekseni için alt şerit
TICKS = [1950, 1970, 1990, 2005, 2010, 2015, 2017, 2018, 2019, 2020, 2022, 2024]
LABEL_PT, GAP = 5.6, 0.035   # etiket yazı boyu ve çizgiye uzaklığı (inç)


def year_to_x(year: float) -> float:
    """Dönem başına eşit genişlik; dönem içinde doğrusal."""
    width = (X1 - X0) / len(ERAS)
    for k, (_, _, lo, hi) in enumerate(ERAS):
        if lo <= year < hi:
            return X0 + width * (k + (year - lo) / (hi - lo))
    raise ValueError(year)


def check_keys() -> None:
    """Şekildeki her anahtar refs.bib'de olmalı (doğrulanmamış kaynak yok).

    refs.bib rapor klasöründedir ve depoda yoktur; bulunamazsa denetim atlanır.
    """
    if not REFS.exists():
        print("  refs.bib bulunamadı; kaynak anahtarı denetimi atlandı")
        return
    keys = set(re.findall(r"@\w+\{([^,]+),", REFS.read_text(encoding="utf-8")))
    missing = sorted({k for e in EVENTS for k in e[3]} - keys)
    if missing:
        raise SystemExit(f"refs.bib'de olmayan anahtarlar: {missing}")


def overlaps(a, b, pad=0.012) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0]
                or a[3] <= b[1] or b[3] <= a[1])


def main() -> None:
    banner("Şekil 00: dört izli zaman çizelgesi")
    check_keys()
    setup_matplotlib()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.canvas.draw()
    inv = ax.transData.inverted()

    def bbox(t):
        e = t.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform([[e.x0, e.y0], [e.x1, e.y1]])
        return (x0, y0, x1, y1)

    # Dönem şeritleri ve başlıkları
    width = (X1 - X0) / len(ERAS)
    for k, (name, span, _, _) in enumerate(ERAS):
        x = X0 + k * width
        if k % 2 == 1:
            ax.add_patch(Rectangle((x, FOOT - 0.02), width, H - HEAD - FOOT + 0.04,
                                   color="#f4f3f0", lw=0, zorder=0))
        ax.text(x + width / 2, H - 0.085, name, ha="center", va="center",
                fontsize=6.3, fontweight="bold", color=COLORS["ink"])
        ax.text(x + width / 2, H - 0.19, span, ha="center", va="center",
                fontsize=5.2, color=COLORS["ink2"])

    # Yıl ekseni: dönem içinde doğrusal, dönemler eşit genişlikte
    for yr in TICKS:
        x = year_to_x(yr)
        ax.plot([x, x], [FOOT - 0.01, H - HEAD + 0.01], color=COLORS["grid"],
                lw=0.4, ls=(0, (1, 1.5)), zorder=0.5)
        ax.text(x, FOOT - 0.04, str(yr), ha="center", va="top", fontsize=4.8,
                color=COLORS["ink2"])

    lane_h = (H - HEAD - FOOT) / len(LANES)
    placed: dict[tuple[int, int], list] = {}
    for i, (title, color) in enumerate(LANES):
        y = H - HEAD - lane_h * (i + 0.5)
        ax.plot([X0, X1], [y, y], color=COLORS["grid"], lw=0.9, zorder=1)
        # İz adı: sol alt köşe, çizginin altı
        t = ax.text(X0 + 0.005, y - GAP, title, ha="left", va="top",
                    fontsize=5.6, fontweight="bold", color=COLORS["ink2"])
        placed[(i, -1)] = [bbox(t)]
        placed[(i, +1)] = []

    for lane, year, label, _, rival in sorted(EVENTS, key=lambda e: (e[0], e[1])):
        color = LANES[lane][1]
        y = H - HEAD - lane_h * (lane + 0.5)
        x = year_to_x(year)
        ax.scatter([x], [y], s=11, zorder=3, linewidths=0.8,
                   facecolors="white" if rival else color, edgecolors=color)
        done = False
        for tier in TIER_ORDER.get(label, (+1, -1)):
            for ha in ("center", "left", "right"):
                dx = {"center": 0, "left": -0.02, "right": 0.02}[ha]
                t = ax.text(x + dx, y + tier * GAP, label, ha=ha,
                            va="bottom" if tier > 0 else "top",
                            fontsize=LABEL_PT, color=COLORS["ink"],
                            style="italic" if rival else "normal", zorder=4)
                b = bbox(t)
                inside = X0 - 0.01 <= b[0] and b[2] <= X1 + 0.02
                if inside and not any(overlaps(b, o)
                                      for o in placed[(lane, tier)]):
                    placed[(lane, tier)].append(b)
                    done = True
                    break
                t.remove()
            if done:
                break
        if not done:
            raise SystemExit(f"Etikete yer bulunamadı: {label}")

    print(f"  {len(EVENTS)} olay, {len(LANES)} iz yerleştirildi")
    save_fig(fig, "fig00_timeline")


if __name__ == "__main__":
    main()
