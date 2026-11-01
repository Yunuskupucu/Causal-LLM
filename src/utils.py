"""Yardımcı fonksiyonlar: tohum sabitleme, cihaz seçimi, grafik stili.

Rapor Bölüm IV.B (Yeniden üretilebilirlik): Tüm rastgelelik kaynakları
(`random`, `numpy`, `torch`) tek bir tohumdan beslenir ve PyTorch'un
deterministik algoritma modu açılır.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

# Dataviz paleti: kategorik slotlar sabit sırayla atanır (asla döngüsel değil).
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "gray": "#9a9892",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e5e4e0",
}
SERIES = [COLORS[k] for k in
          ("blue", "orange", "aqua", "yellow", "magenta", "green", "violet",
           "red")]
# Tek-renk (sequential) mavi rampası: ısı haritaları için.
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]
# Iraksak (diverging) mavi-gri-kırmızı: işaretli skor matrisleri için.
DIV_BLUE_RED = ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f4a9a8",
                "#e34948", "#8f1d1c"]


def set_seed(seed: int) -> None:
    """Tüm rastgelelik tohumlarını sabitler (Kılavuz 4.3-1)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def get_device(name: str = "cpu") -> torch.device:
    """Varsayılan CPU: bit-düzeyinde tekrar üretilebilirlik için."""
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_matplotlib() -> None:
    """Rapor kalitesinde (vektörel, ince çizgi, sessiz ızgara) stil."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["gray"],
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """Her şekli hem vektörel PDF hem 300 DPI PNG olarak kaydeder."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  [şekil] figures/{name}.pdf|png")


def save_json(obj: dict, name: str) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with open(RES_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  [sonuç] results/{name}.json")


def load_json(name: str) -> dict:
    with open(RES_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)
