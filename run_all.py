"""Tek komutla tüm deneyleri ve grafikleri üretir:  python run_all.py

Sıra önemlidir: Deney 02 modeli eğitir ve results/model.pt dosyasını yazar;
diğer deneyler bu dosyayı kullanır. Tüm tohumlar sabittir (config.SEED).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORDER = [
    "exp02_train.py",
    "exp01_mask_verification.py",
    "exp03_sampling_analysis.py",
    "exp04_softmax_temperature.py",
    "exp05_chain_rule.py",
    "exp06_generation_quality.py",
    "exp07_library_comparison.py",
]

if __name__ == "__main__":
    only = sys.argv[1:]            # örn. python run_all.py exp03 exp04
    t0 = time.time()
    for script in ORDER:
        if only and not any(script.startswith(o) for o in only):
            continue
        print(f"\n>>> {script}")
        subprocess.run([sys.executable, "-u",
                        str(ROOT / "experiments" / script)], check=True)
    print(f"\nTamamlandı: {time.time() - t0:.0f} s. Şekiller: figures/, "
          f"sayısal sonuçlar: results/")
