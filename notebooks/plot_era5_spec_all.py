"""
Gera um plot de espectro direcional por hora para todos os arquivos ERA5.
Saída: output_spectra/YYYY-MM-DD_HHZ.png
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend não-interativo para salvar sem abrir janela
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

# Garante que o pacote geosushi em src/ seja usado
_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))
if os.path.isdir(os.path.join(_src, "geosushi")) and _src not in sys.path:
    sys.path.insert(0, _src)

from geosushi.waves.plotting import plot_directional_spectrum

# ── Configurações ──────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_spectra")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Coleta e ordena os arquivos ────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "era5_*_point.nc")))
if not files:
    raise FileNotFoundError(f"Nenhum arquivo era5_*_point.nc encontrado em {DATA_DIR}")

print(f"Encontrados {len(files)} arquivos. Gerando {len(files) * 24} plots...\n")

total = 0
for filepath in files:
    ds = xr.open_dataset(filepath)
    freq = ds["frequency"].values
    dirs = ds["direction"].values
    n_times = ds.dims["time"]

    for t_idx in range(0, n_times, 3):
        espec = ds["EFTH"].isel(time=t_idx)
        timestamp = pd.Timestamp(ds["time"].values[t_idx])
        time_str  = timestamp.strftime("%Y-%m-%d_%HZ")

        out_path = os.path.join(OUTPUT_DIR, f"{time_str}.png")
        if os.path.exists(out_path):
            print(f"  [skip] {time_str}")
            total += 1
            continue

        fig, ax = plot_directional_spectrum(
            espec,
            freq,
            dirs,
            selected_time=timestamp,
            vmin=0,
            vmax=1.5,
        )

        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        total += 1
        print(f"  [{total:04d}] {time_str} → {out_path}")

    ds.close()

print(f"\nConcluído! {total} plots salvos em: {OUTPUT_DIR}")
