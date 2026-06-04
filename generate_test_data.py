"""
Generuje 6 syntetycznych plików CSV w formacie Movebank do testowania filtrów.

Geofence: (11.165-11.175°N, 15.080-15.090°E)
Trajektoria bazowa: sinusoida w środku ogrodzenia, środek (11.170, 15.085)

Użycie:
    python generate_test_data.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

LAT_CENTER = 11.170
LON_CENTER = 15.085
LAT_AMP    = 0.003   # ≈ 330 m
LON_AMP    = 0.002   # ≈ 200 m

LAT_MIN = 11.165
LAT_MAX = 11.175

N = 480              # ~8 h, 1 próbka/min
T_START = datetime(2024, 1, 15, 6, 0, 0)

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def _timestamps(n: int) -> list[str]:
    return [(T_START + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S.000")
            for i in range(n)]


def _base_trajectory(n: int = N):
    t = np.linspace(0, 4 * np.pi, n)
    lat = LAT_CENTER + LAT_AMP * np.sin(t)
    lon = LON_CENTER + LON_AMP * np.sin(2 * t + 0.5)
    speed = 0.3 + 0.4 * np.abs(np.cos(t))
    return lat.copy(), lon.copy(), speed.copy()


def _add_spike_noise(lat, lon, spike_rate=0.05, amplitude=0.002):
    lat, lon = lat.copy(), lon.copy()
    n = len(lat)
    mask = RNG.random(n) < spike_rate
    lat[mask] += RNG.uniform(-amplitude, amplitude, mask.sum())
    lon[mask] += RNG.uniform(-amplitude, amplitude, mask.sum())
    return lat, lon


def _add_gaussian_noise(lat, lon, sigma=0.0002):
    lat = lat + RNG.normal(0, sigma, len(lat))
    lon = lon + RNG.normal(0, sigma, len(lon))
    return lat, lon


def _add_drift_noise(lat, lon, drift_scale=0.000035):
    n = len(lat)
    drift_lat = np.cumsum(RNG.normal(0, drift_scale, n))
    drift_lon = np.cumsum(RNG.normal(0, drift_scale, n))
    # Detrend żeby dryft wracał do zera na końcu
    drift_lat -= np.linspace(drift_lat[0], drift_lat[-1], n)
    drift_lon -= np.linspace(drift_lon[0], drift_lon[-1], n)
    return lat + drift_lat, lon + drift_lon


def _save(lat, lon, speed, filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame({
        "timestamp":      _timestamps(len(lat)),
        "location-lat":   np.round(lat, 6),
        "location-long":  np.round(lon, 6),
        "ground-speed":   np.round(speed, 3),
    })
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Zapisano: {path}  ({len(df)} wierszy)")


def gen_clean():
    lat, lon, spd = _base_trajectory()
    _save(lat, lon, spd, "test_ideal_clean.csv")


def gen_spike():
    lat, lon, spd = _base_trajectory()
    lat, lon = _add_spike_noise(lat, lon, spike_rate=0.05, amplitude=0.002)
    _save(lat, lon, spd, "test_spike_noise.csv")


def gen_gaussian():
    lat, lon, spd = _base_trajectory()
    lat, lon = _add_gaussian_noise(lat, lon, sigma=0.0002)
    _save(lat, lon, spd, "test_gaussian_noise.csv")


def gen_drift():
    lat, lon, spd = _base_trajectory()
    lat, lon = _add_drift_noise(lat, lon, drift_scale=0.000035)
    _save(lat, lon, spd, "test_drift_noise.csv")


def gen_combined():
    lat, lon, spd = _base_trajectory()
    lat, lon = _add_gaussian_noise(lat, lon, sigma=0.00015)
    lat, lon = _add_spike_noise(lat, lon, spike_rate=0.03, amplitude=0.0015)
    lat, lon = _add_drift_noise(lat, lon, drift_scale=0.000020)
    _save(lat, lon, spd, "test_combined_noise.csv")


def gen_breach():
    n = N
    t = np.linspace(0, 4 * np.pi, n)

    lat = LAT_CENTER + LAT_AMP * np.sin(t)
    lon = LON_CENTER + LON_AMP * np.sin(2 * t + 0.5)
    spd = 0.3 + 0.4 * np.abs(np.cos(t))

    # 3 deterministyczne wyjścia poza dolną granicę (lat < 11.165)
    # Każde trwa ~20 próbek
    breach_centers = [80, 200, 360]
    for bc in breach_centers:
        idx = np.arange(max(0, bc - 10), min(n, bc + 10))
        depth = np.sin(np.linspace(0, np.pi, len(idx))) * 0.006  # max 0.006° poniżej granicy
        lat[idx] = LAT_MIN - depth

    # Gaussian noise wzdłuż całej trajektorii (szczególnie widoczny na granicy)
    lat, lon = _add_gaussian_noise(lat, lon, sigma=0.00025)

    _save(lat, lon, spd, "test_breach_events.csv")


if __name__ == "__main__":
    print("Generowanie testowych danych CSV...")
    gen_clean()
    gen_spike()
    gen_gaussian()
    gen_drift()
    gen_combined()
    gen_breach()
    print("Gotowe! 6 plików zapisanych w data/")
