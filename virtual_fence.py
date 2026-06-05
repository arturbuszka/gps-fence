"""
virtual_fence.py
Automatic GPS trajectory analysis – geofence breach detection
Data: cattle, Far North Region, Cameroon (Movebank, CC0)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

# =============================================================================
# SETTINGS
# =============================================================================

SETTINGS = {
    # --- Input data ---
    "datasets": [
        {
            "name": "animal_A",
            "path": "data/animal_A.csv",
            "col_timestamp": "timestamp",
            "col_lat": "location-lat",
            "col_lon": "location-long",
            "col_speed": "ground-speed",
            "col_accuracy": None,
        },
        {
            "name": "animal_B",
            "path": "data/animal_B.csv",
            "col_timestamp": "timestamp",
            "col_lat": "location-lat",
            "col_lon": "location-long",
            "col_speed": "ground-speed",
            "col_accuracy": None,
        },
    ],

    # --- Geofence – sleeping area/water source (northern camp of both animals) ---
    # Covers the area where cattle spend ~5-7h (peak activity in the north)
    # Vertices in WGS84 (lat, lon), clockwise order
    "geofence_polygon_latlon": [
        (11.165, 15.080),
        (11.165, 15.090),
        (11.175, 15.090),
        (11.175, 15.080),
    ],
    "buffer_warning_m": 200.0,  # warning buffer – 200 m from boundary

    # --- Filter parameters ---
    "median_kernel_size": 5,    # must be odd
    "savgol_window": 11,        # must be odd and > polyorder
    "savgol_polyorder": 3,
    "use_kalman": True,
    "kalman_process_noise": 1e-4,
    "kalman_measurement_noise": 1e-2,

    # --- Stability analysis (SG window sweep) ---
    "stability_savgol_windows": [5, 7, 11, 15, 21],

    # --- Output ---
    "output_dir": "wyniki",
    "plot_dpi": 150,
}

# =============================================================================
# 1. DATA LOADING AND VALIDATION
# =============================================================================

def load_and_validate(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["path"])

    rename = {
        cfg["col_timestamp"]: "timestamp",
        cfg["col_lat"]: "lat",
        cfg["col_lon"]: "lon",
    }
    if cfg["col_speed"]:
        rename[cfg["col_speed"]] = "speed"
    df = df.rename(columns=rename)

    required = ["timestamp", "lat", "lon"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "lat", "lon"])
    if len(df) < n_before:
        print(f"  Removed {n_before - len(df)} duplicates")

    nan_mask = df["lat"].isna() | df["lon"].isna()
    if nan_mask.any():
        print(f"  Removed {nan_mask.sum()} rows with NaN in lat/lon")
        df = df[~nan_mask].reset_index(drop=True)

    invalid = (df["lat"].abs() > 90) | (df["lon"].abs() > 180)
    if invalid.any():
        print(f"  Removed {invalid.sum()} rows with invalid coordinates")
        df = df[~invalid].reset_index(drop=True)

    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().astype(float)

    gaps = df["elapsed_s"].diff()
    large_gaps = gaps[gaps > 3600]
    if len(large_gaps) > 0:
        print(f"  Warning: {len(large_gaps)} gaps >1h in data (max {large_gaps.max()/3600:.1f}h)")

    return df[["timestamp", "elapsed_s", "lat", "lon"] + (["speed"] if "speed" in df.columns else [])].copy()


def latlon_to_utm(lat: np.ndarray, lon: np.ndarray):
    median_lon = float(np.median(lon))
    median_lat = float(np.median(lat))
    zone = int((median_lon + 180) / 6) + 1
    epsg = 32600 + zone if median_lat >= 0 else 32700 + zone
    crs = f"EPSG:{epsg}"
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return np.array(easting), np.array(northing), crs, transformer

# =============================================================================
# 2. PREPROCESSING – SIGNAL FILTERING
# =============================================================================

def apply_median_filter(signal: np.ndarray, kernel_size: int) -> np.ndarray:
    return medfilt(signal, kernel_size)


def apply_savgol_filter(signal: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    window = min(window, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    return savgol_filter(signal, window, polyorder)


def apply_kalman_1d(signal: np.ndarray, process_noise: float, measurement_noise: float) -> np.ndarray:
    """Manual Kalman filter implementation for position+velocity model."""
    n = len(signal)
    dt = 1.0

    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = process_noise * np.eye(2)
    R = np.array([[measurement_noise]])

    x = np.array([signal[0], 0.0])
    P = np.eye(2)
    result = np.zeros(n)

    for i in range(n):
        # Prediction
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # Update
        z = np.array([[signal[i]]])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ (z - H @ x_pred).flatten()
        P = (np.eye(2) - K @ H) @ P_pred
        result[i] = x[0]

    return result


def preprocess_trajectory(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    df = df.copy()
    ks = settings["median_kernel_size"]
    sw = settings["savgol_window"]
    sp = settings["savgol_polyorder"]

    lat_med = apply_median_filter(df["lat"].values, ks)
    lon_med = apply_median_filter(df["lon"].values, ks)

    lat_sg = apply_savgol_filter(lat_med, sw, sp)
    lon_sg = apply_savgol_filter(lon_med, sw, sp)

    if settings["use_kalman"]:
        lat_filt = apply_kalman_1d(lat_sg, settings["kalman_process_noise"], settings["kalman_measurement_noise"])
        lon_filt = apply_kalman_1d(lon_sg, settings["kalman_process_noise"], settings["kalman_measurement_noise"])
    else:
        lat_filt = lat_sg
        lon_filt = lon_sg

    df["lat_filt"] = lat_filt
    df["lon_filt"] = lon_filt
    return df

# =============================================================================
# 3. GEOFENCE AND BREACH DETECTION
# =============================================================================

def build_geofence_utm(polygon_latlon: list, transformer) -> np.ndarray:
    lats = [p[0] for p in polygon_latlon]
    lons = [p[1] for p in polygon_latlon]
    east, north = transformer.transform(lons, lats)
    polygon = np.column_stack([east, north])
    # close the polygon
    polygon = np.vstack([polygon, polygon[0]])
    return polygon


def point_in_polygon_raycasting(px: float, py: float, polygon: np.ndarray) -> bool:
    """Ray casting – counts intersections of a horizontal ray with polygon edges."""
    n = len(polygon) - 1
    inside = False
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[i + 1]
        # check if edge crosses the horizontal line y=py
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
    return inside


def shrink_polygon(polygon: np.ndarray, buffer_m: float) -> np.ndarray:
    """Shifts vertices toward centroid by buffer_m (approximation for convex polygons)."""
    vertices = polygon[:-1]  # exclude closing point
    centroid = vertices.mean(axis=0)
    result = []
    for v in vertices:
        direction = centroid - v
        dist = np.linalg.norm(direction)
        if dist > 0:
            new_v = v + direction / dist * min(buffer_m, dist * 0.9)
        else:
            new_v = v
        result.append(new_v)
    result = np.array(result)
    return np.vstack([result, result[0]])


def compute_breach_events(easting: np.ndarray, northing: np.ndarray,
                           elapsed_s: np.ndarray, polygon: np.ndarray,
                           buffer_m: float) -> dict:
    n = len(easting)
    inside_mask = np.array([
        point_in_polygon_raycasting(easting[i], northing[i], polygon)
        for i in range(n)
    ])

    inner_polygon = shrink_polygon(polygon, buffer_m)
    warning_mask = np.array([
        inside_mask[i] and not point_in_polygon_raycasting(easting[i], northing[i], inner_polygon)
        for i in range(n)
    ])

    # Detect entry/exit events
    breach_events = []
    entries = []
    exits = []

    in_breach = False
    breach_start_idx = None

    for i in range(n):
        if not in_breach and not inside_mask[i]:
            in_breach = True
            breach_start_idx = i
            if i > 0:
                exits.append(i - 1)
        elif in_breach and inside_mask[i]:
            in_breach = False
            breach_events.append({
                "start_idx": breach_start_idx,
                "end_idx": i - 1,
                "start_t": elapsed_s[breach_start_idx],
                "end_t": elapsed_s[i - 1],
                "duration_s": elapsed_s[i - 1] - elapsed_s[breach_start_idx],
            })
            entries.append(i)

    if in_breach:
        breach_events.append({
            "start_idx": breach_start_idx,
            "end_idx": n - 1,
            "start_t": elapsed_s[breach_start_idx],
            "end_t": elapsed_s[n - 1],
            "duration_s": elapsed_s[n - 1] - elapsed_s[breach_start_idx],
        })

    return {
        "inside_mask": inside_mask,
        "warning_mask": warning_mask,
        "breach_events": breach_events,
        "entries": entries,
        "exits": exits,
    }

# =============================================================================
# 4. SIGNAL METRICS
# =============================================================================

def compute_haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Haversine formula: distance between GPS points in meters."""
    R = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def compute_total_distance(lat: np.ndarray, lon: np.ndarray) -> float:
    """Total distance as sum of Haversine segments (trapezoidal integration)."""
    dists = compute_haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(np.sum(dists))


def compute_instantaneous_speed(lat: np.ndarray, lon: np.ndarray,
                                 elapsed_s: np.ndarray) -> np.ndarray:
    dists = compute_haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt = np.diff(elapsed_s)
    dt = np.where(dt == 0, np.nan, dt)
    speeds = dists / dt
    # clip unrealistic values (>10 m/s for cattle)
    speeds = np.where(speeds > 10.0, np.nan, speeds)
    return np.append(speeds, speeds[-1] if not np.isnan(speeds[-1]) else np.nanmedian(speeds))


def compute_convex_hull_area(easting: np.ndarray, northing: np.ndarray) -> float:
    points = np.column_stack([easting, northing])
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # in 2D: volume = area
    except Exception:
        return 0.0


def compute_kde_density(easting: np.ndarray, northing: np.ndarray, grid_size: int = 100):
    e_min, e_max = easting.min(), easting.max()
    n_min, n_max = northing.min(), northing.max()
    margin_e = (e_max - e_min) * 0.05
    margin_n = (n_max - n_min) * 0.05
    ee = np.linspace(e_min - margin_e, e_max + margin_e, grid_size)
    nn = np.linspace(n_min - margin_n, n_max + margin_n, grid_size)
    xx, yy = np.meshgrid(ee, nn)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([easting, northing])
    # fallback to hist2d for large datasets
    if len(easting) > 10_000:
        zz, xe, ye = np.histogram2d(easting, northing, bins=grid_size,
                                     range=[[e_min - margin_e, e_max + margin_e],
                                            [n_min - margin_n, n_max + margin_n]])
        xx_h = (xe[:-1] + xe[1:]) / 2
        yy_h = (ye[:-1] + ye[1:]) / 2
        xx2, yy2 = np.meshgrid(xx_h, yy_h)
        return xx2, yy2, zz.T
    else:
        kernel = gaussian_kde(values)
        zz = kernel(positions).reshape(xx.shape)
        return xx, yy, zz


def compute_all_metrics(df: pd.DataFrame, breach_results: dict,
                         easting: np.ndarray, northing: np.ndarray) -> dict:
    speed = compute_instantaneous_speed(df["lat_filt"].values, df["lon_filt"].values, df["elapsed_s"].values)
    inside = breach_results["inside_mask"]
    elapsed = df["elapsed_s"].values

    dt = np.diff(elapsed)
    time_inside = float(np.sum(dt[inside[:-1]]))
    time_outside = float(np.sum(dt[~inside[:-1]]))

    breach_events = breach_results["breach_events"]
    total_breach_dur = sum(e["duration_s"] for e in breach_events)

    return {
        "total_distance_m": compute_total_distance(df["lat_filt"].values, df["lon_filt"].values),
        "time_inside_s": time_inside,
        "time_outside_s": time_outside,
        "breach_count": len(breach_events),
        "total_breach_duration_s": total_breach_dur,
        "mean_speed_ms": float(np.nanmean(speed)),
        "max_speed_ms": float(np.nanmax(speed)),
        "convex_hull_area_m2": compute_convex_hull_area(easting, northing),
        "activity_index": (
            compute_total_distance(df["lat_filt"].values, df["lon_filt"].values)
            / max(compute_convex_hull_area(easting, northing), 1.0)
        ),
    }

# =============================================================================
# 5. STABILITY ANALYSIS
# =============================================================================

def run_stability_analysis(df: pd.DataFrame, polygon_utm: np.ndarray,
                            transformer, settings: dict, out_dir: str) -> pd.DataFrame:
    rows = []
    base_settings = dict(settings)
    ks = settings["median_kernel_size"]

    for window in settings["stability_savgol_windows"]:
        lat_med = apply_median_filter(df["lat"].values, ks)
        lon_med = apply_median_filter(df["lon"].values, ks)
        lat_f = apply_savgol_filter(lat_med, window, settings["savgol_polyorder"])
        lon_f = apply_savgol_filter(lon_med, window, settings["savgol_polyorder"])

        east_f, north_f, _, _ = latlon_to_utm(lat_f, lon_f)
        br = compute_breach_events(east_f, north_f, df["elapsed_s"].values,
                                   polygon_utm, settings["buffer_warning_m"])
        dist = compute_total_distance(lat_f, lon_f)
        dt = np.diff(df["elapsed_s"].values)
        time_in = float(np.sum(dt[br["inside_mask"][:-1]]))

        rows.append({
            "savgol_window": window,
            "breach_count": len(br["breach_events"]),
            "total_distance_m": round(dist, 1),
            "time_inside_s": round(time_in, 1),
        })

    result = pd.DataFrame(rows)
    median_bc = result["breach_count"].median()
    result["deviation"] = (result["breach_count"] - median_bc).abs()
    result.to_csv(os.path.join(out_dir, "stability_table.csv"), index=False)
    return result

# =============================================================================
# 6. VISUALIZATIONS
# =============================================================================

def _draw_polygon(ax, polygon_utm: np.ndarray, color="black", linestyle="-",
                  linewidth=2, label=None, offset_e=0, offset_n=0):
    ax.plot(polygon_utm[:, 0] - offset_e, polygon_utm[:, 1] - offset_n,
            color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def plot_trajectory_map(easting_raw, northing_raw, easting_filt, northing_filt,
                         polygon_utm, inside_mask, output_path, settings):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Raw GPS", "Filtered trajectory"]
    data_pairs = [
        (easting_raw, northing_raw),
        (easting_filt, northing_filt),
    ]

    off_e = easting_filt.mean()
    off_n = northing_filt.mean()

    for ax, title, (e, n) in zip(axes, titles, data_pairs):
        ax.scatter(e[inside_mask] - off_e, n[inside_mask] - off_n,
                   s=2, c="green", alpha=0.4, label="Inside zone")
        ax.scatter(e[~inside_mask] - off_e, n[~inside_mask] - off_n,
                   s=2, c="red", alpha=0.4, label="Outside zone")
        _draw_polygon(ax, polygon_utm, color="black", linewidth=2,
                      label="Geofence", offset_e=off_e, offset_n=off_n)
        ax.set_title(title)
        ax.set_xlabel("Easting – center [m]")
        ax.set_ylabel("Northing – center [m]")
        ax.legend(markerscale=4, fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=settings["plot_dpi"])
    plt.close()


def plot_lat_lon_signals(df: pd.DataFrame, output_path: str, settings: dict):
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    t = df["elapsed_s"].values / 3600

    signals = [
        ("lat", df["lat"].values, df["lat_filt"].values, "Latitude [°]"),
        ("lon", df["lon"].values, df["lon_filt"].values, "Longitude [°]"),
    ]

    for col, (label, raw, filt, ylabel) in enumerate(signals):
        # row 0: raw
        axes[0, col].plot(t, raw, color="tomato", linewidth=0.9)
        axes[0, col].set_title(f"{label} – raw GPS")
        axes[0, col].set_ylabel(ylabel)
        axes[0, col].set_xlabel("Time [h]")
        axes[0, col].grid(True, alpha=0.3)

        # row 1: filtered
        axes[1, col].plot(t, filt, color="steelblue", linewidth=0.9)
        axes[1, col].set_title(f"{label} – filtered (median + SG + Kalman)")
        axes[1, col].set_ylabel(ylabel)
        axes[1, col].set_xlabel("Time [h]")
        axes[1, col].grid(True, alpha=0.3)

        # row 2: residual = removed noise
        res = raw - filt
        std = res.std()
        axes[2, col].plot(t, res, color="darkorange", linewidth=0.7, alpha=0.9,
                          label=f"GPS noise")
        axes[2, col].axhline(0,    color="gray",      linewidth=0.9, linestyle="--")
        axes[2, col].axhline( std, color="steelblue", linewidth=0.9, linestyle=":",
                              label=f"+1σ = {std:.2e}°  (~{std*111000:.1f} m)")
        axes[2, col].axhline(-std, color="steelblue", linewidth=0.9, linestyle=":")
        axes[2, col].set_title(f"Residual {label}  =  raw − filtered  (removed noise)")
        axes[2, col].set_ylabel(f"Δ{label} [°]")
        axes[2, col].set_xlabel("Time [h]")
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

    plt.suptitle("Lat/lon signals: raw / filtered / residual (GPS noise)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=settings["plot_dpi"])
    plt.close()


def plot_speed(elapsed_s: np.ndarray, speed: np.ndarray,
               breach_results: dict, output_path: str, settings: dict):
    t = elapsed_s / 3600
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t, speed, color="steelblue", linewidth=0.8, label="Instantaneous speed")
    mean_speed = float(np.nanmean(speed))
    ax.axhline(mean_speed, color="orange", linestyle="--", linewidth=1,
               label=f"Mean: {mean_speed:.3f} m/s")

    for ev in breach_results["breach_events"]:
        ax.axvspan(ev["start_t"] / 3600, ev["end_t"] / 3600,
                   alpha=0.15, color="red")

    for idx in breach_results["exits"]:
        ax.axvline(elapsed_s[idx] / 3600, color="red", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Instantaneous speed with geofence breach events")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=settings["plot_dpi"])
    plt.close()


def plot_breach_timeline(elapsed_s: np.ndarray, inside_mask: np.ndarray,
                          breach_results: dict, output_path: str, settings: dict):
    t = elapsed_s / 3600
    t_end = t[-1]

    fig, ax = plt.subplots(figsize=(12, 3))

    # green background when inside, red when outside – start with red background
    ax.axhspan(0, 1, color="red", alpha=0.15, label="Outside zone")

    # overwrite with green for each period inside the zone
    in_start = None
    for i in range(len(inside_mask)):
        if inside_mask[i] and in_start is None:
            in_start = t[i]
        elif not inside_mask[i] and in_start is not None:
            ax.axvspan(in_start, t[i], color="green", alpha=0.25, label="_nolegend_")
            in_start = None
    if in_start is not None:
        ax.axvspan(in_start, t_end, color="green", alpha=0.25, label="_nolegend_")

    # blue signal line: 1 = inside zone, 0 = outside zone
    signal = inside_mask.astype(float)
    ax.step(t, signal, where="post", color="steelblue", linewidth=1.5, label="Animal state")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0 – outside zone", "1 – inside zone"])
    ax.set_ylim(-0.1, 1.3)
    ax.set_xlabel("Time [h]")
    ax.set_title(f"Timeline – geofence state  (breaches: {len(breach_results['breach_events'])})"
                 f"   |   green = inside,  red = outside")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", alpha=0.4, label="Inside zone"),
        Patch(facecolor="red",   alpha=0.3, label="Outside zone"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=settings["plot_dpi"])
    plt.close()


def plot_heatmap(easting: np.ndarray, northing: np.ndarray,
                 polygon_utm: np.ndarray, output_path: str, settings: dict):
    xx, yy, zz = compute_kde_density(easting, northing)
    off_e = easting.mean()
    off_n = northing.mean()

    # axis range strictly to data + 5% margin
    margin = 0.05
    e_range = easting.max() - easting.min()
    n_range = northing.max() - northing.min()
    xlim = (easting.min() - margin * e_range - off_e,
            easting.max() + margin * e_range - off_e)
    ylim = (northing.min() - margin * n_range - off_n,
            northing.max() + margin * n_range - off_n)

    fig, ax = plt.subplots(figsize=(8, 8))
    cf = ax.contourf(xx - off_e, yy - off_n, zz, levels=20, cmap="YlOrRd")
    plt.colorbar(cf, ax=ax, label="Presence density")
    _draw_polygon(ax, polygon_utm, color="white", linewidth=2,
                  label="Geofence", offset_e=off_e, offset_n=off_n)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Easting – center [m]")
    ax.set_ylabel("Northing – center [m]")
    ax.set_title("Animal presence heatmap (KDE)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=settings["plot_dpi"])
    plt.close()

# =============================================================================
# 7. SAVING RESULTS
# =============================================================================

def save_metrics_csv(metrics: dict, output_path: str):
    pd.DataFrame([metrics]).to_csv(output_path, index=False)


def save_comparison_csv(all_metrics: list, output_path: str):
    pd.DataFrame(all_metrics).to_csv(output_path, index=False)

# =============================================================================
# 8. ORCHESTRATION
# =============================================================================

def process_single_dataset(cfg: dict, settings: dict) -> dict:
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    out_dir = os.path.join(settings["output_dir"], name)
    os.makedirs(out_dir, exist_ok=True)

    # Load and validate
    print("  [1/6] Loading and validating data...")
    df = load_and_validate(cfg)
    print(f"  Rows: {len(df)}, time span: {df['elapsed_s'].max()/3600:.1f}h")

    # Convert to UTM (raw)
    east_raw, north_raw, crs, transformer = latlon_to_utm(df["lat"].values, df["lon"].values)

    # Filter
    print("  [2/6] Signal filtering (median -> SG -> Kalman)...")
    df = preprocess_trajectory(df, settings)

    # Convert filtered to UTM
    east_filt, north_filt, _, _ = latlon_to_utm(df["lat_filt"].values, df["lon_filt"].values)

    # Geofence
    print("  [3/6] Building geofence and detecting breaches...")
    polygon_utm = build_geofence_utm(settings["geofence_polygon_latlon"], transformer)
    breach_results = compute_breach_events(
        east_filt, north_filt, df["elapsed_s"].values,
        polygon_utm, settings["buffer_warning_m"]
    )
    print(f"  Breaches: {len(breach_results['breach_events'])}")
    pct_inside = breach_results["inside_mask"].mean() * 100
    print(f"  Time inside zone: {pct_inside:.1f}%")

    # Metrics
    print("  [4/6] Computing metrics...")
    speed = compute_instantaneous_speed(df["lat_filt"].values, df["lon_filt"].values, df["elapsed_s"].values)
    metrics = compute_all_metrics(df, breach_results, east_filt, north_filt)
    metrics["dataset_name"] = name
    print(f"  Total distance: {metrics['total_distance_m']/1000:.2f} km")
    print(f"  Convex hull area: {metrics['convex_hull_area_m2']/1e6:.2f} km²")
    print(f"  Activity index: {metrics['activity_index']:.4f} 1/m")

    # Stability analysis
    print("  [5/6] Filter stability analysis...")
    stab = run_stability_analysis(df, polygon_utm, transformer, settings, out_dir)
    print(f"  Stability table saved: {os.path.join(out_dir, 'stability_table.csv')}")

    # Plots
    print("  [6/6] Generating plots...")
    plot_trajectory_map(
        east_raw, north_raw, east_filt, north_filt,
        polygon_utm, breach_results["inside_mask"],
        os.path.join(out_dir, "trajectory_map.png"), settings
    )
    plot_lat_lon_signals(df, os.path.join(out_dir, "lat_lon_signals.png"), settings)
    plot_speed(df["elapsed_s"].values, speed, breach_results,
               os.path.join(out_dir, "speed_plot.png"), settings)
    plot_breach_timeline(df["elapsed_s"].values, breach_results["inside_mask"],
                         breach_results, os.path.join(out_dir, "breach_timeline.png"), settings)
    plot_heatmap(east_filt, north_filt, polygon_utm,
                 os.path.join(out_dir, "heatmap.png"), settings)

    save_metrics_csv(metrics, os.path.join(out_dir, "metrics.csv"))
    print(f"  Results saved in: {out_dir}/")
    return metrics


def main():
    print("\nVirtual Fence – GPS Trajectory Analysis for Cattle")
    print("Data: Daily grazing movements, Far North Region, Cameroon (Movebank CC0)\n")

    all_metrics = []
    for cfg in SETTINGS["datasets"]:
        metrics = process_single_dataset(cfg, SETTINGS)
        all_metrics.append(metrics)

    comparison_path = os.path.join(SETTINGS["output_dir"], "comparison.csv")
    save_comparison_csv(all_metrics, comparison_path)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for m in all_metrics:
        print(f"\n  {m['dataset_name']}:")
        print(f"    Distance:         {m['total_distance_m']/1000:.2f} km")
        print(f"    Time inside:      {m['time_inside_s']/3600:.2f} h  ({m['time_inside_s']/(m['time_inside_s']+m['time_outside_s'])*100:.1f}%)")
        print(f"    Breaches:         {m['breach_count']}")
        print(f"    Mean speed:       {m['mean_speed_ms']:.4f} m/s")
        print(f"    Exploration area: {m['convex_hull_area_m2']/1e6:.2f} km²")
        print(f"    Activity index:   {m['activity_index']:.4f} 1/m")
    print(f"\n  Comparison saved: {comparison_path}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
