"""
dashboard.py
Interaktywny dashboard do analizy trajektorii GPS – Virtual Fence
"""

import os
import sys
import time
import threading
import warnings
import math

# Ustaw backend PRZED jakimkolwiek importem matplotlib / virtual_fence
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as mpatches
from matplotlib.patches import Patch

# Monkey-patch matplotlib.use żeby virtual_fence nie przestawił backendu na Agg
_orig_mpl_use = matplotlib.use
matplotlib.use = lambda *a, **kw: None

import numpy as np
import pandas as pd

# Teraz bezpiecznie importujemy virtual_fence
sys.path.insert(0, os.path.dirname(__file__))
import virtual_fence as vf

matplotlib.use = _orig_mpl_use  # przywróć oryginał

import customtkinter as ctk

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
ACCENT   = "#1f6aa5"
GREEN    = "#2ecc71"
RED      = "#e74c3c"
ORANGE   = "#e67e22"
MUTED    = "#888888"
BG_CARD  = "#2b2b2b"
FG_CARD  = "#ffffff"

FILTER_NAMES = {
    "median": "Median Filter",
    "savgol": "Savitzky-Golay",
    "kalman": "Kalman Filter",
}
FILTER_COLORS = {
    "median": "#3498db",
    "savgol": "#2ecc71",
    "kalman": "#e74c3c",
}

DATASETS = vf.SETTINGS["datasets"]


# =============================================================================
# Pomocnicze: niestandardowy preprocess respektujący pipeline_order
# =============================================================================

def run_pipeline(df: pd.DataFrame, pipeline_order: list, enabled: dict, params: dict) -> dict:
    """
    Przepuszcza sygnał przez filtry w zadanej kolejności.
    Zwraca dict stage_name -> (lat_array, lon_array) dla każdego etapu.
    """
    lat = df["lat"].values.copy()
    lon = df["lon"].values.copy()
    stages = {"raw": (lat.copy(), lon.copy())}

    for stage in pipeline_order:
        if not enabled.get(stage, True):
            stages[stage] = (lat.copy(), lon.copy())
            continue
        if stage == "median":
            ks = params["median_kernel_size"]
            lat = vf.apply_median_filter(lat, ks)
            lon = vf.apply_median_filter(lon, ks)
        elif stage == "savgol":
            sw = params["savgol_window"]
            sp = params["savgol_polyorder"]
            lat = vf.apply_savgol_filter(lat, sw, sp)
            lon = vf.apply_savgol_filter(lon, sw, sp)
        elif stage == "kalman":
            lat = vf.apply_kalman_1d(lat, params["kalman_process_noise"], params["kalman_measurement_noise"])
            lon = vf.apply_kalman_1d(lon, params["kalman_process_noise"], params["kalman_measurement_noise"])
        stages[stage] = (lat.copy(), lon.copy())

    return stages


# =============================================================================
# Worker Thread
# =============================================================================

class ProcessingWorker(threading.Thread):
    def __init__(self, animal_cfg, settings, pipeline_order, enabled, params,
                 on_progress, on_done, on_error):
        super().__init__(daemon=True)
        self.animal_cfg    = animal_cfg
        self.settings      = settings
        self.pipeline_order = pipeline_order
        self.enabled       = enabled
        self.params        = params
        self.on_progress   = on_progress
        self.on_done       = on_done
        self.on_error      = on_error

    def run(self):
        try:
            self.on_progress(1, 6, "Wczytywanie danych…")
            df = vf.load_and_validate(self.animal_cfg)

            self.on_progress(2, 6, "Konwersja UTM (surowe)…")
            east_raw, north_raw, crs, transformer = vf.latlon_to_utm(
                df["lat"].values, df["lon"].values
            )

            self.on_progress(3, 6, "Filtracja sygnału…")
            stages = run_pipeline(df, self.pipeline_order, self.enabled, self.params)
            lat_filt, lon_filt = stages[self.pipeline_order[-1]]
            df["lat_filt"] = lat_filt
            df["lon_filt"] = lon_filt

            self.on_progress(4, 6, "Detekcja przekroczeń geofence…")
            east_filt, north_filt, _, _ = vf.latlon_to_utm(lat_filt, lon_filt)
            polygon_utm = vf.build_geofence_utm(
                self.settings["geofence_polygon_latlon"], transformer
            )
            breach_results = vf.compute_breach_events(
                east_filt, north_filt, df["elapsed_s"].values,
                polygon_utm, self.settings["buffer_warning_m"]
            )

            self.on_progress(5, 6, "Obliczanie metryk…")
            metrics = vf.compute_all_metrics(df, breach_results, east_filt, north_filt)
            speed = vf.compute_instantaneous_speed(
                df["lat_filt"].values, df["lon_filt"].values, df["elapsed_s"].values
            )

            self.on_progress(6, 6, "Gotowe.")
            self.on_done({
                "df": df,
                "east_raw": east_raw,
                "north_raw": north_raw,
                "east_filt": east_filt,
                "north_filt": north_filt,
                "polygon_utm": polygon_utm,
                "breach_results": breach_results,
                "metrics": metrics,
                "speed": speed,
                "stages": stages,
                "pipeline_order": self.pipeline_order,
                "enabled": dict(self.enabled),
            })
        except Exception as exc:
            self.on_error(str(exc))


# =============================================================================
# Główna klasa dashboard
# =============================================================================

class VirtualFenceDashboard(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Virtual Fence Dashboard")
        self.geometry("1440x900")
        self.minsize(1100, 700)

        self._cache: dict = {}
        self._current_animal = 0
        self._pending_after = None
        self._last_result = None
        self._figures: dict = {}
        self._canvases: dict = {}

        self._history: dict[int, list] = {0: [], 1: []}
        self._viewing_snapshot: bool = False
        self._selected_snap_idx: int | None = None
        self._snap_row_widgets: list = []
        self._auto_save_var = ctk.BooleanVar(value=False)

        # pipeline state
        self._pipeline_order = ["median", "savgol", "kalman"]
        self._filter_enabled: dict = {}
        self._filter_frames: dict = {}

        # per-tab state
        self._tab_params: dict[str, dict] = {}
        self._tab_pipeline_order: dict[str, list] = {}
        self._tab_filter_enabled: dict[str, dict] = {}
        self._tab_results: dict[str, dict | None] = {}
        self._active_tab: str = ""

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_ui()
        self._init_tab_state()
        self.after(200, self._trigger_processing)

    # ─────────────────────────────────────────────────────────────────────────
    # UI BUILDERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self._build_top_bar()
        self._build_left_panel()
        self._build_right_panel()
        self._build_status_bar()

    def _build_top_bar(self):
        bar = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color="#1a1a2e")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            bar, text="  Virtual Fence Dashboard",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="#ffffff"
        ).grid(row=0, column=0, padx=10, pady=8, sticky="w")

        mid = ctk.CTkFrame(bar, fg_color="transparent")
        mid.grid(row=0, column=1, sticky="")

        names = [f"  {d['name'].replace('_', ' ').title()}  " for d in DATASETS]
        self._animal_seg = ctk.CTkSegmentedButton(
            mid, values=names,
            command=self._on_animal_switch,
            font=ctk.CTkFont(size=13),
        )
        self._animal_seg.set(names[0])
        self._animal_seg.pack(side="left", padx=6)

        ctk.CTkButton(
            mid, text="+ CSV", width=80, height=34,
            command=self._open_load_csv_dialog,
            font=ctk.CTkFont(size=13),
            fg_color="#2a4a2a", hover_color="#3a6a3a",
        ).pack(side="left", padx=(4, 0))

        self._remove_btn = ctk.CTkButton(
            mid, text="×", width=36, height=34,
            command=self._remove_current_dataset,
            font=ctk.CTkFont(size=16),
            fg_color="#4a2a2a", hover_color="#6a3a3a",
        )
        self._remove_btn.pack(side="left", padx=(4, 0))
        self._remove_btn.pack_forget()

        self._run_btn = ctk.CTkButton(
            bar, text="▶  Apply & Run", width=130, height=34,
            command=lambda: self._trigger_processing(debounce=False),
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self._run_btn.grid(row=0, column=2, padx=(12, 4), pady=8, sticky="e")

        ctk.CTkButton(
            bar, text="↺  Reset", width=90, height=34,
            command=self._reset_params,
            font=ctk.CTkFont(size=13),
            fg_color="#3a3a3a", hover_color="#555555",
        ).grid(row=0, column=3, padx=(0, 12), pady=8, sticky="e")

        self._modified_lbl = ctk.CTkLabel(
            bar, text="", font=ctk.CTkFont(size=11), text_color=ORANGE
        )
        self._modified_lbl.grid(row=0, column=4, padx=(0, 14), pady=8, sticky="e")

    def _build_left_panel(self):
        self._left = ctk.CTkScrollableFrame(
            self, width=370, corner_radius=0,
            fg_color="#1e1e1e",
            scrollbar_button_color="#333",
        )
        self._left.grid(row=1, column=0, sticky="nsew")
        self._left.grid_columnconfigure(0, weight=1)

        self._build_pipeline_section(self._left)
        self._build_geofence_section(self._left)
        self._build_metric_cards(self._left)
        self._build_history_section(self._left)
        self._build_export_btn(self._left)

    def _section_label(self, parent, text):
        ctk.CTkLabel(
            parent, text=text,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=ACCENT,
        ).grid(pady=(14, 4), padx=10, sticky="w")

    # ── Pipeline section ────────────────────────────────────────────────────

    def _build_pipeline_section(self, parent):
        self._section_label(parent, "  ⚙  Signal Processing Pipeline")
        self._pipeline_container = ctk.CTkFrame(parent, fg_color="transparent")
        self._pipeline_container.grid(sticky="ew", padx=6)
        self._pipeline_container.grid_columnconfigure(0, weight=1)
        self._redraw_pipeline_blocks()

    def _redraw_pipeline_blocks(self):
        for w in self._pipeline_container.winfo_children():
            w.destroy()
        self._filter_frames = {}

        for idx, fname in enumerate(self._pipeline_order):
            self._build_filter_block(self._pipeline_container, fname, idx)

    def _build_filter_block(self, parent, fname, idx):
        color = FILTER_COLORS[fname]
        block = ctk.CTkFrame(parent, corner_radius=8, border_width=2, border_color=color,
                             fg_color="#252525")
        block.grid(row=idx, column=0, sticky="ew", pady=4, padx=2)
        block.grid_columnconfigure(1, weight=1)
        self._filter_frames[fname] = block

        # ── header row ──
        hdr = ctk.CTkFrame(block, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 2))
        hdr.grid_columnconfigure(1, weight=1)

        # ▲▼ buttons
        btn_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        btn_frame.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            btn_frame, text="▲", width=26, height=22, fg_color="#333",
            command=lambda f=fname: self._move_filter(f, -1),
            font=ctk.CTkFont(size=10),
        ).pack(side="left", padx=(0, 1))
        ctk.CTkButton(
            btn_frame, text="▼", width=26, height=22, fg_color="#333",
            command=lambda f=fname: self._move_filter(f, +1),
            font=ctk.CTkFont(size=10),
        ).pack(side="left")

        ctk.CTkLabel(
            hdr, text=f"  {FILTER_NAMES[fname]}",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=color,
        ).grid(row=0, column=1, sticky="w")

        # enable switch
        if fname not in self._filter_enabled:
            self._filter_enabled[fname] = ctk.BooleanVar(value=True)
        sw = ctk.CTkSwitch(
            hdr, text="", variable=self._filter_enabled[fname],
            width=46, command=self._on_param_change,
        )
        sw.grid(row=0, column=2, sticky="e")

        # ── params ──
        params_frame = ctk.CTkFrame(block, fg_color="transparent")
        params_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 8))
        params_frame.grid_columnconfigure(1, weight=1)

        if fname == "median":
            self._add_slider(params_frame, "Kernel size", "median_kernel_size",
                             3, 15, 1, odd_snap=True, row=0)
        elif fname == "savgol":
            self._add_slider(params_frame, "Window", "savgol_window",
                             5, 31, 1, odd_snap=True, row=0)
            self._add_slider(params_frame, "Polyorder", "savgol_polyorder",
                             2, 5, 1, row=1)
        elif fname == "kalman":
            self._add_log_slider(params_frame, "Q (process noise)",
                                 "kalman_process_noise", -6, -2, row=0)
            self._add_log_slider(params_frame, "R (meas. noise)",
                                 "kalman_measurement_noise", -4, 0, row=1)

    def _add_slider(self, parent, label, key, lo, hi, step, odd_snap=False, row=0):
        if not hasattr(self, "_slider_vars"):
            self._slider_vars = {}
        if key not in self._slider_vars:
            default_map = {
                "median_kernel_size": 5,
                "savgol_window": 11,
                "savgol_polyorder": 3,
                "buffer_warning_m": 200,
            }
            self._slider_vars[key] = ctk.DoubleVar(value=default_map.get(key, lo))

        var = self._slider_vars[key]
        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=11),
                     text_color="#cccccc").grid(row=row, column=0, sticky="w", pady=2)

        val_lbl = ctk.CTkLabel(parent, text=str(int(var.get())),
                               font=ctk.CTkFont(size=11, weight="bold"),
                               text_color="#ffffff", width=36)
        val_lbl.grid(row=row, column=2, padx=(4, 0))

        def on_change(v, lbl=val_lbl, k=key, snap=odd_snap):
            iv = int(round(float(v)))
            if snap and iv % 2 == 0:
                iv += 1
            self._slider_vars[k].set(iv)
            lbl.configure(text=str(iv))
            self._on_param_change()

        sl = ctk.CTkSlider(parent, from_=lo, to=hi, variable=var,
                           command=on_change, width=170)
        sl.grid(row=row, column=1, padx=6, sticky="ew")

    def _add_log_slider(self, parent, label, key, log_lo, log_hi, row=0):
        if not hasattr(self, "_slider_vars"):
            self._slider_vars = {}
        if not hasattr(self, "_log_slider_vars"):
            self._log_slider_vars = {}

        default_log = {"kalman_process_noise": -4, "kalman_measurement_noise": -2}
        if key not in self._log_slider_vars:
            self._log_slider_vars[key] = ctk.DoubleVar(value=default_log.get(key, log_lo))

        log_var = self._log_slider_vars[key]

        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=11),
                     text_color="#cccccc").grid(row=row, column=0, sticky="w", pady=2)

        val_lbl = ctk.CTkLabel(
            parent, text=f"1e{int(log_var.get())}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#ffffff", width=50,
        )
        val_lbl.grid(row=row, column=2, padx=(4, 0))

        def on_change(v, lbl=val_lbl, k=key):
            iv = int(round(float(v)))
            self._log_slider_vars[k].set(iv)
            lbl.configure(text=f"1e{iv}")
            self._on_param_change()

        sl = ctk.CTkSlider(parent, from_=log_lo, to=log_hi,
                           variable=log_var, command=on_change, width=170)
        sl.grid(row=row, column=1, padx=6, sticky="ew")

    # ── Geofence section ─────────────────────────────────────────────────────

    def _build_geofence_section(self, parent):
        self._section_label(parent, "  📍  Geofence Vertices (lat, lon)")
        gf_frame = ctk.CTkFrame(parent, fg_color="#252525", corner_radius=8)
        gf_frame.grid(sticky="ew", padx=6, pady=4)
        gf_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._gf_entries = []
        default_poly = vf.SETTINGS["geofence_polygon_latlon"]
        for i, (lat, lon) in enumerate(default_poly):
            row_f = ctk.CTkFrame(gf_frame, fg_color="transparent")
            row_f.grid(row=i, column=0, columnspan=4, sticky="ew", padx=8, pady=2)
            row_f.grid_columnconfigure((1, 3), weight=1)

            ctk.CTkLabel(row_f, text=f"V{i+1} ", font=ctk.CTkFont(size=11),
                         text_color=ACCENT, width=24).grid(row=0, column=0)
            e_lat = ctk.CTkEntry(row_f, width=85, placeholder_text="lat",
                                 font=ctk.CTkFont(size=11))
            e_lat.insert(0, str(lat))
            e_lat.grid(row=0, column=1, padx=(0, 4))
            e_lat.bind("<Return>", lambda _: self._on_param_change())

            ctk.CTkLabel(row_f, text=",", font=ctk.CTkFont(size=11)).grid(row=0, column=2)
            e_lon = ctk.CTkEntry(row_f, width=85, placeholder_text="lon",
                                 font=ctk.CTkFont(size=11))
            e_lon.insert(0, str(lon))
            e_lon.grid(row=0, column=3, padx=(4, 0))
            e_lon.bind("<Return>", lambda _: self._on_param_change())

            self._gf_entries.append((e_lat, e_lon))

        buf_row = ctk.CTkFrame(gf_frame, fg_color="transparent")
        buf_row.grid(row=4, column=0, columnspan=4, sticky="ew", padx=8, pady=(4, 8))
        buf_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(buf_row, text="Buffer warning [m]",
                     font=ctk.CTkFont(size=11), text_color="#cccccc").grid(row=0, column=0, sticky="w")

        if not hasattr(self, "_slider_vars"):
            self._slider_vars = {}
        if "buffer_warning_m" not in self._slider_vars:
            self._slider_vars["buffer_warning_m"] = ctk.DoubleVar(value=200)

        buf_lbl = ctk.CTkLabel(buf_row, text="200",
                               font=ctk.CTkFont(size=11, weight="bold"), width=36)
        buf_lbl.grid(row=0, column=2)

        def on_buf(v, lbl=buf_lbl):
            lbl.configure(text=str(int(float(v))))
            self._on_param_change()

        ctk.CTkSlider(buf_row, from_=50, to=500,
                      variable=self._slider_vars["buffer_warning_m"],
                      command=on_buf, width=170).grid(row=0, column=1, padx=6, sticky="ew")

    # ── Metric cards ─────────────────────────────────────────────────────────

    METRIC_DEFS = [
        ("total_distance_m",       "Distance",      lambda v: f"{v/1000:.2f}", "km"),
        ("time_inside_s",          "Time Inside",   lambda v: f"{v/3600:.2f}", "h"),
        ("time_outside_s",         "Time Outside",  lambda v: f"{v/3600:.2f}", "h"),
        ("breach_count",           "Breaches",      lambda v: str(int(v)),     ""),
        ("total_breach_duration_s","Breach Dur.",   lambda v: f"{v/3600:.2f}", "h"),
        ("mean_speed_ms",          "Mean Speed",    lambda v: f"{v:.4f}",      "m/s"),
        ("max_speed_ms",           "Max Speed",     lambda v: f"{v:.3f}",      "m/s"),
        ("convex_hull_area_m2",    "Hull Area",     lambda v: f"{v/1e6:.2f}",  "km²"),
        ("activity_index",         "Activity Idx",  lambda v: f"{v:.5f}",      "1/m"),
    ]

    def _build_metric_cards(self, parent):
        self._section_label(parent, "  📊  Metrics")
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.grid(sticky="ew", padx=6, pady=4)
        for c in range(3):
            grid.grid_columnconfigure(c, weight=1)

        self._metric_widgets = {}
        for i, (key, name, fmt, unit) in enumerate(self.METRIC_DEFS):
            r, c = divmod(i, 3)
            card = ctk.CTkFrame(grid, fg_color=BG_CARD, corner_radius=8,
                                border_width=1, border_color="#444")
            card.grid(row=r, column=c, padx=3, pady=3, sticky="ew", ipadx=4, ipady=4)
            card.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(card, text=name,
                         font=ctk.CTkFont(size=9), text_color=MUTED).grid(
                row=0, column=0, pady=(4, 0))
            val_lbl = ctk.CTkLabel(card, text="—",
                                   font=ctk.CTkFont(size=17, weight="bold"),
                                   text_color=ACCENT)
            val_lbl.grid(row=1, column=0)
            ctk.CTkLabel(card, text=unit,
                         font=ctk.CTkFont(size=9), text_color=MUTED).grid(
                row=2, column=0, pady=(0, 4))

            self._metric_widgets[key] = (card, val_lbl)

    # ── History section ───────────────────────────────────────────────────────

    def _build_history_section(self, parent):
        self._section_label(parent, "  🕑  Historia snapshotów")

        outer = ctk.CTkFrame(parent, fg_color="#252525", corner_radius=8,
                             border_width=1, border_color="#333")
        outer.grid(sticky="ew", padx=6, pady=(0, 4))
        outer.grid_columnconfigure(0, weight=1)

        # Wiersz: entry + przycisk Zapisz
        row0 = ctk.CTkFrame(outer, fg_color="transparent")
        row0.grid(row=0, column=0, sticky="ew", padx=6, pady=(8, 4))
        row0.grid_columnconfigure(0, weight=1)

        self._snap_name_entry = ctk.CTkEntry(
            row0, placeholder_text="Nazwa snapshotu (opcjonalna)…",
            font=ctk.CTkFont(size=11), height=28,
        )
        self._snap_name_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self._snap_save_btn = ctk.CTkButton(
            row0, text="💾 Zapisz", width=80, height=28,
            font=ctk.CTkFont(size=11),
            command=lambda: self._save_snapshot(self._snap_name_entry.get()),
        )
        self._snap_save_btn.grid(row=0, column=1)

        # Checkbox: autozapis
        ctk.CTkCheckBox(
            outer, text="Autozapis po obliczeniu",
            font=ctk.CTkFont(size=11), variable=self._auto_save_var,
            checkbox_width=16, checkbox_height=16,
        ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

        # Przewijalna lista snapshotów
        self._snap_list_frame = ctk.CTkScrollableFrame(
            outer, height=200, fg_color="#1a1a1a",
            scrollbar_button_color="#333",
        )
        self._snap_list_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        self._snap_list_frame.grid_columnconfigure(0, weight=1)

        # Przyciski: Porównaj i Usuń
        row3 = ctk.CTkFrame(outer, fg_color="transparent")
        row3.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 8))
        row3.grid_columnconfigure(0, weight=1)
        row3.grid_columnconfigure(1, weight=1)

        self._snap_compare_btn = ctk.CTkButton(
            row3, text="⚖  Porównaj dwa…", height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#2d3a4a", hover_color="#3a4f6b",
            command=self._open_compare_dialog,
        )
        self._snap_compare_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self._snap_delete_btn = ctk.CTkButton(
            row3, text="🗑 Usuń zaznaczony", height=28,
            font=ctk.CTkFont(size=11),
            fg_color="#4a2d2d", hover_color="#6b3a3a",
            command=self._delete_selected_snap,
        )
        self._snap_delete_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        self._refresh_snap_list()

    def _refresh_snap_list(self):
        for w in self._snap_list_frame.winfo_children():
            w.destroy()
        self._snap_row_widgets = []

        hist = self._history[self._current_animal]
        if not hist:
            ctk.CTkLabel(
                self._snap_list_frame,
                text="Brak zapisanych snapshotów",
                font=ctk.CTkFont(size=11), text_color=MUTED,
            ).pack(pady=10)
            return

        for i, snap in enumerate(reversed(hist)):
            real_idx = len(hist) - 1 - i
            is_sel = (real_idx == self._selected_snap_idx)
            row = ctk.CTkFrame(
                self._snap_list_frame,
                fg_color="#2e2e2e" if is_sel else "#252525",
                corner_radius=6,
                border_width=1,
                border_color=ACCENT if is_sel else "#3a3a3a",
            )
            row.pack(fill="x", padx=4, pady=2)

            ts = time.strftime("%H:%M", time.localtime(snap["timestamp"]))
            lbl_text = snap["label"]
            if len(lbl_text) > 34:
                lbl_text = lbl_text[:31] + "…"

            ctk.CTkLabel(
                row, text=lbl_text,
                font=ctk.CTkFont(size=11), text_color="#dddddd", anchor="w",
            ).pack(side="left", padx=(8, 4), pady=4, fill="x", expand=True)
            ctk.CTkLabel(
                row, text=ts,
                font=ctk.CTkFont(size=10), text_color=MUTED,
            ).pack(side="right", padx=6)

            for widget in [row] + list(row.winfo_children()):
                widget.bind("<Button-1>", lambda e, idx=real_idx: self._on_snap_select(idx))

            self._snap_row_widgets.append(row)

    def _make_snapshot_label(self) -> str:
        name = DATASETS[self._current_animal]["name"]
        hist = self._history[self._current_animal]
        idx = len(hist) + 1
        p = self._collect_params()
        ks = p["median_kernel_size"]
        sw = p["savgol_window"]
        sp = p["savgol_polyorder"]
        kq_raw = p["kalman_process_noise"]
        kq = round(math.log10(kq_raw)) if kq_raw > 0 else -4
        ts = time.strftime("%H:%M")
        return f"{name} – #{idx} – Med:{ks} SG:{sw}/{sp} Kal:1e{kq} – {ts}"

    def _save_snapshot(self, custom_label: str = ""):
        if self._last_result is None:
            self._status_lbl.configure(text="  Brak wyników do zapisania")
            return
        MAX_SNAPS = 10
        hist = self._history[self._current_animal]
        label = custom_label.strip() or self._make_snapshot_label()
        snap = {
            "label":     label,
            "timestamp": time.time(),
            "animal":    self._current_animal,
            "result":    self._last_result,
            "figures":   dict(self._figures),
        }
        hist.append(snap)
        if len(hist) > MAX_SNAPS:
            hist.pop(0)
        self._snap_name_entry.delete(0, "end")
        self._refresh_snap_list()
        short = label[:50]
        self._status_lbl.configure(text=f"  ✓ Snapshot zapisany: {short}")

    def _on_snap_select(self, idx: int):
        hist = self._history[self._current_animal]
        if idx < 0 or idx >= len(hist):
            return
        self._selected_snap_idx = idx
        self._viewing_snapshot = True
        self._refresh_snap_list()

        snap = hist[idx]
        self._last_result = snap["result"]
        self._update_metric_cards(snap["result"]["metrics"])

        fig = snap["figures"].get(self._active_tab)
        if fig is not None:
            self._embed_figure(self._active_tab, fig)

        self._status_lbl.configure(text=f"  Widok: {snap['label'][:55]}")
        self._timing_lbl.configure(text="[snapshot]")

    def _delete_selected_snap(self):
        if self._selected_snap_idx is None:
            return
        hist = self._history[self._current_animal]
        hist.pop(self._selected_snap_idx)
        was_viewing = self._viewing_snapshot
        self._selected_snap_idx = None
        self._viewing_snapshot = False
        self._refresh_snap_list()
        if was_viewing and self._last_result:
            self._update_metric_cards(self._last_result["metrics"])
            self._render_all(self._last_result)
            self._status_lbl.configure(text="  ✓ Ready")

    def _open_compare_dialog(self):
        hist = self._history[self._current_animal]
        if len(hist) < 2:
            self._status_lbl.configure(text="  Potrzebne co najmniej 2 snapshoty do porównania")
            return

        dlg = ctk.CTkToplevel(self)
        dlg.title("Porównaj snapshoty")
        dlg.geometry("860x580")
        dlg.grab_set()
        dlg.configure(fg_color="#1e1e1e")
        dlg.grid_rowconfigure(1, weight=1)
        dlg.grid_columnconfigure(0, weight=1)

        names = [s["label"] for s in hist]
        var_a = ctk.StringVar(value=names[-1])
        var_b = ctk.StringVar(value=names[-2])

        top = ctk.CTkFrame(dlg, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=16, pady=10)

        ctk.CTkLabel(top, text="Snapshot A:", font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkOptionMenu(top, values=names, variable=var_a, width=260,
                          command=lambda _: _refresh_table()).pack(side="left", padx=(6, 16))
        ctk.CTkLabel(top, text="Snapshot B:", font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkOptionMenu(top, values=names, variable=var_b, width=260,
                          command=lambda _: _refresh_table()).pack(side="left", padx=6)

        table_outer = ctk.CTkScrollableFrame(dlg, fg_color="#252525", corner_radius=8)
        table_outer.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))
        table_outer.grid_columnconfigure((0, 1, 2, 3), weight=1)

        def _refresh_table():
            for w in table_outer.winfo_children():
                w.destroy()
            snap_a = hist[names.index(var_a.get())]
            snap_b = hist[names.index(var_b.get())]
            _build_table(snap_a, snap_b)

        def _build_table(snap_a, snap_b):
            headers = ["Metryka",
                       snap_a["label"][:24] + ("…" if len(snap_a["label"]) > 24 else ""),
                       snap_b["label"][:24] + ("…" if len(snap_b["label"]) > 24 else ""),
                       "Różnica"]
            for c, h in enumerate(headers):
                ctk.CTkLabel(
                    table_outer, text=h,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color=ACCENT,
                ).grid(row=0, column=c, padx=12, pady=(8, 4), sticky="w")

            breach_keys = {"breach_count", "total_breach_duration_s"}
            for r, (key, name, fmt, unit) in enumerate(self.METRIC_DEFS, start=1):
                va = snap_a["result"]["metrics"].get(key)
                vb = snap_b["result"]["metrics"].get(key)
                txt_a = (fmt(va) + (" " + unit if unit else "")) if va is not None else "—"
                txt_b = (fmt(vb) + (" " + unit if unit else "")) if vb is not None else "—"

                diff_txt = "—"
                diff_color = "#cccccc"
                if va is not None and vb is not None:
                    delta = vb - va
                    diff_txt = f"{delta:+.3g}" + (" " + unit if unit else "")
                    if key in breach_keys:
                        diff_color = GREEN if delta <= 0 else RED

                row_bg = "#2a2a2a" if r % 2 == 0 else "#252525"
                for c, (txt, color) in enumerate([
                    (name, "#aaaaaa"),
                    (txt_a, "#ffffff"),
                    (txt_b, "#ffffff"),
                    (diff_txt, diff_color),
                ]):
                    cell = ctk.CTkFrame(table_outer, fg_color=row_bg, corner_radius=0)
                    cell.grid(row=r, column=c, sticky="ew", padx=1, pady=1)
                    table_outer.grid_columnconfigure(c, weight=1)
                    ctk.CTkLabel(cell, text=txt, font=ctk.CTkFont(size=11),
                                 text_color=color).pack(padx=10, pady=5, anchor="w")

        _refresh_table()

        ctk.CTkButton(dlg, text="Zamknij", command=dlg.destroy,
                      width=100).grid(row=2, column=0, pady=(0, 12))

    def _build_export_btn(self, parent):
        ctk.CTkButton(
            parent, text="💾  Export PNGs",
            command=self._export_all_pngs,
            fg_color="#2d4a2d", hover_color="#3a6b3a",
        ).grid(pady=(10, 16), padx=16, sticky="ew")

    # ── Right panel (tabs) ────────────────────────────────────────────────────

    def _build_right_panel(self):
        self._tabs = ctk.CTkTabview(self, corner_radius=8)
        self._tabs.grid(row=1, column=1, sticky="nsew", padx=(0, 6), pady=6)

        tab_names = [
            "🗺  Trajectory Map",
            "📈  Lat/Lon Signals",
            "🚀  Speed Plot",
            "⏱  Breach Timeline",
            "🔥  Heatmap",
            "🔬  Pipeline Steps",
        ]
        for t in tab_names:
            self._tabs.add(t)
            frame = self._tabs.tab(t)
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)

        self._tab_names = tab_names
        self.after(150, self._poll_active_tab)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_status_bar(self):
        bar = ctk.CTkFrame(self, height=32, corner_radius=0, fg_color="#111111")
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        self._progress = ctk.CTkProgressBar(bar, width=180, height=12)
        self._progress.set(0)
        self._progress.grid(row=0, column=0, padx=(10, 6), pady=8)
        self._progress.grid_remove()

        self._status_lbl = ctk.CTkLabel(bar, text="Ready",
                                        font=ctk.CTkFont(size=11), text_color=MUTED)
        self._status_lbl.grid(row=0, column=1, sticky="w")

        self._timing_lbl = ctk.CTkLabel(bar, text="",
                                        font=ctk.CTkFont(size=11), text_color=MUTED)
        self._timing_lbl.grid(row=0, column=2, padx=10)

        self._pts_lbl = ctk.CTkLabel(bar, text="",
                                     font=ctk.CTkFont(size=11), text_color=MUTED)
        self._pts_lbl.grid(row=0, column=3, padx=(0, 12))

    # ─────────────────────────────────────────────────────────────────────────
    # STATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_params(self) -> dict:
        if not hasattr(self, "_slider_vars"):
            self._slider_vars = {}
        if not hasattr(self, "_log_slider_vars"):
            self._log_slider_vars = {}

        def sv(k, default):
            v = self._slider_vars.get(k)
            return int(v.get()) if v else default

        def lv(k, default):
            v = self._log_slider_vars.get(k)
            return 10 ** int(v.get()) if v else default

        return {
            "median_kernel_size": sv("median_kernel_size", 5),
            "savgol_window":      sv("savgol_window", 11),
            "savgol_polyorder":   sv("savgol_polyorder", 3),
            "kalman_process_noise":      lv("kalman_process_noise", 1e-4),
            "kalman_measurement_noise":  lv("kalman_measurement_noise", 1e-2),
            "buffer_warning_m":   float(self._slider_vars.get("buffer_warning_m",
                                                               ctk.DoubleVar(value=200)).get()),
        }

    def _collect_settings(self) -> dict:
        params = self._collect_params()
        try:
            poly = [
                (float(e_lat.get()), float(e_lon.get()))
                for e_lat, e_lon in self._gf_entries
            ]
        except (ValueError, AttributeError):
            poly = vf.SETTINGS["geofence_polygon_latlon"]

        s = dict(vf.SETTINGS)
        s.update(params)
        s["geofence_polygon_latlon"] = poly
        return s

    def _get_enabled(self) -> dict:
        return {k: bool(v.get()) for k, v in self._filter_enabled.items()}

    def _open_load_csv_dialog(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Wybierz plik CSV (format Movebank)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        default_name = os.path.splitext(os.path.basename(path))[0]

        dlg = ctk.CTkToplevel(self)
        dlg.title("Wczytaj własne dane")
        dlg.resizable(False, False)
        dlg.grab_set()

        ctk.CTkLabel(dlg, text="Plik:", font=ctk.CTkFont(size=13)).grid(
            row=0, column=0, padx=(16, 8), pady=(16, 6), sticky="w")
        ctk.CTkLabel(dlg, text=path, font=ctk.CTkFont(size=11),
                     text_color="#aaaaaa", wraplength=380).grid(
            row=0, column=1, padx=(0, 16), pady=(16, 6), sticky="w")

        ctk.CTkLabel(dlg, text="Nazwa:", font=ctk.CTkFont(size=13)).grid(
            row=1, column=0, padx=(16, 8), pady=6, sticky="w")
        name_var = ctk.StringVar(value=default_name)
        ctk.CTkEntry(dlg, textvariable=name_var, width=220).grid(
            row=1, column=1, padx=(0, 16), pady=6, sticky="w")

        btn_frame = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(10, 16))

        ctk.CTkButton(btn_frame, text="Anuluj", width=100,
                      fg_color="#3a3a3a", hover_color="#555555",
                      command=dlg.destroy).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Wczytaj", width=100,
                      command=lambda: (
                          self._do_load_custom_csv(name_var.get().strip(), path),
                          dlg.destroy(),
                      )).pack(side="left", padx=8)

    def _do_load_custom_csv(self, name: str, path: str):
        if not name:
            self._status_lbl.configure(text="  Nazwa datasetu nie może być pusta")
            return
        if any(d["name"] == name for d in DATASETS):
            self._status_lbl.configure(text=f"  Dataset '{name}' już istnieje")
            return

        cfg = {
            "name": name,
            "path": path,
            "col_timestamp": "timestamp",
            "col_lat": "location-lat",
            "col_lon": "location-long",
            "col_speed": "ground-speed",
            "col_accuracy": None,
        }
        DATASETS.append(cfg)
        new_idx = len(DATASETS) - 1
        self._history[new_idx] = []

        formatted = [f"  {d['name'].replace('_', ' ').title()}  " for d in DATASETS]
        self._animal_seg.configure(values=formatted)
        self._animal_seg.set(formatted[new_idx])
        self._on_animal_switch(formatted[new_idx])

    def _get_animal_cfg(self) -> dict:
        cfg = dict(DATASETS[self._current_animal])
        if not os.path.isabs(cfg["path"]):
            project_dir = os.path.dirname(os.path.abspath(__file__))
            cfg["path"] = os.path.join(project_dir, cfg["path"])
        return cfg

    # ─────────────────────────────────────────────────────────────────────────
    # PIPELINE BLOCK INTERACTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _move_filter(self, fname, direction):
        idx = self._pipeline_order.index(fname)
        new_idx = idx + direction
        if 0 <= new_idx < len(self._pipeline_order):
            self._pipeline_order[idx], self._pipeline_order[new_idx] = \
                self._pipeline_order[new_idx], self._pipeline_order[idx]
            self._redraw_pipeline_blocks()
            self._on_param_change()

    # ─────────────────────────────────────────────────────────────────────────
    # PROCESSING
    # ─────────────────────────────────────────────────────────────────────────

    def _on_param_change(self):
        self._modified_lbl.configure(text="● modified")

    def _reset_params(self):
        # Suwaki liniowe
        defaults = {
            "median_kernel_size": 5,
            "savgol_window": 11,
            "savgol_polyorder": 3,
            "buffer_warning_m": 200,
        }
        for k, v in defaults.items():
            if k in self._slider_vars:
                self._slider_vars[k].set(v)

        # Suwaki logarytmiczne (Kalman)
        log_defaults = {
            "kalman_process_noise": -4,
            "kalman_measurement_noise": -2,
        }
        for k, v in log_defaults.items():
            if k in self._log_slider_vars:
                self._log_slider_vars[k].set(v)

        # Kolejność i włączenie filtrów
        self._pipeline_order = ["median", "savgol", "kalman"]
        for k in self._filter_enabled:
            self._filter_enabled[k].set(True)
        self._redraw_pipeline_blocks()

        # Geofence
        default_poly = vf.SETTINGS["geofence_polygon_latlon"]
        for i, (e_lat, e_lon) in enumerate(self._gf_entries):
            if i < len(default_poly):
                lat, lon = default_poly[i]
                e_lat.delete(0, "end")
                e_lat.insert(0, str(lat))
                e_lon.delete(0, "end")
                e_lon.insert(0, str(lon))

        if self._active_tab:
            self._tab_params[self._active_tab] = self._default_tab_params()
            self._tab_pipeline_order[self._active_tab] = ["median", "savgol", "kalman"]
            self._tab_filter_enabled[self._active_tab] = {
                "median": True, "savgol": True, "kalman": True
            }
        self._modified_lbl.configure(text="● modified")

    def _trigger_processing(self, debounce=True):
        if self._pending_after is not None:
            self.after_cancel(self._pending_after)
            self._pending_after = None
        if debounce:
            self._pending_after = self.after(800, self._start_worker)
        else:
            self._start_worker()

    def _start_worker(self):
        self._pending_after = None
        self._run_btn.configure(state="disabled")
        self._progress.grid()
        self._progress.set(0)
        self._t_start = time.time()

        worker = ProcessingWorker(
            animal_cfg=self._get_animal_cfg(),
            settings=self._collect_settings(),
            pipeline_order=list(self._pipeline_order),
            enabled=self._get_enabled(),
            params=self._collect_params(),
            on_progress=lambda s, t, m: self.after(0, lambda: self._on_progress(s, t, m)),
            on_done=lambda r: self.after(0, lambda: self._on_done(r)),
            on_error=lambda e: self.after(0, lambda: self._on_error(e)),
        )
        worker.start()

    def _on_progress(self, step, total, msg):
        self._progress.set(step / total)
        self._status_lbl.configure(text=f"  {msg}")

    def _on_done(self, result):
        elapsed = time.time() - self._t_start
        self._viewing_snapshot = False
        self._selected_snap_idx = None
        self._last_result = result
        key = self._current_animal
        self._cache[key] = result

        self._run_btn.configure(state="normal")
        self._progress.grid_remove()
        self._progress.set(0)
        self._status_lbl.configure(text="  ✓ Ready")
        self._timing_lbl.configure(text=f"computed in {elapsed:.1f}s")
        n = len(result["df"])
        self._pts_lbl.configure(text=f"{n:,} GPS points")
        self._modified_lbl.configure(text="")

        self._tab_results[self._active_tab] = result
        self._update_metric_cards(result["metrics"])
        self._render_single(self._active_tab, result)
        self._refresh_snap_list()
        if self._auto_save_var.get():
            self._save_snapshot()

    def _on_error(self, msg):
        self._run_btn.configure(state="normal")
        self._progress.grid_remove()
        self._status_lbl.configure(text=f"  ✗ Error: {msg[:80]}")

    # ─────────────────────────────────────────────────────────────────────────
    # METRIC CARDS
    # ─────────────────────────────────────────────────────────────────────────

    def _update_metric_cards(self, metrics: dict):
        breach_keys = {"breach_count", "total_breach_duration_s"}
        for key, name, fmt, unit in self.METRIC_DEFS:
            card, val_lbl = self._metric_widgets[key]
            v = metrics.get(key)
            if v is None:
                val_lbl.configure(text="—")
                continue
            val_lbl.configure(text=fmt(v))
            # highlight breach cards
            if key in breach_keys and metrics.get("breach_count", 0) > 0:
                card.configure(border_color=RED)
                val_lbl.configure(text_color=RED)
            else:
                card.configure(border_color="#444")
                val_lbl.configure(text_color=ACCENT)

    # ─────────────────────────────────────────────────────────────────────────
    # RENDER FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _embed_figure(self, tab_name, fig):
        old_fig = self._figures.get(tab_name)
        if old_fig is not None:
            plt.close(old_fig)

        tab = self._tabs.tab(tab_name)
        for w in tab.winfo_children():
            w.destroy()

        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ctk.CTkFrame(frame, fg_color="#1e1e1e", height=28)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame, pack_toolbar=False)
        toolbar.config(background="#1e1e1e")
        toolbar.pack(side="left")

        self._canvases[tab_name] = canvas
        self._figures[tab_name] = fig

    def _render_all(self, r):
        self._render_trajectory_map(r)
        self._render_lat_lon_signals(r)
        self._render_speed_plot(r)
        self._render_breach_timeline(r)
        self._render_heatmap(r)
        self._render_pipeline_steps(r)

    def _render_single(self, tab_name: str, r):
        dispatch = {
            self._tab_names[0]: self._render_trajectory_map,
            self._tab_names[1]: self._render_lat_lon_signals,
            self._tab_names[2]: self._render_speed_plot,
            self._tab_names[3]: self._render_breach_timeline,
            self._tab_names[4]: self._render_heatmap,
            self._tab_names[5]: self._render_pipeline_steps,
        }
        fn = dispatch.get(tab_name)
        if fn:
            fn(r)

    def _render_trajectory_map(self, r):
        df     = r["df"]
        er, nr = r["east_raw"], r["north_raw"]
        ef, nf = r["east_filt"], r["north_filt"]
        inside = r["breach_results"]["inside_mask"]
        poly   = r["polygon_utm"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#1e1e1e")
        off_e, off_n = ef.mean(), nf.mean()

        for ax, (e, n), title in zip(axes,
                [(er, nr), (ef, nf)],
                ["Raw GPS", "Filtered Trajectory"]):
            ax.set_facecolor("#1e1e1e")
            ax.scatter(e[inside]-off_e, n[inside]-off_n,
                       s=2, c=GREEN, alpha=0.5, label="In fence")
            ax.scatter(e[~inside]-off_e, n[~inside]-off_n,
                       s=2, c=RED, alpha=0.5, label="Outside")
            vf._draw_polygon(ax, poly, color="white", linewidth=2,
                             label="Geofence", offset_e=off_e, offset_n=off_n)
            ax.set_title(title, color="white")
            ax.set_xlabel("Easting – center [m]", color=MUTED)
            ax.set_ylabel("Northing – center [m]", color=MUTED)
            ax.legend(markerscale=4, fontsize=8)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.tick_params(colors=MUTED)
        fig.tight_layout()
        self._embed_figure(self._tab_names[0], fig)

    def _render_lat_lon_signals(self, r):
        df = r["df"]
        t  = df["elapsed_s"].values / 3600

        fig, axes = plt.subplots(3, 2, figsize=(13, 9), facecolor="#1e1e1e")
        signals = [
            ("lat", df["lat"].values, df["lat_filt"].values, "Latitude [°]"),
            ("lon", df["lon"].values, df["lon_filt"].values, "Longitude [°]"),
        ]
        for col, (label, raw, filt, ylabel) in enumerate(signals):
            for ax in axes[:, col]:
                ax.set_facecolor("#1e1e1e")
                ax.tick_params(colors=MUTED)

            axes[0, col].plot(t, raw,  color="#e74c3c", lw=0.8)
            axes[0, col].set_title(f"{label} – raw", color="white")

            axes[1, col].plot(t, filt, color="#3498db", lw=0.8)
            axes[1, col].set_title(f"{label} – filtered", color="white")

            res = raw - filt
            std = res.std()
            axes[2, col].plot(t, res, color=ORANGE, lw=0.7, alpha=0.9)
            axes[2, col].axhline(0,    color="#555", lw=0.9, ls="--")
            axes[2, col].axhline(std,  color="#3498db", lw=0.9, ls=":",
                                  label=f"+1σ={std:.2e}°")
            axes[2, col].axhline(-std, color="#3498db", lw=0.9, ls=":")
            axes[2, col].set_title(f"Residual {label}  (removed noise)", color="white")
            axes[2, col].legend(fontsize=8)

            for ax in axes[:, col]:
                ax.set_xlabel("Time [h]", color=MUTED)
                ax.set_ylabel(ylabel, color=MUTED)
                ax.grid(True, alpha=0.2)

        fig.suptitle("Lat/Lon Signals: raw / filtered / residual", color="white", fontsize=12)
        fig.tight_layout()
        self._embed_figure(self._tab_names[1], fig)

    def _render_speed_plot(self, r):
        df     = r["df"]
        speed  = r["speed"]
        breach = r["breach_results"]
        t      = df["elapsed_s"].values / 3600

        fig, ax = plt.subplots(figsize=(13, 4), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.plot(t, speed, color="#3498db", lw=0.8, label="Instantaneous speed")
        mean_s = float(np.nanmean(speed))
        ax.axhline(mean_s, color=ORANGE, ls="--", lw=1, label=f"Mean: {mean_s:.4f} m/s")

        for ev in breach["breach_events"]:
            ax.axvspan(ev["start_t"]/3600, ev["end_t"]/3600, alpha=0.18, color=RED)

        ax.set_xlabel("Time [h]", color=MUTED)
        ax.set_ylabel("Speed [m/s]", color=MUTED)
        ax.set_title("Instantaneous speed with geofence breach events", color="white")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors=MUTED)
        fig.tight_layout()
        self._embed_figure(self._tab_names[2], fig)

    def _render_breach_timeline(self, r):
        df     = r["df"]
        inside = r["breach_results"]["inside_mask"]
        breach = r["breach_results"]
        t      = df["elapsed_s"].values / 3600

        fig, ax = plt.subplots(figsize=(13, 3), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.axhspan(0, 1, color=RED, alpha=0.12, label="Outside")

        in_start = None
        for i in range(len(inside)):
            if inside[i] and in_start is None:
                in_start = t[i]
            elif not inside[i] and in_start is not None:
                ax.axvspan(in_start, t[i], color=GREEN, alpha=0.22, label="_nolegend_")
                in_start = None
        if in_start is not None:
            ax.axvspan(in_start, t[-1], color=GREEN, alpha=0.22)

        ax.step(t, inside.astype(float), where="post", color="#3498db", lw=1.5, label="State")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["0 – outside", "1 – inside"], color=MUTED)
        ax.set_ylim(-0.1, 1.3)
        ax.set_xlabel("Time [h]", color=MUTED)
        ax.set_title(
            f"Geofence state timeline  |  breaches: {len(breach['breach_events'])}",
            color="white"
        )
        legend_els = [
            Patch(facecolor=GREEN, alpha=0.4, label="In fence"),
            Patch(facecolor=RED,   alpha=0.3, label="Outside"),
        ]
        ax.legend(handles=legend_els, loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors=MUTED)
        fig.tight_layout()
        self._embed_figure(self._tab_names[3], fig)

    def _render_heatmap(self, r):
        ef   = r["east_filt"]
        nf   = r["north_filt"]
        poly = r["polygon_utm"]
        xx, yy, zz = vf.compute_kde_density(ef, nf)

        off_e, off_n = ef.mean(), nf.mean()
        margin = 0.05
        er = ef.max() - ef.min()
        nr = nf.max() - nf.min()
        xlim = (ef.min() - margin*er - off_e, ef.max() + margin*er - off_e)
        ylim = (nf.min() - margin*nr - off_n, nf.max() + margin*nr - off_n)

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        cf = ax.contourf(xx-off_e, yy-off_n, zz, levels=20, cmap="YlOrRd")
        cbar = plt.colorbar(cf, ax=ax, label="Presence density")
        cbar.ax.yaxis.set_tick_params(color=MUTED)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)
        vf._draw_polygon(ax, poly, color="white", linewidth=2,
                         label="Geofence", offset_e=off_e, offset_n=off_n)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("Easting – center [m]", color=MUTED)
        ax.set_ylabel("Northing – center [m]", color=MUTED)
        ax.set_title("Presence heatmap (KDE)", color="white")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.tick_params(colors=MUTED)
        fig.tight_layout()
        self._embed_figure(self._tab_names[4], fig)

    def _render_pipeline_steps(self, r):
        df     = r["df"]
        stages = r["stages"]
        order  = r["pipeline_order"]
        enabled = r["enabled"]
        t      = df["elapsed_s"].values / 3600

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), facecolor="#1e1e1e",
                                  sharex=True)

        step_colors = ["#888888"] + [FILTER_COLORS[f] for f in order]
        step_labels = ["Raw GPS"] + [
            f"{i+1}. {FILTER_NAMES[f]}" + ("" if enabled.get(f, True) else " (disabled)")
            for i, f in enumerate(order)
        ]
        step_data_lat = [stages["raw"][0]] + [stages[f][0] for f in order]
        step_data_lon = [stages["raw"][1]] + [stages[f][1] for f in order]

        for ax, data_list, ylabel in zip(axes,
                [step_data_lat, step_data_lon],
                ["Latitude [°]", "Longitude [°]"]):
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors=MUTED)
            for sig, color, label in zip(data_list, step_colors, step_labels):
                lw = 0.7 if label != step_labels[0] else 0.5
                alpha = 0.5 if label == step_labels[0] else 0.9
                ax.plot(t, sig, color=color, lw=lw, alpha=alpha, label=label)
            ax.set_ylabel(ylabel, color=MUTED)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc="upper right")

        axes[1].set_xlabel("Time [h]", color=MUTED)
        fig.suptitle("Pipeline step-by-step signal overlay", color="white", fontsize=12)
        fig.tight_layout()
        self._embed_figure(self._tab_names[5], fig)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB STATE
    # ─────────────────────────────────────────────────────────────────────────

    def _default_tab_params(self) -> dict:
        return {
            "median_kernel_size": 5,
            "savgol_window": 11,
            "savgol_polyorder": 3,
            "kalman_process_noise": 1e-4,
            "kalman_measurement_noise": 1e-2,
            "buffer_warning_m": 200.0,
        }

    def _init_tab_state(self):
        for t in self._tab_names:
            self._tab_params[t] = self._default_tab_params()
            self._tab_pipeline_order[t] = ["median", "savgol", "kalman"]
            self._tab_filter_enabled[t] = {"median": True, "savgol": True, "kalman": True}
            self._tab_results[t] = None
        self._active_tab = self._tab_names[0]

    def _save_active_tab_state(self):
        t = self._active_tab
        if not t:
            return
        self._tab_params[t] = self._collect_params()
        self._tab_pipeline_order[t] = list(self._pipeline_order)
        self._tab_filter_enabled[t] = {k: bool(v.get()) for k, v in self._filter_enabled.items()}

    def _load_tab_state(self, tab_name: str):
        params = self._tab_params.get(tab_name, self._default_tab_params())
        log_keys = {"kalman_process_noise", "kalman_measurement_noise"}

        for k, v in params.items():
            if k in log_keys:
                import math
                log_v = round(math.log10(v)) if v > 0 else -4
                if hasattr(self, "_log_slider_vars") and k in self._log_slider_vars:
                    self._log_slider_vars[k].set(log_v)
            else:
                if hasattr(self, "_slider_vars") and k in self._slider_vars:
                    self._slider_vars[k].set(v)

        order = self._tab_pipeline_order.get(tab_name, ["median", "savgol", "kalman"])
        self._pipeline_order[:] = order
        self._redraw_pipeline_blocks()

        enabled = self._tab_filter_enabled.get(tab_name, {})
        for fname, val in enabled.items():
            if fname in self._filter_enabled:
                self._filter_enabled[fname].set(val)

    def _poll_active_tab(self):
        current = self._tabs.get()
        if current and current != self._active_tab:
            self._on_tab_switch(current)
        self.after(150, self._poll_active_tab)

    def _on_tab_switch(self, tab_name: str):
        self._save_active_tab_state()
        self._active_tab = tab_name
        self._load_tab_state(tab_name)

        result = self._tab_results.get(tab_name)
        if result is not None:
            self._last_result = result
            self._update_metric_cards(result["metrics"])
            self._render_single(tab_name, result)
            self._status_lbl.configure(text="  ✓ Loaded from cache")
        else:
            self._trigger_processing(debounce=False)

    # ─────────────────────────────────────────────────────────────────────────
    # ANIMAL SWITCH
    # ─────────────────────────────────────────────────────────────────────────

    def _on_animal_switch(self, value):
        names = [f"  {d['name'].replace('_', ' ').title()}  " for d in DATASETS]
        self._current_animal = names.index(value)
        self._selected_snap_idx = None
        self._viewing_snapshot = False
        for t in self._tab_names:
            self._tab_results[t] = None
        self._active_tab = self._tabs.get() or self._tab_names[0]
        self._trigger_processing(debounce=False)
        self._refresh_snap_list()
        self._refresh_remove_btn()

    def _refresh_remove_btn(self):
        if self._current_animal >= 2:
            self._remove_btn.pack(side="left", padx=(4, 0))
        else:
            self._remove_btn.pack_forget()

    def _remove_current_dataset(self):
        idx = self._current_animal
        if idx < 2:
            return
        DATASETS.pop(idx)
        self._cache.pop(idx, None)
        self._history.pop(idx, None)
        formatted = [f"  {d['name'].replace('_', ' ').title()}  " for d in DATASETS]
        self._animal_seg.configure(values=formatted)
        self._animal_seg.set(formatted[0])
        self._on_animal_switch(formatted[0])

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def _export_all_pngs(self):
        name = DATASETS[self._current_animal]["name"]
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "wyniki", name)
        os.makedirs(out_dir, exist_ok=True)

        tab_file_map = {
            self._tab_names[0]: "trajectory_map.png",
            self._tab_names[1]: "lat_lon_signals.png",
            self._tab_names[2]: "speed_plot.png",
            self._tab_names[3]: "breach_timeline.png",
            self._tab_names[4]: "heatmap.png",
            self._tab_names[5]: "pipeline_steps.png",
        }

        saved_active = self._active_tab
        self._save_active_tab_state()

        exported = 0
        for tab_name, filename in tab_file_map.items():
            result = self._tab_results.get(tab_name)
            if result is None:
                continue
            self._load_tab_state(tab_name)
            self._render_single(tab_name, result)
            fig = self._figures.get(tab_name)
            if fig:
                path = os.path.join(out_dir, filename)
                fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
                exported += 1

        self._load_tab_state(saved_active)
        self._active_tab = saved_active

        if exported == 0:
            self._status_lbl.configure(text="  Nothing to export — run analysis first")
        else:
            self._status_lbl.configure(text=f"  ✓ Exported {exported} tabs to wyniki/{name}/")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    # zmień CWD na katalog skryptu żeby relatywne ścieżki do data/ działały
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    warnings.filterwarnings("ignore")
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = VirtualFenceDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
