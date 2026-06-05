# Architektura projektu — analiza_sygnalow

## 1. Podział modułów

| Moduł | Rola | Eksportuje |
|-------|------|------------|
| `virtual_fence.py` | Backend obliczeniowy: wczytywanie danych, filtracja sygnału GPS, geofence, metryki, wizualizacja | Funkcje `vf.*`, `vf.SETTINGS` |
| `dashboard.py` | GUI (CustomTkinter) + wątek roboczy + zarządzanie stanem per-zakładka | Klasa `VirtualFenceDashboard`, `run_pipeline()` |
| `generate_test_data.py` | Generator syntetycznych plików CSV do testowania filtrów | Nie importowany przez inne moduły |

`dashboard.py` woła `virtual_fence` jako bibliotekę — nie ma logiki obliczeniowej poza `run_pipeline()` (kolejność filtrów konfigurowana przez użytkownika).

---

## 2. Przepływ danych

```
Plik CSV (format Movebank)
  │  kolumny: timestamp, location-lat, location-long, ground-speed
  │
  ▼
vf.load_and_validate(cfg)
  │  wynik: DataFrame [timestamp, elapsed_s, lat, lon, speed]
  │  operacje: parsowanie czasu, usuwanie duplikatów/NaN, walidacja zakresów geograficznych
  │
  ├─► vf.latlon_to_utm(lat_raw, lon_raw)
  │       wynik: east_raw, north_raw  [metry, układ UTM]
  │
  ▼
run_pipeline(df, pipeline_order, enabled, params)          ← dashboard.py
  │  kolejność filtrów konfigurowana per-zakładka
  │  wynik: stages = {"raw": ..., "median": ..., "savgol": ..., "kalman": ...}
  │         każdy stage to (lat_array, lon_array)
  │
  ├─► vf.latlon_to_utm(lat_filt, lon_filt)
  │       wynik: east_filt, north_filt
  │
  ├─► vf.build_geofence_utm(polygon_latlon, transformer)
  │       wynik: polygon_utm  [np.ndarray (n+1, 2)]
  │
  ├─► vf.compute_breach_events(east_filt, north_filt, elapsed_s, polygon_utm, buffer_m)
  │       wynik: breach_results  (patrz §4)
  │
  ├─► vf.compute_all_metrics(df, breach_results, east_filt, north_filt)
  │       wynik: metrics dict  (patrz §4)
  │
  └─► ProcessingWorker.on_done(result)                     ← wątek → UI thread
            │
            ▼
      _render_single(active_tab, result)
            │
            ▼
      matplotlib Figure  →  FigureCanvasTkAgg  →  CTkTabview
```

---

## 3. Wzory matematyczne

### 3.1 Filtr medianowy
**Zastosowanie:** usuwanie szumu impulsowego (pojedyncze błędne odczyty GPS).

```
y[i] = mediana( x[i-k], ..., x[i], ..., x[i+k] )
```

gdzie `kernel_size = 2k+1` (zawsze nieparzyste). Każdy punkt zastępowany medianą sąsiedztwa — skoki nie propagują się dalej.

Implementacja: `scipy.signal.medfilt`

**Parametr UI:** *Kernel size* (3–15, tylko nieparzyste)

---

### 3.2 Filtr Savitzky-Golay
**Zastosowanie:** wygładzanie szumu gaussowskiego z zachowaniem kształtu sygnału (lokalne minima/maksima).

Do każdego okna `w` dopasowywany jest wielomian stopnia `p` metodą najmniejszych kwadratów. Wartość w centrum okna zastępuje oryginalny punkt.

```
y[i] = suma( c_j · x[i+j] )  dla  j = -(w-1)/2 .. (w-1)/2
```

gdzie `c_j` to współczynniki wynikające z dopasowania wielomianu stopnia `p`.

Implementacja: `scipy.signal.savgol_filter`

**Parametry UI:** *Window* (5–31, nieparzyste), *Polyorder* (2–5, musi być < window)

---

### 3.3 Filtr Kalmana 1D
**Zastosowanie:** eliminacja wolno narastającego dryftu GPS; model zakłada że zwierzę porusza się ze stałą prędkością z małym zakłóceniem.

**Model stanu:** wektor `x = [pozycja, prędkość]ᵀ`

**Macierze:**
```
F = [[1, dt],   — przejście stanu (dt = 1.0 s)
     [0,  1]]

H = [[1, 0]]    — obserwujemy tylko pozycję

Q = q · I₂      — szum procesu (q = process_noise)

R = [[r]]       — szum pomiaru (r = measurement_noise)
```

**Pętla (dla każdej próbki i):**
```
Predykcja:
  x_pred = F · x
  P_pred = F · P · Fᵀ + Q

Aktualizacja:
  S = H · P_pred · Hᵀ + R
  K = P_pred · Hᵀ · S⁻¹        (wzmocnienie Kalmana)
  x = x_pred + K · (z_i − H · x_pred)
  P = (I − K · H) · P_pred
```

Wynik: `x[0]` (pozycja) dla każdego kroku.

**Parametry UI:** *Q (process noise)* 10⁻⁶–10⁻², *R (meas. noise)* 10⁻⁴–10⁰ (log-suwaki)

Duże Q/małe R → mocna korekcja (śledzi pomiary). Małe Q/duże R → silne wygładzenie (śledzi model).

---

### 3.4 Odległość Haversine
**Zastosowanie:** odległość między dwoma punktami GPS na sferze (całkowity dystans, prędkość chwilowa).

```
a = sin²(Δφ/2) + cos(φ₁) · cos(φ₂) · sin²(Δλ/2)
d = 2R · arcsin(√a)
```

gdzie `φ` = szerokość, `λ` = długość (radiany), `R = 6 371 000 m`.

---

### 3.5 Ray casting — punkt w geofence
**Zastosowanie:** sprawdzenie czy przefiltrowany punkt GPS leży wewnątrz wielokąta geofence.

Algorytm liczy ile razy poziomy promień z punktu `(px, py)` przecina krawędzie wielokąta. Nieparzysta liczba przecięć → punkt wewnątrz.

```python
dla każdej krawędzi (xi, yi) → (xj, yj):
    if ((yi > py) != (yj > py)) and (px < (xj−xi)·(py−yi)/(yj−yi) + xi):
        inside = not inside
```

---

### 3.6 Shrink polygon — strefa ostrzegawcza
**Zastosowanie:** wewnętrzny bufor geofence — alarm zanim zwierzę przekroczy granicę.

Każdy wierzchołek przesuwa się w kierunku centroidu wielokąta:

```
kierunek = (centroid − v) / ‖centroid − v‖
nowy_v   = v + kierunek · min(buffer_m, 0.9 · ‖centroid − v‖)
```

Ograniczenie do 90% odległości do centroidu zapobiega "przeciśnięciu" wierzchołka za centrum.

**Parametr UI:** *Buffer warning [m]* (50–500 m)

---

### 3.7 KDE — heatmapa gęstości obecności
**Zastosowanie:** wizualizacja gdzie zwierzę spędza najwięcej czasu.

Dla zbiorów ≤10 000 punktów: jądrowa estymacja gęstości (`scipy.stats.gaussian_kde`) z automatycznym doborem szerokości pasma (reguła Silvermana).

Dla zbiorów >10 000 punktów: fallback do histogramu 2D (`numpy.histogram2d`) ze względu na koszt obliczeniowy KDE.

---

### 3.8 Activity index
**Zastosowanie:** miara aktywności ruchowej — ile metrów trasy na metr kwadratowy obszaru.

```
activity_index = total_distance_m / max(convex_hull_area_m2, 1)   [1/m]
```

Pole wypukłego otoczenia (`scipy.spatial.ConvexHull`) przybliża obszar eksplorowany przez zwierzę.

---

## 4. Struktury danych

### DataFrame po `vf.load_and_validate()`

| Kolumna | Typ | Opis |
|---------|-----|------|
| `timestamp` | datetime64 | czas UTC |
| `elapsed_s` | float64 | sekundy od pierwszego pomiaru |
| `lat` | float64 | szerokość WGS84 [°] |
| `lon` | float64 | długość WGS84 [°] |
| `speed` | float64 | prędkość GPS [m/s], opcjonalna |
| `lat_filt` | float64 | po filtracji (dodawana przez pipeline) |
| `lon_filt` | float64 | po filtracji (dodawana przez pipeline) |

---

### `result` dict — wynik `ProcessingWorker`

```python
{
  "df":             pd.DataFrame,     # z kolumnami lat_filt, lon_filt
  "east_raw":       np.ndarray,       # UTM surowy [m]
  "north_raw":      np.ndarray,
  "east_filt":      np.ndarray,       # UTM przefiltrowany [m]
  "north_filt":     np.ndarray,
  "polygon_utm":    np.ndarray,       # geofence (n+1, 2)
  "breach_results": dict,             # patrz niżej
  "metrics":        dict,             # patrz niżej
  "speed":          np.ndarray,       # prędkość chwilowa [m/s]
  "stages":         dict,             # "raw"/"median"/"savgol"/"kalman" → (lat, lon)
  "pipeline_order": list,             # np. ["median", "savgol", "kalman"]
  "enabled":        dict,             # {"median": True, ...}
}
```

---

### `breach_results` dict

```python
{
  "inside_mask":    np.ndarray[bool],  # True jeśli punkt wewnątrz geofence
  "warning_mask":   np.ndarray[bool],  # True jeśli w strefie ostrzegawczej
  "breach_events":  list[dict],        # lista zdarzeń przekroczenia
  "entries":        list[int],         # indeksy wejść do strefy
  "exits":          list[int],         # indeksy wyjść ze strefy
}

# breach_event:
{
  "start_idx": int,
  "end_idx":   int,
  "start_t":   float,   # elapsed_s
  "end_t":     float,
  "duration_s": float,
}
```

---

### `TabState` dataclass (`dashboard.py`)

```python
@dataclass
class TabState:
    params:          dict        # parametry filtrów (kernel_size, window, Q, R, buffer)
    pipeline_order:  list        # kolejność filtrów, np. ["median", "savgol", "kalman"]
    filter_enabled:  dict        # {"median": True, "savgol": True, "kalman": True}
    result:          dict|None   # ostatni wynik dla tej zakładki (cache)
```

Każda z 6 zakładek ma własny `TabState` w `_tabs_state: dict[str, TabState]`.
Przy przełączeniu zakładki: aktywna zakładka jest zapisywana (`_save_active_tab_state`), nowa ładowana (`_load_tab_state`).

---

### `DATASETS` list (`vf.SETTINGS["datasets"]`)

```python
[
  {
    "name":          str,          # wyświetlana nazwa, np. "animal_A"
    "path":          str,          # ścieżka do CSV (względna lub absolutna)
    "col_timestamp": str,          # nazwa kolumny czasu, np. "timestamp"
    "col_lat":       str,          # np. "location-lat"
    "col_lon":       str,          # np. "location-long"
    "col_speed":     str | None,   # np. "ground-speed", opcjonalna
    "col_accuracy":  str | None,   # opcjonalna
  },
  ...
]
```

Nowe datasety ("+CSV") są appendowane do tej listy w czasie działania programu.

---

## 5. Funkcje `vf.*` wołane z dashboardu

| Funkcja | Sygnatura (skrócona) | Zwraca | Gdzie w dashboard.py |
|---------|----------------------|--------|----------------------|
| `load_and_validate` | `(cfg) → DataFrame` | DataFrame z elapsed_s, lat, lon | `ProcessingWorker.run` L.157 |
| `latlon_to_utm` | `(lat, lon) → (east, north, crs, transformer)` | tablice metryczne + obiekt transformacji | `ProcessingWorker.run` L.160, 171 |
| `apply_median_filter` | `(signal, kernel_size) → ndarray` | wygładzony sygnał | `run_pipeline()` L.122–123 |
| `apply_savgol_filter` | `(signal, window, polyorder) → ndarray` | wygładzony sygnał | `run_pipeline()` L.127–128 |
| `apply_kalman_1d` | `(signal, Q, R) → ndarray` | odfiltrowany sygnał | `run_pipeline()` L.130–131 |
| `build_geofence_utm` | `(polygon_latlon, transformer) → ndarray` | wielokąt UTM (n+1, 2) | `ProcessingWorker.run` L.172 |
| `compute_breach_events` | `(east, north, elapsed_s, polygon, buffer_m) → dict` | breach_results | `ProcessingWorker.run` L.175 |
| `compute_all_metrics` | `(df, breach_results, east, north) → dict` | słownik metryk | `ProcessingWorker.run` L.181 |
| `compute_instantaneous_speed` | `(lat, lon, elapsed_s) → ndarray` | prędkość chwilowa [m/s] | `ProcessingWorker.run` L.182 |
| `compute_kde_density` | `(east, north) → (xx, yy, zz)` | siatka gęstości KDE | `_render_heatmap` L.1319 |
| `_draw_polygon` | `(ax, polygon, ...)` | — (rysuje na ax) | `_render_trajectory_map` L.1201, `_render_heatmap` L.1334 |
