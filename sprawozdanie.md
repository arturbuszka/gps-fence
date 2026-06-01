# Sprawozdanie – Automatyczna analiza trajektorii GPS  
## Detekcja przekroczeń wirtualnego ogrodzenia (Virtual Fencing)

**Przedmiot:** Analiza Sygnałów  
**Dane źródłowe:** *Daily grazing movements of cattle in the Far North Region, Cameroon*  
Movebank Data Repository, licencja CC0. Autorzy: Moritz M. et al.  
**Środowisko:** Python 3.12, biblioteki: NumPy, pandas, SciPy, matplotlib, pyproj

---

## 1. Cel i zakres projektu

Celem projektu jest zbudowanie kompletnego potoku przetwarzania sygnałów GPS zwierzęcych, który:

1. wczytuje i waliduje dane telemetryczne (znacznik czasu, szerokość i długość geograficzna),
2. filtruje sygnał GPS w celu usunięcia szumów,
3. sprawdza, czy zwierzę przebywa wewnątrz zdefiniowanego wielokąta (geofence),
4. oblicza metryki ruchu i eksploracji,
5. analizuje wrażliwość detekcji przekroczeń na parametry filtracji.

Analizowano dwa zwierzęta (*cattle0120309day* = animal_A, *cattle1110309day* = animal_B) monitorowane w dniu 12 marca 2009 w okolicach jeziora Logone, Kamerun.

---

## 2. Dane wejściowe

### 2.1 Format CSV

Plik źródłowy zawiera 166 921 rekordów dla 33 osobników. Istotne kolumny:

| Kolumna | Typ | Opis |
|---|---|---|
| `timestamp` | string ISO 8601 | Czas pomiaru UTC |
| `location-lat` | float64 | Szerokość geograficzna [°N] |
| `location-long` | float64 | Długość geograficzna [°E] |
| `ground-speed` | float64 | Prędkość naziemna z odbiornika GPS [m/s] |
| `individual-local-identifier` | string | Identyfikator zwierzęcia |

### 2.2 Wyodrębnione podzbiory

| Zbiór | Identyfikator | Wiersze surowe | Po walidacji | Czas obserwacji |
|---|---|---|---|---|
| animal_A | cattle0120309day | 5 768 | 4 846 | 12.8 h |
| animal_B | cattle1110309day | 4 732 | 4 160 | 10.6 h |

### 2.3 Obszar geograficzny

Oba zwierzęta poruszają się w prostokącie:
- szerokość: 11.140°N – 11.173°N
- długość: 15.081°E – 15.096°E

Pastwisko floodplain nad rzeką Logone, teren otwarty z kałużami sezonowymi.  
Układ odniesienia metryczny: **UTM strefa 33N (EPSG:32633)**.

---

## 3. Moduł 1 – Wczytanie i walidacja danych

### 3.1 Kod: `load_and_validate(cfg)`

```python
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.drop_duplicates(subset=["timestamp", "lat", "lon"])
```

Funkcja wykonuje kolejno:

1. **Wczytanie CSV** przez `pd.read_csv` i przemianowanie kolumn według słownika konfiguracji – umożliwia obsługę dowolnego formatu Movebank bez zmiany kodu.
2. **Parsowanie znaczników czasu** z uwzględnieniem strefy UTC. Sortowanie rosnące gwarantuje monotoniczność osi czasu, co jest wymagane przez filtry sygnałowe.
3. **Usuwanie duplikatów** – rekordy o identycznym znaczniku czasu, lat i lon traktowane jako błąd odbiornika (animal_A: 922 duplikaty = 16% surowych danych).
4. **Walidacja zakresu**: |lat| ≤ 90°, |lon| ≤ 180°. Wykrywa uszkodzone rekordy, w których GPS zwrócił wartość domyślną 0/0.
5. **Kolumna `elapsed_s`**: czas w sekundach od pierwszego pomiaru. Używana przez filtry i metryki zamiast bezwzględnych znaczników czasu.

### 3.2 Konwersja współrzędnych: `latlon_to_utm(lat, lon)`

Współrzędne geograficzne WGS84 (stopnie) są nieliniowe – jeden stopień długości geograficznej ma różną długość metryczną w zależności od szerokości. Aby móc stosować operacje geometryczne (odległość w metrach, algorytm punkt-wielokąt), konwertujemy na układ metryczny UTM:

```python
zone = int((median_lon + 180) / 6) + 1   # auto-detekcja strefy
epsg = 32600 + zone   # EPSG:32633 dla strefy 33N
transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
easting, northing = transformer.transform(lon, lat)
```

Wynikiem są współrzędne *easting* (E) i *northing* (N) w metrach. Dla obszaru Kamerunu (lon ≈ 15°E) strefa 33N jest prawidłowa.

**Źródło błędu nr 1:** Projekcja UTM wprowadza zniekształcenia liniowe rzędu 0.04% na krawędziach strefy. Dla obszaru ~10 km × 10 km błąd odległości wynosi < 4 m – akceptowalny dla skali zadania.

---

## 4. Moduł 2 – Filtracja sygnału GPS

Sygnał GPS jest zaszumiony z kilku powodów: wielościeżkowość (multipath), zasłonięcie nieba przez roślinność, rozproszenie jonosferyczne. Efektem są skokowe zmiany pozycji niemające sensu fizycznego.

Pipeline filtracji: **surowy → filtr medianowy → Savitzky-Golay → Kalman**

### 4.1 Filtr medianowy

```python
from scipy.signal import medfilt
lat_med = medfilt(df["lat"].values, kernel_size=5)
```

Filtr medianowy zastępuje każdą próbkę medianą okna o szerokości *k* próbek. Jest nieliniowy, co czyni go odpornym na impulsy (outliers) – pojedyncze skoki GPS o wartości 5–50 m zostają usunięte bez rozmycia krawędzi trajektorii.

**Parametr:** `median_kernel_size = 5` → każda próbka jest zastępowana medianą z niej samej i 4 sąsiadów (2 przed, 2 po).

**Działanie:** Jeżeli zwierzę w chwili *t* miało odczyt lat = 11.15001, a w sąsiednich próbkach lat ≈ 11.148, filtr zwróci 11.148 – usuwa skok nie wpływając na trend.

### 4.2 Filtr Savitzky-Golay

```python
from scipy.signal import savgol_filter
lat_sg = savgol_filter(lat_med, window_length=11, polyorder=3)
```

Filtr SG dopasowuje w każdym oknie wielomian stopnia *p* metodą najmniejszych kwadratów i zastępuje próbkę centralną wartością wielomianu. Zachowuje momenty wyższe niż filtr dolnoprzepustowy – szczegóły trajektorii nie są nadmiernie wygładzone.

**Wzór (lokalny wielomian stopnia p=3, okno w=11):**

$$\hat{y}[n] = \sum_{k=-(w-1)/2}^{(w-1)/2} c_k \cdot y[n+k]$$

gdzie $c_k$ to współczynniki wyznaczone z pseudoodwrócenia macierzy Vandermonde'a.

**Parametry:** `savgol_window = 11`, `savgol_polyorder = 3`  
Okno 11 próbek przy typowym odstępie ~10 s = ~110 s lokalna aproksymacja. Wystarczające dla usunięcia szumu GPS (składowe > 0.1 Hz) przy zachowaniu ruchów bydła (< 0.01 Hz).

### 4.3 Filtr Kalmana (implementacja ręczna)

Filtr Kalmana to rekurencyjny estymator stanu dla modeli liniowo-gaussowskich. Implementacja własna (bez filterpy) w ~30 wierszach NumPy.

**Model stanu** (pozycja + prędkość, krok czasowy dt=1):

$$\mathbf{x} = \begin{bmatrix} \text{pozycja} \\ \text{prędkość} \end{bmatrix}, \quad
\mathbf{F} = \begin{bmatrix} 1 & dt \\ 0 & 1 \end{bmatrix}, \quad
\mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

**Macierze szumów:**

$$\mathbf{Q} = q \cdot \mathbf{I}_2, \quad \mathbf{R} = [r]$$

gdzie $q = 10^{-4}$ (szum procesu – jak bardzo prędkość może się zmieniać), $r = 10^{-2}$ (szum pomiaru GPS).

**Pętla predykcja–aktualizacja:**

```
# Predykcja
x_pred = F @ x
P_pred = F @ P @ F.T + Q

# Aktualizacja (obserwacja z[i] = surowa próbka)
S = H @ P_pred @ H.T + R           # innowacyjna wariancja
K = P_pred @ H.T @ inv(S)          # wzmocnienie Kalmana
x = x_pred + K @ (z[i] - H @ x_pred)
P = (I - K @ H) @ P_pred
wynik[i] = x[0]                    # estymowana pozycja
```

**Interpretacja:** Wzmocnienie K decyduje, czy bardziej ufamy predykcji modelu czy nowemu pomiarowi. Przy małym R (dokładny GPS) K → duże (ufamy pomiarowi). Przy dużym R (zaszumiony GPS) K → małe (ufamy modelowi ruchu).

**Źródło błędu nr 2:** Agresywne wygładzanie (zbyt duże R lub zbyt duże okno SG) może „pochłonąć" prawdziwe, krótkie przekroczenie granicy. Analiza stabilności (sekcja 7) kwantyfikuje ten efekt.

---

## 5. Moduł 3 – Definicja geofence i detekcja przekroczeń

### 5.1 Definicja wirtualnego ogrodzenia

Geofence zdefiniowany jako prostokąt obejmujący **górne obozowisko** – obszar gdzie bydło spędza czas ~5–7h doby (noclegowisko/wodopój), zidentyfikowany na podstawie heatmapy KDE jako największe skupienie punktów GPS.

```python
"geofence_polygon_latlon": [
    (11.165, 15.080),   # SW
    (11.165, 15.090),   # SE
    (11.175, 15.090),   # NE
    (11.175, 15.080),   # NW
]
```

Po konwersji do UTM strefa 33N: prostokąt ~1 110 m (E-W) × 1 110 m (N-S) = ok. 1.2 km².  
Wierzchołki są zamknięte (ostatni = pierwszy) aby algorytm ray casting działał poprawnie.

**Uzasadnienie wyboru lokalizacji:** Heatmapa KDE wykazała dwa wyraźne centra aktywności na osi N-S połączone wąskim korytarzem przejścia. Geofence obejmuje centrum północne (górne obozowisko). Przekroczenia geofence odpowiadają fizycznym wyjściom bydła na pastwisko i powrotom do miejsca odpoczynku.

### 5.2 Algorytm ray casting (punkt-w-wielokącie)

```python
def point_in_polygon_raycasting(px, py, polygon):
    inside = False
    for i in range(len(polygon) - 1):
        xi, yi = polygon[i]
        xj, yj = polygon[i + 1]
        if ((yi > py) != (yj > py)) and (px < (xj-xi)*(py-yi)/(yj-yi) + xi):
            inside = not inside
    return inside
```

**Zasada działania:** Z badanego punktu P=(px, py) rzucamy poziomy promień w prawo (+x). Liczymy, ile razy promień przecina krawędzie wielokąta. Liczba parzysta → punkt na zewnątrz. Liczba nieparzysta → punkt wewnątrz (twierdzenie Jordana dla krzywych zamkniętych).

Warunek `(yi > py) != (yj > py)` – krawędź przecina poziom py tylko jeśli jeden wierzchołek leży powyżej, drugi poniżej. Zapobiega podwójnemu zliczeniu wierzchołków dokładnie na wysokości py.

**Złożoność:** O(n) dla n wierzchołków, wywoływana dla każdej próbki → O(N·n) dla N próbek. Przy n=4 (prostokąt) i N=5000 próbek: 20 000 operacji – negligibalna.

### 5.3 Strefa ostrzeżenia (buffer)

```python
inner_polygon = shrink_polygon(polygon_utm, buffer_m=200)
warning_mask[i] = inside_mask[i] and not point_in_polygon_raycasting(e[i], n[i], inner_polygon)
```

Wielokąt wewnętrzny powstaje przez przesunięcie każdego wierzchołka o `buffer_m = 200 m` w kierunku centroidu. Punkt leżący w strefie 200 m od granicy jest oznaczony jako ostrzeżenie – zwierzę jest wewnątrz, ale blisko przekroczenia.

### 5.4 Detekcja zdarzeń

```python
if not in_breach and not inside_mask[i]:   # wyjście ze strefy
    in_breach = True
    breach_start_idx = i
elif in_breach and inside_mask[i]:          # powrót do strefy
    in_breach = False
    breach_events.append({...})
```

Automat stanów z dwoma stanami: *wewnątrz* / *poza strefą*. Każde przejście `inside→outside` otwiera zdarzenie, przejście `outside→inside` je zamyka. Jeśli trajektoria kończy się poza strefą, zdarzenie jest zamykane na ostatniej próbce.

**Źródło błędu nr 3:** Nierównomierny krok czasowy. Jeśli próbki mają odstęp 10 s, ale jedna przerwa wynosi 20 minut, jedno zdarzenie przekroczenia może obejmować długi czas w którym tak naprawdę nie wiemy gdzie zwierzę było.

---

## 6. Moduł 4 – Metryki sygnału

### 6.1 Całkowity dystans – całkowanie numeryczne

```python
dists = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
total_distance = np.sum(dists)
```

Interpretacja jako całkowanie trapezów:

$$D = \int_0^T v(t)\,dt \approx \sum_{i=1}^{N-1} d_i$$

gdzie $d_i = \text{Haversine}(P_i, P_{i+1})$ jest odległością między sąsiednimi pomiarami.

**Wzór Haversine** (odległość na sferze o promieniu R = 6 371 000 m):

$$a = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cdot \cos\varphi_2 \cdot \sin^2\!\left(\frac{\Delta\lambda}{2}\right)$$
$$d = 2R \cdot \arcsin\!\sqrt{a}$$

gdzie $\varphi$ – szerokość geograficzna [rad], $\lambda$ – długość geograficzna [rad].

**Weryfikacja:** Haversine dla 1° szerokości na równiku = 111 195 m (wartość tablicowa).

### 6.2 Prędkość chwilowa

$$v_i = \frac{d_i}{\Delta t_i}, \quad \Delta t_i = t_{i+1} - t_i$$

Wartości > 10 m/s zamieniane na NaN (bydło nie biega szybciej niż ~36 km/h; wyższe wartości to artefakty GPS przy dużych odstępach próbkowania lub skok pozycji).

### 6.3 Pole obszaru eksploracji (convex hull)

```python
hull = ConvexHull(np.column_stack([easting, northing]))
area = hull.volume   # w 2D: .volume = pole [m²]
```

Otoczka wypukła (convex hull) to najmniejszy wypukły wielokąt zawierający wszystkie punkty trajektorii. Jej pole jest miarą łącznego obszaru eksplorowanego przez zwierzę.

**Wyniki:**

| Zwierzę | Dystans | Pole hull | Wskaźnik aktywności |
|---|---|---|---|
| animal_A | 20.13 km | 3.87 km² | 0.0052 1/m |
| animal_B | 18.26 km | 4.70 km² | 0.0039 1/m |

### 6.4 Wskaźnik aktywności

$$AI = \frac{D_{\text{total}}}{A_{\text{hull}}} \quad \left[\frac{\text{m}}{\text{m}^2}\right] = \left[\frac{1}{\text{m}}\right]$$

Wyższy wskaźnik → zwierzę przemierzało większą drogę w stosunku do eksplorowanego obszaru (ruch bardziej skoncentrowany, powtarzalne ścieżki). Niższy → szerokie eksplorowanie terenu bez powtarzania trasy.

Animal_A: AI = 0.0052 > animal_B: AI = 0.0039 → A poruszało się bardziej po utartych ścieżkach.

### 6.5 Heatmapa KDE

```python
kernel = gaussian_kde(np.vstack([easting, northing]))
zz = kernel(positions).reshape(grid_shape)
```

Jądrowe estymowanie gęstości (Kernel Density Estimation) tworzy ciągłą funkcję gęstości obecności. Każdy punkt obserwacji jest zastąpiony gaussowskim jądrem, a wyniki sumowane. Paski kolorów na wykresie pokazują obszary, gdzie zwierzę spędzało najwięcej czasu.

---

## 7. Moduł 5 – Analiza stabilności

### 7.1 Cel

Sprawdzenie, jak zmienia się liczba wykrytych przekroczeń przy różnych wartościach okna filtru Savitzky-Golay. Zbyt małe okno → sygnał jest nadal zaszumiony → fałszywe przekroczenia przy granicy geofence. Zbyt duże okno → wygładzenie pochłania prawdziwe, krótkie przekroczenia.

### 7.2 Wyniki – animal_A

| Okno SG | Przekroczenia | Dystans [m] | Czas w strefie [s] | Odchyłka |
|---|---|---|---|---|
| 5 | 4 | 22 967 | 12 480 | 0 |
| 7 | 4 | 22 991 | 12 480 | 0 |
| **11** | **4** | **20 374** | **12 480** | **0** |
| 15 | 4 | 18 172 | 12 480 | 0 |
| 21 | 3 | 14 687 | 12 480 | 1 |

### 7.3 Wyniki – animal_B

| Okno SG | Przekroczenia | Dystans [m] | Czas w strefie [s] | Odchyłka |
|---|---|---|---|---|
| 5 | 3 | 20 914 | 8 520 | 0 |
| 7 | 3 | 21 171 | 8 520 | 0 |
| **11** | **3** | **18 810** | **8 580** | **0** |
| 15 | 3 | 16 451 | 8 580 | 0 |
| 21 | 3 | 12 767 | 8 520 | 0 |

### 7.4 Wnioski

- **animal_A:** Liczba przekroczeń jest stabilna (4) dla okien 5–15. Dopiero okno 21 pochłania jedno zdarzenie (3 przekroczenia) – nadmierne wygładzenie skraca trajektorię o ~35% (22 991 → 14 687 m).
- **animal_B:** Wynik wyjątkowo stabilny – 3 przekroczenia niezależnie od okna SG. Świadczy to, że zdarzenia są dobrze rozdzielone w czasie i filtracja nie ma szans ich „połknąć".
- Czas w strefie pozostaje praktycznie stały dla wszystkich okien – geofence obejmuje obozowisko gdzie zwierzę stoi przez dłuższy czas, więc krótkie skoki pozycji GPS nie wpływają na klasyfikację wewnątrz/poza.

**Wniosek:** Nowy geofence (obozowisko) jest znacznie bardziej odporny na parametry filtracji niż poprzedni (przecięcie trasy). Wybór okna SG=11 pozostaje optymalny jako kompromis między dokładnością dystansu a stabilnością detekcji.

---

## 8. Wyniki końcowe

### 8.1 Tabela porównawcza

| Metryka | animal_A | animal_B |
|---|---|---|
| Dystans całkowity | 20.13 km | 18.26 km |
| Czas w strefie | 3.45 h (27.0%) | 2.38 h (22.5%) |
| Czas poza strefą | 9.35 h (73.0%) | 8.22 h (77.5%) |
| Liczba przekroczeń | 3 | 4 |
| Łączny czas poza strefą | 9.33 h | 8.22 h |
| Średnia prędkość | 0.0640 m/s | 0.0704 m/s |
| Maks. prędkość | 0.397 m/s | 0.338 m/s |
| Pole eksploracji (convex hull) | 3.87 km² | 4.70 km² |
| Wskaźnik aktywności | 0.0052 1/m | 0.0039 1/m |

### 8.2 Interpretacja wyników

Geofence obejmuje **górne obozowisko** – miejsce gdzie bydło nocuje i korzysta z wodopoju, zidentyfikowane na podstawie heatmapy KDE jako obszar najwyższej gęstości obecności (~5–7h doby).

Oba zwierzęta spędzają w strefie **22–27% czasu** (~2.4–3.5 h). Jest to zachowanie typowe dla bydła pasterskiego w systemie transhumance w Kotlinie Logone – zwierzęta wychodzą rano na pastwisko, wracają w środku dnia do obozowiska (odpoczynek/wodopój), a po południu wychodzą ponownie.

**Animal_A (3 przekroczenia):** regularny rytm dobowy – wyjście, powrót, wyjście. Wyższy wskaźnik aktywności (0.0052 vs 0.0039) przy mniejszym polu eksploracji (3.87 km²) wskazuje na bardziej powtarzalne, utarte ścieżki.

**Animal_B (4 przekroczenia):** dodatkowe, krótkie wyjście ze strefy w porównaniu do A. Szerszy obszar eksploracji (4.70 km²) przy niższym wskaźniku aktywności – zwierzę chodziło dalej, ale mniej regularnie. Niższa maks. prędkość (0.338 vs 0.397 m/s) sugeruje spokojniejszy, bardziej rozproszony ruch.

---

## 9. Źródła błędów i ograniczenia metody

### Błąd 1: Szum GPS (multipath, zasłonięcie)
Odbiornik GPS w trybie outdoor w terenie otwartym osiąga dokładność 3–10 m CEP. Przy granicy geofence skok pozycji o 15 m może spowodować fałszywe wykrycie przekroczenia. Filtr medianowy usuwa impulsy, ale nie eliminuje dryfu statystycznego.

### Błąd 2: Nierównomierny krok czasowy
Dane zawierają przerwy > 1h. Podczas przerwy zdarzenie przekroczenia może obejmować czas, w którym pozycja zwierzęcia jest nieznana. Metryka `time_inside_s` oparta na sumowaniu Δt jest wtedy nieprecyzyjna.

### Błąd 3: Aproksymacja rzutowania
Funkcja `shrink_polygon` przesuwa wierzchołki centrycznie – jest aproksymacją prawidłowego offsetu geometrycznego (offsetting algorytm Clipper). Dla prostokąta aproksymacja jest dobra, ale dla nieregularnych wielokątów może dawać błędne wyniki.

### Błąd 4: Agresywna filtracja
Okno SG = 21 skraca trajektorię animal_A o ~35% (22 991 → 14 687 m) i zmniejsza liczbę wykrytych przekroczeń z 4 do 3 – filtr pochłonął jedno prawdziwe zdarzenie. Dla animal_B (geofence obejmuje obozowisko z długimi postojami) efekt jest pomijalny – liczba przekroczeń stabilna dla wszystkich okien. Pokazuje to, że wrażliwość na filtrację zależy od charakteru strefy: strefy tranzytowe (krótkie pobyty) są podatne, strefy postojowe (długie pobyty) – odporne.

---

## 10. Struktura projektu i uruchomienie

```
analiza_sygnalow/
    virtual_fence.py       # cały kod (688 linii)
    requirements.txt       # zależności
    data/
        animal_A.csv       # cattle0120309day (4 846 pomiarów)
        animal_B.csv       # cattle1110309day (4 160 pomiarów)
    wyniki/
        animal_A/
            metrics.csv
            stability_table.csv
            trajectory_map.png
            lat_lon_signals.png
            speed_plot.png
            breach_timeline.png
            heatmap.png
        animal_B/          (ta sama struktura)
        comparison.csv
```

**Uruchomienie:**
```bash
pip install -r requirements.txt
python virtual_fence.py
```

Wszystkie parametry (ścieżki plików, współrzędne geofence, parametry filtrów) konfigurowane są wyłącznie przez słownik `SETTINGS` na początku skryptu.

---

## 11. Bibliografia

1. Moritz M. et al. (2018). *Daily grazing movements of cattle in the Far North Region, Cameroon.* Movebank Data Repository. https://datarepository.movebank.org/handle/10255/move.730 (CC0)
2. Savitzky A., Golay M.J.E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8), 1627–1639.
3. Kalman R.E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
4. Vincenty T. (1975). Direct and inverse solutions of geodesics on the ellipsoid. *Survey Review*, 23(176), 88–93. (podstawa wzoru Haversine)
5. Movable Type Scripts. Haversine formula. https://www.movable-type.co.uk/scripts/latlong.html
