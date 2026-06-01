# Sprawozdanie (skrót) – Automatyczna analiza trajektorii GPS
## Detekcja przekroczeń wirtualnego ogrodzenia (Virtual Fencing)

**Przedmiot:** Analiza Sygnałów  
**Dane źródłowe:** *Daily grazing movements of cattle in the Far North Region, Cameroon* – Movebank Data Repository, licencja CC0 (Moritz M. et al., 2018)  
**Środowisko:** Python 3.12 (NumPy, pandas, SciPy, matplotlib, pyproj)

---

## 1. Cel i dane

Celem projektu jest automatyczna analiza sygnału pozycji GPS bydła pasterskiego w celu wykrycia momentów, kiedy zwierzę opuszcza wyznaczony obszar (geofence). Dane pochodzą z bazy Movebank i zawierają 166 921 rekordów GPS dla 33 osobników z Kamerunu. Do analizy wybrano dwa zwierzęta obserwowane 12 marca 2009 w okolicach jeziora Logone:

| Zwierzę | Identyfikator | Pomiary po walidacji | Czas obserwacji |
|---|---|---|---|
| animal_A | cattle0120309day | 4 846 | 12.8 h |
| animal_B | cattle1110309day | 4 160 | 10.6 h |

Współrzędne GPS (WGS84) zostały przeliczone na układ metryczny UTM strefa 33N, co umożliwia obliczanie odległości w metrach i stosowanie algorytmów geometrycznych.

---

## 2. Przetwarzanie sygnału GPS

Surowy sygnał GPS jest zaszumiony z powodu wielościeżkowości (multipath), zasłonięcia nieba przez roślinność i rozproszenia jonosferycznego. Efektem są skokowe zmiany pozycji niemające sensu fizycznego. Zastosowano trójstopniowy potok filtracji:

**Filtr medianowy** – zastępuje każdą próbkę medianą z 5 sąsiednich pomiarów. Jest odporny na pojedyncze skoki pozycji (impulsy), które filtr uśredniający by tylko osłabił, a nie usunął.

**Filtr Savitzky-Golay (okno 11, stopień 3)** – dopasowuje lokalny wielomian do każdego fragmentu sygnału. Wygładza przebieg zachowując rzeczywiste zmiany kierunku ruchu, w przeciwieństwie do prostego uśredniania, które zaokrągla wszystkie krawędzie.

**Filtr Kalmana** – rekurencyjny estymator łączący model ruchu zwierzęcia (pozycja + prędkość) z bieżącym pomiarem GPS. Automatycznie ważony: gdy pomiar jest wiarygodny, bardziej mu ufa; gdy jest skokowy, opiera się na modelu.

---

## 3. Definicja strefy i detekcja przekroczeń

Wirtualne ogrodzenie (geofence) zostało zdefiniowane jako prostokąt ~1 110 m × 1 110 m (ok. 1.2 km²) obejmujący **górne obozowisko** – obszar gdzie zwierzęta spędzają najwięcej czasu (noclegowisko/wodopój), zidentyfikowany automatycznie na podstawie heatmapy gęstości obecności.

Sprawdzenie, czy punkt należy do wielokąta, realizuje algorytm **ray casting**: z każdego punktu GPS rzucany jest poziomy promień, a wynik zależy od parzystości liczby przecięć z krawędziami wielokąta (nieparzysta liczba = wewnątrz). Przekroczenia wykrywa automat stanów z dwoma stanami (*wewnątrz* / *poza strefą*) – każde przejście między stanami rejestrowane jest jako zdarzenie z datą i czasem trwania.

Dodatkowa **strefa ostrzeżenia** (bufor 200 m wewnątrz granicy) sygnalizuje zbliżanie się do krawędzi zanim do przekroczenia dojdzie.

---

## 4. Metryki sygnału

Całkowity dystans obliczany jest jako suma odległości między kolejnymi pomiarami (wzór Haversine uwzględniający kulistość Ziemi), co jest odpowiednikiem numerycznego całkowania trapezoidal. Pole obszaru eksploracji wyznaczane jest jako otoczka wypukła (convex hull) wszystkich punktów trajektorii.

### Wyniki

| Metryka | animal_A | animal_B |
|---|---|---|
| Dystans całkowity | 20.13 km | 18.26 km |
| Czas w strefie | 3.45 h (27.0%) | 2.38 h (22.5%) |
| Czas poza strefą | 9.35 h (73.0%) | 8.22 h (77.5%) |
| Liczba przekroczeń granicy | 3 | 4 |
| Średnia prędkość | 0.064 m/s | 0.070 m/s |
| Pole eksploracji (convex hull) | 3.87 km² | 4.70 km² |
| Wskaźnik aktywności (dystans/pole) | 0.0052 1/m | 0.0039 1/m |

---

## 5. Analiza stabilności – wpływ parametrów filtracji

Sprawdzono, jak zmiana szerokości okna filtru Savitzky-Golay (5–21 próbek) wpływa na wyniki.

**animal_A:**

| Okno SG | Przekroczenia | Dystans [m] |
|---|---|---|
| 5 | 4 | 22 967 |
| 7 | 4 | 22 991 |
| **11 (domyślne)** | **4** | **20 374** |
| 15 | 4 | 18 172 |
| 21 | 3 | 14 687 |

**animal_B:**

| Okno SG | Przekroczenia | Dystans [m] |
|---|---|---|
| 5 | 3 | 20 914 |
| 7 | 3 | 21 171 |
| **11 (domyślne)** | **3** | **18 810** |
| 15 | 3 | 16 451 |
| 21 | 3 | 12 767 |

Liczba przekroczeń jest stabilna dla szerokiego zakresu parametrów. Jedyny wyjątek: animal_A przy oknie 21 – zbyt agresywne wygładzenie pochłonęło jedno krótkie zdarzenie. Geofence obejmujący obozowisko (długie pobyty) jest z natury odporniejszy na parametry filtracji niż geofence tranzytowy (krótkie przejścia przez granicę).

---

## 6. Źródła błędów

**Szum GPS (multipath, zasłonięcie):** Odbiornik w terenie otwartym osiąga dokładność 3–10 m. Przy granicy geofence skok pozycji o kilkanaście metrów może wygenerować fałszywe zdarzenie przekroczenia. Filtr medianowy ogranicza ten efekt, ale nie eliminuje go całkowicie.

**Nierównomierny krok czasowy:** Dane zawierają przerwy powyżej 1 godziny. W czasie przerwy pozycja zwierzęcia jest nieznana – metryka czasu spędzonego w strefie może być niedoszacowana lub przeszacowana, jeśli przerwa przypada akurat na moment przekroczenia.

**Agresywna filtracja:** Zbyt duże okno filtru wygładza trajektorię tak mocno, że krótkie, prawdziwe wyjścia poza strefę przestają być widoczne w sygnale (animal_A, okno 21: utrata jednego zdarzenia, skrócenie szacowanego dystansu o ~35%).

---

## 7. Wnioski

Oba zwierzęta spędzają w strefie obozowiska **22–27% doby** (~2.4–3.5 h), co odpowiada typowemu rytmowi bydła pasterskiego w systemie transhumance: rano wyjście na pastwisko, powrót w południe na odpoczynek i wodopój, popołudniowe wyjście.

Animal_A porusza się bardziej regularnie po utartych ścieżkach (wyższy wskaźnik aktywności, mniejszy obszar eksploracji). Animal_B eksploruje większy obszar, ale mniej powtarzalnie i z dodatkowym, krótkim wyjściem ze strefy.

Wyniki są stabilne dla szerokiego zakresu parametrów filtracji, co świadczy o wiarygodności metody dla tego typu danych i geometrii geofence.

---

## 8. Bibliografia

1. Moritz M. et al. (2018). *Daily grazing movements of cattle in the Far North Region, Cameroon.* Movebank Data Repository. (CC0)
2. Savitzky A., Golay M.J.E. (1964). Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry*, 36(8).
3. Kalman R.E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1).
