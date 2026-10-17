# Model i planowanie

Model łączy dwie prognozy punktów w proporcji 50/50: bezpośredni model drzew oraz model oparty na minutach.
Używa formy, startów, xG, xA, działań defensywnych, ceny, rywala i terminarza.

Model minut oblicza szanse braku występu, gry przez 1–59 minut oraz gry przez co najmniej 60 minut.
Kalibracja używa późniejszych meczów niż trening. Prognozy minut powstają osobno dla każdego zawodnika.
Status i komunikaty FPL korygują dostępność na datę każdego meczu, w tym przewidywany powrót po kontuzji.

Każda prognoza obejmuje sześć kolejek i używa danych dostępnych w chwili obliczenia.
Podwójna kolejka sumuje punkty obu meczów. Kolejka bez meczu daje zero punktów.

## Planer

HiGHS wybiera transfery, jedenastkę, kapitana i chipy dla całego okresu.
Sprawdza budżet, ceny sprzedaży, limit klubu, formację, wolne transfery oraz koszt dodatkowych transferów.
Wildcard zmienia skład na stałe. Free Hit przywraca poprzedni skład po kolejce.
Bench Boost uwzględnia ławkę. Triple Captain potraja punkty kapitana.
Osobny krok poprawia ławkę i C/VC, uwzględniając prawdopodobieństwo automatycznych zmian.

Planer zachowuje bieżące ceny przez sześć kolejek.
Jeżeli chip pozostaje dostępny poza planem, planer odlicza od jego użycia rezerwę punktów: WC 25, FH 20, BB 18, TC 12.
Rezerwa wynosi zero, gdy chip wygasa w planowanym okresie.

## Wyniki prognoz

Błąd RMSE punktów dla całej puli zawodników. Mniejsza wartość oznacza mniejszy błąd.

| Okres | Model bezpośredni | Model 50/50 |
|---|---:|---:|
| 2025/26, 38 kolejek | 1,949 | 1,930 |
| 2026/27, GW1–GW6 | 2,203 | 2,194 |

Test odtwarza kolejne momenty decyzji na danych historycznych. Sezon 2025/26 służył także do rozwoju modelu.
Parametry, pozostałe miary i wyniki minut są w [`evaluation/`](evaluation/).
