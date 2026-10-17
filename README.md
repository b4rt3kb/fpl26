# fpl26

Model FPL, który co kolejkę proponuje skład, kapitana, ławkę, transfery i chip.
Decyzje znajdziesz w [`gameweeks/`](gameweeks/). Otwórz folder najbliższej kolejki.

## Jak działa

1. Pobiera zawodników, mecze i historię drużyny z API FPL.
2. Odtwarza skład, bank, ceny sprzedaży, wolne transfery i użyte chipy.
3. Prognozuje minuty, dostępność i punkty na sześć kolejek.
4. Planuje transfery, skład i chipy. Uwzględnia podwójne kolejki, zachowanie transferów oraz hity.
5. Zapisuje raport jako README, HTML, JSON i CSV.

GitHub Actions sprawdza deadline co godzinę.
Pełny raport powstaje w oknie 24 godzin i odświeża się w ostatnich dwóch godzinach.
Harmonogram działa na GitHubie i używa Pythona. Nie wymaga otwartego komputera ani API AI.

Punktem wyjścia jest ostatni skład opublikowany po deadline’ie. Transfery i ustawienie składu zatwierdzasz w FPL.

## Uruchomienie

Python 3.11. ID drużyny dla automatyzacji ustaw w zmiennej repozytorium `FPL_ENTRY_ID`.

```bash
pip install -r requirements.txt
python fetch_fpl.py --workers 6
python predict_gw.py --horizon 6 --no-squad
python weekly_decision.py --entry 8863206 --publish-dir gameweeks
```

Ręczne odświeżenie na GitHubie: **Actions → weekly-gameweek-decisions → Run workflow**.

## Pliki

| Folder | Zawartość |
|---|---|
| [`gameweeks/`](gameweeks/) | Decyzje dla każdej kolejki |
| [`predictions/`](predictions/) | Prognozy zawodników i archiwum danych |
| [`evaluation/`](evaluation/) | Wyniki testów modelu |
| [`tests/`](tests/) | Testy danych, modelu i zasad FPL |
| [`data/`](data/) | Opis pobieranych danych |

Model i planowanie opisuje [`README_MODEL.md`](README_MODEL.md).
