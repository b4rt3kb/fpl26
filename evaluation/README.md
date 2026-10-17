# Wyniki testów modelu

- `forecast-2025.json`: prognozy kolejnych kolejek sezonu 2025/26.
- `forecast-2026.json`: prognozy GW1–GW6 sezonu 2026/27.

Pliki zawierają błędy punktów, błędy minut, parametry testu i sumy kontrolne danych.
Każda kolejka używa historii dostępnej przed jej początkiem.

```bash
python evaluate_strategy_models.py --season 2025-26 --output state/evaluation
```

`strategy_backtest.py` odtwarza sezon z transferami, budżetem, chipami i automatycznymi zmianami.
