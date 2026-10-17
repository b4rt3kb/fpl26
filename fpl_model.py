"""Shared fitted models and player-GW evaluation used by training and forecasts."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from fpl_features import FEATS


def hgb():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                         max_depth=6, random_state=42)


class PointsModel:
    def __init__(self, kind="hgb"):
        if kind not in ("hgb", "two"):
            raise ValueError(f"Unknown model: {kind}")
        self.kind = kind

    def fit(self, rows):
        if rows.empty or not np.isfinite(rows["total_points"].to_numpy(float)).all():
            raise ValueError("Training requires completed matches with known targets")
        if np.isinf(rows[FEATS].to_numpy(float)).any():
            raise ValueError("Infinite training feature")
        self.regressor = hgb()
        if self.kind == "hgb":
            self.regressor.fit(rows[FEATS], rows["total_points"])
        else:
            # Both stages use precisely the same appearance condition.
            played = rows["minutes"].gt(0)
            if not played.any():
                raise ValueError("No appearances available for conditional model")
            self.constant_play_probability = float(played.iloc[0]) if played.nunique() == 1 else None
            self.classifier = None
            if self.constant_play_probability is None:
                self.classifier = HistGradientBoostingClassifier(max_iter=200, random_state=42)
                self.classifier.fit(rows[FEATS], played)
            self.regressor.fit(rows.loc[played, FEATS], rows.loc[played, "total_points"])
        return self

    def predict(self, rows):
        if rows.empty:
            return np.array([], dtype=float)
        predictions = self.regressor.predict(rows[FEATS])
        if self.kind == "two":
            probability = (self.constant_play_probability if self.classifier is None
                           else self.classifier.predict_proba(rows[FEATS])[:, 1])
            predictions = probability * predictions
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced non-finite predictions")
        return np.clip(predictions, 0, None)


def player_gameweeks(rows, prediction_col):
    """Sum double fixtures before ranking players or reporting GW errors."""
    return rows.groupby(["gw", "key"], as_index=False).agg(
        pred=(prediction_col, "sum"), actual=("total_points", "sum"))


def ranking_metrics(rows, prediction_col):
    if rows.empty or not np.isfinite(rows[[prediction_col, "total_points"]].to_numpy(float)).all():
        raise ValueError("Evaluation requires finite predictions and actual points")
    grouped = player_gameweeks(rows, prediction_col)
    correlations, precisions = [], []
    for _, gameweek in grouped.groupby("gw"):
        if gameweek.pred.nunique() > 1 and gameweek.actual.nunique() > 1:
            correlations.append(spearmanr(gameweek.pred, gameweek.actual).statistic)
        k = min(10, len(gameweek))
        predicted = set(gameweek.nlargest(k, "pred")["key"])
        actual = set(gameweek.nlargest(k, "actual")["key"])
        precisions.append(len(predicted & actual) / k)
    return {"mae_player_gw": float(mean_absolute_error(grouped.actual, grouped.pred)),
            "spearman_player_gw": float(np.mean(correlations)) if correlations else None,
            "precision_at_10_player_gw": float(np.mean(precisions))}
