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
        if kind not in ("hgb", "two", "minutes", "ensemble"):
            raise ValueError(f"Unknown model: {kind}")
        self.kind = kind

    def fit(self, rows, minutes_training=None):
        if rows.empty or not np.isfinite(rows["total_points"].to_numpy(float)).all():
            raise ValueError("Training requires completed matches with known targets")
        if np.isinf(rows[FEATS].to_numpy(float)).any():
            raise ValueError("Infinite training feature")
        if self.kind == "ensemble":
            self.baseline = PointsModel("hgb").fit(rows)
            self.components = PointsModel("minutes").fit(rows if minutes_training is None else minutes_training)
            self.minutes_model = self.components.minutes_model
            return self
        if self.kind == "minutes":
            from minutes_model import MinutesModel
            from fpl_features import MINUTES_FEATURES, EXTENDED_FEATURES
            self.minutes_model = MinutesModel().fit(rows)
            self.point_features = list(dict.fromkeys(FEATS + MINUTES_FEATURES + EXTENDED_FEATURES))
            self.conditional_models = {}
            for band, mask in (("p_1_59", rows.minutes.between(1, 59)),
                               ("p_60_plus", rows.minutes.ge(60))):
                if not mask.any():
                    self.conditional_models[band] = 0.0
                else:
                    estimator = hgb()
                    # Fixed iterations avoid random internal validation across time.
                    estimator.set_params(max_iter=220, early_stopping=False, l2_regularization=2.0)
                    estimator.fit(rows.loc[mask].reindex(columns=self.point_features), rows.loc[mask, "total_points"])
                    self.conditional_models[band] = estimator
            return self
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
        if self.kind in ("minutes", "ensemble"):
            return self.predict_components(rows)["pred"].to_numpy()
        predictions = self.regressor.predict(rows[FEATS])
        if self.kind == "two":
            probability = (self.constant_play_probability if self.classifier is None
                           else self.classifier.predict_proba(rows[FEATS])[:, 1])
            predictions = probability * predictions
        if not np.isfinite(predictions).all():
            raise ValueError("Model produced non-finite predictions")
        return np.clip(predictions, 0, None)

    def predict_components(self, rows):
        """Expected points from conditional returns in mutually exclusive minute bands.

        The minute labels are used only when fitting. Forecast feature matrices
        contain no actual future minutes, starts, points or other match outcomes.
        News availability is fused later, once, at the individual fixture level.
        """
        if self.kind == "ensemble":
            result = self.components.predict_components(rows)
            result["component_pred"] = result.pred
            result["direct_pred"] = self.baseline.predict(rows)
            result["pred"] = .5 * result.component_pred + .5 * result.direct_pred
            return result
        if self.kind != "minutes":
            raise ValueError("Component forecasts require the minutes model")
        result = self.minutes_model.predict(rows)
        prediction = np.zeros(len(rows), dtype=float)
        for band, estimator in self.conditional_models.items():
            conditional = (np.full(len(rows), estimator) if isinstance(estimator, float)
                           else estimator.predict(rows.reindex(columns=self.point_features)))
            result["points_given_" + band] = conditional
            prediction += result[band].to_numpy() * conditional
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite component prediction")
        result["pred"] = np.clip(prediction, 0, None)
        return result


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
