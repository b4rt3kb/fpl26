"""Chronological, three-state minutes model with held-out probability calibration.

States are no appearance (0), cameo (1–59), and 60+ (60–90 minutes).
Calibration reserves the latest complete deadline blocks; neither base learners
nor conditional-minute regressors fit their labels. Overlapping earlier-origin
targets are purged. Calibration diagnostics are fitting diagnostics, not a test
set estimate: assess quality on a later, untouched forecast origin.

The model accepts frozen-origin training rows from build_season(as_of=...).
Ordinary weekly rows alone do not validate long-horizon forecasts. Diagnostics
report the fitted lead range, and predictions flag inputs outside that range.
Schedule features cover the supplied PL calendar only, not cups/internationals;
injury information must come from an origin-time availability layer.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss

from fpl_features import MINUTES_FEATURES

PROBABILITY_COLUMNS = ["p_no_appearance", "p_1_59", "p_60_plus"]
OUTPUT_COLUMNS = PROBABILITY_COLUMNS + ["expected_minutes", "p_start",
                                       "minutes_if_1_59", "minutes_if_60_plus",
                                       "lead_out_of_support"]


def minute_states(minutes):
    values = np.asarray(minutes, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 90)).any():
        raise ValueError("Minutes targets must be finite and between 0 and 90")
    return np.where(values == 0, 0, np.where(values < 60, 1, 2))


class _ProbabilityCalibrator:
    """Low-dimensional temperature + class-intercept calibration on later data.

    The reference-class intercept is fixed at zero. L2 shrinkage toward the
    identity map limits sensitivity to small samples; no class reweighting is
    used, because the output must retain the real appearance probabilities.
    """
    def fit(self, probabilities, labels):
        p = np.clip(np.asarray(probabilities, float), 1e-8, 1)
        y = np.asarray(labels, int)
        self.parameters_ = np.zeros(p.shape[1])
        logits = np.log(p)

        def objective(theta):
            temperature = np.exp(theta[0])
            offsets = np.r_[0., theta[1:]]
            q = softmax(logits/temperature + offsets, axis=1)
            return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1)).mean() + .005*np.square(theta).sum()

        result = minimize(objective, self.parameters_, method="L-BFGS-B",
                          bounds=[(-1.6, 1.6)] + [(-3., 3.)]*(p.shape[1]-1))
        self.success_ = bool(result.success and np.isfinite(result.x).all())
        if self.success_:
            self.parameters_ = result.x
        return self

    def predict(self, probabilities):
        p = np.clip(np.asarray(probabilities, float), 1e-8, 1)
        return softmax(np.log(p)/np.exp(self.parameters_[0]) + np.r_[0., self.parameters_[1:]], axis=1)


class MinutesModel:
    """fit(rows) -> self; predict(rows) -> indexed per-fixture DataFrame.

    Unknown optional features remain NaN, as they do in build_season. Target
    timestamps are mandatory during fitting. Set calibrate=False explicitly for
    a controlled ablation; small data automatically uses an uncalibrated model.
    """
    def __init__(self, *, calibrate=True, calibration_events=4, min_train_rows=1500,
                 min_calibration_rows=500, min_class_count=20, max_iter=150,
                 random_state=42):
        if calibration_events < 1 or min_train_rows < 1 or min_calibration_rows < 1 or min_class_count < 1 or max_iter < 1:
            raise ValueError("Minutes-model training limits must be positive")
        self.calibrate = calibrate
        self.calibration_events = calibration_events
        self.min_train_rows = min_train_rows
        self.min_calibration_rows = min_calibration_rows
        self.min_class_count = min_class_count
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _features(rows):
        features = rows.reindex(columns=MINUTES_FEATURES).apply(pd.to_numeric, errors="raise")
        if np.isinf(features.to_numpy(float)).any():
            raise ValueError("Infinite minutes feature")
        return features

    def _classifier(self):
        # Fixed rounds avoid a random internal validation split across time.
        return HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=.06,
                max_leaf_nodes=15, max_depth=4, min_samples_leaf=40, l2_regularization=5.,
                early_stopping=False, random_state=self.random_state)

    def _regressor(self):
        return HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=.06,
                max_leaf_nodes=15, max_depth=4, min_samples_leaf=40, l2_regularization=5.,
                early_stopping=False, random_state=self.random_state)

    @staticmethod
    def _expanded_probability(model, prior, features, n_classes):
        if model is None:
            return np.tile(prior, (len(features), 1))
        probability = np.zeros((len(features), n_classes))
        probability[:, model.classes_.astype(int)] = model.predict_proba(features)
        return probability

    def fit(self, rows):
        if rows.empty or "minutes" not in rows:
            raise ValueError("Minutes model needs completed training rows")
        states = minute_states(rows.minutes)
        if not {"feature_cutoff", "kickoff_time"}.issubset(rows.columns):
            raise ValueError("Chronological minutes fit needs feature_cutoff and kickoff_time")
        if "_forecast" in rows and rows._forecast.fillna(False).astype(bool).any():
            raise ValueError("Synthetic forecast targets cannot enter minutes training")
        if "finished" in rows and (rows.finished.notna() & ~rows.finished.isin([True, 1, "True"])).any():
            raise ValueError("Unfinished fixtures cannot enter minutes training")
        cutoff = pd.to_datetime(rows.feature_cutoff, utc=True, errors="raise")
        kickoff = pd.to_datetime(rows.kickoff_time, utc=True, errors="raise")
        if cutoff.isna().any() or kickoff.isna().any() or (cutoff > kickoff).any():
            raise ValueError("Invalid chronology for minutes features")
        X = self._features(rows)
        base = np.ones(len(rows), dtype=bool)
        calibration = np.zeros(len(rows), dtype=bool)
        reason = "disabled" if not self.calibrate else "insufficient chronological calibration data"
        unique = np.sort(cutoff.unique())
        boundary = None
        purged = 0
        if self.calibrate and len(unique) > self.calibration_events:
            boundary = pd.Timestamp(unique[-self.calibration_events])
            candidate_base = (cutoff < boundary).to_numpy()
            completed = (kickoff + pd.Timedelta(hours=2) <= boundary).to_numpy()
            purged = int((candidate_base & ~completed).sum())
            candidate_base &= completed
            candidate_cal = (cutoff >= boundary).to_numpy()
            counts_base = np.bincount(states[candidate_base], minlength=3)
            counts_cal = np.bincount(states[candidate_cal], minlength=3)
            if (candidate_base.sum() >= self.min_train_rows and candidate_cal.sum() >= self.min_calibration_rows
                    and (counts_base >= self.min_class_count).all() and (counts_cal >= self.min_class_count).all()):
                base, calibration = candidate_base, candidate_cal
                reason = "temporal holdout reserved"
        self.base_indices_ = rows.index[base].to_numpy()
        self.calibration_indices_ = rows.index[calibration].to_numpy()
        self.classifier_ = None
        counts = np.bincount(states[base], minlength=3)
        self.class_prior_ = (counts+.5)/(counts.sum()+1.5)
        if np.unique(states[base]).size > 1:
            self.classifier_ = self._classifier().fit(X.loc[base], states[base])
        self.calibrator_ = None
        if calibration.any():
            raw = self._expanded_probability(self.classifier_, self.class_prior_, X.loc[calibration], 3)
            candidate = _ProbabilityCalibrator().fit(raw, states[calibration])
            if candidate.success_:
                self.calibrator_ = candidate
                reason = "temporal holdout temperature and class intercepts"
            else:
                reason = "calibration optimization failed; raw probabilities retained"

        self.minute_regressors_ = {}
        self.minute_means_ = {}
        for state, prior_mean in ((1, 30.), (2, 85.)):
            selected = base & (states == state)
            minutes = rows.minutes.to_numpy(float)[selected]
            self.minute_means_[state] = float((minutes.sum()+20*prior_mean)/(len(minutes)+20))
            self.minute_regressors_[state] = (self._regressor().fit(X.loc[selected], minutes)
                                                if len(minutes) >= 100 else None)

        # Start is conditional on an appearance so P(start) cannot exceed P(play).
        self.start_classifier_, self.start_calibrator_, self.start_prior_ = None, None, None
        starts = pd.to_numeric(rows.get("starts", pd.Series(np.nan, index=rows.index)), errors="coerce")
        known_starts = starts.notna().to_numpy()
        if not starts[starts.notna()].isin([0, 1]).all():
            raise ValueError("Known starts targets must be binary")
        if (known_starts & starts.eq(1).to_numpy() & (states == 0)).any():
            raise ValueError("Start recorded with zero minutes; inspect source before fitting")
        start_fit = base & (states > 0) & known_starts
        if start_fit.sum() >= self.min_class_count:
            labels = starts.to_numpy(float)[start_fit].astype(int)
            c = np.bincount(labels, minlength=2)
            self.start_prior_ = (c+.5)/(c.sum()+1)
            if len(np.unique(labels)) > 1:
                self.start_classifier_ = self._classifier().fit(X.loc[start_fit], labels)
            start_cal = calibration & (states > 0) & known_starts
            counts_cal = np.bincount(starts.to_numpy(float)[start_cal].astype(int), minlength=2)
            if (counts_cal >= self.min_class_count).all():
                p = self._expanded_probability(self.start_classifier_, self.start_prior_, X.loc[start_cal], 2)
                candidate = _ProbabilityCalibrator().fit(p, starts.to_numpy(float)[start_cal].astype(int))
                if candidate.success_:
                    self.start_calibrator_ = candidate

        lead = X.loc[base, "forecast_lead_days"].dropna()
        self.lead_range_ = (float(lead.min()), float(lead.max())) if len(lead) else None
        self.diagnostics_ = {"base_rows": int(base.sum()), "calibration_rows": int(calibration.sum()),
            "purged_overlap_rows": purged, "calibration_applied": self.calibrator_ is not None,
            "calibration_reason": reason, "calibration_start": boundary.isoformat() if calibration.any() else None,
            "base_latest_completion": (kickoff[base].max()+pd.Timedelta(hours=2)).isoformat(),
            "training_lead_days_range": self.lead_range_, "start_supported": self.start_prior_ is not None,
            "class_counts_base": counts.tolist(),
            "calibration_metrics_scope": "calibrator fitting data; not independent test performance"}
        if calibration.any():
            calibrated = self.predict(rows.loc[calibration])
            truth = np.eye(3)[states[calibration]]
            for name, p in (("raw", raw), ("calibrated", calibrated[PROBABILITY_COLUMNS].to_numpy())):
                self.diagnostics_[f"{name}_log_loss"] = float(log_loss(states[calibration], p, labels=[0, 1, 2]))
                self.diagnostics_[f"{name}_brier"] = float(np.square(p-truth).sum(axis=1).mean())
            self.diagnostics_["calibration_minutes_mae"] = float(np.abs(
                calibrated.expected_minutes.to_numpy()-rows.minutes.to_numpy(float)[calibration]).mean())
        return self

    def predict(self, rows):
        if not hasattr(self, "class_prior_"):
            raise RuntimeError("Fit the minutes model before predicting")
        if rows.empty:
            return pd.DataFrame(index=rows.index, columns=OUTPUT_COLUMNS)
        X = self._features(rows)
        p = self._expanded_probability(self.classifier_, self.class_prior_, X, 3)
        if self.calibrator_ is not None:
            p = self.calibrator_.predict(p)
        p = np.clip(p, 0, 1)
        p /= p.sum(axis=1, keepdims=True)
        output = pd.DataFrame(p, index=rows.index, columns=PROBABILITY_COLUMNS)
        means = {}
        for state, lo, hi in ((1, 1., 59.), (2, 60., 90.)):
            regressor = self.minute_regressors_[state]
            prediction = regressor.predict(X) if regressor is not None else np.full(len(X), self.minute_means_[state])
            means[state] = np.clip(prediction, lo, hi)
        output["minutes_if_1_59"] = means[1]
        output["minutes_if_60_plus"] = means[2]
        output["expected_minutes"] = p[:, 1]*means[1] + p[:, 2]*means[2]
        output["p_start"] = np.nan
        if self.start_prior_ is not None:
            start = self._expanded_probability(self.start_classifier_, self.start_prior_, X, 2)
            if self.start_calibrator_ is not None:
                start = self.start_calibrator_.predict(start)
            output["p_start"] = (1-p[:, 0])*start[:, 1]
        lead = X.forecast_lead_days
        output["lead_out_of_support"] = (lead.lt(self.lead_range_[0]) | lead.gt(self.lead_range_[1])) if self.lead_range_ else lead.notna()
        if not np.isfinite(output[PROBABILITY_COLUMNS + ["expected_minutes"]].to_numpy()).all():
            raise ValueError("Non-finite minutes prediction")
        return output[OUTPUT_COLUMNS]
