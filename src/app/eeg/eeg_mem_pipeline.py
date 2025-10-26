# -*- coding: utf-8 -*-
"""
eeg_mem_pipeline.py

A combined pipeline for EEG-based memory readiness classification that supports:
- B: Out-of-fold (OOF) probability evaluation, calibration curve plotting, and threshold search
- A: Full retrain with calibrated LinearSVC, save/load, and online predict_proba()

Usage
-----
As library:
    from eeg_mem_pipeline import (
        oof_probabilities, scan_threshold, train_calibrated_linear_svc,
        evaluate_metrics, save_model_bundle, load_model_bundle,
        plot_calibration, online_predict, pick_cv_splitter
    )
    # X: np.ndarray (n_samples, n_features), y: np.ndarray (n_samples,)
    proba_oof = oof_probabilities(X, y, n_splits=5, splitter='stratified')
    thr, thr_info = scan_threshold(y_true=y, proba=proba_oof, metric='f1_macro')
    model = train_calibrated_linear_svc(X, y, cv_calibration=3, pca_mode='keep_components', pca_param=20)
    save_model_bundle('eeg_mem_model.pkl', model, thr, meta={'pca_mode':'keep_components', 'pca_param':20})
    model2, thr2, meta = load_model_bundle('eeg_mem_model.pkl')
    p = online_predict(model2, x_new=np.random.randn(X.shape[1]), threshold=thr2)

As script (demo with synthetic data):
    python eeg_mem_pipeline.py

Dependencies
------------
numpy, pandas, scikit-learn, matplotlib, joblib
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Literal, List

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix,
    roc_curve, balanced_accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, TimeSeriesSplit, cross_val_predict, train_test_split
)
import matplotlib.pyplot as plt

# -------------------------------
# Feature documentation constants
# -------------------------------
# Memory readiness features are engineered EEG descriptors grouped as follows:
# 1. Spectral power per band (theta, alpha, beta, gamma) per channel and window statistics.
# 2. Cross-band ratios (theta/alpha, beta/alpha, engagement indices) capturing cognitive state shifts.
# 3. Temporal stability metrics (rolling variance, EWMA drift, spectral entropy) indicating sustained readiness.
# 4. Signal hygiene flags such as contact quality, motion gating, blink artefact counts, and impedance surrogates.
MEMORY_READINESS_FEATURE_NOTES: List[str] = [
    "Band power aggregations for theta (4-7 Hz), alpha (8-12 Hz), beta (13-30 Hz), gamma (30-45 Hz) per electrode.",
    "Cross-band ratios (theta/alpha, beta/alpha, beta/(alpha+theta)) capturing engagement vs. drowsiness balance.",
    "Temporal descriptors including rolling variance, spectral entropy, peak frequency drift, and EWMA trend deltas.",
    "Signal-quality indicators: contact impedance proxies, motion/IMU veto flags, blink/artefact density, noise floors.",
]


@dataclass
class TrainingConfig:
    """Hyperparameter bundle used to train the memory readiness model.

    Inputs:
        n_splits: Cross-validation folds for out-of-fold probability estimation.
        splitter: Cross-validation splitter strategy ('stratified'|'group'|'timeseries').
        pca_mode: PCA strategy passed to the base pipeline ('keep_components'|'variance_ratio'|'off').
        pca_param: PCA parameter interpreted according to `pca_mode`.
        calibration_method: Platt scaling strategy for `CalibratedClassifierCV`.
        cv_for_calibration: Inner folds used for calibration.
        C: LinearSVC regularization strength.
        max_iter: Maximum iterations for LinearSVC optimization.
        threshold_method: Strategy employed by `find_optimal_threshold`.
        threshold_step: Resolution when scanning thresholds for non-ROC methods.
        min_positive: Minimum positives required before threshold search.
        random_state: Reproducibility seed for stratified CV.
    Outputs:
        Values are consumed by `train_memory_readiness_bundle` to orchestrate training.
    """
    n_splits: int = 5
    splitter: str = "stratified"
    pca_mode: str = "keep_components"
    pca_param: int | float = 20
    calibration_method: str = "sigmoid"
    cv_for_calibration: int = 3
    C: float = 1.0
    max_iter: int = 5000
    threshold_method: Method = "youden"
    threshold_step: float = 0.01
    min_positive: int = 1
    random_state: int = 42


# -----------------------------
# Utility: choose a CV splitter
# -----------------------------
def pick_cv_splitter(n_splits: int = 5,
                     splitter: str = 'stratified',
                     groups: Optional[np.ndarray] = None,
                     random_state: int = 42):
    """
    Choose a cross-validation splitter.
    splitter: 'stratified' | 'group' | 'timeseries'
    """
    splitter = splitter.lower()
    if splitter == 'group':
        if groups is None:
            raise ValueError("groups must be provided when splitter='group'")
        return GroupKFold(n_splits=n_splits), groups
    elif splitter == 'timeseries':
        return TimeSeriesSplit(n_splits=n_splits), None
    else:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), None


# ----------------------------------
# B) Get Out-of-Fold probabilities
# ----------------------------------
def _make_base_pipeline(n_features: int,
                        pca_mode: str = 'keep_components',
                        pca_param: int | float = 20,
                        C: float = 1.0,
                        max_iter: int = 5000):
    """
    Build the preprocessing + LinearSVC pipeline (without calibration).
    pca_mode:
        - 'keep_components' : keep exactly int(pca_param) components
        - 'variance_ratio'  : keep float(pca_param) variance (e.g., 0.95)
        - 'off'             : no PCA
    """
    steps = [('scaler', StandardScaler())]

    if pca_mode == 'keep_components':
        n_comp = max(1, int(min(pca_param, n_features)))
        steps.append(('pca', PCA(n_components=n_comp)))
    elif pca_mode == 'variance_ratio':
        keep = float(pca_param)
        steps.append(('pca', PCA(n_components=keep)))
    elif pca_mode == 'off':
        pass
    else:
        raise ValueError("pca_mode must be one of {'keep_components','variance_ratio','off'}")

    # dual=False is faster when n_samples > n_features; we don't know it here, will set at fit time via wrapper
    clf = LinearSVC(C=C, class_weight='balanced', dual=True, max_iter=max_iter)
    steps.append(('svm', clf))
    return Pipeline(steps)


class _DualFlagLinearSVC(LinearSVC):
    """Wrapper to set dual flag at fit time based on X shape."""
    def fit(self, X, y, sample_weight=None):
        # if n_samples > n_features -> dual=False is faster
        n_samples, n_features = X.shape
        self.dual = not (n_samples > n_features)
        return super().fit(X, y, sample_weight=sample_weight)


def _make_base_pipeline_dualaware(n_features: int,
                                  pca_mode: str = 'keep_components',
                                  pca_param: int | float = 20,
                                  C: float = 1.0,
                                  max_iter: int = 5000):
    steps = [('scaler', StandardScaler())]
    if pca_mode == 'keep_components':
        n_comp = max(1, int(min(pca_param, n_features)))
        steps.append(('pca', PCA(n_components=n_comp)))
    elif pca_mode == 'variance_ratio':
        keep = float(pca_param)
        steps.append(('pca', PCA(n_components=keep)))
    elif pca_mode == 'off':
        pass
    else:
        raise ValueError("pca_mode must be one of {'keep_components','variance_ratio','off'}")

    clf = _DualFlagLinearSVC(C=C, class_weight='balanced', max_iter=max_iter)
    steps.append(('svm', clf))
    return Pipeline(steps)


def oof_probabilities(X: np.ndarray,
                      y: np.ndarray,
                      n_splits: int = 5,
                      splitter: str = 'stratified',
                      groups: Optional[np.ndarray] = None,
                      pca_mode: str = 'keep_components',
                      pca_param: int | float = 20,
                      calibration_method: str = 'sigmoid',
                      cv_for_calibration: int = 3,
                      C: float = 1.0,
                      max_iter: int = 5000,
                      random_state: int = 42) -> np.ndarray:
    """
    Compute Out-Of-Fold probabilities with a calibrated LinearSVC.
    Returns a vector proba_oof with length n_samples (probability of positive class for binary).
    For multiclass, returns the probability of the last class; adapt as needed.
    C and max_iter mirror the LinearSVC hyperparameters so the CV pipeline matches the final model.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.ndim != 2:
        raise ValueError("X must be 2D array (n_samples, n_features)")

    base = _make_base_pipeline_dualaware(
        n_features=X.shape[1],
        pca_mode=pca_mode,
        pca_param=pca_param,
        C=C,
        max_iter=max_iter
    )
    calibrated = CalibratedClassifierCV(base, method=calibration_method, cv=cv_for_calibration)

    cv_splitter, g = pick_cv_splitter(
        n_splits=n_splits,
        splitter=splitter,
        groups=groups,
        random_state=random_state
    )
    proba_oof = cross_val_predict(
        calibrated, X, y,
        cv=cv_splitter,
        method='predict_proba',
        groups=g,
        n_jobs=None
    )
    # Return probability of the last class (works for binary OvR)
    return proba_oof[:, -1]


# ----------------------------------------
# Threshold search & basic metric summaries
# ----------------------------------------
def _youden_index(tpr: float, fpr: float) -> float:
    return tpr - fpr

def scan_threshold(y_true: np.ndarray,
                   proba: np.ndarray,
                   metric: str = 'f1_macro') -> Tuple[float, Dict[str, float]]:
    """
    Scan threshold from 0.05..0.95 to select the best one by metric:
    - 'f1_macro': maximize macro F1
    - 'youden'  : maximize Youden's J statistic (TPR - FPR)
    Returns: (best_threshold, info_dict)
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()

    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr, best_score = 0.5, -1.0
    best_info = {}

    for thr in thresholds:
        y_hat = (proba >= thr).astype(int)
        acc = accuracy_score(y_true, y_hat)
        f1m = f1_score(y_true, y_hat, average='macro')

        if metric == 'youden':
            # need confusion entries
            tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            score = _youden_index(tpr, fpr)
        else:
            score = f1m

        if score > best_score:
            best_score = score
            best_thr = thr
            best_info = {'acc': acc, 'f1_macro': f1m, 'score': score}

    return float(best_thr), best_info


Method = Literal["youden", "f1", "balanced_accuracy"]


def find_optimal_threshold(
    scores: np.ndarray,       # model output probabilities/scores in [0,1]
    labels: np.ndarray,       # true labels (0/1)
    method: Method = "youden",
    step: float = 0.01,       # threshold resolution
    min_pos: int = 1          # minimum positive samples required
) -> Tuple[float, Dict[str, float]]:
    """
    Return the optimal threshold and corresponding metrics.
    method:
      - "youden": maximize TPR - FPR using ROC curve (recommended)
      - "f1": scan for maximum F1
      - "balanced_accuracy": scan for maximum balanced accuracy
    """
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    assert scores.shape == labels.shape, "scores/labels shape mismatch"
    assert set(np.unique(labels)).issubset({0, 1}), "labels must be 0/1"

    # Safety: if not enough positive samples, return default 0.5
    if labels.sum() < min_pos:
        return 0.5, {"reason": "too_few_positives"}

    # Method 1: Youden (uses ROC, smooth and recommended)
    if method == "youden":
        fpr, tpr, ths = roc_curve(labels, scores)
        j = tpr - fpr
        i = int(np.argmax(j))
        thr = ths[i]
        y_pred = (scores >= thr).astype(int)
        return float(thr), {
            "tpr": float(tpr[i]),
            "fpr": float(fpr[i]),
            "youden_j": float(j[i]),
            "f1": float(f1_score(labels, y_pred)),
            "balanced_acc": float(balanced_accuracy_score(labels, y_pred)),
            "acc": float(accuracy_score(labels, y_pred)),
            "precision": float(precision_score(labels, y_pred, zero_division=0)),
            "recall": float(recall_score(labels, y_pred, zero_division=0)),
        }

    # Methods 2/3: grid search
    grid = np.arange(0.05, 0.95 + 1e-9, step)
    best_thr, best_val, best_stats = 0.5, -1.0, {}
    for thr in grid:
        y_pred = (scores >= thr).astype(int)
        prec = precision_score(labels, y_pred, zero_division=0)
        rec = recall_score(labels, y_pred, zero_division=0)
        f1v = f1_score(labels, y_pred, zero_division=0)
        ba = balanced_accuracy_score(labels, y_pred)
        acc = accuracy_score(labels, y_pred)

        val = f1v if method == "f1" else ba
        if val > best_val:
            best_val = val
            best_thr = thr
            best_stats = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1v),
                "balanced_acc": float(ba),
                "acc": float(acc),
            }
    return float(best_thr), best_stats


def evaluate_metrics(y_true: np.ndarray,
                     proba: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, float]:
    """Compute standard metrics (ACC, F1, AUC, Brier) at a given threshold."""
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()
    y_hat = (proba >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, y_hat),
        'f1_macro': f1_score(y_true, y_hat, average='macro'),
        'brier': brier_score_loss(y_true, proba)
    }
    # AUC only for binary labels with at least one pos/neg
    try:
        metrics['auc'] = roc_auc_score(y_true, proba)
    except Exception:
        metrics['auc'] = float('nan')
    return metrics


# -----------------------------
# Plot calibration curve helper
# -----------------------------
def plot_calibration(y_true: np.ndarray,
                     proba: np.ndarray,
                     save_path: Optional[str] = None,
                     n_bins: int = 10) -> None:
    """
    Plot calibration curve (reliability diagram). If save_path is provided, save to file.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy='uniform')

    plt.figure(figsize=(5, 5))
    # Perfectly calibrated
    plt.plot([0, 1], [0, 1])
    # Model
    plt.plot(mean_pred, frac_pos, marker='o')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.grid(True)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


# ------------------------------------
# A) Train full calibrated LinearSVC
# ------------------------------------
def train_calibrated_linear_svc(X: np.ndarray,
                                y: np.ndarray,
                                cv_calibration: int = 3,
                                pca_mode: str = 'keep_components',
                                pca_param: int | float = 20,
                                calibration_method: str = 'sigmoid',
                                C: float = 1.0,
                                max_iter: int = 5000) -> CalibratedClassifierCV:
    """
    Train a calibrated LinearSVC on full data. Returns a model that supports predict_proba().
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    base = _make_base_pipeline_dualaware(
        n_features=X.shape[1],
        pca_mode=pca_mode,
        pca_param=pca_param,
        C=C,
        max_iter=max_iter
    )
    model = CalibratedClassifierCV(base, method=calibration_method, cv=cv_calibration)
    model.fit(X, y)
    return model


def train_memory_readiness_bundle(
    X: np.ndarray,
    y: np.ndarray,
    config: Optional[TrainingConfig] = None,
    groups: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    feature_notes: Optional[List[str]] = None
) -> ModelBundle:
    """Train, calibrate, and package the memory readiness model in one call.

    Inputs:
        X: Feature matrix shaped (n_samples, n_features) with engineered EEG descriptors.
        y: Binary labels (0/1) indicating memory readiness ground truth.
        config: Optional `TrainingConfig` controlling CV, PCA, and threshold behaviour.
        groups: Optional grouping array, forwarded when `config.splitter == 'group'`.
        save_path: Optional prefix for persisted bundle (`<path>.model`/`.json`).
        meta_path: Optional path for the human-readable metadata JSON; defaults to `<save_path>.meta.json`.
        feature_notes: Override for the default `MEMORY_READINESS_FEATURE_NOTES`.
    Outputs:
        ModelBundle containing the calibrated classifier, optimal threshold, and metadata dict.
    """
    cfg = config or TrainingConfig()
    notes = feature_notes or MEMORY_READINESS_FEATURE_NOTES

    proba_oof = oof_probabilities(
        X=X,
        y=y,
        n_splits=cfg.n_splits,
        splitter=cfg.splitter,
        groups=groups,
        pca_mode=cfg.pca_mode,
        pca_param=cfg.pca_param,
        calibration_method=cfg.calibration_method,
        cv_for_calibration=cfg.cv_for_calibration,
        C=cfg.C,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state
    )
    thr, thr_stats = find_optimal_threshold(
        scores=proba_oof,
        labels=y,
        method=cfg.threshold_method,
        step=cfg.threshold_step,
        min_pos=cfg.min_positive
    )
    eval_metrics = evaluate_metrics(y_true=y, proba=proba_oof, threshold=thr)
    metrics_payload: Dict[str, object] = {k: float(v) for k, v in eval_metrics.items()}
    for key, val in thr_stats.items():
        if key not in metrics_payload:
            if isinstance(val, (float, int, np.floating, np.integer)):
                metrics_payload[key] = float(val)
            else:
                metrics_payload[key] = val

    model = train_calibrated_linear_svc(
        X=X,
        y=y,
        cv_calibration=cfg.cv_for_calibration,
        pca_mode=cfg.pca_mode,
        pca_param=cfg.pca_param,
        calibration_method=cfg.calibration_method,
        C=cfg.C,
        max_iter=cfg.max_iter
    )

    trained_on = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: Dict[str, object] = {
        "model_name": "EEG Memory Readiness (Calibrated LinearSVC)",
        "trained_on": trained_on,
        "features": int(X.shape[1]),
        "classifier": "LinearSVC (StandardScaler+PCA, calibrated)",
        "best_thr": float(thr),
        "threshold_method": cfg.threshold_method,
        "metrics": metrics_payload,
        "feature_notes": notes,
        "config": asdict(cfg),
    }

    bundle = ModelBundle(model=model, threshold=float(thr), meta=meta)

    if save_path:
        save_model_bundle(save_path, model, thr, meta=meta)
        meta_out = meta_path or f"{save_path}.meta.json"
        save_model_meta(
            thr,
            stats=metrics_payload,
            method=cfg.threshold_method,
            features=int(X.shape[1]),
            trained_on=trained_on,
            model_name=meta["model_name"],  # type: ignore[index]
            out_path=meta_out,
            feature_notes=notes,
        )

    return bundle


# -----------------
# Model persistence
# -----------------
@dataclass
class ModelBundle:
    model: CalibratedClassifierCV
    threshold: float
    meta: Dict

def save_model_bundle(path: str,
                      model: CalibratedClassifierCV,
                      threshold: float,
                      meta: Optional[Dict] = None) -> None:
    """Save model + threshold + meta to a single file path."""
    bundle = {
        'threshold': float(threshold),
        'meta': meta or {}
    }
    dump(model, path + '.model')
    with open(path + '.json', 'w', encoding='utf-8') as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

def load_model_bundle(base_path: str = "model_store/eeg_mem_model", strict: bool = False) -> Tuple[Optional[CalibratedClassifierCV], float, Dict]:
    """
    Load a trained scikit-learn Pipeline model and its JSON metadata as a 'bundle'.

    Parameters
    ----------
    base_path : str
        Path prefix of the bundle (without extension). The function will look for:
        - <base_path>.model (joblib dump)
        - <base_path>.json  (metadata)
    strict : bool
        If True, missing files will raise FileNotFoundError. If False, the function
        returns (None, 0.5, {}) or fills reasonable defaults.

    Returns
    -------
    (model, best_thr, meta) : tuple
        model : object | None
            The loaded sklearn Pipeline (or None if not found and strict=False).
        best_thr : float
            The chosen/default threshold (meta.best_thr or 0.5)
        meta : dict
            Metadata dictionary; guaranteed to contain at least:
              - "model_name"
              - "trained_on"
              - "features" (best-effort)
              - "classifier"
              - "threshold_method"
              - "best_thr"
              - "metrics" (dict)
              - "path_model"
              - "path_meta"
    """
    import os
    import json
    import datetime as _dt

    try:
        import joblib
    except Exception as e:
        # If joblib isn't available, fall back to previously imported `load` where possible
        joblib = None

    path_model = f"{base_path}.model"
    path_meta = f"{base_path}.json"

    # 1) Load model
    model = None
    if os.path.exists(path_model):
        try:
            if joblib is not None:
                model = joblib.load(path_model)
            else:
                # fallback to the already-imported load from joblib
                model = load(path_model)
        except Exception:
            if strict:
                raise
            model = None
    elif strict:
        raise FileNotFoundError(f"Model file not found: {path_model}")

    # 2) Load metadata
    meta = {}
    if os.path.exists(path_meta):
        try:
            with open(path_meta, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception:
            if strict:
                raise
            meta = {}
    elif strict:
        raise FileNotFoundError(f"Meta json not found: {path_meta}")

    # 3) Fill reasonable defaults / best-effort inference
    meta.setdefault("model_name", "EEG Memory Readiness v1.0")
    meta.setdefault("trained_on", _dt.date.today().isoformat())
    meta.setdefault("threshold_method", meta.get("threshold_method", "youden"))

    # threshold fallback
    best_thr = meta.get("best_thr", 0.5)
    try:
        best_thr = float(best_thr)
    except Exception:
        best_thr = 0.5
    meta["best_thr"] = best_thr

    # classifier name inference
    def _infer_classifier(m):
        try:
            if hasattr(m, "named_steps") and m.named_steps:
                last_est = list(m.named_steps.values())[-1]
                return type(last_est).__name__
            return type(m).__name__
        except Exception:
            return meta.get("classifier", "UnknownClassifier")

    if model is not None:
        meta.setdefault("classifier", _infer_classifier(model))
    else:
        meta.setdefault("classifier", meta.get("classifier", "UnknownClassifier"))

    # features inference
    def _infer_features(m):
        if m is None:
            return meta.get("features", None)
        try:
            if hasattr(m, "named_steps") and "scaler" in m.named_steps:
                n = getattr(m.named_steps["scaler"], "n_features_in_", None)
                if n is not None:
                    return int(n)
            if hasattr(m, "steps") and m.steps:
                n = getattr(m.steps[0][1], "n_features_in_", None)
                if n is not None:
                    return int(n)
            n = getattr(m, "n_features_in_", None)
            if n is not None:
                return int(n)
        except Exception:
            pass
        return meta.get("features", None)

    features = meta.get("features", None)
    if features is None:
        features = _infer_features(model)
    meta["features"] = int(features) if features is not None else 0

    # metrics fallback
    metrics = meta.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    meta["metrics"] = metrics

    # attach paths
    meta["path_model"] = path_model
    meta["path_meta"] = path_meta

    return model, float(best_thr), meta


def save_model_meta(thr,
                    stats: Dict,
                    method: str = "youden",
                    features: Optional[int] = None,
                    trained_on: Optional[str] = None,
                    model_name: str = "EEG Memory Readiness v1.0",
                    out_path: str = "model_store/eeg_mem_model.json",
                    feature_notes: Optional[List[str]] = None) -> None:
    """
    Save simple JSON metadata about the trained model (human-readable summary).

    Parameters
    - thr: chosen threshold
    - stats: metric dict returned from find_optimal_threshold or evaluate_metrics
    - method: threshold selection method name
    - features: optional number of features (inferred from X if provided)
    - trained_on: ISO date string; defaults to now
    - model_name: friendly name for the model
    - out_path: file path to write JSON metadata
    - feature_notes: optional list documenting the engineered feature families
    """
    if trained_on is None:
        trained_on = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    meta = {
        "model_name": model_name,
        "trained_on": trained_on,
        "features": int(features) if features is not None else None,
        "classifier": "LinearSVC (calibrated)",
        "best_thr": float(thr),
        "threshold_method": method,
        "metrics": stats,
        "feature_notes": feature_notes or MEMORY_READINESS_FEATURE_NOTES,
    }
    Path(os.path.dirname(out_path) or "model_store").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"✅ Saved model meta to {out_path}")


def train_and_save_pipeline(X_train: np.ndarray,
                            y_train: np.ndarray,
                            path: str = "model_store/eeg_mem_model",
                            config: Optional[TrainingConfig] = None,
                            groups: Optional[np.ndarray] = None) -> ModelBundle:
    """Convenience wrapper that trains and persists the memory readiness model.

    Inputs:
        X_train: Training feature matrix for EEG-derived memory readiness signals.
        y_train: Corresponding binary targets (0=not ready, 1=ready).
        path: Output prefix for persisted artifacts.
        config: Optional training configuration override.
        groups: Optional grouping used when performing grouped CV splits.
    Outputs:
        ModelBundle mirroring what was saved on disk for downstream reuse.
    """
    Path(os.path.dirname(path) or "model_store").mkdir(parents=True, exist_ok=True)
    bundle = train_memory_readiness_bundle(
        X=X_train,
        y=y_train,
        config=config,
        groups=groups,
        save_path=path,
        meta_path=path + ".meta.json",
        feature_notes=MEMORY_READINESS_FEATURE_NOTES,
    )
    print(f"✅ Saved EEG model to {path}.model and metadata to {path}.json / {path}.meta.json")
    return bundle


# --------------
# Online predict
# --------------
def online_predict(model: CalibratedClassifierCV,
                   x_new: np.ndarray,
                   threshold: float,
                   user_threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Predict online with probability & decision.
    Returns dict with prob, threshold_used, label_hat, confidence(|p-0.5|).
    """
    x_new = np.asarray(x_new).reshape(1, -1)
    p = float(model.predict_proba(x_new)[0, -1])
    thr = float(user_threshold) if user_threshold is not None else float(threshold)
    label_hat = int(p >= thr)
    confidence = abs(p - 0.5) * 2.0  # 0..1 scale
    return {
        'prob': p,
        'threshold_used': thr,
        'label_hat': label_hat,
        'confidence': confidence
    }


# ------------------------
# Demo: end-to-end example
# ------------------------
def _demo_synthetic(seed: int = 7) -> None:
    """
    Create a synthetic EEG-like dataset and run B + A pipeline demonstration.
    """
    rng = np.random.default_rng(seed)
    n_samples = 400
    n_features = 30
    X = rng.normal(size=(n_samples, n_features))
    # simulate a linear+nonlinear signal for binary target
    w = rng.normal(size=n_features)
    z = X @ w + 0.5 * np.tanh(X[:, :3].sum(axis=1))
    y = (z > np.median(z)).astype(int)

    # --- B: OOF probabilities & threshold scan
    proba_oof = oof_probabilities(X, y, n_splits=5, splitter='stratified',
                                  pca_mode='keep_components', pca_param=20,
                                  calibration_method='sigmoid', cv_for_calibration=3)
    thr, info = scan_threshold(y_true=y, proba=proba_oof, metric='f1_macro')
    metrics = evaluate_metrics(y_true=y, proba=proba_oof, threshold=thr)
    print("[B] Best threshold:", thr, "| info:", info)
    print("[B] Metrics @thr:", metrics)

    # plot calibration curve
    plot_path = 'calibration_curve_demo.png'
    plot_calibration(y_true=y, proba=proba_oof, save_path=plot_path, n_bins=10)
    print(f"[B] Calibration curve saved to: {plot_path}")

    # --- A: Full retrain with calibration & save
    model = train_calibrated_linear_svc(X, y, cv_calibration=3,
                                        pca_mode='keep_components', pca_param=20)
    save_model_bundle('eeg_mem_model', model, thr, meta={'pca_mode':'keep_components','pca_param':20})
    print("[A] Model saved to: eeg_mem_model.model + eeg_mem_model.json")

    # --- Online predict
    x_new = rng.normal(size=n_features)
    out = online_predict(model, x_new, threshold=thr)
    print("[Online] prob={prob:.3f}, label={label_hat}, thr={threshold_used:.2f}, conf={confidence:.2f}".format(**out))


if __name__ == "__main__":
    _demo_synthetic()
