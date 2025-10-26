# src/app/eeg/train_and_save.py
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

from app.eeg import algorithms as algo  # alias for convenient calls

def _now():
    return datetime.now().strftime("%Y-%m-%d")

def train_linear_svm_bayes(X: np.ndarray, y: np.ndarray) -> Tuple[object, Dict[str, Any]]:
    pipe = algo.make_linear_svm_pipeline(pca_dim=20, C=1.0)
    # Parameter/search space for BayesSearchCV. If skopt is not installed this
    # will fall back to GridSearchCV (in that case provide discrete lists).
    space = {
        "pca__n_components": (5, 60),                   # 连续/整数搜索
        "clf__C": (1e-3, 10.0, "log-uniform"),
    }
    search, used_bayes = algo.bayes_optimize(pipe, space, X, y, cv_splits=5, n_iter=25, scoring="f1")
    best = search.best_estimator_
    # 评估（这里简单用 CV 的 best_score_；也可用 hold-out）
    meta = {
        "model_name": "EEG Memory Readiness (Bayes-tuned LinearSVC)",
        "trained_on": _now(),
        "classifier": "LinearSVC (PCA+StandardScaler)",
        "features": X.shape[1],
        "threshold_method": "youden",
        "metrics": {
            "f1": float(search.best_score_)
        },
        "search": "BayesSearchCV" if used_bayes else "GridSearchCV",
        "param_best": search.best_params_,
    }
    # We do not compute best_thr here; compute it in your offline evaluation
    # script and write it into the meta if desired.
    return best, meta

def train_mlp_bayes(X: np.ndarray, y: np.ndarray) -> Tuple[object, Dict[str, Any]]:
    pipe = algo.make_mlp_pipeline(pca_keep=0.95, hidden=(64, 32), alpha=1e-3)
    space = {
        "pca__n_components": (0.70, 0.99),              # 保留比例
        "clf__alpha": (1e-5, 1e-2, "log-uniform"),
    }
    search, used_bayes = algo.bayes_optimize(pipe, space, X, y, cv_splits=5, n_iter=25, scoring="f1")
    best = search.best_estimator_
    meta = {
        "model_name": "EEG Memory Readiness (Bayes-tuned MLP)",
        "trained_on": _now(),
        "classifier": "MLP (PCA+StandardScaler)",
        "features": X.shape[1],
        "threshold_method": "youden",
        "metrics": {
            "f1": float(search.best_score_)
        },
        "search": "BayesSearchCV" if used_bayes else "GridSearchCV",
        "param_best": search.best_params_,
    }
    return best, meta

def save_bundle(model, meta, model_dir="model_store/eeg_mem_model"):
    # Save to the canonical model path used by the Streamlit UI.
    algo.save_model_bundle(model, meta, model_dir=model_dir)
