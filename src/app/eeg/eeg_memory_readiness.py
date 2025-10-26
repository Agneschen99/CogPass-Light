"""Lightweight loader and predictor for EEG memory readiness scores."""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from .eeg_mem_pipeline import load_model_bundle

_MODEL: Optional[CalibratedClassifierCV] = None
_THRESHOLD: float = 0.5
_META: Dict[str, object] = {}
_LOADED_PATH: Optional[str] = None
_LAST_ERROR: Optional[Exception] = None
_LOCK = threading.Lock()


def load_once(base_path: str = "model_store/eeg_mem_model") -> Tuple[bool, Optional[Exception]]:
    """Load and cache the memory readiness model bundle if not already available.

    Parameters
    ----------
    base_path : str, optional
        Path prefix pointing to the stored model bundle (`.model`/`.json` files).

    Returns
    -------
    Tuple[bool, Optional[Exception]]
        Boolean flag indicating success, plus the raised exception (if any).
    """
    global _MODEL, _THRESHOLD, _META, _LOADED_PATH, _LAST_ERROR
    with _LOCK:
        if _MODEL is not None and _LOADED_PATH == base_path:
            return True, None
        try:
            model, thr, meta = load_model_bundle(base_path, strict=True)
            if model is None:
                raise FileNotFoundError(f"No model bundle found at '{base_path}'")
            _MODEL = model
            _THRESHOLD = float(thr)
            _META = meta or {}
            _LOADED_PATH = base_path
            _LAST_ERROR = None
            return True, None
        except Exception as exc:  # pragma: no cover - defensive
            _MODEL = None
            _THRESHOLD = 0.5
            _META = {}
            _LOADED_PATH = None
            _LAST_ERROR = exc
            return False, exc


def predict_scores(X: np.ndarray) -> Dict[str, float]:
    """Return memory readiness probabilities for the provided feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix shaped `(n_samples, n_features)` compatible with the trained model.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the mean probability under the key `"memory"`.

    Raises
    ------
    RuntimeError
        If the model bundle could not be loaded before prediction.
    ValueError
        If the provided feature array does not resolve to two dimensions.
    """
    global _MODEL
    ok, err = load_once(_LOADED_PATH or "model_store/eeg_mem_model")
    if not ok or _MODEL is None:
        raise RuntimeError("Memory readiness model unavailable") from err

    feats = np.asarray(X, dtype=float)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    if feats.ndim != 2:
        raise ValueError("X must be 2-dimensional after reshaping")

    probs = _MODEL.predict_proba(feats)
    memory_prob = float(np.mean(probs[:, -1]))
    return {"memory": memory_prob}


__all__ = ["load_once", "predict_scores"]
