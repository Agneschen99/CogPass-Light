from __future__ import annotations

# src/app/eeg/algorithms.py
import os
import json
import time
from pathlib import Path
import numpy as np

"""
This file contains **drop‑in upgrades** for your current src/app/eeg/algorithms.py.
You can either:
  A) paste functions into your existing file and call them where noted below, or
  B) replace your file and re‑wire imports minimally (function names kept compatible where possible).

What’s included (all pure‑NumPy/Sklearn, no new heavy deps):
  1) Robust calibration stats (median/MAD) + winsorize.
  2) Dual smoothing: EWMA + median filter (anti‑spike, low‑latency).
  3) Two‑state HMM smoother for stable on/off states (pure NumPy Viterbi / forward‑backward style).
  4) Covariance (4×4) Log‑Euclidean feature for extra robustness on Muse 4‑ch.
  5) Quality gate: IMU + amplitude guard; confidence helper.
  6) Improved adaptive threshold (EWMA drift + hysteresis + min‑duration).
  7) DP utilities for arrays and post‑smoothed scores (UI vs export split).
  8) Calibrated LinearSVC (Platt) + safer pipelines (dual=False, probabilities via calibration).
  9) Ensemble: optional probability calibration and disagreement‑based confidence.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# =============================
# 1) Robust stats & winsorizer
# =============================

def robust_stats(x: np.ndarray) -> Tuple[float, float]:
    """Return (median, MAD_scaled). MAD is scaled to be comparable to sigma.
    If data too small, fall back to (mean, std/1.2533) approximately.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 5:
        mu = float(np.mean(x)) if x.size else 0.0
        sd = float(np.std(x)) if x.size else 1.0
        return mu, max(sd / 1.2533, 1e-6)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mad_scaled = 1.4826 * mad
    return med, max(mad_scaled, 1e-6)


def winsorize(x: np.ndarray, p: float = 0.01) -> np.ndarray:
    """Clip extremes to [p,1-p] quantiles to tame outliers without heavy distortion."""
    if x.size == 0:
        return x.astype(float)
    lo, hi = np.quantile(x, [p, 1.0 - p])
    return np.clip(x, lo, hi).astype(float)


# =====================================
# 2) EWMA + Median filter score smoother
# =====================================

def ewma_1d(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    a = float(alpha)
    for i in range(1, x.size):
        y[i] = a * x[i] + (1.0 - a) * y[i-1]
    return y


def median_filter_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if k <= 1 or x.size == 0:
        return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.median(xp[i:i+k])
    return out


def dual_smooth_scores(scores: np.ndarray, alpha: float = 0.2, k: int = 5) -> np.ndarray:
    """EWMA first (anti‑noise) then median filter (anti‑spike). Low‑latency and robust."""
    return median_filter_1d(ewma_1d(scores, alpha=alpha), k=k)


# ===============================
# 3) Tiny two‑state HMM smoother
# ===============================
@dataclass
class HMMParams:
    mu0: float
    sd0: float
    mu1: float
    sd1: float
    p_stay: float = 0.9  # probability of staying in same state


def _log_norm_pdf(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-6)
    z = (x - float(mu)) / sd
    return -0.5 * (z*z + np.log(2*np.pi*sd*sd))


def hmm_two_state_viterbi(scores: np.ndarray, p: HMMParams) -> Tuple[np.ndarray, np.ndarray]:
    """Return (state_path, state_prob) for two states {0,1} using Viterbi on Gaussian emissions.
    state_prob is a simple normalized likelihood over states at each t (not full forward‑backward),
    good enough for UI smoothing/thresholding.
    """
    x = np.asarray(scores, dtype=float)
    T = x.size
    if T == 0:
        return x.astype(int), x

    ps = float(p.p_stay)
    pt = 1.0 - ps
    # transition log‑probs
    A = np.log(np.array([[ps, pt], [pt, ps]], dtype=float))  # shape (2,2)

    # emission log‑probs per state
    e0 = _log_norm_pdf(x, p.mu0, p.sd0)
    e1 = _log_norm_pdf(x, p.mu1, p.sd1)
    E = np.vstack([e0, e1])  # (2,T)

    # init: assume equal prior
    V = np.empty((2, T), dtype=float)
    B = np.zeros((2, T), dtype=int)  # backpointers
    V[:, 0] = np.log(0.5) + E[:, 0]

    for t in range(1, T):
        for s in range(2):
            trans = V[:, t-1] + A[:, s]
            j = int(np.argmax(trans))
            V[s, t] = trans[j] + E[s, t]
            B[s, t] = j

    # backtrace
    S = np.empty(T, dtype=int)
    S[-1] = int(np.argmax(V[:, -1]))
    for t in range(T-2, -1, -1):
        S[t] = B[S[t+1], t+1]

    # crude state probability from emissions at each t (for UI opacity)
    L0 = np.exp(e0 - np.maximum(e0, e1))
    L1 = np.exp(e1 - np.maximum(e0, e1))
    P1 = L1 / (L0 + L1 + 1e-9)
    return S, P1


# ======================================
# 4) Covariance Log‑Euclidean feature (4×4)
# ======================================

def cov_logeuclid_feature(window: np.ndarray) -> np.ndarray:
    """window: shape [N_samples, N_channels] (Muse: channels last = 4)
    Returns upper‑triangular (including diag) of log‑Euclidean mapped covariance.
    """
    if window.ndim != 2:
        window = np.atleast_2d(window)
    C = np.cov(window, rowvar=False)  # [C,C]
    C += 1e-6 * np.eye(C.shape[0])
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-9)
    logC = V @ np.diag(np.log(w)) @ V.T
    iu = np.triu_indices(logC.shape[0])
    return logC[iu].astype(float)


# ==================================
# 5) Quality gate (IMU + amplitude)
# ==================================

def quality_gate(eeg_window: np.ndarray,
                 fs: float,
                 imu_acc_rms: Optional[float] = None,
                 amp_thresh_uV: float = 150.0,
                 acc_thresh_g: float = 1.5) -> bool:
    """Return True if window is OK. Uses amplitude (µV) and optional IMU accel (g)."""
    if eeg_window.size == 0:
        return False
    # amplitude in microvolts: assume input already converted to µV; if not, scale before calling
    amp_ok = (np.max(np.abs(eeg_window)) <= float(amp_thresh_uV))
    acc_ok = True if imu_acc_rms is None else (float(imu_acc_rms) <= float(acc_thresh_g))
    return bool(amp_ok and acc_ok)


def compute_confidence_simple(duration_s: float,
                              snr_est: float,
                              contact_ok: bool = True) -> float:
    # Simpler confidence, keep within [0,1]
    dur = min(1.0, float(duration_s) / 30.0)
    snr = float(np.clip(snr_est / (snr_est + 1.0), 0.0, 1.0))
    contact = 1.0 if contact_ok else 0.5
    return float(np.clip(dur * snr * contact, 0.0, 1.0))


def _bandpower_fft(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Return crude band power for the interval [f_lo, f_hi] using an FFT-based PSD."""
    x = np.asarray(sig, dtype=float)
    if x.size == 0 or fs <= 0:
        return 0.0
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(x)) ** 2) / max(x.size, 1)
    sel = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    if not np.any(sel):
        return 0.0
    return float(np.sum(psd[sel]))


def build_online_feats_from_raw(raw: np.ndarray, fs: float) -> np.ndarray:
    """Extract seven canonical memory/attention band features from a raw EEG window.

    Parameters
    ----------
    raw : np.ndarray
        One-dimensional EEG window in microvolts (or consistent units) sampled at `fs`.
    fs : float
        Sampling frequency in Hertz.

    Returns
    -------
    np.ndarray
        Seven-element feature vector `[delta, theta, alpha, beta, alpha/theta, beta/(alpha+theta), theta/beta]`.
    """
    sig = np.asarray(raw, dtype=float).ravel()
    delta = _bandpower_fft(sig, fs, 1.0, 4.0)
    theta = _bandpower_fft(sig, fs, 4.0, 8.0)
    alpha = _bandpower_fft(sig, fs, 8.0, 13.0)
    beta = _bandpower_fft(sig, fs, 13.0, 30.0)
    # Derived ratios with safe denominators
    alpha_theta = alpha / max(theta, 1e-9)
    beta_alpha_theta = beta / max(alpha + theta, 1e-9)
    theta_beta = theta / max(beta, 1e-9)
    return np.array([delta, theta, alpha, beta, alpha_theta, beta_alpha_theta, theta_beta], dtype=float)


# ======================================
# 6) Improved adaptive threshold (class)
# ======================================
class AdaptiveThreshold:
    """EWMA drift on mean only; fixed spread from calibration; hysteresis + min duration.
    Use with scores in [0,1]. Call update(score, conf) each step; query .state (0/1) for lamp.
    """
    def __init__(self,
                 mu_init: float,
                 spread_ref: float,
                 k: float = 0.7,
                 lmbda: float = 0.02,
                 hysteresis: float = 0.15,
                 min_duration_s: float = 1.0,
                 step_s: float = 0.5):
        self.mu = float(mu_init)
        self.spread = max(float(spread_ref), 1e-6)
        self.k = float(k)
        self.lmbda = float(lmbda)
        self.hyst = float(hysteresis)
        self.min_steps = max(1, int(round(min_duration_s / max(step_s, 1e-6))))
        self.state = 0
        self._counter = 0

    @property
    def threshold(self) -> float:
        return float(self.mu + self.k * self.spread)

    def update(self, score: float, conf_ok: bool = True) -> int:
        # slow mean drift only when confidence is OK
        if conf_ok:
            self.mu = (1.0 - self.lmbda) * self.mu + self.lmbda * float(score)
        T = self.threshold
        up = T + self.hyst * self.spread
        dn = T - self.hyst * self.spread
        desired = 1 if score >= up else (0 if score <= dn else self.state)
        if desired != self.state:
            self._counter += 1
            if self._counter >= self.min_steps:
                self.state = desired
                self._counter = 0
        else:
            self._counter = 0
        return self.state


# =======================================
# 7) DP utilities (array‑level, export‑only)
# =======================================

def add_dp_noise_array(arr: np.ndarray, eps: float = 3.0, delta: float = 1e-5, S: float = 1.0) -> np.ndarray:
    sigma = float(S) * np.sqrt(2.0 * np.log(1.25 / float(delta))) / max(float(eps), 1e-6)
    return arr.astype(float) + np.random.normal(0.0, sigma, size=arr.shape)


# ================================
# 8) Calibrated Linear SVM pipeline
# ================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def make_calibrated_linear_svm(pca_keep: float = 0.95, C: float = 0.8, class_weight="balanced") -> Pipeline:
    """LinearSVC -> Platt scaling to get calibrated probabilities for UI/thresholding."""
    base = LinearSVC(C=C, class_weight=class_weight, dual=False)  # dual=False is faster when n_samples>n_features
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_keep)),
        ("clf", CalibratedClassifierCV(base, method="sigmoid", cv=3)),
    ])


# ==========================================
# 9) Ensemble with disagreement confidence
# ==========================================
@dataclass
class TrainedModel:
    name: str
    model: Any  # must expose predict_proba after calibration


def _prob_from_estimator(est, X) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else np.max(p, axis=1)
    elif hasattr(est, "decision_function"):
        z = est.decision_function(X)
        if z.ndim > 1:
            z = np.max(z, axis=1)
        return 1.0 / (1.0 + np.exp(-z))
    else:
        return est.predict(X).astype(float)


def ensemble_probs(models: List[TrainedModel], X) -> Tuple[np.ndarray, np.ndarray]:
    probs = [ _prob_from_estimator(m.model, X) for m in models ]
    P = np.vstack(probs)  # [M,N]
    # disagreement (std across models); confidence inversely related to disagreement
    disag = np.std(P, axis=0)
    conf = 1.0 - (disag / (np.max(disag) + 1e-9))
    conf = np.clip(conf, 0.0, 1.0)
    # weight by distance to 0.5 too
    mag = np.abs(P - 0.5) * 2.0
    w = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-9)
    p_ens = np.sum(w * P, axis=0)
    return p_ens, conf


def confidence_weighted_predict(models: List[TrainedModel], X) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, confidences) using disagreement-aware ensemble aggregation.

    Parameters
    ----------
    models : List[TrainedModel]
        Sequence of trained estimators wrapped in `TrainedModel`.
    X : array-like
        Feature matrix to score; passed directly to the wrapped estimators.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        First array contains ensemble probabilities; second array is confidence weights.
    """
    return ensemble_probs(models, X)


def dp_confidence_weighted_predict(*args, **kwargs):
    """Backward-compatible alias for `confidence_weighted_predict`.

    Parameters
    ----------
    *args, **kwargs :
        Forwarded to `confidence_weighted_predict` so legacy call sites continue to work.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Same output as `confidence_weighted_predict`.
    """
    return confidence_weighted_predict(*args, **kwargs)


# ======================================================
# >>> INTEGRATE HERE: usage patterns in your main runner
# ======================================================
"""
Example wiring (pseudocode inside your streaming loop):

# After you compute per‑window score_raw in [0,1]:
score_s = dual_smooth_scores(np.array([*recent_scores, score_raw]))[-1]

# HMM smoothing (optional, after you have a short buffer):
if len(score_buffer) >= 20:
    S, P1 = hmm_two_state_viterbi(np.array(score_buffer), hmm_params)
    score_hmm = float(P1[-1])
else:
    score_hmm = score_s

# Quality / confidence:
conf = compute_confidence_simple(duration_s=session_duration, snr_est=snr, contact_ok=contact)

# Adaptive threshold:
state = thr.update(score_hmm, conf_ok=(conf >= 0.4))

# DP for export only:
export_series = add_dp_noise_array(np.array(history_scores), eps=3.0, delta=1e-5, S=1.0)

# Ensemble (if you keep multiple small models):
p_ens, conf_ens = ensemble_probs(models, X_eval)
"""
# End of patch
