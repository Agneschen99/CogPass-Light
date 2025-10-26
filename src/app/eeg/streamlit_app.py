from __future__ import annotations  # postpone evaluation of annotations for forward refs

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app.eeg.adp.adaptive_dp import AdaptiveDPController
except ModuleNotFoundError:
    from src.app.eeg.adp.adaptive_dp import AdaptiveDPController

import io
import time
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import os, sys
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
# --- bootstrap import path so "app" becomes importable
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import streamlit as st
import numpy as np
import json
import warnings
import pandas as pd

try:
    from pylsl import StreamInlet, resolve_stream
except Exception:
    StreamInlet = None
    resolve_stream = None

def get_privacy_controls(prefix: str = "main") -> Tuple[bool, float]:
    enable = st.sidebar.toggle(
        "Enable Differential Privacy",
        value=True,
        key=f"{prefix}_dp_toggle",
        help="Applies DP to cloud summary only. Differential Privacy helps protect your data by adding noise to results."
    )
    eps = st.sidebar.slider(
    "ε (privacy budget, lower=more private)",
    0.1, 5.0, 1.0,
    key=f"{prefix}_dp_epsilon",
    help="Epsilon controls the strength of differential privacy: lower values mean stronger privacy but less accurate results."
    )
    return enable, eps

# NOTE: Do not call Streamlit UI helpers at import time. Use defaults here and
# call `get_privacy_controls` from `main()` when the UI is active.
ENABLE_DP = False
EPSILON = 1.0
DELTA = 1e-5

def _dp_sigma(sensitivity: float, epsilon: float, delta: float=1e-5) -> float:
    """Compute Gaussian noise scale for differential privacy."""
    return float(sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon)

def dp_release_01(value_01: float, epsilon: float, delta: float=1e-5) -> float:
    """Add Gaussian noise to 0–1 values and clip result."""
    v = float(np.clip(value_01, 0.0, 1.0))
    sigma = _dp_sigma(sensitivity=1.0, epsilon=epsilon, delta=delta)
    noisy = v + np.random.normal(0.0, sigma)
    return float(np.clip(noisy, 0.0, 1.0))

# ==== helper: get latest epoch EEG feature vector ====

def get_latest_epoch_features() -> np.ndarray:
    """
    Return a numpy vector shaped (n_features,).
    Currently returns random data as a placeholder for real EEG features.
    """
    # TODO: replace with real epoch feature extraction logic
    return np.random.randn(20)  # e.g. if your model PCA dimension is 20
import altair as alt
import plotly.graph_objects as go
# ==== EEG Memory Readiness – imports & config ====
from app.eeg.eeg_mem_pipeline import load_model_bundle, online_predict
# Optional: high-level algorithm helpers and training convenience functions.
# Import guarded so the UI still loads if train_and_save.py isn't present.
from app.eeg import algorithms as algo
try:
    from app.eeg.train_and_save import train_linear_svm_bayes, train_mlp_bayes, save_bundle
except Exception:
    train_linear_svm_bayes = None
    train_mlp_bayes = None
    save_bundle = None


def apply_auto_dp_and_render_module(X_now, bundle, thr: float):
    """Module-level Auto-DP helper for tests/use outside Streamlit.

    Returns a dict with prob_raw, prob_private, eps_used, eps_total, label.
    """
    # 1) build model wrappers (tolerant to missing models)
    models = []
    if isinstance(bundle, dict):
        if "svm" in bundle and bundle["svm"] is not None:
            models.append(getattr(algo, 'TrainedModel')("svm", bundle["svm"]))
        if "mlp" in bundle and bundle["mlp"] is not None:
            models.append(getattr(algo, 'TrainedModel')("mlp", bundle["mlp"]))
    else:
        # accept a single model object
        models.append(getattr(algo, 'TrainedModel')("m", bundle))

    if not models:
        raise ValueError("No model available in bundle")

    # 2) ensemble probability
    try:
        # time the ensemble/prediction step
        with infer_timer():
            probs = algo.confidence_weighted_ensemble(models, X_now)
    except Exception:
        probs = np.asarray([0.5])
    prob_raw = float(np.mean(probs)) if np.size(probs) else 0.5

    # 3) Auto-DP: request epsilon and add noise (controller accounts spent epsilon)
    try:
        dp = AdaptiveDPController.from_settings()
    except Exception:
        dp = AdaptiveDPController(total_epsilon=5.0, delta=1e-5, strategy="per_call", eps_per_unit=1.0, S=1.0)

    eps_t = None
    try:
        eps_t = dp.request_epsilon(units=1.0)
    except Exception:
        eps_t = None

    if eps_t and eps_t > 0:
        prob_private = dp.add_noise_to_scalar(prob_raw, epsilon=eps_t)
    else:
        prob_private = prob_raw

    # Clip privatized output into [0, 1] to keep it a valid probability for downstream use
    try:
        prob_private = float(np.clip(prob_private, 0.0, 1.0))
    except Exception:
        # fallback: coerce numerically if np is unavailable for some reason
        prob_private = max(0.0, min(1.0, float(prob_private)))

    return {
        "prob_raw": float(prob_raw),
        "prob_private": float(prob_private),
        "eps_used": float(getattr(dp, 'spent_epsilon', 0.0)),
        "eps_total": float(getattr(dp, 'total_epsilon', 0.0)),
        "label": int(prob_raw >= float(thr)),
    }


MODEL_PATH = "model_store/eeg_mem_model"   # Path where your model is saved (.model & .json)
import io
import time
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple



# Constants
WINDOW_SECS = 60.0
STEP_SECS = 0.20
DRAW_EVERY_SEC = 5.0
MAX_PLOT_POINTS = 900
MAX_POINTS_CAP = 40000
HIST_MAX = 1200
RATIO_WIN_SEC = 2.0
SAVE_RECORD_EVERY = 30.0

EMA_ALPHA = 0.2
TOPN = 3

APP_HOME = Path.home() / ".neuroplan"
SETTINGS_FILE = APP_HOME / "settings.json"
HISTORY_PATH = APP_HOME / "history.csv"
BASELINE_PATH = APP_HOME / "baseline.json"
SESSIONS_PATH = APP_HOME / "sessions.csv"

# Module-level holder for active LSL inlet is delegated to app.eeg.lsl
def _set_current_inlet(inlet):
    """Set or clear the current inlet in the central LSL module.

    If `inlet` is None the LSL module's clear function is called.
    This avoids storing non-serializable StreamInlet objects in st.session_state.
    """
    try:
        from app.eeg import lsl as _lsl
        if inlet is None:
            try:
                _lsl.lsl_clear()
            except Exception:
                _lsl._LSL_INLET = None
        else:
            # assign directly into the lsl module (it keeps module-level state)
            try:
                _lsl._LSL_INLET = inlet
            except Exception:
                # best-effort fallback
                globals()['_LSL_INLET'] = inlet
    except Exception:
        # fallback: keep a local reference (shouldn't normally happen)
        globals()['_LSL_INLET'] = inlet


def _get_current_inlet():
    try:
        from app.eeg import lsl as _lsl
        return getattr(_lsl, '_LSL_INLET', None)
    except Exception:
        return globals().get('_LSL_INLET', None)


# =============== LSL 连接与 Live View 辅助 ===============

def _toast(msg: str, icon: str = "ℹ️"):
    try:
        st.toast(msg, icon=icon)
    except Exception:
        st.write(f"{icon} {msg}")


def resolve_stream_or_toast(timeout: float = 2.0) -> Optional[StreamInlet]:
    """尝试发现 EEG LSL 流并创建 inlet；成功后放到 session_state['lsl_inlet']。"""
    if resolve_stream is None or StreamInlet is None:
        _toast("pylsl 未安装，无法连接 LSL。请在 venv 里安装 pylsl。", "❌")
        return None
    try:
        # Prefer resolve_byprop for EEG streams; fall back to resolve_streams (any stream)
        try:
            from pylsl import resolve_byprop, resolve_streams
        except Exception:
            resolve_byprop = globals().get('resolve_byprop')
            resolve_streams = globals().get('resolve_streams')

        xs = []
        if resolve_byprop is not None:
            try:
                xs = resolve_byprop('type', 'EEG', timeout=timeout)
            except Exception:
                xs = []

        if not xs and resolve_streams is not None:
            try:
                xs = resolve_streams(wait_time=timeout)
            except Exception:
                xs = []

        if not xs:
            _toast("未发现 EEG LSL 流（检查 muselsl 是否在推流）", "⚠️")
            return None

        inlet = StreamInlet(xs[0], max_buflen=60)   # 60s 环形缓冲
        st.session_state['lsl_inlet'] = inlet
        _toast(f"LSL 已连接：{inlet.info().name()}", "✅")
        return inlet
    except Exception as e:
        _toast(f"连接 LSL 失败：{e}", "❌")
        return None


def ensure_live_state():
    """初始化 session_state 用到的键。"""
    ss = st.session_state
    ss.setdefault('lsl_inlet', None)
    ss.setdefault('live_running', False)
    ss.setdefault('live_cfg', dict(window_sec=10, refresh_hz=8, run_sec=None))
    ss.setdefault('live_buf', [])       # List[Tuple[ts, sample(list/np.ndarray)]]
    ss.setdefault('last_refresh', 0.0)


def start_live_view(window_sec: int = 10, refresh_hz: int = 8, run_sec: Optional[int] = None):
    """开始非阻塞 Live View：只设置状态并立即 rerun。"""
    ensure_live_state()
    ss = st.session_state
    ss['live_cfg'] = dict(window_sec=window_sec, refresh_hz=refresh_hz, run_sec=run_sec, t0=time.time())
    if ss.get('lsl_inlet') is None:
        if resolve_stream_or_toast() is None:
            return
    ss['live_running'] = True
    st.rerun()


def stop_live_view():
    """停止 Live View。"""
    st.session_state['live_running'] = False
    _toast("Live view 停止", "🟥")


def _pull_nonblocking():
    """从 inlet 拉取所有可用样本（非阻塞），并写入 live_buf。"""
    ss = st.session_state
    inlet = ss.get('lsl_inlet')
    if inlet is None:
        return
    buf: List[Tuple[float, List[float]]] = ss.get('live_buf', [])
    # 拉取尽可能多的样本（timeout=0）
    while True:
        sample, ts = inlet.pull_sample(timeout=0.0)
        if sample is None:
            break
        buf.append((ts, sample))
    # 只保留窗口期内的
    win = ss['live_cfg']['window_sec']
    tcut = time.time() - win
    if buf:
        # buf 是按时间递增的，做一次切片更快
        i = 0
        n = len(buf)
        while i < n and buf[i][0] < tcut:
            i += 1
        if i > 0:
            buf = buf[i:]
    ss['live_buf'] = buf


def live_tick_and_plot(placeholder):
    """
    每次 rerun 调用一次：
    1) 拉样本
    2) 画图
    3) 到刷新时机就安排下一次 rerun（非阻塞）
    """
    ensure_live_state()
    ss = st.session_state
    if not ss.get('live_running'):
        placeholder.info("点击 Start 开始 Live EEG")
        return

    # inlet 心跳：如果丢了，尝试重连一次
    if ss.get('lsl_inlet') is None:
        if resolve_stream_or_toast() is None:
            placeholder.warning("等待 LSL 流（muselsl stream 是否正在运行？）")
            return

    # 拉样本并维护窗口
    _pull_nonblocking()

    # 绘图（使用第 1 个通道作为示例；你可以改成均值或自定义处理）
    buf = ss.get('live_buf', [])
    if not buf:
        placeholder.info("Waiting for EEG…")
    else:
        ts = np.array([t for t, _ in buf], dtype=float)
        rel = ts - ts[0]
        ch0 = np.array([s[0] for _, s in buf], dtype=float)  # 第一个通道
        df = pd.DataFrame({"t": rel, "ch0": ch0}).set_index("t")
        with placeholder.container():
            st.line_chart(df, height=260)
            # 这里你也可以再画功率谱或 EMA 等

    # 运行时长到点自动停止
    run_sec = ss['live_cfg'].get('run_sec')
    if run_sec:
        if time.time() - ss['live_cfg'].get('t0', time.time()) >= run_sec:
            stop_live_view()
            return

    # 非阻塞节流刷新（基于 refresh_hz）
    now = time.time()
    refresh = max(1, int(ss['live_cfg']['refresh_hz']))  # Hz
    min_dt = 1.0 / float(refresh)
    if now - ss.get('last_refresh', 0.0) >= min_dt:
        ss['last_refresh'] = now
        # 触发下一次渲染
        st.rerun()

# ---- CSV schemas (memory_index added) ----
HISTORY_COLS = ["ts", "local_time", "hour", "ratio", "memory_index", "task", "difficulty"]
SESSIONS_COLS = ["ts", "local_time", "duration_sec", "rel_focus_mean", "confidence", "task", "difficulty"]

MODEL_PATH = "model_store/eeg_mem_model"
PREDICT_LOG = "/mnt/data/predict_log.csv"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Backwards-compatible loader: try plain `eeg_mem_pipeline` import, otherwise use app.eeg.eeg_mem_pipeline
_alt_loader = None
try:
    # try the simple import if someone's running this file from a different working dir
    from eeg_mem_pipeline import load_model_bundle as _alt_loader  # type: ignore
except Exception:
    _alt_loader = None

# Try to load model now (non-fatal). load_model_bundle returns (model, threshold, meta)
_model_top, _thr_top, _meta_top = None, 0.5, {}
for _loader in (_alt_loader, load_model_bundle):
    if _loader is None:
        continue
    try:
        _model_top, _thr_top, _meta_top = _loader(MODEL_PATH)
        break
    except Exception:
        continue

# Expose module-level variables (safe defaults if loading failed)
MODEL_LOADED = _model_top
MODEL_LOADED_THRESHOLD = _thr_top
MODEL_LOADED_META = _meta_top

# DP defaults
DELTA = 1e-5
DEFAULT_EPSILON = 1.0

# session placeholders (will be initialized in main)
ss: Any = None
status = None
timer_ph = None
chart_ph = None
spectrum_ph = None
feedback_ph = None
info_ph = None


# -------------------- Utility / signal processing --------------------
def _dp_sigma(sensitivity: float, epsilon: float, delta: float = 1e-5) -> float:
    return float(sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon)


def dp_release_01(value_01: float, epsilon: float, delta: float = 1e-5) -> float:
    v = float(np.clip(value_01, 0.0, 1.0))
    sigma = _dp_sigma(sensitivity=1.0, epsilon=epsilon, delta=delta)
    noisy = v + np.random.normal(0.0, sigma)
    return float(np.clip(noisy, 0.0, 1.0))


def get_latest_epoch_features() -> np.ndarray:
    # TODO: replace with real epoch feature extraction
    return np.random.randn(20)





# -------------------- CSV / persistence helpers --------------------
def _append_csv(p: Path, rows: list, cols: list) -> None:
    try:
        APP_HOME.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        if not p.exists():
            df.to_csv(p, index=False, columns=cols)
        else:
            df.to_csv(p, mode="a", header=False, index=False, columns=cols)
    except Exception:
        pass


def _load_csv(p: Path, cols: list):
    try:
        if p.exists():
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame(columns=cols)


# -------------------- JSON / meta helpers (module-level) --------------------
def _load_json(p: Path) -> dict:
    # Deprecated helper: prefer using explicit JSON loader or the module-level
    # persistence APIs. Emit a warning for now and delegate to json.load.
    try:
        warnings.warn("_load_json is deprecated; prefer using explicit _save_json/_load_json or higher-level persistence APIs", DeprecationWarning)
    except Exception:
        pass
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_json(p: Path, data: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_meta_from_disk(base_path: str) -> dict:
    mp = Path(f"{base_path}.json")
    meta = _load_json(mp)
    if meta:
        return meta
    mp2 = Path(f"{base_path}.meta.json")
    return _load_json(mp2)


def _safe_best_thr(meta: dict, fallback: float = 0.5) -> float:
    try:
        v = float(meta.get("best_thr", fallback))
        return max(0.0, min(1.0, v))
    except Exception:
        return fallback



@st.cache_resource(show_spinner=False)
def _load_eeg_model():
    try:
        model, threshold, meta = load_model_bundle(MODEL_PATH)
        return model, threshold, meta
    except Exception as e:
        st.warning(f"EEG memory model not loaded yet: {e}")
        return None, 0.5, {}


# -------------------- Storage helpers --------------------
def _ensure_dirs():
    APP_HOME.mkdir(parents=True, exist_ok=True)


# ---------- Baseline helpers (session + settings persistence) ----------
def _read_settings() -> dict:
    """Read settings.json from ~/.neuroplan if present."""
    try:
        app_home = Path.home() / ".neuroplan"
        settings_file = app_home / "settings.json"
        if settings_file.exists():
            return json.loads(settings_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_settings(d: dict) -> None:
    """Write settings dict to ~/.neuroplan/settings.json (best-effort)."""
    # DEPRECATED: prefer using the module-level _save_json with an explicit path.
    try:
        app_home = Path.home() / ".neuroplan"
        settings_file = app_home / "settings.json"
        try:
            # delegate to canonical JSON saver
            _save_json(settings_file, d)
        except Exception:
            # fallback to previous behavior
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            settings_file.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # persistence failure should not block the UI


def _guess_default_baseline() -> float:
    """Prefer model meta best_thr, then session threshold, then 0.5."""
    ss = st.session_state
    meta = ss.get("model_meta")
    if isinstance(meta, dict):
        bt = meta.get("best_thr")
        if isinstance(bt, (int, float)):
            return float(bt)
    thr = ss.get("threshold")
    if isinstance(thr, (int, float)):
        return float(thr)
    return 0.5


def _load_baseline() -> float:
    """Return baseline as a dict: {mu, sigma, thr, ts}.

    Migrate legacy float baselines to the dict form using session_state['baseline_sigma']
    as a fallback for sigma. Persist migration back to settings.json.
    """
    ss = st.session_state
    # If session has dict baseline already, return it
    if "baseline" in ss and isinstance(ss["baseline"], dict):
        return ss["baseline"]

    # Read from settings.json
    cfg = _read_settings()
    b = cfg.get("baseline")

    # If settings has dict baseline, prefer that
    if isinstance(b, dict):
        baseline = {
            "mu": float(b.get("mu", b.get("thr", _guess_default_baseline()))),
            "sigma": float(b.get("sigma", 0.1)),
            "thr": float(b.get("thr", b.get("mu", _guess_default_baseline()))),
            "ts": b.get("ts"),
        }
        ss["baseline"] = baseline
        return baseline


def load_baseline_from_disk():
    """Best-effort load of baseline into session_state from SETTINGS_FILE."""
    try:
        data = _load_json(SETTINGS_FILE, {})
    except Exception:
        data = {}
    b = data.get("baseline", None)
    if b:
        try:
            st.session_state["baseline_mu"] = float(b.get("mu", 0.47))
        except Exception:
            st.session_state["baseline_mu"] = 0.47
        try:
            st.session_state["baseline_sigma"] = float(b.get("sigma", 0.10))
        except Exception:
            st.session_state["baseline_sigma"] = 0.10
        try:
            st.session_state["baseline_ts"] = float(b.get("ts", 0.0))
        except Exception:
            st.session_state["baseline_ts"] = float(time.time())
    return b


def save_baseline_to_disk(mu: float, sigma: float):
    """Persist baseline (mu, sigma, ts) into SETTINGS_FILE (best-effort)."""
    try:
        mu_v = float(mu)
    except Exception:
        mu_v = _guess_default_baseline()
    try:
        sigma_v = float(sigma)
    except Exception:
        sigma_v = float(st.session_state.get("baseline_sigma", 0.1))

    ts_v = time.time()

    # Update session state so callers don't need to set these themselves
    try:
        st.session_state["baseline_mu"] = mu_v
        st.session_state["baseline_sigma"] = sigma_v
        st.session_state["baseline_ts"] = ts_v
        st.session_state["baseline"] = {"mu": mu_v, "sigma": sigma_v, "thr": mu_v, "ts": ts_v}
    except Exception:
        pass

    try:
        data = _load_json(SETTINGS_FILE, {}) or {}
    except Exception:
        data = {}
    data["baseline"] = {"mu": mu_v, "sigma": sigma_v, "ts": ts_v}
    try:
        _save_json(SETTINGS_FILE, data)
    except Exception:
        # best-effort persistence; swallow errors so UI isn't blocked
        pass


# --- Baseline calibration helpers (non-blocking) ---
def _safe_rerun(delay_s: float = 0.15):
    """Lightweight self-refresh: sleep briefly then trigger a single rerun.

    This avoids blocking the UI thread and is tolerant to Streamlit versions.
    """
    try:
        time.sleep(max(0.0, float(delay_s)))
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    except Exception:
        pass


def _to_float(x) -> float:
    """Coerce various collect_step return shapes into a single float.

    If x is a list/tuple/ndarray, take the last element; otherwise cast to float.
    """
    try:
        import numpy as _np
        if isinstance(x, (list, tuple, _np.ndarray)):
            x = x[-1]
    except Exception:
        if isinstance(x, (list, tuple)):
            x = x[-1]
    return float(x)


def _persist_baseline(mu: float, sigma: float):
    """Persist baseline to session state and disk using the canonical saver.

    Attempts to call the project's persistence function(s) if present.
    """
    try:
        st.session_state["baseline_mu"] = float(mu)
    except Exception:
        st.session_state["baseline_mu"] = _guess_default_baseline()
    try:
        st.session_state["baseline_sigma"] = float(sigma)
    except Exception:
        st.session_state["baseline_sigma"] = float(st.session_state.get("baseline_sigma", 0.1))
    try:
        st.session_state["baseline"] = {"mu": float(mu), "sigma": float(sigma), "ts": time.time()}
    except Exception:
        pass
    try:
        st.session_state["baseline_ts"] = time.time()
    except Exception:
        pass

    # Try canonical saver(s) if present (backwards compatible)
    for name in ("save_baseline_to_disk", "_save_baseline"):
        fn = globals().get(name)
        if callable(fn):
            try:
                fn(mu, sigma)
            except TypeError:
                try:
                    fn(mu)
                except Exception:
                    pass
            except Exception:
                pass


def _baseline_init(duration_s: int = 120, step_hz: float = 8.0):
    """初始化一次非阻塞校准状态。"""
    st.session_state["bl_state"] = {
        "status": "running",           # running | done | canceled
        "t0": time.time(),
        "dur": int(duration_s),
        "step_hz": float(step_hz),
        "n": 0,
        "sum": 0.0,
        "sum2": 0.0,
        "last_step": 0.0,
    }


def _baseline_tick(collector):
    """
    进行一次“刻”的采样-累加（由 UI 每次 rerun 触发一次），不阻塞。
    collector: 你的采样函数（如 collect_step），返回单个数或 (t, v)。
    """
    s = st.session_state.get("bl_state")
    if not s or s.get("status") != "running":
        return

    now = time.time()
    elapsed = now - s["t0"]

    # 控制步频（例如 8Hz）：避免一次 rerun 采样太多
    min_dt = 1.0 / max(1e-3, s["step_hz"])
    if (now - s.get("last_step", 0.0)) >= min_dt:
        try:
            x = collector()           # 可能是 v 或 (t, v)
            v = _to_float(x)
            s["n"] += 1
            s["sum"]  += v
            s["sum2"] += v * v
            s["last_step"] = now
        except Exception:
            # 采样失败不中断本轮
            pass

    # 更新进度
    s["progress"] = max(0.0, min(1.0, elapsed / max(1.0, s["dur"])))

    # 判断是否完成
    if elapsed >= s["dur"]:
        try:
            import math
            if s["n"] > 1:
                mu = s["sum"] / s["n"]
                # 无偏估计：Var = (sum2 - n*mu^2)/(n-1)
                var = max(0.0, (s["sum2"] - s["n"] * mu * mu) / (s["n"] - 1))
                sigma = math.sqrt(var)
            else:
                mu, sigma = 0.0, 0.1      # 极端 fallback

            _persist_baseline(mu, sigma)
            s["status"] = "done"
            s["mu"] = float(mu)
            s["sigma"] = float(sigma)
        except Exception:
            # finalize failure should not crash UI
            s["status"] = "done"

    # If settings has legacy float baseline, migrate it
    if isinstance(b, (int, float)):
        mu = float(b)
        sigma = float(ss.get("baseline_sigma", 0.1))
        baseline = {"mu": mu, "sigma": sigma, "thr": mu, "ts": None}
        # Persist via canonical helper
        try:
            _persist_baseline(mu, sigma)
        except Exception:
            # fallback to writing session dict directly
            ss["baseline"] = baseline
            cfg["baseline"] = baseline
            _write_settings(cfg)
        return baseline

    # If session holds a float baseline (legacy), migrate it similarly
    if "baseline" in ss and isinstance(ss["baseline"], (int, float)):
        mu = float(ss["baseline"])
        sigma = float(ss.get("baseline_sigma", 0.1))
        baseline = {"mu": mu, "sigma": sigma, "thr": mu, "ts": None}
        try:
            _persist_baseline(mu, sigma)
        except Exception:
            ss["baseline"] = baseline
            cfg["baseline"] = baseline
            _write_settings(cfg)
        return baseline

    # Fallback: construct from guesses
    mu = _guess_default_baseline()
    sigma = float(ss.get("baseline_sigma", 0.1))
    baseline = {"mu": float(mu), "sigma": sigma, "thr": float(mu), "ts": None}
    try:
        _persist_baseline(baseline["mu"], baseline["sigma"])
    except Exception:
        ss["baseline"] = baseline
    return baseline


# Note: previous code had a thin wrapper `_save_baseline`.
# We intentionally removed it to keep a single canonical
# persistence entry: `save_baseline_to_disk(mu, sigma)`.


def _coerce_baseline(bl):
    """Return (mu, sigma) for either float or dict baseline.

    Accepts legacy float baselines or the new dict form and returns two floats.
    """
    import streamlit as st
    if isinstance(bl, dict):
        mu = float(bl.get("mu", bl.get("thr", 0.5)))
        sigma = float(bl.get("sigma", st.session_state.get("baseline_sigma", 0.1)))
    else:
        mu = float(bl)
        sigma = float(st.session_state.get("baseline_sigma", 0.1))
    return mu, sigma


def run_baseline_calibration(seconds: int = 120):
    """
    Non-blocking baseline calibration with progress bar.
    Collects relaxed-period β/α ratio for `seconds`,
    saves mean (mu) and std (sigma) to session state.
    """
    import time
    import numpy as _np
    import streamlit as _st

    if _st.session_state.get("is_calibrating", False):
        _st.warning("Calibration already running…")
        return

    _st.session_state["is_calibrating"] = True
    _st.session_state.pop("baseline_mu", None)
    _st.session_state.pop("baseline_sigma", None)

    # 进度与状态
    status = _st.status("Calibrating baseline… relax 2 min 🧘", expanded=True)
    prog = _st.progress(0, text="Collecting… 0%")

    # 数据缓存
    vals_tmp, ts_tmp = [], []
    t_end = time.time() + seconds
    steps = max(1, int(seconds * 10))  # 0.1s 步进

    for i in range(steps):
        if time.time() >= t_end:
            break

        # 从你现有的数据通道捞一次值（示例：st.session_state['ss'].buf）
        # 如果你有 collect_step() 就调用它；否则用你现有的“抓一次值”的函数。
        try:
            collect_step()
        except Exception:
            pass

        ss_local = _st.session_state.get("ss")
        if ss_local and getattr(ss_local, "buf", None):
            try:
                vals_tmp.extend(list(ss_local.buf))
                ts_tmp.extend(list(ss_local.ts))
                ss_local.buf.clear(); ss_local.ts.clear()
            except Exception:
                pass

        # 每 0.1s 更新一次进度 & 让前端渲染
        pct = int(100 * (i + 1) / steps)
        prog.progress(min(pct, 100), text=f"Collecting… {pct}%")
        time.sleep(0.1)

    # 计算 μ / σ 并保存
    if len(vals_tmp) > 10:
        vals_np = _np.asarray(vals_tmp, dtype=float)
        mu = float(_np.mean(vals_np))
        sigma = float(_np.std(vals_np))
        _st.session_state["baseline_mu"] = mu
        _st.session_state["baseline_sigma"] = sigma
        try:
            status.update(state="complete", label=f"Baseline done ✓  μ={mu:.3f}, σ={sigma:.3f}")
        except Exception:
            pass
        try:
            _st.toast(f"Baseline updated: μ={mu:.3f}, σ={sigma:.3f}", icon="✅")
        except Exception:
            pass
        # persist baseline to disk and update timestamps
        try:
            st.session_state["baseline_mu"] = mu
            st.session_state["baseline_sigma"] = sigma
            st.session_state["baseline_ts"] = time.time()
            save_baseline_to_disk(mu, sigma)
            try:
                _st.toast(f"Baseline saved: μ={mu:.3f}, σ={sigma:.3f}", icon="💾")
            except Exception:
                pass
        except Exception:
            pass
    else:
        try:
            status.update(state="error", label="Not enough data for baseline. Try again.")
        except Exception:
            pass
        _st.error("Not enough data collected. Please try again.")

    _st.session_state["is_calibrating"] = False


def _calibration_collect_once():
    """Collect one small step for calibration: call collect_step, absorb ss.buf/ss.ts into calib accumulators and clear them."""
    ss = st.session_state
    if not ss.get("is_calibrating", False):
        return
    try:
        collect_step()
        if len(ss.get("buf", [])) > 0:
            ss["calib_vals"].extend(list(ss.buf))
            ss["calib_ts"].extend(list(ss.ts))
            # clear session buffers to avoid duplication
            ss.buf.clear()
            ss.ts.clear()
    except Exception:
        pass

# -----------------------------------------------------------------------

# ==== Diagnostics & Shape Contracts =========================================
import time as _diag_time
import numpy as _diag_np
from collections import deque as _diag_deque


def _metrics_init():
    import streamlit as st
    st.session_state.setdefault("metrics", {
        "pull_wall_dt": _diag_deque(maxlen=128),     # wall-clock time between pulls (s)
        "arrivals": _diag_deque(maxlen=128),         # number of samples returned per pull
        "ts_recent": _diag_deque(maxlen=4096),       # recent timestamps (for measured fs)
        "latencies_ms": _diag_deque(maxlen=256),     # inference latencies (ms)
        "expected_fs": None,                         # nominal sampling rate from LSL info
        "last_pull_wall": None,
    })


def diagnostics_record_pull(ts_list, lsl_info: dict):
    """Call after each LSL pull to record arrival counts and timestamps.

    ts_list: sequence of timestamps returned from inlet.pull_chunk()
    lsl_info: dict returned from central lsl module (may contain 'sfreq')
    """
    import streamlit as st
    _metrics_init()
    ms = st.session_state["metrics"]
    now = _diag_time.perf_counter()
    if ms["last_pull_wall"] is None:
        ms["last_pull_wall"] = now
    wall_dt = max(1e-6, now - ms["last_pull_wall"])
    ms["pull_wall_dt"].append(wall_dt)
    ms["arrivals"].append(len(ts_list))
    ms["last_pull_wall"] = now
    for t in ts_list:
        ms["ts_recent"].append(t)
    if ms["expected_fs"] is None and lsl_info and lsl_info.get("sfreq"):
        try:
            ms["expected_fs"] = float(lsl_info["sfreq"])
        except Exception:
            pass


class infer_timer:
    """Context manager to time inference blocks and record latency (ms).

    Usage:
        with infer_timer():
            do_inference()
    """
    def __enter__(self):
        self.t0 = _diag_time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        import streamlit as st
        _metrics_init()
        dt_ms = (_diag_time.perf_counter() - self.t0) * 1000.0
        st.session_state["metrics"]["latencies_ms"].append(dt_ms)


def _compute_rates():
    """Return (arrival_rate_hz, fs_meas_hz, fs_expected_hz, drop_rate, last_latency_ms, avg_latency_ms)."""
    import streamlit as st
    _metrics_init()
    ms = st.session_state["metrics"]
    if ms["pull_wall_dt"]:
        rates = [a / max(1e-6, dt) for a, dt in zip(ms["arrivals"], ms["pull_wall_dt"])]
        arrival_rate = float(_diag_np.mean(rates))
    else:
        arrival_rate = 0.0

    if len(ms["ts_recent"]) >= 5:
        diffs = _diag_np.diff(_diag_np.asarray(ms["ts_recent"], dtype=float))
        med = float(_diag_np.median(diffs)) if diffs.size else 0.0
        fs_meas = (1.0 / med) if med > 0 else 0.0
    else:
        fs_meas = 0.0

    fs_expected = ms.get("expected_fs") or fs_meas or 0.0
    drop_rate = None
    if fs_expected > 0:
        drop_rate = max(0.0, 1.0 - (arrival_rate / fs_expected))

    lat_list = list(ms["latencies_ms"])
    lat_last = float(lat_list[-1]) if lat_list else None
    lat_avg = float(_diag_np.mean(lat_list)) if lat_list else None
    return arrival_rate, fs_meas, fs_expected, drop_rate, lat_last, lat_avg


def _status_color(ok: bool):
    return "🟢" if ok else "🔴"


def _safe_shape(x, want_nd=None, want_nfeat=None):
    """Return (ok, msg) without raising. Verifies ndarray shape/contracts."""
    try:
        arr = _diag_np.asarray(x)
        shape = arr.shape
        nd_ok = (want_nd is None) or (arr.ndim == want_nd)
        nf_ok = True
        if want_nfeat is not None:
            nf_ok = (arr.ndim >= 2 and arr.shape[1] == want_nfeat)
        ok = bool(nd_ok and nf_ok)
        msg = f"shape={shape}"
        return ok, msg
    except Exception as e:
        return False, f"unavailable ({type(e).__name__})"


def _has_nan(x):
    try:
        return bool(_diag_np.isnan(_diag_np.asarray(x)).any())
    except Exception:
        return True


def diagnostics_panel_plus(n_features_expected: int = None):
    """Render an expanded diagnostics panel: arrival rates, measured fs, drop rate, latency, and shape contracts."""
    import streamlit as st
    _metrics_init()
    arrival_rate, fs_meas, fs_exp, drop_rate, lat_last, lat_avg = _compute_rates()

    with st.expander("🔎 System Diagnostics (I/O & Shape Contracts)", expanded=False):
        # ---- I/O rates & latency
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Arrival Rate", f"{arrival_rate:.1f} Hz")
        c2.metric("Measured fs", f"{fs_meas:.1f} Hz")
        c3.metric("Expected fs", f"{fs_exp:.1f} Hz")
        c4.metric("Drop Rate", "-" if drop_rate is None else f"{drop_rate*100:.1f} %")
        if lat_last is not None or lat_avg is not None:
            st.caption(f"Inference latency: last={lat_last:.1f} ms · avg={lat_avg:.1f} ms" if lat_last else "")

        st.divider()

        # ---- shape contracts & NaN checks
        ss = st.session_state
        X = ss.get("X_window_features")
        probs = ss.get("probs") or ss.get("focus_probs")
        focus_hist = ss.get("focus_history")

        okX, msgX = _safe_shape(X, want_nd=2, want_nfeat=n_features_expected)
        okP, msgP = _safe_shape(probs, want_nd=1)
        okF, msgF = _safe_shape(focus_hist, want_nd=1)

        nanX = _has_nan(X) if X is not None else True
        nanP = _has_nan(probs) if probs is not None else False

        st.write(
            f"{_status_color(okX and not nanX)} **Feature Matrix X** — {msgX}"
            + ("" if n_features_expected is None else f" · expect n_features={n_features_expected}")
            + (" · NaN detected" if nanX else "")
        )
        st.write(
            f"{_status_color(okP and not nanP)} **Model Probs** — {msgP}"
            + (" · NaN detected" if nanP else "")
        )
        st.write(
            f"{_status_color(okF)} **Focus History** — {msgF}"
        )

        # Gentle warnings (non-blocking)
        if not (okX and not nanX):
            st.warning("Feature matrix X invalid — check windowing, feature extraction, or NaN sources.", icon="⚠️")
        if probs is not None and not (okP and not nanP):
            st.warning("Model probabilities invalid — check model input shape & scaler parameters.", icon="⚠️")

# ==== /Diagnostics & Shape Contracts ========================================

# -----------------------------------------------------------------------


def render_model_info_panel_v2(*, model_path: str, meta_path: str):
    """
    更简洁的模型信息面板（已修复 HTML 被当作文本显示的问题）：
    - 顶部：操作按钮（Reload / Clear / Download / Reset threshold）
    - 中间：卡片式关键指标（Model, Date, Classifier, Features, Best Thr, Method）
    - 下方：彩色高亮的指标栅格（acc/f1/precision/recall/tpr/fpr/youden_j/balanced_acc）
    - 最底：原始 JSON（收起）
    """
    # 基础样式（一次注入即可）
    st.markdown(
        """
        <style>
        .np-wrap{margin-top:6px;}
        .np-cards{
          display:grid;grid-template-columns:repeat(6, minmax(0,1fr));
          gap:12px;margin-top:6px
        }
        .np-card{
          background:var(--background-color,#fff);
          border:1px solid rgba(0,0,0,0.06);
          border-radius:12px;padding:12px 14px
        }
        .np-k{font-size:12px;color:#6b7280;margin-bottom:6px}
        .np-v{font-size:20px;font-weight:600;line-height:1.1}
        .np-badges{
          display:grid;grid-template-columns:repeat(6, minmax(0,1fr));
          gap:12px;margin-top:12px
        }
        .np-badge{
          background:#f8fafc;border:1px solid rgba(0,0,0,0.05);
          border-radius:10px;padding:10px
        }
        .np-badge .k{font-size:12px;color:#6b7280}
        .np-badge .v{font-size:18px;font-weight:600}
        .np-good{color:#10b981} .np-warn{color:#f59e0b} .np-bad{color:#ef4444}
        </style>
        """,
        unsafe_allow_html=True,
    )

    ss = st.session_state
    meta_file = Path(meta_path)

    # ========= actions bar =========
    c1, c2, c3, c4, csp = st.columns([1,1,1,1,6])
    with c1:
        if st.button("🔄 Reload", use_container_width=True):
            if meta_file.exists():
                ss["model_meta"] = _load_json(meta_file)
                st.toast("Meta reloaded from disk.")
            else:
                st.warning(f"Meta file not found: {meta_file}")
    with c2:
        if st.button("🧹 Clear", use_container_width=True):
            ss.pop("model_meta", None)
            st.toast("Cleared meta from session.")
    with c3:
        # 下载按钮（使用当前 session 或磁盘文件)
        _meta_json = (ss.get("model_meta") or (_load_json(meta_file) if meta_file.exists() else {}))
        st.download_button(
            "⬇️ Download",
            data=json.dumps(_meta_json, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="model_meta.json",
            mime="application/json",
            use_container_width=True
        )
    with c4:
        if st.button("🎯 Reset threshold", use_container_width=True):
            meta = ss.get("model_meta")
            if not meta and meta_file.exists():
                meta = _load_json(meta_file)
                ss["model_meta"] = meta
            best_thr = (meta or {}).get("best_thr", None)
            if isinstance(best_thr, (int, float)):
                ss["threshold"] = float(best_thr)
                st.toast(f"Threshold reset to model default: {best_thr:.2f}")
            else:
                st.warning("No best_thr found in meta.")

    # ========= load meta (pref session) =========
    meta = ss.get("model_meta")
    if meta is None and meta_file.exists():
        meta = _load_json(meta_file)
        ss["model_meta"] = meta
    if not meta:
        st.info("No model metadata available.")
        return

    # 基本字段
    name = meta.get("model_name", "—")
    date = meta.get("trained_on", "—")
    clf = meta.get("classifier", "—")
    feats = meta.get("features", "—")
    best_thr = meta.get("best_thr", None)
    thr_method = meta.get("threshold_method", "—")
    m = meta.get("metrics", {}) or {}

    # ---------- headline cards ----------
    cards_html = f"""
    <div class="np-wrap">
      <div class="np-cards">
        <div class="np-card"><div class="np-k">Model</div><div class="np-v">{name}</div></div>
        <div class="np-card"><div class="np-k">Trained on</div><div class="np-v">{date}</div></div>
        <div class="np-card"><div class="np-k">Classifier</div><div class="np-v">{clf}</div></div>
        <div class="np-card"><div class="np-k">Features</div><div class="np-v">{feats}</div></div>
        <div class="np-card"><div class="np-k">Best threshold</div><div class="np-v">{(f"{best_thr:.2f}" if isinstance(best_thr,(int,float)) else "—")}</div></div>
        <div class="np-card"><div class="np-k">Method</div><div class="np-v">{thr_method}</div></div>
      </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    # ---------- metric badges ----------
    def _cls(v):
        if not isinstance(v, (int, float)): return ""
        if v >= 0.80: return "np-good"
        if v >= 0.60: return "np-warn"
        return "np-bad"

    order = ["acc","f1","precision","recall","tpr","fpr","youden_j","balanced_acc"]
    badge_items = []
    for k in order:
        v = m.get(k, None)
        vv = f"{v:.2f}" if isinstance(v,(int,float)) else "—"
        badge_items.append(
            f'<div class="np-badge"><div class="k">{k}</div><div class="v {_cls(v)}">{vv}</div></div>'
        )
    badges_html = f'<div class="np-badges">' + "".join(badge_items) + "</div>"
    st.markdown(badges_html, unsafe_allow_html=True)

    # ---------- raw JSON (collapsed) ----------
    with st.expander("🧾 Full meta JSON", expanded=False):
        st.code(json.dumps(meta, ensure_ascii=False, indent=2), language="json")
    # NOTE: removed misplaced LSL connection snippet that was accidentally pasted here.
    # LSL connection logic lives in `_lsl_connect_or_error` and `render_live_eeg_panel`.
    return None


# -------------------- Acquisition --------------------
def collect_step():
    # Use centralized LSL pull helper which handles timeouts and reconnects
    try:
        from app.eeg import lsl as _lsl
    except Exception:
        _lsl = None

    if _lsl is None:
        inlet = _get_current_inlet()
        if inlet is None:
            return
        try:
            chunk, stamps = inlet.pull_chunk(timeout=STEP_SECS, max_samples=int(1024 * STEP_SECS))
        except Exception:
            return
    else:
        chunk, stamps = _lsl.lsl_safe_pull(max_samples=int(1024 * STEP_SECS))
        try:
            from app.eeg.lsl import lsl_info
            diagnostics_record_pull(stamps, lsl_info())
        except Exception:
            pass

    if not stamps:
        return
    vals = [row[0] for row in chunk]
    t0 = ss.started_at or time.time()
    ts_rel = [s - t0 for s in stamps]
    ss.buf.extend(vals)
    ss.ts.extend(ts_rel)

    now_rel = time.time() - t0
    while ss.ts and (now_rel - ss.ts[0] > WINDOW_SECS):
        ss.ts.popleft()
        ss.buf.popleft()
    while len(ss.buf) > MAX_POINTS_CAP:
        ss.ts.popleft()
        ss.buf.popleft()


def _downsample(ts: np.ndarray, ys: np.ndarray, max_points: int):
    if ts.size <= max_points:
        return ts, ys
    idx = np.linspace(0, ts.size - 1, max_points).astype(int)
    return ts[idx], ys[idx]


def _compute_series_ratio(ts: np.ndarray, vals: np.ndarray, fs: float, win_sec=RATIO_WIN_SEC, step_sec=0.5):
    if ts.size == 0:
        return np.array([]), np.array([])
    t_min, t_max = ts.min(), ts.max()
    if t_max - t_min < 0.5:
        return np.array([]), np.array([])
    out_t, out_r = [], []
    t = t_min + win_sec
    while t <= t_max:
        idx = np.where((ts >= t - win_sec) & (ts <= t))[0]
        if idx.size > 10:
            out_t.append(t)
            out_r.append(algo.ema_update(algo.beta_alpha_ratio(vals[idx], fs)))
        t += step_sec
    return np.asarray(out_t), np.asarray(out_r)


def _state_label(r_rel: float):
    if np.isnan(r_rel):
        return "Unknown", "⚪"
    if r_rel < 0.45:
        return "Low", "🟡"
    if r_rel < 0.65:
        return "Medium", "🟢"
    return "High", "🔵"


# -------------------- Focus/Memory Meters --------------------
def render_bar_meter(rel_now: float, container, label: str = "Focus Meter"):
    n = 10
    k = int(round(np.clip(rel_now, 0, 1) * n))
    seg_html = []
    for i in range(n):
        filled = (i < k)
        color = f"hsl({210 - 90 * (i / (n-1))}, 90%, 50%)" if filled else "hsl(0,0%,88%)"
        seg_html.append(
            f'<div style="width:9%;height:18px;border-radius:6px;background:{color};margin-right:1%"></div>'
        )
    meter = "".join(seg_html)
    pct = int(np.clip(rel_now, 0, 1) * 100)
    hint = (
        "Excellent!" if rel_now >= 0.7 else ("Good, keep it steady." if rel_now >= 0.5 else "Try a short break or slow breathing.")
    )
    container.markdown(
        f"""
        <div style="margin:8px 0 4px 0;font-weight:600;">{label}: {k}/{n} ({pct}%)</div>
        <div style="display:flex;align-items:center;">{meter}</div>
        <div style="opacity:0.7;margin-top:6px;">Hint: {hint}</div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- Charts & metrics --------------------
def draw_chart_and_metrics() -> Tuple[float, float, float]:
    if len(ss.buf) == 0:
        return 0.0, 0.0, 0.0
    vals = np.asarray(ss.buf, dtype=float)
    ts = np.asarray(ss.ts, dtype=float)
    fs = algo._estimate_fs(ts)

    t_plot, r_plot = _compute_series_ratio(ts, vals, fs)
    if t_plot.size == 0:
        return 0.0, 0.0, 0.0
    t_plot, r_plot = _downsample(t_plot, r_plot, MAX_PLOT_POINTS)

    bl = _load_baseline()
    mu, sigma = _coerce_baseline(bl)

    idx_tail = np.where(ts >= ts.max() - 5.0)[0]
    ratio_now = algo.beta_alpha_ratio(vals[idx_tail] if idx_tail.size > 0 else vals, fs)
    ratio_now = algo.ema_update(ratio_now)
    rel_now = algo.relative_focus(ratio_now, mu, sigma)

    mi_now = algo.theta_gamma_index(vals[idx_tail] if idx_tail.size > 0 else vals, fs)
    mi_now = algo.ema_update(mi_now)
    rel_mem = algo.relative_memory_score(mi_now, mu, sigma)

    snr_est = float(np.clip(1.0 / (np.var(r_plot) + 1e-6), 0, 10))
    duration = float(ss.ts[-1] - ss.ts[0]) if len(ss.ts) > 1 else 0.0
    conf = algo.compute_confidence(duration, snr_est, contact_ok=True)

    df = pd.DataFrame({"Time (s)": t_plot, "β/α": r_plot})
    mean_val = float(np.mean(r_plot))
    line = alt.Chart(df).mark_line().encode(
        x=alt.X("Time (s):Q", title="Time (seconds)"),
        y=alt.Y("β/α:Q", title="EEG β/α (EMA)"),
    ).properties(height=300)
    avg_rule = alt.Chart(pd.DataFrame({"y": [mean_val]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
    chart = (line + avg_rule).interactive()
    chart_ph.altair_chart(chart, use_container_width=True)

    r_short = float(np.mean(df["β/α"].tail(10))) if df.shape[0] >= 10 else float(np.mean(df["β/α"]))
    r_long = float(np.mean(df["β/α"].tail(60))) if df.shape[0] >= 60 else r_short
    delta_pct = 0.0 if r_long == 0 else (r_short - r_long) / r_long * 100.0

    colA, colB, colC, colD = st.columns(4)
    label, emoji = _state_label(rel_now)
    colA.metric("Relative Focus (0–1)", f"{rel_now:.2f}", delta=f"{emoji} {label}")
    colB.metric("Current β/α (EMA)", f"{ratio_now:.2f}", delta=f"{delta_pct:+.1f}%")
    colC.metric("Confidence", f"{conf:.2f}")
    colD.metric("Mean β/α (window)", f"{mean_val:.2f}")

    colE, colF = st.columns(2)
    colE.metric("Memory Index (θ+γ)/α", f"{mi_now:.2f}")
    colF.metric("Relative Memory (0–1)", f"{rel_mem:.2f}")

    with feedback_ph.container():
        render_bar_meter(rel_now, feedback_ph, label="Focus Meter")
        st.markdown("---")
        render_bar_meter(rel_mem, feedback_ph, label="Memory Meter (encoding/retrieval)")

    return rel_now, ratio_now, conf


def safe_draw_chart_and_metrics() -> Tuple[float, float, float]:
    """Call draw_chart_and_metrics and catch/display exceptions to avoid white-screen reruns."""
    try:
        return draw_chart_and_metrics()
    except Exception as e:
        try:
            st.error(f"Chart error: {e}")
        except Exception:
            pass
        return 0.0, 0.0, 0.0


def draw_power_spectrum(max_hz=40):
    if len(ss.buf) == 0:
        return
    vals = np.asarray(ss.buf, dtype=float)
    ts = np.asarray(ss.ts, dtype=float)
    fs = algo._estimate_fs(ts)
    if fs <= 0 or vals.size < int(fs):
        return
    idx = np.where(ts >= ts.max() - 2.0)[0]
    if idx.size < int(0.5 * fs):
        return
    x = vals[idx]
    win = np.hanning(x.size)
    xw = x * win
    spec = np.fft.rfft(xw)
    psd = (np.abs(spec) ** 2) / np.sum(win ** 2)
    freqs = np.fft.rfftfreq(xw.size, d=1.0 / fs)
    m = freqs <= max_hz
    df = pd.DataFrame({"Freq (Hz)": freqs[m], "Power": psd[m]})
    chart = (alt.Chart(df).mark_bar()
             .encode(x=alt.X("Freq (Hz):Q", bin=alt.Bin(maxbins=80), title="Frequency (Hz)"),
                     y=alt.Y("sum(Power):Q", title="Power"),
                     tooltip=["Freq (Hz):Q", "Power:Q"]) 
             .properties(height=220, title="Power Spectrum (0–40 Hz, recent 2 s)"))
    spectrum_ph.altair_chart(chart, use_container_width=True)

    # --- Robust PSD metrics (band power, peak freq, spectral entropy, focus cue)
    try:
        def _band_power(psd_arr: np.ndarray, freq_arr: np.ndarray, lo: float, hi: float, mode="mean") -> float:
            mband = (freq_arr >= lo) & (freq_arr < hi)
            if not np.any(mband):
                return 0.0
            seg = psd_arr[mband].astype(float)
            return float(seg.mean() if mode == "mean" else seg.sum())

        def _spectral_entropy(psd_arr: np.ndarray) -> float:
            p = psd_arr.astype(float)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                return float("nan")
            p = p / s
            return float(-(p * np.log(p + 1e-12)).sum())

        def _metric_value(v, fmt):
            try:
                return fmt.format(v) if np.isfinite(v) else "—"
            except Exception:
                return "—"

        psd_vis = psd[m]
        freqs_vis = freqs[m]

        P_alpha = _band_power(psd_vis, freqs_vis, 8.0, 13.0, mode="mean")
        P_beta = _band_power(psd_vis, freqs_vis, 13.0, 30.0, mode="mean")
        beta_alpha = P_beta / (P_alpha + 1e-12)

        peak_f = float(freqs_vis[int(np.nanargmax(psd_vis))]) if psd_vis.size else float("nan")
        spec_entropy = _spectral_entropy(psd_vis)

        focus_t = st.session_state.get("focus_t")
        focus_display = _metric_value(focus_t, "{:.2f}") if isinstance(focus_t, (int, float)) else "—"

        def focus_delta_text(f):
            if not isinstance(f, (int, float)) or not np.isfinite(f):
                return None
            if f >= 0.70:
                return "🟢 high"
            if f <= 0.40:
                return "🔴 low"
            return "🟡 mid"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("β/α (13–30 / 8–13 Hz)", _metric_value(beta_alpha, "{:.2f}"))
        c2.metric("Peak freq (Hz)", _metric_value(peak_f, "{:.1f}"))
        c3.metric("Spectral entropy", _metric_value(spec_entropy, "{:.2f}"))
        c4.metric("Focus(t)", focus_display, delta=focus_delta_text(focus_t))
    except Exception:
        pass


# -------------------- History / Sessions --------------------
def save_session_history():
    if len(ss.buf) == 0 or ss.started_at is None:
        return
    vals = np.asarray(ss.buf, dtype=float)
    ts = np.asarray(ss.ts, dtype=float)
    fs = algo._estimate_fs(ts)
    if ts.size < 2:
        return
    rows = []
    t_min, t_max = ts.min(), ts.max()
    t = t_min + SAVE_RECORD_EVERY
    while t <= t_max:
        idx = np.where((ts >= t - SAVE_RECORD_EVERY) & (ts <= t))[0]
        if idx.size > 50:
            r = algo.beta_alpha_ratio(vals[idx], fs)
            mi = algo.theta_gamma_index(vals[idx], fs)
            epoch = ss.started_at + t
            lt = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
            hour = int(datetime.fromtimestamp(epoch).strftime("%H"))
            rows.append({
                "ts": epoch, "local_time": lt, "hour": hour,
                "ratio": r, "memory_index": mi,
                "task": ss.selected_task, "difficulty": ss.selected_diff,
            })
        t += SAVE_RECORD_EVERY
    _append_csv(HISTORY_PATH, rows, HISTORY_COLS)


def record_session_summary(rel_mean: float, conf: float):
    row = {"ts": time.time(), "local_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "duration_sec": float(ss.last_duration_secs or 0.0),
           "rel_focus_mean": rel_mean, "confidence": conf,
           "task": ss.selected_task, "difficulty": ss.selected_diff}
    _append_csv(SESSIONS_PATH, [row], SESSIONS_COLS)


def export_history_csv():
    df = _load_csv(HISTORY_PATH, HISTORY_COLS)
    if df.empty:
        st.warning("No history yet.")
        return
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button("⬇️ Export History CSV", data=buf.getvalue(),
                       file_name="eeg_history.csv", mime="text/csv")


# -------------------- Personalized advice helpers --------------------
def _recent_task_window_from_history(task: str, minutes: float) -> Tuple[Optional[float], int]:
    hist = _load_csv(HISTORY_PATH, HISTORY_COLS)
    if hist.empty or task == "None":
        return None, 0
    cutoff = time.time() - minutes * 60.0
    df = hist[(hist["ts"] >= cutoff) & (hist["task"] == task)]
    if df.empty:
        return None, 0
    return float(df["ratio"].mean()), int(df.shape[0])


def _recent_window_from_buffer(max_minutes: float = 10.0) -> Optional[float]:
    if len(ss.buf) == 0:
        return None
    vals = np.asarray(ss.buf, dtype=float)
    ts = np.asarray(ss.ts, dtype=float)
    fs = algo._estimate_fs(ts)
    window = min(max_minutes * 60.0, float(ts.max() - ts.min()))
    if window < 5.0:
        return None
    idx = np.where(ts >= ts.max() - window)[0]
    if idx.size < 20:
        return None
    return float(algo.beta_alpha_ratio(vals[idx], fs))


def _overall_daily_mean() -> Optional[float]:
    hist = _load_csv(HISTORY_PATH, HISTORY_COLS)
    if hist.empty:
        return None
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        df = hist[hist["local_time"].str.startswith(today)]
        if not df.empty:
            return float(df["ratio"].mean())
    except Exception:
        pass
    return float(hist["ratio"].mean()) if not hist.empty else None


def _best_tasks_for_current_hour(topk: int = 3) -> List[Tuple[str, float]]:
    hist = _load_csv(HISTORY_PATH, HISTORY_COLS)
    if hist.empty:
        return []
    hour_now = int(datetime.now().strftime("%H"))
    df = hist[hist["hour"] == hour_now]
    if df.empty or "task" not in df.columns:
        return []
    g = df.groupby("task")["ratio"].mean().sort_values(ascending=False).head(topk)
    return [(t, float(s)) for t, s in g.items()]


def _advice_from_scores(rel_focus_est: Optional[float], improve_pct: Optional[float]) -> str:
    imp = improve_pct if improve_pct is not None else 0.0
    rf = rel_focus_est if rel_focus_est is not None else 0.5
    if rf >= 0.70 and imp >= 8.0:
        return "Great slot for this task — keep it here, or try a more challenging version."
    if rf >= 0.55 and imp >= 0.0:
        return "Good for this task. You can continue, or do focused reading/review."
    if rf >= 0.45 and imp < 0.0:
        return "Borderline for this task. Prefer lighter tasks (reading/review) in this hour."
    return "Not ideal for high-focus work now. Consider rest, a walk, or very light tasks."


# -------------------- Recommendation (with improvement %) + Advice --------------------
def ai_recommendation_block():
    ss = st.session_state
    ss.setdefault("min_history_mode", "Quick (10 min)")
    ss.setdefault("min_history_min", 30.0)

    st.markdown("### 🧠 EEG Memory Readiness (Overview + Confidence)")

    model, default_thr, _meta = _load_eeg_model()
    # Use threshold from session_state (set by the main UI block) if present,
    # otherwise fall back to the model's default returned by _load_eeg_model().
    user_thr = float(ss.get("threshold", float(default_thr)))

    if st.button("🧠 Capture & Predict This Epoch (30–60s)"):
        if model is None:
            st.error("No trained and calibrated EEG model found. Please train offline and save to model_store first.")
        else:
            x_new = get_latest_epoch_features()
            # time the online predict call
            with infer_timer():
                out = online_predict(model, x_new=x_new, threshold=user_thr)
            p = out["prob"]
            label = out["label_hat"]
            conf = out["confidence"]

            st.metric("Memory Readiness (probability)", f"{p*100:.1f}%")
            st.write(f"**Decision:** {'✅ Good for learning new material' if label==1 else '🟡 Better for review/light tasks'}")
            st.write(f"**Model confidence:** {conf:.2f} (0~1, higher is more stable)")
            st.caption(f"Threshold used: {user_thr:.2f}")

            # write log here (ensures variables are defined)
            try:
                row = pd.DataFrame([{
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "prob": float(p),
                    "label_hat": int(label),
                    "confidence": float(conf),
                    "thr": float(user_thr),
                }])
                if os.path.exists(PREDICT_LOG):
                    row.to_csv(PREDICT_LOG, mode="a", header=False, index=False)
                else:
                    row.to_csv(PREDICT_LOG, index=False)
            except Exception:
                pass

    st.markdown("### 📊 AI Study Time Recommendation (Task Difficulty)")

    col_mode, col_info = st.columns((1, 1))
    options = ["Quick (10 min)", "Standard (30 min)", "Research (60 min)"]
    idx = options.index(ss.min_history_mode) if ss.min_history_mode in options else 0
    mode = col_mode.selectbox("History Requirement", options, index=idx, key="history_req")
    ss.min_history_mode = mode

    mins_map = {"Quick (10 min)": 10.0, "Standard (30 min)": 30.0, "Research (60 min)": 60.0}
    ss.min_history_min = mins_map.get(mode, 30.0)
    col_info.caption("Choose a shorter requirement for demos, longer for more robust estimates.")


# ---- Pretty "Model Info" panel ----
import pandas as pd

def _get(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k, None)
    return d if d is not None else default


def render_model_info(meta: dict):
    st.subheader("Model summary & training metadata", divider=False)

    # ---- KPI ----
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    best_thr = _get(meta, "best_thr", default=None)
    features = _get(meta, "features", default=None)
    trained_on = _get(meta, "trained_on", default="—")
    thr_method = _get(meta, "threshold_method", default="—")

    with c1:
        st.metric("best_thr", f"{best_thr:.2f}" if best_thr is not None else "—")
    with c2:
        st.metric("features", str(features) if features is not None else "—")
    with c3:
        st.metric("trained_on", trained_on)
    with c4:
        st.caption("Threshold method")
        tag = thr_method.title() if isinstance(thr_method, str) else "—"
        st.markdown(
            f"<div style='display:inline-block;padding:.25rem .5rem;border:1px solid #e5e7eb;"
            f"border-radius:.5rem;background:#f8fafc;font-size:0.85rem;'>{tag}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ---- common metrics table ----
    metrics = _get(meta, "metrics", default={}) or {}
    rows = {
        "accuracy (acc)": metrics.get("acc", metrics.get("metric/acc", None)),
        "f1": metrics.get("f1", metrics.get("metric/f1", None)),
        "precision": metrics.get("precision", metrics.get("metric/precision", None)),
        "recall": metrics.get("recall", metrics.get("metric/recall", None)),
        "tpr (sensitivity)": metrics.get("tpr", metrics.get("metric/tpr", None)),
        "fpr": metrics.get("fpr", metrics.get("metric/fpr", None)),
        "AUC": metrics.get("auc", metrics.get("metric/auc", None)),
        "Brier": metrics.get("brier", metrics.get("metric/brier", None)),
        "Youden J": metrics.get("youden_j", metrics.get("metric/youden_j", None)),
        "Balanced acc": metrics.get("balanced_acc", metrics.get("metric/balanced_acc", None)),
    }
    table = (
        pd.DataFrame(
            [
                {"metric": k, "value": (None if v is None else float(v))}
                for k, v in rows.items()
                if v is not None
            ]
        )
        .sort_values("metric")
        .reset_index(drop=True)
    )
    if not table.empty:
        table["value"] = table["value"].map(lambda x: f"{x:.2f}")
        st.caption("Validation metrics (at best_thr)")
        st.table(table)
    else:
        st.info("No validation metrics found in model meta.")

    # export / raw
    cdl, _ = st.columns([1, 3])
    with cdl:
        st.download_button(
            "Download meta.json",
            data=json.dumps(meta, ensure_ascii=False, indent=2),
            file_name="eeg_mem_model.meta.json",
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Raw meta (JSON)", expanded=False):
        st.json(meta, expanded=False)


def render_model_info_panel_v2(*, model_path: str, meta_path: str):
    """
    更简洁的模型信息面板：
    - 顶部：操作按钮（Reload / Clear / Download / Reset threshold）
    - 中间：卡片式关键指标（Model, Date, Classifier, Features, Best Thr, ACC, F1）
    - 下方：彩色高亮的指标栅格（而不是大表格）
    - 最底：原始 JSON（收起）
    """
    st.markdown(
        """
        <style>
        .np-cards{display:grid;grid-template-columns:repeat(6, minmax(0,1fr));gap:12px;margin-top:6px;}
        .np-card{background:var(--background-color,#ffffff);border:1px solid rgba(0,0,0,0.06);
                 border-radius:12px;padding:12px 14px;}
        .np-k{font-size:12px;color:#6b7280;margin-bottom:6px;}
        .np-v{font-size:20px;font-weight:600;line-height:1.1;}
        .np-badges{display:grid;grid-template-columns:repeat(6, minmax(0,1fr));gap:12px;margin-top:12px;}
        .np-badge{background:#f8fafc;border:1px solid rgba(0,0,0,0.05);border-radius:10px;padding:10px;}
        .np-badge .k{font-size:12px;color:#6b7280;}
        .np-badge .v{font-size:18px;font-weight:600;}
        .np-good{color:#10b981;} .np-warn{color:#f59e0b;} .np-bad{color:#ef4444;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    ss = st.session_state
    meta_file = Path(meta_path)

    # ========= actions bar =========
    c1, c2, c3, c4, csp = st.columns([1,1,1,1,6])
    with c1:
        if st.button("🔄 Reload", use_container_width=True):
            if meta_file.exists():
                ss["model_meta"] = json.loads(meta_file.read_text(encoding="utf-8"))
                st.toast("Meta reloaded from disk.")
            else:
                st.warning(f"Meta file not found: {meta_file}")
    with c2:
        if st.button("🧹 Clear", use_container_width=True):
            ss.pop("model_meta", None)
            st.toast("Cleared meta from session.")
    with c3:
        # 下载按钮
        _meta_json = (ss.get("model_meta") or
                      (json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}))
        st.download_button(
            "⬇️ Download",
            data=json.dumps(_meta_json, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="model_meta.json",
            mime="application/json",
            use_container_width=True
        )
    with c4:
        # 一键恢复阈值到模型默认
        if st.button("🎯 Reset threshold", use_container_width=True):
            meta = ss.get("model_meta")
            if not meta and meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                ss["model_meta"] = meta
            best_thr = (meta or {}).get("best_thr", None)
            if isinstance(best_thr, (int, float)):
                ss["threshold"] = float(best_thr)
                st.toast(f"Threshold reset to model default: {best_thr:.2f}")
            else:
                st.warning("No best_thr found in meta.")

    # ========= load meta (pref session) =========
    meta = ss.get("model_meta")
    if meta is None and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        ss["model_meta"] = meta
    if not meta:
        st.info("No model metadata available.")
        return

    # basic fields
    name = meta.get("model_name", "—")
    date = meta.get("trained_on", "—")
    clf = meta.get("classifier", "—")
    feats = meta.get("features", "—")
    best_thr = meta.get("best_thr", None)
    thr_method = meta.get("threshold_method", "—")
    m = meta.get("metrics", {}) or {}

    # headline values
    acc = m.get("acc", meta.get("acc"))
    f1  = m.get("f1",  meta.get("f1"))

    # ---------- headline cards ----------
    cards_html = """
    <div class="np-cards">
      <div class="np-card"><div class="np-k">Model</div><div class="np-v">{name}</div></div>
      <div class="np-card"><div class="np-k">Trained on</div><div class="np-v">{date}</div></div>
      <div class="np-card"><div class="np-k">Classifier</div><div class="np-v">{clf}</div></div>
      <div class="np-card"><div class="np-k">Features</div><div class="np-v">{feats}</div></div>
      <div class="np-card"><div class="np-k">Best threshold</div><div class="np-v">{best_thr}</div></div>
      <div class="np-card"><div class="np-k">Method</div><div class="np-v">{thr_method}</div></div>
    </div>
    """.format(
        name=name,
        date=date,
        clf=clf,
        feats=feats,
        best_thr=(f"{best_thr:.2f}" if isinstance(best_thr, (int, float)) else "—"),
        thr_method=thr_method,
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    # ---------- metric badges ----------
    def _cls(v):
        if not isinstance(v, (int, float)): return ""
        if v >= 0.80: return "np-good"
        if v >= 0.60: return "np-warn"
        return "np-bad"

    order = ["acc","f1","precision","recall","tpr","fpr","youden_j","balanced_acc"]
    badges = []
    for k in order:
        v = m.get(k, None)
        vv = f"{v:.2f}" if isinstance(v, (int, float)) else "—"
        badges.append(
            f'<div class="np-badge"><div class="k">{k}</div><div class="v {_cls(v)}">{vv}</div></div>'
        )

    badges_html = '<div class="np-badges">' + ''.join(badges) + '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)

    # ========= raw JSON (collapsed) =========
    with st.container():
        st.code(json.dumps(meta, ensure_ascii=False, indent=2), language="json")


# -------------------- Main UI / Controls --------------------
# --- Live EEG (Muse via LSL) panel -------------------------------------------
from collections import deque
from typing import List
import time
import numpy as np
import pandas as pd


def pull_short_chunk(inlet, max_dur=0.5):
    """Pull up to max_dur seconds of samples using pull_sample and append to
    a small session-state ring buffer. Returns number of samples appended.

    This is a lightweight, single-shot grab (non-blocking long loops).
    """
    ss = st.session_state
    # ensure ring buffer exists (keep last ~2k points)
    if "ring" not in ss:
        ss["ring"] = deque(maxlen=2000)

    t_end = time.time() + float(max_dur)
    cnt = 0
    try:
        while time.time() < t_end:
            sample, ts = inlet.pull_sample(timeout=0.05)
            if sample is not None:
                ss["ring"].append(sample)
                cnt += 1
            else:
                time.sleep(0.01)
    except Exception:
        # swallow LSL errors; caller will display a message
        pass
    return cnt

def _lsl_connect_or_error(stream_type: str = 'EEG', timeout: float = 2.0, session_state: dict = None):
    """Try to resolve an LSL stream by property and return (inlet, error_message).

    Returns (inlet, None) on success, or (None, error_str) on failure.
    """
    try:
        from pylsl import resolve_byprop, StreamInlet
    except Exception as e:
        return None, f"Missing dependency pylsl: {e}"

    try:
        streams = resolve_byprop('type', stream_type, timeout=timeout)
        if not streams:
            return None, f"No LSL stream of type {stream_type} found within {timeout}s."
        inlet = StreamInlet(streams[0], max_buflen=360)

        # try to populate a minimal info dict and optionally write into session_state
        # Prefer centralized lsl module info if available
        try:
            from app.eeg import lsl as _lsl
        except Exception:
            _lsl = None

        sfreq = None
        labels = []
        n_ch = None
        try:
            if _lsl is not None:
                # assign inlet into central module and use its info cache
                try:
                    _lsl._LSL_INLET = inlet
                except Exception:
                    pass
                info = getattr(_lsl, '_LSL_INFO', {}) or {}
                try:
                    sfreq = int(info.get('sfreq')) if info.get('sfreq') is not None else None
                except Exception:
                    sfreq = None
                try:
                    n_ch = int(info.get('n_ch')) if info.get('n_ch') is not None else None
                except Exception:
                    n_ch = None
                labels = info.get('labels') or []
            else:
                # fallback to direct inlet query
                info = inlet.info()
                sfreq = int(info.nominal_srate())
                n_ch = info.channel_count()
                try:
                    chs = info.desc().child('channels')
                    for i in range(n_ch):
                        lab = chs.child('channel', i).child_value('label')
                        labels.append(lab or f"Ch{i+1}")
                except Exception:
                    labels = [f"Ch{i+1}" for i in range(n_ch)]
        except Exception:
            sfreq = None
            labels = []

        if session_state is not None:
            try:
                # Don't store the StreamInlet object in session_state (it's not serializable).
                _set_current_inlet(inlet)
                if sfreq is not None:
                    session_state["lsl_sfreq"] = sfreq
                if labels:
                    session_state["lsl_labels"] = labels
                # create a ring buffer placeholder (10s default)
                try:
                    from collections import deque
                    maxlen = (sfreq * 10) if sfreq else 2560
                    session_state.setdefault("lsl_ring", deque(maxlen=maxlen))
                except Exception:
                    pass
            except Exception:
                pass

        return inlet, None
    except Exception as e:
        return None, f"LSL connect error: {e}"


def render_live_eeg_panel():
    # Deprecated: blocking live panel replaced by non-blocking Live EEG helpers.
    # Keep a lightweight wrapper to preserve compatibility for any callers.
    try:
        st.warning(
            "render_live_eeg_panel() is deprecated — the app now uses a non-blocking Live EEG panel."
        )
        ensure_live_state()
        ph = st.empty()
        # Delegate to new non-blocking helper which will manage pulls and reruns.
        live_tick_and_plot(ph)
    except Exception:
        try:
            st.info("Live EEG unavailable (deprecated blocking panel).")
        except Exception:
            pass

# Legacy blocking live-stream helpers removed: replaced by the non-blocking
# live_tick_and_plot / start_live_view / stop_live_view helpers earlier in
# this file. The removed functions were: _init_stream_state, _bg_pull_loop,
# _auto_start_stream, _stop_stream, and show_live_chart. Kept file tidy.

def render_main_controls():
    global ss, status, timer_ph, chart_ph, spectrum_ph, feedback_ph, info_ph

    st.set_page_config(page_title="Muse EEG — Live Attention (optimized)", layout="wide")
    st.caption(
        "Stream EEG from Muse via LSL. Throttled UI refresh to reduce flicker. "
        "Includes: EMA smoothing, baseline calibration, confidence, power spectrum, "
        "task × difficulty logging, hourly recommendations with improvement %, and a Focus Meter (0–100%)."
    )

    # Ensure we attempt a connection the first time the page loads (best-effort)
    try:
        from app.eeg import lsl as _lsl
    except Exception:
        _lsl = None
    try:
        if _lsl is not None and getattr(_lsl, '_LSL_INLET', None) is None:
            _lsl.lsl_connect()
    except Exception:
        pass

    # Show LSL connection status/controls near the top
    try:
        lsl_status_block()
    except Exception:
        pass

    
    # session state
    ss = st.session_state
    ss.setdefault("collecting", False)
    ss.setdefault("inlet", None)
    ss.setdefault("buf", deque())
    ss.setdefault("ts", deque())
    ss.setdefault("started_at", None)
    ss.setdefault("stopped_at", None)
    ss.setdefault("last_draw", 0.0)
    ss.setdefault("last_duration_secs", None)
    ss.setdefault("ema_state", None)
    # init calibration flag
    ss.setdefault("is_calibrating", False)
    # Ensure baseline values are present in session_state on first load
    if "baseline_mu" not in ss:
        try:
            load_baseline_from_disk()
        except Exception:
            pass
    ss.setdefault("selected_task", "None")
    ss.setdefault("selected_diff", "Moderate")
    ss.setdefault("min_history_mode", "Standard (30 min)")
    ss.setdefault("min_history_min", 30.0)

    # placeholders
    status = st.empty()
    timer_ph = st.empty()
    chart_ph = st.empty()
    spectrum_ph = st.empty()
    feedback_ph = st.empty()
    info_ph = st.empty()

    # Try to load saved model bundle (convenience; non-fatal)
    try:
        model_loaded, thr_loaded, meta_loaded = load_model_bundle(MODEL_PATH)
        ss.setdefault("model_meta", meta_loaded)
    except Exception:
        ss.setdefault("model_meta", {})

    # One-Button EEG app (embedded) — local import to avoid side-effects
    try:
        with st.expander("🎛️ One-Button EEG demo", expanded=False):
            try:
                # local import so that this module doesn't trigger page config in the one-button file
                from app.eeg.one_button_app import render_one_button_app
                render_one_button_app()
            except Exception as e:
                st.info(f"One-button app unavailable: {e}")
    except Exception:
        # swallow UI errors
        pass

    # top controls
    col_task, col_diff, col_start, col_stop, col_export = st.columns([2, 1.3, 1, 1, 1])
    ss.selected_task = col_task.selectbox("Current Task",
                                          ["None", "Reading", "Coding", "Writing", "Memorization", "Review"],
                                          index=0)
    ss.selected_diff = col_diff.selectbox("Difficulty", ["Easy", "Moderate", "Challenging"], index=1)
    start_clicked = col_start.button("▶ Start", type="primary")
    stop_clicked = col_stop.button("▣ Stop")
    export_clicked = col_export.button("Export CSV")

    with st.expander("🧪 Baseline Calibration", expanded=False):
        st.caption("Run a relaxed 2–3 minute session to estimate your personal baseline.")
    
        bl = _load_baseline() or {"mu": 0.5, "sigma": 0.1, "ts": None}
        mu_now, sigma_now = _coerce_baseline(bl)
        ts_now = bl.get("ts", None)

        # Display current calibrated baseline
        st.write(f"Current baseline μ = {mu_now:.3f} , σ = {sigma_now:.3f}")
        if ts_now:
            st.caption(f"Last saved: {ts_now}")

        # Manual baseline slider (safe variable name: baseline_slider_val)
        baseline_slider_val = st.slider(
            "Adjust baseline (manual)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=mu_now,
            key="baseline_slider",
            help="Manually set baseline; press 'Save Baseline (slider)' to persist.")

        s_col1, s_col2 = st.columns([1, 1])
        with s_col1:
            if st.button("Save Baseline (slider)", key="save_baseline_slider"):
                # Save mu and keep current sigma via centralized persistence
                try:
                    mu_save = float(baseline_slider_val)
                except Exception:
                    mu_save = _guess_default_baseline()
                sigma_save = float(st.session_state.get("baseline_sigma", sigma_now))
                try:
                    st.session_state["baseline_mu"] = mu_save
                    st.session_state["baseline_sigma"] = sigma_save
                    st.session_state["baseline_ts"] = time.time()
                    save_baseline_to_disk(mu_save, sigma_save)
                    st.toast(f"Baseline saved: μ={mu_save:.3f}, σ={sigma_save:.3f}", icon="💾")
                except Exception:
                    st.toast("Failed to save baseline", icon="⚠️")
        with s_col2:
            if st.button("Reset Baseline (session)", key="reset_baseline_session"):
                # reset session baseline to guess default (migrate to dict)
                mu_def = _guess_default_baseline()
                # Update session state and persist
                mu_tmp = float(mu_def)
                sigma_tmp = float(st.session_state.get("baseline_sigma", 0.1))
                st.session_state["baseline_mu"] = mu_tmp
                st.session_state["baseline_sigma"] = sigma_tmp
                st.session_state["baseline_ts"] = time.time()
                save_baseline_to_disk(mu_tmp, sigma_tmp)
                st.toast("Baseline reset in session", icon="🔄")

        # Calibration controls: duration selector + run button
        col_a, col_b = st.columns([1, 1])
        with col_a:
            dur_min = st.selectbox("Calibration duration (min)", [2, 3], index=0, help="Recommended: 2–3 minutes")
        duration_sec = dur_min * 60

        with col_b:
            run_btn = st.button("🧪 Run calibration", type="primary", key="baseline_start")

        if run_btn:
            # initialize our non-blocking baseline state machine
            _baseline_init(duration_s=duration_sec, step_hz=8.0)
            try:
                st.toast("Calibration started…", icon="🧪")
            except Exception:
                pass
            _safe_rerun(0.1)

        # drive the non-blocking state machine on each rerun
        s = st.session_state.get("bl_state")
        if s and s.get("status") == "running":
            try:
                _baseline_tick(collector=collect_step)
            except Exception:
                pass
            st.progress(s.get("progress", 0.0))
            # allow user to cancel
            if st.button("Cancel calibration", key="baseline_stop"):
                try:
                    s["status"] = "canceled"
                    st.toast("Calibration canceled", icon="✖️")
                except Exception:
                    pass
            _safe_rerun(0.15)
        elif s and s.get("status") == "done":
            # ensure persisted and show success
            try:
                _persist_baseline(s.get("mu", 0.0), s.get("sigma", 0.1))
            except Exception:
                pass
            try:
                st.toast(f"Baseline done: μ={s['mu']:.3f}, σ={s['sigma']:.3f}", icon="✅")
            except Exception:
                st.success(f"Baseline done ✓ μ={s['mu']:.3f}, σ={s['sigma']:.3f}")
            # clear state after showing result
            try:
                st.session_state.pop("bl_state", None)
            except Exception:
                pass
        elif s and s.get("status") == "canceled":
            try:
                st.toast("Calibration stopped", icon="✖️")
                st.session_state.pop("bl_state", None)
            except Exception:
                pass

    # Start / Stop
    if start_clicked and not ss.collecting:
        try:
            inlet, err = _lsl_connect_or_error('EEG', timeout=1.5)
            if err is not None:
                status.warning(err)
                _set_current_inlet(None)
            else:
                try:
                    from app.eeg import lsl as _lsl2
                    _lsl2.lsl_set_inlet(inlet)
                except Exception:
                    _set_current_inlet(inlet)
                # store some convenient fields
                try:
                    # Prefer centralized LSL info cache when available
                    try:
                        info_cache = getattr(_lsl, '_LSL_INFO', {}) if _lsl is not None else {}
                    except Exception:
                        info_cache = {}

                    if info_cache and info_cache.get('sfreq'):
                        ss["lsl_sfreq"] = int(float(info_cache.get('sfreq')))
                    else:
                        info = inlet.info()
                        ss["lsl_sfreq"] = int(info.nominal_srate())

                    if info_cache and info_cache.get('labels'):
                        ss["lsl_labels"] = list(info_cache.get('labels'))
                    else:
                        # fallback to querying the inlet for channel labels
                        try:
                            chs = info.desc().child('channels')
                            labels = []
                            for i in range(info.channel_count()):
                                labels.append(chs.child('channel', i).child_value('label') or f"Ch{i+1}")
                            ss["lsl_labels"] = labels
                        except Exception:
                            pass
                except Exception:
                    pass
            ss.buf.clear()
            ss.ts.clear()
            ss.collecting = True
            ss.started_at = time.time()
            ss.stopped_at = None
            ss.last_draw = 0.0
            ss.last_duration_secs = None
            ss.ema_state = None
            status.success("Collecting EEG…")
        except Exception as e:
            ss.collecting = False
            _set_current_inlet(None)
            status.error(f"Could not open EEG stream: {e}")

    if stop_clicked and ss.collecting:
        ss.collecting = False
        _set_current_inlet(None)
        if len(ss.ts) > 0:
            ss.last_duration_secs = float(ss.ts[-1] - ss.ts[0])
        elif ss.started_at is not None:
            ss.last_duration_secs = float(time.time() - ss.started_at)
        else:
            ss.last_duration_secs = 0.0
        mm = int(ss.last_duration_secs // 60)
        ss_ = int(ss.last_duration_secs % 60)
        status.info(f"Stopped. Total duration: {mm:02d}:{ss_:02d}")
        try:
            save_session_history()
            record_session_summary(0.0, 0.0)
        except Exception as e:
            st.warning(f"Error saving history: {e}")

    if export_clicked:
        export_history_csv()

    # Main loop behaviour (driven by Streamlit reruns)
    if ss.collecting:
        if _get_current_inlet() is None:
            t0 = ss.started_at
            t = time.time() - t0
            sim = 0.5 * np.sin(0.2 * t) + 0.2 * np.random.randn(50)
            start_t = (ss.ts[-1] + STEP_SECS) if ss.ts else 0.0
            ss.buf.extend(sim)
            ss.ts.extend(np.linspace(start_t, start_t + STEP_SECS, 50))
        else:
            collect_step()

        elapsed = time.time() - (ss.started_at or time.time())
        timer_ph.caption(f"⏱ Elapsed: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

        if time.time() - ss.last_draw >= DRAW_EVERY_SEC:
            rel_now, ratio_now, conf = draw_chart_and_metrics()
            draw_power_spectrum()
            ss.last_draw = time.time()

        time.sleep(STEP_SECS)
        st.rerun()

    # Non-collecting final frame
    if not ss.collecting and len(ss.buf) > 0:
        rel_now, ratio_now, conf = draw_chart_and_metrics()
        draw_power_spectrum()

def main():
    # Render main controls (Live EEG UI is inside render_main_controls)
    render_main_controls()

    # --------- Threshold with default = best_thr + reset & persist & source badge ---------
    import json
    from pathlib import Path

    # Ensure APP_HOME exists
    APP_HOME.mkdir(parents=True, exist_ok=True)
    # Use module-level JSON helpers (deprecated local aliases kept for backward compatibility)
    _load_json = globals().get("_load_json") or (lambda p: {})
    _save_json = globals().get("_save_json") or (lambda p, data: None)
    _load_meta_from_disk = globals().get("_load_meta_from_disk") or (lambda base_path: {})
    _safe_best_thr = globals().get("_safe_best_thr") or (lambda meta, fallback=0.5: fallback)

    # 1) load meta (session -> disk)
    _meta = st.session_state.get("model_meta") or _load_meta_from_disk(MODEL_PATH)
    model_best_thr = _safe_best_thr(_meta, fallback=0.5)
    st.session_state.setdefault("model_best_thr", model_best_thr)

    # 2) try saved user threshold (cross-session)
    _settings = _load_json(SETTINGS_FILE)
    saved_thr = _settings.get("custom_threshold")

    # 3) compute default priority
    default_source = "From meta"
    _default_thr = model_best_thr

    if isinstance(saved_thr, (int, float)):
        _default_thr = float(saved_thr)
        default_source = "From saved setting"
    elif "MODEL_LOADED_THRESHOLD" in globals():
        try:
            _default_thr = float(globals()["MODEL_LOADED_THRESHOLD"])
            default_source = "From loaded model"
        except Exception:
            pass

    if "threshold" in st.session_state:
        try:
            _default_thr = float(st.session_state["threshold"])
            default_source = "From session"
        except Exception:
            pass

    # 4) UI: left slider + right info
    thr_col, info_col = st.columns([1, 0.55])

    with thr_col:
        st.session_state["threshold"] = st.slider(
            "Threshold (Adjustable)",
            min_value=0.0, max_value=1.0, step=0.01,
            value=float(_default_thr),
            help="Default comes from model meta ('best_thr'); you can still adjust it here."
        )

    with info_col:
        c1, c2, c3 = st.columns([0.45, 0.4, 0.6])
        with c1:
            st.caption("Model default")
            st.metric("best_thr", f"{st.session_state['model_best_thr']:.2f}")
        with c2:
            st.caption("Default source")
            st.write(f"**{default_source}**")
        with c3:
            st.caption("Actions")
            if st.button("Reset to model default", use_container_width=True):
                st.session_state["threshold"] = float(st.session_state["model_best_thr"])
                st.toast("Threshold reset to model default", icon="🔄")

    # ---- Baseline UI (show current baseline and quick actions) ----
    try:
        baseline_now = _load_baseline()
        # normalize for display
        baseline_mu, baseline_sigma = _coerce_baseline(baseline_now)
    except Exception:
        # Fallback numeric defaults if anything goes wrong
        baseline_now = 0.5
        baseline_mu = float(baseline_now)
        baseline_sigma = float(st.session_state.get("baseline_sigma", 0.1))

    bcol1, bcol2, bcol3 = st.columns([1, 0.6, 0.6])
    with bcol1:
        st.metric("Baseline", f"{baseline_mu:.2f}", help="Baseline used for focus/memory normalization")
    with bcol2:
        if st.button("Save Baseline", use_container_width=True):
            try:
                mu_tmp = float(baseline_now)
            except Exception:
                mu_tmp = _guess_default_baseline()
            sigma_tmp = float(st.session_state.get("baseline_sigma", 0.1))
            st.session_state["baseline_mu"] = mu_tmp
            st.session_state["baseline_sigma"] = sigma_tmp
            st.session_state["baseline_ts"] = time.time()
            save_baseline_to_disk(mu_tmp, sigma_tmp)
            st.toast("Baseline saved to settings", icon="💾")
    with bcol3:
        if st.button("Reset Baseline", use_container_width=True):
            # Reset baseline using the safe writer which persists and updates session
            mu_def = _guess_default_baseline()
            mu_tmp = float(mu_def)
            sigma_tmp = float(st.session_state.get("baseline_sigma", 0.1))
            st.session_state["baseline_mu"] = mu_tmp
            st.session_state["baseline_sigma"] = sigma_tmp
            st.session_state["baseline_ts"] = time.time()
            save_baseline_to_disk(mu_tmp, sigma_tmp)
            st.toast("Baseline reset in session", icon="🔄")

    # 5) save / clear persisted default
    save_col, clear_col = st.columns([0.25, 0.2])
    with save_col:
        if st.button("Save as default", help="Persist current threshold across sessions"):
            _settings = _load_json(SETTINGS_FILE)
            _settings["custom_threshold"] = float(st.session_state["threshold"])
            _save_json(SETTINGS_FILE, _settings)
            st.toast(f"Saved {st.session_state['threshold']:.2f} as default", icon="💾")

    with clear_col:
        if st.button("Clear saved default", help="Remove persisted custom threshold"):
            _settings = _load_json(SETTINGS_FILE)
            if "custom_threshold" in _settings:
                _settings.pop("custom_threshold")
                _save_json(SETTINGS_FILE, _settings)
                st.toast("Cleared saved default", icon="🧹")
    # --------------------------------------------------------------------

    # Always show recommendation block
    # ------- Model Info (expander) -------
    import json
    from pathlib import Path

    def _load_meta_from_disk(base_path: str) -> dict:
        """Best-effort load of model meta JSON from disk."""
        meta_path = Path(f"{base_path}.json")
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        # compatibility: enhanced save may write .meta.json
        meta_path2 = Path(f"{base_path}.meta.json")
        if meta_path2.exists():
            try:
                with open(meta_path2, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # Prefer session-state meta (loaded earlier) or try disk
    meta = st.session_state.get("model_meta") or _load_meta_from_disk(MODEL_PATH)

    # Render compact model info panel (separate helper)
    with st.expander("📄 View Model Info", expanded=False):
        render_model_info_panel_v2(model_path=f"{MODEL_PATH}.model", meta_path=f"{MODEL_PATH}.json")

    # ==== Auto-DP (minimal) ====
    from app.eeg.algorithms import dp_confidence_weighted_predict

    def _get_dp_from_session() -> 'AdaptiveDPController':
        # Return the session DP controller, creating from persisted settings if needed
        if "dp_controller" not in st.session_state:
            # Prefer persisted settings; fallback to a canonical default
            try:
                dp = AdaptiveDPController.from_settings()
            except Exception:
                dp = AdaptiveDPController(total_epsilon=5.0, delta=1e-5, strategy="per_call", eps_per_unit=1.0, S=1.0)
            st.session_state["dp_controller"] = dp
        return st.session_state["dp_controller"]


    # ---- DP Controller accessor (unify session key) ----
    def get_dp_controller():
        ss = st.session_state
        if "dp_controller" not in ss:
            # create a sensible default controller compatible with current AdaptiveDPController API
            ss["dp_controller"] = AdaptiveDPController(total_epsilon=5.0, delta=1e-5, strategy="per_call", eps_per_unit=1.0, S=1.0)
        return ss["dp_controller"]


    def apply_auto_dp_and_render(X_now, bundle, thr: float):
        """
        X_now: current window features (n_samples, n_features)
        bundle: dict with keys like 'svm' and/or 'mlp' mapping to fitted estimators
        thr: decision threshold
        """
        # 1) build model wrappers (tolerant to missing models)
        models = []
        if "svm" in bundle and bundle["svm"] is not None:
            models.append(algo.TrainedModel("svm", bundle["svm"]))
        if "mlp" in bundle and bundle["mlp"] is not None:
            models.append(algo.TrainedModel("mlp", bundle["mlp"]))
        if not models:
            st.error("No model available in bundle.")
            return

        # 2) ensemble probability
        try:
            probs = algo.confidence_weighted_ensemble(models, X_now)
        except Exception:
            probs = np.asarray([0.5])
        prob_raw = float(np.mean(probs)) if np.size(probs) else 0.5

        # 3) Auto-DP: request epsilon and add noise (controller accounts spent epsilon)
        dp = get_dp_controller()
        # request epsilon for this call (controller enforces strategy)
        eps_t = dp.request_epsilon(units=1.0)
        if eps_t and eps_t > 0:
            prob_private = dp.add_noise_to_scalar(prob_raw, epsilon=eps_t)
        else:
            prob_private = prob_raw
        eps_used = float(dp.spent_epsilon)
        eps_total = float(dp.total_epsilon)

        # 4) decision (use raw or private probability as desired)
        label = int(prob_raw >= float(thr))

        # 5) render
        st.metric("Predicted Probability (Private Output)", f"{prob_private:.2f}")
        st.write("Decision:", "**HIGH**" if label == 1 else "LOW")

        with st.container():
            conf = abs(prob_raw - 0.5) * 2.0
            st.caption("系统已自动按置信度与剩余预算分配 ε 并加噪。")
            st.write(f"• Model confidence: **{conf:.2f}**")
            st.write(f"• ε_t used this step: **{eps_t:.3f}" if eps_t is not None else "• ε_t used this step: N/A")
            prog = min(1.0, eps_used / max(1e-6, eps_total))
            st.progress(prog, text=f"Privacy budget {eps_used:.2f} / {eps_total:.2f}")
            if dp.exhausted:
                st.warning("Privacy budget exhausted — switched to safe mode.")


        def safe_apply_auto_dp_and_render(X_now, bundle, thr: float):
            try:
                return apply_auto_dp_and_render(X_now, bundle, thr)
            except Exception as e:
                try:
                    st.error(f"Prediction / Auto-DP error: {e}")
                except Exception:
                    pass
                return None

    with st.expander("🔒 Auto-DP (adaptive differential privacy)", expanded=False):
        dp = _get_dp_from_session()
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            dp.strategy = st.selectbox("Strategy", ["per_call", "per_epoch", "per_minute"], index=["per_call", "per_epoch", "per_minute"].index(dp.strategy))
            dp.total_epsilon = st.number_input("Total ε budget", min_value=0.1, max_value=50.0, step=0.1, value=float(dp.total_epsilon))
            dp.eps_per_unit = st.number_input("ε per unit (auto if unsure)", min_value=0.001, max_value=5.0, step=0.001, value=float(dp.eps_per_unit))
            dp.delta = st.number_input("δ (Gaussian)", min_value=1e-12, max_value=1e-1, step=1e-6, value=float(dp.delta), format="%.1e")
        with col2:
            st.metric("Remaining ε", f"{dp.remaining:.3f}", help="Leftover privacy budget")
            st.metric("Spent ε", f"{dp.spent_epsilon:.3f}")
        with col3:
            st.button("Save DP settings", on_click=lambda: dp.save_settings())
            st.button("Clear DP settings", on_click=lambda: (dp.clear_settings(), st.rerun()))

        # optional visual progress
        st.progress(min(1.0, dp.spent_epsilon / max(1e-9, dp.total_epsilon)))

    # ---- Advanced: Train / Optimize UI (optional) ----
    with st.expander("🔬 Train / Optimize (advanced)", expanded=False):
        st.write("Bayesian Optimization / confidence-weighted ensemble / adaptive threshold / DP noise")

        # Example: pull training data from session_state (replace with your real source)
        X = st.session_state.get("X_train_demo", None)
        y = st.session_state.get("y_train_demo", None)

        if X is None or y is None:
            st.info("No training data found. Provide X/y (e.g., from your capture history) and store them in st.session_state['X_train_demo'], ['y_train_demo'].")
        else:
            algo_name = st.selectbox("Algorithm", ["LinearSVC (Bayes)", "MLP (Bayes)"])
            colA, colB, colC = st.columns(3)
            with colA:
                do_adapt = st.checkbox("Adaptive threshold", value=False, help="Self-tune threshold by target accuracy")
            with colB:
                do_dp = st.checkbox("DP noise on output", value=False, help="Add Gaussian DP noise to output prob/score")
            with colC:
                target_acc = st.slider("Target acc for adapt", 0.5, 0.95, 0.80, 0.01)

            # DP controller (optional) — load from saved settings and allow editing
            dp = None
            try:
                dp = AdaptiveDPController.from_settings()
            except Exception:
                dp = None

            with st.container():
                if dp is None:
                    st.info("DP controller unavailable. Ensure `app.eeg.adp.adaptive_dp.AdaptiveDPController` is importable.")
                else:
                    dcol1, dcol2, dcol3 = st.columns(3)
                    with dcol1:
                        total_eps = st.number_input("Total ε (budget)", min_value=0.0, value=float(dp.total_epsilon), format="%.3f")
                        strategy_opts = ["per_call", "per_epoch", "per_minute"]
                        strategy_idx = strategy_opts.index(dp.strategy) if dp.strategy in strategy_opts else 0
                        strategy_sel = st.selectbox("Strategy", strategy_opts, index=strategy_idx)
                    with dcol2:
                        eps_unit = st.number_input("ε per unit", min_value=0.0, value=float(dp.eps_per_unit), format="%.4f")
                        sens = st.number_input("Sensitivity S", min_value=0.0, value=float(dp.S), format="%.4f")
                    with dcol3:
                        st.metric("Remaining ε", f"{dp.remaining:.4f}")
                        if st.button("Save DP settings"):
                            dp.total_epsilon = float(total_eps)
                            dp.strategy = strategy_sel
                            dp.eps_per_unit = float(eps_unit)
                            dp.S = float(sens)
                            dp.save_settings()
                            st.toast("DP settings saved", icon="💾")
                        if st.button("Clear DP settings"):
                            dp.clear_settings()
                            st.toast("DP settings cleared", icon="🧹")

            can_train = (train_linear_svm_bayes is not None and train_mlp_bayes is not None and save_bundle is not None)
            if not can_train:
                st.warning("Training helpers not available in this environment. Add `src/app/eeg/train_and_save.py` to enable training in the UI.")

            if st.button("🚀 Train & Save bundle"):
                if not can_train:
                    st.error("Training helpers missing. Cannot run training.")
                else:
                    with st.spinner("Training..."):
                        try:
                            if algo_name.startswith("LinearSVC"):
                                model, meta2 = train_linear_svm_bayes(X, y)
                            else:
                                model, meta2 = train_mlp_bayes(X, y)

                            # Optionally write a chosen threshold into meta2 here
                            # meta2["best_thr"] = your_best_thr
                            # Save training data into session_state for later demos/ensembles
                            st.session_state["X_train_demo"] = X   # shape [n_samples, n_features]
                            st.session_state["y_train_demo"] = y   # shape [n_samples]

                            save_bundle(model, meta2)
                            # Auto-update session meta and refresh UI so model info panel shows new meta
                            try:
                                new_meta = _load_meta_from_disk("model_store/eeg_memory")
                                st.session_state["model_meta"] = new_meta
                                st.success("Model + meta saved and UI updated to reflect new metadata.")
                                st.rerun()
                            except Exception:
                                st.success("Model + meta saved to `model_store/eeg_memory/`")
                        except Exception as e:
                            st.error(f"Training failed: {e}")

            st.divider()
            st.markdown("**Ensemble demo**")

            try:
                m1, meta1 = algo.load_model_bundle("model_store/eeg_memory")
                m2, meta2 = m1, meta1  # demo: duplicate model
                # wrap into algo.TrainedModel if available otherwise use tuples compatible with algo
                models = [getattr(algo, 'TrainedModel')("m1", m1), getattr(algo, 'TrainedModel')("m2", m2)]
                X_eval = X[:32]

                # use DP-aware ensemble predictor if requested
                thr = st.session_state.get("threshold", 0.5)
                acc_demo = 0.80
                if do_adapt:
                    thr = algo.update_threshold(thr, acc_demo, target_acc=target_acc)
                    st.write(f"Adaptive thr → {thr:.2f}")

                # --- DP-aware ensemble using helper runner ---
                dp = st.session_state.get("dp_controller", None)
                thr = st.session_state.get("threshold", st.session_state.get("model_best_thr", 0.5))

                try:
                    probs_to_show, labels, info = algo.run_ensemble_with_dp(
                        models=models,
                        X_eval=X_eval,
                        dp_controller=dp,
                        threshold=float(thr),
                        low_conf_gate=0.10,
                        safe_mode_trend=0.20,
                    )
                except Exception as e:
                    st.error(f"Ensemble helper failed: {e}. Falling back to basic ensemble.")
                    with infer_timer():
                        raw = algo.confidence_weighted_ensemble(models, X_eval)
                    probs_to_show = np.asarray(raw, dtype=float)
                    labels = (probs_to_show >= float(thr)).astype(int)
                    info = {"used_dp": bool(dp), "safe_mode": False, "raw_probs": probs_to_show}

                # UI hints
                if info.get("used_dp"):
                    st.success("DP controller available — running DP ensemble.")
                else:
                    st.info("DP controller not found — running standard ensemble.")

                if info.get("safe_mode"):
                    st.warning("Privacy budget exhausted. Safe Mode: showing trends only.")

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Probabilities (display)**")
                    st.write(np.round(probs_to_show, 3))
                with col2:
                    st.metric("Model default best_thr", st.session_state.get("model_best_thr", 0.5))
                    st.metric("Current threshold", float(thr))

                st.session_state["last_probs"] = info.get("raw_probs", np.asarray(probs_to_show, dtype=float))
                st.session_state["last_labels"] = labels
            except Exception as e:
                st.info(f"Load/ensemble demo: {e}")

    # 🟢 Live EEG panel (Muse via LSL)
    def lsl_status_block():
        import streamlit as st
        try:
            from app.eeg import lsl as _lsl
        except Exception:
            _lsl = None

        col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.2])

        # 状态
        if _lsl is None:
            status = "disconnected"
            info = {}
        else:
            try:
                status, info = _lsl.lsl_health()
            except Exception:
                status, info = "error", {"last_err": "lsl module error"}

        if status == "ok":
            pill = "🟢 Connected"
            help_msg = f"{info.get('name','?')} | ch={info.get('n_ch','?')} | fs={info.get('sfreq','?')}Hz"
        elif status == "disconnected":
            pill = "⚪️ Disconnected"
            help_msg = "No inlet"
        else:
            pill = "🔴 Error"
            help_msg = info.get("last_err", "unknown")

        with col1:
            st.markdown(f"**LSL:** {pill}")
            st.caption(help_msg)

        with col2:
            if st.button("🔌 Reconnect", use_container_width=True):
                try:
                    if _lsl is not None:
                        _lsl.lsl_clear()
                        ok = _lsl.lsl_connect()
                    else:
                        ok = False
                    st.toast("Reconnected ✅" if ok else "Reconnect failed ❌", icon="🔁")
                except Exception as e:
                    st.toast(f"Reconnect failed: {e}")

        with col3:
            if st.button("🧹 Clear inlet", use_container_width=True):
                try:
                    if _lsl is not None:
                        _lsl.lsl_clear()
                    st.toast("Cleared inlet", icon="🧹")
                except Exception as e:
                    st.toast(f"Clear failed: {e}")

        with col4:
            try:
                consec = _lsl._LSL_CONSEC_TIMEOUTS if _lsl is not None else 0
            except Exception:
                consec = 0
            st.caption(f"timeouts: {consec}  ·  ε-DP: {st.session_state.get('epsilon','–')}")

        # --- DEBUG: list discovered LSL streams (1s timeout) ---
        try:
            import streamlit as st
            try:
                from pylsl import resolve_streams
                streams = resolve_streams(wait_time=1.0)
                st.caption("🔎 LSL streams seen:")
                st.write([(s.name(), s.type(), s.hostname(), s.source_id()) for s in streams])
            except Exception as e:
                st.caption(f"LSL resolve error: {e!r}")
        except Exception:
            # Best-effort; don't let debug block the UI
            pass

        # --- DEBUG: inlet & pull stats ---
        try:
            ss = st.session_state
            inlet_ok = bool(ss.get("_LSL_INLET"))  # if you keep inlet in session_state; module-level inlet lives in app.eeg.lsl
            ring_len = len(ss.get("ring", []))
            st.write({
                "inlet_ok": inlet_ok,
                "ring_len": ring_len,
                "last_pull_wall": ss.get("metrics", {}).get("last_pull_wall"),
            })
        except Exception:
            pass

    # =============== Live EEG (Muse via LSL) UI ===============
    st.subheader("🧠 Live EEG (Muse via LSL)")
    ensure_live_state()

    # 顶部操作区（给按钮加唯一 key）
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        if st.button("Reconnect", key="live_reconnect"):
            resolve_stream_or_toast()
    with c2:
        if st.button("Start", type="primary", key="live_start"):
            cfg = st.session_state.get('live_cfg', {})
            start_live_view(cfg.get('window_sec', 10), cfg.get('refresh_hz', 8), cfg.get('run_sec', None))
    with c3:
        if st.button("Stop", key="live_stop"):
            stop_live_view()
    with c4:
        live = "🟢 Running" if st.session_state.get('live_running') else "⚪ Stopped"
        st.write(live)

    # 控件：窗口 / 刷新率 / 运行时长（也加唯一 key，防止和其它面板的 slider 撞名）
    cfg = st.session_state.get('live_cfg', {})
    ws = st.slider("Window (sec)", 2, 60, int(cfg.get('window_sec', 10)), 1,
                   help="波形显示的时间窗口长度", key="live_win")
    rh = st.slider("Refresh (Hz)", 1, 20, int(cfg.get('refresh_hz', 8)), 1,
                   help="UI 刷新频率（非阻塞）", key="live_hz")
    rs = st.slider("Run for (sec)", 0, 300, int(cfg.get('run_sec') or 0), 5,
                   help="到点自动停止；0 表示不限时", key="live_run")

    st.session_state['live_cfg'] = dict(
        window_sec=ws, refresh_hz=rh, run_sec=(rs or None), t0=cfg.get('t0', time.time())
    )

    # 波形占位与 tick（占位不需要 key）
    plot_ph = st.empty()
    live_tick_and_plot(plot_ph)

    # Diagnostics panel: show I/O & shape contracts. Pass expected feature dim (20) per user's note.
    try:
        diagnostics_panel_plus(n_features_expected=20)
    except Exception:
        # keep UI resilient if diagnostics rendering fails for any reason
        pass

    # ===== Auto-DP + prediction wiring (uses existing `algo` helpers and loaded model) =====
    # Accessor for an 'adp' session controller (uses the canonical adp module)
    def _get_adp():
        ss = st.session_state
        if "adp" not in ss:
            try:
                adp = AdaptiveDPController.from_settings()
            except Exception:
                adp = AdaptiveDPController(
                    total_epsilon=float(ss.get("dp_eps_total", 5.0)),
                    delta=1e-5,
                    strategy=str(ss.get("dp_strategy", "per_call")),
                    eps_per_unit=float(ss.get("dp_eps_base", 1.0)),
                    S=float(ss.get("dp_S", 1.0)),
                    user_id=str(ss.get("user_id", "guest")),
                    session_id=str(ss.get("session_id", "sess001")),
                    alpha=float(ss.get("dp_alpha", 0.8)),
                    eps_floor=float(ss.get("dp_eps_floor", 0.01)),
                )
            ss["adp"] = adp
        return ss["adp"]

    adp = _get_adp()

    # Use helper to apply auto-DP and render prediction (reads bundle/thr from session when available)
    if "X_window_features" in st.session_state:
        X_now = st.session_state["X_window_features"]
        # bundle compatibility: prefer st.session_state['bundle'] if present
        bundle = st.session_state.get("bundle", None)
        if bundle is None:
            # fall back to loaded model(s)
            if MODEL_LOADED is not None:
                bundle = {"svm": MODEL_LOADED, "mlp": MODEL_LOADED}
            else:
                m, t, md = _load_eeg_model()
                bundle = {"svm": m, "mlp": m}
        # thr unified from session
        thr = float(st.session_state.get('threshold', 0.5))
        try:
            apply_auto_dp_and_render(X_now, bundle, thr)
        except Exception as e:
            try:
                st.error(f"Prediction / Auto-DP error: {e}")
            except Exception:
                pass
    else:
        st.info("Waiting for EEG features (X_window_features) …")

    ai_recommendation_block()


if __name__ == "__main__":
    main()

