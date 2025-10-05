import io
import time
from collections import deque
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from pylsl import StreamInlet, resolve_stream

# -------------------- Page --------------------
st.set_page_config(page_title="Muse EEG — Live Attention (optimized)", layout="wide")
st.markdown("## 🧠 Muse EEG — Live Attention (optimized)")
st.caption(
    "Stream EEG from Muse via LSL. Throttled UI refresh to reduce flicker. "
    "Includes: EMA smoothing, baseline calibration, confidence, power spectrum, "
    "task × difficulty logging, hourly recommendations with improvement %, and a Focus Meter (0–100%)."
)

# -------------------- Parameters --------------------
WINDOW_SECS       = 60.0
STEP_SECS         = 0.20
DRAW_EVERY_SEC    = 5.0
MAX_PLOT_POINTS   = 900
MAX_POINTS_CAP    = 40000
RATIO_WIN_SEC     = 2.0
SAVE_RECORD_EVERY = 30.0          # history aggregation (s)

EMA_ALPHA         = 0.2
TOPN              = 3

APP_HOME          = Path.home() / ".eeg_app"
HISTORY_PATH      = APP_HOME / "history.csv"
BASELINE_PATH     = APP_HOME / "baseline.json"
SESSIONS_PATH     = APP_HOME / "sessions.csv"

# ---- CSV schemas (memory_index added) ----
HISTORY_COLS  = ["ts","local_time","hour","ratio","memory_index","task","difficulty"]
SESSIONS_COLS = ["ts","local_time","duration_sec","rel_focus_mean","confidence","task","difficulty"]

# -------------------- Session State --------------------
ss = st.session_state
if "collecting"         not in ss: ss.collecting  = False
if "inlet"              not in ss: ss.inlet       = None
if "buf"                not in ss: ss.buf         = deque()
if "ts"                 not in ss: ss.ts          = deque()
if "started_at"         not in ss: ss.started_at  = None
if "stopped_at"         not in ss: ss.stopped_at  = None
if "last_draw"          not in ss: ss.last_draw   = 0.0
if "last_duration_secs" not in ss: ss.last_duration_secs = None
if "ema_state"          not in ss: ss.ema_state   = None
if "selected_task"      not in ss: ss.selected_task = "None"
if "selected_diff"      not in ss: ss.selected_diff = "Moderate"
# History requirement mode
if "min_history_mode"   not in ss: ss.min_history_mode = "Standard (30 min)"
if "min_history_min"    not in ss: ss.min_history_min  = 30.0

# -------------------- Storage helpers --------------------
def _ensure_dirs():
    APP_HOME.mkdir(parents=True, exist_ok=True)

def _ensure_csv(path: Path, cols: List[str]):
    _ensure_dirs()
    if not path.exists():
        pd.DataFrame(columns=cols).to_csv(path, index=False)

def _load_csv(path: Path, cols: List[str]) -> pd.DataFrame:
    _ensure_csv(path, cols)
    try:
        df = pd.read_csv(path)
        for c in cols:
            if c not in df.columns: df[c] = np.nan
        return df[cols] if not df.empty else pd.DataFrame(columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)

def _append_csv(path: Path, rows: List[Dict[str,Any]], cols: List[str]):
    if not rows: return
    _ensure_dirs()
    df_new = pd.DataFrame(rows)[cols]
    if path.exists() and path.stat().st_size > 0:
        try:
            df_old = pd.read_csv(path)
            for c in cols:
                if c not in df_old.columns: df_old[c] = np.nan
            df = pd.concat([df_old[cols], df_new], ignore_index=True)
        except Exception:
            df = df_new
    else:
        df = df_new
    df.to_csv(path, index=False)

# -------------------- Signal processing --------------------
def _estimate_fs(ts: np.ndarray) -> float:
    if ts.size < 2: return 256.0
    diffs = np.diff(ts); diffs = diffs[diffs > 0]
    return float(1.0/np.median(diffs)) if diffs.size else 256.0

def _bandpower(sig: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    if sig.size == 0 or fs <= 0: return 0.0
    win = np.hanning(sig.size); x = sig*win
    spec = np.fft.rfft(x); psd = (np.abs(spec)**2)/np.sum(win**2)
    freqs = np.fft.rfftfreq(sig.size, d=1.0/fs)
    idx = (freqs >= fmin) & (freqs <= fmax)
    return float(np.mean(psd[idx])) if np.any(idx) else 0.0

def beta_alpha_ratio(sig: np.ndarray, fs: float) -> float:
    beta  = _bandpower(sig, fs, 13.0, 30.0)
    alpha = _bandpower(sig, fs, 8.0, 13.0)
    return float(beta/(alpha+1e-8))

# ---- Memory-related indices (NEW) ----
def theta_gamma_index(sig: np.ndarray, fs: float) -> float:
    """Memory encoding/retrieval proxy: (theta + low-gamma) normalized by alpha.
    Muse 对高频较弱，采用较保守的 low-gamma 30–45Hz；若噪声较大，可将上限改为 40Hz。
    """
    theta = _bandpower(sig, fs, 4.0, 7.0)
    gamma = _bandpower(sig, fs, 30.0, 45.0)
    alpha = _bandpower(sig, fs, 8.0, 12.0)
    return float((theta + gamma) / (alpha + 1e-8))

def ema_update(x: float) -> float:
    ss.ema_state = x if ss.ema_state is None else EMA_ALPHA*x + (1-EMA_ALPHA)*ss.ema_state
    return ss.ema_state

# --- Baseline helpers ---
def _load_baseline():
    import json
    _ensure_dirs()
    if not BASELINE_PATH.exists(): return None
    try: return json.loads(BASELINE_PATH.read_text())
    except Exception: return None

def _save_baseline(mu: float, sigma: float):
    import json
    _ensure_dirs()
    BASELINE_PATH.write_text(json.dumps({"mu": float(mu), "sigma": float(sigma)}))

def relative_focus(cur_ba: float, base_mu: Optional[float], base_sigma: Optional[float]) -> float:
    if base_mu is None or base_sigma is None or base_sigma < 1e-6:
        z = cur_ba - 1.0  # neutral center if no baseline
    else:
        z = (cur_ba - base_mu) / base_sigma
    return float(1/(1+np.exp(-z)))  # 0~1

# ---- Relative Memory score (NEW) ----
def relative_memory_score(cur_mi: float, base_mu: Optional[float], base_sigma: Optional[float]) -> float:
    """Map memory index to 0–1 using the same z→sigmoid scheme as focus.
    If no baseline, center at 1.0 as neutral.
    """
    if base_mu is None or base_sigma is None or base_sigma < 1e-6:
        z = cur_mi - 1.0
    else:
        z = (cur_mi - base_mu) / base_sigma
    return float(1/(1+np.exp(-z)))

def compute_confidence(duration_sec: float, snr_est: float, contact_ok: bool=True) -> float:
    t_score = min(duration_sec/180.0, 1.0)
    snr_score = float(np.clip((snr_est-3)/7, 0, 1))
    c_score = 1.0 if contact_ok else 0.3
    return float(np.clip(0.5*t_score + 0.4*snr_score + 0.1*c_score, 0, 1))

# -------------------- LSL --------------------
def connect_lsl(kind="EEG", timeout=2.5, retries=3):
    for _ in range(retries):
        try:
            streams = resolve_stream("type", kind, timeout=timeout)
            if streams: return StreamInlet(streams[0])
        except Exception:
            pass
        time.sleep(1.0)
    return None

# -------------------- Acquisition --------------------
def collect_step():
    if ss.inlet is None: return
    chunk, stamps = ss.inlet.pull_chunk(timeout=STEP_SECS, max_samples=int(1024*STEP_SECS))
    if not chunk: return
    vals = [row[0] for row in chunk]
    t0 = ss.started_at or time.time()
    ts_rel = [s - t0 for s in stamps]
    ss.buf.extend(vals); ss.ts.extend(ts_rel)

    now_rel = time.time() - t0
    while ss.ts and (now_rel - ss.ts[0] > WINDOW_SECS):
        ss.ts.popleft(); ss.buf.popleft()
    while len(ss.buf) > MAX_POINTS_CAP:
        ss.ts.popleft(); ss.buf.popleft()

def _downsample(ts: np.ndarray, ys: np.ndarray, max_points: int):
    if ts.size <= max_points: return ts, ys
    idx = np.linspace(0, ts.size-1, max_points).astype(int)
    return ts[idx], ys[idx]

def _compute_series_ratio(ts: np.ndarray, vals: np.ndarray, fs: float,
                          win_sec=RATIO_WIN_SEC, step_sec=0.5):
    if ts.size == 0: return np.array([]), np.array([])
    t_min, t_max = ts.min(), ts.max()
    if t_max - t_min < 0.5: return np.array([]), np.array([])
    out_t, out_r = [], []
    t = t_min + win_sec
    while t <= t_max:
        idx = np.where((ts >= t-win_sec) & (ts <= t))[0]
        if idx.size > 10:
            out_t.append(t)
            out_r.append(ema_update(beta_alpha_ratio(vals[idx], fs)))
        t += step_sec
    return np.asarray(out_t), np.asarray(out_r)

def _state_label(r_rel: float):
    if np.isnan(r_rel): return "Unknown","⚪"
    if r_rel < 0.45:    return "Low","🟡"
    if r_rel < 0.65:    return "Medium","🟢"
    return "High","🔵"

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
    pct = int(np.clip(rel_now,0,1)*100)
    hint = (
        "Excellent!" if rel_now >= 0.7 else (
        "Good, keep it steady." if rel_now >= 0.5 else "Try a short break or slow breathing.")
    )
    container.markdown(
        f"""
        <div style="margin:8px 0 4px 0;font-weight:600;">{label}: {k}/{n} ({pct}%)</div>
        <div style="display:flex;align-items:center;">{meter}</div>
        <div style="opacity:0.7;margin-top:6px;">Hint: {hint}</div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- UI placeholders --------------------
status     = st.empty()
timer_ph   = st.empty()
chart_ph   = st.empty()
spectrum_ph= st.empty()
feedback_ph= st.empty()
info_ph    = st.empty()

# -------------------- Charts & metrics --------------------
def draw_chart_and_metrics() -> Tuple[float, float, float]:
    if len(ss.buf) == 0: return 0.0, 0.0, 0.0
    vals = np.asarray(ss.buf, dtype=float)
    ts   = np.asarray(ss.ts,  dtype=float)
    fs   = _estimate_fs(ts)

    t_plot, r_plot = _compute_series_ratio(ts, vals, fs)
    if t_plot.size == 0: return 0.0, 0.0, 0.0
    t_plot, r_plot = _downsample(t_plot, r_plot, MAX_PLOT_POINTS)

    bl = _load_baseline() or {"mu": None, "sigma": None}

    # ---- Current Focus (β/α) ----
    idx_tail = np.where(ts >= ts.max() - 5.0)[0]
    ratio_now = beta_alpha_ratio(vals[idx_tail] if idx_tail.size > 0 else vals, fs)
    ratio_now = ema_update(ratio_now)
    rel_now   = relative_focus(ratio_now, bl.get("mu"), bl.get("sigma"))

    # ---- Current Memory index (θ+γ)/α (NEW) ----
    mi_now = theta_gamma_index(vals[idx_tail] if idx_tail.size > 0 else vals, fs)
    mi_now = ema_update(mi_now)  # reuse EMA smoothing
    rel_mem = relative_memory_score(mi_now, bl.get("mu"), bl.get("sigma"))

    # SNR/Confidence (unchanged)
    snr_est = float(np.clip(1.0 / (np.var(r_plot) + 1e-6), 0, 10))
    duration = float(ss.ts[-1] - ss.ts[0]) if len(ss.ts) > 1 else 0.0
    conf = compute_confidence(duration, snr_est, contact_ok=True)

    # Focus trend line
    df = pd.DataFrame({"Time (s)": t_plot, "β/α": r_plot})
    mean_val = float(np.mean(r_plot))
    line = alt.Chart(df).mark_line().encode(
        x=alt.X("Time (s):Q", title="Time (seconds)"),
        y=alt.Y("β/α:Q", title="EEG β/α (EMA)")
    ).properties(height=300)
    avg_rule = alt.Chart(pd.DataFrame({"y":[mean_val]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    chart = (line + avg_rule).interactive()
    chart_ph.altair_chart(chart, use_container_width=True)

    # Trend delta: last 5s vs last 30s mean
    r_short = float(np.mean(df["β/α"].tail(10))) if df.shape[0] >= 10 else float(np.mean(df["β/α"]))
    r_long  = float(np.mean(df["β/α"].tail(60))) if df.shape[0] >= 60 else r_short
    delta_pct = 0.0 if r_long == 0 else (r_short - r_long) / r_long * 100.0

    colA, colB, colC, colD = st.columns(4)
    label, emoji = _state_label(rel_now)
    colA.metric("Relative Focus (0–1)", f"{rel_now:.2f}", delta=f"{emoji} {label}")
    colB.metric("Current β/α (EMA)", f"{ratio_now:.2f}", delta=f"{delta_pct:+.1f}%")
    colC.metric("Confidence", f"{conf:.2f}")
    colD.metric("Mean β/α (window)", f"{mean_val:.2f}")

    # Memory metrics (NEW)
    colE, colF = st.columns(2)
    colE.metric("Memory Index (θ+γ)/α", f"{mi_now:.2f}")
    colF.metric("Relative Memory (0–1)", f"{rel_mem:.2f}")

    # Two meters
    with feedback_ph.container():
        render_bar_meter(rel_now, feedback_ph, label="Focus Meter")
        st.markdown("---")
        render_bar_meter(rel_mem, feedback_ph, label="Memory Meter (encoding/retrieval)")

    return rel_now, ratio_now, conf

def draw_power_spectrum(max_hz=40):
    if len(ss.buf) == 0: return
    vals = np.asarray(ss.buf, dtype=float); ts = np.asarray(ss.ts, dtype=float)
    fs = _estimate_fs(ts)
    if fs <= 0 or vals.size < int(fs): return
    idx = np.where(ts >= ts.max() - 2.0)[0]
    if idx.size < int(0.5 * fs): return
    x = vals[idx]; win = np.hanning(x.size); xw = x*win
    spec = np.fft.rfft(xw); psd = (np.abs(spec)**2)/np.sum(win**2)
    freqs = np.fft.rfftfreq(xw.size, d=1.0/fs)
    m = freqs <= max_hz
    df = pd.DataFrame({"Freq (Hz)": freqs[m], "Power": psd[m]})
    chart = (alt.Chart(df).mark_bar()
             .encode(x=alt.X("Freq (Hz):Q", bin=alt.Bin(maxbins=80), title="Frequency (Hz)"),
                     y=alt.Y("sum(Power):Q", title="Power"),
                     tooltip=["Freq (Hz):Q","Power:Q"]) 
             .properties(height=220, title="Power Spectrum (0–40 Hz, recent 2 s)"))
    spectrum_ph.altair_chart(chart, use_container_width=True)

# -------------------- History / Sessions --------------------
def save_session_history():
    if len(ss.buf) == 0 or ss.started_at is None: return
    vals = np.asarray(ss.buf, dtype=float); ts = np.asarray(ss.ts, dtype=float)
    fs = _estimate_fs(ts)
    if ts.size < 2: return
    rows = []
    t_min, t_max = ts.min(), ts.max()
    t = t_min + SAVE_RECORD_EVERY
    while t <= t_max:
        idx = np.where((ts >= t - SAVE_RECORD_EVERY) & (ts <= t))[0]
        if idx.size > 50:
            r  = beta_alpha_ratio(vals[idx], fs)
            mi = theta_gamma_index(vals[idx], fs)  # NEW
            epoch = ss.started_at + t
            lt = datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")
            hour = int(datetime.fromtimestamp(epoch).strftime("%H"))
            rows.append({
                "ts":epoch,"local_time":lt,"hour":hour,
                "ratio":r,"memory_index":mi,
                "task":ss.selected_task,"difficulty":ss.selected_diff
            })
        t += SAVE_RECORD_EVERY
    _append_csv(HISTORY_PATH, rows, HISTORY_COLS)

def record_session_summary(rel_mean: float, conf: float):
    row = {"ts":time.time(),"local_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "duration_sec":float(ss.last_duration_secs or 0.0),
           "rel_focus_mean":rel_mean,"confidence":conf,
           "task":ss.selected_task,"difficulty":ss.selected_diff}
    _append_csv(SESSIONS_PATH, [row], SESSIONS_COLS)

def export_history_csv():
    df = _load_csv(HISTORY_PATH, HISTORY_COLS)
    if df.empty:
        st.warning("No history yet."); return
    buf = io.BytesIO(); df.to_csv(buf, index=False)
    st.download_button("⬇️ Export History CSV", data=buf.getvalue(),
                       file_name="eeg_history.csv", mime="text/csv")

# -------------------- Personalized advice helpers --------------------
def _recent_task_window_from_history(task: str, minutes: float) -> Tuple[Optional[float], int]:
    """Mean β/α from history rows within the last `minutes` for the given task."""
    hist = _load_csv(HISTORY_PATH, HISTORY_COLS)
    if hist.empty or task == "None": return None, 0
    cutoff = time.time() - minutes * 60.0
    df = hist[(hist["ts"] >= cutoff) & (hist["task"] == task)]
    if df.empty: return None, 0
    return float(df["ratio"].mean()), int(df.shape[0])

def _recent_window_from_buffer(max_minutes: float = 10.0) -> Optional[float]:
    """Mean β/α from the live buffer (up to max_minutes)."""
    if len(ss.buf) == 0: return None
    vals = np.asarray(ss.buf, dtype=float)
    ts   = np.asarray(ss.ts,  dtype=float)
    fs   = _estimate_fs(ts)
    window = min(max_minutes * 60.0, float(ts.max() - ts.min()))
    if window < 5.0: return None
    idx = np.where(ts >= ts.max() - window)[0]
    if idx.size < 20: return None
    return float(beta_alpha_ratio(vals[idx], fs))

def _overall_daily_mean() -> Optional[float]:
    hist = _load_csv(HISTORY_PATH, HISTORY_COLS)
    if hist.empty: return None
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
    if hist.empty: return []
    hour_now = int(datetime.now().strftime("%H"))
    df = hist[hist["hour"] == hour_now]
    if df.empty or "task" not in df.columns: return []
    g = df.groupby("task")["ratio"].mean().sort_values(ascending=False).head(topk)
    return [(t, float(s)) for t, s in g.items()]

def _advice_from_scores(rel_focus_est: Optional[float], improve_pct: Optional[float]) -> str:
    """Craft simple actionable advice from relative focus and improvement vs daily mean."""
    imp = improve_pct if improve_pct is not None else 0.0
    rf  = rel_focus_est if rel_focus_est is not None else 0.5
    if rf >= 0.70 and imp >= 8.0:
        return "Great slot for this task — keep it here, or try a more challenging version."
    if rf >= 0.55 and imp >= 0.0:
        return "Good for this task. You can continue, or do focused reading/review."
    if rf >= 0.45 and imp < 0.0:
        return "Borderline for this task. Prefer lighter tasks (reading/review) in this hour."
    return "Not ideal for high-focus work now. Consider rest, a walk, or very light tasks."

# -------------------- Recommendation (with improvement %) + Advice --------------------
def ai_recommendation_block():
    st.markdown("### 📊 AI Study Time Recommendation (Task × Difficulty)")

    # Mode selector for required history minutes
    col_mode, col_info = st.columns([1.4, 2])
    mode = col_mode.selectbox(
        "History Requirement",
        ["Quick (10 min)", "Standard (30 min)", "Research (60 min)"],
        index=["Quick (10 min)", "Standard (30 min)", "Research (60 min)"].index(ss.min_history_mode)
    )
    ss.min_history_mode = mode
    mins_map = {"Quick (10 min)": 10.0, "Standard (30 min)": 30.0, "Research (60 min)": 60.0}
    ss.min_history_min = mins_map.get(mode, 30.0)
    col_info.caption("Choose a shorter requirement for demos, longer for more robust estimates.")

    hist = _load_csv(HISTORY_PATH, HISTORY_COLS)

    task_filter = ss.selected_task if ss.selected_task != "None" else None
    diff_filter = ss.selected_diff

    if task_filter: hist = hist[hist["task"] == task_filter]
    if diff_filter: hist = hist[hist["difficulty"] == diff_filter]

    hist_min_total = (len(hist) * SAVE_RECORD_EVERY) / 60.0
    target_min = float(ss.min_history_min)

    if hist_min_total < target_min:
        pct = min(hist_min_total / target_min, 1.0)
        st.progress(pct, text=f"Collected {hist_min_total:.1f} min / target {target_min:.0f} min")
        st.info("Not enough history for personalized recommendation yet.")
        st.caption("Tip: collect more data with the task & difficulty you plan to study next.")
        return
    if hist.empty:
        st.info("No history found."); return

    # ----- Focus-based hourly stats (β/α) -----
    overall_mean = float(hist["ratio"].mean())
    stats = (hist.groupby("hour", dropna=True)
                 .agg(ratio=("ratio","mean"), n=("ratio","count"))
                 .reset_index())
    stats["improve_pct"] = (stats["ratio"] - overall_mean) / (overall_mean + 1e-9) * 100.0
    stats = stats.sort_values("ratio", ascending=False).reset_index(drop=True)

    tag = f"Task: {task_filter or 'Any'}; Difficulty: {diff_filter}"
    st.success(f"🎯 Recommended Hours (Focus, {tag}, requirement: {int(target_min)} min)")
    for rank, row in enumerate(stats.head(TOPN).itertuples(index=False), 1):
        h = int(row.hour); sc = float(row.ratio); n = int(row.n); imp = float(row.improve_pct)
        start = f"{h:02d}:00"; end = f"{(h+1)%24:02d}:00"
        st.write(f"**#{rank}**  {start}–{end} · β/α ≈ **{sc:.2f}** · {imp:+.1f}% vs daily mean · n={n}")

    chart = (alt.Chart(stats).mark_bar()
             .encode(x=alt.X("hour:O", title="Hour of Day"),
                     y=alt.Y("ratio:Q", title="Mean β/α (higher = more focused)"),
                     tooltip=["hour:O","ratio:Q","n:Q","improve_pct:Q"]) 
             .properties(height=240, title=f"Hourly β/α ( {tag} )"))
    st.altair_chart(chart, use_container_width=True)

    # ----- Memory-based hourly stats (NEW) -----
    if "memory_index" in hist.columns and not hist["memory_index"].dropna().empty:
        stats_mem = (hist.groupby("hour", dropna=True)
                     .agg(memory_index=("memory_index","mean"), n=("memory_index","count"))
                     .reset_index())
        overall_mem_mean = float(hist["memory_index"].mean())
        stats_mem["improve_pct"] = (stats_mem["memory_index"] - overall_mem_mean) / (overall_mem_mean + 1e-9) * 100.0
        stats_mem = stats_mem.sort_values("memory_index", ascending=False).reset_index(drop=True)

        st.success("🧠 Recommended Hours for **Memory Tasks** (Memorization/Review)")
        for rank, row in enumerate(stats_mem.head(TOPN).itertuples(index=False), 1):
            h = int(row.hour); sc = float(row.memory_index); n = int(row.n); imp = float(row.improve_pct)
            start = f"{h:02d}:00"; end = f"{(h+1)%24:02d}:00"
            st.write(f"**#{rank}**  {start}–{end} · (θ+γ)/α ≈ **{sc:.2f}** · {imp:+.1f}% vs daily mean · n={n}")

        chart_mem = (alt.Chart(stats_mem).mark_bar()
                     .encode(x=alt.X("hour:O", title="Hour of Day"),
                             y=alt.Y("memory_index:Q", title="Mean (θ+γ)/α (higher = better for memory)"),
                             tooltip=["hour:O","memory_index:Q","n:Q","improve_pct:Q"]) 
                     .properties(height=220, title="Hourly Memory Index"))
        st.altair_chart(chart_mem, use_container_width=True)

    # ---------- Personalized Task Advice (this window) ----------
    st.markdown("### 📌 Personalized Task Advice (this window)")

    use_mem = ss.selected_task in ("Memorization", "Review") and "memory_index" in hist.columns

    # Compute recent mean for this task within target_min window
    recent_hist_mean, recent_n = None, 0
    if not hist.empty:
        cutoff = time.time() - target_min * 60.0
        dfw = hist[hist["ts"] >= cutoff]
        col = "memory_index" if use_mem else "ratio"
        if not dfw[col].dropna().empty:
            recent_hist_mean = float(dfw[col].mean())
            recent_n = int(dfw[col].shape[0])

    # Compute daily mean for comparison
    daily_mean = None
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        col = "memory_index" if use_mem else "ratio"
        df_today = hist[hist["local_time"].str.startswith(today)]
        if not df_today.empty and not df_today[col].dropna().empty:
            daily_mean = float(df_today[col].mean())
        else:
            daily_mean = float(hist[col].mean()) if not hist[col].dropna().empty else None
    except Exception:
        pass

    # Relative estimate
    bl = _load_baseline() or {"mu": None, "sigma": None}
    if recent_hist_mean is not None:
        rel_est = (relative_memory_score(recent_hist_mean, bl.get("mu"), bl.get("sigma"))
                   if use_mem else
                   relative_focus(recent_hist_mean, bl.get("mu"), bl.get("sigma")))
    else:
        rel_est = None

    improve = None if (recent_hist_mean is None or daily_mean is None or daily_mean == 0) \
        else (recent_hist_mean - daily_mean) / daily_mean * 100.0

    # Compose advice text
    if recent_hist_mean is None:
        st.info("Not enough recent data for this task to judge this window. Keep collecting, then check back.")
    else:
        bullets = []
        bullets.append(f"**Current task**: {ss.selected_task} / **{ss.selected_diff}")
        label = "(θ+γ)/α" if use_mem else "β/α"
        bullets.append(f"**This-window {label} mean**: {recent_hist_mean:.2f}")
        if rel_est is not None: bullets.append(f"**Relative estimate**: {rel_est:.2f}")
        if improve is not None: bullets.append(f"**vs daily mean**: {improve:+.1f}%")
        if recent_n: bullets.append(f"**samples**: n={recent_n}")
        st.success(" · ".join(bullets))

        advice = _advice_from_scores(rel_est, improve)
        st.write(f"**Recommendation:** {advice}")

        # Better tasks for the current hour (based on focus history).
        # 可扩展为基于 memory_index 的榜单，这里先保持一致性
        top_tasks = _best_tasks_for_current_hour(topk=3)
        if top_tasks:
            nice = ", ".join([f"{t} (β/α {s:.2f})" for t, s in top_tasks])
            st.caption(f"Better-performing tasks for **this hour** (from your history): {nice}")

# -------------------- Top controls --------------------
col_task, col_diff, col_start, col_stop, col_export = st.columns([2,1.3,1,1,1])
ss.selected_task = col_task.selectbox("Current Task",
    ["None","Reading","Coding","Writing","Memorization","Review"], index=0)
ss.selected_diff = col_diff.selectbox("Difficulty", ["Easy","Moderate","Challenging"], index=1)
start_clicked = col_start.button("▶ Start", type="primary")
stop_clicked  = col_stop.button("▣ Stop")
export_clicked = col_export.button("Export CSV")

with st.expander("🧪 Baseline Calibration", expanded=False):
    st.caption("Run a relaxed 2–3 minute session to estimate your personal baseline.")
    st.write("Current baseline:", _load_baseline())
    if st.button("📏 Run 2-min Calibration"):
        if not ss.collecting:
            st.warning("Please start acquisition first.")
        else:
            st.info("Collecting 120 s for baseline...")
            t_end = time.time() + 120
            vals_tmp, ts_tmp = [], []
            while time.time() < t_end:
                collect_step()
                if len(ss.buf) > 0:
                    vals_tmp.extend(list(ss.buf)); ts_tmp.extend(list(ss.ts))
                time.sleep(0.1)
            if len(ts_tmp) > 10:
                vals_tmp = np.asarray(vals_tmp, dtype=float)
                ts_tmp   = np.asarray(ts_tmp,  dtype=float)
                fs = _estimate_fs(ts_tmp)
                t_plot, r_plot = _compute_series_ratio(ts_tmp, vals_tmp, fs)
                if len(r_plot) > 5:
                    _save_baseline(float(np.mean(r_plot)), float(np.std(r_plot)))
                    st.success("Baseline updated ✅")
                else:
                    st.warning("Not enough data. Please try again.")
            else:
                st.warning("Not enough data. Please try again.")

# -------------------- Start / Stop --------------------
if start_clicked and not ss.collecting:
    try:
        inlet = connect_lsl("EEG", timeout=2.0, retries=3)
        if inlet is None:
            status.warning("No EEG LSL stream detected. Falling back to offline demo (random data).")
        ss.inlet = inlet
        ss.buf.clear(); ss.ts.clear()
        ss.collecting  = True
        ss.started_at  = time.time()
        ss.stopped_at  = None
        ss.last_draw   = 0.0
        ss.last_duration_secs = None
        ss.ema_state = None
        status.success("Collecting EEG…")
    except Exception as e:
        ss.collecting = False; ss.inlet = None
        status.error(f"Could not open EEG stream: {e}")

if stop_clicked and ss.collecting:
    ss.collecting = False; ss.inlet = None
    if len(ss.ts) > 0: ss.last_duration_secs = float(ss.ts[-1] - ss.ts[0])
    elif ss.started_at is not None: ss.last_duration_secs = float(time.time() - ss.started_at)
    else: ss.last_duration_secs = 0.0
    mm = int(ss.last_duration_secs//60); ss_ = int(ss.last_duration_secs%60)
    status.info(f"Stopped. Total duration: {mm:02d}:{ss_:02d}")
    try:
        save_session_history()
        # summary gets updated on final render below with real rel_now/conf
        record_session_summary(0.0, 0.0)
    except Exception as e:
        st.warning(f"Error saving history: {e}")

if export_clicked:
    export_history_csv()

# -------------------- Main loop --------------------
if ss.collecting:
    if ss.inlet is None:
        # Offline demo signal so the UI works without a device
        t0 = ss.started_at; t = time.time() - t0
        sim = 0.5*np.sin(0.2*t) + 0.2*np.random.randn(50)
        start_t = (ss.ts[-1] + STEP_SECS) if ss.ts else 0.0
        ss.buf.extend(sim); ss.ts.extend(np.linspace(start_t, start_t+STEP_SECS, 50))
    else:
        collect_step()

    elapsed = time.time() - (ss.started_at or time.time())
    timer_ph.caption(f"⏱ Elapsed: {int(elapsed//60):02d}:{int(elapsed%60):02d}")

    if time.time() - ss.last_draw >= DRAW_EVERY_SEC:
        rel_now, ratio_now, conf = draw_chart_and_metrics()
        draw_power_spectrum()
        ss.last_draw = time.time()

    time.sleep(STEP_SECS)
    st.rerun()

# -------------------- Non-collecting: final frame + recommendations --------------------
if not ss.collecting and len(ss.buf) > 0:
    rel_now, ratio_now, conf = draw_chart_and_metrics()
    draw_power_spectrum()
    if ss.last_duration_secs is not None:
        mm = int(ss.last_duration_secs // 60); ss_ = int(ss.last_duration_secs % 60)
        info_ph.info(f"Stopped. Total duration recorded: **{mm:02d}:{ss_:02d}**")
        try: record_session_summary(float(rel_now), float(conf))
        except Exception: pass
    else:
        info_ph.info("Stopped.")

# Always show recommendations/advice once per page render
ai_recommendation_block()
