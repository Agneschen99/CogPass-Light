# -*- coding: utf-8 -*-
# One-button EEG demo (clean, single-definition, embeddable)

"""
Embeddable one-button Attention+Memory demo for Streamlit.

Public API:
    render_one_button_app(call_set_page_config: bool = False)

Notes:
 - This module must NOT call st.set_page_config() at import time so it can be
   embedded safely. If run as __main__ or when call_set_page_config=True,
   render_one_button_app will set a benign page config.
 - If pylsl is unavailable, the app falls back to simulated EEG.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Iterable, List, Tuple

import numpy as np
import streamlit as st

try:
    from pylsl import StreamInlet, resolve_stream
    LSL_AVAILABLE = True
except Exception:
    StreamInlet = None  # type: ignore
    resolve_stream = None
    LSL_AVAILABLE = False

try:
    from app.eeg import algorithms_backup as algo
except Exception:  # pragma: no cover - optional dependency
    algo = None  # type: ignore

try:
    from app.eeg import eeg_memory_readiness as mem_model
except Exception:  # pragma: no cover - optional dependency
    mem_model = None  # type: ignore

def _welch_bandpower(sig: Iterable[float], fs: float, fmin: float, fmax: float) -> float:
    """Estimate band power using an FFT-based PSD (no scipy dependency).

    sig: 1-D iterable of samples
    fs: sampling frequency
    returns: power in the band [fmin, fmax]
    """
    x = np.asarray(sig, dtype=float)
    if x.size == 0:
        return 0.0
    n = x.size
    # Detrend
    x = x - x.mean()
    # One-sided FFT
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd = (np.abs(np.fft.rfft(x)) ** 2) / n
    # Integrate PSD over the band
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx.size == 0:
        return 0.0
    return float(psd[idx].sum())


def compute_features(chunk: Iterable[float], fs: float):
    """Return base band powers and derived attention/memory/fatigue metrics."""
    feature_vec = None
    if algo is not None and hasattr(algo, "build_online_feats_from_raw"):
        try:
            feature_vec = np.asarray(algo.build_online_feats_from_raw(chunk, fs), dtype=float).ravel()
        except Exception:
            feature_vec = None

    if feature_vec is None:
        delta_band = (1.0, 4.0)
        theta_band = (4.0, 8.0)
        alpha_band = (8.0, 13.0)
        beta_band = (13.0, 30.0)

        d = _welch_bandpower(chunk, fs, *delta_band)
        t = _welch_bandpower(chunk, fs, *theta_band)
        a = _welch_bandpower(chunk, fs, *alpha_band)
        b = _welch_bandpower(chunk, fs, *beta_band)
        alpha_theta = a / max(t, 1e-9)
        beta_alpha_theta = b / max(a + t, 1e-9)
        theta_beta = t / max(b, 1e-9)
        feature_vec = np.array([d, t, a, b, alpha_theta, beta_alpha_theta, theta_beta], dtype=float)
    else:
        d, t, a, b, alpha_theta, beta_alpha_theta, theta_beta = feature_vec.tolist()

    attention = beta_alpha_theta
    memory_ratio = alpha_theta
    fatigue = theta_beta

    return {
        "delta": d,
        "theta": t,
        "alpha": a,
        "beta": b,
        "attention": attention,
        "memory_ratio": memory_ratio,
        "memory": memory_ratio,  # backward-compat: default to ratio before model inference
        "fatigue": fatigue,
        "feature_vector": feature_vec,
    }


def find_best_windows(timeline: List[datetime], series: List[float], merge_win: int = 2, topk: int = 3) -> List[Tuple[datetime, datetime]]:
    """Find top-k windows in series and return corresponding timeline intervals.

    This is a tiny heuristic: pick largest points, merge close indices.
    """
    if not series:
        return []
    arr = np.asarray(series)
    order = np.argsort(arr)[::-1]
    used = np.zeros_like(arr, dtype=bool)
    spans: List[Tuple[int, int]] = []
    for i in order:
        if used[i]:
            continue
        s = e = int(i)
        for j in range(max(0, i - merge_win), min(len(arr), i + merge_win + 1)):
            if not used[j]:
                used[j] = True
                s = min(s, j)
                e = max(e, j)
        spans.append((s, e))
        if len(spans) >= topk:
            break
    spans.sort(key=lambda se: se[0])
    merged: List[Tuple[datetime, datetime]] = []
    for s, e in spans:
        if not merged or timeline[s] > merged[-1][1]:
            merged.append((timeline[s], timeline[e]))
        else:
            # extend
            merged[-1] = (merged[-1][0], max(merged[-1][1], timeline[e]))
    return merged


def fmt_interval(iv: Tuple[datetime, datetime]) -> str:
    s, e = iv
    return f"{s.strftime('%H:%M:%S')} – {e.strftime('%H:%M:%S')}"


def _simulate_chunk(n_samples: int, fs: float, t0: float) -> List[float]:
    # simple simulated EEG-like mixture
    base = np.random.randn(n_samples) * 0.12
    base += 0.5 * np.sin(2 * np.pi * 10 * np.arange(n_samples) / fs) * (0.2 if 15 < (t0 % 60) < 30 else 0.08)
    base += 0.35 * np.sin(2 * np.pi * 18 * np.arange(n_samples) / fs) * (0.25 if 35 < (t0 % 60) < 50 else 0.06)
    if 50 < (t0 % 90) < 70:
        base += 0.6 * np.sin(2 * np.pi * 6 * np.arange(n_samples) / fs)
    return base.tolist()


def render_one_button_app(call_set_page_config: bool = False) -> None:
    """Render the embeddable one-button Attention+Memory app into the current Streamlit page.

    If call_set_page_config is True, set a minimal page config (safe when run as main).
    """
    if call_set_page_config:
        st.set_page_config(page_title="One-Button EEG Demo", layout="centered")

    st.header("One-Button: Attention + Memory (EEG)")
    st.write("小型演示：采集一段 EEG 窗口并给出基于频带的注意力/记忆/疲劳建议。")

    # Controls
    c1, c2 = st.columns([3, 1])
    with c1:
        run_secs = st.number_input("Total duration (s)", value=30, min_value=5, max_value=600, step=5)
        win_len = st.number_input("Window length (s)", value=4.0, min_value=1.0, max_value=30.0, step=0.5)
        step_sec = st.number_input("Step (s)", value=2.0, min_value=0.2, max_value=30.0, step=0.1)
        fs = st.number_input("Sampling rate (Hz)", value=100.0, min_value=10.0, max_value=1000.0, step=1.0)
    with c2:
        attn_thr = st.number_input("Attention threshold", value=0.8)
        fatigue_theta_beta = st.number_input("Fatigue θ/β thresh", value=0.9)
        fatigue_low_attn_repeat = st.number_input("Low-attn repeats", value=3, min_value=1)

    # Small UI pieces
    cols = st.columns(2)
    start_btn = cols[0].button("▶ Start (Attention + Memory)", use_container_width=True)
    stop_btn = cols[1].button("⏹ Stop", use_container_width=True)

    status = st.empty()
    progress = st.progress(0)
    attn_chart = st.empty()
    mem_chart = st.empty()
    fatigue_box = st.empty()

    # Internals
    if start_btn:
        status.info("🔴 开始采集…")
        inlet = None
        if LSL_AVAILABLE:
            try:
                streams = resolve_stream('type', 'EEG', timeout=1.0)
                if streams:
                    inlet = StreamInlet(streams[0], max_buflen=60)
                    status.success("✅ 已连接 LSL EEG")
                else:
                    status.warning("⚠️ 未发现 LSL EEG 流，启用模拟数据")
            except Exception:
                status.warning("⚠️ LSL 连接失败，启用模拟数据")
        else:
            status.warning("⚠️ pylsl 不可用，使用模拟数据")

        start_t = datetime.now()
        attn_series: List[float] = []
        mem_series: List[float] = []
        fatigue_series: List[float] = []
        t_stamps: List[datetime] = []

        steps = int(max(1, (run_secs - win_len) // step_sec + 1))
        for k in range(steps):
            if stop_btn:
                status.warning("⏸ 已手动停止")
                break

            n_needed = int(max(1, win_len * fs))
            if inlet is not None:
                chunk = []
                deadline = time.time() + win_len + 0.5
                while len(chunk) < n_needed and time.time() < deadline:
                    sample, _ = inlet.pull_sample(timeout=0.05)
                    if sample is not None:
                        # average channels
                        try:
                            chunk.append(float(np.mean(sample)))
                        except Exception:
                            chunk.append(float(sample[0]))
                if len(chunk) < n_needed:
                    chunk += [0.0] * (n_needed - len(chunk))
            else:
                chunk = _simulate_chunk(n_needed, fs, float(k * step_sec))

            feats = compute_features(chunk, fs)
            attn_series.append(feats["attention"])

            mem_value = feats.get("memory", 0.0)
            fv = feats.get("feature_vector")
            if mem_model is not None and fv is not None:
                try:
                    scores = mem_model.predict_scores(np.asarray(fv, dtype=float))
                    mem_value = float(scores.get("memory", mem_value))
                except Exception:
                    mem_value = feats.get("memory_ratio", mem_value)
            mem_series.append(mem_value)
            feats["memory"] = mem_value
            fatigue_series.append(feats["fatigue"])
            t_stamps.append(start_t + timedelta(seconds=k * step_sec))

            # quick fatigue heuristic
            low_attn = feats["attention"] < attn_thr or feats["fatigue"] > fatigue_theta_beta
            if low_attn and len(attn_series) >= fatigue_low_attn_repeat:
                recent = attn_series[-fatigue_low_attn_repeat:]
                if all(x < attn_thr for x in recent):
                    fatigue_box.warning("😮‍💨 检测到疲劳/注意力下降：建议休息 3–5 分钟或切换轻任务。")

            attn_chart.line_chart({"attention": attn_series}, height=160)
            mem_chart.line_chart({"memory": mem_series}, height=160)

            progress.progress(int((k + 1) / steps * 100))
            time.sleep(0.01)

        # Summary
        if t_stamps:
            attn_best = find_best_windows(t_stamps, attn_series, merge_win=2, topk=3)
            mem_best = find_best_windows(t_stamps, mem_series, merge_win=2, topk=3)

            st.subheader("📊 EEG-Based Recommendations")
            if attn_best:
                st.markdown("**🔥 注意力高峰（适合攻坚）：** " + "； ".join(fmt_interval(x) for x in attn_best))
            else:
                st.markdown("**注意力高峰**：未检测到显著峰值，建议先做轻任务或稍后再试。")

            if mem_best:
                st.markdown("**🧷 记忆力就绪（适合背诵/复盘）：** " + "； ".join(fmt_interval(x) for x in mem_best))
            else:
                st.markdown("**记忆力就绪**：未检测到显著峰值，可在注意力高峰结束后安排 10–15 分钟复盘。")


if __name__ == "__main__":
    render_one_button_app(call_set_page_config=True)
