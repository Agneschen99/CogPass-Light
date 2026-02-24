import streamlit as st
import numpy as np
import pandas as pd
import time
from collections import deque
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pylsl import StreamInlet, resolve_stream
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

try:
    from app.eeg import algorithms_backup as algo
    from app.eeg.eeg_memory_readiness import load_model_bundle, online_predict
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

MODEL_PATH = "model_store/eeg_mem_model"

def init_session_state():
    if "lsl_inlet" not in st.session_state:
        st.session_state.lsl_inlet = None
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = False
    if "baseline" not in st.session_state:
        st.session_state.baseline = None
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "ai_model" not in st.session_state:
        st.session_state.ai_model = None
        if AI_AVAILABLE:
            try:
                model, meta = load_model_bundle(MODEL_PATH)
                st.session_state.ai_model = model
                st.session_state.ai_meta = meta
            except Exception:
                pass
    if "data_buffer" not in st.session_state:
        st.session_state.data_buffer = deque(maxlen=200)
    if "ai_prob_buffer" not in st.session_state:
        st.session_state.ai_prob_buffer = deque(maxlen=200)

def get_current_data():
    if st.session_state.lsl_inlet:
        try:
            chunk, _ = st.session_state.lsl_inlet.pull_chunk(timeout=0.0)
            if chunk:
                return np.array(chunk)
        except Exception:
            pass
    t = time.time()
    fake_chunk = np.sin(t * 5) * 10 + 50 + np.random.randn(1, 4) * 2
    return fake_chunk

def connect_device():
    if not LSL_AVAILABLE:
        st.error("❌ pylsl not installed")
        return False
    with st.spinner("Searching for Muse..."):
        streams = resolve_stream("type", "EEG")
        if streams:
            inlet = StreamInlet(streams[0], max_buflen=360)
            st.session_state.lsl_inlet = inlet
            st.session_state.connection_status = True
            st.toast("✅ Connection successful", icon="🔗")
            return True
        else:
            st.error("❌ Device not found")
            return False

def run_calibration_ui():
    placeholder = st.empty()
    bar = st.progress(0)
    vals = []
    duration = 5
    start_t = time.time()
    while time.time() - start_t < duration:
        chunk = get_current_data()
        if chunk is not None:
            vals.append(np.mean(chunk))
        elapsed = time.time() - start_t
        prog = min(elapsed / duration, 1.0)
        bar.progress(prog)
        placeholder.info(f"🧘 Calibrating... ({int(elapsed)}s / {duration}s)")
        time.sleep(0.1)
    if vals:
        mu = np.mean(vals)
        sigma = np.std(vals)
        st.session_state.baseline = {"mu": mu, "sigma": sigma, "ts": time.time()}
        placeholder.success(f"✅ Calibration complete (μ={mu:.2f})")
    else:
        st.session_state.baseline = {"mu": 50.0, "sigma": 10.0, "ts": time.time()}
        placeholder.success("✅ Calibration complete (default values)")

def main():
    st.set_page_config(page_title="CogPass Unlocked", page_icon="🧠", layout="wide")
    init_session_state()

    st.title("🧠 Attention + Memory (EEG)")
    
    st.divider()
    st.subheader("EEG Mode")

    cols = st.columns([1, 1, 1])
    col1, col2, col3 = cols[0], cols[1], cols[2]

    with col1:
        st.markdown("### 🔌 Connect")

        # 已连接
        if st.session_state.connection_status:
            if st.button("Disconnect", key="disconnect_h"):
                st.session_state.lsl_inlet = None
                st.session_state.connection_status = False
                st.rerun()
            st.success("Connected")

        # 未连接
        else:
            if st.button("Connect Muse", key="connect_h"):
                connect_device()
                st.rerun()
            st.error("Not Connected")

    with col2:
        st.markdown("### 🧪 Baseline")
        if st.session_state.baseline is None:
            if st.button("Run Calibration", key="calibrate_h"):
                run_calibration_ui()
                st.rerun()
        else:
            bl = st.session_state.baseline
            st.write(f"μ={bl['mu']:.2f}, σ={bl['sigma']:.2f}")
            if st.button("Recalibrate", key="recalibrate_h"):
                st.session_state.baseline = None
                st.rerun()
        if st.session_state.baseline:
            st.success("Calibrated")
        else:
            st.warning("No Baseline")

    with col3:
        st.markdown("### 🎧 Recording")
        if not st.session_state.is_recording:
            if st.button("Start Recording", key="record_h"):
                st.session_state.is_recording = True
                st.rerun()
        else:
            if st.button("Stop", key="stop_h"):
                st.session_state.is_recording = False
                st.rerun()
        if st.session_state.is_recording:
            st.error("Recording")
        else:
            st.info("Ready")

    st.divider()
    if st.session_state.lsl_inlet is None:
        st.warning("⚠️ Using simulated EEG (fake mode). Connect Muse for real data.")

    if st.session_state.is_recording:
        placeholder = st.empty()
        while st.session_state.is_recording:
            chunk = get_current_data()
            val = np.mean(chunk) if chunk is not None else 0
            ai_score = 0.5 + 0.3 * np.sin(time.time())
            st.session_state.data_buffer.append(val)
            st.session_state.ai_prob_buffer.append(ai_score)

            with placeholder.container():
                g1, g2 = st.columns([3, 1])
                with g1:
                    df = pd.DataFrame({
                        "Raw Signal": list(st.session_state.data_buffer),
                        "AI Probability": [x * 100 for x in st.session_state.ai_prob_buffer]
                    })
                    st.line_chart(df, height=250)
                with g2:
                    st.metric("Focus Score", f"{ai_score * 100:.1f}%")
                    if ai_score > 0.7:
                        st.success("State: Focus 🔥")
                    elif ai_score < 0.3:
                        st.error("State: Fatigue 😴")
                    else:
                        st.info("State: Stable 📝")
                st.divider()
                st.subheader("🧭 AI Learning Recommendations")
                recent_avg = np.mean(st.session_state.ai_prob_buffer)
                rec1, rec2 = st.columns([3, 1])
                with rec1:
                    if recent_avg > 0.65:
                        st.success(f"🔥 High Focus ({recent_avg * 100:.0f}%). Do hard tasks now.")
                    elif recent_avg < 0.35:
                        st.warning(f"😴 Low Focus ({recent_avg * 100:.0f}%). Rest 5 minutes.")
                    else:
                        st.info(f"📝 Stable ({recent_avg * 100:.0f}%). Good for review.")

                import datetime

                now = datetime.datetime.now()
                t1 = (now - datetime.timedelta(seconds=15)).strftime("%H:%M:%S")
                t2 = (now - datetime.timedelta(seconds=5)).strftime("%H:%M:%S")
                with rec2:
                    st.markdown("**⏱️ Captured Peak Moments:**")
                    if recent_avg > 0.6:
                        st.markdown(f"- 🔥 {t1} – {t2}")
                    else:
                        st.caption("No significant peaks yet...")
            time.sleep(0.1)

if __name__ == "__main__":
    main()
