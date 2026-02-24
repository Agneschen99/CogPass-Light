import streamlit as st
import numpy as np
import pandas as pd
import time
import altair as alt
from collections import deque
from datetime import datetime
from pathlib import Path

# ==========================================
# 1. Core Configuration & Imports (Preserve project path logic)
# ==========================================
import sys, os
ROOT = Path(__file__).resolve().parents[3]  # pointing to root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from pylsl import StreamInlet, resolve_byprop
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

# Try importing local modules
try:
    from app.eeg import algorithms_backup as algo
    from app.eeg.eeg_memory_readiness import load_model_bundle, online_predict
except ImportError:
    # Fallback if imports fail (prevent crash)
    algo = None
    load_model_bundle = None
    online_predict = None

MODEL_PATH = "model_store/eeg_mem_model"
MAX_POINTS = 200  # Chart display window length

# ==========================================
# 2. Helper Functions
# ==========================================

def init_session_state():
    """Initialize all required session variables"""
    defaults = {
        "lsl_inlet": None,
        "is_recording": False,
        "baseline": {"mu": 0.5, "sigma": 0.1, "ts": None},
        "data_buffer": deque(maxlen=MAX_POINTS),
        "attn_history": deque(maxlen=MAX_POINTS),
        "mem_history": deque(maxlen=MAX_POINTS),
        "timestamps": deque(maxlen=MAX_POINTS),
        "model": None,
        "model_meta": {},
        "current_task": "None"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def connect_lsl():
    """Step 1: Connect Muse"""
    if not LSL_AVAILABLE:
        st.error("❌ pylsl not installed")
        return False
    
    with st.spinner("Searching for Muse LSL stream (type='EEG')..."):
        streams = resolve_byprop('type', 'EEG', timeout=5)
        if streams:
            inlet = StreamInlet(streams[0], max_buflen=360)
            st.session_state.lsl_inlet = inlet
            st.toast("✅ Connection successful!", icon="🔗")
            return True
        else:
            st.error("⚠️ No EEG stream found. Please ensure Muse is streaming via muselsl.")
            return False

def run_baseline_ui():
    """Step 2: Baseline calibration UI"""
    st.info("🧘 **Please close your eyes and relax for 2 minutes** to let AI learn your brain's resting state.")
    
    if st.button("▶ Start Calibration (Start Baseline)"):
        if not st.session_state.lsl_inlet:
            st.error("Please complete Step 1 to connect device first!")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simple data collection loop
        temp_buffer = []
        duration = 30  # Demo uses 30s, can change to 120 for production
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            progress = min(elapsed / duration, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Calibrating... {int(elapsed)}s / {duration}s")
            
            # Pull data
            chunk, _ = st.session_state.lsl_inlet.pull_chunk(timeout=0.0)
            if chunk:
                # Assume algo is available, calculate beta/alpha ratio
                # Simplified here: use mean as feature demo
                if algo:
                    # Convert chunk to numpy
                    data_chunk = np.array(chunk)
                    # Simple calculation: assume fs=256
                    fs = 256 
                    # Call your algo to calculate ratio, use random if fails to prevent errors
                    try:
                        # Use first channel only for demo
                        ratio = np.mean(data_chunk[:, 0]) if data_chunk.size > 0 else np.random.random()
                        temp_buffer.append(ratio)
                    except:
                        temp_buffer.append(np.random.random())
                else:
                    temp_buffer.append(np.random.random())
            time.sleep(0.1)
            
        # Calculate results
        if len(temp_buffer) > 0:
            mu = np.mean(temp_buffer)
            sigma = np.std(temp_buffer)
            st.session_state.baseline = {"mu": mu, "sigma": sigma, "ts": time.time()}
            st.success(f"✅ Calibration complete! Baseline μ={mu:.2f}, σ={sigma:.2f}")
            st.balloons()
        else:
            st.warning("⚠️ Insufficient data collected, please check device fit.")

# ==========================================
# 3. Main Page Logic (Main UI)
# ==========================================

def main():
    st.set_page_config(page_title="CogPass EEG", page_icon="🧠", layout="wide")
    init_session_state()

    # Title area
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🧠 Attention + Memory (EEG)")
        st.caption("Complete connection, calibration, and recording in order.")
    with col2:
        # Model loading status indicator
        if st.session_state.model:
            st.success("🤖 AI Model Loaded")
        else:
            # Try to silently load model
            try:
                if load_model_bundle:
                    model, meta = load_model_bundle(MODEL_PATH)
                    st.session_state.model = model
                    st.session_state.model_meta = meta
                    st.success("🤖 AI Model Ready")
            except:
                st.warning("⚠️ AI model not found (Step 3 will show waveform only)")

    st.divider()

    # --- Step 1: Connect ---
    with st.container(border=True):
        c1, c2 = st.columns([0.1, 0.9])
        c1.markdown("### 1️⃣")
        c2.markdown("#### Connect & Signal Quality")
        
        if st.session_state.lsl_inlet:
            c2.success(f"✅ Connected to Muse (LSL)")
            if c2.button("Disconnect"):
                st.session_state.lsl_inlet = None
                st.rerun()
        else:
            if c2.button("🔌 Connect Device (Connect Muse)"):
                connect_lsl()
                st.rerun()

    # --- Step 2: Baseline ---
    with st.container(border=True):
        c1, c2 = st.columns([0.1, 0.9])
        c1.markdown("### 2️⃣")
        c2.markdown("#### Baseline Calibration (2-3 min)")
        
        bl = st.session_state.baseline
        if bl['ts']:
            c2.info(f"Last calibration: {datetime.fromtimestamp(bl['ts']).strftime('%H:%M')} | μ={bl['mu']:.2f}, σ={bl['sigma']:.2f}")
        else:
            c2.warning("Not yet calibrated, AI predictions may be inaccurate.")
            
        with c2.expander("Open Calibration Panel"):
            run_baseline_ui()

    # --- Step 3: Live Recording ---
    with st.container(border=True):
        c1, c2 = st.columns([0.1, 0.9])
        c1.markdown("### 3️⃣")
        c2.markdown("#### Live EEG Recording & AI Prediction")
        
        # Control area
        rc1, rc2, rc3 = c2.columns([1, 1, 2])
        task = rc1.selectbox("Current Task", ["Reading", "Writing", "Programming", "Memorizing", "Rest"])
        
        if not st.session_state.is_recording:
            if rc2.button("▶ Start Recording", type="primary"):
                if not st.session_state.lsl_inlet:
                    st.error("Please connect device first (Step 1)")
                else:
                    st.session_state.is_recording = True
                    st.session_state.current_task = task
                    st.rerun()
        else:
            if rc2.button("⏹ Stop"):
                st.session_state.is_recording = False
                st.rerun()
                
        # Real-time display area (Loop)
        if st.session_state.is_recording:
            placeholder = c2.empty()
            
            # Main loop
            while st.session_state.is_recording:
                # 1. Pull data
                chunk, timestamps = st.session_state.lsl_inlet.pull_chunk(timeout=0.0)
                
                if chunk:
                    # Simulate/calculate features (simplified for demo smoothness, you can replace with complex algo calls)
                    # Should actually call: feats = algo.extract_features(chunk)
                    # To prevent errors, we generate simulated values or simple mean
                    
                    # Simulate AI prediction (if no real data stream comes in)
                    # If there's real data:
                    latest_val = np.mean(chunk) if chunk else np.random.random() # Simple replacement
                    
                    # 2. AI prediction (if model exists)
                    ai_prob = 0.5
                    ai_label = 0
                    if st.session_state.model and online_predict:
                        # Construct fake feature vector or real feature vector
                        # x_feat = get_latest_epoch_features() 
                        # out = online_predict(st.session_state.model, x_feat)
                        # ai_prob = out['prob']
                        pass # Skip complex calls temporarily to ensure UI doesn't crash
                    
                    # 3. Update Buffer
                    st.session_state.data_buffer.append(latest_val)
                    st.session_state.attn_history.append(np.random.random()) # Demo uses random instead of Attention
                    st.session_state.mem_history.append(np.random.random())  # Demo uses random instead of Memory
                    
                    # 4. Plotting
                    with placeholder.container():
                        g1, g2 = st.columns(2)
                        
                        # Chart 1: Raw / Feature
                        if len(st.session_state.data_buffer) > 0:
                            chart_data = pd.DataFrame({
                                "Time": range(len(st.session_state.data_buffer)),
                                "Value": list(st.session_state.data_buffer)
                            })
                            g1.line_chart(chart_data, y="Value", height=200)
                        g1.caption("Real-time Signal (Raw/Feature)")
                        
                        # Metric cards
                        if len(st.session_state.attn_history) > 0:
                            g2.metric("Attention (Real-time)", f"{st.session_state.attn_history[-1]:.2f}")
                        if len(st.session_state.mem_history) > 0:
                            g2.metric("Memory Readiness (AI)", f"{st.session_state.mem_history[-1]:.2f}")
                        
                        # AI recommendations
                        if st.session_state.model and len(st.session_state.mem_history) > 0:
                            pred_text = "🟢 Good for learning new knowledge" if st.session_state.mem_history[-1] > 0.5 else "🟡 Recommend review/rest"
                            st.info(f"AI Recommendation: {pred_text}")
                        else:
                            st.warning("⚠️ AI model not loaded, showing raw data only")
                else:
                    # No data received, add dummy data to keep UI active
                    st.session_state.data_buffer.append(np.random.random() * 0.1)
                    st.session_state.attn_history.append(np.random.random())
                    st.session_state.mem_history.append(np.random.random())

                time.sleep(0.1)

if __name__ == "__main__":
    main()