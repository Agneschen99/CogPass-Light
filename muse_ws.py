
import asyncio
import json
import os
import random
import time
from typing import Dict, Any

import websockets
from websockets.server import WebSocketServerProtocol

PORT = int(os.getenv("EEG_WS_PORT", "8765"))
HOST = os.getenv("EEG_WS_HOST", "0.0.0.0")

# --- Optional LSL imports (only used in --mode lsl) ---
HAS_LSL = False
try:
    from pylsl import StreamInlet, resolve_stream
    HAS_LSL = True
except Exception:
    HAS_LSL = False

# -------------------------------
# Helpers
# -------------------------------
def build_packet(payload: Dict[str, Any]) -> str:
    """Serialize a message with a common envelope."""
    msg = {
        "source": "muse-ws",
        "ts": time.time(),
        **payload,
    }
    return json.dumps(msg)

# -------------------------------
# Mock data generator
# -------------------------------
async def mock_stream(ws: WebSocketServerProtocol):
    """
    Send mock EEG “band powers” at ~10 Hz:
    delta, theta, alpha, beta, gamma + a naive focus_index (0~1)
    """
    while True:
        # Fake “band power” numbers
        delta = random.uniform(0.4, 0.8)
        theta = random.uniform(0.3, 0.7)
        alpha = random.uniform(0.2, 0.6)
        beta  = random.uniform(0.1, 0.5)
        gamma = random.uniform(0.05, 0.3)

        focus_index = beta / (alpha + theta + 1e-6)
        focus_index = max(0.0, min(1.0, focus_index))  # clamp 0~1

        payload = {
            "mode": "mock",
            "bands": {
                "delta": delta,
                "theta": theta,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            },
            "focus_index": focus_index,
        }
        await ws.send(build_packet(payload))
        await asyncio.sleep(0.1)  # 10 Hz

# -------------------------------
# LSL streaming (real Muse)
# -------------------------------
async def lsl_stream(ws: WebSocketServerProtocol):
    """
    Read Muse EEG samples via LSL and forward light-weight features.
    Note: we forward raw channels + a naive rolling-avg “focus_index”.
    You can compute richer metrics in the frontend or here later.
    """
    if not HAS_LSL:
        await ws.send(build_packet({"mode": "lsl", "error": "pylsl not installed"}))
        return

    await ws.send(build_packet({"mode": "lsl", "status": "resolving LSL stream..."}))
    streams = resolve_stream("type", "EEG", timeout=5)
    if not streams:
        await ws.send(build_packet({"mode": "lsl", "error": "no EEG LSL stream found"}))
        return

    inlet = StreamInlet(streams[0], max_buflen=5)
    await ws.send(build_packet({"mode": "lsl", "status": "connected"}))

    # naive focus index from absolute amplitude ratio on the fly
    import collections
    win = collections.deque(maxlen=50)  # ~50 samples rolling window

    while True:
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue

        # sample: [TP9, AF7, AF8, TP10, AUX] (不同设备/配置可能不同)
    
        s_abs = sum(abs(x) for x in sample[:2]) / 2.0
        win.append(s_abs)
        rolling = sum(win) / (len(win) or 1)

        payload = {
            "mode": "lsl",
            "raw": sample,            
            "amp_mean": rolling,      
            "focus_index": min(1.0, rolling / 100.0),  
        }
        await ws.send(build_packet(payload))

# -------------------------------
# WebSocket handler
# -------------------------------
async def handler(ws: WebSocketServerProtocol):
    # Decide mode by query string or env
    # e.g. ws://localhost:8765/?mode=lsl
    mode = "mock"
    if ws.path and "mode=lsl" in ws.path:
        mode = "lsl"
    if os.getenv("EEG_MODE", "").lower() == "lsl":
        mode = "lsl"

    try:
        await ws.send(build_packet({"hello": True, "mode": mode}))
        if mode == "lsl":
            await lsl_stream(ws)
        else:
            await mock_stream(ws)
    except websockets.exceptions.ConnectionClosed:
        # client disconnected
        return

async def main():
    print(f"[EEG] WebSocket server on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT, ping_interval=20, ping_timeout=20):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EEG] server stopped.")
