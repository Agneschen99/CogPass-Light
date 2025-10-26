from typing import Optional, Tuple, Dict, Any
import time

try:
    from pylsl import StreamInlet, resolve_stream
except Exception:
    # pylsl may be absent in test environments
    StreamInlet = None  # type: ignore
    resolve_stream = None  # type: ignore

# Module-level state (不要放到 session_state 里)
# Use Any for the runtime-optional StreamInlet to avoid typing issues when pylsl is missing
_LSL_INLET: Optional[Any] = None
_LSL_INFO: Dict[str, str] = {}
_LSL_LAST_ERR: Optional[str] = None
_LSL_CONSEC_TIMEOUTS: int = 0

# 可调参数
_LSL_TIMEOUT_SEC = 0.25     # 单次拉取超时
_LSL_MAX_CONSEC_TIMEOUTS = 8  # 连续超时达到该值时尝试重连
_LSL_STREAM_TYPE = "EEG"    # Muse 默认类型


def lsl_connect() -> bool:
    """Resolve and connect to the first EEG stream. Returns True if OK."""
    global _LSL_INLET, _LSL_INFO, _LSL_LAST_ERR, _LSL_CONSEC_TIMEOUTS
    _LSL_LAST_ERR = None
    _LSL_CONSEC_TIMEOUTS = 0
    try:
        if resolve_stream is None:
            raise RuntimeError("pylsl not available")
        streams = resolve_stream('type', _LSL_STREAM_TYPE, timeout=1.5)
        if not streams:
            raise RuntimeError(f"No LSL streams of type '{_LSL_STREAM_TYPE}'")
        inlet = StreamInlet(streams[0], max_buflen=5, processing_flags=0)
        # 轻量健康检测：time_correction 相当于 ping
        _ = inlet.time_correction(timeout=_LSL_TIMEOUT_SEC)
        # 保存
        _LSL_INLET = inlet
        info = streams[0]
        _LSL_INFO = {
            "name": info.name(),
            "type": info.type(),
            "n_ch": str(info.channel_count()),
            "sfreq": str(int(info.nominal_srate())),
            "hostname": info.hostname(),
        }
        return True
    except Exception as e:
        _LSL_INLET = None
        _LSL_INFO = {}
        _LSL_LAST_ERR = f"{type(e).__name__}: {e}"
        return False


def lsl_clear():
    """Drop current inlet and reset state."""
    global _LSL_INLET, _LSL_INFO, _LSL_LAST_ERR, _LSL_CONSEC_TIMEOUTS
    try:
        if _LSL_INLET is not None:
            _LSL_INLET.close_stream()
    except Exception:
        pass
    _LSL_INLET = None
    _LSL_INFO = {}
    _LSL_LAST_ERR = None
    _LSL_CONSEC_TIMEOUTS = 0


def lsl_set_inlet(inlet) -> None:
    """
    Set an existing StreamInlet into the central module and attempt to
    populate the info cache from it. Safe to call when pylsl is present.
    """
    global _LSL_INLET, _LSL_INFO, _LSL_LAST_ERR, _LSL_CONSEC_TIMEOUTS
    try:
        _LSL_INLET = inlet
        _LSL_CONSEC_TIMEOUTS = 0
        if inlet is None:
            _LSL_INFO = {}
            return
        try:
            info = inlet.info()
            _LSL_INFO = {
                "name": info.name(),
                "type": info.type(),
                "n_ch": str(info.channel_count()),
                "sfreq": str(int(info.nominal_srate())),
                "hostname": info.hostname(),
            }
        except Exception:
            # best-effort: leave info empty if access fails
            _LSL_INFO = {}
    except Exception:
        _LSL_INLET = None
        _LSL_INFO = {}


def lsl_health() -> Tuple[str, Dict[str, str]]:
    """
    Returns (status, info).
    status: 'ok' | 'disconnected' | 'error'
    """
    if _LSL_INLET is None:
        return "disconnected", {}
    try:
        # 快速健康检测，不消耗数据
        _ = _LSL_INLET.time_correction(timeout=0.05)
        return "ok", _LSL_INFO
    except Exception as e:
        return "error", {"last_err": f"{type(e).__name__}: {e}"}


def lsl_safe_pull(max_samples: int = 1):
    """
    拉取样本；自动处理超时并在连续超时过多时尝试重连。
    返回 (chunk, timestamps)；若无数据则返回 ([], [])。
    """
    global _LSL_CONSEC_TIMEOUTS, _LSL_LAST_ERR, _LSL_INLET
    if _LSL_INLET is None:
        return [], []
    try:
        chunk, ts = _LSL_INLET.pull_chunk(
            timeout=_LSL_TIMEOUT_SEC, max_samples=max_samples
        )
        if ts:
            _LSL_CONSEC_TIMEOUTS = 0
            # Try to record diagnostics centrally (best-effort). Import dynamically
            try:
                # diagnostics_record_pull lives in the streamlit UI module; import at runtime
                from app.eeg.streamlit_app import diagnostics_record_pull
                try:
                    diagnostics_record_pull(ts, _LSL_INFO)
                except Exception:
                    # swallow any diagnostics errors
                    pass
            except Exception:
                # not available (e.g., running outside Streamlit) — ignore
                pass
            return chunk, ts
        # 超时但无异常
        _LSL_CONSEC_TIMEOUTS += 1
        if _LSL_CONSEC_TIMEOUTS >= _LSL_MAX_CONSEC_TIMEOUTS:
            # 尝试重连
            lsl_clear()
            ok = lsl_connect()
            if not ok:
                _LSL_LAST_ERR = "Auto-reconnect failed"
        return [], []
    except Exception as e:
        _LSL_CONSEC_TIMEOUTS += 1
        _LSL_LAST_ERR = f"{type(e).__name__}: {e}"
        if _LSL_CONSEC_TIMEOUTS >= _LSL_MAX_CONSEC_TIMEOUTS:
            lsl_clear()
            lsl_connect()
        return [], []

# Expose a simple public API
__all__ = [
    'lsl_connect',
    'lsl_clear',
    'lsl_health',
    'lsl_safe_pull',
]


def lsl_info() -> Dict[str, str]:
    """Return the last-known LSL stream info cache (may be empty)."""
    return _LSL_INFO


__all__.append('lsl_info')
