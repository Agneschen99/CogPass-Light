from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import time
from typing import Dict, Any, Optional
import numpy as np

SETTINGS_PATH = Path.home() / ".neuroplan" / "settings.json"

def _load_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_settings(cfg: Dict[str, Any]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


class AdaptiveDPController:
    """
    Lightweight controller for managing differential privacy budget.
    - total_epsilon: total ε budget
    - delta: composition delta (for Gaussian mechanism)
    - strategy:
        'per_call'   : consume fixed ε per call
        'per_epoch'  : consume ε per training epoch
        'per_minute' : minute-based replenishment
    - S: sensitivity (default 1.0)
    - Settings are persisted to ~/.neuroplan/settings.json under key 'dp'
    """
    def __init__(
        self,
        total_epsilon: float = 3.0,
        delta: float = 1e-5,
        strategy: str = "per_call",
        eps_per_unit: Optional[float] = None,
        S: float = 1.0,
        # session-style / backward-compatible aliases
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        eps_total: Optional[float] = None,
        eps_base: Optional[float] = None,
        alpha: float = 0.8,
        eps_floor: float = 0.01,
    ):
        # Accept alias names for backward compatibility
        if eps_total is not None:
            total_epsilon = float(eps_total)

        if eps_base is not None:
            eps_per_unit = float(eps_base)

        self.total_epsilon = float(total_epsilon)
        self.delta = float(delta)
        self.strategy = strategy
        self.S = float(S)

        # Keep session metadata for bookkeeping
        self.user_id = user_id
        self.session_id = session_id
        self.alpha = float(alpha)
        self.eps_floor = float(eps_floor)

        # unit epsilon (default: 1% of total, but at least 0.01; enforce eps_floor)
        if eps_per_unit is not None:
            self.eps_per_unit = float(max(eps_per_unit, self.eps_floor))
        else:
            self.eps_per_unit = float(max(self.eps_floor, max(0.01, 0.01 * self.total_epsilon)))

        # runtime state
        self.spent_epsilon = 0.0
        self.last_timestamp = time.time()

    @classmethod
    def from_settings(cls) -> "AdaptiveDPController":
        cfg = _load_settings().get("dp", {})
        # support both legacy keys and newer session-style keys
        total_eps = cfg.get("total_epsilon", cfg.get("eps_total", 3.0))
        eps_unit = cfg.get("eps_per_unit", cfg.get("eps_base", None))
        return cls(
            total_epsilon=total_eps,
            delta=cfg.get("delta", 1e-5),
            strategy=cfg.get("strategy", "per_call"),
            eps_per_unit=eps_unit,
            S=cfg.get("S", 1.0),
            user_id=cfg.get("user_id"),
            session_id=cfg.get("session_id"),
            alpha=cfg.get("alpha", 0.8),
            eps_floor=cfg.get("eps_floor", 0.01),
        )._restore_runtime(cfg)

    def _restore_runtime(self, cfg: Dict[str, Any]) -> "AdaptiveDPController":
        self.spent_epsilon = float(cfg.get("spent_epsilon", 0.0))
        self.last_timestamp = float(cfg.get("last_timestamp", time.time()))
        return self

    def save_settings(self) -> None:
        all_cfg = _load_settings()
        all_cfg["dp"] = {
            "total_epsilon": self.total_epsilon,
            "delta": self.delta,
            "strategy": self.strategy,
            # persist both canonical and session-style fields for compatibility
            "eps_per_unit": self.eps_per_unit,
            "eps_base": self.eps_per_unit,
            "eps_total": self.total_epsilon,
            "S": self.S,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "alpha": getattr(self, "alpha", 0.8),
            "eps_floor": getattr(self, "eps_floor", 0.01),
            "spent_epsilon": self.spent_epsilon,
            "last_timestamp": self.last_timestamp,
        }
        _save_settings(all_cfg)

    def clear_settings(self) -> None:
        all_cfg = _load_settings()
        if "dp" in all_cfg:
            del all_cfg["dp"]
            _save_settings(all_cfg)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 1e-9

    def _consume(self, eps: float) -> float:
        eps = float(max(0.0, eps))
        if eps > self.remaining:
            eps = self.remaining
        self.spent_epsilon += eps
        self.last_timestamp = time.time()
        return eps

    def _alloc_per_minute(self) -> float:
        """Allocate per-minute budget: simple linear refill based on time elapsed."""
        now = time.time()
        elapsed = max(0.0, now - self.last_timestamp)
        bucket = (elapsed / 60.0) * self.eps_per_unit
        give_back = min(bucket, self.spent_epsilon)
        if give_back > 0:
            self.spent_epsilon -= give_back
            self.last_timestamp = now
        return min(self.eps_per_unit, self.remaining)

    def request_epsilon(self, units: float = 1.0) -> float:
        """Return and deduct available ε according to strategy."""
        if self.exhausted:
            return 0.0
        if self.strategy == "per_call":
            need = self.eps_per_unit * units
            return self._consume(need)
        elif self.strategy == "per_epoch":
            need = self.eps_per_unit * max(1.0, units)
            return self._consume(need)
        elif self.strategy == "per_minute":
            alloc = self._alloc_per_minute()
            return self._consume(alloc)
        else:
            return self._consume(self.eps_per_unit * units)

    def gaussian_sigma(self, epsilon: float) -> float:
        """σ = S * sqrt(2 ln(1.25/δ)) / ε. If ε==0 return inf."""
        if epsilon <= 0:
            return float("inf")
        return self.S * np.sqrt(2.0 * np.log(1.25 / self.delta)) / epsilon

    def add_noise_to_scalar(self, value: float, epsilon: Optional[float] = None) -> float:
        """Add Gaussian noise to a scalar; if epsilon is None, request from the controller."""
        if epsilon is None:
            epsilon = self.request_epsilon(1.0)
        if epsilon <= 0:
            return float(value)
        sigma = self.gaussian_sigma(epsilon)
        return float(value + np.random.normal(0.0, sigma))

    def add_noise_to_probs(self, probs: np.ndarray, epsilon: Optional[float] = None) -> np.ndarray:
        """Add Gaussian noise to a probability vector and clip to [0,1]."""
        if epsilon is None:
            epsilon = self.request_epsilon(1.0)
        if epsilon <= 0:
            return probs
        sigma = self.gaussian_sigma(epsilon)
        noisy = probs + np.random.normal(0.0, sigma, size=probs.shape)
        return np.clip(noisy, 0.0, 1.0)
