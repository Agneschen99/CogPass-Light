import sys, os
import numpy as np

# Ensure 'src' is on sys.path so `import app...` works when running tests from repo root
ROOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if ROOT_SRC not in sys.path:
    sys.path.insert(0, ROOT_SRC)

from app.eeg.streamlit_app import apply_auto_dp_and_render_module
from app.eeg import algorithms as algo


class DummyEstimator:
    """Simple estimator implementing predict_proba for testing."""
    def predict_proba(self, X):
        # Return probability 0.7 for any sample
        X = np.asarray(X)
        n = X.shape[0] if X.ndim > 1 else 1
        proba = np.zeros((n, 2), dtype=float)
        proba[:, 1] = 0.7
        proba[:, 0] = 0.3
        return proba


def test_auto_dp_smoke():
    # Create tiny feature matrix (2 samples, 3 features)
    X = np.array([[0.1, 0.2, 0.3], [0.0, -0.1, 0.2]])
    dummy = DummyEstimator()
    # Bundle as dict with 'svm' key
    bundle = {"svm": dummy}
    out = apply_auto_dp_and_render_module(X, bundle, thr=0.5)
    # Assertions: probabilities in [0,1], eps_used <= eps_total, label in {0,1}
    assert 0.0 <= out["prob_raw"] <= 1.0, f"prob_raw out of range: {out['prob_raw']}"
    assert 0.0 <= out["prob_private"] <= 1.0, f"prob_private out of range: {out['prob_private']}"
    assert out["eps_used"] <= out["eps_total"] + 1e-9, "spent epsilon exceeds total"
    assert out["label"] in (0, 1), "label not 0/1"
    print("Smoke test passed: DP wiring returns probabilities in [0,1] and eps bookkeeping is consistent.")


if __name__ == '__main__':
    test_auto_dp_smoke()
