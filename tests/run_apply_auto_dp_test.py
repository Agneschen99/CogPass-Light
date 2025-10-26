# Small runtime tests for ensemble + DP behavior
import numpy as np
from app.eeg import algorithms as algo
try:
    from app.eeg.adp.adaptive_dp import AdaptiveDPController
except ModuleNotFoundError:
    from src.app.eeg.adp.adaptive_dp import AdaptiveDPController

class MockModel:
    def __init__(self, prob):
        self.prob = float(prob)
    def predict_proba(self, X):
        # X is ignored; return shape (n_samples, 2)
        n = np.asarray(X).shape[0]
        p = np.full((n,), self.prob, dtype=float)
        return np.vstack([1-p, p]).T


def approx(a, b, tol=1e-6):
    return abs(a-b) <= tol


def test_single_model():
    X = np.zeros((1, 3))
    m = MockModel(0.8)
    tm = algo.TrainedModel("svm", m)
    p = algo.confidence_weighted_ensemble([tm], X)
    p0 = float(np.mean(p))
    print("SVM-only ensemble ->", p0)
    assert approx(p0, 0.8), f"Expected 0.8, got {p0}"


def test_mlp_only():
    X = np.zeros((1, 3))
    m = MockModel(0.3)
    tm = algo.TrainedModel("mlp", m)
    p = algo.confidence_weighted_ensemble([tm], X)
    p0 = float(np.mean(p))
    print("MLP-only ensemble ->", p0)
    assert approx(p0, 0.3), f"Expected 0.3, got {p0}"


def test_combined():
    X = np.zeros((1, 3))
    m1 = MockModel(0.8)
    m2 = MockModel(0.3)
    t1 = algo.TrainedModel("svm", m1)
    t2 = algo.TrainedModel("mlp", m2)
    p = algo.confidence_weighted_ensemble([t1, t2], X)
    p0 = float(np.mean(p))
    print("Combined ensemble ->", p0)
    # expected 0.6 from manual calc
    assert approx(p0, 0.6, tol=1e-6), f"Expected 0.6, got {p0}"


def test_dp_no_noise_when_eps_zero():
    dp = AdaptiveDPController(total_epsilon=1.0, delta=1e-5, strategy='per_call', eps_per_unit=0.0)
    val = 0.42
    noisy = dp.add_noise_to_scalar(val, epsilon=0.0)
    print("DP with eps=0 ->", noisy)
    assert approx(noisy, val), f"Expected no change when epsilon=0, got {noisy}"


if __name__ == '__main__':
    test_single_model()
    test_mlp_only()
    test_combined()
    test_dp_no_noise_when_eps_zero()
    print('\nAll tests passed')
