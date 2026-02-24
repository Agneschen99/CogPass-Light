from __future__ import annotations
import os
import json
import time
import pickle
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# =========================================================================
# 1. 基础信号处理 (Signal Processing) - clean_app.py 画图必需
# =========================================================================

def _bandpower_fft(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    # 转为 float，消除 NaN、inf
    x = np.nan_to_num(np.asarray(sig, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    
    # 空数组或非法采样率直接返回 0
    if x.size == 0 or fs <= 0:
        return 0.0

    # 去直流分量
    x = x - np.mean(x)

    # FFT + PSD
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(x))**2) / max(x.size, 1)

    # 频段选择
    sel = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    if not np.any(sel):
        return 0.0

    # 求 bandpower
    return float(np.sum(psd[sel]))


def _bandpower_multich(raw: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """
    多通道 bandpower：
    - 如果是 1D: 当成单通道
    - 如果是 2D: 对每个通道算 bandpower，然后取平均
      Muse: shape ~ (N_samples, 4) -> AF7, AF8, TP9, TP10
    """
    arr = np.asarray(raw, dtype=float)

    if arr.ndim == 1:
        return _bandpower_fft(arr, fs, f_lo, f_hi)

    if arr.ndim == 2:
        powers = []
        n_ch = arr.shape[1]
        for ch in range(n_ch):
            powers.append(_bandpower_fft(arr[:, ch], fs, f_lo, f_hi))
        return float(np.mean(powers)) if len(powers) > 0 else 0.0

    # 其它奇怪维度：先压成 (N, C)，再平均
    arr2 = arr.reshape(arr.shape[0], -1)
    powers = []
    for ch in range(arr2.shape[1]):
        powers.append(_bandpower_fft(arr2[:, ch], fs, f_lo, f_hi))
    return float(np.mean(powers)) if len(powers) > 0 else 0.0


def beta_alpha_ratio(eeg_data: np.ndarray, fs: float = 256.0) -> float:
    """
    计算 Beta/Alpha 比率 (专注度指标)
    兼容：
    - 单通道: 1D (N,)
    - 多通道: 2D (N, C)，如 Muse 4 通道
    """
    beta = _bandpower_multich(eeg_data, fs, 13.0, 30.0)
    alpha = _bandpower_multich(eeg_data, fs, 8.0, 13.0)

    # 防止除以 0 & 极端爆炸
    alpha = max(alpha, 1e-3)
    return beta / alpha


def build_online_feats_from_raw(raw: np.ndarray, fs: float) -> np.ndarray:
    """
    从 Muse 原始窗口提取 7 个核心特征喂给 AI 模型。

    raw:
      - 推荐: shape = (N_samples, 4)  # AF7, AF8, TP9, TP10
      - 也兼容: (N_samples,) 单通道
    特征:
      [delta, theta, alpha, beta,
       alpha/theta, beta/(alpha+theta), theta/beta]
    """
    # 多通道 bandpower（对每个通道算，再取平均）
    delta = _bandpower_multich(raw, fs, 1.0, 4.0)
    theta = _bandpower_multich(raw, fs, 4.0, 8.0)
    alpha = _bandpower_multich(raw, fs, 8.0, 13.0)
    beta  = _bandpower_multich(raw, fs, 13.0, 30.0)

    # 数值稳定处理（避免 ratio 爆炸）
    alpha = max(alpha, 1e-6)
    theta = max(theta, 1e-6)
    beta  = max(beta, 1e-6)

    alpha_theta      = alpha / theta
    beta_alpha_theta = beta / (alpha + theta)
    theta_beta       = theta / beta

    return np.array(
        [delta, theta, alpha, beta,
         alpha_theta, beta_alpha_theta, theta_beta],
        dtype=float
    )

# =========================================================================
# 2. 训练辅助 (Training Helpers) - bootstrap_model.py 必需
# =========================================================================

def make_linear_svm_pipeline(pca_dim: int = 20, C: float = 1.0):
    """
    创建 AI 流水线:
    StandardScaler -> PCA -> Calibrated LinearSVC

    注意:
      实际使用时，pca__n_components 会在 bayes_optimize 里
      根据 X.shape[1] 自动裁剪，不会超过特征维度。
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    
    base_clf = LinearSVC(C=C, class_weight='balanced', dual=False, random_state=42)
    calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_dim)),
        ('clf', calibrated_clf)
    ])


def bayes_optimize(pipeline, param_space, X, y, cv_splits=5, n_iter=25, scoring='f1'):
    """
    自动调参 (这里用 GridSearchCV 的简化版)，并对 PCA 维度做安全裁剪：
    - 如果搜索空间里有 'pca__n_components': (low, high)
      会自动把 high 裁剪到 <= X.shape[1]，防止 n_components > n_features 报错。
    """
    from sklearn.model_selection import GridSearchCV
    
    n_features = X.shape[1]
    grid_space: Dict[str, Any] = {}

    for key, value in param_space.items():
        # 连续取值区间 Tuple -> 离散点
        if isinstance(value, tuple) and len(value) >= 2:
            # 对 PCA 维度做特别处理
            if key.startswith("pca__n_components"):
                # 假设是整数范围 (low, high)
                v0, v1 = value[0], value[1]
                # 限制在 [1, n_features]
                low = int(max(1, min(v0, n_features)))
                high = int(max(1, min(v1, n_features)))
                # 去重 + 排序
                grid_space[key] = sorted(set([low, high]))
            else:
                # 其它参数：整数就取两个端点，浮点就取 3 个均匀点
                if isinstance(value[0], int):
                    grid_space[key] = [value[0], value[1]]
                else:
                    grid_space[key] = np.linspace(value[0], value[1], 3).tolist()
        else:
            grid_space[key] = value
            
    search = GridSearchCV(pipeline, grid_space, cv=cv_splits, scoring=scoring, n_jobs=-1)
    search.fit(X, y)
    return search, False


def save_model_bundle(model, meta, model_dir="model_store/eeg_mem_model"):
    """保存模型 + meta 到统一路径"""
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "eeg_mem_model.model")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    meta_path = os.path.join(model_dir, "eeg_mem_model.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ 模型已保存: {model_path}")

# =========================================================================
# 3. 预测辅助 (Prediction Helpers) - AI 预测必需
# =========================================================================

@dataclass
class TrainedModel:
    name: str
    model: Any

def confidence_weighted_predict(models: List[TrainedModel], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    多模型投票预测：
      - 如果模型有 predict_proba: 使用概率的第 1 类 (positive class)
      - 否则使用 predict 的结果当作分数
      返回:
        p_ens: 集成后的平均概率/分数
        conf: 1 - std(各模型输出) 作为"置信度"近似
    """
    probs = []
    for m in models:
        if hasattr(m.model, "predict_proba"):
            p = m.model.predict_proba(X)
            probs.append(p[:, 1] if p.shape[1] == 2 else p[:, 0])
        else:
            probs.append(m.model.predict(X).astype(float))
            
    P = np.vstack(probs)          # shape: (n_models, n_samples)
    p_ens = np.mean(P, axis=0)    # 集成概率
    conf = 1.0 - np.std(P, axis=0)  # 波动越小 -> 置信度越高
    return p_ens, conf

# 兼容旧代码调用
dp_confidence_weighted_predict = confidence_weighted_predict
