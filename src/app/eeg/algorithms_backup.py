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
# 1. 基础信号处理 (Signal Processing Helpers) - 用于 clean_app.py
# =========================================================================

def _bandpower_fft(sig: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """计算指定频段的功率 (基于 FFT)"""
    x = np.asarray(sig, dtype=float)
    if x.size == 0 or fs <= 0:
        return 0.0
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(x)) ** 2) / max(x.size, 1)
    sel = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    if not np.any(sel):
        return 0.0
    return float(np.sum(psd[sel]))

def beta_alpha_ratio(eeg_data: np.ndarray, fs: float = 256.0) -> float:
    """
    计算 Beta/Alpha 比率 (专注度简单指标)
    供 clean_app.py 实时显示用
    """
    # 确保输入是 1D 数组
    sig = np.asarray(eeg_data).ravel()
    beta = _bandpower_fft(sig, fs, 13.0, 30.0)
    alpha = _bandpower_fft(sig, fs, 8.0, 13.0)
    # 防止除以零
    return beta / max(alpha, 1e-6)

def build_online_feats_from_raw(raw: np.ndarray, fs: float) -> np.ndarray:
    """
    从原始脑波提取 7 个核心特征 (用于 AI 模型输入)
    Returns: [delta, theta, alpha, beta, A/T, B/(A+T), T/B]
    """
    sig = np.asarray(raw, dtype=float).ravel()
    delta = _bandpower_fft(sig, fs, 1.0, 4.0)
    theta = _bandpower_fft(sig, fs, 4.0, 8.0)
    alpha = _bandpower_fft(sig, fs, 8.0, 13.0)
    beta = _bandpower_fft(sig, fs, 13.0, 30.0)
    
    alpha_theta = alpha / max(theta, 1e-9)
    beta_alpha_theta = beta / max(alpha + theta, 1e-9)
    theta_beta = theta / max(beta, 1e-9)
    
    return np.array([delta, theta, alpha, beta, alpha_theta, beta_alpha_theta, theta_beta], dtype=float)

# =========================================================================
# 2. 高级统计与平滑 (Advanced Stats & Smoothing)
# =========================================================================

def robust_stats(x: np.ndarray) -> Tuple[float, float]:
    """返回 (median, scaled_MAD)，比 mean/std 更抗干扰"""
    x = np.asarray(x, dtype=float)
    if x.size < 5:
        mu = float(np.mean(x)) if x.size else 0.0
        sd = float(np.std(x)) if x.size else 1.0
        return mu, max(sd / 1.2533, 1e-6)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return med, max(1.4826 * mad, 1e-6)

def ewma_1d(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """指数加权移动平均 (平滑曲线)"""
    x = np.asarray(x, dtype=float)
    if x.size == 0: return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i-1]
    return y

# =========================================================================
# 3. 训练与模型辅助 (Training Helpers) - 修复 bootstrap_model 报错的关键
# =========================================================================

def make_linear_svm_pipeline(pca_dim: int = 20, C: float = 1.0):
    """创建 AI 流水线: 标准化 -> PCA降维 -> LinearSVC分类器"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    
    # 基础分类器
    base_clf = LinearSVC(C=C, class_weight='balanced', dual=False, random_state=42)
    # 包装 calibration 以输出概率 (predict_proba)
    calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=pca_dim)),
        ('clf', calibrated_clf)
    ])

def bayes_optimize(pipeline, param_space, X, y, cv_splits=5, n_iter=25, scoring='f1'):
    """自动调参 (Bayes 或 Grid Search)"""
    from sklearn.model_selection import GridSearchCV
    
    # 简化版：直接使用 Grid Search 作为 fallback，保证能跑通
    # 将连续参数转为离散点
    grid_space = {}
    for key, value in param_space.items():
        if isinstance(value, tuple) and len(value) >= 2:
            if isinstance(value[0], int):
                grid_space[key] = [value[0], value[1]] # 简化为端点
            else:
                grid_space[key] = np.linspace(value[0], value[1], 3).tolist() # 取3个点
        else:
            grid_space[key] = value
            
    search = GridSearchCV(pipeline, grid_space, cv=cv_splits, scoring=scoring, n_jobs=-1)
    search.fit(X, y)
    return search, False

def save_model_bundle(model, meta, model_dir="model_store/eeg_mem_model"):
    """保存模型和元数据到硬盘"""
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型二进制
    model_path = os.path.join(model_dir, "eeg_mem_model.model")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 保存元数据 JSON
    meta_path = os.path.join(model_dir, "eeg_mem_model.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"✅ 模型已保存: {model_path}")

# =========================================================================
# 4. 集成预测 (Ensemble & Prediction)
# =========================================================================

@dataclass
class TrainedModel:
    name: str
    model: Any

def confidence_weighted_predict(models: List[TrainedModel], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """多模型投票预测 (如果你未来有多个模型)"""
    probs = []
    for m in models:
        if hasattr(m.model, "predict_proba"):
            p = m.model.predict_proba(X)
            probs.append(p[:, 1] if p.shape[1] == 2 else p[:, 0])
        else:
            probs.append(m.model.predict(X).astype(float))
            
    P = np.vstack(probs)
    p_ens = np.mean(P, axis=0) # 简单平均
    conf = 1.0 - np.std(P, axis=0) # 分歧越小，置信度越高
    return p_ens, conf

# 兼容旧代码调用
dp_confidence_weighted_predict = confidence_weighted_predict
