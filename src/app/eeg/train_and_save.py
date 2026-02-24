from typing import Dict, Any, Tuple
import numpy as np
from datetime import datetime

# 修正 1: 导入正牌算法库 (包含 4通道平均逻辑), 弃用 algorithms_backup
from app.eeg import algorithms as algo

def _now():
    return datetime.now().strftime("%Y-%m-%d")

def train_linear_svm_bayes(X: np.ndarray, y: np.ndarray) -> Tuple[object, Dict[str, Any]]:
    """
    训练 LinearSVC 模型，自动适应输入特征数量，防止 PCA 崩溃。
    """
    # 获取真实特征维度 (例如 7)
    n_samples, n_features = X.shape
    
    print(f"正在训练模型... 样本数: {n_samples}, 特征数: {n_features}")

    # 修正 2: 动态设定 PCA 默认值
    # 如果特征很少 (比如7个)，默认保留 5个；绝不能超过 n_features
    default_pca = min(5, n_features)
    
    # 创建 pipeline
    pipe = algo.make_linear_svm_pipeline(pca_dim=default_pca, C=1.0)
    
    # 修正 3: 动态设定搜索空间 (Grid Search Space)
    # 搜索上限不能超过特征总数 n_features
    max_pca = n_features
    min_pca = max(1, min(3, n_features)) # 至少保留1个，最多从3开始搜
    
    # 构造安全的搜索区间
    if min_pca < max_pca:
        # 如果空间足够 (比如 3 到 7)，就搜个范围
        pca_range = (min_pca, max_pca)
    else:
        # 如果特征极少 (比如一共就3个)，就固定只试这一个值
        pca_range = [max_pca]

    print(f"PCA 搜索范围设定为: {pca_range}")

    space = {
        "pca__n_components": pca_range, 
        "clf__C": (1e-3, 10.0, "log-uniform"),
    }
    
    # 开始优化 (调用 algorithms.py 里的优化器)
    search, used_bayes = algo.bayes_optimize(pipe, space, X, y, cv_splits=5, n_iter=25, scoring="f1")
    best = search.best_estimator_
    
    meta = {
        "model_name": "EEG Memory (4-ch Avg + Robust PCA)",
        "trained_on": _now(),
        "classifier": "LinearSVC (PCA+Calibrated)",
        "input_features": int(n_features),  # 新字段
        "features": int(n_features),        # 兼容老代码
        "metrics": {
            "f1": float(search.best_score_)
        },
        "search": "BayesSearchCV" if used_bayes else "GridSearchCV",
        "param_best": search.best_params_,
    }
    
    return best, meta

def save_bundle(model, meta, model_dir="model_store/eeg_mem_model"):
    # 调用算法库的保存函数
    algo.save_model_bundle(model, meta, model_dir=model_dir)
