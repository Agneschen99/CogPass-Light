# bootstrap_model.py
import numpy as np
import sys
import os

# 确保能找到 src 目录
sys.path.append(os.path.join(os.getcwd(), 'src'))

from app.eeg.train_and_save import train_linear_svm_bayes, save_bundle

def main():
    print("正在生成初始模型 (适应新的 7 特征算法)...")
    
    # 🔥 关键修改：特征数改为 7，必须与 algorithms.py 的输出一致！
    n_samples = 100
    n_features = 7  # <--- 改成 7
    
    # 模拟数据 (100个样本, 7个特征)
    X_dummy = np.random.rand(n_samples, n_features)
    y_dummy = np.random.randint(0, 2, n_samples)
    
    try:
        print("开始训练 LinearSVC...")
        # 这里的 train_linear_svm_bayes 会调用你刚才更新的 algorithms.py
        # 也就是会自动使用 pca_dim=5 的新配置
        model, meta = train_linear_svm_bayes(X_dummy, y_dummy)
        
        save_bundle(model, meta)
        print("✅ 新模型已保存！现在可以运行 clean_app.py 了。")
        
    except Exception as e:
        print(f"❌ 出错了: {e}")

if __name__ == "__main__":
    main()
