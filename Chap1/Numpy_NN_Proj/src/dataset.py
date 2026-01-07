import numpy as np

def create_data(n_samples=300, n_features=2, n_classes=3):
    """
    生成演示用数据
    这里使用简单的随机正态分布数据
    """
    print(f"正在生成数据: {n_samples} 样本, {n_features} 特征, {n_classes} 类别...")
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=(n_samples,))
    return X, y
