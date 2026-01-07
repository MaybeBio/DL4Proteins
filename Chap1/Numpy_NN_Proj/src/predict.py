import numpy as np

def predict_single(model, sample):
    """对单个样本进行预测"""
    # 确保样本是 2D 数组 (1, n_features)
    if sample.ndim == 1:
        sample = sample.reshape(1, -1)

    probs = model.predict(sample)
    pred_class = np.argmax(probs, axis=1)[0]
    return pred_class, probs
