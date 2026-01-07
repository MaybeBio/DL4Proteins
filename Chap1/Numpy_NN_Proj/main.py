import numpy as np
from src.dataset import create_data
from src.model import UniversalDeepModel, Optimizer_Adam
from src.trainer import Trainer
from src.utils import plot_history
from src.predict import predict_single

# 1. 准备数据
X_data, y_data = create_data(n_samples=500, n_features=2, n_classes=3)

# 2. 配置模型与优化器
model = UniversalDeepModel(
    input_dim=2, 
    hidden_layer_sizes=[64, 64], 
    output_dim=3
)
optimizer = Optimizer_Adam(learning_rate=0.05, decay=1e-4)

# 3. 初始化训练器
trainer = Trainer(model, optimizer)

# 4. 开始训练
print("开始训练...")
trainer.train(X_data, y_data, epochs=2000, print_every=100)

# 5. 可视化结果 (如果运行环境支持)
# plot_history(trainer.history)

# 6. 推理示例
print("\n模型推理示例:")
new_sample = np.array([0.5, -1.2])
pred_class, probs = predict_single(model, new_sample)
print(f"输入: {new_sample}")
print(f"预测类别: {pred_class}")
print(f"置信度: {probs}")
