import numpy as np

class Trainer:
    """
    训练器类
    负责: 管理训练循环, 梯度更新, 日志记录
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.history = {'loss': [], 'acc': [], 'lr': [], 'grad_norm': []}

    def train(self, X_train, y_train, epochs=1000, batch_size=None, print_every=100):
        n_samples = len(X_train)

        for epoch in range(epochs):
            # --- 1. 数据准备 (支持 Mini-batch 扩展预留) ---
            # 简单起见，这里演示全量梯度下降 (Full Batch)
            # 如果需要 SGD，可以在这里加 Batch 循环

            # --- 2. 前向传播 ---
            # 计算网络输出 (Logits)
            logits = self.model.forward(X_train)
            # 计算损失
            loss = self.model.loss_activation.forward(logits, y_train)

            # --- 3. 统计指标 ---
            predictions = np.argmax(self.model.loss_activation.output, axis=1)
            y_labels = np.argmax(y_train, axis=1) if len(y_train.shape) == 2 else y_train
            accuracy = np.mean(predictions == y_labels)

            # 记录历史
            self.history['loss'].append(loss)
            self.history['acc'].append(accuracy)
            self.history['lr'].append(self.optimizer.current_learning_rate)

            # --- 4. 反向传播 ---
            # 从 Loss 开始反向
            self.model.loss_activation.backward(self.model.loss_activation.output, y_train)
            self.model.final_dense.backward(self.model.loss_activation.dinputs)

            # 反向流经隐藏层
            back_gradient = self.model.final_dense.dinputs
            for layer in reversed(self.model.layers):
                layer.backward(back_gradient)
                back_gradient = layer.dinputs

            # 记录梯度范数 (用于监控)
            grad_norms = [np.linalg.norm(layer.dweights) for layer in self.model.layers if hasattr(layer, "dweights")]
            self.history['grad_norm'].append(np.mean(grad_norms) if grad_norms else 0)

            # --- 5. 参数更新 ---
            self.optimizer.pre_update_params()
            for layer in self.model.layers:
                if hasattr(layer, 'weights'):
                    self.optimizer.update_params(layer)
            self.optimizer.update_params(self.model.final_dense)
            self.optimizer.post_update_params()

            # --- 6. 日志打印 ---
            if not epoch % print_every:
                print(f'Epoch: {epoch}, Acc: {accuracy:.3f}, Loss: {loss:.3f}, ' +
                      f'LR: {self.optimizer.current_learning_rate:.5f}, ' +
                      f'Grad Norm: {self.history["grad_norm"][-1]:.3f}')
