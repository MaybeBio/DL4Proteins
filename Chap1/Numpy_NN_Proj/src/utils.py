import matplotlib.pyplot as plt

def plot_history(history):
    """可视化训练历史"""
    print("绘制训练曲线...")
    epochs = range(len(history['loss']))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], label='Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['acc'], label='Accuracy', color='green')
    plt.title('Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['grad_norm'], label='Grad Norm', color='orange')
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')

    plt.tight_layout()
    plt.show() # 在 Notebook 中会显示，在脚本中可能需要保存
    # plt.savefig('training_result.png')
