Notes For deep learning tools for biomolecular structure prediction and design, focus on Protein Models Only

For original material, please refer to [Colab Notebooks covering deep learning tools for biomolecular structure prediction and design](https://github.com/Graylab/DL4Proteins-notebooks)

# Utils

- [create_project_structure.py](./create_project_structure.py) 
  
```python
python create_project_structure.py YOUR_PROJECT_NAME
```
It will create a detailed and structured project folder for your first deep learning project organization

```bash
project_root/
│
├── data/                   # Store raw data and processed data/存放原始数据和处理后的数据
│   ├── raw/
│   └── processed/
│
├── configs/                # Configuration file (hyperparameters, paths)/配置文件 (超参数、路径)
│   └── config.yaml
│
├── src/                    # Core Source Code/核心源代码
│   ├── __init__.py
│   ├── dataset.py          # 1. Data Loader/数据处理
│   ├── model.py            # 2. Network Architecture/模型定义
│   ├── trainer.py          # 3. Training Loop/训练逻辑
│   ├── utils.py            # 4. Metrics, Logging/辅助函数
│   └── predict.py          # 5. Reasoning, Prediction Logic/推理,预测逻辑
│
├── main.py                 # Project Entry (Integrating All Modules)/项目入口 (整合所有模块)
├── requirements.txt        # Dependency/依赖库
└── README.md

```


# Chap1: Neural Networks with NumPy

## What this chapter holds?

OBJECTIVES: By the end of this workshop you should be able to understand the following concepts:

- Neuron (in the context of machine learning)
- Forward pass / backward pass
- ReLU activation
- Softmax activation loss and categorical cross entropy
- Adam optimizer
- Weights and biases and updating each
- Training and epochs

In short, you will know how to create a simple fully connected neural network only using Numpy, and what parts/modules a Neural Network should or generally will have, and what roles these parts/modules play accordingly.

We also provide a structured project file organization for our NumPy code (available at [Numpy_NN_Proj](./Chap1/Numpy_NN_Proj/)), which is quite similar to the framework we currently use for building deep learning projects with PyTorch. You should start from this foundation, and try your best to restore the original logic of this NumPy file organization for any complex network you encounter—because in essence, these complex networks are nothing more than the intricate extensions and abstract encapsulations of each module in the NumPy-based neural network.

## What you should do after learning this chapter?

### Learn/Review :
* Numpy modules, at least you should know how to use it 

    https://numpy.org/
* Matrix calculus and some necessary Linear Algebra Knowledge

    Learn from Wiki and start from wiki is a good choice
* Basic Neural Network Knowledge: You should have a grasp of the tensor flow in a simple neural network—such as the complete pipeline of input → forward propagation → loss calculation → gradient computation → backpropagation → parameter update. Additionally, you need to understand the fundamentals of training a standard neural network, including core concepts like batch, iteration, and epoch, as well as the common challenges encountered during training and their corresponding solutions.

    We strongly recommend you to watch Online Courses taught by 
Hung-yi Lee in YouTube
    
    [2021 Courses](https://www.youtube.com/playlist?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J)

    ![alt text](./Figs/1.png)

    [2022 Courses](https://www.youtube.com/playlist?list=PLJV_el3uVTsPM2mM-OQzJXziCGJa8nJL8)

    ![alt text](image.png)

### Practice :

- Build your own codebase for general regression and classification tasks, and learn to construct engineering-oriented deep learning projects. In other words, we need to learn to abstracting an existing program into a universal code framework. 
  
  Additionally, we should learn how others organize project files and identify the necessary file components for a deep learning project. 
  
  In accordance with Online courses taught by Hung-yi Lee above, please refer to https://github.com/virginiakm1988/ML2022-Spring and do the HW1 and HW2 yourself.

# Chap2: Neural Networks with PyTorch

