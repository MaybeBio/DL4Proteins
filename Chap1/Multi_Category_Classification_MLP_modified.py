import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 第一部分：底层核心类库 
# =============================================================================

# 1. 全连接层
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        Description
        -----------
        初始化全连接层的权重和偏置, 注意这是一个抽象的全连接层类, 不是输入层！

        Args
        ----
        n_inputs : int
            当前层输入特征的数量(例, 输入28x28图像, 则n_inputs=784, 就是feature维度)
        n_neurons : int
            当前层输入神经元的数量

        Notes
        -----
        - 1, 解释: 输入层数据格式是「样本 x 特征(n_samples, n_features), 隐藏层核心是「神经元数量(n_neurons), 权重矩阵用「特征 x 神经元(n_features, n_neurons)」的维度设计,正是为了通过矩阵乘法让两者高效衔接;
        输入数据形状是 (n_samples, n_features)(比如 100 个样本, 每个样本 784 个特征 → (100, 784)),权重矩阵是 (n_features, n_neurons)(784 个特征 x 10 个神经元 → (784, 10));
        (n_samples, n_features) @ (n_features, n_neurons) = (n_samples, n_neurons)
        - 2, 当前全连接层单层的构建未涉及激活函数, 只是单纯的线性变换(矩阵乘法+偏置), 激活函数会在后续单独实现, 所以所有的output都是线性变换的结果, 我们直接考虑loss计算和反向传播即可, 不需要考虑激活函数的非线性影响, 就是将output作为当前层的最终输出(类比激活函数之后的输出)
        """

        # 初始化(n_inputs, n_neurons)形状的权重矩阵, 采用随机的标准正态分布
        # 缩放0.01以防止权重过大, 避免前向传播时输出过大导致梯度消失/爆炸
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        # 初始为(1, n_neurons)形状的全0偏置向量, 每个神经元对应一个偏置值
        # 每个神经元只有 1 个偏置（管 “神经元的偏移”）；所有样本共享这组偏置（管 “规则通用”）
        # 偏置的本质是 “与样本无关的神经元偏移”，所有样本共享同一组偏置（1 个神经元 1 个偏置）
        # 偏置的第 2 维（神经元数）必须和「输入 × 权重」结果的第 2 维（神经元数）完全一致（比如都是 10）—— 因为要给每个神经元加专属偏移，维度不匹配就加错了；
        # 偏置的第 1 维（样本数）用 1，是因为广播机制会自动把 1 扩展成实际样本数（比如 100）—— 既满足 “所有样本共享偏置”，又避免存储冗余（不用存 100 份重复的偏置）。
        # 所以这里偏置形状是 (1, n_neurons), 而不是反过来(n_neurons, 1)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        """
        Description
        -----------
        前向传播, 计算当前层的输出(输入的线性变换)

        Args
        ----
        inputs : np.ndarray
            输入数据, 上一层的输出, 形状为(上一层输出样本数, 上一层输出特征数), 也就是(n_samples, n_inputs/features)

            
        Notes
        -----
        - 1, forward方法依然是在前面整体抽象的全连接层中定义的,不特指输入层到第一层隐藏层, 而是可以作为任意两层之间的全连接层;
        只能说是当前层, 无论是哪一层, 全连接层的输出形状都是 (n_samples, 上一层神经元数);
        不管是输入层后的第一层，还是隐藏层之间，只要传入符合维度的 inputs(上一层输出/上一层神经元数/当前层输入feature数), 并提前定义好对应维度的 weights(n_in_features x 当前层神经元数)和 biases(1 x 当前层神经元数），就能自动完成前向传播计算。
        - 2, 理解抽象全连接层中inputs的维度:
            - 通用维度: inputs.shape = (样本数, 当前层输入特征数)，与层位置无关；
            - 样本数不变：所有层的 inputs 第一维度都是同一批样本数，贯穿网络；——》从矩阵乘法角度来看, 任意中间层的行数=第1个矩阵的行数
            - 输入特征数来源：当前层的 “输入特征数” = 上一层的输出特征数 = 上一层的神经元数量；
            - 抽象复用性：正因为维度规则通用，这个 forward 方法才能作为任意全连接层使用，只需匹配上一层输出和自身权重维度即可
        - 3, 此处的全连接层不包括激活函数, 只是单纯的线性变换(矩阵乘法+偏置), 激活函数会在后续单独实现, 所以所有的output都是线性变换的结果, 我们直接考虑loss计算和反向传播即可, 不需要考虑激活函数的非线性影响, 就是将output作为当前层的最终输出(类比激活函数之后的输出)
        """
        # 保存当前层的输入, 用于后续反向传播计算梯度
        self.inputs = inputs

        # 计算当前层的输出: output = inputs @ weights + biases (矩阵乘法+广播机制)
        # 输入数据 X：100 个样本，每个样本 784 个特征 → 形状 (100, 784)；
        # 权重 weights：784 个特征 × 10 个神经元 → 形状 (784, 10)；——》X @ weights → 形状 (100, 784) @ (784, 10) = (100, 10)
        # 偏置 biases：1 行 × 10 个神经元 → 形状 (1, 10)（全 0 初始化，即 [[0,0,0,...,0]]）。——》NumPy 的广播机制会自动把 (1,10) 的偏置 “复制扩展” 成 (100,10)
        # 刚好实现了 “给每个神经元的所有样本输出，都加同一个偏移量”—— 这正是 “与样本无关、每个神经元 1 个偏置” 
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        """
        Description
        -----------
        反向传播, 计算当前层的梯度 (更新梯度值用于优化器更新参数)

        Args
        ----
        dvalues : np.ndarray
            下一层传递过来的梯度, 损失对当前层输出的偏导, 作为当前层需要计算梯度的起点;
            dvalues(当前层梯度起点) = ∂L(整体loss)/∂out(当前层输出, 也就是下一层输入)

        
        Notes
        -----
        - 1, 我们这里的表述是下一层(靠近output)传到上一层(靠近input), 从loss传到输入的反向顺序说法, 所谓的上下是按照正常正向数据传递的说法表述
        - 2, 对于矩阵微积分求导部分的数学符号以及规则说明, 可以参考: https://blog.csdn.net/weixin_62528784/article/details/156519242?spm=1001.2014.3001.5501
        """
        # 计算权重的梯度：损失对权重的偏导 = 输入的转置 @ 下一层梯度
        # 原理: 依据链式法则, ∂L/∂W = ∂L/∂out * ∂out/∂W
        # 其中 ∂out/∂W = inputs.T, 因为 out = inputs @ weights + biases——》这一点可以从矩阵求导的分母布局法理解(输入的转置的形状正好和权重形状匹配)
        # 而 ∂L/∂out 就是 dvalues, 因为 dvalues = ∂L/∂out, dvalues就是定义为loss对这一层输出的梯度, 所以dvalues是我们计算的起点
        self.dweights = np.dot(self.inputs.T, dvalues)

        # 计算偏置的梯度：损失对偏置的偏导 = 下一层梯度的求和(下一层沿样本轴求和)
        # 原理: 依据链式法则, ∂L/∂b = ∂L/∂out * ∂out/∂b
        # 其中 ∂out/∂b = 1 (因为偏置是加法项, 对每个样本都一样), 因为 out = inputs @ weights + biases
        # 所以 ∂L/∂b = sum(∂L/∂out) = sum(dvalues)
        # keepdims=True保持维度为(1, n_neurons)，与偏置形状一致
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # 计算输入的梯度：损失对输入的偏导 = 下一层梯度 @ 权重的转置
        # 原理: 依据链式法则, ∂L/∂inputs = ∂L/∂out * ∂out/∂inputs
        # 其中 ∂out/∂inputs = weights.T, 因为 out = inputs @ weights + biases ——》和前面一样可以从矩阵求导的分母布局法理解(权重的转置的形状正好和输入形状匹配)
        # 而 ∂L/∂out 就是 dvalues, 因为 dvalues = ∂L/∂out
        # 所以 ∂L/∂inputs = dvalues @ weights.T
        # 原理: 将梯度反向传给上一层, 用于前一层的参数更新
        self.dinputs = np.dot(dvalues, self.weights.T)


# 2. ReLU 激活函数
class Activation_ReLU:
    """
    Description
    -----------
    ReLU(Rectified Linear Unit) 激活函数
    用于引入非线性, 解决线性模型无法拟合复杂数据的问题

    Notes
    -----
    - 1, 此处单独实现ReLU激活函数类, 作为独立的激活层使用, 不考虑与全连接层耦合
    """
    def forward(self, inputs):
        """
        Description
        -----------
        前向传播, 计算ReLU激活函数的输出(out_relu)

        Args
        ----
        inputs : np.ndarray
            in_relu, 输入数据, 上一层的输出, 全连接层的线性输出, 形状与全连接层输出一致, 也就是不考虑与激活函数耦合时的全连接层输出;
            
        Notes
        -----
        - 1, 理论上全连接层输出inputs+本层激活函数之后的输出才是当前层的最终输出, 但由于此处不考虑耦合, 此处只是独立的1个激活层, 所以直接将ReLU的输出作为当前层的最终输出
        """

        # 保存当前层的输入(in_relu), 用于后续反向传播判断梯度是否为0
        self.inputs = inputs
        # 计算ReLU激活函数的输出(out_relu): output = max(0, inputs), ReLU函数将负值置0, 保持正值不变
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """ 
        Description
        -----------
        反向传播, 计算ReLU激活函数的梯度, 用于更新前一层的梯度(ReLU层链式法则传递), 
        计算公式: ∂L/∂in_relu = ∂L/∂out_relu * ∂out_relu/∂in_relu
        
        Args
        ----
        dvalues : np.ndarray
            out_relu, 下一层传递过来的梯度, 损失对当前层输出的偏导, 也就是损失对ReLU层输出的偏导, 作为当前层需要计算梯度的起点;
            dvalues(当前层梯度起点) = ∂L(整体loss)/∂out(当前层输出, 也就是下一层输入)

        Notes
        -----
        - 1, ReLU层在反向传播时的核心任务: 
            - 接收下一层传递过来的梯度 dvalues = ∂L/∂out_relu (损失对ReLU层输出的偏导);
            - 计算当前层的梯度 dinputs = ∂L/∂in_relu (损失对ReLU层输入的偏导), 传递给前一层用于更新梯度;
            - 将调整后的梯度传递给上一层(通常是全连接层), 供上一层计算参数(权重/偏置)的梯度;
            - 依据ReLU的梯度规则调整梯度值(输入<=0位置的梯度置0, 输入>0位置的梯度保持不变);
        - 2, 激活函数层都是剥离开来, 单独实现的, 上一个全连接层的输出作为当前ReLU层的输入, 当前ReLU层的输出作为下一个全连接层的输入;
        """

        # 复制下一层传递过来的梯度, 作为当前层的梯度初始值
        # 若ReLU层后接全连接层, 则dvalues就是全连接层backward方法计算出的self.dinputs(全连接层的输入梯度, 对应ReLU层的输出梯度)
        # 为什么复制? 因为我们需要修改梯度值, 不能直接修改传入的dvalues, 因为下一层的梯度结果dvalues可能还会被其他层使用
        # 这样可以避免影响到其他层的梯度计算
        # ReLU的梯度计算需要根据输入值是否大于0来决定(也就是需要基于自身输入调整)
        self.dinputs = dvalues.copy()

        # ReLU的梯度规则: 当输入值<=0时, 梯度为0; 当输入值>0时, 梯度保持不变, 为1
        # 此处self.inputs就是ReLU层的输入值(in_relu), 要计算其梯度, ∂L/∂in_relu = ∂L/∂out_relu * ∂out_relu/∂in_relu
        # - 当in_relu <= 0, 也就是 self.inputs <= 0, 则 ∂out_relu/∂in_relu = 0, 因为ReLU函数在该区间的梯度为0 ——> 整体梯度 ∂L/∂in_relu = ∂L/∂out_relu * 0 = 0, 即把self.dinputs对应位置置0
        # - 当in_relu > 0, 也就是 self.inputs > 0, 则 ∂out_relu/∂in_relu = 1, 因为ReLU函数在该区间的梯度为1 ——> 整体梯度 ∂L/∂in_relu = ∂L/∂out_relu * 1 = ∂L/∂out_relu, 即self.dinputs保持不变
        # 因此我们需要将输入值<=0的位置的梯度置0, 不传递该位置的梯度
        self.dinputs[self.inputs <= 0] = 0

# 3. Softmax 激活函数 (用于预测)
class Activation_Softmax:
    """
    Description
    -----------
    softmax激活函数层, 用于多分类任务的输出层, 将线性输出转化为概率分布
    

    Notes
    -----
    - 1, 本类中未实现反向传播, 通常与交叉熵损失结合使用, 以简化反向传播计算    

    """

    def forward(self, inputs):
        """ 
        Description
        -----------
        前向传播, 计算Softmax激活函数的输出概率分布, 计算公式: softmax(xi) = exp(xi) / sum(exp(xj))

        Args
        ----
        inputs : np.ndarray
            输入数据, 上一层的输出, 全连接层的线性输出, 形状与全连接层输出一致, 简单理解为in_softmax;
            输出层全连接层的线性输出, 称为Logits, 形状为(样本数, 类别数)

        output : np.ndarray (在状态中保存)
            softmax激活函数的输出概率分布, 形状与输入一致, (样本数, 类别数);
            inputs是softmax层输入, output是softmax层输出
        """

        # 保存当前层的输入, 用于后续计算(反向传播时需要用到, 本类中未实现反向传播, 通常与交叉熵损失结合)
        self.inputs = inputs
        # step1: 为了数值稳定性, 减去每行(每个样本)的最大值, 防止指数函数溢出(np.exp过大可能返回inf)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # step2: 计算每行(每个样本)的指数和, softmax概率=每个样本的指数值/该样本所有指数值的和
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# 4. 通用 Loss 父类
class Loss:
    """
    Description
    -----------
    定义损失函数的统一接口, 所有具体损失函数类均继承自该父类

    """
    def calculate(self, output, y):
        """
        Description
        -----------
        计算损失的公共方法: 返回平均损失(数据损失)

        Args
        ----
        output : np.ndarray
            模型的预测输出, 形状为(样本数, 类别数)
        y : np.ndarray
            真实标签, 可为类别索引或独热编码, 对应形状为(样本数,)或(样本数, 类别数) (独热编码)

        """

        # 调用子类实现的forward方法, 计算每个样本的损失(样本损失)
        # 调用具体损失函数的前向传播方法, 计算每个样本的损失, forward方法在子类中具体实现, 父类中只是定义抽象接口
        sample_losses = self.forward(output, y)

        # 计算所有样本的平均损失(数据损失), 作为模型优化的目标
        data_loss = np.mean(sample_losses)
        return data_loss

# 5. 交叉熵损失 (含 Softmax)
class Loss_CategoricalCrossentropy(Loss):
    """       
    Description
    -----------
    交叉熵损失函数, 用于多分类任务, 与softmax配合使用

    Args
    ----
    继承自 Loss 父类, 实现具体的前向和反向传播方法
    """
    def forward(self, y_pred, y_true):
        """  
        Description
        -----------
        前向传播, 计算每个样本的交叉熵损失
        
        Args
        ----
        y_pred : np.ndarray
            模型的预测输出概率, 形状为(样本数, 类别数), softmax层的输出概率
        y_true : np.ndarray
            真实标签, 可为类别索引或独热编码, 对应形状为(样本数,)或(样本数, 类别数) (独热编码)
        """
        
        # 获取样本数量, 用于后续平均损失计算
        samples = len(y_pred)
        # 防止log(0)导致数值不稳定, 对预测概率进行裁剪
        # 裁剪范围在[1e-7, 1-1e-7]之间
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # 从模型输出的概率矩阵 y_pred_clipped 中，提取每个样本「真实类别对应的预测概率」
        # 用于后续计算交叉熵损失: -log(正确类别的预测概率)
        # 分两种情况处理真实标签: 类别索引或独热编码
        # case1: 真实标签为类别索引 (一维数组), 如 y_true = [0, 2, 1] 其shape为(3,)
        if len(y_true.shape) == 1:
            # 列表索引取出每个样本对应的正确类别的预测概率
            # 前面提到过, 列表索引index是按位置配对取元素
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # case2: 真实标签为独热编码 (二维数组), 如 y_true = [[1,0,0], [0,0,1], [0,1,0]]
        elif len(y_true.shape) == 2:
            # 通过逐元素相乘并沿类别轴求和, 获取每个样本对应的正确类别的预测概率
            # 注意 * 就是逐元素乘法, 就是线性代数中的Hadamard积, 哈达玛乘积(维度相同)
            # np.dot 或 @ 是矩阵乘法
            # 预测概率与独热编码逐元素乘法, 等价于取真实类别的概率(独热编码只有真实类别为1)
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # correct_confidences中保存了每个样本「真实类别对应的预测概率」, 形状为(samples,)
        # 计算交叉熵损失: -log(正确类别的预测概率)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        """   
        Description
        -----------
        反向传播, 计算损失对模型输出y_pred的梯度, 记为∂L/∂y_pred, 并将结果存入self.dinputs, 最终传递给前一层(通常是输出层的全连接层, 当然最后一层一般是softmax层), 
        用于后续参数(权重, 偏置)的梯度计算和更新
        
        Args
        ----
        dvalues : np.ndarray
            下一层传递过来的梯度, 损失对当前层输出的偏导;
            dvalues(当前层梯度起点) = ∂L(整体loss)/∂out(当前层输出, 也就是下一层输入);
            当然, 在此处独立的交叉熵损失层中, dvalues其实就是y_pred, 因为损失层是网络的最后一层, 没有更下游的层了, 所以dvalues就是损失对y_pred的偏导的起点;
            即损失计算的输入, 也就是模型的预测输出y_pred
        y_true : np.ndarray
            真实标签, 可为类别索引或独热编码, 对应形状为(样本数,)或(样本数, 类别数) (独热编码)

        Notes
        -----
        - 1, 公式推导的是「单个样本的梯度」，但代码中计算的是所有样本的平均梯度（因损失 L 通常取平均值），因此需要除以样本数
        - 2, 该层中的dvalues其实就是y_pred, dinputs就是损失对y_pred的梯度, 也就是∂L/∂y_pred
        """

        # 获取样本数和类别数
        # 注意此处dvalues就是y_pred(模型的预测输出), 因为交叉熵损失层是最后一层, 没有更下游的层了
        samples = len(dvalues)
        labels = len(dvalues[0])
        # 如果输入是类别索引, 则转换为独热编码形式
        if len(y_true.shape) == 1:
            # np.eys(label)生成单位矩阵,y_true作为索引取出对应行, 即独热编码
            # np.eye(labels)：生成 (类别数, 类别数) 的单位矩阵（对角线上为 1，其余为 0）, 每行对应1个类别的独热编码
            # [y_true]：用真实类别索引取单位矩阵的对应行，得到独热编码形式
            y_true = np.eye(labels)[y_true]

        # 交叉熵损失对y_pred的梯度公式: -真实标签/预测概率
        # 计算公式为 ∂L/∂y_pred = ∂(-sum(y_true * log(y_pred)))/∂y_pred 
        # 一般为了计算方便, log的底数取e, 其实就是ln, 求导就是1/x
        # 然后我们以样本的某一个类别为例子来推导:
        # L = -y_true  * log(y_pred), 对y_pred,c求导:
        # ∂L/∂y_pred,c = ∂(-sumk(y_true,k * log(y_pred,k)))/∂y_pred,c
        # 只有当k=c时, 导数不为0, 其他k≠c时导数为0, 所以:
        # ∂L/∂y_pred,c = -y_true,c/y_pred,c 
        # 推广到所有类别, 就是 ∂L/∂y_pred = -y_true / y_pred
        self.dinputs = -y_true / dvalues
        # 归一化梯度, 除以样本数, 保持梯度规模稳定(确保梯度规模与样本数量无关)
        self.dinputs = self.dinputs / samples

# 6. Softmax + Loss 组合类 (为了反向传播更稳定)
class Activation_Softmax_Loss_CategoricalCrossentropy():
    """  
    Description
    -----------
    将softmax激活函数与交叉熵损失结合, 优化反向传播稳定性;
    因为单独计算softmax和交叉熵的梯度会有数值不稳定问题, 组合后可简化梯度计算

    Notes
    -----
    - 1, 该类封装了softmax激活和交叉熵损失, 提供前向和反向传播方法, 具体调用类的实现细节见前面softmax和交叉熵损失类
    
    """
    def __init__(self):
        """
        Description
        -----------
        初始化组合类, 创建softmax激活和交叉熵损失实例
        """
        
        # 实现见前面
        # softmax激活层
        self.activation = Activation_Softmax()
        # 交叉熵损失层
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        """ 
        Description
        -----------
        前向传播, 计算softmax输出和交叉熵损失(先激活再计算损失)

        Args
        ----
        inputs : np.ndarray
            输入数据, 上一层的输出, 全连接层的线性输出, 形状与全连接层输出一致;
            输出层全连接层的线性输出, 称为Logits, 形状为(样本数, 类别数), 
            也就是in_softmax, softmax层的输入

        y_true : np.ndarray
            真实标签, 可为类别索引或独热编码, 对应形状为(样本数,)或(样本数, 类别数) (独热编码), 同交叉熵损失
        """

        # 对Logits做softmax激活, 得到概率分布
        # 也就是Activation_Softmax()类的forward方法
        self.activation.forward(inputs)

        # 保存激活后的输出(概率分布), 用于后续反向传播
        # 也就是softmax激活函数的输出概率分布, 形状与输入一致, (样本数, 类别数);
        # inputs是softmax层输入, output是softmax层输出
        self.output = self.activation.output

        # 计算平均交叉熵损失，调用交叉熵损失的calculate方法
        # 实际实现细节上调用的是Loss_CategoricalCrossentropy类的forward方法, 也就是将softmax层的输出作为输入计算损失
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        """
        Description
        -----------
        反向传播, 计算组合层的梯度, 直接计算简化后的梯度公式(也就是直接计算损失对Logits的梯度, 避免数值不稳定)
        Args
        ----
        dvalues : np.ndarray
            下一层传递过来的梯度, 损失对当前层输出的偏导;
            dvalues(当前层梯度起点) = ∂L(整体loss)/∂out(当前层输出, 也就是下一层输入);
            当然, 在此处softmax+交叉熵组合层中, dvalues其实就是y_pred, 因为组合层是网络的最后一层, 没有更下游的层了, 所以dvalues就是损失对y_pred的偏导的起点;
            即损失计算的输入, 也就是模型的预测输出y_pred
        y_true : np.ndarray
            真实标签, 可为类别索引或独热编码, 对应形状为(样本数,)或(样本数, 类别数) (独热编码)
        
        
        Notes
        -----
        - 1, 注意该组合层backward的目标是计算损失对Logits的梯度(也就是损失对全连接层输出的梯度), 以便传递给前一层(通常是输出层的全连接层), 用于更新参数;
        也就是计算 ∂L/∂Logits, 而不是单独计算softmax层或交叉熵损失层的梯度;
        """

        # 获取样本数量
        # 注意这里的dvalues就是y_pred(模型的预测输出), 因为softmax+交叉熵组合层是最后一层, 没有更下游的层了
        samples = len(dvalues)  
        if len(y_true.shape) == 2:
            # 若真实标签为独热编码, 则转换为类别索引(方便后续索引操作), 独热变索引
            y_true = np.argmax(y_true, axis=1)
        # 复制dvalues(此处dvalues是softmax的输出， 即概率分布), 因为y_pred可能有其他作用, 比如说计算准确率等, 不能直接修改
        self.dinputs = dvalues.copy()

        # 关键步骤：简化之后的梯度也就是softmax+交叉熵的联合梯度 = 预测概率 - 真实标签(独热编码形式)
        # 对每个样本的 "真实类别对应的概率" 减 1, 等价于将独热编码的 y_true 中 "1" 的位置减 1, "0" 的位置不变
        # 用的还是列表索引
        self.dinputs[range(samples), y_true] -= 1
        # 梯度归一化: 除以样本数, 保持梯度规模稳定(确保梯度规模与样本数量无关)
        self.dinputs = self.dinputs / samples

# 7. Adam 优化器
class Optimizer_Adam:
    """
    Description
    -----------
    Adam优化器类, 用于更新神经网络的权重和偏置参数, 结合动量(Momentum)和自适应学习率(RMSProp)的优化算法, 通过积累历史梯度信息动态调整参数更新策略，实现更快收敛和更稳定的训练;
    收敛快, 稳定, 适合大多数神经网络训练

    Adam的核心是维护两个关键变量:
    - 动量(momentum): 积累历史梯度的"方向", 缓解SGD在局部最优附近的震荡, 加速沿稳定方向的收敛;量纲是"梯度的累积", 梯度是损失对参数的偏导, 量纲是损失值/参数值
    - 自适应缓存(cache): 积累历史梯度的"幅度", 为每个参数动态调整学习率(梯度大的参数用小步长, 梯度小的参数用大步长);量纲是"梯度平方的累积", 量纲是(损失值/参数值)^2
    其中偏差修正: 初始阶段动量和缓存接近0, 通过修正项使其更接近真实值, 避免初期更新过慢

    注意, 优化器维护的动量和缓存只是辅助变量, 与此处model的权重/偏置核心参数不同;
    辅助变量的初始化≠核心参数的初始化

    """
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        """  
        Description
        -----------
        初始化Adam优化器的超参数

        Args
        ----
        learning_rate : float
            初始学习率, 控制参数更新的步长大小, 默认0.001
        decay : float
            学习率衰减率, 控制学习率随迭代次数逐渐减小, 默认0.0(不衰减)
        epsilon : float
            防止除0错误的小常数, 用于数值稳定性, 默认1e-7
        beta_1 : float
            动量项的指数衰减率(控制动量的历史贡献， 默认0.9)
        beta_2 : float
            自适应学习率项的指数衰减率(控制平方梯度的历史贡献， 默认0.999)
        """
        
        # 初始学习率, 即初始步长, 控制参数更新的幅度
        self.learning_rate = learning_rate
        # 当前学习率(可能会随迭代次数逐渐衰减), 实际用于更新的学习率
        self.current_learning_rate = learning_rate
        # 学习率衰减率(控制实际用于更新的学习率随迭代次数减小, 避免后期震荡), 默认0不衰减
        self.decay = decay
        # 迭代次数(用于学习率衰减计算和偏差修正)
        self.iterations = 0
        # 防止除0错误的小常数, 数值稳定项
        self.epsilon = epsilon
        # 动量项系数, 控制动量的历史贡献; 即动量项的指数衰减率, 控制历史动量的"记忆比例"(β1越大, 记忆越久)
        self.beta_1 = beta_1
        # 自适应学习率项系数, 控制平方梯度的历史贡献; 即自适应学习率项/缓存项的指数衰减率, 控制历史平方梯度的"记忆比例"(β2越大, 记忆越久)
        self.beta_2 = beta_2
    
    def pre_update_params(self):
        """
        Description
        -----------
        在更新参数前调用, 用于调整当前学习率(如果设置了衰减);
        在每次参数更新之前, 根绝迭代次数调整当前学习率(current_learning_rate), 仅当decay>0时生效
        """
        if self.decay:
            # 若启动学习率衰减, 则根据迭代次数调整当前学习率/按公式更新当前学习率
            # 衰减公式: lr = initial_lr / (1 + decay * iterations)
            # 训练前期, 迭代次数小, 当前学习率接近初始值, 大步长快速收敛; 训练后期, 迭代次数大, 当前学习率减小, 小步长精细调整参数, 避免在最优解附近震荡
            # 目的: 训练前期使用较大学习率快速收敛, 后期使用较小学习率精细调整
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        """ 
        Description
        ---------
        接收1个全连接层(Layer_Dense示例), 利用该层的梯度(dweights/dbiases)更新其相应参数(权重和偏置)
        
        Args
        ----
        layer : Layer_Dense
            需要更新参数的层实例, 该层必须包含dweights和dbiases属性(梯度)
        """
        
        # 如果该层没有weight_cache属性(没有历史信息, 说明是首次更新), 则初始化动量和缓存数组(与参数形状一致, 因为动量公式中每一轮迭代中的动量本质是历史梯度的一个加权平均, 所以形状要和梯度一致, 而参数的梯度形状和参数本身一致)
        # 首次更新某层时，没有历史梯度数据, 所以创建与权重 / 偏置形状完全一致的全 0 数组，用于存储历史动量（weight_momentums）和历史平方梯度（weight_cache）
        if not hasattr(layer, 'weight_cache'):

            # layer.weights 是该层的权重参数
            # layer.weight_momentums: 是该层的权重动量, 量纲是"梯度的累积"
            # layer.weight_cache: 是该层的权重缓存, 量纲是"梯度平方的累积"

            # 动量数组(momentums): 存储历史梯度的指数移动平均(用于动量更新)
            layer.weight_momentums = np.zeros_like(layer.weights)
            # np.zeros_like: 创建与layer.weights形状相同的全0数组
            layer.weight_cache = np.zeros_like(layer.weights)

            # 缓存数组(cache): 存储历史平方梯度的指数移动平均(用于自适应学习率)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # 1, 更新动量数组(权重与偏置) ——> ⚠️ 动量法体现
        # 公式: momentum = beta_1 * previous_momentum + (1 - beta_1) * current_gradient
        # mt=β1⋅mt-1+(1-β1)⋅∇Wt  当前迭代的动量=β1乘以前一迭代的动量+（1-β1）乘以当前迭代的权重梯度
        # 作用: 积累历史梯度方向, 加速收敛(如沿同一方向则步长变大)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        # 2, 动量偏差修正: 初始迭代时momentum接近0, 需修正为更精确的估计 ——> 矫正加权和为1
        # 公式: corrected_momentum = momentum / (1 - beta_1^(iterations + 1))
        # 原因: beta1接近1, 迭代初期1 = beta1^t 较小, 修正后momentum更接近真实值
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        # 3, 更新缓存数组(权重与偏置) ——> ⚠️ 自适应学习率体现(也就是RMSProp)
        # 公式: cache = beta_2 * previous_cache + (1 - beta_2) * current_gradient^2
        # 作用: 记录梯度平方的历史, 用于自适应调整学习率(梯度大则步长小, 反之则步长大)
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        # 4, 缓存偏差修正: 同动量修正, 解决初始迭代时cache接近0的问题 ——> 矫正加权和为1 不能
        # 公式: corrected_cache = cache / (1 - beta_2^(iterations + 1))
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        # 5, 最终参数更新: 结合动量修正和自适应学习率
        # 公式: 参数 = 参数 - 学习率*修正动量/(sqrt(修正缓存) + 小常数)
        # 修正动量: 提供梯度的方向和累计效应(解决SGD震荡问题)
        # 修正缓存: 调整学习率以适应不同参数的梯度规模(自适应调整每个参数的学习率, 梯度大则步长小, 避免震荡)
        # 小常数: 防止除0错误, 保持数值稳定
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
        # 迭代次数更新: 每次参数更新后迭代次数+1, 用于下一次衰减和偏差修正
        self.iterations += 1

# =============================================================================
# 第二部分：通用深度网络封装 (UniversalDeepModel)
# =============================================================================

class UniversalDeepModel:
    """
    Description
    -----------
    这是一个通用的、支持任意深度的全连接神经网络封装类; 它自动管理层的创建、前向传播、反向传播和参数更新.
    本质是组装工具类, 根据后续用户配置动态搭建网络(隐藏层数量/神经元数, 输入输出维度), 并自动协调"前向传播-损失计算-反向传播-参数更新"的流程.

    Notes
    -----
    - 1, 封装目的是为了降低使用复杂度, 用户只需指定网络结构和训练参数(数据和网络结构), 无需手动管理每一层和训练细节(也就是无需关注梯度计算/层间维度匹配等细节)
    """
    def __init__(self, input_dim, hidden_layer_sizes, output_dim, learning_rate=0.001, decay=0.):
        """
        Description
        -----------
            初始化模型架构, 动态创建隐藏层和输出层, 并设置优化器和损失函数.
            只是设计网络结构, 并未进行训练, 也就是层的初始化和连接.
        
        Args
        ----
            input_dim (int): 输入数据的特征数量 (例如 2, 784), 主要是每一层的输入维度, 可以视为上一层的输出维度/神经元数量, 如 28x28 图像展平后为 784
            hidden_layer_sizes (list of int): 一个列表，定义隐藏层的结构
                例如 [64] 表示 1 个隐藏层，有 64 个神经元
                例如 [128, 64] 表示 2 个隐藏层，第一层 128, 第二层 64
            output_dim (int): 输出类别的数量, 例如 10 表示有 10 个类别 (0-9)
            learning_rate (float): 初始学习率, 控制参数更新步长, 默认0.001
            decay (float): 学习率衰减率, 防止训练后期震荡, 默认0.0(不衰减)
        """
        self.layers = [] # 用于存储所有的网络层 (Dense 和 ReLU), 仅存隐藏层, 输出层单独处理, 因输出层激活函数与隐藏层不同
        self.optimizer = Optimizer_Adam(learning_rate=learning_rate, decay=decay)
        
        # --- 1. 动态构建隐藏层 ---
        # current_input_dim 记录当前层的输入维度，初始为数据的输入维度
        current_input_dim = input_dim 
        
        for i, n_neurons in enumerate(hidden_layer_sizes):
            # 创建全连接层：输入维度current_input_dim -> 当前隐藏层神经元数n_neurons
            # 隐藏层的固定结构：全连接层（线性变换）+ ReLU 层（非线性激活）
            # 示例, 若hidden_layer_sizes=[64,32], 则self.layers会依次添加: Dense(input_dim->64)->ReLU->Dense(64->32)->ReLU
            dense_layer = Layer_Dense(current_input_dim, n_neurons)
            self.layers.append(dense_layer)
            
            # 创建激活函数层：每个全连接层后通常接一个 ReLU, 引入非线性
            activation_layer = Activation_ReLU()
            self.layers.append(activation_layer)
            
            # 更新下一层的输入维度为当前层的神经元数
            current_input_dim = n_neurons
            
            print(f"构建层 {i+1}: Dense({dense_layer.weights.shape[0]}->{n_neurons}) + ReLU")

        # --- 2. 构建输出层 ---
        # 最后一层全连接：最后一个隐藏层神经元数 -> 输出类别数
        # 注意：最后一层通常不接 ReLU，而是接 Softmax (包含在 loss_activation 中)
        # 单独存储输出层全连接层（因后续反向传播需优先处理输出层梯度）
        self.final_dense = Layer_Dense(current_input_dim, output_dim)
        print(f"构建输出层: Dense({current_input_dim}->{output_dim}) + Softmax")
        
        # --- 3. 定义损失函数和输出激活 ---
        # 使用 Softmax + CrossEntropy 的组合类
        # 输出层激活+损失：Softmax（概率化）+ 交叉熵（损失计算）组合类
        self.loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

    def forward(self, X):
        """
        Description
        -----------
            前向传播：数据从输入流向输出, 只是计算输出, 不进行训练.
            前向传播是 “数据流转阶段”:输入数据从隐藏层逐层传递,最终经过输出层全连接层,得到线性输出(Logits)(Softmax 在损失计算时才执行);
            层间传递：每个层的 forward 方法会更新自身的 self.output, 并作为下一层的输入, 无需用户手动处理维度匹配(底层 Layer_Dense 已保证维度正确)
        
        Args
        ----
            X (np.ndarray): 输入数据, 形状为(样本数, 特征数)
        """
        # 1. 数据先流经所有隐藏层
        current_output = X # 初始输入为原始数据
        for layer in self.layers:
            layer.forward(current_output) # 调用层的前向传播（Dense算线性输出，ReLU做激活）
            current_output = layer.output # 将当前层的输出作为下一层的输入
            
        # 2. 流经最后一个全连接层
        self.final_dense.forward(current_output)
        
        # 3. 返回最终层的输出 (Logits)，注意此时还没过 Softmax
        # Softmax 激活被封装在 loss_activation 中，会在计算损失时自动执行（避免重复计算）
        return self.final_dense.output

    def train(self, X, y, epochs=1000, print_every=100):
        """
        Description
        -----------
            训练循环, 是模型的 “迭代优化阶段”，整合了 “前向传播→损失计算→梯度反向传播→参数更新” 的全流程，是训练模型的入口.

        Args
        ----
            X (np.ndarray): 输入数据, 形状为(样本数, 特征数)
            y (np.ndarray): 真实标签, 可为类别索引或独热编码, 形状为(样本数,)或(样本数, 类别数) (独热编码)
            epochs (int): 训练轮数, 控制训练的迭代次数, 默认1000
            print_every (int): 每多少轮打印一次训练状态, 默认100

        """

        # 初始化历史记录字典 (用于可视化)
        self.history = {
            'loss': [], 'acc': [], 
            'lr': [], 'grad_norm_first': [],
            'grad_norm_last': [], 'grad_norm_mean': []
        }


        for epoch in range(epochs):
            # ====================
            # A. 前向传播 (Forward)
            # ====================
            
            # 1. 计算网络主体输出, 线性输出（Logits）
            final_outputs = self.forward(X)
            
            # 2. 计算损失 (同时做 Softmax)
            # self.loss_activation.forward 接收Logits，先做Softmax得到概率，再算交叉熵损失
            # forward 返回的是 loss 值
            loss = self.loss_activation.forward(final_outputs, y)

            # ====================
            # B. 打印状态 (Logging)
            # ====================
            if not epoch % print_every:
                # 1. 计算预测类别（从概率分布取最大值索引）
                predictions = np.argmax(self.loss_activation.output, axis=1)
                # 2. 处理标签（若为独热编码，转成类别索引）
                if len(y.shape) == 2:
                    y_labels = np.argmax(y, axis=1)
                else:
                    y_labels = y
                # 3. 计算准确率（预测正确的样本数 / 总样本数）
                accuracy = np.mean(predictions == y_labels)
                
                # 4. 打印当前轮数, 准确率, 损失, 学习率, 梯度范数⚠️
                # 对于梯度范数, 此处收集所有含有权重的层(隐藏层, 输出层)
                # 注意: self.layers里面混杂了Dense和ReLU层, 只有Dense层有dweights属性, 我们可以由此过滤出含权重的全连接层
                all_dense_layers = [layer for layer in self.layers if hasattr(layer, "dweights")]
                all_dense_layers.append(self.final_dense)  # 加上输出层, 都是layer_dense实例

                # 收集梯度范数
                grad_norms = []
                for layer in all_dense_layers:
                    # 注意：在第0轮训练还没做backward时，dweights可能不存在，做个保护
                    if hasattr(layer, 'dweights'):
                        norm = np.linalg.norm(layer.dweights)
                        grad_norms.append(norm)
                    else:
                        grad_norms.append(0.0)

                # 保存历史记录
                self.history['loss'].append(loss)
                self.history['acc'].append(accuracy)
                self.history['lr'].append(self.optimizer.current_learning_rate)
                if grad_norms:
                    self.history['grad_norm_first'].append(grad_norms[0])
                    self.history['grad_norm_last'].append(grad_norms[-1])
                    self.history['grad_norm_mean'].append(np.mean(grad_norms))

                # 打印信息
                # grad_norms[0] -> 第一层 (最靠近输入，最容易梯度消失)
                # grad_norms[-1] -> 最后一层 (最靠近输出，最容易梯度爆炸)
                if grad_norms:
                    print(f"Epoch {epoch}, " +
                          f"Acc {accuracy:.4f}, " +
                          f"Loss {loss:.4f}, " +
                          f"LR {self.optimizer.current_learning_rate:.6f} | " +
                        f"Grad Norm First: {grad_norms[0]:.4e}, " + # 监控底层是否学得动
                        f"Last: {grad_norms[-1]:.4e}, " +           # 监控源头信号强不强
                        f"Mean: {np.mean(grad_norms):.4e}")          # 监控整体

            # ====================
            # C. 反向传播 (Backward)
            # 反向传播是 “梯度回流阶段”，核心是根据损失计算所有参数（权重、偏置）的梯度，遵循 “从输出层→隐藏层→输入层” 的顺序（梯度链式法则）
            # ====================
            
            # 1. 从 Loss 开始反向传播
            # 1. 从损失层开始反向传播（计算对Logits的梯度）
            # self.loss_activation.backward 直接返回损失对输出层全连接层输出（Logits）的梯度
            self.loss_activation.backward(self.loss_activation.output, y)
            
            # 2. 反向传播经过输出层
            # 输入是 loss 层的梯度 (dinputs)
            # 2. 输出层全连接层反向传播（计算对输出层权重/偏置的梯度，及对隐藏层输出的梯度）
            self.final_dense.backward(self.loss_activation.dinputs)
            
            # 3. 反向传播经过所有隐藏层 (需要倒序遍历！因为梯度从后往前传)
            # 这里的梯度链是：上一层的 dinputs -> 当前层的 backward
            # 初始梯度：输出层对隐藏层输出的梯度
            back_gradient = self.final_dense.dinputs
            
            for layer in reversed(self.layers):
                layer.backward(back_gradient) # 调用层的反向传播（计算当前层梯度）
                back_gradient = layer.dinputs # 更新梯度：当前层对前一层输入的梯度 → 传给前一层

            # ====================
            # D. 参数更新 (Optimize)
            # ====================
            
            # 1. 预更新：处理学习率衰减（若开启）
            self.optimizer.pre_update_params()
            
            # 2. 更新隐藏层的参数（仅全连接层有参数，ReLU无参数）
            for layer in self.layers:
                # 只有 Layer_Dense 有参数(weights/biases)，ReLU 没有
                if hasattr(layer, 'weights'):
                    self.optimizer.update_params(layer)
            
            # 3. 更新输出层的参数
            self.optimizer.update_params(self.final_dense)
            
            # 4. 后更新：迭代次数+1（用于下一轮衰减计算和Adam的偏差修正）
            self.optimizer.post_update_params()

    def predict(self, X):
        """
        Description
        -----------
        使用训练好的模型进行预测, 返回类别概率分布.
        训练完成后，通过 predict 方法对新数据进行预测，输出类别概率分布（方便用户判断预测置信度）
        预测函数：输入数据，输出概率分布
        推理流程：新数据→前向传播(Logits)→Softmax(概率)
        

        Args
        ----
        X : np.ndarray
            输入数据, 形状为(样本数, 特征数)

        Notes
        -----
        - 1, 输出解读：例如输出 [[0.05, 0.9, 0.05]] 表示样本属于第 2 类的概率为 90%，可通过 np.argmax(probs, axis=1) 得到最终预测类别
        """
        # 1. 前向传播得到Logits（线性输出）
        logits = self.forward(X)
        # 2. 对Logits执行Softmax，转为概率分布
        self.loss_activation.activation.forward(logits)
        # 3. 返回概率分布（形状：(样本数, 类别数)）
        return self.loss_activation.activation.output


# 定义一个可视化函数
# --- 可视化函数 ---
def plot_training_history(history):
    epochs = range(len(history['loss']))
    
    plt.figure(figsize=(15, 5))
    
    # 图1: Loss 曲线
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['loss'])
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()
    
    # 图2: Accuracy 曲线
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['acc'])
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.legend()
    
    # 图3: Learning Rate 曲线
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['lr'])
    plt.title('Learning Rate Decay')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    # 图4: First Layer Gradient Norm 曲线 (诊断用)
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['grad_norm_first'], color='green')
    plt.title('First Layer Gradient Norm (Stability Check)')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')

    # 图5: Last Layer Gradient Norm 曲线 (诊断用)
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['grad_norm_last'], color='red')
    plt.title('Last Layer Gradient Norm (Stability Check)')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')

    # 图6: Mean Gradient Norm 曲线 (诊断用)
    plt.subplot(2, 3, 6)
    plt.plot(epochs, history['grad_norm_mean'])
    plt.title('Mean Gradient Norm (Stability Check)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Norm')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 第三部分：【用户配置区】 (万金油模板接口)
# =============================================================================

# --- 1. 数据准备 (Data Preparation) ---
# 这里是唯一需要你根据实际任务修改数据加载逻辑的地方
# 示例：生成 300 个样本，2 个特征，3 分类
print("正在生成数据...")
N_SAMPLES = 300
INPUT_FEATURES = 2
NUM_CLASSES = 3
X_data = np.random.randn(N_SAMPLES, INPUT_FEATURES) # 你的输入数据, 随机生成标准正态分布数据, 形状 (300, 2)
y_data = np.random.randint(0, NUM_CLASSES, size=(N_SAMPLES,)) # 你的标签, 随机生成 0,1,2 三类标签, 形状 (300,)

# --- 2. 模型配置 (Model Configuration) ---
# 只要修改这里，就能改变网络的深度和宽度
# 场景 A: 简单网络 -> hidden_layers = [64]
# 场景 B: 深层网络 -> hidden_layers = [128, 128, 64]
MY_HIDDEN_LAYERS = [64, 64]  # 2个隐藏层，每层64个神经元

model = UniversalDeepModel(
    input_dim=INPUT_FEATURES,       # 自动适配输入数据
    hidden_layer_sizes=MY_HIDDEN_LAYERS, # 在这里定义你有多少层，每层多大
    output_dim=NUM_CLASSES,         # 自动适配输出类别
    learning_rate=0.05,             # 学习率
    decay=1e-4                      # 学习率衰减 (防止后期震荡)
)

# --- 3. 训练 (Training) ---
print("\n开始训练...")
model.train(
    X_data, 
    y_data, 
    epochs=2000,    # 训练轮数
    print_every=1 # 每多少轮打印一次
)

# 可视化训练过程
plot_training_history(model.history)

# --- 4. 验证/使用 (Inference) ---
print("\n模型使用示例:")
# 假设来了一条新数据
new_sample = np.array([[0.5, -1.2]]) # 新样本, 形状 (1, 2)
probs = model.predict(new_sample) # 预测类别概率分布, 形状 (1, 3)
pred_class = np.argmax(probs, axis=1) # 预测类别索引

print(f"输入: {new_sample}")
print(f"各类别概率: {probs}")
print(f"预测类别: {pred_class}")