# PyTorch 2.3笔记

- ### 默认float32, int64

- ### PyTorch中常用乘法：
  
  |                      | 对位乘 | 矩阵乘 | 内积  |
  |:--------------------:|:---:|:---:|:---:|
  | r1 * r2              | √   |     |     |
  | torch.mul(r1, r2)    | √   |     |     |
  | r1 @ r2              |     | √   | √   |
  | torch.matmul(r1, r2) |     | √   | √   |
  | torch.dot(r1, r2)    |     |     | √   |
  
  ### 稍微注意一下torch.rand(n)和torch.rand(1,n) or torch.rand(n,1)是不一样的，前者才是一维vector

- ### xxx.shape vs xxx.size()，注意括号

- ### 一直忽视了一个很基础但重要的点：在PyTorch中，torch.nn.Module类重载了\_\_call\_\_方法，而\_\_call\_\_会调用forward方法。
  
  ### e.g.，这种典型代码：
  
  ```python
  class WtfNet(nn.Module):
      def __init__(self):
          ...
          self.conv1 = nn.Conv2d(in, out, kernel)
          ...
  
      def forward(self, x):
          ...
          x = self.conv1(x)
          ...
          return x
  
  input = ... 
  model = WtfNet()
  output = model(input)
  ```

### 具体的运作流程是这样的：

### ① WtfNet类继承nn.Module类并实例化一个对象model;

### ② model(input)触发WtfNet类继承自nn.Module类的__call__方法，即调用WtfNet类的forward方法；

### ③ 类似地，self.conv1是nn.Conv2d类（同样继承自nn.Module）的实例。self.conv1(x)触发__call__方法，再进一步调用nn.Conv2d的forward方法，最终是调用torch.nn.functional.conv2d()进行卷积计算。

### 所以其实可以不在构造函数中定义网络层，而是在forward中直接调用F.xxx()进行计算，效果一样，效率应该还高一些。不过代码可能会丑一些

### 再提一句，要实例化的对象才有这种机制，因此不能直接写成 x = nn.Conv2d(...)
<br>

- ### PyCharm调试小技巧：对想要查看的变量，右键->对表达式求值，然后用np.array(xxx)把想要查看的xxx括起来，求值，最后点击“作为Array查看”

<br>

- ### 总结PyTorch的正向、反向传播中tensor的一些重要属性：
  
  ### ① requires_grad：默认为false（但torch.nn网络参数的requires_grad=True）。正向传播过程中，只要参与计算的tensor有requires_grad=True的，那么后续结果都为True（有传递性）。
  
  ### ② is_leaf：计算图中的叶子节点是那些在前向传播过程中被用户直接创建的tensor（包括各种网络参数），非叶子节点是由叶子节点通过各种操作创建的中间张量。
  
  ### ③ grad: 在反向传播中记录梯度。在反向传播过程中对grad的计算和保留如下：
  
  |                     | is_leaf=True | is_leaf=False |
  |:-------------------:|:------------:|:-------------:|
  | requires_grad=True  | 计算grad并保留    | 计算grad但不保留    |
  | requires_grad=False | 不计算grad      | 不计算grad       |
  
  ### ④ grad_fn: 对于requires_grad=True的tensor，正向传播时会在grad_fn指向创建该tensor的Function对象（各种{xxxBackWard0}）。对于叶子节点，grad_fn=None。

- ### PyTorch中的AutoGrad: 每个Function对象都实现了其"backward"方法，据此可以计算输入tensor的梯度。如果需要自定义操作，可以通过继承torch.autograd.Function类并实现forward和backward方法。

- ### 对于像 torch.abs 这样的函数，在输入为 0 时确实是不可导的。PyTorch 处理这种情况的方式是定义一个子梯度（subgradient），以确保梯度计算过程不会出现错误。允许在不可导点选取一个合理的梯度值。在实践中，对于大多数自动微分框架，包括 PyTorch，在 torch.abs 的 0 点会返回一个默认的梯度值。在 torch.abs 的实现中，PyTorch 返回的梯度在输入为 0 时是 0

<br>

- ### 关于画图的一些问题：
  - ### PIL图像像素值范围是整数[0, 255]；
  - ### transforms.ToTensor()会将PIL图像映射为[0.0, 1.0]并调整通道顺序为[C, H, W]；如果是numpy数组则只映射范围[0.0, 1.0]。类型都会转为浮点；
  - ### plt绘制的对象是numpy数组，数值范围要求整数[0, 255]，浮点数[0.0, 1.0]，且数据维度是(H, W, C)。
  
  ### 综上，如果要对经过了预处理的tensor进行绘制，一般需要经过这些步骤：
  - ### 反标准化、转为numpy数组(x = x.numpy())、调整通道顺序

<br>

- ### nn.CrossEntropy自带softmax，网络输出不要再额外添加了
<br>

- ### 加载部分权重：以ViT为例，假设num_classes不是1000，相应地要去掉最后Head的权重：
  ```python
  # 创建模型
  model = create_vit_model()

  # 加载部分权重
  checkpoint = torch.load(r'vit_base_patch16_224.pth')
  checkpoint.pop('head.weight', None)  # 具体网络层的名字可以点进调试里面看，隐藏了就展开
  checkpoint.pop('head.bias', None)

  model_state_dict = model.state_dict()  # 读取模型状态到字典
  model_state_dict.update(checkpoint)  # 更新字典
  model.load_state_dict(model_state_dict)  # 根据字典加载权重
  ```
- ### 用x = torch.flatten(x, start_dim=1)保留batch_size进行展平，进行后续的fc等操作
- ### 卷积输出计算：
  $$
  H, W_{out} = \frac{H, W_{in} + 2 \times padding - dilation \times (kernel - 1) - 1}{stride} + 1
  $$
  ### 不考虑空洞卷积则简化为：
  $$
  H, W_{out} = \frac{H, W_{in} + 2 \times padding - kernel}{stride} + 1
  $$
  ### 默认：padding=0, dilation=1, stride=1
  ### 常见情况：
  - ### 3x3卷积不填充，尺寸减小2；
  - ### 3x3卷积填充，尺寸不变；
  
<br>

- ### model.train(), model.eval(), torch.no_grad()：
  - ### 训练模式：启用Dropout；BN会使用当前批次的running_mean和running_std处理当前样本，同时会不断更新已有样本的mean和std。
  - ### 验证模式：禁用Dropout；BN使用已计算的mean和std处理当前样本。
  - ### no_grad模式：停止Autograd，减小开销和加快推理速度。
    ### 关于第3点多解释一下：① 首先Autograd≠backward，Autograd大概是指的整个前向和反向传播的机制； ② no_grad模式下pytorch仍然会构建计算图，但会减少占用。具体而言，虽然前向传播本身就不计算梯度，但pytorch仍会分配为其内存，启用no_grad后则不再分配；此外还可以减少计算图结点所需缓冲区等等（“具体而言”的内容不是很确定，来自网络和GPT。但结论肯定是会减少开销没错，因此在推理中要开启no_grad）。

    