# 从一次训练迭代到 64 卡扩展：分布式训练与显存优化复习笔记

准备大模型训练与推理优化面试时，我在这一模块最需要厘清的是三个区别：保存了什么，不等于计算了什么；合并梯度贡献，不一定要除以卡数；模型状态能放下，不等于整个训练过程能放下。这篇笔记把基础概念、贯穿案例、答题中的误区和主动追问整理在一起，保留本轮学习的 27 张互动图，并为自测题附上答案。

内容来自本模块的互动复习，也衔接了相邻会话《1-模型基础与计算量：P0，先建立后续优化的基础》和《2-推理引擎与服务优化：P0，优先深入》中的张量轴、归一化、因果 Attention、重计算和性能评测概念。以下是原理复习，不是声称所有问题都已在同岗位面经中出现，也不是实际多 GPU 性能实验。图中显存、时间与吞吐数字均为教学数据。

## 1. 用一个训练任务贯穿知识链

假设我要训练一个 7B 参数的 Decoder-only Transformer，采用 Adam，目标是先在 8 张 GPU 上运行，再分析扩到 64 张 GPU 的效率。先约定一套便于手算的模型状态配置：

| 项目 | 每参数字节数 | 7B 模型总量 |
|---|---:|---:|
| 低精度模型参数 | 2 | 14 GB |
| 低精度梯度 | 2 | 14 GB |
| FP32 主参数 | 4 | 28 GB |
| Adam 一阶矩 m | 4 | 28 GB |
| Adam 二阶矩 v | 4 | 28 GB |
| 合计 | 16 | 112 GB |

这里 P=7×10⁹，GB 按 10⁹ 字节计算。把主参数、m、v 合称“优化器相关状态”，占 12P 字节。**16 字节/参数是本文的题设，不是所有 PyTorch 或 BF16 训练的默认布局。** 实际还应确认参数与梯度 dtype、优化器实现、是否存在主参数、状态压缩和 offload。

训练任务沿着下面的顺序推进：

```text
拿到数据 → 前向 → 计算 loss → 反向生成梯度
        → 合并各数据分片的贡献 → 完成梯度累积与必要处理
        → 优化器更新 → 进入下一轮
```

任务中依次遇到的问题是：112 GB 状态放不下；减小 batch 后仍放不下；DDP 并没有把显存池合起来；ZeRO/FSDP 可以分片存储，但计算时仍需参数；单层矩阵太大时再考虑 TP；整段层堆叠可用 PP；长序列带来激活和 Attention 压力，需要 checkpoint、SP、CP；数值计算需要正确精度与归一化；最后，扩卡收益必须从有效吞吐与关键路径验证。

这些技术可以组合，但不是在同一个维度重复“除以 GPU 总数”。后文会分别说明每一种 GPU 组分担的工作。

## 2. 一次训练迭代：梯度计算与参数更新分开

### 2.1 backward 不会自动更新参数

前向用当前参数计算输出与 loss。`loss.backward()` 通过链式法则计算梯度，并累积到参数的 `.grad`；`optimizer.step()` 再使用梯度与优化器状态更新参数。`zero_grad()` 清理之前的梯度，避免不需要的跨迭代累积。[PyTorch 自动微分教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)、[清理梯度教程](https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)

因此“已经反向、还没 step”的时刻，参数尚未改变，梯度已经生成。若连续执行两次 backward 而不清理，得到的是两次梯度之和，不是后一次覆盖前一次。

<iframe src="/content/interactive/distributed-training/training-step.html?v=height-v2" title="训练迭代：前向、反向、更新与清理" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="training-step"></iframe>

[单独打开此图](/content/interactive/distributed-training/training-step.html?v=height-v2)

### 2.2 梯度累积为什么通常能模拟更大 batch

把一个大 batch 分成 k 个等大小 micro-batch。若每个 micro-batch 的 loss 都是均值，目标是整个大 batch 的均值，则应累积各个梯度的 1/k：可以先把各 loss 除以 k，再 backward；完成 k 次后只更新一次。

参数梯度的形状由参数形状决定。一个含 100 万可训练标量的模型，普通稠密训练中对应 100 万梯度元素；micro-batch 从 8 降到 1，并不会变成 12.5 万梯度元素。累积多次是在同一份梯度位置上加值。

纯 DP 下，若按样本计数：

$$B_{global}=B_{micro,per\ rank}\times k_{accum}\times N_{DP}.$$

TP 或 PP 内部合作处理同一批数据，不能把它们的卡数再当作独立样本份数乘进去。变长语言模型任务还应另行记录实际有效 token 数。

“梯度累积等价于大 batch”有条件：loss 权重正确、累积期间参数固定、优化器只更新一次、非线性梯度处理放在合适位置。BatchNorm 等依赖 batch 统计的操作、不同 Dropout 随机轨迹以及浮点求和次序，都可能让实际结果不能逐位一致。

## 3. 显存拆账：减小 micro-batch 减少什么

### 3.1 持久状态、激活与临时空间

模型参数、参数梯度与优化器状态通常形成较大的持续占用。激活是前向产生、反向需要的中间结果；临时空间还包括算子 workspace、参数聚合、通信 buffer、输入、框架与分配器开销。

实际峰值应按时间取最大值：

$$M_{peak}=\max_t\left[M_{state}(t)+M_{activation}(t)+M_{gather}(t)+M_{workspace}(t)+M_{other}(t)\right].$$

不要把不同时间出现的每一项独立峰值机械相加，也不要只计算持久状态就宣布能训练。allocated、reserved 与设备可用显存也是不同的观察口径。

<iframe src="/content/interactive/distributed-training/adam-memory.html?v=height-v2" title="Adam 训练模型状态的显存组成" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="adam-memory"></iframe>

[单独打开此图](/content/interactive/distributed-training/adam-memory.html?v=height-v2)

### 3.2 micro-batch 不是模型状态的缩放系数

主例的参数仍是 14 GB，梯度仍是 14 GB，优化器相关状态仍是 84 GB。减少 micro-batch 主要减少同时保存的激活和部分输入、临时张量，不能把梯度与优化器状态也按 batch 比例缩小。激活也未必精确线性缩放：序列长度、Attention 实现、padding、保存策略和算子 workspace 都会影响结果。

<iframe src="/content/interactive/distributed-training/microbatch-memory.html?v=height-v2" title="micro-batch 改变激活但不改变参数状态形状" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="microbatch-memory"></iframe>

[单独打开此图](/content/interactive/distributed-training/microbatch-memory.html?v=height-v2)

## 4. DP 与 DDP：数据不同，模型状态仍复制

数据并行让各 rank 使用相同的完整模型处理不同样本。DDP 在反向中同步梯度；参数以相同初始值开始，再使用一致的梯度更新，保持模型副本一致。DP 是策略名；PyTorch 的 `DataParallel` 与 `DistributedDataParallel` 是不同实现，面试中不要把策略与类名混为一谈。

当各 rank 的本地 loss 都是等大小 batch 的均值时，跨卡平均本地梯度对应全局均值目标。若不同 rank 有不同数量的有效 token，则需要按目标定义正确加权，不能默认等权平均一定正确。

DDP 不是把 8 张 24 GB GPU 合成一张 192 GB GPU。本文配置下，每卡仍要容纳约 112 GB 模型状态。它提高数据处理吞吐，但没有消除模型副本的冗余。DDP 通常按 bucket 同步，梯度桶就绪后可以启动 AllReduce，与之后的反向计算重叠。[DDP 设计说明](https://docs.pytorch.org/docs/stable/notes/ddp.html)

<iframe src="/content/interactive/distributed-training/ddp-gradient-sync.html?v=height-v2" title="DDP 本地样本与跨卡梯度同步" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="ddp-gradient-sync"></iframe>

[单独打开此图](/content/interactive/distributed-training/ddp-gradient-sync.html?v=height-v2)

## 5. 三种集合通信：先认清张量位置

| 操作 | 合并的对象 | 每卡最后得到什么 |
|---|---|---|
| AllGather | 各卡不同位置的分片 | 拼接后的完整张量 |
| AllReduce | 各卡相同位置的数值 | 完整的归约结果 |
| ReduceScatter | 相同位置先归约，再分配位置 | 归约结果的一部分 |

例如 GPU 0 梯度 `[2,4]`，GPU 1 梯度 `[6,8]`，两列分别对应两个参数。跨卡平均是 `[(2+6)/2,(4+8)/2]=[4,6]`。ReduceScatter 后，GPU 0 可以只保留 4，GPU 1 只保留 6。

把 GPU 0 内部的 `(2+4)/2` 算成 3，把 GPU 1 内部的 `(6+8)/2` 算成 7，是混合不同参数的梯度，改变了问题本身。**归约对齐的是相同参数或相同输出位置。**

<iframe src="/content/interactive/distributed-training/gradient-position-reduction.html?v=height-v2" title="按参数位置跨 rank 归约" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="gradient-position-reduction"></iframe>

[单独打开此图](/content/interactive/distributed-training/gradient-position-reduction.html?v=height-v2)

“归约”不自动等于“求平均”。对于 TP 同一输出的部分贡献，应求和恢复输出；对于已包含全局 loss 权重的梯度贡献，也不应重复除以卡数。数据并行目标中的平均缩放，需要依据 loss 定义处理。[NCCL 集合通信说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

## 6. ZeRO：逐步去掉模型状态冗余

沿用参数 2P、梯度 2P、优化器相关状态 12P 的假设，均匀分到 N 个数据并行 rank：

| 策略 | 参数 | 梯度 | 优化器相关状态 | 每卡持久状态 |
|---|---|---|---|---|
| DDP | 完整 | 完整 | 完整 | 16P |
| ZeRO-1 | 完整 | 完整 | 分片 | 4P+12P/N |
| ZeRO-2 | 完整 | 分片 | 分片 | 2P+14P/N |
| ZeRO-3 | 分片 | 分片 | 分片 | 16P/N |

这些是状态账本，不包含运行峰值的其他项目。阶段定义参考 [DeepSpeed ZeRO 官方教程](https://www.deepspeed.ai/tutorials/zero/) 与 [ZeRO 论文](https://arxiv.org/abs/1910.02054)。

### 6.1 ZeRO-1：先分片优化器状态

每个 rank 负责部分参数的优化器状态与更新。更新后，各 rank 交换更新的参数，使完整参数副本保持一致。主例 4 卡时：14+14+84/4=49 GB/卡。即使 N 趋于很大，完整参数与梯度仍至少 28 GB，因此本文题设的 24 GB 卡不能只靠 ZeRO-1 解决状态容量问题。

<iframe src="/content/interactive/distributed-training/zero-one-partition.html?v=height-v2" title="ZeRO-1 分片优化器相关状态" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="zero-one-partition"></iframe>

[单独打开此图](/content/interactive/distributed-training/zero-one-partition.html?v=height-v2)

### 6.2 ZeRO-2：再分片归约后的梯度

4 卡时：14+14/4+84/4=38.5 GB/卡。8 卡时：14+1.75+10.5=26.25 GB/卡，仍超过 24 GB。

**梯度分片不等于只计算自己负责的参数梯度。** 每个 rank 仍对自己的本地数据执行完整模型的前后向，并产生本地数据对各参数的贡献；随后归约、分片保存。可以按 bucket 边计算边通信释放，不要求先持有整个模型所有梯度。

ReduceScatter 操作的是梯度张量时，只改变梯度的结果布局，不会因此把 ZeRO-2 的完整模型参数切掉。

<iframe src="/content/interactive/distributed-training/zero-two-memory.html?v=height-v2" title="ZeRO-2 梯度和优化器状态分片" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="zero-two-memory"></iframe>

[单独打开此图](/content/interactive/distributed-training/zero-two-memory.html?v=height-v2)

### 6.3 ZeRO-3：参数平时也分片

8 卡时持久模型状态为 112/8=14 GB/卡。需要计算某一层时，聚合该层参数；用完后按策略重新分片。额外参数通信换来了更低的驻留显存。

<iframe src="/content/interactive/distributed-training/zero-three-lifecycle.html?v=height-v2" title="ZeRO-3 当前层参数聚合与释放" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="zero-three-lifecycle"></iframe>

[单独打开此图](/content/interactive/distributed-training/zero-three-lifecycle.html?v=height-v2)

## 7. FSDP：把参数的生命周期讲完整

### 7.1 分片单位与通信分组是两回事

全分片 FSDP 与 ZeRO-3 共享“参数、梯度和优化器状态分片”的核心思路，但不能把两个框架的 API、调度和实现细节完全等同。FSDP2 教程中参数表示为分片 DTensor，默认沿参数 dim 0 分片；把 Transformer block 设为 FSDP 单元，则让一组参数一起参与相关聚合调度。它并不是把整个 block 永久交给某一个 rank 独立计算。[FSDP2 官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

较小通信单元通常降低一次聚合的完整参数量，但增加 collective 启动与调度次数。较大单元则可能减少启动次数，却需要更大的临时完整参数。预取可以减少等待，也可能让相邻单元的完整参数同时驻留。选择粒度需要测量。

<iframe src="/content/interactive/distributed-training/fsdp-gather-granularity.html?v=height-v2" title="FSDP 参数聚合粒度与临时驻留量" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="fsdp-gather-granularity"></iframe>

[单独打开此图](/content/interactive/distributed-training/fsdp-gather-granularity.html?v=height-v2)

### 7.2 一个 block 的完整计算过程

以下采用“前向后重新分片、反向需要输入梯度、不预取”的明确题设：

1. **前向前：** 每张卡原本只有参数分片，AllGather 临时恢复当前 block 的完整参数。
2. **前向：** 各卡用完整参数处理自己的本地数据，保存反向需要的激活。其他 block 暂时仍是分片。
3. **前向后：** 释放临时完整参数，保留参数分片和必要激活。释放参数不等于释放激活。
4. **反向前：** 再次 AllGather，恢复当前 block 的完整参数。各卡计算本地数据对该 block 所有可训练参数的贡献和输入梯度。
5. **梯度就绪：** ReduceScatter 合并相同参数位置的贡献，按拥有者保留梯度分片；根据 loss 定义完成正确缩放。用完的完整参数与激活可以释放。
6. **更新：** 本轮所有必要反向、累积与梯度处理完成后，各卡更新自己的参数分片，使用对应梯度和优化器状态。无需长期保存完整参数；下一轮 AllGather 会聚合更新后的分片。

具体释放时机取决于 reshard 配置、预取、梯度累积及实现；这里讲的是指定策略的生命周期。

### 7.3 用小矩阵确认“完整计算”是什么意思

设 W=[[1,2],[3,4]]，GPU 0 平时保存第一行，GPU 1 保存第二行。采用列向量 y=Wx。GPU 0 的本地 x₀=[1,1]，GPU 1 的本地 x₁=[2,1]。

AllGather 后，两卡都使用完整 W：GPU 0 得到 y₀=[3,7]；GPU 1 得到 y₁=[4,10]。这两个输出不同，原因是本地数据不同，不是输出被拆成一半。

若各卡 loss 为 ½‖y‖²，则本地参数梯度是 yxᵀ：

```text
GPU 0：[[ 3, 3], [ 7, 7]]
GPU 1：[[ 8, 4], [20,10]]
跨卡均值：[[5.5,3.5], [13.5,8.5]]
```

GPU 0 保留平均梯度第一行，GPU 1 保留第二行。用学习率 0.1 的 SGD 方便手算，更新后的 W=[[0.45,1.65],[1.65,3.15]]。主训练案例仍使用 Adam；这里更换优化器只是展示分片的拥有与更新，不改变计算范围。

<iframe src="/content/interactive/distributed-training/fsdp-matrix-walkthrough.html?v=height-v2" title="FSDP 小矩阵：聚合、前后向、归约与更新" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="fsdp-matrix-walkthrough"></iframe>

[单独打开此图](/content/interactive/distributed-training/fsdp-matrix-walkthrough.html?v=height-v2)

我的最终理解是：**计算本地数据对完整 block 的梯度贡献，保存和更新本卡负责的分片。**

## 8. TP：切分的是算子的计算

### 8.1 输出维度切分与输入维度切分

为避免“行并行、列并行”的命名造成混淆，本文小矩阵统一用 **y=Wx，W 的形状为 [out,in]**。常见框架用行向量 Y=XW，此时画出来的行列方向会相反，应优先说清切的是输入还是输出特征。

按输出维度切 W 的行：每张卡产生不同输出元素，若需要完整输出则拼接。按输入维度切 W 的列：每张卡只算相同输出的一部分贡献，必须求和恢复输出。

例如 W=[[1,2],[3,4]]、x=[1,1]，输出切分时两卡得到 [3] 和 [7]，拼接为 [3,7]。输入切分时两卡分别计算 [1,3] 与 [2,4]，求和仍是 [3,7]。后者不能除以 2。

<iframe src="/content/interactive/distributed-training/tensor-parallel-output-split.html?v=height-v2" title="TP 按输出特征切分" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="tensor-parallel-output-split"></iframe>

[单独打开此图](/content/interactive/distributed-training/tensor-parallel-output-split.html?v=height-v2)

<iframe src="/content/interactive/distributed-training/tensor-parallel-input-split.html?v=height-v2" title="TP 按输入特征切分并求和" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="tensor-parallel-input-split"></iframe>

[单独打开此图](/content/interactive/distributed-training/tensor-parallel-input-split.html?v=height-v2)

FSDP 聚合完整算子参数后，各卡为自己的数据执行完整算子；TP 用参数子矩阵合作执行同一个算子。这就是“存储分片”与“计算分片”的差异。

### 8.2 MLP 两层配对，避免中间立刻拼全

第一层按输出切分，得到不同 hidden 元素；ReLU 是逐元素函数，各卡可以对自己的完整 hidden 元素独立处理。第二层按输入切分，直接消费本卡 hidden 分片，再求和得到完整输出。无需在两层之间先 AllGather hidden。[PyTorch TP 教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

一个 2→4→2 示例：x=[2,1]，W₁ 的行是 [1,0]、[0,1]、[1,1]、[-1,1]。第一层输出为 [2,1,3,-1]。两卡 ReLU 后分别是 [2,1] 与 [3,0]。第二层对应子矩阵为 I 和 2I，贡献为 [2,1]、[6,0]，最终输出 [8,1]。

但输入切分产生的“部分贡献”不能随意先过 ReLU。若两卡部分值为 [2,-3] 与 [-1,4]，先加再 ReLU 得到 [1,1]；先各自 ReLU 再加得到 [2,4]。非线性通常不能跨越求和移动。

<iframe src="/content/interactive/distributed-training/tensor-parallel-mlp-pair.html?v=height-v2" title="TP MLP 输出切分与输入切分配对" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="tensor-parallel-mlp-pair"></iframe>

[单独打开此图](/content/interactive/distributed-training/tensor-parallel-mlp-pair.html?v=height-v2)

### 8.3 Attention 按完整 head 切分

普通 MHA 可以让不同 rank 负责不同完整 head。QKV 投影按输出特征切分，各卡完成所拥有 head 的 Attention，再通过输入切分的输出投影 Wₒ 合并贡献。

只按 head 切分时，每个 rank 的本地 head 可以拥有完整序列的 Q/K/V。一个 head 的 Softmax 沿 key 位置进行，不需要其他 head 的 K/V。GQA、MQA 的 KV 头分配需要结合框架配置，不能机械套用每个 rank 都拥有独立相同数量 KV head 的假设。

<iframe src="/content/interactive/distributed-training/tensor-parallel-attention-heads.html?v=height-v2" title="Attention TP：完整 head 的序列计算" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="tensor-parallel-attention-heads"></iframe>

[单独打开此图](/content/interactive/distributed-training/tensor-parallel-attention-heads.html?v=height-v2)

### 8.4 反向为什么有时求和，有时拼接

对于 y=Wx，记 G=∂L/∂y：

$$\frac{\partial L}{\partial x}=W^\top G,\qquad \frac{\partial L}{\partial W}=Gx^\top.$$

输出切分时，各卡得到对**同一个完整输入 x** 的不同贡献，需要求和。例如 [2,4]+[3,4]=[5,8]。

输入切分时，各卡得到**不同输入元素**的梯度，若需要完整 dX 则拼接。例如 W 两列为 [1,3]、[2,4]，G=[1,2]：GPU 0 的 dX₀=1×1+3×2=7，GPU 1 的 dX₁=2×1+4×2=10，完整输入梯度是 [7,10]。

判断通信的办法是先标注每个数代表哪个位置：相同位置的贡献求和，不同位置的结果拼接。不能只凭“在不同 GPU 上”就决定求平均。

## 9. PP：按层分工，用 micro-batch 填流水线

### 9.1 stage 边界传什么

假设 GPU 0 负责 L₁、L₂，GPU 1 负责 L₃、L₄。前向 GPU 0 将边界激活 A 发给 GPU 1；反向 GPU 1 将 ∂L/∂A 发回 GPU 0。GPU 0 再通过链式法则计算 L₂、L₁ 的梯度。L₃、L₄ 的权重梯度由 GPU 1 保留，不需要当作边界输入梯度发给 GPU 0。

<iframe src="/content/interactive/distributed-training/pipeline-single-microbatch.html?v=height-v2" title="单 micro-batch 流水线的激活与输入梯度传递" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="pipeline-single-microbatch"></iframe>

[单独打开此图](/content/interactive/distributed-training/pipeline-single-microbatch.html?v=height-v2)

### 9.2 利用率公式的前提

把一个训练 batch 拆成 m 个 micro-batch，在 p 个 stage 之间错开执行。理想 GPipe 示意中，各 stage 平衡，前向与反向各占一个单位槽，忽略通信与其他开销：总长度 2(m+p−1)，每卡忙碌 2m。

$$U=\frac{m}{m+p-1},\qquad f_{bubble}=\frac{p-1}{m+p-1}.$$

这里气泡占比的分母是实际时间。某些材料给出 (p−1)/m，分母是理想计算时间，不能与上式当作同一个比例。

p=4、m=8 时 U=8/11，气泡 3/11。m 不变而 p=8 时 U=8/15。更多 stage 不自动让这个调度更高效；增加 micro-batch 可摊薄填充排空开销，但会影响激活、启动次数及单个 micro-batch 的算子效率。

<iframe src="/content/interactive/distributed-training/pipeline-microbatch-overlap.html?v=height-v2" title="多 micro-batch 的流水线填充与排空" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="pipeline-microbatch-overlap"></iframe>

[单独打开此图](/content/interactive/distributed-training/pipeline-microbatch-overlap.html?v=height-v2)

### 9.3 GPipe 与同步 1F1B

GPipe 先完成这一批全部前向，再做反向，许多 micro-batch 的激活需要同时保留。同步 1F1B 在预热后交替执行一个前向、一个反向，尽早释放使用完的激活，最后排空。

图中 2 stage、4 micro-batch 的等时案例，两种调度总长都是 10 槽、每卡忙 8 槽；GPipe 的激活份数峰值为 [4,4]，1F1B 为 [2,1]。这些份数由图中排程决定，不是任意模型的 GB 显存预测。普通 1F1B 的主要收益是降低激活驻留，不保证消除气泡。[PyTorch PP 教程](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)、[NVIDIA Megatron 调度说明](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)

**1F1B 不意味着每做完一个 micro-batch 的反向就 step。** 当前同步题设要等本轮 micro-batch 的梯度完成累积，再统一更新；否则其他 micro-batch 可能用旧参数前向、新参数反向，破坏参数版本一致性。

<iframe src="/content/interactive/distributed-training/pipeline-gpipe-one-f-one-b.html?v=height-v2" title="GPipe 与同步 1F1B 的调度和激活驻留" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="pipeline-gpipe-one-f-one-b"></iframe>

[单独打开此图](/content/interactive/distributed-training/pipeline-gpipe-one-f-one-b.html?v=height-v2)

## 10. Activation Checkpointing：少保存，反向时重算

保存区段输入与必要信息，而不保留全部内部激活；反向需要时重新执行必要的前向，重建中间张量，再完成反向。减少的是保存的激活，代价是额外前向计算，不是“把反向再执行一遍”。

它与保存到磁盘、用于恢复训练的 checkpoint 不同。后者可以保存模型、优化器、随机状态、步数等；前者改变单次训练过程的内存与计算安排。

若区段包含 Dropout，重算必须与原前向的随机行为一致。一般需要保存、恢复 RNG 状态；只设置一次相同种子不等于恢复到当前随机序列位置。PyTorch checkpoint 默认处理相关 RNG 状态，但设备迁移、状态变化和具体模式仍需遵守 API 条件。非重入实现可在必要重算完成后提前停止，并非总是完整重跑整个区段。[checkpoint API](https://docs.pytorch.org/docs/stable/checkpoint.html)

<iframe src="/content/interactive/distributed-training/activation-checkpoint-recompute.html?v=height-v2" title="激活保存与反向时的前向重计算" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="activation-checkpoint-recompute"></iframe>

[单独打开此图](/content/interactive/distributed-training/activation-checkpoint-recompute.html?v=height-v2)

这与相邻推理模块讨论的“历史 KV 按需重算”共享以计算换驻留空间的思想，但重建对象与生命周期不同。训练 checkpoint 重建反向需要的激活；推理 KV 重算重建当前 Attention 所需的历史状态。二者都要评估重算阶段的临时峰值，不能只比较长期保存量。

## 11. 混合精度与数值稳定：范围、精度、缩放分开

### 11.1 FP16、BF16、FP32

| 格式 | 符号位 | 指数位 | 显式小数位 | 主要特点 |
|---|---:|---:|---:|---|
| FP16 | 1 | 5 | 10 | 常见正常范围内比 BF16 精细，最大有限值 65504 |
| BF16 | 1 | 8 | 7 | 数值范围接近 FP32，尾数较粗 |
| FP32 | 1 | 8 | 23 | 范围大，尾数精度更高 |

100000 在 FP16 中可能溢出，BF16 可表示为附近的 99840；1.001 在 FP16 中约为 1.0009765625，BF16 则会舍入到 1。范围更大不等于小数更精细。极小数可能 underflow 或舍入为 0；超范围可能成为 Inf；Inf/Inf、0/0 等可能生成 NaN。[NVIDIA 精度说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)

autocast 根据算子选择合适 dtype，不应简单理解为所有张量都转成 FP16。是否用主参数、梯度 dtype 与优化器布局需要另查。[PyTorch AMP 文档](https://docs.pytorch.org/docs/stable/amp.html)

<iframe src="/content/interactive/distributed-training/floating-point-range-precision.html?v=height-v2" title="浮点范围与有效精度比较" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="floating-point-range-precision"></iframe>

[单独打开此图](/content/interactive/distributed-training/floating-point-range-precision.html?v=height-v2)

### 11.2 Loss Scaling 必须发生在梯度丢失之前

对 loss 乘 S 后再反向，链式法则使梯度也乘 S，帮助小梯度在 FP16 计算过程中保持可表示；更新前再 unscale。已经舍入为 0 的梯度事后乘 S 仍是 0，信息无法恢复。

S 过大会让其他梯度溢出。动态 GradScaler 会检测非有限梯度，必要时跳过更新并调整 scale。BF16 常因范围较大而不需要同样的缩放策略，但它仍不能修复公式本身的数值问题。[AMP 实践教程](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

<iframe src="/content/interactive/distributed-training/loss-scaling-gradient-range.html?v=height-v2" title="Loss Scaling：反向前放大与更新前还原" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="loss-scaling-gradient-range"></iframe>

[单独打开此图](/content/interactive/distributed-training/loss-scaling-gradient-range.html?v=height-v2)

### 11.3 先还原，再裁剪；先正确聚合，再做非线性处理

梯度 g=[3,4]、scale=10、裁剪阈值 1：先 unscale，再裁剪得到 [0.6,0.8]。若对 [30,40] 先裁剪成 [0.6,0.8]，再除以 10，会错误得到 [0.06,0.08]。

梯度裁剪与 micro-batch 平均也不交换。g₁=[3,0]、g₂=[0,4]：平均为 [1.5,2]，范数 2.5，裁剪为 [0.6,0.8]；若先各自裁剪，再平均则是 [0.5,0.5]，方向与大小都变了。

分片训练还要使用适合分布式布局的全局 norm 计算，不能只把某卡梯度分片的 norm 当作整个模型的 norm。逻辑上先得到本轮目标所需的聚合梯度、还原 scale，再按定义裁剪并更新；具体通信可以与反向重叠。

### 11.4 不同有效 token 数，需要不同权重

两个 micro-batch 有效 token 数为 2、8，各自 mean loss 的梯度为 6、2。全局 token 平均目标应使用：

$$g=\frac{2\times6+8\times2}{2+8}=2.8.$$

直接 `(6+2)/2=4` 把只占 20% token 的第一批赋予 50% 权重。跨 rank 也一样：先明确是 token 平均、样本平均还是其他目标，处理 mask 与总计数，再决定归一化。并行计算中的“sum”与 loss 目标中的“mean”是两个层次。

### 11.5 Softmax 稳定计算不能靠换 BF16 代替

手写 `exp(z)/sum(exp(z))`，在 FP32 下对 [100,102] 直接计算指数也会溢出，随后 Inf/Inf 得到 NaN。稳定算法先减去最大值 m：

$$p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j.$$

减去公共常数不改变数学结果，指数输入不大于 0，避免这里的正向指数溢出。[200,203] 应变为 [-3,0]。BF16 的范围接近 FP32，直接算 e²⁰⁰ 仍会溢出。

这是手写公式的教学示例，不是在说 PyTorch 内置 Softmax 对这些正常有限输入会采用不稳定实现。若输入本身已有 Inf、NaN 或 mask 导致整行无有效位置，仍要按实际算法处理；减最大值不是所有异常的通用修复。排查时定位第一个产生非有限值的算子。Loss Scaling 无法修复前向已经生成的 NaN。

<iframe src="/content/interactive/distributed-training/softmax-numerical-stability.html?v=height-v2" title="直接指数与减最大值的 Softmax" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="softmax-numerical-stability"></iframe>

[单独打开此图](/content/interactive/distributed-training/softmax-numerical-stability.html?v=height-v2)

## 12. SP：让每个本地 token 保有完整 hidden

### 12.1 先认清 LayerNorm 的统计轴

常见 LayerNorm(H) 对每个 token 的 hidden 向量计算均值与方差，统计轴是 H，不是 batch 或序列。x=[1,3] 时均值 2、总体方差 1，忽略 epsilon 且 γ=1、β=0，结果为 [-1,1]。[LayerNorm API](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)

如果按 hidden 切分，各卡只有一个元素，不能仅凭本地均值方差得到正确全向量的统计量；需要跨卡合并统计信息。无需因此断言一定要把整份激活 AllGather，统计量归约也可能足够。

若按序列切分，每张卡拥有部分 token 的完整 hidden，各 token 的 LayerNorm 前向就可独立完成。归一化统计是局部的，不意味着共享 γ、β 的参数梯度永远不需要同步。

<iframe src="/content/interactive/distributed-training/sequence-parallel-layernorm-axis.html?v=height-v2" title="LayerNorm hidden 轴与序列分片" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="sequence-parallel-layernorm-axis"></iframe>

[单独打开此图](/content/interactive/distributed-training/sequence-parallel-layernorm-axis.html?v=height-v2)

### 12.2 本文 SP 的具体范围

这里讨论常见 Megatron 风格、配合 TP 的 Sequence Parallel：在 LayerNorm、Dropout 等区段按序列保存激活，减少 TP rank 上的重复驻留。进入相关 TP 计算前 AllGather 序列激活；输出投影产生同位置的部分贡献后，用 ReduceScatter 求和并重新按序列分片。[PyTorch TP 与 SP 教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)

完整激活形状 [B,S,H]=[2,8,4096]，4 卡沿 S 均匀切分，局部是 [2,2,4096]：每条序列 2 个 token，batch 有 2 条，总计本卡 4 个位置。不是 [2,8,1024]，后者切了 hidden。

<iframe src="/content/interactive/distributed-training/sequence-parallel-tp-communication.html?v=height-v2" title="SP 与 TP 的 AllGather、ReduceScatter 布局转换" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="sequence-parallel-tp-communication"></iframe>

[单独打开此图](/content/interactive/distributed-training/sequence-parallel-tp-communication.html?v=height-v2)

## 13. CP：Attention 也持续按序列分工

### 13.1 本地 Query 为什么需要远端 K/V

Context Parallel 将序列分布到不同 rank，Attention 中保留本地 Query，交换计算所需的其他位置 K/V。与上一节的 SP 布局不同，这里 Attention 的序列计算本身也分布执行。[Megatron Core CP 说明](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

设序列位置 1、2 在 GPU 0，位置 3、4 在 GPU 1。因果 Attention 下：Q₄ 需要 K/V₁…₄，GPU 1 仅有 3、4 不够；Q₁ 只需要 K/V₁，不需要未来位置。全局位置与 causal mask 必须正确，不能按物理卡上的局部编号重新定义可见性。

简图可以用 AllGather KV 说明需要的信息；实际可用环形、分块点对点交换，边传边算，避免一次复制全部 KV。

<iframe src="/content/interactive/distributed-training/context-parallel-local-query.html?v=height-v2" title="CP 本地 Query 与远端 K/V 的因果可见性" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="context-parallel-local-query"></iframe>

[单独打开此图](/content/interactive/distributed-training/context-parallel-local-query.html?v=height-v2)

### 13.2 KV 反向要合并各 Query 的贡献

同一个 K₁ 被不同 rank 的 Query 使用后，它的梯度来自这些 Query 的贡献。若本地贡献为 [2,1]、远端为 [4,3]，且均已包含目标 loss 的归一化系数，则 K₁ 的拥有者应得到 [6,4]，不能再次平均成 [3,2]。这是链式法则的贡献合并。

### 13.3 因果计算量为什么容易不均

以有效 Query–Key 配对数作简化工作量代理，长度 8 的位置 i 需要看 i 个 key。连续分配 1…4 与 5…8 时，计数为 10 与 26；首尾配对 1、2、7、8 与 3、4、5、6 时，都是 18。

首尾配对在这个代理上更均衡，但不能据此保证实际耗时严格相同。kernel 的 tile、padding、通信、MLP 与硬件状态都会影响性能。重新分配不改变原 token 的全局位置或因果关系。

<iframe src="/content/interactive/distributed-training/context-parallel-causal-balance.html?v=height-v2" title="CP 因果工作量：连续与首尾配对分配" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="context-parallel-causal-balance"></iframe>

[单独打开此图](/content/interactive/distributed-training/context-parallel-causal-balance.html?v=height-v2)

## 14. 回到 7B 任务：把显存与一次 step 连起来

### 14.1 8 张 24 GB 卡的容量判定

ZeRO-2 持久状态 26.25 GB/卡，不能放入题设的 24 GB。ZeRO-3 持久状态 14 GB/卡，状态账本能放下。若**同一时刻**还有 3 GB 额外参数聚合、6 GB 保存激活、2 GB 通信与其他 buffer，则峰值为：

$$14+3+6+2=25\text{ GB}.$$

仍然不够。若 checkpoint 将该时刻激活从 6 GB 降到 2 GB，其他项不变，则峰值为 21 GB。ZeRO-3 改变模型状态冗余，checkpoint 改变激活保存量，因此两者可以组合。这只是在给定预算表下通过，不保证真实实现里未列出的开销为零。

### 14.2 一次可讲给面试官的完整训练时序

本轮先按数据并行分配本地 micro-batch，明确全局 loss 权重。在每个 block 前，FSDP AllGather 当前 block 参数；前向处理本地数据。对 checkpoint 区段只保留必要输入与随机状态；普通区段保留所需激活。前向后按题设释放完整参数。

反向逆序遍历 block：重新聚合必要参数，重算 checkpoint 区段的前向，计算本地数据对完整 block 的参数梯度和输入梯度。就绪的梯度参与 ReduceScatter，合并相同参数位置的跨数据贡献并保留分片。完成本轮累积后，正确 unscale、计算分布式全局 norm、按需裁剪，再由各 rank 更新自己的分片。下一轮聚合新的参数值。

若单层算子还太大，可以在 block 内引入 TP；若按层分布整模型，可以增加 PP；长序列可引入 SP/CP。这些是后续配置分支，不表示刚才“8 rank 全做 FSDP”的数字还能直接照搬。

例如简单、互不重叠的 64 卡设备网格可以取 DP=4、TP=4、PP=2、CP=2。此时 TP/PP/CP rank 合作处理同一份数据，真正独立的数据副本数是 4。若 FSDP 在这 4 个 DP rank 间分片，显存账本先考虑 PP 分层、TP 算子分片，再按实际 FSDP 组分摊；不能直接把任意状态都除以 64。实际 hybrid sharding 或共享通信组还要以框架设备网格为准。

## 15. 从 8 卡到 64 卡：看吞吐，再找关键路径

### 15.1 扩展效率怎么计算

有效 tokens/s=每 step 全局有效 token 数÷step 秒数。例子中，8 卡每步 8192 token、1 秒，吞吐 8192 tokens/s；64 卡每步 65536 token、2 秒，吞吐 32768 tokens/s。吞吐提高 4 倍，GPU 数提高 8 倍，扩展效率为 4/8=50%。

64 卡 step 更慢，但每步工作量变大，吞吐仍提高。比较时必须区分固定全局工作量的 strong scaling 与固定每卡工作量的 weak scaling；不能把两种测试口径混为一谈。

还应记录模型与训练算法、精度、GPU 和网络拓扑、序列长度分布、有效 token 与 padding、每卡 micro-batch、累积次数、DP/TP/PP/CP/FSDP 组配置。固定 global batch 时，增加 DP 数通常要调整每卡 batch 或累积次数；固定后两者时，global batch 则增大。“batch 和并发都相同”不是足够清楚的口径。

### 15.2 原始通信更短，不保证 step 更短

只对这张简化时间线，可写 step=准备+计算+没有被隐藏的通信。准备 100 ms，计算 800 ms：A 总通信 1200 ms，重叠 300 ms，step=1800 ms；B 总通信 1400 ms，重叠 700 ms，step=1600 ms。

B 总通信更长，却减少了关键路径上的暴露通信。在实际 trace 中不能把所有 NCCL 事件时长简单相加或机械减去一个“重叠比例”，要根据依赖、不同 stream 和各 rank 时间轴识别真正延长 step 的部分。

<iframe src="/content/interactive/distributed-training/training-communication-overlap.html?v=height-v2" title="原始通信、重叠与关键路径中的暴露通信" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="training-communication-overlap"></iframe>

[单独打开此图](/content/interactive/distributed-training/training-communication-overlap.html?v=height-v2)

### 15.3 通信记录长，也可能是在等其他 rank

设时间轴已对齐：GPU 0 在 100 ms 梯度就绪并进入 AllReduce，GPU 1 到 300 ms 才就绪，均在 330 ms 完成。GPU 0 的记录长 230 ms，但不能宣布网络传输了 230 ms；其中可能包含等待 GPU 1。

| 观察证据 | 优先调查 | 可验证的优化与指标 |
|---|---|---|
| 输入晚就绪，之后计算正常 | 读取、预处理、H2D 与预取 | 改善输入供给；输入等待下降、吞吐上升 |
| 输入相近，但某卡计算更长 | 工作量、算子形状/调用次数、执行效率 | 均衡实际工作或优化已定位算子；计算与就绪差距缩小 |
| 各卡同时就绪，通信仍长 | 拓扑、通信实现、带宽、消息大小 | 独立通信测试与端到端比较；暴露通信下降 |
| 一卡早进通信，其他卡晚进 | 追查慢 rank 的上游阶段 | 修复等待来源；等待下降且 step 缩短 |

可用 PyTorch Profiler 标记数据、前向、反向区段，记录 CPU/CUDA 活动、算子形状和调用次数，导出 trace 并对照各 rank。跨主机比较需确保时钟或时间轴对齐。[Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

<iframe src="/content/interactive/distributed-training/rank-timeline-bottleneck-diagnosis.html?v=height-v2" title="同样 step 耗时下的数据、计算与通信瓶颈" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="rank-timeline-bottleneck-diagnosis"></iframe>

[单独打开此图](/content/interactive/distributed-training/rank-timeline-bottleneck-diagnosis.html?v=height-v2)

图中三种原因都得到 step=230 ms，GPU 0 的通信记录也都是 130 ms，但分别由输入慢、计算不均、通信阶段长造成。灰色等待是依据就绪时间解释的教学区段，不代表 profiler 必然输出一个独立、准确同名的事件。

### 15.4 优化慢卡，而不是要求每张卡都更快

两卡数据准备各 20 ms，通信在两卡计算完成后耗时 30 ms。原计算耗时 [80,180] ms，step=230 ms；重新分配为 [130,130] ms，step=180 ms。

GPU 0 自己变慢，但整体关键路径缩短，固定工作量吞吐提高 230/180≈1.28 倍。负载均衡的目标是降低最慢路径与等待，不是每个 rank 都独立刷新最快时间。“相同 token 数”也不一定等于相同 Attention 工作量。

面试回答可以按“**观察 → 假设原因 → 验证 → 优化 → 端到端复测**”组织。优化前先确认因果证据，优化后报告 step、有效吞吐、就绪差距、暴露通信和峰值显存的变化。最终训练效率还应考虑数值正确性、收敛与达到目标质量所需的总时间。

## 16. 我的错误回答与不够完整的地方

下表保留真实学习过程。后来答对的题不再列为“仍不会”，只是留下复习检查点。

| 当时的回答或遗漏 | 问题在哪里 | 现在的准确表述 |
|---|---|---|
| micro-batch 缩成 1/8 后，认为梯度和“优化器保留的激活”也缩成 1/8 | 混淆参数状态与激活；优化器状态不是通常所说的前向激活 | 稠密参数梯度与状态形状不变，主要减少保存激活 |
| ReduceScatter 平均 `[2,4]` 与 `[6,8]`，答 3、7 | 平均了同 rank 的不同参数 | 对齐相同位置跨 rank 归约，得到 4、6 |
| 说 ReduceScatter 后各卡保留部分“参数” | 没说清此次通信的张量类型 | 梯度通信改变梯度布局；ZeRO-2 参数仍完整 |
| ZeRO-2 前向可以，反向只能算自己保留那部分 | 把保存/更新责任等同于本地贡献的计算范围 | 本地数据对完整模型的贡献都要计算，再归约分片 |
| ZeRO-3 只给出 `7×16/8` | 状态账本正确，尚未回答能否保证运行 | 14 GB 不是峰值，还需聚合参数、激活与临时空间 |
| TP 输入切分反向答 `[2,6]`、`[2,4]`，再求和 `[4,10]` | 套用前向元素运算，未先写 dX=WᵀG；混淆不同输入位置 | 两列与 G 点积得到 7、10；不同位置拼接为 `[7,10]` |
| PP 利用率答 4/5，但解释只说“首次要激活最开始阶段” | 数值正确，填充、排空与总槽数解释不完整 | 先写 2(m+p−1) 与 2m，再说明气泡来源 |
| 扩卡比较认为 batch、并发都相同 | 没区分 global 与 per-rank，也没说明 strong/weak scaling | 记录 global batch、micro-batch、累积、并行组和有效 token |
| 各 rank 时序如何区分瓶颈，答“不太讲得清楚” | 缺少逐阶段就绪证据 | 比较输入、算子计算、梯度就绪、通信启动与完成 |
| 通信重叠、负载均衡方向正确，但未给验证指标 | 优化建议尚未形成闭环 | 暴露通信/等待下降，并用相同工作量下 step 与有效吞吐验证 |
| 计算不均时提出让序列长度相等、优化算子 | 可作为候选，但 trace 尚未证明具体原因 | 先对照实际工作量、形状与算子调用；再采取对应优化 |

后续已经能正确回答：FSDP 完整生命周期、TP 输入梯度、loss 缩放与裁剪顺序、CP 因果负载、8→64 卡吞吐，以及 ZeRO-3 与 checkpoint 的峰值组合题。下一步源码与真实多卡实践可以继续检验这些原理是否能用于实际系统。

## 17. 我主动追问的几个问题

### “没看懂 FSDP 具体怎么分，以及计算过程怎样”

关键是把两层概念分开：参数平时以分片形式驻留；计算当前单元前聚合完整参数。block 分组控制参数通信的集合与调度，不意味着每个 rank 只算半个 block。小矩阵示例的输出 `[3,7]` 是本地样本的完整输出。梯度归约后才保留拥有者的分片。

### “是不是输出维度按行切，算完拼接；输入维度按列切，算完融合”

在本文 y=Wx、W[out,in] 约定下成立。但“融合”应具体说成对相同输出位置的线性贡献求和；不能任意使用平均或其他函数。框架若用 Y=XW，图上的行列方向会反转，应该用输入/输出轴说明。

### “ReLU 能在各卡独立处理，第二层直接用对应 hidden，再求和吗”

第一层输出切分时，每个 hidden 元素已完整，逐元素 ReLU 可以本地执行；第二层输入切分与本地 hidden 匹配，可直接计算贡献并求和。若第一层本身只是同一元素的部分贡献，则必须先恢复完整值再做 ReLU。

### “这里说的应该是 Attention 计算吧”

是。CP 中 Q₄ 缺少远端 K/V 的讨论发生在 Attention；LayerNorm、逐 token MLP 对序列其他位置没有同样的读取要求。因果 mask 限定可见位置，Q₁ 无需未来 KV，而 Q₄ 必须得到前面位置的信息。

### 从相邻会话延伸：重计算是不是不可行的删缓存

在推理模块，我曾提出保留最新 KV、历史 KV 按需重建；最终澄清这是可行的计算换显存思路，不是删除历史语义。这里训练 checkpoint 也遵循“需要时正确重建”的原则。但训练激活与推理 KV 的依赖和用途不同，应分别计算额外工作与临时峰值。

## 18. 从浅到深自测：先作答，再展开

### A. 训练状态与显存

**1. 已执行 backward，但还没 step，参数是否改变？**

<details><summary>答案</summary>

通常尚未改变。backward 计算并累积梯度；optimizer.step 才执行参数更新。

</details>

**2. 两次 backward 的梯度分别是 −10 与 −6，未清理时得到多少？只保留后一次要怎样做？**

<details><summary>答案</summary>

得到 −16。若不希望累积，应在第二次 backward 前清理 `.grad`；只得到 −6。

</details>

**3. 100 万可训练参数，micro-batch 从 8 变为 1，梯度元素变成多少？**

<details><summary>答案</summary>

普通稠密梯度仍有 100 万元素。micro-batch 改变数据与激活，不改变参数形状。

</details>

**4. 本文 7B Adam 状态总量是多少？8 卡 DDP 每卡多少？**

<details><summary>答案</summary>

16×7=112 GB。DDP 复制完整状态，每卡仍为 112 GB，不是 14 GB。

</details>

**5. 4 卡 ZeRO-1、ZeRO-2 的状态量分别是多少？**

<details><summary>答案</summary>

ZeRO-1：14+14+84/4=49 GB；ZeRO-2：14+14/4+84/4=38.5 GB。

</details>

**6. 8 张 24 GB 卡，ZeRO-2 与 ZeRO-3 是否通过持久状态容量检查？**

<details><summary>答案</summary>

ZeRO-2 为 26.25 GB，不能；ZeRO-3 为 14 GB，持久状态能放下，但尚不能保证运行峰值。

</details>

**7. ZeRO-3 状态 14 GB，同一时刻聚合 3 GB、激活 6 GB、其他 2 GB，峰值多少？checkpoint 使激活降到 2 GB 后呢？**

<details><summary>答案</summary>

原峰值 25 GB，超 24 GB；新峰值 21 GB。状态仍 14 GB。分片与重计算作用在不同项目上。

</details>

### B. 归约与 FSDP

**8. `[2,4]` 与 `[6,8]` 跨卡平均并 ReduceScatter，一卡负责一列，各得什么？**

<details><summary>答案</summary>

GPU 0 得 4，GPU 1 得 6。跨卡对齐相同参数位置，而不是平均一张卡内部的两个不同位置。

</details>

**9. `[1,10]` 与 `[5,2]`，同样的操作结果呢？会把参数也分片吗？**

<details><summary>答案</summary>

分别 3、6。若通信对象是梯度，就只改变归约后梯度的布局，不会自动切分参数。

</details>

**10. ZeRO-2 中一个参数由 GPU 1 更新，GPU 0 能跳过本地数据对它的梯度吗？**

<details><summary>答案</summary>

不能。否则缺少 GPU 0 样本的贡献，更新不再代表目标全局数据。

</details>

**11. FSDP 以 block 聚合且前向后 reshard，反向需要输入梯度，讲出生命周期。**

<details><summary>答案</summary>

分片驻留→前向前 AllGather→完整参数计算本地前向并保存必要激活→前向后释放完整参数→反向前再次 AllGather→计算本地数据对完整 block 的参数梯度与输入梯度→ReduceScatter 正确归约并保存梯度分片→用完后释放完整参数/激活→本轮累积结束后更新自己的分片。其他策略与预取可能调整驻留时机。

</details>

**12. FSDP 的通信单元越小就一定越快吗？**

<details><summary>答案</summary>

不一定。可能降低临时参数量，但增加 collective 启动与调度次数；需要比较暴露通信、step 与显存峰值。

</details>

### C. TP 与反向

**13. y=Wx、W[out,in]，输出切分与输入切分分别切哪里，怎样恢复完整输出？**

<details><summary>答案</summary>

输出切分切行，结果是不同输出位置，需要完整输出时拼接；输入切分切列，结果是同一输出的部分贡献，求和恢复。

</details>

**14. FSDP 和 TP 都切了权重，计算过程区别是什么？**

<details><summary>答案</summary>

本文 FSDP 聚合当前单元完整参数后，各卡计算各自数据的完整算子；TP 各卡用参数子矩阵合作计算同一算子。

</details>

**15. 两层 TP MLP 为什么可以不在中间拼出完整 hidden？**

<details><summary>答案</summary>

第一层输出切分后各 hidden 元素完整，逐元素函数可本地执行；第二层输入切分匹配本地 hidden，直接算部分输出贡献，最后求和。

</details>

**16. 部分贡献 `[2,−3]` 和 `[−1,4]`，ReLU 是否与相加交换？**

<details><summary>答案</summary>

不交换。先加再 ReLU 为 [1,1]；先 ReLU 再加为 [2,4]。

</details>

**17. 只按 Attention head 切分，GPU 0 的 Head 1 是否一定缺少远端 token 的 K/V？**

<details><summary>答案</summary>

不是。题设未按序列切分，它持有本地 head 的完整序列 Q/K/V；其他 head 的 KV 不用于这个 head 的 Attention。

</details>

**18. 输出切分反向，各卡对同一输入贡献为 `[2,4]` 和 `[3,4]`，完整 dX 是什么？**

<details><summary>答案</summary>

求和为 [5,8]，不除以 2。这是同一输入位置的链式法则贡献合并。

</details>

**19. 输入切分 W 两列为 `[1,3]` 与 `[2,4]`，G=[1,2]，完整 dX 是什么？**

<details><summary>答案</summary>

各卡计算 W 的本地列与 G 的点积：7 与 10，对应不同输入元素，拼接成 [7,10]。

</details>

### D. PP 与重计算

**20. 后 stage 应向前 stage 发送自己的权重梯度，还是边界激活的梯度？**

<details><summary>答案</summary>

发送 ∂L/∂A，供前 stage 继续链式反向。后 stage 自己保存其权重梯度。

</details>

**21. 理想 GPipe，p=4、m=8，利用率与按实际时长算的气泡比例？p=8 时呢？**

<details><summary>答案</summary>

p=4：8/11、3/11；p=8：8/15、7/15。前提是平衡、前后向等时、忽略通信。

</details>

**22. 同步 1F1B 的主要显存收益是什么？每个 micro-batch 完成反向后可以立即 step 吗？**

<details><summary>答案</summary>

更早使用并释放保存激活，减少在途激活数量。通常要完成本轮所有累积再更新，避免前后向参数版本不一致。

</details>

**23. checkpoint 省模型参数和 Adam 状态吗？重算的是反向吗？**

<details><summary>答案</summary>

通常不减少这些状态。它少保存内部激活，反向需要时重算必要的前向，再进行反向。

</details>

**24. checkpoint 中含 Dropout，只设一次相同 seed 就一定保证重算一致吗？**

<details><summary>答案</summary>

不一定。需要恢复到原前向相应的 RNG 状态位置，并保证其他影响计算的状态一致。

</details>

### E. 精度与 loss 归一化

**25. 100000 与 1.001，BF16 和 FP16 各有什么优势？**

<details><summary>答案</summary>

100000 用 BF16 更不易溢出，FP16 超过最大有限值；1.001 用 FP16 保留更细，BF16 在此会舍入到 1。

</details>

**26. 极小梯度已经变成 0，再乘 1024 能救回来吗？**

<details><summary>答案</summary>

不能。应在 backward 前放大 loss，使梯度形成时就处于较可表示范围，更新前再还原。

</details>

**27. g=[3,4]、scale=10、裁剪阈值 1，两种先后顺序各得到什么？**

<details><summary>答案</summary>

先还原再裁剪为 [0.6,0.8]；先裁剪放大梯度再还原为 [0.06,0.08]。应使用前者。

</details>

**28. g₁=[3,0]、g₂=[0,4]，阈值 1，先均值再裁剪与先裁剪再均值是否相同？**

<details><summary>答案</summary>

不相同，分别 [0.6,0.8] 与 [0.5,0.5]。裁剪是非线性处理。

</details>

**29. 两批有效 token 数 2、8，本地 mean 梯度 6、2，全局 token mean 梯度多少？**

<details><summary>答案</summary>

(2×6+8×2)/10=2.8。直接平均为 4，会过度加权第一批。

</details>

**30. `[200,203]` 稳定 Softmax 的平移是什么？直接换 BF16 可以代替吗？**

<details><summary>答案</summary>

减最大值 203 得 [-3,0]，概率约 [0.047426,0.952574]。不能用 BF16 替代稳定计算，直接指数仍会溢出。

</details>

### F. SP 与 CP

**31. x=[1,3] 的 LayerNorm 均值、总体方差、归一化结果？忽略 epsilon 和仿射参数。**

<details><summary>答案</summary>

2、1、[-1,1]。按 hidden 切分时需要合并统计信息，按 token 切分保留完整 hidden 时可本地计算。

</details>

**32. `[2,8,4096]` 在 4 卡沿 sequence 切分，本地形状是什么？每卡总共多少 token 位置？**

<details><summary>答案</summary>

[2,2,4096]，每条序列 2 个位置、batch 2，总共 4 个本地位置。

</details>

**33. SP 配合 TP 的输出投影为何用 ReduceScatter 求和而不是平均？**

<details><summary>答案</summary>

各卡是相同输出位置的线性部分贡献，应求和恢复，再沿序列散布。它不是多份独立完整输出的平均。

</details>

**34. CP，GPU 0 存 1、2，GPU 1 存 3、4。Q₄ 只用本地 KV 是否足够？Q₁ 需要远端吗？**

<details><summary>答案</summary>

Q₄ 不够，缺少位置 1、2；因果 Attention 的 Q₁ 仅需位置 1，不需要未来远端 KV。

</details>

**35. 已归一化的 K₁ 梯度贡献 `[2,1]`、`[4,3]`，最终多少？**

<details><summary>答案</summary>

[6,4]，求和，不重复平均。

</details>

**36. 长度 8 的因果配对代理，连续 1…4 / 5…8 与首尾 1、2、7、8 / 3、4、5、6 的工作量各多少？**

<details><summary>答案</summary>

连续 10/26，首尾 18/18。代理更均衡不等于真实 kernel 耗时一定完全相等。

</details>

### G. 扩卡与诊断

**37. 8 卡：8192 token、1 s/step；64 卡：65536 token、2 s/step。吞吐、加速、扩展效率？**

<details><summary>答案</summary>

8192 与 32768 tokens/s，加速 4 倍，效率 50%。step 慢不等于吞吐低。

</details>

**38. 固定每卡 micro-batch 与累积次数，DP 从 8 到 64，global batch 还能固定吗？**

<details><summary>答案</summary>

不能，在纯 DP 中会扩大 8 倍。固定 global batch 就要调整每卡 micro-batch 或累积次数等。

</details>

**39. 准备 100 ms、计算 800 ms；A 通信 1200、重叠 300；B 通信 1400、重叠 700。谁快？**

<details><summary>答案</summary>

A 为 1800 ms，B 为 1600 ms。B 的暴露通信更少。公式只适用于给定简化时间线。

</details>

**40. GPU 0 在 100 ms 进入归约，GPU 1 在 300 ms，均于 330 ms 完成，能断言网络用了 230 ms 吗？**

<details><summary>答案</summary>

不能。GPU 0 可能在等 GPU 1。优先调查 GPU 1 输入或前后向为何晚就绪，再分析实际传输。

</details>

**41. 计算 `[80,180]` 变为 `[130,130]`，准备 20、后续通信 30，优化失败了吗？**

<details><summary>答案</summary>

没有。step 从 230 到 180 ms，固定工作量吞吐提高约 1.28 倍。单卡变慢不等于整体退化。

</details>

**42. 用一两分钟回答：64 卡吞吐只达到 8 卡的 4 倍，怎么查？**

<details><summary>答案</summary>

先明确 strong/weak scaling、模型与精度、硬件拓扑、global batch、micro-batch、累积、序列分布和有效 token；计算吞吐与扩展效率。对齐各 rank trace，检查输入、算子、梯度就绪、通信启动和完成，区分输入慢、计算不均、真实通信长和等待。根据证据改善输入预取、按实际工作量均衡、优化具体算子、调整通信粒度/拓扑或提高重叠。最后在清楚的工作量口径下复测 step、有效 tokens/s、等待与暴露通信，并检查峰值显存、数值和训练质量。

</details>

## 19. 复习时保留的判断方法

每遇到一种并行方案，先问：切的是数据、层、特征、序列，还是驻留状态？每张卡本地拿到的张量代表完整结果、不同位置，还是同位置的部分贡献？然后才能判断使用 AllGather、拼接、求和或平均。

显存题写出 dtype 与字节数，再区分持久状态和同一时刻的额外占用。性能题先固定比较口径，从各 rank 的时序找到整体关键路径。能说清“省了哪项显存、增加哪项通信或计算、如何验证最终收益”，比只背策略名称更有用。

## 参考资料

- [PyTorch Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：训练状态与梯度累积。
- [DeepSpeed ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/) 与 [ZeRO 论文](https://arxiv.org/abs/1910.02054)：状态分片的阶段划分。
- [PyTorch FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)：分片、聚合、预取与混合精度策略。
- [PyTorch TP/SP](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)：线性层、Attention 与序列布局。
- [PyTorch PP](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)：阶段划分与流水线调度。
- [NVIDIA Megatron](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)：同步 1F1B 与激活驻留。
- [PyTorch Activation Checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)：重计算、随机状态与实现差异。
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html) 与 [AMP 实践](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)：autocast、GradScaler 与裁剪顺序。
- [Megatron Core CP](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)：KV 通信与因果负载。
- [NCCL Collectives](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)：集合通信的结果布局。
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：算子与时间轴证据。

<script src="/js/transformer-interactives.js" defer></script>

<style>
@media(max-width:400px){#post-meta .post-meta-date{display:grid;grid-template-columns:max-content max-content;gap:4px 6px;white-space:normal}#post-meta .post-meta-date>.post-meta-label:first-child{grid-column:1;grid-row:1}#post-meta .post-meta-date>.post-meta-date-created{grid-column:2;grid-row:1}#post-meta .post-meta-date>.post-meta-label:last-of-type{grid-column:1;grid-row:2}#post-meta .post-meta-date>.post-meta-date-updated{grid-column:2;grid-row:2}#post-meta .post-meta-date>.post-meta-separator{display:none}}
</style>
