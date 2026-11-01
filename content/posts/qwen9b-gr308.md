# Qwen3.5-9B 推理调优笔记：从 Gated DeltaNet 到 GR308 的调度与双副本

看到 Qwen3.5-9B 在 GR308 上留下的 TP1×DP2、MTP4、Graph [4,8,16,32] 和 batch token 4096，最容易记住的是参数组合，最难回答的是这些选择为什么成立。本篇从模型内部的一次状态更新开始，用一条 3072 token 输入、最多生成 1024 token 的文本请求，串起算子、缓存、调度、通信和最终实验。

这是一篇结构与工程复习笔记。基础公式按官方 Qwen3.5-9B 配置和实现解释；性能数据来自 2026 年 9 月的 GR308 历史记录，2026 年 11 月 1 日进行的是整理与复核。教学估算、互动演示和真实测量分别标明，避免把一次演示当作硬件剖析。涉及其他模型的结论只用于比较优化思路，详细历史见 [GR308 语言模型总览](/2026/10/02/gr308-language/)、[图像模型复盘](/2026/10/02/gr308-image/) 与 [视频模型复盘](/2026/10/02/gr308-video/)。

## 1. 一条请求串起整条知识链

固定研究对象为 Qwen3.5-9B 的文本执行路径，BF16 权重，服务最大上下文 4096。单条请求输入 3072 token，最多输出 1024 token；并发测试使用 C1 和 C16。C16 是客户端最多同时发起 16 条请求，并不保证服务的每一轮都恰好运行 16 条，也不保证 DP2 的两个副本始终各处理 8 条。

| 符号 | 数值 | 含义 |
| --- | ---: | --- |
| $D$ | 4096 | 隐藏维度 |
| $F$ | 12288 | 稠密 SwiGLU 中间维度 |
| $L$ | 32 | 主干 Decoder 层数 |
| $L_G/L_A$ | 24/8 | GDN 层数与完整注意力层数 |
| $H_Q/H_{KV}/d_A$ | 16/4/256 | 完整注意力的头配置 |
| $H_K/H_V/d_G$ | 16/32/128 | GDN 的 QK 头、V 头与每头维度 |
| $T_{in}/T_{out}$ | 3072/1024 | 输入长度与目标输出长度 |
| $C$ | 1 或 16 | 外部请求并发 |

请求先进入服务队列，调度器判断序列槽位和缓存是否足够；随后将一部分或全部 prompt 送入 prefill。模型把 token ID 查表成 4096 维向量，经过 32 层混合主干，建立各层的历史状态。Prefill 通常从最后一个输入位置的 logits 采样得到第一个输出；此后普通 decode 将新生成的 token 作为输入，继续预测下一个 token。MTP 则插入候选提出、主模型验证和状态提交的过程。

整个生命周期可以读成：

```text
3072 token prompt → 排队与容量检查 → token预算决定本轮prefill多少
→ 24层GDN更新状态、8层Attention建立KV → 最终Norm与LM Head
→ 首个输出token → 普通decode或MTP验证 → 更新缓存、继续生成
→ 达到停止条件 → 释放请求占用的缓存块和状态槽位
```

### 互动图：追踪请求的生命周期

[单独打开请求链路演示](/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=chain&v=e5788bdda5ce)。按步骤查看当前处理什么、读取哪些历史、更新哪些状态。演示不计算真实 GPU 延迟。

<iframe src="/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=chain&v=e5788bdda5ce" title="Qwen9B 请求生命周期" class="transformer-interactive" data-widget="qwen9b-chain" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:800px;border:0"></iframe>

## 2. 模型骨架：混合的是序列交互方式

Qwen3.5-9B 是带视觉编码器的多模态模型，GR308 文本实验设置了 `--language-model-only`，因此这里讨论语言主干。32 层的排列为：

```text
[GDN + FFN，GDN + FFN，GDN + FFN，Gated Full Attention + FFN] × 8
```

每个 Block 的外部骨架仍是 Pre-Norm 与残差：

$$
u=x+\operatorname{Mixer}(\operatorname{Norm}(x)),\qquad
y=u+\operatorname{FFN}(\operatorname{Norm}(u)).
$$

Mixer 在 24 层使用 Gated DeltaNet，在 8 层使用完整注意力。每层 FFN 都是稠密 SwiGLU，不存在专家路由，EP 不是这一个模型的优化方向。代码复用的类名即便含有 `Moe`，也要继续检查配置和具体分支，不能凭类名断定模型是 MoE。

完整注意力采用 GQA，16 个 Q 头共享 4 个 KV 头，执行 Q/K 归一化、位置编码与因果注意力，再由输出 gate 调节：

$$
A=\operatorname{softmax}(QK^\top/\sqrt{256}+M)V,\qquad
Y=W_O(A\odot\sigma(g)).
$$

这里的 gate 与 GDN 后面使用 SiLU 的输出门不是同一个具体算子。GQA 减少 KV 头数，8 层完整注意力仍要保存随上下文增长的 K/V，不能说混合模型完全没有 KV Cache。

稠密 FFN 为：

$$
\operatorname{FFN}(x)=W_{down}[\operatorname{SiLU}(W_{gate}x)\odot W_{up}x].
$$

两路从 4096 扩到 12288，再降回 4096。仅这三个矩阵，每层约 $3\times4096\times12288=1.51$ 亿参数，32 层约 48.3 亿。这个数量级估算说明减少完整注意力层数后，模型仍有大量稠密矩阵工作；它没有证明运行时一定受哪一类瓶颈限制。

## 3. GDN 前向：局部卷积怎样接到长期状态

### 3.1 一次投影生成六类信号

令 GDN 模块输入为 $X:[B,T,4096]$，它已经经过该子层前的归一化。投影得到 Q、K、V、Z，以及控制遗忘和更新的 A、B 信号：

| 分支 | 每个 token 的形状 | 职责 |
| --- | --- | --- |
| Q、K | 各 $[16,128]$ | 读取条件、写入位置 |
| V、Z | 各 $[32,128]$ | 待写入内容、输出通道门控 |
| $a_t,b_t$ | 各 $[32]$ | 每个 value head 的遗忘与更新控制 |

投影权重是训练得到的模型参数。实现可以将多个投影打包成更大 GEMM；项目代码中的 `in_proj_qkvz` 与 `in_proj_ba` 体现了这种组织，六类信号不等于必须启动六次矩阵乘法。

### 3.2 因果卷积计算近邻位置的加权组合

Q/K/V 总计 8192 个通道，经过沿序列方向的 depthwise Conv1D，kernel size 为 4。对一个通道，其计算可示意为：

$$
c_t=\operatorname{SiLU}(w_0r_t+w_1r_{t-1}+w_2r_{t-2}+w_3r_{t-3}).
$$

$r$ 是投影后的通道特征，$w$ 是固定的已训练卷积权重。每个通道有自己的卷积核；这一步只直接读取当前与前三个位置，不读取未来，也不负责跨通道混合。跨通道信息已经通过线性投影发生交互。Z、A、B 分支不走这段 QKV 卷积。

短卷积让当前位置的 Q/K/V 带上局部模式。Decode 为它维护 conv state，以复用短窗口所需的投影历史。输入隐藏状态本来就可能携带远处上下文，因此“卷积直接看四个位置”不等于模型只能看四个 token。

### 3.3 头对齐、QK 归一化与三类门

卷积后，参考实现把 16 个 Q/K 头对应到 32 个 V 头，每个 Q/K 头被用于两个 value head。Q/K 做 L2 归一化，Q 额外乘 $1/\sqrt{128}$。这些是 GDN 状态计算使用的 Q/K，不要与完整注意力中 head dimension 为 256 的配置混用。

门控关系为：

$$
\beta_t=\sigma(b_t),\quad
g_t=-\exp(A_{log})\operatorname{softplus}(a_t+dt_{bias}),\quad
\alpha_t=\exp(g_t).
$$

$A_{log}$ 与 $dt_{bias}$ 是模型参数，$a_t,b_t$ 来自当前输入。$\alpha$ 控制旧状态的衰减，$\beta$ 控制当前纠错量写入的强度，Z 控制最后输出的通道。这三类门作用位置不同。

### 3.4 状态矩阵按请求、层、头分别保存

为统一公式，使用 $S:[d_K,d_V]$ 的方向。每个请求、每个 GDN 层、每个 value head 维护自己的 $128\times128$ 矩阵。逻辑上可以把全部状态写成：

$$
[24,B,32,128,128].
$$

物理布局不必如此排列；SGLang 与厂商内核还可能分别使用 $[N,H,V,K]$ 与 $[N,H,K,V]$，需要在接口边界转换。单看两个轴长度相同看不出语义转置，必须理解矩阵方向。

全新请求通常从零状态开始；带已有上下文的继续执行从对应请求的缓存恢复。固定层在不同 token 步之间复用并更新状态，通常不会为每个历史 token 永久保留一整份矩阵。投机验证等路径可能另外保存中间状态，不能把“通常只需当前递推状态”当作全部运行时内存承诺。

### 3.5 先衰减、再预测、再纠错，最后读取

对一个请求、一个头，使用归一化后的 $k_t$ 和已缩放的 $q_t$：

$$
\widetilde S_t=\alpha_tS_{t-1},\qquad
\hat v_t=k_t^\top\widetilde S_t,
$$

$$
e_t=v_t-\hat v_t,\qquad
S_t=\widetilde S_t+\beta_tk_te_t^\top,\qquad
o_t=q_t^\top S_t.
$$

旧状态先遗忘一部分；当前 K 从旧状态读出 V 的预测；真实 V 与预测的差成为纠错量；K 与纠错量做外积，得到能写回状态的矩阵；Q 再从新状态读取输出。状态更新使用 K/V 和门控，Q 主要负责读取。

外积 $k_te_t^\top$ 的输出是 $128\times128$，而 $q_t^\top S_t$ 的输出是 128 维。状态保存的是动态的 key→value 关联，不是训练权重，也不是一个随便 reshape 出来的 QK 注意力分数矩阵。推理中的这种“更新记忆”没有反向传播，不会修改模型权重。

用一维教学例子检查算术：$S_{old}=0.6,\alpha=0.9,k=1,v=1,\beta=0.5$。衰减后是 0.54，预测是 0.54，误差是 0.46，写入量是 0.23，新状态是 0.77。若直接加整个 V，会得到不同结果。这个例子为解释更新规则，省略了真实模型的维度与归一化。

### 3.6 输出门控与 prefill/decode 执行差别

每个头读出的向量先执行 RMSNorm，再乘 $\operatorname{SiLU}(z_t)$，拼接 32 个头为 4096 维，最后输出投影：

$$
y_t=W_O\operatorname{Concat}[\operatorname{RMSNorm}(o_t)\odot\operatorname{SiLU}(z_t)].
$$

SiLU 门不是严格位于 0～1 的 sigmoid 开关。这里的 Norm 在头的 128 维上进行，也不能与 Block 外部的 4096 维 Norm 混为一谈。

Decode 通常执行一步递推；prefill 可以把块内更新组织成矩阵运算、块间传递状态。数学上有递推关系，不等于实现必须逐 token 启动 Python 循环。这个区别正是 GDN 优化内核有价值的地方。

### 互动图：观察状态怎样被纠错

[单独打开二维 GDN 状态演示](/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=gdn&v=e5788bdda5ce)。它用二维状态显示旧矩阵、衰减、预测误差、外积修正和新矩阵，可调整遗忘与更新系数。二维数值是教学输入，不是某个真实 checkpoint 的激活。

<iframe src="/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=gdn&v=e5788bdda5ce" title="GDN 状态更新二维演示" class="transformer-interactive" data-widget="qwen9b-gdn" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:900px;border:0"></iframe>

## 4. 先做容量账，再选择 TP 或 DP

### 4.1 缓存必须按类型计算

按 90 亿参数、BF16 估算，语言权重约 18 GB，即 16.8 GiB。这是数量级估算，真实加载还受具体权重身份、MTP 和框架实现影响。共享显存上部署两份独立权重时，这一项约翻倍；不能因为看到两个逻辑设备，就假设它们各自独占一套物理容量。

8 层完整注意力的 BF16 KV，每请求每 token 理论为：

$$
8\times2\times4\times256\times2=32768\text{ bytes}=32\text{ KiB}.
$$

占满 4096 个位置时约 128 MiB/请求，16 请求合计约 2 GiB。24 层 GDN 的 FP32 矩阵状态为：

$$
24\times32\times128\times128\times4=48\text{ MiB/请求},
$$

16 请求合计 768 MiB。还要加入卷积缓存、分页对齐、激活、算子工作区、Graph 和投机状态。目标输出 1024 加输入 3072 是上下文容量规划边界；实际生成时最后一个已采样 token 是否再被送入模型，取决于停止位置，不能把估算当作精确分配日志。

这些数字说明单逻辑设备执行一个完整 9B 副本值得测试，但实际能否装下，要经过加载、prefill、捕图与完整输出验证。

### 4.2 一张双芯物理卡怎样被使用

历史交接记录为 4 张 GR308 物理卡、8 个逻辑设备，同卡 .0/.2 共享物理资源，每物理卡记为 56 GiB。后期报告部分环境字段按“每逻辑设备 56 GiB”列出，存在记账口径差异。因此容量以去重后的物理资源和实际可分配结果为准，不能将两个报表数字直接相加为 112 GiB。

9B 使用逻辑设备 2、3。基线 TP2 让它们共同计算一个模型，输入投影和 FFN 按通道/头切分，行并行输出需要汇总。常规布局中 mixer 输出与 FFN 输出都会产生归约位置；具体 collective 次数取决于融合与延迟归约，未通过真实 trace 核对时不宣称固定次数。

一个候选通信时间模型是：

$$
T_{comm}\approx\sum_i(\tau_{launch,i}+M_i/BW_i+T_{wait,i}).
$$

例如 BF16 的 $[16,4096]$ 输出有效载荷仅 128 KiB，反复的小规模归约可能受启动和同步影响。PCIe Gen5 x16、P2P capability 或小 all-reduce 正确返回，并不能证明模型实际路径达到了某个带宽。也不能把图像模型的通信微基准直接当作这次文本实验的测量。

TP1×DP2 则在两个逻辑设备上各放一个完整副本，把不同请求分给它们。每个副本有自己的 KV、GDN 和卷积状态，通常不需要逐层交换隐藏状态，也不进行训练式梯度同步；服务仍有调度、进程协调和共享物理资源的开销。

历史 CPU 记录为双路 EPYC、GPU 归 NUMA 0；9B 入口绑定 CPU 8–15。它固定了一个主机侧条件，但没有独立 A/B 证明该绑核贡献多少。NUMA、通信实现和拓扑应分别作为实验变量。

## 5. 三个容易混淆的预算

### 5.1 batch token 预算是一轮工作量上限

`max-num-seqs=16` 控制每引擎可运行序列数量；`max-num-batched-tokens=4096` 控制一轮调度的 token 预算。普通 decode 中 16 条请求通常只产生 16 个待计算位置，一条 3072 token 的 prefill 却可能占 3072 个位置。MTP 验证会改变每请求本轮 query 长度。

在启用 chunked prefill、无其他任务且忽略对齐等约束时，3072 token 的 prompt 在预算 2048 下至少拆为 2048+1024 两轮，在预算 4096 下有机会一轮完成。这是上限，调度器不必每轮填满，也不保证一定按这个整数切分。

若已有 16 条普通 decode 请求，本轮预算 2048，在允许混合调度的示意中可先消耗 16，再将最多 2032 的剩余预算用于 prefill。还要满足序列槽位、缓存和后端能力约束。更大预算可能减少 prefill 分块、提高矩阵效率，也可能拉长单轮计算和 decode 等待；TTFT 包含排队，不能只按一个 prefill kernel 推断。

### 互动图：预算怎样改变 prefill 轮数

[单独打开调度预算演示](/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=scheduler&v=e5788bdda5ce)。改变 prompt 长度、已有 decode 数和预算，观察剩余容量与分块。它假设普通 decode，每轮每请求一个位置，不模拟真实 GPU 时间或完整 vLLM 调度器。

<iframe src="/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=scheduler&v=e5788bdda5ce" title="Prefill 与 decode token 预算" class="transformer-interactive" data-widget="qwen9b-scheduler" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:800px;border:0"></iframe>

### 5.2 Graph 桶是一组可回放的执行形状

Graph 捕获操作及其依赖，并用稳定缓冲区重复执行，减少 CPU 重复提交开销。它不缓存旧输出；输入 token、位置和缓存内容每轮仍然变化。捕获需要适配的形状、布局和执行路径，所以为多个规模准备不同桶。

假设普通单 token decode，后端允许向上 padding，实际规模 5 时，只有 [32] 可能补到 32；有 [4,8,16,32] 时可以补到 8，减少额外工作。这里 32 指执行规模，不是历史上下文长度 32，更不是把模型隐藏维度改成 32。

增加 64 桶可以覆盖更大的执行规模，对原本命中 8 桶的请求通常没有直接帮助；额外桶会增加捕获、图对象与缓冲区管理成本，内存池也可能共享，不能简单将各桶内存相加。没有合适且兼容的桶时可能回退 eager，具体由 dispatcher 和后端决定。

MTP 验证中，一个请求可能对应多个 query token。Graph 分派还会考虑 token 数、请求数、query 长度是否一致等信息。C16 不自动等于使用 16 桶，必须查看厂商日志。项目配置 `FULL_DECODE_ONLY` 只捕获符合条件的 decode，prefill/mixed 不应直接套用这一整图路径。

### 互动图：桶越大和桶越多有什么代价

[单独打开 Graph 桶演示](/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=graph&v=e5788bdda5ce)。比较单桶、分级桶与增加 64 桶，查看真实位置、padding 和桶外回退。仅演示普通单 token decode；不按此推算 MTP4 的实际捕图尺寸。

<iframe src="/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=graph&v=e5788bdda5ce" title="Graph 捕获桶与 padding" class="transformer-interactive" data-widget="qwen9b-graph" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:800px;border:0"></iframe>

### 5.3 固定 8 GiB 与自动 KV 是容量确定方式

`--kv-cache-memory-bytes 8589934592` 显式指定 8 GiB 的预算，上游定义是每个 GPU worker 的 KV 预算，不是整个 TP/DP 服务合计 8 GiB。移除该参数后，框架依据显存利用率设置和启动内存评估推导容量。项目代码里“自动”的具体操作就是删除固定字节参数。

自动通常发生在启动阶段，随后从已建立的缓存池给请求分块，并不等于每条请求都临时向系统申请或扩容整个池。固定值被设置时，上游不再依赖 `gpu_memory_utilization` 推导 KV 容量；厂商混合缓存的实际组织仍需检查日志。

假设引擎允许预算 36 GiB、非缓存开销 24 GiB，自动可能规划约 12 GiB，而固定仍按 8 GiB，数字只为解释方式。更多缓存可能容纳更多历史 token、减少等待与抢占；原先容量足够时，不会直接加速 GEMM。Qwen9B 的 GDN/conv、Graph、MTP 都占空间，不能无限扩大缓存。

### 互动图：分开计算权重、KV 和递推状态

[单独打开缓存与拓扑容量演示](/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=memory&v=e5788bdda5ce)。改变总请求数、上下文长度和副本数，查看理论状态预算；另用明确标注的假设引擎预算比较固定与自动 KV。它不模拟厂商分配器，也不判断真实机器必然能装下。

<iframe src="/content/interactive/qwen9b-gr308/qwen9b-lab.html?panel=memory&v=e5788bdda5ce" title="Qwen9B 缓存容量与副本估算" class="transformer-interactive" data-widget="qwen9b-memory" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:900px;border:0"></iframe>

## 6. 瓶颈分析怎样转化为逐步调优

### 6.1 先固定负载与证据口径

固定 BF16、3072→1024、C1/C16、checkpoint、镜像、物理放置和 CPU 亲和性。关闭 prefix cache，避免命中率干扰。记录实际输出长度、failed、抢占、显存、Graph 回放以及 MTP draft/accepted 计数。短测 C1 16 请求、C16 32 请求；本项目最终正式窗口 C1/C16 各 64 请求。

aggregate output tok/s 是全部实际输出 token 除以测试墙钟；报告中的 Decode Avg TPS 为 $1000/mean\ TPOT(ms)$，与平均每请求速率有关。平均 TPOT 的倒数不必等于逐请求速率的平均，更不能直接把它乘并发当作墙钟吞吐。

### 6.2 结构提出假设，剖析与 A/B 决定归因

| 观察或假设 | 优先检查 | 调整方向 | 什么证据才支持保留 |
| --- | --- | --- | --- |
| 模型可能单芯容纳 | 加载、prefill、Graph/MTP峰值 | TP2、TP1、TP1×DP2 | 相同物理资源下完整请求的延迟与吞吐 |
| Decode 可能受发射/同步影响 | CPU间隙、kernel与collective时间线 | Graph、减少层间通信 | 回放生效与端到端收益，不能只看利用率 |
| Prompt被切得过碎 | 每轮token数、prefill块、排队 | 2048/4096预算 | 吞吐和TTFT共同变化 |
| 缓存不足或状态放置异常 | block数、抢占、状态索引/stride | 自动KV、布局修正 | 不再抢占且请求完整；更大池不自动算加速 |
| 投机每轮产出偏低 | 接受长度、draft/verify和状态开销 | MTP1/2/3/4 | 有效提交token/总时间，而非单看接受率 |

单 token 稠密 decode 可以用“每轮读取权重字节、有效 batch 重用与通信成本”建立模型，但只有真实 profiler 与硬件计数器才能确认瓶颈。历史报告的理论屋顶不是利用率实测，不应把“接近 98% 理论单请求上限”写成“GPU 已利用 98%”。

### 6.3 实际历史扫描顺序

早期入口是 TP2、MTP1、Graph 最大桶 32、batch2048、固定 8 GiB KV、同步调度。一次 MTP1→3 对照得到 C16 205.71→231.81 tok/s。之后自适应脚本从 MTP3 候选出发，依次比较 Graph32/64、batch2048/4096、固定/自动 KV、MTP2/3/4，最后测试单 TP1 与双 TP1。

这与“先分析拓扑，再介绍 MTP”的讲解顺序不同。不能在复盘中把分析顺序写成真实时间顺序，也不能将各阶段数字拼成同一个受控、单调提升的实验。

| 局部对照 | 已记录结果 | 判断边界 |
| --- | --- | --- |
| MTP1→3 | C16 205.71→231.81 tok/s | 对应阶段正收益 |
| batch2048→4096 | 约215.8→237–239 tok/s | 对应配置有效，不与其他提升率相乘 |
| 固定8 GiB→自动KV | C16约+0.9% | 小幅变化，不能擅自归因为消除抢占 |
| MTP3→4 | C1改善较明显、C16解码小幅提高 | 最终采用4，不意味着每轮必收4个候选 |
| Graph32→64 | 无足以改变决策的稳定收益 | 保留[4,8,16,32] |
| TP2→TP1×DP2 | 拓扑短测157.67→247.87，+57.2% | 支持副本方案，不等于通信时间减少57.2% |

MTP 的目标近似为 $N_{committed}/(T_{draft}+T_{verify}+T_{state}+T_{runtime})$。接受率高而验证慢，仍可能没有净收益；拒绝候选后，GDN/conv 的状态提交也必须正确。`mtp_num_hidden_layers=1` 描述模块层数，运行参数 MTP4 描述 speculative token 数，两者不矛盾。

### 6.4 正式窗口保留了短测收益

最终选用 TP1×DP2、MTP4、Graph [4,8,16,32]、batch4096、block-size800、自动 KV、同步调度。外部 C16 是总并发，不代表每个副本各 C16。

| 测试 | 聚合输出吞吐 | 请求数/失败 | 解释 |
| --- | ---: | --- | --- |
| 最终 C1 | 48.62 tok/s | 64/0 | 一个活跃请求不能利用双副本的总容量 |
| 最终 C16 | 247.16 tok/s | 64/0 | 正式测试窗口结果 |
| 后续 C32 | 287.02 tok/s | 64/0 | 原始JSON保留，仍为短测 |

247.87 是拓扑短测，247.16 是正式结果。C32 的接受率约 83.21%、接受长度约 4.326，只属于这次 C32 样本，不能当作 C16 接受率。正式窗口证明该负载完成，不代替长期服务稳定性或业务质量验收。最终 C16 TTFT 约 3.50 秒；与外部参考表比吞吐更好而 TTFT 更高，参考表不是本次严格控制变量的前后对照。

### 6.5 当前最值得继续检查的方向

TP1×DP2 已减少逐层 TP 汇总，后续应先统计两个副本各自的请求数、队列时间、prefill/decode占比、空闲时间和尾延迟，再决定是否比较内部 DP 与双端口轮询。固定总 C16 时可以设计 8+8 与其他分配的对照，但当前没有证据证明已完成该比较。自动 KV 冷启动重复、MTP3/4 配对复测和跨卡重复也应单独登记，不能把计划写成已交付收益。

## 7. 把主会话的其他模型放回同一分析框架

主会话先核对了简历和整个 GR308 项目，侧边讨论进一步聚焦9B。跨模型最有价值的共同点，是每种选择都由实际瓶颈决定，而不是复制9B的开关。

| 分支 | 收敛路线与实测 | 能迁移的分析方法 |
| --- | --- | --- |
| Qwen30B-A3B | TP2、关闭EP、async；C16正式225.67，C32短测320.90 tok/s | 专家通信与调度成本可能抵消计算节省，必须比较无EP |
| Qwen35B-A3B | TP2/MTP4/batch4096；C16短测180.64 | MTP与预算联调；最佳D组不同于较早正式脚本中的MTP3 |
| QwenNext80B-A3B | TP4/MTP3；C16短测255.93 tok/s，占两物理卡 | 更大模型要先满足容量，再检查投机与跨卡同步；C16原始独立入口尚未完全确认 |
| Qwen-Image-2.1 | 1024²/4步DiT TP2：163.33→89.60秒 | 真正并行每层与16+16层串行切分不同；1.82倍只属于DiT短测 |
| Qwen-Image VAE | 串行瓦片154.70→双芯143.50秒 | 整图1241.47到瓦片143.50包含算法变化，不能全部归因于扩芯 |
| Wan2.2 | S2混合精度：1190.20→672.33秒 | 先发现VAE约占96.2%，再逐模块扩大FP16；有效5帧/3步、首帧PSNR39.36dB，121帧未成功 |
| H3原生W4A8 | 基础算子通过，原生门未通过 | 输出在fant且数值正确，不证明走原生kernel；没有完整模型吞吐 |
| 早期27B | 后续受控BF16基线约53.6，修正早期18–19 tok/s判断 | 基线环境、冷启动顺序与生命周期可以改变结论；MTP1冷C16两次死锁不能忽略 |

图像 Block 的 RMSNorm/RoPE 组合约6.98%耗时下降只验证到单Block，默认关闭；不能叠加到DiT的1.82倍上。H3的原生W4A8门未过，与博客其他历史阶段出现的不同精度/解码实验不是同一条验收链，必须按各自时间与路径阅读。

## 8. 我主动问了什么，哪些理解还没有被作答验证

这次可见记录中，我主要在追问和听解释，没有留下逐题独立作答。因此不能将下面的缺口写成“我答错过这些题”，更不能编造一次模拟面试。更准确的记录是：这些问题暴露了需要建立的连接，是否掌握还要靠脱离笔记的复述和演算验证。

| 我的原问题或追问 | 当时需要澄清的连接 | 完整回答应当覆盖 |
| --- | --- | --- |
| 先从Qwen9B模型结构入手 | 型号与执行模块 | 稠密FFN、24GDN+8Attention、文本路径与视觉塔区别 |
| 详细说说GDN这个过程 | 投影、局部特征、状态更新怎样相连 | QKVZ/AB、因果卷积、门、纠错、读出与输出投影 |
| 状态矩阵是每一层保存一个吗 | 状态的所有权与数量 | 每请求×每GDN层×每V头；另有conv state，8层保存KV |
| 因果卷积主要算什么，状态矩阵通过什么算出来 | 固定参数与动态缓存 | 训练好的卷积核处理近邻；K/V与门递推更新S；Q用于读取 |
| 针对模型、GPU和通信怎样一步步调优 | 结构分析怎样变成实验 | 容量账、拓扑、通信模型、固定负载、局部消融和正式复测 |
| batch token预算是什么意思 | 请求数、token数与调度轮 | 3072输入在2048预算下分块；混合decode占用预算；不是4096请求 |
| Graph桶不同形状有什么区别 | 捕获模板与真实batch | padding、捕获成本、上下文与桶区别、MTP query长度 |
| 固定8 GiB改为自动是什么意思 | 缓存池容量与分配时机 | 每worker预算、启动profile、池内分页、容量收益不等于kernel加速 |

还有几处应当修正的是讲解口径，而非我的既有作答：讲解先谈拓扑不代表历史先做拓扑；常规Graph例子不能直接用于MTP4；自动KV不是动态无限扩容；53.6的后续27B基线不能被旧19的结果覆盖；每逻辑设备的容量字段不能直接相加。把这些修正保留在笔记中，比只保留最后一个简洁答案更有用。

## 9. 脱离笔记后的自测与参考回答

### 问题一：写出一层GDN的两类缓存，并说明请求换序后怎么办

参考回答：conv state保存短卷积历史，recurrent state保存各头的key→value映射。状态按请求和层独立，batch换序后通过槽位映射找到原请求状态，不能按当前batch下标盲目覆盖。新请求的槽位需要正确初始化，结束后释放。

### 问题二：为什么状态是128×128，完整注意力却有S×S分数

参考回答：$S_{GDN}$的轴是key/value特征维，外积$ke^\top$得到128×128；完整注意力$QK^\top$的轴是query/key位置。二者的轴语义不同，不能因都称“矩阵”就直接替代。

### 问题三：一条3072输入在2048预算下会发生什么

参考回答：启用chunked prefill且无其他约束时至少两轮，可能2048+1024；若同轮有decode则可用预算减少。4096有机会一轮完成，但实际还受缓存、序列槽位和策略限制。

### 问题四：五条decode请求为什么可能执行八个位置

参考回答：普通单token decode中，为复用8桶Graph可向上padding；只有32桶时可能补到32。MTP验证的query长度可能大于1，实际分派需看运行时描述与日志，不能直接套“五请求=五位置”。

### 问题五：缓存加倍为什么可能完全不加速

参考回答：容量只在原先导致等待、抢占、重算或限制并发时带来相应收益。没有这些问题，单次算子计算不因池子更大而变快。更大池还可能压缩工作区与Graph余量。

### 问题六：9B的57.2%到底说明什么

参考回答：同一拓扑短测中聚合吞吐从157.67到247.87，说明该配置下双TP1副本优于TP2。它不是通信时间减少57.2%，最终正式值247.16也不是那次短测的原始数。请求长度、CPU、MTP、Graph和输出完整率都要随实验记录。

### 问题七：MTP4为何不等于模型有四层草稿模块

参考回答：模型MTP层数与运行时提出多少候选是不同配置。收益需要有效提交token覆盖draft、verify及状态提交成本；拒绝候选后必须保持主路径状态正确。

### 问题八：怎样证明自己找到了通信瓶颈

参考回答：先提出包含启动、传输和等待的时间模型，再看真实kernel/collective时间线，并做相同负载的拓扑对照。P2P能力、链路标称带宽、GPU利用率或单个小通信测试都不足以证明整模型瓶颈。

## 10. 来源与证据范围

基础结构参考 [Qwen3.5-9B官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)、[官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json)与[Transformers模型实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。调度与Graph概念参考 [vLLM分块prefill说明](https://docs.vllm.ai/en/stable/configuration/optimization/)、[vLLM 0.18 Graph设计](https://docs.vllm.ai/en/v0.18.0/design/cuda_graphs/)及[缓存参数定义](https://docs.vllm.ai/en/v0.19.1/api/vllm/config/cache/)。当前上游文档用于解释参数语义，不替代历史厂商实现的实际日志。

历史性能和扫描顺序参考公开整理的 [GR308 Lab](https://github.com/KunyuanComputing/gr308-lab) 中四模型总结、指标CSV、`gr308_adaptive_short_matrix.py` 与 `gr308_final_high_throughput.py`，以及本地原始报告的交叉核对。本文没有重新运行GR308性能实验，没有将计划中的跨卡、多seed、长稳测试写成已完成事项。

可见主会话完成各模型梳理，侧边会话继续讨论9B结构、GDN与三个预算。核对时，会话对应的可视化目录没有可复用原文件；本文互动图据这些讨论重新制作，用于教学，不声称是原图的逐像素恢复。所有图的假设与数值范围均在图中标明。

<script src="/js/transformer-interactives.js" defer></script>
