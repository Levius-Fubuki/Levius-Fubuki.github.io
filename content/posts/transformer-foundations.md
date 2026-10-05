# 从 Transformer 张量形状到 MLA：一次 AI Infra 面试复习笔记

准备大模型训练与推理优化面试时，我最先遇到的问题不是记不住公式，而是公式里的轴没有对应到实际计算：多头到底拆了哪个维度，参数量为什么不乘序列长度，Decode 为什么只有一行 Query，缓存更小为什么仍可能多做点积。这篇笔记把这次互动学习中的推导、答错的地方和后续追问放在一起，方便下次从头独立算一遍。

全文采用一个固定的教学配置串起知识链。MLA 部分另设低秩与位置维度，用来比较计算路径；它不是将同一个 GQA 模型直接无损转换成 MLA，也不是某个线上模型的真实配置。

## 1. 用一个配置走完整条链

| 符号 | 教学数值 | 含义 |
| --- | ---: | --- |
| $B$ | 1 | batch 中序列数 |
| $S$ | 100 | Prefill 的序列长度；估算缓存时指已存 token 数 |
| $D$ | 512 | 每个 token 的隐藏维度 |
| $H_Q$ | 8 | Query heads |
| $H_{KV}$ | 2 | GQA 的 KV heads |
| $d_h=D/H_Q$ | 64 | 每个 head 的特征维度 |
| $F$ | 1408 | SwiGLU 中间维度 |
| $L$ | 12 | Transformer 层数 |
| $b$ | 2 bytes | BF16/FP16 每个缓存元素占用 |
| $d_c$ | 128 | MLA 联合 KV 潜变量维度 |
| $d_R$ | 16 | MLA 的共享位置 Key 维度 |

行向量与权重相乘时，权重统一写成“输入维度 × 输出维度”。实际框架可能按相反顺序存储线性权重，阅读代码时需要检查接口约定。

以 Pre-Norm Decoder-only Transformer 为例，完整路径是：

```text
Token IDs [B,S]
  → Embedding [B,S,D]
  → Norm → Attention → 与该子层原始输入残差相加
  → Norm → SwiGLU MLP → 与该子层原始输入残差相加
  → 重复 L 层
  → 最终 Norm → LM Head [B,S,Vocab]
```

Embedding 将编号查表变成向量。Attention 汇总不同位置的信息；MLP 对每个位置分别变换特征。两个子层的最终形状都回到 $[B,S,D]$，才能与各自的残差输入逐元素相加。LM Head 把隐藏特征变成词表 logits，每个位置用于预测下一个 token。

### 互动图：结构与开销

[打开 Transformer 结构与开销互动图](/content/interactive/transformer-foundations/transformer-foundations.html)。可切换前向传播、因果掩码、head 共享、归一化、执行阶段与开销估算。图中的开销面板采用它自己标明的 MHA、SwiGLU 配置，不要把面板默认值与本文教学配置混用。

<div class="transformer-interactive-viewport" style="width:100%;max-width:100%;overflow-x:auto"><iframe class="transformer-interactive" data-widget="transformer-foundations" src="/content/interactive/transformer-foundations/transformer-foundations.html" title="Transformer 结构与计算开销" loading="lazy" style="display:block;width:100%;min-width:320px;height:760px;border:0" referrerpolicy="no-referrer"></iframe></div>

## 2. Attention 的每个轴都对应什么

### 2.1 投影和拆头是两件事

输入 $X:[1,100,512]$。GQA 的投影权重为：

| 权重 | 形状 | 参数数 |
| --- | --- | ---: |
| $W_Q$ | $[512,512]$ | 262,144 |
| $W_K$ | $[512,128]$ | 65,536 |
| $W_V$ | $[512,128]$ | 65,536 |
| $W_O$ | $[512,512]$ | 262,144 |

Q 投影得到 $[1,100,512]$，reshape 为 $[1,100,8,64]$，再交换序列与 head 轴，得到：

$$
Q:[B,H_Q,S,d_h]=[1,8,100,64].
$$

K、V 的投影输出各为 $[1,100,128]$，拆成两个 64 维 head：

$$
K,V:[B,H_{KV},S,d_h]=[1,2,100,64].
$$

这里 $D=H_Qd_h$。KV head 数减少，不意味着每个 KV head 的维度改成 $D/H_{KV}$。GQA 中，每四个 Query heads 共享一组 K、V；每个 Query head 仍有自己的 Query 和匹配权重。共享可以通过分组索引实现，不必真的复制四份缓存。

### 2.2 从匹配分数到加权汇总

先看单个 batch、单个 head：

$$
Q:[S,d_h],\quad K^\top:[d_h,S],\quad QK^\top:[S,S].
$$

行代表 Query 位置，列代表 Key 位置。点积把 $d_h$ 个特征求和成一个分数，所以输出最后两维是位置对 $S\times S$，不是特征对 $d_h\times d_h$。

标准 Attention 写作：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),\qquad O=AV.
$$

缩放在常见初始化假设下控制点积方差，避免分数过大导致 softmax 过度饱和。$M$ 是掩码；softmax 沿最后的 Key 轴执行，每行可访问位置的权重之和为 1。

本例 GQA 的 Attention 权重为 $[1,8,100,100]$。KV heads 可以共享，但八个 Query heads 仍分别产生分数。加权汇总后输出 $[1,8,100,64]$，交换轴并拼接 heads 得到 $[1,100,512]$，再经 $W_O$ 混合各个 head 的信息。

### 2.3 因果掩码允许读取自身

```text
输入 [A, B, C]
Query A → K/V A
Query B → K/V A、B
Query C → K/V A、B、C
```

B 位置预测的是 C，因此读取当前输入 B 不泄漏目标；读取 C 才会。因果关系是“历史和当前位置可见、未来不可见”。训练使用已知完整序列进行 teacher forcing，因此可以并行计算所有位置，但必须遵守这个可见范围。

RoPE 让分数含有位置关系，并不会自动屏蔽未来内容。位置编码和因果掩码承担不同职责。

### 互动图：逐步追踪形状

[打开 Attention 张量形状互动图](/content/interactive/transformer-foundations/attention-shapes.html)。它使用 $B=2,S=3,D=8,H=2$ 的小例子，将投影、reshape、transpose、匹配、掩码、softmax、汇总与拼接拆成十步。

<div class="transformer-interactive-viewport" style="width:100%;max-width:100%;overflow-x:auto"><iframe class="transformer-interactive" data-widget="attention-shapes" src="/content/interactive/transformer-foundations/attention-shapes.html" title="Attention 张量形状逐步推导" loading="lazy" style="display:block;width:100%;min-width:320px;height:760px;border:0" referrerpolicy="no-referrer"></iframe></div>

## 3. 训练、Prefill、Decode：变化的是哪些轴

训练的完整序列已知，前向后还要计算 loss、反向传播和更新参数。Prefill 并行处理 prompt，建立每层的 KV Cache，并通常从最后一个位置的 logits 产生第一个生成 token。

普通自回归 Decode 每条序列每步输入一个新 token，用它预测下一个 token。设此前已有 $S$ 个缓存 token；追加当前 K、V 后，单个 head 的计算为：

$$
Q_{\text{新}}:[1,d_h],\qquad K_{\text{全部}}:[S+1,d_h],
$$

$$
Q_{\text{新}}K_{\text{全部}}^\top:[1,S+1],\qquad AV:[1,d_h].
$$

历史 Query 不重新计算，也通常不需要缓存。新 Query 读取历史与当前的 K、V，然后只追加当前 token 的 K、V。如果估算中将 $S$ 定义为“已经包含当前 token 的可访问长度”，则上述形状写成 $[1,S]$；必须先明确计数口径。

单步 Decode 的核心 Attention FLOPs 随上下文长度线性增长。连续生成很多步时，上下文也在增长，累计工作量需要逐步求和，不能把“单步线性”误解成生成任意长度都只需固定工作量。

普通单步 Decode 通常不存在尚未生成的未来 K、V，未必需要显式的未来位置掩码矩阵，但仍需处理有效缓存、padding 等范围。训练、Prefill、分块 Decode 或多 token 验证中的掩码需求要分别判断。

## 4. 参数、FLOPs、缓存：三个数分别数什么

参数量数的是可学习权重；FLOPs 数的是本次工作负载的运算；缓存字节数数的是运行时要保存的数据。增加 batch 或序列长度会使用同一组权重更多次，不会增加固定架构的参数量。

### 4.1 从矩阵乘法推 FLOPs

$$
[m,k]\times[k,n]\rightarrow[m,n].
$$

输出共有 $mn$ 个元素，每个元素需要一个长度 $k$ 的点积。按一次乘加计 2 FLOPs：

$$
\operatorname{FLOPs}\approx2mkn.
$$

中间维度 $k$ 只乘一次。一个投影处理 $BS$ 个 token 时，还有一个方便记法：

$$
\text{投影 FLOPs}=2BS\times\text{该权重矩阵参数数}.
$$

### 4.2 本例 GQA 与 SwiGLU 参数量

GQA 四个投影的参数为：

$$
P_{\text{Attn}}=2D^2+2D H_{KV}d_h=655,360.
$$

只有标准 MHA 的四个投影都为 $[D,D]$，才可以简化成 $4D^2$。

SwiGLU 有 gate、up、down 三个投影：前两个为 $[D,F]$，最后一个为 $[F,D]$：

$$
\operatorname{SwiGLU}(X)
=\bigl(\operatorname{SiLU}(XW_{\text{gate}})\odot XW_{\text{up}}\bigr)W_{\text{down}}.
$$

前两路逐元素相乘，形状仍为 $[B,S,F]$，不是拼接成 $2F$。最后投影回 $D$ 以便残差相加。三个矩阵元素数都为 $DF$：

$$
P_{\text{MLP}}=3DF=2,162,688.
$$

因此单层主要线性参数为 2,818,048；12 层合计 33,816,576。这个数不包含 Embedding、LM Head、Norm、bias，也不能当作完整模型总参数。若采用普通两矩阵 FFN，参数公式为 $2DF$，不能直接套用 SwiGLU 的公式。

### 4.3 本例整段 Prefill 的前向计算量

| 组成 | 公式 | FLOPs |
| --- | --- | ---: |
| Q/K/V/O 投影 | $2BS P_{\text{Attn}}$ | 131,072,000 |
| $QK^\top$ | $2B H_QS^2d_h$ | 10,240,000 |
| $AV$ | $2B H_QS^2d_h$ | 10,240,000 |
| 三个 MLP 投影 | $6BSDF$ | 432,537,600 |
| 单层合计 | 上述相加 | 584,089,600 |

单层约 584.090 MFLOPs，12 层约 7.009 GFLOPs。这里采用完整稠密 $S\times S$ 的矩阵计算口径，不计 Norm、softmax、门控和词表输出等操作；能跳过未来区域的因果 kernel，其核心计算可能低于这个估算。训练的反向与参数更新也未包含其中。

GQA 减少 KV heads 后，K、V 投影变窄；但 Query heads 不变，因此 $QK^\top$ 和 $AV$ 的上述理论 FLOPs 不按 $H_{KV}/H_Q$ 缩小。实际运行速度还与访存复用和 kernel 有关。

### 4.4 KV Cache 用元素数乘字节数

普通 MHA/GQA/MQA 缓存为：

$$
\text{KV bytes}=2BSL H_{KV}d_hb.
$$

最前面的 2 是 K、V 两份，最后的 $b=2$ 是 BF16 的字节数，两个系数含义不同。

本例 GQA：

$$
2\times1\times100\times12\times2\times64\times2
=614,400\text{ bytes}=600\text{ KiB}.
$$

保持 Query heads、每头维度和其他设置不变：MHA 的 8 个 KV heads 需 2,457,600 bytes，MQA 的 1 个 KV head 需 307,200 bytes。$1\text{ KiB}=1024\text{ bytes}$，$1\text{ MiB}=1024^2\text{ bytes}$；它们与十进制 KB、MB 不同。

缓存不是总显存。权重、激活、临时 buffer、分页元数据、预留块，以及训练时的梯度和优化器状态，需要另外计入。

### 4.5 改参数时怎样增长

| 改动 | 固定架构参数量 | 投影/MLP FLOPs | 整段稠密 Attention 核心 FLOPs | 普通 KV Cache |
| --- | --- | --- | --- | --- |
| $B$ 翻倍 | 不变 | ×2 | ×2 | ×2 |
| $S$ 翻倍 | 不变 | ×2 | ×4 | ×2 |
| $H_{KV}$ 减半，$H_Q,d_h$ 固定 | K/V 投影参数减少 | K/V 投影计算减少 | 不变 | ×1/2 |

## 5. Norm：统计范围、数值不变性与残差路径

### 5.1 每个 token 沿特征轴归一化

对于 $[B,S,D]=[2,3,4]$，常见沿最后一维的 LayerNorm/RMSNorm 独立处理 6 组向量，每组 4 个特征。不是沿 batch 或所有 token 混合统计，输出 shape 不变。

对向量 $x$：

$$
\mu=\frac1D\sum_i x_i,\quad
\sigma^2=\frac1D\sum_i(x_i-\mu)^2,
$$

$$
\operatorname{LN}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i,
$$

$$
\operatorname{RMSNorm}(x)_i
=\gamma_i\frac{x_i}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}.
$$

LayerNorm 减均值并按标准差缩放；RMSNorm 不减均值，按均方根缩放。常见 LN 有 $D$ 个缩放和 $D$ 个偏移参数，常见 RMSNorm 只有 $D$ 个缩放参数，具体实现应以配置为准。

例如 $x=[1,3,5,7]$，均值为 4、方差为 5、均方为 21。取 $\gamma=1,\beta=0$，忽略 $\epsilon$：LN 先中心化再除以 $\sqrt5$；RMSNorm 直接除以 $\sqrt{21}$。LN 此时有正负值，RMSNorm 此时仍全为正。

整体平移 $x\to x+a$ 时，LN 不变，RMSNorm 通常改变。整体正数倍缩放 $x\to ax$ 时，忽略 $\epsilon$，两者都不变；若 $a<0$，归一化部分会翻转符号，不能直接套用正数缩放结论。可学习仿射参数和 $\epsilon$ 也要纳入具体分析。

### 5.2 一次互动图疑问：为什么改缩放后 RMSNorm 变了

我看到“平移 4、缩放 1”对应输入 $[1,3,5,7]$，将缩放改为 0.25 后，图中输入变为 $[3.25,3.75,4.25,4.75]$，便问：“是 $\epsilon$ 的原因吗？”

原因是旧图使用了：

$$
x=a[-3,-1,1,3]+4.
$$

它保持平移量 4 不变，只缩放中心化部分。真正将完整输入乘 0.25，应该得到 $[0.25,0.75,1.25,1.75]$。两种操作不是同一个实验；即使 $\epsilon=0$，旧图的 RMSNorm 输出也会变化。

旧图还随着输入变化自动调整三个图的共同横轴，导致 LN 数值几乎不变时，条形长度看上去改变。现在互动图区分两种变换，并固定归一化输出横轴，可另外切换 $\epsilon$ 检查影响。这个例子提醒我：先核对输入值和坐标轴，再用理论解释图形。

### 互动图：归一化与残差

[打开归一化与残差互动图](/content/interactive/transformer-foundations/norms-and-residuals.html)。在数值视图中切换“先平移，再缩放完整向量”与“先缩放，再加固定平移量”；在路径视图中切换 Pre-Norm、Post-Norm。

<div class="transformer-interactive-viewport" style="width:100%;max-width:100%;overflow-x:auto"><iframe class="transformer-interactive" data-widget="norms-and-residuals" src="/content/interactive/transformer-foundations/norms-and-residuals.html" title="归一化数值与残差路径" loading="lazy" style="display:block;width:100%;min-width:320px;height:760px;border:0" referrerpolicy="no-referrer"></iframe></div>

### 5.3 Norm 类型与放置位置是两个问题

$$
\text{Pre-Norm}:\quad y=x+F(\operatorname{Norm}(x)),
$$

$$
\text{Post-Norm}:\quad y=\operatorname{Norm}(x+F(x)).
$$

这里 $F$ 可以是 Attention 或 MLP，不是前面的中间维度符号。Pre-Norm 残差加回原始输入；Post-Norm 相加后的整体再经过 Norm。Norm 类型可以独立选择，因此可以采用 Pre-Norm 加 RMSNorm。

### 5.4 为什么恒等路径有助于梯度传播，但不是保证

对 $y=x+u$，输出传回的梯度 $g$ 沿 $x$ 的直接旁路贡献仍是 $g$。若 $u$ 也由 $x$ 计算，输入总梯度还要加上子层分支贡献。

用 Jacobian 表示单层结构：

$$
J_{\text{Pre}}=I+J_FJ_{\text{Norm}},\qquad
J_{\text{Post}}=J_{\text{Norm}}(I+J_F).
$$

Pre-Norm 为梯度保留不经过本子层 Norm 的直接路径。Post-Norm 则要求反向梯度先经过输出处的 Norm。多层堆叠时，这种差异会影响训练稳定性；最终输出层等其他变换仍然存在。

普通串联中的梯度变换会连续相乘，若某方向每层乘 0.5，20 层后约为 $10^{-6}$；每层乘 2，则约为一百万。Pre-Norm 的直接路径有助于减少对这些分支变换的依赖，但不能推导出“总梯度一直等于 1”。旁路贡献为 3、分支贡献为 −3 时，总梯度仍为 0；分支贡献很大时也可能爆炸。初始化、学习率和梯度裁剪仍需关注。

“所有特征同时加 1，而 LN 输出不变”的例子说明某些输入变化会被 Norm 消除，不是声称训练总在这个极端状态下运行。LN 会消除统一偏移，但这个方向本身也可能是归一化设计有意忽略的信息，不能用这个例子认定 Post-Norm 无法学习。

## 6. RoPE：位置通过旋转参与匹配

标准 RoPE 将 Q、K 的特征两两分组，根据位置旋转每一对特征。对一对特征：

$$
\begin{bmatrix}x'_1\\x'_2\end{bmatrix}
=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix},\qquad \phi=m\theta_i.
$$

不同特征对使用不同频率。标准旋转保持向量长度；Q、K 的点积中，位置因素表现为相对位置：

$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k.
$$

固定原始 Q、K，将位置 $(m,n)=(2,6)$ 同时平移成 $(5,9)$，位置差仍为 4，所以点积不变。只改变 Key 位置则可能改变分数，但距离增大不保证分数减小：点积是向量长度乘夹角余弦，余弦并不单调；多对特征还要共同贡献分数。

一般不因这个公式宣称模型可以无限外推上下文长度。实际频率、缩放方案、训练长度和实现都影响长上下文表现。

### 互动图：旋转与相对位置

[打开 RoPE 旋转互动图](/content/interactive/transformer-foundations/rope-rotation.html)。固定原始向量，单独移动 Key 位置，再将 Q、K 同时移动一格；分别观察点积是否改变。

<div class="transformer-interactive-viewport" style="width:100%;max-width:100%;overflow-x:auto"><iframe class="transformer-interactive" data-widget="rope-rotation" src="/content/interactive/transformer-foundations/rope-rotation.html" title="RoPE 旋转与相对位置" loading="lazy" style="display:block;width:100%;min-width:320px;height:760px;border:0" referrerpolicy="no-referrer"></iframe></div>

## 7. MLA：内容和位置怎样共同工作

### 7.1 一个联合潜变量同时提供内容 K、V

MLA 通过学习到的下投影生成潜变量，再通过不同上投影生成内容 K、V：

$$
c_j^{KV}=h_jW_{\text{down}},\qquad
K_j^C=c_j^{KV}U_K,\qquad V_j^C=c_j^{KV}U_V.
$$

共享一个潜变量不意味着 K、V 相同。低秩参数化是模型训练得到的结构，不能当作对任意现成 MHA 做无损压缩。

### 7.2 无位置旋转时为何能吸收 Key 投影

对一个 head，用行向量写：$c_j:[1,d_c]$、$U_K:[d_c,d_h]$、$q_t:[1,d_h]$。内容分数为：

$$
q_t(c_jU_K)^\top=(q_tU_K^\top)c_j^\top.
$$

先变换当前 Query $\widetilde q_t=q_tU_K^\top$，就可直接匹配历史潜变量。每个 head 的上投影不同，但固定模型下不随历史位置改变。

### 7.3 为什么内容 Key 上的 RoPE 妨碍吸收

将本轮 Query 自己的旋转算入 $q_t$。如果在展开的内容 Key 上做位置旋转：

$$
k'_j=c_jU_KR_j,\qquad
q_t(k'_j)^\top=q_tR_j^\top U_K^\top c_j^\top.
$$

若仍移到 Query 一侧，变换后的 Query 为 $\widetilde q_{t,j}=q_tR_j^\top U_K^\top$，它取决于被匹配的历史位置 $j$。矩阵结合律仍成立，但不能一般性地用一个固定 Query 变换匹配所有历史潜变量。

最小反例：两个潜变量都为 1，展开后的内容 Key 都为 $[1,0]$；固定 Query 为 $[1,0]$。若两个 Key 分别旋转 0°、90°，点积分别为 1、0。同一个变换后的 Query 与两份相同潜变量相乘却只能给出相同结果，因此位置因素必须另行参与。角度只是教学示例，不是指定模型的频率配置。

### 7.4 解耦 RoPE：位置相关仍然可以批量计算

DeepSeek-V2 的 MLA 把内容与位置两部分的分数相加，再缩放、softmax：

$$
\text{score}_{t,j}
=q_t^C(k_j^C)^\top+q_t^R(k_j^R)^\top.
$$

位置分数仍依赖 $t,j$。历史 $k_j^R$ 在写入缓存时完成旋转，当前 $q_t^R$ 在本轮完成旋转；不是每次匹配一个历史位置都重新旋转 Query。

忽略 batch，本例的位置部分可以一次计算：

$$
Q_t^R:[8,16],\quad K_{\text{缓存}}^R:[100,16],\quad
Q_t^R(K_{\text{缓存}}^R)^\top:[8,100].
$$

“固定投影吸收”与“一次矩阵乘法批量匹配”是两个概念。后者允许每行缓存携带不同的位置特征。

### 7.5 MLA 缓存为何不乘 K/V 两份或 Query head 数

本例每个 token、每层缓存 128 个潜变量元素，加 16 个共享位置 Key 元素：

$$
(128+16)\times2=288\text{ bytes}.
$$

100 个 token、12 层：

$$
BSL(d_c+d_R)b=100\times12\times144\times2
=345,600\text{ bytes}=337.5\text{ KiB}.
$$

内容 K、V 使用同一份潜变量，不额外乘 2；位置 Key 与潜变量都跨 Query heads 共享，不额外乘 $H_Q$。缓存的张量可写为 $C:[B,S,d_c]$ 与 $K^R:[B,S,d_R]$。

这一教学例子中，MLA 缓存比 GQA 的 600 KiB 小，但比 MQA 的 300 KiB 大。不同架构的容量比较不等价于质量比较，更不等价于端到端速度比较。

### 7.6 内容投影吸收省什么，不保证省什么

只缓存潜变量时，若每步重建全部历史内容 Key，开销是：

$$
2S d_cH_Qd_h=13,107,200\text{ FLOPs}.
$$

把变换移到当前 Query，显式执行该变换的开销为：

$$
2H_Qd_hd_c=131,072\text{ FLOPs}.
$$

一个处理历史 100 个 token，一个处理当前一个 token。实际实现还可能把固定投影预先合并到模型权重，具体算子账本会随实现变化。上述比较不是声称普通 MHA 每步重建历史 Key；缓存完整 K 的 MHA 本就能复用它。

Value 侧也可调整顺序：

$$
\sum_j a_j(c_jU_V)=\left(\sum_j a_jc_j\right)U_V.
$$

然而，潜空间点积未必更少：展开内容 Key 的单步点积为 $2H_QSd_h=102,400$ FLOPs，潜变量点积为 $2H_QSd_c=204,800$ FLOPs。本例 $d_c>d_h$，后者反而更大。

MLA 可以减少历史展开与缓存开销，而核心点积 FLOPs 可能增加。Decode 若主要受 HBM 访存限制，合理实现对共享缓存的复用能降低实际读取量，节省的访存时间可能超过额外计算开销。粗略看：

$$
\text{访存耗时}\approx\frac{\text{实际读取字节数}}{\text{有效带宽}}.
$$

静态缓存容量与实际读取流量相关但不相等。batch、布局、重复读取、融合、并行与 kernel 效率都会影响结果，最终必须 benchmark。

### 互动图：缓存、投影吸收、位置分离

[打开 MLA 缓存互动图](/content/interactive/transformer-foundations/mla-cache.html)。逐步查看联合压缩、内容展开、投影吸收、RoPE 分离，以及不同潜变量维度下的缓存对比。

<div class="transformer-interactive-viewport" style="width:100%;max-width:100%;overflow-x:auto"><iframe class="transformer-interactive" data-widget="mla-cache" src="/content/interactive/transformer-foundations/mla-cache.html" title="MLA 压缩缓存与解耦位置" loading="lazy" style="display:block;width:100%;min-width:320px;height:760px;border:0" referrerpolicy="no-referrer"></iframe></div>

## 8. 答错与答得不完整的地方

以下保留我的实际回答或接近原句的表述，再记录纠正后的理解。它们是复习入口，不作为模型已经稳定掌握的证明。

| 问题 | 我的回答/遗漏 | 纠正后的理解 |
| --- | --- | --- |
| 拆头后的 Q | 写成 $[B,S,D,H]$ | $D$ 被拆成 $H_Qd_h$；常用布局为 $[B,H_Q,S,d_h]$ |
| 单步 Decode 的分数矩阵 | 写成 $[S+1,S]$ | 当前只有一个 Query；含自身时为 $[1,S+1]$ |
| 缓存保存什么 | 说保存 q、k | 普通 KV Cache 保存 K、V，通常不保存历史 Q |
| GQA 为什么核心点积不变 | 把“head 数减少”与 $Hd_h=D$ 混在一起 | 减少的是 KV heads；Query heads 保持不变 |
| GQA 的四个投影 | 认为全是 $[D,D]$ | Q/O 为 $[D,D]$；K/V 输出宽度为 $H_{KV}d_h$ |
| K 投影 FLOPs | 列成 $2\times512\times512\times100\times128$ | 多乘了一个 512；应为 $2\times100\times512\times128$ |
| 总投影 FLOPs | 只加一个 Q 和一个 K 的开销 | 还要包含 O 和 V，不能把“乘加系数 2”当作“两个矩阵” |
| SwiGLU 的三个权重 | 认为全是 $[D,F]$ | gate/up 为 $[D,F]$，down 为 $[F,D]$ |
| 因果可见方向 | 说后面的 token 无法看前面 | 后面的可看历史及自身；前面的不能看未来 |
| B 位置可以看谁 | 只回答 A | 还可看 B 自己，因为 B 位置预测下一 token |
| Pre-Norm 梯度优势 | 能复述顺序，但解释不清训练原因 | 旁路提供恒等贡献；分支仍会叠加或抵消 |
| 解释 MLA 性能 | 口误写成 MLP，并说只缓存潜变量 | 是 MLA；还包含共享位置 Key，速度需分析实际访存与计算 |

后续问答中，我能独立写出 GQA 的 Q/K/V/A 形状、KV 缓存列式、四个投影参数和核心点积 FLOPs，也能区分 token 数与权重维度变化的影响。需要持续复习的是它们的条件与边界，不是只记住某一次答对的数字。

## 9. 我主动追问的问题

### “这部分现在掌握如何？”

最初能判断平方与线性增长，但 Decode 形状、Query/KV heads 和参数/FLOPs 边界仍需提示。之后通过同一个 GQA 配置独立列式，才逐步补上。能够听懂推导与能够自行推导是两个阶段。

### “缩放后图怎么变了，是 epsilon 的原因吗？”

先检查变换顺序和输入值。旧图是缩放中心化部分再加固定平移，RMSNorm 改变不是主要由 epsilon 引起；LN 条形长度还受到动态横轴的影响。后来明确区分两种操作，并固定输出图尺度。

### “Pre-Norm 为什么能减缓梯度消失、爆炸？”

恒等路径帮助传递梯度，不能保证总梯度不消失或不爆炸。把旁路与分支的贡献分别写出来，比笼统说“归一化让训练稳定”更准确。

### “RoPE 为什么妨碍吸收上投影？”

因为要移到 Query 一侧的变换含有历史位置相关的 $R_j$；变换后的 Query 也就带上了 $j$，无法只做一次固定变换来匹配全部潜变量。关联顺序可以改变，但不能任意交换矩阵或消掉位置依赖。

### “分离之后，位置匹配不是仍然无法统一吗？”

位置匹配的参数化仍与位置有关，但已经旋转的位置 Key 可缓存；当前 Query 一次旋转后，与整张位置缓存做矩阵乘法。它能批量计算，不要求所有 token 的位置特征相同。

### “内容统一后是不是一定减少 FLOPs？”

它能省去低秩缓存方案中的历史 K/V 展开，但潜空间点积可能更大。要区分“比每步重建历史 Key 更省”和“比完整 K 缓存的标准 Attention 更少 FLOPs”。缓存压缩、算子计算量和端到端性能需要分别衡量。

## 10. 从浅到深自测

先独立作答，再展开每题的参考答案。除第 1 题另给形状外，数值题沿用本文教学配置；FLOPs 按一次乘加计 2 次、完整稠密矩阵乘法估算。

**1. 给定 $[B,S,D]=[2,3,4]$，Norm 计算几组统计，每组多少元素？**

<details class="self-check-answer" id="self-check-answer-1">
<summary>第 1 题参考答案</summary>

沿最后一个隐藏维度计算，所以有 $BS=2\times3=6$ 组，每组 $D=4$ 个元素。每个 token 独立计算自己的均值、方差或均方，不把不同 token 或不同 batch 的元素混在一起。

例如 $X[0,1,:]$ 的四个元素属于同一组；$X[0,2,:]$ 属于另一组。本文只讨论归一化轴为最后一维的配置，若框架显式指定其他归一化轴，统计范围也会改变。

</details>

**2. 整体加常数、整体乘正数，两种 Norm 分别如何变化？epsilon 和仿射参数有哪些边界？**

<details class="self-check-answer" id="self-check-answer-2">
<summary>第 2 题参考答案</summary>

固定同一组 $\gamma,\beta$ 时，LayerNorm 对整体平移 $x+c\mathbf{1}$ 不变：均值也加 $c$，减均值后分子和方差都不变。RMSNorm 不减均值，通常会随整体平移而改变。

对整体正比例缩放 $ax$，忽略 $\epsilon$ 且分母非零时，两种 Norm 的缩放都在分子与分母中抵消。加入固定的 $\epsilon>0$ 后，一般只有近似不变。例如 RMSNorm 的归一化部分为：

$$
\frac{ax}{\sqrt{a^2\operatorname{mean}(x^2)+\epsilon}}
=\frac{x}{\sqrt{\operatorname{mean}(x^2)+\epsilon/a^2}},\qquad a>0.
$$

LayerNorm 同理，把均方替换为方差、分子替换为 $x-\mu$。如果输入尺度很小，$\epsilon$ 的影响不能忽略；负比例缩放还会翻转归一化部分的符号。固定的仿射参数不会破坏已经成立的不变性，但改变 $\gamma,\beta$ 是另一项操作；经过仿射变换的输出也不必均值为 0、方差为 1。

旧图操作的是 $a[-3,-1,1,3]+4$，并非把完整输入 $[1,3,5,7]$ 乘 $a$。因此 RMSNorm 的变化首先来自平移与缩放的操作顺序，不能全归因于 $\epsilon$。

</details>

**3. 为什么 Pre-Norm 的残差加回原始输入？旁路梯度与总梯度有何区别？**

<details class="self-check-answer" id="self-check-answer-3">
<summary>第 3 题参考答案</summary>

Pre-Norm 定义为 $y=x+F(\operatorname{Norm}(x))$：Norm 只在子层分支上，残差分支直接传递该子层的原始输入 $x$。若把 Norm 后的值加回来，就改变了这条恒等路径。

设上游梯度为列向量 $g=\partial\mathcal{L}/\partial y$，则：

$$
\frac{\partial\mathcal{L}}{\partial x}
=g+J_{\operatorname{Norm}}^\top J_F^\top g.
$$

第一项是旁路原样传回的梯度，第二项是子层分支贡献。旁路为 3、分支为 $-3$ 时，总梯度仍为 0；分支很大时，总梯度仍可能很大。因此恒等路径通常有利于深层梯度传播，但不保证总梯度非零或不爆炸。

Post-Norm 是 $y=\operatorname{Norm}(x+F(x))$，其反向梯度为 $(I+J_F)^\top J_{\operatorname{Norm}}^\top g$；连残差方向也要经过 Norm 的 Jacobian。深层训练还需考虑初始化、学习率和梯度裁剪等条件。

</details>

**4. GQA 的 Query/KV heads 数为何分别出现在不同公式中？**

<details class="self-check-answer" id="self-check-answer-4">
<summary>第 4 题参考答案</summary>

每个 Query head 都要独立计算分数和加权结果，因此：

$$
Q:[B,H_Q,S,d_h],\quad A:[B,H_Q,S,S],
$$

$$
\operatorname{FLOPs}(QK^\top)=2BH_QS^2d_h.
$$

共享的是 K、V，所以它们的形状为 $[B,H_{KV},S,d_h]$，缓存字节数为 $2BSLH_{KV}d_hb$，K/V 投影各有 $DH_{KV}d_h$ 个参数。

本例 $H_Q=8,H_{KV}=2,d_h=64$：四个 Query heads 共用一组 KV，K/V 投影宽度是 $2\times64=128$。只减少 KV heads 时，Query heads 和每头维度保持不变，核心点积 FLOPs 不因此按 KV head 比例下降。

</details>

**5. 为什么因果 Attention 可以看自己，而不会泄漏下一 token？**

<details class="self-check-answer" id="self-check-answer-5">
<summary>第 5 题参考答案</summary>

因果掩码允许 $j\le i$。位置 $i$ 已知的输入 token 用于预测位置 $i+1$ 的目标，因此看自身是合法输入，看下一位置才会泄漏目标。

输入 $[A,B,C]$ 时，B 位置可读 A 和 B，用于预测 C；它不能读 C。训练虽然一次提供完整序列，仍要通过掩码保持这一关系。普通单步 Decode 中未来 token 尚未进入缓存，但仍需正确限制有效缓存和 padding 范围。

</details>

**6. Prefill 与单步 Decode 的分数矩阵、核心 FLOPs 怎样变化？**

<details class="self-check-answer" id="self-check-answer-6">
<summary>第 6 题参考答案</summary>

Prefill 同时计算 $S$ 个 Query，分数形状为 $[B,H_Q,S,S]$。按本文稠密口径，$QK^\top$ 与 $AV$ 各需 $2BH_QS^2d_h$，合计 $4BH_QS^2d_h$，随 $S$ 平方增长。

单步 Decode 若已有 $S$ 个历史 token，加入当前 token 后：

$$
Q:[B,H_Q,1,d_h],\qquad A:[B,H_Q,1,S+1].
$$

两个核心矩阵乘各需 $2BH_Q(S+1)d_h$，合计 $4BH_Q(S+1)d_h$，单步随历史长度线性增长。若 $S$ 已经包含当前 token，就写 $[B,H_Q,1,S]$ 与 $4BH_QSd_h$。

本例 Prefill 核心合计 20,480,000 FLOPs；若历史已有 100 个 token，一步 Decode 含当前位置共 101 个，核心合计 206,848 FLOPs。两者都未计投影、MLP、Norm、softmax，也不包含反向传播。

</details>

**7. 从每个输出元素的点积推导 $2mkn$，再算出本文四个投影与 MLP FLOPs。**

<details class="self-check-answer" id="self-check-answer-7">
<summary>第 7 题参考答案</summary>

$[m,k]\times[k,n]$ 产生 $mn$ 个输出，每个输出做长度 $k$ 的点积。按一次乘加计 2 FLOPs，得到 $2mkn$。这里是常用的乘加计数约定；若分别精确数乘法与加法，一个非空点积通常为 $k+(k-1)$ 次。

本例 $BS=100,D=512,H_{KV}d_h=128,F=1408$：

<div class="table-scroll self-check-table" tabindex="0" role="region" aria-label="投影 FLOPs 答案表，可横向滚动">

| 矩阵乘法 | 列式 | FLOPs |
| --- | --- | ---: |
| Q 投影 | $2\times100\times512\times512$ | 52,428,800 |
| K 投影 | $2\times100\times512\times128$ | 13,107,200 |
| V 投影 | 与 K 相同 | 13,107,200 |
| O 投影 | 与 Q 相同 | 52,428,800 |
| 四个 Attention 投影合计 | 上述相加 | 131,072,000 |
| SwiGLU gate/up/down | $3\times2\times100\times512\times1408$ | 432,537,600 |

</div>

MLP 三个矩阵形状分别为 $[D,F],[D,F],[F,D]$，元素数都为 $DF$。加上 $QK^\top$、$AV$ 各 10,240,000，单层主要矩阵乘合计 584,089,600 FLOPs。

不要把点积的中间轴再乘一次，也不要把“乘加系数 2”当成两个投影矩阵；Q、K、V、O 四个投影必须分别计入。

</details>

**8. 为什么 RoPE 同时平移两个位置时点积不变，而距离增大不保证分数减小？**

<details class="self-check-answer" id="self-check-answer-8">
<summary>第 8 题参考答案</summary>

暂用列向量表示，固定旋转前的内容向量 $q,k$ 和旋转频率：

$$
(R_mq)^\top(R_nk)=q^\top R_m^\top R_nk=q^\top R_{n-m}k.
$$

同时给两个位置加 $c$，差值 $(n+c)-(m+c)=n-m$ 不变，所以点积不变。这个结论针对固定内容向量的 RoPE 匹配项，不意味着把真实网络中整个序列移动后，所有隐藏状态一定不变。

旋转保持向量长度，单个二维分量对的分数随相对旋转角呈正弦/余弦变化；多个分量对还会叠加。因此单对 token 的分数可能变大、变小或振荡，不能推出距离越远分数越小，也不能据此保证无限长上下文的外推效果。

</details>

**9. 用矩阵形状推导 MLA 的 Key/Value 投影吸收，并找出直接加 RoPE 后的位置依赖。**

<details class="self-check-answer" id="self-check-answer-9">
<summary>第 9 题参考答案</summary>

回到行向量约定，单个 head 有 $c_j:[1,d_c]$、$U_K,U_V:[d_c,d_h]$、$q_t:[1,d_h]$。内容 Key 为 $k_j^C=c_jU_K$，所以：

$$
q_t(k_j^C)^\top=(q_tU_K^\top)c_j^\top.
$$

先把当前 Query 变成 $[1,d_c]$，再与所有历史潜变量匹配，无需逐条展开历史内容 Key。Value 侧在权重 $a_{tj}$ 已确定后利用线性性：

$$
\sum_j a_{tj}(c_jU_V)=\left(\sum_j a_{tj}c_j\right)U_V.
$$

先在潜空间加权汇总，再做一次上投影；固定的后续输出投影也可与它组合。这是同一低秩参数化内的等价改写，不能宣称任意现有 MHA 都能无损压缩。

若直接令旋转后的内容 Key 为 $k'_j=c_jU_KR_j$，将当前 Query 的旋转记在 $\widetilde q_t$ 中，则：

$$
\widetilde q_t(k'_j)^\top
=\left(\widetilde q_tR_j^\top U_K^\top\right)c_j^\top.
$$

移到 Query 侧的变换含有历史位置 $j$，不同 Key 需要不同变换，不能只对当前 Query 做一次固定变换匹配全部缓存。问题在于位置依赖及不能任意交换矩阵，矩阵乘法的结合律仍然成立。

</details>

**10. 为什么 MLA 缓存不乘 K/V 两份？位置分数如何批量计算？**

<details class="self-check-answer" id="self-check-answer-10">
<summary>第 10 题参考答案</summary>

本文 MLA 方案只缓存一份联合 KV 潜变量 $c_j$，它通过不同上投影提供内容 K、V，另存一份各个 head 共享的旋转后位置 Key。缓存形状为 $[B,S,d_c]$ 与 $[B,S,d_R]$，总字节数为：

$$
BSL(d_c+d_R)b.
$$

因此每个 token、每层为 $(128+16)\times2=288$ bytes；100 个 token、12 层、batch 为 1 时，为 $100\times12\times144\times2=345,600$ bytes，即 337.5 KiB。联合潜变量没有分成 K、V 两份，也没有给每个 Query head 复制一份，所以不额外乘 2 或 $H_Q$。

当前 Query 的位置部分旋转后为 $Q^R:[8,16]$，缓存位置 Key 为 $K^R:[100,16]$，一次矩阵乘：

$$
Q^R(K^R)^\top:[8,100].
$$

每个历史位置的 Key 各自不同，仍然能放在同一个矩阵里批量计算。位置分数与内容分数相加后，按模型定义缩放、掩码并做 softmax；两部分不是各自 softmax 后再相加。

</details>

**11. 如果 $d_c>d_h$，为什么潜空间点积更多，整体 Decode 仍可能更快？**

<details class="self-check-answer" id="self-check-answer-11">
<summary>第 11 题参考答案</summary>

本例取可访问长度 $S=100$：展开后的内容 Key 匹配需 $2H_QSd_h=102,400$ FLOPs，潜空间匹配需 $2H_QSd_c=204,800$ FLOPs。因此潜空间点积本身多一倍。

但端到端时间同时受计算、访存和其他开销影响。在缓存读取受带宽限制、压缩缓存实际减少 HBM 读取且 kernel 能有效复用时，额外计算仍可能小于节省的传输时间。与“每步从潜变量重新展开所有历史 K”的方案相比，吸收还避免了随 $S$ 增长的历史展开；完整 K 缓存的标准 Attention 本来就不做这项重建，不能把这一收益重复算给它。

缓存容量更小也不自动等于读取流量按同一比例减少，需要看 head 复用、batch、布局和 kernel。本例 MLA 缓存 337.5 KiB，小于 GQA 的 600 KiB，却大于 MQA 的 300 KiB；压缩与速度优势必须说明比较对象。没有实际工作负载的测量，结论只能是“可能更快”。

</details>

**12. 若历史长度翻倍，当前 Query 的固定变换与缓存匹配，分别怎样增长？**

<details class="self-check-answer" id="self-check-answer-12">
<summary>第 12 题参考答案</summary>

固定 $B,H_Q,d_h,d_c,d_R$，且每步只处理一个当前 Query。Query 的内容变换 $q_tU_K^\top$ 需 $2BH_Qd_hd_c$，与历史长度无关；内容匹配需 $2BH_QSd_c$，位置匹配需 $2BH_QSd_R$，都随可访问缓存长度 $S$ 线性增长。

本例 $S$ 从 100 增至 200 时：

<div class="table-scroll self-check-table" tabindex="0" role="region" aria-label="历史长度与计算量答案表，可横向滚动">

| 部分 | $S=100$ | $S=200$ |
| --- | ---: | ---: |
| 当前 Query 内容变换 | 131,072 FLOPs | 131,072 FLOPs |
| 潜变量内容匹配 | 204,800 FLOPs | 409,600 FLOPs |
| 位置匹配 | 25,600 FLOPs | 51,200 FLOPs |

</div>

Value 的潜空间加权汇总也随长度线性增长，汇总后的固定上投影不因历史长度增加而翻倍。缓存字节数同样翻倍。如果“历史长度”严格指当前 token 之前的长度 $S$，加入自身后匹配长度从 $S+1$ 变成 $2S+1$，比例为 $(2S+1)/(S+1)$，接近而非精确等于 2。

这是单步 Decode 的增长规律；整段 Prefill 的 Query 数与 Key 数一起增长，核心 Attention 则是平方增长。

</details>

基础题应能独立列式；进阶题应能说清比较对象、缓存策略和适用条件。判断性能时，最后还要落实到可测量的工作负载。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Attention 的原始定义与缩放点积。
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：RMSNorm 的定义与不变性。
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)：Pre-Norm/Post-Norm 的初始化梯度分析。
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)：RoPE 与相对位置性质。
- [DeepSeek-V2 技术报告，第 2.1 节](https://arxiv.org/html/2405.04434v5#S2.SS1)：MLA、联合 KV 压缩、投影吸收与解耦 RoPE。
- [FlashMLA 官方实现](https://github.com/deepseek-ai/FlashMLA)：实际 MLA kernel 的实现入口；性能数字应按其测试配置理解。

本文来自本次面试复习问答、批注与互动实验。所有计算示例均为教学估算，不作为指定职位的固定考题或真实模型性能结论。

<style>
body:has(#transformer-foundations) #post-meta { white-space: normal; overflow-wrap: anywhere; }
#article-container .self-check-answer { margin-bottom: 1.5rem; }
#article-container .self-check-answer > summary { cursor: pointer; }
#article-container .self-check-table { position: relative; max-width: 100%; overflow-x: auto; margin: 1rem 0; }
#article-container .self-check-table table { min-width: 540px; }
#article-container .self-check-table th, #article-container .self-check-table td { white-space: nowrap; word-break: normal; overflow-wrap: normal; }
</style>
<script src="/js/transformer-interactives.js" defer></script>
