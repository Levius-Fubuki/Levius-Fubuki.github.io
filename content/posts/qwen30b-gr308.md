# Qwen3-30B-A3B 推理调优笔记：从 GQA 与 MoE 到 GR308 瓶颈闭环

Qwen3-30B-A3B 的调优不能只记住「TP2、关闭 EP、Graph、batch4096、async」几个开关。要解释这些选择，需要先知道一个 token 如何经过全注意力与八个专家，权重和 KV 怎样落在双芯显存里，再用实验回答：这一步究竟在等计算、等数据、等通信，还是等 CPU 准备下一批任务？

本文用同一条请求贯穿基础概念和工程过程：3072 token 输入、1024 token 输出，BF16 权重，一张 GR308 双芯卡运行 TP2，重点比较 C1、C16，补充 C32。性能记录发生在 2026 年 9 月；本次笔记整理于 2026 年 11 月 2 日，没有重新运行 GPU 实验。官方结构、历史测量、教学估算和待验证假设会分别标明。

本篇承接 [GR308 语言模型总览](/2026/10/02/gr308-language/)。9B 的 GDN 与双副本路线见 [Qwen3.5-9B 专题](/2026/11/01/qwen9b-gr308/)，图像与视频的阶段优化见 [GR308 图像模型](/2026/10/02/gr308-image/) 和 [GR308 视频模型](/2026/10/02/gr308-video/)。不同模型的结果用于比较方法，不能拼成一次共同的加速实验。

## 1. 用一条请求串起整个知识链

请求到达后，CPU 调度器先检查序列槽位与缓存容量。Prefill 把 prompt 中的 token 送入模型，建立各层 KV；第一次输出通常由最后一个 prompt 位置的 logits 采样得到。后续普通 Decode 每个活跃请求每步输入一个新 token，读取历史 KV，再预测下一个 token。请求完成后释放缓存资源。

```text
3072 token prompt
→ 请求队列与容量检查
→ token 预算决定本轮处理多少 Prefill
→ Embedding
→ 48 层「GQA 全注意力 + Top-8 MoE」
→ Norm、LM Head、采样
→ 第一个输出 token
→ 多轮 Decode：读写 KV、路由专家、执行 TP 通信
→ 达到停止条件，释放请求资源
```

C16 表示客户端最多保持 16 条请求在途，不保证每一步都正好执行 16 条。本文在解释普通 Decode 时把「本步 token 数」写成 $T$；若这一轮有 13 条请求各生成一个 token，$T=13$。它与客户端并发上限、prompt 长度、token 预算是不同变量。

### 互动图：从请求到下一枚 token

这组演示根据本次讨论重新整理，不是恢复出的历史 GPU trace，也不执行真实模型。选择 Prefill、Decode、TP 通信和状态释放，可看到每一步读取与更新的对象。

[单独打开请求链路演示](/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=chain&v=970d8157da13)。

<iframe src="/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=chain&v=970d8157da13" title="Qwen30B 请求与执行链路" class="transformer-interactive" data-widget="qwen30b-chain" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:800px;border:0"></iframe>

## 2. 模型结构：全注意力、GQA 与 MoE 分别是什么

| 参数 | 数值 | 解释 |
| --- | ---: | --- |
| 总参数量 | 约 30.5B | 部署时要考虑全部权重 |
| 每 token 激活参数量 | 约 3.3B | 按官方模型口径，不是实际显存读取字节数 |
| Decoder 层数 $L$ | 48 | 各层拥有独立权重 |
| 隐藏维度 $D$ | 2048 | 层之间传递的 token 表示维度 |
| Q 头 / KV 头 | 32 / 4 | GQA，每八个 Q 头共享一组 K/V |
| 每头维度 $d$ | 128 | 显式配置，不能用 $2048/32$ 代替 |
| 每层专家数 $E$ | 128 | 48 层分别有自己的专家集合 |
| 每 token、每层选中专家 $K$ | 8 | 每层重新路由，不是整条请求固定八个 |
| 专家中间维度 $F$ | 768 | 单个 SwiGLU 专家的扩展通道 |

结构参数来自 [Qwen 官方模型说明](https://huggingface.co/Qwen/Qwen3-30B-A3B) 和 [config.json](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)。

全注意力描述「当前 token 能关注哪些历史位置」；GQA 描述「Q 头如何共享 KV 头」；MoE 描述「FFN 使用哪些专家」。这三个词不在同一个分类维度。

**这个 30B 模型的 48 层使用因果全注意力，未启用滑动窗口，不使用 GDN。**它的 FFN 为稀疏 MoE。项目中 FLA/GDN 的状态适配属于其他模型路径，不能搬到这个模型身上。

单个 Block 可以写成：

$$
h=x+\operatorname{Attention}(\operatorname{RMSNorm}(x)),\qquad
y=h+\operatorname{MoE}(\operatorname{RMSNorm}(h)).
$$

完整模型为 Embedding、48 个 Block、最终 RMSNorm 和 LM Head。Attention 与 MoE 都在每层执行，稀疏化专家没有省掉注意力计算。

### 2.1 注意力的张量形状

对输入 $X:[T,2048]$，各投影得到：

| 张量 | 总投影维度 | 按头展开 |
| --- | ---: | --- |
| Q | $32\times128=4096$ | $[T,32,128]$ |
| K | $4\times128=512$ | $[T,4,128]$ |
| V | $4\times128=512$ | $[T,4,128]$ |

Q、K 做头维度的 RMSNorm，再应用 RoPE。因果注意力计算可以示意为：

$$
A=\operatorname{softmax}(QK^\top/\sqrt{128}+M)V.
$$

$M$ 屏蔽未来位置。输出拼接 Q 头后为 4096 维，再通过输出投影回到 2048 维。隐藏维度和 Q 投影总宽度不必相等。实现依据见 [Transformers Qwen3-MoE Attention](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)。

### 2.2 Router 和专家内部的 gate 不是一回事

Router 用 $[2048,128]$ 的权重为一个 token 计算专家分数，经 softmax、Top-8 和选中权重归一化后，决定调用哪些专家。

$$
r=xW_r,\qquad p=\operatorname{softmax}(r),\qquad
\operatorname{MoE}(x)=\sum_{e\in\operatorname{Top8}(p)}\widetilde p_eE_e(x).
$$

同一份 token 表示分别进入八个专家，输出加权求和；不是把 2048 维向量切成八份交给八个专家。每个专家内部还有 SwiGLU：

$$
E_e(x)=\left[\operatorname{SiLU}(xW_{gate,e})\odot(xW_{up,e})\right]W_{down,e}.
$$

gate/up 均为 $2048\rightarrow768$，down 为 $768\rightarrow2048$。Router gate 负责「选专家」，专家内部 gate projection 负责门控特征，两者作用不同。

单个专家的主要参数约为 $3\times2048\times768=4.72M$，全部专家约为 $48\times128\times4.72M=28.99B$。一个 token 经过的专家参数约为 $48\times8\times4.72M=1.81B$，再加公共注意力、嵌入与输出等部分，形成官方的激活参数口径。因此不能直接用 $30.5B\times8/128$ 计算总激活参数。

## 3. GR308 的四份账：容量、计算、显存带宽、通信

### 3.1 本机配置与产品规格分开记录

本轮服务器有四张双芯物理卡，共八个逻辑设备；30B 使用同一卡上的逻辑设备 4、5。每逻辑设备报告 57,344 MiB，也就是 56 GiB。CPU 为双 AMD EPYC 9654，主存约 1 TiB。文本实验主要使用 vLLM 0.18.0、vllm_fant 1.1.0.20 和 BF16。

| 资源 | 本次记录 | 可支持的判断 |
| --- | --- | --- |
| 显存容量 | 56 GiB / 逻辑设备 | 做权重、KV、激活及工作区容量预算 |
| 显存带宽 | 报告采用约 1008 GB/s / 逻辑设备 | 硬件配置口径，不能当作应用测量值 |
| FP16/BF16 峰值 | 当前可核验资料未给出数值 | 无法计算可靠的峰值算力利用率 |
| 同卡通信 | 厂商 collective 与 custom all-reduce 路径 | 必须按消息大小、拓扑、运行时验证 |
| CPU 绑定 | 30B 使用 CPU 集合 16–23 | 证明固定了 CPU 集合，不证明完成最优 NUMA 扫描 |

官方现售 GR308S 标注最高 1120 GB/s，GR308S MAX 标注最高 2240 GB/s；这些是相应产品规格。不能将 MAX 的带宽直接赋给实验设备，也不能把两份独立显存看作无通信成本的统一内存池。参考 [GR308S](https://www.fantasyxpu.com/product/gr380s) 与 [GR308S MAX](https://www.fantasyxpu.com/product/gr308smax)。

### 3.2 TFLOPS 是运算速率，不是模型 token 速率

矩阵乘 $[M,K]\times[K,N]$ 通常按 $2MNK$ FLOPs 计数，一次乘加算两次浮点运算。有效 TFLOPS 为：

$$
P_{effective}=\frac{2MNK}{t}\times10^{-12}.
$$

早期基础实验记录过 BF16 矩阵 $M=2048,K=5120,N=1024$，事件中位时间约 1.999 ms，主机墙钟约 5.554 ms。按公式分别约为 10.74 TFLOPS 与 3.87 TFLOPS。这只是该形状、该环境和该计时边界的有效性能，不是 GPU 峰值，也不是 30B 当前层的性能。事件和墙钟差额的原因没有完整闭环，不能全部划为 CPU 调度。

FP32、BF16 矩阵计算、稀疏峰值和 INT8 TOPS 的口径不能混用。当前资料无法核验本机 BF16 峰值，就应保留未知，而不是从显存、A100 比例或宣传中的其他精度反推。

### 3.3 三种数据通路

| 数据移动 | 主要通路 | 常见优化对象 |
| --- | --- | --- |
| GPU 读取本地专家权重、KV | 本地显存 | 复用、布局、缓存、融合、减少临时张量 |
| TP/EP 的 rank 交换数据 | 芯片间传输与 collective | 消息尺寸、固定延迟、rank 等待、分发汇总 |
| CPU 送入输入、取回结果 | H2D/D2H | pinned memory、同步、NUMA、传输粒度 |

模型权重常驻后，不会在每个 Decode 步都从 CPU 搬整套模型。若服务使用 offload，需要重新分析主机传输路径，不能沿用常驻假设。

早期 copy/triad 微基准分别记录约 379.36/350.60 GB/s 的逻辑读写带宽。没有 DRAM 流量计数器，重复数组可能受缓存影响，因此不能用 $379/1008$ 当作已测带宽利用率。

另一轮图像实验在同卡双芯测到：32 MiB BF16 all-reduce 约 1.57–1.58 ms，64 MiB 约 2.68 ms，对应报告的总线带宽约 21.2 和 25 GB/s。这些是不同运行时、不同消息尺寸下的参考，不是本次 30B 的通信剖析。

通信时间常用简化模型：

$$
T_{comm}\approx\alpha+\frac nB.
$$

其中 $\alpha$ 是启动与同步等固定成本，$n$ 为数据量，$B$ 为对应路径的有效带宽。小消息更容易受 $\alpha$ 与 rank 等待影响，不能只拿大消息 GB/s 预测。

## 4. 同一个 Decode 步的容量与计算预算

### 4.1 为什么采用双芯 TP2

30.5B 参数按 BF16 两字节估算，原始权重约 61 GB，即 56.8 GiB；单逻辑设备只有 56 GiB，尚未计入 KV、激活、Graph 与工作区。因此当前 BF16 常驻方案先采用 TP2，将主要权重分到两颗芯片，理想份额约 28.4 GiB/rank。

TP2 后每 rank 通常负责 16 个 Q 头、2 个 KV 头。不开 EP 时，各专家的中间通道从 768 分为两份，每 rank 负责约 384 通道，再合并输出。关闭 EP 不会关闭 MoE，也不会让所有 token 改走同一个专家。

| 切分方式 | rank 0 | rank 1 |
| --- | --- | --- |
| TP2、不开 EP | 128 个专家各自的一部分矩阵 | 128 个专家各自的其余矩阵 |
| 同两 rank 启用 EP2 | 一部分完整专家 | 剩余完整专家 |

EP2 均匀放置时可理解为各持有 64 个完整专家。注意力仍可采用 TP2，因此「TP2+EP2」在这里不是四个设备的乘法。两种方式都分摊专家权重，开启 EP 不会天然再把专家显存减半。标准分片逻辑参考 [vLLM Qwen3-MoE](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/model_executor/models/qwen3_moe.py)，厂商实现可能进一步融合具体步骤。

### 4.2 KV 容量怎么算

按 BF16、48 层、4 个 KV 头、头维度128，未分片时每个历史 token 的原始 KV 大小：

$$
48\times2_{K/V}\times4\times128\times2_{bytes}=96\ \mathrm{KiB}.
$$

TP2 每 rank 两个 KV 头，约 48 KiB/token/rank。C16 每条请求都达到 4096 token 时，理想原始 KV 约为：

$$
16\times4096\times48\ \mathrm{KiB}=3\ \mathrm{GiB/rank}.
$$

运行时还涉及页块、对齐、预留与布局，因此理想值不能替代启动日志。项目固定 8 GiB KV 配置预算，并检查后端报告的并发容量；配置预算通常按设备生效，仍以厂商版本行为为准。

### 4.3 Top-8 为何仍然会读很多专家权重

一个 token 只选八个专家，但不同请求未必选相同专家。假设独立均匀路由，$T$ 个 token 本层触及的不同专家数期望为：

$$
E_{used}=128\left[1-\left(1-\frac8{128}\right)^T\right].
$$

| 本步 token 数 | 总 assignment | 期望触及专家数 | 每个被用专家平均 assignment |
| --- | ---: | ---: | ---: |
| 1 | 8 | 8.00 | 1.00 |
| 16 | 128 | 82.42 | 1.55 |
| 32 | 256 | 111.77 | 2.29 |

这是教学模型，不是实际路由采样。它说明增加 batch 同时有两面：更多 token 可以复用权重，但也可能触及更多专家；小并发下，各专家 GEMM 的 M 维仍很小。激活参数量不能直接视为实际 DRAM 字节数，不能从 A3B 推导「等同于稠密3B速度」。

### 互动图：路由与 TP/EP 的区别

调整本步 token 数和路由偏斜，观察 assignment、被使用专家与负载分布；切换 TP/EP，观察改变的是专家内部通道还是专家所有权。路由示例为固定种子的合成输入。

[单独打开 MoE 路由演示](/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=moe&v=970d8157da13)。

<iframe src="/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=moe&v=970d8157da13" title="Top8 路由与 TP EP 分布" class="transformer-interactive" data-widget="qwen30b-moe" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:950px;border:0"></iframe>

## 5. Graph 桶、token 预算和 async 各改哪一步

### 5.1 Graph 桶是预捕获的执行尺寸档位

Graph 捕获一套 GPU 操作及依赖关系，之后更新输入并回放，减少重复提交成本。它不缓存答案，也不意味着把所有算子融合成一个 kernel。回放通常要求相关形状、内存地址和执行结构满足约束，但缓冲区中的 token、位置、KV 元数据等可以变化。

本轮使用 `FULL_DECODE_ONLY`，捕获尺寸 `[4,8,16,32]`。对未启用 MTP 的普通 Decode，可以近似理解为本步 token/序列槽位档位。采用常见的向上匹配填充方式举例：

| 本步实际 token | 选择桶 | 填充槽位 |
| --- | ---: | ---: |
| 3 | 4 | 1 |
| 7 | 8 | 1 |
| 13 | 16 | 3 |
| 16 | 16 | 0 |
| 25 | 32 | 7 |

13 个请求不用等到凑齐16个才运行，可以按框架规则填充、屏蔽无效槽位。桶不是上下文长度；历史KV由缓存和元数据寻址，不能把所有不同长度请求都理解为同长整段序列。厂商的确切匹配和回退策略以日志为准，表格解释的是常见方式。

Graph 桶过少，可能增加填充或回退；过多会增加捕获时间和内存。这里沿用配置名 `cudagraph`，并不表示 GR308 是 NVIDIA 设备。项目使用厂商后端的捕获回放路径。参考 [vLLM Graph 设计](https://docs.vllm.ai/en/v0.18.0/design/cuda_graphs/)。

### 5.2 三个容易混淆的上限

| 参数 | 实验中的含义 | 不代表什么 |
| --- | --- | --- |
| `max-model-len=4096` | 单请求上下文上限 | 4096个并发请求 |
| `max-num-seqs=16` | 调度序列数量上限 | 每轮一定执行16条 |
| `max-num-batched-tokens=4096` | 一轮调度处理的 token 预算 | C16 Decode 自动产生4096个新token |

3072 token prompt 配合 2048 或4096预算，会影响 Prefill 切块及其与 Decode 的混合。预算变大可能提高局部批量，也可能增加等待。C32 补测必须使用能容纳相应序列数的单独配置，不能把正式 C16 脚本中的 `max-num-seqs=16` 原样当成C32复现命令。

### 5.3 简历中的 async 指引擎异步调度

项目通过 `--async-scheduling` 替换 `--no-async-scheduling`。它让不必等待当前结果的调度、结果处理或后续准备尽量与 GPU 执行重叠，减少 GPU 等待。自回归的数据依赖仍然存在，下一枚token不能跳过必要的前一步结果。

API 的 `async/await`、引擎异步调度和 GPU 多stream是不同层次。开启已有开关并验证收益，不等于实现了新的异步调度器。参考 [vLLM 调度配置](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/config/scheduler.py)。

### 互动图：桶匹配与调度重叠

第一部分显示真实槽位与填充槽位；第二部分比较串行调度和理想重叠。CPU/GPU时长可调，但数值是教学输入，不能当作GR308实测，也不能将重叠收益与Graph收益简单相乘。

[单独打开 Graph 与 async 演示](/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=scheduling&v=970d8157da13)。

<iframe src="/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=scheduling&v=970d8157da13" title="Graph 桶与异步调度演示" class="transformer-interactive" data-widget="qwen30b-scheduling" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:1050px;border:0"></iframe>

## 6. Grouped GEMM 与 auto backend

GEMM 是通用矩阵乘。Router 分组后，不同专家收到的 token 数不同，可以写成多个独立任务 $Y_e=X_eW_e$。Grouped GEMM 将这些大小可能不同的矩阵任务统一组织，争取减少启动开销、改善设备任务分配。专家权重和输出仍然独立，不能当作所有专家共享同一权重。

Triton 在这里是编写 GPU kernel 的语言与编译工具，不是同名推理服务器。Triton grouped GEMM 是一种实现选择。`auto` 则是后端选择策略：让框架/插件按支持条件选择默认路径，它不等于「没有优化」，不保证每次请求现场测速，也可能选择到某种Triton实现。确切内核须查看日志和调用路径。概念参考 [Triton Group GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)。

TP/EP决定权重与任务怎样分布，grouped GEMM决定矩阵任务怎样计算，Graph决定操作怎样提交回放，async决定调度工作怎样重叠。这些可以组合，但应逐项验证，不能把所有开关一起改后再分别宣称贡献。

## 7. 瓶颈分析如何形成实验闭环

先固定checkpoint、镜像、厂商设置文件、卡对、CPU集合、请求长度、seed和计时口径。关闭prefix caching避免命中差异，记录权重与配置身份，确认Graph捕获和输出完整。这里没有原生MTP权重，实验未注入MTP配置；它不是通过MTP得到最终结果的模型。

以下按因果问题整理实验过程，不假装所有实验在一条随机交错的A/B中连续发生。

### 7.1 容量不足：先让双芯TP2可运行

**假设：**单芯BF16权重及执行缓冲难以容纳。**动作：**同卡两逻辑设备TP2，保留KV与工作区。**分析：**容量得以安排，但引入每层通信；后续需要在可运行基线上评估通信代价，不能把「放得下」等同于「最快」。

### 7.2 小并发路径不佳：先补Graph小桶

| 配置 | C1聚合tok/s | C16聚合tok/s |
| --- | ---: | ---: |
| Graph `[32]`，auto，batch2048 | 10.88 | 215.33 |
| Graph `[4,8,16,32]`，其余主要项保持 | 24.39 | 216.77 |

C1约2.24倍，C16约+0.67%。**分析：**补桶是小批次的有效方向；还不能把全部差额归于launch，因为填充和执行路径也可能变化。下一步继续提高C16，需要看调度预算和算子效率。

### 7.3 Prefill组织有改进空间：token预算2048到4096

对应C16从216.77到218.80 tok/s，约+0.94%。继续到6144时吞吐变化不到1%，TTFT却增加约30%–47%。**分析：**4096是本轮折中点，继续加大已出现明显的延迟代价。它不是通用最优值。

### 7.4 调度间隙：同步到async

对应短测C16从218.80到221.52 tok/s，约+1.24%。**分析：**观察到小幅正收益，保留async；差异较小，没有多轮配对统计，不能声称稳定的百分之一提升，更不能据此证明CPU是主瓶颈。

### 7.5 容量还是速度：固定KV到自动KV

C16约+0.9%，C32约−0.35%，没有方向一致的收益。**分析：**固定8 GiB已是可用选择，扩大容量没有显示出明确加速。KV容量不限制吞吐，不等于DRAM访问不受带宽影响。block800到1088同样未见足够收益，保留800。

### 7.6 专家GEMM较碎：强制Triton后端

C32记录auto为320.90，Triton为321.35 tok/s，仅约+0.14%。**分析：**当前候选端到端基本持平，保留auto。可能是内核没有明显变快，也可能是局部收益被路由/重排/合并抵消；需要分阶段时间线才能区分。不能将报告的跨阶段汇总表当成逐项相邻严格配对的完整原始记录。

### 7.7 完整专家会不会更高效：重新比较EP

EP2 linear得到C16 185.86、C32 230.52 tok/s，在报告对应对照下分别回归约16%和28%。**分析：**本轮关闭EP。结构上，完整专家计算的潜在收益需要覆盖token分发、汇总、同步和负载偏斜。当前证据没有将损失精确拆分，也未证明此路径一定使用原生FHCCL all-to-all。

常规未融合TP2的示意中，attention与MoE输出可能分别归约。C16隐藏状态负载约为 $16\times2048\times2=64$ KiB；48层反复调用，小消息固定延迟和rank等待可能很重要。每层两次归约只是说明常见结构，不是厂商实际collective次数的trace测量。不能拿64 MiB大消息带宽直接推断这里的真实耗时。

### 7.8 固定候选，正式复测

最终C16正式配置：BF16，TP2，无EP，无MTP，auto MoE，Graph `[4,8,16,32]`，token预算4096，block800，固定8 GiB KV，async。测试输入3072、输出1024，每组64请求。

| 并发 | 聚合输出tok/s | TTFT中位数 | mean TPOT | 请求完成/失败 | 证据 |
| --- | ---: | ---: | ---: | --- | --- |
| C1 | 25.7611 | 1054.43 ms | 37.8295 ms | 64/0 | 正式窗口 |
| C16 | 225.6720 | 4225.69 ms | 65.6860 ms | 64/0 | 正式窗口 |
| C32 | 320.8978 | 5616.56 ms | 88.2858 ms | 64/0 | 独立短测 |

C32吞吐更高，但单请求等待与解码间隔也更大。按业务延迟要求选并发，而不是机械选择最高TPS。正式窗口通过不等于长期生产SLA，也不证明所有prompt质量等价。

### 互动图：逐轮查看假设、结果与下一步

选择一次实验，可同时看到它针对哪类开销、实际记录是什么、哪些解释仍待验证。时间模型单独标为教学估算；历史性能不会随滑块变化而冒充新实测。

[单独打开瓶颈与实验复盘](/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=bottleneck&v=970d8157da13)。

<iframe src="/content/interactive/qwen30b-gr308/qwen30b-lab.html?panel=bottleneck&v=970d8157da13" title="GR308 瓶颈假设与优化实验" class="transformer-interactive" data-widget="qwen30b-bottleneck" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:1050px;border:0"></iframe>

## 8. 怎样判断瓶颈，而不是只背参数

$$
T_{step}\approx T_{operator}+T_{unhidden\ communication}+T_{gap},\qquad
T_{kernel}\gtrsim\max\left(\frac F{P},\frac D{B_{mem}}\right).
$$

这些是简化分析式，重叠部分不能重复相加。算术强度 $I=F/D$ 是每搬一字节做多少运算，Roofline上界为 $P_{effective}\le\min(P_{peak},B_{mem}I)$。这里的 $D$ 应是同一边界的实际数据流量，不是随手拿激活参数量代替。

| 观察 | 可提出的假设 | 应补的证据 | 不足以直接断言 |
| --- | --- | --- | --- |
| C1补小桶后大幅改善 | 小批次执行/填充/提交开销重要 | 桶命中、padding量、GPU时间线 | CPU launch占了全部差额 |
| 增大并发后吞吐提高 | 权重复用与批处理效率提高 | 实际批次、专家token直方图 | 一定已经打满显存带宽 |
| async小幅改善 | 可重叠调度工作存在 | CPU/GPU关联时间线 | CPU是主瓶颈 |
| EP回归 | 分发汇总或偏斜抵消计算收益 | collective消息、rank等待、专家分布 | 物理链路带宽一定不足 |
| 换GEMM后端持平 | 候选没有净收益 | 真实形状microbenchmark与分阶段耗时 | 专家GEMM完全不值得优化 |
| 自动KV持平 | 扩容不是当前有效方向 | 实际块数、驻留、排队与抢占 | 内存系统不可能是瓶颈 |

下一轮应在同一个30B服务里同步采集：GPU空隙、attention/router/GEMM/combine时间、collective大小与等待、专家token分布、Graph回放和抢占。若先做kernel微基准，要使用实际专家矩阵形状，而非只测很大的正方形GEMM。当前没有这些完整证据，所以结论应停在「已通过配置实验收敛可用方案」，不宣称完成硬件瓶颈唯一归因。

## 9. 指标口径与实验算术

聚合输出吞吐使用：

$$
TPS_{aggregate}=\frac{\sum_i output\_tokens_i}{wall\_time}.
$$

64条请求各输出1024 token，总计65,536 token；若聚合吞吐225.672 tok/s，对应测试墙钟约290.4秒。这是整个测试窗口的产能，不能当作单请求速度。

本报告的Decode Avg TPS按 $1000/mean\ TPOT(ms)$ 计算。C16的65.686 ms对应约15.224 tok/s；它与225.672是不同指标，均值的倒数也不等于每条请求速率的算术平均。不要将Decode TPS再乘并发，冒充统一墙钟聚合吞吐。

| 比较 | 数字 | 正确解释 |
| --- | --- | --- |
| 本地早期纯TP2到最终C16 | 215.33→225.67，约+4.8% | 跨阶段记录比较，不是完整随机交错配对实验 |
| Graph补桶的C1变化 | 10.88→24.39 | 小批次路径改善，主要收益不在C16 |
| 外部参考Decode TPS到最终 | 11.70→15.22，约+30.1% | 指标为每请求解码口径，且系统配置不同 |
| C16到C32 | 225.67→320.90 | 改变了并发与延迟，不能写成免费加速 |
| 理论屋顶比例 | 225.67/800≈28.2% | 规划估算比值，不是硬件利用率计数器 |

脚本核对completed/failed、每条实际输入输出长度、总token数、吞吐算术和分位延迟；固定随机合成输入、ignore_eos用于性能诊断。质量检查须独立进行，不能把「1024 token完整输出」当作语义正确或正常业务分布。

## 10. 会话里的问题，以及怎样回答得更准确

本次可见讨论主要是提问与解释，没有完整的个人作答、评分或追问记录。下面保留真实问题，整理它们涉及的概念边界与可练习的回答；不把示例答案伪装成「我当时答错的原话」。

### 问题一：注意力是全注意力还是GDN？

准确回答：Qwen3-30B-A3B的48层使用因果全注意力，采用32Q/4KV的GQA，FFN是128选8的MoE。GQA描述头共享，GDN是另一类递推状态机制。不能根据同项目中其他模型用了GDN，就给30B添加GDN状态路径。

### 问题二：Graph桶到底是什么？

准确回答：它是预捕获执行形状的尺寸档位。普通Decode可近似理解为本步token槽位。13个真实槽位可以在向上填充策略下用16桶，填充位置按实现屏蔽，不必等待新请求。桶不是上下文长度，Graph不缓存答案，也不等同算子融合。多桶的收益要结合命中、填充和回退验证。

### 问题三：简历里的async就是异步调度吗？

准确回答：这里是vLLM的`--async-scheduling`，尽量重叠CPU调度准备与GPU执行。项目做了开关对照，C16观察到小幅收益；这不等于API异步、不等于GPU多流，也不能说自己实现了新调度器。

### 问题四：Triton grouped GEMM是什么，auto又是什么？

准确回答：前者是用Triton组织一组专家矩阵乘的实现，后者是框架/插件的后端选择策略。auto不等于eager或无优化，可能选到相近内核。强制Triton的端到端差异约0.14%，当前按持平处理，真实路径需要日志核实。

### 问题五：GR308有多少TFLOPS，带宽和通信怎么影响优化？

准确回答：先报本机56 GiB/逻辑设备和报告约1008 GB/s/逻辑设备的配置，再区分本地显存、芯片间collective和主机传输。当前没有可核验的本机BF16峰值TFLOPS；特定矩阵有效TFLOPS不能当峰值。基于模型结构提出低复用、小矩阵或小消息延迟假设，再用匹配形状与同负载实验验证。

### 需要改进的解释方式

只报最终参数，缺少「为什么怀疑这里、改了哪个变量、结果支持到哪」的因果链；只说「MoE是3B计算量」，忽略常驻权重、批量专家并集与路由成本；只说「EP通信慢」，没有区分小消息延迟、rank等待与带宽；把Graph、async、kernel都叫成「GPU加速」，又没有说明它们处于不同层次。这些是回答应补齐的内容，不是本次会话中用户已说过的错误原话。

讲解中也需要修正：早期回复将9B的157.67 tok/s称作TP2基线，但已归档细表把它标为单TP1的F1组，对应双TP1的F2为247.87。该对照不属于30B，也不应用来解释30B的收益。对于EP通信，不能仅凭MoE的概念流程，就断定厂商具体调用了原生all-to-all；图像运行时中alltoall_base不支持的记录也不能无条件推广到另一版本文本后端。

### 一段可以练习的完整回答

「我先确认30B是全注意力加MoE，约30.5B总参数、3.3B激活，48层，每层128选8。BF16权重超过单逻辑芯片的可用空间，所以采用同卡双芯TP2，分别安排权重、KV和工作区。之后从固定3072/1024负载建立C1/C16基线：补齐Graph小桶主要改善C1，token预算4096和async为C16带来小幅收益；自动KV和Triton后端没有稳定净收益，EP2反而回归，因此保留纯TP2。最终C16正式窗口225.67 output tok/s、64请求全完成。当前还没有完整profiler将硬件瓶颈唯一归因，不能声称打满算力或带宽。」

## 11. 放回整个GR308项目看方法

| 方向 | 有效决策 | 证据范围 |
| --- | --- | --- |
| 9B | 单芯副本与DP2、MTP联合选择 | C16正式247.16 tok/s；拓扑候选与最终复测分开 |
| 30B | TP2、关闭EP、Graph多桶、async | C16正式225.67；C32短测320.90 |
| 35B | TP2、MTP4、batch4096 | C16最佳短测180.64，仍需正式多轮复测 |
| 80B | TP4、MTP3 | C16最佳短测255.93；输入长度和卡数不同 |
| 早期27B | 基线环境与冷启动状态必须重新核验 | 旧19 tok/s不能当上限；后续非投机约53.6，MTP冷C16有死锁 |
| Qwen-Image-2.1 | 阶段拆分、DiT TP2、VAE瓦片 | DiT 1024²/4步163.33→89.60秒；非完整长步数E2E倍率 |
| Wan2.2 | 按VAE模块选择混合精度 | 有效5帧/3步1190.20→672.33秒；首帧PSNR39.36dB，非全视频等价 |
| H3 | 数值、设备与原生分派分别检查 | W4A8原生门禁未通过，没有完整性能结论 |

跨模型可复用的是实验方法：先定位阶段与约束，提出可证伪假设，用最小实验筛选，再回到完整负载复测。短微基准、容量修复、整模型吞吐和质量验收分别记录，不将它们相互替代。

## 12. 来源与复核入口

- [Qwen3-30B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B) 与 [官方配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)：结构与参数。
- [Transformers v4.51.3 Qwen3-MoE](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)：Attention、Router与SwiGLU参考实现。
- [vLLM v0.18.0 Qwen3-MoE](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/model_executor/models/qwen3_moe.py)、[Graph设计](https://docs.vllm.ai/en/v0.18.0/design/cuda_graphs/)、[调度配置](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/config/scheduler.py)：框架概念与标准执行结构，不替代厂商trace。
- [Triton Group GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)：分组矩阵乘概念。
- [GR308 Lab公开归档](https://github.com/KunyuanComputing/gr308-lab)：四模型调优报告、结果CSV、运行配置与图像通信测试。历史报告及脚本具有各自版本边界，仓库并非每次远端原始数据的完整镜像。
- [GR308S官方规格](https://www.fantasyxpu.com/product/gr380s)、[GR308S MAX官方规格](https://www.fantasyxpu.com/product/gr308smax)：对应产品规格，不能替代本机配置。

本篇的互动演示是根据讨论重建的教学资产。模型结构与历史表格固定展示；合成路由、Graph填充、理想调度重叠和Roofline计算均明确标记为示意，不生成新的GPU实验结论。

<script src="/js/transformer-interactives.js" defer></script>
