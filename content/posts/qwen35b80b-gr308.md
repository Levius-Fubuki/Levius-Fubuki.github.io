# Qwen35B 与 Qwen80B 推理调优笔记：从混合注意力到 GR308 瓶颈分析

Qwen3.6-35B-A3B 与 Qwen3-Next-80B-A3B 的调优，需要把四件事连起来：一个 token 经过什么结构，完整权重和历史状态占多少显存，模型如何切到 GR308 的逻辑设备上，以及一次调度究竟推进多少有效 token。理解了这些关系，TP2、TP4、MTP4、MTP3、Graph64 和 batch token 预算才有明确含义。

本文以 35B 的一条「3072 token 输入、最多 1024 token 输出」请求为主线，再把相同方法迁移到 80B。结构参数取自官方模型说明与配置；性能来自 2026 年 9 月 GR308 历史实验。本次于 2026 年 11 月 3 日整理基础知识、提问与工程记录，没有重新运行 GPU 实验。教学估算、历史测量和待验证解释分别标明。

本篇承接 [GR308 语言模型调优总览](/2026/10/02/gr308-language/)。9B 的双副本路线见 [Qwen9B 深入笔记](/2026/11/01/qwen9b-gr308/)，纯全注意力 MoE 的对照见 [Qwen30B 专题](/2026/11/02/qwen30b-gr308/)。本文不把不同模型、输入长度和实验阶段的数据拼成一次共同的加速实验。

## 1. 用一条请求串起整个知识链

设想一个 3072 token 的 prompt 到达 35B 服务。CPU 调度器先检查请求槽位、全注意力 KV、GDN 状态池和执行预算。随后 Prefill 处理输入：Embedding 把 token ID 变成隐藏向量；40 个 Block 交替执行 GDN 或全注意力，每个 Block 还有 MoE；最后经过归一化、LM Head 和采样，得到第一个输出 token。

后续 Decode 每轮把新 token 输入模型，读取和更新历史状态，逐步预测下一个 token。启用 MTP 后，辅助路径先提出候选，目标模型验证后，只提交可接受的前缀，并把缓存与递归状态推进到相同位置。TP2 的两个 rank 需要共同执行相应的分片计算和集合通信。请求完成后，释放它占用的缓存和状态槽位。

```text
3072 token prompt
→ 排队与容量检查
→ 本轮 token 预算决定处理多少输入
→ Embedding
→ 10 组「GDN+MoE，GDN+MoE，GDN+MoE，全注意力+MoE」
→ Norm、LM Head、采样
→ 首个输出
→ 普通 Decode 或 MTP 候选与验证
→ 提交有效 token，同步推进 KV、GDN 和卷积状态
→ 达到停止条件，释放资源
```

这里有四个不同的「数量」：单个请求的输入长度、服务中的在途请求数、一次执行实际处理的 token 数、调度器允许的 token 上限。C16 只表示客户端最多保持 16 条请求在途，不保证每轮都执行 16 条。MTP4 也不表示一次请求必定多产出四个 token。

### 互动图：一个请求经过哪些对象

演示可以切换 35B 与 80B，并逐步查看排队、Prefill、Decode、MTP 验证和资源回收。它是本次知识整理的教学图，不是真实 GPU 时间线。当前可用的会话资料没有保存可直接导出的原互动图，因此这里重新组织讨论中的关键关系。

[单独打开请求链演示](/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=chain&v=56818fa1cd19)。

<iframe src="/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=chain&v=56818fa1cd19" title="Qwen35B 与 Qwen80B 请求执行链" class="transformer-interactive" data-widget="qwen35b80b-chain" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:820px;border:0"></iframe>

## 2. 先读懂模型：混合注意力和稀疏专家

| 结构 | Qwen3.6-35B-A3B | Qwen3-Next-80B-A3B |
| --- | ---: | ---: |
| 总参数 / 激活参数 | 约 35B / 3B | 约 80B / 3B |
| 主干层数 | 40 | 48 |
| 隐藏维度 | 2048 | 2048 |
| GDN / 全注意力层数 | 30 / 10 | 36 / 12 |
| 每层路由专家总数 | 256 | 512 |
| 每 token、每层选中路由专家 | 8 | 10 |
| 共享专家 | 1 | 1 |
| 专家中间维度 | 512 | 512 |
| 全注意力 Q / KV 头数 | 16 / 2 | 16 / 2 |
| 全注意力每头维度 | 256 | 256 |
| GDN QK / V 头数 | 16 / 32 | 16 / 32 |
| GDN K / V 每头维度 | 128 / 128 | 128 / 128 |

参数来自 [35B 官方模型说明](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)、[35B 配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)、[80B 官方模型说明](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)和 [80B 配置](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)。服务器历史 checkpoint 配置没有完整随归档交付，不能把当前官方文件当成历史权重的逐字段证明。

35B 配置的架构标识仍使用 `qwen3_5_moe`，这是实现复用。它带视觉编码器，但本次 GR308 文本实验设置 `--language-model-only`，不能把文字吞吐解释成多模态端到端性能。

### 2.1 四层一组，每层都经过自己的 MoE

两个模型每四层安排三层 Gated DeltaNet 和一层 Gated Attention。GDN 或 Attention 是 token 间的信息混合模块；MoE 是前馈变换模块。每个 Block 还包含归一化和残差连接，各层拥有各自的专家权重。

这里有两种门控，需要区分。注意力门控调节信息混合模块的输出；MoE router 决定一个 token 被送到哪些专家。它们既不共享同一个含义，也不能用一个「gate」概括性能瓶颈。

80B 比 35B 增大容量，主要依靠更多层和更多专家，隐藏维度与单专家宽度相同。两者都叫 A3B，只能说明官方激活参数量约为 3B；不能由此推断延迟相同、访存量相同或只需要存储 3B 权重。

### 2.2 全注意力、GQA 与 KV cache

全注意力层中，当前 Q 访问历史 K/V。16 个 Q 头共享 2 个 KV 头，即每八个 Q 头共享一组 K/V。Q/K/V 投影空间由显式头数和 head dimension 决定，不能机械地用隐藏维度除头数代替配置中的 256。

Prefill 建立历史 K/V；普通 Decode 为新位置追加 K/V，并访问已有缓存。其占用随上下文长度增长。按 BF16、2 个 KV 头、每头 256 维，单层每个历史 token 的 K/V 数据粗估为：

$$2\times2\times256\times2=2048\ \text{bytes}.$$

前一个 2 是 K 与 V，后一个 2 是 BF16 字节数。35B 只有 10 层全注意力，80B 为 12 层。若按 4096 个位置计算，完整模型未分片、无分页开销时，每请求全注意力 KV 分别约为 80 MiB 和 96 MiB。这是教学容量估算，不是每个 rank 的实际分配量。

TP4 超过两个 KV 头时，实际实现可能复制 KV 头或采用其他支持方式，因此不能简单把 KV 估算除以四。应检查运行时布局和日志。

### 2.3 GDN：递归状态、遗忘门与纠错更新

GDN 将历史信息写入持续更新的矩阵状态。输入先经过投影、短因果卷积，再用门控递归计算输出。省略归一化、多头与输出门控，用 $V\times K$ 状态方向写成概念形式：

$$\widetilde S_t=\alpha_t S_{t-1},$$
$$S_t=\widetilde S_t+\beta_t(v_t-\widetilde S_t k_t)k_t^\top,$$
$$o_t=S_tq_t.$$

$\alpha_t$ 控制旧记忆保留多少，$\beta_t$ 控制新信息写入多少，括号内是当前状态对 $v_t$ 的预测误差。它通过递归状态承载历史，不需要在 GDN 层保存所有历史位置的完整 K/V。普通 Decode 中，单请求递归状态和短卷积历史的形状通常固定。

固定形状仍有实际开销。若仅按每层 32 个 V 头、128×128 状态和 FP32 存储粗算，一层约 2 MiB；35B 的 30 个 GDN 层约 60 MiB／请求，80B 的 36 层约 72 MiB／请求。这里没有计入 TP 分配、卷积状态、padding、池管理和 MTP 临时状态，不能作为实际显存准入值。

这也解释了为什么容量评估不能只看 KV：短上下文下，递归状态本身就可能是不可忽略的一项。

### 2.4 MoE：稀疏激活与小 GEMM

一个 token 在每层经过 router，选择 Top-K 路由专家，并执行共享专家分支。输出概念上是所选专家的加权和，加上共享专家分支。每个专家使用 SwiGLU：2048 维输入分别经过 gate 和 up 投影到 512 维，再做 `SiLU(gate) × up`，最后由 down 投影回 2048 维。

只看普通 C16 Decode，若这轮恰有 16 个 token，35B 每层产生 128 个路由 assignment，80B 为 160 个。assignment 是「某个 token 送到某个专家执行一次」，不等于独立请求数。实际 assignment 可能集中在少数专家，也可能分散，必须从路由分布确认。

专家收到的 token 很少时，GEMM 的行数很小。此时排序、分组、dispatch、kernel launch、combine 和同步可能占据明显比例。grouped GEMM 可以组织多个专家计算，但是否更快仍取决于后端、实际形状和数据重排成本。

## 3. 三种预算：并发、每轮 token 与历史状态

本次对话中特别澄清了 `max-num-batched-tokens`。它表示一次调度迭代中，执行 batch 最多处理多少个 token。称它为「调度 token 预算」是为了说明它限制本轮工作量；直接称为「每个执行 batch 的 token 数上限」也正确。

| 概念 | 约束对象 | 不代表什么 |
| --- | --- | --- |
| 客户端 C16 | 最多 16 条在途请求 | GPU 每轮必定执行 16 条 |
| `max-num-seqs` | 调度器的序列数量上限 | 每请求上下文长度 |
| `max-num-batched-tokens` | 一次调度迭代处理的 token 总量 | 整个请求输入加输出的总量 |
| `max-model-len` | 单请求允许的上下文长度 | 本轮必须一次处理的输入量 |
| KV / state pool | 可驻留的历史状态 | 空闲显存可以全部拿来做缓存 |

普通 Decode 中，每个活跃请求本轮通常输入一个 token；Prefill 则可能处理一整段输入。设预算 2048，本轮有 8 条普通 Decode，理想算术余量为 2040 个 Prefill token；真正分配仍受到序列上限、分页对齐、优先级和后端限制约束。

对于 3072-token 的新 prompt，启用 chunked prefill 且本轮预算只有 2048 时，需要分多轮推进。把上限改为 4096，可能让输入不再因这个 token 上限而被迫拆成同样的块，但是否一次完成仍由实际调度决定。MTP 的 draft/verify 计数要查后端，不能把普通 Decode 算法直接套过去。

更大的预算可能减少部分工作切分，也可能增加单轮耗时、临时显存和请求等待。它不会自动提高所有专家的 GEMM batch，更不会保证 TTFT 单调改善。

### 互动图：调整 token 预算和显存粗估

演示把普通调度预算与静态权重容量分开：一侧观察 Decode token 如何占据本轮预算，另一侧比较 35B/80B 在 TP1/2/4 下的权重粗估。容量图不包含全部运行开销，也不会给出未经实测的 TPS 预测。

[单独打开预算演示](/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=budget&v=56818fa1cd19)。

<iframe src="/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=budget&v=56818fa1cd19" title="调度 token 预算与权重容量粗估" class="transformer-interactive" data-widget="qwen35b80b-budget" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:840px;border:0"></iframe>

## 4. 从模型结构走到 GR308 拓扑

本轮环境有四张双芯物理卡，八个逻辑设备；测试脚本解析的每逻辑设备容量约 56 GiB。TP2 使用同一物理卡的两个逻辑设备仍然需要通信。程序不能把双芯卡当作一个无需同步的统一显存池。

BF16 完整权重粗估：35B 为 70 GB，约 65.2 GiB；80B 为 160 GB，约 149.0 GiB。理想均分时，35B TP2 约 32.6 GiB／rank，80B TP2 约 74.5 GiB／rank，80B TP4 约 37.3 GiB／rank。

这给出了初始可行拓扑：35B 从单物理卡 TP2 开始；80B 的 TP2 权重粗估已超过单逻辑设备容量，因此使用跨两物理卡的 TP4。实际准入还要加入复制参数、KV、GDN、激活、Graph、通信缓冲与工作区。

$$M_{rank}\approx M_{weights}+M_{KV}+M_{GDN}+M_{conv}+M_{workspace}+M_{graph}+M_{comm}+M_{runtime}.$$

上述各项可能存在复用或时间重叠，不能把多个独立峰值简单相加。完整权重也未必严格均分。最终应以启动、峰值与完整请求测试确认。

TP、DP、EP、PP 切分的对象不同。TP 切层内张量；DP 放多个副本接不同请求；EP 分配专家并搬运 token；PP 分层并在阶段间传激活。35B/80B 的最终路线使用 TP，后期尝试过 EP，但没有保留。不能把 9B 单芯可容纳时有效的 DP2 原样搬到 80B。

80B 的 TP4 还使层内本地矩阵更窄。在常见列切分示意下，512 个专家中间通道经 TP2 对应 256 个，经 TP4 对应 128 个。本地算量减少，但 collective 和同步不一定同比减少。实际厂商 kernel 的布局仍需核对，不能只凭示意尺寸宣布通信已经成为主瓶颈。

## 5. 第一步：冻结环境，建立可以比较的基线

先固定模型身份、权重精度、MTP 权重证据、镜像 digest、驱动、运行时配置、设备编号和 CPU 核集合。启动脚本保存 `fantHipSettings.cfg` 和完整参数，不把启动环境当作无关细节。

35B 用 3072→1024，80B 用 1024→1024；短测 C1 使用 16 请求、C16 使用 32 请求。关闭 prefix cache，避免命中率改变成绩。保持选定 custom all-reduce 路径，测试过程中不把它与其他参数一起切换。

CPU 绑定诸如 `0-7`、`8-15`、`16-31` 的核集合，是减少线程迁移和竞争的控制措施，不等于已经证明最优 NUMA。NUMA 放置还需核对 CPU、内存和设备关系。communicator 初始化也要控制顺序，避免同时启动多个模型产生额外初始化波动。

一次准入至少确认：权重身份正确，API 返回目标模型；基础输出检查通过；KV 与状态足以驻留目标并发；没有致命运行时错误；Graph 真正捕获或回放；请求完成后资源能释放。

有运行脚本只证明实现了测试流程，不证明所有分支都成功执行。35B 的脚本包含跨卡复测和正式阶段，但 MTP4 最优点目前仍以短测候选记录，不能据脚本名字补出正式验收。

## 6. 第二步：用观测结果决定先改哪里

| 现象 | 首先检查 | 可以安排的实验 |
| --- | --- | --- |
| 启动 OOM | 权重、状态池、工作区、Graph 和通信缓冲 | 固定其余条件，降低或固定缓存预算 |
| 有抢占、请求反复重算 | KV/state 驻留容量 | 记录实际 block/slot 与抢占，验证容量修正 |
| C1 慢、C16 扩展明显 | 固定启动成本、小矩阵、通信延迟 | eager/Graph、MTP 深度配对比较 |
| MTP 接受率高但吞吐低 | draft、verify、状态维护和同步 | 分开计时，比较每轮有效提交量 |
| TTFT 高、Decode 接近 | prefill 分块、排队与调度预算 | 比较 token 上限，观察请求时间线 |
| TP 扩大后无收益 | 本地矩阵缩小、collective 和等待 | 固定模型与负载，比较可行拓扑 |
| GPU 利用率降到零却请求未完成 | host 等待、进程、死锁与固件错误 | 检查调用栈与日志，先恢复正确执行 |
| C64 在算子处报错 | 索引、布局与状态容量 | 构造最小兼容性复现，不记录无效 TPS |

这是诊断入口表，不是根因判定器。GPU 核心利用率高不能证明达到 DRAM 带宽上限；collective API 时间可能包含等待其他 rank；微基准带宽也不能直接替代模型中的通信成本。没有对应计数器和分项计时，就应保留假设。

## 7. 第三步：MTP 和 Graph 联合筛选

MTP 辅助路径提出多个候选，目标模型验证并提交可接受前缀。35B 配置中一个 MTP 隐藏层，与推理参数 MTP4 的四个候选 token 不是同一个数量。80B 的最优候选为 MTP3，也不能据此认为它新增了三个主干层。

评估目标可以写为概念式：

$$R(k)\approx\frac{\mathbb{E}[\text{每轮有效提交 token}]}{T_{draft}+T_{verify}+T_{state}+T_{schedule}+T_{communication}}.$$

这里是辅助理解的成本账本，实际路径可能重叠，计时时要避免重复相加。增大 $k$ 会增加候选，也可能增加拒绝、验证、状态和工作区开销。接受率与平均接受长度还受计数口径影响，框架可能把额外 token 计入某些长度指标。

### 7.1 混合状态模型的 MTP 正确性

全注意力需要管理候选位置的 KV 有效长度；GDN 和短卷积也必须保证正式状态只推进到接受前缀。若四个候选只接受前两个，不能让第 3、4 个候选对应的递归更新继续污染下一轮。

实现可以采用检查点、候选状态、中间状态选择或其他策略；具体方式由后端决定，不能宣称仓库已经统一实现某一种机制。需要检验的结果是：token 提交位置、KV 有效位置与 recurrent state 对应位置一致。

本地 FLA 适配还存在框架与厂商状态布局差异：SGLang 使用 `[N,H,V,K]`，厂商算子使用 `[N,H,K,V]`。适配器按请求索引抽取状态，转置并保证连续，执行后写回状态池。它说明状态语义和布局本身就是兼容工作的一部分。

### 互动图：候选被拒绝后，状态停在哪里

演示采用一条线性的候选序列，选择接受前缀长度，观察「已提交」和「未提交」的位置。它不模拟真实 GDN 数值，也不代表厂商采用树形草稿或某一种具体回滚实现。

[单独打开 MTP 状态演示](/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=state&v=56818fa1cd19)。

<iframe src="/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=state&v=56818fa1cd19" title="MTP 接受前缀与 KV GDN 状态位置" class="transformer-interactive" data-widget="qwen35b80b-state" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:790px;border:0"></iframe>

### 7.2 Graph 需要匹配执行形状

最终路径使用 `FULL_DECODE_ONLY`；编译配置中的 `mode=NONE` 不代表 Graph 关闭，也不能把 Graph 收益叫作全模型编译收益。Graph 通过复用已捕获的执行路径减少重复发射等成本，但不会自动消除所有计算与通信。

应检查 capture sizes、实际 replay 和 eager fallback，而不是只看启动命令。C16 是请求并发，不能直接推出 Graph16；同样不能用 C16×MTP4 就认定 64 桶必然覆盖验证形状。是否包含额外位置、如何 padding 以及 runner 怎样定义 batch 都需要看后端。

## 8. 35B 的实际调优：MTP4 和 4096 token 预算

35B 从 TP2、MTP1 和 Graph32 路线开始，逐步比较 MTP3、Graph 桶、调度预算与 KV。后续 MTP4 矩阵同时放置不同候选到不同物理卡筛选，并保留 MTP3 控制组。

| 对比 | 记录 | 决策 |
| --- | --- | --- |
| MTP3→MTP4，batch2048 | C16 164.38→175.57 tok/s | 保留 MTP4 候选 |
| MTP4，预算2048→4096 | C16 175.57→180.64；TTFT 5593.57→3321.47 ms | 选择 4096 |
| Graph64→128 | C16 总吞吐约 +0.5%，Decode TPS 略降 | 保留到 64 |
| 预算4096→6144 | C16 −0.50%，C32 +0.48%，TTFT 更高 | 不扩大 |
| Triton grouped GEMM＋融合 | C16 167.69，C32 188.63 | 相对候选回归 |
| EP2 linear | C16 167.81，C32 185.50 | 关闭 EP |
| round-robin 放置 | 实际回退 linear | 没有有效 A/B |
| C64 | SSM kernel 因状态索引非连续失败 | 无有效吞吐 |

不同阶段的行不是连续累乘的加速链；MTP4 预算2048和4096的筛选还存在物理卡不同这一变量。它支持下一步复测谁，不足以精确断言全部差异由预算造成。

最终 D 组为 TP2、MTP4、Graph `[4,8,16,32,64]`、预算4096、block1088、自动 KV、同步调度、auto MoE backend。C16 为 180.6351 tok/s，32 请求、0 失败；C32 为 208.0804，64 请求、0 失败，均保留短测分类。

复现要找 `gr308_35b_mtp4_short_matrix.py` 的 `D-mtp4-g64-b4096`。通用正式脚本里的35B仍是较早的MTP3/预算2048，不应拿错配置。

35B 没有因调大 token 预算持续变快，Graph128 也没有稳定优势。此时更合理的研究方向是核对真实专家形状、排序/分组/汇总开销和 MTP verify，而不是继续无边界扩参数。

## 9. 80B 的实际调优：先用 MTP，再处理跨卡成本

80B 的第一轮脚本清楚地排列了 eager无MTP、eager+MTP1、Graph32和Graph64，再尝试自动KV、预算4096与异步调度。最初阶段只建立MTP1能力；后期记录才继续出现MTP2和MTP3，应按时间区分。

| 对比 | C16 记录 | 决策 |
| --- | --- | --- |
| 无MTP→MTP1 | 177.71→231.30 tok/s | 主要正收益 |
| MTP1＋Graph64 | 231.98 tok/s | 用作后续候选 |
| MTP1→MTP2 | 231.98→251.49 | 保留 |
| MTP2→MTP3 | 251.49→255.93 | MTP3 为最佳短测 |
| MTP1下预算2048→4096 | 吞吐接近，TTFT下降 | 可考虑延迟档 |
| 同步→异步 | C16略降，TTFT恶化 | 保留同步 |
| TP4＋EP4 | 233.93，较255.93低约8.6% | 不采用EP |
| TP2×DP2×EP | 设备类型问题后又出现固件同步超时 | 无可用候选 |

MTP2→MTP3 的C16增加约1.8%，但对应C1总输出由36.75升到47.13 tok/s。是否值得仍要以重复实验确认，不能把一次小增幅说成统计显著。

最终候选为TP4、MTP3、Graph `[4,8,16,32,64]`、预算2048、block1088、固定8 GiB KV、同步。自动KV在初始化阶段失败，不能因35B成功就直接照搬。

C16最佳为255.9308 tok/s、32请求、0失败。后续C32原始JSON记录339.2602 tok/s、64/64完成、65536个输出token，接受率约86.09%、接受长度指标约3.58。这些接受率只对应C32窗口。

归档还保留一个复现缺口：80B MTP3的C16独立原始入口尚未确认，报告和CSV保留了结果；已有MTP3入口覆盖C32补测。不能把报告完整等同于所有原始运行材料完整。

## 10. 第四步：为什么 EP 与通信融合没有自动获胜

纯TP将层内权重或中间通道切分，相关行并行输出需要归约。EP把专家分配给不同rank，可能让每个专家在本地保持更完整的矩阵，但要付出token派发、接收、输出汇总和负载不均的成本。

因此应该比较的是两条完整路径：

```text
纯TP：本地专家分片计算＋相应归约
EP：路由＋跨rank派发＋专家计算＋跨rank汇总
```

collective的具体组合由后端决定，不必然等于直接调用某个`alltoall_base`。图像模型一次FHCCL能力探测的结果，不能无条件搬到文本EP运行路径。需要保存实际backend、进程组、专家放置和分派日志。

本轮35B与80B的EP对照都没有赢，支持保留纯TP。要进一步判断损失来自通信传输、rank等待、专家小GEMM还是重排，需要分项profile。仅凭端到端回归无法唯一归因。

同样，`fuse_gemm_comms`、`fuse_allreduce_rms` 在部分候选中启动失败，说明先要处理兼容性，不能把配置名称当成已经获得的融合收益。后期实验保持custom all-reduce开启；早期27B关闭该路径的记录属于不同环境，不应混用。

## 11. 互动实验台：沿历史候选查看证据

实验台只切换已记录的候选数据，不用滑块编造性能曲线。每个点展示并发、吞吐、条件和局限；没有有效结果的路径显示失败状态。图中相邻点不保证构成严格同环境A/B，各步收益也不能相乘。

[单独打开调优证据实验台](/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=evidence&v=56818fa1cd19)。

<iframe src="/content/interactive/qwen35b80b-gr308/qwen35b80b-lab.html?panel=evidence&v=56818fa1cd19" title="35B 与 80B 历史调优候选与证据范围" class="transformer-interactive" data-widget="qwen35b80b-evidence" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:850px;border:0"></iframe>

## 12. 怎样判断优化结果可以交付

aggregate output tok/s是完整测试窗口内实际输出token除以wall time。报告的Decode Avg TPS按`1000 / mean TPOT(ms)`换算。TTFT还可能包含排队和调度等待，不是单个Prefill kernel时间；TPOT是每个输出token的摊销时间，MTP下也不能直接等同于一次verify耗时。

验收应同时检查输出数量、请求完整率、基础正确性、性能和资源生命周期。固定答案检查不等于普遍逐token等价，零HTTP错误也不等于业务质量通过。35B/80B最高点仍是短测候选，后续至少应在相同配置做配对重复，固定物理卡和启动顺序，并报告波动。

优先补齐三类证据：

1. 每轮候选数、接受数、提交数，以及draft/verify/state计时。
2. 实际Graph回放、专家token分布、collective传输与rank等待。
3. 多轮完整请求、冷启动与先C1后C16两种生命周期、结束后的资源释放。

小于3%的单轮差异在本轮按工程持平处理，这是筛选规则，不是统计显著性检验。只有一次失败的路径也不能直接外推成所有驱动和负载的永久限制。

## 13. 从整个 GR308 项目保留下来的方法

主聊天还涉及9B、30B、图像、视频和H3。它们提供的是分析方法的补充，不应混入35B/80B的加速分母。

| 分支 | 已记录结果 | 可迁移的方法 |
| --- | --- | --- |
| 9B | TP2→双TP1短测157.67→247.87；正式C16为247.16 | 先判断单芯能否容纳，再决定是否需要层内通信 |
| 30B | TP2无EP、预算4096、async，正式C16为225.67 | 异步和EP必须逐模型验证 |
| 27B历史 | 早期约19 tok/s后被厂商环境复现的53.574修正 | 启动环境和生命周期属于实验变量 |
| Qwen-Image | DiT短测163.33→89.60秒 | 阶段加速与端到端加速分开；单算子收益不能直接叠加 |
| Wan2.2 | 有效5帧/3步1190.20→672.33秒，首帧PSNR39.36 | 混合精度必须同时验证性能和质量；不外推到长视频 |
| H3 | 原生W4A8门禁未通过 | 输出在设备上、数值正确和原生加速是不同结论 |

27B的KV144→152曾消除抢占并大幅提高吞吐，但在后续无抢占环境强制KV152几乎不改变性能。两段记录并不冲突：容量不足时解决抢占，与容量已充足时继续调整KV，是不同问题。

图像和视频的详细复盘已独立成文，见 [Qwen Image](/2026/10/02/gr308-image/) 与 [Wan/H3](/2026/10/02/gr308-video/)。这些边界提醒我们，先确定瓶颈在哪个阶段，再选择合适的优化对象。

## 14. 本次提问与表达复盘

当前可见会话主要是主动提问、概念澄清和项目梳理，没有完整的模拟面试问答。因此这里不虚构「我曾回答错了什么」。可以确证的是提出了哪些问题，以及哪些表述需要补全条件。

| 实际提出的问题或澄清 | 它指向的知识缺口 | 更完整的表达方式 |
| --- | --- | --- |
| 结合简历与项目，梳理各模型完整调优 | 简历成绩与实现、证据如何对应 | 先报负载和验证范围，再说明修改与对照，不混算各阶段收益 |
| 先从35B、80B的模型结构入手 | 为什么相同A3B采用不同拓扑 | 结合总权重、层数、专家、混合状态与设备容量说明 |
| 针对模型结构和GPU、通信配置，如何逐步分析调优 | 从原理走到实验决策 | 按容量→基线→观测→单变量候选→复测讲，不只列开关 |
| max-num-batched-tokens是每个batch的token最大数 | 上限限定的是哪个batch、哪些token | 一次调度迭代的执行batch token总数上限；普通Prefill/Decode合计，MTP查后端口径 |

最后一项澄清方向是正确的。「每个batch」还可以补上「一次调度迭代」和「实际处理token」两个限定，避免被误解成所有在途请求的输入加输出总量。此前把它称为预算是同一概念的另一种表述，不应把这个澄清记成错误。

### 14.1 容易说得过满的句子

| 容易出现的简写 | 应补上的条件 |
| --- | --- |
| A3B，所以显存只要3B权重 | 激活参数与完整存储不同；普通部署仍要容纳全部专家权重 |
| 两个芯片在一张卡，所以不用通信 | 它们是独立逻辑设备，层内分片仍需同步 |
| batch变大，所以所有专家GEMM变大 | 该参数是调度上限，实际batch和路由分布要观测 |
| MTP接受率高，所以一定快 | 同时看每轮有效提交量与draft/verify/state成本 |
| EP慢，是all-to-all带宽低 | 端到端只确认本轮回归；具体损失需profile |
| Graph64覆盖C16×MTP4 | 并发、候选数、执行形状和桶定义不能直接相乘替代日志 |
| 64请求成功，所以是正式稳定验收 | 实验类别、重复、负载和生命周期覆盖都要确认 |

### 14.2 一段可以复述的项目回答

> 我先根据BF16完整权重与运行状态估算设备容量：35B从单物理卡TP2开始，80B使用跨两物理卡TP4。然后冻结镜像、运行时、CPU核集合与输入输出条件，分别观察C1和C16。由于模型包含大量GDN层和稀疏专家，我优先比较MTP与Graph，再调整每轮token上限和状态容量，最后比较专家backend、同步调度与EP。35B收敛到MTP4和4096预算，80B收敛到MTP3和2048预算；两者都没有保留EP。最高C16分别为180.64和255.93 output tok/s，目前仍是固定负载下的短测候选。性能方向有记录，细分瓶颈归因与多轮稳定性仍需补齐。

这段回答的后续追问，应能展开到一个具体实验：提出了什么假设、固定哪些条件、观测什么、结果是否超过波动、下一步为何保留或淘汰。这样结构知识和工程经历才真正连在一起。

## 15. 资料与可复查入口

- [Qwen3.6-35B-A3B 官方模型说明](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)与[结构配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)。
- [Qwen3-Next-80B-A3B-Instruct 官方模型说明](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)与[结构配置](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)。
- [GR308 Lab 公开代码与报告](https://github.com/KunyuanComputing/gr308-lab)：README中的四模型结果表、报告索引、35B MTP4矩阵和80B短测入口可以交叉核对。
- [35B MTP4短测矩阵](https://github.com/KunyuanComputing/gr308-lab/blob/main/projects/gr308-three-card-mtp1-graph32/scripts/gr308_35b_mtp4_short_matrix.py)。
- [80B早期短测矩阵](https://github.com/KunyuanComputing/gr308-lab/blob/main/projects/gr308-three-card-mtp1-graph32/scripts/gr308_80b_short_tuning.py)。
- [公共测试启动器](https://github.com/KunyuanComputing/gr308-lab/blob/main/projects/gr308-three-card-mtp1-graph32/scripts/gr308_parallel_goal.py)：用于核对设备放置、精度、Graph和启动环境；代码存在不代表每个实验都完成。

所有互动图都是教学演示；历史性能点保留其负载和证据等级。文章不提供未经测量的组合TPS，不把失败候选或理论容量当成部署验收。

<script src="/js/transformer-interactives.js" defer></script>
