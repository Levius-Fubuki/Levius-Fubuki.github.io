# 从 KV Cache 到有效吞吐：大模型推理引擎与服务优化复习笔记

这份笔记整理了“大模型推理引擎与服务优化”模块的互动复习，也合并了侧边会话《Continuous原因解析》中关于调度轮次、batch slot 和 KV 位置的解释。它保留基础概念、计算过程、我的易错回答与主动追问，以及会话中的 18 张互动图。文末自测题均附答案。

本文所有耗时、带宽和吞吐数字均用于教学。它们不是 vLLM、SGLang 或某款 GPU 的实测结果；串行时间线也不能直接当作真实混合 batch 的性能预测。

## 1. 用一个服务案例贯穿整条知识链

假设我要部署一个支持完整因果 Attention 的 Decoder-only 模型。只为计算 KV Cache，使用下面的参数：

| 参数 | 含义 | 主例取值 |
|---|---|---:|
| L | Transformer 层数 | 32 |
| H_Q | Query head 数 | 32 |
| H_KV | KV head 数，主例为 GQA | 8 |
| d | 每个 head 的维度 | 128 |
| s | 每个缓存元素的字节数，BF16 | 2 |
| T | 某请求当前已经缓存的 token 位置数 | 4096 |
| S | 每个 KV 逻辑块可容纳的 token 数 | 16 |
| M_KV | 留给 KV Cache 的显存预算 | 8 GiB |

请求 A 已完成 4096-token prompt 的 Prefill，正在生成。请求 B 与 A 有相同的前 64 个 prompt token；请求 C 是新到达的 2000-token 长 prompt。调度器需要兼顾它们的首 token 等待时间、后续输出间隔和整体吞吐。

一条请求的生命周期可以写成：

```text
请求到达 → 排队 → 查找可复用前缀 → Prefill
         → 采样首 token → 逐轮 Decode → 完成并释放请求状态
```

这条流程自然引出一系列问题：历史计算怎样复用？缓存放不下怎么办？如何给新请求腾出执行位置？长 Prefill 会不会打断正在生成的请求？分离 Prefill/Decode 是否值得？怎样让目标模型一次验证多个 token？最终怎样证明服务变好了？

## 2. KV Cache：新 Q 为什么仍要读历史 K/V

### 2.1 Prefill 与 Decode 的区别

Prefill 一次处理 prompt 中的多个 token，建立各层的 K/V，并通过最后一个 prompt 位置的输出预测首个生成 token。普通 Decode 每轮将一个新位置送入模型，继续预测下一个 token。

假设 prompt 有 100 个位置，Prefill 后采样得到 y₁。下一轮模型处理 y₁：计算它在每一层的 Q/K/V，将新 K/V 加入缓存，再用当前 Q 读取历史以及当前位置的 K/V。此时通常有 101 个有效 KV 位置；随后采样得到的 y₂ 尚未被模型处理，不能提前算作已缓存位置。

<iframe src="/content/interactive/inference-serving/kv-cache-walkthrough.html?v=height-v2" title="KV Cache 与逐 token 推理" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="kv-cache-walkthrough"></iframe>

[单独打开此图](/content/interactive/inference-serving/kv-cache-walkthrough.html?v=height-v2)

### 2.2 为什么通常缓存 K/V，而不缓存历史 Q

单个 attention head 的计算为：

$$
S=\frac{QK^\top}{\sqrt d},\qquad P=\operatorname{softmax}(S),\qquad O=PV.
$$

当前输出使用当前 Q，与已有 K 匹配后对已有 V 加权。历史 Q 是此前位置产生输出时使用的查询，普通逐 token Decode 不需要再次生成那些历史输出，因此一般不保存它们作为后续 Attention 的缓存。

KV Cache 避免重新执行历史位置的前向计算来取得 K/V；当前新位置的 Q/K/V、Attention、MLP 等仍需计算。当前 Q 读取历史 K/V 的工作也仍然存在。

### 2.3 历史 KV 为什么能在追加 token 后复用

在因果模型里，旧位置看不到后来追加的 token。因此，在模型、位置与输入前缀保持一致时，追加一个 token 不会改变已有位置的表示和 KV。

但这不等于“每个 token 的 KV 与其他 token 无关”。第一层的 K/V 投影通常来自该位置的初始表示；后续层的输入已融合此前上下文。因此把前缀中的 b 换成 x，后面的 c、d 即使 token ID 相同，其后续层 KV 一般也会变化。

```text
A：a b c d
B：a x c d

可复用相同前缀 a 的整套 KV；不能因为 c、d 相同就复用它们的整套 KV。
```

如果使用双向 Attention，追加内容也可能改变旧位置表示。上述“追加后旧 KV 不变”的解释依赖因果可见性。

<iframe src="/content/interactive/inference-serving/context-and-kv.html?v=height-v2" title="因果可见性与各层 KV 的上下文依赖" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="context-and-kv"></iframe>

[单独打开此图](/content/interactive/inference-serving/context-and-kv.html?v=height-v2)

### 2.4 Q、K 与得分矩阵的形状

单个 head、一次处理的 Q 位置数为 n_q，总 K 位置数为 n_k 时：

$$
Q\in\mathbb R^{n_q\times d},\quad K\in\mathbb R^{n_k\times d},\quad QK^\top\in\mathbb R^{n_q\times n_k}.
$$

| 场景 | Q 的行数 | K 的行数 | 得分矩阵 |
|---|---:|---:|---|
| 整段 N-token Prefill | N | N | N×N |
| 单 token Decode，总 KV 为 N | 1 | N | 1×N |
| Prefill 一个 c-token chunk，之前有 T 个 KV | c | T+c | c×(T+c)，含因果遮罩 |

主例中，4096 个历史位置后处理一个新 token，得分矩阵是 1×4097，而不是 4097×4097。增加历史长度主要增加列数；只有增加本次处理的 Q 位置数，才增加行数。

## 3. 算清 KV 显存，再讨论并发

### 3.1 公式、单位与边界

对于等长请求、缓存所有层 K/V 的主例：

$$
M=2LBTH_{KV}ds.
$$

开头的 2 表示 K 和 V 两份；s=2 表示 BF16 每元素两个字节。这两个 2 的含义不同。GQA 的缓存使用 H_KV，而不是 H_Q。

主例每 token 的缓存大小是：

$$
2\times32\times8\times128\times2=131072\text{ bytes}=128\text{ KiB}.
$$

于是，4096 个位置的单请求缓存为：

$$
4096\times128\text{ KiB}=512\text{ MiB}.
$$

| 条件 | 理论 KV 数据量 |
|---|---:|
| B=1，T=4096 | 512 MiB |
| B=2，T=4096 | 1024 MiB=1 GiB |
| B=3，T=4096 | 1536 MiB=1.5 GiB |
| B=1，T=8192 | 1024 MiB=1 GiB |

8 GiB 的 KV 专用预算，理论上可容纳 16 个 4096-position 请求，或 8 个 8192-position 请求。不能把 B=2 的合计 1 GiB 当成“每请求占用 1 GiB”。

请求长度不同时，应使用总有效位置数：

$$
M=2LH_{KV}ds\sum_{i=1}^{B}T_i.
$$

这些公式计算逻辑 KV 数据，不包含权重、工作区、激活、元数据、对齐与分页尾部空槽。多卡分片后每卡存储量还依赖具体并行和 KV 复制方式，不能无条件把结果除以卡数。

<iframe src="/content/interactive/inference-serving/kv-memory.html?v=height-v2" title="请求数量、上下文长度与 KV head 数对显存的影响" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="kv-memory"></iframe>

[单独打开此图](/content/interactive/inference-serving/kv-memory.html?v=height-v2)

### 3.2 显存容量与显存带宽

容量决定能放多少数据；带宽决定数据搬得多快。小 batch Decode 常因权重或 KV 读取而受访存限制，但不能只看计算单元利用率就下结论，需要结合带宽、算子耗时与执行空隙分析。

额外做一个权重读取练习：假设每轮读取 14 GB 权重，有效带宽为 2 TB/s，即 2000 GB/s，读取下界为 7 ms；权重降到 7 GB 后为 3.5 ms。这不保证整个 Decode 加速两倍，因为 KV 访问、计算、通信等部分未必同比下降，量化也可能引入反量化开销。这个权重数字是独立教学条件，不是在推导主例模型的参数量。

增大 batch 可以让更多 token 在同一次矩阵运算中使用相同权重，摊薄部分读取和启动开销；同时也会增加计算、KV 读写和显存需求。吞吐可能增加，单请求输出间隔也可能增加。

## 4. PagedAttention：逻辑顺序与物理位置分离

### 4.1 块数、尾部空槽与地址计算

PagedAttention 将 KV 按固定 token 数分块，用 block table 将逻辑块映射到物理块。物理块无需连续，逻辑 token 顺序通过映射保持。

设每块 S 个位置，当前长度 T：

$$
n_{blocks}=\left\lceil T/S\right\rceil,\qquad
\text{空槽}=n_{blocks}S-T.
$$

T=40、S=16 时，需要 3 块，总容量 48 个槽，剩 8 个空槽。这不是整个显存池还有 8 个槽，而是该请求已分配块内的未用槽。

把“第几个 token”和“从零开始的索引”分开：

$$
i=n-1,\qquad L=\lfloor i/S\rfloor,\qquad offset=i\bmod S.
$$

S=16 时：

| 第 n 个 token | 全局索引 i | 逻辑块 | 块内偏移 |
|---:|---:|---:|---:|
| 17 | 16 | L1 | 0 |
| 40 | 39 | L2 | 7 |
| 49 | 48 | L3 | 0 |

假设 block table 为 L0→P7、L1→P2、L2→P11，则全局索引 36 位于 L2，偏移 4，去 P11 读取该位置 KV。第 49 个 token 需要 L3 的新物理块，已有三个块的数据无需搬迁。

<iframe src="/content/interactive/inference-serving/paged-kv-blocks.html?v=height-v2" title="分页分配、释放与物理块复用" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="paged-kv-blocks"></iframe>

[单独打开此图](/content/interactive/inference-serving/paged-kv-blocks.html?v=height-v2)

<iframe src="/content/interactive/inference-serving/token-block-index.html?v=height-v2" title="token 序号、全局索引、逻辑块与偏移" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="token-block-index"></iframe>

[单独打开此图](/content/interactive/inference-serving/token-block-index.html?v=height-v2)

### 4.2 块不是越小越好

较小的块往往减少尾部浪费，但需要更多映射项、分配操作和管理工作。较大的块管理更粗，尾部空槽可能更多。性能要结合访问布局、算子和调度实现评估。

例如 T=17：S=16 时两块、15 个空槽；S=32 时一块、也有 15 个空槽。单个样本并不能证明小块总更省。

<iframe src="/content/interactive/inference-serving/kv-block-size.html?v=height-v2" title="块大小、尾部浪费与映射数量" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="kv-block-size"></iframe>

[单独打开此图](/content/interactive/inference-serving/kv-block-size.html?v=height-v2)

### 4.3 分页不突破逻辑数据量下限

分页减少碎片和按最大长度预留的浪费，不自动减少每个有效位置的 K/V 元素数量。若 KV 预算 8 GiB、12 个请求实际各需 1 GiB，且没有共享、压缩或重计算，仅分页也放不下 12 GiB。FlashAttention 同样不直接缩小这些 KV 数据。

## 5. Prefix Cache：相同前缀跨请求复用

Prefix Cache 缓存此前前缀在各层计算出的 KV。兼容的后续请求可跳过已命中前缀的 Prefill 工作。一般需要一致的模型权重、token ID、位置、适配器和影响缓存语义的配置；文本看着相同，不保证实际 token 化或模型条件相同。

主例 A、B 共享前 64 个位置，每块 16 个位置，可以复用 4 个完整块。每块的数据为 16×128 KiB=2 MiB，所以共享前缀共 8 MiB。B 的 4096-position prompt 在简化的完整块命中条件下还需处理 4032 个位置；引擎为生成首 token 可能还需要边界位置或 logits 相关工作，具体以实现为准。

若 A、B 此刻都缓存 4096 个位置且完全共享这 64 个位置，物理 KV 数据可从 1024 MiB 降到 1016 MiB。逻辑上每请求仍有完整上下文，物理上相同块共享，不能把“共享”理解成某请求缺少历史。

命中主要减少 Prefill。B 生成的新 Q 仍要读取其有效历史 KV，不会因为前缀命中而不再做 Decode Attention。完整块复用的例子也不应泛化为所有引擎只支持完整块。

<iframe src="/content/interactive/inference-serving/prefix-cache-sharing.html?v=height-v2" title="相同前缀的物理块共享与 Prefill 复用" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="prefix-cache-sharing"></iframe>

[单独打开此图](/content/interactive/inference-serving/prefix-cache-sharing.html?v=height-v2)

复用从开头连续匹配的位置开始。A=a b c d、B=a x c d，只能复用 a 的完整前缀结果；c、d 相同不足以抵消此前 x 的上下文变化。

## 6. Continuous Batching：slot 不属于某个请求

### 6.1 三种组批方式

| 方式 | 调整粒度 | 需要理解的重点 |
|---|---|---|
| Static Batching | 整批请求 | 固定组合可能等待最长请求结束 |
| 请求级 Dynamic Batching | 请求到达与组批窗口 | 把可用请求汇成 batch，定义随服务框架而异 |
| Continuous Batching | 每次生成迭代 | 已完成请求退出，新的就绪请求加入 |

侧边会话《Continuous原因解析》的关键解释是：batch 是本轮一起执行的临时组合。A 的连续生成依赖 A 的 token 序列、KV 与采样等状态，不依赖上一轮与它同批的请求必须继续存在。

用会话里的小例子：最多两个执行位置，A 需 4 轮、B 需 2 轮、C 需 3 轮，并假设它们都已经 Decode-ready。

| 轮次 | 整批换批 | Continuous |
|---:|---|---|
| 1 | A+B | A+B |
| 2 | A+B，B 完成 | A+B，B 完成 |
| 3 | A+空位 | A+C |
| 4 | A+空位，A 完成 | A+C，A 完成 |
| 5 | C | C，完成 |
| 6 | C | — |
| 7 | C，完成 | — |

C 在两种方式下分别从第 5、3 轮开始。这里是调度轮数比较，不假设不同 batch 的单轮毫秒数相同。

<iframe src="/content/interactive/inference-serving/decode-batching.html?v=height-v2" title="整批换批与 Continuous Batching 的执行轮次" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="decode-batching"></iframe>

[单独打开此图](/content/interactive/inference-serving/decode-batching.html?v=height-v2)

### 6.2 batch slot 与 KV block 是两件事

```text
第 2 轮：slot 0 → A → A 的 KV blocks
         slot 1 → B → B 的 KV blocks

第 3 轮：slot 0 → A → A 原来的 KV blocks
         slot 1 → C → C 自己的 KV blocks
```

slot 1 是本轮执行位置，不是 B 独占的一块 GPU、一个 SM，也不是 B 的 KV 存储地址。替换 B 不需要重算或搬迁 A 的缓存；每个请求的 Attention 上下文仍相互隔离。

主例的新请求 C 尚未 Prefill，因此不能把它最后一个 prompt token 输入模型就直接开始正常 Decode。必须先建立所需历史 KV，或在计算时正确重建它们。Continuous Batching 提供加入执行的机会，并不消除 Prefill 依赖。

## 7. Chunked Prefill、混合 batch 与串行交错

### 7.1 拆分长 prompt，不丢掉前面的上下文

主例 C 的 2000-token prompt 可分成四个 500-token chunk。后续 chunk 的 Q 仍需关注此前 chunk 的有效 K/V，因此不是四段彼此独立的短 prompt。

分块限制每轮的 Prefill 工作量，让正在 Decode 的 A 获得更频繁的调度机会，但不能保证 A 与独自 Decode 时耗时完全一样。

会话中的串行教学时间线假设：A 每次 Decode 5 ms；C 总 Prefill 80 ms；拆成四块、每块 20 ms；忽略调度开销。时间从 C 到达且 A 已在生成时开始计，不是 A 自身的到达时刻。

| 调度 | A 的四次输出时刻 | C 首 token 时刻 |
|---|---|---:|
| 完整 C Prefill 插在两次 A Decode 之间 | 5、90、95、100 ms | 85 ms |
| 每块 C Prefill 与 A Decode 串行交错 | 5、30、55、80 ms | 100 ms |

分块让 A 不必一次等待 80 ms，但 C 自己的 Prefill 被插入 A 的工作，首 token 可能更晚。不是“分块之后每个请求的所有延迟都改善”。

<iframe src="/content/interactive/inference-serving/chunked-prefill-timeline.html?v=height-v2" title="长 Prefill 的串行教学时间线" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="chunked-prefill-timeline"></iframe>

[单独打开此图](/content/interactive/inference-serving/chunked-prefill-timeline.html?v=height-v2)

### 7.2 串行交错和混合 batch 的计算组织不同

串行交错的示意：

```text
A 第1层 → A 第2层 → …完成 A 前向
C 第1层 → C 第2层 → …完成 C 前向
```

混合 batch 将 A 的一个 Decode token 与 C 的 500 个 Prefill token 放进一次前向：

```text
第1层：处理 A+C → 第2层：处理 A+C → …
```

线性层可将它们堆叠成 501 行输入，与同一权重矩阵相乘；Attention 仍使用各请求自己的上下文与遮罩。两种方式可能得到相同的逐请求数学结果，但 kernel 启动、权重复用、运算形状和资源利用不同。

混合 batch 不代表所有操作在物理上完全重叠，也不代表 C 的计算免费。实际迭代耗时既不能直接用 5+20 ms 推出，也不能无条件用 max(5,20) ms 推出。

<iframe src="/content/interactive/inference-serving/separate-forward-vs-batch.html?v=height-v2" title="两次独立前向与一次混合 batch 的层级顺序" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="separate-forward-vs-batch"></iframe>

[单独打开此图](/content/interactive/inference-serving/separate-forward-vs-batch.html?v=height-v2)

<iframe src="/content/interactive/inference-serving/batching-and-pd-scope.html?v=height-v2" title="串行交错、共置混合 batch 和 PD 分离的资源范围" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="batching-and-pd-scope"></iframe>

[单独打开此图](/content/interactive/inference-serving/batching-and-pd-scope.html?v=height-v2)

## 8. TTFT、TPOT、ITL、吞吐与 Goodput

### 8.1 不同指标回答不同问题

| 指标 | 定义或常见测量口径 | 对应体验 |
|---|---|---|
| TTFT | 请求起点到收到首个输出的时间 | 等多久开始回答 |
| ITL | 相邻流式输出之间的间隔 | 是否出现停顿 |
| TPOT | 对一个请求，首输出之后的总时间除以后续输出 token 数 | 平均生成速度 |
| E2E latency | 请求起点到最后输出的时间 | 整个回答多久完成 |
| requests/s | 完成请求数/测试时长 | 整体处理请求能力 |
| output tokens/s | 输出 token 数/测试时长 | 整体生成吞吐 |
| Goodput | 满足指定要求的成功请求数/测试时长 | 合格服务能力 |

客户端、网关与服务端的起点可能不同；比较时必须统一测量位置和公式。对输出 N>1 的请求：

$$
TPOT=\frac{t_{last}-t_{first}}{N-1}
=\frac{E2E-TTFT}{N-1}.
$$

N=1 时该公式没有定义，不能随意填成一个可比较的 TPOT。p95 TPOT 通常是每请求 TPOT 的分位数，不等于把所有 ITL 混在一起后的 p95。

在普通逐 token 流式输出下，ITL 可对应相邻 token 的间隔；投机解码可能一次返回多个 token，此时有些工具记录的是流式事件之间的间隔。必须检查实现，不能把多 token 事件内的 token 都默认为零间隔后混入统计。

### 8.2 排队可以改变 TTFT，而不改变后续输出间隔

若请求到达 t=0，收到输出的时刻为 100、130、160、190 ms：TTFT=100 ms，ITL=30、30、30 ms，TPOT=30 ms，E2E=190 ms。将所有输出时刻推迟 50 ms，TTFT=150 ms、TPOT 仍为 30 ms、E2E=240 ms。

<iframe src="/content/interactive/inference-serving/queue-and-token-latency.html?v=height-v2" title="整体等待偏移与 token 输出间隔" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="queue-and-token-latency"></iframe>

[单独打开此图](/content/interactive/inference-serving/queue-and-token-latency.html?v=height-v2)

在前述 A 的未分块串行时间线中，ITL 按时间顺序为 85、5、5 ms，平均为 95/3≈31.67 ms。平均值可能掩盖一次很长的停顿，应同时观察尾部间隔与分布。

### 8.3 吞吐增加，不保证满足延迟要求

会话中的教学数据：

| batch | 输出吞吐 | p95 TPOT |
|---:|---:|---:|
| 1 | 50 tokens/s | 20 ms |
| 4 | 160 tokens/s | 25 ms |
| 8 | 220 tokens/s | 45 ms |

若 p95 TPOT 要求不超过 30 ms，batch 1、4 都合格；追求合格配置中的最高吞吐时，应选 4。不是“batch 8 总吞吐最高就选 8”。

<iframe src="/content/interactive/inference-serving/throughput-latency-budget.html?v=height-v2" title="吞吐与 p95 TPOT 预算的权衡" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="throughput-latency-budget"></iframe>

[单独打开此图](/content/interactive/inference-serving/throughput-latency-budget.html?v=height-v2)

Goodput 的一个明确口径是：

$$
Goodput=\frac{\#\{\text{成功且同时满足所有逐请求 SLO 的请求}\}}{\text{测试时长}}.
$$

10 秒内，A 完成 100 请求、其中 70 合格；B 完成 90 请求、其中 85 合格。普通请求吞吐分别为 10、9 requests/s，Goodput 分别为 7、8.5 requests/s。若目标是最多的合格请求，选 B。

分位数层面的业务约束与逐请求 Goodput 的统计口径应分别说明。知道 p95 TTFT 和 p95 TPOT，不能直接推出同时满足两项逐请求约束的请求比例。

## 9. PD 分离：用资源隔离换取 KV 交接

### 9.1 资源组一般是什么

PD 分离把 Prefill、Decode 交给不同推理实例或资源池。常见实现是不同 GPU 集合，例如 Prefill 使用 GPU0/1，Decode 使用 GPU2/3；每组可以是一张卡或多张卡。它通常不是在一张 GPU 内简单把几个 SM 固定分给 P、剩余 SM 分给 D。

两组都需要执行相应阶段的完整模型前向；每组整体持有或访问所需模型权重，而不是 P 只做前几层、D 只做后几层。它们可以采用不同的并行方式和资源配置。

C 的 Prefill 与 A 的 Decode 可以在两个池中同时推进；但 C 自己仍需等待必要的 Prefill 结果及 KV 交接后，才能正常继续 Decode。请求之间可以并行，同一请求的依赖并未消失。

### 9.2 交接时间必须进入成本账

主例缓存大小 512 MiB=0.5 GiB，假设有效传输带宽 4 GiB/s，则：

$$
t_{transfer}=0.5/4\text{ s}=125\text{ ms}.
$$

同机可能通过 NVLink、PCIe 等传输，跨机器可能使用 RDMA 等方式；“有效带宽”需实际测量，不能直接用设备标称峰值。

<iframe src="/content/interactive/inference-serving/prefill-decode-handoff.html?v=height-v2" title="PD 分离与 KV 交接成本" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="prefill-decode-handoff"></iframe>

[单独打开此图](/content/interactive/inference-serving/prefill-decode-handoff.html?v=height-v2)

若其他计算不变、传输无法隐藏，且只避免 20 ms 等待，125 ms 交接会让时间增加约 105 ms；若避免 200 ms 等待，则净减少约 75 ms。独立并行配置、调度变化、网络争用和传输重叠都可能改变结果。

因此“PD 多一次传输，所以总耗时一定更长”以及“PD 分离一定更快”都不成立。更适合比较的是，在相同资源预算下，能否改善目标负载的 TTFT、ITL 和 SLO 内吞吐。

### 9.3 PD 与组批不是一个维度

严格分离的 P/D 执行对不会再组成同一设备上的 P+D 混合 batch，但整个系统仍能组合多种机制：P 池组批多个 Prefill chunk；D 池对多个请求 Continuous Batching；系统也可以同时有共置池和分离池。

Chunked Prefill 也可以用于 P 池。它改善调度粒度，并不天然提供硬件隔离。PD 的干扰隔离同样有边界：共享网络、主机或其他资源仍可能产生争用。

## 10. 投机解码：一次目标前向验证多个候选

### 10.1 草稿、验证与提交是不同计数

草稿模型提出 A、B、C、D；目标模型用因果遮罩一次计算验证所需的条件分布，并按位置顺序决定接受前缀。

如果只接受 A、B，而第三个位置目标结果为 X，提交的是 A、B、X，共 3 个；不是只提交两个。C 后面的 D 原先建立在 S+A+B+C 上，不能直接当作 S+A+B+X 后的有效候选继续提交。

如果四个草稿全部接受，在经典方案的一轮验证中，还可以从目标模型的下一位置分布采样 bonus token，最多提交 5 个。实际还受 EOS、输出长度上限和算法实现约束。

<iframe src="/content/interactive/inference-serving/speculative-decoding-greedy.html?v=height-v2" title="草稿接受前缀、修正 token 与 bonus token" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="speculative-decoding-greedy"></iframe>

[单独打开此图](/content/interactive/inference-serving/speculative-decoding-greedy.html?v=height-v2)

验证 C 的条件分布是 p(C|S,A,B)。把 C 作为某位置输入后，该位置输出用于预测下一 token；不能把输入位置与被预测位置混淆。p 是目标模型的条件概率，不是一个抽象的“正确/错误概率”。

### 10.2 为什么高接受率才有收益

教学条件：草稿 4 ms、目标验证 12 ms，一轮总计 16 ms；普通 Decode 每 token 10 ms。

| 接受草稿数 | 本轮提交数 | 普通方法产出同样数量的时间 | 投机时间 |
|---:|---:|---:|---:|
| 0 | 1 个修正 token | 10 ms | 16 ms，更慢 |
| 2 | 2 草稿+1 修正 | 30 ms | 16 ms，更快 |
| 4 | 4 草稿+1 bonus | 50 ms | 16 ms，更快 |

比较时要按实际提交的有效输出数量，不能固定拿 4 个草稿对应的普通 40 ms 比较。16/5=3.2 ms/token 是本轮成本摊销，不代表用户每 3.2 ms 收到一个 token。

<iframe src="/content/interactive/inference-serving/speculative-decoding-cost.html?v=height-v2" title="接受率、提交数量与投机解码成本" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="speculative-decoding-cost"></iframe>

[单独打开此图](/content/interactive/inference-serving/speculative-decoding-cost.html?v=height-v2)

收益还取决于草稿速度、验证代价、候选长度、上下文长度、batch 与剩余算力。高并发下目标设备已饱和时，投机额外计算未必有利。

### 10.3 随机采样版本不能直接照搬贪心对齐

经典精确投机采样：在同一条件前缀下，草稿 x 来自 q，目标分布为 p，接受概率为：

$$
\alpha(x)=\min(1,p(x)/q(x)).
$$

若均匀随机数 u 不超过接受概率，则接受。首次拒绝后，从修正分布采样：

$$
r(x)=\frac{\max(p(x)-q(x),0)}{\sum_y\max(p(y)-q(y),0)}.
$$

例：p(A)=0.2、p(B)=0.8；q(A)=0.6、q(B)=0.4。

- 草稿 A：接受率 1/3，u=0.2 接受，u=0.5 拒绝。
- 拒绝 A 后：正残差为 (0,0.4)，归一化得到 (0,1)，修正 token 为 B。
- 草稿 B：接受率 min(1,0.8/0.4)=1；u=0.9 也接受，不需要修正采样。

按期望计，100 次草稿约提出 60 次 A、40 次 B；20 次 A 被接受、40 次 B 被接受、40 次拒绝修正为 B，最终为 20 A、80 B，与目标分布一致。这是分布一致，不保证与单独运行目标模型得到完全相同的随机序列。

<iframe src="/content/interactive/inference-serving/speculative-sampling-acceptance.html?v=height-v2" title="目标/草稿分布、接受概率与正残差修正" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="speculative-sampling-acceptance"></iframe>

[单独打开此图](/content/interactive/inference-serving/speculative-sampling-acceptance.html?v=height-v2)

## 11. FlashAttention 与 Online Softmax

### 11.1 省 S/P 的物化，不省有效 Attention 的语义

朴素 Attention 会把得分 S 和概率 P 等大中间矩阵写入 HBM，再由后续算子读取。FlashAttention 使用分块与片上计算，避免完整 S/P 的 HBM 存储和反复读写，仍计算有效位置的精确 Attention。不同浮点运算顺序可能产生数值差异，不应宣称逐位一致。

FlashAttention 没有通过少看历史 token 来加速，也没有消除 KV Cache。稠密 Prefill 的 Attention 算术量仍为 O(N²d)；不能因为使用了 FlashAttention 就说计算复杂度降成线性。单 token Decode 的得分形状是 1×N，其数据访问和并行特点也与完整 Prefill 不同。

<iframe src="/content/interactive/inference-serving/flashattention-tiling.html?v=height-v2" title="FlashAttention 的得分分块与片上累计状态" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:650px;border:0" data-widget="flashattention-tiling"></iframe>

[单独打开此图](/content/interactive/inference-serving/flashattention-tiling.html?v=height-v2)

### 11.2 为什么分块 Softmax 不能各块算完再平均

对一个 Q，令 $w_i=e^{s_i}$，暂时忽略数值稳定性：

$$
O=\frac{\sum_iw_iV_i}{\sum_iw_i}.
$$

| 块 | 未归一化权重 | V | 分母 | 分子 |
|---|---|---|---:|---:|
| 1 | 1、1 | 2、4 | 2 | 6 |
| 2 | 3、3 | 6、8 | 6 | 42 |

累计分母 8、分子 48，所以 O=6。块 1 自己的输出为 3，块 2 为 7，直接平均得到 5，是错的；两个块应按总权重 2 和 6 合并。

### 11.3 更新最大值时，旧累计值一起缩放

为避免指数溢出，Softmax 使用 $e^{s_i-m}$，其中 m 是当前已看见的最大得分。保存三个量：最大值 m、累计分母 ℓ、累计分子 z；实际 z 是与 V 同维度的向量，前面的标量例子只是简化。

新块最大值为 m_b 时，更新：

$$
m'=\max(m,m_b),\qquad a=e^{m-m'},
$$

$$
\ell'=a\ell+\sum_{i\in\text{新块}}e^{s_i-m'},
$$

$$
z'=az+\sum_{i\in\text{新块}}e^{s_i-m'}V_i,\qquad O=z'/\ell'.
$$

旧值需要缩放，因为之前减的是旧 m；新块直接按新 m′ 计算，无需重复调整。

会话中的练习：旧 m=2、ℓ=2、z=8，新块只有 s=4、V=10，则 m′=4、a=e⁻²，新权重为 $e^{4-4}$=e⁰=1，因此：

$$
\ell'=2e^{-2}+1,\qquad z'=8e^{-2}+1\times10.
$$

我最初把 e⁻² 乘到了新块上，应当改为缩放旧累计值。之后若出现得分 6，当前两项都已经属于旧累计结果，应把整个 (2e⁻²+1) 乘 e^{4-6}，再加新块贡献。若新块最大值只有 1，最大值仍是 2，缩放系数 $e^{2-2}$=1，旧值不变。

### 11.4 当前 Q 的累计状态不能代替历史 KV

这些累计状态属于当前 Q。下一个 Q 改变后，QKᵀ 得分改变，历史 V 的权重也改变，不能直接复用上一 Q 的分子和分母，跳过历史 K/V。

| 技术 | 主要复用或优化的对象 |
|---|---|
| KV Cache | 历史位置在各层的 K/V |
| PagedAttention | KV 的物理分配、寻址与共享管理 |
| Prefix Cache | 兼容请求的相同前缀 KV |
| FlashAttention | 一次 Attention 内中间结果的存储与读写 |

## 12. 量化、按需分配与历史重计算

### 12.1 权重量化和 KV 量化不是同一件事

权重量化减少模型参数存储；KV 量化减少缓存元素存储。一个实例中的请求通常共享权重，而 KV 随请求数量和上下文增长。收益应同时评估量化格式、元数据、反量化代价与输出质量。

独立显存练习：总显存 24 GiB、权重 14 GiB、其他固定开销 2 GiB、每请求 KV 1 GiB。原配置可容纳 8 个请求；仅权重降到 7 GiB，可容纳 15 个；仅每请求 KV 降到 0.5 GiB、权重不变，可容纳 16 个。权重占用减半不等于并发翻倍。

### 12.2 我的“只保留最新 KV”究竟指什么

会话里我提出：“可以不一定每个请求都保存完整的 KV……只保存最新的。”这句话最初被理解成删除历史后不再参与 Attention。我的后续澄清是：**只保留最新的 KV，之前的在需要时重新计算。**

这是用计算换显存的可行思路，不能记录成“错误地认为历史 KV 不需要”。应区分三个方案：

| 方案 | 是否保留完整上下文语义 | 主要代价或限制 |
|---|---|---|
| 按当前实际长度分配，不预留最大长度 | 可以 | 管理开销，实际 KV 数据仍存在 |
| 缺失历史 KV 时正确重计算 | 可以 | 重复前向计算、Decode 延迟、临时显存 |
| 丢弃历史 KV 且不再重建 | 普通完整 Attention 下不等价 | 上下文可见性改变 |

深层历史 KV 依赖此前上下文表示，重建通常需要执行相应前缀计算，不是只拿历史 embedding 乘一次 K/V 投影。重计算也不是零显存：如果先重建所有历史 KV，再一起使用，仍会产生临时存储需求；是否分块生成并及时释放、如何组批，会影响峰值。

还可以把部分缓存放到 CPU 等存储，再按需传回，但这属于 offload，代价主要是传输与调度，和重新计算不同。实际滑动窗口模型可按模型规定只看窗口；不能无条件将完整 Attention 模型改成窗口后仍宣称语义相同。

题目“没有做 KV 量化”是初始条件，不是禁止提出量化。KV 量化本身可以做；若此前表述让人理解为不能量化，应修正这一点。

## 13. 怎样公平比较 vLLM 与 SGLang

### 13.1 固定外部条件，允许合理内部调优

比较前，先确定目标是单请求延迟、饱和吞吐、SLO 内吞吐，还是成本。不能先看一张吞吐表再倒推业务目标。

应记录和控制：

1. 模型与语义：相同权重、tokenizer、chat template、精度、输出约束与采样配置；量化和投机方案若不同，需明确质量与额外资源条件。
2. 硬件预算：GPU 型号和数量、拓扑、主机与网络；草稿模型或 PD 额外资源也要计入，不能只比较目标模型卡数。
3. 工作负载：相同请求或输入/输出长度分布、共享前缀比例、上下文和输出长度限制。相同平均长度不等于相同长尾负载。
4. 缓存与预热：运行时预热与 Prefix Cache 状态分别管理。
5. 加压方式：到达速率、并发数、突发方式、测试时长与错误处理。
6. 指标口径：TTFT、ITL、TPOT、E2E、失败率、分位数与 Goodput；保存逐请求数据。
7. 可重复性：固定软件版本，记录关键参数，重复测试并检查波动。允许各引擎在同一资源和业务约束内分别调优，不要求内部参数逐项完全相同。

### 13.2 预热不等于没有缓存命中

模型加载、编译、kernel 准备等运行时预热可以排除初始化影响；用测试 prompt 预热，则可能把它的 KV 留进 Prefix Cache。

无前缀复用场景可在运行时预热后清空前缀缓存，或在两边关闭前缀缓存；还要避免正式测试内出现未设计的相同前缀。有前缀复用场景则保持相同共享前缀分布与缓存准备条件，并单独报告命中情况。

4096-token prompt 只修改最后一个 token，前 4095 个位置仍相同。假设仅完整块可复用且 S=16，可命中前 4080 个位置、255 个完整块，剩 16 个位置重新处理。“请求不完全相同”不能保证冷缓存。

### 13.3 固定并发与固定到达速率

固定并发：始终保持指定数量的在途请求，完成一个再补一个；服务越快，客户端自动发送得越快。

固定到达速率：按设定速率发请求，不因旧请求尚未完成而自动停发。若实际处理能力 8 requests/s，到达 10 requests/s，且无拒绝、超时或背压，则队列平均每秒增加约 2 个请求，后续 TTFT 持续增加。

要逐步提高负载，观察吞吐与延迟曲线。不能把发送速率当成成功完成吞吐，也不能把过载时 TTFT 变长直接归因于 Prefill 算子变慢。

### 13.4 面试回答范例

> 我会先确定线上 SLO 与资源预算，统一模型、精度、硬件和请求长度及前缀分布。运行时预热后，分别控制未命中和命中前缀缓存的场景。用固定到达速率和固定并发方式逐步加压，记录 TTFT、TPOT、ITL、E2E、失败率和满足要求的有效吞吐。允许两个引擎分别合理调优，保存版本与配置，重复验证后，选择目标负载下有效吞吐更高的方案。

本文没有根据虚构数据给 vLLM 或 SGLang 排名，也没有把本轮原理复习当成对两者源码的完整审计。

## 14. 回到主例：一条请求怎样串起所有优化

| 主例中的问题 | 处理路径 | 需要验证的边界 |
|---|---|---|
| A 每轮继续生成 | 计算新 Q/K/V，读取历史 KV | 不重算历史 Q；缓存位置数随处理新位置增长 |
| 4096-position KV 多大 | 128 KiB/token，512 MiB/request | H_KV 与 H_Q、单请求与 batch 合计不能混淆 |
| 缓存物理分配不连续 | 16-position 逻辑块与 block table | 减少碎片，不压缩实际数据 |
| B 共享 A 的前 64 个位置 | 复用 4 个块、8 MiB 前缀 | 后续 Decode 仍读历史；兼容性与边界处理依实现 |
| 短请求完成、C 等待 | iteration 边界重组 batch | slot 不等于 KV 地址；C 先建立 Prefill 状态 |
| C 的 2000-token Prefill 干扰 A | 拆成四个 500-token chunk，考虑混合 batch | 数学结果与资源组织分开；分块不保证零干扰 |
| A 的输出出现长停顿 | 看 ITL 分布与尾部，不只看平均 TPOT | A 已开始生成，不能把这里的首个示意点当其 TTFT |
| 干扰仍难控制 | 评估 PD 资源池与 KV 交接 | 0.5 GiB/4 GiB/s=125 ms，额外传输是否值得 |
| Decode 目标前向重复多 | 草稿+并行验证+顺序接受 | 成本按有效提交数计算；接受率与负载决定收益 |
| Attention 中间数据搬运多 | FlashAttention 与 Online Softmax | 精确语义；不消除 KV，不把复杂度误写为线性 |
| KV 实际需求仍超预算 | 量化、降低活跃并发、重计算或 offload | 显存、计算、传输和质量分别评估 |
| 需要上线选择引擎 | 相同条件下扫描负载并统计 Goodput | 不用最高裸吞吐替代业务要求 |

对于“首 token 很慢，开始后输出顺畅”，先拆分排队、Prefill 与其他开销。教学例 TTFT=800 ms 排队+150 ms Prefill+50 ms 其他=1000 ms；Prefill 减半后为 925 ms，而排队降到 200 ms 后为 400 ms。TTFT 不总由 Prefill 主导，投机解码也不是这类问题的默认答案。

## 15. 我的易错回答，以及应该怎样修正

以下保留会话中真实出现的理解偏差与不完整回答。已经澄清的合理方案单独说明，不把它们继续列为错误。

| 当时的回答或理解 | 不够准确的地方 | 复习时的正确表述 |
|---|---|---|
| “KV 都是 token 自己拥有的，与其他 token 无关” | 忽略深层表示对前文的依赖 | 追加不改因果历史；修改前缀会影响后续层 KV |
| “省掉之前算好的 KV，保留新 token 的 QKV” | 漏掉新 token 的 Attention、MLP 与历史读取 | 避免历史前向重算，新位置完整计算与历史 KV 访问仍存在 |
| 用包含 B=2 的大小推单请求容量 | 把 batch 合计当每请求 | 主例单请求 512 MiB，B=2 合计 1 GiB；8 GiB 可放16请求 |
| token 序号、索引、块号混用 | 没先做 n−1 | 先 i=n−1，再整除求块号、取模求偏移 |
| “C 走槽位2，所以可以加入” | 把执行位置当资源或 KV 地址 | batch成员可在轮次边界更新；槽位与KV地址解耦 |
| “分 chunk 就不影响 A” | 粒度控制不等于资源隔离 | C仍消耗算力与带宽，混合迭代可能变长 |
| “组批就是 A、C 交替进行” | 串行独立前向与混合前向不同 | 看每层是否共同处理A+C，而不只看两请求是否都推进 |
| “这两种方式其实一样” | 数学结果相同不代表执行组织相同 | kernel启动、权重访问、矩阵形状与利用率不同 |
| ITL 回答“5、85、5” | 未沿实际输出时刻依次相减 | 5/90/95/100的ITL为85/5/5；均值95/3 |
| 整体等待偏移后认为 TTFT 不变、TPOT 变小 | 混淆请求起点与相邻输出间隔 | 全部时刻等量推迟：TTFT/E2E增加，TPOT不变 |
| SLO内回答“batch1、4” | 给出了合格集合，未完成吞吐最优选择 | 都合格，但在该表中最高合格吞吐是batch4 |
| 投机拒绝第三个后“只提交前两个” | 少算目标修正 token | 经典方案提交A、B、X共3个 |
| 第一位置拒绝时拿普通40ms比较 | 用草稿长度代替实际产出 | 本轮仅提交1个，对照普通10ms；16ms投机更慢 |
| 选p(C\|S,A,B)，理由是“预测是否正确” | 条件分布不是二分类正确率 | p是目标模型在该前缀下生成各token的分布 |
| “先热身一遍”解决缓存公平性 | 同prompt预热可能制造命中 | 运行时预热与前缀缓存状态分别控制 |
| Goodput算7、8.5，漏普通吞吐 | 未完整回答两种指标 | 普通10、9；有效7、8.5 requests/s |
| “正常TTFT最大占比就是Prefill” | 高负载可能主要在排队 | 先测量请求耗时分解，不能默认瓶颈 |
| 最大值更新时把e⁻²乘新块 | 缩放方向反了 | 旧分母2e⁻²+1；旧分子8e⁻²+10 |
| 单token Decode先写“4097、4097” | 历史K数量误作Q行数 | 1×4097；历史只增加列，不重新产生历史Q行 |
| “Flash只加快读写，不能降低显存占用” | 忽略减少S/P物化 | 可减少Attention中间存储，但不直接缩小KV本身 |
| 12GiB实际KV、8GiB预算回答“不一定能” | 明确数据下不应保留模糊性 | 无共享/压缩/重计算时，不能全部驻GPU |

最终澄清：“只保留最新 KV，之前重新算”是可行的计算换显存方案；我理解历史 K/V 供当前 Q 使用。此前教学把这句话解释为永久丢弃历史，是沟通误读，不是需要继续纠正的知识错误。

## 16. 我主动追问的问题

### 问：PD 的不同 GPU 资源组，是不同卡还是 SM？

答：常见 PD 分离指不同实例使用不同 GPU 集合，可每组单卡或多卡；通常不是简单把一张卡的 SM 切成 P/D 两组。具体实现可能有其他共享方式，不能靠“组”这个词推断硬件隔离。

### 问：PD 多了 KV 交接，所以 Prefill+Decode 总耗时一定更长吗？

答：在计算不变、没有竞争且传输不能隐藏时，会多出交接成本。在线负载下，避免的等待、不同并行配置与资源争用变化可能超过交接成本，所以总体结论需要实测。

### 问：之前的串行时间图属于未分离状态吗？A Decode 和 C Prefill 能同时进行吗？

答：那张图明确假设同资源串行执行，是解释分块如何缩短一次阻塞的教学模型。共置混合 batch 可以在同一模型前向推进 A/C，不等于全部 kernel 物理并行；PD 两池可独立推进两请求，但 C 的自身 Decode 仍依赖其 Prefill。

### 问：共置混批与 PD 分离无法共存吗？

答：同一严格隔离的 P/D 执行对不会又属于同一设备的混合batch；整个系统可以同时有不同池。P池可组批/分块，D池可Continuous Batching，组批技术并没有被PD禁用。

### 问：串行交错与混合 batch 感觉一样，为什么要区分？

答：串行是完成一请求的全部层后，再完成另一请求；混合是同一层处理两请求后进入下一层。它们的逐请求数学结果可相同，但执行边界和资源复用不同。

### 问：为什么 $e^{4-4}$=1？新加的1和1×10不用缩放？

答：新块使用更新后的最大值4，得分也为4，所以指数为0，e⁰=1。它已在新尺度下，无需重复缩放；若下一块把最大值提高到6，现在的所有累计贡献都变成旧值，需整体乘$e^{4-6}$。

### 问：能否只保存最新 KV，历史按需重算？之前不是不能量化吗？

答：正确重算可保留语义，但增加前缀计算与临时显存，需衡量Decode代价。KV量化本身可用，“题设未量化”不是禁止提出量化。记录这次追问时应保留我的最终澄清。

## 17. 自测题与参考答案

建议先遮住答案。计算题说明单位；场景题说清条件、瓶颈、措施与代价，不只背技术名称。

### 题1：为什么只缓存 K/V，通常不缓存历史 Q？

**答案：** 当前输出使用新Q读取历史K/V；历史Q是旧位置的查询，普通Decode不需要重算旧位置输出。新token的Q仍需计算。

### 题2：100-token prompt处理完，采样得到y₁；处理y₁时有多少KV位置？

**答案：** 加上y₁自己的K/V后通常是101。随后采样的y₂尚未前向，不能提前计入缓存。

### 题3：主例B=3、T=4096的KV是多少？8GiB预算在T=8192时理论可容纳多少请求？

**答案：** 3×512MiB=1536MiB=1.5GiB；T翻倍后单请求1GiB，可容纳8请求。不含其他开销。

### 题4：S=16，40个token需要几块、多少尾部空槽？第40个token在哪里？

**答案：** ceil(40/16)=3块，3×16−40=8空槽；索引39，L2，偏移7。

### 题5：i=36，S=16，L2→P11，读哪里？第49个token新增时，旧块要搬吗？

**答案：** L2、offset4，读取P11对应位置；第49个token索引48位于L3、offset0，需要新块但旧块不用搬。

### 题6：A=a b c d，B=a x c d，哪些位置的整套KV可直接复用？

**答案：** 相同前缀a；后续c/d的深层KV受不同前文影响，不能只按token相同复用。物理块粒度可能进一步限制复用。

### 题7：主例B共享64token、S=16，可复用几块？主要省哪个阶段？

**答案：** 4完整块、8MiB；主要省Prefill，不省新Q对有效历史KV的读取。

### 题8：Continuous Batching换掉B，会不会让A丢缓存？C没Prefill能直接加入正常Decode吗？

**答案：** 不会，执行slot与KV地址解耦；不能，C需要建立或正确重建历史KV。加入batch不消除依赖。

### 题9：混合batch含A的1个Decode位置与C的500个Prefill位置，线性层有多少输入位置？是否允许A关注C？

**答案：** 合计501；不同请求Attention上下文隔离，不能相互关注。

### 题10：A输出时刻5/90/95/100ms，ITL与平均TPOT是多少？

**答案：** 85/5/5ms；(100−5)/(4−1)=95/3≈31.67ms。

### 题11：到达0，输出100/130/160/190ms，全体推迟50ms，TTFT、TPOT、E2E呢？

**答案：** 150ms、30ms、240ms；排队偏移不改变相邻输出间隔。

### 题12：10秒完成20请求，每请求50个输出token，两种吞吐是多少？

**答案：** 2requests/s；100output tokens/s。不能用单请求1/TPOT替代整体吞吐。

### 题13：batch1/4/8的吞吐为50/160/220，p95TPOT为20/25/45ms，预算≤30ms选谁？

**答案：** 合格集合为1和4；追求最高合格吞吐选4。

### 题14：10秒内A成功100、合格70；B成功90、合格85，普通吞吐与Goodput？

**答案：** A为10与7；B为9与8.5requests/s；合格请求最多选B。

### 题15：到达10requests/s、处理8，无拒绝或超时，队列与TTFT怎样变？

**答案：** 平均每秒积累约2请求，后续等待和TTFT增大；不表示Prefill算子本身变慢。

### 题16：TTFT由800ms排队、150msPrefill、50ms其他组成。Prefill降到75或排队降到200，结果？

**答案：** 分别925ms和400ms，后一目标收益更大；实现减少排队仍需具体资源或调度措施。

### 题17：把最大batch从8调到32，一定同时改善TTFT、TPOT和Goodput吗？

**答案：** 不一定。吞吐可能提高、排队可能减少，单轮工作和输出间隔也可能增大；超过SLO时Goodput可能下降。

### 题18：512MiB KV以4GiB/s交接需要多久？避免20ms或200ms等待时的净变化？

**答案：** 125ms；其他不变且不重叠时，分别多105ms、少75ms。

### 题19：PD让C的Prefill与A的Decode并行，是否也让C在Prefill未就绪时Decode？

**答案：** 否，不同请求可并行，同一请求仍有数据依赖。

### 题20：草稿ABCD接受AB，第三位置被修正为X，提交几个？全接受时呢？

**答案：** 经典方案提交ABX共3个；全接受可加bonus，共5个，受终止条件约束。

### 题21：投机总16ms、普通10ms/token，首次拒绝时哪个更快？接受2个时呢？

**答案：** 首次拒绝仅产1个，普通10ms比16快；接受2个加修正产3个，普通30ms比16慢。

### 题22：验证草稿C用p(C|S)、p(C|S,A)还是p(C|S,A,B)？

**答案：** p(C|S,A,B)，即在此前候选前缀下的目标条件分布。

### 题23：p=(0.2,0.8)、q=(0.6,0.4)，草稿A接受率、拒绝后修正分布？草稿B呢？

**答案：** A接受率1/3；正残差(0,0.4)归一化为(0,1)，修正B。草稿B接受率1，不进行拒绝修正。

### 题24：FlashAttention删掉了S/P的计算吗？删掉KV了吗？

**答案：** 都没有。它避免完整S/P物化与大量HBM读写，逐块完成有效Attention；历史KV仍需提供。

### 题25：两块权重(1,1)/(3,3)，V(2,4)/(6,8)，最终输出是多少？

**答案：** 分母8、分子48、输出6；不能把块输出3和7直接平均为5。

### 题26：旧m=2、ℓ=2、z=8，新位置s=4、V=10，更新公式？

**答案：** m′=4，a=e⁻²；ℓ′=2e⁻²+1，z′=8e⁻²+10。新权重$e^{4-4}$=1。

### 题27：下一块最大值6时，之前新加的1需要一起缩放吗？最大值没变时呢？

**答案：** 要，整个旧累计值乘$e^{4-6}$；最大值不变时旧值乘1，无需改变。

### 题28：当前总K位置8192，单token Decode得分矩阵有几行几列？

**答案：** 1行8192列。旧Q不重新计算；若多个Q位置一起处理，行数才相应增加。

### 题29：下一Q能直接复用上一Q的Online Softmax分子、分母而不读历史KV吗？

**答案：** 不能。Q改变，历史位置的得分和V权重改变；上一Q的累计状态不通用。

### 题30：24GiB总显存、14GiB权重、2GiB其他、每请求KV1GiB，原配置/权重7GiB/KV0.5GiB各能放几请求？

**答案：** 8、15、16；两种量化的作用对象不同。

### 题31：8GiB KV预算、12请求各实际1GiB，Paged+Flash保证放得下吗？可怎样处理？

**答案：** 不可全部驻GPU；可限制活跃并发、KV量化、增加预算，或设计按需重计算/offload。后两者减少驻留但引入计算/传输与临时存储代价。

### 题32：只保留最新KV、历史重算，与永久丢弃历史有什么区别？

**答案：** 正确重算能提供完整Attention所需KV，保持语义；永久丢弃且不重建改变可见上下文。重算深层KV通常依赖前缀前向，不是独立embedding投影。

### 题33：只改4096-token prompt最后一个token，能保证冷缓存吗？

**答案：** 不能，前4095token仍相同；S=16完整块条件下可复用4080token、255块。

### 题34：A测输入128/输出1024、B测输入8192/输出128，A输出吞吐更高，能给引擎排名吗？

**答案：** 不能，Prefill/Decode工作比例与长度负载不同。先统一模型、资源、工作负载、缓存与指标，再比较。

### 题35：Compute利用率低且HBM带宽接近实测上限，换算力更强但带宽相同GPU保证加速吗？

**答案：** 更可能访存带宽受限，不能保证明显加速；需结合算子profiling，而不是只比较FLOPs峰值。

### 题36：14GB权重、2000GB/s带宽，权重减半后读取下界与整轮加速？

**答案：** 7ms降到3.5ms；整轮不保证两倍，其他耗时与重叠、量化开销都需评估。

## 18. 互动图索引

正文中的图可以操作；单独打开便于集中练习。部分图使用更小的token数、4-token块或不同默认batch以便看清结构，计算时以各图条件为准，不把这些缩尺默认值混入主例。

1. [KV Cache逐token流程](/content/interactive/inference-serving/kv-cache-walkthrough.html)
2. [上下文与深层KV](/content/interactive/inference-serving/context-and-kv.html)
3. [KV显存计算](/content/interactive/inference-serving/kv-memory.html)
4. [分页分配和物理块](/content/interactive/inference-serving/paged-kv-blocks.html)
5. [token索引与偏移](/content/interactive/inference-serving/token-block-index.html)
6. [块大小与尾部浪费](/content/interactive/inference-serving/kv-block-size.html)
7. [Prefix Cache共享](/content/interactive/inference-serving/prefix-cache-sharing.html)
8. [Continuous Batching轮次](/content/interactive/inference-serving/decode-batching.html)
9. [Chunked Prefill串行时间线](/content/interactive/inference-serving/chunked-prefill-timeline.html)
10. [排队与token延迟](/content/interactive/inference-serving/queue-and-token-latency.html)
11. [吞吐与延迟预算](/content/interactive/inference-serving/throughput-latency-budget.html)
12. [PD交接成本](/content/interactive/inference-serving/prefill-decode-handoff.html)
13. [组批与PD范围](/content/interactive/inference-serving/batching-and-pd-scope.html)
14. [独立前向与混合batch](/content/interactive/inference-serving/separate-forward-vs-batch.html)
15. [投机接受与提交](/content/interactive/inference-serving/speculative-decoding-greedy.html)
16. [投机成本](/content/interactive/inference-serving/speculative-decoding-cost.html)
17. [投机随机接受与修正](/content/interactive/inference-serving/speculative-sampling-acceptance.html)
18. [FlashAttention分块](/content/interactive/inference-serving/flashattention-tiling.html)

## 19. 官方资料与论文

- [Hugging Face：Cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation)：因果缓存、Prefill与Decode。
- [vLLM：Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/)：分页KV访问设计。
- [vLLM：Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)：前缀复用与收益范围。
- [vLLM：Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)：Chunked Prefill与调度预算。
- [Hugging Face：Continuous batching](https://huggingface.co/docs/transformers/main/continuous_batching)：在线组批。
- [NVIDIA Triton：Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html)：请求级Dynamic Batching。
- [vLLM：Disaggregated Prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)：P/D实例与KV交接。
- [DistServe论文](https://arxiv.org/abs/2401.09670)：按延迟约束评估分离服务。
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：经典接受、拒绝与修正采样。
- [FlashAttention论文](https://arxiv.org/abs/2205.14135)：IO-aware分块精确Attention。
- [vLLM Benchmark CLI文档源文件](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)：指标口径、负载与缓存公平性。
- [NVIDIA：Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)：推理计算与访存权衡。

官方latest/main文档会变化；实际压测应记录使用的版本、配置与测量日期。本文的基础公式与教学算例不依赖某个最新版本的默认参数。

<script src="/js/transformer-interactives.js" defer></script>

<style>
@media(max-width:400px){#post-meta .post-meta-date{display:grid;grid-template-columns:max-content max-content;gap:4px 6px;white-space:normal}#post-meta .post-meta-date>.post-meta-label:first-child{grid-column:1;grid-row:1}#post-meta .post-meta-date>.post-meta-date-created{grid-column:2;grid-row:1}#post-meta .post-meta-date>.post-meta-label:last-of-type{grid-column:1;grid-row:2}#post-meta .post-meta-date>.post-meta-date-updated{grid-column:2;grid-row:2}#post-meta .post-meta-date>.post-meta-separator{display:none}}
</style>
