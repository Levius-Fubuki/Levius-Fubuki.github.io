# 从关键路径到达标吞吐：大模型性能分析与实验方法复习笔记

这篇笔记整理“性能分析与实验方法”模块的互动复习，结合模型基础、GPU/CUDA、分布式训练和推理服务会话中的相关解释。正文先用一个服务案例串起测量、定位、实验和验收，再展开基础概念、我的真实易错点与主动提问，最后附 63 道自测及答案。

本文的耗时、带宽和吞吐数字都是教学假设，不是某款 GPU、vLLM 或某个线上服务的实测报告。互动图表示各自明确限定的模型；不同图的基线不能混在一起相减。代码是课堂示范，本次整理只做语法、数值模型和浏览器验证，不把它包装成 CUDA 实机测试。

## 1. 一条主线：怎样证明一次优化有效

性能优化回答应形成一条证据链：

```text
工作负载与目标 → 可信测量 → 时间线与关键路径
              → 可验证的瓶颈假设 → 最小对照实验
              → 速度、显存、质量与资源验收 → 适用边界
```

“某个指标很高”和“已经找到瓶颈”之间，还隔着证据与实验。“kernel 更快”和“业务更好”之间，还隔着关键路径、工作负载、SLO、质量和资源预算。

### 1.1 用一个服务案例贯穿知识链

假设我要优化一个流式生成服务。模型、权重、精度、输出内容与硬件保持一致，先固定一类输入。一次首 token 路径被抽象为：

| 阶段 | 教学耗时 | 依赖关系 |
| --- | ---: | --- |
| 准备 | 10 ms | 最先完成 |
| 计算分支 | 80 ms | 准备后启动 |
| 必需的通信分支 | 50 ms | 与计算分支重叠 |
| 收尾与首 token 返回 | 5 ms | 等待两条分支全部完成 |

这里假设两条分支可重叠，且暂时不争抢资源。首 token 延迟为：

$$
T_{\mathrm{TTFT}}=10+\max(80,50)+5=95\ \mathrm{ms}.
$$

我首先要用正常运行的计时确认 95 ms，而不是只测 CPU 提交返回。然后在系统时间线上核对上述依赖，确认没有遗漏拷贝、CPU 等待或跨 rank 就绪差异。

<details class="interactive-figure" id="figure-compute-communication-overlap">
<summary>互动图：计算与通信重叠的关键路径</summary>

<iframe src="/content/interactive/performance-analysis/compute-communication-overlap.html?v=height-v2" title="计算与通信重叠的关键路径" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="compute-communication-overlap"></iframe>

[单独打开此图](/content/interactive/performance-analysis/compute-communication-overlap.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

### 1.2 从长 kernel 追到原因

假设 80 ms 计算分支包含 10 次同类 kernel，每次 8 ms。单个 kernel 做 200 GFLOP，实际 HBM 流量为 2 GB；对应计算路径的峰值为 100 TFLOP/s，HBM 峰值为 1 TB/s。

$$
T_{\rm compute}\geq\frac{200\times10^9}{100\times10^{12}}=2\ \mathrm{ms},\qquad
T_{\rm HBM}\geq\frac{2\times10^9}{10^{12}}=2\ \mathrm{ms}.
$$

实测 8 ms 对应 25 TFLOP/s 和 250 GB/s，都是匹配峰值的 25%。这些数字只能说明还没有贴近这两条理想屋顶，不能直接断定 HBM 跑满或算力不足。下一步检查数据依赖、可并行访存、指令发射、寄存器、同步及实际缓存流量。

### 1.3 kernel 快两倍，首 token 只快约 1.46 倍

假设经过证据验证，某项实现修改让每个 kernel 从 8 ms 降到 4 ms，计算分支从 80 ms 降到 40 ms，其他条件不变：

$$
T_{\mathrm{TTFT,new}}=10+\max(40,50)+5=65\ \mathrm{ms}.
$$

首 token 加速比为 95/65≈1.46。再单独压缩计算，通信分支仍可能让总耗时停在 65 ms。瓶颈已经转移，需要重新分析。

若通信 kernel 很长，我还要对齐所有 rank 的同一次 collective：它是在真正传输，还是在等其他 rank？CPU 已提交通信，不代表 GPU 上的输入已经就绪。仅凭 NCCL kernel 时间不能决定升级网络。

### 1.4 首 token 加速，不等于整条回答同比例加速

假设输出 100 个 token，每个 token 单独送达；首 token 后仍为平均 10 ms 一个间隔：

$$
T_{\rm old}=95+99\times10=1085\ \mathrm{ms},\qquad
T_{\rm new}=65+99\times10=1055\ \mathrm{ms}.
$$

此时完整回答只加速约 1.028 倍。优化报告应明确收益发生在首 token 路径，而不是笼统写“推理加速 1.46 倍”。

### 1.5 上线验收把整条链闭合

接下来回放真实请求组成、长度、前缀复用和突发到达。检查 TTFT、生成停顿、联合 SLO 达标吞吐、显存和任务质量。若长 Prefill 让其他请求的 Decode 停顿，不能用总 token/s 提高掩盖体验回归；若只是服务层聚合输出块，则应改查发送与缓冲。

还要把构图、编译、静态输入拷贝和冷启动成本纳入对应场景；确认新请求确实写入了图读取的输入地址，输出没有在复制完成前被下一轮覆盖。最终结论应是“在什么工作负载和资源约束下、哪项指标改善、代价是什么”，而不是“GPU 更忙，所以优化成功”。

## 2. 基础概念：延迟、吞吐与关键路径

### 2.1 加速比与耗时下降比例

$$
S=\frac{T_{\rm old}}{T_{\rm new}},\qquad
R_T=1-\frac{T_{\rm new}}{T_{\rm old}}.
$$

10 ms 降到 7 ms，是耗时下降 30%、加速约 1.43 倍。不要把“耗时降低”和“速度提高”的百分比混用。

### 2.2 Amdahl 定律有明确前提

被优化部分原占比为 f，局部加速 s 倍；串行、其余工作不变时：

$$
S=\frac{1}{(1-f)+f/s},\qquad S_{\max}=\frac{1}{1-f}.
$$

80 ms kernel 加 20 ms 其他工作，kernel 加速两倍后为 60 ms；即使 kernel 趋近零，总耗时也趋近 20 ms，最大加速比趋近 5 倍。

<details class="interactive-figure" id="figure-performance-critical-path">
<summary>互动图：串行阶段与Amdahl定律</summary>

<iframe src="/content/interactive/performance-analysis/performance-critical-path.html?v=height-v2" title="串行阶段与Amdahl定律" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="performance-critical-path"></iframe>

[单独打开此图](/content/interactive/performance-analysis/performance-critical-path.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

有重叠时，要按依赖图计算完成时间。累计 kernel 耗时不是端到端时间，优化关键路径外的工作不一定缩短完成时间。重叠还可能改变独立运行时的耗时：计算和通信争用 SM 或 HBM 时，两者都可能变慢。

### 2.3 单次延迟与稳态完成间隔

CPU 准备 8 ms、GPU 处理 2 ms，第一批从开始到完成需要 10 ms。允许流水重叠后，相邻完成间隔可为 8 ms，稳态吞吐为 125 batch/s。首批延迟和稳态节拍是两个量。

稳定运行、统计边界一致时，Little 定律为：

$$
L=\lambda W.
$$

L 是平均在途请求数，λ 是平均完成吞吐，W 是平均延迟；不能把 P99 直接代入。增加并发可以抬高吞吐，同时使平均延迟增加。

## 3. Roofline、带宽与访存延迟

### 3.1 先统一单位和计数口径

F 为 FLOP 数，D 为 Byte 数，P 为 FLOP/s，B 为 Byte/s，t 为秒：

$$
I=F/D,\quad t\geq\max(F/P_{\max},D/B_{\max}),\quad
P_{\rm measured}=F/t,\quad B_{\rm measured}=D/t.
$$

本文 GB、TB 用十进制；GiB、MiB 用二进制。GFLOP 与 TFLOP 相差 1000 倍。先统一换成秒再转毫秒，避免只看数字除法。

计算峰值还要匹配精度、CUDA Core/Tensor Core 路径、稠密/稀疏条件和 FMA 计数。不能用 FP16 稀疏 Tensor Core 峰值评价 FP32 普通指令路径。

<details class="interactive-figure" id="figure-kernel-time-bounds">
<summary>互动图：计算/访存理想下界与算术强度</summary>

<iframe src="/content/interactive/performance-analysis/kernel-time-bounds.html?v=height-v2" title="计算/访存理想下界与算术强度" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="kernel-time-bounds"></iframe>

[单独打开此图](/content/interactive/performance-analysis/kernel-time-bounds.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

### 3.2 三种“字节数”不要混用

| 量 | 回答的问题 |
| --- | --- |
| 逻辑访问量 | 算法或代码发出了多少读写需求 |
| 某缓存层级实际流量 | 有多少字节真正经过该层级 |
| 有用数据量 | 业务或算法真正使用了多少字节 |

逻辑访问可能命中 L1/L2，因此逻辑带宽可以超过 HBM 峰值。另一方面，访问不合并可能搬运大量未使用的字节，实际 HBM 带宽很高，有用数据吞吐仍然低。

<details class="interactive-figure" id="figure-coalesced-memory-access">
<summary>互动图：合并访问与有效字节</summary>

<iframe src="/content/interactive/performance-analysis/coalesced-memory-access.html?v=height-v2" title="合并访问与有效字节" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="coalesced-memory-access"></iframe>

[单独打开此图](/content/interactive/performance-analysis/coalesced-memory-access.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

<details class="interactive-figure" id="figure-cuda-bandwidth-metrics">
<summary>互动图：有效带宽与实际带宽</summary>

<iframe src="/content/interactive/gpu-cuda/cuda-bandwidth-metrics.html?v=height-v2" title="有效带宽与实际带宽" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="cuda-bandwidth-metrics"></iframe>

[单独打开此图](/content/interactive/gpu-cuda/cuda-bandwidth-metrics.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

### 3.3 带宽低也可能受访存限制

在指针追逐中，下一次地址依赖上一次读取的返回值，无法提前发出同一条链的后续请求。若单次延迟 200 ns，四次依赖读取需 800 ns；四个地址独立且可完全并行时，总时间可为 200 ns，但单次延迟仍是 200 ns。

多条独立访问链可以增加 memory-level parallelism，隐藏部分等待；这不是让单次显存访问本身变快。算法不允许独立化时，不能凭空消除数据依赖。

### 3.4 融合和 occupancy 都需要结果验证

融合可能减少中间张量读写和 kernel 提交，却增加寄存器压力；限制寄存器又可能产生 spilling。local memory 流量增加后，部分请求还可能继续进入 HBM。

occupancy 通常指活跃 warp 数占该 SM 所支持最大活跃 warp 数的比例，受寄存器、共享内存等资源限制。它不直接等于有效执行效率。还要看就绪 warp、发射、等待、实际流量和最终耗时，不能只追求更高 occupancy 或更高带宽。

## 4. 两种 Profiler 各回答什么问题

| 工具与证据 | 先回答的问题 |
| --- | --- |
| Nsight Systems、CPU 调用栈、调度日志 | 时间花在哪里；哪个依赖拖住完成；CPU/GPU/通信/拷贝怎样重叠 |
| Nsight Compute、运算量、对应层级流量 | 选定 kernel 为什么慢；计算路径、访存、指令、warp 和资源限制是什么 |
| 正常运行的独立 benchmark | 去除分析工具干扰后，端到端收益是否仍成立 |

GPU-Util 主要表示采样期间 GPU 上有 kernel 执行的时间占比，不是实际 FLOP/s 占峰值的比例。CPU 总占用也可能掩盖单个关键线程跑满。它们适合发现现象，不足以单独证明根因。

若 10 ms 内只有 2 ms kernel 执行，应先查剩余区间的拷贝、数据供给、提交与同步；“没有 kernel”不自动等于整张卡完全空闲。

Profiler 可能带来追踪、重放、计数采集和缓存状态变化。分析时 B 更快、正常运行时 B 更慢并不矛盾。最终性能验收必须回到关闭 Profiler 的运行。

## 5. CUDA 异步计时与完整 benchmark

### 5.1 CPU 返回不等于 GPU 完成

CPU 时钟只包围一次异步调用，可能测到提交开销。若要测从调用到完成，应在起点前排除旧工作，在终点前等待目标工作完成；同步范围取决于被测对象。

<details class="interactive-figure" id="figure-cuda-async-timing">
<summary>互动图：异步提交与完成计时</summary>

<iframe src="/content/interactive/performance-analysis/cuda-async-timing.html?v=height-v2" title="异步提交与完成计时" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="cuda-async-timing"></iframe>

[单独打开此图](/content/interactive/performance-analysis/cuda-async-timing.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

Event 的时间差是两个 GPU 事件之间经过的时间，不会因为 CPU 晚读取而增加，但事件之间的空闲会被计入。多 stream 任务需要按目标范围汇合；单条 stream 的 end 不代表整个任务结束。

<details class="interactive-figure" id="figure-cuda-sync-scope">
<summary>互动图：同步范围与Event时间</summary>

<iframe src="/content/interactive/performance-analysis/cuda-sync-scope.html?v=height-v2" title="同步范围与Event时间" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="cuda-sync-scope"></iframe>

[单独打开此图](/content/interactive/performance-analysis/cuda-sync-scope.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

单次独立调用允许逐轮等待；持续流水吞吐应整体计时，避免每轮全设备同步破坏 CPU/GPU 重叠。十批 CPU 准备 3 ms、GPU 执行 7 ms 的理想模型，逐批同步为 100 ms，流水可为 73 ms。

### 5.2 我要求助手直接写出的代码

课堂里我提出“你直接把代码写出来然后讲解”。下面是该示范整理版：输入和参数已在同一 GPU，fn 只使用当前 stream，测单次独立推理调用延迟。它不是通用多 stream 服务吞吐测试器。

```python
import statistics
import torch


@torch.inference_mode()
def benchmark(fn, x, warmup=20, repeats=100):
    if not x.is_cuda:
        raise ValueError("x 必须是 CUDA Tensor")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup 必须 >= 0，repeats 必须 > 0")

    device = x.device
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        end.record(stream)

        for _ in range(warmup):
            output = fn(x)
            del output
        torch.cuda.synchronize(device)

        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        times_ms = []
        for _ in range(repeats):
            start.record(stream)
            output = fn(x)
            end.record(stream)
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
            del output
        peak = torch.cuda.max_memory_allocated(device)

    return {
        "median_ms": statistics.median(times_ms),
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "extra_peak_allocated_bytes": peak - baseline,
    }


if __name__ == "__main__":
    x = torch.randn(4096, 4096, device="cuda")
    print(benchmark(torch.relu, x))
```

Event 预先 record，让其延迟初始化位于正式计时之外。预热后同步，再读取显存基线、重置峰值。每轮只保留 Python 时间值，不长期持有 GPU 输出。峰值减基线表示相对常驻张量的额外峰值，不表示设备全部额外内存。

warmup=20 不是普适的充分条件；要确认性能稳定。模型还应按任务设置 eval 模式，inference_mode 与 eval 并非同一件事。代码中的逐轮等待有意对应独立调用延迟，不适用于直接推断流水吞吐。

## 6. 公平实验：比较的是实现，还是其他条件

### 6.1 需要记录的变量

模型和权重、精度、输入输出长度分布、batch、micro-batch、梯度累积、DP/TP/PP、GPU 型号和拓扑、软件版本、频率状态、其他负载、到达轨迹、并发上限、缓存容量和初始状态都可能改变结果。

batch=8、8 ms 与 batch=32、16 ms 对应 1000 与 2000 条/s，但不能把差异直接归因于代码。应补齐相同 batch 下的比较，必要时做实现×batch 的二维对照。

### 6.2 冷启动、缓存与测试顺序

冷启动和稳态分别报告。反复用同一 prompt 会提高前缀复用，反复读同一个输入缓冲区也可能改变数据缓存状态。代码路径已预热，不代表每次访问的数据都处于相同状态。

如果 A→B 为 10/7 ms，而 B→A 仍为 10/7 ms，最明确的证据是“第二次更快”，不是 B 更快。还需流量和缓存计数等证据，才能进一步归因于 L2。真实业务复用输入时测相应热状态，持续新输入时用轮换缓冲区和匹配工作集。

前缀缓存要分别报告命中率口径、复用长度、命中和未命中耗时、淘汰和整体分布。不能把同 prompt 的高命中 benchmark 直接推广到低复用业务。

### 6.3 构图开销需要摊销

假设 A 每请求 10 ms，B 每请求执行 5 ms，每次新 shape 首次另需 100 ms 构图，缓存初始为空且不淘汰。N 个请求、K 次首次构图：

$$
T_A=10N,\qquad T_B=5N+100K.
$$

B 更快需 K<N/20。1000 个请求、一个 shape 时约加速 1.96 倍；50 个 shape 时正好无收益。真实系统还需记录重新构图、缓存淘汰、内存和长尾请求。

<details class="interactive-figure" id="figure-graph-build-amortization">
<summary>互动图：构图成本摊销</summary>

<iframe src="/content/interactive/performance-analysis/graph-build-amortization.html?v=height-v2" title="构图成本摊销" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="graph-build-amortization"></iframe>

[单独打开此图](/content/interactive/performance-analysis/graph-build-amortization.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

## 7. 统计、排队和流量模型

### 7.1 分位数要在正确粒度上计算

先按同一请求合成端到端耗时，再计算分位数。阶段 P99 不能相加；请求平均 TPOT 的 P99、逐 token 间隔的 P99、响应块间隔的 P99也不能混称。

平均 TPOT 可以掩盖一次长停顿；P99 也可能漏掉低于其尾部比例的事件。报告较高分位数、最大值及超出业务停顿阈值的次数/比例，并注明样本量和分位数算法。

若两项请求级 SLO 各有 90% 达标，联合比例可能为 80%～90%。应关联请求 ID，统计同时满足所有条件的成功请求，得到达标吞吐。

### 7.2 零次超标不等于没有尾部

在独立、稳定、具有代表性的理想样本中，真实超标率 p、样本数 n 时：

$$
P(\text{零次超标})=(1-p)^n.
$$

p=1%、n=100 时，零次超标仍有约 36.6% 的概率。使这个概率小于 5%，需要 n≥299 且观察到零次超标；这不是未来绝不超标的保证。

如果每个窗口 200 个请求共享同一状态，50 个窗口独立，则 10000 个请求全部未超标的概率是 0.99⁵⁰≈60.5%，不是 0.99¹⁰⁰⁰⁰。增加窗口内请求数不能替代增加独立状态覆盖。

### 7.3 完成吞吐稳定，不代表队列稳定

持续到达 1200 请求/s、完成 1000 请求/s，队列每秒增长 200。10 秒后未完成 2000 个，停止发送后还需 2 秒清空。只统计已完成请求的“成功率100%”不能代表全部到达请求；也不能把未完成一律当失败，必须单独报告。

<details class="interactive-figure" id="figure-arrival-completion-backlog">
<summary>互动图：到达、完成与积压</summary>

<iframe src="/content/interactive/performance-analysis/arrival-completion-backlog.html?v=height-v2" title="到达、完成与积压" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="arrival-completion-backlog"></iframe>

[单独打开此图](/content/interactive/performance-analysis/arrival-completion-backlog.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

均匀到达和突发到达，即使平均都是 500 请求/s，P99 也可相差巨大。每请求固定处理 1 ms：每 2 ms 到一个请求时延迟为 1 ms；每 2 s 同时到 1000 个时，该批延迟为 1～1000 ms，nearest-rank P99 为 990 ms。

### 7.4 开环与闭环要匹配业务

闭环请求完成后补发，服务变慢时发送速率也下降；开环按计划节奏产生请求。若真实到达不依赖响应，闭环可能出现协调遗漏，漏掉本应在暂停期间到达并排队的请求。

记录计划到达、实际发送、服务端关键阶段与完成时间，明确最终采用的延迟边界，同时单独报告压测端发送滞后。开环固定到达轨迹和并发上限，实际在途并发允许随延迟变化。超时、失败和未完成请求都要保留统计。

## 8. LLM 流式服务：首 token、停顿与有效工作

### 8.1 TTFT 与 TPOT 不是同一个优化目标

每次恰好返回一个 token、输出 N>1 时：

$$
T_{\rm last}=\mathrm{TTFT}+(N-1)\mathrm{TPOT}.
$$

A 的 TTFT/TPOT 为 200/10 ms，B 为 100/15 ms：输出 100 token 时 A 更快，输出 10 token 时 B 更快，N=21 时均为 400 ms。需要真实输出长度分布和业务目标才能决定取舍。

### 8.2 Prefill、Decode 与客户端缓冲

长 Prefill 可推迟已有请求的 Decode。结合调度日志、CPU 提交和 GPU 时间线，区分尚未调度与已提交后等待。减少每轮 Prefill token 预算是一个可测试因素，但要同时检查 TTFT、吞吐和联合 SLO，不能只看停顿改善。

<details class="interactive-figure" id="figure-chunked-prefill-timeline">
<summary>互动图：长Prefill分块与Decode停顿</summary>

<iframe src="/content/interactive/inference-serving/chunked-prefill-timeline.html?v=height-v2" title="长Prefill分块与Decode停顿" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="chunked-prefill-timeline"></iframe>

[单独打开此图](/content/interactive/inference-serving/chunked-prefill-timeline.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

若引擎每 10 ms 生成一个 token，服务端输出队列也及时收到，而客户端每 100 ms 才收到 10 个 token，应继续检查输出缓冲、代理与传输。块内同时送达不代表同时生成，块间隔也不能直接报告成逐 token 生成间隔。

### 8.3 Padding、分桶与请求混合

同样处理四条 128-token 与四条 1024-token 输入，有效 token 为 4608。混合 batch 补齐到 1024 共 8192 槽位；按长度分两批仅 4608 槽位。含 padding token/s 可能奖励了额外无效工作。

在线分桶还要检查凑批等待、TTFT、尾延迟、饥饿和达标吞吐。短请求很快、长请求退化时，也应按真实混合工作量汇总，不能平均各类加速比。

<details class="interactive-figure" id="figure-workload-mix-speedup">
<summary>互动图：请求混合与整体加速</summary>

<iframe src="/content/interactive/performance-analysis/workload-mix-speedup.html?v=height-v2" title="请求混合与整体加速" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="workload-mix-speedup"></iframe>

[单独打开此图](/content/interactive/performance-analysis/workload-mix-speedup.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

## 9. 分布式训练：慢 rank、重叠和资源竞争

### 9.1 扩容比较先统一口径

先确认固定总工作量还是随卡数增加工作量；核对 global batch、micro-batch、梯度累积与 DP/TP/PP。再检查是否跨机器、互联拓扑和软件版本是否变化；有效 token 是否包含 padding；计时是否覆盖全任务而非最快 rank。

对齐时间线并匹配同次 collective。早到 rank 的 NCCL kernel 很长，可能是在等待其他 rank 的梯度或数据。仅在各 rank 及时就绪、传输仍慢并有网络/拓扑证据时，再加强“通信路径受限”的判断。

<details class="interactive-figure" id="figure-rank-timeline-bottleneck-diagnosis">
<summary>互动图：跨rank就绪与通信瓶颈</summary>

<iframe src="/content/interactive/distributed-training/rank-timeline-bottleneck-diagnosis.html?v=height-v2" title="跨rank就绪与通信瓶颈" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="rank-timeline-bottleneck-diagnosis"></iframe>

[单独打开此图](/content/interactive/distributed-training/rank-timeline-bottleneck-diagnosis.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

最小实验可以预置相同真实输入到 GPU，绕过数据加载、CPU 预处理和拷贝，但保留内容、长度、mask、rank 分配以及 MoE 路由分布。记录预置规模带来的显存和缓存变化；实验无改善，只能削弱数据供给假设，不能直接证明网络故障。

### 9.2 通信合并不总是减少完成时间

反向计算 0～30 ms，三组梯度在 10/20/30 ms 就绪，各通信 8 ms：分别启动的累计通信 24 ms，完成时间 38 ms。合并后虽然只通信 18 ms，却等到 30 ms 才开始，完成于 48 ms。

<details class="interactive-figure" id="figure-gradient-bucket-overlap">
<summary>互动图：梯度分桶的启动与重叠</summary>

<iframe src="/content/interactive/performance-analysis/gradient-bucket-overlap.html?v=height-v2" title="梯度分桶的启动与重叠" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="gradient-bucket-overlap"></iframe>

[单独打开此图](/content/interactive/performance-analysis/gradient-bucket-overlap.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

优化暴露在尾部的 G3，可能比优化已被计算隐藏的 G1/G2 更有效。真实系统中还需测重叠后的资源竞争，不能沿用独立运行的耗时。

### 9.3 MFU、HFU 与达到质量目标的时间

本文口径：MFU 的分子为不含额外重算的模型 FLOP 数；HFU 的分子包含额外重算的实际 FLOP 数；两者除以匹配硬件峰值×耗时。提高 HFU 可能只是多做了重计算。

<details class="interactive-figure" id="figure-activation-checkpoint-recompute">
<summary>互动图：激活重计算的时间与显存</summary>

<iframe src="/content/interactive/distributed-training/activation-checkpoint-recompute.html?v=height-v2" title="激活重计算的时间与显存" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="activation-checkpoint-recompute"></iframe>

[单独打开此图](/content/interactive/distributed-training/activation-checkpoint-recompute.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

是否采用重计算，要综合显存可行性、micro-batch、卸载/通信代价及达标时间。训练吞吐提高，也可能因为需要更多有效 token 才收敛而延长训练。比较同一验证集与目标质量，不能只看 step 数。

侧边分布式会话的关键补充是：FSDP 的存储分片与 TP 的计算分片不同。各卡做了哪些计算、暂时聚合了哪些状态，会改变计算量、通信和峰值显存，不能仅从“参数分片”推断每卡工作量。

## 10. 显存：计数、碎片、在途存储与泄漏

allocated 关注张量分配；reserved 是分配器保留的内存。两者差值不等于随时可满足任意申请的可用块。设备总占用还可能包含分配器之外的分配。

<details class="interactive-figure" id="figure-cuda-memory-accounting">
<summary>互动图：allocated、reserved与峰值</summary>

<iframe src="/content/interactive/performance-analysis/cuda-memory-accounting.html?v=height-v2" title="allocated、reserved与峰值" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="cuda-memory-accounting"></iframe>

[单独打开此图](/content/interactive/performance-analysis/cuda-memory-accounting.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

| 情况 | 关键证据 | 不能直接下的结论 |
| --- | --- | --- |
| 活跃张量峰值高 | 分配历史、峰值、引用与工作集 | 不能只看测量结束后的 allocated |
| 在途存储尚未回收 | active_awaiting_free、相关 stream/event | 不一定是引用泄漏 |
| 缓存与碎片 | segment/block 布局、inactive_split、最大可用块 | reserved−allocated 不保证可拼成大块 |
| 泄漏或有意保留 | 同条件重复后增长趋势、持有对象和缓存策略 | 单次 reserved 升高不能证明泄漏 |

empty_cache 不能释放活跃张量，也不能任意移动活跃块以拼接空闲。普通保留段里夹着活跃分配时，空闲碎片未必能整段归还驱动。使用 memory_snapshot、memory_stats 时还要确认分配器后端和指标适用范围。

## 11. CPU 提交、CUDA Graph 与跨 stream 生命周期

### 11.1 跨核平均掩盖提交线程

32 核中一个关键线程跑满，总平均约 3.1%。50 次提交各 40 μs、kernel 各 10 μs，按“提交结束才能启动”的教学模型，总完成 2010 μs。GPU 执行减半只变为 2005 μs；提交耗时减半则为 1010 μs。

<details class="interactive-figure" id="figure-cpu-kernel-launch-bottleneck">
<summary>互动图：CPU提交开销与GPU空洞</summary>

<iframe src="/content/interactive/performance-analysis/cpu-kernel-launch-bottleneck.html?v=height-v2" title="CPU提交开销与GPU空洞" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="cpu-kernel-launch-bottleneck"></iframe>

[单独打开此图](/content/interactive/performance-analysis/cpu-kernel-launch-bottleneck.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

验证时检查 CPU 调用栈、kernel 启动间隙、数据准备和同步。CUDA Graph 减少提交开销后，若 kernel 本身耗时基本不变，而空洞和总耗时下降，就支持提交瓶颈假设；仍需正常运行复测。

### 11.2 图记录的是地址与操作，不是 Python 变量名

捕获时读取 A，后来把 Python 变量改指向 B，并不自动让图读取 B。保持结构、形状和类型不变时，应正确更新静态输入 A，并保证拷贝与重放的依赖。只计 replay 时间，却拿它与普通执行的“输入准备到输出可用”时间相比，并不公平。

### 11.3 两条执行依赖与一条生命周期约束

```text
第一次写 O 完成 → 将 O 复制到独立结果 R 完成 → 下一次覆盖 O
```

只保证“写完再读”不够，还要“读完再覆盖”。跨 stream 事件可约束相关操作，不必每次阻塞 CPU 或同步整张设备。

<details class="interactive-figure" id="figure-cuda-stream-event-dependency">
<summary>互动图：跨stream事件依赖与完成时刻</summary>

<iframe src="/content/interactive/gpu-cuda/cuda-stream-event-dependency.html?v=height-v2" title="跨stream事件依赖与完成时刻" class="transformer-interactive" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:520px;border:0" data-widget="cuda-stream-event-dependency"></iframe>

[单独打开此图](/content/interactive/gpu-cuda/cuda-stream-event-dependency.html?v=height-v2)。窄屏可在图区域横向查看。

</details>

此外，CPU 释放 A 的最后一个引用时，侧 stream 可能还在读 A。要保留引用到使用完成，或正确向分配器登记 record_stream，或把完成依赖同步回创建流后安全释放。record_stream 防止存储过早复用，不负责建立写后读依赖。

正确的生命周期管理仍可能让大量在途输入占满显存。生产每 1 ms 一份、消费每 4 ms 一份时，限制预取/提交数量、按完成事件循环复用缓冲区，可以降低峰值；只要供给跟得上，不必牺牲 GPU 稳态吞吐。

## 12. 后台 checkpoint：异步没有消除守恒关系

每步 1 s、每 50 步提交一份快照、后台串行每份写 80 s。500 步结束时提交 10 份，只写完 5 份，最新可恢复状态是第 250 步；全部写完到 850 s。

前台 step 不变，不能证明存盘没有代价。需报告提交到持久化的延迟、待写数量与字节、快照内存、恢复点滞后、写入失败和最终排空时间。

若每份都保留且缓冲有界，长期平均每步至少 80/50=1.6 s。若允许丢弃中间快照，可保留一份正在写入的不可变快照，以及一份最新待写快照；新快照只替换尚未开写的旧快照。正在写入的内容不能被原地覆盖。

## 13. 正确性、质量和资源预算共同验收

### 13.1 数值容差不是任务质量保证

$$
|\mathrm{candidate}-\mathrm{reference}|
\leq atol+rtol\,|\mathrm{reference}|.
$$

即使单步 logits 通过容差，也可能改变 greedy 的 argmax，使后续前缀分叉。固定同一前缀对比 logits，可隔离局部误差；再用代表性评测、长序列、解码参数和质量阈值检查最终影响。

### 13.2 达标吞吐与单位 GPU 效率

$$
G=\frac{\text{成功且同时满足全部请求级 SLO 的请求数}}{\text{测量时长}},\qquad
G_{\rm per\ GPU}=G/n_{\rm GPU}.
$$

4 卡 A 的完成吞吐1000、联合达标率95%，得950请求/s、237.5请求/s/GPU。8卡 B 为1600、90%，得1440请求/s、180请求/s/GPU。固定8卡预算、假设双A互不干扰，则1900高于1440，但实际必须验证流量路由、共享CPU/网络和前缀缓存命中变化，不能直接把单副本乘二当结论。

## 14. 我的真实易错点、逐步补全与主动提问

### 14.1 主会话中能确认的薄弱点

| 当时的回答 | 复盘结论 | 现在应怎样说 |
| --- | --- | --- |
| Roofline 时间回答“0.2，0.4” | 数值与单位不正确，但“访存约束更强”的判断正确 | 200 GFLOP/100 TFLOP/s=2 ms；4 GB/1 TB/s=4 ms，先换到秒 |
| 指针追逐选择 A：增加计算吞吐 | 忽略下一次地址依赖前次结果 | 算力增加不能解除地址依赖；算法允许时用独立链增加并行访存 |
| 最初只回答在 op 后同步 | 原题排除了此前工作，这个回答在原题下成立 | 后续加入同 stream 旧任务后，再在 t0 前同步；这是边界补全，不能把原回答简单标错 |
| “8 ms 没有 kernel，先分析等待” | 排查优先级正确 | 补充核对拷贝与其他 stream，“无 kernel”不等于设备完全空闲 |
| 开环实验中说固定并发 | 可理解为固定限制，但需说清口径 | 固定到达轨迹和并发上限；实际在途数量随延迟变化 |

后期对联合 SLO、相关样本、重计算、异步存盘、跨 stream 生命周期、碎片和资源效率的回答大多正确。本篇把它们列为已建立的判断边界，不为了制造“错题本”而改写成个人错误。

### 14.2 我主动提出的问题

**“你直接把代码写出来然后讲解。”** 代码题由助手示范，我没有现场独立写完，不能登记为已经通过的手写能力。本文保留完整函数和适用条件，便于下次遮住实现重写。

**“代码都可以先放一放。”** 当时选择先稳固概念与实验设计，再回到实现。这个学习顺序不等于不会写代码，也不等于示范代码已经得到 GPU 验证。

**“还有多少道题？”** 原互动没有预先设定总量，最后收束到63个编号知识单元。本篇把主线和延伸分开，并提供完整答案，避免无边界增加同类题。

### 14.3 侧边会话对本章最有帮助的补充

| 来源与实际疑问 | 与本章的连接 |
| --- | --- |
| GPU/CUDA：shared scratch 覆盖前的 block 屏障，我曾漏掉“写入前已同步”的条件 | 判断生命周期要沿依赖顺序，而不是看屏障离某行代码远近；与静态输出“读完再覆盖”相通 |
| GPU/CUDA：三遍扫描是逻辑流量，不一定是 HBM 流量或同比例耗时 | 对应 Roofline 和有效带宽的层级口径 |
| 分布式训练：“没看懂 FSDP 具体怎么分，以及计算过程怎样” | 区分存储分片、算子计算、临时聚合和真实峰值，才能比较扩容 |
| 推理服务：“先热身一遍”是否就公平 | 运行时预热与前缀缓存状态必须分别控制 |
| 推理服务：PD 多出 KV 交接，是否总耗时一定更长 | 局部新增开销不能决定端到端；还要看等待、竞争和重叠 |
| 推理服务：串行交错和混合 batch 感觉相同 | 数学结果相同不代表启动、权重复用、矩阵形状和关键路径相同 |
| 推理服务：“只保留最新 KV，之前需要时重新算” | 经澄清是可行的计算换显存方案，不是永久忽略历史；应比较完整语义、重算成本与可行显存 |

相关完整笔记：[GPU、CUDA 与算子优化](/2026/10/20/gpu-cuda/)、[分布式训练与显存优化](/2026/10/12/distributed-training/)、[推理引擎与服务优化](/2026/10/08/inference-serving/)、[模型基础](/2026/10/05/transformer-foundations/)。

## 15. 常见误区速查

| 容易脱口而出的结论 | 应保留的边界 |
| --- | --- |
| GPU-Util高，所以算力跑满 | 看匹配计算吞吐及等待证据 |
| CPU总占用低，所以无CPU瓶颈 | 看关键线程与提交时间线 |
| 带宽低，所以不是访存问题 | 可能受延迟、依赖与并行请求数限制 |
| 带宽高，所以搬运高效 | 看实际流量中有多少是有用字节 |
| occupancy高，所以kernel快 | 看发射、等待、spill与最终时间 |
| 通信总时长减少，所以step更快 | 看启动时间、重叠与暴露尾部 |
| NCCL长，所以升级网络 | 先区分慢rank和真实传输瓶颈 |
| Event只统计kernel执行 | 事件间空闲与等待也可能计入 |
| P99相加或SLO比例相乘即可 | 请求级关联；未说明独立性不能直接相乘 |
| 完成吞吐稳定，所以没有过载 | 还要看未完成队列和超时 |
| reserved比allocated高就是泄漏 | 查缓存、碎片、在途状态与引用 |
| record_stream替代事件等待 | 生命周期登记与执行顺序不同 |
| HFU提高，所以有效训练更快 | 可能只是额外重算增加 |
| 局部allclose通过，所以质量不变 | token选择和任务质量仍需验证 |
| 缓存/固定shape稳态收益可直接推广 | 真实复用、构图成本和淘汰可能改变结论 |
| 总吞吐越高，固定预算越划算 | 比较相同硬件预算内的达标吞吐 |

## 16. 面试回答模板与实验记录

### 16.1 通用回答

> 我先明确真实工作负载、目标指标和质量、显存、资源约束，再建立可信基线。随后用系统时间线定位关键路径，区分计算、访存、通信、CPU提交和排队问题，对关键kernel做细粒度分析。针对证据提出可验证假设，通过最小对照实验检验。最后关闭Profiler检查端到端收益、SLO、显存和任务质量，报告适用范围与回归。

### 16.2 kernel快两倍，整体收益很小

> 我先看kernel在原始端到端中的占比与是否处于关键路径。重叠区间的节省未必缩短完成时间；还要核对其他阶段、拷贝、同步和资源竞争是否增加。最终用实际时间线解释收益，不能简单相加累计kernel耗时。

### 16.3 多卡扩容不线性

> 我先统一总工作量、batch和并行配置，再核对硬件拓扑与有效token口径。随后对齐各rank同一次collective，区分就绪等待与有效传输。用保留真实输入内容和rank分配的最小实验检验数据供给或负载不均假设，再结合匹配拓扑的通信基准评价网络。

### 16.4 证明优化真实有效

> 保持工作负载和环境可比，统一计时边界，分开冷启动与稳态，重复并交换顺序。最终在Profiler关闭时比较端到端延迟、联合SLO达标吞吐、显存和质量；覆盖不同长度、负载及缓存状态，避免把条件变化当作实现收益。

### 16.5 一次优化报告的可填写骨架

| 字段 | 需要填写的证据 |
| --- | --- |
| 场景 | 模型、精度、GPU、拓扑、软件、长度、到达轨迹与缓存状态 |
| 目标 | 指标、SLO、显存、质量与资源预算 |
| 基线 | 正常运行的分布、吞吐、显存、失败与未完成请求 |
| 假设 | 具体瓶颈、时间线或计数证据，以及仍未确定的部分 |
| 实验 | 只改什么、控制什么、如何排除主要替代解释 |
| 结果 | 指标前后值、重复范围、端到端收益与质量验收 |
| 边界 | 适用负载、启动成本、显存/质量代价及回归 |

## 17. 自测题与答案

以下按课堂编号整理63个知识单元。指针追逐中的两次补问并入第7题；第31题是助手示范代码，不能视为我现场独立实现。各题的不同时间线和硬件数字都是各自的教学假设。

<details class="self-test" id="self-test-1">
<summary>Q1：串行任务100 ms，其中kernel 80 ms、其他20 ms。kernel加速两倍，整体怎样？</summary>

答：新耗时80/2+20=60 ms，加速100/60≈1.67倍。其余20 ms不变，所以整体不到两倍。

</details>

<details class="self-test" id="self-test-2">
<summary>Q2：上一题只持续优化kernel，总耗时与加速比的极限是什么？</summary>

答：kernel趋近零时总耗时趋近20 ms，加速比趋近5倍；这是串行、其他耗时不变的极限。

</details>

<details class="self-test" id="self-test-3">
<summary>Q3：准备10 ms，计算80 ms与通信50 ms重叠，收尾5 ms。计算加速两倍后怎样？</summary>

答：10+max(40,50)+5=65 ms。原耗时95 ms；通信分支成为关键约束，单独继续压缩计算不再缩短此模型的完成时间。

</details>

<details class="self-test" id="self-test-4">
<summary>Q4：F=200 GFLOP、D=4 GB，匹配峰值100 TFLOP/s与1 TB/s。两种下界是多少？</summary>

答：先转秒：计算200e9/100e12=0.002 s=2 ms；访存4e9/1e12=0.004 s=4 ms。理想下界4 ms，计算峰值翻倍后仍为4 ms。

</details>

<details class="self-test" id="self-test-5">
<summary>Q5：保持F与峰值不变，将实际HBM流量从4 GB降到2 GB。算术强度与下界怎样？</summary>

答：强度从50变为100 FLOP/B；计算和HBM下界都为2 ms，整体理想下界为2 ms。

</details>

<details class="self-test" id="self-test-6">
<summary>Q6：F=200 GFLOP、HBM流量2 GB，实测8 ms，峰值100 TFLOP/s与1 TB/s。HBM跑满了吗？</summary>

答：实测25 TFLOP/s和250 GB/s，都为匹配峰值25%。不支持HBM跑满；还需检查依赖、并行请求、指令发射和同步等限制。

</details>

<details class="self-test" id="self-test-7">
<summary>Q7：指针追逐的下一地址依赖前次返回。优先增加纯算力，还是算法允许时增加独立访问链？四次200 ns读取呢？</summary>

答：选独立访问链。纯算力不解除地址依赖。同一链四次读取为800 ns；完全独立且可全并行时为200 ns，但单次读取仍为200 ns，不是50 ns。

</details>

<details class="self-test" id="self-test-8">
<summary>Q8：GPU-Util接近100%，实测计算吞吐只有匹配峰值25%，矛盾吗？</summary>

答：不矛盾。前者关注采样期间是否有kernel执行，后者是FLOP/s相对峰值。执行中的kernel可能等待访存、通信或同步，不能据此认定算力跑满。

</details>

<details class="self-test" id="self-test-9">
<summary>Q9：10 ms内累计只有2 ms kernel，其余8 ms无kernel，先查什么？</summary>

答：先看系统时间线与关键路径，检查CPU供给、提交、同步、拷贝和其他stream。无kernel不自动等于设备完全空闲，不宜先优化只占2 ms的计算。

</details>

<details class="self-test" id="self-test-10">
<summary>Q10：CPU准备每批8 ms、GPU执行2 ms，允许流水重叠。首批延迟、完成节拍与吞吐？</summary>

答：首批10 ms；理想稳态每8 ms完成一批，即125 batch/s。首批延迟与稳态完成间隔不是一个量。

</details>

<details class="self-test" id="self-test-11">
<summary>Q11：CPU时钟包围异步op得到0.1 ms，能认为GPU计算完成吗？</summary>

答：不能，通常只测到调用与提交。读取结束时间前要等待目标工作；要排除旧工作，还应先划清起点。同步范围需匹配测量对象。

</details>

<details class="self-test" id="self-test-12">
<summary>Q12：起点时同stream旧任务还需3 ms，新op需2 ms，只有op后同步，测到多少？</summary>

答：约5 ms。若只测新op，应在开始计时前先等旧工作完成；再执行op并在读取结束时间前等待目标完成。

</details>

<details class="self-test" id="self-test-13">
<summary>Q13：A冷启动80 ms、稳态4 ms；B冷启动20 ms、稳态5 ms，B是否全面加速？</summary>

答：不是。冷启动快4倍，但稳态耗时增加25%。长期重复服务与短生命周期任务要分别评价，报告冷启动和稳态。

</details>

<details class="self-test" id="self-test-14">
<summary>Q14：旧实现batch8耗时8 ms，新实现batch32耗时16 ms，能说代码吞吐翻倍吗？</summary>

答：分别1000与2000条/s，但batch变化混杂了结论。补齐两种实现各自在batch8和32的对照，固定其他条件再归因。

</details>

<details class="self-test" id="self-test-15">
<summary>Q15：A五次都是10 ms；B为8、10、10、10、10 ms，可报告稳定加速吗？</summary>

答：不能。两组中位数都是10 ms。增加重复、交换顺序，并检查同步、频率、缓存与负载，验证8 ms能否复现以及来源。

</details>

<details class="self-test" id="self-test-16">
<summary>Q16：SLO为P99≤20 ms且吞吐≥900/s。A为18 ms与1000/s，B为35 ms与1300/s，选谁？</summary>

答：A同时达标，B不满足尾延迟。优化B需追慢请求的排队、输入长度、batch组成、CPU/GPU时间线，不能只看更高总吞吐。

</details>

<details class="self-test" id="self-test-17">
<summary>Q17：reference=[0,1000]，candidate=[5e-7,1002]，atol=1e-6、rtol=1e-3，是否通过？</summary>

答：第一项允许1e-6，误差5e-7，通过；第二项允许1.000001，误差2，不通过。整体未通过，不能只凭耗时减少采用。

</details>

<details class="self-test" id="self-test-18">
<summary>Q18：A本轮allocated峰值6 GB、B为5 GB；B reserved更高，说明什么？</summary>

答：B张量分配峰值降低1 GB。reserved更高不能单独证明泄漏；还需观察同条件重复运行、缓存策略、引用和在途存储。

</details>

<details class="self-test" id="self-test-19">
<summary>Q19：Rank0在10 ms进入通信，Rank1在40 ms才就绪，随后传输5 ms。传输时间减半能省多少？</summary>

答：Rank0区间含等待30 ms和有效传输5 ms。完成从45变42.5 ms，只省2.5 ms；应先追晚rank上游计算、数据准备与负载不均。

</details>

<details class="self-test" id="self-test-20">
<summary>Q20：S1事件时间戳为0与2 ms，其他stream到8 ms才结束；全设备同步后读取事件，差是多少？</summary>

答：仍为2 ms。CPU等到8 ms不改变已记录时间戳；但这两个事件也不能自动代表整个设备任务的8 ms。

</details>

<details class="self-test" id="self-test-21">
<summary>Q21：start事件0 ms，kernel在0～2 ms；CPU到6 ms才提交end，之后GPU立即记录。事件差呢？</summary>

答：约6 ms，包含2～6 ms的空闲。Event测的是两个事件之间经过的时间，不是只累加kernel忙碌时间。

</details>

<details class="self-test" id="self-test-22">
<summary>Q22：10批，CPU准备3 ms、GPU执行7 ms。逐批同步与理想流水各需多久？</summary>

答：逐批同步10×(3+7)=100 ms；流水首批10 ms，之后9×7 ms，共73 ms。持续吞吐应避免逐轮全设备同步破坏重叠。

</details>

<details class="self-test" id="self-test-23">
<summary>Q23：关闭Profiler时A10 ms、B11 ms；开启时A30 ms、B20 ms，应报告什么？</summary>

答：正常运行B耗时增加10%，没有加速。Profiler对两种实现的扰动可不同；用它定位，再关闭验证最终收益。

</details>

<details class="self-test" id="self-test-24">
<summary>Q24：N=1e8个FP32元素，y=a+b、z=y*c，无缓存且融合中间值留寄存器，带宽1 TB/s。理想流量和时间？</summary>

答：未融合六次数组读写，共2.4 GB、2.4 ms；融合四次，共1.6 GB、1.6 ms，理想加速1.5倍。真实结果仍需测。

</details>

<details class="self-test" id="self-test-25">
<summary>Q25：未融合2.4 GB/2.4 ms；融合1.6 GB/3.2 ms，寄存器32→128、活跃warp32→8，如何判断？</summary>

答：平均HBM带宽1000→500 GB/s，融合反而更慢。寄存器压力和延迟隐藏不足是待验证假设；查就绪warp、等待和发射，不能只凭occupancy定论。

</details>

<details class="self-test" id="self-test-26">
<summary>Q26：限制寄存器后，HBM流量1.6→3.2 GB、耗时3.2→4 ms，活跃warp增加，是否成功？</summary>

答：新带宽800 GB/s，但耗时更长。可能spilling增加local memory及部分HBM流量；需计数和源码证据，不把更高带宽或occupancy当收益。

</details>

<details class="self-test" id="self-test-27">
<summary>Q27：90个短请求、10个长请求，串行。A短1/长10 ms，B短0.5/长20 ms，整体怎样？</summary>

答：A=190 ms，B=245 ms，整体加速190/245≈0.78，实际变慢。不能直接平均不同类别的加速比。

</details>

<details class="self-test" id="self-test-28">
<summary>Q28：稳态A平均在途64、吞吐2000/s；B为128、2200/s。平均延迟呢？</summary>

答：由L=λW得A32 ms、B约58.2 ms。吞吐提高10%并不说明请求更快；需相同负载点与多负载曲线比较，边界必须一致。

</details>

<details class="self-test" id="self-test-29">
<summary>Q29：greedy logits：参考A5/B4.9999，优化A4.9998/B5，已通过局部容差。是否质量不变？</summary>

答：参考选A，优化选B。局部容差不能保证选词和后续质量；固定相同前缀查logits，并在代表性任务、长度和解码配置下评估质量。

</details>

<details class="self-test" id="self-test-30">
<summary>Q30：8卡扩16卡吞吐仅增20%，GPU-Util95%、某rank NCCL变长。如何检验网络瓶颈？</summary>

答：先核对工作负载/并行配置、硬件拓扑/环境、有效token和计时口径。匹配同次collective，区分晚就绪与有效传输。可预置相同真实GPU输入，保留mask、rank分配和MoE路由，记录缓存/显存变化；无改善不等于证明网络故障。

</details>

<details class="self-test" id="self-test-31">
<summary>Q31：怎样实现同GPU、同stream的独立推理调用benchmark，并测额外显存峰值？</summary>

答：第5节给出完整函数。关键是Event初始化和预热移出测量，完成后取allocated基线并重置峰值，op两侧记录Event、等end完成，释放各轮输出；返回中位数、峰值和峰值减基线。本题由助手示范，不等于我已独立手写或GPU实测。

</details>

<details class="self-test" id="self-test-32">
<summary>Q32：逻辑访问10 GB、实际HBM流量1 GB，耗时2 ms，HBM峰值1 TB/s。逻辑带宽超过峰值是错吗？</summary>

答：实际HBM带宽500 GB/s，为峰值50%；逻辑带宽5 TB/s可因缓存复用出现，不必然是计时错误。应匹配流量层级。

</details>

<details class="self-test" id="self-test-33">
<summary>Q33：A有用/传输128/128 B，B为128/1024 B，实际带宽均900 GB/s，谁更有效？</summary>

答：A字节利用率100%，B12.5%；有用吞吐A900、B112.5 GB/s。B实际带宽高仍可能大量无效搬运，应检查布局和合并访问。

</details>

<details class="self-test" id="self-test-34">
<summary>Q34：累计算子耗时40→20 ms，端到端100→98 ms，优化无效吗？</summary>

答：不能这样判断。若全部在串行关键路径且其他不变，本应80 ms；实际98 ms提示重叠、关键路径外工作或其他开销变化，需对比跨stream和CPU/拷贝/同步时间线。

</details>

<details class="self-test" id="self-test-35">
<summary>Q35：10000请求：9880个排队1+计算1 ms；60个100+1；60个1+100。nearest-rank P99是什么？</summary>

答：两阶段各有9940个1 ms样本，阶段P99均1 ms；端到端9880个2 ms、120个101 ms，P99为101 ms。先按同一请求汇总再求分位数。

</details>

<details class="self-test" id="self-test-36">
<summary>Q36：真实每10 ms到一请求，服务暂停0～100 ms；闭环脚本只等t=0请求完成后补发，有何遗漏？</summary>

答：10～90 ms本应到达的请求会排队，闭环未发出它们，可能低估尾延迟。匹配到达模型，记录计划到达/实际发送/完成及超时失败和未完成请求。

</details>

<details class="self-test" id="self-test-37">
<summary>Q37：1200/s到达、1000/s完成，持续10秒后停止发送。队列与成功率怎样解释？</summary>

答：到达12000、完成10000、未完成2000，排空需2秒。已完成全成功不代表全部请求；也不能把83.3%直接当最终成功率。队列每秒增200，不能持续承载。

</details>

<details class="self-test" id="self-test-38">
<summary>Q38：单请求固定1 ms，A每2 ms来一个，B每2秒突发1000个。均值和P99怎样？</summary>

答：长期到达与完成吞吐均500/s。A延迟P99=1 ms；B每批1～1000 ms，nearest-rank P99=990 ms。平均流量相同不保证尾延迟相同，需保留突发性。

</details>

<details class="self-test" id="self-test-39">
<summary>Q39：A TTFT200/TPOT10 ms，B100/15 ms；输出100、10、21 token时如何比较？</summary>

答：100个时A1190、B1585 ms，A快；10个时A290、B235 ms，B快；21个时都400 ms。结合输出长度分布和SLO，不能宣称B全面更快。

</details>

<details class="self-test" id="self-test-40">
<summary>Q40：101个输出token，TTFT100 ms。A的100个间隔均10 ms；B的99个5 ms、一个505 ms，平均与P99呢？</summary>

答：两者平均TPOT10 ms、最后token均1100 ms。B逐token间隔nearest-rank P99=5 ms，但仍有505 ms停顿；补最大值、更高分位数和阈值超标比例。

</details>

<details class="self-test" id="self-test-41">
<summary>Q41：A完成1000/s且联合SLO95%；B完成1100/s，两项SLO各90%但未知重合。达标吞吐？</summary>

答：A950/s。B联合比例80%～90%，达标880～990/s，不能认定B更好；需按请求关联两项指标。

</details>

<details class="self-test" id="self-test-42">
<summary>Q42：真实超标率1%，100次独立测量零次超标的概率？要把该概率降到5%以下需多少样本？</summary>

答：0.99^100≈36.6%。要求0.99^n<0.05，n>约298.07，至少299且零次超标。样本P99不能直接保证总体；独立性、稳定性和代表性是前提。

</details>

<details class="self-test" id="self-test-43">
<summary>Q43：50个独立窗口，每窗口200请求共享99%正常/1%异常状态。一万个全正常的概率？增加窗口内请求有用吗？</summary>

答：概率0.99^50≈60.5%。独立单元是窗口，窗口数不变时增加每窗口请求不会提升发现异常状态的概率，应增加独立窗口与运行状态覆盖。

</details>

<details class="self-test" id="self-test-44">
<summary>Q44：FP32普通路径峰值25 TFLOP/s，FP16稀疏400 TFLOP/s；FP32做100 GFLOP需5 ms，利用率和极限呢？</summary>

答：实测20 TFLOP/s应对比25，为80%。保持计算量、精度和执行路径不变，计算下界4 ms，加速上限1.25倍；80%仍不能单独证明compute-bound。

</details>

<details class="self-test" id="self-test-45">
<summary>Q45：A TTFT100 ms；B命中20、未命中120 ms。100次同prompt先miss再99hit，以及真实10%hit时呢？</summary>

答：重复压测B均值21 ms，加速约4.76倍；真实10%hit时B均值110 ms，比A慢10%。使用真实复用与到达顺序，控制缓存容量/初态/淘汰，分别报告命中和未命中。

</details>

<details class="self-test" id="self-test-46">
<summary>Q46：A每步8000 token/1s、1000步达标；B16000/1.5s、800步达标，谁更快？</summary>

答：吞吐A8000、B10666.7 token/s，提高33.3%；A需800万token、1000s，B1280万、1200s。B慢20%；仍800步时每步需低于1.25s才比A快。

</details>

<details class="self-test" id="self-test-47">
<summary>Q47：峰值100 TFLOP/s，每步1000有效token。A模型量100 TFLOP、额外0、2s；B额外100、3s，MFU/HFU？</summary>

答：A MFU/HFU均50%；B MFU33.3%、HFU66.7%。有效吞吐500→333.3 token/s。HFU增加可来自重算；若节省显存使目标可行，仍可能值得采用，须比较完整目标。

</details>

<details class="self-test" id="self-test-48">
<summary>Q48：反向0～30 ms，G1/G2/G3在10/20/30就绪。分别通信各8 ms，或全部合并18 ms，结果？</summary>

答：分别累计24 ms、阶段38 ms；合并累计18 ms、阶段48 ms。后者启动晚、失去重叠。单独把G3从8降4可让阶段38→34 ms；优化已隐藏的前两组未必改完成时间。

</details>

<details class="self-test" id="self-test-49">
<summary>Q49：A计算30+通信20 ms串行；B重叠后实测42/35；C限制通信资源后重叠34/40，总耗时与验证？</summary>

答：A50、B42、C40 ms，B相对A约1.19倍。不能用独立30/20预测重叠；需对照独立与重叠、资源限制，观察SM、发射、HBM与等待，并排除频率和负载变化。

</details>

<details class="self-test" id="self-test-50">
<summary>Q50：调度修改使吞吐+15%、TTFT P99降20%，逐token间隔P99升到180 ms，如何诊断？</summary>

答：怀疑大Prefill推迟Decode，但先区分未调度与提交后等待。固定到达轨迹、内容、模型、缓存和并发上限，只调每轮Prefill预算；检查Decode等待及间隔，并同时验收TTFT、达标吞吐、显存和质量。

</details>

<details class="self-test" id="self-test-51">
<summary>Q51：引擎每10 ms生成、输出队列立即收到，客户端每100 ms收到10 token，应该先优化GPU吗？</summary>

答：证据不支持。查输出缓冲、代理和网络；块间隔不等于token生成间隔。只改变缓冲/flush策略，关联生成、发送和接收时间；生成不变而客户端停顿缩短支持缓冲假设。

</details>

<details class="self-test" id="self-test-52">
<summary>Q52：每步1s、每50步提交快照、后台每份80s，500步结束时与全部写完时刻？</summary>

答：提交10份，完成时间130/210/290/370/450s共5份，最新可恢复第250步；全部到850s。长期后台跟不上，报告持久化延迟、积压内存、恢复点滞后、失败与最终排空。

</details>

<details class="self-test" id="self-test-53">
<summary>Q53：上题若内存有界又要每份保留，可否维持1s/步？允许丢旧待写时怎么办？</summary>

答：不能三者兼得；背压下长期平均至少80/50=1.6s/步。可保留一份不可变正在写的快照和一份最新待写；只能替换未开始写的旧快照，不能覆盖正在写的内容。

</details>

<details class="self-test" id="self-test-54">
<summary>Q54：32核关键提交线程满载；50次提交各40μs，kernel各10μs。GPU减半或提交减半怎样？</summary>

答：跨核总利用率约3.1%不能排除CPU瓶颈。基准2010μs；GPU减半2005μs；提交减半1010μs。结合CPU栈与时间线排除数据/同步，再用保持工作一致的图重放对照。

</details>

<details class="self-test" id="self-test-55">
<summary>Q55：图捕获读取A，Python变量改指向同shape的B后重放；普通端到端1ms、只测replay0.4ms，能说2.5倍吗？</summary>

答：图仍读A。把新数据正确复制到静态A并建立依赖。先修正输入正确性，再统一输入准备到输出可用的范围，包含必要拷贝、同步、输出处理与对应启动成本，不能直接比1和0.4。

</details>

<details class="self-test" id="self-test-56">
<summary>Q56：S第一次写O为0～2ms、第二次覆盖2～4；T等第一次完成后在2～3复制O，安全吗？</summary>

答：不安全，T读O与S第二次写O竞争。复制完成后记事件，S等它再覆盖；第二次最早3～5ms。只需相关跨stream依赖，不必每轮全设备同步。

</details>

<details class="self-test" id="self-test-57">
<summary>Q57：A在S写0～2ms，T等后读2～6ms；CPU在3ms释放最后引用并在S分配B，风险是什么？</summary>

答：A存储可能被B复用，S写B与T读A冲突。保留A引用到读取结束，或正确record_stream，或手工同步生命周期。record_stream不能替代T等待S写完的执行依赖。

</details>

<details class="self-test" id="self-test-58">
<summary>Q58：t=0～7ms各提交1GB输入，共8份；GPU依次各读4ms。t=7ms存储多少，2GB预算怎么办？</summary>

答：仅第一份完成，其余7份占7GB，是在途积压不必然泄漏。用两个缓冲区按完成事件循环复用，没有空闲就背压。限制提前提交不必然降低GPU吞吐，检查空洞与稳态吞吐。

</details>

<details class="self-test" id="self-test-59">
<summary>Q59：24GB卡reserved22、allocated14；8GB空闲碎片最大1GB且无可整段释放的段，驱动空闲2GB。申请3GB失败为何？</summary>

答：在题设禁止迁移/重映射的模型中，缓存无3GB块、驱动也不足3GB，合计10GB不能直接拼起来。不证明泄漏；empty_cache不能释放活跃张量或直接合并隔开的块。查快照、inactive_split、峰值与active_awaiting_free。

</details>

<details class="self-test" id="self-test-60">
<summary>Q60：A4卡完成1000/s、联合95%；B8卡1600/s、联合90%。8卡预算如何选？</summary>

答：A达标950、每GPU237.5；B达标1440、每GPU180。双A若互不干扰可达1900，高于单B。实测验证路由均衡、请求组成/突发、共享资源和前缀缓存变化。

</details>

<details class="self-test" id="self-test-61">
<summary>Q61：A→B为10/7ms；B→A也为10/7ms，能说B更快或直接归因L2吗？</summary>

答：不能；最明确的是第二次更快。配合缓存命中与实际HBM流量等验证原因，匹配输入及缓存状态并交换顺序。复用业务测相应热态，新输入业务用轮换缓冲区复现工作集。

</details>

<details class="self-test" id="self-test-62">
<summary>Q62：A每请求10ms，B执行5ms且每新shape构图额外100ms。1000请求、1或50种shape，以及一般式？</summary>

答：1种时A10000、B5100ms，约1.96倍；50种时B10000ms，无加速。一般B=5N+100K，K<N/20才更快。报告构图次数/耗时、命中、淘汰、重构与包含构图的总耗时。

</details>

<details class="self-test" id="self-test-63">
<summary>Q63：四条128和四条1024输入，混批补齐耗10ms，分短长两批共8ms，含padding与有效吞吐？</summary>

答：有效4608 token；A槽位8192、含padding819200/s、有效460800/s；B槽位4608、两种均576000/s。B有效吞吐+25%。在线分桶还需检查凑批、TTFT、尾延迟、联合达标吞吐与饥饿。

</details>

## 18. 互动图索引

正文保留12张本模块正式互动图，并引用5张相关侧边会话图。图中的状态和答案用于理解各自模型；不会把滑块状态当成新的答题记录。

1. [串行阶段与Amdahl定律](/content/interactive/performance-analysis/performance-critical-path.html?v=height-v2)（本模块原图）。
2. [计算与通信重叠的关键路径](/content/interactive/performance-analysis/compute-communication-overlap.html?v=height-v2)（本模块原图）。
3. [计算/访存理想下界与算术强度](/content/interactive/performance-analysis/kernel-time-bounds.html?v=height-v2)（本模块原图）。
4. [异步提交与完成计时](/content/interactive/performance-analysis/cuda-async-timing.html?v=height-v2)（本模块原图）。
5. [allocated、reserved与峰值](/content/interactive/performance-analysis/cuda-memory-accounting.html?v=height-v2)（本模块原图）。
6. [同步范围与Event时间](/content/interactive/performance-analysis/cuda-sync-scope.html?v=height-v2)（本模块原图）。
7. [请求混合与整体加速](/content/interactive/performance-analysis/workload-mix-speedup.html?v=height-v2)（本模块原图）。
8. [合并访问与有效字节](/content/interactive/performance-analysis/coalesced-memory-access.html?v=height-v2)（本模块原图）。
9. [到达、完成与积压](/content/interactive/performance-analysis/arrival-completion-backlog.html?v=height-v2)（本模块原图）。
10. [梯度分桶的启动与重叠](/content/interactive/performance-analysis/gradient-bucket-overlap.html?v=height-v2)（本模块原图）。
11. [CPU提交开销与GPU空洞](/content/interactive/performance-analysis/cpu-kernel-launch-bottleneck.html?v=height-v2)（本模块原图）。
12. [构图成本摊销](/content/interactive/performance-analysis/graph-build-amortization.html?v=height-v2)（本模块原图）。
13. [有效带宽与实际带宽](/content/interactive/gpu-cuda/cuda-bandwidth-metrics.html?v=height-v2)（关联会话原图）。
14. [长Prefill分块与Decode停顿](/content/interactive/inference-serving/chunked-prefill-timeline.html?v=height-v2)（关联会话原图）。
15. [跨rank就绪与通信瓶颈](/content/interactive/distributed-training/rank-timeline-bottleneck-diagnosis.html?v=height-v2)（关联会话原图）。
16. [激活重计算的时间与显存](/content/interactive/distributed-training/activation-checkpoint-recompute.html?v=height-v2)（关联会话原图）。
17. [跨stream事件依赖与完成时刻](/content/interactive/gpu-cuda/cuda-stream-event-dependency.html?v=height-v2)（关联会话原图）。

## 19. 参考资料与进一步核对

优先使用官方文档核对定义、API边界和工具行为；本文的课堂算例和个人复盘来自本次学习对话。

- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：系统时间线、CPU/GPU及通信分析。
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：Roofline、执行与访存指标、分析工具影响。
- [NVIDIA NVML utilization定义](https://docs.nvidia.com/deploy/archive/R550/nvml-api/structnvmlUtilization__t.html)：GPU-Util的统计口径。
- [PyTorch CUDA语义](https://docs.pytorch.org/docs/2.14/notes/cuda.html)：异步、stream、缓存分配器与CUDA Graph。
- [PyTorch CUDA Event](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.streams.Event.html)：事件、计时和等待。
- [PyTorch record_stream](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.record_stream.html)：侧stream使用与分配器生命周期。
- [PyTorch CUDA内存分析](https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html)：快照、segment/block和在途状态。
- [NVIDIA AIPerf指标定义](https://docs.nvidia.com/aiperf/dev/reference/ai-perf-metrics-reference)：TTFT、ITL、响应块与计时边界。
- [wrk2对协调遗漏的解释](https://github.com/giltene/wrk2)：计划到达与响应驱动的压测差异。
- [PyTorch FSDP训练吞吐案例](https://pytorch.org/blog/maximizing-training/)：重计算、MFU/HFU和端到端训练表现。
- [vLLM Chunked Prefill调优](https://docs.vllm.ai/en/v0.18.0/configuration/optimization/)：Prefill/Decode调度中的吞吐与延迟取舍，具体默认值依版本核对。

<script src="/js/transformer-interactives.js" defer></script>
