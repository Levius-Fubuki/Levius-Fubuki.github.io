# GPU、CUDA 与算子优化：从线程、访存到稳定 Softmax 和 FlashAttention

这篇笔记整理大模型训练与推理优化面试中的 GPU / CUDA 模块，也收录了相关性能分析会话的重点。目标是能解释数据怎样移动、线程怎样合作、结果为什么正确，以及优化为什么可能有效。文中的耗时、带宽和资源配置如未特别说明，均为教学假设，不代表真实 GPU 测量。

学习顺序是：执行模型 → 存储与访存 → 性能上限 → GEMM → reduction → stable softmax → FlashAttention → LayerNorm / RMSNorm → 融合与异步执行 → Graph → profiling。后面的优化都建立在前面的正确性条件上。

## 1. 用一个 Attention 子层串起整个知识链

考虑单个 attention head 的一次 prefill，序列长度 N=4096，head dimension d=64，Q、K、V 的形状均为 [4096,64]。为方便推导，基线分数与概率使用 FP32 存储；乘法输入精度、累加精度在实际实验中另外明确。

计算链是：

```text
输入准备 / 必要的 H2D
  → S = Q Kᵀ / √64
  → 因果 mask
  → P = 每行 stable softmax(S)
  → O = P V
  → T = O + residual
  → Y = RMSNorm(T; γ)
  → 后续 GPU 算子 / 必要的 D2H
```

这是教学用的 attention 子层骨架，省略了 QKV 投影、输出投影和多头拼接；不能直接当成完整 Transformer block。真实推理应尽量让中间数据留在 GPU，不是每层都做 H2D / D2H。

从这一个例子出发，每类知识都有具体位置：

| 问题 | 本例中的落点 | 优化依据 |
| --- | --- | --- |
| 谁负责输出 | block 负责矩阵块或 softmax 的一行，线程负责若干元素 | thread / warp / block / SM 映射 |
| 数据怎样移动 | Q/K/V 从 global memory 进入 shared memory，再进入寄存器 | 合并访存、bank、缓存和复用 |
| 矩阵乘法怎样提速 | QKᵀ 和 PV | tiling、寄存器分块、双缓冲、Tensor Core |
| 行统计量怎样算 | softmax 的最大值与指数和 | 线程局部 → warp → block 归约 |
| 为什么可能 NaN | 直接 exp 大分数，或全遮挡行做 −∞−(−∞) | 数值稳定与有效元素约定 |
| 为什么能不保存 P | 按 K/V 块更新每行 m、ℓ、U | online softmax 与 FlashAttention |
| 为什么融合有收益 | residual 与 RMSNorm 之间的 T | 减少中间写回和重读 |
| 为什么多 Stream 仍可能慢 | 数据依赖、默认流隐式顺序、资源竞争 | Event、真实时间线 |
| 为什么重放可能更快 | 同一计算链重复执行 | CUDA Graph 减少提交开销 |
| 怎样证明优化有效 | 从本子层端到端耗时，再定位热点 kernel | 对照实验、误差验证和 profiling |

两个 N×N FP32 中间矩阵 S、P 各占 4096²×4=64 MiB。同时物化时二者合计 128 MiB，还没有计入 Q/K/V/O、workspace 等。若忽略缩放与 softmax，两个 GEMM 合计约 4N²d=4,294,967,296 FLOP；因果计算若有效跳过未来区域会减少常数，但稠密注意力仍随 N 二次增长。

因此本例有两类不同优化：提高矩阵乘加效率，以及避免大中间矩阵进入 HBM。只背“用了 Tensor Core”或“用了融合”都不能解释整条计算链。

把同一计算链缩成一个可手算的 query：d=2，q=[1,0]，四个 keys 为 [0,0]、[√2,0]、[2√2,0]、[3√2,0]。缩放后的分数为 [0,1,2,3]。若这是位置 2 的 query，因果 mask 后只允许前三项，稳定 softmax 的分母为 ℓ=e⁻²+e⁻¹+1。对应 values 为 [1,0]、[0,1]、[1,1]，于是 O=[(e⁻²+1)/ℓ,(e⁻¹+1)/ℓ]≈[0.755272,0.909969]。加 residual=[0.1,−0.1]，得到 T≈[0.855272,0.809969]；取 γ=[1,1]、忽略 ε 后，RMSNorm 以 sqrt((T₀²+T₁²)/2) ≈0.832929 为公共分母，得到 Y≈[1.026825,0.972436]。GEMM 负责分数和加权和，归约负责 m/ℓ 与均方；FlashAttention 改变中间数据的生命周期，融合保留 T 的寄存器值，Stream / Graph 改变执行组织，而不应改变上述数学目标。这里用实数 / FP64 手算近似，实际 FP16 路径须单独验证舍入误差。

## 2. 执行模型：逻辑线程怎样落在 GPU 上

**Thread** 是一次 kernel 中的逻辑执行实例，有自己的索引和私有变量。**Warp** 通常包含 32 个线程，是理解指令发射、分支和访存的基本分组。**Block** 包含多个 warp，同一 block 的线程可以通过 shared memory 和 block 屏障协作。**Grid** 是本次 kernel 启动的全部 blocks。**SM** 是承载 blocks / warps 的硬件执行单元，包含计算管线、寄存器等资源。

一个普通 block 的线程在同一 SM 上执行；多个 blocks 可以分批驻留和调度。逻辑线程数不等于物理 CUDA 核心数，也不意味着所有线程同时运行。普通 kernel 不能假定不同 blocks 的执行顺序，更不能随意用全局忙等模拟安全的 grid 屏障。

处理 1000 个元素、每 block 128 线程：需要 ceil(1000/128)=8 个 blocks，每 block 4 个 warp，总共启动 1024 个线程。最后 24 个线程只负责边界处理。全局索引：

```cpp
std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
if (i < n) out[i] = a[i] + b[i];
```

block 7 的 thread 110 对应 i=1006，线程已经启动，但不会访问数组。二维 block 的线性线程编号按 x 维优先展开，不能把所有二维块都想象成“一行恰好一个 warp”：例如 blockDim=(16,16) 时，一个 warp 跨两行。

<details class="interactive-figure">
<summary>互动图 01 · 线程、线程块与 SM 调度</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-execution-basics.html?v=height-v2" title="线程、线程块与 SM 调度" class="transformer-interactive" data-widget="cuda-execution-basics" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-execution-basics.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 02 · 全局索引与边界线程</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-thread-index.html?v=height-v2" title="全局索引与边界线程" class="transformer-interactive" data-widget="cuda-thread-index" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-thread-index.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 3. 存储层次与同步：先分清“在哪里”和“谁能看到”

| 存储 | 典型用途与共享范围 | 必须注意 |
| --- | --- | --- |
| Register | 单线程索引、累加器、临时值 | 数量有限，使用量影响驻留；数组不保证一定分配在寄存器 |
| Shared memory | 同 block 协作暂存 tile / warp 部分结果 | 显式管理，存在 bank conflict，需要正确同步 |
| L1 / L2 cache | 硬件缓存，改善访问局部性 | 缓存命中会改变真实 DRAM 流量；作用范围依架构 |
| Global memory | 大数组、跨 kernel 的中间结果 | 地址可由不同线程访问，可见性与执行顺序需另行保证 |
| Local memory | 线程私有地址空间，如溢出的寄存器或部分私有数组 | “私有”不等于“片上”，其后备通常是设备内存 |

GPU 性能规格至少要看：对应精度与模式的计算吞吐、HBM 带宽、HBM 容量、设备互联带宽与延迟。容量决定能否装下模型与工作集，带宽决定数据移动上限，互联影响跨卡通信；“显存大”不能直接推出“kernel 快”。

`__syncthreads()` 建立 block 内的会合与先前相关内存操作对后续访问的可见性。它不是跨 block 同步，也不会把某个线程的寄存器值自动复制给其他线程。`__syncwarp()` 只覆盖其指定参与范围的 warp 内同步，不能替代跨 warp 的 block 屏障。

```cpp
if (tid == 0) s[0] = 42;
__syncthreads();
if (tid == 64) out[blockIdx.x] = s[0];
```

把屏障仅放进 `if (tid==0)`，而其他仍在执行的线程不参与对应屏障，不满足这个协作程序的参与条件，可能挂起或产生未定义行为。边界处理也应结合算法判断：独立逐元素 kernel 可以让无效线程退出；后续仍要合作的归约线程通常应该填单位元并继续参与。不能把“无效线程一律 return”当通用模板。

<details class="interactive-figure">
<summary>互动图 03 · 寄存器、共享内存与可见性</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-memory-scope.html?v=height-v2" title="寄存器、共享内存与可见性" class="transformer-interactive" data-widget="cuda-memory-scope" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-memory-scope.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 4. 合并访存、转置与 bank conflict

### 4.1 合并访存看的是同一条访问指令的 warp 地址集合

在课堂约定的 32B sector 模型中，32 个 lane 各读一个 FP32：若起点对齐且地址连续，共请求 128B，覆盖 4 个 sectors，段内有效数据利用率 100%。若相邻 lane 相隔 32 个 float，即 128B，每人落在不同 sector，则覆盖 32 个 sectors，每段只用 4B，利用率 12.5%。

这比较的是某次请求的地址覆盖，不是在保证整个 kernel 的实际 HBM 流量相差 8 倍。缓存、重复访问、读写路径和其他指令都可能改变实际流量。单线程跨循环迭代的步长很大，也不能直接判定“不合并”：grid-stride loop 的同一次迭代中，相邻 lane 仍可读取连续位置。

<details class="interactive-figure">
<summary>互动图 04 · 连续访问与跨步访问</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-coalescing.html?v=height-v2" title="连续访问与跨步访问" class="transformer-interactive" data-widget="cuda-coalescing" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-coalescing.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 4.2 矩阵转置需要改变线程之间的数据交接方式

按行存储的矩阵，地址是 row×width+col。直接转置 `out[col][row]=in[row][col]`，让 warp 连续读输入时，往往会跨行写输出；反过来安排则变成连续写、跨行读，只是交换了两侧的访存效率。

可以先合作把输入矩阵块写入 shared tile，再转置读取 tile 并连续写输出。载入元素与写回该元素的线程不必相同。课堂中 `tile[2][7]` 由 (tx=7,ty=2) 载入，再由 (tx=2,ty=7) 写到 out[7][2]。中间需要 block 同步，保证其他线程已经写完。

<details class="interactive-figure">
<summary>互动图 05 · 矩阵转置的读写布局</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-matrix-access.html?v=height-v2" title="矩阵转置的读写布局" class="transformer-interactive" data-widget="cuda-matrix-access" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-matrix-access.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 06 · 共享 tile 在线程间交接</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-tiled-transpose.html?v=height-v2" title="共享 tile 在线程间交接" class="transformer-interactive" data-widget="cuda-tiled-transpose" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-tiled-transpose.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 4.3 Shared memory 也有访问布局问题

在本节 32 banks、每个 FP32 占一个 4B word 的简化条件下：bank=word_index mod 32。`float tile[32][32]` 的同列访问步长为 32，容易把不同地址集中到同一个 bank；改成 `[32][33]` 后，列访问 bank 随 lane 轮转。步长 34 时 bank=(lane×34)%32，前四个为 0、2、4、6，只使用 16 个 banks，每个 bank 两个不同地址，属于 2 路冲突。

同一个 warp 读取同一个地址时可以广播，不能把它误报为 32 路冲突。64 位数据、向量化访问及具体指令事务划分要按实际情况分析，不能机械套这个 FP32 模型。

<details class="interactive-figure">
<summary>互动图 07 · Shared memory 的 bank 冲突</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-bank-conflicts.html?v=height-v2" title="Shared memory 的 bank 冲突" class="transformer-interactive" data-widget="cuda-bank-conflicts" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-bank-conflicts.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 5. 分支发散、occupancy 与 Roofline

### 5.1 发散发生在 warp 内

如果同一 warp 的部分 lane 走 A、另一部分走 B，就可能分路径执行，并用活跃掩码限制参与线程。若 A、B 各 8 条指令，则这个简化模型中一个发散 warp 要经历两条路径的 16 条指令；若 warp 0 全走 A、warp 1 全走 B，每个 warp 各执行 8 条。不同 warp 选择不同分支不叫 warp 内发散。具体生成指令还可能使用 predication，必须结合编译结果判断实际开销。

<details class="interactive-figure">
<summary>互动图 08 · Warp 内分支发散</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-warp-divergence.html?v=height-v2" title="Warp 内分支发散" class="transformer-interactive" data-widget="cuda-warp-divergence" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-warp-divergence.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 5.2 Occupancy 是驻留 warp 比例，不是性能分数

理论 occupancy=SM 可驻留 warp 数 / 该 SM 最大支持的驻留 warp 数。限制可能来自寄存器、shared memory、线程数、warp 数和 block 数上限，还存在硬件分配粒度。

假设一个 block 有 256 线程，即 8 warps，寄存器和 shared memory 都最多支持 4 个 blocks 驻留，SM 最多支持 64 warps，则 occupancy=4×8/64=50%。即便把每线程寄存器从 64 降到 32，使寄存器允许 8 blocks，shared memory 仍只允许 4 blocks，occupancy 不变。

更多驻留 warp 可能隐藏等待，但强行降低寄存器可能减少数据复用或导致 spilling；更高 occupancy 不保证更快。

<details class="interactive-figure">
<summary>互动图 09 · 资源约束与 occupancy</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-occupancy.html?v=height-v2" title="资源约束与 occupancy" class="transformer-interactive" data-widget="cuda-occupancy" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-occupancy.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 5.3 Roofline 先估计算力和带宽哪个更紧

算术强度 I=FLOP / 搬运字节数；性能上限的简化模型为 P≤min(P_peak,BW×I)。必须说明字节数在哪一层统计：源码逻辑访问、HBM 访问还是 shared memory 访问。

FP32 向量加法每元素 1 FLOP，读两个 float、写一个 float，逻辑流量 12B，I=1/12 FLOP/B。假设 HBM 带宽 3 TB/s，带宽屋顶是 0.25 TFLOP/s；即使算力屋顶提高到 120 TFLOP/s，也不会改变这个带宽屋顶。Roofline 是上限推理，延迟、低并行度、依赖链等仍可能让实际性能远低于它。

<details class="interactive-figure">
<summary>互动图 10 · 算术强度与 Roofline</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-roofline.html?v=height-v2" title="算术强度与 Roofline" class="transformer-interactive" data-widget="cuda-roofline" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-roofline.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 6. GEMM：从一个线程一个输出，到多层数据复用

### 6.1 朴素 GEMM 与线程内累加

对 A[M,K]、B[K,N]，C[row,col]=Σk A[row,k]B[k,col]。一个线程负责一个输出，`acc` 留在寄存器中，沿 K 维累加。

例如 K=N=64，warp 的线程负责第 2 行、连续 32 列，在 k=5 这一轮，所有线程读 A[2×64+5]=A[133]，B 的索引为 352～383。这次 B 读取是连续的，虽然单线程下次 k 的地址会跳 256B。合并访存仍不能解决相邻线程重复读取同一批 A/B 的复用问题。

下面补齐一个完整的朴素 GEMM 及启动包装，便于与后面的优化版本比较。它是本文补充的正确性基线，尚未经过 nvcc 编译或 GPU 实测；输入按行连续存储，维度非负，缓冲区合法且不重叠。

```cpp
#include <cuda_runtime.h>
#include <cstddef>

// Contiguous row-major matrices; M,N,K >= 0; non-overlapping buffers.
// One thread computes one output. This is a correctness baseline.
__global__ void gemm_naive_fp32(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    const std::size_t row =
        std::size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t col =
        std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= std::size_t(M) || col >= std::size_t(N)) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * std::size_t(K) + k]
             * B[std::size_t(k) * N + col];
    }
    C[row * std::size_t(N) + col] = acc;
}

// Buffers and stream are already prepared; API/launch errors are checked by caller.
void launch_gemm_baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream
) {
    if (M == 0 || N == 0) return;
    dim3 block(16, 16);
    dim3 grid((unsigned(N) + 15u) / 16u,
              (unsigned(M) + 15u) / 16u);
    gemm_naive_fp32<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
```

<details class="interactive-figure">
<summary>互动图 11 · 朴素 GEMM 地址集合</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-naive-gemm.html?v=height-v2" title="朴素 GEMM 地址集合" class="transformer-interactive" data-widget="cuda-naive-gemm" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-naive-gemm.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 6.2 Shared-memory tiling 的两道屏障

一个 block 负责 T×T 输出，每轮沿 K 读取一个 A tile 和 B tile，线程合作载入后，重复利用片上数据。

```cpp
// 示意：As、Bs 为 shared memory；全 block 遵循相同迭代次数。
for (int base = 0; base < K; base += T) {
    As[ty][tx] = valid_A ? A[a_index] : 0.0f;
    Bs[ty][tx] = valid_B ? B[b_index] : 0.0f;
    __syncthreads();          // 载入完成后再消费
    for (int q = 0; q < T; ++q)
        acc += As[ty][q] * Bs[q][tx];
    __syncthreads();          // 消费完成后再覆盖同一缓冲区
}
```

这里是展示同步结构的伪代码，`valid_A / a_index` 等由完整实现的边界映射提供。一次 q 只给当前线程的 C[ty,tx] 增加一个乘积；一轮 base 增加一段 K 维贡献，全部 base 结束才得到完整结果。删除第二道屏障，快 warp 可能载入下一块并覆盖慢 warp 尚在读取的当前块。

<details class="interactive-figure">
<summary>互动图 12 · GEMM 的 tile 复用与同步</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-gemm-reuse.html?v=height-v2" title="GEMM 的 tile 复用与同步" class="transformer-interactive" data-widget="cuda-gemm-reuse" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-gemm-reuse.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 6.3 为什么扩大 Bₖ 不一定提高算术强度

输出 tile 为 T×T，每轮 K 长度 Bₖ，输入读数为 2TBₖ 个 FP32，即 8TBₖ 字节；计算量为 2T²Bₖ FLOP，所以这个仅统计输入读取的模型给出 I=T/4 FLOP/B。T=16、Bₖ=16：读 512 个 float=2048B，算 8192 FLOP，I=4。只把 Bₖ 变成 32，读取与计算都翻倍，I 不变；把 T 变成 32，则 I=8。

这是说明输入复用的局部模型，未计 C 的读写、缓存、边界等，不应直接当成完整 kernel 的 HBM 算术强度。

<details class="interactive-figure">
<summary>互动图 13 · 输出 tile 与输入算术强度</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-gemm-intensity.html?v=height-v2" title="输出 tile 与输入算术强度" class="transformer-interactive" data-widget="cuda-gemm-intensity" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-gemm-intensity.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 6.4 寄存器分块、向量化、双缓冲和 Tensor Core

让一个线程负责 4×4 输出时，每次 q 取 4 个 A、4 个 B，就能做 16 次 FMA；若逐输出分别取两个操作数，需要 32 次读取，现在是 8 次，显式读取数为原来的 1/4。代价至少有 16 个 FP32 累加器，另有操作数、地址等寄存器需求。

双缓冲让下一块载入与当前块计算重叠。4 块数据，载入 20μs、计算 30μs，串行为 200μs；若重叠理想，流水线为 20+4×30=140μs。额外缓冲区可能降低 occupancy；实现还必须确保生产者没有覆盖仍被消费的缓冲区。普通同步 load 加两块数组不自动等于真正的异步搬运流水线。

向量化 `float4` 表达每线程一次 16B 访问，可减少相关指令，但不自动减少总字节。32 个线程各取 4 个 float，合计 512B，在本节对齐条件下覆盖 16 个 32B sectors。即使 x 原本对齐，x+1 已偏移 4B，不满足 float4 的 16B 对齐要求。130 个元素可处理前 128 个，再用标量处理尾部两个；不能越界读满一个向量。

WMMA 的一个 16×16×16 矩阵乘加包含 4096 FMA=8192 FLOP。它是 warp 协作接口，不能只让 thread 0 调用 `mma_sync`。输入类型、布局、对齐、支持的 shape 和架构都要满足接口约束。FP16 输入、FP32 累加仍可能与纯 FP32 GEMM 不同，因为输入转换已经损失精度，舍入与累加顺序也会不同。

<details class="interactive-figure">
<summary>互动图 14 · 寄存器分块的复用与成本</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-register-tiling.html?v=height-v2" title="寄存器分块的复用与成本" class="transformer-interactive" data-widget="cuda-register-tiling" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-register-tiling.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 15 · 双缓冲的数据生命周期</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-double-buffering.html?v=height-v2" title="双缓冲的数据生命周期" class="transformer-interactive" data-widget="cuda-double-buffering" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-double-buffering.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 16 · 向量化载入、对齐与尾部</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-vector-loads.html?v=height-v2" title="向量化载入、对齐与尾部" class="transformer-interactive" data-widget="cuda-vector-loads" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-vector-loads.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 17 · Tensor Core 与 WMMA 协作</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-tensor-core-wmma.html?v=height-v2" title="Tensor Core 与 WMMA 协作" class="transformer-interactive" data-widget="cuda-tensor-core-wmma" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-tensor-core-wmma.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 7. Reduction：线程私有 → warp → block → grid

### 7.1 先理解单位元与依赖深度

256 项二叉树求和需要 log₂256=8 轮、共 255 次加法。8 是依赖深度，不是只执行了 8 次加法。短输入需要让无效槽位贡献求和单位元 0；例如 8 槽位处理 N=6，若后两个槽位未初始化，第一轮就可能读到未定义值。

Warp shuffle 让同 warp 的线程交换寄存器值。`__shfl_down_sync(mask,v,offset)` 只负责交换，不自动做加法，也不自动把完整结果广播给每个 lane。对下移树归约，最终仅 lane 0 保证拥有完整结果；例如初值 lane+1、offset=16，lane 7 读到 lane 23 的 24，自身 8+24=32。

使用 FULL_MASK 的完整 warp 模板时，让所有 32 个 lanes 参与，无效数据填单位元。准确规则包括“mask 指定的尚未退出线程按要求执行相应 intrinsic”和“被读取的源 lane 必须有效参与”；不能仅凭部分线程退出就断言一定死锁，也不能因为没有死锁就认为从 inactive 源 lane 取值正确。[CUDA warp intrinsic 约束](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints)

<details class="interactive-figure">
<summary>互动图 18 · Shared memory 树归约</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-shared-reduction.html?v=height-v2" title="Shared memory 树归约" class="transformer-interactive" data-widget="cuda-shared-reduction" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-shared-reduction.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 19 · Warp shuffle 归约</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-warp-shuffle.html?v=height-v2" title="Warp shuffle 归约" class="transformer-interactive" data-widget="cuda-warp-shuffle" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-warp-shuffle.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 7.2 跨 warp 必须经过共享结果和正确的会合

256 线程有 8 warps，每个 warp 的 lane 0 写一个 FP32 部分和，只需 8×4=32B shared memory。所有线程经过 block 屏障后，warp 0 的 lane 0～7 取部分和，其余 lane 填 0，再归约一次。512 线程则有 16 个部分和，需要 64B。

<details class="interactive-figure">
<summary>互动图 20 · 跨 warp 的 block 归约</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-block-reduction.html?v=height-v2" title="跨 warp 的 block 归约" class="transformer-interactive" data-widget="cuda-block-reduction" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-block-reduction.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 7.3 多 block 与 grid-stride loop

若每线程只读一项，N=1,000,000、block=256，三轮可得到 3907→16→1。若第一轮固定启动 128 blocks，让每个线程使用 `i+=gridDim.x*blockDim.x` 累加多个输入，则第一轮只输出 128 个数，第二轮一个 block 足够。第一轮与第二轮放在同一个 Stream，按顺序执行，中间不需要 CPU 同步；不同 Stream 则需要显式依赖。

例如 grid=4、block=256、blockIdx=2、threadIdx=5，起点 517，stride=1024；若 N=2000，则读取 517、1541，下一项 2565 越界。同轮相邻线程依旧读连续地址。

<details class="interactive-figure">
<summary>互动图 21 · 跨 block 的分轮归约</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-grid-reduction.html?v=height-v2" title="跨 block 的分轮归约" class="transformer-interactive" data-widget="cuda-grid-reduction" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-grid-reduction.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 22 · Grid-stride loop 归约</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-grid-stride-reduction.html?v=height-v2" title="Grid-stride loop 归约" class="transformer-interactive" data-widget="cuda-grid-stride-reduction" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-grid-stride-reduction.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 7.4 完整 reduction 基础实现

约定：FP32 输入输出、一维启动、每 block 固定 256 线程、n≥0，输入与部分和缓冲区不重叠。空输入可用非零 grid 启动，得到零部分和，再归约成零。下面保留会话中的实现；它是算法学习代码，没有声称经过 GPU 性能调优。

```cpp
#include <cuda_runtime.h>
#include <cstddef>

// 同一 warp 的 32 个线程都必须调用。
// 返回后，仅 lane 0 保证持有整个 warp 的和。
__device__ __forceinline__
float warp_sum(float v) {
    constexpr unsigned FULL_MASK = 0xffffffffu;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(FULL_MASK, v, offset);
    }

    return v;
}

__global__
void reduce_sum(const float* x, float* partial, int n) {
    __shared__ float warp_sums[8];  // 8 个 warp，共 32B

    const unsigned tid     = threadIdx.x;
    const unsigned lane    = tid & 31u;
    const unsigned warp_id = tid >> 5;

    // 使用较宽的索引类型，避免索引递增时发生 int 溢出。
    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    // 1. 每个线程独立累加；没有输入可读时保持 0。
    float sum = 0.0f;
    for (std::size_t i = start;
         i < static_cast<std::size_t>(n);
         i += stride) {
        sum += x[i];
    }

    // 2. 每个 warp 内归约。
    sum = warp_sum(sum);

    // 3. 每个 warp 的 lane 0 保存部分和。
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }

    // 4. 等待所有 warp 写完 shared memory。
    __syncthreads();

    // 5. Warp 0 归约 8 个部分和。
    if (warp_id == 0) {
        float block_sum = (lane < 8) ? warp_sums[lane] : 0.0f;

        // Warp 0 的全部 32 个 lanes 都参与。
        block_sum = warp_sum(block_sum);

        if (lane == 0) {
            partial[blockIdx.x] = block_sum;
        }
    }
}
```

两次启动使用同一个已经创建的 Stream s；d_partial 至少容纳 128 个 float，d_result 容纳一个 float：

```cpp
reduce_sum<<<128, 256, 0, s>>>(d_x, d_partial, n);
reduce_sum<<<1, 256, 0, s>>>(d_partial, d_result, 128);
```

正式程序应检查 API 返回码、kernel launch 错误，并在需要观察 GPU 结果的位置同步和检查异步错误。FP32 求和不满足实数加法结合律，与 CPU 串行求和不同顺序可能产生不同舍入。

<details class="interactive-figure">
<summary>互动图 23 · 归约代码与尾部检查</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-reduction-code-challenge.html?v=height-v2" title="归约代码与尾部检查" class="transformer-interactive" data-widget="cuda-reduction-code-challenge" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-reduction-code-challenge.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 8. Stable softmax：先保证归一化范围正确

对一行有限的实数输入：

$$
m=\max_j x_j,\qquad e_i=\exp(x_i-m),\qquad
\ell=\sum_j e_j,\qquad y_i=e_i/\ell.
$$

例如 [1000,1002,1004]，减去 1004 得到 [−4,−2,0]，分母是 e⁻⁴+e⁻²+1。指数输入不大于 0，避免直接 exp 大正数的上溢；至少一个有效项为 1，所以在有限、非空输入这个约定下分母不会变成 0。很小的项仍可能下溢为 0。

这不是 NaN、+∞ 或全为 −∞ 输入的通用处理方案。全遮挡行需要明确输出策略，例如由上层约定输出零，或单独分支处理；不能无条件计算 −∞−(−∞)。

Softmax 最容易写错的不是 exp，而是整行统计量：线程各自对 4 项做 softmax 会产生 256 个局部归一化组，而不是一个 1024 项的归一化。每组和为 1，则整行和可能变成 256。

<details class="interactive-figure">
<summary>互动图 24 · 稳定 Softmax 的指数范围</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-stable-softmax.html?v=height-v2" title="稳定 Softmax 的指数范围" class="transformer-interactive" data-widget="cuda-stable-softmax" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-stable-softmax.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 25 · Block Softmax 的归一化范围</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-block-softmax.html?v=height-v2" title="Block Softmax 的归一化范围" class="transformer-interactive" data-widget="cuda-block-softmax" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-block-softmax.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 9. 三种 CUDA softmax：warp、缓存 block、三遍扫描

### 9.1 一个 warp 一行：两次归约，两次广播

适用于每行 1～32 个有限 FP32 输入。无效位置读作 −∞，指数贡献 0；完整 32 个 lanes 参与归约。求最大值后广播 lane 0 的完整 m，求指数和后再次广播 lane 0 的完整分母。`__syncwarp()` 只同步，不能替代广播。

例如输入 [1004,1000,1002,999,1001]，省去最大值广播，lane 1 的下移归约局部最大值是 1002，它会算 e^(1000−1002)=e⁻²，而正确应使用 1004 得到 e⁻⁴。局部归约结果看起来合理，也不是全行结果。

<details class="interactive-figure">
<summary>互动图 26 · Warp Softmax 与无效槽位</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-warp-softmax.html?v=height-v2" title="Warp Softmax 与无效槽位" class="transformer-interactive" data-widget="cuda-warp-softmax" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-warp-softmax.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 27 · Warp Softmax 代码逐步演示</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-softmax-code-walkthrough.html?v=height-v2" title="Warp Softmax 代码逐步演示" class="transformer-interactive" data-widget="cuda-softmax-code-walkthrough" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-softmax-code-walkthrough.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

```cpp
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>

// 32 个 lanes 全部参与；仅 lane 0 保证得到完整最大值。
__device__ __forceinline__
float warp_max_32(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

// 32 个 lanes 全部参与；仅 lane 0 保证得到完整和。
__device__ __forceinline__
float warp_sum_32(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// 约定：一维 block，blockDim.x == 256。
// x、y 为连续存储的 [rows, cols] 矩阵。
// rows >= 0，1 <= cols <= 32，有效输入均为有限数。
__global__
void softmax_warp_fp32(
    const float* x,
    float* y,
    int rows,
    int cols
) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_id = threadIdx.x >> 5;

    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * 8 + warp_id;

    // 同一 warp 的 row 相同，因此整个 warp 一起退出。
    if (row >= static_cast<std::size_t>(rows)) {
        return;
    }

    const std::size_t base =
        row * static_cast<std::size_t>(cols);
    const bool valid = lane < static_cast<unsigned>(cols);

    // 1. 载入输入；无效位置不影响最大值。
    float v = valid ? x[base + lane] : -CUDART_INF_F;

    // 2. 归约最大值。
    float m = warp_max_32(v);

    // 3. 广播最大值，再计算指数。
    m = __shfl_sync(0xffffffffu, m, 0);
    float e = valid ? expf(v - m) : 0.0f;

    // 4. 归约指数和。
    float denominator = warp_sum_32(e);

    // 5. 广播分母，再归一化。
    denominator = __shfl_sync(0xffffffffu, denominator, 0);

    if (valid) {
        y[base + lane] = e / denominator;
    }
}
```

启动方式：`rows>0` 时，grid=(rows+7)/8，block=256；每个 block 的 8 个 warps 各处理一行。行越界时整个 warp 一起退出，而且后续不存在 block 协作，因此与“只让一个行内的无效 lane 提前退出”不同。

### 9.2 一个 block 一行、每线程 4 项：四处屏障各保护什么

256 线程处理最多 1024 列，tid 负责 tid、tid+256、tid+512、tid+768。每线程缓存四项，再完成线程内、warp 内、block 内的最大值归约；指数和也做相同层级归约。

| 屏障 | 生产者 → 消费者 | 屏障前必须完成 |
| --- | --- | --- |
| ① | 各 warp → warp 0 | 八个 warp 最大值写入 max_parts |
| ② | warp 0 → 全 block | 整行最大值发布到 max_parts[0] |
| ③ | 各 warp → warp 0 | 八个指数部分和写入 sum_parts |
| ④ | warp 0 → 全 block | 整行分母发布到 sum_parts[0] |

①不能代替②：全行最大值的汇总和发布发生在①之后。③也不能挽救已经用错误最大值算出来的指数。

cols=1000、tid=250 时，四个槽位为 250、506、762、1018，只有前三个有效；1018 在 max 阶段贡献 −∞，exp 后贡献 0，不写回。

<details class="interactive-figure">
<summary>互动图 28 · Block Softmax 的四处屏障</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-softmax-block-code.html?v=height-v2" title="Block Softmax 的四处屏障" class="transformer-interactive" data-widget="cuda-softmax-block-code" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-softmax-block-code.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

```cpp
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>

// 全 warp 参与；仅 lane 0 保证得到完整结果。
__device__ __forceinline__
float warp_max_32(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__
float warp_sum_32(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// 一个 block 处理一行。
// 要求：rows >= 0，1 <= cols <= 1024，blockDim.x == 256。
__global__
void softmax_block_fp32(
    const float* x,
    float* y,
    int rows,
    int cols
) {
    __shared__ float max_parts[8];
    __shared__ float sum_parts[8];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp_id = tid >> 5;
    const unsigned row = blockIdx.x;

    // 整个 block 一起退出。
    if (row >= static_cast<unsigned>(rows)) {
        return;
    }

    const std::size_t base =
        static_cast<std::size_t>(row) * cols;

    // 1. 每线程缓存 4 项，同时求局部最大值。
    float v[4];
    float local_max = -CUDART_INF_F;

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const unsigned col = tid + q * 256;
        v[q] = (col < static_cast<unsigned>(cols))
             ? x[base + col]
             : -CUDART_INF_F;

        local_max = fmaxf(local_max, v[q]);
    }

    // 2. 每个 warp 求最大值，lane 0 写出。
    float warp_max = warp_max_32(local_max);

    if (lane == 0) {
        max_parts[warp_id] = warp_max;
    }

    __syncthreads();  // ① 等 8 个 warp 的最大值写完

    // 3. Warp 0 汇总，线程 0 发布整行最大值。
    if (warp_id == 0) {
        float m = (lane < 8)
                ? max_parts[lane]
                : -CUDART_INF_F;

        m = warp_max_32(m);

        if (lane == 0) {
            max_parts[0] = m;
        }
    }

    __syncthreads();  // ② 等整行最大值发布完成

    const float row_max = max_parts[0];

    // 4. 原地把线程私有缓存改为指数，并求局部和。
    float local_sum = 0.0f;

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        v[q] = expf(v[q] - row_max);
        local_sum += v[q];
    }

    // 5. 每个 warp 求指数和，lane 0 写出。
    float warp_sum = warp_sum_32(local_sum);

    if (lane == 0) {
        sum_parts[warp_id] = warp_sum;
    }

    __syncthreads();  // ③ 等 8 个 warp 的指数和写完

    // 6. Warp 0 汇总，线程 0 发布整行分母。
    if (warp_id == 0) {
        float denominator = (lane < 8)
                          ? sum_parts[lane]
                          : 0.0f;

        denominator = warp_sum_32(denominator);

        if (lane == 0) {
            sum_parts[0] = denominator;
        }
    }

    __syncthreads();  // ④ 等整行分母发布完成

    const float denominator = sum_parts[0];

    // 7. 每个线程归一化自己缓存的项，仅写有效列。
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const unsigned col = tid + q * 256;

        if (col < static_cast<unsigned>(cols)) {
            y[base + col] = v[q] / denominator;
        }
    }
}
```

`rows>0` 时以 `<<<rows,256,0,s>>>` 启动。两个 shared 数组各 8 个 float，共 64B。`v[4]` 是线程私有缓存，但编译器最终放在哪里仍应查看编译与 profiling 结果。

### 9.3 复用 scratch 的读后写风险：本次真正需要修正的理解

若把 max_parts 和 sum_parts 合并为 scratch[8]，warp 0 可能在慢 warp 读取 `scratch[0]` 前，就把它改成指数部分和。原来②只保证“最大值已经写好”，不保证“所有线程已经读走”。

安全顺序可以是：

```cpp
// 整行最大值已经由原屏障②发布。
const float row_max = scratch[0];
__syncthreads();              // 全 block 读走旧值后才允许复用
// 计算 local_sum、warp_sum ……
if (lane == 0) scratch[warp_id] = warp_sum;
```

也可以把新增屏障放在所有线程算完 local_sum 之后，但必须在任何覆盖写入之前：

```cpp
const float row_max = scratch[0];
float local_sum = /* 使用寄存器中的 row_max 计算 */;
__syncthreads();              // 同样保证所有线程都已经完成旧值读取
// 随后才允许归约、覆盖 scratch。
```

第二段只展示屏障位置，省略表达式不是可独立编译的 kernel。把屏障放在 `row_max=scratch[0]` 之前则不能保护后续读取：快线程过屏障后仍可能先读、先算、先覆盖，慢线程再读到新值。

当时我把“算完 local_sum 后、写 scratch 前加屏障”判断为不安全，理由是快线程可能先覆盖。这个回答忽略了题目已经指定“所有覆盖写入都在屏障之后”。快 warp 会在屏障处等待，所有参与线程到达时，它们已读走最大值并完成前面的计算，所以这处位置也是安全的。后来我已自行纠正。这个例子说明，同步分析要画出具体读、写和屏障的先后，不能只记“这附近需要同步”。

这里只为节省 32B shared memory 却多了一次全 block 屏障，不保证值得做；这是正确性训练，不是默认优化建议。

<details class="interactive-figure">
<summary>互动图 29 · 复用 scratch 的读后写风险</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-softmax-scratch-reuse.html?v=height-v2" title="复用 scratch 的读后写风险" class="transformer-interactive" data-widget="cuda-softmax-scratch-reuse" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-softmax-scratch-reuse.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 9.4 宽行：三遍扫描用更多读取换较少缓存

固定 4 项版本不能只把 cols 从 1024 改成 2048：它仍只处理前 1024 列。若输入全零、输出预先清零，错误版本前半行每项 1/1024，后半行 0；它依然非负、整行和为 1。正确结果应为每项 1/2048。因此“非负且和为 1”是必要检查的一部分，却不是充分验证。

三遍扫描每遍使用 `for(col=tid;col<cols;col+=256)`：第一遍读 x 求 m，第二遍读 x 求 ℓ，第三遍再读 x、重算指数并写出。没有固定的 1024 列覆盖上限，但一个 block 处理极长行也不一定是最佳并行方案。

<details class="interactive-figure">
<summary>互动图 30 · 三遍扫描 Softmax 的流量取舍</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-softmax-three-pass.html?v=height-v2" title="三遍扫描 Softmax 的流量取舍" class="transformer-interactive" data-widget="cuda-softmax-three-pass" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-softmax-three-pass.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

```cpp
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>

__device__ __forceinline__
float scan_warp_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__
float scan_warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

// 必须由整个 256 线程 block 调用。
// 返回后，每个线程都得到整行最大值。
__device__ __forceinline__
float scan_block_max(float v, float* scratch) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_id = threadIdx.x >> 5;

    v = scan_warp_max(v);
    if (lane == 0) {
        scratch[warp_id] = v;
    }

    __syncthreads();

    if (warp_id == 0) {
        float m = (lane < 8) ? scratch[lane] : -CUDART_INF_F;
        m = scan_warp_max(m);

        if (lane == 0) {
            scratch[0] = m;
        }
    }

    __syncthreads();
    return scratch[0];
}

// 返回后，每个线程都得到整行指数和。
__device__ __forceinline__
float scan_block_sum(float v, float* scratch) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp_id = threadIdx.x >> 5;

    v = scan_warp_sum(v);
    if (lane == 0) {
        scratch[warp_id] = v;
    }

    __syncthreads();

    if (warp_id == 0) {
        float sum = (lane < 8) ? scratch[lane] : 0.0f;
        sum = scan_warp_sum(sum);

        if (lane == 0) {
            scratch[0] = sum;
        }
    }

    __syncthreads();
    return scratch[0];
}

__global__
void softmax_scan_fp32(
    const float* x,
    float* y,
    int rows,
    int cols
) {
    __shared__ float max_scratch[8];
    __shared__ float sum_scratch[8];

    const unsigned row = blockIdx.x;
    if (row >= static_cast<unsigned>(rows)) {
        return;  // 整个 block 一起退出
    }

    const std::size_t width = static_cast<std::size_t>(cols);
    const std::size_t base =
        static_cast<std::size_t>(row) * width;
    const std::size_t tid = threadIdx.x;

    // 第一遍：求最大值。
    float local_max = -CUDART_INF_F;
    for (std::size_t col = tid; col < width; col += 256) {
        local_max = fmaxf(local_max, x[base + col]);
    }

    const float row_max = scan_block_max(
        local_max, max_scratch
    );

    // 第二遍：求指数和。
    float local_sum = 0.0f;
    for (std::size_t col = tid; col < width; col += 256) {
        local_sum += expf(x[base + col] - row_max);
    }

    const float denominator = scan_block_sum(
        local_sum, sum_scratch
    );

    // 第三遍：重算指数并写回。
    for (std::size_t col = tid; col < width; col += 256) {
        y[base + col] =
            expf(x[base + col] - row_max) / denominator;
    }
}
```

启动为 `rows>0` 时 `<<<rows,256,0,s>>>`，要求 cols≥1。两个 scratch 分开，避免统计量复用风险。cols=8193 时，线程 0 每遍读取 33 项，最后是 8192；线程 1 读取 32 项，最后是 7937。

| 版本 | 跨阶段保存输入 / 指数 | 每有效元素逻辑访问 | 指数计算次数 |
| --- | --- | --- | --- |
| 完整缓存 | 保存本线程负责的所有项 | 1 次读 x、1 次写 y | 1 |
| 三遍扫描 | 不跨阶段缓存整行输入项 | 3 次读 x、1 次写 y | 2 |

8192 个 FP32 的三遍扫描流量是 8192×4×4=128 KiB，完整缓存理想逻辑流量是 64 KiB。前者不一定有两倍 HBM 流量或两倍耗时：缓存可能吸收重读，后者可能增加寄存器压力、降低并发甚至 spilling。三遍扫描也不是零寄存器，统计量、地址和临时变量仍占资源。

## 10. Online softmax 与 FlashAttention：避免物化整张概率矩阵

### 10.1 两组统计量为什么不能直接相加

对一组分数 A 保存 m_A=max(A)、ℓ_A=Σ exp(s−m_A)。合并另一组 B 时：

$$
m=\max(m_A,m_B),\quad
\alpha=e^{m_A-m},\quad \beta=e^{m_B-m},\quad
\ell=\alpha\ell_A+\beta\ell_B.
$$

例如 m_A=2、ℓ_A=3、m_B=5、ℓ_B=2，则 m=5、ℓ=3e⁻³+2。旧、新指数和原本基于不同的最大值，必须统一尺度。若新块最大值更小，旧项缩放因子为 1，但新项仍要缩放。块 A 已经归一化的概率通常不是最终概率，因为加入 B 后分母改变。

<details class="interactive-figure">
<summary>互动图 31 · Online Softmax 的尺度合并</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-online-softmax.html?v=height-v2" title="Online Softmax 的尺度合并" class="transformer-interactive" data-widget="cuda-online-softmax" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-online-softmax.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 10.2 Attention 还可以同时累计加权分子

最终只需要 O=Σ softmax(s_j)V_j，可维护 U=Σ exp(s_j−m)V_j，并按相同缩放因子更新：

$$
U=\alpha U_A+\beta U_B,\qquad O=U/\ell.
$$

m、ℓ 对一个 query 是标量，U 是 d_v 维向量。若上例 U_A=12、U_B=18，则 O=(12e⁻³+18)/(3e⁻³+2)；真实 d_v=64 时，对 U 的 64 个分量分别应用同样的缩放。

<details class="interactive-figure">
<summary>互动图 32 · 加权分子与 Online Attention</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-online-attention.html?v=height-v2" title="加权分子与 Online Attention" class="transformer-interactive" data-widget="cuda-online-attention" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-online-attention.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 10.3 把统一案例中的 S/P 换成分块生命周期

固定 64 个 queries，遍历 64 个 keys / values 的块：算 64×64 分数块、更新每行 m/ℓ、累计 U，再丢弃当前分数 / 概率块。FP32 分数块 16 KiB；m、ℓ 各 64 个数，U 为 64×64=4096 个数。它们的实际存储分布与布局依实现而定，不能把所有累计量都简单认定在 shared memory。

每个 query 有自己的 m/ℓ/U：即使关注同一组 keys，各行分数也可能不同，所以分母通常不同。“因为每行归一化范围不同”解释不完整，未加 mask 的 attention 完全可以具有相同的 key 范围。

把 K/V 块从 64 增为 128，循环次数虽减半，每次分数块也翻倍，计算量不会凭空减半；更大的临时存储还可能限制并发。不同 Q blocks 也可能重读 K/V，不能把“按块复用”说成“整个 kernel 的 K/V 只从 HBM 读一次”。

[FlashAttention 原论文](https://arxiv.org/abs/2205.14135) 的核心是利用分块和重算减少 HBM 读写及完整中间矩阵的存储，保持标准 attention 的数学结果；浮点执行路径仍可能有舍入差异。稠密主计算仍为 O(N²d)，不是线性 attention。序列长度翻倍，主计算量约变为四倍。

<details class="interactive-figure">
<summary>互动图 33 · FlashAttention 分块数据流</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-flashattention-tiles.html?v=height-v2" title="FlashAttention 分块数据流" class="transformer-interactive" data-widget="cuda-flashattention-tiles" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-flashattention-tiles.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 10.4 Causal mask 与全遮挡块

位置对齐、允许关注自身时，第 i 个 query 仅能看 key≤i。query 70 在 key 块 [64,127] 中可见 64～70 共 7 项，在整行可见 0～70 共 71 项。块 [128,191] 完全位于未来，可跳过分数和加权累加，保持已有 m/ℓ/U 不变。

空历史和全遮挡块要专门处理：若 m_old=m_block=−∞，直接套 `exp(m_old−m_new)` 就会出现 NaN。初始化空状态与跳过无效块，是 online 公式的实现前提之一。

<details class="interactive-figure">
<summary>互动图 34 · Causal mask 与未来块跳过</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-causal-mask.html?v=height-v2" title="Causal mask 与未来块跳过" class="transformer-interactive" data-widget="cuda-causal-mask" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-causal-mask.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 10.5 反向用 LSE 重算 P

前向结束后保存每行最终 LSE=m+lnℓ。反向重新计算一小块 S，再用 P_ij=exp(S_ij−LSE_i) 恢复概率块。若 m=5、ℓ=2、S_ij=4，则 LSE=5+ln2，P_ij=e⁻¹/2。

必须使用整行最终 LSE，而非当前 K 块自己的 LSE；缩放、mask 等也必须一致。本节讨论不含 dropout 的基础版本；有 dropout 时还需可重现相同的随机掩码。重算增加 FLOP，但可能通过省掉完整 P 的 HBM 写入和读取而更快，不能只看计算量。

<details class="interactive-figure">
<summary>互动图 35 · 最终 LSE 与概率重算</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-attention-recompute.html?v=height-v2" title="最终 LSE 与概率重算" class="transformer-interactive" data-widget="cuda-attention-recompute" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-attention-recompute.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 11. LayerNorm、RMSNorm 与稳定方差

### 11.1 相同点是归约，不同点是统计量

对一行 D 维特征，LayerNorm 使用均值 μ=(Σx)/D 和总体方差 σ²=Σ(x−μ)²/D；RMSNorm 使用均方 q=(Σx²)/D：

$$
\operatorname{LN}(x_i)=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i,
\qquad
\operatorname{RMSNorm}(x_i)=\gamma_i\frac{x_i}{\sqrt{q+\epsilon}}.
$$

对 [1,3]，γ=1、β=0、忽略 ε：均值 2、方差 1，LN=[−1,1]，RMSNorm=[1/√5,3/√5]。全体加 100，在精确算术下 LN 不变，RMSNorm 一般改变。对常量 [3,3,3,3]，LN 全零，RMSNorm 每项接近 1。有限精度下的平移也可能引入额外舍入，不能把数学性质说成所有实现都逐位不变。

<details class="interactive-figure">
<summary>互动图 36 · LayerNorm 与 RMSNorm 对比</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-layernorm-rmsnorm.html?v=height-v2" title="LayerNorm 与 RMSNorm 对比" class="transformer-interactive" data-widget="cuda-layernorm-rmsnorm" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-layernorm-rmsnorm.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 11.2 Welford 避免“大数相减丢掉小方差”

数学公式 E[x²]−E[x]² 在浮点中可能灾难性抵消。课堂 FP32 例子 [10001,10002,10003,10004] 的真实总体方差是 1.25，但某些求值 / 累加顺序会得到错误值，甚至负数；不能把具体负值当成跨平台保证。把结果 clamp 到 0 或只加 ε，无法恢复已经丢失的方差信息。

Welford 状态是 (n,μ,M₂)，M₂=Σ(x−μ)²。加入 x 时：

$$
n'=n+1,\quad\delta=x-\mu,\quad\mu'=\mu+\delta/n',\quad
M_2'=M_2+\delta(x-\mu').
$$

[1,3] 的状态为 (2,2,2)，加入 5 后为 (3,3,8)，总体方差 8/3。两组状态合并时，δ=μ_B−μ_A：

$$
n=n_A+n_B,\quad
\mu=\mu_A+\delta\frac{n_B}{n},\quad
M_2=M_{2A}+M_{2B}+\delta^2\frac{n_A n_B}{n}.
$$

A=(2,2,2)、B=(3,7,8)，合并得到 n=5、μ=5、M₂=40、总体方差 8。空线程必须使用 n=0 的空状态，合并时专门返回非空一侧，避免 0/0；若用 (1,0,0)，就等于增加了一个真实零元素。

Welford 更稳定，但不是无限精度；并行合并顺序仍可能改变舍入。LayerNorm 的总体方差除 D，不是统计学无偏估计的 D−1。

<details class="interactive-figure">
<summary>互动图 37 · 方差公式的浮点稳定性</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-variance-stability.html?v=height-v2" title="方差公式的浮点稳定性" class="transformer-interactive" data-widget="cuda-variance-stability" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-variance-stability.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 38 · Welford 状态并行合并</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-welford-merge.html?v=height-v2" title="Welford 状态并行合并" class="transformer-interactive" data-widget="cuda-welford-merge" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-welford-merge.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 11.3 混合精度先转换，再平方

FP16 的 300 平方为 90000，超出 FP16 有限范围；先在 FP16 平方溢出，再转 FP32 已经太迟。正确计算路径是先将输入转 FP32，再平方和归约。

若 x、γ、y 均为 FP16，每元素逻辑访问是 2B+2B+2B=6B；若另重读一次 x，则为 8B。D=1000 即便按 1024 个槽位启动、无效项补零，均方也应除真实长度 1000。

<details class="interactive-figure">
<summary>互动图 39 · RMSNorm 的精度与访问量</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-rmsnorm-precision.html?v=height-v2" title="RMSNorm 的精度与访问量" class="transformer-interactive" data-widget="cuda-rmsnorm-precision" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-rmsnorm-precision.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 12. Residual + RMSNorm 融合：省访问，也改变资源与舍入

回到统一案例的 T=O+residual，Y=RMSNorm(T)。先假设 x、r、T、γ、y 都用 FP16 存储，归约用 FP32：

| 方案 | 每元素逻辑字节 | 来源 |
| --- | --- | --- |
| 分开执行 | 12B | 加法读 x/r、写 T 共 6B；归一化读 T/γ、写 y 共 6B |
| 融合，不输出 T | 8B | 读 x/r/γ，写 y |
| 融合，仍输出 T | 10B | 上项再写 T |
| γ 改用 FP32，分开 / 融合不输出 / 融合输出 T | 14 / 10 / 12B | 相比各方案多读 2B γ |

融合保留 T 的线程私有值，再完成 RMS 统计与归一化，可省中间结果的写回和重读，也减少一次 kernel 提交。但更大寄存器需求、spilling、并行度变化都可能抵消收益；若训练反向仍需要 T，不能在流量模型中假设它不输出。

融合还有数值语义：分开路径可能先把 x+r 舍入为 FP16，融合一直留 FP32 会改变后续结果。若要保留这一步语义，可以在寄存器中做 FP16 舍入再转回 FP32，无需写入 global memory。不过只保留这一处舍入也不能自动保证整个实现逐位等价，还需检查后续计算顺序。

<details class="interactive-figure">
<summary>互动图 40 · 残差与 RMSNorm 融合</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-residual-rmsnorm-fusion.html?v=height-v2" title="残差与 RMSNorm 融合" class="transformer-interactive" data-widget="cuda-residual-rmsnorm-fusion" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-residual-rmsnorm-fusion.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 13. Stream、Event、拷贝流水线和默认 Stream

### 13.1 Event 应该恰好覆盖真正的生产者

Stream 是有顺序的提交队列，不是独占硬件引擎。不同 Stream 的独立工作可能并发，但依赖、资源和硬件限制可能阻止重叠。

假设 A 产生 tmp，B 读取 tmp，C 与 B 独立，A/C 位于 s1，B 位于 s2：

```cpp
A<<<grid, block, 0, s1>>>(tmp);
cudaEventRecord(ready, s1);
C<<<grid, block, 0, s1>>>(other);
cudaStreamWaitEvent(s2, ready, 0);
B<<<grid, block, 0, s2>>>(tmp);
```

这是依赖结构示意，kernel 参数与启动尺寸应由实际实现提供。`cudaStreamWaitEvent` 让 s2 后续工作在 GPU 端等待，并不要求 CPU 等 A 执行完成。

A=20μs、C=50μs、B=40μs，B/C 理想重叠时为 20+max(50,40)=70μs。Event 若记录在 C 后，会让 B 额外等 C，变为 110μs；放在 A 前则不能保证 A 已完成。CPU 先提交 A 再提交 B，本身也不能保证不同 Stream 的数据依赖。

<details class="interactive-figure">
<summary>互动图 41 · Stream 与 Event 的依赖位置</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-stream-event-dependency.html?v=height-v2" title="Stream 与 Event 的依赖位置" class="transformer-interactive" data-widget="cuda-stream-event-dependency" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-stream-event-dependency.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 13.2 Async 返回与缓冲区生命周期不是同一件事

异步 H2D 返回后不能立即改写仍被拷贝读取的 host 输入；可以在该 H2D 后记录 Event，CPU 在 `cudaEventSynchronize` 成功后复用输入。D2H 的输出也要等待拷贝完成才读。

若四批任务每批 H2D=10μs、计算=30μs、D2H=10μs，串行为 200μs。假设拷贝方向的硬件引擎、计算能力、独立缓冲区与调度允许理想重叠，流水线为 10+4×30+10=140μs。同一 Stream 按每批 H2D→Kernel→D2H 的顺序提交，会串起各批次，不能自动形成这种跨批重叠。

Pinned host memory 是常见异步传输与重叠的重要条件，但仍要看设备能力与资源。`Async` 名称不保证每种参数组合都立即返回；pageable memory 可能引入 staging 或阻塞。[CUDA API 同步行为](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

<details class="interactive-figure">
<summary>互动图 42 · H2D、计算与 D2H 流水线</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-copy-compute-pipeline.html?v=height-v2" title="H2D、计算与 D2H 流水线" class="transformer-interactive" data-widget="cuda-copy-compute-pipeline" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-copy-compute-pipeline.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 13.3 Legacy 默认流有隐式顺序，NonBlocking 不是 CPU 不阻塞承诺

Legacy 默认 Stream 与普通 blocking Streams 存在隐式同步关系；若按 A(s1)、X(legacy default)、B(s2) 提交，课堂独立任务可能形成 A→X→B。A=30μs、X=10μs、B=20μs，总共 60μs。将 s1/s2 创建为 `cudaStreamNonBlocking`，可避开这类 legacy 隐式依赖，理想独立执行总长为 30μs。

这里的 NonBlocking 描述 Stream 与 legacy 默认流的同步关系，不是承诺所有 host API 调用都不阻塞。Per-thread default stream 的行为也不同于 legacy 模式；读代码和测时间时要明确使用哪种模式。[CUDA Stream 同步语义](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)

<details class="interactive-figure">
<summary>互动图 43 · Legacy 默认流的隐式顺序</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-default-stream.html?v=height-v2" title="Legacy 默认流的隐式顺序" class="transformer-interactive" data-widget="cuda-default-stream" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-default-stream.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 14. CUDA Graph：减少提交间隙，正确处理参数变化

Graph 把操作及依赖组织起来，准备、实例化后多次重放。它主要减少重复调度与提交开销，不自动融合所有 kernel，也不自动减少 HBM 访问。Stream capture 记录 GPU 工作及其依赖，不等于把任意 CPU 控制流变成每次重放都会执行的程序。[CUDA Graph 官方说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

课堂模型：5 个串行短 kernels，每个 2μs，普通 CPU 每 6μs 提交一个，首个在 6μs 到 GPU，最后一个在 30μs 到达并于 32μs 完成；Graph 提交一次 6μs，再连续执行 10μs，总长 16μs。若每 kernel 改为 50μs，普通提交已被 GPU 工作覆盖，两者都为 6+5×50=256μs。不能用“5×提交+5×执行”一律相加，它忽略 CPU/GPU 重叠。

准备成本需要多次重放摊薄，不能每轮都重新 capture / instantiate 却忽略成本。真正的收益取决于工作链、提交间隙、图大小、软件与硬件，课堂数字不是 benchmark。

Graph 的 kernel 参数包括指针值。它保存的是“地址 A”，不是把 A 指向的数据冻结下来：修改 A 内的数据，重放会读取新内容；只把 CPU 变量 x 改指向 B，不会自动更新执行图中的地址。需要按支持的 API 更新节点参数、执行图或重新准备合适的图。

同样，shape 变化还牵涉 grid/block、缓冲区容量、workspace 和图结构。课堂中每线程处理一项、block=4：n 从 4 更新成 8，但 grid 仍为 1，只能覆盖 4 项；还要把 grid 更新到 2。实际项目可按 shape 桶缓存图，但必须满足更新与捕获约束。

<details class="interactive-figure">
<summary>互动图 44 · CUDA Graph 的提交间隙</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-graph-launch.html?v=height-v2" title="CUDA Graph 的提交间隙" class="transformer-interactive" data-widget="cuda-graph-launch" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-graph-launch.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 45 · Graph 参数与地址变化</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-graph-parameters.html?v=height-v2" title="Graph 参数与地址变化" class="transformer-interactive" data-widget="cuda-graph-parameters" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-graph-parameters.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 15. 性能分析：从任务时间线，到 kernel 内部证据

### 15.1 局部加速不等于整体加速

侧边性能会话中，原任务 100ms，某 kernel 80ms，其他阶段 20ms，均串行。只把 kernel 加速 2 倍，总时长为 80/2+20=60ms，整体加速比约 1.67，而非 2。即便这个 kernel 趋近于零，总时长仍趋近 20ms，最高趋近 5 倍。一般串行模型 S=1/((1−p)+p/s)。这是我已答对的部分，不是待纠正错题。

有重叠时先看关键路径：A=80μs、B=50μs 并行，C=20μs 等待两者，原任务 100μs。只把 B 减半仍为 100μs；只把 A 减半则为 max(40,50)+20=70μs，关键路径切换到 B→C。计算与通信重叠也应用同样思路，但前提是不存在依赖或资源竞争，不能把理想 max 模型当实测保证。

侧边会话的另一张图把准备 / 收尾与并行计算 / 通信分开。它提出了尚未作答的追问，本文在自测中补上答案，并明确不是我已经完成的课堂回答。

<details class="interactive-figure">
<summary>互动图 46 · Amdahl 定律与整体收益</summary>
<iframe src="/content/interactive/gpu-cuda/performance-critical-path.html?v=height-v2" title="Amdahl 定律与整体收益" class="transformer-interactive" data-widget="performance-critical-path" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/performance-critical-path.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 47 · 计算与通信的重叠时间线</summary>
<iframe src="/content/interactive/gpu-cuda/compute-communication-overlap.html?v=height-v2" title="计算与通信的重叠时间线" class="transformer-interactive" data-widget="compute-communication-overlap" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/compute-communication-overlap.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>
<details class="interactive-figure">
<summary>互动图 48 · 并行分支与关键路径</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-profiling-critical-path.html?v=height-v2" title="并行分支与关键路径" class="transformer-interactive" data-widget="cuda-profiling-critical-path" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-profiling-critical-path.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 15.2 两种 Nsight 工具回答不同的问题

[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-trace) 用于看 CPU API、kernel、拷贝、等待以及各 Stream 的时间线，定位提交间隙和端到端关键路径；[Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) 用于看单个 kernel 的计算、访存、资源使用等，检验更具体的瓶颈假设。

多个 Stream 的 kernel 时间直接相加，可能重复计算重叠区域，也可能漏掉 GPU 空闲间隙。Profiler 采样、replay 或序列化也可能扰动执行，最终收益应回到正常运行条件的重复测量。

### 15.3 有效带宽与实际 DRAM 带宽不要混淆

固定同一任务逻辑字节量为 0.3GB。A 实际 DRAM 流量 1.2GB、耗时 0.5ms，实际带宽为 2.4TB/s，有效带宽 0.6TB/s。B 实际流量 0.6GB、耗时 0.3ms，实际带宽为 2TB/s，有效带宽 1TB/s。B 更快，即使它实际 DRAM 带宽数字更低。

某条指令的 sector 数下降到 1/8，不能直接推出整个 kernel 的 HBM 流量也变为 1/8。Nsight Compute 中某个 Memory Throughput 百分比也未必只指 HBM，它可能反映不同内存层级或管线。计算与 DRAM 利用率都低，仍可能受并行度不足、长依赖链、访存延迟或调度间隙影响。

<details class="interactive-figure">
<summary>互动图 49 · 实际带宽与有效带宽</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-bandwidth-metrics.html?v=height-v2" title="实际带宽与有效带宽" class="transformer-interactive" data-widget="cuda-bandwidth-metrics" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-bandwidth-metrics.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

### 15.4 正确计时、正确比较

对同一 Stream 的重复 kernel 测量，可先 warm-up，再在循环前后记录 CUDA Events，等待结束 Event，然后用 elapsed time 除以重复次数。跨 Stream 的端到端 GPU 区间需构造覆盖所有分支的依赖；请求端到端时间通常还要用 host 时钟测包含目标阶段的完整流程，不能只测某一个 Stream 的局部区间。

1024³ GEMM，100 次合计 20ms，每次 200μs，按 2MNK 统计约 10.74TFLOP/s。这是计算练习。未同步的 CPU 启动计时主要反映提交开销，不能代表 GPU 执行完成时间。

至少固定并记录：GPU / 软件版本、shape、布局、batch / 并发、数据类型、累加精度、计算模式、warm-up、重复次数、是否包含分配与拷贝。比较绝对误差与相对误差，例如 `|got-ref|≤atol+rtol|ref|`；接近零值时不可只看相对误差。更低精度的高 TFLOP/s 若不满足质量要求，也不算同一目标下的有效优化。

<details class="interactive-figure">
<summary>互动图 50 · GEMM 的计时与计算精度</summary>
<iframe src="/content/interactive/gpu-cuda/cuda-gemm-timing.html?v=height-v2" title="GEMM 的计时与计算精度" class="transformer-interactive" data-widget="cuda-gemm-timing" loading="lazy" referrerpolicy="no-referrer" style="display:block;width:100%;height:640px;border:0"></iframe>
<p><a href="/content/interactive/gpu-cuda/cuda-gemm-timing.html" target="_blank" rel="noopener">独立打开互动图</a></p>
</details>

## 16. 我的追问、答得不够好的地方和已掌握的边界

这一节只记录会话中能确认的提问与回答，不把后来补充的通用风险冒充个人错题。

| 会话中的疑问 / 回答 | 应保留的理解 | 需要修正或补齐 |
| --- | --- | --- |
| “每个线程 [ty,tx] 的单循环只算 C tile 内该位置的贡献？” | 一个线程可以负责一个输出；acc 跨迭代累加 | 说明 q 一轮、K 分块一轮、完整 K 三种层级，避免把单个乘积当最终 C |
| “归约的思路已经有了，你来给出代码吧” | grid-stride 局部累加 → warp shuffle → shared 部分和 → block 汇总，思路正确 | 从口述落到完整 mask、边界、同步、第二轮启动的代码；不应标成不会归约 |
| “继续讲稳定 softmax，先图再代码和考核” | 从数学稳定性转入实际数据流 | 归约结果只有 lane 0 完整，广播和 block 发布不能省略 |
| scratch 屏障放在 local_sum 后、覆盖前，被我判为不安全 | 我注意到了旧值可能被覆盖 | 这个判断已纠正：覆盖前的全 block 屏障会等所有线程读完旧值；不能忽略“写入前”条件 |
| cols=1000、tid=250，只回答写 250/506/762、不写 1018 | 输出边界判断正确 | 补齐无效槽位在 max 中为 −∞、指数和中为 0，不只说“不写回” |
| “每个 query 归一化范围不同，所以不能共用分母” | 每个 query 确实需要自己的分母 | 同范围也可能有不同分数和分母，原因不能只归结为范围不同 |
| 无效 softmax 槽位填 0 的分析 | 我指出了无效 exp(0)=1 混进分母，方向正确 | 最大值变成 0 本身不是理想实数 softmax 错误的充分理由；关键是无效项是否进分母及数值稳定性 |
| 三遍扫描流量 128KiB，“不一定 HBM / 耗时翻倍” | 已区分逻辑访问与实际性能 | 保留缓存、寄存器、并发和 exp 代价这些条件，不把推导包装成实测 |

反复出现的“我的理解对吗”大多得到了正确确认，包括：边界线程、合并访存、bank 广播、occupancy 限制、双缓冲生命周期、Graph 参数、Event 位置、关键路径和数值误差。这些应作为复习时需要能再次解释的掌握点。

## 17. 自测题与答案

建议先遮住答案，写清假设和推导，再对照。以下答案包含尚未在侧边会话作答的追问，不代表全部已经由我口头通过。

### 17.1 执行、存储与性能模型

<details class="self-test">
<summary>Q1：1000 项、block=128，需要多少 blocks / warps？block 7 thread 110 是否存在？</summary>

答：8 blocks，每 block 4 warps，共 32 warps。该线程存在，i=1006，但通过 `i<1000` 守卫跳过数组访问。不需要 1024 个物理 CUDA 核心同时执行。

</details>

<details class="self-test">
<summary>Q2：只有 tid=0 调用 block 屏障，tid=64 随后读它写入的 shared 值，能保证正确吗？</summary>

答：不能。把写、全 block 参与的屏障、读按顺序分开。单个写线程的程序顺序不保证另一个 warp 已能正确读取。

</details>

<details class="self-test">
<summary>Q3：32 lanes 连读 32 个对齐 FP32，与相邻 lane 跨 128B，各覆盖多少 32B sectors？</summary>

答：4 与 32。有效字节均 128B，段内利用率分别 100% 与 12.5%；不据此直接宣称整个 kernel 的 DRAM 流量比是 1:8。

</details>

<details class="self-test">
<summary>Q4：改变转置线程映射，让输出连续就同时解决输入了吗？</summary>

答：没有。朴素映射通常只是交换连续读与连续写的一侧。shared tile 允许线程先合并读，再换线程映射合并写，需要载入完成后的同步。

</details>

<details class="self-test">
<summary>Q5：FP32 访问步长 34 时的 bank 分布如何？所有 lanes 读同址呢？</summary>

答：bank=(34×lane)%32，用 16 banks，每个有两个不同地址，是 2 路冲突。同址读取可广播，不是 32 路冲突。

</details>

<details class="self-test">
<summary>Q6：两个 warps 分别统一走 A / B 算发散吗？</summary>

答：不属于 warp 内发散。发散看同一个 warp 是否需要处理不同路径；具体成本还看编译出的指令。

</details>

<details class="self-test">
<summary>Q7：block=256，每 SM 最多 64 warps，shared 限 4 blocks，寄存器优化后可容 8 blocks，occupancy 多少？</summary>

答：仍为 4×8/64=50%，shared 仍是限制。不能只算寄存器一种资源。

</details>

<details class="self-test">
<summary>Q8：FP32 向量加法在 3TB/s 带宽下的带宽屋顶？提高纯算力是否解决？</summary>

答：I=1/12 FLOP/B，带宽屋顶 0.25TFLOP/s。只提高算力不改变该屋顶；实际还可能低于此值。

</details>

### 17.2 GEMM 与归约

<details class="self-test">
<summary>Q9：GEMM 的每次 q、每轮 base 和全部 K 分别贡献什么？</summary>

答：q 增加一次乘积；base 增加一段 K 的点积；全部 K 结束才得到当前输出的完整 acc。

</details>

<details class="self-test">
<summary>Q10：为什么 tiled GEMM 消费 tile 后还需要第二道屏障？</summary>

答：防止快 warp 提前载入下一 tile，覆盖慢 warp 尚在读取的 shared 数据。第一道屏障保护生产后消费，第二道保护消费后复用。

</details>

<details class="self-test">
<summary>Q11：T=16、Bₖ=16 的局部输入算术强度是多少？Bₖ 翻倍、T 翻倍各怎样？</summary>

答：8192 FLOP /2048B=4 FLOP/B。只把 Bₖ 翻倍不变；把 T 翻倍变为 8。此模型未包含输出访问。

</details>

<details class="self-test">
<summary>Q12：4×4 寄存器 tile 每轮 q 读多少操作数，做多少 FMA，为什么不无限增大？</summary>

答：4A+4B=8 个值、16 次 FMA，相比逐输出 32 次操作数读取降为 1/4。至少 16 个累加值增加寄存器压力，可能降低驻留或 spilling。

</details>

<details class="self-test">
<summary>Q13：4 tiles，load=20μs、compute=30μs，串行与理想双缓冲分别多久？</summary>

答：200μs 与 20+4×30=140μs。复用 buffer 前必须等旧消费者完成；重叠与额外资源成本需实测。

</details>

<details class="self-test">
<summary>Q14：float4 能否从 x+1 直接读？130 项如何处理尾部？</summary>

答：若 x 为 16B 对齐，x+1 偏移 4B 后不满足 float4 对齐。前 128 项可按合法向量访问，余下两个标量处理；不能越界取完整向量。

</details>

<details class="self-test">
<summary>Q15：16³ WMMA 做多少 FLOP？thread 0 单独调用是否可行？</summary>

答：4096 FMA=8192 FLOP。不可，需满足 warp 协作和接口布局等约束；FP32 累加不能恢复 FP16 输入精度。

</details>

<details class="self-test">
<summary>Q16：256 项树归约为什么是 8 轮、255 次加法？N=6 放到 8 槽位怎么补？</summary>

答：树深 8，各层加法数之和 128+64+…+1=255。无效项填 0 并按算法继续同步，不能让未初始化槽位参与。

</details>

<details class="self-test">
<summary>Q17：FULL_MASK shuffle 只让前 20 lanes 进入是否正确？</summary>

答：对这里的完整 warp 模板不正确。mask 指定的尚未退出线程必须满足调用条件，读取的源 lane 也必须有效参与。最简单方案是全部 32 lanes 进入，无效数据填单位元。

</details>

<details class="self-test">
<summary>Q18：512 线程 block 归约需多少 shared 部分和？warp 0 的剩余 lanes 放什么？</summary>

答：16 个 float=64B。lane 0～15 读部分和，其余填 0。跨 warp 写后读用 block 屏障，不能用各自的 __syncwarp 替代。

</details>

<details class="self-test">
<summary>Q19：N=1,000,000，每线程一项、block=256，需几轮？固定第一轮 128 blocks 用 grid-stride 呢？</summary>

答：3907→16→1，共三轮；固定 128 blocks 则 128→1，共两轮。同 Stream 两次 launch 之间无需 CPU 等待。

</details>

<details class="self-test">
<summary>Q20：N=1,000,003 全为 1，固定 grid=128、block=256，第一轮和最终结果是什么？</summary>

答：partial[0]=7936、partial[66]=7747、partial[127]=7680，最终 1000003。每个 block 获得的尾部长度不同，不能简单平均分配。

</details>

### 17.3 Softmax 与 FlashAttention

<details class="self-test">
<summary>Q21：[1000,1002,1004] 的稳定 softmax 如何计算？会完全避免下溢吗？</summary>

答：用 [−4,−2,0] 的指数除以 e⁻⁴+e⁻²+1。不能完全避免下溢，但有限非空输入中至少一个指数为 1，分母非零。

</details>

<details class="self-test">
<summary>Q22：无效 softmax lane 如何填？能否只同步而不广播最大值？</summary>

答：max 阶段 −∞，指数和阶段 0，并按归约模板参与。同步不复制寄存器，需明确广播完整最大值和分母。

</details>

<details class="self-test">
<summary>Q23：tid=250、cols=1000，每线程 4 项时如何处理？</summary>

答：槽位 250、506、762 有效并写回；1018 在 max 贡献 −∞、exp 后贡献 0，不写回。

</details>

<details class="self-test">
<summary>Q24：删除 block softmax 的屏障②，只保留①和③行不行？</summary>

答：不行。①之后才汇总 / 发布整行 max；其他 warps 可能读到部分结果。③发生在指数已经计算之后，不能修复早先错误。

</details>

<details class="self-test">
<summary>Q25：scratch 最大值读走后能否立即被另一个 warp 写部分和？屏障放哪里安全？</summary>

答：不能仅凭一个 warp 读完就覆盖。全 block 读走后、任何覆盖写入前加屏障；可紧跟 row_max 读取，也可位于 local_sum 计算后且所有覆盖写之前。只放在读取之前不够。

</details>

<details class="self-test">
<summary>Q26：固定 4 项版本直接传 cols=2048，为什么非负和行和检查可能漏错？</summary>

答：只归一化 / 写出前 1024 项。全零输入、输出预清零时，前半项为 1/1024、后半为 0，整行和仍是 1。要逐元素对比参考，并检查完整写覆盖。

</details>

<details class="self-test">
<summary>Q27：三遍扫描 cols=8193，thread 0 / 1 每遍读几项，最后一项？8192 列逻辑流量？</summary>

答：33 项到 8192；32 项到 7937。8192×16B=128KiB，不能直接等同两倍 HBM 或耗时。

</details>

<details class="self-test">
<summary>Q28：为什么 online softmax 的两个 ℓ 不能直接加？m_A=2、ℓ_A=3、m_B=5、ℓ_B=2 呢？</summary>

答：指数参考最大值不同。统一到 m=5 后，ℓ=3e⁻³+2。U 也按同样因子缩放；若 U_A=12、U_B=18，O=(12e⁻³+18)/(3e⁻³+2)。

</details>

<details class="self-test">
<summary>Q29：64 queries、d_v=64 的 tile，m、ℓ、U 的形状各是多少？各行可否共用分母？</summary>

答：m/ℓ 各 64，U=[64,64]。不能一般性共用，即使 key 范围相同，各 query 的分数也可能不同。

</details>

<details class="self-test">
<summary>Q30：4096² FP32 分数矩阵多大？64² 分数块多大？FlashAttention 把复杂度变线性了吗？</summary>

答：64MiB 与 16KiB。没有，标准稠密主计算仍为 O(N²d)；它减少中间矩阵存储与 HBM IO。

</details>

<details class="self-test">
<summary>Q31：query 70 对 key 块 [64,127] 能看几项？块 [128,191] 怎样？</summary>

答：当前块 7 项，整行 71 项；未来整块可跳过并保持累计状态。全遮挡块不能无条件执行 −∞−(−∞)。

</details>

<details class="self-test">
<summary>Q32：m=5、ℓ=2、S_ij=4，重算 P 是多少？能用当前块 LSE 吗？</summary>

答：LSE=5+ln2，P=e⁻¹/2。必须使用整行最终 LSE；当前块只在块内归一化。

</details>

### 17.4 归一化、融合与异步执行

<details class="self-test">
<summary>Q33：[1,3] 的 LN 与 RMSNorm 分别是什么？公共平移有何影响？</summary>

答：γ=1、β=0、忽略 ε 时分别 [−1,1] 与 [1/√5,3/√5]。精确算术中 LN 消去公共偏移，RMSNorm 一般不消去。

</details>

<details class="self-test">
<summary>Q34：Welford 的 (2,2,2) 加入 5 得到什么？与 (3,7,8) 合并呢？</summary>

答：加入后为 (3,3,8)，方差 8/3；合并原 (2,2,2) 与 (3,7,8) 得 (5,5,40)，方差 8。无效项须为 n=0 的空状态，不能算一个零样本。

</details>

<details class="self-test">
<summary>Q35：FP16 平方溢出后转 FP32 是否有用？D=1000 补到 1024 后除多少？</summary>

答：转换不能恢复已溢出的值，先转 FP32 再平方。除真实长度 1000。

</details>

<details class="self-test">
<summary>Q36：γ 为 FP32，其余输入 / 中间存储 FP16，residual+RMSNorm 分开 / 融合的字节数？</summary>

答：分开 14B；融合不输出 T 为 10B；仍输出 T 为 12B。均为所述存储约定下的逻辑访问，需检查融合资源与舍入语义。

</details>

<details class="self-test">
<summary>Q37：A20→C50 在 s1，B40 依赖 A 在 s2，Event 放 A 后 / C 后分别多久？</summary>

答：理想 B/C 可重叠时分别 70μs、110μs。Event 放 A 前不能覆盖 A；CPU 的 StreamWaitEvent 调用不必等待 A 完成。

</details>

<details class="self-test">
<summary>Q38：H2D Async 返回就能改 host 输入吗？四批 10/30/10 的理想流水线呢？</summary>

答：不能，以 H2D 完成事件等确认生命周期。串行 200μs，具备各项重叠条件时理想 140μs。

</details>

<details class="self-test">
<summary>Q39：legacy 默认流为什么会让 A30、X10、B20 变 60μs？NonBlocking 就保证 CPU 不阻塞吗？</summary>

答：普通 blocking streams 与 legacy 默认流隐式同步可能建立 A→X→B。NonBlocking 可避开这种依赖，理想独立并发为 30μs，但不承诺所有 API 的 host 调用不阻塞。

</details>

### 17.5 Graph、profiling 与整体验证

<details class="self-test">
<summary>Q40：五个 2μs kernel、每次 CPU 提交 6μs，普通与 Graph 的课堂模型多久？kernel 变 50μs 呢？</summary>

答：32μs 与 16μs；长 kernel 两者均 256μs。普通提交和 GPU 执行可重叠，不应总是线性相加全部提交时间。

</details>

<details class="self-test">
<summary>Q41：Graph 捕获指针 A 后，改 A 内容 / 改 CPU 指针变量为 B，有何区别？n4→8 还要改什么？</summary>

答：前者重放读取 A 的新数据；后者不会自动改变执行图节点参数，需更新地址。每线程一项、block=4 的例子中还需把 grid 从 1 更新为 2，并保证容量等条件。

</details>

<details class="self-test">
<summary>Q42：80ms kernel+20ms 其他串行，kernel 加速两倍 / 无限倍，整体怎样？</summary>

答：总时间 60ms、整体约 1.67 倍；极限总时间 20ms、最高趋近 5 倍。局部收益与整体收益不同。

</details>

<details class="self-test">
<summary>Q43：A80/B50 并行，C20 等待两者，只优化 A 或 B 减半呢？</summary>

答：A 减半总时长 70μs；B 减半仍 100μs。先看关键路径，不是优先优化最容易改的 kernel。

</details>

<details class="self-test">
<summary>Q44：侧边计算 / 通信重叠图中 kernel 加速两倍后怎样？</summary>

答：按图中默认参数，CPU 准备 10ms、计算 80ms、通信 50ms、收尾 5ms，原长 95ms；计算减为 40ms 后，总长 10+max(40,50)+5=65ms。继续只缩短计算已不能降低这个理想模型的总时长，因为通信成为较长分支。

</details>

<details class="self-test">
<summary>Q45：A 实际带宽 2.4TB/s、有效 0.6TB/s；B 实际 2TB/s、有效 1TB/s，谁快？</summary>

答：相同任务下 B 更快。实际搬运越多不代表有效工作越多，需同时看时间和流量定义。

</details>

<details class="self-test">
<summary>Q46：GPU 利用率低、计算与 DRAM 百分比都低，说明无优化空间吗？</summary>

答：不说明。检查小 grid、长依赖、访存延迟、CPU 提交间隙、同步和资源限制，再选择合适工具验证假设。

</details>

<details class="self-test">
<summary>Q47：如何发现“softmax 总和正确、实现仍错误”？</summary>

答：用非整倍数和超容量行宽、逐元素参考比较、预填 NaN 等输出覆盖哨兵（有限输入约定下）、边界与偏移测试。测试全零、重复值、大正负偏移、随机值等；只检查行和不够。

</details>

<details class="self-test">
<summary>Q48：怎么证明统一案例的优化有效，而非只让一个 kernel 好看？</summary>

答：固定 shape / 精度 / 版本等变量，验证子层逐元素结果与业务质量，先测完整流程与关键路径，再比较 kernel 和资源证据。报告绝对耗时、加速比、误差、显存及适用边界，不把教学图和 CPU 模型当 GPU 实测。

</details>

## 18. 复习时怎样把它讲成一次完整优化

面试里可以按同一案例回答：“我先实现 QKᵀ、按行 stable softmax、PV 和 residual+RMSNorm 的正确基线，明确输入、累加精度与 mask。用 CPU 参考和覆盖测试验证，特别检查宽行、尾部与全遮挡策略。随后看 Systems 时间线确定热点；若大矩阵乘法主导，就分析 tile 复用、Tensor Core 和资源；若 S/P 的物化造成 IO 与显存压力，就讨论 FlashAttention；若 residual 与归一化重复搬运中间值，就评估融合；若短 kernel 之间有提交空隙，就评估 Graph。每一步均比较端到端延迟、误差与资源变化。”

这是一条复习和实验方案，不是对个人已经完成真实优化项目的宣称。已有会话提供了概念推导、代码阅读和 CPU 语义检查；这里的 CUDA 样例尚未在当前环境进行 nvcc 编译或 GPU 实测。

练习时至少保留三份完整实现：简单 GEMM、并行 reduction、数值稳定 softmax。先保证结果正确，再解释每次优化具体减少了哪些访问、隐藏了哪些等待、增加了哪些资源需求。真正有说服力的回答，是能把概念、代码、数值语义和测量证据连在一起。

## 19. 延伸资料与互动图说明

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：优化与验证的一般方法、访存及带宽概念。
- [CUDA 线程与存储基础](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)：kernel、线程层级与存储说明。
- [CUDA C++ 扩展与同步约束](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)：block 屏障与 warp intrinsic 的准确参与条件。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：在实际 CUDA 环境进一步检查非法访问、竞争及同步问题。

文中保留本模块 48 张正式课堂互动图，以及相关性能会话的 2 张图。各图保留课堂自身的教学尺寸，未强行改成主案例的 N=4096；图内数字用于验证具体假设、数据流和调度，不是 benchmark。原先的试排预览不重复收录。相关代码题会话只开启了 LRU 练习，尚未产生本模块的新算子结论，因此没有把 LRU 扩写进本篇。

<script src="/js/transformer-interactives.js" defer></script>
