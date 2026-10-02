# AMD GPU 架构笔记：从 Wavefront 到大模型性能分析

这篇长文由 AMD 架构课程的五份中文学习材料整理而成，依次连接编程模型、CU 资源、内存、架构家族与 LLM 工作负载。材料来源为 AMD 官方文档，采集于 2026-09-05；这里不把教学关系图当成特定芯片实测，也不将 AMD 术语直接套用到 GR308。

阅读顺序是先区分软件线程组织与硬件资源，再看数据如何移动，最后讨论推理阶段的瓶颈。每节保留“来源直接说明”和“教学推导”的区别；产品特定参数仍需对照实际 gfx 目标。


## AMD GPU 基础：从 CPU 到 HIP 执行层次

### EXTRACTED｜来源直接说明的概念

#### CPU 与 GPU：优化目标不同

CPU 的核心目标是让单个、可能包含复杂分支的指令序列尽快完成；这通常被称为**低延迟**。GPU 的目标则是让大量相似操作同时推进，以提高单位时间完成的工作量，即**高吞吐量**。延迟是一次操作从开始到结果出现的等待时间；吞吐量是每秒（或每周期）完成的操作数。二者并非互斥，但对硬件资源的取舍不同。

因此，复杂控制流、少量相互依赖的任务通常更适合 CPU；大量数据上执行相同或近似相同操作、分支较少的部分更适合 GPU。HIP 把这类数据并行 C/C++ 程序映射到 GPU 的 SIMT 执行模型：源代码看起来像许多独立线程，硬件把一组线程的同一条指令送入 SIMD 执行资源。

#### 并行工作的 HIP 层次

一个 `__global__` kernel 是为设备编译的函数定义；**kernel launch** 则是主机对该函数的一次具体发射，会创建一个待执行的 grid。函数定义本身不属于线程包含层次。一次 launch 的执行实例从大到小严格包含为：`grid → work-group → wavefront → thread/work-item`。

| HIP / 常用术语 | 含义 | 与硬件的关系 |
| --- | --- | --- |
| grid | 此次 launch 的全部线程块 | 工作组被调度到整个 GPU 的可用 CU |
| block / work-group | 能协作的一组线程 | 在本课采用的 CU/CDNA 教学模型中，视为驻留在一个 CU，能使用 LDS 与块内同步；RDNA 的 WGP 例外见下文 |
| warp / wavefront | 最紧密的锁步执行线程组 | 由多条 lane 以 SIMT/SIMD 方式执行 |
| thread / work-item | wavefront 中的一个独立执行实例 | 具有自己的索引和寄存器状态 |

HIP 文档同时使用 CUDA 风格的 `thread/block/grid` 与 AMD/HSA 风格的 `work-item/work-group`。教学中可把它们分别视为同一层的常用别名，但不要把 API 名称与底层实现细节混为一谈。块和 grid 都可为一、二或三维；常见的一维索引为 `global_id = blockIdx.x * blockDim.x + threadIdx.x`。

**架构范围说明：**“一个 work-group 在一个 CU”是本课为说明 CU/LDS 与块内同步采用的 CU-based（尤其 CDNA）模型，并非 AMD 各代的统一物理放置规则。AMD 的 RDNA 架构资料说明，一个 WGP 由两个 CU 的资源共同服务于单个 work-group；在该 WGP 模式下，同一 work-group 的 wave 可以分布在 WGP 的两个 CU 上。初学时仍可把 work-group 看作共享 LDS、可使用块内 barrier 的协作单元；分析 RDNA 实现和性能时，应把 WGP 而不是单个 CU 作为这一层的硬件边界。补充来源：<https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Architecture_public.pdf>。

#### Wavefront 与分歧

wavefront 中的 lane 通常执行同一条指令。若分支让不同 lane 走不同路径，硬件会以掩码方式分时执行这些路径，未参与当前路径的 lane 不产生有用结果；这称为**控制流分歧**。它不是程序错误，却可能降低有效吞吐量。若相邻线程处理的数据遵循相近的控制路径，通常更容易发挥 SIMT 硬件。

当一个 wavefront 等待内存数据或其他操作数时，CU 可以从其他已驻留且就绪的 wavefront 发射指令。GPU 不需要像传统 CPU 线程切换那样把寄存器状态写回全局内存；这种快速选择就绪 wave 的能力是隐藏延迟的基础。

来源：<https://rocm.docs.amd.com/projects/HIP/en/develop/understand/programming_model.html>

### INFERRED｜面向初学者的教学推导

把 kernel 想成一张待分发的练习题：grid 是整套题，work-group 是被分到同一张桌子的小组，work-item 是每位学生，wavefront 是一排会同时听到同一条指令的学生。LDS 只方便同桌协作；不同桌不能用块内 barrier 直接会合。

选择线程块大小时，先保证问题能被足够多的 work-group 覆盖，再用测量判断资源是否足够。`256` 线程并非普遍最佳值；寄存器、LDS、wave 大小和具体 GPU 都会改变结果。也不要把“GPU 有很多线程”理解为每个线程都在同一时刻发射指令：驻留、就绪和实际发射是不同状态。

#### 学习检查

1. 为什么高吞吐量不等于单次操作低延迟？
2. 哪一层可以使用 LDS 和块内同步？
3. 如果一个 wavefront 的一半 lane 进入 `if`、另一半进入 `else`，为什么可能变慢？


## AMD Compute Unit：调度、执行管线与片上状态

### EXTRACTED｜来源直接说明的概念

Compute Unit（CU）是 AMD GPU 上执行 work-group/wavefront 的基本计算组织。一个 CU 包含用于执行指令、保存线程状态、访问片上共享存储和连接缓存/内存系统的资源。具体数量、互连方式与哪些单元共享资源会因 GCN、RDNA、CDNA 和产品代际而变化；以下术语不应被读成所有 AMD GPU 的固定方块图。

#### 调度器与“换一个就绪 wave”

wavefront 的指令会在 CU 内被选择和发射。若某个 wave 由于内存读取或数据依赖尚未就绪，调度器可选择另一个就绪的驻留 wave。这是 GPU 以并发 work 隐藏等待时间的核心机制。Profiler 的管线指标把瓶颈分为不同管线和等待原因；解释计数器时需要结合目标架构和该版本工具的指标定义。

#### 指令与寄存器资源

| 名称 | 教学上的作用 | 适用边界 |
| --- | --- | --- |
| VALU | Vector ALU：对 wave 中各 lane 的向量/逐线程数据做算术与逻辑运算 | AMD ISA/Profiler 术语；吞吐和指令支持随架构变化 |
| SALU | Scalar ALU：处理 wave 统一的标量控制或标量数据操作 | 不代表每条源代码标量表达式都只走 SALU |
| VMEM | Vector memory：由 lane 地址参与的内存访问路径/指令类别 | “VMEM 忙”不能单独证明带宽已饱和 |
| VGPR | 每 lane 的向量通用寄存器状态 | 用量影响可同时驻留的 wave 数 |
| SGPR | wave 共享的标量通用寄存器状态 | 可用量和分配规则依架构/编译器而异 |
| LDS | 一个 work-group 内可协作使用的片上 Local Data Share | 容量、bank 组织和带宽是架构特定的 |
| vL1D | 向量 L1 数据缓存相关的层级/指标命名 | 命名与层级组织须按目标 GPU 文档核对 |
| MFMA | Matrix Fused Multiply-Add 指令族 | 属于支持该指令的 AMD 数据中心架构/目标；数据类型和形状受 ISA 限制 |

在 CDNA 的 ROCm Compute Profiler 文档中，管线描述讨论 VALU、SALU、VMEM、LDS 与 MFMA 等活动；CDNA Matrix Core 指令的工作由 MFMA 相关管线呈现。它们是性能归因的有用坐标系，不是“每周期固定完成多少工作”的跨代承诺。

来源：<https://rocm.docs.amd.com/projects/rocprofiler-compute/en/docs-6.3.2/conceptual/compute-unit.html>；<https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html>

### INFERRED｜面向初学者的教学推导

可把 CU 看作一间有多种工位的车间：VALU 处理逐 lane 的普通计算，SALU 处理全 wave 一致的控制/标量工作，VMEM 负责把分散的逐 lane 数据请求送往内存路径，LDS 是同一 work-group 的共享白板，VGPR/SGPR 则是各自的随手状态。此比喻用于建立直觉，不能替代 ISA 或计数器定义。

优化的第一问不应是“增加多少线程”，而是“哪个资源使 wave 不就绪或某条管线拥堵”。例如，高 VGPR 使用可能降低驻留 wave 数，因而减少隐藏内存延迟的机会；但强行压低寄存器也可能导致 spill 或更多指令。合理流程是：在目标 GPU 上编译，使用 ROCm profiler 观察占用率、管线活动和内存指标，然后只改一个因素并重新测量。

#### 学习检查

1. VGPR 的增加为什么可能影响 latency hiding？
2. 为什么看到 VMEM 指标较高还不足以断定优化方向？
3. MFMA 是否等同于所有 AMD GPU 都有同样的矩阵执行单元？为什么？


## AMD GPU 内存与性能：层次、延迟隐藏和数据布局

### EXTRACTED｜来源直接说明的概念

#### 从近到远的存储视图

GPU 程序通常会接触如下层次：每个线程/wave 的寄存器（VGPR、SGPR），供 work-group 协作的 LDS，缓存层次（常以 L1/vL1D、L2 等名称讨论），以及容量最大、延迟相对高的设备全局内存（数据中心产品常配 HBM）。寄存器和 LDS 是片上显式或近显式管理的重要资源；全局内存面向整个 grid 可见。

“L1、L2、HBM 的确切容量、延迟、带宽、是否共享以及 LDS 与缓存的物理关系”均为架构和产品特定参数。课程中不把某一芯片的数字当作 AMD GPU 通用常数；应查询目标型号的 AMD 文档，或以 profiler 的实际测量为准。

#### 隐藏延迟与 occupancy

当一个 wave 等待内存操作时，CU 可发射另一个就绪 wave 的指令。**occupancy** 是相对于目标架构上限的驻留 active wave 比率：`active/resident waves ÷ 该 CU 可驻留的最大 waves`。它受 VGPR、SGPR、LDS、work-group 大小和硬件限制共同影响。

这里要分清两种数值：**理论/估算 occupancy** 是根据 launch 配置、寄存器/LDS 申请量和目标硬件上限计算的上界；**实测/achieved occupancy** 是 profiler 在 kernel 执行期间观察到的实际 active wave 程度。二者都不表示“下一条指令一定可发射”，因为 active wave 还可能因数据依赖、内存或 barrier 而不 eligible。更高 occupancy 可能有助于提供足够多的候选 wave 来隐藏等待，但不等于性能必然更高：带宽、依赖链、分歧和执行管线同样可能是瓶颈。

#### 合并访问、分歧、bank conflict 与 spill

- **合并（coalescing）**：同一 wave 的 lane 若访问相邻或可合并的地址，内存系统通常能以更少、更有效的事务服务它们。具体事务粒度和规则依 ISA/架构而变。
- **分歧（divergence）**：同一 wave 的不同控制路径会以掩码执行，可能浪费 lane 执行机会；数据相关的地址模式也可能使内存访问更难高效合并。
- **LDS bank conflict**：LDS 按 bank 组织。同一访问中多个 lane 竞争同一个 bank 的不同地址时，访问可能序列化或产生冲突开销；若读取同一地址则可能是广播而非同一种冲突。Composable Kernel 文档明确要求按目标硬件的 bank 映射理解该问题。
- **register spill**：编译器无法把所有活跃值保留在寄存器时，会把一部分私有状态放入更远的内存位置并插入额外读写。它会增加指令与内存流量，且未必只由“声明了很多变量”决定；应看编译结果和 profiler。

#### 算术强度与 roofline

AMD HIP 的性能文档将算术强度（arithmetic intensity）定义为 kernel 的浮点操作数与搬运字节数之比，即 FLOPs/byte；该数值既可理论计算，也可用实测流量计算。该文档还将 Roofline 定义为用算术强度为横轴、性能为纵轴，并以计算峰值与内存带宽构成两个上限的分析框架：较低算术强度的 kernel 往往更容易受数据移动限制，较高算术强度的 kernel 则可能接近计算侧限制。Roofline 有意忽略部分延迟和控制流影响，因此是定位起点而不是性能保证。峰值、有效带宽和数据计数边界必须针对具体硬件与精度定义。

来源：<https://rocm.docs.amd.com/projects/HIP/en/develop/understand/programming_model.html>；<https://rocmdocs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html>；<https://rocm.docs.amd.com/projects/composable_kernel/en/docs-7.1.1/conceptual/ck_tile/hardware/lds_bank_conflicts.html>；<https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html>

### INFERRED｜面向初学者的教学推导

可按下面顺序审视一个慢 kernel：先确认正确性与测量边界；再看全局内存访问是否连续、是否重复读取；随后评估能否以寄存器或 LDS 重用数据；最后检查寄存器压力、LDS bank 映射、分歧与实际 occupancy。这个顺序是排查启发式，不是对所有 kernel 的固定优化配方。

例如，矩阵分块会尝试让一个 work-group 协作把数据块读入 LDS，并在多个乘加中重用它，从而提升算法层面的算术强度；但 tile 过大也可能占用太多 LDS/VGPR，降低并发度。应在目标 GPU、目标数据类型、真实输入大小和真实库/编译选项下测量。

#### 学习检查

1. 为什么 “更高 occupancy” 不能单独作为优化成功的证据？
2. LDS bank conflict 与两个 lane 读取相同 LDS 地址是一回事吗？
3. 一个 kernel 算术强度高时，还可能被哪些因素限制？


## AMD GPU 家族：GCN、RDNA 与 CDNA 的定位边界

### EXTRACTED｜来源直接说明的概念

AMD 文档把 GPU 架构支持按 gfx 目标与产品代际描述；同一个“AMD GPU”并不是单一微架构。写 kernel、解释 profiler 或选择库实现前，应先确认 `gfx` 目标、产品型号和 ROCm 版本。

| 家族 | 官方材料中的定位 | 本文可安全采用的代际范围 |
| --- | --- | --- |
| GCN | AMD 早期 GPU 架构家族，资料常以 CU 与 Wave64 语境描述 | 仅在该代或明确兼容的目标上引用其组织细节 |
| RDNA | 面向图形/游戏工作负载的架构家族；RDNA 公开资料介绍 Work Group Processor（WGP）和 Wave32 执行能力 | WGP、Wave32 等描述须按 RDNA 代际/产品查证 |
| CDNA | 面向数据中心与计算的架构家族；ROCm profiler 对 CDNA 管线和 MFMA 提供专门说明 | CDNA 细节不可直接套给 RDNA 或 GCN |

#### RDNA：WGP 与 Wave32

AMD 的 RDNA 公开架构资料将 Work Group Processor（WGP）作为组织单位来说明，并强调 Wave32 对图形和计算调度的意义。RDNA 性能指南还讨论 wavefront size、occupancy、缓存、LDS 和 SIMD 使用效率。这里的正确读法是：Wave32 是 RDNA 文档强调的执行模式/能力；具体支持的 wave 模式、吞吐与最优分组仍由所用 RDNA 代、着色器/编译目标和工作负载决定。

#### CDNA / Instinct：XCD、Matrix Core 与 MFMA

MI300 系列架构资料以 chiplet 组织说明 XCD（Accelerator Complex Die）和其计算资源，并以 Infinity Fabric 连接相关组件。ROCm 的 CDNA 管线说明把 MFMA（矩阵 fused multiply-add）作为矩阵计算路径的一部分讨论。Matrix Core/MFMA 的确切数据类型、矩阵形状和指令可用性需要以目标 GPU 的 ISA、ROCm 支持矩阵与库文档为准。

#### Infinity Fabric

Infinity Fabric 是 AMD 用于连接系统/封装内组件的互连技术。在 MI300 的说明中，它参与连接 chiplet、内存和平台组件的架构叙述。它不是一个可由单个 HIP kernel 直接替代或绕过的“内存层级”，也不能由其名称直接推导出固定通信带宽或延迟。

来源：<https://gpuopen.com/learn/rdna-performance-guide/>；<https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Architecture_public.pdf>；<https://instinct.docs.amd.com/develop/gpu-arch/mi300.html>；<https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html>

### INFERRED｜面向初学者的教学推导

可用“产品定位”而非“谁更快”来建立第一印象：RDNA 材料优先讲图形/游戏效率与 WGP；CDNA/Instinct 材料优先讲数据中心计算、矩阵工作负载和多芯粒系统；GCN 则常用于理解较早 AMD GPU 的历史与兼容背景。这是阅读路径，不是性能排名，也不表示任一架构不能运行另一类工作负载。

把代际标记写在笔记里能避免常见错误：

- `Wave32`：RDNA 公开资料的语境；不要据此宣布所有 AMD GPU 的 wave 大小都为 32。
- `Wave64`：在 GCN 与数据中心/CDNA 语境常见；具体支持与编译目标仍须确认。
- `WGP`：RDNA 组织术语；不能替换所有代的 CU 解释。
- `XCD`、`Matrix Core`、`MFMA`：以 MI300/CDNA 或已确认支持的目标为前提。
- `Infinity Fabric`：系统/封装互连语境；需要按拓扑和产品资料分析通信。

#### 学习检查

1. 为什么用 RDNA 的 WGP 图解释 CDNA profiler 指标可能误导？
2. 在看到 MFMA 优化建议前，应核实哪三类信息？
3. Infinity Fabric 为什么不等同于某一级 GPU cache？


## LLM 工作负载到 AMD GPU 的教学映射

### EXTRACTED｜可作为映射边界的硬件事实

HIP 将数据并行 kernel 组织为 grid、work-group、work-item 和 wavefront；CU 通过寄存器、LDS、缓存/内存路径及不同执行管线推进这些工作。ROCm Compute Profiler 的 CDNA 管线说明包含 VALU、VMEM、LDS 与 MFMA 等类别。AMD GPU 架构参考要求按具体 `gfx` 目标确定支持能力。

这些资料给出的是硬件与编程模型的词汇边界；它们**没有**把某个 LLM 算法名与某一条硬件管线一一绑定，也没有给出适用于所有模型、批量、精度和 AMD 产品的定量性能结论。

来源：<https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/pipeline-descriptions.html>；<https://rocm.docs.amd.com/projects/HIP/en/develop/understand/programming_model.html>；<https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html>

### INFERRED｜跨层教学映射（非硬件等价关系）

以下全部是为帮助定位测量问题的**推导式映射**。实际 kernel 划分、数据布局、通信、融合策略和瓶颈由框架、模型形状、精度、库版本、GPU 架构和运行时决定。

| LLM 概念 | 常见计算/数据动作 | 可先观察的 GPU 侧线索 | 必须保留的边界 |
| --- | --- | --- | --- |
| GEMM | 大规模矩阵乘加，常有分块与数据复用 | 支持的目标上可观察矩阵指令/VALU 活动、VGPR/LDS 和内存流量 | GEMM 不等于 MFMA；库可能采用不同 kernel 或回退路径 |
| Attention | QKᵀ、softmax、与 V 的乘法及中间数据处理 | 多阶段 kernel 的读写、归约、分歧、算术强度 | “Attention 是内存受限/计算受限”取决于形状与实现 |
| KV Cache | 保存并读取历史 token 的 K/V 状态 | 全局内存访问、缓存命中、地址连续性、容量压力 | cache 的逻辑含义不等于硬件 L1/L2 cache |
| Prefill | 同时处理一段 prompt，通常有较大并行度 | GEMM/attention 的大批工作、数据重用和矩阵路径 | 不能从阶段名推导固定吞吐或占用率 |
| Decode | 每步生成少量新 token，并读取历史 KV | 小工作量调度、KV 读取、通信与批处理策略 | 不等价于“只有内存访问”；实现仍含计算和同步 |
| Continuous batching | 在服务端动态合并/调度请求 | kernel 发射粒度、形状变化、队列和资源利用 | 它是运行时服务策略，不是 CU 硬件功能 |
| Tensor parallel | 张量维度跨设备划分，并在需要处汇总 | 设备间通信、拓扑、计算/通信重叠 | 不是某一种 Infinity Fabric 拓扑的同义词 |
| RCCL | ROCm 生态中的集合通信库 | all-reduce 等通信调用、拓扑和同步 | RCCL 调用并不自动保证最佳拓扑或重叠 |

#### 一条可操作的分析链

从模型层的“decode 延迟高”开始，不应跳到“某个硬件单元太慢”。可依次确认：请求形状与批处理策略 → 框架实际发出的 kernel/通信 → 每个热点 kernel 的时间和内存/管线指标 → 目标 `gfx` 的可用指令与内存拓扑。这样能区分计算、内存、发射开销和跨设备通信，而非把一个 LLM 名词误当作单一硬件模块。

#### 学习检查

1. 为什么 KV cache 不是硬件 cache？
2. 为什么 tensor parallel 的性能分析必须包含通信拓扑？
3. 哪些条件会改变 prefill 与 decode 的相对瓶颈？

## 工作区补记：四个线程不等于四条物理 Lane

对应学习工作区进一步澄清了一个容易误读的教学例子。启动一个只有四个 work-item 的 work-group，只是在描述四份逻辑工作，并不会把硬件配置成只有四条 lane。若目标使用 Wave32，这四个 work-item 占据一个部分填充的 wave，其他位置被屏蔽；若使用 Wave64，仍需按对应执行模式理解。

例如计算 `output[i] = input[i] * 2 + 1`，输入为十、二十、三十、四十，输出应为二十一、四十一、六十一、八十一。图中只画四行寄存器值，是省略了无效位置，不是在宣布真实 wave 大小为四。VGPR 编号对应一组逐 lane 的值，也不是每条向量指令临时创建一个四元素寄存器数组。

同样，一百个 work-item 在假设的 Wave32 模式下需要四个 wave，最后一个只有四个有效位置。这种尾部利用率损失与分支分歧有所区别：前者来自工作量未填满，后者来自有效工作在执行时走向不同控制路径。理解这点之后，才能正确阅读线程块大小、掩码与占用率之间的关系。

工作区还讨论了部署配方与硬件说明的分工。框架启动参数可以说明如何调用某个后端，但不能仅从 HIP 兼容接口或 `torch.cuda` 这个上层名字，推断底层设备就是某种 NVIDIA 或 AMD 微架构。底层资源与编译目标仍需查对应实现；本文保留 AMD 文档自身的适用边界。
