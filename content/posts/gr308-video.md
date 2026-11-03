# GR308 视频推理笔记：Wan2.2 生成链路、混合精度与 DiT 对照

<link rel="stylesheet" href="/css/qwen-image-notes.css?v=8cdbf2c3b938">
<script src="/content/interactive/wan22-gr308/article.js?v=53224b331185" defer></script>
<script src="/js/qwen-image-notes.js?v=a563ea6d2614" defer></script>

复盘 Wan2.2 的时候，我最先记住的是 1190.20→672.33 秒。继续追问才发现，真正需要掌握的是这些秒数对应哪段执行：文字怎样变成条件，随机 latent 怎样经过 DiT 更新，视频时间维和采样时间步有何区别，以及最后的 VAE 为什么比去噪主干更慢。

这篇笔记以“一艘木制帆船缓慢驶过蓝色山间湖泊，清晨阳光，镜头稳定”为例，串起 Wan2.2-TI2V-5B 的完整文生视频链路，并与 [Qwen-Image-2.1 图像篇](/2026/10/02/gr308-image/)对照。基准使用 1280×704、有效 5 帧、3 个采样步、UniPC、guidance 5、seed 42。121 帧用于说明目标规格与形状扩展，未取得完整 GR308 验收结果。

2026-11-03 的更新整合项目复盘与这轮追问。性能数字来自 2026 年 9 月历史实验；本轮整理没有新增 GPU 性能测试。另保留 H3 的两条历史诊断，分别说明原生 W4A8 能力门与固定输入数值门，避免把不同路径的阶段成果混成一次完整视频生成。

<span class="qwen-legacy-anchor" id="wan-先把目标和短测条件分清"></span>

9月18日报告的服务器快照为8张GR308物理卡、16个逻辑设备，驱动1.1.0.21；有效短测使用两张物理卡的TP2路径。Qwen图像TP2则使用同一张双芯物理卡的两个逻辑设备，两者的“2”不能直接当作相同硬件拓扑。模型与运行环境沿用对应报告，不把其他日期的四卡快照混入本次配置。

流程图支持放大、缩小、展开和键盘滚动；窄屏初始显示主要节点，可横向滚动查看分支。互动观察器另外提供可调整的数值与状态。

## 一条请求经过哪些组件

用户提交 prompt、negative prompt、宽高、帧数、步数、seed、guidance 和输出 fps。文字条件编码通常每请求做一次；DiT 和 Scheduler 每个采样步执行；VAE 在采样结束后解码，内部可以按时间块或空间区域执行。

<figure class="qwen-diagram" id="wan-figure-1">
<figcaption>图 1 · 文字到视频的请求全链路</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 1">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 1">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 1 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 995" role="img" aria-label="文字到视频的请求全链路"><title>文字到视频的请求全链路</title><defs><marker id="wan-arrow-1" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" d="M370 91 C370 108.0 370 108.0 370 125" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M370 201 C370 218.0 370 218.0 370 235" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M370 311 C370 328.0 370 328.0 370 345" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M130 311 C130 328.0 370 328.0 370 345" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M610 311 C610 328.0 370 328.0 370 345" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M370 421 C370 438.0 370 438.0 370 455" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M370 531 C370 548.0 370 548.0 370 565" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M370 641 C370 658.0 370 658.0 370 675" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M370 751 C370 768.0 370 768.0 370 785" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M370 861 C370 878.0 370 878.0 370 895" marker-end="url(#wan-arrow-1)"/><path class="qi-edge" d="M260 713 H18 V383 H260" marker-end="url(#wan-arrow-1)"/><text class="qi-edge-label" x="28" y="510.0">继续</text><g><rect class="qi-node" x="260" y="15" width="220" height="76" rx="8"/><text x="370" y="59" text-anchor="middle"><tspan x="370" dy="0">文字与负向提示词</tspan></text></g><g><rect class="qi-node" x="260" y="125" width="220" height="76" rx="8"/><text x="370" y="169" text-anchor="middle"><tspan x="370" dy="0">UMT5 条件编码器</tspan></text></g><g><rect class="qi-node" x="260" y="235" width="220" height="76" rx="8"/><text x="370" y="279" text-anchor="middle"><tspan x="370" dy="0">固定文本条件</tspan></text></g><g><rect class="qi-node" x="20" y="235" width="220" height="76" rx="8"/><text x="130" y="266" text-anchor="middle"><tspan x="130" dy="0">seed 与目标规格</tspan><tspan x="130" dy="21">随机视频 latent</tspan></text></g><g><rect class="qi-node" x="500" y="235" width="220" height="76" rx="8"/><text x="610" y="279" text-anchor="middle"><tspan x="610" dy="0">当前采样时间 t</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="76" rx="8"/><text x="370" y="389" text-anchor="middle"><tspan x="370" dy="0">DiT：30 层 Transformer</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="76" rx="8"/><text x="370" y="499" text-anchor="middle"><tspan x="370" dy="0">正负预测与 CFG</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="76" rx="8"/><text x="370" y="609" text-anchor="middle"><tspan x="370" dy="0">UniPC 更新 latent</tspan></text></g><g><rect class="qi-node" x="260" y="675" width="220" height="76" rx="8"/><text x="370" y="719" text-anchor="middle"><tspan x="370" dy="0">采样是否完成</tspan></text></g><g><rect class="qi-node" x="260" y="785" width="220" height="76" rx="8"/><text x="370" y="829" text-anchor="middle"><tspan x="370" dy="0">VAE Decoder</tspan></text></g><g><rect class="qi-node" x="260" y="895" width="220" height="76" rx="8"/><text x="370" y="939" text-anchor="middle"><tspan x="370" dy="0">RGB 帧与视频编码</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  P[文字与负向提示词] --&gt; E[UMT5 条件编码器]
  E --&gt; C[固定文本条件]
  R[seed 与宽高帧数] --&gt; Z[随机视频 latent]
  C --&gt; D[DiT 30 层 Transformer]
  Z --&gt; D
  T[当前采样时间 t] --&gt; D
  D --&gt; V[预测 latent 更新量]
  V --&gt; G[CFG 正负条件组合]
  G --&gt; S[UniPC Scheduler 更新 latent]
  S --&gt; Q{采样完成}
  Q --&gt;|继续| D
  Q --&gt;|完成| VA[VAE Decoder]
  VA --&gt; RGB[RGB 视频帧]
  RGB --&gt; O[按 fps 编码为视频文件]</code></pre></details>
</figure>

UMT5 输出隐藏特征供 DiT 读取，不是直接输出视频。目标 latent 从高斯噪声初始化；纯文生视频不需要先用 VAE Encoder 把文字变成 latent。参考图输入属于另一条条件分支，才会使用图像编码路径。

模型名称中的 5B 主要描述约 50 亿参数的 DiT 主干，配套还有文本编码器和 VAE。按 5B 估算，主干 FP16/BF16 权重约 10 GB，FP32 约 20 GB；这不包括其余组件、激活、卷积工作区和通信缓冲。不能据此判断整条视频链路只需 10 GB，或把 VAE 也叫作“5B 解码器”。

## latent 张量与两种时间

视频 latent 的逻辑布局为 `[B,C,F,H,W]`：B 是批次，C 是特征通道，F 是压缩后的内容时间，H/W 是潜空间尺寸。Wan2.2-TI2V-5B 的 latent 通道为 48，VAE 时空步幅为 `(4,16,16)`。

为单个 1280×704、5 帧样本初始化的目标是：

```text
最终 RGB：[1, 3,  5, 704, 1280]
对应 latent：[1,48, 2,  44,   80]

F_latent = (F_video - 1) // 4 + 1
5 帧 → 2 个 latent 时间位置
121 帧 → 31 个 latent 时间位置
```

一个完整 latent 张量表示整段视频的压缩状态；时间切片是其中的一部分，不等于一张独立生成的 RGB 图。DiT 联合处理这些时空位置，VAE 再恢复完整帧。这里应区分“完整 latent 张量”“latent 时间切片”和“进入 Transformer 的 patch token”。

我之前学 Qwen-Image 时也遇到 timestep，它控制的是噪声水平与采样进度。视频的 F 与采样的 t 是两条轴：前者决定表示多少内容时间，后者决定生成过程推进到哪里。静态图片实现也可能保留 F=1；三轴位置编码中的 frame 坐标同样不等于去噪 timestep。

<figure class="qwen-diagram" id="wan-figure-2">
<figcaption>图 2 · 三次采样更新与视频内容时间</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 2">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 2">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 2 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 775" role="img" aria-label="三次采样更新与视频内容时间"><title>三次采样更新与视频内容时间</title><defs><marker id="wan-arrow-2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" d="M370 91 C370 108.0 370 108.0 370 125" marker-end="url(#wan-arrow-2)"/><path class="qi-edge" d="M370 201 C370 218.0 370 218.0 370 235" marker-end="url(#wan-arrow-2)"/><path class="qi-edge" d="M370 311 C370 328.0 370 328.0 370 345" marker-end="url(#wan-arrow-2)"/><path class="qi-edge" d="M370 421 C370 438.0 370 438.0 370 455" marker-end="url(#wan-arrow-2)"/><path class="qi-edge" d="M370 531 C370 548.0 370 548.0 370 565" marker-end="url(#wan-arrow-2)"/><path class="qi-edge" d="M370 641 C370 658.0 370 658.0 370 675" marker-end="url(#wan-arrow-2)"/><g><rect class="qi-node" x="260" y="15" width="220" height="76" rx="8"/><text x="370" y="46" text-anchor="middle"><tspan x="370" dy="0">随机 latent</tspan><tspan x="370" dy="21">1×48×2×44×80</tspan></text></g><g><rect class="qi-node" x="260" y="125" width="220" height="76" rx="8"/><text x="370" y="169" text-anchor="middle"><tspan x="370" dy="0">采样 1：DiT + Scheduler</tspan></text></g><g><rect class="qi-node" x="260" y="235" width="220" height="76" rx="8"/><text x="370" y="279" text-anchor="middle"><tspan x="370" dy="0">采样 2：DiT + Scheduler</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="76" rx="8"/><text x="370" y="389" text-anchor="middle"><tspan x="370" dy="0">采样 3：DiT + Scheduler</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="76" rx="8"/><text x="370" y="486" text-anchor="middle"><tspan x="370" dy="0">最终 latent</tspan><tspan x="370" dy="21">1×48×2×44×80</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="76" rx="8"/><text x="370" y="609" text-anchor="middle"><tspan x="370" dy="0">VAE 解码</tspan></text></g><g><rect class="qi-node" x="260" y="675" width="220" height="76" rx="8"/><text x="370" y="706" text-anchor="middle"><tspan x="370" dy="0">5 帧视频</tspan><tspan x="370" dy="21">1×3×5×704×1280</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  Z0[初始随机 latent：1×48×2×44×80] --&gt; A[采样步 1：DiT 与 Scheduler]
  A --&gt; Z1[更新 latent：形状不变]
  Z1 --&gt; B[采样步 2：DiT 与 Scheduler]
  B --&gt; Z2[更新 latent：形状不变]
  Z2 --&gt; C[采样步 3：DiT 与 Scheduler]
  C --&gt; Z3[最终 latent：1×48×2×44×80]
  Z3 --&gt; V[VAE 解码]
  V --&gt; F[5 帧：1×3×5×704×1280]</code></pre></details>
</figure>

3 步表示完整 latent 更新三轮，5 帧表示最终内容长度，30 层表示一次 Wan DiT 前向经过的 Block 数。第一步并非生成第一帧；采样推进时通常改变张量内容，而不改变 F/H/W。启用 CFG 后，一次采样步逻辑上还有正负两条条件预测；串行、合批或并行是执行安排，不能把步数直接当 DiT 前向次数。

## 从时空网格进入 DiT

Transformer 处理 token 序列，latent 则是多通道时空网格。Wan 的 patch 为 `(1,2,2)`，每个 patch 包含同一时间位置的 2×2 个空间格点及全部 48 个通道，即 192 个输入数。patch embedding 将它投影到 3072 维，再将位置轴展平成序列。

```text
[1,48,2,44,80]
 → patch embedding，kernel/stride=(1,2,2)
[1,3072,2,22,40]
 → flatten 位置轴并转置
[1,1760,3072]

N = F_latent × (H_latent/2) × (W_latent/2)
5 帧：N = 2×22×40 = 1760
121 帧：N = 31×22×40 = 27280
```

展平只是改变位置的组织方式，每个 token 的特征维仍然保留。时空 RoPE 让模型知道 token 原来属于哪个时间和空间位置。这里的 N 是逻辑长度，未计入分片对齐可能新增的 padding。

Qwen-Image-2.1 的 DiT `patch_size=1`。1024² 图片对应 `[1,64,1,64,64]`，展平为 `[1,4096,64]`，再经 Linear(64,4096) 变成 `[1,4096,4096]`。它没有在这个位置继续合并 2×2 latent。Qwen3-VL 条件编码器自己的像素 patch，是另一条路径，不能拿来替代 DiT patch 的定义。

<figure class="qwen-diagram" id="wan-figure-3">
<figcaption>图 3 · Wan 与 Qwen 的 patch 和序列</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 3">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 3">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 3 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 445" role="img" aria-label="Wan 与 Qwen 的 patch 和序列"><title>Wan 与 Qwen 的 patch 和序列</title><defs><marker id="wan-arrow-3" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" d="M130 91 C130 108.0 130 108.0 130 125" marker-end="url(#wan-arrow-3)"/><path class="qi-edge" d="M130 201 C130 218.0 130 218.0 130 235" marker-end="url(#wan-arrow-3)"/><path class="qi-edge" d="M130 311 C130 328.0 130 328.0 130 345" marker-end="url(#wan-arrow-3)"/><path class="qi-edge" d="M610 91 C610 108.0 610 108.0 610 125" marker-end="url(#wan-arrow-3)"/><path class="qi-edge" d="M610 201 C610 218.0 610 218.0 610 235" marker-end="url(#wan-arrow-3)"/><path class="qi-edge" d="M610 311 C610 328.0 610 328.0 610 345" marker-end="url(#wan-arrow-3)"/><g><rect class="qi-node" x="20" y="15" width="220" height="76" rx="8"/><text x="130" y="46" text-anchor="middle"><tspan x="130" dy="0">Wan：5 帧视频 latent</tspan><tspan x="130" dy="21">1×48×2×44×80</tspan></text></g><g><rect class="qi-node" x="20" y="125" width="220" height="76" rx="8"/><text x="130" y="169" text-anchor="middle"><tspan x="130" dy="0">1×2×2 patch 投影</tspan></text></g><g><rect class="qi-node" x="20" y="235" width="220" height="76" rx="8"/><text x="130" y="279" text-anchor="middle"><tspan x="130" dy="0">位置网格：2×22×40</tspan></text></g><g><rect class="qi-node" x="20" y="345" width="220" height="76" rx="8"/><text x="130" y="376" text-anchor="middle"><tspan x="130" dy="0">序列</tspan><tspan x="130" dy="21">1×1760×3072</tspan></text></g><g><rect class="qi-node" x="500" y="15" width="220" height="76" rx="8"/><text x="610" y="46" text-anchor="middle"><tspan x="610" dy="0">Qwen：1024²图像 latent</tspan><tspan x="610" dy="21">1×64×1×64×64</tspan></text></g><g><rect class="qi-node" x="500" y="125" width="220" height="76" rx="8"/><text x="610" y="169" text-anchor="middle"><tspan x="610" dy="0">patch 1：空间展平</tspan></text></g><g><rect class="qi-node" x="500" y="235" width="220" height="76" rx="8"/><text x="610" y="279" text-anchor="middle"><tspan x="610" dy="0">序列：1×4096×64</tspan></text></g><g><rect class="qi-node" x="500" y="345" width="220" height="76" rx="8"/><text x="610" y="376" text-anchor="middle"><tspan x="610" dy="0">64→4096 投影</tspan><tspan x="610" dy="21">1×4096×4096</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  subgraph W[Wan 视频目标]
    W0[latent：1×48×2×44×80] --&gt; W1[1×2×2 patch 投影]
    W1 --&gt; W2[位置网格：2×22×40]
    W2 --&gt; W3[序列：1×1760×3072]
  end
  subgraph Q[Qwen 图像目标]
    Q0[latent：1×64×1×64×64] --&gt; Q1[patch_size 1：空间展平]
    Q1 --&gt; Q2[序列：1×4096×64]
    Q2 --&gt; Q3[64→4096 投影]
    Q3 --&gt; Q4[序列：1×4096×4096]
  end</code></pre></details>
</figure>

## 两种 DiT 怎样读文字

两者都使用固定文本条件，但结构不同。Qwen-Image-2.1 将投影后的文字和图像目标放入统一序列；Wan 保留视频主序列，通过独立交叉注意力读取文字。

| 组件 | Qwen-Image-2.1 | Wan2.2-TI2V-5B |
| --- | --- | --- |
| 编码器 | Qwen3-VL | UMT5 |
| DiT 隐藏宽度 | 4096 | 3072 |
| Block 数 | 32 | 30 |
| 注意力头 | 32×128 | 24×128 |
| 目标输入 | 图像 latent，64 通道 | 视频 latent，48 通道 |
| 条件交互 | 统一序列、带条件 mask | 视频 Self-Attention，再 Cross-Attention |
| MLP | SwiGLU，扩展到 12288 | GELU，扩展到 14336 |
| 本次配置的 CFG | 默认 true_cfg_scale=1，不启用双分支 | guidance=5，启用双条件组合 |

以 L=128 个文字位置演示 Qwen，首步统一序列是 128+4096=4224；L=128 只是形状示例，不是本次历史请求实测长度。目标图像可读取条件与自身全部位置，条件前缀不能读取未来目标图像。参考图编辑另增加参考图 latent，图像篇同尺寸单参考图示例为 8320 个统一位置。

Wan 的 UMT5 特征经投影后为 `[B,L,3072]`，视频主序列为 `[B,N,3072]`。Self-Attention 的 Q/K/V 都来自视频；Cross-Attention 的 Q 来自视频，K/V 来自文字。视频主序列仍有 N 个位置，不因为文字长度 L 就变成 N+L。

<figure class="qwen-diagram" id="wan-figure-4">
<figcaption>图 4 · Wan 单个 DiT Block 的三条计算分支</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 4">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 4">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 4 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 885" role="img" aria-label="Wan 单个 DiT Block 的三条计算分支"><title>Wan 单个 DiT Block 的三条计算分支</title><defs><marker id="wan-arrow-4" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" d="M370 91 C370 108.0 370 108.0 370 125" marker-end="url(#wan-arrow-4)"/><path class="qi-edge" d="M370 201 C370 218.0 370 218.0 370 235" marker-end="url(#wan-arrow-4)"/><path class="qi-edge" d="M370 311 C370 328.0 370 328.0 370 345" marker-end="url(#wan-arrow-4)"/><path class="qi-edge" d="M370 421 C370 438.0 370 438.0 370 455" marker-end="url(#wan-arrow-4)"/><path class="qi-edge" d="M240 493 L260 493" marker-end="url(#wan-arrow-4)"/><path class="qi-edge" d="M370 531 C370 548.0 370 548.0 370 565" marker-end="url(#wan-arrow-4)"/><path class="qi-edge" d="M370 641 C370 658.0 370 658.0 370 675" marker-end="url(#wan-arrow-4)"/><path class="qi-edge" d="M370 751 C370 768.0 370 768.0 370 785" marker-end="url(#wan-arrow-4)"/><g><rect class="qi-node" x="260" y="15" width="220" height="76" rx="8"/><text x="370" y="59" text-anchor="middle"><tspan x="370" dy="0">视频特征 X</tspan></text></g><g><rect class="qi-node" x="260" y="125" width="220" height="76" rx="8"/><text x="370" y="169" text-anchor="middle"><tspan x="370" dy="0">归一化与时间调制</tspan></text></g><g><rect class="qi-node" x="260" y="235" width="220" height="76" rx="8"/><text x="370" y="279" text-anchor="middle"><tspan x="370" dy="0">时空 Self-Attention</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="76" rx="8"/><text x="370" y="389" text-anchor="middle"><tspan x="370" dy="0">门控残差</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="76" rx="8"/><text x="370" y="486" text-anchor="middle"><tspan x="370" dy="0">Cross-Attention</tspan><tspan x="370" dy="21">Q 来自视频</tspan></text></g><g><rect class="qi-node" x="20" y="455" width="220" height="76" rx="8"/><text x="130" y="486" text-anchor="middle"><tspan x="130" dy="0">固定文字 context</tspan><tspan x="130" dy="21">提供 K / V</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="76" rx="8"/><text x="370" y="609" text-anchor="middle"><tspan x="370" dy="0">残差与归一化调制</tspan></text></g><g><rect class="qi-node" x="260" y="675" width="220" height="76" rx="8"/><text x="370" y="706" text-anchor="middle"><tspan x="370" dy="0">GELU MLP</tspan><tspan x="370" dy="21">3072→14336→3072</tspan></text></g><g><rect class="qi-node" x="260" y="785" width="220" height="76" rx="8"/><text x="370" y="829" text-anchor="middle"><tspan x="370" dy="0">门控残差，下一层</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  X[Wan 视频特征 X] --&gt; N1[归一化与采样时间调制]
  N1 --&gt; A[视频时空 Self-Attention]
  A --&gt; R1[门控与残差]
  R1 --&gt; C[Cross-Attention：Q 来自视频]
  TXT[固定文字 context：提供 K/V] --&gt; C
  C --&gt; R2[残差与归一化调制]
  R2 --&gt; M[GELU MLP：3072→14336→3072]
  M --&gt; R3[门控残差，进入下一层]</code></pre></details>
</figure>

Wan 的采样时间经 embedding 产生 shift、scale、gate，叠加每层自己的调制参数；Self-Attention 和 MLP 分支使用相应门控。Qwen 的结构则是联合 Attention 与 SwiGLU 两个主要残差分支，时间网络生成 scale 与 tanh gate，同一步的调制供各层使用。gate 作用于隐藏特征，Scheduler 的步长作用于模型外的 latent，两者不可混淆。

Qwen 条件使用固定 t=0 调制，配合条件前缀无法读取目标的 mask，使每层条件 K/V 可以跨采样步复用。文生图示例首步 Query 长度为 4224；后续 Query 为 4096，K/V 仍包含缓存文字128和当前目标4096。缓存不保存上一轮目标图像来冒充本轮目标，也不让生成变成逐像素自回归。Wan 普通路径每步重算视频的自注意力状态；固定文字的交叉注意力 K/V 在数学上有复用空间，但不能直接宣称已实现 Qwen 式前缀缓存。

## 预测量怎样变成下一步 latent

DiT 的隐藏状态经过全部 Block 后，还要投影回与 latent 匹配的表示。Qwen 取目标位置，经输出头得到 `[1,4096,64]`；Wan 将每个 3072 维 token 投影为 192 个 patch 输出，再 unpatchify 为 `[1,48,2,44,80]`。

DiT 输出的是交给采样器的预测量，而非本步直接完成的 RGB 媒体。Qwen 的 FlowMatchEulerDiscreteScheduler 可用 `z_next = z + (sigma_next - sigma) * v` 理解；实际 sigma 时间表受配置与序列长度等因素影响。Wan 本次使用 UniPC，具体更新依赖该采样器，不能机械套用 Euler 公式。

开启 CFG 时，逻辑上分别得到正向条件预测 p_pos 和负向或空条件预测 p_neg，再计算 `p = p_neg + guidance * (p_pos - p_neg)`。Qwen-Image-2.1 管线默认 true_cfg_scale=1；启用双分支还需要匹配的负向条件。Wan 短测 guidance=5。相同步数不保证两模型具有相同前向次数或工作量。

循环结束后，才将最终 latent 反归一化并交给 VAE。Qwen 的 DiT 输出需要恢复图像 latent 网格；Wan 的 DiT unpatchify 也只是恢复视频 latent，之后还有真正的 VAE 像素解码。这两次“还原”处于不同层次。

## up block 和 residual 在 VAE 的哪里

Wan VAE Decoder 的主要结构是 post_quant_conv、输入卷积、Mid Block、四个 up block、输出归一化、激活与输出卷积；此版本还需要 unpatchify 将输出 patch 表示恢复为 RGB。名为 post_quant_conv 的 1×1×1 卷积沿用 VAE 架构命名，不能据名字推断使用了整数低比特量化。

up block 是尺度阶段，包含多个 residual block 和可选上采样器；residual block 是阶段内部的特征加工单元。最后一级虽然位于 up_blocks 列表里，但不再执行上采样。

<figure class="qwen-diagram" id="wan-figure-5">
<figcaption>图 5 · VAE Decoder 中的 up block 与 residual</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 5">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 5">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 5 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 995" role="img" aria-label="VAE Decoder 中的 up block 与 residual"><title>VAE Decoder 中的 up block 与 residual</title><defs><marker id="wan-arrow-5" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" d="M370 91 C370 108.0 370 108.0 370 125" marker-end="url(#wan-arrow-5)"/><path class="qi-edge" d="M370 201 C370 218.0 370 218.0 370 235" marker-end="url(#wan-arrow-5)"/><path class="qi-edge" d="M370 311 C370 328.0 370 328.0 370 345" marker-end="url(#wan-arrow-5)"/><path class="qi-edge" d="M370 421 C370 438.0 370 438.0 370 455" marker-end="url(#wan-arrow-5)"/><path class="qi-edge" d="M370 531 C370 548.0 370 548.0 370 565" marker-end="url(#wan-arrow-5)"/><path class="qi-edge" d="M370 641 C370 658.0 370 658.0 370 675" marker-end="url(#wan-arrow-5)"/><path class="qi-edge" d="M370 751 C370 768.0 370 768.0 370 785" marker-end="url(#wan-arrow-5)"/><path class="qi-edge" d="M370 861 C370 878.0 370 878.0 370 895" marker-end="url(#wan-arrow-5)"/><g><rect class="qi-node" x="260" y="15" width="220" height="76" rx="8"/><text x="370" y="59" text-anchor="middle"><tspan x="370" dy="0">最终 latent</tspan></text></g><g><rect class="qi-node" x="260" y="125" width="220" height="76" rx="8"/><text x="370" y="156" text-anchor="middle"><tspan x="370" dy="0">post_quant_conv</tspan><tspan x="370" dy="21">与输入卷积</tspan></text></g><g><rect class="qi-node" x="260" y="235" width="220" height="76" rx="8"/><text x="370" y="266" text-anchor="middle"><tspan x="370" dy="0">Mid：Residual</tspan><tspan x="370" dy="21">Attention / Residual</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="76" rx="8"/><text x="370" y="389" text-anchor="middle"><tspan x="370" dy="0">Up 0：Residual 与上采样</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="76" rx="8"/><text x="370" y="499" text-anchor="middle"><tspan x="370" dy="0">Up 1：Residual 与上采样</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="76" rx="8"/><text x="370" y="609" text-anchor="middle"><tspan x="370" dy="0">Up 2：Residual 与上采样</tspan></text></g><g><rect class="qi-node" x="260" y="675" width="220" height="76" rx="8"/><text x="370" y="706" text-anchor="middle"><tspan x="370" dy="0">最终 Up：Residual</tspan><tspan x="370" dy="21">无上采样</tspan></text></g><g><rect class="qi-node" x="260" y="785" width="220" height="76" rx="8"/><text x="370" y="829" text-anchor="middle"><tspan x="370" dy="0">输出 norm / 激活 / 卷积</tspan></text></g><g><rect class="qi-node" x="260" y="895" width="220" height="76" rx="8"/><text x="370" y="939" text-anchor="middle"><tspan x="370" dy="0">unpatchify 与像素后处理</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  Z[最终 latent] --&gt; P[post_quant_conv 与 conv_in]
  P --&gt; MID[Mid：Residual / Attention / Residual]
  MID --&gt; U0[Up 0：多个 Residual 与上采样]
  U0 --&gt; U1[Up 1：多个 Residual 与上采样]
  U1 --&gt; U2[Up 2：多个 Residual 与上采样]
  U2 --&gt; U3[最终 Up：多个 Residual，无上采样]
  U3 --&gt; H[norm_out / 激活 / conv_out]
  H --&gt; O[unpatchify 与像素后处理]</code></pre></details>
</figure>

单个 residual 的主支路为 RMSNorm→SiLU→因果3D卷积→RMSNorm→SiLU→Dropout→因果3D卷积，然后加上 shortcut。通道相同时捷径可直接传递输入，变化时用 1×1×1 卷积对齐，数学上是 `y = F(x) + S(x)`。residual 通常在当前尺度加工特征，上采样器才负责恢复空间或时间分辨率。Wan2.2 的 residual up block 还可以有跨整个阶段的上采样捷径，不能把它与内部 residual 当作同一条连接。

因果3D卷积沿时间方向使用当前及过去特征，并可缓存历史块的输入。因此视频分块解码需维护特征缓存、首块标记和 dtype。VAE 的因果执行不意味着 DiT 也按同样顺序逐帧生成；DiT 在潜空间建立整段视频的时空关系。

## 空间分片与瓦片解码的区别

| 路径 | 切分对象与执行 | 主要关注点 |
| --- | --- | --- |
| 独立瓦片 tiled decode | 将 latent 切成通常带重叠的小块，每块完整解码后拼接 | overlap、blend、局部感受野与质量变化 |
| 空间分片 spatial shard | 将一次解码的中间特征按空间区域分到多个设备 | 卷积边界信息、所需通信与最终汇总 |
| 时间分块 causal decode | 沿 latent 时间维逐块执行，缓存历史特征 | 首帧规则、缓存更新与跨块连续性 |

瓦片可以单卡顺序处理，也可以分给多卡；它和空间分片都在分空间，但不是完全相同的执行算法。当前 Wan 代码包含按高度切分与空间并行卷积路径，还继承 tiled VAE 能力，运行时是否启用空间 shard 由配置与形状判定。开启 parallel decode 不足以证明每个请求都走了某个具体分支，应检查实际 dispatch。

Qwen 图像实验明确分配36个重叠瓦片，每芯18块，之后汇总和blend。其串行瓦片154.70秒与双芯143.50秒的差异约7.2%；整图到瓦片还改变了Attention和边界计算，不能全部解释成多卡加速，也不能将图像瓦片数字移植成Wan的视频成绩。

## 先量哪里慢，再决定调什么

Wan 短测最初的阶段时间为文本编码20.64秒、DiT21.90秒、VAE1144.41秒、端到端1190.20秒。阶段相加1186.95秒，与总时间差3.25秒，说明还有其他计时开销；不能强行把端到端写成三个数字的精确和。VAE占约96.2%。

四次TP2基线端到端均值1192.43秒、标准差约2.00秒；VAE均值1146.46秒、标准差约0.85秒。这个结果支持VAE是当前短负载的稳定主瓶颈，不证明所有帧数和步数下比例相同。

如果只将21.90秒的DiT加速两倍，按其他阶段不变估算，只节省10.95秒，约0.92%；即使完全消除该阶段，端到端上限也只节省约1.84%。这是一项Amdahl式预算推导。121帧与更多采样步会改变主干规模、计算与解码比例，不能把该预算推广到长视频。

调优顺序因此是：先固定输入、测阶段与重复波动；再在主瓶颈内定位模块；为候选同时计时和检查质量；最后回到完整负载验收。DiT TP、VAE空间分片、混合精度分别改变不同阶段和执行维度，不应把多个开关当作等效方案。

<span class="qwen-legacy-anchor" id="全局-fp16-很快-却先过不了质量"></span>

## 从全局 FP16 到模块级混合精度

全局FP16让端到端下降到377.14秒，但首帧出现条带、块状断裂和颜色漂移。PSNR只有14.254dB，亮度SSIM0.7019，RGB SSIM0.5996，MAE38.50/255。三次成功输出哈希一致，损坏可重复，但现有证据不足以把根因唯一归到某个溢出算子。

后续按模块扩大FP16范围，得到下面的轨迹。

| 方案 | 精度范围概要 | 端到端 / s | VAE / s |
| --- | --- | ---: | ---: |
| FP32 baseline | FP32-effective 工程参考 | 1190.20 | 1144.41 |
| Global FP16 | 全局VAE解码FP16，质量不合格 | 377.14 | 333.47 |
| M0 | 仅早期路径 | 1184.34 | 1138.27 |
| M1 | 早期路径加1个up block | 1151.72 | 1106.83 |
| M2 | 扩到2个up block | 1029.46 | 987.54 |
| M3 | 扩到3个up block | 756.04 | 712.45 |
| S1 | M3加最终1个residual | 710.02 | 668.04 |
| S2 | M3加最终2个residual | 672.33 | 630.31 |

<figure class="qwen-diagram" id="wan-figure-6">
<figcaption>图 6 · 性能候选必须同时经过质量检查</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 6">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 6">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 6 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 885" role="img" aria-label="性能候选必须同时经过质量检查"><title>性能候选必须同时经过质量检查</title><defs><marker id="wan-arrow-6" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" d="M370 91 C370 108.0 370 108.0 370 125" marker-end="url(#wan-arrow-6)"/><path class="qi-edge" d="M370 201 C370 218.0 370 218.0 370 235" marker-end="url(#wan-arrow-6)"/><path class="qi-edge" d="M370 311 C370 328.0 370 328.0 370 345" marker-end="url(#wan-arrow-6)"/><path class="qi-edge" d="M370 421 C370 438.0 370 438.0 370 455" marker-end="url(#wan-arrow-6)"/><path class="qi-edge" d="M370 531 C370 548.0 370 548.0 370 565" marker-end="url(#wan-arrow-6)"/><path class="qi-edge" d="M370 641 C370 658.0 370 658.0 370 675" marker-end="url(#wan-arrow-6)"/><path class="qi-edge" d="M370 751 C370 768.0 370 768.0 370 785" marker-end="url(#wan-arrow-6)"/><g><rect class="qi-node" x="260" y="15" width="220" height="76" rx="8"/><text x="370" y="59" text-anchor="middle"><tspan x="370" dy="0">FP32 基线与参考帧</tspan></text></g><g><rect class="qi-node" x="260" y="125" width="220" height="76" rx="8"/><text x="370" y="169" text-anchor="middle"><tspan x="370" dy="0">全局 FP16：377.14 秒</tspan></text></g><g><rect class="qi-node" x="260" y="235" width="220" height="76" rx="8"/><text x="370" y="266" text-anchor="middle"><tspan x="370" dy="0">质量失败</tspan><tspan x="370" dy="21">PSNR 14.25 dB</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="76" rx="8"/><text x="370" y="389" text-anchor="middle"><tspan x="370" dy="0">逐模块精度消融</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="76" rx="8"/><text x="370" y="486" text-anchor="middle"><tspan x="370" dy="0">3 个 up block +</tspan><tspan x="370" dy="21">最后 2 个 residual</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="76" rx="8"/><text x="370" y="596" text-anchor="middle"><tspan x="370" dy="0">S2：672.33 秒</tspan><tspan x="370" dy="21">首帧 PSNR 39.36 dB</tspan></text></g><g><rect class="qi-node" x="260" y="675" width="220" height="76" rx="8"/><text x="370" y="719" text-anchor="middle"><tspan x="370" dy="0">121 帧、多样本验收</tspan></text></g><g><rect class="qi-node" x="260" y="785" width="220" height="76" rx="8"/><text x="370" y="829" text-anchor="middle"><tspan x="370" dy="0">当前仍受运行时问题阻塞</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  B[固定 FP32 基线与参考帧] --&gt; G[全局 FP16]
  G --&gt; Q{画面及数值检查}
  Q --&gt;|失败：PSNR 14.25| A[逐模块精度消融]
  A --&gt; M[逐步扩大到3个up block]
  M --&gt; R[继续选择最终residual]
  R --&gt; S[S2：耗时672.33秒，首帧PSNR39.36]
  S --&gt; L[完整121帧与多样本验收]
  L --&gt; F[当前仍受固件、分配与后端问题阻塞]</code></pre></details>
</figure>

“精度消融”指实验方法：逐组改变模块dtype或autocast边界，看时间和误差如何变化。本次属于FP16/FP32推理混合精度，没有整数scale/zero-point校准或量化训练，也没有修改训练权重的学习过程。它涉及浮点格式与执行边界，不能简单说成只改一个算术开关。

S2的实现选择前三个up block与最后一级的最后两个residual使用FP16路径，其余选定敏感路径保留FP32。代码明确将输出归一化norm_out和输出卷积conv_out设为FP32，并在相应段关闭autocast。简历中更准确的表述是“输出归一化及输出头保留FP32”，不应扩大成每个residual内部所有norm都独立强制FP32。

配置入口是 `SGLANG_WAN_VAE_FP16_UP_BLOCKS=3` 与 `SGLANG_WAN_VAE_FP16_FINAL_RESNETS=-2`。负号表示选择最终residual堆栈的后缀。这个行为依赖定制VAE代码，单独启用原生全局FP16解码无法得到S2。

“低风险”来自固定样本的逐步对照，不是对所有prompt和运动场景的先验保证。后段H/W扩大后激活与卷积工作区可能变大，但真正的耗时、精度敏感性和工作区算法仍由测量决定，不能只靠模型示意图下结论。

S2相对基线端到端减少43.5%，VAE减少44.9%，对应VAE约1.82倍加速；这与Qwen图像DiT的1.82倍恰好接近，却属于不同模型、阶段和工作负载。首帧指标为PSNR39.3606dB、亮度SSIM0.974300、RGB SSIM0.961718、MAE1.66397/255、P95绝对误差6。对照使用同prompt、seed的FP32工程参考，不是绝对审美分数，也不能证明视频时序等价。

<span class="qwen-legacy-anchor" id="为什么还需要长视频验收"></span>
<span class="qwen-legacy-anchor" id="增加低精度覆盖或设备数-也有负结果"></span>

## TP 已经使用，扩卡也有失败分支

短测基线明确使用TP2，并打开VAE parallel decode相关配置；当时并非完全没有考虑并行。DiT TP切分注意力和MLP张量计算，VAE空间分片则处理卷积特征区域，二者需要各自实现和验证。

为继续扩大S2，实验分别尝试仅目标Conv3D autocast、预先转换卷积权重、扩到四卡、扩大最终高分辨率residual的FP16范围。均在相近阶段失败，代表性错误为申请约5.80GiB工作区失败，监控仍报告约28.92GiB空闲。日志未完成唯一根因隔离，allocator、虚拟地址、算法工作区与后端边界应继续区分。四卡尝试还将对应去噪从约57秒增加到86.88秒，增加设备没有在该实验中解决问题。

| 完整121帧、20步路径 | 失败位置与现象 | 当前能确认的范围 |
| --- | --- | --- |
| S2 / TP2 | denoise发生DMA与固件semaphore超时 | 完整任务未完成 |
| S2 / SP2 | step0申请160MiB失败，报告约23.34GiB空闲 | 非普通总容量数字能独立解释 |
| S2 / SP4 | step0申请80MiB失败，报告约23.15GiB空闲 | 扩大分片未解决分配失败 |
| SP8 ring | native fallback与attention backend能力限制 | 未取得有效性能结果 |

下一轮应先隔离最小Conv3D工作区复现与长序列同步问题，确认低workspace算法或带halo的tiled Conv3D是否可用；这些是后续路线，尚未实现验收。完整链路跑通后，再开展固定拓扑的TP/SP/CFG矩阵，并增加多prompt、多seed和全视频时序检查。

现有RTX5090记录中的121帧、3步U8+CFG1到U4+CFG2，57.206→37.222秒，属于另一套部署实验。GR308这里只完成有效5帧短测，不组成硬件速度倍率，也不将两边加速比相乘。



## 互动观察器

下面四个面板将讨论中的流程、维度、缓存与精度表转成交互示意。内容时间与采样步数分别控制；框内的历史秒数保持固定，不随教学参数外推。每个面板均可单独打开，文章正文提供同样的关键结论。

<iframe class="transformer-interactive" data-widget="wan22-chain" title="请求执行与两条时间轴" src="/content/interactive/wan22-gr308/wan22-lab.html?panel=chain&amp;v=20261103" loading="lazy" style="width:100%;height:850px;border:0;display:block" sandbox="allow-scripts allow-same-origin"></iframe>

[单独打开：请求执行与两条时间轴](/content/interactive/wan22-gr308/wan22-lab.html?panel=chain&v=20261103)。

<iframe class="transformer-interactive" data-widget="wan22-shape" title="Wan 与 Qwen 的维度对照" src="/content/interactive/wan22-gr308/wan22-lab.html?panel=shape&amp;v=20261103" loading="lazy" style="width:100%;height:950px;border:0;display:block" sandbox="allow-scripts allow-same-origin"></iframe>

[单独打开：Wan 与 Qwen 的维度对照](/content/interactive/wan22-gr308/wan22-lab.html?panel=shape&v=20261103)。

<iframe class="transformer-interactive" data-widget="wan22-cache" title="统一序列缓存与交叉注意力" src="/content/interactive/wan22-gr308/wan22-lab.html?panel=cache&amp;v=20261103" loading="lazy" style="width:100%;height:900px;border:0;display:block" sandbox="allow-scripts allow-same-origin"></iframe>

[单独打开：统一序列缓存与交叉注意力](/content/interactive/wan22-gr308/wan22-lab.html?panel=cache&v=20261103)。

<iframe class="transformer-interactive" data-widget="wan22-precision" title="混合精度历史结果与瓶颈预算" src="/content/interactive/wan22-gr308/wan22-lab.html?panel=precision&amp;v=20261103" loading="lazy" style="width:100%;height:1000px;border:0;display:block" sandbox="allow-scripts allow-same-origin"></iframe>

[单独打开：混合精度历史结果与瓶颈预算](/content/interactive/wan22-gr308/wan22-lab.html?panel=precision&v=20261103)。

<script src="/js/transformer-interactives.js?v=c999d3775011" defer></script>

## 我问过的问题与需要练习的表达

这轮材料主要是我的连续追问，没有完整的模拟面试作答记录。下面归纳的是需要补全的概念和可练习表达，不把问题杜撰成我曾经说过的错误结论。

| 我的追问 | 容易省略或混淆的部分 | 可以怎样讲完整 |
| --- | --- | --- |
| 视频精度优化主要做了什么？ | 只报43.5%，没有阶段和质量 | 先定位VAE占96.2%，再逐模块消融，S2仅支持5帧3步与首帧参考 |
| up block和residual是什么？ | 把尺度阶段与内部单元并列 | up block包含多个residual及可选上采样；最后一级不再上采样 |
| 精度消融是量化吗？ | 将FP16转换与INT8/INT4混为一谈 | 本次是浮点混合精度；消融指逐组替换的实验方法 |
| 模型多大，为什么没用TP？ | 只看5B权重，忽略基线 | 5B描述DiT；TP2已用，VAE并行单独实现，短测先处理主瓶颈 |
| 空间分片是不是瓦片解码？ | 只要分空间就认为同算法 | 独立瓦片完整解码再blend；空间shard在解码层内处理区域及边界 |
| 一个latent有5帧，还是5个latent？ | latent张量、时间切片与token混称 | 整段5帧由[1,48,2,44,80]表示，时间切片不等于RGB帧 |
| patch是不是把二维latent摊平？ | 漏掉视频时间轴和局部投影 | Wan先做1×2×2 patch投影，再将F/H/W位置展平；Qwen2.1 patch为1 |
| 生图的时间步和视频时间维一样吗？ | 将采样循环当播放时间 | F表示内容长度，t/σ表示采样进度，步数通常不改变latent形状 |
| 两者DiT全过程有什么区别？ | 只说都用了Transformer | 对照条件编码、patch、统一序列/交叉注意力、缓存、输出与Scheduler |

讲解中还修正了三处容易误导的简写：生图latent也可保留F=1，不能只用二维布局概括所有实现；S2明确保护的是输出norm/head，不能扩大成所有内部norm；Qwen2.1默认没有正负双分支CFG，不能把Wan的guidance=5直接搬过去。

一段可练习的口述是：用户文字先经UMT5得到固定条件，按目标宽高和帧数创建随机latent。5帧1280×704对应48通道、2×44×80网格，经1×2×2 patch投影后成为1760个3072维token。每个采样步，30层DiT通过视频自注意力和文本交叉注意力预测更新量，CFG组合后由UniPC更新整段latent。完成采样才进入VAE，逐级通过residual和up block恢复视频像素。我在TP2基线上发现VAE占短测约96%，采用模块级FP16/FP32边界将端到端1190.20降到672.33秒；质量结论只覆盖同一首帧参考，121帧目标仍未通过。

## 与主会话其他模型的联系

文本模型经历提供了同样的方法论：先固定负载与环境，检查实际分派，再筛选局部候选，最后回到请求级测试。9B主要收益来自TP2改TP1×DP2；30B选择关闭EP并使用异步调度；35B与80B用MTP深度和batch组合取得短测候选。这些参数影响的执行位置不同，不能直接借用到卷积解码器。

| 模型方向 | 历史选定结果 | 验证范围 |
| --- | --- | --- |
| Qwen3.5-9B | C16聚合247.16 output tok/s | 64请求正式；拓扑157.67→247.87为另一组短测 |
| Qwen3-30B-A3B | C16聚合225.67，C32 320.90 | C16正式，C32短测 |
| Qwen3.6-35B-A3B | TP2/MTP4，C16 180.64 | 32请求短测候选 |
| Qwen3-Next-80B-A3B | TP4/MTP3，C16 255.93 | 32请求短测，两物理卡，输入长度不同 |
| Qwen-Image-2.1 | 1024²四步DiT 163.33→89.60秒 | 采样阶段，不含最终VAE |
| Wan2.2 | 端到端1190.20→672.33秒 | 有效5帧3步，首帧参考质量 |

27B还提醒我检查基线：早期环境单副本约18—19 tok/s，后续复现厂商环境为53.574；无抢占时强制KV152几乎不改变速度，冷启动MTP1 C16却两次死锁。环境与生命周期是实验变量，旧数字不能当作硬件上限。不同命名的27B材料在未确认权重身份前分别引用。更详细的文本结果放在[GR308文本篇](/2026/10/02/gr308-language/)，图片缓存与TP2在[图像篇](/2026/10/02/gr308-image/)展开。

## H3 两条诊断分支需要分别阅读

9月10日SGLang原生W4A8门禁验证了FANT基础FP16/BF16 matmul、SDPA和Conv3D；真实H3投影虽然返回fant张量且与CPU参考一致，实际Comfy Kitchen选择eager，PrivateUse1没有专用int_mm kernel，因此没有证明原生W4A8加速。该次门禁没有启动正式H3服务。下面保留的50层固定输入诊断属于另一条历史路径，不能拿其耗时替代本次门禁，也不能合并成完整E2E通过。

## H3：先把量化语义还原，再判断是否能加速

MiniMax-H3 的困难从权重执行语义开始。当前分支使用 W4A8 相关模型制品。名称中的 W4 描述权重存储，A8 描述量化路径的一部分，但它们并不保证目标设备已经具备对应的原生 INT8 GEMM。适配必须同时处理解包、缩放、布局、激活量化与投影。

本次平台上没有完成原生 INT8 加速。实际候选用受控的 FP32 分块 fallback 执行相关矩阵运算，并缓存解码后的 W4 权重。它可以帮助验证模型语义和减少重复处理，但需要用更多显存换取时间，不能宣传成“GR308 已经完成 H3 原生 W4A8 高速推理”。

早期真实 projection 的局部测试，例如 block0 的 attention output projection，曾约 0.4858 秒并与 CPU 参考一致。它只验证一个投影，尚未覆盖完整 block 的残差、归一化、模态交互与累积误差。更早 tiny 50 层约 174 秒的路径同样不构成有效视频生成。

后续把诊断推进到固定输入的 50 层 DiT。条件使用两个物理 GR308、TP2，真实 text features 为 [10,5120] BF16，latent 为 [2,8,8]，audio length 9，packed sequence length 64，seed 20260906，timestep 1.0。输入被刻意固定，用于比较数值和权重处理，不是正常长视频工作负载。

## 解码权重缓存：三倍阶段加速的代价

旧路径每轮都要将打包权重转换为可执行形式。候选为每个 rank 缓存 200 个解码权重张量，后续重复使用。计时采用一次 warmup、两次正式测量，每次取较慢 rank 的耗时，再取两次中位数；样本量很小，结论只适用于当前固定输入诊断。

| 固定输入 50 层 DiT | 基线 | 解码权重缓存 |
| --- | ---: | ---: |
| 耗时 | 38.7363 s | 12.4207 s |
| 每 rank 当轮解码次数 | 200 | 0 |
| 每 rank 缓存命中 | 0 | 200 |
| 设备峰值 | 10.5075 GiB | 19.3730 GiB |

阶段速度约为原来的 3.1187 倍，延迟下降 67.94%，额外解码权重约 8.97 GiB/rank。这个收益的来源很明确：避免重复解包与转换。它不能被归因于新的原生低精度矩阵单元，也不能从这张表推算生产吞吐。

缓存还引入生命周期问题。若模型权重、量化参数或设备布局改变，缓存必须失效；若不同执行阶段争抢显存，常驻缓存可能导致其他组件无法加载。当前固定模型、固定输入对照让这些因素保持不变，因此可以隔离收益，但不能替代长期服务中的资源管理验证。

## 性能过门，数值没有一起过门

完整诊断同时比较 video 与 audio 分支的误差。报告值分别为 4.7474% 和 6.7403%，设定门槛为 5%。video 低于门槛，audio 超过，因此最终状态是 `gate_failed`。不能只引用 video 数字，把整个 H3 宣布为通过。

这里的“误差百分比”沿用报告的诊断指标，不是视频质量评分，也不是生成内容正确率。两个模态还可能以不同方式传播误差；有限的最终张量和较低平均误差，不足以保证解码后视频、音频及其同步效果。

该阶段记录明确为 `video_e2e=false`、媒体文件数 0、`native_int8_acceleration=false`。这些状态与 12.4207 秒应当一起出现。后续 SGLang 兼容方案已经有设计文档和准备工作，但没有相应完整 E2E 成绩；尚在推进的路径不能替代已失败的数值门。

从工程角度看，这并不是一份空结果。它定位出重复权重处理是固定输入路径中的重要成本，证明缓存可以降低耗时，并把剩余问题收敛到数值正确性和完整模态链路。下一步应围绕错误传播定位，而不是继续追求更好看的阶段时间。

## 如何理解两组结果中的精度与误差

Wan 的混合精度与 H3 的量化适配用了不同技术，不能都简写成“降低精度”。前者主要改变部分 VAE 运算和中间值的浮点格式，其目标是避免整条路径使用更昂贵的高精度计算；后者先处理已量化权重的存储和解码，再通过当前设备能够执行的路径验证语义。权重小了、计算快了和结果接近参考，是三个需要分别检查的命题。

例如，H3 缓存解码权重以后，常驻设备中的表示反而更大。这并没有违背量化文件体积较小的事实：磁盘制品、传输表示与执行表示可以不同。节省重复转换带来的时间，要以额外常驻内存为代价。若最终还要加载视频解码器与音频组件，这部分代价会进入整个服务的容量预算。

Wan 的结果则提醒人们，误差必须在真正使用的输出上观察。某层输出的均方误差小，不保证经历多个上采样、归一化和残差步骤后仍然小。项目通过实际参考帧比较来约束候选，这是比只检查中间张量更接近用户目标的一步；但只看首帧仍留下时序缺口。

PSNR 是由均方误差及像素范围计算的指标，对照时必须使用相同值域。MAE 使用像素级绝对误差，文中明确写为零到二百五十五的尺度，避免被误解成零到一归一化后的差值。SSIM 关注局部结构，其亮度与 RGB 口径也分别保留。几种指标互补，却都无法独立证明视频的叙事或动作符合提示词。

H3 的百分比来自另外一套张量误差门，它没有图像像素值域，也不是 Wan 的 MAE。把两边数字放在一列然后比较“谁更准”，会丢掉参考张量、归一化方式和任务目标。博客因此保留各自的原指标名与通过状态，不制造跨模型质量排行。

最后，计时中位数与重复次数需要一起阅读。H3 的两次测量可以支持固定输入下的明显改善，却无法估计生产尾延迟；Wan 的四次稳定基线说明短任务的阶段瓶颈可重复，却没有自动覆盖候选的长时稳定性。现有数据足够支持有条件的工程判断，但新增目标仍应产生新增证据。

## 两条路线共同形成的验收方法

这两个项目可以按同一顺序理解，但每一步都留下了各自证据。先验证制品和权重布局，再完成真实组件，随后用固定输入对照，最后才进入生成媒体、质量与完整服务。每一层成功，都只是允许进入下一层。

Wan 的优化以数值敏感性为线索，逐块改变 VAE 精度，并用参考帧限制退化。H3 的优化以执行成本为线索，缓存解码权重，随后在双模态数值门前停止。前者得到有限范围的有效混合精度配置，后者得到有性能价值但尚不合格的诊断候选。

如果以后要继续这两条支线，最有价值的新增材料也很明确：Wan 需要完整帧数和多样本的时序质量，H3 需要音频误差根因、合格的完整 DiT 与实际媒体输出。无需再次用下载完成、HTTP 成功或单层正确去代替这些缺口。

本文依据《风华 GR308 Wan2.2 视频生成性能测试与调优报告》（2026-09-18）、H3 独立工作树诊断报告与《部署 MiniMax H3 性能测试》《wan2.2》工作区后续记录。所有运行时、精度和形状条件按各自记录保留；本轮没有启动远端实验，也没有把未完成方案计为交付成果。

结果文件也应保留对应参数和参考身份。相同文件名如果被后续实验覆盖，就可能把新候选的图片与旧基线的指标错误配对。报告中的阶段时间、精度矩阵与质量对照应作为同一次运行的证据共同读取，而不是从不同目录各取一个最优数字重新拼表。

## 公开归档不等于完整运行时已经恢复

2026-10-18 核验的 [GR308 Lab 固定修订](https://github.com/KunyuanComputing/gr308-lab/tree/51a0d34beab550c484492c36fc09a143df9624b4)公开了 Wan 工具、H3 平台工程和报告。[恢复清单](https://github.com/KunyuanComputing/gr308-lab/blob/51a0d34beab550c484492c36fc09a143df9624b4/archive/recovered/README.md)说明：H3 有五个历史文件的原始哈希与旧索引一致，但仍缺 31 个顶层 Python 文件；完整 DiT 探针仅有前 300 行，在 `finally` 中途截断，不能执行。公开整理版另有脱敏后的哈希，不能拿它与原始哈希混比。

因此，归档改善了代码和证据的可访问性，没有补成一次新 GPU 验收，也没有消除音频误差门、实际媒体输出及长视频质量缺口。历史证据包是否仍存在尚未确认。通信方面，本轮相关会话也明确没有同芯/跨卡大消息带宽实测；P2P 可访问、单元素 collective 正常、显存带宽或 PCIe 理论上限，都不能写成已测的卡间有效带宽。


## 配置与复现资料

- [Wan TI2V 5B config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py), [DiT](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py), [Sampling](https://github.com/Wan-Video/Wan2.2/blob/main/wan/textimage2video.py).
- [Wan DiT config](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/main/transformer/config.json), [VAE config](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/blob/main/vae/config.json).
- [Qwen-Image-2.1 config](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/transformer/config.json), [Scheduler](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json).
- [GR308 Lab pinned revision](https://github.com/KunyuanComputing/gr308-lab/tree/51a0d34beab550c484492c36fc09a143df9624b4).
