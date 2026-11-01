# Qwen Image 2.1 移植 GR308：从 latent 与扩散到 TP2、分块 VAE

复盘图像推理优化时，我最容易记住的是一个数字：1024×1024、4 步短测，DiT 从 163.33 秒降到 89.60 秒，约 1.82 倍加速。真正需要讲清楚的却是这串数字背后的链路：像素怎样进入潜空间，DiT 每一步在计算什么，两颗芯片切走了哪些计算，以及 VAE 分块为什么会改变实际工作量。

这篇笔记把 GR308 项目复盘与后续追问连成一个例子：输入“画一个印有‘风华三号’的红色陶瓷茶壶”，生成一张 1024×1024 图片。默认 batch 为 1，保留文字条件长度用 T=128 演示；这个长度是为了说明形状而设，不是历史请求的实测 token 数。图片编辑分支另外假设一张预处理后为 1024×1024 的参考图。

文中的 GPU 性能来自 2026 年 9 月历史实验。本次 2026-11-01 更新整理模型原理、代码和讨论中的认识修正，没有重跑 GR308，也没有新增端到端性能成绩。实测、数学推导与待验证判断在相应位置分别说明。

## 先看整个请求怎样工作

完整管线包含 Qwen3-VL 条件编码器、DiT 图像生成主干、VAE 和 Scheduler。编码器理解指令；DiT 在潜空间里反复预测更新方向；Scheduler 按时间表推进 latent；VAE 最后将 latent 解码成像素。

<figure class="qwen-diagram" id="qwen-figure-1">
<figcaption>图 1 · 请求全链路：条件、DiT 迭代与 VAE</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 1">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 1">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 1 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 750" role="img" aria-label="请求全链路：条件、DiT 迭代与 VAE"><title>请求全链路：条件、DiT 迭代与 VAE</title><defs><marker id="qwen-diagram-arrow-0" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M130 87 C130 106.0 130 106.0 130 125"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M370 87 C370 106.0 130 106.0 130 125"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M370 87 C370 106.0 370 106.0 370 125"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M130 197 C130 216.0 130 216.0 130 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M370 197 C370 216.0 370 216.0 370 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M130 307 C130 326.0 370 326.0 370 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M370 307 C370 326.0 370 326.0 370 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M610 307 C610 326.0 370 326.0 370 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M500 381 L480 381"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M370 417 C370 436.0 370 436.0 370 455"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M480 491 L500 491"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M610 527 C610 546.0 610 546.0 610 565"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M500 601 L480 601"/><text class="qi-edge-label" x="490.0" y="590">完成</text><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M370 637 C370 656.0 370 656.0 370 675"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-0)" d="M720 601 H734 V381 H480"/><text class="qi-edge-label" x="686" y="455.0">继续</text><g><rect class="qi-node" x="20" y="15" width="220" height="72" rx="8"/><text x="130" y="57" text-anchor="middle"><tspan x="130" dy="0">文字提示词</tspan></text></g><g><rect class="qi-node" x="260" y="15" width="220" height="72" rx="8"/><text x="370" y="57" text-anchor="middle"><tspan x="370" dy="0">参考图：编辑时输入</tspan></text></g><g><rect class="qi-node" x="20" y="125" width="220" height="72" rx="8"/><text x="130" y="167" text-anchor="middle"><tspan x="130" dy="0">Qwen3-VL 条件编码器</tspan></text></g><g><rect class="qi-node" x="260" y="125" width="220" height="72" rx="8"/><text x="370" y="167" text-anchor="middle"><tspan x="370" dy="0">VAE Encoder</tspan></text></g><g><rect class="qi-node" x="20" y="235" width="220" height="72" rx="8"/><text x="130" y="277" text-anchor="middle"><tspan x="130" dy="0">条件特征</tspan></text></g><g><rect class="qi-node" x="260" y="235" width="220" height="72" rx="8"/><text x="370" y="277" text-anchor="middle"><tspan x="370" dy="0">参考图 latent</tspan></text></g><g><rect class="qi-node" x="500" y="235" width="220" height="72" rx="8"/><text x="610" y="277" text-anchor="middle"><tspan x="610" dy="0">随机目标 latent</tspan></text></g><g><rect class="qi-node" x="500" y="345" width="220" height="72" rx="8"/><text x="610" y="387" text-anchor="middle"><tspan x="610" dy="0">当前噪声时间步 σ</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="72" rx="8"/><text x="370" y="387" text-anchor="middle"><tspan x="370" dy="0">DiT：32 层 Transformer</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="72" rx="8"/><text x="370" y="497" text-anchor="middle"><tspan x="370" dy="0">预测方向 v</tspan></text></g><g><rect class="qi-node" x="500" y="455" width="220" height="72" rx="8"/><text x="610" y="497" text-anchor="middle"><tspan x="610" dy="0">Scheduler 更新 latent</tspan></text></g><g><rect class="qi-node" x="500" y="565" width="220" height="72" rx="8"/><text x="610" y="607" text-anchor="middle"><tspan x="610" dy="0">采样完成？</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="72" rx="8"/><text x="370" y="607" text-anchor="middle"><tspan x="370" dy="0">VAE Decoder</tspan></text></g><g><rect class="qi-node" x="260" y="675" width="220" height="72" rx="8"/><text x="370" y="717" text-anchor="middle"><tspan x="370" dy="0">RGBA 图像</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  P[文字提示词] --&gt; E[Qwen3-VL 条件编码器]
  R[参考图：编辑时输入] --&gt; E
  R --&gt; VE[VAE Encoder]
  E --&gt; C[条件特征]
  VE --&gt; RL[参考图 latent]
  N[随机目标 latent] --&gt; D[DiT：32 层 Transformer]
  C --&gt; D
  RL --&gt; D
  T[当前噪声时间步 sigma] --&gt; D
  D --&gt; V[预测方向 v]
  V --&gt; S[Scheduler 更新目标 latent]
  S --&gt; Q{采样完成？}
  Q --&gt;|继续| D
  Q --&gt;|完成| VD[VAE Decoder]
  VD --&gt; O[RGBA 图像]
</code></pre></details>
</figure>

参考图有两条编码路径。Qwen3-VL 提供融合了图像内容的语义上下文；VAE 提供参考图的空间 latent。纯文生图跳过参考图路径。编码器通常每请求计算一次；DiT 每个采样步计算一次；最终 VAE 解码通常只执行一次。

### 参数量应按组件说清楚

| 组件 | 参数量 | 统计范围 |
| --- | ---: | --- |
| DiT | 7.115B | 图像生成 Transformer |
| Qwen3-VL 编码器 | 8.767B | 包含视觉塔及语言模型相关权重 |
| VAE | 0.338B | Encoder 与 Decoder 合计 |
| 完整管线 | 16.220B | 上述权重合计，约 162.2 亿参数 |

官方常用的“7B”指视觉生成主干。根据官方 checkpoint 索引、DiT 的 BF16 参数字节和 VAE 权重头，合计约 16.22B。这个总量不表示每个生成步骤都会执行全部组件。

## 用一张图片理解 latent 与各个维度

VAE 的逻辑布局是 `[B,C,F,H,W]`：B 是批次，C 是特征通道，F 是帧数，H/W 是潜空间的高和宽。静态图的 F=1。

1024×1024 图片经过 16 倍空间压缩，潜空间为 64×64；每个格点有 64 个数：

```text
空间 latent：[1,64,1,64,64]
               B C  F H  W

展平空间轴、将通道移到最后：
序列 latent：[1,4096,64]
               B   N  C

N = H×W = 64×64 = 4096
```

“二维”描述空间网格，完整张量还有批次、通道和帧轴。展平不产生新的内容，位置 `(h,w)` 对应序列位置 `h×64+w`，原来的 64 维特征仍然保留。

DiT 的 `patch_size=1`，每个 latent 格点直接成为一个图像 token。这里没有再把 2×2 latent 合并。前面 Qwen3-VL 使用的 16×16 像素 patch 属于另一条视觉编码路径。

输入线性投影进一步把每个 token 从 64 维变成 4096 维：

```text
[1,4096,64] → Linear(64,4096) → [1,4096,4096]
```

最后两个 4096 分别表示空间 token 数与隐藏特征宽度。RoPE 让展平后的 token 仍携带空间位置信息。

### 维度与并行观察器

切换分辨率、任务和执行阶段，观察目标 token、条件前缀及每颗芯片的计算范围。文字位置固定用 128 演示；参考图在本演示中与输出同尺寸。实际预处理、模板长度和 padding 可能改变序列长度。

<div class="qwen-image-lab" data-qwen-image-lab>
<div class="qwen-lab-controls">
<label>分辨率 <select data-qwen-resolution><option value="512">512×512</option><option value="1024" selected>1024×1024</option><option value="2048">2048×2048</option></select></label>
<label>任务 <select data-qwen-task><option value="text">文生图</option><option value="edit">单参考图编辑</option></select></label>
<label>阶段 <select data-qwen-phase><option value="prefill">首步</option><option value="cached" selected>缓存后续步</option></select></label>
<label>并行 <select data-qwen-parallel><option value="1">单芯</option><option value="2" selected>TP2</option></select></label>
</div>
<div data-qwen-shape-diagram role="img" aria-label="latent、DiT 注意力与 MLP 维度变化"></div>
<p data-qwen-shape-detail aria-live="polite">1024×1024 文生图、后续步、TP2：目标 latent 为 [1,4096,64]；每芯 16 个注意力头，MLP 中间宽度 6144。</p>
</div>

## 条件编码器内部怎样产生 4096 维特征

文字经 tokenizer 得到 `[1,T₀]` 的 ID，Embedding 后为 `[1,T₀,4096]`。T₀ 包含模板前缀，与裁剪后的 T 不一定相同。

参考图进入视觉塔时使用 RGB；RGBA 输入会为这条分支生成白底 RGB 副本，而 VAE 分支保留 alpha。假设预处理尺寸保持 1024×1024：

```text
16×16 像素 patch，空间网格 64×64
→ 4096 个 patch
→ 每个输入 patch 含 3×2×16×16=1536 个数
→ Patch Embedding
→ [4096,1152]
```

其中时间 patch 大小为 2，静态图预处理补齐时间轴，不代表输入了两张不同图片。

视觉塔有 27 层。每层的 Attention 使用 16 个头、每头 72 维，MLP 为 1152→4304→1152。随后 Merger 将空间相邻 2×2 patch 的特征拼接：

```text
[4096,1152] → [1024,4608] → MLP → [1024,4096]
```

视觉向量填入语言序列的图像槽位，进入 36 层语言 Transformer。该部分隐藏宽度为 4096，Q 有 32 个头，K/V 各 8 个头，每头 128 维；MLP 中间宽度为 12288。视觉中间层还通过 DeepStack 注入前几个语言层，主形状不变。

管线读取最后层隐藏状态，去除固定模板前缀与 padding。它取隐藏向量作为 DiT 条件，词表 logits 不进入图像生成主干。

| 场景 | 演示条件形状 |
| --- | --- |
| 文生图 | `[1,128,4096]` |
| 单参考图编辑 | `[1,128+1024,4096]` |

## 参考图的 VAE 路径与序列组装

参考图的 VAE Encoder 从 `[1,4,1,1024,1024]` 开始，主要边界为：

| 位置 | 通道与空间形状，省略 B=1、F=1 |
| --- | --- |
| 输入 | 4×1024×1024 |
| 输入卷积 | 96×1024×1024 |
| Down Block 0 | 96×512×512 |
| Down Block 1 | 192×256×256 |
| Down Block 2 | 384×128×128 |
| Down Block 3 | 768×64×64 |
| Down Block 4 与 Mid Block | 768×64×64 |
| 输出卷积与 quant_conv | 128×64×64 |

128 个输出通道表示 64 通道的均值和 64 通道的 log variance。图片编辑代码取分布的 mode，再用 checkpoint 固定的逐通道 mean/std 归一化，得到 `[1,64,1,64,64]`，展平为 `[1,4096,64]`。分布参数与固定归一化统计量是不同的量。

DiT 入口还有一处重要组装：VLM 的每个图像槽位对应 2×2 个 latent 位置。代码将图像槽位展开四倍，再将 VAE latent 的输入投影填进去。保留的文字隐藏状态已融合图像上下文，图像槽位本身则被 latent 特征替换。

```text
文生图：128 个文字位置 + 4096 个目标位置 = 4224

编辑：128 个文字位置 + 4096 个参考图位置
                         + 4096 个目标位置 = 8320
```

这是统一序列的布局，不能只把 VLM 的 1024 个图像槽位原样当作 DiT 的参考图 token 数。

## DiT 一个 Block 的完整计算

DiT 是 32 层 Single-Stream Transformer。隐藏宽度 D=4096，注意力为 32×128，SwiGLU 中间宽度为 12288。设本层输入为 `[1,S,4096]`：

<figure class="qwen-diagram" id="qwen-figure-2">
<figcaption>图 2 · 一个 DiT Block：注意力、门控与 SwiGLU</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 2">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 2">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 2 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 1210" role="img" aria-label="一个 DiT Block：注意力、门控与 SwiGLU"><title>一个 DiT Block：注意力、门控与 SwiGLU</title><defs><marker id="qwen-diagram-arrow-1" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 87 C370 101.0 370 101.0 370 115"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 187 C370 203.5 130 203.5 130 220"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 187 C370 203.5 370 203.5 370 220"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 187 C370 203.5 610 203.5 610 220"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M130 292 C130 308.5 250 308.5 250 325"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 292 C370 308.5 250 308.5 250 325"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M250 397 C250 416.0 370 416.0 370 435"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M610 292 C610 363.5 370 363.5 370 435"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 507 C370 521.0 370 521.0 370 535"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 607 C370 621.0 370 621.0 370 635"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 707 C370 721.0 370 721.0 370 735"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M260 51 H18 V771 H260"/><text class="qi-edge-label" x="28" y="375.0">残差</text><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 807 C370 821.0 370 821.0 370 835"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 907 C370 923.5 130 923.5 130 940"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 907 C370 923.5 610 923.5 610 940"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M130 1012 C130 1023.5 370 1023.5 370 1035"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M610 1012 C610 1023.5 370 1023.5 370 1035"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M370 1107 C370 1118.5 370 1118.5 370 1130"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-1)" d="M260 771 H8 V1166 H260"/><text class="qi-edge-label" x="18" y="932.5">残差</text><g><rect class="qi-node" x="260" y="15" width="220" height="72" rx="8"/><text x="370" y="57" text-anchor="middle"><tspan x="370" dy="0">X：[1,S,4096]</tspan></text></g><g><rect class="qi-node" x="260" y="115" width="220" height="72" rx="8"/><text x="370" y="157" text-anchor="middle"><tspan x="370" dy="0">LayerNorm + 时间 scale</tspan></text></g><g><rect class="qi-node" x="20" y="220" width="220" height="72" rx="8"/><text x="130" y="262" text-anchor="middle"><tspan x="130" dy="0">Q：[1,S,32,128]</tspan></text></g><g><rect class="qi-node" x="260" y="220" width="220" height="72" rx="8"/><text x="370" y="262" text-anchor="middle"><tspan x="370" dy="0">K：[1,S,32,128]</tspan></text></g><g><rect class="qi-node" x="500" y="220" width="220" height="72" rx="8"/><text x="610" y="262" text-anchor="middle"><tspan x="610" dy="0">V：[1,S,32,128]</tspan></text></g><g><rect class="qi-node" x="140" y="325" width="220" height="72" rx="8"/><text x="250" y="367" text-anchor="middle"><tspan x="250" dy="0">Q/K RMSNorm + RoPE</tspan></text></g><g><rect class="qi-node" x="260" y="435" width="220" height="72" rx="8"/><text x="370" y="477" text-anchor="middle"><tspan x="370" dy="0">带 mask 的 Attention</tspan></text></g><g><rect class="qi-node" x="260" y="535" width="220" height="72" rx="8"/><text x="370" y="564" text-anchor="middle"><tspan x="370" dy="0">合头 + 输出投影</tspan><tspan x="370" dy="21">[1,S,4096]</tspan></text></g><g><rect class="qi-node" x="260" y="635" width="220" height="72" rx="8"/><text x="370" y="677" text-anchor="middle"><tspan x="370" dy="0">乘 tanh Attention gate</tspan></text></g><g><rect class="qi-node" x="260" y="735" width="220" height="72" rx="8"/><text x="370" y="777" text-anchor="middle"><tspan x="370" dy="0">第一次残差：X + 分支</tspan></text></g><g><rect class="qi-node" x="260" y="835" width="220" height="72" rx="8"/><text x="370" y="877" text-anchor="middle"><tspan x="370" dy="0">LayerNorm + 时间 scale</tspan></text></g><g><rect class="qi-node" x="20" y="940" width="220" height="72" rx="8"/><text x="130" y="982" text-anchor="middle"><tspan x="130" dy="0">proj：4096 → 12288</tspan></text></g><g><rect class="qi-node" x="500" y="940" width="220" height="72" rx="8"/><text x="610" y="969" text-anchor="middle"><tspan x="610" dy="0">gate：4096 → 12288</tspan><tspan x="610" dy="21">SiLU</tspan></text></g><g><rect class="qi-node" x="260" y="1035" width="220" height="72" rx="8"/><text x="370" y="1064" text-anchor="middle"><tspan x="370" dy="0">两路逐元素相乘</tspan><tspan x="370" dy="21">[1,S,12288]</tspan></text></g><g><rect class="qi-node" x="260" y="1130" width="220" height="72" rx="8"/><text x="370" y="1159" text-anchor="middle"><tspan x="370" dy="0">out：12288 → 4096</tspan><tspan x="370" dy="21">时间 gate + 第二次残差</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  X[输入 X：1×S×4096] --&gt; N1[LayerNorm 与时间 scale]
  N1 --&gt; Q[Q：1×S×32×128]
  N1 --&gt; K[K：1×S×32×128]
  N1 --&gt; V[V：1×S×32×128]
  Q --&gt; R[Q/K RMSNorm 与 RoPE]
  K --&gt; R
  R --&gt; A[带条件 mask 的 Attention]
  V --&gt; A
  A --&gt; O[合头与输出投影：1×S×4096]
  O --&gt; G[乘 tanh Attention gate]
  X --&gt; ADD[第一次残差相加]
  G --&gt; ADD
  ADD --&gt; N2[LayerNorm 与时间 scale]
  N2 --&gt; U[proj：4096→12288]
  N2 --&gt; W[gate：4096→12288，SiLU]
  U --&gt; M[逐元素相乘]
  W --&gt; M
  M --&gt; D[out：12288→4096]
  D --&gt; G2[乘 tanh MLP gate]
  ADD --&gt; Y[第二次残差相加]
  G2 --&gt; Y
</code></pre></details>
</figure>

Q/K/V 分别投影，再把 4096 维拆成 32 个头。Q/K 在每头的 128 维上做 RMSNorm，随后使用三轴 RoPE；轴维度为 16、56、56，总计 128。注意力的数学形状是 `[B,heads,Sq,Sk]`，实际后端可能分段或分块执行，不一定一次分配完整 score 矩阵。

MLP 的两路投影为：

$$
\operatorname{MLP}(X)=\left(\operatorname{SiLU}(XW_g)\odot XW_u\right)W_d
$$

两路各产生 `[1,S,12288]`，逐元素相乘，再投影回 `[1,S,4096]`。Block 入口和出口的形状一致，变化的是特征内容。

## 扩散时间步、Flow Matching 与 Euler 更新

这个 checkpoint 使用 Flow Matching 与 Euler 调度器。训练原理可以用一条插值路径理解：

$$
z_\sigma=(1-\sigma)z_{\mathrm{clean}}+\sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
$$

sigma=0 是干净 latent，sigma=1 是纯噪声。沿这条训练路径的目标方向为 epsilon−z_clean；DiT 根据当前状态、时间与条件学习预测方向。这里描述预训练模型原理，GR308 项目没有重新训练这套权重。

推理从随机噪声开始，按递减时间表执行：

$$
z_{i+1}=z_i+(\sigma_{i+1}-\sigma_i)v_\theta(z_i,\sigma_i,c)
$$

循环编号 i 与 sigma 不同。i 只是执行次数，sigma 描述当前噪声阶段。第一次 i=0、sigma=1，初始 z 是高斯噪声；代码可能用 t=1000 表示同一时刻，再在进入 DiT 前除以 1000。动态 shifting 会调整中间时间点，不能把示意用的等间隔表当作真实调度记录。

<figure class="qwen-diagram" id="qwen-figure-3">
<figcaption>图 3 · 采样循环：方向预测与 Euler 更新</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 3">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 3">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 3 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 740" role="img" aria-label="采样循环：方向预测与 Euler 更新"><title>采样循环：方向预测与 Euler 更新</title><defs><marker id="qwen-diagram-arrow-2" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M370 87 C370 161.0 370 161.0 370 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M130 197 C130 216.0 370 216.0 370 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M610 197 C610 216.0 370 216.0 370 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M370 307 C370 326.0 370 326.0 370 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M370 417 C370 436.0 370 436.0 370 455"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M260 51 H18 V491 H260"/><text class="qi-edge-label" x="28" y="235.0">当前 latent</text><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M370 527 C370 546.0 370 546.0 370 565"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M480 601 H734 V271 H480"/><text class="qi-edge-label" x="686" y="400.0">继续</text><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-2)" d="M370 637 C370 651.0 370 651.0 370 665"/><text class="qi-edge-label" x="378.0" y="645.0">完成</text><g><rect class="qi-node" x="260" y="15" width="220" height="72" rx="8"/><text x="370" y="57" text-anchor="middle"><tspan x="370" dy="0">当前 zi：[1,4096,64]</tspan></text></g><g><rect class="qi-node" x="20" y="125" width="220" height="72" rx="8"/><text x="130" y="167" text-anchor="middle"><tspan x="130" dy="0">文字 / 参考图条件</tspan></text></g><g><rect class="qi-node" x="500" y="125" width="220" height="72" rx="8"/><text x="610" y="167" text-anchor="middle"><tspan x="610" dy="0">当前时间步 σi</tspan></text></g><g><rect class="qi-node" x="260" y="235" width="220" height="72" rx="8"/><text x="370" y="277" text-anchor="middle"><tspan x="370" dy="0">DiT：32 层 Block</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="72" rx="8"/><text x="370" y="387" text-anchor="middle"><tspan x="370" dy="0">方向 vi：[1,4096,64]</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="72" rx="8"/><text x="370" y="497" text-anchor="middle"><tspan x="370" dy="0">Euler：zi+1 = zi + Δσ·vi</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="72" rx="8"/><text x="370" y="607" text-anchor="middle"><tspan x="370" dy="0">是否完成？</tspan></text></g><g><rect class="qi-node" x="260" y="665" width="220" height="72" rx="8"/><text x="370" y="707" text-anchor="middle"><tspan x="370" dy="0">VAE 解码</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  Z[当前 latent zi：1×4096×64] --&gt; D[DiT：32 层 Block]
  C[文字或参考图条件] --&gt; D
  T[当前时间步 sigma_i] --&gt; D
  D --&gt; V[方向预测 vi：1×4096×64]
  V --&gt; U[Euler 更新 zi+1 = zi + delta_sigma × vi]
  Z --&gt; U
  U --&gt; Q{是否完成？}
  Q --&gt;|继续，更新时间步| D
  Q --&gt;|完成| O[VAE 解码]
</code></pre></details>
</figure>

每次 DiT 都执行全部 32 层，采样步骤之间重复使用同一套权重。在当前确定性配置下，随机性主要来自初始噪声，每一步不需要再随机加一份噪声。

一个标量教学例子：干净值为 0.5、噪声值为 2.0，理想方向为 1.5。每次 sigma 减少 0.25，状态依次是 2.0、1.625、1.25、0.875、0.5。真实模型同时预测 262144 个 latent 元素的方向，并根据每一步的新状态重新计算。

## 时间 gate、残差和采样步长分别控制什么

当前时间经过正余弦编码、时间 MLP 和调制投影，产生 Attention scale/gate 与 MLP scale/gate，各为 4096 维。在 B=1、开启 causal_condition 时，还额外编码一个 t=0 的条件行，因此形状是 `[2,256]→[2,4096]→[2,16384]`。

两条残差分支可以写成：

$$
U=\operatorname{LN}(X)\odot(1+s_A(t)),\quad X_1=X+\tanh(g_A(t))\odot\operatorname{Attention}(U)
$$

$$
V=\operatorname{LN}(X_1)\odot(1+s_M(t)),\quad X_2=X_1+\tanh(g_M(t))\odot\operatorname{MLP}(V)
$$

scale 调整分支输入，gate 控制分支输出加回主干的幅度与符号，残差保留原输入。gate 是逐通道向量，tanh 后在 −1 到 1 之间，不是一个控制循环次数的开关，也不能预设噪声越大门控一定越大。

| 概念 | 作用对象 | 来源 |
| --- | --- | --- |
| 时间 gate | DiT 内部 4096 维隐藏状态 | 时间嵌入网络生成 |
| 残差连接 | Block 输入与分支输出 | 模型结构 |
| delta sigma | 模型外 64 维 latent 的更新步长 | Scheduler 时间表 |

本模型的调制参数由同一采样步的全部 32 层共享；每层 Attention/MLP 权重仍各自独立。文字与参考图条件使用固定 t=0 的调制，目标图像使用当前时间。t=0 不代表 gate=0，它仍经过网络计算。

## 首步缓存为什么改变后续计算

条件前缀不能读取未来目标图像，且其时间调制固定，所以在首步建立每层条件 K/V 缓存后，后续可以复用。这里缓存的是 DiT 的条件 K/V，与 Qwen3-VL 自己的语言模型缓存是两套状态。

设条件前缀长度为 P、目标图像 token 数为 N=4096：

| 场景 | P | 首步 Query 长度 | 后续 Query 长度 | 后续 K/V 长度 |
| --- | ---: | ---: | ---: | ---: |
| 文生图，文字 T=128 | 128 | 4224 | 4096 | 4224 |
| 单参考图编辑 | 4224 | 8320 | 4096 | 8320 |

每层 K/V cache 各为 `[1,P,32,128]`。后续重新计算目标的 Q/K/V，把目标 K/V 与缓存拼起来。32 层 BF16 缓存按此示例计算，文字前缀为 64 MiB，单参考图编辑前缀为 2112 MiB；这是理论参数量推导，不含 allocator 等其他成本。

首步与后续步的执行范围不同，也解释了为什么 Graph 需要分别组织捕获。首次分配、mask 构建和缓存状态都应进入性能记录。

<span class="qwen-legacy-anchor" id="第一阶段-把模型框架与设备后端接起来"></span>
<span class="qwen-legacy-anchor" id="第二阶段-用阶段接口隔离运行时差异"></span>

## 四阶段怎样帮助 GR308 优化

最初的任务是先跑通完整链路。FANT 对 complex64、部分 repeat_interleave、高维 permute、pixel shuffle 的支持存在限制，项目使用实数 RoPE、索引选择、较低维重排和切片 copy 等兼容改写。形状相同不足以证明等价，还要检查坐标、head 顺序及数值。

DiT 与 VAE 同时驻留时，约 47.45 GiB 已分配后再申请约 5.06 GiB 会失败，因而采用保存 latent、释放 Transformer、单独加载 VAE 的执行方式。阶段加载解决可完成性，同时会引入切换开销。

| 接口 | 工作 | 观测边界 |
| --- | --- | --- |
| prepare_encode | 条件编码、初始 latent、时间表 | 编码与准备耗时 |
| denoise_step | DiT 方向预测 | 首步与后续步分别计时 |
| step_scheduler | 更新 latent 与索引 | 数值更新与调度开销 |
| post_decode | 反归一化、VAE 与图片后处理 | 解码、合并与输出耗时 |

四阶段提供设备、状态和生命周期边界，阶段之间仍有数据依赖。它们不等于四张卡，也不意味着四段同时运行。已有实现由独立的 DiT 与 VAE 脚本分阶段验证，不能据此声称完整优化组合的 40 步服务验收已完成。

<span class="qwen-legacy-anchor" id="按层流水线-容量收益成立-延迟收益没有成立"></span>
<span class="qwen-legacy-anchor" id="真正的-tp2-切分每一层的计算"></span>

## TP2：每层同时切注意力头与 MLP 中间通道

硬件是一张双芯物理卡中的两个逻辑设备，rank 0 和 rank 1 各控制一芯。FHCCL 的 broadcast、all-reduce、all-gather 已通过探测，all-to-all 尚不支持，因此采用依赖 all-reduce 的 TP2，原版 Ulysses 路径暂时受限。

此前 16+16 层切分使用 CPU 中转，1024²、4 步由 163.33 秒变为 164.07 秒，单芯峰值降低约 19%。它分散了容量压力，但单图仍按前后层顺序执行。

TP2 则让两颗芯片同时参与每一层，两边都持有完整的层入口隐藏向量：

| 位置 | 完整计算 | TP2 每 rank |
| --- | --- | --- |
| 输入 | `[1,S,4096]` | `[1,S,4096]`，复制 |
| Q/K/V | 各 `[1,S,32,128]` | 各 `[1,S,16,128]` |
| Attention 合头 | `[1,S,4096]` | `[1,S,2048]` |
| Attention 输出投影 | 4096→4096 | 2048→4096 的局部贡献 |
| 第一处 all-reduce 后 | `[1,S,4096]` | 两边得到相同完整结果 |
| SwiGLU gate/proj | 各 `[1,S,12288]` | 各 `[1,S,6144]` |
| MLP 输出投影 | 12288→4096 | 6144→4096 的局部贡献 |
| 第二处 all-reduce 后 | `[1,S,4096]` | 两边得到相同完整结果 |

注意力头切分是 TP2 的一部分；MLP 切的是扩展后的中间特征，不是把层间 4096 维状态永久截成两半。两边仍处理同一批 token。相关权重也同步切分，不先计算完整激活再丢弃一半。

输出投影为什么求和？按数学 `XW` 的写法，若 Attention 输出按头分为 H0、H1，输出权重按输入轴分为 W0、W1，则：

$$
[H_0,H_1]\begin{bmatrix}W_0\\W_1\end{bmatrix}=H_0W_0+H_1W_1
$$

两边分别产生完整宽度的局部贡献，FHCCL all-reduce SUM 将其相加。SwiGLU 也按同一组中间通道切 gate/proj，再在 down projection 后求和。PyTorch 权重存成 `[out,in]`，因此列并行代码切权重第一维，行并行切第二维。

<figure class="qwen-diagram" id="qwen-figure-4">
<figcaption>图 4 · TP2：每层两次 FHCCL all-reduce</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 4">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 4">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 4 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 1030" role="img" aria-label="TP2：每层两次 FHCCL all-reduce"><title>TP2：每层两次 FHCCL all-reduce</title><defs><marker id="qwen-diagram-arrow-3" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M370 87 C370 106.0 130 106.0 130 125"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M370 87 C370 106.0 610 106.0 610 125"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M130 197 C130 216.0 130 216.0 130 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M610 197 C610 216.0 610 216.0 610 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M130 307 C130 326.0 370 326.0 370 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M610 307 C610 326.0 370 326.0 370 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M370 417 C370 436.0 370 436.0 370 455"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M370 527 C370 546.0 130 546.0 130 565"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M370 527 C370 546.0 610 546.0 610 565"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M130 637 C130 656.0 130 656.0 130 675"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M610 637 C610 656.0 610 656.0 610 675"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M130 747 C130 766.0 370 766.0 370 785"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M610 747 C610 766.0 370 766.0 370 785"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-3)" d="M370 857 C370 876.0 370 876.0 370 895"/><g><rect class="qi-node" x="260" y="15" width="220" height="72" rx="8"/><text x="370" y="57" text-anchor="middle"><tspan x="370" dy="0">两边持有完整 X</tspan></text></g><g><rect class="qi-node" x="20" y="125" width="220" height="72" rx="8"/><text x="130" y="167" text-anchor="middle"><tspan x="130" dy="0">rank 0：16 个头</tspan></text></g><g><rect class="qi-node" x="500" y="125" width="220" height="72" rx="8"/><text x="610" y="167" text-anchor="middle"><tspan x="610" dy="0">rank 1：另外 16 个头</tspan></text></g><g><rect class="qi-node" x="20" y="235" width="220" height="72" rx="8"/><text x="130" y="277" text-anchor="middle"><tspan x="130" dy="0">输出局部贡献 Y0</tspan></text></g><g><rect class="qi-node" x="500" y="235" width="220" height="72" rx="8"/><text x="610" y="277" text-anchor="middle"><tspan x="610" dy="0">输出局部贡献 Y1</tspan></text></g><g><rect class="qi-node" x="260" y="345" width="220" height="72" rx="8"/><text x="370" y="387" text-anchor="middle"><tspan x="370" dy="0">AllReduce：Y0 + Y1</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="72" rx="8"/><text x="370" y="497" text-anchor="middle"><tspan x="370" dy="0">时间 gate + 残差</tspan></text></g><g><rect class="qi-node" x="20" y="565" width="220" height="72" rx="8"/><text x="130" y="607" text-anchor="middle"><tspan x="130" dy="0">rank 0：MLP 6144 通道</tspan></text></g><g><rect class="qi-node" x="500" y="565" width="220" height="72" rx="8"/><text x="610" y="607" text-anchor="middle"><tspan x="610" dy="0">rank 1：MLP 6144 通道</tspan></text></g><g><rect class="qi-node" x="20" y="675" width="220" height="72" rx="8"/><text x="130" y="717" text-anchor="middle"><tspan x="130" dy="0">down 局部贡献</tspan></text></g><g><rect class="qi-node" x="500" y="675" width="220" height="72" rx="8"/><text x="610" y="717" text-anchor="middle"><tspan x="610" dy="0">down 局部贡献</tspan></text></g><g><rect class="qi-node" x="260" y="785" width="220" height="72" rx="8"/><text x="370" y="827" text-anchor="middle"><tspan x="370" dy="0">第二次 AllReduce SUM</tspan></text></g><g><rect class="qi-node" x="260" y="895" width="220" height="72" rx="8"/><text x="370" y="924" text-anchor="middle"><tspan x="370" dy="0">时间 gate + 残差</tspan><tspan x="370" dy="21">进入下一层</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  X[两边持有完整 X] --&gt; A0[rank 0：16 个 Attention 头]
  X --&gt; A1[rank 1：另外 16 个头]
  A0 --&gt; O0[输出局部贡献 Y0]
  A1 --&gt; O1[输出局部贡献 Y1]
  O0 --&gt; AR1[AllReduce：Y0+Y1]
  O1 --&gt; AR1
  AR1 --&gt; R[时间 gate 与残差]
  R --&gt; M0[rank 0：6144 个 MLP 通道]
  R --&gt; M1[rank 1：另外 6144 个通道]
  M0 --&gt; D0[down projection 局部贡献]
  M1 --&gt; D1[down projection 局部贡献]
  D0 --&gt; AR2[第二次 AllReduce SUM]
  D1 --&gt; AR2
  AR2 --&gt; Y[时间 gate 与残差，进入下一层]
</code></pre></details>
</figure>

残差应在合并完整分支输出后正确添加。如果先让两个 rank 各加一次 X、再对整个结果求和，会把残差加成两份。每层的两个 all-reduce 对应 Attention 和 MLP 两个分支。

## VAE 解码与分布式瓦片

最终 latent 先还原为 `[1,64,1,64,64]` 并反归一化。Decoder 的主要维度为：

| 位置 | 通道与空间形状，省略 B、F |
| --- | --- |
| latent 与 post_quant_conv | 64×64×64 |
| conv_in 与 Mid Block | 1152×64×64 |
| Up Block 0 | 1152×128×128 |
| Up Block 1 | 1152×256×256 |
| Up Block 2 | 576×512×512 |
| Up Block 3 | 288×1024×1024 |
| Up Block 4 | 144×1024×1024 |
| 归一化、SiLU、输出卷积 | 4×1024×1024 |

Residual Block 使用归一化、激活、两次卷积及 shortcut；Mid Block 还包含空间 Attention。后段高分辨率激活会很大，例如 BF16 的 288×1024×1024 单个激活约 576 MiB，尚不含其他张量和算子工作区。

瓦片配置使用 256×256 像素区域、192×192 步长，对应 latent 的 16×16 区域、12×12 步长。64 维 latent 网格上，起点为 0、12、24、36、48、60，两个方向组合为 36 块，边缘区域较小。双芯每 rank 处理 18 块，每颗芯片拥有完整 VAE 权重。

<figure class="qwen-diagram" id="qwen-figure-5">
<figcaption>图 5 · VAE：瓦片分工、汇总与空间融合</figcaption>
<div class="qwen-diagram-tools"><button type="button" data-qwen-action="zoom-in" aria-label="放大图 5">放大</button><button type="button" data-qwen-action="zoom-out" aria-label="缩小图 5">缩小</button><button type="button" data-qwen-action="reset">还原</button><button type="button" data-qwen-action="expand" aria-expanded="false">完整高度</button><span data-qwen-zoom-label aria-live="polite">100%</span></div>
<div class="qwen-diagram-viewport" tabindex="0" role="region" aria-label="图 5 可滚动区域"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 810" role="img" aria-label="VAE：瓦片分工、汇总与空间融合"><title>VAE：瓦片分工、汇总与空间融合</title><defs><marker id="qwen-diagram-arrow-4" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path class="qi-arrow" d="M0 0L8 4L0 8Z"/></marker></defs><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M370 87 C370 106.0 370 106.0 370 125"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M370 197 C370 216.0 130 216.0 130 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M370 197 C370 216.0 610 216.0 610 235"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M130 307 C130 326.0 130 326.0 130 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M610 307 C610 326.0 610 326.0 610 345"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M130 417 C130 436.0 370 436.0 370 455"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M610 417 C610 436.0 370 436.0 370 455"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M370 527 C370 546.0 370 546.0 370 565"/><path class="qi-edge" marker-end="url(#qwen-diagram-arrow-4)" d="M370 637 C370 656.0 370 656.0 370 675"/><g><rect class="qi-node" x="260" y="15" width="220" height="72" rx="8"/><text x="370" y="57" text-anchor="middle"><tspan x="370" dy="0">完整 64×64 latent 网格</tspan></text></g><g><rect class="qi-node" x="260" y="125" width="220" height="72" rx="8"/><text x="370" y="167" text-anchor="middle"><tspan x="370" dy="0">36 个重叠瓦片</tspan></text></g><g><rect class="qi-node" x="20" y="235" width="220" height="72" rx="8"/><text x="130" y="264" text-anchor="middle"><tspan x="130" dy="0">rank 0：18 块</tspan><tspan x="130" dy="21">完整 VAE 权重</tspan></text></g><g><rect class="qi-node" x="500" y="235" width="220" height="72" rx="8"/><text x="610" y="264" text-anchor="middle"><tspan x="610" dy="0">rank 1：18 块</tspan><tspan x="610" dy="21">完整 VAE 权重</tspan></text></g><g><rect class="qi-node" x="20" y="345" width="220" height="72" rx="8"/><text x="130" y="387" text-anchor="middle"><tspan x="130" dy="0">写入对应 buffer 槽位</tspan></text></g><g><rect class="qi-node" x="500" y="345" width="220" height="72" rx="8"/><text x="610" y="387" text-anchor="middle"><tspan x="610" dy="0">写入对应 buffer 槽位</tspan></text></g><g><rect class="qi-node" x="260" y="455" width="220" height="72" rx="8"/><text x="370" y="497" text-anchor="middle"><tspan x="370" dy="0">AllReduce 汇总瓦片集合</tspan></text></g><g><rect class="qi-node" x="260" y="565" width="220" height="72" rx="8"/><text x="370" y="607" text-anchor="middle"><tspan x="370" dy="0">rank 0：overlap / blend</tspan></text></g><g><rect class="qi-node" x="260" y="675" width="220" height="72" rx="8"/><text x="370" y="704" text-anchor="middle"><tspan x="370" dy="0">裁剪输出</tspan><tspan x="370" dy="21">1024×1024 RGBA</tspan></text></g></svg></div>
<details><summary>查看本图 Mermaid 源码</summary><pre><code class="language-mermaid">flowchart TD
  L[完整 64×64 latent 网格] --&gt; T[切成 36 个重叠瓦片]
  T --&gt; R0[rank 0：18 块，完整 VAE]
  T --&gt; R1[rank 1：18 块，完整 VAE]
  R0 --&gt; B0[写入对应 tile buffer 槽位]
  R1 --&gt; B1[写入对应 tile buffer 槽位]
  B0 --&gt; A[AllReduce SUM 汇总瓦片集合]
  B1 --&gt; A
  A --&gt; M[rank 0：overlap 与 blend]
  M --&gt; O[裁剪后输出 1024×1024 RGBA]
</code></pre></details>
</figure>

all-reduce 汇总的是互斥槽位：负责该块的 rank 写数据，另一边对应位置为零；相邻瓦片的空间重叠由后续 blend 处理。

<span class="qwen-legacy-anchor" id="分块-vae-主要收益先来自算法路径"></span>

## 同硬件下，瓦片化为什么能快这么多

这是本次追问中最重要的修正。早先只强调工作集与显存，没有充分展开 VAE 的 Attention：整图解码时，Mid Block 的 64×64 网格有 4096 个位置，Attention 交互对数为 4096²=16777216；一个 16×16 瓦片只有 256 个位置。

按上述 36 块的真实边缘尺寸，全部瓦片的 Attention 位置对数合计为 1679616，约为整图十分之一。与此同时，重叠瓦片的空间位置总量为整图约 1.72 倍，卷积等线性空间计算可能增加。因此瓦片化同时减少跨区域 Attention、增加重叠区计算，并改变算子形状和工作区选择。

| 路径 | 历史耗时 | 能支持的判断 |
| --- | ---: | --- |
| 单芯整图 VAE | 约 1241 s | 原执行路径 |
| 单芯串行瓦片 | 154.70 s | 改变解码组织后约 8 倍差异 |
| 双芯瓦片 | 143.50 s | 相对串行瓦片再缩短约 7.2% |

Attention 交互减少约十倍，并不能单独证明整个 VAE 的八倍收益来源。若 Attention 只占一半时间，即使这部分十倍加速，按其他部分不变估算整体也只有约 1.82 倍。记录同时指出上采样兼容路径较慢，所以必须继续比较整图/瓦片的逐算子时间，才能分摊各因素贡献。

单芯串行瓦片的记录峰值约 39.64 GiB，甚至高于整图约 30.15 GiB；双芯则约为 17.4/23.0 GiB。因此“小块一定降低整个进程峰值，所以必然更快”也不是现有记录支持的因果结论。

### 瓦片化还改变了数值语义

整图 Attention 可访问所有位置，瓦片只在本块内交互；卷积边界上下文也变化。overlap/blend 可以缓和接缝，不能恢复全部整图交互，所以整图与瓦片不保证逐元素等价。

已记录的 MAE 0.0435/255、99.9815% 通道差不超过 1、最大差 11，比较的是同一 latent 的单芯串行瓦片与双芯瓦片。它验证分布式瓦片执行的一致性，不能替代整图对瓦片的质量验证。后续应固定同一 latent 比较整图与不同 tile/overlap，检查接缝、纹理、文字与透明边缘。

<span class="qwen-legacy-anchor" id="单卡基线-真正出图以后再计时"></span>

## 1.82 倍究竟测量了什么

| DiT 负载 | 单芯 | TP2 | 加速比 |
| --- | ---: | ---: | ---: |
| 512×512，2 步 | 9.28 s | 5.51 s | 1.68× |
| 1024×1024，4 步 | 163.33 s | 89.60 s | 1.82× |

TP2 脚本先编码提示词，再开始去噪计时，最终输出 latent。因此这个数字覆盖采样阶段的墙钟时间与相关循环开销，不包括权重加载、提示词编码和 VAE 解码。耗时下降约 45.1%，不应说成耗时下降 82%。

每 rank 的 DiT 参数字节由约 14.230 GB 变成 7.251 GB，保留了一些复制模块，所以没有严格减半。1024 短测每芯记录峰值约 29.495 GB。两个 rank 最终 latent 一致；相对单芯基线 MAE 0.00575、最大差 0.09375，所有值有限。跨 rank 一致与对单芯参考接近，是两项不同检查。

按短测线性外推的 40 步 DiT 约 896 秒，以及与独立 VAE 时间相加得到的约 17.3 分钟，只能用于规划预算。它们没有形成一次完整优化组合的 40 步端到端实测，不能与旧完整出图直接组成已实现的总加速比。

<span class="qwen-legacy-anchor" id="剩余瓶颈-名字叫-sdpa-实际走哪条路径"></span>
<span class="qwen-legacy-anchor" id="小算子融合-单点很快-整块提升更小"></span>
<span class="qwen-legacy-anchor" id="用阶段数据重建一次请求的成本"></span>

## 按瓶颈迁移安排下一轮优化

阶段成本可以写为准备＋首步＋(N−1)个后续步＋VAE＋加载切换。原单芯四步计算中，DiT/VAE 约占 11.6%/88.4%；40 步约为 56.7%/43.3%。比例只针对这两个阶段，不含完整冷启动。步数改变瓶颈优先级，四步不能替代正常步数的质量和响应时间验收。

TP2 后真实 Block 的剖析外推如下：

| 部分 | 约占完整一步 |
| --- | ---: |
| SDPA | 69%～70% |
| 归一化、残差及重排 | 12%～13% |
| GEMM | 2.8% |
| all-reduce 调用，含 rank 等待 | 0.9%～2.1% |

独立 32 MiB BF16 all-reduce 约 1.57–1.58 ms；每层两次、32 层的纯传输估算约 0.10 秒。真实调用还可能等待前序计算与其他 rank，不能把全部墙钟都算成链路慢。现有证据支持优先处理 SDPA，缺少硬件计数器时仍不能断言纯带宽或纯计算饱和。

强制 FLASH/EFFICIENT 选项仍落到约 484 ms 的同一路径；显式 matmul-softmax-matmul 约 569 ms；Python 分块在线 softmax 在 1024 序列约 76–85 ms，对照约 34.7 ms，反而更慢。减少理论中间存储不保证由多个小算子拼出的实现更快。

原生 FTUCA SDPA 属于候选方向，现有材料没有完整性能验收。即使固定方形、无 mask 微基准通过，还需要覆盖真实缓存后续步的非等长 Q/K、首步 segment/mask、真实 Block 和完整采样。

RMSNorm 与 RoPE 单算子分别约 19×、38.5×，组合真实 Block 为 0.67934→0.63190 秒，下降约 6.98%。该组合只完成单算子和 Block 门，默认关闭，不能叠加到已实测 1.82× 上。SiLU 单点虽然快，放回完整 MLP 前半段只有 9.35→9.21 ms，未据此改动布局。

Graph 适合在执行路径稳定后推进：分别捕获首步和 cached 后续步，预分配 K/V、RoPE 和 mask，消除循环内动态 CPU/Python 构造，再验证状态生命周期。它改善执行组织，不能预先消除 SDPA 本体成本。

## 从主会话保留下来的跨模型经验

图像模型是这篇笔记的主线，主会话中的其他模型提供了几个有用对照。这里的数值按各自历史口径保留，不组成跨模型性能排名。

这些文本实验涉及的基础术语也要分清：TP 把同一层计算分给多个设备，DP 用多个完整副本处理不同请求，EP 分配 MoE 专家并搬运路由 token；MTP 提出多个候选 token 后由目标模型验证，收益取决于接受率与验证成本；KV cache 保存注意力的历史 K/V，混合架构还可能有独立 recurrent state；Graph 重放固定执行区域，不能自动修复动态状态或不支持的算子。聚合 output tok/s 是实际输出 token 总数除以墙钟时间，报告中的 Decode Avg TPS 则按平均 TPOT 的倒数计算。

| 方向 | 选定路线与结果 | 关键范围 |
| --- | --- | --- |
| Qwen3.5-9B | TP1×DP2、MTP4；C16 正式 247.16 output tok/s | 64 请求、0 失败；拓扑短测 157.67→247.87 为另一组对照 |
| Qwen3-30B-A3B | TP2、关闭 EP、异步调度；C16 正式 225.67 | 64 请求、0 失败；C32 320.90 是短测 |
| Qwen3.6-35B-A3B | TP2、MTP4、batch4096；C16 180.64 | 32 请求短测候选，最佳入口为 D 组 |
| Qwen3-Next-80B-A3B | TP4、MTP3；C16 255.93 | 32 请求短测、两物理卡；输入长度也不同 |
| Wan2.2 | VAE S2 混合精度；1190.20→672.33 s | 有效 5 帧、3 步；PSNR39.36仅固定首帧参考 |
| MiniMax H3 | 平台基础算子通过，W4A8 原生分派未证实 | fallback 数值正确不构成原生加速；没有整模型性能结论 |

27B 还有基线修正：早期环境单副本约 18–19 tok/s，后续复现厂商环境得到 53.574。此时强制 KV152 几乎不改变吞吐，冷启动 MTP1 C16 两次死锁。早期速度不是硬件上限；先跑 C1 与直接冷启动 C16 的状态也不能混为一谈。Qwen3.8-27B 与 Qwen3.6-27B 是材料中的不同命名，未核对权重身份前分别叙述。

完整文本与视频流程分别保留在本站的 [GR308 文本篇](/2026/10/02/gr308-language/)、[GR308 视频篇](/2026/10/02/gr308-video/)。共同原则是测实际 dispatch、区分阶段与全链、保留失败实验、把质量与稳定性放在性能结果旁边。

<span class="qwen-legacy-anchor" id="这次实现留下了什么"></span>

## 我的追问与需要练习的回答

这次讨论以我的连续追问为主，没有一份完整的模拟面试答卷。因此下面记录的是容易讲不完整的概念及改进表达，不把追问杜撰成曾经答错的原话。

| 我问过的问题 | 容易漏掉的点 | 更准确的回答 |
| --- | --- | --- |
| 参数量是多少，总共呢？ | 只报 DiT 的 7B | DiT约7.115B，完整权重合计约16.22B，需说明组件范围 |
| 整体工作架构是什么？ | 只说“输入文字生成图片” | 条件编码一次，DiT与Scheduler迭代，VAE最后解码；编辑另有参考图双分支 |
| 中间量怎样变化？ | 混淆 token 数与特征宽度 | 从 `[B,C,F,H,W]` 展平为 `[B,N,C]`，再投影到模型隐藏宽度 |
| DiT 怎样扩散？ | 把方向预测当成一次直接出图 | 当前latent与时间条件进入32层DiT，Scheduler按预测方向多步更新 |
| TP2和注意力头切分是一个东西吗？ | 只讲一半Attention | 本实现同时切Attention头和MLP中间通道，每层两次all-reduce |
| MLP切的是隐藏向量吗？ | 把4096层间状态永久切半 | 每芯保留完整入口，切12288扩展中间通道，各6144，再归约局部输出 |
| 时间gate和时间步有什么关系？ | 把gate当成循环次数或delta sigma | 时间网络生成逐通道scale/gate；Scheduler步长另控制latent移动距离 |
| 第一次sigma如何初始化？ | 把循环i当sigma | 文生图i=0时sigma=1、latent为高斯噪声，后续按调度表递减 |
| latent二维为何写4096×64？ | 把展平误认为重新压缩 | 64×64空间展成4096位置，每个位置仍有64个特征 |
| B/C/F/H/W分别是什么？ | 相同数字看起来像同一轴 | 批次、通道、帧、高、宽；C=64与H=W=64含义不同 |
| 相同硬件分块为什么快8倍？ | 只解释为内存更小 | 局部Attention改变工作量，形状和工作区也变化；实际贡献需剖析 |

我还要主动补充三句：1.82×只覆盖四步DiT；瓦片/整图不保证严格等价；单算子通过、单Block通过和完整成图通过是不同层级。回答时先说明对象、形状、执行次数和验证范围，再报提升数字。

### 一段能够连通整条知识链的口述

以1024×1024文生图为例，Qwen3-VL先把提示词编码成4096维条件。目标从64×64网格、每格64维的随机latent开始，展平成4096个token，再投影到4096维。每个采样步，时间嵌入控制DiT的scale和gate，32层Transformer预测latent更新方向，Scheduler沿递减sigma推进状态。我的TP2在每层把32个注意力头分为两组，并把12288维SwiGLU中间通道分成两半；两处行并行输出通过FHCCL求和。采样结束后恢复latent空间布局，交给分布式重叠瓦片VAE解码并拼接。四步DiT短测从163.33秒降到89.60秒；VAE的路径变化另行测试，完整40步优化组合仍需单独验收。

<span class="qwen-legacy-anchor" id="公开工程与恢复版本的边界"></span>

## 复现时保留的边界与资料

调优顺序是先修兼容和容量，再对固定输入比较替代路径，随后下钻真实Block与算子，最后回到完整请求。至少固定checkpoint、dtype、分辨率、步数、prompt、seed、逻辑/物理设备与计时范围；保存首步、后续步、加载切换、显存和误差。只有单轮短测时，不声称统计显著或长期稳定。

历史恢复代码包含失败尝试，不应把多个补丁无条件顺序执行。现有本地与公开材料并未完整包含所有远端原始日志；“代码找回”不代表最终质量门已通过。公开工程固定修订为 `51a0d34beab550c484492c36fc09a143df9624b4`。

- [Qwen-Image-2.1 官方模型与配置](https://huggingface.co/Qwen/Qwen-Image-2.1)。
- [官方 Transformer 配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/transformer/config.json)、[编码器配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/text_encoder/config.json)、[VAE 配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/config.json)、[Scheduler 配置](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json)。
- [GR308 Lab 固定修订](https://github.com/KunyuanComputing/gr308-lab/tree/51a0d34beab550c484492c36fc09a143df9624b4)。
- [图像并行测试记录](https://github.com/KunyuanComputing/gr308-lab/blob/51a0d34beab550c484492c36fc09a143df9624b4/projects/qwen-image21-omni-fant/docs/QWEN_IMAGE_21_OMNI_FANT_TEST_20260922.md)。
- [TP2 实现](https://github.com/KunyuanComputing/gr308-lab/blob/51a0d34beab550c484492c36fc09a143df9624b4/projects/qwen-image21-omni-fant/scripts/tp2_full_dit.py)、[四阶段接口](https://github.com/KunyuanComputing/gr308-lab/blob/51a0d34beab550c484492c36fc09a143df9624b4/projects/qwen-image21-omni-fant/src/qwen_image21_omni_fant/staged.py)、[分布式 VAE](https://github.com/KunyuanComputing/gr308-lab/blob/51a0d34beab550c484492c36fc09a143df9624b4/projects/qwen-image21-omni-fant/scripts/distributed_vae_decode.py)。
- [恢复版本说明](https://github.com/KunyuanComputing/gr308-lab/blob/51a0d34beab550c484492c36fc09a143df9624b4/archive/recovered/README.md)。

所有GR308性能数值属于项目历史测量，不代表Diffusers上游的性能声明；本文对Attention交互数、缓存大小与维度的计算属于代码层面的推导。

<link rel="stylesheet" href="/css/qwen-image-notes.css?v=8cdbf2c3b938">

<script src="/js/qwen-image-notes.js?v=a563ea6d2614" defer></script>
