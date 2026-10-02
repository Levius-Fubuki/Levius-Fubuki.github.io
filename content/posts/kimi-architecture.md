# Kimi K3 架构笔记：KDA、深度残差与稀疏专家

本文由既有深读笔记整理入博客，并核对了对应 Codex 学习工作区。内容区分论文事实、教学推导与实验边界；论文中的性能与规模数据属于作者报告，不是我在本地复现的成绩。原笔记的公式、示例和推导在此保留。

<div class="study-note">



<p class="lead">把 2.78T 参数、百万 Token、原生视觉放在一起并不是简单堆规模。K3 分别沿着序列、深度和通道三个方向设计信息通路，再用系统优化让这些结构真的跑得起来。</p>

<section class="searchable-section" id="overview">
<h2>01 · 先看整体：三条信息轴</h2>
<div class="grid">
<div class="card"><span class="tag">SEQUENCE / TOKEN MIXING</span><h3>序列方向</h3><p><strong>3 层 KDA + 1 层 Gated MLA</strong>。KDA 低成本地压缩历史，MLA 周期性地进行全局内容寻址。</p></div>
<div class="card"><span class="tag">DEPTH / LAYER MIXING</span><h3>深度方向</h3><p><strong>Block Attention Residuals</strong> 让后层有选择地读取嵌入和此前各块表示，而非只接收一条累计残差。</p></div>
<div class="card"><span class="tag">WIDTH / CHANNEL MIXING</span><h3>宽度方向</h3><p><strong>Stable LatentMoE</strong> 提供 896 个路由专家；每个 Token 只激活其中 16 个，并始终经过 2 个共享专家。</p></div>
<div class="card"><span class="tag">MODALITY / INPUT</span><h3>模态入口</h3><p><strong>MoonViT-V2</strong> 把图片和视频压成视觉 Token，经 MLP 投影后与文本进入同一个主干。</p></div>
</div>
<div aria-label="Kimi K3 总体数据流" class="flow">
<div class="flow-row"><div class="node"><b>文本</b>Token Embedding</div><div class="arrow">↘</div><div class="node"><b>统一 Token 流</b>text + vision</div><div class="arrow">→</div><div class="node"><b>93 层主干</b>KDA / MLA + MoE</div><div class="arrow">→</div><div class="node"><b>输出</b>下一个 Token</div></div>
<div class="flow-row" style="margin-top:8px"><div class="node"><b>图片 / 视频</b>MoonViT-V2</div><div class="arrow">→</div><div class="node"><b>压缩与投影</b>Pixel Shuffle + MLP</div><div class="arrow">↗</div></div>
</div>
<div class="callout claim"><span class="tag">论文披露</span>每个注意力层之后都配一个 Stable LatentMoE 前馈层；主干按 3:1 混合 KDA 与 MLA，并在末尾追加全局 MLA，使最终输出一定经过全局交互。</div>
</section>
<section class="searchable-section" id="specifications">
<h2>02 · 关键规格怎么读</h2>
<div class="table-wrap"><table><thead><tr><th>指标</th><th>Kimi K3</th><th>初学者应该怎么理解</th></tr></thead><tbody>
<tr><td>总参数</td><td><strong>2.78T</strong></td><td>模型全部权重容量，不等于每个 Token 都计算全部权重。</td></tr>
<tr><td>激活参数</td><td><strong>104.2B</strong></td><td>一个 Token 通过路由后实际参与主要计算的参数规模。</td></tr>
<tr><td>层与隐藏维度</td><td>93；7168</td><td>深度为 93，残差流中每个 Token 的主体表示为 7168 维。</td></tr>
<tr><td>注意力组成</td><td><strong>69 KDA + 24 MLA</strong></td><td>大部分层用固定状态记忆，少数层做全局注意力。</td></tr>
<tr><td>MoE</td><td><strong>896 路由 / Top-16 / 2 共享</strong></td><td>专家库很大，但每个 Token 只选择一小组专门专家。</td></tr>
<tr><td>上下文</td><td><strong>1M Token</strong></td><td>可接受的最大序列窗口，不等于所有 1M 内容都被无损记住。</td></tr>
<tr><td>视觉塔</td><td>401M；27 层；Patch 14</td><td>图片/视频先被独立编码，再进入语言主干。</td></tr>
</tbody></table></div>
<div class="callout analogy"><span class="tag">教学类比</span>把总参数看作一所拥有 896 个专业科室的医院；激活参数是这位病人实际调用的科室与通用部门。医院很大，不表示一次问诊要让所有医生同时工作。</div>
</section>
<section class="searchable-section" id="attention-prerequisite">
<h2>03 · Attention 与 KV Cache：只补必要地基</h2>
<p>每个 Token 经过线性投影得到 Query、Key、Value。可以粗略理解为：<strong>Q 表示“我要找什么”，K 表示“我能被怎样匹配”，V 是匹配后取出的内容</strong>。标准全局注意力让 Q 与此前所有 K 比较，再对 V 加权求和。</p>
<div class="grid"><div class="card"><h3>Prefill</h3><p>一次处理已有提示词。例如先读完 40 万 Token 的代码仓库上下文。Token 维度有较强并行性。</p></div><div class="card"><h3>Decode</h3><p>自回归地逐个生成新 Token。每一步只有少量新 Query，却要读取很长的历史缓存。</p></div></div>
<p>KV Cache 保存各历史 Token 的 Key/Value，长度越长占用越大。如果 93 层全部采用标准注意力，1M Token 下缓存与计算都会非常昂贵。K3 的目标不是彻底消灭全局注意力，而是让大多数层用固定状态 KDA，只让少数 MLA 层保留全局访问。</p>
</section>
<section class="searchable-section" id="kda">
<h2>04 · KDA：会遗忘、会纠错的在线记忆</h2>
<p>Kimi Delta Attention 为每个头维护固定形状的状态矩阵 <code>S<sub>t</sub>∈R<sup>dₖ×dᵥ</sup></code>。它不逐 Token 保存所有 K/V，而是把历史不断写进这个矩阵。</p>
<div class="equation">Sₜ = (I − βₜ kₜ kₜᵀ) Diag(αₜ) Sₜ₋₁ + βₜ kₜ vₜᵀ
õₜ = Sₜᵀ qₜ</div>
<p>把第一式展开，就得到更直观的四步：</p>
<div class="steps"><div class="step"><strong>按通道遗忘：</strong><code>D = Diag(αₜ)Sₜ₋₁</code>。每个 Key 通道都有自己的保持率。</div><div class="step"><strong>检查旧记忆：</strong><code>prediction = kₜᵀD</code>，看看当前 Key 原来能从状态中读出什么。</div><div class="step"><strong>只写误差：</strong><code>correction = βₜkₜ(vₜᵀ − prediction)</code>。旧答案正确就少改，错误才重点修正。</div><div class="step"><strong>按 Query 读取：</strong><code>õₜ=Sₜᵀqₜ</code>，Query 决定从更新后的联想记忆里读什么。</div></div>
<h3>四个 Token 的玩具例子</h3>
<div class="callout analogy"><span class="tag">教学类比 · 非真实张量</span>
            Token 1 写入“城市：Paris”；Token 2 写入“国家：France”；Token 3 再遇到相近 Key，却给出修正信息“会议城市：Lyon”。若旧状态预测 Paris，而新 Value 是 Lyon，Delta 项主要写入 Lyon−Paris 的误差。Token 4 的 Query 问“会议城市”，便从修正后的状态读出更接近 Lyon 的表示。真实 KDA 是连续高维向量，不会以字符串槽位保存事实。
          </div>
<h3>α 与 β 分别控制什么</h3>
<div class="grid"><div class="card"><h3>α：保持率</h3><p>逐 Key 通道控制旧状态保留多少。它决定记忆时间尺度，不同通道可以快忘或慢忘。</p></div><div class="card"><h3>β：写入强度</h3><p>一个标量门，控制当前 Delta 修正有多强。β 小表示这次输入不应大改记忆。</p></div></div>
<p>Q/K/V 投影前后还包括短因果卷积、Swish 与 Q/K 的 L2Norm；输出经过逐头 RMSNorm，再用输入相关的全秩门：</p>
<div class="equation">yₜ = Wₒ [ Sigmoid(Wg xₜ) ⊙ RMSNorm(õₜ) ]</div>
<h3>为什么还要 Chunkwise</h3>
<p>逐 Token 递推与 GPU 喜欢的大块并行相冲突。KDA 将序列分块：<strong>块间传播状态，块内改写为因果矩阵乘法</strong>。块内输出由“进入该块的历史状态”和“本块 Token 间交互”两部分组成。</p>
<div class="flow"><div class="flow-row"><div class="node"><b>Chunk 1</b>块内并行</div><div class="arrow">S₁ →</div><div class="node"><b>Chunk 2</b>块内并行</div><div class="arrow">S₂ →</div><div class="node"><b>Chunk 3</b>块内并行</div></div></div>
<h3>Lower-bounded decay：一个硬件友好的数值设计</h3>
<div class="equation">gₜ = gmin · Sigmoid(eᴬ zₜ),  gmin = −5
αₜ = exp(gₜ),  因而 e⁻⁵ &lt; αₜ &lt; 1</div>
<p>Chunk 并行式会使用累计保持率的倒数。若 α 可无限接近 0，连续相乘后倒数会溢出。K3 把 16-Token Tile 内累计 log-decay 限制在 (−80,0)，倒数小于 e⁸⁰，仍在 BF16 动态范围内。结果是对角 Tile 和非对角 Tile 都能走密集 Tensor Core GEMM，而不需要逐位置特殊路径。</p>
<div class="callout derive"><span class="tag">本文推导</span>固定形状状态降低的是“随序列长度增长的状态量”。它并不意味着历史无损：许多 Token 的信息必须被压进同一状态矩阵，精确原文寻址仍需要 MLA。</div>
</section>
<section class="searchable-section" id="mla">
<h2>05 · Gated MLA：定期查全局档案</h2>
<p>Multi-head Latent Attention 先把每个 Token 压缩为低维潜变量：</p>
<div class="equation">cₜ = Wc xₜ
缓存 cₜ → 需要时上投影重建各头 K/V → 全局 Softmax Attention</div>
<p>它仍能访问完整前缀，只是缓存的不是各头完整 K/V，而是共享的压缩表示。K3 再加一个输入相关全秩输出门：</p>
<div class="equation">yₜ = Wₒ [ Sigmoid(Wg xₜ) ⊙ õₜ ]</div>
<div class="table-wrap"><table><thead><tr><th></th><th>KDA</th><th>Gated MLA</th></tr></thead><tbody><tr><td>历史表示</td><td>固定大小递归状态</td><td>随序列增长的低维潜变量缓存</td></tr><tr><td>优势</td><td>长序列持续混合便宜</td><td>可对完整历史做精确内容寻址</td></tr><tr><td>局限</td><td>压缩可能丢失细节</td><td>仍随序列长度增加缓存和注意力成本</td></tr><tr><td>位置</td><td>69 层</td><td>24 层，周期性出现</td></tr></tbody></table></div>
<h3>NoPE 不等于没有顺序</h3><p>MLA 的 Q/K 不加显式位置编码；位置敏感性和近因性主要由其间 KDA 的因果递推、短卷积和衰减提供。这避免了把上下文扩到 1M 时重新调 RoPE 频率，但模型仍然要通过长上下文课程学习掌握远距离依赖。</p>
<div class="stack"><div class="layer kda"><b>KDA</b><span>压缩并更新历史</span></div><div class="layer kda"><b>KDA</b><span>压缩并更新历史</span></div><div class="layer kda"><b>KDA</b><span>压缩并更新历史</span></div><div class="layer mla"><b>Gated MLA</b><span>全局内容检索</span></div><div class="layer kda"><b>重复该节奏</b><span>最终再以 MLA 收尾</span></div></div>
</section>
<section class="searchable-section" id="attnres">
<h2>06 · Attention Residuals：沿深度检索</h2>
<p>标准残差把历史层不断相加进一个状态，像“沿深度运行的 RNN”。后层看到的是累计结果，不能明确选择某个浅层或中层表示。AttnRes 把注意力思想从 Token 轴搬到层轴。</p>
<div class="equation">qₗ = wₗ                         （第 l 层的可学习伪查询）
kᵢ = vᵢ = fᵢ(hᵢ)                 （此前层的输出）
αᵢ→ₗ = softmaxᵢ[wₗᵀ RMSNorm(kᵢ)]
hₗ = Σᵢ αᵢ→ₗ vᵢ</div>
<p><code>wₗ</code> 对层固定，但 <code>kᵢ</code> 来自当前 Token 的历史层表示，所以不同 Token 的跨层权重仍然不同。</p>
<div class="callout analogy"><span class="tag">教学例子</span>在 6 层玩具网络中，第 6 层处理变量名 Token 时，可能给第 1 层的字面/拼写特征 0.35 权重，给第 5 层的程序语义 0.50 权重，其余层共 0.15。另一个 Token 会得到不同分配。</div>
<h3>为什么使用 Block AttnRes</h3>
<p>保存全部 93 层输出会带来 O(Ld) 激活和流水线通信。K3 约每 12 层组成一个块，块内输出先求和，后层只注意嵌入、已完成块和当前块的部分和。论文称 8 个主块可恢复大部分收益；计入嵌入来源为 9 个来源。</p>
<div class="equation">Full AttnRes：保存每层表示 → O(Ld)
Block AttnRes：保存 N 个块表示 → O(Nd)，N ≈ 8</div>
</section>
<section class="searchable-section" id="moe">
<h2>07 · Stable LatentMoE：重点拆开路由全过程</h2>
<h3>先从 Dense FFN 讲起</h3>
<p>普通 Transformer 的每个 Token 都通过同一套 FFN 权重。MoE 把 FFN 换成多个“专家”，Router 根据当前 Token 选择少数专家。这样可以增加总参数容量，而不让每个 Token 运行全部专家。</p>
<div class="flow"><div class="flow-row"><div class="node"><b>Token x</b>7168 维</div><div class="arrow">↗</div><div class="node"><b>2 个共享专家</b>总是执行</div><div class="arrow">→</div><div class="node"><b>相加</b>通用 + 专门</div></div><div class="flow-row" style="margin-top:8px"><div class="node"><b>Token x</b>7168 维</div><div class="arrow">→</div><div class="node"><b>降维 + Router</b>3584 维 / Top-16</div><div class="arrow">→</div><div class="node"><b>16 / 896 路由专家</b>加权聚合</div><div class="arrow">↗</div></div></div>
<h3>一个完整的 8 Token 路由例子</h3>
<p>假设只有 4 个专家、每个 Token 选 1 个。Router 为每行产生分数，当前最大值用 ● 标出：</p>
<div class="flow"><div class="route-grid"><div>Token</div><div>E1</div><div>E2</div><div>E3</div><div>E4</div>
<div>T1</div><div class="chosen hot">.91 ●</div><div>.30</div><div>.20</div><div>.10</div>
<div>T2</div><div class="chosen hot">.83 ●</div><div>.42</div><div>.33</div><div>.18</div>
<div>T3</div><div class="chosen hot">.78 ●</div><div>.70</div><div>.25</div><div>.21</div>
<div>T4</div><div class="chosen hot">.73 ●</div><div>.69</div><div>.28</div><div>.25</div>
<div>T5</div><div>.55</div><div class="chosen hot">.88 ●</div><div>.40</div><div>.32</div>
<div>T6</div><div>.43</div><div class="chosen hot">.81 ●</div><div>.58</div><div>.40</div>
<div>T7</div><div>.38</div><div class="chosen hot">.71 ●</div><div>.69</div><div>.63</div>
<div>T8</div><div>.20</div><div>.54</div><div class="chosen">.76 ●</div><div>.74</div>
</div></div>
<p>负载为 <strong>(4,3,1,0)</strong>。E1 成为慢点，E4 没有训练样本。目标负载：</p>
<div class="equation">m = 8, n = 4, k = 1
q = m·k/n = 8·1/4 = 2 Token / expert</div>
<h3>Router 偏置只管“派给谁”</h3>
<div class="equation">sᵢ = Sigmoid(Wr xᵢ)
Tᵢ = argtop-k(sᵢ + b)                 ← 选专家时使用偏置
pᵢⱼ = sᵢⱼ / Σᵣ∈Tᵢ sᵢᵣ              ← 混合权重不用偏置</div>
<p>这很关键：偏置 <code>b</code> 是调度旋钮，不直接污染专家输出的相对权重，也不通过辅助损失扭曲 Router 的主优化目标。</p>
<h3>Quantile Balancing 怎么一步找阈值</h3>
<p>对每个 Token 先取 Top-(k+1)，第 k+1 名分数作为“要挤进 Top-k 必须超过的截止线” αᵢ。对专家 j，计算每个 Token 的 margin：<code>sᵢⱼ−αᵢ</code>。若要让该专家接收比例 k/n 的 Token，就把偏置设置为这些 margin 的 <code>1−k/n</code> 分位数的相反数。</p>
<div class="equation">b̂ⱼ⁽ᵗ⁺¹⁾ = −quantile₍₁₋ₖ/ₙ₎(s:,j − α⁽ᵗ⁾)
b⁽ᵗ⁺¹⁾ = b̂⁽ᵗ⁺¹⁾ − mean(b̂⁽ᵗ⁺¹⁾)·1</div>
<div class="steps"><div class="step">E1 竞争力过强：提高它进入 Top-k 的有效门槛。</div><div class="step">E3/E4 边缘 Token 与截止线非常接近：降低它们的有效门槛。</div><div class="step">下一 Batch 才启用新偏置，避免用当前 Batch 的结果反过来路由自身。</div><div class="step">教学示例可得到 (2,2,2,2)；真实训练用全局分布近似目标负载，不保证每个微批绝对相等。</div></div>
<p>全局 Batch 有数百万 margin，无法全部 Gather。实现改为各 Rank 统计每专家直方图，对 Bin 计数做一次 All-Reduce，再从累计计数恢复分位数，误差由 Bin 宽控制。最终路由偏置在推理时冻结。</p>
<h3>为什么叫 LatentMoE</h3>
<div class="equation">z = W↓x ∈ R³⁵⁸⁴
u = Σᵢ∈Top-16 pᵢ Eᵢʳᵒᵘᵗᵉᵈ(z)
y = Σⱼ₌₁² Eⱼˢʰᵃʳᵉᵈ(x) + W↑ RMSNorm(u)</div>
<p>路由专家只处理半宽潜空间，降低一份 Token 同时发送给 16 个专家时的通信与权重流量。共享专家保留完整宽度通路，负责通用变换。</p>
<h3>Stable 的另外两个部件</h3>
<div class="grid"><div class="card"><h3>Normalized LatentMoE</h3><p>不同专家组合的聚合幅值可能差异很大。升维前 RMSNorm 先统一尺度，避免 routed path 冲击共享路径。</p></div><div class="card"><h3>SiTU-GLU</h3><p>对门分支与 up 分支分别用平滑 tanh 限幅，在原点附近保持近似 SwiGLU，极大输入时不再无限增长。</p></div></div>
<div class="equation">SiTU-GLU(x) = [β₁ tanh(Wg x/β₁) ⊙ σ(Wg x)]
                 ⊙ [β₂ tanh(Wu x/β₂)]
β₁ = 4, β₂ = 25, 因而标量乘积受平滑上界控制</div>
<div class="callout claim"><span class="tag">不要混淆</span><strong>QB</strong> 调整“Token 逻辑上选哪个专家”；第二篇的 <strong>MoonEP</strong> 处理“这些专家实例物理上怎样复制、放置和执行”。一个是模型路由规则，一个是集群执行系统。</div>
</section>
<section class="searchable-section" id="vision">
<h2>08 · 原生视觉：进入同一条 Token 河流</h2>
<div class="flow"><div class="flow-row"><div class="node"><b>图像 / 视频</b>像素与帧</div><div class="arrow">→</div><div class="node"><b>MoonViT-V2</b>27 层 / ≈401M</div><div class="arrow">→</div><div class="node"><b>Pixel Shuffle</b>2×2，Token ÷4</div><div class="arrow">→</div><div class="node"><b>MLP Projector</b>映射到 LLM</div><div class="arrow">→</div><div class="node"><b>共享主干</b>与文本交错</div></div></div>
<p>MoonViT-V2 使用 Patch size 14。图像和视频共享参数；视频注意力分解为空间帧内与时间帧间两步，并做时间池化。投影前 2×2 Pixel Shuffle 把视觉 Token 数降到四分之一，使最高约 3584×3584 像素的输入在长上下文中仍可承受。</p>
<p>视觉塔从零开始，与语言模型一起以 next-token prediction 训练。论文报告它比 SigLIP 初始化的视觉塔梯度更平稳，视觉评测表现相当。</p>
<div class="callout derive"><span class="tag">准确表述</span>“原生多模态”不是没有独立视觉编码器，而是视觉塔、投影器与 LLM 从预训练之初联合优化，不依赖事后嫁接和单独对齐阶段。</div>
</section>
<section class="searchable-section" id="muon">
<h2>09 · Per-Head Muon：各注意力头独立校准步幅</h2>
<p>Muon 对矩阵参数的动量做 Newton–Schulz 正交化。若把完整 Q/K/V 投影作为一个大矩阵处理，梯度尺度大的头可能支配共同更新方向。K3 将动量沿 Head 维切开，每个头分别正交化。</p>
<div class="grid"><div class="card"><span class="tag">FULL MATRIX</span><h3>[Head 1 | Head 2 | …]</h3><p>所有头耦合成一个正交化问题；大尺度头影响更强。</p></div><div class="card"><span class="tag">PER HEAD</span><h3>H₁ ↻　H₂ ↻　H₃ ↻</h3><p>每头获得更均衡的更新尺度，长矩阵的小块运算也略便宜。</p></div></div>
<p>它主要改善大规模训练稳定性，不改变前向推理结构。</p>
</section>
<section class="searchable-section" id="summary">
<h2>10 · 把设计逻辑重新串起来</h2>
<div class="table-wrap"><table><thead><tr><th>瓶颈</th><th>结构</th><th>代价 / 补偿</th></tr></thead><tbody>
<tr><td>1M 序列太长</td><td>KDA 固定状态承担 69 层</td><td>压缩可能丢细节，因此周期性插入 24 层 MLA。</td></tr>
<tr><td>93 层残差压缩历史</td><td>Block AttnRes 跨块选择</td><td>以块为粒度，牺牲逐层精确选择换取内存与通信可控。</td></tr>
<tr><td>总容量要大、单 Token 成本要小</td><td>LatentMoE 896 选 16</td><td>引入路由与负载问题，用 QB、RMSNorm、SiTU-GLU 稳定。</td></tr>
<tr><td>视觉要参与长工具链</td><td>MoonViT-V2 原生联合训练</td><td>视觉编码带来额外计算，用 Token 压缩和流水线隐藏。</td></tr>
</tbody></table></div>
<div class="callout claim"><span class="tag">核心结论</span>Kimi K3 不是靠一个“神奇注意力”完成百万上下文。它用 KDA、MLA、AttnRes、LatentMoE 和视觉通路分别解决不同的信息瓶颈，再依赖第二篇中的长上下文训练与系统基础设施让组合成立。</div>
</section>
<section class="searchable-section" id="glossary">
<h2>11 · 概念速查与易错点</h2>
<div class="table-wrap"><table><thead><tr><th>概念</th><th>一句话定义</th></tr></thead><tbody><tr><td>Activated Parameters</td><td>单个 Token 实际经过的主要权重规模，不等于总参数。</td></tr><tr><td>Router</td><td>为 Token 计算专家分数并进行 Top-k 选择的模块。</td></tr><tr><td>Shared Expert</td><td>所有 Token 都执行、承载通用能力的专家。</td></tr><tr><td>Routed Expert</td><td>只被特定 Token 选中、形成专业化的专家。</td></tr><tr><td>LatentMoE</td><td>在低维潜空间中进行多专家计算，再升回模型宽度。</td></tr><tr><td>KDA State</td><td>固定形状的递归联想状态，不是逐 Token KV 列表。</td></tr><tr><td>MLA Cache</td><td>随序列增长的低维潜变量缓存，可重建 K/V 做全局注意力。</td></tr><tr><td>AttnRes</td><td>沿网络深度对历史层或块表示做选择性加权。</td></tr></tbody></table></div>
<div class="misconception"><div class="cross">×</div><div>“2.78T 模型每生成一个 Token 都运行 2.78T 参数。”</div><div class="check">✓</div><div>每 Token 激活约 104.2B 参数；2.78T 是全部专家等权重的总量。</div></div>
<div class="misconception"><div class="cross">×</div><div>“KDA 状态固定，所以能无损记住无限历史。”</div><div class="check">✓</div><div>固定状态意味着压缩；MLA 用于补回精确全局内容寻址。</div></div>
<div class="misconception"><div class="cross">×</div><div>“NoPE 表示模型不知道 Token 顺序。”</div><div class="check">✓</div><div>KDA 的因果递推、短卷积和衰减提供顺序信息；NoPE 指 MLA Q/K 无显式位置编码。</div></div>
<div class="misconception"><div class="cross">×</div><div>“Quantile Balancing 直接改专家输出权重。”</div><div class="check">✓</div><div>偏置只参与 Top-k 派发；归一化混合权重使用未加偏置的 Router 分数。</div></div>
</section>
<footer class="footer">
<p><strong>主要来源：</strong><a href="https://arxiv.org/html/2607.24653v2">Kimi K3: Open Frontier Intelligence（HTML）</a> · <a href="https://arxiv.org/pdf/2607.24653v2">PDF</a> · <a href="https://huggingface.co/moonshotai/Kimi-K3">模型权重</a>。</p>
<p>本文数字与公式以论文 v2（2026-08-07）为准。教学例子用于理解，不代表真实激活、路由结果或权重内容。</p>
</footer>

</div>


## 阅读架构与阅读实现的边界

公式描述的是模型允许怎样处理信息，真实实现还需要决定张量布局、精度、分块和通信。这里保留论文的结构与教学例子，但不从总参数或激活参数推算某台机器的实际吞吐。专家虽然稀疏激活，全部权重仍需有存储位置；递归状态虽然不随序列线性增长，仍要为每个活跃请求保存。把容量与计算分开，才能把这篇结构笔记连接到下一篇系统笔记。

此外，路由偏置控制专家选择，混合权重控制已选专家的输出组合，物理调度则决定专家在哪里执行。三者可以共同影响负载，却分别属于模型定义、数值计算和运行系统。学习时应分别标注，避免把一种负载均衡机制误认成另一种实现。
