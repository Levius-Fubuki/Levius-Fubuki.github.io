# Kimi K3 系统笔记：长上下文训练、专家调度与混合缓存

本文由既有深读笔记整理入博客，并核对了对应 Codex 学习工作区。内容区分论文事实、教学推导与实验边界；论文中的性能与规模数据属于作者报告，不是我在本地复现的成绩。原笔记的公式、示例和推导在此保留。

<div class="study-note">


<p class="lead">模型公式只是起点。要让 3T 级 MoE 在百万上下文上完成数百次工具调用，还需要长上下文课程、可恢复 RL、五种并行方式、动态专家复制、分层缓存和部署感知后训练。</p>

<section class="searchable-section" id="lifecycle">
<h2>01 · 一条完整生命周期</h2>
<div class="flow"><div class="flow-row"><div class="node"><b>数据</b>文本 + 视觉</div><div class="arrow">→</div><div class="node"><b>预训练</b>8K / 64K</div><div class="arrow">→</div><div class="node"><b>Cooldown</b>256K / 1M</div><div class="arrow">→</div><div class="node"><b>SFT</b>Agent 冷启动</div><div class="arrow">→</div><div class="node"><b>9 个 RL 教师</b>3 领域 × 3 effort</div><div class="arrow">→</div><div class="node"><b>MOPD</b>统一模型</div><div class="arrow">→</div><div class="node"><b>部署</b>QAT + Draft + Cache</div></div></div>
<p>架构篇解决“信息怎样在模型内部流动”；本篇解决三个现实问题：</p>
<div class="triple"><div class="card"><span class="tag">TRAIN</span><h3>怎样稳定训练</h3><p>参数、激活和专家通信不能让 GPU 内存或慢 Rank 拖垮训练。</p></div><div class="card"><span class="tag">ALIGN</span><h3>怎样学会工作</h3><p>模型要在工具环境中行动、验证、失败恢复，而不是只生成一次答案。</p></div><div class="card"><span class="tag">SERVE</span><h3>怎样经济上线</h3><p>百万前缀必须可复用，FP4 专家要保持能力，长请求不能饿死短请求。</p></div></div>
</section>
<section class="searchable-section" id="pretraining">
<h2>02 · 预训练：先把基本世界模型做强</h2>
<h3>五个基础词</h3>
<div class="table-wrap"><table><thead><tr><th>词</th><th>初学者解释</th></tr></thead><tbody><tr><td>Next-token prediction</td><td>给定此前序列，预测下一个 Token。视觉 Token 与文本 Token 也进入同一目标。</td></tr><tr><td>Batch</td><td>一次参与前向/反向的样本或 Token 集合；扩大 Batch 会改变梯度统计和通信。</td></tr><tr><td>Learning rate</td><td>参数每次更新的步幅。过大易发散，过小训练慢。</td></tr><tr><td>Warmup</td><td>训练开头逐步升高学习率，避免初始梯度不稳定。</td></tr><tr><td>Cooldown</td><td>训练末段降低学习率，并可集中加入高质量、长上下文数据以精修能力。</td></tr></tbody></table></div>
<p>语料覆盖 Web、代码、数学、知识以及图文、OCR、视频、视觉编程等。SVG、网页、游戏、CAD、3D 资产等“代码—渲染结果”对，让模型把代码、截图和修改结果放在同一因果链上。</p>
<div class="callout claim"><span class="tag">论文披露</span>语言与视觉从训练开始联合优化；优化器为 Per-Head Muon，使用权重裁剪、Quantile Balancing、1% 线性 Warmup、Cosine 学习率衰减和 0.1 Weight Decay。</div>
</section>
<section class="searchable-section" id="long-context">
<h2>03 · 百万上下文不是把位置上限改大</h2>
<div class="flow"><div class="timeline"><div class="phase"><b>8K</b>预训练起点</div><div class="phase"><b>64K</b>预训练后段</div><div class="phase"><b>256K</b>Cooldown</div><div class="phase"><b>1M</b>Cooldown 最终阶段</div></div></div>
<p>NoPE 避免 RoPE 外推和频率重标定，但不会自动让模型学会跨百万 Token 找证据。真正的长能力还需要：</p>
<div class="steps"><div class="step"><strong>清理自然长数据：</strong>去重、帧感知哈希、质量筛选和结构验证。</div><div class="step"><strong>提高长样本比例：</strong>避免大量短文本淹没长上下文分布。</div><div class="step"><strong>合成长依赖：</strong>把互相关联的信息散布到序列远端，使局部阅读无法解题。</div><div class="step"><strong>逐级扩窗：</strong>只在训练的一小部分使用昂贵 256K/1M 计算。</div></div>
<div class="callout derive"><span class="tag">教学例子</span>假设任务要求核对一家公司：定义在 20K，审计数据在 410K，例外条款在 930K。答案必须同时引用三处。简单把三篇无关文档拼到 1M，只会训练“长输入中的局部回答”，不会迫使模型跨长距离组合。</div>
<h3>2.5× 到底是什么意思</h3>
<p>论文在持出 OOD 验证数据上拟合 Scaling Law，称架构、数据和训练配方整体相较 Kimi K2 获得约 <strong>2.5× scaling efficiency</strong>。它不是“所有基准分数乘 2.5”，也不是“KDA 单独提升 2.5 倍”。论文未提供足以拆出每项贡献的大规模逐模块消融。</p>
</section>
<section class="searchable-section" id="agent-rl">
<h2>04 · Agent RL：训练的是闭环，不是长作文</h2>
<div class="table-wrap"><table><thead><tr><th>概念</th><th>含义</th></tr></thead><tbody><tr><td>Policy</td><td>当前模型在观察到上下文后选择下一个 Token、工具调用或行动的概率分布。</td></tr><tr><td>Rollout</td><td>让 Policy 在环境中实际运行一次，直到完成、失败或预算耗尽。</td></tr><tr><td>Trajectory</td><td>整个观察—行动—结果序列。</td></tr><tr><td>Reward</td><td>验证器或奖励模型对轨迹结果给出的训练信号。</td></tr><tr><td>On-policy</td><td>数据由当前正在训练的 Policy 生成。</td></tr><tr><td>Off-policy / stale</td><td>生成数据的旧 Policy 与当前更新后的模型已有差异。</td></tr></tbody></table></div>
<h3>一个研究 Agent 的轨迹</h3>
<div class="flow"><div class="flow-row"><div class="node"><b>目标</b>核对一项技术主张</div><div class="arrow">→</div><div class="node"><b>搜索</b>找到候选来源</div><div class="arrow">→</div><div class="node"><b>阅读</b>提取证据</div><div class="arrow">→</div><div class="node"><b>验证失败</b>日期不匹配</div><div class="arrow">→</div><div class="node"><b>调整</b>追加限定查询</div><div class="arrow">→</div><div class="node"><b>最终状态</b>证据可核验</div></div></div>
<p>K3 的 RL 覆盖三大领域：</p>
<div class="triple"><div class="card"><h3>General</h3><p>知识、推理、视觉、搜索、可信度与知识工作。</p></div><div class="card"><h3>General Agents</h3><p>长时助理、深度研究与长文写作。</p></div><div class="card"><h3>Coding Agents</h3><p>SWE、Kernel、编码体验与 Web 开发。</p></div></div>
<p>每个领域分别训练 low、high、max 三种推理强度，共 <strong>3×3=9 个教师策略</strong>。</p>
<h3>Reasoning Effort 是怎样学出来的</h3>
<div class="equation">基础模型估计问题预算 b₀(x)
若轨迹总预算 T(y) &gt; τ · b₀(x)，则任务奖励覆盖为 −1</div>
<p>先用较大的 τ 训练 max，再逐渐减小得到 high 与 low。通用任务统计思考 Token；Agent 任务统计推理和工具参数在内的累计输出 Token。它在训练“给定预算下完成任务”，而不只是要求回复变短。</p>
<h3>Agentic GRM</h3><p>不可直接验证的任务由生成式奖励模型进行组内二元比较，强制遵循：读结果 → 生成 Rubric → 逐候选评分 → 写入 Scorepad。若输出超过冷启动长度估计的 σ 倍，则自动输掉比较，以限制通过冗长骗取奖励。</p>
<h3>Partial Rollout：不等最慢轨迹</h3>
<div class="flow"><div class="flow-row"><div class="node"><b>A 完成</b>进入优化</div><div class="node"><b>B 完成</b>进入优化</div><div class="node"><b>C 暂停</b>保留模型+环境状态</div><div class="node"><b>D 继续</b>下一迭代优先恢复</div></div></div>
<p>当比例 λ 的轨迹完成，系统就开始 Policy 优化，不被长尾任务拖住。未完成轨迹暂停并进入下一轮，因此一条轨迹可能跨越多次模型更新，变成极端陈旧的 off-policy 数据。</p>
<div class="callout warn"><span class="tag">论文披露边界</span>报告说明其 Policy 优化用逐 Token 正则把更新限制在局部邻域，从而容忍陈旧轨迹，但本节没有给出完整优化公式。这里不补写未披露算法细节。</div>
</section>
<section class="searchable-section" id="mopd">
<h2>05 · MOPD：把九位老师合进一个学生</h2>
<p>部署九个独立模型既昂贵，也会割裂跨域能力。Multi-Teacher On-Policy Distillation（<strong>MOPD</strong>）按样本的领域 d 和 effort e 选择对应教师，对学生自己生成的 Token 给密集奖励：</p>
<div class="equation">r_opdᵈ(yₜ | e,x,y&lt;t)
= clip( stopgrad[ log( π_teacherᵈ,ᵉ(yₜ) / π_student(yₜ) ) ], −Rmax, Rmax )</div>
<div class="steps"><div class="step">学生在“低努力代码补全”样本上生成 Token，由 low/coding 教师给概率比信号。</div><div class="step">学生在“最大努力长助手”样本上运行时，切换 max/general-agent 教师。</div><div class="step">教师更偏好的 Token 获得正信号；极端概率比通过 Clip 限制。</div><div class="step">stop-gradient 表示概率比作为奖励，不沿教师或该奖励计算图反向传播。</div></div>
<p>最终得到一个由 effort 指令条件化的统一模型，而不是九个在线模型。</p>
</section>
<section class="searchable-section" id="parallelism">
<h2>06 · 五种并行：到底切的是什么</h2>
<div class="table-wrap"><table><thead><tr><th>方式</th><th>切分对象</th><th>典型通信</th><th>K3 中的意义</th></tr></thead><tbody><tr><td>DP</td><td>同一模型处理不同数据</td><td>梯度 All-Reduce / Reduce-Scatter</td><td>扩大总 Batch；结合 ZeRO 分片状态。</td></tr><tr><td>TP</td><td>一层中的矩阵、Head 或通道</td><td>层内 All-Reduce / All-Gather</td><td>单层权重放不下一张卡时横向切分。</td></tr><tr><td>PP</td><td>不同层</td><td>阶段间激活 P2P</td><td>93 层按流水线阶段分配；用 Virtual Pipeline 减少气泡。</td></tr><tr><td>EP</td><td>不同 MoE 专家</td><td>Token All-to-All</td><td>896 专家分散到 Rank；MoonEP 管理热点。</td></tr><tr><td>CP</td><td>同一序列的不同位置</td><td>上下文状态或 K/V 交换</td><td>把百万 Token 沿长度切开；KCP 处理 KDA 递归。</td></tr></tbody></table></div>
<h3>GPU 执行层级</h3>
<p>软件执行实例按 grid → CTA / block → warp → thread 组织。SM 是承载 CTA 的硬件执行资源，Tensor Core 是 SM 内的矩阵运算单元；Tensor Core 不是 warp 的子级。</p>
<div class="callout derive"><span class="tag">基础概念 · 非论文启动参数</span>CTA≈CUDA Thread Block，是软件协作单元；SM 是执行 CTA 的硬件。Tile 是大矩阵切出的计算块。K3 论文描述 Tensor Core 与 SM 级规划，但没有在这些段落公布某个统一的 CTA/warp 配置，不能从概念图反推真实 Kernel Launch 参数。</div>
</section>
<section class="searchable-section" id="kcp">
<h2>07 · FlashKDA 与 KDA Context Parallelism</h2>
<h3>FlashKDA：在单设备内部隐藏递归等待</h3>
<p>KDA Chunk 内是 Token 并行矩阵计算，Chunk 间却必须传播状态。FlashKDA 将 Token-parallel 的块内阶段与 Head-parallel 的状态递推重叠，避免两者交替时 SM 闲置。超长 Prefill 还会在单 Rank 内按序列段分配 SM，先独立计算段变换，再精确合并入口状态。</p>
<h3>为什么普通线性注意力的“状态求和”不够</h3>
<p>KDA 每个序列段对入口状态做的不是简单加法，而是仿射变换：</p>
<div class="equation">S_out = M · S_in + L
M：该段全部 Token 对旧状态的累计变换
L：从 S=0 开始时，该段自己产生的状态</div>
<p>三个设备各处理一段：</p>
<div class="equation">S₁ = M₁S₀ + L₁
S₂ = M₂S₁ + L₂ = M₂M₁S₀ + M₂L₁ + L₂
S₃ = M₃S₂ + L₃</div>
<p>直接做 <code>L₁+L₂+L₃</code> 会漏掉后续段对前段状态的 M 变换。好消息是仿射对可以结合：</p>
<div class="equation">(M₂,L₂) ∘ (M₁,L₁) = (M₂M₁, M₂L₁ + L₂)</div>
<p>因此各 Rank 可以先独立算本地 <code>(Mᵢ,Lᵢ)</code>，一次 All-Gather 后用 Prefix Scan 按文档顺序组合，恢复各段准确入口状态。通信的是固定大小片段，而不是随 1M 长度增长的全部 KV 块。这就是 <strong>KDA Context Parallelism</strong>（KCP）的关键。</p>
</section>
<section class="searchable-section" id="moonep">
<h2>08 · MoonEP：让逻辑专家映射到均衡硬件执行</h2>
<div class="grid"><div class="card"><span class="tag">MODEL / QB</span><h3>Quantile Balancing</h3><p>跨 Batch 调 Router 偏置，改善 Token 选择专家的统计分布。回答“逻辑上选谁”。</p></div><div class="card"><span class="tag">SYSTEM / MOONEP</span><h3>MoonEP</h3><p>根据当前微批实际路由动态复制热点专家、迁移执行并归并梯度。回答“物理上在哪算”。</p></div></div>
<p>即使长期平均平衡，某层某个微批仍可能突然集中到少数专家。传统 EP 中，持有热点专家的 Rank 最慢，全体同步等待，动态 Token 数还会造成碎片化。</p>
<div class="steps"><div class="step"><strong>在线规划：</strong>读取当前微批、当前层 Router 输出。</div><div class="step"><strong>有限冗余：</strong>选出需要复制的热点专家，预取权重到有余力的 Rank。</div><div class="step"><strong>静态 Shape 执行：</strong>重排 Token，使各 Rank 获得完全平衡工作量，减少同步和碎片。</div><div class="step"><strong>反向归并：</strong>副本的梯度先写本地 Reduce Buffer，结束后还原到 Home Rank。</div></div>
<p>MoonEP 还将专家派发/合并做零拷贝通信，并让通信、Shared Expert 与 Group GEMM 相互重叠。</p>
<h3>内存不是一个池子</h3>
<div class="memory"><div class="mem reg">寄存器：线程私有，最快，容量最小</div><div class="mem smem">Shared Memory：CTA 内共享</div><div class="mem hbm">GPU HBM：权重、激活、KV/状态</div><div class="mem dram">CPU DRAM：梯度分片、外部 Cache</div><div class="mem nvme">NVMe：训练/推理阶段切换时的更慢 Offload</div></div>
<div class="callout derive"><span class="tag">来自侧边学习内容的基础补充</span>一个 Kernel 同时保持的中间变量越多，线程寄存器压力越高，可能降低一个 SM 可驻留的 Warp 数或发生 Spill。这解释了为什么融合和重计算不能只看 FLOPs，还要看变量生命周期。但论文没有给出所有 K3 Kernel 的寄存器数，本笔记不把该一般规律写成具体测量结果。</div>
<p>训练侧还使用统一激活管理、Checkpoint/Recompute、PP Rank 间远程平衡激活、Pipeline ZeRO-2 梯度分片与 CPU Offload，以及只为本 Rank 所属参数做 P2P Muon Gather，避免完整参数缓冲。</p>
</section>
<section class="searchable-section" id="agentenv">
<h2>09 · AgentENV：模型状态和世界状态一起恢复</h2>
<p>Partial Rollout 只保存模型 KV/KDA 状态还不够。Agent 可能已经改了文件、启动服务、写入数据库；恢复时外部世界也必须回到同一状态。</p>
<div class="grid"><div class="card"><h3>Pause / Resume</h3><p>等待模型推理时暂停 microVM，不占 CPU 与内存；之后从相同环境继续。</p></div><div class="card"><h3>Fork</h3><p>从精确状态分叉一个副本供奖励判断，不污染原环境。</p></div><div class="card"><h3>Snapshot</h3><p>定期保存状态，长任务失败后从中间恢复。</p></div><div class="card"><h3>Isolation</h3><p>基于 Firecracker microVM，允许更真实的系统操作，同时比普通容器隔离更强。</p></div></div>
<p>增量 Checkpoint 只保存上次之后变脏的内存页。论文报告最低 Checkpoint/Resume 延迟约 <strong>133ms / 49ms</strong>，真实工作负载内存超售最高 6.5×；训练和评估共创建 51,219,741 个沙箱、覆盖 1,505,678 个镜像。这些是论文报告值，不代表任意部署环境都能复现。</p>
</section>
<section class="searchable-section" id="serving-cache">
<h2>10 · 混合缓存：KDA State 与 MLA KV 必须对齐</h2>
<div class="grid"><div class="card"><h3>KDA Cache</h3><p>每个请求只有一份固定大小递归状态，会在 Decode 时原地更新。</p></div><div class="card"><h3>MLA KV Cache</h3><p>按 Token 增长、分页存储的潜变量缓存。</p></div></div>
<p>共享前缀只有在同一 Token 边界上同时恢复二者才有效。只有 MLA 命中但 KDA 没有对应快照，后续 KDA 状态就不代表该前缀。</p>
<h3>论文中的 6144 / 512 / 2560 例子</h3>
<div class="equation">1 个物理块 = 6144 Token
内部包含 12 个哈希块，每个 512 Token
请求前 2800 Token 与缓存一致
可复用边界 B = 2560 = 5 × 512</div>
<p>物理块保持大粒度，减少分配和 KDA 大状态管理成本；哈希块保持细粒度，使部分填充物理块也可命中。只有某哈希端点同时存在所有 KDA Cache Group 的状态快照时才可复用。</p>
<div class="steps"><div class="step">MLA 用链式哈希确认 [0,2560) 完整一致。</div><div class="step">KDA 在 2560 边界存在持久快照。</div><div class="step">将只读共享快照复制成请求私有运行状态，避免并发修改。</div><div class="step">从 2560 继续 Prefill，而不是重算 [0,2560)。</div></div>
<h3>RL 外部状态池</h3><p>活跃 Decode Block 留在 GPU；被逐出的可复用前缀才 Write-back 到 CPU DRAM，复用前 Prefetch。KDA State 与对应 MLA Block 同步迁移。Rollout 结束后释放外部池；训练权重和优化器状态可暂存 NVMe，让 DRAM 腾给缓存。</p>
<h3>调度</h3><p>Auto-throttling 根据活跃请求、排队数和 KV 利用率动态降并发。生产侧通过缓存亲和把会话路由到持有前缀的主集群，并预分配第二集群用于故障接管；短请求与超长请求分配独立资源预算，避免 1M 请求突发拖垮全局 TTFT。</p>
</section>
<section class="searchable-section" id="quantization">
<h2>11 · 部署感知量化：训练时就接受低精度</h2>
<div class="table-wrap"><table><thead><tr><th>组件</th><th>精度</th><th>原因</th></tr></thead><tbody><tr><td>MoE 路由专家权重</td><td><strong>MXFP4</strong></td><td>专家权重占总参数内存主体，压缩收益最大。</td></tr><tr><td>专家输入激活</td><td><strong>MXFP8</strong></td><td>匹配低精度专家计算，同时保留比 FP4 更大动态范围。</td></tr><tr><td>注意力、潜投影、共享专家、Router</td><td>更高精度</td><td>规模相对小或对路由/状态更敏感。</td></tr></tbody></table></div>
<p>QAT 从 SFT 开始贯穿 RL；Rollout 与训练使用相同量化方案，避免“训练看到高精度 Policy，采样却来自低精度部署模型”的不匹配。</p>
</section>
<section class="searchable-section" id="speculative-decoding">
<h2>12 · MTP / EAGLE-3：先草拟，再由大模型验收</h2>
<h3>先理解 Target 与 Draft</h3>
<div class="flow"><div class="flow-row"><div class="node"><b>Draft</b>快速猜多个 Token</div><div class="arrow">→</div><div class="node"><b>Target</b>K3 一次并行验证</div><div class="arrow">→</div><div class="node"><b>接受前缀</b>分布一致则连续提交</div><div class="arrow">→</div><div class="node"><b>拒绝处</b>由 Target 纠正</div></div></div>
<p>Target 就是决定最终输出分布的主模型；Draft 只加速候选生成。正确的无损投机采样不会把 Draft 的近似分布当最终答案。</p>
<p>K3 预训练含一个结构类似主干 Block 的 MTP 层。后训练把它改造成 EAGLE-3 单层 Draft：Target 冻结，只更新 Draft 和特征融合投影。输入融合第 1、4 和最终 AttnRes Block 的低、中、高层特征；训练中展开 7 步。</p>
<h3>为什么不只最小化 KL</h3>
<p>Draft 容量有限，KL 小不保证投机采样接受率最高。论文直接优化：</p>
<div class="equation">acceptance = Σx∈V min(p(x), q(x))
L_LK = −log acceptance</div>
<p><code>p</code> 是 Target 的下一个 Token 分布，<code>q</code> 是 Draft 分布。该损失直接推动两者可共同接受的概率质量增大。</p>
<div class="callout warn"><span class="tag">KDA 特有难点</span>Draft Token 若被拒绝，KDA 状态可能已经原地前进，不能像丢弃几块 KV 那样简单回滚。因此线上 Kernel 需要为推测状态与已确认状态设计专门管理路径。</div>
</section>
<section class="searchable-section" id="evaluation">
<h2>13 · 实验怎么读，论文哪里没回答</h2>
<div class="table-wrap"><table><thead><tr><th>方向</th><th>代表结果</th><th>说明</th></tr></thead><tbody><tr><td>推理</td><td>GPQA Diamond 93.5</td><td>K3 max effort；并非所有推理集都第一。</td></tr><tr><td>编码</td><td>Terminal-Bench 2.1：88.3</td><td>报告各模型多个 Harness 中最佳结果。</td></tr><tr><td>Agent</td><td>BrowseComp 91.2；MCPMark 94.5</td><td>上下文管理和工具配置会显著影响成绩。</td></tr><tr><td>视觉</td><td>OmniDocBench 91.1；Video-MME 90.0</td><td>部分视觉集另报告 Python 工具增强结果。</td></tr></tbody></table></div>
<p>横向比较要同时检查：</p>
<ul><li>reasoning effort 是否都是 max；</li><li>使用 Kimi Code、Claude Code、Codex 还是其他 Harness；</li><li>是否允许搜索、Python 或上下文压缩；</li><li>对手是否触发 fallback、拒答或安全拦截；</li><li>结果来自公开榜单还是作者内部基准。</li></ul>
<div class="callout warn"><span class="tag">局限</span>论文没有披露完整预训练 Token 总量，也没有提供足够的 2.8T 规模逐组件消融来分解 KDA、AttnRes、Stable LatentMoE、数据和训练配方各自贡献。2.5× 是组合系统的 Scaling Law 声明，不能用于单模块归因。</div>
</section>
<section class="searchable-section" id="glossary">
<h2>14 · 瓶颈—方案速查</h2>
<div class="table-wrap"><table><thead><tr><th>瓶颈</th><th>K3 方案</th><th>你应记住的因果关系</th></tr></thead><tbody><tr><td>百万序列训练</td><td>课程学习 + 合成长依赖 + KCP</td><td>可表示长度、学会用长度、算得动长度是三件事。</td></tr><tr><td>长轨迹拖尾</td><td>Partial Rollout + AgentENV</td><td>暂停模型轨迹必须同时暂停外部世界状态。</td></tr><tr><td>九个专门 Policy</td><td>MOPD</td><td>按领域与 effort 选教师，用逐 Token 信号合回统一学生。</td></tr><tr><td>专家 Rank 不均衡</td><td>QB + MoonEP</td><td>QB 调逻辑路由；MoonEP 调物理副本和执行。</td></tr><tr><td>混合注意力前缀</td><td>KDA-aware cache</td><td>MLA KV 与 KDA State 必须在相同边界共同命中。</td></tr><tr><td>2.78T 权重成本</td><td>MXFP4 QAT</td><td>只压最占内存的 routed experts，敏感模块保留高精度。</td></tr><tr><td>自回归串行</td><td>MTP → EAGLE-3</td><td>Draft 猜、Target 验；接受率而非 Draft 单独准确率决定加速。</td></tr></tbody></table></div>
<div class="callout claim"><span class="tag">最终心智模型</span>K3 的端到端路线是：模型结构先把长序列和大专家库变得可能；训练数据与 RL 教会它使用这些容量；KCP、MoonEP、AgentENV 和混合缓存再把训练与服务成本压进现实边界。</div>
</section>
<footer class="footer"><p><strong>主要来源：</strong><a href="https://arxiv.org/html/2607.24653v2">Kimi K3: Open Frontier Intelligence（HTML）</a> · <a href="https://arxiv.org/pdf/2607.24653v2">PDF</a> · <a href="https://github.com/MoonshotAI/MoonEP">MoonEP</a> · <a href="https://github.com/kvcache-ai/AgentENV">AgentENV</a> · <a href="https://github.com/fla-org/flash-linear-attention">Flash Linear Attention</a>。</p><p>论文事实、本文推导和教学示例已分别标记；侧边聊天只用于补充 GPU/下一 Token 等基础概念，没有被当作 K3 实现证据。</p></footer>

</div>
