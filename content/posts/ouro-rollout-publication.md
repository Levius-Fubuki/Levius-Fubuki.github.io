# 把循环模型接入 RL 框架：Ouro 的轨迹重放、权重发布与恢复

一个推理引擎能够返回 token，离强化学习训练可用还有多远？这次把 Ouro-1.4B 接入 verl 时，真正花时间的地方是把生成、反向传播、权重更新和恢复连成同一条可检查的数据流。循环模型让问题更具体：同一个 token 可以执行不同轮数，历史 K/V 还可能来自较浅的退出位置。只保存最终文本和一组概率，无法完整描述生成它的策略。

项目最后取得了两组范围明确的结果：双卡上完成实际 BF16/FSDP 更新、权重发布与新进程恢复；另一轮单卡实验接通真实 Ray 客户端和单轮 AgentLoop，验证取消、发布失败与 actor 重启。学习得到的 N+2 退出头也进入了在线参考路径，但没有战胜固定三轮或原始退出门。完整的分布式 Ray 训练闭环仍未完成。

以下依据 2026 年 10 月 28、29 日归档的代码与实验记录。工程实现位于本人的 [verl fork 草稿 PR #1](https://github.com/Levius-Fubuki/verl/pull/1)，本文固定到 `fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d`；没有提交上游 verl PR，也不把自动化实验表述成人工逐行审阅或独立人工测试。基础物理层复用 vllm-rlt、ByteDance/Ouro 与相关 Hugging Face 实现，开发和实验使用了 Codex 辅助。本次博客整理只读核对材料，没有连接服务器或重跑 GPU。

## 一、先固定模型语义，再接训练框架

普通前馈模型的层数通常写在结构里；循环模型还需要描述一次输入实际走过几轮。这里保留一套物理上的 24 层核心，最多复用四轮。四轮计算不意味着构造四份独立参数，反向传播也不能把它改写成四个互不相关的模型。

我把接口拆为三类对象：训练模型负责可微重放，rollout 负责真实生成及轨迹，publisher 负责把更新后的物理参数交给推理引擎。训练侧使用实际 verl 的 `DataProto`、FSDP engine、损失和 checkpoint 接口，先用直接 torchrun 配方验证这些连接，再另做 Ray 服务入口。这样，训练数学和 actor 调度可以分别定位。

核心数据流是：

```text
真实生成：token、退出深度、有效采样参数、raw score、权重版本
    → 对齐 response 目标与实际输入深度
    → 依原轨迹重放，计算 old-policy 概率与损失
    → FSDP 梯度更新
    → 按物理单元导出完整参数
    → 完整发布并确认真实版本
    → 下一轮生成，或保存恢复所需状态
```

这条链有一个很容易忽略的前提：训练端重放的是 rollout 已经选择的离散轨迹，而不是重新运行退出门，再期待碰巧得到同样的路径。退出门在这份训练实现中冻结；它没有因为接通 PPO/GRPO 就自动成为可训练的 N+2 策略。

### 输出深度必须移到正确的输入位置

自回归模型在当前位置计算的 logits 预测下一个 token。假设 prompt 有三个 token，四个输出的执行深度记录为 `[4, 2, 3, 4]`，输入深度对应为：

```text
输入位置：p0 p1 p2 y0 y1 y2 y3
执行深度： 4  4  4  2  3  4  1
预测目标：p1 p2 y0 y1 y2 y3  —
```

第一个回答由最后一个 prompt 输入产生，因此它必须符合全深度 prefill；`y0` 输入执行两轮，产生 `y1`。最后一个输入没有回答目标，必须被 loss mask 排除，示例中的深度 1 不能被解释为又测到一次有效浅层生成。

实现把这次移位集中在 `input_execution_depths()`，拒绝非法轮数和不符合全深度 prefill 的首输出，而不在后面的 loss converter 中再次“修正一位”。实际响应长度也必须进入 padding 和 mask：短 EOS 请求不能沿用固定长响应的位置假设。[轨迹对齐源码](https://github.com/Levius-Fubuki/verl/blob/fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d/examples/ouro/model.py)

## 二、退出以后，历史 K/V 仍然参与后面的计算

last-exited K/V 的含义是：某个历史输入在第 d 轮退出后，更深轮的注意力继续使用它最后执行那一轮的 K/V。不能给它补零，不能换成另一条全深度历史，也不能只把最终 hidden state 接到语言模型头，就声称重放了原轨迹。

训练实现逐轮维护 K/V。对本轮仍活跃的位置使用新投影，对已经退出的位置使用上一轮保留下来的值，概念上相当于：

```python
k = torch.where(active_mask, current_k, previous_k)
v = torch.where(active_mask, current_v, previous_v)
```

这里保留的是可微计算关系。若把 `previous_k/v` detach，或者照搬推理缓存的原地写入方式，前向输出可能看似合理，梯度却不再覆盖被后续注意力复用的历史。相同输出和正确训练梯度，需要各自的对照。

所有 rank 还要执行相同的物理层、循环调用顺序。不能因本 rank 的若干输入提前退出，就自行少调用一个 FSDP 单元，让其他 rank 卡在参数集合通信里。当前实现用掩码保留语义，dense 重放仍会计算部分不活跃行，因此这里没有早退训练加速的结论。

packed 输入采用先还原为独立 padded 行、执行后再展平的桥接。它有利于先验证轨迹和目标对齐，但不等于已经实现高效 varlen attention。sequence parallel 和 fused loss 也没有在这份接口中悄悄接管不支持的路径。[实现与适用范围](https://github.com/Levius-Fubuki/verl/blob/fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d/examples/ouro/README.md)

## 三、raw score、训练步和发布版本分别保存

两个看起来差不多的推理接口，可能使用不同的概率字段和配置位置。这次保留了显式 profile：

| profile | 固定引擎与概率约定 | 验证范围 |
| --- | --- | --- |
| `rollout_outputs` | #83 `c7abc6d`，`logprobs`、引擎级 raw 模式 | 独立 rollout 验证 |
| `publication_outputs` | #78 `e6fae1e`，`log_probs`、请求级 raw 模式 | 本文当前训练与发布实验 |
| `prototype` | 历史 `bdce287` | 明确选择后的历史复现 |

适配器没有看到一个字段缺失就尝试另一套 API，也没有把 #78 和 #83 在本地合成一个未经上游确认的运行时。缺少权重发布能力时，在启动训练 worker 或初始化 CUDA/FSDP 前就拒绝。能够生成，并不能补足训练所需的发布契约。

raw rollout score、重算的 old-policy score 和经过采样处理的分布也要区分。当前训练配方约束 `temperature=1, top_p=1, top_k=-1`，不能把 greedy 或过滤后的行为概率直接塞进相同的 PPO 比率。每条输出保存实际生效的采样参数，而不只保存客户端最初请求的默认值。

训练 `global_step` 与 `weight_version` 则是两条时间线。一次训练更新可以经历发布失败和完整重试；一次 checkpoint 恢复也需要重新发布。因此，`weight_version - 1` 无法可靠还原优化器步数。生成结果必须同时携带两者，消费端才能判断它到底来自哪个训练状态。[profile 与迁移记录](https://github.com/Levius-Fubuki/verl/blob/fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d/examples/ouro/MIGRATION.md)

## 四、完整权重发布是一段有失败状态的协议

权重发布不是循环调用几次 `copy_`。调用方需要证明参数完整、名称唯一、shape 正确，接收方需要确认实际提交的版本；中途失败时，生成入口必须保持关闭。

publisher 先选择一个严格更大的尝试版本，其计算关系是：

```python
version = max(global_step + 1,
              last_attempted_version + 1,
              backend_version + 1)
```

在开始调用前就登记该版本，因为失败可能发生在某个 P/D 对端已经收到开始消息以后。随后检查物理参数名称、完整 manifest 与 tensor 类型，传输全部 bucket，调用 finish，再读回 backend 的真实 committed version。只有读回值与本次版本一致，调用方才更新自己的 committed 状态。

FSDP1 导出按物理单元顺序 gather。所有 rank 以相同顺序参与，只由 rank 0 发布；每个 yielded BF16 tensor 拥有自己的存储，避免 gather 临时缓冲释放后，publisher bucket 仍引用无效内容。共享核心参数按对象身份去重，不能按循环轮次重复导出。

bucket 上限是软限制：单个超大参数单独发送，生产者还可能同时持有当前 gather 单元与一个前瞻 tensor。因此“没有全模型 state_dict 快照”不等于“显存峰值严格只有一个 bucket”。这条导出路径要求 `use_orig_params=True`，没有验收 FSDP2、LoRA 或手动参数 offload。[发布实现](https://github.com/Levius-Fubuki/verl/blob/fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d/examples/ouro/publication.py)、[物理单元导出](https://github.com/Levius-Fubuki/verl/blob/fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d/examples/ouro/worker.py)

在单卡 Ray 验证中，首次成功提交版本 1；故意输入重复参数名使版本 2 失败，入口持续关闭；完整重试提交版本 3。新 actor 恢复计数以后仍然不能生成，必须重新完整发布，得到版本 4。恢复发布计数不等于恢复模型权重，`resume_generation()` 也不能绕过失败或 draining 状态。

双设备 P/D 还需要更谨慎：某一端已经提交后，不应假装全局回滚已经完成。当前协议要求保持准入屏障，再走新版本的完整流。多个副本之间的生产级一致确认和分布式容错，仍在本次验收之外。

## 五、用新进程的下一步证明恢复，而不是只看文件存在

10 月 28 日实验使用两张报告为 RTX 4090、各 49,140 MiB 的设备，SM89、PCIe、无 CUDA peer access；运行 Ouro-1.4B，使用继承镜像 torch 的环境。这个显存配置不能直接简称为“普通双 24GB 4090”。环境实际可执行，但依赖元数据仍存在已披露的不兼容项，不是一个从零干净求解的安装锁文件。

固定四个算术 prompt、每个两个样本、响应上限 32、microbatch 1，分别完成三步全深度和三步自适应 BF16/FSDP 更新。微型奖励用于让梯度与优化器链路实际工作，不用于证明模型收敛或通用算术能力。

| 观察 | 已归档结果 | 解释范围 |
| --- | --- | --- |
| 全深度／自适应训练 | 各三次实际更新及发布 | 小规模同步配方跑通 |
| 最大 selected-score 差异 | 0.202963／0.233159 | 不满足严格 1e-3 等价 |
| rank 0 allocated 峰值 | 约 16.161 GiB | 不包含独立 rollout worker |
| 新进程恢复后的下一步 | 八项记录逐项相同 | 同 world size、兼容配置 |
| 恢复后的发布版本 | 3 → 4 | 重新发布有意消耗新版本 |

恢复比较的是第 3 步的 token、深度、seed、文本、reward、loss、梯度范数和分片状态哈希，不只比较“load 没报错”。这能够捕获 sampler cursor、RNG 或优化器恢复不完整的问题，但不覆盖改变 world size 后的重分片。

数值验收也没有被“训练完成”覆盖。固定的 pre-TIS smoke 门为最大绝对 score 差不超过 0.5、比率在 `[0.5, 2]` 内，然后使用上限为 2 的 token TIS。它是明确的 BF16 集成容许范围，不是逐位一致或严格概率等价。第一层对照将最早观察到的差异定位到 attention 输出；强制 math SDPA 修正了这个局部对照，也没有证明整个模型的所有差异都因此消失。[训练、恢复与数值报告](https://github.com/Levius-Fubuki/verl/blob/fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d/examples/ouro/GPU_20261028.md)

### 开关开启还不够，需要实际执行证据

独立发布矩阵覆盖 full、early exit、prefix、async、Graph、speculation 及其已支持组合，共十种。各模式比较暖引擎更新后与新鲜发布策略的结果，token 和深度相同、selected-score 最大差为 0；这不等于任意 feature 与 eager 基线都已经逐位相同。

P/D 生产端分别记录 12,599,296 bytes、四次实际 NIXL 写入，结束后 worker KV 和协调器 reservation 归零。这支持同机传输与回收正确性，不能变成跨机 RDMA 带宽成绩。

Graph 也另有 profiler 证据：两个发布后的独立 capture 各观察到 28 次实际 `cudaGraphLaunch`，没有只拿 `graphs_enabled=True` 当执行证明。prefix 检查最初看错了计数器：used blocks 不含可淘汰的 prefix-only 页。改用 KV owner's `lookup_prefix`，才验证出发布前一页缓存、开始更新后清空，以及新策略输出与新鲜参考一致；失败的原探针仍保留在归档中。

## 六、Ray 接入最难的是执行所有权和准入顺序

verl 的 `BaseRollout` 与 `RolloutReplicaRegistry` 是两个入口。实现了前者，不代表 AgentLoop 已经能找到、启动和管理 native 服务。后续单卡增量显式注册 `OuroReplica`，让真实 `LLMServerClient` 和 `SingleTurnAgentLoop` 进入 token-in/token-out 路径。

若在 async actor 里调用一个长时间阻塞的 generate，abort 和更新请求可能一直得不到执行。直接丢进任意后台线程也不够：pipe、CUDA 上下文和控制调用需要明确的所有者。

`OuroTokenServer` 采用一个 actor event-loop owner。add、step、abort、发布、清缓存和 close 都由它处理；完整 native step 之间让出事件循环，使取消和控制命令可以前进。单次 native step 仍是同步的，其时长会限制响应速度。这是调度边界，不能称为可随时抢占的设备执行。

两处竞争通过失败回归暴露：FIFO 中尚未准入的请求与 abort 的顺序，以及发布正在 drain 时另一调用尝试 resume。修复后，失败或 updating 状态不能被普通 resume 清掉。每条请求还携带 actor epoch 与唯一 generation，外部 request ID 复用不会让旧结果自动获得新请求的身份。

实际检查覆盖两个并发客户端请求、in-flight abort、完整发布失败重试、新 actor 恢复及重新发布。重启前后 token、深度与 selected score 精确相同，actor epoch 则应改变；最终 active requests 与 used KV blocks 均为零。`TokenOutput.extra_fields` 保存真实深度、采样参数、raw profile、权重版本和优化器步数，真实 `SingleTurnAgentLoop.as_dict()` 之后这些元数据仍在。[单卡服务报告](https://github.com/Levius-Fubuki/verl/blob/fd26c25f49e1d114ca27f4662ab9cda2aaeb3b9d/examples/ouro/GPU_20261029.md)

这轮只验证单文本 turn、显式模板和 standalone 单副本。工具调用、多模态、HTTP 服务、混合/共置显存切换、资源池放置、多副本发布确认，以及完整 AgentLoopManager 与训练器的数据转换都没有因此过门。两个训练 rank 加一个独立 rollout 的专用拓扑需要三张卡；这是下一阶段资源划分，不是已经完成的三卡成绩。

## 七、N+2 真的执行到目标深度，也可能不值得部署

N+2 的参考增量在第 1 轮读取 hidden，预测是否在第 3 轮退出。预测“可以退出”以后，仍必须实际执行到第 3 轮，再用该 hidden 做 coda；不能直接拿第 1 轮的 hidden 充当第 3 轮输出。浅层结束后还要 finalize last-exited KV，影响下一输入所见的历史。

训练头使用冻结全深度 teacher：48 个训练 prompt 组、16 个验证组，分别得到 768、256 个状态；另有两个各 24 prompt 的测试集。归一化只用训练数据，模型修订、目标深度与阈值写入 artifact，加载时拒绝错误元数据、非有限权重、shape 或 scale。数据按 prompt 分组，不能把同一个算术问题的相邻 token 随机分到训练与验证两侧。

验证分类准确率为 242/256，即 94.53%，训练多数类基线为 235/256，即 91.80%。这只是 teacher hazard 的不平衡分类任务。真实在线策略会改变自己经历的退出轨迹，离线分类更准，并不直接保证回答更好或计算更省。

参考实现做了强制正/负决策对照，分别匹配固定三轮与全深度；独立 StageEngine trace 对照中的五个 token、执行深度和 selected score 也完全相同。它证明这条短参考路径执行了所选深度，没有把 N+2 变成整个模型训练/推理数值等价的证明。

较难的 24 prompt 算术测试给出了明确的保留决策：

| 策略 | 回答正确数 | 平均 decode-input 深度 | 完整串行请求输出速率 |
| --- | ---: | ---: | ---: |
| 四轮全深度 | 23/24 | 4.000 | 22.50 tokens/s |
| 固定三轮 | 24/24 | 3.000 | 29.25 tokens/s |
| 原始退出门 | 24/24 | 3.153 | 27.47 tokens/s |
| 学习 N+2 头 | 23/24 | 3.308 | 26.11 tokens/s |

各模式为 BF16/Triton、greedy、并发 1、强制输出 16 token，两个未加 profiler 的 pass 反转策略顺序。计时包含新 KV 分配、prefill 和 decode，排除模型加载与预热，因此表中是完整串行参考请求口径，不能直接当作批量在线服务吞吐。

学习头执行轮数更多，还增加归一化、投影和决策读回；实测较慢与新增工作一致，但没有完成每部分 wall time 的因果分解。小测试集上的答案相同也不代表生成序列相同。当前头既没有速度优势，又漏掉固定三轮和原门能答对的一题，因而没有部署依据。保留这个负结果，比只展示 94.53% 的分类准确率更能解释下一步应改哪里。

## 八、交付物和下一步仍沿着同一条证据链

这个项目把问题分成了可分别验收的层：模型重放、概率与元数据、训练更新、物理参数导出、完整发布、服务生命周期、checkpoint 恢复，以及退出策略的质量/成本。某层通过后，后面仍有独立的门槛。

当前可公开复核的材料包括固定源码、两轮报告、原始失败与通过记录、结果 JSON、源文件 hash、profile、预测头和完整证据包。[10 月 28 日双卡归档](https://github.com/Levius-Fubuki/verl/releases/tag/ouro-gpu-evidence-20261028)、[10 月 29 日单卡归档](https://github.com/Levius-Fubuki/verl/releases/tag/ouro-single-gpu-evidence-20261029)分别保留各自环境与范围，不能把两轮合成一次更大的“全功能验收”。

下一阶段的重点是统一上游 rollout/publication 接口，并让实际分布式训练器消费 Ray 返回的完整轨迹，执行跨副本发布屏障及新进程恢复。N+2 则需要更好的任务与轨迹分布、校准和真正的批量 lookahead 实现。本文能确认的进展是具体链路及恢复条件已经有实测依据；完整生产训练、改变 world size 的恢复、广泛质量、收敛和训练加速仍需要自己的实验。

## 相关笔记

- [从并行切分到训练闭环：分布式训练笔记](/2026/10/12/distributed-training/)
- [从关键路径到达标吞吐：性能分析与实验方法](/2026/10/23/performance-analysis/)
- [推理优化的实验与验收](/2026/10/02/inference-evidence/)
