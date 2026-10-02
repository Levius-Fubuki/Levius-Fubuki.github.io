# AI Infra 算子笔记：双流调度、QKV/RoPE 融合与投机验证

这篇笔记沿着三种不同优化阅读源码：双流调整执行依赖，RoPE 融合减少中间数据物化，投机解码改变一次验证后可以提交的 token 数。材料来自 8×RTX 5090 学习包和对应 Codex 逐算子学习记录；以下历史实验由参考材料提供，不将整包实现归为我的原创，也不把它们写成 GR308 成绩。

学习时应先固定张量契约，再画依赖，最后看候选在真实服务路径上是否通过验收。三个主题因此保留各自的输入形状、测量口径与拒绝条件。QKV/RoPE 的 kernel 实现体未随学习包提供，文中融合内部数据流是依据调用契约的推断，不能当作逐行源码复现。

## Qwen3.6 MoE Router-First Dual Stream 学习笔记

### 1. 结论

该优化没有改变 MoE 数学计算，也没有新增计算 kernel。它把 routed experts 放到当前主 CUDA stream，把 shared expert 放到 alternate stream，从而改变两条独立工作流的调度顺序和临界路径。

在固定 `4×TP2、NUMA-local、NCCL auto、decode Graph、Triton linear attention` 的强基线上，严格 A-B-B-A 结果为：

```text
aggregate output tok/s：3904.68 → 3990.23，+2.19%
C1 TPOT P50：4.2609 → 4.2058 ms，-1.29%
C1 TPOT P99：4.482 → 4.351 ms，-2.91%
```

该候选只在 Qwen3.6 的精确 MoE geometry 下开启。

### 2. 模型结构背景

Qwen3.6-35B-A3B-FP8 的关键结构：

```text
40 层
30 层 GDN + 10 层 full attention
hidden size = 2048
routed experts = 256
每 token TopK = 8
另有 shared expert
生产拓扑 = 4 个 TP2 replica
```

每个 MoE block 有两条相对独立的分支：

```text
hidden states
   ├─ router / TopK → routed experts ─┐
   └─ shared expert ──────────────────┤
                                     ▼
                                   combine
```

两条分支读取同一份输入，分别产生输出，最终再组合。因此它们具备并行执行的基础。

### 3. 候选源码

新增构造参数：

```python
prefer_router_on_main_stream: bool = False
```

候选分支：

```python
current_stream = torch.cuda.current_stream()
self.alt_stream.wait_stream(current_stream)

with torch.cuda.stream(self.alt_stream):
    shared_output = self._forward_shared_experts(
        hidden_states.clone(),
        apply_gate=not use_fused_gate,
    )

router_output = self._forward_router_experts(hidden_states)

current_stream.wait_stream(self.alt_stream)
return router_output, shared_output
```

### 4. Stream 依赖关系

第一处同步：

```python
self.alt_stream.wait_stream(current_stream)
```

含义是：alternate stream 在读取 `hidden_states` 前，等待主 stream 上此前产生输入的工作完成。它向 GPU stream 插入依赖，不是让 CPU 同步等待。

随后两条分支并行排队：

```text
main stream： routed experts ───────────────┐
                                            ├─ join → downstream
alt stream ： shared expert ────────────────┘
```

第二处同步：

```python
current_stream.wait_stream(self.alt_stream)
```

含义是：主 stream 的后续工作必须等 shared expert 完成。函数返回后，下游 combine 可以安全读取两份输出。

完整 happens-before 关系：

```text
上游产生 hidden states
          │
          ├─────────────┐
          ▼             ▼
 routed experts    shared expert
          │             │
          └──────┬──────┘
                 ▼
               combine
```

### 5. 为什么 router-first 可能更快

原先的双流方向是：

```text
main stream：shared expert
alt stream ：routed experts
```

候选方向是：

```text
main stream：routed experts
alt stream ：shared expert
```

Routed 分支通常包含 router、TopK、多个 expert 的计算以及相关 TP/调度工作，临界路径更长、更复杂。把它留在主 stream，可减少它与主执行链、CUDA Graph 和后续通信调度之间的额外制约；较短的 shared 分支放在 alternate stream 作为被覆盖的支路。

但“更长分支放主 stream”不是通用定律。实际效果还取决于：

- routed/shared expert 的计算量比例；
- token 数和 TopK；
- TP/EP geometry；
- 两条分支的 SM、带宽和 workspace 竞争；
- CUDA stream 优先级；
- Graph 捕获后的调度 DAG；
- 下游 combine 和 collective 的依赖位置。

因此报告只在精确 geometry 下保留。

### 6. 为什么数学结果不变

优化前后仍计算相同的两个函数：

```text
router_output = routed_experts(hidden_states)
shared_output = shared_expert(hidden_states)
```

变化的是二者被 enqueue 到哪个 stream，不是权重、TopK、激活函数或 combine 公式。

正确性的必要条件：

1. 两个分支不能对同一个输入或输出发生冲突写入；
2. alternate stream 读取输入前必须等待输入生产完成；
3. 下游读取 shared output 前，主 stream 必须等待 alternate stream；
4. tensor 生命周期必须覆盖两个 stream 的使用期；
5. CUDA Graph capture/replay 中依赖和地址必须稳定。

`hidden_states.clone()` 给 shared 分支独立输入存储，可避免 shared 路径中的潜在原地操作或别名影响 routed 分支。由于 clone 在 alternate stream 中执行，它也受到前置 `alt.wait_stream(main)` 的保护。

### 7. Geometry gate

环境变量：

```text
SGLANG_QWEN36_SWAP_MOE_DUAL_STREAM=1
```

不会直接无条件启用候选，而是经过：

```python
def _qwen36_swap_moe_dual_stream_covered(config):
    return (
        _qwen36_swap_moe_dual_stream
        and _qwen36_exact_moe_layout(config)
    )
```

当前 bundle 没有展开 `_qwen36_exact_moe_layout()` 的完整函数体，但报告明确其目标是只覆盖 Qwen3.6 的精确 expert/TP/hidden/layer geometry。

这是 fail-closed 设计：环境变量只表达“请求启用”，模型结构 gate 决定“是否允许启用”。

### 8. A-B-B-A 单变量实验

顺序：

```text
A1：baseline
B1：router-first
B2：router-first
A2：baseline
```

唯一变化的变量：

```bash
SGLANG_QWEN36_SWAP_MOE_DUAL_STREAM=0/1
```

其他候选明确关闭：

```text
Qwen vision QKV/RoPE fusion = 0
GDN corrected merge = 0
CUDA shared expert fusion = 0
disable dual stream = 0
```

固定实验条件：

```text
profile = qwen36-fp8
layout = 4×TP2
Graph = on
linear attention decode backend = Triton
aggregate prompts = 32
C1 prompts = 8
input/output = 128 → 128
CPU offload = 关闭
```

整个 bracket 持有同一个 GPU lock，避免其他实验插入 A-B-B-A 中间。

### 9. Summarizer 审计

Summarizer 不只计算性能，还检查实验身份：

```text
profile/scope
主模型源码 SHA-256
qwen2_moe.py 支持源码 SHA-256
4×TP2 + Graph-on
四个 replica
无 offload 参数
prefill/decode 均为 Triton
运行时 layout audit
四个 server 日志
候选日志 marker
生成文本 SHA-256
```

候选日志必须出现：

```text
Qwen3.6 MoE router-first dual-stream candidate active:
routed experts on main stream, shared expert on alternate stream.
```

A 组不能出现 marker，B 组四个 replica 必须全部出现 marker。这样可以防止“设置了环境变量，但候选因 gate/fallback 没真正运行”的假阳性。

### 10. Keep gate

最终保留条件：

```python
keep = (
    generated_text_equal
    and output_speedup >= 1.01
    and candidate_c1_tpot_p99
        <= baseline_c1_tpot_p99 * 1.05
)
```

即：

```text
所有 A/B 输出摘要一致；
aggregate output throughput 至少提升 1%；
C1 TPOT P99 退化不超过 5%。
```

测试还显式验证：输出摘要不同、P99 超限或候选 marker 缺失时必须拒绝或 fail closed。

### 11. 源码入口

- 候选补丁：`qwen36_moe_router_first_remote.patch`（学习包文件）
- A-B-B-A 队列：`qwen36_shared_expert_ab_queue.sh`（学习包文件）
- Summarizer：`qwen36_shared_expert_ab_summarize.py`（学习包文件）
- 合同测试：`test_qwen36_shared_expert_ab_summarize.py`（学习包文件）

### 12. 记忆卡

```text
类型：调度型算子优化，不是新数学 kernel
主 stream：routed experts
alternate stream：shared expert
开始依赖：alt.wait_stream(main)
结束依赖：main.wait_stream(alt)
正确性：两分支数学不变，join 后再读取
保护：环境变量 + exact geometry gate
证据：强 Graph 基线上的严格 A-B-B-A
收益：throughput +2.19%，C1 TPOT P99 -2.91%
```

## Qwen3-VL QKV/RoPE Fusion 学习笔记

### 1. 结论

该算子把视觉注意力中的以下操作融合起来：

```text
QKV 拆分
→ Q/K/V reshape 与连续布局整理
→ Q/K 的 NeoX RoPE
→ 直接写出最终 Q、K、V
```

融合的主要价值是减少中间张量、显存重复读写和 kernel launch。它不会消除最终 Q、K、V 的输出写入，也不会优化整个视觉编码器的所有阶段。

在最终 `2×TP4 + CPU feature transport + DP encoder + Graph` 路径上，严格 A-B-B-A 的 aggregate output throughput 为：

```text
76.702 → 77.772 output tok/s，+1.40%
```

TTFT P50 降低约 1.69%，因此在该精确部署路径上保留。

### 2. 输入输出契约

真实 shape gate 默认使用：

```text
tokens   = 1024 / 4096 / 16384
heads    = 16
head_dim = 72
```

输入：

```text
qkv: [1, T, 3 × 16 × 72] = [1, T, 3456]，BF16
cos: [T, 36]，FP32
sin: [T, 36]，FP32
```

输出：

```text
Q': [T, 16, 72]，已应用 RoPE
K': [T, 16, 72]，已应用 RoPE
V : [T, 16, 72]，只完成拆分和布局整理
```

调用接口：

```python
q, k, v = fused_qkv_split_rope(qkv, cos, sin, heads, head_dim)
```

### 3. Baseline 数据流

基线实现：

```python
q, k, v = qkv.split([heads * head_dim] * 3, dim=-1)
q = q.reshape(-1, heads, head_dim).contiguous()
k = k.reshape(-1, heads, head_dim).contiguous()
v = v.reshape(-1, heads, head_dim).contiguous()

if cos.shape[-1] * 2 == head_dim:
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)

q, k = apply_rotary_pos_emb_native_eager(q, k, cos, sin)
```

`split()` 通常只产生 view，但 Q、K、V 仍继承原始 QKV 的行跨度。后续 `.contiguous()` 会真正分配和写入连续的 Q、K、V。

基线近似数据流：

```text
QKV
 ├─ 读取并写出 contiguous Q
 ├─ 读取并写出 contiguous K
 ├─ 读取并写出 contiguous V
 ├─ 物化扩展后的 cos
 ├─ 物化扩展后的 sin
 └─ RoPE 再次读取 Q/K，并写出旋转后的 Q'/K'
```

### 4. NeoX RoPE 数学

对于一个 `head_dim=72` 的向量，将其分为两个 36 维半区：

```text
x = [x_left | x_right]
```

NeoX 风格旋转：

```text
rotate_half(x) = [-x_right | x_left]
x_rope = x × cos + rotate_half(x) × sin
```

等价的逐元素形式：

```text
x_left'  = x_left × cos - x_right × sin
x_right' = x_right × cos + x_left  × sin
```

该操作分别应用于 Q 和 K，V 不旋转。

### 5. 融合后的预期数据流

当前学习 bundle 没有保存 `qwen3_vl_qkv_rope.py` 的 kernel 实现体，因此下面是根据调用接口、baseline 和候选名称得到的实现级推断：

```text
读取 row-major QKV 与半宽 cos/sin
→ 在寄存器中定位 Q/K/V 和 head 内维度
→ 在寄存器中计算 Q/K 的 NeoX RoPE
→ 直接写出最终 Q'/K'/V
```

它可以避免：

- 单独物化连续 Q/K 中间结果；
- 显式生成 `[T, 72]` 的 cos/sin；
- RoPE 重新读取刚写出的 Q/K；
- 多个独立 PyTorch/CUDA kernel launch。

### 6. 正确性 gate

Gate 固定随机种子，并对 Q、K、V 三个输出逐一比较：

```python
torch.manual_seed(20260825)

torch.testing.assert_close(
    got,
    expected,
    rtol=1e-2,
    atol=2e-2,
)
```

同时记录：

```text
max_abs：发现局部索引、半区映射、符号或尾部 mask 错误
mean_abs：发现整体数值的系统性偏移
```

当前 gate 的重点是报告对应的真实 geometry，而不是证明该算子支持所有 shape。若扩展为通用算子，还应增加 batch、非 2 次幂 token、不同 heads/head_dim、非连续输入、真实 cos/sin、NaN/Inf 和 CUDA Graph capture 测试。

### 7. Microbenchmark

计时方法：

```text
warmup 20 次
→ CUDA synchronize
→ CUDA Event 重复计时 100 次
→ 使用中位数
```

Microbenchmark 只能说明该 exact-shape 算子调用更快，不能直接作为模型 E2E 收益。

### 8. E2E A-B-B-A

实验顺序：

```text
A1：关闭融合
B1：开启融合
B2：开启融合
A2：关闭融合
```

开关：

```bash
SGLANG_QWEN_VIT_FUSED_QKV_ROPE=0/1
```

固定条件包括：

```text
layout = 2×TP4
Graph = on
aggregate prompts = 32
C1 prompts = 8
input/output = 128 → 64
```

Summarizer 的保留条件：

```text
日志确认融合路径实际命中；
aggregate output throughput 至少提升 1%；
C1 TPOT P99 退化不超过 5%。
```

日志命中检查很重要：只设置环境变量并不能证明候选算子真正执行，shape/backend 不支持时可能发生 fallback。

### 9. 为什么 E2E 只有 +1.40%

完整 Qwen3-VL 请求还包括：

```text
图片预处理
→ patch embedding
→ 多层 ViT attention/MLP
→ 视觉特征传输
→ language prefill
→ language decode
→ TP collective
```

QKV/RoPE fusion 只覆盖视觉链路中的一个局部阶段。即使局部 kernel 加速明显，最终收益仍受该阶段在总延迟中占比限制，即 Amdahl 定律。

### 10. 源码入口

- Gate：`qwen_vit_qkv_rope_gate.py`（学习包文件）
- A-B-B-A 队列：`qwen_vit_fusion_ab_queue.sh`（学习包文件）
- 汇总与 keep gate：`qwen_vit_fusion_ab_summarize.py`（学习包文件）
- E2E runner：`qwen_vit_fusion_e2e_runner.py`（学习包文件）

### 11. 记忆卡

```text
融合对象：QKV split + contiguous/layout + Q/K NeoX RoPE
核心收益：减少中间张量、显存读写和 launch
真实 shape：H=16，D=72，T=1024/4096/16384
正确性：Q/K/V 全量 assert_close
实验方法：最终 topology 上 A-B-B-A
模型收益：+1.40%，TTFT P50 -1.69%
证据边界：不能与通信拓扑的 1.932× 直接相乘
```


## 投机解码：提出候选、验证与提交

下面的连续匹配伪代码描述 greedy 合同。随机采样的无损验证还涉及接受概率与拒绝后的分布修正，不能仅用 token 相等条件代替。全候选接受时的额外 token 来自 target 的下一位置；首次不匹配时由 target 给出纠正 token。

### 1. 总体机制

投机解码使用较便宜的 draft 一次提出多个候选 token，再由完整 target 一次并行验证：

```text
draft 生成候选块
→ target 并行 verify
→ 接受从左到右连续匹配的前缀
→ 追加一个 target bonus token
→ 更新 target/draft KV Cache
```

正确实现不会直接提交未经 target 验证的 draft token。

### 2. 两条实验线

#### EAGLE

Qwen3.6 实验配置：

```text
algorithm = EAGLE
num_steps = 3
topk = 1
num_draft_tokens = 4
draft attention = Triton
```

`topk=1` 表示生成线性候选链，不构造分叉 token tree。

结果：

```text
aggregate output tok/s：3808.20 → 3915.29，+2.81%
C1 TPOT P50：4.2155 → 2.3030 ms，基线约为候选的 1.83 倍
C1 TTFT P50：66.91 → 78.18 ms，退化约 16.8%
```

输出摘要未通过严格一致性 gate，因此 `keep=false`。当前只能作为低延迟候选。

#### DFlash2

DFlash2 draft 没有自己的 embedding 和 lm_head：

```text
target embedding
→ DFlash2 draft hidden states
→ 复用 target lm_head 选择候选 token
→ target verify
```

在 TP 环境中，每个 rank 先取本地 TopK，然后只 AllGather `TP × K` 个候选分数和 ID，再做全局 TopK，避免聚合完整 vocabulary logits。

结果：

```text
C1 TPOT P50：12.8667 → 3.8852 ms，基线约为候选的 3.31 倍
接受长度中位数：约 3.034
C32 output tok/s：1011.77 → 782.89，退化 22.6%
C1 TTFT P50：退化约 5.7%
```

因此 DFlash2 更适合作为低并发 latency lane，不适合作为 C32 吞吐默认。

### 3. DFlash2 关键源码结构

#### Draft model

```text
target 多层 hidden states
→ concat
→ fc
→ RMSNorm
→ draft transformer layers
→ candidate hidden states
```

Draft forward 返回 hidden states，不直接返回 logits。

#### Candidate selector

每个位置先产生 K 个基础候选，然后根据 token 自身分数与相邻候选转移分数选择一条候选路径：

```text
score = unary logit + context-conditioned transition score
```

#### Verify and commit

Greedy 验证从左到右比较：

```python
accept_len = 0
for i in range(num_draft_tokens):
    if draft[i] == target[i]:
        accept_len += 1
    else:
        break
```

提交长度：

```text
commit_len = accept_len + 1
```

额外的 1 是 target 在第一个不匹配位置提供的 bonus token。因此即使 draft 第一个 token 就失败，每轮仍至少前进一个正确 token。

#### KV Cache

Verify 完成后，只能将已提交位置对应的 target hidden states 写入 draft KV Cache。被拒绝的候选不能留在缓存中，否则后续生成会使用错误上下文。

### 4. 验收指标

投机解码不能只看 TPOT，还要同时检查：

```text
accept length
aggregate throughput
C1 TPOT P50/P99
TTFT
输出正确性
greedy 与 stochastic sampling 的不同合同
```

随机 sampling 下输出摘要不同不一定代表错误，但成为默认配置前必须补充 greedy 条件下的确定性一致性 gate。

### 5. 关键源码入口

- DFlash draft model：`dflash.py`（学习包文件）
- DFlash worker：`dflash_worker_v2.py`（学习包文件）
- Accept/prepare Triton kernel：`dflash.py`（学习包文件）
- EAGLE 实验队列：`qwen36_spec_decode_ab_queue.sh`（学习包文件）
- EAGLE 汇总 gate：`qwen36_spec_decode_ab_summarize.py`（学习包文件）


## 把三种优化放回同一条服务路径

双流优化主要改变关键路径，融合主要改变数据搬运，投机解码主要改变验证与提交比例。三者都可能在某个局部指标上获益，却在整体服务中被资源竞争、首 token 等待或错误状态抵消。不能将各自的单点倍率相乘。

这里的 gate 是实验预设的选择条件，不是全形状、全采样分布的证明。输出摘要一致支持当前输入和执行配置；未提供的 kernel 源码、未测的 dtype 与未覆盖的并发仍是边界。系统层配置与拓扑背景见 [八卡推理系统学习笔记](/2026/10/02/infra-topology/)。公开框架入口为 [SGLang](https://github.com/sgl-project/sglang)，文中源码文件名对应保存于本地的历史学习包，不能假设上游 main 仍是相同实现。

## 按输入契约手算一次正确性检查

以视觉 QKV 为例，一个 token 的最后一维先排 Q，再排 K，再排 V。每段又含多个 head，每个 head 内是旋转前的通道。如果融合实现只保证输出元素总数相同，却把 head 和 token 的顺序交换，后续矩阵乘仍可能接受这个张量，最终结果却已经改变。因而 shape、stride、元素映射和 dtype 都属于契约。

NeoX 旋转把一个 head 的前半与后半配对。可以用四维玩具向量理解：先取前两个数和后两个数，后半取负后放到前面，前半移到后面，再分别乘正弦和余弦。若改成相邻两个元素配对，得到的是另一种布局合同，即使公式也叫 RoPE，仍不能互换。真实算子检查因此必须覆盖所有位置，而不能只打印头几个数。

误差门还要同时保存绝对误差与参考幅值。相对误差在参考值接近零时可能放大，绝对容差则可能掩盖大幅值处的比例偏移。学习包同时使用相对与绝对容差，并记录最大值与平均值，正是为了让极端索引错误和整体数值偏移都有机会被看见。通过这些检查依然只覆盖当前几何形状，不能把未测试的非连续输入自动列为支持。

双流例子关注的则是时间契约。同一个输入 tensor 可以被两条流读取，但输入生产必须先完成；两份输出可以分别计算，但 combine 必须等两边都完成。两个 wait 分别位于分叉之前与汇合之前，作用不同。将第二个 wait 提前到分叉处会让两条支路失去重叠，完全删除它又可能产生读写竞争。

这里的 clone 也不是免费操作。它增加内存读写，却可以隔离潜在原地写入或别名关系。是否能安全去掉，要检查真实子函数的写行为、生命周期与图捕获约束，不能因为数学上两条分支独立就直接删除。调度优化最终仍受共享执行资源约束：两条流表示允许并发，不保证硬件一定同时执行。

投机验证再增加了序列契约。假设草稿提出三个 token，前两个与 target 一致，第三个不一致，本轮只能提交前两个和 target 在第三个位置的纠正结果。被拒绝位置之后的候选既不能出现在输出里，也不能残留为下一轮可见缓存。若全接受，额外 token 则来自 target 对下一位置的预测，不能把两种情况混成同一个数组索引。

这三种合同分别落在空间、时间和序列上。优化代码有时只改几行，却可能破坏其中任意一层。阅读源码时，先找输入何时有效、输出采用什么布局、哪些状态允许提交，再去看计时结果，可以更快理解为什么某个候选必须受精确 shape 或并发条件限制。
