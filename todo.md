# myvllm 轻量开发路线

## 1. 项目目标

myvllm 是一个用于学习现代 LLM 推理核心机制的轻量框架：

- 以 nano-vLLM 为模型和计算层基线。
- 参考 vLLM V1 的 scheduler、paged KV cache 和 prefix cache 语义。
- 优先跑通单进程、单 GPU、Qwen3 的完整离线推理链路。
- 支持 continuous batching、chunked prefill 和 mixed prefill/decode。
- 在单 Engine 正确后完成两个独立进程之间的 Prefill/Decode（PD）分离和真实
  KV cache 传输。
- 保留足够清晰的边界，方便后续研究外部 C++ KV engine 和稀疏 Attention。

第一版不是 vLLM 的缩小复制品，不追求生产系统的全部兼容性、插件能力和
异常恢复能力。

## 2. 轻量实现约束

后续代码继续采用当前 `KVCacheManager` 的实现风格：职责完整，但不堆叠生产级
抽象和重复校验。

### 2.1 只实现当前主链路需要的职责

- 一个需求只有一种实现时，直接使用函数或具体类，不提前定义 backend、registry、
  protocol 或 factory。
- 固定一种 block hash、一个 cache layout、一个 Qwen3 模型族和一个 Attention
  路径。
- 每个组件完整负责一件事，不先写临时无缓存推理路径，也不写随后必然删除的
  placeholder backend。
- 优先让核心组件保持在约 100--250 行；超过这一规模时先检查是否混入其他职责。

### 2.2 信任内部调用边界

- 外部配置和用户输入进行必要校验。
- Scheduler、ModelRunner、KVCacheManager 之间约定好的内部参数不重复做
  `ValueError` 防御。
- 引用计数、free queue、容量原子性等会破坏核心状态的不变量必须保留。
- 内部不可能分支优先使用少量 `assert`，不为每个误用路径设计异常体系。

### 2.3 明确不进入轻量主线的功能

第一版不实现：

- vLLM worker 增量同步协议和通用多进程 RPC 层。
- KV cache group、Mamba cache、null block、KV events 和 offload。
- 通用 Attention backend 注册体系。
- LoRA、量化、structured output 和分布式容错。
- 通用 KV connector 插件体系、远程多节点传输和外部 C++ engine。
- 稀疏 Attention/SnapKV/AdaKV；主链路完成后再作为研究扩展加入。

### 2.4 核心所有权

```text
LLMEngine
├── Request table                 # 请求生命周期的唯一所有者
├── Scheduler                     # 只决定本轮运行什么
│   └── KVCacheManager            # block table 和 prefix cache
└── ModelRunner                   # 只准备 batch 并执行模型
    ├── KVCacheStorage            # 实际 GPU K/V tensor
    └── Qwen3 / Attention
```

- `Request.num_computed_tokens` 是唯一计算进度。
- Engine 只在模型执行成功后推进 Request。
- Scheduler 不修改计算进度，只分配预算和 KV blocks。
- ModelRunner 不持有另一份 Request 状态。
- 模型层不读取全局 `Context`；ModelRunner 显式传递 execution metadata。

PD 模式复用上面的完整 Engine，不再实现另一套 scheduler/cache/model：

```text
PDCoordinator
├── PREFILL PDEngineProcess       # 独立进程和独立 Engine/KV cache
├── DECODE PDEngineProcess        # 独立进程和独立 Engine/KV cache
└── LocalKVConnector              # 经 CPU shared memory 实际传输 KV
```

- `PDCoordinator` 只路由请求和 handoff，不执行模型、不持有 GPU KV cache。
- 两个 `PDEngineProcess` 各自拥有模型、Scheduler、KVCacheManager 和
  KVCacheStorage，不能共享 Python Engine 对象。
- `LocalKVConnector` 只负责 tensor 传输和传输生命周期，不拥有 Request 或
  调度策略。
- 第一版只实现单机跨进程、CPU shared-memory staging 这一种正确性路径；直接
  GPU/NCCL、RDMA 和 C++ connector 留到性能扩展。

## 3. 当前进度

### 3.1 工程和核心数据结构：已完成

- [x] uv、src layout、pytest、ruff 和 mypy。
- [x] `SamplingParams`。
- [x] `ModelConfig`、`SchedulerConfig`、`CacheConfig`、`ParallelConfig`。
- [x] `RequestStatus`、`Request` 和状态转换。
- [x] `ScheduledRequest`、`SchedulerOutput` 和 `RequestOutput`。
- [x] `num_computed_tokens` 作为唯一计算进度。
- [x] `KVCacheSpec` 及 dtype item size、单 block 字节数和总容量计算。

### 3.2 Paged KV cache 和 prefix cache：已完成

- [x] `KVCacheBlock` 和完整 block hash boundary。
- [x] `FreeKVCacheBlockQueue` 双向链表及批量头尾操作。
- [x] `BlockHashToBlockMap`，支持相同 hash 的多个物理 block。
- [x] `BlockPool` 的分配、touch、free、cache、eviction 和 reset。
- [x] Request 自己增量维护固定 SHA-256 链式 block hashes。
- [x] `KVCacheManager` 权威维护 `request_id -> block table`。
- [x] 增量 `allocate_slots()` 和容量不足时的原子失败。
- [x] 最长连续 prefix lookup，且保留至少一个 token 产生 logits。
- [x] 完整 block 注册、共享引用、释放和抢占接口。
- [x] duplicate hash、free queue 顺序、共享引用、prefix hit 和 eviction 测试。

轻量取舍：prefix hash 只用于当前单模型 Engine 生命周期，不加入持久化 cache 的
model revision、cache group 或多算法 hasher。

### 3.3 GPU KV storage：主体已完成

- [x] `KVCacheStorage` 分配真实 tensor：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

- [x] storage 只提供每层 K/V view，不保存 Request 或 block table。
- [x] CPU shape/view 测试和 CUDA bf16 分配、写入 smoke test。
- [x] 默认 `CacheConfig.block_size = 256`。
- [ ] Attention 完成后，将每层绑定到对应 K/V view。

### 3.4 nano-vLLM 基础计算层：已完成

- [x] `SiluAndMul`。
- [x] `RMSNorm` 和 fused residual add。
- [x] RoPE。
- [x] replicated/column/row tensor-parallel linear。
- [x] merged gate/up 和 packed QKV weight loader。
- [x] vocab embedding 和 LM head。
- [x] greedy/temperature sampler。
- [x] distributed 未初始化时使用 rank 0/world size 1。
- [x] 移除 LM head 对 nano-vLLM 全局 `Context` 的依赖。
- [x] 记录 nano-vLLM 来源 commit 和 MIT 许可证。

### 3.5 依赖与验证检查点

- [x] PyTorch、NumPy 和 Triton 环境。
- [x] FlashAttention 2.8.4 固定到官方 commit `d16e381f...`。
- [x] FA2 paged KV 对应的默认 block size 256。
- [x] 当前全量测试：203 passed。
- [x] 当前 ruff、mypy 和 `uv lock --check` 通过。

说明：FA2 已声明并写入 lockfile，但 `myvllm/.venv` 仍需通过 `uv sync --dev`
实际构建安装；`nano-vllm/.venv` 中的安装不会被其他虚拟环境共享。

## 4. 后续轻量主线

Attention 暂缓到完成 Triton 学习后实现。现在先做不会被 Attention 设计推翻的
完整组件。

### 4.1 下一步：Continuous batching Scheduler

职责：维护 waiting/running 队列，在 token、sequence 和 KV block 预算内决定
本轮运行哪些请求。它可以调用 `KVCacheManager`，但不执行模型，也不推进
Request 的计算进度。

- [ ] 维护 waiting/running 请求队列。
- [ ] 接受 `max_num_batched_tokens` 和 `max_num_seqs` 预算。
- [ ] running 请求优先推进，再使用剩余预算接纳 waiting 请求。
- [ ] 统一计算：

```text
remaining = request.num_tokens - request.num_computed_tokens
scheduled = min(remaining, remaining_token_budget)
```

- [ ] decode 通常调度 1 token；prefill 可按剩余 budget 切 chunk。
- [ ] 新请求先查询 prefix cache，再调用 `KVCacheManager.allocate_slots()`。
- [ ] KV 不足时抢占 running 队尾请求并释放其 block table。
- [ ] Scheduler 只返回 `SchedulerOutput`，不推进 Request 进度。
- [ ] `should_sample` 只在本轮计算到请求当前尾部时为真。
- [ ] 完成 token budget、chunk boundary、prefix hit、KV 不足、preemption 和
  无 block 泄漏测试。

第一版不实现 priority policy、复杂公平性算法、async scheduler、metrics
collector 和 scheduler/worker delta protocol。

### 4.2 通用 safetensors weight loader

职责：把本地 safetensors checkpoint 映射到模型参数，包括当前 Qwen3 需要的
packed projection；不负责下载模型、创建模型或执行 forward。

- [ ] 增加 `safetensors` 依赖。
- [ ] 实现普通 parameter copy。
- [ ] 根据 `packed_modules_mapping` 加载 Q/K/V 和 gate/up shards。
- [ ] 忽略 checkpoint 中模型明确不使用的额外参数。
- [ ] 使用小型临时 safetensors checkpoint 测试普通、merged 和 QKV 加载。

第一版不实现远程下载、量化权重、LoRA、多个 checkpoint 格式或 vLLM 的
通用 weight loader registry。

### 4.3 暂缓：AttentionMetadata

职责：作为一次 model execution 的只读数据容器，保存已经算好的 batch 边界、
block table 和 slot mapping。它不从 Request 构建自己，也不执行 attention。

- [ ] `AttentionMetadata` 只保存 ModelRunner 已计算的数据，不包含 builder：
  - [ ] `query_start_loc`。
  - [ ] sequence lengths / sequence start locations。
  - [ ] `slot_mapping`。
  - [ ] padded `block_tables`。
  - [ ] 最大 query/sequence 长度。
- [ ] ModelRunner 负责从 Request 和 `ScheduledRequest` 计算上述 tensor。
- [ ] 使用 CPU 单元测试验证 full prefill、partial prefill、decode 和 mixed batch
  对应的边界与映射。

### 4.4 暂缓：Attention

职责：把当前层产生的 K/V 写入 paged KV storage，并用固定的
FlashAttention-2 路径计算 attention 输出；不判断请求状态、不分配 block，也不
构造 batch metadata。

完成 Triton 学习后实现，不在此前加入临时 PyTorch production backend。

- [ ] Triton kernel 根据 `slot_mapping` 写入当前 K/V。
- [ ] 第一版 Attention 主体采用 nano-vLLM 路线：FlashAttention-2 varlen 和
  KV-cache decode API。
- [ ] 同时支持 full prefill、chunked prefill、decode 和 mixed batch。
- [ ] 每层 Attention 绑定 `KVCacheStorage.get_layer_kv_cache(layer_idx)`。
- [ ] GPU 数值测试与 PyTorch reference 对齐。

第一版固定 FA2，不建立多 backend 抽象；如果目标 GPU 不支持 FA2，再在主链路
完成后评估 FA4 或自写 Triton Paged Attention。

### 4.5 Qwen3 模型

职责：组合现有 layer、Attention 和 LM head，实现 Qwen3 dense causal LM 的
forward；不管理 Request、KV block 分配或采样循环。

- [ ] 迁移 `Qwen3Attention`、`Qwen3MLP` 和 decoder layer。
- [ ] 迁移 `Qwen3Model` 和 `Qwen3ForCausalLM`。
- [ ] 使用已经完成的 packed QKV、merged gate/up、RoPE 和 RMSNorm。
- [ ] 将 execution metadata 显式传到 Attention，不读取全局状态。
- [ ] LM head 只计算 ModelRunner 选择后的 hidden-state rows。
- [ ] 使用小尺寸随机 Qwen3 config 完成 GPU forward 测试。

第一版只支持 Qwen3 dense 模型，不建立模型 registry。

### 4.6 ModelRunner

职责：把 `SchedulerOutput` 和 Request 转成本轮 GPU batch，构造
`AttentionMetadata`，执行 Qwen3 和 sampler，并返回计算结果；不修改 Request，
不维护调度队列。

- [ ] 持有 Qwen3、KVCacheStorage 和 sampler。
- [ ] `prepare_batch()` flatten 本轮 token IDs 和 positions。
- [ ] 构造 Attention metadata、padded block tables 和 slot mapping。
- [ ] 同时接受 full/partial prefill 和 decode 请求。
- [ ] partial prefill 中间 chunk 不选择 logits row。
- [ ] 最后一个 prompt chunk和 decode 请求选择各自最后一个 logits row。
- [ ] 执行模型并返回 sampled token IDs，不修改 Request。
- [ ] mixed batch 输出与逐请求执行一致。

第一版不实现 CUDA graph、worker RPC、pinned-buffer 复用和异步 copy。

### 4.7 LLMEngine 和端到端生成

职责：作为单 Engine 请求生命周期的唯一所有者，组织
`schedule -> execute -> update` 循环，处理终止和输出；不实现具体 attention，
也不把调度逻辑复制到 Engine 中。

- [ ] Engine 持有 Request table、Scheduler 和 ModelRunner。
- [ ] 接受 prompt/token IDs，创建并提交 Request。
- [ ] 循环执行 `schedule -> model -> update request`。
- [ ] 模型执行成功后推进 `num_computed_tokens` 并追加 sampled token。
- [ ] 处理 EOS、stop token、`max_tokens`、abort、failed 和 finished。
- [ ] 所有 terminal 路径最终释放 KV blocks。
- [ ] 返回稳定的累计 `RequestOutput`。
- [ ] 完成单请求多 token 和多请求动态 batch 生成。
- [ ] greedy 输出与 Hugging Face/nano-vLLM reference 一致。
- [ ] chunked prefill 与完整 prefill 输出一致。
- [ ] mixed batch 与逐请求输出一致。

第一版 Engine 是同步离线 API，不实现 HTTP server、OpenAI API 或异步 streaming。

### 4.8 单 Engine 正确性和性能基线

这不是新运行组件，而是进入 PD 阶段前的验收门槛。

- [ ] greedy 输出与 Hugging Face/nano-vLLM reference 一致。
- [ ] chunked prefill 与完整 prefill 一致。
- [ ] mixed batch 与逐请求串行执行一致。
- [ ] abort、failed、finished 和 preemption 后无 KV block 泄漏。
- [ ] 记录 TTFT、TPOT、吞吐、峰值显存、KV usage、prefix hit rate 和
  preemption 次数。
- [ ] 使用相同模型、输入、batch 和生成长度与 nano-vLLM/vLLM 对比。

## 5. 必做主线：独立进程 PD 分离

单 Engine 验收完成后再进入本阶段。最终完成标准是 Prefill Engine 和 Decode
Engine 运行在两个独立进程中，真实传输 prompt KV，生成结果与单 Engine 一致。

PD 只新增 3 个运行组件。`EngineRole`、`PDHandoff` 和
`KVTransferMetadata` 是这些组件之间的轻量数据结构，不单独算组件，也不建立
通用 RPC/connector 协议层。

### 5.1 LocalKVConnector

职责：在两个本机进程之间传输选定请求的 KV tensor，并维护
save/load/ack/cancel 的传输生命周期；不拥有 Request，不分配逻辑 block，也不
决定何时调度请求。

- [ ] 第一版固定使用 CPU shared-memory staging：Prefill GPU KV 复制到共享
  CPU tensor，Decode 再复制到自己的 GPU KV blocks。
- [ ] `save()` 按请求的逻辑 block 顺序导出所有层 K/V，而不是传递只在 Prefill
  Engine 内有效的物理 block ID。
- [ ] `load()` 接受 Decode Engine 已分配的目标 block IDs，并写入对应 storage。
- [ ] 提供最小 `poll()`、`ack()` 和 `cancel()`，清理 shared-memory payload。
- [ ] metadata 携带 transfer/request ID、有效 token 数、block 数以及 cache spec
  fingerprint，拒绝形状、dtype、model revision 不一致的传输。
- [ ] 使用小 KV tensor 完成两个 spawn 进程之间的真实传输测试。

第一版不定义抽象 `KVConnector` 基类，不实现 NCCL、RDMA、远程节点或多个
connector backend。出现第二种传输实现时再提取公共接口。

### 5.2 PDEngineProcess

职责：在一个独立子进程中运行现有 `LLMEngine`，根据 `PREFILL` 或 `DECODE`
role 执行对应阶段，并负责本地 Request/KV block 与 transfer 状态的衔接；不负责
客户端路由或跨进程 tensor 搬运细节。

- [ ] 每个进程独立创建模型、Scheduler、KVCacheManager、KVCacheStorage 和
  LLMEngine，不共享 Python Engine 对象。
- [ ] Prefill role 计算完整 prompt、采样第一个 output token、调用 connector
  保存 prompt KV，然后返回 `PDHandoff`。
- [ ] Prefill 进程在 Decode load 成功并 ack 前保持源 blocks 引用；ack、cancel
  或失败后释放。
- [ ] Decode role 重建 Request，先分配自己的目标 blocks，再调用 connector
  load；KV ready 前不能进入 running queue。
- [ ] Decode 从 `num_computed_tokens == num_prompt_tokens` 和已追加的第一个
  output token 继续生成。
- [ ] 通过最小 command/result queue 处理 submit、handoff、abort、error 和
  shutdown，不复制 vLLM worker 增量同步协议。

`BOTH` 只表示普通单 Engine 模式，可复用已有 `LLMEngine`，不需要第三个进程。

### 5.3 PDCoordinator

职责：作为 PD 模式的客户端入口，把请求依次路由到 Prefill 和 Decode 进程，转发
handoff、abort 和最终输出；不执行模型、不访问 GPU KV tensor，也不保存第二份
Engine Request 状态。

- [ ] 启动或连接一个 Prefill 进程和一个 Decode 进程。
- [ ] 为每个 request/transfer 维护最小路由状态，拒绝重复 handoff。
- [ ] 收到 Prefill handoff 后提交给 Decode；收到 Decode ready 后通知 Prefill
  释放源 KV。
- [ ] abort 和子进程错误必须同时取消 transfer，并通知两个 Engine 清理本地
  Request 和 blocks。
- [ ] 对外返回与单 Engine 相同的累计 `RequestOutput`。
- [ ] 独立进程端到端测试覆盖正常生成、并发请求、abort、transfer failure 和
  进程退出清理。

PD 数据流固定为：

```text
client -> PDCoordinator -> Prefill process
       <- first token + PDHandoff
       -> Decode process allocates destination blocks
       -> LocalKVConnector copies prompt KV
       <- Decode ready / transfer ack -> Prefill releases source blocks
       <- Decode continues generation -> final RequestOutput
```

PD 验收条件：两个进程具有独立 PID 和独立 Engine/KV storage；传输期间源 KV
不会提前释放，Decode 不会提前调度；正常、取消和失败路径均无 shared-memory
或 block 泄漏；输出与同配置单 Engine 一致。

## 6. 剩余组件和实现顺序

完整项目主线还剩 **10 个主要组件**，其中前 7 个组成单 Engine，后 3 个完成
独立进程 PD：

| # | 组件 | 唯一职责 | 当前状态 |
|---|---|---|---|
| 1 | Continuous batching Scheduler | 队列、预算、KV 分配和抢占决策 | 下一步 |
| 2 | safetensors weight loader | checkpoint 到现有 parameter 的映射 | 未实现 |
| 3 | AttentionMetadata | 保存一次执行的 attention 数据 | 空文件，暂缓 |
| 4 | Attention | Triton KV 写入和固定 FA2 计算 | 空文件，暂缓 |
| 5 | Qwen3 model | 组合 layer，完成模型 forward | 未实现 |
| 6 | ModelRunner | 准备 batch、执行模型和采样 | 未实现 |
| 7 | LLMEngine | Request 生命周期和执行循环 | 未实现 |
| 8 | LocalKVConnector | 本机跨进程实际传输 KV tensor | 未实现 |
| 9 | PDEngineProcess | 在独立进程按 P/D role 运行 Engine | 未实现 |
| 10 | PDCoordinator | 请求路由、handoff 和错误协调 | 未实现 |

推荐实现顺序：

```text
现在：Scheduler -> weight loader

完成 Triton 学习后：
AttentionMetadata -> Attention -> Qwen3 -> ModelRunner -> LLMEngine
-> 单 Engine correctness/performance baseline
-> LocalKVConnector -> PDEngineProcess -> PDCoordinator
-> 独立进程 PD correctness/performance baseline
```

`EngineRole`、`PDHandoff`、`KVTransferMetadata`、进程 command/result 等只是
支持上述组件的数据类，不为了数量统计而把它们包装成 manager 或 protocol。

## 7. 主链路完成后的可选扩展

### 7.1 性能优化

- [ ] 在 profiling 证明有收益后优化 CUDA graph、pinned buffer 和 metadata
  copy。
- [ ] 为 `LocalKVConnector` 增加 GPU-direct/NCCL 实现时，再提取窄 connector
  接口。

### 7.2 外部 C++ KV engine

- [ ] 保持 Scheduler 拥有逻辑 block table。
- [ ] 只在出现第二种 storage 实现时抽象窄 `KVCacheBackend` 接口。
- [ ] 使用 pybind11、nanobind 或稳定 C ABI 接入。

### 7.3 稀疏 KV/Attention 研究

- [ ] 在 dense correctness 和性能基线之后选择 SnapKV、AdaKV 等单一方案。
- [ ] 优先扩展 block selection/metadata，不侵入 Request 和 Scheduler 主状态机。

## 8. 全局正确性要求

```text
0 <= request.num_computed_tokens <= request.num_tokens
prefill := num_computed_tokens < num_prompt_tokens
decode  := num_computed_tokens >= num_prompt_tokens
```

- free queue 中的真实 block 必须满足 `ref_cnt == 0`。
- `ref_cnt > 0` 的 block 不能进入 free queue。
- 同一物理 block 不能重复入队或重复释放。
- free cached block 可以保留 hash；物理 block 重用前必须驱逐旧 hash。
- KV 分配失败不能留下部分 block table 或错误引用计数。
- Scheduler 不推进 Request；Engine 只在执行成功后推进。
- ModelRunner 不修改 Request。
- 完成、取消、失败和抢占路径最终释放 KV blocks。
- greedy、chunked prefill 和 mixed batch 必须与 reference 对齐。
- Prefill transfer ack 前源 blocks 必须保持引用，不能被 free queue 复用。
- Decode load 完成前请求不能进入 running queue，也不能推进计算进度。
- 每个 transfer 最终只能进入 ack、cancel 或 failed 中的一种 terminal 状态。
- PD 正常、abort、transfer failure 和进程退出路径最终释放两侧 blocks 与共享
  内存。
- PD 输出必须与相同配置、prompt 和 sampling 参数的单 Engine 输出一致。

## 9. 规模目标

- Scheduler：约 150--250 行。
- Weight loader：约 50--100 行。
- Attention metadata + nano-vLLM 风格 Attention：约 150--250 行。
- ModelRunner：约 250--400 行。
- LLMEngine：约 150--250 行。
- LocalKVConnector：约 150--250 行。
- PDEngineProcess：约 150--250 行。
- PDCoordinator：约 100--200 行。
- PD 数据类和控制面合计尽量控制在约 400--700 行。
- 不含迁移的模型/layer，myvllm 自有核心尽量控制在约 2,500--4,000 行。

规模目标用于提醒职责边界，不为了压行数牺牲核心状态正确性。
