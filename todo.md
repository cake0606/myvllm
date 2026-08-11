# myvllm 轻量开发计划

## 1. 项目目标

myvllm 以 nano-vLLM 为计算实现基线，以 vLLM V1 为调度和 KV cache
设计参考，完成：

- 正确、可测试的离线推理。
- Continuous batching。
- 可与 decode 混合调度的 chunked prefill。
- 增量式 paged KV cache 和 prefix cache。
- Prefill/Decode（PD）分离。
- 可选的外部 C++ KV cache/storage/transfer engine。

项目不复制 vLLM V1 的多进程 worker 协议、cache group 层次和通用插件
体系。模型、layer 和 loader 优先迁移 nano-vLLM 的现有实现，不重复实现一套
dense 模型。

## 2. 轻量设计原则

- 单进程、单 GPU 的完整路径优先；真正跨进程时才增加序列化 DTO。
- `Request` 使用 `num_computed_tokens` 表示唯一计算进度。
- prefill/decode 由请求进度推导，不保存可变的 `is_prefill` 状态。
- scheduler 只返回本轮执行的请求、token 数、block IDs 和采样标记。
- Engine 持有并更新 Request；Scheduler 只决策，ModelRunner 只执行计算。
- KV cache 的逻辑 block、请求 block table 和 GPU tensor storage 分离。
- 模型层不读取全局 `Context`；执行 metadata 由 ModelRunner 显式构造。
- 一种正确实现跑通后再抽象 backend，不提前搭建注册或插件体系。
- 不实现随后会被 paged KV/mixed batch 删除的临时无缓存 decode 路径。
- 不在当前目标中加入稀疏执行框架。

核心链路：

```text
LLMEngine
├── Request table
├── Scheduler
│   └── KVCacheManager
│       ├── BlockPool / PrefixCache
│       └── request_id -> block table
└── ModelRunner
    ├── KVCacheStorage
    └── Qwen3 / Attention
```

单进程核心协议：

```text
Request
  -> SchedulerOutput
  -> ModelRunner
  -> sampled token IDs
  -> Engine 更新 Request
  -> RequestOutput
```

## 3. 开发阶段

### 阶段 0：冻结 nano-vLLM/Hugging Face 基线

- [ ] 记录 nano-vLLM commit、本地 patch、模型 revision 和依赖版本。
- [ ] 固定 greedy 输入、随机种子和期望 token IDs。
- [ ] 保存 Hugging Face 与 nano-vLLM 的首 token 和完整输出。
- [ ] 保存吞吐、TTFT、TPOT 和峰值显存基线。

完成标准：可以独立于 myvllm 重复获得参考输出和性能数据。

### 阶段 1：最小工程骨架和核心数据结构

- [x] 配置 `pyproject.toml`、uv、src layout、pytest、ruff 和 mypy。
- [x] 实现 `SamplingParams`。
- [x] 实现 `ModelConfig`、`SchedulerConfig`、`CacheConfig` 和
  `ParallelConfig`。
- [x] 实现 `RequestStatus` 和 `Request`。
- [x] 使用 `num_computed_tokens` 作为唯一计算进度。
- [x] 实现只描述格式、不分配 tensor 的 `KVCacheSpec`。
- [x] 实现轻量 `ScheduledRequest` 和 `SchedulerOutput`。
- [x] 实现稳定的 `RequestOutput`。
- [x] 导出阶段 1 的公共类型。
- [x] 为上述类型提供无 GPU 单元测试。

阶段 1 明确不实现：

- scheduler/worker 首次全量、后续增量同步协议。
- `NewRequestData`、`CachedRequestUpdate`、`WorkerRequestState`。
- KV connector、PD 状态机和 worker RPC。
- 多 attention backend 注册和稀疏框架。

完成标准：核心类型和状态转换在无 GPU 环境通过 pytest、ruff 和 mypy。

当前检查点：143 个测试通过，ruff 和 mypy 通过。

### 阶段 2：完整单进程 Engine

阶段 2 合并原计划中的模型迁移、paged KV cache、continuous batching 和
chunked prefill。目标不是先构造只能生成首 token 的临时链路，而是按依赖顺序
完成可持续运行的单进程 Engine。

实现顺序：

```text
Block metadata / free queue
  -> BlockHashToBlockMap / BlockPool
  -> KVCacheManager / PrefixCache / KVCacheStorage
  -> nano-vLLM layers / Qwen3 / loader
  -> Attention metadata / ModelRunner
  -> Scheduler
  -> LLMEngine / RequestOutput
```

#### 2.1 物理 block pool

- [x] 实现 `KVCacheBlock`：block ID、ref count、hash 和 hash token boundary。
- [x] 实现 `FreeKVCacheBlockQueue`：
  - [x] 双向链表和 head/tail sentinel。
  - [x] `popleft`、`popleft_n`、`remove`。
  - [x] `append`、`append_n`、`prepend_n`。
  - [x] 保持 eviction order，并支持 prefix hit 时 O(1) 中间删除。
  - [x] 覆盖空队列、批量顺序、指针一致性和重复入队测试。
- [ ] 实现 `BlockHashToBlockMap`：允许相同 hash 对应多个物理 block。
- [ ] 实现 `BlockPool`：
  - [ ] 初始化全部物理 block 和 free queue。
  - [ ] `get_new_blocks(num_blocks)`。
  - [ ] `touch(blocks)`，命中 free cached block 时从 queue 移除。
  - [ ] `free_blocks(blocks)`，引用归零后按 cache 价值重新入队。
  - [ ] 注册、查找和驱逐 block hash。
  - [ ] 重用物理 block 前移除旧 prefix hash。
  - [ ] free block 数量、usage 和 reset prefix cache。
- [ ] 为 duplicate hash、共享引用、重复释放、LRU 顺序和 hash eviction
  编写无 GPU 测试。

本阶段不迁移 vLLM V1 的 null block、KV cache group、Mamba、KV events、
metrics collector、offload 和 copy-on-write hash move。

#### 2.2 请求级 KV cache 管理

- [ ] 在 `KVCacheSpec` 中增加 dtype item size、单 block 字节数和总容量纯计算。
- [ ] 实现 `KVCacheManager`，权威持有 `request_id -> block table`。
- [ ] 实现 `allocate_slots(request, num_scheduled_tokens)`：只增量分配本轮需要的
  block。
- [ ] 实现 `get_block_table(request_id)`、`free(request_id)` 和
  `preempt(request_id)`。
- [ ] KV 不足时分配必须原子失败，不能留下部分 block table。
- [ ] terminal、abort、failed 和 preempted 路径最终释放引用。
- [ ] 为跨 block、共享 block、增量 decode、抢占和无泄漏编写测试。

#### 2.3 Prefix cache

- [ ] 使用稳定 bytes hash，输入至少包含 parent hash、当前完整 block token IDs、
  model revision 和 cache format。
- [ ] Request 或独立 hasher 生成链式 block hashes；BlockPool 只保存和索引 hash。
- [ ] 只将完整 block 注册为第一版 prefix entry。
- [ ] 查找最长可复用 prefix，并通过 `BlockPool.touch()` 获取引用。
- [ ] free cached block 保留 hash 和 KV 内容；物理 block 被重新分配时才驱逐。
- [ ] 增加 miss、完整命中、部分 block 不命中、重复 hash 和 eviction 测试。

第一版不实现 vLLM V1 的多 cache group 和一个物理 block 的多个 partial hash。

#### 2.4 GPU KV storage

- [ ] 实现 `KVCacheStorage`，分配实际 KV tensor：

```text
[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

- [ ] storage 只按 layer/block 提供 view，不保存 Request 或 block table。
- [ ] 将每层 attention 绑定到对应 K/V view。
- [ ] 提供写入 slot 和按 block table 读取 KV 的 correctness 路径。
- [ ] storage 的 block 数量和 layout 必须与 `KVCacheSpec/BlockPool` 一致。

#### 2.5 迁移 nano-vLLM 计算组件

- [ ] 保留 nano-vLLM 来源说明和许可证。
- [ ] 迁移 `activation.py`、`linear.py`、`layernorm.py`、
  `rotary_embedding.py`、`embed_head.py` 和 `sampler.py`。
- [ ] 迁移 `models/qwen3.py` 和 `utils/loader.py`。
- [ ] 保留 tensor-parallel linear、packed QKV、merged gate/up 和权重加载逻辑。
- [ ] distributed 未初始化时支持 rank 0/world size 1。
- [ ] LM head 不读取全局 prefill 状态；ModelRunner 显式选择 logits rows。
- [ ] sampler 提供确定性的 greedy 路径。
- [ ] 不迁移 nano-vLLM 的 `Sequence`、旧 Scheduler、BlockManager、全局
  `Context`、worker RPC 和 CUDA graph 控制逻辑。

#### 2.6 Attention metadata 和 ModelRunner

- [ ] 定义一次执行真正需要的 metadata：
  - [ ] flattened `input_ids` 和 `positions`。
  - [ ] `query_start_loc`、sequence lengths 和 computed token counts。
  - [ ] `slot_mapping` 和 padded block tables。
  - [ ] `logits_indices` 和 sampled request IDs。
- [ ] 实现统一 `prepare_batch()`，同时接受 full/partial prefill 和 decode。
- [ ] partial prefill 中间 chunk 不产生 logits row。
- [ ] 最后一个 prompt chunk 和 decode 请求只选择各自最后一个 logits row。
- [ ] ModelRunner 不修改 Request，不维护重复的 worker request state。
- [ ] 实现 correctness attention；正确后再接 FlashAttention/Triton 优化路径。
- [ ] mixed batch 输出必须与逐请求执行一致。

#### 2.7 Continuous batching 和 chunked prefill Scheduler

- [ ] Scheduler 维护 waiting/running 队列。
- [ ] 使用统一 `max_num_batched_tokens` budget，同时限制 `max_num_seqs` 和
  可用 KV blocks。
- [ ] running/decode 请求优先推进，再使用剩余 budget 接纳新 prefill。
- [ ] 使用统一公式：

```text
remaining = request.num_tokens - request.num_computed_tokens
scheduled = min(remaining, request_chunk_limit, remaining_step_budget)
```

- [ ] decode 通常调度 1 token，prefill 调度一个 chunk。
- [ ] `should_sample` 只在本轮计算到当前请求尾部时为真。
- [ ] Scheduler 调用 KVCacheManager 分配 block，但不推进 Request 进度。
- [ ] 支持动态加入、abort、KV 不足时的 preemption 和恢复。
- [ ] 防止持续到来的 prefill 饿死 decode。
- [ ] 为 token budget、mixed batch、chunk boundary、KV 不足和队列公平性编写
  无 GPU 测试。

#### 2.8 LLMEngine 和端到端生成

- [ ] Engine 持有 Request table、Scheduler、KVCacheManager 和 ModelRunner。
- [ ] 模型执行成功后才推进 `num_computed_tokens` 并 append sampled token。
- [ ] 处理 EOS、stop token、`max_tokens`、abort、failed 和 finished。
- [ ] 所有 terminal 路径释放 KV blocks。
- [ ] 返回稳定的累计 `RequestOutput`。
- [ ] 完成单请求多 token 和多请求动态 batch 生成。
- [ ] greedy 输出与 Hugging Face/nano-vLLM reference 一致。
- [ ] chunked prefill 输出与完整 prefill 一致。
- [ ] mixed batch 输出与逐请求执行一致。

阶段 2 完成标准：单进程 Engine 可以持续接收请求，通过增量 paged KV cache
执行 mixed prefill/decode batch，完成多 token 生成；不同 chunk size 和 batch
组合的 greedy 输出与 reference 一致，且所有终止路径无 block 泄漏。

### 阶段 3：稳定性和性能基线

- [ ] 增加长时间运行、abort 竞态、OOM、preemption 和 KV 泄漏测试。
- [ ] 记录 TTFT、TPOT、吞吐、p50/p95/p99 和峰值显存。
- [ ] 记录每步 prefill/decode token 数、KV usage、prefix hit rate 和 preemption。
- [ ] 添加 FlashAttention/Triton KV 写入路径，并保留 correctness fallback。
- [ ] 正确后再优化 CUDA graph、pinned buffer 和异步 metadata copy。

完成标准：核心状态机可长期运行且可观测，性能优化不改变公共协议和输出。

### 阶段 4：PD 分离

- [ ] 增加 `PREFILL`、`DECODE`、`BOTH` engine role。
- [ ] 此时才定义 `KVTransferConfig` 和 `KVTransferMetadata`。
- [ ] 定义最小 `KVConnector`：save、load、poll、cancel。
- [ ] producer save 完成前保持 block 引用；consumer load 完成前不可执行。
- [ ] 先用进程内/local connector 验证状态机。
- [ ] 再实现独立 P/D engine 和 CPU/GPU transfer backend。
- [ ] 验证 abort、失败、超时和所有 block 回收路径。

完成标准：P 生成 prompt KV 和首 token，D 加载后继续 decode，输出与单
engine 一致。

### 阶段 5：外部 C++ KV engine

- [ ] 保持 Python Scheduler 拥有 Request 和逻辑 block table。
- [ ] 定义窄接口 `KVCacheBackend`，隔离 allocate/view/export/import/transfer。
- [ ] 保留 Python/Torch backend 作为 correctness reference。
- [ ] 通过 pybind11、nanobind 或稳定 C ABI 接入 C++ backend。
- [ ] 用 `KVCacheSpec` 和 model revision 做握手兼容性校验。
- [ ] 异步完成通过 handle + poll/callback 表达，不把 C++ 状态暴露给 Request。

完成标准：切换 C++ backend 不修改 Scheduler、Request、ModelRunner 和
Attention 的公共协议。

## 4. 全局正确性要求

```text
0 <= request.num_computed_tokens <= request.num_tokens
prefill := num_computed_tokens < num_prompt_tokens
decode  := num_computed_tokens >= num_prompt_tokens
```

- free queue 中的真实 block 必须满足 `ref_cnt == 0`。
- `ref_cnt > 0` 的 block 不能进入 free queue。
- 同一物理 block 不能重复入队或重复释放。
- free cached block 可以保留 hash；物理 block 重用前必须先驱逐 hash。
- Scheduler 不推进 Request；Engine 只在执行成功后推进进度。
- greedy 输出与 Hugging Face/nano-vLLM reference 一致。
- chunked prefill 输出与完整 prefill 一致。
- mixed batch 输出与逐请求串行输出一致。
- PD 输出与单 engine 输出一致。
- 完成、抢占、取消和失败最终都释放 KV block。

## 5. 规模控制

- 阶段 1 核心源码目标：约 300--500 行 Python。
- 单进程 Scheduler/cache/engine 自有代码目标：约 1,200--2,000 行。
- PD Python 控制面与 connector 目标：约 300--600 行。
- 不含迁移的模型/layer 代码，myvllm 自有 Python 核心目标控制在
  2,500--4,000 行。
- 不为了对齐 vLLM V1 而提前加入多 cache group、Mamba、通用插件和多进程
  worker 协议。
