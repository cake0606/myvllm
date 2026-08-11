# myvllm architecture

myvllm is a small, single-process-first extension of nano-vLLM. It borrows
vLLM's request progress and token-budget ideas without reproducing the V1
worker protocol or cache-group hierarchy.

## Core path

```text
LLMEngine
├── Scheduler
│   ├── Request
│   └── KVCacheManager
└── ModelRunner
    └── Attention
```

The scheduler returns only a tuple of `ScheduledRequest` values. A scheduled
request identifies the request, the number of tokens to execute, its current
block IDs, and whether this step should sample. The engine owns the mutable
`Request` table; no duplicate worker-side request state exists in the
single-process implementation.

## KV cache boundary

`KVCacheSpec` is an immutable format description. It contains no tensor and
performs no allocation. The future `KVCacheManager` owns logical block tables,
while a storage/backend object owns tensor allocation and transfer.

This split leaves a narrow integration point for a future C++ KV cache engine:
the backend can change without changing `Request`, `Scheduler`, or
`ScheduledRequest`.

## Deferred boundaries

The following abstractions are introduced only when the feature needs them:

- scheduler/worker serialization: when worker processes or RPC are added;
- KV transfer metadata and connectors: when PD separation is implemented;
- multiple attention backends: after one correctness path works.

Global or writable `is_prefill` state is forbidden. Prefill and decode are
always derived from `num_computed_tokens` and `num_prompt_tokens`.
