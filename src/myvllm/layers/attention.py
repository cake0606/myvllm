import torch
import triton
import triton.language as tl
from flash_attn import (
        flash_attn_varlen_func, 
        flash_attn_with_kvcache,
        )
from torch import nn

from myvllm.attention.metadata import AttentionMetadata

@triton.jit
def _store_kv_cache_kernel(
        key_ptr, 
        )
