from __future__ import annotations

import functools
from typing import Optional

import torch

USE_TORCH_JIT = True

# TODO (yaikwy) : use tvm ffi load_jit
from torch.utils.cpp_extension import load as load_jit

from jit_kernel.utils import KERNEL_PATH

_CPP_ENTRY = "fast_topk"
_PY_SYMBOL = "fast_topk"

common_cuda_flags = ["-O2"]
if USE_TORCH_JIT:
    major = torch.cuda.get_device_capability()[0]
    if major > 9:
        common_cuda_flags += ["-DENABLE_HOPPER=1", "-arch=sm_90"]

@functools.cache
def _jit_fast_topk_v3_module():
    if USE_TORCH_JIT:
        return load_jit(
            name="topk_indexer_radix",
            sources=[str(KERNEL_PATH / "csrc" / "topk_indexer/topk_indexer_radix.cu")],
            extra_cflags=["-O2"],
            extra_cuda_cflags=common_cuda_flags,
            verbose=True,
        )
    else:
        return load_jit(
            "fast_topk_v3",
            cuda_files=["topk_indexer/topk_indexer_radix.cu"],
            cuda_wrappers=[(_PY_SYMBOL, _CPP_ENTRY)],
        )


def fast_topk_v3(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert (
        topk == 2048
    ), "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2

    # topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    topk_indices = torch.full(
        (score.size(0), topk), -1, dtype=torch.int32, device=score.device
    )

    module = _jit_fast_topk_v3_module()

    module.fast_topk(score, topk_indices, lengths, row_starts)
    return topk_indices


def fast_topk_transform_fused_v3(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert (
        topk == 2048
    ), "fast_topk_transform_fused_v3 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2

    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    module = _jit_fast_topk_v3_module()
    module.fast_topk_transform_fused(
        score, lengths, dst_page_table, page_table_size_1, cu_seqlens_q, row_starts
    )
    return dst_page_table


def fast_topk_transform_ragged_fused_v3(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert (
        topk == 2048
    ), "fast_topk_transform_ragged_fused_v3 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2

    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    module = _jit_fast_topk_v3_module()
    module.fast_topk_transform_ragged_fused(
        score, lengths, topk_indices_ragged, topk_indices_offset, row_starts
    )
    return topk_indices_ragged
