import itertools
import os
from typing import Optional

import torch
import triton
import triton.testing
import sgl_kernel  # noqa: F401

from jit_kernel.topk_indexer import (
    fast_topk_transform_fused_v3,
    fast_topk_transform_ragged_fused_v3,
)

SEED = 42
MAX_SEQ_LEN = 131072

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    topk_indices_offset: Optional[torch.Tensor] = None,
    max_permit_error: int = 0,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()

    wrong_values = 0
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i

        offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0

        if len(more) > 0 or len(less) > 0:
            more_values = sorted(score[i, idx - offset].item() for idx in more)
            less_values = sorted(score[i, idx - offset].item() for idx in less)
            if more_values != less_values:
                wrong_values += len(more)
                print(
                    f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, "
                    f"with {more_values=}, {less_values=}"
                )
        assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


def _normalize_score(score: torch.Tensor) -> torch.Tensor:
    score_max = score.max()
    score_min = score.min()
    return (score - score_min) / (score_max - score_min + 1e-6) * 255


def _make_score(bs: int, seq_len: int, has_row_starts: bool) -> torch.Tensor:
    shape = (bs, MAX_SEQ_LEN) if has_row_starts else (bs, seq_len)
    return _normalize_score(torch.randn(*shape, dtype=torch.float32, device="cuda"))


def _make_paged_inputs(bs: int, seq_len: int, mode: str):
    has_row_starts = mode == "extend"
    score = _make_score(bs, seq_len, has_row_starts)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
        if has_row_starts
        else None
    )

    if mode == "decode":
        prefill_bs = bs
        cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    else:
        prefill_bs = max(1, bs // 4)
        boundaries = torch.linspace(
            0, bs, prefill_bs + 1, dtype=torch.float32, device="cuda"
        ).to(torch.int32)
        boundaries[-1] = bs
        cu_seqlens_q = boundaries

    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(prefill_bs, -1)
    return score, lengths, src_page_table, cu_seqlens_q, row_starts


def _make_ragged_inputs(bs: int, seq_len: int, has_row_starts: bool):
    score = _make_score(bs, seq_len, has_row_starts)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    row_starts = (
        torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
        if has_row_starts
        else None
    )
    topk_indices_offset = torch.randint(
        0, 1024, (bs,), dtype=torch.int32, device="cuda"
    )
    return score, lengths, topk_indices_offset, row_starts


def sgl_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        score, lengths, dst_page_table, page_table_size_1, cu_seqlens_q, row_starts
    )
    return dst_page_table


def sgl_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
        score, lengths, topk_indices_ragged, topk_indices_offset, row_starts
    )
    return topk_indices_ragged


def calculate_diff(bs: int, k: int, seq_len: int, mode: str):
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    if mode == "ragged":
        score, lengths, topk_indices_offset, row_starts = _make_ragged_inputs(
            bs, seq_len, has_row_starts=True
        )
        indices_ref = sgl_topk_transform_ragged_fused(
            score, lengths, topk_indices_offset, k, row_starts=row_starts
        )
        indices_our = fast_topk_transform_ragged_fused_v3(
            score, lengths, topk_indices_offset, k, row_starts=row_starts
        )
    else:
        score, lengths, src_page_table, cu_seqlens_q, row_starts = _make_paged_inputs(
            bs, seq_len, mode
        )
        topk_indices_offset = None
        indices_ref = sgl_topk_transform_fused(
            score, lengths, src_page_table, cu_seqlens_q, k, row_starts=row_starts
        )
        indices_our = fast_topk_transform_fused_v3(
            score, lengths, src_page_table, cu_seqlens_q, k, row_starts=row_starts
        )

    indices_ref = torch.sort(indices_ref, dim=-1).values
    indices_our = torch.sort(indices_our, dim=-1).values

    assert_equal(
        score,
        indices_ref,
        indices_our,
        bs,
        k,
        seq_len,
        topk_indices_offset=topk_indices_offset,
        max_permit_error=5,
    )


bs = [1, 2, 4, 8]
k = [2048]
seq_len = [
    16384,
    65536,
    98304,
    120000,
]
mode = ["decode", "extend", "target_verify", "ragged"]

configs = list(itertools.product(bs, k, seq_len, mode))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs", "k", "seq_len", "mode"],
        x_vals=configs,
        line_arg="provider",
        line_vals=["sgl", "radix_cluster"],
        line_names=["sgl", "radix_cluster"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Latency",
        plot_name="top2048-transform-performance",
        args={},
    )
)
def benchmark(bs: int, k: int, seq_len: int, mode: str, provider) -> None:
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    quantiles = [0.5, 0.2, 0.8]

    if mode == "ragged":
        score, lengths, topk_indices_offset, row_starts = _make_ragged_inputs(
            bs, seq_len, has_row_starts=True
        )
        if provider == "sgl":
            fn = lambda: sgl_topk_transform_ragged_fused(
                score, lengths, topk_indices_offset, k, row_starts=row_starts
            )
        else:
            fn = lambda: fast_topk_transform_ragged_fused_v3(
                score, lengths, topk_indices_offset, k, row_starts=row_starts
            )
    else:
        score, lengths, src_page_table, cu_seqlens_q, row_starts = _make_paged_inputs(
            bs, seq_len, mode
        )
        if provider == "sgl":
            fn = lambda: sgl_topk_transform_fused(
                score, lengths, src_page_table, cu_seqlens_q, k, row_starts=row_starts
            )
        else:
            fn = lambda: fast_topk_transform_fused_v3(
                score, lengths, src_page_table, cu_seqlens_q, k, row_starts=row_starts
            )

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    test_configs = [configs[0]] if IS_CI else configs
    for cfg in test_configs:
        print(f"cfg : {cfg}")
        calculate_diff(*cfg)

    print("\n" + "=" * 60)
    print("Starting transform performance benchmark...")
    benchmark.run(print_data=True)
