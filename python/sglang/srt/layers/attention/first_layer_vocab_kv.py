"""Serve the first attention layer's K/V from a vocabulary table instead of the KV pool.

Everything before the first cross-token operation is tokenwise:

    K_unrot[t] = k_norm(W_K RMSNorm(E[t])),   V[t] = W_V RMSNorm(E[t])

so both are functions of the token id alone. With
`SGLANG_FIRST_LAYER_VOCAB_KV=1` the model builds [V, Hkv, d] tables once after weight loading,
the KV pool allocates no layer-0 K/V (only one int32 token id per slot), and layer-0 attention
reads table rows by token id and applies RoPE per key position on read:

* decode: a split-KV Triton kernel walks req_to_token -> slot -> token id -> table row;
* extend: the visible keys are materialized post-RoPE into a temporary buffer and passed to
  FlashAttention-4 varlen (bottom-right causal), then freed.

Scope (enforced by 'check_first_layer_vocab_kv_server_args' and the pool configurator):
text-only Qwen3ForCausalLM on CUDA with an unquantized bf16 KV cache, standard RoPE whose
key position equals the index within the request, and no feature that stores, transfers, or
re-derives layer-0 KV elsewhere (HiCache, PD disaggregation, speculative decoding, LoRA,
pipeline parallelism, DP attention, dLLM, HND layout, prefill CUDA graphs). Only attention
TP == 1 has been tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs

SUPPORTED_ARCHITECTURES = ("Qwen3ForCausalLM",)


def first_layer_vocab_kv_enabled() -> bool:
    return envs.SGLANG_FIRST_LAYER_VOCAB_KV.get()


def uses_vocab_kv(layer_id: int) -> bool:
    return layer_id == 0 and first_layer_vocab_kv_enabled()


def check_first_layer_vocab_kv_server_args(cfg: Any) -> None:
    """Reject server features that would read, write, move, or ship layer-0 KV slots."""
    if not first_layer_vocab_kv_enabled():
        return
    conflicts = {
        "--device cuda": cfg.device not in ("cuda", None),
        "no --speculative-algorithm": cfg.speculative_algorithm is not None,
        "no --enable-hierarchical-cache": cfg.enable_hierarchical_cache,
        "no --disaggregation-mode": cfg.disaggregation_mode not in (None, "null"),
        "--pp-size 1": cfg.pp_size != 1,
        "no --enable-lora": bool(cfg.enable_lora),
        "no --quantization": cfg.quantization is not None,
        "--kv-cache-dtype auto/bf16": cfg.kv_cache_dtype
        not in ("auto", "bf16", "bfloat16"),
        "no --dllm-algorithm": cfg.dllm_algorithm is not None,
        "--disable-prefill-cuda-graph": not cfg.disable_prefill_cuda_graph,
        "no SGLANG_USE_HND_KVCACHE": envs.SGLANG_USE_HND_KVCACHE.get(),
    }
    violated = [need for need, bad in conflicts.items() if bad]
    if violated:
        raise ValueError(
            "SGLANG_FIRST_LAYER_VOCAB_KV=1 requires: " + ", ".join(violated)
        )


def check_first_layer_vocab_kv_model(architectures: list[str] | None) -> None:
    if first_layer_vocab_kv_enabled() and list(architectures or []) != list(
        SUPPORTED_ARCHITECTURES
    ):
        raise ValueError(
            f"SGLANG_FIRST_LAYER_VOCAB_KV=1 supports {SUPPORTED_ARCHITECTURES}, "
            f"got {architectures}"
        )


@triton.jit
def _vocab_decode_stage1(
    Q,
    KTab,
    VTab,
    ReqToToken,
    ReqPoolIdx,
    SeqLens,
    SlotTokens,
    CosSin,
    PartialO,
    PartialLse,
    stride_qb,
    stride_qh,
    stride_kr,
    stride_kh,
    stride_vr,
    stride_vh,
    stride_r2t,
    stride_cs,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    sm_scale,
    GROUP: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)

    req = tl.load(ReqPoolIdx + b).to(tl.int64)
    seq_len = tl.load(SeqLens + b)
    split_len = tl.cdiv(seq_len, NUM_SPLITS)
    start = split * split_len
    end = tl.minimum(start + split_len, seq_len)

    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < GROUP
    heads = kvh * GROUP + offs_h
    offs_d = tl.arange(0, HALF)
    q_ptr = Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :]
    q1 = tl.load(q_ptr, mask=h_mask[:, None], other=0.0)
    q2 = tl.load(q_ptr + HALF, mask=h_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, HALF], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, HALF], dtype=tl.float32)

    for start_n in range(start, end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < end
        slots = tl.load(
            ReqToToken + req * stride_r2t + offs_n, mask=n_mask, other=0
        ).to(tl.int64)
        tok = tl.load(SlotTokens + slots, mask=n_mask, other=0).to(tl.int64)

        k_ptr = KTab + tok[:, None] * stride_kr + kvh * stride_kh + offs_d[None, :]
        k1 = tl.load(k_ptr, mask=n_mask[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + HALF, mask=n_mask[:, None], other=0.0).to(tl.float32)
        cs_ptr = CosSin + offs_n[:, None] * stride_cs + offs_d[None, :]
        cos = tl.load(cs_ptr, mask=n_mask[:, None], other=0.0).to(tl.float32)
        sin = tl.load(cs_ptr + HALF, mask=n_mask[:, None], other=0.0).to(tl.float32)
        k1r = (k1 * cos - k2 * sin).to(q1.dtype)
        k2r = (k2 * cos + k1 * sin).to(q1.dtype)

        s = tl.dot(q1, tl.trans(k1r)) + tl.dot(q2, tl.trans(k2r))
        s = tl.where(n_mask[None, :], s * sm_scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptr = VTab + tok[:, None] * stride_vr + kvh * stride_vh + offs_d[None, :]
        v1 = tl.load(v_ptr, mask=n_mask[:, None], other=0.0)
        v2 = tl.load(v_ptr + HALF, mask=n_mask[:, None], other=0.0)
        p = p.to(v1.dtype)
        acc1 = acc1 * alpha[:, None] + tl.dot(p, v1)
        acc2 = acc2 * alpha[:, None] + tl.dot(p, v2)
        m_i = m_new

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_ptr = (
        PartialO
        + b * stride_ob
        + heads[:, None] * stride_oh
        + split * stride_os
        + offs_d[None, :]
    )
    tl.store(o_ptr, acc1 / l_safe[:, None], mask=h_mask[:, None])
    tl.store(o_ptr + HALF, acc2 / l_safe[:, None], mask=h_mask[:, None])
    lse = tl.where(end > start, m_i + tl.log(l_safe), float("-inf"))
    tl.store(PartialLse + b * stride_lb + heads * stride_lh + split, lse, mask=h_mask)


@triton.jit
def _vocab_decode_stage2(
    PartialO,
    PartialLse,
    O,
    stride_ob,
    stride_oh,
    stride_os,
    stride_lb,
    stride_lh,
    stride_out_b,
    stride_out_h,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)
    s_mask = offs_s < NUM_SPLITS
    offs_d = tl.arange(0, HEAD_DIM)
    lse = tl.load(
        PartialLse + b * stride_lb + h * stride_lh + offs_s,
        mask=s_mask,
        other=float("-inf"),
    )
    m = tl.max(lse, 0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse - m)
    denom = tl.sum(w, 0)
    po = tl.load(
        PartialO
        + b * stride_ob
        + h * stride_oh
        + offs_s[:, None] * stride_os
        + offs_d[None, :],
        mask=s_mask[:, None],
        other=0.0,
    )
    out = tl.sum(po * w[:, None], 0) / tl.where(denom > 0, denom, 1.0)
    tl.store(
        O + b * stride_out_b + h * stride_out_h + offs_d, out.to(O.dtype.element_ty)
    )


@triton.jit
def _gather_rope_kv(
    KTab,
    VTab,
    ReqToToken,
    SlotTokens,
    CosSin,
    KeyReq,
    KeyPos,
    KOut,
    VOut,
    stride_kr,
    stride_vr,
    stride_r2t,
    stride_cs,
    num_keys,
    KV_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Materialize post-RoPE K and V rows for (request, position) pairs."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_keys
    req = tl.load(KeyReq + offs, mask=mask, other=0).to(tl.int64)
    pos = tl.load(KeyPos + offs, mask=mask, other=0).to(tl.int64)
    slots = tl.load(ReqToToken + req * stride_r2t + pos, mask=mask, other=0).to(
        tl.int64
    )
    tok = tl.load(SlotTokens + slots, mask=mask, other=0).to(tl.int64)
    offs_d = tl.arange(0, HALF)
    cos = tl.load(
        CosSin + pos[:, None] * stride_cs + offs_d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        CosSin + pos[:, None] * stride_cs + HALF + offs_d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    row_out = offs.to(tl.int64)[:, None] * (KV_HEADS * 2 * HALF)
    for h in tl.static_range(KV_HEADS):
        k_ptr = KTab + tok[:, None] * stride_kr + h * 2 * HALF + offs_d[None, :]
        k1 = tl.load(k_ptr, mask=mask[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(k_ptr + HALF, mask=mask[:, None], other=0.0).to(tl.float32)
        out_ptr = KOut + row_out + h * 2 * HALF + offs_d[None, :]
        tl.store(
            out_ptr, (k1 * cos - k2 * sin).to(KOut.dtype.element_ty), mask=mask[:, None]
        )
        tl.store(
            out_ptr + HALF,
            (k2 * cos + k1 * sin).to(KOut.dtype.element_ty),
            mask=mask[:, None],
        )
        v_ptr = VTab + tok[:, None] * stride_vr + h * 2 * HALF + offs_d[None, :]
        v_out = VOut + row_out + h * 2 * HALF + offs_d[None, :]
        tl.store(
            v_out, tl.load(v_ptr, mask=mask[:, None], other=0.0), mask=mask[:, None]
        )
        tl.store(
            v_out + HALF,
            tl.load(v_ptr + HALF, mask=mask[:, None], other=0.0),
            mask=mask[:, None],
        )


@dataclass
class VocabKVTables:
    k: torch.Tensor  # [V, Hkv, d], after k_norm, before RoPE
    v: torch.Tensor  # [V, Hkv, d]

    def nbytes(self) -> int:
        return (
            self.k.numel() * self.k.element_size()
            + self.v.numel() * self.v.element_size()
        )


class FirstLayerVocabKV:
    """Layer-0 attention over vocabulary tables; owned by the model's layer-0 attention module."""

    def __init__(
        self, num_heads: int, num_kv_heads: int, head_dim: int, scaling: float
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.tables: VocabKVTables | None = None
        self._num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

    @torch.no_grad()
    def build(
        self, project_kv, vocab_size: int, device: torch.device, chunk: int = 16384
    ) -> None:
        """`project_kv(token_ids) -> (k [n, Hkv, d], v [n, Hkv, d])` must be the model's exact
        layer-0 path (embedding -> input norm -> K/V projection -> k_norm)."""
        ks, vs = [], []
        for start in range(0, vocab_size, chunk):
            ids = torch.arange(start, min(start + chunk, vocab_size), device=device)
            k, v = project_kv(ids)
            ks.append(k.reshape(-1, self.num_kv_heads, self.head_dim))
            vs.append(v.reshape(-1, self.num_kv_heads, self.head_dim))
        k, v = torch.cat(ks), torch.cat(vs)
        if self.tables is None:
            self.tables = VocabKVTables(k.contiguous(), v.contiguous())
        else:
            # Weight updates rebuild in place: captured CUDA graphs hold the table pointers.
            self.tables.k.copy_(k)
            self.tables.v.copy_(v)

    @staticmethod
    def record_tokens(forward_batch, slot_tokens: torch.Tensor) -> None:
        slot_tokens[forward_batch.out_cache_loc] = forward_batch.input_ids.to(
            torch.int32
        )

    def forward(
        self,
        q: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        forward_batch,
        req_to_token: torch.Tensor,
        slot_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """q: [N, Hq * d] already rotated. Returns [N, Hq * d]."""
        assert self.tables is not None, "vocab KV tables were not built"
        self.record_tokens(forward_batch, slot_tokens)
        q = q.view(-1, self.num_heads, self.head_dim)
        mode = forward_batch.forward_mode
        if mode.is_decode():
            out = self._decode(
                q, cos_sin_cache, forward_batch, req_to_token, slot_tokens
            )
        elif mode.is_extend():
            out = self._extend(
                q, cos_sin_cache, forward_batch, req_to_token, slot_tokens
            )
        else:
            raise NotImplementedError(f"first-layer vocab KV does not support {mode}")
        return out.reshape(-1, self.num_heads * self.head_dim)

    def _decode(self, q, cos_sin_cache, forward_batch, req_to_token, slot_tokens):
        batch = q.shape[0]
        hkv, d = self.num_kv_heads, self.head_dim
        group = self.num_heads // hkv
        # Fixed per batch size so the launch is CUDA-graph capturable for any sequence length.
        splits = max(1, min(triton.cdiv(4 * self._num_sms, batch * hkv), 64))
        partial_o = torch.empty(
            batch, self.num_heads, splits, d, device=q.device, dtype=torch.float32
        )
        partial_lse = torch.empty(
            batch, self.num_heads, splits, device=q.device, dtype=torch.float32
        )
        out = torch.empty_like(q)
        k_tab, v_tab = self.tables.k, self.tables.v
        _vocab_decode_stage1[(batch, hkv, splits)](
            q,
            k_tab,
            v_tab,
            req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            slot_tokens,
            cos_sin_cache,
            partial_o,
            partial_lse,
            q.stride(0),
            q.stride(1),
            k_tab.stride(0),
            k_tab.stride(1),
            v_tab.stride(0),
            v_tab.stride(1),
            req_to_token.stride(0),
            cos_sin_cache.stride(0),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            self.scaling,
            GROUP=group,
            BLOCK_H=max(16, triton.next_power_of_2(group)),
            HALF=d // 2,
            BLOCK_N=64,
            NUM_SPLITS=splits,
            num_warps=4,
            num_stages=2,
        )
        _vocab_decode_stage2[(batch, self.num_heads)](
            partial_o,
            partial_lse,
            out,
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            out.stride(0),
            out.stride(1),
            HEAD_DIM=d,
            NUM_SPLITS=splits,
            BLOCK_S=triton.next_power_of_2(splits),
            num_warps=4,
        )
        return out

    def _extend(self, q, cos_sin_cache, forward_batch, req_to_token, slot_tokens):
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            flash_attn_varlen_func,
        )

        device = q.device
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        extend_lens = forward_batch.extend_seq_lens_cpu
        num_keys = sum(seq_lens)
        lens_k = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        lens_q = torch.tensor(extend_lens, dtype=torch.int32, device=device)
        cu_k = torch.nn.functional.pad(lens_k.cumsum(0, dtype=torch.int32), (1, 0))
        cu_q = torch.nn.functional.pad(lens_q.cumsum(0, dtype=torch.int32), (1, 0))
        key_req = torch.repeat_interleave(
            forward_batch.req_pool_indices, lens_k, output_size=num_keys
        )
        key_pos = torch.arange(
            num_keys, device=device, dtype=torch.int32
        ) - torch.repeat_interleave(cu_k[:-1], lens_k, output_size=num_keys)
        hkv, d = self.num_kv_heads, self.head_dim
        k = torch.empty(num_keys, hkv, d, device=device, dtype=q.dtype)
        v = torch.empty_like(k)
        block = 64
        _gather_rope_kv[(triton.cdiv(num_keys, block),)](
            self.tables.k,
            self.tables.v,
            req_to_token,
            slot_tokens,
            cos_sin_cache,
            key_req,
            key_pos,
            k,
            v,
            self.tables.k.stride(0),
            self.tables.v.stride(0),
            req_to_token.stride(0),
            cos_sin_cache.stride(0),
            num_keys,
            KV_HEADS=hkv,
            HALF=d // 2,
            BLOCK=block,
            num_warps=4,
        )
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max(extend_lens),
            max_seqlen_k=max(seq_lens),
            softmax_scale=self.scaling,
            causal=True,
        )
        return out[0] if isinstance(out, tuple) else out
