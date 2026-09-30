"""
Qwen3.5 (qwen3_5) linear-attention custom-op extraction.

Extracts GatedDeltaRule and ShortConv from the inlined free-function logic in
transformers `models/qwen3_5/modeling_qwen3_5.py` into standalone nn.Modules so
they can be exported via pnnx `moduleop=GatedDeltaRule,ShortConv` as unexpanded
custom ops (matching the C++ runtime contract in src/kernel/gdr.cpp + shortconv.cpp).

The torch math below is copied verbatim from the reference modeling file so that a
refactored layer is numerically identical to the original (validated by
qwen35_test_parity.py). During pnnx export these modules become black-box custom
ops; the C++ runtime re-implements the same math.

Input/output contract (must match assets/qwen3.5_0.8b decoder param):
    ShortConv.forward(weight, mixed_qkv, initial_state) -> (out, new_state)
      weight      : conv1d weight [conv_dim, 1, kernel]  (registered param -> MemoryData blob)
      mixed_qkv   : [B, conv_dim, T]
      initial_state: [B, conv_dim, kernel]
    GatedDeltaRule.forward(A_log, dt_bias, b, a, query, key, value, initial_state) -> (out, new_state)
      A_log,dt_bias: buffers [num_v_heads]  (MemoryData blobs, come first)
      b,a         : [B, T, num_v_heads]
      query,key   : [B, T, num_k_heads, head_k_dim]
      value       : [B, T, num_v_heads, head_v_dim]
      initial_state: [B, num_v_heads, head_k_dim, head_v_dim]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- copied verbatim from transformers models/qwen3_5/modeling_qwen3_5.py ----
def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    if query.dtype == torch.float32:
        query, key, value, beta, g = [x.transpose(1, 2).contiguous()
                                      for x in (query, key, value, beta, g)]
    else:
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32)
            for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous()
    if initial_dtype != torch.float32:
        core_attn_out = core_attn_out.to(initial_dtype)
    return core_attn_out, last_recurrent_state


class ShortConv(nn.Module):
    """Causal grouped conv1d + SiLU, with explicit conv state.

    Matches the model's `F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])` for prefill
    and the `causal_conv1d_update` recurrence for decode, unified via left-concat of
    the state (as the exported graph does: conv_in = cat(state, mixed_qkv)).
    """

    def __init__(self, conv_weight, kernel_size):
        super().__init__()
        # Keep the HF depthwise-convolution layout [conv_dim, 1, kernel].
        # Accept a squeezed [conv_dim, kernel] tensor as well for old callers.
        weight = conv_weight.detach().clone()
        if weight.ndim == 2:
            weight = weight.unsqueeze(1)
        if weight.ndim != 3:
            raise ValueError(f"expected conv weight with 2 or 3 dims, got {tuple(weight.shape)}")
        self.weight = nn.Parameter(weight)
        self.kernel_size = kernel_size
        self.conv_dim = conv_weight.shape[0]

    def forward(self, weight, mixed_qkv, initial_state):
        # NCNN stores a full kernel-sized state and ignores its oldest sample.
        conv_in = torch.cat([initial_state, mixed_qkv], dim=-1)  # [B, conv_dim, kernel+T]
        out = F.conv1d(conv_in, weight, None, padding=0, groups=self.conv_dim)
        out = F.silu(out[:, :, -mixed_qkv.shape[-1]:])
        new_state = conv_in[:, :, -self.kernel_size:]  # [B, conv_dim, kernel]
        return out, new_state


class GatedDeltaRule(nn.Module):
    """Gated delta rule linear attention with explicit recurrent state."""

    def __init__(self, A_log, dt_bias, head_k_dim, head_v_dim, num_k_heads, num_v_heads, eps=1e-6):
        super().__init__()
        self.A_log = nn.Parameter(A_log.detach().clone())
        self.dt_bias = nn.Parameter(dt_bias.detach().clone())
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.eps = eps

    def forward(self, A_log, dt_bias, b, a, query, key, value, initial_state):
        beta = b.sigmoid()
        if A_log.dtype == torch.float32:
            g = -A_log.exp() * F.softplus(a + dt_bias)
        else:
            g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
        # Derive the GQA ratio from the actual tensors. Some Transformers
        # revisions expose stale num_v_heads metadata even though in_proj_b/a
        # and value carry the correct runtime head count.
        num_k_heads = query.shape[2]
        num_v_heads = value.shape[2]
        if num_k_heads <= 0 or num_v_heads <= 0 or num_v_heads % num_k_heads != 0:
            raise ValueError(
                f"GDR head count mismatch: query={num_k_heads}, value={num_v_heads}")
        if query.shape[-1] != key.shape[-1]:
            raise ValueError("GDR query/key head dimensions must match")
        if num_v_heads != num_k_heads:
            repeat = num_v_heads // num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        return core_attn_out, last_recurrent_state
