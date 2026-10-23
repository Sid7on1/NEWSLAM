"""Correctness tests for SLAM3 (Mac NEWSLAM slam3_pytorch.py, 715 lines).

Run: venv/bin/python -m pytest tests/test_slam3.py -v
Some tests are EXPECTED TO FAIL — they encode properties the architecture
claims (causality, MoE attention routing, sparse compute). A failure is a
finding, reported as-is.
"""
import sys, math
sys.path.insert(0, "code")
import torch
import torch.nn.functional as F
import pytest

from slam3_pytorch_mac import (
    SLAM3Model, SLAM3Config, RoPEPositionalEncoding, apply_rope,
    MoEAttentionModule, SparseTopKMoEFFN, Expert,
)

torch.manual_seed(0)
CFG = SLAM3Config()


@pytest.fixture(scope="module")
def model():
    m = SLAM3Model(CFG)
    m.eval()
    return m


# ---------- RoPE ----------

def test_rope_preserves_norms():
    rope = RoPEPositionalEncoding(32)
    x = torch.randn(2, 40, 4, 32)
    cos, sin = rope(40, x.device)
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_relative_position_property():
    """dot(R_m q, R_n k) must depend only on (m - n)."""
    rope = RoPEPositionalEncoding(32)
    q = torch.randn(32); k = torch.randn(32)
    cos, sin = rope(64, q.device)
    def rot(v, pos):
        return apply_rope(v.view(1, 1, 1, 32), cos[pos:pos+1].unsqueeze(1), sin[pos:pos+1].unsqueeze(1)).flatten()
    scores = {}
    for (m, n) in [(10, 4), (20, 14), (33, 27), (5, 0), (21, 16)]:
        scores.setdefault(m - n, []).append(torch.dot(rot(q, m), rot(k, n)).item())
    for off, vals in scores.items():
        assert max(vals) - min(vals) < 1e-4, f"offset {off}: scores vary {vals}"


# ---------- MoE FFN routing ----------

def test_moe_topk_selects_exactly_k():
    ffn = SparseTopKMoEFFN(128, 256, num_experts=4, top_k=2)
    x = torch.randn(2, 16, 128)
    logits = ffn.gate(x)
    _, idx = torch.topk(logits, ffn.top_k, dim=-1)
    assert idx.shape == (2, 16, 2)
    # every token has exactly k distinct experts
    for b in range(2):
        for t in range(16):
            assert len(set(idx[b, t].tolist())) == 2


def test_moe_expert_weights_sum_to_one():
    ffn = SparseTopKMoEFFN(128, 256, num_experts=4, top_k=2)
    ffn.eval()
    x = torch.randn(1, 8, 128)
    logits = ffn.gate(x)
    top_logits, top_idx = torch.topk(logits, 2, dim=-1)
    probs = F.softmax(top_logits, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(1, 8), atol=1e-6)


def test_lb_loss_finite_and_trainable():
    """Load-balancing loss must be finite and reducible by training the gate."""
    torch.manual_seed(1)
    ffn = SparseTopKMoEFFN(128, 256, num_experts=4, top_k=2)
    ffn.train()
    # skew the gate so routing starts imbalanced
    with torch.no_grad():
        ffn.gate.bias.copy_(torch.tensor([3.0, 0.0, 0.0, 0.0]))
    x = torch.randn(4, 32, 128)
    opt = torch.optim.Adam(ffn.gate.parameters(), lr=0.05)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        _, lb = ffn(x, return_loss=True)
        assert torch.isfinite(lb), "LB loss not finite"
        lb.backward()
        opt.step()
        losses.append(lb.item())
    assert losses[-1] < losses[0], f"LB loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_moe_ffn_actually_sparse_compute():
    """A genuinely sparse MoE must not run every expert on every token.
    Hooks count how many tokens each Expert module actually processes."""
    ffn = SparseTopKMoEFFN(128, 256, num_experts=4, top_k=2)
    ffn.eval()
    seen = {}
    handles = []
    def mk(name):
        def hook(mod, inp, out):
            seen[name] = seen.get(name, 0) + inp[0].shape[0] * inp[0].shape[1]
        return hook
    for i, e in enumerate(ffn.experts):
        handles.append(e.register_forward_hook(mk(i)))
    x = torch.randn(2, 16, 128)
    with torch.no_grad():
        ffn(x)
    for h in handles:
        h.remove()
    total_tokens = 2 * 16
    # sparse would be: each expert sees << total tokens (avg total_tokens*top_k/num_experts = 16)
    assert all(v <= total_tokens for v in seen.values())
    # THE FINDING-CHECK: if every expert processed ALL tokens, compute is dense.
    dense = all(v == total_tokens for v in seen.values())
    assert not dense, (
        f"FINDING: every expert processed all {total_tokens} tokens ({seen}); "
        "routing only re-weights dense outputs — compute is NOT sparse"
    )


# ---------- MoE attention ----------

def test_moe_attention_gate_gets_gradients():
    """MoEAttentionModule defines a router (self.gate). If 'MoE attention' is
    real, that router must participate in the forward pass."""
    attn = MoEAttentionModule(128, 4, num_experts=4, top_k=2)
    attn.train()
    x = torch.randn(2, 16, 128, requires_grad=True)
    out = attn(x)
    out.sum().backward()
    g = attn.gate.weight.grad
    assert g is not None and g.abs().sum() > 0, (
        "FINDING: MoEAttentionModule.gate receives NO gradient — it is never "
        "called in forward(); the 'MoE attention' is plain multi-head attention"
    )


# ---------- KV cache ----------

def test_kv_cache_module_equivalence():
    """Module level: forward with pre-cached K/V from the SAME input must
    reproduce the uncached forward exactly."""
    attn = MoEAttentionModule(128, 4, num_experts=4, top_k=2)
    attn.eval()
    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        out_uncached = attn(x, use_cached_kv=False)
        attn.cache_kv_from_level1(x)
        out_cached = attn(x, use_cached_kv=True)
    assert torch.allclose(out_uncached, out_cached, atol=1e-5), (
        f"max diff {(out_uncached - out_cached).abs().max().item()}"
    )


def test_model_is_causal(model):
    """A language model must not let position t depend on tokens after t.
    Perturb ONLY the last quarter of the input; outputs in the first
    quarter must be unchanged."""
    torch.manual_seed(2)
    x = torch.randint(0, CFG.VOCAB_SIZE, (1, 128))
    x2 = x.clone()
    x2[:, 96:] = torch.randint(0, CFG.VOCAB_SIZE, (1, 32))
    with torch.no_grad():
        o1 = model(x, training=False)
        o2 = model(x2, training=False)
    diff = (o1[:, :32] - o2[:, :32]).abs().max().item()
    assert diff < 1e-4, (
        f"FINDING: causality violated — changing tokens at positions 96-127 "
        f"changed logits at positions 0-31 by up to {diff:.4f}. The EF blocks "
        "attend (unmasked) to Level-1 K/V computed from the FULL sequence, "
        "including future tokens"
    )


# ---------- Whole model ----------

def test_forward_backward_shapes_and_grads(model):
    model.train()
    x = torch.randint(0, CFG.VOCAB_SIZE, (2, 128))
    logits = model(x, training=True)
    assert logits.shape == (2, 128, CFG.VOCAB_SIZE)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, CFG.VOCAB_SIZE), x[:, 1:].reshape(-1))
    loss = loss + 0.01 * model.get_load_balance_loss() + 0.01 * model.get_specialization_loss()
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients in: {bad[:5]}"
    assert torch.isfinite(loss)


def test_parameter_count(model):
    n = model.count_parameters()
    assert n == 15_837_250, n


def test_specialization_loss_finite(model):
    model.train()
    x = torch.randint(0, CFG.VOCAB_SIZE, (1, 64))
    _ = model(x, training=True)
    assert torch.isfinite(model.get_specialization_loss())
    assert torch.isfinite(model.get_load_balance_loss())


def test_overfit_one_batch():
    """Classic sanity check: the model must be able to overfit one tiny
    fixed batch (loss drop > 50% in 120 steps)."""
    torch.manual_seed(3)
    m = SLAM3Model(CFG)
    m.train()
    x = torch.randint(0, CFG.VOCAB_SIZE, (4, 64))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    def total_loss():
        logits = m(x, training=True)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, CFG.VOCAB_SIZE), x[:, 1:].reshape(-1))
        return ce + 0.01 * m.get_load_balance_loss() + 0.01 * m.get_specialization_loss()
    first = total_loss().item()
    for _ in range(120):
        opt.zero_grad()
        l = total_loss()
        l.backward()
        opt.step()
    last = total_loss().item()
    assert last < 0.5 * first, f"overfit failed: {first:.2f} -> {last:.2f}"
