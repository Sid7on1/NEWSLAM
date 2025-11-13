import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
import time

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Optimized for faster training on Colab
# ═══════════════════════════════════════════════════════════════════════════

class SLAM3Config:
    """
    Centralized configuration for SLAM3 model
    Optimized for Colab GPU/TPU training
    """
    # Model architecture (REDUCED for faster training)
    EMBED_DIM = 128          # Was 64, increased for better capacity
    SEQ_LEN = 128            # Was 32, increased for longer context
    NUM_HEADS = 4            # Same
    NUM_EXPERTS = 4          # Same
    FF_DIM = 256             # Was 128, doubled
    VOCAB_SIZE = 50257       # GPT-2 vocab
    TOP_K_EXPERTS = 2        # Same
    NUM_ROTATIONS = 4        # Same
    MAX_EF_CYCLES = 2        # Was 4, reduced for speed

    # Loss weights
    LOAD_BALANCE_WEIGHT = 0.01
    SPECIALIZATION_WEIGHT = 0.01
    ENTROPY_THRESHOLD = 0.1

    # Training
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 16          # Adjust based on GPU memory
    DROPOUT = 0.1
    GRAD_CLIP = 1.0
    WARMUP_STEPS = 500
    MAX_STEPS = 50000

    # Efficiency
    USE_FLASH_ATTENTION = False
    CHECKPOINT_GRADIENTS = True

    # Checkpointing
    SAVE_EVERY = 500
    EVAL_EVERY = 100
    LOG_EVERY = 10

    def validate(self):
        """Validate configuration"""
        assert self.EMBED_DIM % self.NUM_HEADS == 0, "embed_dim must be divisible by num_heads"
        assert self.SEQ_LEN % 4 == 0, "seq_len must be divisible by 4 for rotations"
        assert self.TOP_K_EXPERTS <= self.NUM_EXPERTS, "top_k must be <= num_experts"

config = SLAM3Config()
config.validate()

# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def efficient_rotation(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Efficient rotation using indexing"""
    if shift == 0:
        return x
    batch_size, seq_len, embed_dim = x.shape
    shift = shift % seq_len
    return torch.cat([x[:, shift:, :], x[:, :shift, :]], dim=1)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask"""
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)
    return mask.bool()


class MetricsTracker:
    """Track training metrics"""
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'ce_loss': [],
            'lb_loss': [],
            'spec_loss': [],
            'step_time': [],
            'learning_rate': [],
        }

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.metrics:
                if torch.is_tensor(value):
                    value = value.item()
                self.metrics[key].append(float(value))

    def get_stats(self, key: str, last_n: int = 100):
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        values = self.metrics[key][-last_n:]
        values_tensor = torch.tensor(values)
        return {
            'mean': values_tensor.mean().item(),
            'std': values_tensor.std().item(),
            'min': values_tensor.min().item(),
            'max': values_tensor.max().item(),
        }

# ═══════════════════════════════════════════════════════════════════════════
# RoPE POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════════════

class RoPEPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(position, self.inv_freq)
        sin_freqs = torch.sin(freqs)
        cos_freqs = torch.cos(freqs)
        return cos_freqs, sin_freqs


def apply_rope(x, cos_freqs, sin_freqs):
    """Apply RoPE to input tensor"""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos_freqs - x_odd * sin_freqs
    rotated_odd = x_even * sin_freqs + x_odd * cos_freqs

    output = torch.zeros_like(x)
    output[..., ::2] = rotated_even
    output[..., 1::2] = rotated_odd

    return output

# ═══════════════════════════════════════════════════════════════════════════
# EXPERT MODULE
# ═══════════════════════════════════════════════════════════════════════════

class Expert(nn.Module):
    """Expert module with better initialization"""
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ═══════════════════════════════════════════════════════════════════════════
# MOE ATTENTION MODULE
# ═══════════════════════════════════════════════════════════════════════════

class MoEAttentionModule(nn.Module):
    """MoE Attention with KV caching"""
    def __init__(self, embed_dim, num_heads, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.gate = nn.Linear(embed_dim, num_experts)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.rope = RoPEPositionalEncoding(self.head_dim)

        # Cached K/V
        self.cached_k = None
        self.cached_v = None
        self.use_cache = False

    def cache_kv_from_level1(self, x):
        """Cache K/V from Level 1"""
        batch_size, seq_len, _ = x.shape

        k = self.k_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE to K
        cos_freqs, sin_freqs = self.rope(seq_len, x.device)
        cos_freqs = cos_freqs.unsqueeze(1)
        sin_freqs = sin_freqs.unsqueeze(1)
        k = apply_rope(k, cos_freqs, sin_freqs)

        self.cached_k = k
        self.cached_v = v
        self.use_cache = True

        return k, v

    def forward(self, x, use_cached_kv=False):
        batch_size, seq_len, _ = x.shape

        # Get or compute K, V
        if use_cached_kv and self.cached_k is not None:
            k, v = self.cached_k, self.cached_v
        else:
            k, v = self.cache_kv_from_level1(x)

        # Always compute fresh Q
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE to Q
        cos_freqs, sin_freqs = self.rope(seq_len, x.device)
        cos_freqs = cos_freqs.unsqueeze(1)
        sin_freqs = sin_freqs.unsqueeze(1)
        q = apply_rope(q, cos_freqs, sin_freqs)

        # Multi-head attention
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Get K/V sequence length (may differ from Q if using cached)
        kv_seq_len = k.shape[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask - handle different Q and K/V lengths
        if seq_len == kv_seq_len:
            # Standard case: same length
            mask = create_causal_mask(seq_len, x.device)
            scores = scores.masked_fill(mask, float('-inf'))
        else:
            # When using cached K/V: Q is shorter, K/V is full sequence
            # Allow Q to attend to all of K/V (no causal mask needed for cross-attention)
            # Or create appropriate mask shape [seq_len, kv_seq_len]
            pass  # No masking when using cached K/V from different sequence

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(out)

# ═══════════════════════════════════════════════════════════════════════════
# SPARSE TOP-K MOE FFN
# ═══════════════════════════════════════════════════════════════════════════

class SparseTopKMoEFFN(nn.Module):
    """Sparse MoE FFN with load balancing"""
    def __init__(self, embed_dim, hidden_dim, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.embed_dim = embed_dim

        self.experts = nn.ModuleList([
            Expert(embed_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(embed_dim, num_experts)
        self.last_load_balance_loss = 0.0

    def compute_load_balance_loss(self, gate_logits, expert_indices):
        """Compute load balancing loss"""
        batch_size, seq_len, _ = gate_logits.shape

        # Expert usage frequency
        expert_mask = torch.zeros(batch_size * seq_len * self.top_k, self.num_experts, device=gate_logits.device)
        flat_indices = expert_indices.reshape(-1)
        expert_mask.scatter_(1, flat_indices.unsqueeze(1).long(), 1.0)

        expert_freq = expert_mask.sum(0) / (batch_size * seq_len * self.top_k + 1e-8)

        # Importance weights
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_importance = gate_probs.mean(dim=(0, 1))

        # Load balance loss
        load_balance_loss = self.num_experts * (expert_freq * expert_importance).sum()

        return load_balance_loss

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Compute gating scores
        gate_logits = self.gate(x)

        # Add noise during training
        if self.training:
            noise = torch.randn_like(gate_logits) * 0.01
            gate_logits = gate_logits + noise

        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Compute load balance loss
        self.last_load_balance_loss = self.compute_load_balance_loss(gate_logits, top_k_indices)

        # Sparse computation
        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)

            if expert_mask.sum() > 0:
                # Compute expert weights
                expert_weights = torch.zeros(batch_size, seq_len, device=x.device)
                for k in range(self.top_k):
                    k_mask = (top_k_indices[..., k] == expert_idx)
                    expert_weights += k_mask.float() * top_k_probs[..., k]

                # Process through expert
                expert_output = self.experts[expert_idx](x)
                output += expert_output * expert_weights.unsqueeze(-1)

        return output

# ═══════════════════════════════════════════════════════════════════════════
# EF BLOCK
# ═══════════════════════════════════════════════════════════════════════════

class EFBlock(nn.Module):
    """EF Block with enforced specialization"""
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, block_id, dropout=0.1):
        super().__init__()
        self.block_id = block_id
        self.embed_dim = embed_dim

        self.attn = MoEAttentionModule(embed_dim, num_heads, num_experts, top_k=config.TOP_K_EXPERTS, dropout=dropout)
        self.ffn = SparseTopKMoEFFN(embed_dim, ff_dim, num_experts, top_k=config.TOP_K_EXPERTS, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable specialization
        self.specialization_proj = nn.Linear(embed_dim, embed_dim)
        self.frequency_analyzer = nn.Linear(embed_dim, 4)

        # Target frequency profiles
        frequency_profiles = {
            0: ([1.0, 0.5, 0.2, 0.1], "Structure"),
            1: ([0.3, 1.0, 0.8, 0.3], "Patterns"),
            2: ([0.1, 0.4, 1.0, 0.7], "Transitions"),
            3: ([0.1, 0.2, 0.5, 1.0], "Details"),
        }

        profile, name = frequency_profiles.get(block_id, ([0.5]*4, "General"))
        target_freq_profile = torch.tensor(profile, dtype=torch.float32)
        target_freq_profile = target_freq_profile / target_freq_profile.sum()
        self.register_buffer('target_freq_profile', target_freq_profile)
        self.block_name = name

        self.last_specialization_loss = 0.0
        self.last_freq_profile = None

    def compute_specialization_loss(self, x):
        """Compute specialization loss"""
        freq_scores = self.frequency_analyzer(x)
        freq_profile = freq_scores.mean(dim=(0, 1))
        freq_profile = F.softmax(freq_profile, dim=-1)

        self.last_freq_profile = freq_profile

        # KL divergence
        epsilon = 1e-8
        kl_loss = (self.target_freq_profile * torch.log(
            (self.target_freq_profile + epsilon) / (freq_profile + epsilon)
        )).sum()

        return kl_loss

    def forward(self, x, cached_k=None, cached_v=None, use_cached_kv=False):
        # Specialization transformation
        x_normed = self.norm1(x)
        x_specialized = self.specialization_proj(x_normed)

        # Attention with cached K/V
        if use_cached_kv and cached_k is not None:
            self.attn.cached_k = cached_k
            self.attn.cached_v = cached_v

        attn_out = self.attn(x_specialized, use_cached_kv=use_cached_kv)
        attn_out = self.dropout(attn_out)
        x = x + attn_out

        # Compute specialization loss
        self.last_specialization_loss = self.compute_specialization_loss(attn_out)

        # FFN
        ffn_out = self.ffn(self.norm2(x))
        ffn_out = self.dropout(ffn_out)
        x = x + ffn_out

        return x

# ═══════════════════════════════════════════════════════════════════════════
# MULTI-ROTATION EF CYCLE
# ═══════════════════════════════════════════════════════════════════════════

class MultiRotationEFCycle(nn.Module):
    """Multi-rotation EF cycle with efficient rotation"""
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, num_rotations=4, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            EFBlock(embed_dim, num_heads, ff_dim, num_experts, i, dropout)
            for i in range(4)
        ])
        self.num_rotations = num_rotations
        self.embed_dim = embed_dim

        self.fusion_gate = nn.Linear(embed_dim, num_rotations)

    def forward(self, x, cached_k=None, cached_v=None, use_cached_kv=False):
        batch_size, seq_len, embed_dim = x.shape

        # Ensure divisibility
        original_seq_len = seq_len
        if seq_len % 4 != 0:
            pad_len = 4 - (seq_len % 4)
            x = F.pad(x, (0, 0, 0, pad_len))
            seq_len = x.shape[1]

        seg_len = seq_len // 4

        # Process rotations
        all_rotation_outputs = []

        for rotation_idx in range(self.num_rotations):
            shift = (rotation_idx * seq_len // self.num_rotations) % seq_len
            rotated_x = efficient_rotation(x, shift)

            # Process segments
            rotation_outputs = []
            for block_idx, block in enumerate(self.blocks):
                start = block_idx * seg_len
                end = start + seg_len
                seg = rotated_x[:, start:end, :]

                processed_seg = block(seg, cached_k, cached_v, use_cached_kv)
                rotation_outputs.append(processed_seg)

            rotation_result = torch.cat(rotation_outputs, dim=1)

            # Unrotate
            reverse_shift = seq_len - shift if shift > 0 else 0
            unrotated = efficient_rotation(rotation_result, reverse_shift)

            all_rotation_outputs.append(unrotated)

        # Learnable fusion
        stacked_outputs = torch.stack(all_rotation_outputs, dim=0)
        fusion_scores = self.fusion_gate(x)
        fusion_weights = F.softmax(fusion_scores, dim=-1)
        fusion_weights = fusion_weights.unsqueeze(-1)
        stacked_outputs = stacked_outputs.permute(1, 2, 0, 3)
        final_output = (stacked_outputs * fusion_weights).sum(dim=2)

        # Trim to original length
        if final_output.shape[1] != original_seq_len:
            final_output = final_output[:, :original_seq_len, :]

        return final_output

    def get_specialization_loss(self):
        """Collect specialization losses"""
        total_loss = 0.0
        for block in self.blocks:
            total_loss += block.last_specialization_loss
        return total_loss / len(self.blocks)

# ═══════════════════════════════════════════════════════════════════════════
# LEVEL 1 ATTENTION
# ═══════════════════════════════════════════════════════════════════════════

class Level1Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts, dropout=0.1):
        super().__init__()
        self.attn = MoEAttentionModule(embed_dim, num_heads, num_experts, top_k=config.TOP_K_EXPERTS, dropout=dropout)

    def forward(self, x):
        return self.attn(x, use_cached_kv=False)

    def get_cached_kv(self):
        return self.attn.cached_k, self.attn.cached_v

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC SLAM3 ENCODER
# ═══════════════════════════════════════════════════════════════════════════

class DynamicSLAM3Encoder(nn.Module):
    """Complete SLAM3 Encoder"""
    def __init__(self, cfg: SLAM3Config):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.VOCAB_SIZE, cfg.EMBED_DIM)
        self.max_ef_cycles = cfg.MAX_EF_CYCLES

        self.dropout = nn.Dropout(cfg.DROPOUT)

        # Level 1
        self.level1_attn = Level1Attention(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.NUM_EXPERTS, cfg.DROPOUT)
        self.norm1 = nn.LayerNorm(cfg.EMBED_DIM)

        # Level 2: EF cycles
        self.ef_cycles = nn.ModuleList([
            MultiRotationEFCycle(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.FF_DIM, cfg.NUM_EXPERTS, cfg.NUM_ROTATIONS, cfg.DROPOUT)
            for _ in range(cfg.MAX_EF_CYCLES)
        ])

        # Level 3
        self.level3_attn = Level1Attention(cfg.EMBED_DIM, cfg.NUM_HEADS, cfg.NUM_EXPERTS, cfg.DROPOUT)
        self.norm2 = nn.LayerNorm(cfg.EMBED_DIM)

        # Output
        self.output_proj = nn.Linear(cfg.EMBED_DIM, cfg.VOCAB_SIZE)

        # Early stopping
        self.entropy_proj = nn.Linear(cfg.EMBED_DIM, 1)

    def compute_entropy(self, x):
        """Compute entropy for early stopping"""
        entropy_scores = self.entropy_proj(x)
        return entropy_scores.abs().mean()

    def forward(self, x, training=True, return_info=False):
        batch_size, seq_len = x.shape

        # Token embeddings
        x = self.embed(x)
        x = self.dropout(x)

        # Level 1
        x = x + self.level1_attn(self.norm1(x))
        cached_k, cached_v = self.level1_attn.get_cached_kv()

        # Level 2: EF cycles
        num_cycles = self.max_ef_cycles if training else self.max_ef_cycles
        cycles_executed = 0

        for cycle_idx in range(num_cycles):
            prev_x = x
            x = self.ef_cycles[cycle_idx](x, cached_k, cached_v, use_cached_kv=True)
            cycles_executed += 1

            # Early stopping in inference
            if not training and cycle_idx > 0:
                entropy = self.compute_entropy(x - prev_x)
                if entropy < self.config.ENTROPY_THRESHOLD:
                    break

        # Level 3
        x = x + self.level3_attn(self.norm2(x))

        if return_info:
            return x, {'cycles_executed': cycles_executed}
        return x

    def forward_with_logits(self, x, training=True):
        """Forward pass returning logits"""
        encoded = self(x, training=training)
        return self.output_proj(encoded)

    def get_load_balance_loss(self):
        """Collect load balance losses"""
        total_loss = 0.0
        count = 0
        for cycle in self.ef_cycles:
            for block in cycle.blocks:
                if hasattr(block.ffn, 'last_load_balance_loss'):
                    total_loss += block.ffn.last_load_balance_loss
                    count += 1
        return total_loss / max(count, 1)

    def get_specialization_loss(self):
        """Collect specialization losses"""
        total_loss = 0.0
        for cycle in self.ef_cycles:
            total_loss += cycle.get_specialization_loss()
        return total_loss / len(self.ef_cycles)

# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE SLAM3 MODEL
# ═══════════════════════════════════════════════════════════════════════════

class SLAM3Model(nn.Module):
    """Complete SLAM3 Model - PyTorch version"""
    def __init__(self, cfg: SLAM3Config = None):
        super().__init__()
        self.config = cfg or SLAM3Config()
        self.encoder = DynamicSLAM3Encoder(self.config)

    def forward(self, src, training=True):
        memory = self.encoder(src, training=training)
        return self.encoder.output_proj(memory)

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SLAM3 - PyTorch Implementation")
    print("="*80 + "\n")

    # Test model
    cfg = SLAM3Config()
    model = SLAM3Model(cfg)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Config: {cfg.NUM_ROTATIONS} rotations, {cfg.MAX_EF_CYCLES} cycles")
    print(f"Sequence length: {cfg.SEQ_LEN}")
    print(f"Embed dim: {cfg.EMBED_DIM}")

    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 2
    tokens = torch.randint(0, cfg.VOCAB_SIZE, (batch_size, cfg.SEQ_LEN), device=device)

    print(f"\nTesting forward pass on {device}...")
    output = model(tokens, training=False)
    print(f"Output shape: {output.shape}")
    print("\n✓ Model test successful!")
