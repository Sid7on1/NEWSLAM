# VSLAM (MLX) static audit — NOT run (MLX is Apple-only; this VM is Linux). No numbers were faked or re-derived.

## The v6 model code (models/slam_v6_jamba_killer.py, 637 lines, MLX)
Real code, not a skeleton:
- `CortexREnhancedGatedAttention` — MHA with a **correct causal mask** (mx.triu, -inf) and a sigmoid gate on the attention output (`attn_out * sigmoid(gate_proj(x))`). A legitimate, simple gated-attention variant.
- GEGLU FFN, RMSNorm, parallel attn/FFN norm layer with concat+project — all standard modern parts, correctly assembled.
- `JambaKillerEFBlock` "signature" q/k/v transforms are three Linear layers averaged and scaled by **0.01** then added to x — i.e. a near-identity nudge wearing a grand name.
- Cortex-R domain router (8 domains, rank-64 LoRA-style adapters, "model soup" fusion) and an **emotion-bias** module (7 emotion types injected into queries) are real code but unevaluated speculation: no experiment in the repo isolates their effect. Feature-stacking without ablations.

## The harness "results" (comprehensive_model_results.json) — not evidence
1. **parameter_count = 0 for every model** (SLAM v6, Jamba, DeepSeek, Hunyuan, standard transformer). The instrumentation never counted a single parameter; any size/efficiency comparison in the file is void.
2. **Training histories have exactly 5 points.** The headline "final_perplexity 4.46 / accuracy 98.5%" comes after FIVE optimizer steps on an unspecified task.
3. **Internally inconsistent:** for SLAM v6, metrics.final_perplexity = 4.46 while histories.perplexity ends at 7.33 and exp(final loss 1.992) = 7.33. The "final" metric does not match the model's own history.
4. **The task is synthetic and trivially learnable:** accuracy goes 1.5% → 98.5% in 5 steps (loss 10.15 → 1.99). README config: embed 256, vocab 5,000, seq 32, random-token inputs (mx.random.randint in the README quick-start). This measures memorization of a toy target, not language modelling.
5. **Scores are a homemade rubric** ("overall_score 74/100", V4 Champion "83/100, Production Ready") with no external validity. The reference "Jamba 1.5 / DeepSeek V2.5 / Hunyuan Large" entries are small local re-implementations in the same toy harness — the names of frontier models attached to ~5K-vocab toy clones. Comparing against them proves nothing about the real systems.
6. README's "2,970 combinations swept" (6 attn × 11 FFN × 5 norm × 9 optimizers) is plausible as a grid count but no per-combination results file was found in the folder listing to verify outcomes; the champion claims (93.75% acc, 0.98 ms) come from the same toy harness.

Verdict on VSLAM: a genuine, energetic architecture-playground with real MLX code — and a results file that must never be quoted anywhere. The v6 code deserves a real evaluation; the harness deserves deletion or a rewrite.
