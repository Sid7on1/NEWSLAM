# SLAM — Honest Audit & Benchmark Report

**Date:** 2026-10-23 (overnight run) · **Code audited:** Mac NEWSLAM `slam3_pytorch.py` (715 lines, 2025-11-17) — the newest SLAM in existence, verified by direct reads of the Mac, GitHub, and the code itself
**Hardware for all numbers:** Linux VM, 2 vCPU, PyTorch 2.14 CPU, 2 threads, **no GPU**. Everything below was run tonight; nothing is quoted from Sid's old harness files except where explicitly flagged as his unverified artifacts.

---

## 1. The verdict, blunt

**SLAM is a real, ambitious student research prototype with one genuinely interesting idea — and its current headline claims do not survive contact with its own code.**

It is **not** "just jargon": RoPE is correctly implemented, the MoE FFN routing and load-balancing loss are genuine, the Encoder-Fusion structure does mechanically what it says, and the whole thing trains and overfits properly. A first-year-to-third-year undergraduate wrote a working, non-trivial transformer variant from scratch. That is real.

It is also **not "crazy"** — not yet, and not close:

- The flagship **"MoE attention" contains no MoE at all** — its router is dead code (Finding 1).
- The **"sparse" MoE FFN computes densely** — it runs every expert on every token, then masks (Finding 2).
- The model **breaks causality** — it can attend to future tokens, so it is not currently a valid language model (Finding 3, test-measured).
- In tonight's matched benchmark, **a plain vanilla transformer beat it** on final perplexity (1064.8 vs 1121.3) and trained 26% faster — despite SLAM enjoying the future-information leak as a handicap in its favour.

### Grades (separate, as they should be)
| Aspect | Grade | Why |
|---|---|---|
| **Architecture ideas** | B− | One real idea (cross-layer K/V sharing + segment fusion across rotations), surrounded by renamed standard parts and two mechanisms that are names without mechanisms (entropy exit, frequency specialization) |
| **Implementation** | C+ | Working, trainable, mostly clean PyTorch — undermined by a dead router, dense-instead-of-sparse MoE, position bugs in the EF path, and no training loop or decoding path in the repo |
| **Evidence** | F (before tonight) / C− (after) | Before tonight: zero completed training runs, zero valid benchmarks anywhere (README says "Incomplete"). After tonight: one honest small-budget comparison — which SLAM loses. The evidence base is now real but unfavourable |

**One-sentence version for Sid:** you built a real research prototype with a real idea — and right now its three most quotable claims (MoE attention, sparse experts, KV-cache efficiency) are respectively absent, inverted, and inseparable from a causality bug. Fix those three things and this becomes a legitimately strong project; leave them and any good research interviewer will find them in ten minutes, the way this audit did.

---

## 2. Version lineage

Full table in `results/version_map.md`. The short version:

| Version | Where | Reality |
|---|---|---|
| v1 (Apr–May 2025) | Mac ~/SLAM + GitHub SLAM-v1 (different, smaller file) | BERT-tokenizer encoder, EF cycles, MLM/IMDB heads. No RoPE/MoE/KV. No results. |
| v2 (Apr–May 2025) | Mac slam-bigg + GitHub SLAM-v2 | MoE-attention encoder, 20 experts. Has checkpoints, no verified results. |
| "v3" skeleton (May 2025) | Mac ~/SALM-MUTLIMODEL | **Placeholder by its own comments** — no-op RoPE, uniform MoE, fake tokenizer. Never cite. |
| **NEWSLAM = PyTorch v3 (Nov 2025)** | **Mac (715 lines) — CANONICAL**; GitHub copy is older (650 lines; May 2026 commit touched only the README) | The real v3: RoPE, MoE FFN, L1-KV/fresh-Q, EF rotations. Audited + benchmarked below. |
| slam-v4 (Jul 2025) | Mac slam-v4/ | PyTorch iteration with its own tests folder; superseded. |
| VSLAM v4/v5/v6 (Jul–Nov 2025) | Mac VSLAM/, MLX | Real MLX code incl. a "Cortex-R" v6 — but its results file is broken (below). GitHub "SLAM-v3" is actually this v6 MLX code, mislabeled. |

---

## 3. Audit findings (full detail: `results/code_audit.md`)

**The three that matter:**

1. **MoE attention's router is never called.** `MoEAttentionModule.gate` (line 189) — grep proves the only `.gate(` call in the file belongs to the FFN. Attention is plain MHA. Gradient test confirms: the gate receives **no gradient, ever**.
2. **MoE FFN is dense, not sparse.** Every expert processes the full [B,T,D] tensor; routing only re-weights outputs. Compute ≈ 4-expert dense FFN to use 2 experts' worth of signal.
3. **Causality leak (measured, not suspected).** EF blocks attend *unmasked* to Level-1 K/V computed from the **full sequence**. Perturbing only tokens 96–127 changed logits at positions 0–31 by up to **0.1208** (should be 0.0000). Root cause is architectural: the "KV cache" is a one-shot full-sequence K/V bank transplanted into other modules — it is cross-layer K/V sharing, not an inference cache, and there is no autoregressive decoding path in the file at all.

**The rest, quickly:** RoPE positions restart per EF segment while cached K keeps global positions → Q/K relative offsets wrong for segments 2–4 · "entropy early exit" computes no entropy (it's |Linear(Δx)|.mean() vs an unvalidated 0.1) · "frequency specialization" uses no frequency analysis (a Linear(128→4) forced by KL toward hand-invented target vectors labelled Structure/Patterns/Transitions/Details) · no training loop in the model file — training lived in unexecuted notebooks · `forward(training=True)` default ignores `model.eval()` conventions · 81.6% of the 15.84M parameters are embedding + output head; the entire novel architecture is ~2.92M params · no custom tokenizer (GPT-2 BPE), FlashAttention flag is decorative.

**What is genuinely sound:** RoPE (passes norm + relative-position property tests) · top-k routing and the Switch-style load-balancing loss (real, finite, trainable) · the rotation/segment fusion machinery · overall trainability (overfit-one-batch passes cleanly).

**VSLAM (static audit only — MLX cannot run on Linux; full notes `results/vslam_audit.md`):** the v6 code is real (gated attention with a *correct* causal mask, GEGLU, RMSNorm, a domain-router with rank-64 adapters), but its "results" are void: **parameter_count = 0 for every model**, training histories of exactly **5 steps**, headline PPL (4.46) inconsistent with its own history (7.33), on a synthetic toy task (vocab 5,000, seq 32, random tokens). The "Jamba/DeepSeek/Hunyuan" comparisons are toy clones in the same harness. **Never quote any VSLAM number.**

---

## 4. Test results (`tests/test_slam3.py`, log: `logs/pytest_full.log`)

**13 tests: 10 passed, 3 failed. The 3 failures are findings, kept deliberately.**

| Test | Result |
|---|---|
| RoPE preserves norms | ✅ PASS |
| RoPE relative-position property | ✅ PASS |
| MoE top-k selects exactly k=2 experts | ✅ PASS |
| MoE expert weights sum to 1 | ✅ PASS |
| Load-balancing loss finite + decreases under gate training | ✅ PASS |
| KV-cache module equivalence (cached == uncached, same input) | ✅ PASS (atol 1e-5) |
| Forward/backward shapes + all gradients finite | ✅ PASS |
| Parameter count == 15,837,250 | ✅ PASS |
| Specialization + LB losses finite in full model | ✅ PASS |
| Overfit-one-batch (loss must drop >50% in 120 steps) | ✅ PASS |
| **MoE FFN is actually sparse in compute** | ❌ **FAIL — every expert processed all tokens; compute is dense** |
| **MoE attention router receives gradients** | ❌ **FAIL — gate is dead code; 'MoE attention' is plain MHA** |
| **Model is causal (future perturbation must not change past logits)** | ❌ **FAIL — max change 0.1208 at early positions** |

---

## 5. Benchmarks vs matched baseline

**Baseline:** vanilla decoder-only transformer written for this audit (`code/baseline.py`): RoPE causal attention, dense GELU FFN, 15 layers, d=128, ff=512 — **15,890,385 params vs SLAM's 15,837,250 (+0.34%)**, same vocab (GPT-2, 50,257).
**Learning task:** WikiText-2, GPT-2 tokenizer, identical budget for both: **400 steps × batch 8 × seq 128 = 409,600 tokens**, AdamW lr 3e-4 (SLAM's own config value), warmup 40 + cosine, grad-clip 1.0, seed 42, val = 20 fixed batches every 25 steps. SLAM trained with its aux losses (CE + 0.01·LB + 0.01·spec). Scripts: `code/train_compare.py`; data: `results/train_*.csv`, `results/train_summary.json`, `results/efficiency.json`.

### Learning (validation perplexity ↓)
| Step | SLAM | Baseline |
|---|---|---|
| 0 | 57,319 | 56,848 |
| 50 | 7,263 | 8,492 |
| 100 | 1,599 | 1,895 |
| 150 | 1,381 | 1,340 |
| 250 | 1,229 | 1,145 |
| **400 (final)** | **1,121.3** | **1,064.8** |

SLAM descends faster early (steps ~50–125 — plausibly helped by its extra per-token compute *and* its future leak), then plateaus; the baseline overtakes at ~step 150 and keeps widening. **Final: baseline 5.0% lower perplexity.** And SLAM's number is the *optimistic* one — its val loss is computed with the future leak active.

### Efficiency
| Metric | SLAM | Baseline |
|---|---|---|
| Train throughput | 460 tok/s | 581 tok/s (+26%) |
| Inference fwd, batch 1, seq 128 | 0.0955 s (0.75 ms/tok) | 0.0706 s (0.55 ms/tok) — SLAM 1.35× slower |
| Inference fwd, batch 1, seq 512 | 0.310 s | 0.295 s — SLAM 1.05× slower |
| Persistent KV memory @ seq 4096 | 4.19 MB | 62.9 MB (standard per-layer cache) — SLAM's single bank is 15× smaller |

The KV number needs its caveat printed next to it forever: SLAM's "cache" is one K/V bank computed from the **whole input at once**. A real autoregressive cache must be built token-by-token; SLAM has no such path, and the single-bank design *is* the causality bug. It is a memory saving the way removing the brakes is a weight saving.

**Budget caveat (important):** 410k tokens is a smoke budget, not a training run — both models sit at PPL ~1,100, nowhere near converged (a trained 124M GPT-2 scores ~25–30 on WikiText-2). Tonight's result licenses exactly one claim: *at matched parameters and a small equal budget, on CPU, SLAM did not beat a vanilla decoder and was slower to train.* It does not prove SLAM could never win at scale — it proves the burden of proof is unmet, and the early-plateau shape is not encouraging.

---

## 6. What to fix/build next (ordered, concrete)

1. **Decide what SLAM is.** As an encoder (masked/bidirectional modelling, classification, retrieval), the full-sequence K/V bank is legal and the causality finding evaporates. As a decoder LM, it is currently invalid. Pick one and rename things honestly. *(Highest-value decision; costs nothing.)*
2. **If decoder: fix causality properly** — mask EF attention to each segment's causal past (or drop the cross-layer bank and give each block its own K/V). Re-run `test_model_is_causal` until it passes at 0.0000.
3. **Make MoE attention real or rename it.** Wire the router (per-head or per-token expert attention variants) or call the module what it is: MHA. A dead Linear in the flagship module is the worst look in the repo.
4. **Make the MoE FFN actually sparse** — gather/scatter token dispatch (or at minimum index-select expert inputs). This is a real ~2× FFN FLOP cut and turns a false claim into a true one; benchmark before/after.
5. **Fix EF RoPE positions** — rotate segment Q with global offsets (start = segment_start) to match cached K.
6. **Kill or substantiate the named mechanisms** — replace "entropy" with an actual residual-norm stopping rule and validate the threshold, or delete it; replace the "frequency analyzer" with a real spectral feature (FFT band energies of hidden states) or rename the loss "profile-matching loss" and soften the claims.
7. **Ship a training loop + a completed run.** One script, one dataset (WikiText-103 as the notebooks intended), checkpoints, logged curves. Target: a *causal-fixed* SLAM vs tonight's baseline recipe at ≥10M tokens on Kaggle/Colab GPU. This single artifact converts the whole project from claims to evidence.
8. **Ablate the one real idea.** L1-KV-sharing on/off, EF rotations 1 vs 4, specialization loss on/off — three small runs, one table. Ablations are what separates a research project from a pile of features, and they're cheap at this scale.
9. **VSLAM: delete or quarantine the results file**, fix the parameter counter, and re-run the sweep on a real task (even tiny-shakespeare LM) before quoting anything. Fix the GitHub "SLAM-v3" label while you're there — it points at v6 MLX code.
10. **Tie the embedding and output head** (saves 6.48M params — 41% of the model) the moment parameter efficiency is claimed anywhere.

---

## 7. Resume-safe claims (post-audit)

**Now SAFE (verified tonight, with numbers):**
- "Implemented a transformer architecture from scratch in PyTorch (15.8M params): RoPE, sparse top-2-of-4 MoE feed-forward with load-balancing loss, multi-level segment-fusion (Encoder-Fusion) cycles with a KL specialization loss" — all code-verified.
- "Wrote a correctness test-suite (13 tests) and a parameter-matched benchmark vs a vanilla decoder baseline on WikiText-2 (400-step equal budget): SLAM reached val PPL 1121 vs baseline 1065; identified and measured a causality leak (max logit deviation 0.12) in its cross-layer K/V design" — *this framing (found my own bug, measured it) is stronger in a research interview than a fake win.*
- "Implemented RoPE verified by norm-preservation and relative-position property tests; load-balancing loss verified trainable."
- Honest scope line: "architecture implemented and benchmarked at prototype scale; a full training run is the next phase."

**Still NOT safe (do not print):**
- ❌ "MoE attention" (does not exist in the code)
- ❌ "sparse MoE" as an efficiency claim (compute is dense)
- ❌ "efficient KV cache" / any inference-speed or long-context claim (no autoregressive path; the memory figure is inseparable from the causality bug)
- ❌ Any perplexity/accuracy from NEWSLAM notebooks, VSLAM, or GitHub READMEs (unexecuted, synthetic, or broken instrumentation)
- ❌ "custom tokenizer" (GPT-2 BPE), "Q-only updates" as a *novel mechanism* claim without the caveat that K/V come from layer-1 embeddings of the full sequence
- ❌ Q-only pattern phrasing as originally drafted ("later levels compute fresh Q against cached K/V") is mechanically true — but printing it without the causality caveat would mislead; keep it only alongside the encoder framing or after Fix 2.

---

## Artifacts in this folder
- `code/` — Mac + GitHub copies of slam3_pytorch.py, SALM placeholder, VSLAM v6, `baseline.py`, `train_compare.py`, `bench_efficiency.py`
- `tests/test_slam3.py` — 13 tests, rerunnable: `venv/bin/python -m pytest tests/test_slam3.py -v`
- `results/` — version_map.md, code_audit.md, vslam_audit.md, train_slam.csv, train_baseline.csv, train_summary.json, efficiency.json, vslam_harness_results_mac.json
- `logs/` — pytest_full.log, train_compare.log, efficiency_run.log
