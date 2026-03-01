---
layout: post
title: "CS336 Spring 2025, End to End: A Technical Summary of Stanford's Complete LLMs-From-Scratch Offering"
date: 2026-02-27 00:00:00 +0530
categories: [genai, systems]
tags: [cs336, llm, transformer, scaling-laws, data, alignment]
excerpt: "A comprehensive technical walkthrough of Stanford CS336 Spring 2025: lectures, assignments, and the core engineering lessons behind building language models from scratch."
---

As of **February 27, 2026**, Stanford's CS336 site labels the current term as **Spring 2026**, but that page is marked as under active construction and lists a tentative schedule.  
This post summarizes the **most recent complete offering**, which is **Spring 2025 (archived)**.

---

## What This Course Is Actually About

CS336 is less "learn transformer equations" and more "learn to engineer language models under hard resource constraints."  
The organizing principle across the quarter is:

> Maximize model quality under limited compute, memory, communication bandwidth, and data quality.

In Spring 2025, that objective is developed lecture by lecture, then forced into practice through assignments.

---

## Lecture-by-Lecture Long Summary (Spring 2025)

The complete offering runs from **April 1, 2025 to June 6, 2025**.  
Below is a long, technical summary of each scheduled lecture.

## Lecture 1: Overview and Tokenization

The opening lecture frames the core motivation for CS336: frontier models are increasingly "industrialized" and opaque, while researchers still need mechanistic understanding to do serious work.  
The lecture distinguishes three kinds of knowledge:

- **Mechanics** (what components do)
- **Mindset** (optimize efficiency under scale)
- **Intuition** (which design choices actually work)

Key framing arguments:

- "Scale alone" is a bad interpretation of progress; **scalable algorithms** matter.
- Smaller (<1B) course models are imperfect proxies for frontier systems, but they still teach transferable mechanics.
- The right mental model is to optimize **accuracy = efficiency x resources**.

The lecture also surveys LM history (n-grams -> neural LM -> seq2seq -> attention -> transformer -> scaling era -> open models), then maps out the full course stack: tokenization, architecture, training, kernels, distributed systems, inference, scaling laws, data curation, evaluation, and alignment.

## Lecture 2: PyTorch Primitives and Resource Accounting

This lecture is a bottom-up systems tutorial for training loops:

- Tensor basics, storage, views, strides, and contiguous layouts
- Float formats (`fp32`, `fp16`, `bf16`, `fp8`) and stability/range trade-offs
- FLOPs accounting for linear layers and backprop
- Model FLOPs utilization (MFU) and throughput interpretation
- Parameter initialization and training stability intuition
- Data loading/memmap/pinned memory
- Optimizer mechanics (Adam-family lineage) and loop structure
- Mixed precision and practical checkpointing

The core habit it tries to build: do rough accounting before coding expensive runs.

Instead of hand-wavy "this model is big," students are trained to ask:

- How much memory does each state tensor need?
- Where do forward/backward FLOPs concentrate?
- Which dtype decisions change stability vs speed?

## Lecture 3: Architectures and Hyperparameters

Lecture 3 moves from canonical transformers to modern LM design conventions.

Major architecture topics:

- **Pre-norm vs post-norm**: why pre-norm became dominant
- **LayerNorm vs RMSNorm**: runtime/memory movement considerations
- **Bias removal** in many modern stacks
- **FFN variants**: ReLU/GeLU vs gated variants (SwiGLU/GeGLU)
- **Serial vs parallel block variants**
- **Positional schemes**: absolute/relative/RoPE

One major takeaway is methodological: architecture choices are often empirical and ecosystem-driven, but there are still recurring patterns (for example, pre-norm + RMSNorm + SwiGLU-like stacks in many recent open models).

## Lecture 4: Mixture of Experts (MoE)

This lecture explains why sparse MoEs became mainstream:

- Higher parameter capacity at similar active FLOPs
- Strong open-model results at large scale
- Good fit for multi-device parallelism

Detailed topics include:

- Routing families (top-k token choice, hashing, assignment variants)
- Expert granularity and shared-expert patterns
- Load balancing heuristics and auxiliary losses
- Training stability issues (router behavior, precision choices)
- Systems concerns (sparse kernels, communication patterns)
- Upcycling dense checkpoints into MoE models

The lecture uses contemporary model families (including DeepSeek/Qwen-era designs) to show how "paper MoE" differs from production MoE.

## Lecture 5: GPUs

Lecture 5 is the hardware foundations lecture:

- GPU vs CPU execution model (throughput-oriented SIMT)
- SMs, warps, memory hierarchy, and bandwidth constraints
- Why matmuls dominate and why memory still bottlenecks end-to-end speed

Optimization concepts covered:

- Roofline intuition
- Control divergence
- Precision effects on arithmetic intensity
- Operator fusion
- Recomputation trade-offs
- Coalesced memory access
- Tiling and wave quantization effects

The lecture closes by connecting these ideas to attention acceleration (FlashAttention-style reasoning), preparing students for kernel-level optimization work.

## Lecture 6: Kernels and Triton

This is the practical kernel engineering lecture.

Workflow emphasized:

1. Benchmark end-to-end behavior.
2. Profile to find bottlenecks/kernels.
3. Inspect lower-level behavior (including PTX-level clues).
4. Re-implement critical paths (CUDA/Triton/compiled variants).

Key technical segments:

- Profiling matrix ops and MLP traces
- Kernel fusion demonstrations (e.g., GeLU variants)
- Writing CUDA extensions and validating correctness
- Writing Triton kernels and interpreting generated code
- Softmax and matmul optimization patterns
- Shared-memory tiling and L2-aware traversal behavior

This lecture is the bridge between "I know the algorithm" and "I can make it fast on real hardware."

## Lecture 7: Parallelism Basics

Lecture 7 is the conceptual map of large-scale training.

It starts with communication primitives (all-reduce, reduce-scatter, all-gather, etc.) and then compares parallelization strategies:

- Naive data parallelism
- ZeRO stages 1-3 / FSDP-style sharding
- Pipeline parallelism
- Tensor parallelism
- Sequence/activation parallel approaches

Important engineering points:

- Why naive DP is memory-inefficient
- Where ZeRO gives "almost free" memory wins vs added comm complexity
- Why pipeline bubbles depend heavily on microbatching
- Why tensor parallelism is typically intra-node (fast interconnect)
- Why activation memory remains a separate scaling challenge

## Lecture 8: Distributed Training in PyTorch

Lecture 8 operationalizes distributed concepts in code:

- Collective primitives in `torch.distributed`
- NCCL/backend behavior
- Hardware topology constraints (NVLink/NVSwitch context)
- Communication benchmarking
- Bare-bones implementations of data/tensor/pipeline parallel workflows

This lecture emphasizes that distributed strategy is fundamentally communication scheduling plus sharding decisions.  
It also reinforces a recurring course theme: simple abstractions hide real costs, so you must measure on your hardware.

## Lecture 9: Scaling Laws Basics

Lecture 9 introduces scaling laws as practical planning tools, not just curves in papers.

Core topics:

- Historical context from sample-complexity ideas to neural scaling observations
- Power-law behavior for loss vs data/model/compute
- Why exponent estimation matters
- Hyperparameter scaling questions (batch, LR, depth/width tradeoffs)
- Joint data-model scaling and compute-optimal decision-making
- Chinchilla-style methods (lower envelope, IsoFLOPs, joint fits)

Practical lesson: use small-to-medium runs to predict large-run choices instead of tuning directly at full scale.

## Lecture 10: Inference

Lecture 10 is a strong systems lecture on serving and decoding.

Main split:

- **Prefill**: parallelizable and usually more compute-friendly
- **Decode/generation**: sequential and typically memory-limited

Topics covered:

- Latency/throughput/TTFT trade-offs
- KV cache accounting and why decode becomes memory-bound
- Architecture-level cache-reduction methods:
  - GQA
  - MLA
  - cross-layer KV sharing
  - local/hybrid attention
- Quantization and pruning/distillation paths
- Speculative decoding (draft/target verification framing)
- Continuous/selective batching under dynamic request shapes

The key systems insight is that inference optimization is not one trick; it is a layered stack of architecture, caching, precision, and scheduling choices.

## Lecture 11: Scaling Details and Case Studies

Lecture 11 moves from scaling-law theory to real model recipes.

Case studies discussed include public scaling disclosures from modern model families (e.g., CerebrasGPT, MiniCPM, DeepSeek-style analyses), with emphasis on:

- Practical hyperparameter scaling procedures
- Batch/LR scaling estimation
- Data-to-model ratio selection
- Compute-saving schedule choices (e.g., warmup-stable-decay variants)
- Comparing scaling-fit methods in practice

It also dives into **muP (maximum update parameterization)** concepts and why scale-invariant hyperparameter transfer is attractive (and where it can fail or need adaptation in modern stacks).

## Lecture 12: Evaluation

Lecture 12 is a comprehensive "how to evaluate responsibly" framework.

It argues that evaluation is task-dependent and must specify:

- Inputs and coverage
- Model-calling protocol (prompting/tools/system setup)
- Output scoring methodology
- Interpretation assumptions

Benchmark families discussed include:

- Knowledge-style tests (MMLU, MMLU-Pro, GPQA, etc.)
- Instruction-following and preference-style evaluations (Arena-like setups, IFEval, AlpacaEval, WildBench)
- Agentic benchmarks (SWE-bench/CyBench/MLE-bench category)
- Safety/red-team evaluation styles

Additional focus areas:

- Realism vs benchmark convenience
- Train-test overlap and contamination concerns
- "Method vs system" evaluation ambiguity

## Lecture 13: Data (Sources and Governance)

Lecture 13 broadens from "how to train" to "what to train on."

Major content:

- Data lifecycle: live service -> dump/crawl -> processed corpus -> aggregated dataset
- Training stages:
  - pretraining
  - mid-training
  - post-training
- Historical dataset tour (BERT/GPT-era through newer open-data pipelines)
- Common Crawl ecosystem and conversion/filtering concerns
- Synthetic data and distillation roles

It also includes legal/compliance framing:

- Copyright scope and constraints
- Licensing realities
- Fair use factors and ambiguity
- Terms-of-service constraints beyond copyright

The lecture's central point is that data quality, provenance, and policy constraints are core model-quality variables.

## Lecture 14: Data Processing (Filtering and Deduplication Mechanics)

Lecture 14 is algorithmic and implementation-focused.

Filtering toolkit:

- n-gram/KenLM-style scoring
- fastText-style classifiers
- distributional/importance-resampling ideas (DSIR framing)

Applications:

- Language identification
- Quality filtering
- Toxicity/harm filters
- Task/domain targeting (e.g., math-heavy corpora)

Deduplication mechanics:

- Exact hashing workflows
- Bloom filters for approximate membership
- MinHash + LSH for near-duplicate detection under Jaccard similarity

This lecture makes explicit that filtering is an optimization problem over quality, diversity, coverage, and compute budget.

## Lecture 15: Alignment via SFT and RLHF

Lecture 15 transitions from pretraining behavior to instruction-following/aligned behavior.

Part 1 (SFT emphasis):

- What instruction datasets contain in practice
- Style effects (length/tone/bulleting) and how they influence model preference metrics
- Safety tuning with relatively small targeted data
- Why factual injection via finetuning can have non-trivial side effects
- Mid-training/instruction-mix strategies to scale alignment without catastrophic forgetting

Part 2 (RLHF setup):

- Why preference optimization is used after SFT
- Annotation pipeline realities (crowdsourcing quality, demographic effects, label consistency)
- Human-vs-AI feedback trade-offs in cost/scale
- PPO-era RLHF pipeline vs DPO simplifications
- Known failure modes (reward overoptimization, calibration/mode-collapse issues)

## Lecture 16: RL from Verifiable Rewards (RLVR)

Lecture 16 continues DPO/PPO framing, then focuses on reasoning-era RL:

- PPO recap and complexity
- GRPO motivation (remove value function, use group-relative normalization)
- Objective behavior and known caveats (including normalization-induced quirks)
- Length-control concerns in reasoning training

The lecture then surveys successful open reasoning pipelines (R1/Kimi/Qwen-era trends) and highlights what repeatedly appears in working recipes:

- Strong SFT bootstrap
- Difficulty-aware data filtering
- Verifiable reward design
- RL loops with careful systems handling (rollout efficiency, uneven sequence lengths)
- Distillation from strong reasoning traces

## Lecture 17: Policy Gradient Mechanics

Lecture 17 gives the mathematical and practical mechanics behind reasoning RL updates:

- Policy-gradient derivation for sequence generation
- Sparse reward variance issues
- Baselines and advantage intuition
- Group-relative normalization logic
- KL-regularized update intuition
- Toy experimental walkthrough showing how reward shaping choices alter training behavior

It is the "close the loop" lecture that makes A5-style algorithms legible from first principles rather than just library calls.

## Lecture 18: Guest Lecture (Junyang Lin)

The Spring 2025 schedule lists this as a guest lecture; publicly linked detailed slides/materials are not attached on the archive page.  
So, for this post, the concrete lecture-level summary is limited to the schedule metadata.

## Lecture 19: Guest Lecture (Mike Lewis)

Likewise, the final guest lecture is listed in the schedule without a public lecture artifact linked directly from the archive table.  
Summary here is therefore constrained to official schedule-level information.

---

## Assignment Arc (How the Lectures Land in Code)

The lecture progression maps directly onto the five assignments:

- **A1 Basics**: tokenizer + transformer + optimizer + loop/checkpointing
- **A2 Systems**: profiling, FlashAttention-style optimization, DDP, optimizer sharding
- **A3 Scaling**: fit scaling laws and choose compute-optimal model setup under budget
- **A4 Data**: crawl-processing filters, PII/harm filtering, dedup, train/evaluate filtered corpus
- **A5 Alignment/Reasoning RL**: SFT + expert iteration + GRPO-style methods on reasoning tasks  
  Optional supplement: safety/instruction tuning + DPO-like preference optimization

Together they force students to integrate theory, infra, and evaluation in one continuous engineering pipeline.

---

## Why This Offering Is Valuable

Spring 2025 CS336 is unusually strong because it does not separate "ML ideas" from "systems reality."

- Architecture choices are taught alongside memory/throughput implications.
- Parallelism is taught with concrete communication primitives and failure modes.
- Scaling laws are treated as planning tools tied to actual budgets.
- Data quality and legal constraints are treated as first-class technical constraints.
- Alignment methods are taught with implementation details and known failure patterns.

That combination is exactly what most LLM work in practice demands.

---

## Sources (Official Materials)

- Archived Spring 2025 course website (schedule, lecture links, assignments):  
  https://cs336.stanford.edu/spring2025/
- Current CS336 site (Spring 2026 page under active construction):  
  https://cs336.stanford.edu/
- Spring 2025 lecture materials repository:  
  https://github.com/stanford-cs336/spring2025-lectures
- Assignment 1 (Basics):  
  https://github.com/stanford-cs336/assignment1-basics
- Assignment 2 (Systems):  
  https://github.com/stanford-cs336/assignment2-systems
- Assignment 3 (Scaling):  
  https://github.com/stanford-cs336/assignment3-scaling
- Assignment 4 (Data):  
  https://github.com/stanford-cs336/assignment4-data
- Assignment 5 (Alignment and Reasoning RL, including optional supplement):  
  https://github.com/stanford-cs336/assignment5-alignment
