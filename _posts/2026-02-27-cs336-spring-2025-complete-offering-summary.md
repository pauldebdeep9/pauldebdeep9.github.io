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

## What CS336 Tries to Teach (Beyond "How Transformers Work")

CS336 is not a survey class and not an API-first class. It is a systems-first, implementation-heavy class built around one central question:

> Given fixed resources (compute, memory, bandwidth, and data), how do you build the best language model you can?

The Spring 2025 version emphasizes four things in parallel:

- **Mechanics**: how tokenization, attention, optimization, and decoding work mathematically and in code.
- **Systems**: where time and memory go on real hardware, and how to recover performance.
- **Scaling discipline**: using empirical laws and constrained budgets instead of wishful extrapolation.
- **Alignment/evaluation realism**: how post-training and evaluation actually behave in practice, including reasoning-focused RL.

This makes the course unusually close to real research engineering workflows.

---

## Course Arc (Spring 2025 Schedule)

The complete offering runs from **April 1, 2025 to June 6, 2025**, with 19 scheduled meetings (including guest lectures) and five major assignments.

### 1) Foundations and Transformer internals (Lectures 1-4)

The first block sets framing and fundamentals:

- Overview and tokenization
- PyTorch fundamentals and resource accounting
- Architecture and hyperparameters
- Mixture-of-Experts (MoE)

The key move here is that tokenization and architecture are treated as **first-order optimization variables**, not boilerplate. Students are asked to reason about representation granularity, sequence length trade-offs, and parameter/compute budgets early.

### 2) Hardware and single-node performance (Lectures 5-6)

This block transitions from model math to kernels:

- GPU architecture and memory hierarchy
- Kernels and Triton

The practical objective is explicit: map transformer primitives onto hardware efficiently. In course terms, arithmetic intensity, memory traffic, and kernel fusion become core modeling concerns, not implementation afterthoughts.

### 3) Distributed training and parallelism (Lectures 7-8)

Parallelism is treated as a design space:

- Collective communication primitives (broadcast, reduce, all-reduce, all-gather, reduce-scatter)
- Distributed data parallel behavior in PyTorch/NCCL
- Data, tensor, and pipeline parallelism

The major learning outcome is to understand **communication as a bottlenecked system** and to engineer overlap between compute and communication.

### 4) Scaling and deployment path (Lectures 9-12)

The next block bridges training-time and serving-time optimization:

- Scaling laws (intro + details)
- Inference systems
- Evaluation

Highlights include compute-optimal scaling intuition, inference latency/throughput trade-offs, and a broad benchmark taxonomy (knowledge, instruction following, agentic capability, reasoning, and safety).

### 5) Data curation and alignment (Lectures 13-17)

The final core block addresses data and post-training:

- Data sources and legal constraints
- Data filtering and deduplication
- Alignment via SFT/RLHF
- RL-style alignment and reasoning-focused policy optimization

This is where students connect upstream data decisions with downstream model behavior and then connect reward design with policy behavior.

### 6) Guest perspectives (Lectures 18-19)

The course ends with guest lectures (Junyang Lin and Mike Lewis), extending course themes to real-world system and model development contexts.

---

## Assignment-by-Assignment Breakdown

Spring 2025 CS336 is best understood through its assignment sequence. Each assignment pushes one part of the stack while preserving continuity with earlier components.

## Assignment 1: Basics (released April 1; due April 15)

This is a full from-scratch LM build:

- Implement byte-pair encoding (BPE) tokenizer
- Implement transformer LM components
- Implement cross-entropy and AdamW
- Implement training loop plus checkpointing
- Train on TinyStories/OpenWebText-style data and report perplexity

Technical emphasis:

- Tokenizer behavior is not treated as preprocessing trivia; it is treated as a compression and optimization problem.
- Model construction includes attention, RoPE, normalization, feedforward blocks, and full LM assembly.
- Resource accounting appears early, forcing students to relate FLOPs/memory to training feasibility.

A1 is intentionally heavy: it establishes the codebase and mental model for all later systems work.

## Assignment 2: Systems (released April 15; due April 30)

A2 is a performance engineering assignment with four pillars:

- Profiling and benchmarking harnesses
- FlashAttention2 implementation (PyTorch and Triton autograd-function paths)
- Distributed data parallel implementations (including gradient synchronization strategies)
- Optimizer state sharding

Technical emphasis:

- Use profiling tools to identify dominant kernels and memory behavior.
- Implement attention kernels and reason about forward/backward memory footprints.
- Engineer communication overlap and bucketization in DDP.
- Reduce optimizer-state memory pressure via sharding.

Conceptually, A2 turns "my model trains" into "my model trains efficiently at scale."

## Assignment 3: Scaling (released April 29; due May 6)

A3 is small in code surface area but high in strategic depth:

- Fit scaling-law relationships under a fixed experimentation budget
- Query a training API with model hyperparameters + target FLOPs
- Use a limited budget (additional FLOPs cap) to predict compute-optimal model size and training setup for a larger run

Technical emphasis:

- Experiment design under constrained budgets
- IsoFLOPs reasoning
- Practical scaling-law fitting and extrapolation

This assignment teaches "how to spend compute" rather than "how to run more compute."

## Assignment 4: Data (released May 6; due May 23)

A4 operationalizes the data pipeline:

- Convert Common Crawl HTML/WET content into training text
- Run language identification
- Detect/mask PII patterns (emails, phone numbers, IPs)
- Apply harmful-content and quality filtering
- Perform exact and fuzzy deduplication (including MinHash-style workflows)
- Train and evaluate models on filtered corpora

Technical emphasis:

- Data quality is measured by downstream impact, not just clean-looking text.
- Filtering is multi-objective: legality, safety, quality, and utility all compete.
- Deduplication is treated as both quality control and contamination-risk mitigation.

A4 makes the case that dataset construction is one of the highest-leverage model interventions.

## Assignment 5: Alignment and Reasoning RL (released May 23; due June 6)

A5 focuses on math reasoning behavior and post-training:

- Zero-shot baseline on MATH
- Supervised fine-tuning from stronger-model reasoning traces
- Expert iteration with verified rewards
- GRPO (group-relative policy optimization) with verified rewards

Technical emphasis:

- Build and evaluate prompt/response tokenization and masked objectives
- Compute response log-probs and token-level statistics
- Implement policy-gradient-style losses and group-normalized reward pipelines
- Compare SFT, expert iteration, and GRPO behavior empirically

This brings together policy optimization and practical reasoning evaluation in one assignment track.

## Optional Supplement: Safety, Instruction Tuning, and Preference Learning

The optional A5 supplement extends alignment from reasoning to broader assistant behavior:

- Instruction tuning on instruction-response pairs
- Baselines and evaluations on MMLU, GSM8K, AlpacaEval, and simple safety tests
- DPO-style preference optimization from pairwise data

Technical emphasis:

- Post-training objective choice directly shapes behavior style and safety/quality trade-offs.
- Evaluation must remain multi-axis; single-metric wins can hide regressions.

---

## What the Lecture + Assignment Design Teaches as a System

A useful way to read Spring 2025 is as one end-to-end pipeline:

1. **Represent text** (tokenizer, corpus decisions)
2. **Build model internals** (attention/FFN/normalization/optimization)
3. **Make training fast and stable** (kernels, mixed precision, distributed execution)
4. **Choose scale intelligently** (scaling-law projections, compute allocation)
5. **Serve efficiently** (inference optimization and decoding systems)
6. **Evaluate honestly** (capability, safety, contamination, realism)
7. **Post-train for behavior** (SFT, preference methods, reasoning RL)

In other words, the course treats modern LLM development as an integrated optimization problem across data, algorithms, systems, and objectives.

---

## Why This Offering Stands Out

Three characteristics make the Spring 2025 offering unusually strong:

- **Execution over abstraction**: students ship working implementations rather than only discussing architectures.
- **Performance accountability**: profiling, communication costs, and memory pressure are explicit grading targets.
- **Alignment grounded in implementation**: SFT/RLHF/GRPO are taught as concrete pipelines with measurable failure modes, not slogans.

A practical inference from the full syllabus and handouts: CS336 is designed to produce engineers who can move from whiteboard idea to scalable training/evaluation loop without outsourcing core understanding to frameworks.

---

## Final Takeaway

The completed Spring 2025 CS336 offering is a full-stack LLM engineering curriculum:

- From tokenizers to kernels
- From single-GPU training to distributed systems
- From scaling-law planning to inference optimization
- From data curation to alignment and reasoning RL

If your goal is to understand language models deeply enough to build and optimize them under real-world constraints, this course structure is one of the clearest public blueprints available.

---

## Sources (Official Materials)

- Archived Spring 2025 course website (schedule, lecture links, assignment links):  
  https://cs336.stanford.edu/spring2025/
- Current CS336 site (shows Spring 2026 as active-construction, tentative schedule):  
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
- Assignment 5 (Alignment and Reasoning RL, plus optional supplement):  
  https://github.com/stanford-cs336/assignment5-alignment
