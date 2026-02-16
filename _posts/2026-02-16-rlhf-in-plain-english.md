---
layout: post
title: "RLHF in Plain English: How LLMs Learn to Follow Humans (Without Becoming Robots)"
date: 2026-02-16 22:00:00 +0530
categories: [genai, alignment]
tags: [rlhf, llm, dpo, post-training]
excerpt: "A practical high-level guide to RLHF: what it is, why it exists, and how it works."
---

If you've used an LLM that feels "helpful" rather than merely "fluent," you've already experienced the result of **RLHF** - **Reinforcement Learning from Human Feedback**. RLHF is one of the core post-training techniques that turns a raw next-token predictor into an assistant that better matches what people *actually want*.

This post explains RLHF at a practical, high-level level: what it is, why it exists, and how it works.

---

## The problem RLHF tries to solve

A base language model is trained to predict the next token on large text corpora. That's great for language fluency, but it doesn't guarantee:

- **Helpfulness**: answering the question you meant, not the question you literally wrote
- **Harmlessness**: avoiding unsafe outputs
- **Honesty**: saying "I don't know" instead of improvising
- **Instruction following**: doing what you ask in the format you want

In short: *next-token prediction isn't the same as being a good assistant.*

RLHF is one way to bridge that gap by injecting a human preference signal.

---

## The RLHF pipeline (the 3-stage mental model)

Most RLHF systems can be understood as three stages:

### 1) Supervised Fine-Tuning (SFT): "Imitate good answers"

You start with a base model and fine-tune it on examples of *good* assistant behavior:

- prompt -> ideal response

This teaches the model the general shape of "assistant-style" outputs. Think of it as a strong behavioral prior.

### 2) Reward Modeling (RM): "Learn what humans prefer"

Now we need a way to score responses automatically.

Humans are asked to compare outputs:

- prompt + response A vs prompt + response B
- "Which is better?"

From many such comparisons, you train a **reward model** that predicts which response a human would prefer. It becomes a learned scoring function:

> Reward(prompt, response) -> scalar score

This reward model is the key trick: it turns messy human judgments into something you can optimize.

### 3) Reinforcement Learning (RL): "Optimize the policy to score higher"

Finally, you update the assistant model (the **policy**) to produce responses that get higher scores from the reward model - while keeping it close to the SFT model so it doesn't drift into weird behavior.

This typically looks like:

- generate responses
- score with reward model
- update model to increase expected reward
- apply a "don't drift too far" constraint (commonly via a KL penalty)

The outcome: a model that tends to produce outputs humans rate as better.

---

## What makes RLHF *work* (and what can go wrong)

### Why it works

RLHF focuses training on what humans care about most:

- clarity, completeness, correctness (to some extent)
- style/tone alignment
- refusal behavior and policy compliance
- instruction adherence

It's targeted signal, not just broad web text.

### Where it breaks

RLHF is also famous for "you get what you measure" behavior:

- **Reward hacking**: model learns patterns that score well but aren't truly better
  (e.g., overly verbose answers, excessive hedging, generic safety disclaimers)
- **Preference bias**: if labelers prefer a style, the model converges to that style
- **Goodhart's law**: once the reward score becomes the goal, it stops being a perfect proxy for quality
- **Hallucination isn't magically solved**: RLHF can reduce it, but it doesn't create ground-truth knowledge

This is why modern alignment stacks often combine RLHF-style approaches with data curation, tool use, retrieval, policy constraints, and evaluation.

---

## RLHF vs "preference tuning" methods like DPO

RLHF is a family idea: "optimize against human preference."

Some teams now use alternatives that skip explicit RL loops (e.g., **DPO**: Direct Preference Optimization). Conceptually:

- RLHF: learn reward -> do RL optimization
- DPO: directly optimize the model using preference pairs without a separate RL step

You'll still see RLHF used as the umbrella term in casual conversation, even when the method is technically DPO or something adjacent.

---

## A simple intuition you can remember

- **SFT teaches** the model how a good answer *looks*.
- **Reward modeling teaches** what humans *like more*.
- **RL teaches** the model to *choose* those better-liked behaviors more often.

That's RLHF.

---

## TL;DR

RLHF is a post-training method that aligns LLM behavior with human preferences. It typically involves (1) supervised fine-tuning on good examples, (2) learning a reward model from human comparisons, and (3) optimizing the assistant model to score higher on that reward while staying close to the original behavior.
