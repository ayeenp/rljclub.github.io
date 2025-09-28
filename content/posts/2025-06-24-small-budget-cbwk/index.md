---
author: ["Arian Aghamohseni", "Kia Joolai"]
title: "Learning Safely on a Shoestring: Small-Budget Contextual Bandits with Knapsacks"
date: "2025-06-24"
description: "A dual price–based contextual bandit method that enforces hard budgets in the small-budget regime (on the order of 1 over the square root of T) and achieves instance-sensitive, square-root-in-T regret."
summary: "We present a principled, price-based algorithm for Contextual Bandits with Knapsacks (CBwK) that keeps hard budgets safe—even when the per-round allowance is tiny—while achieving regret on the order of (1 + norm of lambda-star) times sqrt(T). The approach pairs penalized optimism, a small safety buffer, and an adaptive step-size schedule."
tags: ["contextual-bandits", "knapsacks", "duality", "fairness", "online-learning", "experiments"]
categories: ["Reinforcement Learning"]
cover:
  image: cover.png
  caption: "Price-based control for constrained contextual bandits."
  hiddenInSingle: true
ShowToc: true
TocOpen: true
math: true
unsafe: True
---

## Introduction

Real deployments often impose **hard limits**: spend caps, exposure quotas, risk budgets, or **fairness** constraints. In such settings we observe **context**, choose **actions**, collect **reward**, and also **consume resources**. Over $T$ rounds, the total consumption must stay within a fixed allowance. This is the **Contextual Bandits with Knapsacks (CBwK)** framework.

The **small-budget regime**—per-round allowance $B=\Theta(1/\sqrt{T})$—is statistically fragile: random fluctuations of cumulative cost are themselves $\Theta(\sqrt{T})$. A learner that “aims at $B$” in expectation can **still overshoot** in realization.

This post distills a simple method that achieves both:
- **Hard feasibility** with high probability;
- **Instance-sensitive regret** $\widetilde{O}\!\big((1+\lVert \lambda^\star\rVert)\sqrt{T}\big)$, where $\lVert \lambda^\star\rVert$ reflects true constraint tightness.

The design is minimal:
1) **Shadow prices** (dual variables) for each constrained resource;
2) **Penalized optimistic** action selection;
3) A tiny **safety buffer** $b_T=\widetilde{O}(1/\sqrt{T})$ internally;
4) **Adaptive step-size** (doubling) for the price update—no prior tuning of the price scale.

![Poster snippet](poster.png#center "Poster snippet")

---

## CBwK in one paragraph

At round $t$: see $x_t$, pick $a_t\in\mathcal{A}$, get reward $r(x_t,a_t)\in[0,1]$, incur cost vector $c(x_t,a_t)\in[0,1]^d$.
**Hard budget** (component-wise):
$$
\sum_{t=1}^{T} c(x_t,a_t)\ \le\ T\,B.
$$
Goal: **maximize** total reward while satisfying the budget **ex post** (with high probability).

When $B\approx 1/\sqrt{T}$, cumulative-cost noise is the same order as $T\,B$. Safety requires explicit control, not just good expectations.

---

## Method: prices + buffer + adaptive responsiveness

### Lagrangian view

For a price vector $\lambda\ge 0$, define
$$
s(x,a;\lambda)\;=\; r(x,a)\;-\;\langle\, c(x,a)-B,\ \lambda \,\rangle.
$$
Strong duality yields
$$
\mathrm{OPT}(B)\;=\;\min_{\lambda\ge 0}\ \mathbb{E}\!\left[\ \max_{a\in\mathcal{A}} s(X,a;\lambda)\ \right].
$$
If we knew $\lambda^\star$, the action $\arg\max_{a} s(x,a;\lambda^\star)$ is optimal among feasible policies.
**Algorithmic plan:** learn $\lambda$ online; act greedily with current $\lambda$.

### Penalized optimism (safe exploration)

Use UCB for reward and LCB for cost. Choose

$$
a_t \in \arg\max_{a}\ \widehat r^{\mathrm{UCB}}_t(x_t,a)\;-\;\Big\langle \lambda_{t-1},\ \widehat c^{\mathrm{LCB}}_t(x_t,a)\;-\;(B-b_T)\Big\rangle .
$$

Then update prices by projected gradient

$$
\lambda_t \leftarrow \Big[\, \lambda_{t-1} + \gamma\big(\widehat c^{\mathrm{LCB}}_t(x_t,a_t) - (B-b_T)\big)\,\Big]_+ .
$$

Prices rise if spending exceeds the buffered target; otherwise they relax. Projection enforces $\lambda \ge 0$.

### Safety buffer $b_T$

Aim **below** the nominal budget: use $B-b_T$ inside the learner, with $b_T\approx \widetilde{O}(1/\sqrt{T})$. This matches concentration of sums and converts average control into **high-probability feasibility**.

### Adaptive step-size (doubling)

The right responsiveness depends on the unknown $\lVert \lambda^\star\rVert$. Start conservative; **double $\gamma$** only if the positive drift of cumulative costs (vs. $B-b_T$) crosses a calibrated threshold. This **auto-calibrates** the controller to the instance.

---

## What the guarantees say

- **Feasibility (w.h.p.)**
  $$
  \sum_{t=1}^{T} c(x_t,a_t)\ \le\ T\,B \quad \text{(coordinate-wise)}.
  $$
- **Regret**
  $$
  \mathrm{Regret}(T)\;=\;\widetilde{O}\!\big((1+\lVert \lambda^\star\rVert)\sqrt{T}\big),
  $$
  with $\lVert \lambda^\star\rVert$ the magnitude of optimal prices (a *tightness* measure). When constraints are loose, $\lVert \lambda^\star\rVert$ is small—an instance-sensitive improvement over crude $1/\min_j B_j$ constants.

---

## Empirical illustration (Appendix G reproduction): rideshare + fairness

The figure below (your `paper.png`) reproduces the **rideshare assistance** example from the public Stanford “learning-to-be-fair” repository. We track **Average Rewards** (left) and **Rideshare Costs** (right) as a function of horizon $T$ for several **fixed** step sizes (legend “PGD Gamma …”) and for the **adaptive** variant.

- **Fairness threshold:** $\tau = 10^{-7}$.
- **Dashed lines:** left: optimistic benchmarks $\mathrm{OPT}(\cdot, B)$; right: the **rideshare budget** (target cost).

![Appendix G reproduction: average reward (left) and rideshare cost (right) vs. T](paper.png#center "Appendix G reproduction: fixed vs. adaptive step-size")

### What the plot shows

- **Tuning sensitivity of fixed step sizes.**
  Some fixed $\gamma$ choices produce attractive average rewards but **linger above** the budget line for long horizons (see the top curve on the right panel). Others enforce cost more tightly but settle at **lower reward**. This is the classic reward–cost trade-off when responsiveness is mis-tuned.

- **Adaptive schedule (Box C).**
  The adaptive method **navigates regimes**: it begins with $\gamma=0.01$ (regime $k=0$), then increases to $\gamma=0.02$ ($k=1$), and finally to $\gamma=0.04$ ($k=2$), where it stabilizes.
  This **auto-scaling** tracks the budget while retaining competitive reward—**without** a priori knowing the correct price scale.

- **Takeaway.**
  Fixed $\gamma$ trades off reward and feasibility in a brittle way; the **adaptive** approach discovers a good responsiveness level from the data and keeps the controller near the budget while maintaining solid reward.

> In words: the right panel is the *safety barometer*. The adaptive curve homes in on the budget line and stays there; some fixed-$\gamma$ curves hug above it (overspending for long), others overcorrect below it (leaving reward on the table).

---

## Practical guidance

- **Estimators.** Use linear/GLM models with regularization that give valid **UCB/LCB**. Keep confidence radii conservative for cost.
- **Monitoring.** Plot (estimated and realized) cumulative cost vs. $T(B-b_T)$; trigger doubling when the **positive drift** passes the regime threshold.
- **Initialization.** Start with small $\gamma \propto 1/\sqrt{T}$; let the regime logic escalate only when the data insist.
- **Cheap fallback.** If possible, include a low-cost baseline action; it effectively caps $\lVert \lambda^\star\rVert$ and improves constants.

---

## Limits (and what’s next)

- **Very tiny budgets** $B \ll 1/\sqrt{T}$: noise dominates—hard feasibility becomes fragile for any learner.
- **Nonlinear constraints** (ratios, quantiles) need new relaxations.
- **Unaware fairness** (no group attribute) calls for robust/proxy constraints and new theory.
- **Non-stationary/adversarial contexts**: pair the dual controller with drift detection/restarts or adversarial OCO techniques.

**Theory-forward directions**
1) **Coordinate-wise adaptive dual steps** (AdaGrad-style): better constants and scaling with many constraints; requires new martingale analysis for data-dependent steps.
2) **Covering constraints** (minimum service): analyze a dual controller that *pushes up* when behind schedule.
3) **Instance-optimal rates**: under non-degenerate LP optima, aim for $o(\sqrt{T})$ or polylog regret; prove matching lower bounds in the small-budget regime.

---

## References

- E. Chzhen, C. Giraud, Z. Li, G. Stoltz. *Small Total-Cost Constraints in Contextual Bandits with Knapsacks, with Application to Fairness.* arXiv:2305.15807.
- M. Badanidiyuru et al. *Bandits with Knapsacks.* FOCS 2013.
- A. Slivkins. *Introduction to Multi-Armed Bandits.* FnT ML, 2019.
