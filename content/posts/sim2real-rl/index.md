---
title: "RL in Robotics: From Simulation to Reality"
date: 2025-06-24
authors: ["Matin M. Babaei", "Ali Ekhterachian"]
tags: ["reinforcement-learning", "robotics", "sim2real"]
description: "Why sim-to-real matters, what methods work in practice, and clear takeaways for RL practitioners."
draft: true
ShowToc: true
TocOpen: true
cover:
  # <!-- image: "cover.png"     # put cover.png next to this file (optional) -->
  hidden: true
  # alt: "Sim-to-Real overview"
  # caption: "Simulation → Real transfer at a glance"
  # relative: true
---

## Motivation
Real robots face noisy sensors, delays, non-ideal actuators, and safety constraints. Simulators are cheap and safe, but policies trained in sim often underperform on hardware — the **sim-to-real gap**.

## Methods (What actually helps)
- **System Identification** — calibrate dynamics to reduce transition mismatch.
- **Domain Randomization** — vary visuals/physics (masses, friction, lighting, textures) to learn robust policies.
- **Domain Adaptation** — align feature spaces between sim and real (discrepancy/adversarial/reconstruction).
- **Adversarial RL** — train a protagonist vs. perturbing adversary to harden policies against worst-case disturbances.
- **Reward/Observation engineering** — robust sensors, action delay modeling, careful reward design.

## Key Insights
- Too little randomization → overfitting to sim; too much → unstable training. Curriculum/ADR helps.
- DA + DR > either alone for vision-based control.
- Modeling **action delay** and **latency** explicitly often yields large real-world gains.
- Safety constraints must be designed up-front; retrofit is costly.

## Figure
<!-- ![Sim-to-Real gap factors](fig1.png) -->

*Caption:* Observation, action, transition, and reward gaps compound to create performance drop from sim to real.

## Practical Tips
- Start with ID on a few key parameters (mass/friction); then add DR ranges around them.
- Log/measure real-world latencies; emulate them in sim.
- Keep visuals simple; prefer invariances over brittle perception stacks.
- Track offline metrics that correlate with real rollouts (latency-aware returns, robustness tests).

## Conclusion
Bridging sim-to-real is a **pipeline** problem: calibrated sims, structured randomization, feature alignment, and robust training. Combining these moves policies from lab demos to reliable deployments.

## Acknowledgements
Thanks to RLJ Club members and instructors for feedback.
