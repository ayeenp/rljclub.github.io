---
title: "Interpretable & Efficient Deep RL for Autonomous Driving"
date: 2025-09-11
authors: ["Danial Parnian", "Mohammadamin Kiani"]
description: "Poster session summary: interpretable perception and white-box control with efficient deep RL for urban driving."
summary: "Two complementary angles: (A) latent MaxEnt RL that explains what the agent sees; (B) ICCT trees that explain why it acts — with training and parameter efficiency."
tags: ["Autonomous Driving", "Deep RL", "Interpretability", "Efficiency"]
categories: ["Reinforcement Learning"]
cover:
  image: figs/poster-thumb.png
  caption: "Poster snippet — Interpretable & Efficient Deep RL for AD"
  alt: "Poster snippet showing key modules and results"
image: "figs/poster-thumb.png"
caption: "Poster snippet — Interpretable & Efficient Deep RL for AD"
alt: "Poster snippet showing key modules and results"
ShowToc: true
TocOpen: true
math: true
lastmod: 2025-09-11
draft: false
---

We explore **interpretable** and **compute-efficient** Deep RL for autonomous driving. The poster highlights two complementary contributions:

* **(A) Interpretable perception** with a *latent world model* trained under **maximum-entropy RL**. The latent state summarizes multi-modal inputs (camera + LiDAR) and is decoded into a human-readable **bird’s-eye semantic mask**, letting us inspect *what the agent believes it sees*:contentReference[oaicite:2]{index=2}.
* **(B) Interpretable control** with **ICCT** — *Interpretable Continuous Control Trees*. These are compact decision trees where each leaf is a *sparse linear controller*, allowing us to trace *why the agent chose a particular action*:contentReference[oaicite:3]{index=3}.

{{< figure src="figs/poster-thumb.png" caption="Poster overview: the complete pipeline, including perception via latent MaxEnt RL and control via ICCT, with interpretability probes and comparative results." >}}

---

## Motivation & Problem

Deep RL promises adaptive, end-to-end driving policies, but practical deployment is hindered by two issues:

* **Black-box behavior.** Policies based on deep networks are notoriously hard to audit. For a safety-critical application like autonomous driving, stakeholders need **human-auditable explanations**.
* **Scalability.** Urban driving involves complex interactions and dense traffic, requiring algorithms that can **learn efficiently** from limited simulation data.

Our goal is to build policies that are both **trustworthy** and **robust**, while offering **fast learning** and **transparent decision-making**. We approach this from two angles:  
(A) interpretable **perception** and (B) interpretable **control**.

---

## (A) Latent MaxEnt RL — Interpretable Perception

Traditional end-to-end RL maps raw sensor data directly to control, leaving little room for interpretation. We instead insert a **latent world model**:

* The environment is modeled as an MDP  
  $\mathcal{M}=\langle\mathcal{S},\mathcal{A},R,T,\gamma,\rho_0\rangle$.
* A compact latent state $z_t$ encodes the scene, learned from multi-modal inputs (camera + LiDAR).
* Control is optimized with **maximum-entropy RL** (SAC in the latent space):

$$
\max_{\phi}\; \mathbb{E}\Big[\sum_{t=1}^H r(z_t,a_t)\; -\; \log\pi_\phi(a_t\mid z_t)\Big].
$$

A **decoder** reconstructs a semantic bird’s-eye mask from $z_t$, used both as supervision during training and as an interpretability tool at test time.

{{< figure src="figs/1.png" caption="Sequential latent model: the agent acts in compact latent states $z_t$, while a decoder reconstructs a semantic mask for interpretability." >}}  
{{< figure src="figs/2.png" caption="Example decoded semantic mask. It highlights the drivable map, planned route, surrounding vehicles, and the ego car — making the agent’s perception directly interpretable." >}}

**Mask quality metric** (average pixel difference, lower is better):  
$$
e = \frac{1}{N} \sum_i \frac{\|\hat m_i - m_i\|_1}{W \times H \times C}.
$$

---

### Results & Insights

{{< figure src="figs/7.png" caption="Reconstructions: decoded masks closely match ground-truth labels, capturing lanes, objects, and planned routes." >}}  
{{< figure src="figs/8.png" caption="Learning curves: latent MaxEnt RL converges faster and achieves higher asymptotic performance than standard end-to-end RL baselines (DQN, DDPG, TD3, SAC)." >}}

**Findings:**

* **Efficiency.** Learning in the latent space reduces sample complexity and accelerates training.  
* **Faithfulness.** Decoded masks remain accurate across varied traffic scenes, enabling **post-hoc auditing**.  
* **Failure modes.** Rare or occluded objects can lead to imperfect masks, often foreshadowing downstream control errors:contentReference[oaicite:4]{index=4}.

---

## (B) ICCT — Interpretable Control via Differentiable Trees

While the latent model explains *what is seen*, the controller is still typically a black-box MLP. To address this, we introduce **Interpretable Continuous Control Trees (ICCTs)**:contentReference[oaicite:5]{index=5}.

### Core Design

* **Decision nodes:** crisp rules on a single feature, e.g. *if speed > threshold*.  
* **Leaves:** sparse linear controllers $a_d = \sum_j \beta_{dj} x_j + \delta_d$.  
* **Differentiable training:** start with “soft” (fuzzy) splits and gradually crispify into human-readable rules.

{{< figure src="figs/fig1_2.png" caption="ICCT pipeline: differentiable decision-tree training with crispification. Each node splits on one interpretable feature; leaves are sparse linear controllers." >}}

### Algorithm (simplified)

```text
1) NODE_CRISP: pick a single feature x_k and threshold b
2) OUTCOME_CRISP: branch left/right with a hard decision
3) ROUTE: follow nodes to reach a leaf controller
4) SPARSIFY: enforce k-hot selection for interpretability
5) ACTION: sample during training (exploration) or use leaf mean (exploitation)
```

# Results

{{< figure src="figs/fig5_2.png" caption="Physical demonstration: ICCT controlling the ego vehicle in a 14-car traffic scenario, showing interpretable real-world feasibility." >}}

{{< figure src="figs/learning_curves.png" caption="ICCT training curves across urban driving tasks. ICCT matches or surpasses black-box MLP baselines, despite orders-of-magnitude fewer parameters." >}}

## Key Highlights

- **Performance.** ICCT matches or even outperforms deep MLPs, with up to **33% gains** in some driving benchmarks.  
- **Efficiency.** Policies use **300×–600× fewer parameters**, reducing memory and compute costs.  
- **Interpretability.** Small trees with sparse linear controllers are **auditable** and amenable to formal verification.  

---

# Comparison & Synergy

The two approaches address complementary gaps:

- **Latent MaxEnt RL** → explains *what the agent perceives*.  
  *Limitation:* the policy remains a black-box controller.  

- **ICCT** → explains *why the agent acts*.  
  *Limitation:* assumes access to meaningful features.  

**Synergy.** By chaining them — feeding semantic features from (A) into interpretable controllers in (B) — we can achieve a fully **white-box perception-to-action pipeline**.

---

# Downloads

- 📄 **Poster (PDF):** [Download](./poster.pdf)  
- 🖥️ **Slides (PDF):** [Download](./slides.pdf) | **PPTX:** [Download](./slides.pptx)

---

# References

1. Chen, Li, Tomizuka. *Interpretable End-to-End Urban Autonomous Driving with Latent Deep RL*. arXiv:2001.08726 (2020).  
2. Paleja, Niu, Silva, et al. *Learning Interpretable, High-Performing Policies for Autonomous Driving*. arXiv:2202.02352 (2023).  
3. Prakash, Avi, et al. *Efficient and Generalized End-to-End Autonomous Driving with Latent Deep RL and Demonstrations*. arXiv:2205.15805 (2022).
