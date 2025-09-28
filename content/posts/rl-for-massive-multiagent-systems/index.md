---
author: ["Armin Khosravi", "Mohammad Amin Abbasfar"]
title: "Reinforcement Learning for Massive Multi-Agent Systems"
date: "2025-09-11"
description: "A review on scalable reinforcement learning methods for massive multi-agent systems."
# summary: ""
tags: ["Multi-Agent RL", "Massive Multi-Agent Systems", "Scalability", "Mean-Field", "Parameter Sharing", "Transfer Learning", "Curriculum Learning", "Graph Neural Networks", "Model-Based RL"]
# categories: []
# series: []
# aliases: []
cover:
  image: cover.png
  caption: "paper cover"
  hiddenInSingle: true
ShowToc: true
TocOpen: true
---

## Introduction

In many real-world systems, we do not just have one agent learning to act, but hundreds or even thousands. Think of traffic lights in a city, drones in a swarm, energy grid nodes, or even massive team-based video games like Google Football or SMACv2. All of these require agents to coordinate in a scalable way.

{{< figure src="./SMACv2.png" alt="SMACv2 image"
caption="Screenshots from SMACv2 showing agents battling the built-in AI." >}}


The difficulty comes from the explosion of interactions as the number of agents grows. Communication bandwidth is limited, agents often have only partial information, and centralized methods become computationally intractable. That is why there is a push for new reinforcement learning methods that scale gracefully to thousands of agents—balancing efficiency, decentralization, and coordination. This article explores some approaches for addressing these scalability challenges in massive multi-agent systems (MMAS).


## Background

### Single-Agent Reinforcement Learning

A Markov decision process (MDP) is defined by the tuple $(S, \mathcal{A}, P, R, \gamma, \rho_0)$ where:
- $S$ is the state space.
- $\mathcal{A}$ is the action space.
- $P: S \times \mathcal{A} \times S \rightarrow [0, 1]$ is the transition function defining the probability of going to the next state $s^\prime$ given the current state $s$ and action $a$.
- $\mathcal{R}: S \times \mathcal{A} \times S \rightarrow \mathbb{R}$ is the reward function that assigns a real number to each experience $(s, a, s^\prime)$ showing how good the experience is.
- $\gamma \in [0, 1)$ is the discount factor.
- $\rho_0$ is the initial state distribution.

The agent’s goal is to learn a policy $\pi:S \times \mathcal{A} \rightarrow [0, 1]$ to act in such a way that maximizes the expected return from the initial state distribution $\rho_0$, where the return is defined as the cumulative discounted reward:

$$ G_t = \mathbb{E_{\tau \sim p_\pi}} \left[\sum_{t=0}^{\infty} \gamma^t \mathcal{R}(s_t, a_t, s_{t+1}) \right]. $$

### Multi-Agent Reinforcement Learning

In Multi-Agent Reinforcement Learning, the single MDP framework is extended to a Markov Game (also called a Stochastic Game) to support multiple agents. A Markov Game for $N$ agents is defined by the tuple $(S, \mathcal{A}^1, \ldots, \mathcal{A}^N, P, \mathcal{R}^1, \ldots, \mathcal{R}^N, \gamma, \rho_0)$, where:
- $S$ is the set of states common to all agents.
- $\mathcal{A}^i$ is the action space of agent $i$.
- $P: S \times \mathcal{A}^1 \times \ldots \times \mathcal{A}^N \times S \rightarrow [0, 1]$ is the transition function that defines the probability of moving to state $s^\prime$ given the joint action $(a^1, \ldots, a^N)$ taken in state $s$.
- $\mathcal{R}^i: S \times \mathcal{A}^1 \times \ldots \times \mathcal{A}^N \times S \rightarrow \mathbb{R}$ is the reward function for agent $i$, which depends on the state and the joint action of all agents.
- $\gamma \in [0, 1)$ and $\rho_0$ remain the same. 

The goal of each agent $i$ is to find a policy $\pi^i:S \times \mathcal{A^i} \rightarrow [0, 1]$ to maximize its own expected cumulative discounted return:
$$ G^i_t = \mathbb{E_{\tau \sim p_{\pi^i}}} \left[\sum_{t=0}^{\infty} \gamma^t \mathcal{R}^i(s_t, a_t, s_{t+1}) \right]. $$


## Local Interactions and Mean-Field Approaches

In massive multi-agent systems, if every agent tries to directly model every other agent’s behavior, the complexity grows quadratically ($\mathcal{O}(N^2)$), quickly becoming impossible to compute or learn. Local interaction models solve this by assuming that an agent only needs to consider its immediate neighborhood—like a traffic light only needing information from nearby intersections, not the entire city.

Another solution is **mean-field reinforcement learning (MFRL)**, where instead of modeling each agent individually, an agent treats the rest of the population as a “statistical average.” This average (the mean field) represents the expected behavior of other agents, turning a huge multi-agent problem into something closer to a single-agent RL problem with an extra input (the mean action or state distribution).
This approach has its roots in **statistical physics** (where large particle systems are simplified via mean-field theory), and it has become a powerful way to make reinforcement learning scalable for thousands of agents.

An example of such algorithms is Mean-Field Q-learning (MFQ), where the collective effect of other agents is treated as an “average policy,” letting each agent adapt without tracking everyone else.


{{< figure src="./mean_field.png" alt="mean field image"
caption="Mean field approximation. Each agent is represented as a node in the grid, which is only affected by the mean effect from its neighbors (the blue area). Many-agent interactions are effectively converted into two-agent interactions." >}}


### How it helps

- **Reduces complexity**: Interaction costs go from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$ or even constant-time approximations.

- **Supports decentralization**: Each agent only needs partial, local information.

- **Improves communication efficiency**: Avoids overwhelming networks with messages between thousands of agents.

### Advantages

- **Dramatically reduce complexity** from quadratic to linear or constant-time approximations.

- **Enable truly decentralized learning**, since each agent only needs local information.

- **Proven to scale** to hundreds or thousands of agents in swarm robotics and traffic networks.

### Limitations

- **Oversimplifies interactions**—agents far away might still matter in certain domains (e.g., cascading failures in power grids).

- **Mean-field assumptions can break** when populations are small or highly heterogeneous.







## Knowledge Reuse and Training Acceleration

Training reinforcement learning agents usually requires millions of interactions with the environment. For massive systems with hundreds or thousands of agents, this cost explodes. Knowledge reuse methods aim to **accelerate learning** by not starting from scratch for every agent or environment. There have been multiple methods attempted to use the notion of knowledge reuse.

One idea is to use **Parameter sharing**. If agents are homogeneous (e.g., drones in a swarm, or traffic lights in a grid), they can share a single policy network. Each agent just feeds in its own local observation and gets an action, but the parameters are globally shared. This drastically reduces training complexity and stabilizes learning.

Another approach is **Transfer learning**. A policy trained in one scenario (like a smaller traffic network) can be reused in a larger scenario, avoiding relearning from zero.

**Curriculum learning** can also be used for learning acceleration. The environment’s complexity is scaled gradually—first train with fewer agents or easier tasks, then introduce more agents and harder conditions. This mirrors how humans learn by tackling simpler problems first.

Together, these strategies make it feasible to scale from small experiments to large systems without an exponential blowup in training time.

{{< figure src="./parameter_sharing.png" alt="parameter sharing image"
caption="An illustration of parameter sharing in neural networks. (Source: [www.researchgate.net](https://www.researchgate.net/figure/Hard-parameter-sharing-for-Multi-Task-Learning-integrated-in-neural-networks_fig2_336935196))" >}}

### How it helps

- Parameter sharing: Homogeneous agents (e.g., traffic lights at intersections) can share one neural policy, dramatically reducing the number of trainable parameters.

- Transfer learning: Policies from smaller or simpler environments can be transferred to larger, more complex setups.

- Curriculum learning: Start with a small number of agents or simple tasks, then gradually scale up.

### Advantages

- **Cuts down training time** by reusing parameters or pre-trained policies.

- Curriculum learning **stabilizes training** by gradually increasing task difficulty.

- **Makes experiments feasible** that would otherwise require enormous computation.

### Limitations

- **Works best for homogeneous agents**; harder when agents are diverse with unique roles.

- Transfer learning **may cause negative transfer** if the source and target environments differ too much.

- Curriculum design often **requires manual tuning and domain knowledge**.










## Complexity Reduction and Decentralized Optimization

One of the biggest barriers in MMAS is the curse of dimensionality: as the number of agents grows, the joint state and action spaces grow exponentially. Complexity reduction techniques explicitly address this by simplifying the optimization problem. This can be done using several techniques.

- **Localized optimization**: Many real-world systems have a locality principle: the actions of far-away agents matter very little to a given agent. For example, in traffic control, what happens 20 intersections away barely affects your local intersection. Algorithms exploit this by ignoring distant influences.

- **Scalable actor-critic methods**: Actor-critic algorithms (policy gradient + value estimation) can be adapted for large systems by decomposing the value function into local components and stabilizing updates across many agents.

- **Value factorization**: Instead of learning a single global value function (which is intractable), methods like QMIX or its extensions factorize the value into per-agent or per-group utilities that can be optimized separately.

This way, each agent deals with a smaller, tractable optimization problem, while the collective behavior still leads to coordinated system-level performance.

### How it helps

- Localized optimization: Agents **focus only on relevant parts** of the system—e.g., distant agents in a traffic grid have negligible impact on a local intersection.

- Scalable actor-critic methods: **Reduce variance and stabilize training** when dealing with large populations.

- Factorization methods: **Decompose global value functions** into local utilities that can be optimized independently.


### Advantages

- **Tackles the curse of dimensionality directly** by ignoring irrelevant interactions.

- Value factorization methods allow **learning tractable local utilities** while preserving global coordination.

- **More scalable in memory and compute**, enabling use in domains like large-scale network control.

### Limitations

- Local optimization **may miss emergent global behaviors** (e.g., city-wide traffic waves).

- Decomposition methods **can introduce bias** if the true global value function is not easily factorized.

- Balancing locality vs. global optimality is **still an open research problem**.






## Architectural Innovations

Traditional neural networks like MLPs or CNNs are not well-suited to massive multi-agent systems because they don’t naturally handle relational structure or variable numbers of agents. Architectural innovations in deep learning address this gap. Some of those are listed below. 

- **Graph Neural Networks (GNNs)**: Agents are nodes, and interactions are edges. By passing “messages” along graph connections, GNNs can capture relational dynamics efficiently (e.g., traffic lights connected by road segments). They scale naturally as the number of agents grows.

- **Attention mechanisms**: Attention lets agents selectively focus on the most relevant other agents, rather than treating all equally. For instance, in a swarm of 1000 drones, an agent might only “attend” to its 3 closest neighbors.

- **Transformers**: Originally designed for language, transformers can model long-range dependencies. Memory-efficient variants (like Sable) allow transformers to be applied to thousands of agents and long time horizons.

- **Permutation-invariant models**: Architectures like SPECTra ensure that the model’s output does not depend on the arbitrary ordering of agents—important since in MMAS, agents are often interchangeable.

These innovations make neural architectures expressive enough to capture coordination in huge agent populations, while still being computationally feasible.

{{< figure src="./gnn.png" alt="gnn image"
caption="An example structure of a graph neural network (Source: [www.geeksforgeeks.org](https://www.geeksforgeeks.org/deep-learning/what-are-graph-neural-networks/))" >}}


### How it helps

- Graph Neural Networks (GNNs): Encode agents as graph nodes and interactions as edges. **Efficient for structured environments** (traffic grids, communication networks).

- Attention mechanisms: Decide which agents are most relevant for coordination, **avoiding unnecessary communication**.

- Transformers: **Memory-efficient** variants (e.g., Sable) handle thousands of agents across long time horizons.

- Permutation invariance: Architectures like SPECTra **generalize to variable team sizes**, which is essential when the number of agents is not fixed.


### Advantages

- GNNs and attention models naturally **capture the relational structure** between agents.

- Transformers and permutation-invariant models **generalize to variable team sizes**.

- **Proven to scale** to 1000+ agents in simulations, enabling flexible coordination strategies.

### Limitations

- **High computational cost**—Transformers and GNNs still struggle with very large graphs or very long horizons.

- **Require careful design** of graph structure or attention scope.

- **Risk of overfitting** to training environments, limiting generalization.






## Model-Based Approaches

Most RL methods are model-free: they learn directly from trial and error, requiring vast amounts of data. In MMAS, this is often infeasible because simulating millions of interactions is too costly. Model-based approaches add efficiency by having agents **learn local models of the environment’s dynamics**.

Each agent tries to predict how its local environment evolves—how its neighbors’ actions will affect future states—and then uses this predictive model to plan ahead (e.g., via tree search, planning rollouts, or policy improvement).

The key idea is **decentralized modeling**: instead of requiring a single massive model of the entire system, each agent learns its own simplified dynamics model that only requires local information. These models can be combined or shared where needed, but remain scalable because they do not attempt to represent the entire multi-agent environment.
By reducing the reliance on raw trial-and-error, model-based RL enables more sample-efficient learning, faster convergence, and better generalization to new scenarios.


{{< figure src="./model_based.png" alt="model based image"
caption="In decentralized model-based methods, each agent has its own version of the world model, which can be updated using the Communication Block. It is sufficient to send only stochastic state $z^i_{t-1}$ and action $a^i_{t-1}$ from the previous step for each agent in order to obtain feature vectors $e^i_t$. Agent $i$ then can use the updated world model and its current observation $o^i_t$ to output its next action $a^i_t$." >}}


### How it helps

- **Sample efficiency**: Agents can simulate outcomes without requiring millions of environmental interactions.

- **Better coordination**: Agents can plan ahead while still respecting decentralized constraints.

- **Scalable learning**: Local models are easier to learn and share than global dynamics across thousands of agents.


### Advantages

- Much higher **sample efficiency**—fewer interactions needed to learn effective policies.

- Agents **can plan ahead**, leading to smoother coordination.

- Local models are modular and **easier to learn** than global dynamics.

### Limitations

- Learning accurate models is **difficult in highly stochastic** or partially observable environments.

- **Computational overhead** of planning can grow quickly.

- **Requires strong assumptions about locality**; errors in the learned model can cascade across the system.



## Conclusion

Massive multi-agent systems are increasingly relevant in domains like smart cities, robotics, and network control. Traditional RL methods break down in these settings, but new advances in mean-field approximations, local communication, knowledge reuse, and scalable neural architectures are pushing the frontier.

The shift toward practical, scalable reinforcement learning is opening the door to controlling systems once thought impossible to handle at scale.



## References

[1] [Yang, Yaodong, Rui Luo, Minne Li, Ming Zhou, Weinan Zhang, and Jun Wang. “Mean Field Multi-Agent Reinforcement Learning.” Proceedings of the 35th International Conference on Machine Learning (ICML), vol. 80, edited by Jennifer Dy and Andreas Krause, PMLR, July 2018, pp. 5567–5576.](https://proceedings.mlr.press/v80/yang18d.html)

[2] [Ellis, Benjamin, Jonathan Cook, Skander Moalla, Mikayel Samvelyan, Mingfei Sun, Anuj Mahajan, Jakob N. Foerster, and Shimon Whiteson. “SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning.” Advances in Neural Information Processing Systems 36 (NeurIPS 2023) Datasets & Benchmarks Track, edited by Alexander Oh et al., NeurIPS Foundation, 2024, pp. 37567–37593.](https://research.manchester.ac.uk/en/publications/smacv2-an-improved-benchmark-for-cooperative-multi-agentreinforce)

[3] [Yang, Yufeng, Adrian Kneip, and Charlotte Frenkel. “EvGNN: An Event-driven Graph Neural Network Accelerator for Edge Vision.” arXiv preprint, 30 Apr. 2024.](https://arxiv.org/abs/2404.19489)

[4] [Egorov, Vladimir, and Aleksei Shpilman. “Scalable Multi-Agent Model-Based Reinforcement Learning.” Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2022), ACM/IFAAMAS, May 2022, pp. 381–390.](https://spacefrontiers.org/r/10.48550/arxiv.2205.15023)

[5] [Paolo, Giuseppe, Abdelhakim Benechehab, Hamza Cherkaoui, Albert Thomas, and Balázs Kégl. “TAG: A Decentralized Framework for Multi-Agent Hierarchical Reinforcement Learning.” arXiv preprint, 21 Feb. 2025.](https://arxiv.org/abs/2502.15425)

[6] [Chu, Tianshu, Jie Wang, Lara Codecà, and Zhaojian Li. “Multi-Agent Deep Reinforcement Learning for Large-scale Traffic Signal Control.” arXiv preprint, 11 Mar. 2019.](https://arxiv.org/abs/1903.04527)

[7] [Ma, Chengdong, Aming Li, Yali Du, Hao Dong, and Yaodong Yang. “Efficient and Scalable Reinforcement Learning for Large-Scale Network Control.” Nature Machine Intelligence, vol. 6, no. 9, Sept. 2024, pp. 1006–1020.](https://kclpure.kcl.ac.uk/portal/en/publications/efficient-and-scalable-reinforcement-learning-for-large-scale-net)