---
author: "Mahshid Dehghani, Homa GhaffarZadeh"
title: "Emergent Communication in Multi-Agent RL"
date: "2025-09-06"
# description: ""
# summary: ""
tags: ["multi-agent", "communication", "emergent language", "Comm_MARL"]
# categories: []
# series: []
# aliases: []
cover:
  image: ma_family.png
  caption: "paper cover"
  hiddenInSingle: true
ShowToc: true
TocOpen: true
---

## Introduction

> Imagine there is a team of robots on a mission. Some are searching for an object, others are navigating a complex maze. They can't see what their teammates see, and they can't shout instructions across the field. This is the challenge of multi-agent reinforcement learning (MARL), where coordination is key but communication is often an afterthought.

Traditional MARL often faces a major hurdle: **_partial observability_**. Each agent only sees a small part of the environment, making it difficult to make globally optimal decisions. Another problem is **_non-stationarity_**, where the environment is constantly changing due to the actions of other agents. But what if we could teach these agents to talk to each other? That’s where the fascinating field of multi-agent deep reinforcement learning with communication **(Comm-MADRL)** comes in.


This field isn't just about sharing raw data; it's about learning a language from scratch. Instead of simply processing text like a large language model, these agents develop a deeper understanding of the world by experiencing the benefits of communication through goal-oriented tasks.

## The Nine Dimensions of Communication
As the field of Comm-MADRL grows, it’s essential to have a framework for understanding and classifying the different approaches. The paper *"A survey of multi-agent deep reinforcement learning with communication"* proposes a new, systematic way to do this using nine dimensions. This framework helps researchers analyze, develop, and compare different systems.


The following table shows the key dimensions to consider:

![Linguistic Elements](Dim2.png#center)

## Teaching Agents to Talk: The Two Core Approaches
One of the foundational steps in this research was figuring out how to let agents learn to communicate from the ground up. The paper *"Learning to Communicate with Deep Multi-Agent Reinforcement Learning"* introduced two innovative methods to do just that.

**Reinforced Inter-Agent Learning (RIAL):** This approach uses a single deep Q-learning model, shared by all agents, to make decisions. The agents' actions include sending messages, which are treated just like any other action in the environment. This is a more traditional, decentralized approach to learning.
![RIAL - RL based communication](RIAL.PNG#center)

**Differentiable Inter-Agent Learning (DIAL):** This is where things get really interesting. DIAL allows for a more direct, end-to-end learning process. During training, error derivatives—the signals that tell a neural network how to adjust—can be backpropagated through the communication channel itself. Think of it like agents whispering advice to each other, and the network can immediately figure out if the advice was helpful or not. The training is centralized, but the learned policies can be executed in a decentralized way.
![DIAL - Differentiable communication](DIAL.PNG#center)

## Emergent Language
Sometimes, the goal isn’t just to solve a task, but to see if agents can develop something that looks and feels like human language. This is the field of emergent language (EL), which a third paper, *"Emergent Language: A Survey and Taxonomy,"* explores in detail. EL research is distinct from traditional natural language processing (NLP) because it focuses on how agents develop and learn their own language through a process of grounded, goal-oriented interaction.

The research breaks down communication into a "semiotic cycle" between a speaker and a listener. The speaker takes a goal and their world model, conceptualizes a message, produces an utterance, and sends it to the listener. The listener then comprehends the utterance, interprets it, and takes an action in their environment. This continuous cycle of meaning-making and action is crucial to understanding how language emerges.
![Linguistic Elements](SL.png#center)

**EL Metrics** check if EL shows language-like qualities. Grounding measures alignment with the environment, compositionality tracks how structure maps to meaning, consistency ensures reliable use of words, generalization tests adaptability to new tasks, and pragmatics captures efficiency, predictability, and cooperation in communication.
![Linguistic Elements](el_metrics.png#center)

## What's Next?
The research papers suggest several exciting future directions:

- **Non-Cooperative Settings:** Most research has focused on cooperative agents, but what happens when agents have mixed or even competitive goals? How would communication evolve then?

- **Heterogeneous Agents:** Current work often assumes all agents are the same. Future research could explore communication between agents with different capabilities and goals.

- **Beyond Text:** Can agents communicate using non-textual data like voice or gestures?

In the end, teaching agents to talk isn't just a technical challenge; it's a step towards creating more flexible and useful AI systems that can work seamlessly with each other—and perhaps one day, with us.

## References

[1] [Zhu, Changxi, Mehdi Dastani, and Shihan Wang. A Survey of Multi-Agent Deep Reinforcement Learning with Communication. 2024. arXiv, https://arxiv.org/abs/2203.08975.](https://arxiv.org/abs/2407.10583)

[2] [Foerster, Jakob N., Yannis M. Assael, Nando de Freitas, and Shimon Whiteson. Learning to Communicate with Deep Multi-Agent Reinforcement Learning. 2016. arXiv, https://arxiv.org/abs/1605.06676.](https://arxiv.org/abs/2212.10420)

[3] [Peters, Jannik, et al. “Emergent Language: A Survey and Taxonomy.” Autonomous Agents and Multi-Agent Systems, vol. 39, no. 1, Springer, 2025, https://doi.org/10.1007/s10458-025-09691-y.](https://arxiv.org/abs/2407.10583)
