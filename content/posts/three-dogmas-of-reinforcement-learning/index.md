---
author: "Arash Alikhani"
title: "Three Dogmas of Reinforcement Learning"
date: "2024-08-06"
description: "by David Abel, Mark K. Ho, and Anna Harutyunyan"
# summary: ""
tags: ["dogmas", "reward-hypothesis", "philosophy", "paradigm", "ICML", "RLC", "2024", "RLC24"]
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

This paper critically examines the assumptions commonly accepted in modeling reinforcement learning problems, suggesting that these assumptions may impede progress in the field. The authors refer to these assumptions as "dogmas."

> **Dogma: A fixed, especially religious, belief or set of beliefs that people are expected to accept without any doubts.** *—From the [Cambridge Advanced Learner's Dictionary & Thesaurus](https://dictionary.cambridge.org/dictionary/english)*

The paper introduces three central dogmas:

1. The Environment Spotlight
2. Learning as Finding a Solution
3. The Reward Hypothesis (although not exactly a dogma)

The author argues the true reinforcement learning landscape is actualy like this,

![rl landscape](rl1.png#center)

In the words of Rich Sutton:

> **RL can be viewed as a microcosm of the whole AI problem.**

However, today's RL landscape is overly simplified,

![rl landscape](rl2.png#center)

These three dogmas are responsible for narrowing the potential of RL,

![rl landscape](rl3.png#center)

The authors propose that we consider moving beyond these dogmas,

![rl landscape](rl4.png#center)

To reclaim the true landscape of RL,

![rl landscape](rl5.png#center)

## Background

The authors reference Thomas Kuhn's book, "The Structure of Scientific Revolutions",

![The Structure of Scientific Revolution](book1.jpg#center)

Kuhn distinguishes between two phases of scientific activity,

![scientific activity](science.png#center)

- **Normal Science:** Resembling puzzle-solving.
- **Revolutionary Phase:** Involving a fundamental rethinking of the values, methods, and commitments of science, which Kuhn calls a "paradigm."

Here's an example of a previous paradigm shift in science:

![paradigm shift](science-example.png#center)

The authors explore the paradigm shift needed in RL:

![rl paradigm shift](rl-paradigm-shift.png#center)

## Dogma One: The Environment Spotlight

The first dogma we call *the environment spotlight*, which refers to our collective focus on modeling environments and environment-centric concepts rather than agents.

![Dogma One: The Environment Spotlight](dogma1pic.png#center)

What do we mean when we say that we focus on environments? We suggest that it is easy to answer only one of the following two questions:

1. What is at least one canonical mathematical model of an **environment** in RL?
    - MDP and its variants! And we define everything in terms of it. By embracing the MDP, we are allowed to import a variety of fundamental results and algorithms that define much of our primary research objectives and pathways. For example, we know every MDP has at least one deterministic, optimal, stationary policy, and that dynamic programming can be used to identify this policy.

    ![Dogma One: The Environment Spotlight](dogma1-env.png#center)

2. What is at least one canonical mathematical model of an **agent** in RL?
    - In contrast, this question has no clear answer!

The author suggests it is important to define, model, and analyse agents in addition to environments. We should build toward a canonical mathematical model of an agent that can open us to the possibility of discovering general laws governing agents (if they exist).

![Dogma One: The Environment Spotlight](dogma1-agent.png#center)

![Dogma One: The Environment Spotlight](dogma1.png#center)

## Dogma Two: Learning as Finding a Solution

The second dogma is embedded in the way we treat the concept of learning. We tend to view learning as a finite process involving the search for—and eventual discovery of—a solution to a given task.

![Dogma Two: Learning as Finding a Solution](dogma2pic.png#center)

We tend to implicitly assume that the learning agents we design will eventually find a solution to the task at hand, at which point learning can cease. Such agents can be understood as searching through a space of representable functions that captures the possible action-selection strategies available to an agent, similar to the Problem Space Hypothesis, and, critically, this space contains at least one function—such as the optimal policy of an MDP—that is of sufficient quality to consider the task of interested solved. Often, we are then interested in designing learning agents that are guaranteed to converge to such an endpoint, at which point the agent can stop its search (and thus, stop its learning).

![standard rl](standard_rl.png#center)

The author suggests to embrace the view that learning can also be treated as adaptation. As a consequence, our focus will drift away from optimality and toward a version of the RL problem in which agents continually improve, rather than focus on agents that are trying to solve a specific problem.

![continual rl](continual_rl.png#center)

When we move away from optimality,

- How do we think about evaluation?
- How, precisely, can we define this form of learning, and differentiate it from others?
- What are the basic algorithmic building blocks that carry out this form of learning, and how are they different from the algorithms we use today?
- Do our standard analysis tools such as regret and sample complexity still apply?

These questions are important, and require reorienting around this alternate view of learning.

![standard rl vs continual rl](standard_continual_rl.png#center)

The authors introduce the book "Finite and Infinite Games",

![Finite and Infinite Games](book2.jpg#center)

And the concept of Finite and Infinite Games is summarized in the following quote,

> There are at least two kinds of games, One could be called finite; the other infinite. A finite game is played for the purpose of winning, an infinite game for the purpose of continuing the play.

And argues alignment is an infinite game.

![alignment is an infinite games](alignment.png#center)

![Dogma Two: Learning as Finding a Solution](dogma2.png#center)

## Dogma Three: The Reward Hypothesis

The third dogma is the reward hypothesis, which states "All of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward)."

![reward hypothesis](rewardhyp1.png#center)

![Dogma Three: The Reward Hypothesis](dogma3pic.png#center)

The authors argue that the reward hypothesis is not truly a dogma. Nevertheless, it is crucial to understand its nuances as we continue to design intelligent agents.

The reward hypothesis basically says,

![reward hypothesis](rewardhyp2.png#center)

In recent analysis by [2] fully characterizes the implicit conditions required for the hypothesis to be true. These conditions come in two forms. First, [2] provide
a pair of interpretative assumptions that clarify what it would mean for the reward hypothesis to be true or false—roughly, these amount to saying two things (brwon doors).

![reward hypothesis conditions](alldoors.png#center)

- First, that "goals and purposes" can be understood in terms of a preference relation on possible outcomes.

![food outcome](food_outcome.png#center)
![chess outcome](chess_outcome.png#center)

- Second, that a reward function captures these preferences if the ordering over agents induced by value functions matches that of the ordering induced by preference on agent outcomes.

![reward hypothesis conditions](reward1.png#center)
![reward hypothesis conditions](reward2.png#center)

This leads to the following conjecture,

![reward conjecture](conjecture.png#center)

Then, under this interpretation, a Markov reward function exists to capture a preference relation if and only if the preference relation satisfies the four von Neumann-Morgenstern axioms, and a fifth Bowling et al. call $\gamma$-Temporal Indifference.

![reward conjecture](conjecture-prefs.png#center)
![reward conjecture conditions](constraintdoors.png#center)

- **Axiom 1: Completeness >** You have a preference between every outcome pair.
  - You can always compare any two choices.

- **Axiom 2: Transitivity >** No preference cycles.
  - If you like chocolate more than vanilla, and vanilla more than strawberry, you must like chocolate more than strawberry.

- **Axiom 3: Independence >** Independent alternatives can't change your preference.
  - If you like pizza more than salad, and you have to choose between a lottery of pizza or ice cream and a lottery of salad or ice cream, you should still prefer the pizza lottery over the salad lottery.

- **Axiom 4: Continuity >** There is always a break even chance.
  - Imagine you like a 100 dollar bill more than a 50 dollar bill, and a 50 dollar bill more than a 1 dollar bill. There should be a scenario where getting a chance at 100 dollar and 1 dollar, with certain probabilities, is equally good as getting the 50 dollar for sure.  

These 4 axioms are called the von Neumann-Morgenstern axioms.

![von Neumann-Morgenstern](photo.png#center)

- **Axiom 5: Temporal $\boldsymbol{\gamma}$-Indifference >** Discounting is consistent throughout time.
  - Temporal $\gamma$-indifference says that if you are indifferent between receiving a reward at time $t$ and receiving the same reward at time $t+1$, then your preference should not change if we move both time points by the same amount. For instance, if you don't care whether you get a candy today or tomorrow, then you should also not care whether you get the candy next week or the week after.

Taking these axioms into account, the reward conjecture becomes the reward theorem,

![reward theorem](theorem.png#center)

It is essential to consider that people do not always conform to these axioms, and human preferences can vary.

![human preference](rational_human.png#center)

It is important that we are aware of the implicit restrictions we are placing on the viable goals and purposes under consideration when we represent a goal or purpose through a reward signal. We should become familiar with the requirements imposed by the five axioms, and be aware of what specifically we might be giving up when we choose to write down a reward function.

## See Also

- [David Abel Presentation @ ICML 2023](https://slideslive.com/39015036)
- [David Abel Personal Website](https://david-abel.github.io)
- [Mark Ho Personal Website](https://markkho.github.io)
- [Anna Harutyunyan Personal Website](https://anna.harutyunyan.net)

## References

[1] [Abel, David, Mark K. Ho, and Anna Harutyunyan. "Three Dogmas of Reinforcement Learning." arXiv preprint arXiv:2407.10583 (2024).](https://arxiv.org/abs/2407.10583)

[2] [Bowling, Michael, et al. "Settling the reward hypothesis." International Conference on Machine Learning. PMLR, 2023.](https://arxiv.org/abs/2212.10420)