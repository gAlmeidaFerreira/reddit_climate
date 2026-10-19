# Measuring Real-World Opinion Dynamics: Dyadic and Group Influence on Reddit

Master's research project, Tsinghua SIGS (Institute of Data and Information).
Author: Guilherme Almeida Ferreira. Status: lab presentation of 2026-10-15; measurement step done, model design next.

## Goal

The overall question is: **how can we build an opinion dynamics model of a social network that reproduces the opinion changes observed in real data?** It splits into four steps:

1. How to measure the dynamics in the real world?
2. How to design a model from real-world observations?
3. How to compare the model with real data?
4. Which adjustments bring the model closer to reality?

This repository covers **step 1**. Classical models (DeGroot, Friedkin-Johnsen, bounded confidence) and the control-theoretic literature are rarely validated against data. Here the opinion dynamics are measured on real data first, *without assuming a model beforehand*, with a Kozitsin-style framework applied to Reddit data where opinions are scored directly from the comments.

## Data

Reddit climate-change data of Antelmi, Spagnuolo & Aiello (2026). A *submission* starts a discussion and a *thread* holds the comments under it. Each comment has an author, a score $x \in [0, 1]$ (agreement with "Climate change is a real concern", scored by GPT-3.5) and a parent (the comment it replies to).

## Methodology

Following Kozitsin (2022), opinions are discretized into $m$ classes and the dynamics are described by **transition probabilities** $p(k \mid s, \cdot)$ of a new opinion $k$ given the own previous opinion $s$ and an influence variable. This presentation uses **$m = 3$** (against $[0, 0.35)$, neutral $[0.35, 0.65)$, in favour $[0.65, 1]$).

Each event is one comment:

- **$s$**: the author's last scored comment (own opinion)
- **$k$**: the comment itself (new opinion)
- **Dyadic influence, $l$**: the score of the parent comment
- **Group influence, $g$**: summary of the comments published before the author's reply to the same parent, with at least 2 scored members

Group summaries:

- **Tendency**: the mean (3 levels: against, neutral, in favour)
- **Dispersion**: the unbiased variance, or the share of opposed pairs (the share of pairs of members in opposite classes, a polarization measure), each split into low and high
- Together these give **6 group categories** (3 mean levels $\times$ 2 dispersion levels), so the group tensor is $m \times 6 \times m$ and the dyadic tensor is $m \times m \times m$

Evaluation uses information theory:

- **Information gain**: $I(K;X \mid S) = H(K \mid S) - H(K \mid S, X)$, the bits about the new opinion that $X$ adds beyond the own opinion.
- **Held-out log-loss** of a lookup table $p(k \mid s, \text{summary})$, with user-grouped folds and shrinkage chosen in inner folds, to compare group summaries (R3, R4).
- **Chain-rule decomposition** with Miller-Madow bias correction, user bootstrap intervals and permutation p-values for $I(K;S)$, $I(K;L \mid S)$ and $I(K;G \mid S)$ (R5).

## Results ($m = 3$)

| | |
|---|---|
| Events | 87,032 dyadic $(s, l, k)$; 42,268 group $(s, g, k)$ |
| Output opinion $k$ (dyadic) | against 43,087; neutral 23,179; in favour 20,766 |
| Entropy | $H(K) = 1.506$ bits; $I(K;S) = 0.0115$ bits (dyadic sample) |

- **Own opinion** alone explains little: $H(K \mid S)$ is about 1.49 bits, and every row of $p(k \mid s)$ leans towards "against" because most opinions are against.
- **Influencer**: knowing $l$ raises the probability that the new opinion matches $l$ (diagonal +3.3 to +6.4 percentage points); $I(K;L \mid S) = 0.0091$ bits, significantly above zero.
- **Group summary**: mean + dispersion is the most informative (about 0.0177 bits per event beyond the own opinion); variance and share of opposed pairs are about equal; group size alone is about 0.
- **Group vs influencer** (38,685 events with both): $I(K;S) = 0.0067$, $I(K;L \mid S) = 0.0036$, $\mathbf{I(K;G \mid S) = 0.0172}$ bits, so the group adds the most information about the new opinion.
- **Group transition tensor**: the mean drives the shift (against groups raise "against" and lower "in favour", favour groups the opposite, neutral groups give small shifts); the share of opposed pairs changes the size of the shift by a few points, less than the mean does. Rows with few events are marked † and should be read with care.

Caveats: the information gained is small (roughly 1-2% of $H(K)$), and the results show an association between opinions, not a causal effect.

## Steps completed

- [x] Motivation, research goal and literature review
- [x] Event extraction and discretization ($m = 3$), giving the dyadic and group transition tensors
- [x] Group representations (mean, variance, share of opposed pairs) compared by held-out log-loss
- [x] Information analysis of own, influencer and group opinion, with the group transition tensor and the analysis notebook

## Still to do

- [ ] **Robustness**: other class cuts and $m$, a larger minimum group size, other datasets
- [ ] **Confound checks**: thread-level controls or placebo tests, since a group's mean may also reflect the thread's overall leaning
- [ ] **Model design and long-run analysis**: build the opinion dynamics model from the measured transition tensors, extend Kozitsin's model with the group information, and analyse its long-run behaviour (mean-field equations, equilibria and their stability)
- [ ] **Network topology**: add topology, for example hypergraphs representing the groups
- [ ] **Time slices**: transition probabilities by period of time, and whether their change can be approximated
- [ ] Next research questions: design a model from the observations, compare it with data, adjust it

## Repository layout

```
config.py                settings: class cuts, group size, folds, bootstrap and permutation counts
events.py                builds dyadic and group events from the dataset
tensors.py               transition tensors, summary comparison, information decomposition
notebook/analysis.ipynb  results R1-R5 and the final group tensor
```

## Main references

1. Kozitsin, I. V. (2022). *A general framework to link theory and empirics in opinion formation models.*
2. Antelmi, A., Spagnuolo, C. & Aiello, L. M. (2026). *Modeling the impact of group interactions on climate-related opinion change in Reddit.*