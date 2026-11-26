# Paper notes

Owner: rl-researcher.

## Source tiers

| Tier | Meaning |
| --- | --- |
| **[V]** | **Verified.** PDF downloaded from arXiv this session (2026-11-26), converted with `pdftotext`, and the quoted number read in the text |
| **[K]** | **Keyword-checked.** PDF downloaded and searched, but the claim was not read in full context |
| **[W]** | **WebFetch summary.** Taken from an automated summary of a web page, not read verbatim |
| **[P]** | **Prior knowledge.** Not fetched. Verify before quoting in the brief |

Local copies, which are not part of the repo and may be cleared, are in `/private/tmp/claude-501/-Users-raghavsharma-Desktop-DOOMSDAY/8af00418-f4ee-48ce-a8af-028a5b4c7dbc/scratchpad/papers/`.

Code links are from prior knowledge [P] unless the paper text states them.

---

## Core RL

### PPO: Proximal Policy Optimization Algorithms
Schulman, Wolski, Dhariwal, Radford, Klimov. arXiv 2017. https://arxiv.org/abs/1707.06347 · code: https://github.com/openai/baselines

**Atari settings (Table 5) [V]**

| Setting | Value |
| --- | --- |
| Horizon T | 128 |
| Adam lr | 2.5e-4·α |
| Epochs | 3 |
| Minibatch | 32×8 |
| γ | 0.99 |
| λ | 0.95 |
| Actors | 8 |
| Clip | 0.1·α |
| Value-function coefficient | 1 |
| Entropy | 0.01 |

α is annealed linearly from 1 to 0. MuJoCo used T = 2048, lr 3e-4, 10 epochs and minibatch 64 [V].

**Use:** this is the core learner.

### GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation
Schulman, Moritz, Levine, Jordan, Abbeel. ICLR 2016. https://arxiv.org/abs/1506.02438 · code: none official

- [V] The best results came at intermediate values, γ ∈ [0.96, 0.99] and λ ∈ [0.92, 0.99].
- [V] "The best value of λ is much lower than the best value of γ."

**Use:** λ = 0.95. γ is chosen by horizon arithmetic (RL_RESEARCH.md §3).

### The 37 Implementation Details of PPO
Huang, Dossa, Raffin, Kanervisto, Wang. ICLR Blog Track 2022. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ · code: https://github.com/vwxyzjn/cleanrl

Fetched with WebFetch, which read the first 100k characters. These items are tagged **[W]**: they come from the tool's summary of the page, not from verbatim reading.

- [W] Orthogonal initialisation: gain √2, 0.01 for the policy head, 1 for the value head.
- [W] Adam ε = 1e-5.
- [W] Per-minibatch advantage normalisation.
- [W] Global gradient clip 0.5.
- [W] Value clipping: Engstrom et al. found no benefit, and Andrychowicz et al. found it may hurt.
- [W] LSTM details: zero initial state, reset at done, sequential minibatches built from whole sub-environment sequences, and states reconstructed during training. The LSTM run reuses Atari settings (128 steps, 4 minibatches, 4 epochs).
- [W] MultiDiscrete: the joint log-probability is the sum of the per-component log-probabilities.
- [W] The post did **not** implement truncation bootstrapping, so it is not a source for that.

### What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study
Andrychowicz et al. ICLR 2021. https://arxiv.org/abs/2006.05990 · code: none official [P]

Recommendations [V]:
- PPO loss, starting with clip 0.25.
- Last policy layer initialised 100× smaller.
- Always normalise observations.
- GAE with λ = 0.9, and no PPO-style value clipping.
- γ is among the most important hyperparameters (start at 0.99).
- Adam with β1 = 0.9.

**Caveat:** the study used continuous control only.

### Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO
Engstrom et al. ICLR 2020. https://arxiv.org/abs/2005.12729 · code: https://github.com/MadryLab/implementation-matters [P]

- [K] Code-level optimisations, such as reward scaling, initialisation and lr annealing, drive much of PPO's advantage.

### Time Limits in Reinforcement Learning
Pardo, Tavakoli, Levdik, Kormushev. ICML 2018. https://arxiv.org/abs/1712.00378 · code: none [P]

- [K] If the time limit is not part of the task, bootstrap at truncation ("partial-episode bootstrapping").
- [K] If it is part of the task, make the agent time-aware.

**Use:** RL_RESEARCH.md §3.3.

### DQN: Playing Atari with Deep Reinforcement Learning
Mnih et al. NIPS DL Workshop 2013. https://arxiv.org/abs/1312.5602 · code: https://github.com/deepmind/dqn (Lua, for the 2015 Nature paper) [P]

- [K] Experience replay, last N transitions, uniform sampling, minibatch Q-learning.

**Use:** background only. Rejected as the core (ALGORITHM_COMPARISON.md).

### R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning
Kapturowski, Ostrovski, Quan, Munos, Dabney. ICLR 2019. https://openreview.net/forum?id=r1lyTjAqYX · code: none official. SEED RL has an R2D2 implementation: https://github.com/google-research/seed_rl [P]

**The PDF could not be obtained:** OpenReview showed a bot check, and the typeset.io mirror returned HTML.

- [P, plus a secondary source from a web search] Sequences of 80 with burn-in of 20–40, and stored state plus burn-in work best.
- [V, via the NGU table, which reuses R2D2 settings] Trace length 80, replay period 40.

**Use:** the stored-vs-recomputed hidden-state reasoning. Applying it to PPO is a HYPOTHESIS.

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
Espeholt et al. ICML 2018. https://arxiv.org/abs/1802.01561 · code: https://github.com/deepmind/scalable_agent [P]

- [K] V-trace off-policy correction with truncation levels ρ̄ ≥ c̄. ρ̄ sets which fixed point is reached, and c̄ sets the contraction speed.
- [K] About 250,000 frames/s ("21 billion frames/day") on a large distributed setup.

**Use:** the upgrade path only.

### Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous RL
Petrenko, Huang, Kumar, Sukhatme, Koltun. ICML 2020. https://arxiv.org/abs/2006.11751 · code: https://github.com/alex-petrenko/sample-factory

**Throughput**

| System | VizDoom 128×72, env frames/s | Source |
| --- | --- | --- |
| System #1: 10 cores, 1 GPU | 16,985 with 20 envs up to 59,525 with 640 envs | Table A.2 [V] |
| System #2: 36 cores | 146,551 | Table 1 [V] |

These are frames, i.e. decisions × 4.

**Hyperparameters (Table A.5) [V]**

| Setting | Value |
| --- | --- |
| lr | 1e-4 |
| Frameskip | 2 or 4 |
| γ | 0.995 or 0.99 |
| Gradient clip | 4.0 |
| Rollout T | 32 |
| Batch | 2048 |
| Epochs | 1 |
| V-trace | ρ̄ = c̄ = 1 |
| PPO clip | [1/1.1, 1.1] |
| Entropy | 0.003 |
| Critic coefficient | 0.5 |

Architecture and setup:
- [V] GRU-512 chosen over LSTM, with "similar sample efficiency" and "slightly less computation".
- [V] 24–25 environments per core give the highest VizDoom throughput.
- [V] Policy lag of 5–10 SGD steps was stable.
- [V] Factorised action heads, 12,096 joint actions in the multi-agent setup.
- [V] The game-information vector includes only what a human sees in the UI.
- [V] The Battle training axis runs to about 4e9 frames, which is about 1e9 decisions.
- [K] Fig. 7 compares against DFP's reported Battle and Battle2 scores. Only the legend and caption were read.

**Use:** throughput design, the factorised categorical heads, and hyperparameter anchors. Not a dependency.

---

## Exploration

### ICM: Curiosity-driven Exploration by Self-supervised Prediction
Pathak, Agrawal, Efros, Darrell. ICML 2017. https://arxiv.org/abs/1705.05363 · code: https://github.com/pathak22/noreward-rl

- [V] VizDoom "My Way Home". In the sparse and very sparse settings the agent spawns about 350 steps from the goal under the optimal policy.
- [V] A3C with 20 workers, action repeat 4.
- [V] β = 0.2 (weight of the forward loss against the inverse loss), λ = 0.1 (weight of the policy loss), lr 1e-3.
- [V] r_int = (η/2)‖φ̂(s_{t+1}) − φ(s_{t+1})‖².
- [V] Pixel-based curiosity fails in the very sparse setting. ICM is robust to a white-noise patch.
- [V] The training-step axis runs to about 9M.

### RND: Exploration by Random Network Distillation
Burda, Edwards, Storkey, Klimov. ICLR 2019. https://arxiv.org/abs/1810.12894 · code: https://github.com/openai/random-network-distillation

**Hyperparameters (Table 5) [V]**

| Setting | Value |
| --- | --- |
| Rollout | 128 |
| Minibatches | 4 |
| Epochs | 4 |
| Extrinsic coefficient | 2 |
| Intrinsic coefficient | 1 |
| Envs | 128 |
| lr | 1e-4 |
| λ | 0.95 |
| Entropy | 0.001 |
| Predictor trained on | 25 % of experience |
| γ_E (extrinsic) | **0.999** |
| γ_I (intrinsic) | **0.99** |
| Clip | [0.9, 1.1] |

Preprocessing [V]:
- Target and predictor inputs are normalised as clip((x − μ)/σ, −5, 5).
- Sticky actions with p = 0.25.

Findings:
- [V] Raising γ_I to 0.999 hurts.
- [V] Non-episodic intrinsic returns explore more.
- [V] Dynamics-based curiosity oscillates at room boundaries, a form of the noisy-TV problem.
- [V] RNN policies need more environments and frames, e.g. 1.6B frames to reach a return of 7,570.

### Unifying Count-Based Exploration and Intrinsic Motivation
Bellemare et al. NeurIPS 2016. https://arxiv.org/abs/1606.01868 · code: none official [P]

- [K] Pseudo-counts derived from a density model.

**Use:** superseded for us by exact true-cell counts in training.

### Episodic Curiosity through Reachability
Savinov, Raichuk, Marinier, Vincent, Pollefeys, Lillicrap, Gelly. ICLR 2019. https://arxiv.org/abs/1810.02274 · code: https://github.com/google-research/episodic-curiosity

- [K] A siamese R-network predicts whether two observations are within k steps of each other, and novelty means being unreachable from the episodic memory.
- [K] Targets the "couch-potato" problem, and was evaluated on VizDoom, DMLab and MuJoCo.

**Use:** skipped. Training-time true positions do this better for us.

### NGU: Never Give Up: Learning Directed Exploration Strategies
Badia et al. ICLR 2020. https://arxiv.org/abs/2002.06038 · code: none official

Intrinsic reward [V]:
- r_i = r_episodic · min(max(α, 1), L) with L = 5.
- r_episodic = 1/√(Σ_kNN K + c) with c = 0.001.
- The embedding f is trained with an inverse-dynamics model, giving "controllable state".

**Hyperparameters (Table 6) [V]**

| Setting | Value |
| --- | --- |
| Neighbours k | 10 |
| ξ | 0.008 |
| Kernel ε | 1e-4 |
| β | 0.3 |
| Episodic memory | 30,000 |
| γ_e | 0.997 |
| γ_i | 0.99 |
| Trace length | 80 |
| Replay period | 40 |
| lr (R2D2) | 1e-4 |
| lr (RND and action prediction) | 5e-4 |

[V] Baselines were run for 35B frames.

**Use:** the NGU-lite structure (episodic count × clipped RND multiplier).

### Agent57: Outperforming the Atari Human Benchmark
Badia et al. ICML 2020. https://arxiv.org/abs/2003.13350 · code: none

- [K] Downloaded but not examined.
- [P] Adds a meta-controller over (β, γ) mixtures to NGU.

Not relevant at our compute scale.

### Go-Explore: a New Approach for Hard-Exploration Problems
Ecoffet et al. 2019 (later published in Nature 2021). https://arxiv.org/abs/1901.10995 · code: https://github.com/uber-research/go-explore [P]

- [K] Downloaded only.
- [P] Return to promising states, then explore. This needs simulator restore.
- **Note:** a ViZDoom state restore is possible in training, but "return then explore" from saved states does not train a policy that explores from spawn without extra work. Not recommended.

---

## Auxiliary tasks, measurement prediction and memory

### UNREAL: Reinforcement Learning with Unsupervised Auxiliary Tasks
Jaderberg et al. ICLR 2017. https://arxiv.org/abs/1611.05397 · code: none official

- [V] Auxiliary tasks: pixel control, reward prediction and value replay.
- [V] On Labyrinth, mean human-normalised performance rose from 54 % to 87%, and data efficiency improved about 10× (up to 18×).
- [V] Task weights: λ_RP = 1 and λ_VR = 1. λ_PC was sampled log-uniformly in [0.01, 0.1] for Labyrinth.
- [V] 20-step unrolls.

### Lample & Chaplot: Playing FPS Games with Deep Reinforcement Learning
Lample, Chaplot. AAAI 2017. https://arxiv.org/abs/1609.05521 · code: https://github.com/glample/Arnold [P]

- [V] DRQN plus a game-feature head trained jointly, with engine-provided labels at training time only.
- [V] Separate navigation and action networks.
- [V] Shaping:
  - positive reward for item pickup;
  - negative reward for losing health or ammo;
  - a small reward for distance travelled ("avoids turning in circles").
- [V] Sequential DRQN updates: minimum history 4, 5 states updated.
- [V] Frame skip 4 was the best trade-off.
- [V] After 65 h of training, the best K/D is below 2.0 without game features and reaches a maximum above 4.0 with them.
- [V] DFP's paper names the Lample & Chaplot DRQN a strong baseline.

### DFP: Learning to Act by Predicting the Future
Dosovitskiy, Koltun. ICLR 2017. https://arxiv.org/abs/1611.01779 · code: https://github.com/intel-isl/DirectFuturePrediction [P]

- [V] Predicts Δmeasurements at offsets 1, 2, 4, 8, 16 and 32 steps. Only the last 3 offsets enter the objective, with weights (0.5, 0.5, 1).
- [V] Episodes of 525 steps, equal to 1 minute of play, so one step is 4 tics.
- [V] Won the Full Deathmatch track of the September 2016 Visual Doom AI Competition, which was held on previously unseen environments.

[V] Results (Table 1):

| Method | D3 frags | Steps per day |
| --- | --- | --- |
| DFP | 33.5 | 70M |
| A3C | 5.6 | 80M |

- [V] DFP is reactive, with "no explicit facilities for memory".

### Recurrent Models of Visual Attention (RAM)
Mnih, Heess, Graves, Kavukcuoglu. NeurIPS 2014. https://arxiv.org/abs/1406.6247 · code: none official

- [K] A recurrent glimpse policy trained with REINFORCE.

**Use:** conceptual only. Entity attention is our equivalent.

### Asymmetric Actor Critic for Image-Based Robot Learning
Pinto, Andrychowicz, Welinder, Zaremba, Abbeel. RSS 2018. https://arxiv.org/abs/1710.06542 [P]

- [P] The critic gets full state and the actor gets observations.

**Use:** the privileged-critic hypothesis (H5).

---

## Long horizon and hierarchy (all [P], not fetched)

- **Ng, Harada, Russell (1999).** Policy invariance under reward transformations. ICML. Potential-based shaping preserves optimal policies.
- **FeUdal Networks (FuN).** Vezhnevets et al., ICML 2017, https://arxiv.org/abs/1703.01161. A manager emits goal directions at a lower temporal resolution.
- **HIRO.** Nachum et al., NeurIPS 2018, https://arxiv.org/abs/1805.08296 · code: https://github.com/tensorflow/models/tree/master/research/efficient-hrl. Off-policy goal relabelling.
- **Option-Critic.** Bacon, Harb, Precup, AAAI 2017, https://arxiv.org/abs/1609.05140.
- **RUDDER.** Arjona-Medina et al., NeurIPS 2019, https://arxiv.org/abs/1806.07857 · code: https://github.com/ml-jku/rudder. Return decomposition with an LSTM.
- **Active Neural SLAM.** Chaplot et al., ICLR 2020, https://arxiv.org/abs/2004.05155. Its global planner is analytic, which our rules ban.

## World models

### DreamerV3: Mastering Diverse Domains through World Models
Hafner, Pasukonis, Ba, Lillicrap. arXiv 2023 (later published in Nature 2025). https://arxiv.org/abs/2301.04104 · code: https://github.com/danijar/dreamerv3

- [K] Fixed hyperparameters across domains.
- [K] Model sizes from 12M to 400M parameters.
- [K] One A100 per run.
- [K] Atari budget of 200M frames.

## Generalisation

### Procgen: Leveraging Procedural Generation to Benchmark RL
Cobbe, Hesse, Hilton, Schulman. ICML 2020. https://arxiv.org/abs/1912.01588 · code: https://github.com/openai/procgen

- [V] "Agents strongly overfit to small training sets." Closing the generalisation gap can require "as many as 10,000 levels".
- [V] 200M timesteps per training set.
- [V] Training performance can *rise* with more levels, an implicit curriculum.

### Learning Invariances for Policy Generalization
Tachet des Combes, Bachman, van Seijen. ICLR 2018 Workshop. https://arxiv.org/abs/1809.02591

- [K] Downloaded only. From the title and abstract only, it concerns learning invariances for generalisation. Not examined further.
