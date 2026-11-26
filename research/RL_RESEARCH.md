# RL Research: recommendations for the DOOMSDAY Track 1 agent

Owner: rl-researcher. Status: Phase 1 research. This file builds on `docs/RESEARCH_NOTES.md` but does not edit it.
Companion files: [ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md), [PAPER_NOTES.md](PAPER_NOTES.md),
[EXPERIMENT_HYPOTHESES.md](EXPERIMENT_HYPOTHESES.md).

Labels used throughout:
- **ESTABLISHED**: a published finding, with a citation. The source tier is given in PAPER_NOTES.md:
  - **[V]**: the number was verified in the paper text this session.
  - **[K]**: the paper was downloaded but only keyword-checked.
  - **[W]**: taken from a WebFetch summary of a web page (the 37-details blog), not read verbatim.
  - **[P]**: from prior knowledge, not fetched.
- **HYPOTHESIS**: my own reasoning for our setting. It has not been tested.
- **MEASURED**: run in this repo this session. Commands and outputs are in §8.
- **ESTIMATE**: arithmetic on measured numbers plus stated assumptions.

---

## 0. Ground truth this rests on (from the fixed contract)

| Fact | Source |
| --- | --- |
| One decision lasts 4 tics. Episodes run 2,100–4,200 tics, which is 525–1,050 decisions | `docs/CONTRACT.md` §5, `env/doom_env/limits.py:7-9` |
| Actions are `MultiBinary(9)`. USE and SELECT_NEXT_WEAPON fire on the rising edge only. Pressing USE on a moving door reverses it | `docs/CONTRACT.md` §3 |
| Observations: 64 rays (distance and hit type 0–5), 32 entities (5 features, type, mask) and 17 vitals. There is **no heading and no global position** | `docs/CONTRACT.md` §2 |
| `terminated` means exit reached or death. `truncated` means the tick cap was hit | `docs/CONTRACT.md` §7 |
| Inference: 1 CPU thread and 40 ms × steps per map in total, so the budget is an average, not a per-step limit. Any single `act()` must finish within 5 s | `env/doom_env/limits.py:5-8`, `docs/EVALUATION.md` |
| Score categories: exit (largest), progression (keys and doors), exploration (unique areas), combat and survival. Harder tiers weigh more | `docs/EVALUATION.md` |
| No hackathon duration or deadline is given in the problem statement (https://rentry.co/uio6w6kn, fetched). Budgets below are therefore stated in decisions and hours | rentry (fetched) |

---

## 1. Corrections to the brief and to `docs/RESEARCH_NOTES.md` (please route to the main session)

1. **Throughput depends heavily on the map, and real maps are 5–10× slower than `two_rooms`.** The figure of "2,300 steps/s per core" was measured on the toy map. MEASURED this session (§8, with machine load average 7–12 because other agents were running):

   | Map class | 1 process | 8 processes (aggregate) |
   | --- | --- | --- |
   | `maps/two_rooms.wad` | 2.8k–4.8k/s | 13k–17k/s |
   | `tests/fixtures/*.wad` (small multi-room maps) | 2.5k–2.9k/s | 5.4k–9.0k/s |
   | Freedoom2 MAP01 | 0.79k/s | 3.0k/s |
   | Freedoom2 MAP10 | 0.37k/s | 1.4k/s |

   Scaling flattens at 6–8 processes. The M4 has 4 performance cores and 6 efficiency cores (`sysctl hw.perflevel0/1.physicalcpu` = 4/6). The runs were noisy, so an **idle-machine rerun is needed** before anyone quotes these as final.

   A profiler run suggests the wrapper's Python sensor code (`env/doom_env/sensors.py`) is a large part of each step on Freedoom maps. However, the profile and the benchmark disagree (0.67 ms of observation time per step against 1.26 ms per step in total), so **I am not claiming a percentage**. code-review-and-optimization should profile this properly.
2. **RND discounts are reversed in RESEARCH_NOTES §5.** RND Table 5 gives γ_E = 0.999 for extrinsic reward and γ_I = 0.99 for intrinsic reward [V]. The paper also shows that raising γ_I to 0.999 *hurts* performance [V].
3. **PPO's Atari settings** in Table 5 are 3 epochs, clip 0.1·α and lr 2.5e-4·α, with α annealed from 1 to 0, 8 actors, T = 128 and entropy 0.01 [V]. RESEARCH_NOTES quotes the clip as "0.1–0.2".
4. **The Sample Factory "~100k+ fps" figure** is environment *frames* per second (decisions × 4) on a 36-core machine. VizDoom APPO reached 146,551 frames/s on 36 cores and up to 59,525 frames/s on a 10-core machine (Tables 1 and A.2) [V]. In our units, the 10-core figure is about 15k decisions/s with an 128×72 RGB renderer, and that is close to our toy-map numbers.
5. **Lample & Chaplot's "game features" are mostly already in our observation.** Their target, "an enemy is visible", is something the wrapper gives us directly (`visible_entities`). Auxiliary heads that only repeat the current observation add almost nothing. Useful auxiliary targets for us are things the agent **cannot** see in the current frame: memory, pose and the future (§5). HYPOTHESIS.

---

## 2. Core algorithm recommendation

**DECISION (proposed): a custom synchronous recurrent PPO.** The core is a GRU with 256 units, a GAE advantage, and N CPU worker processes each stepping several environments. Rollouts are taken from the single current policy.

The full comparison is in [ALGORITHM_COMPARISON.md](ALGORITHM_COMPARISON.md). The short version:

- **We are bound by the environment, not by data efficiency per GPU-hour.** Every candidate gets the same 1.5k–9k decisions/s. What separates the candidates is how much signal each gets per decision, how stable it is, and how much engineering it costs. PPO with a GRU is the configuration Sample Factory used to train VizDoom agents to strong performance (APPO with PPO clipping, GRU-512, rollout length 32) [V]. Recurrent PPO is also the method CleanRL documents in detail [W].
- **IMPALA/V-trace** pays off when a learner on a GPU consumes data from hundreds of actors. With 8–10 cores and a CPU learner, asynchrony gains us roughly the learner's share of wall time, about 20–40 % (ESTIMATE). It does so at the cost of correcting for policy lag. Sample Factory itself ran V-trace and PPO clipping *together* [V]. A later "APPO-lite" option is to overlap the update with the next rollout, which makes the policy one iteration stale. Do that only if profiling shows the update takes more than about 30 % of wall time.
- **R2D2-style recurrent Q-learning** would need either a 216-way joint Q-function or a branching Q-head, a replay buffer that stores recurrent state, burn-in, and ε-exploration. The original setup ran 256 actors [P], and NGU, which is built on R2D2, used 35B frames [V]. There is too much machinery for our budget, and ε-greedy exploration is poor at sparse, long-horizon tasks.
- **DFP** won the Full Deathmatch track of the 2016 Visual Doom AI Competition, held on previously unseen maps, and reported 70M training steps/day [V]. However, it is **reactive**, with no recurrent memory [V], and it acts by predicting measurements. Our largest score terms, the exit and multi-room navigation, need memory. **Use DFP's measurement-prediction idea as an auxiliary head, not as the core** (§5).
- **A Dreamer-style world model** is built for sample efficiency when environments are expensive. DreamerV3's results used one A100 per run and models from 12M to 400M parameters [K]. Our environment is cheap and our hardware is a laptop CPU, so a world model shifts the bottleneck from simulation to model training. It also adds a second network to debug. It is not recommended.

**ESTIMATE of achievable steps.** These assume a synchronous loop in which policy inference and the PPO update take about 40 % of wall time. That 40 % is an assumption, because torch is not installed and could not be measured.

| Training map mix | Raw rate, 8 workers | Effective rate | Per hour | Per 24 h |
| --- | --- | --- | --- | --- |
| Small or generated multi-room maps (fixture-like) | 5.4k–9k/s | 3.2k–5.4k/s | 12M–19M decisions | 0.28B–0.47B |
| Freedoom-scale full levels | 1.4k–3k/s | 0.8k–1.8k/s | 3M–6.5M decisions | 0.07B–0.16B |
| Planning figure for a mixed curriculum | | ~2k–3k/s | **~8M decisions/h** | **~0.2B/day** |

At about 1,000 decisions per episode, that is roughly 8k episodes per hour.

To compare with the literature, every figure below is converted to decisions:
- **ICM on My Way Home:** the plot axis runs 0–9M training steps [V, read from the figure]. ICM used action repeat 4 [V], so these are probably agent steps, i.e. about 9M decisions. The paper does not state the unit (unconverted).
- **DFP:** 50M steps [V]. Its 525-step episodes equal 1 minute of play, which is 4 tics per step, so this is 50M decisions.
- **UNREAL:** about 10× fewer steps than A3C on Labyrinth [V].
- **RND on Montezuma:** 1.97B frames, which is about 0.49B decisions [V].
- **Sample Factory Battle:** an axis up to 4e9 frames, which is about 1e9 decisions [V].
- **Procgen:** 200M timesteps per environment [V]. These are agent steps in a different game, with no tic equivalent, so the figure is not directly comparable.

**Our realistic total is 50M–500M decisions across all experiments.** That rules out "billion-scale" recipes and favours dense shaping, auxiliary losses, and a curriculum that front-loads easy maps.

**CPU vs MPS.** HYPOTHESIS, to be measured once torch is installed. A GRU-256 and MLP model with 0.5–1M parameters, run at batch 64 for acting and about 2k samples per minibatch for updates, is dominated by kernel-launch overhead on MPS. The learner should run on CPU with `torch.set_num_threads(2)` and leave 8 cores for the environment workers. Benchmark MPS once for the update phase only; the main session already holds this view (RESEARCH_NOTES §1 correction 1).

---

## 3. Starting hyperparameters (for the rl-training-engineer)

The rationale column cites the source tier. Anything marked "ours" is a HYPOTHESIS to be tuned in EXPERIMENT_HYPOTHESES.md.

| Parameter | Start value | Range to try | Source / rationale |
| --- | --- | --- | --- |
| Discount γ | **0.995** | 0.99 / 0.995 / 0.998 (H3) | The choice rests on horizon arithmetic, a HYPOTHESIS:<br>• γ = 0.99: 0.99^1000 = 4.3e-5, and the effective horizon 1/(1−γ) is 100 decisions (about 11 s). That is short for key → door → exit chains, which are likely several hundred decisions.<br>• γ = 0.995: a 200-decision horizon, and 0.995^1000 = 6.7e-3.<br>• γ = 0.998: a 500-decision horizon.<br>Supporting evidence: RND raised γ_E from 0.99 to 0.999 on sparse Montezuma and it "greatly improves performance" [V]. Andrychowicz et al. call γ one of the most important hyperparameters and recommend tuning it per environment [V].<br>**Counter-evidence:** Sample Factory used **0.99 at frameskip 4**, which is our setting. It used 0.995 only at frameskip 2 [V], but that was for dense deathmatch rewards. H3 settles the choice |
| GAE λ | **0.95** | 0.9–0.98 | PPO [V]. RND [V]. GAE's best range was λ ∈ [0.92, 0.98] [V]. Andrychowicz et al. recommend 0.9 [V], but that was on continuous control with dense rewards |
| PPO clip ε | **0.2** | 0.1–0.25 | PPO Atari used 0.1·α [V]. Sample Factory and RND used a ratio range of [0.9, 1.1], i.e. 0.1 [V]. Andrychowicz et al. suggest starting at 0.25 [V]. Choose 0.2 and also stop the epoch early if approx_kl > 0.03 (ours) |
| Adam lr | **3e-4**, held constant during curriculum stages, with a linear decay to 1e-4 over the final stage | 1e-4 to 5e-4 | Andrychowicz et al.: Adam with a tuned lr, 3e-4 as default [V]. Sample Factory: 1e-4 [V]. RND: 1e-4 [V]. Don't anneal to 0 while the map distribution is still shifting (ours) |
| Adam ε | **1e-5** | | 37 implementation details [V] |
| Gradient-norm clip | **0.5** (global) | 0.5–4.0 | 37 details [W]. Sample Factory used 4.0 [V]. Log the pre-clip norm |
| Parallel environments | **64** (8 workers × 8 envs) | 32–128 | Sized from MEASURED saturation at 6–8 processes. RND found that more parallel environments help [V] |
| Rollout length T | **128** | 64–256 | PPO Atari [V], RND [V]. Each iteration then collects 8,192 decisions |
| BPTT chunk | **the full rollout (128)**, with minibatches formed **over environments** (16 envs × 128 steps = 2,048 samples) | 32 / 64 / 128 (H9) | CleanRL's LSTM recipe: sequential minibatches built from whole sub-environment sequences [W]. Sample Factory used T = 32 and a batch of 2,048 [V]. A chunk of 32 is the fallback if updates are too slow |
| Minibatches per epoch | **4** | 4–8 | PPO Atari (32×8 = 256 per minibatch) and RND (4) [V] |
| Epochs | **3** | 2–4 | PPO Atari: 3 [V]. RND: 4 [V]. Sample Factory: 1, because it is asynchronous [V]. Recurrent state goes stale with more epochs (§3.2) |
| Value loss coefficient | **0.5** | 0.25–1 | Sample Factory [V]. PPO Atari used 1 [V] |
| Value clipping | **off** | | Andrychowicz et al.: value clipping hurts [V]. Engstrom et al.: no evidence that it helps [K] |
| Value / return normalisation | **on**: divide rewards by a running std of discounted returns, clip ±10. Upgrade to PopArt if curriculum stages change reward scale a lot | | 37 details (reward scaling) [W]. PopArt [P]. Andrychowicz et al.: "check if value function normalization helps" [V] |
| Advantage normalisation | **Per minibatch, on the combined advantage** | | 37 details #7 [W]. With two streams (only from H7 on): A = 2·A_E + 1·A_I, which uses RND's reward coefficients and its "combined advantages" step (Table 5 and Algorithm 1) [V]. RND normalises the intrinsic *reward* by a running std of intrinsic returns [V]. It does **not** normalise each stream's advantage separately, and neither should we: that would scale pure-noise intrinsic advantages up to unit variance (HYPOTHESIS) |
| Observation normalisation | Running mean and std for vitals and entity velocities, clipped to ±5. Rays are already in [0, 1] | | Andrychowicz et al.: "always use observation normalization" [V]. RND clipped to [−5, 5] [V] |
| Initialisation | Orthogonal, gain √2 for the hidden layers, **0.01 for the policy head**, 1.0 for the value head | | 37 details #2 [W]. Andrychowicz et al.: last policy layer 100× smaller [V] |
| Entropy | **Sum over action heads. Coefficient 0.005, linearly decayed to 0.001 over training**. Log entropy per head | 0.001–0.01 | See §3.1 |
| GRU | 1 layer, 256 units. h = 0 at reset. Reset inside sequences using the done mask | 128–512 | Sample Factory chose GRU over LSTM for its similar sample efficiency and lower cost [V]. 37 details LSTM #2–#3 [W] |

### 3.1 Entropy and action parameterisation (factorised heads)

- **Entropy scale.** Nine Bernoulli heads have a maximum entropy of 9 ln 2 = 6.24 nats. A single 18-way Atari softmax has a maximum of ln 18 = 2.89 nats. If we **sum** the per-head entropies (which is the correct joint entropy of a factorised policy), Atari's 0.01 is roughly twice as strong in effect. Sample Factory used 0.003 with 7 factorised heads [V]. Start at 0.005 summed (HYPOTHESIS). Averaging over heads instead would make the coefficient independent of the head count, but whatever choice we make must be fixed and documented in `docs/TRAINING.md`.
- **p = 0.5 at initialisation conflicts with the action semantics.** A uniform Bernoulli start gives:
  - USE pressed on 50 % of steps. Because USE fires on the rising edge, that is a new press about 25 % of the time, and each one reverses any moving door (CONTRACT §3).
  - FORWARD and BACK held together on 25 % of steps.
  - TURN_LEFT and TURN_RIGHT held together on 25 % of steps. MEASURED in RESEARCH_NOTES §2: this produces a 0° turn.
- **PROPOSED CONTRACT** (for the project-architect, in `docs/ACTION_SPACE.md`). Use **categorical factor heads** of the kind Sample Factory used [V]:

  | Head | Choices |
  | --- | --- |
  | move | none / fwd / back |
  | strafe | none / L / R |
  | turn | none / L / R |
  | attack | 0 / 1 |
  | use | 0 / 1 |
  | next_weapon | 0 / 1 |

  This gives 3·3·3·2·2·2 = 216 joint actions with a maximum entropy of 3 ln 3 + 3 ln 2 = 5.37 nats. It loses no expressivity, because pressing both opposing buttons only cancels out. It maps one-to-one onto `MultiBinary(9)`. This is architecture, not a gameplay rule. Testing it is H2.
- **USE initial bias (HYPOTHESIS).** Initialise the USE head's logit bias at about −2, which gives p ≈ 0.12. That reduces door bouncing early in training. It is an initial parameter, not a rule. The policy can learn to move it anywhere.
- **Feed back the previous action and a "steps since USE went down" feature.** RESEARCH_NOTES §1 already proposes this. It is required for the edge semantics to be learnable at all.

### 3.2 Hidden state: stored vs recomputed

- **ESTABLISHED practice for recurrent PPO** (37 details LSTM #5 [W]; Sample Factory [K]):
  1. Store the hidden state at the start of each training sequence during the rollout.
  2. During the update, recompute the hidden states *within* the sequence with the current parameters, resetting them at done flags.
- **What R2D2 showed (ESTABLISHED, but for replay).** Stored state goes stale ("representational drift"), and a burn-in prefix of 20–40 steps fixes it, with sequences of 80 [P / secondary source; PDF not obtained]. Applying this to *on-policy* PPO is **HYPOTHESIS**. Our staleness comes only from 3 epochs of updates within a single iteration, which is far smaller than replay lag. Using whole-rollout sequences (128 steps) makes the stored state exact for epoch 1, and it is only one or two optimiser steps stale after that.
- **Optional refresh.** Before each epoch, run one no-grad forward pass over the whole rollout to refresh the stored chunk-start states. With a 0.5M-parameter model on CPU this costs well under a second per iteration (ESTIMATE). Do this only if H9 shows short chunks win on speed.

### 3.3 Termination vs truncation (implementation details)

- **ESTABLISHED (Pardo et al. 2018) [K].** When a time limit is not part of the task, bootstrap from V(s_T) at truncation ("partial-episode bootstrapping"). When the time limit *is* part of the task, the agent should instead be time-aware.
- **Our case is mixed (HYPOTHESIS).** The evaluation cap is real. The agent can count its own steps, which is legal because it is internal state, but it does not know the map's cap (2,100–4,200 tics). The recommendation is to bootstrap on truncation and to randomise `episode_ticks` across 2,100–4,200 in training. As an option, add a normalised own-step-counter input (t / 1,050). That input is H12.
- **Code requirements for the training engineer:**
  1. On truncation, keep the **final observation before the auto-reset** (the gymnasium `final_obs` convention) and the **hidden state produced at that step**. Evaluate V(s_T, h_T) and only then reset h.
  2. Compute δ_t = r_t + γ·V(s_{t+1})·(1 − terminated_t) − V(s_t).
  3. Cut the GAE recursion at **both** kinds of episode end (`done = terminated | truncated`).
  4. At the end of the rollout, bootstrap with V(next_obs, h_next).
  5. Write a unit test: feed a constant-reward environment with a truncation and check that the advantage equals the analytic value.
- Death is `terminated`, so there is no bootstrap after it. That is correct.

---

## 4. Exploration recommendation

**DECISION (proposed), in order:**

1. **Baseline exploration bonus = episodic count of true-position cells**, computed from `privileged_state()` (training only). This is already in the reward plan (RESEARCH_NOTES §6: +0.05 per new 64×64 cell, once per episode). It is the "oracle" version of NGU's episodic novelty: it is drift-free and immune to the noisy-TV problem, because it counts positions, not observations, so animated monsters, projectiles and moving doors cannot inflate it. Every other exploration method must beat **this**, not "no exploration". The bonus also matches the scorer's "unique areas visited" category directly (`docs/EVALUATION.md`).
2. **Make the bonus learnable.** The bonus depends on the history of the current episode (whether a cell is new depends on where the agent has been), so a memoryless critic cannot predict it.
   - (a) The GRU partly carries that history.
   - (b) The legal ego-map from the spatial team gives the policy and critic a visitation channel. This is the main fix.
   - (c) **HYPOTHESIS, main-session decision:** an *asymmetric, privileged critic* (as in the asymmetric actor-critic of Pinto et al. 2017 [P]). It is a training-only value MLP that additionally receives privileged features, such as the true visited-cell count around the agent and the geodesic distance to the next objective. The actor never sees these, and the critic is not shipped in `submission/`. The rules allow "any training-time scaffolding" (`docs/RULES.md` §1). The risk is that the critic and actor share a torso. The privileged inputs must enter only after the shared torso, so the actor's features stay legal at evaluation.
   - This is H5.
3. **Add lifelong novelty only after (1) and (2) work. NGU-lite** (H7), using the multiplier min(max(α, 1), L) with L = 5 from NGU [V].
   - **Reward streams (to avoid double counting):**
     - **Before H7:** there is only one reward stream and **one value head, V_ext**. The privileged-critic branch from H5 is optional. The cell bonus is part of the extrinsic training reward: r_ext = … + w_cell·1[new cell] (w_cell = 0.05, constant).
     - **From H7 on:** add a second head, **V_int**, with γ_I = 0.99. The intrinsic reward is r_int = w_cell · 1[new cell] · (min(max(α_RND, 1), 5) − 1). That is the *extra* bonus a new cell earns when the surrounding situation is also novel across episodes. The total per-new-cell incentive is w_cell·(1 + β·(clip(α) − 1)), so the base bonus is never counted twice.
   - α_RND is an RND predictor error, normalised as in NGU [K]. It is computed on a *normalised compact embedding* of ray distances and hit types, entity type counts, and key flags. Entity velocities and health are excluded because they are noisy.
   - What it adds: pushes toward situations that are rare across episodes, such as locked doors, key rooms and lifts.
   - Normalise r_int by a running std of intrinsic returns [V]. Use RND's coefficients and combine the raw advantages as A = 2·A_ext + 1·A_int [V]. Then apply per-minibatch normalisation to the combined A (§3).
   - Train the predictor on 25 % of each batch [V].
   - Episodic vs non-episodic intrinsic returns: RND found non-episodic intrinsic returns explore more [V]. **HYPOTHESIS:** for us, keep the intrinsic stream *episodic*, since it is gated by the per-episode new-cell indicator and evaluation is per-episode on unseen maps.
4. **Do not prioritise ICM.** ICM's own VizDoom result is strong: My Way Home, sparse and very-sparse rewards, and robust to a white-noise patch thanks to inverse-model features [V]. But:
   - RND found dynamics-based curiosity worse than RND on Montezuma because of aleatoric oscillation at room boundaries [V].
   - In our environment the forward-model error is largest exactly where the agent **can cause randomness itself**: re-pressing USE reverses a door (CONTRACT §3), and firing at monsters causes stochastic monster reactions. Both are controllable, so the inverse-model features keep them. **HYPOTHESIS: ICM would reward door bouncing.** This is a concrete reward-hacking risk. H11 tests it only if there is spare budget.
5. **Skip episodic curiosity via reachability (Savinov et al.)** for now. It needs a trained siamese R-network and a memory comparison at every step [K]. In training we have true positions, which give a better reachability signal (cells), and at evaluation no intrinsic reward runs at all.
6. **Skip pseudo-counts from a density model (Bellemare et al. 2016 [K]).** They are dominated by exact true-cell counts in training.

**Decay.** Intrinsic weight β starts at 1.0 relative to the cell bonus, decays linearly to 0.1 over the final 30 % of each curriculum stage, and is never 0 during exploration-tier stages (HYPOTHESIS). Keep the cell bonus **itself** at a constant, small weight, because it mirrors a real score category. Its decay should come from reward balancing against the exit and key rewards, not from annealing.

**What matters at evaluation (HYPOTHESIS).** No bonus runs at evaluation. What has to transfer is a learned *exploration policy* that is conditioned on memory, i.e. "I haven't been over there". That is why (2b), the ego-map, matters more than any choice of bonus.

---

## 5. Long-horizon and keycard recommendation

**Try first: an auxiliary prediction pack (cheap, dense, independent of the policy loss).** Each head is a small MLP on the GRU output. All targets are training-only labels, and the heads are dropped at evaluation (Lample & Chaplot did the same [V]). Ranked by expected value per unit of cost (HYPOTHESIS):

1. **Future-measurement prediction (DFP-style).** Predict Δ over horizons {8, 32, 128} decisions for:
   - new true cells visited;
   - key-flag sum (from vitals);
   - kills;
   - health;
   - and as a binary, "exit within the horizon".

   DFP used offsets {1, 2, 4, 8, 16, 32} with a goal-weighted objective [V]. Our episodes are about 1,000 steps, so the horizons are stretched. These targets give dense gradient toward exactly what the scorer rewards.
2. **Memory probes** (targets that are absent from the current observation):
   - (a) "a locked door has been seen this episode" (derivable from observation history);
   - (b) the bearing and distance, in the ego frame, to the last-seen locked door and to the episode start (privileged x, y, angle);
   - (c) "the current cell has been visited before" (privileged).

   These force the GRU or map to carry exactly the key → door → exit memory. HYPOTHESIS. Analogous evidence: UNREAL's auxiliary tasks gave a mean **10× speed-up** and raised Labyrinth performance from 54 % to 87 % human-normalised [V]. Lample & Chaplot report that after 65 h of training the best K/D without game features is below 2.0, while with them it reaches a maximum above 4.0 [V].
3. **Next-step ray-distance prediction** (UNREAL-like "feature control" on rays). Self-supervised and cheap. Lowest priority: it helps the representation, not long-horizon credit.

Loss weight: 0.1–0.5 per head group, with a global auxiliary weight of 0.25 to start. Log each head's loss and accuracy on validation maps. That doubles as an interpretability tool for the judges.

**Shaping for credit assignment.** ESTABLISHED: potential-based shaping F = γΦ(s′) − Φ(s) preserves optimal policies (Ng et al. 1999 [P]). Proposed potential (HYPOTHESIS):

- **Φ(s) = −D_remaining(s) / D_scale.** D_remaining is the geodesic distance from the agent to the next unmet objective **plus** the geodesic lengths of the remaining chain, e.g. key → locked door → exit, chosen from the keys already held (vitals) and the map's lock structure.
- **Continuity:** using the *total remaining path* keeps Φ continuous when a subgoal is reached. At key pickup the agent stands on the key, so the remaining length is unchanged.
- **The naive version penalises picking up the key.** "Distance to the *current* objective" jumps from about 0 to −distance(door) at pickup, which is a negative shaping reward at exactly the moment we want to reward.
- **Terminal handling** (Φ is negative, so a zero terminal potential would pay +D/D_scale for dying far from the goal):
  - **At death:** use F = γ·Φ(s_death) − Φ(s_prev), i.e. the potential of the last real state, **not** 0.
  - **At exit:** D_remaining = 0, so Φ = 0 naturally.
  - **At truncation:** compute F normally from the real final state, and bootstrap V as in §3.3.
  - This gives up exact policy invariance at terminal transitions, which is accepted. Log the summed shaping reward per episode: it should telescope to about Φ(s_T) − Φ(s_0).
- Precomputed offline from the WAD, in training only. The environment-engineer or a reward owner would implement it. It is probably the single strongest long-horizon lever.
- **Needs a main-session ruling.** `CLAUDE.md` bans "classical pathfinding … ever". Here the geodesic distance is computed by the *training reward*, offline, and never selects an action or ships in `submission/`. `docs/RULES.md` §2 allows any training-time modification, but this should be confirmed (or reviewed with Fable) before anyone builds it. A fallback with no pathfinding is the Euclidean distance to the next objective, but that is misleading in mazes.

**Hierarchy: defer.** A high-level head that emits a latent goal every k = 8 decisions (in the FuN / HIRO style [P]) or an option-critic [P] adds non-stationarity and new hyperparameters. The evidence in their papers comes from 1e8–1e9-step regimes [P]. Try it only if H6 (auxiliary pack) and H8 (curriculum) leave tier-3 keycard maps unsolved (H10). The judging criteria reward "hierarchical control" (`docs/RULES.md` §5), so a *simple*, ablated option head is worth having late if it is at least neutral.

**Return decomposition (RUDDER, Arjona-Medina et al. 2019 [P]).** Not recommended. It needs an LSTM return predictor plus contribution analysis. Potential-based shaping with privileged geodesic distance gives the same credit-redistribution benefit far more cheaply in our training-only-privileged setting.

---

## 6. Generalisation to unseen maps

- **ESTABLISHED (Procgen, Cobbe et al. 2020) [V].** PPO agents "strongly overfit to small training sets", and closing the gap can take as many as **10,000 levels**.
  - What we have: Freedoom 1 and 2 (about 68 maps), 13 fixtures, and the ViZDoom scenarios. That is about **100 maps, two orders of magnitude short**.
  - Consequence: a procedural multi-room generator (keys, locked doors, exits) plus domain randomisation (spawn point, monster mix, item positions and key placement on fixed geometry) is **mandatory** for tiers 2–4, not optional. This belongs to the environment-engineer and the main session.
- Hold out maps from day 1: train / val / test splits per tier (the evaluation-and-generalization-researcher owns `maps/splits/`). Report the gap = train score − val score every evaluation cycle.
- Randomise `episode_ticks` (2,100–4,200) and spawn points. HYPOTHESIS: this prevents memorising open-loop routes. RND used sticky actions for the same reason [V].
- Sample at evaluation, don't argmax, unless the validation score says otherwise. Sampling breaks deterministic loops (HYPOTHESIS, already in RESEARCH_NOTES §9). Testing it is H13.

---

## 7. Failure modes and how to detect them in logs

The training engineer should log every quantity here per iteration, per curriculum stage and per map class.

| Failure | Detect it with | Threshold / response |
| --- | --- | --- |
| **Entropy collapse** | Entropy per head, in particular `use`, `turn` and `move` | Any head below 0.05 nats while val score is flat → raise entropy or check reward scale. USE collapsing to never-press means doors are never opened (check `doors_opened` per episode) |
| **Value divergence** | Explained variance of V_ext and V_int, value loss, pre-clip grad norm | Explained variance < 0 for more than 20 iterations, or grad norm 10× its median → lower lr, check return normalisation, check bootstrapping |
| **Policy step too large** | approx_kl, clipfrac | approx_kl > 0.05 or clipfrac > 0.3 → lower lr or epochs. Early-stop at KL 0.03 |
| **Recurrent state leaking across episodes** | Unit test: two env slots, reset one mid-rollout, and assert its h is 0 after reset in *both* the rollout and the update recomputation. Also log V(s_0) at episode start | V(s_0) should be the same on every map with the same spawn statistics. Variance tied to the previous episode's outcome means a leak |
| **Hidden state mismatch between rollout and update** | log π_new/π_old at epoch 1, minibatch 1, *before* any gradient step | It must be exactly 1.0 (|ratio − 1| < 1e-5). Any other value points to a sequence, mask or state bug |
| **Stale policy lag** (only if asynchronous) | Policy version number of each sample minus the learner's version | Sample Factory saw 5–10 SGD steps of lag as stable [V]. Alarm above 20 |
| **Reward hacking: door bouncing** | USE rising edges per episode, and door reversal count (privileged) | More than about 5 reversals of the same door per episode |
| **Reward hacking: spinning or jitter in place** | Cell bonus is 0 by design. Watch turn fraction and the "stuck rate" (no new cell for 100 steps) | Stuck rate rising while return rises → shaping exploit |
| **Reward hacking: kill farming / item dancing** | Kills and pickups per 100 steps against exit rate | Rising combat reward with a falling exit rate → reduce weights |
| **Death-avoidance passivity** | Ratio of episode length to cap, distance travelled | Agent hides because the death penalty outweighs the cell bonus → rebalance |
| **Overfitting to training maps** | Train − val score gap per tier, with ≥ 10 val maps | Gap growing while train rises → more maps or randomisation, stop that run |
| **Curriculum forgetting** | Score on earlier-stage val maps | A drop above 20 % → replay of about 20 % from earlier stages (already in RESEARCH_NOTES §7) |
| **Truncation bug** | Mean advantage at the last step before a truncation vs at mid-episode | A systematic negative spike means V(s_T) was zeroed |
| **Throughput regression** | Decisions/s per worker, learner share of wall time | Alarm at a 20 % drop |
| **Inference budget** | p50/p99 of `act()` on 1 thread (`validate.py`) | p99 > 15 ms → shrink the model |

---

## 8. Measurements run this session (commands and output)

All scripts are in the scratchpad (`/private/tmp/claude-501/-Users-raghavsharma-Desktop-DOOMSDAY/8af00418-f4ee-48ce-a8af-028a5b4c7dbc/scratchpad/`) and only read from the repository.

```
$ sysctl -n hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu
10 / 4 / 6
```

**bench3.py: barrier-synced, 10 s fixed duration, random actions with p = 0.3, `render_buffers=False`**

Load average was 9.68 at the start.

```
MAP01 two_rooms   P=1 4825/s | P=4 11903 | P=6 14184 | P=8 14619 | P=10 13522, 13022
```

Load average 12.49.

```
Freedoom2 MAP01   P=1 794/s  | P=6 2523  | P=8 2970
Freedoom2 MAP10   P=1 366/s  | P=8 1436
```

Load average 7.02.

```
fixture 2f617ad2  P=1 2922/s | P=8 5426
fixture 509aca7c  P=1 2481/s | P=8 6936
fixture 6a392a90  P=1 2854/s | P=8 8964
```

**Earlier runs (bench.py and bench2.py)**

These used a pool, the times include startup, and the results were inconsistent: two_rooms gave an aggregate of 6.1k at P=4 but 16.3k at P=6. They are reported only to show how noisy the measurements were.

`torch` is not installed, so the policy-inference cost, the update cost and the CPU-vs-MPS comparison were **not measured**.
