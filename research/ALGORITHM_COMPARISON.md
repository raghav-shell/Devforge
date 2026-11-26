# Algorithm comparison for our constraints

Owner: rl-researcher. Source tiers ([V] verified in the paper text, [K] keyword-checked, [P] prior knowledge) are
explained in [PAPER_NOTES.md](PAPER_NOTES.md). All step counts are converted to **decisions** (one decision = 4 tics).

## Our constraints (the yardstick)

| Constraint | Value | Source |
| --- | --- | --- |
| Environment throughput | 1.4k–3k decisions/s on Freedoom-scale maps and 5.4k–9k/s on small multi-room maps, both at 8 processes; saturates at 6–8 processes | MEASURED, RL_RESEARCH.md §8 |
| Total budget | Roughly 50M–500M decisions across all experiments (no deadline is published) | ESTIMATE, RL_RESEARCH.md §2 |
| Learner hardware | M4 CPU, 10 cores (4 performance + 6 efficiency), 16 GB, MPS available but unmeasured | `CLAUDE.md` |
| Inference | 1 CPU thread at 40 ms average per decision. Model about 1M parameters or less | `env/doom_env/limits.py` |
| Task | Partially observable, about 1,000-decision episodes, sparse terminal rewards, factorised `MultiBinary(9)` actions with edge semantics | `docs/CONTRACT.md` |
| Rules | Our own pipeline. No default off-the-shelf library policies | `docs/RULES.md` §1 |

## Comparison

Score: ++ strong fit, + fit, 0 neutral, − poor fit, −− very poor fit.

| Criterion | Recurrent PPO (sync) | APPO / IMPALA V-trace | R2D2-style recurrent Q | DFP | DreamerV3-style world model |
| --- | --- | --- | --- | --- | --- |
| Handles partial observability | ++ GRU on-policy, stored state exact per rollout | ++ same model, with lagged state | + stored state needs burn-in to work [P] | −− reactive, no memory [V] | ++ recurrent world-model state |
| Factorised 216 or 2^9 actions | ++ product of heads, trivial | ++ same | − needs a joint 216-way Q or a branching head [P] | − predicts per action, so the output layer has 216 × measurements × offsets entries | + any actor |
| Sparse, long-horizon rewards | + with shaping, auxiliary losses and γ 0.995 | + same | + n-step plus value rescaling, ε-greedy exploration is weak [P] | 0 future offsets reach only 32 steps in the paper [V] | + imagination horizon is only about 15 steps [P], so the critic still does the long-horizon work |
| Sample efficiency per decision | 0 | 0 to + (more data per hour, not per sample) | + replay reuse | + (supervised targets) | ++ |
| Throughput on 8–10 CPU cores | + env-bound; learner idles while sampling (about 20–40 % loss, ESTIMATE) | ++ overlaps learner and actors | 0 actor/learner split plus a replay buffer | + | −− world-model training dominates wall time on CPU or MPS |
| Implementation risk in our timeframe | ++ low. Well documented (37 details [W], CleanRL [W]) | 0 policy lag, V-trace, process orchestration | −− replay of recurrent state, priorities, target network, burn-in, value rescaling | 0 simple, but needs goal and measurement design | −− RSSM, KL balancing, symlog, two-hot critic, replay |
| Evidence in ViZDoom | ++ Sample Factory APPO+PPO clipping is plotted above DFP's reported Battle and Battle2 scores (Fig. 7) [K: legend and caption read, curves not checked]; ICM used A3C [V] | ++ the same Sample Factory work | 0 Lample & Chaplot used DRQN (ViZDoom 2016) [V] | ++ won ViZDoom 2016 full deathmatch [V] | 0 not shown on ViZDoom navigation |
| Inference cost | ++ about 1–5 ms (ESTIMATE) | ++ the same model | ++ | ++ | 0 actor is small, but the world model is unused at evaluation (fine) |
| **Verdict** | **Core algorithm** | **Upgrade path only if the learner exceeds about 30 % of wall time** | **Reject** | **Use as an auxiliary head (future-measurement prediction)** | **Reject** |

## Rationale per method

### Recurrent PPO (chosen)
- PPO with a clipped ratio and several epochs per fresh rollout. Atari settings: T = 128, 8 actors, 3 epochs, clip 0.1·α, entropy 0.01 [V].
- A GRU core is ESTABLISHED for VizDoom: Sample Factory picked GRU over LSTM, finding "similar sample efficiency" at "slightly less computation" [V].
- On-policy learning means an exact stored-state, recompute-in-sequence update for epoch 1. Staleness after that is bounded by 2–3 optimiser passes (HYPOTHESIS: negligible next to replay staleness).
- The intrinsic reward and auxiliary losses plug in cleanly: two value heads and extra heads on the GRU output.

### APPO / IMPALA V-trace
- IMPALA's V-trace corrects for actor–learner policy lag with truncated importance weights ρ̄ and c̄ [K]. Sample Factory used ρ̄ = c̄ = 1 *and* PPO clipping [V]. It reported VizDoom throughput on a 10-core machine of 16,985–59,525 env frames/s depending on the number of environments, which is about 4k–15k decisions/s [V].
- **For us:** the simulator is the bottleneck (MEASURED). Asynchrony only recovers the learner-busy fraction of wall time. Cheapest version: double-buffered sampling, i.e. step half the environments while the policy acts on the other half (Sample Factory §3 [K]), and/or run the update of iteration k while iteration k+1 rolls out with the policy from k−1. PPO clipping already tolerates one iteration of lag. Add V-trace only if lag is more than 1 iteration.

### R2D2-style recurrent value methods
- Stored state with burn-in of 20–40 steps on sequences of 80 fixes representational drift from replay [P, secondary source]. NGU builds on R2D2: trace length 80, replay period 40, γ 0.997, 256 actors, 35B frames [V].
- **For us:** the mechanism that makes R2D2 good (replay reuse, many actors) needs a lot of infrastructure, and its exploration (ε) is the weakest part of our problem. A 216-action joint Q plus value rescaling plus priorities is a week of debugging. Reject.

### DFP (Direct Future Prediction)
- Predicts changes in a measurement vector at offsets {1, 2, 4, 8, 16, 32}. Acts by maximising a goal-weighted sum. Won the ViZDoom 2016 full-deathmatch track. 70M steps/day against 80M/day for A3C in their setup [V].
- Fully reactive: its discussion states the model uses the current frame only, "with no explicit facilities for memory" [V]. Our tiers 2–4 need memory.
- **For us:** keep the *idea* as an auxiliary head (RL_RESEARCH.md §5). The future-measurement targets map directly onto the scorer's categories.

### DreamerV3-style world models
- DreamerV3 used fixed hyperparameters across domains, models of 12M–400M parameters, a single A100 per run, and Atari budgets of 200M frames [K].
- **For us:** world models trade compute for environment steps. Our environment steps are cheap and our compute is a laptop. The world model would also have to model monsters and doors from rays and entities, a new research problem. Reject. It is not worth the risk even as an auxiliary task. Next-ray prediction (an UNREAL-style auxiliary) gets most of the representation benefit.

### DQN variants
- DQN: replay, a target network and ε-greedy [K]. Lample & Chaplot's DRQN with game features is the main ViZDoom DQN line: replay of 1M frames, RMSProp, batch 32, ε annealed 1 → 0.1 over 1M steps, γ = 0.99, frame skip 4 chosen as the best trade-off [V].
- Same verdict as R2D2: rejected as the core, though the frame-skip-4 finding confirms the contract's choice is sensible.

## Throughput upgrade decision tree (for the rl-training-engineer)

1. Build synchronous recurrent PPO. Log the fraction of wall time spent in the environment, in policy inference and in the update.
2. If the update takes more than 30 % of wall time: overlap it with the next rollout (one-iteration lag; PPO clipping handles it).
3. If policy inference inside the rollout takes more than 20 %: batch across all environments in one forward pass, and keep worker processes for the environments only (Sample Factory's rollout-worker / policy-worker split [K]).
4. Use MPS only if a measured benchmark shows the update getting faster at our batch size (RL_RESEARCH.md §2).
