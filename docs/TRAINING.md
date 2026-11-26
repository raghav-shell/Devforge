# Training (custom recurrent PPO)

Owner: rl-training-engineer. Code: `training/ppo/`, `training/train.py`, `training/configs/`,
`training/benchmarks/train_*.py`. Tests: `tests/training/`. Contracts: `docs/MODULE_CONTRACTS.md` sections 3.4, 4, 8;
`docs/ACTION_SPACE.md` section 2; rulings in `docs/DECISIONS.md` (D1, D9, D10, D11).

No SB3 or other off-the-shelf RL policy code is used. The engine calls the model only through the
`DoomPolicy` public API (`initial_state`, `step`, `forward_sequence`) and the action API in
`doom_agent.models.{distributions,action_codec}`.

## 1. Reproducible commands

```bash
# unit + integration tests for this area
uv run --no-sync python -m pytest tests/training -q --basetemp=/tmp/ptq_train

# engine smoke without ViZDoom (fake vec env + fake policy; marked "fake": true in run.json)
uv run --no-sync python -m training.train --fake --updates 5 --run-id fake-smoke \
  --set ppo.num_envs=8 --set ppo.rollout_len=32 --set ppo.num_minibatches=2 --set env.num_workers=1

# H1 map pool: generator family a only, 200 seeds x 5 stages, under runs/ (gitignored; never maps/generated/)
uv run --no-sync python -m training.envs.mapgen --stage all --family a --seed 0 --count 200 --out runs/maps/gen-a-v1
uv run --no-sync python -m training.configs.make_manifest --root runs/maps/gen-a-v1 --family a \
  --out runs/maps/gen-a-v1/train_gen_a.json     # refuses family b / seed >= 1000 / anything in val or test

# H1 baseline (ratio_check_fail=true is the default, D17). --max-minutes stops on wall time and checkpoints.
uv run --no-sync python -m training.train --config training/configs/h1.json --run-id h1-s0 --set run.seed=0 \
  --set env.manifests='["runs/maps/gen-a-v1/train_gen_a.json"]' --set run.ckpt_every=100

# load an export through a COPY of submission/agent.py (strict mode) and play N steps; submission/weights/ untouched
uv run --no-sync python -m training.ppo.agent_check --export runs/rl-training-engineer/h1-s0/export --steps 1000

# real-env smoke on the smoke manifest with the registered progression terms only (works today)
uv run --no-sync python -m training.train --run-id real-two-rooms-smoke --updates 6 \
  --set ppo.num_envs=8 --set ppo.num_minibatches=2 --set env.num_workers=2 --set env.envs_per_worker=4 \
  --set ppo.ratio_check_fail=true --set run.ckpt_every=3 \
  --set 'reward.terms=[{"name":"exit_reached","weight":10.0},{"name":"key_pickup","weight":3.0},{"name":"door_open","weight":1.0},{"name":"locked_door_open","weight":3.0},{"name":"death","weight":-3.0}]'

# resume (latest checkpoint in the run dir, or a specific file)
uv run --no-sync python -m training.train --resume runs/rl-training-engineer/h1-s0 --updates 100

# export weights for submission/weights/ (main session copies + commits via LFS)
uv run --no-sync python -m training.ppo.export --ckpt runs/rl-training-engineer/h1-s0/checkpoints/ckpt_000100.pt \
  --out runs/rl-training-engineer/h1-s0/export

# learner / inference throughput (CPU threads vs MPS, N envs)
uv run --no-sync python -m training.benchmarks.train_throughput --envs 16 64 --threads 1 2 4 --devices cpu mps
```

Overrides use `--set group.key=value` (JSON values). Unknown keys are rejected. Every run writes
`runs/rl-training-engineer/<run-id>/{run.json, metrics.jsonl, tb/, checkpoints/ckpt_<update:06d>.pt}`.
`run.json` records the full config, git commit and dirty flag, `obs_pipeline_hash`, training manifest ids,
the action-API source, TensorBoard status, device, thread count and torch version.

## 2. Algorithm (what the code does)

Per iteration: collect T = 128 decisions from N = 64 envs, compute GAE, then run 3 epochs × 4 minibatches.
Each minibatch is **16 whole env sequences × 128 steps**, replayed with `forward_sequence`.

| Item | Implementation |
| --- | --- |
| Hidden state | The trainer carries `new_state` from `step` and never zeroes it. The policy masks it with `episode_start` before it consumes obs_t. `state0` is stored **pre-mask** at every chunk start, with shape (T/chunk, N, S). `next_state` becomes the next rollout's `state0[0]` |
| Chunked BPTT | `ppo.bptt_chunk` (0 means the whole rollout, which is v1). With chunk < T, each env sequence is split into T/chunk sub-sequences, each starting from its stored `state0[c]` |
| GAE | `training/ppo/gae.py`: done = terminated \| truncated cuts the recursion. δ_t = r_t + γ[(1−done_t)V_{t+1} + truncated_t·¬terminated_t·trunc_value_t] − V_t. V_T = V(next_obs, next_state) |
| Truncation | For truncated & ¬terminated & ¬crashed, the trainer runs `step(final_feats, h_t_post, episode_start=False)` under no_grad and stores the value in `trunc_value[t]`, discarding the state it returns. If both flags are set, terminated wins. A non-crash truncation without `final_feats` **raises**. If `episode_start != terminated \| truncated`, the trainer **raises** |
| Crashes | `info["crashed"]` slots must be truncated. They get trunc_value 0, are counted per iteration and cumulatively (`env/crashed`, `env/crashed_total`) and are stored in `buffer.crashed`. Crashed episodes are not mixed into episode statistics. Done steps without `info["episode"]` are counted in `env/episodes_missing_info` |
| Reward scaling | `ppo.return_norm`: each reward is divided by the running std of the per-env discounted return, then clipped to ±10. The value head learns the normalised scale. The raw reward is kept in `buffer.reward` and logged next to the task metrics |
| Policy loss | Clipped surrogate, ε = 0.2. Per-minibatch advantage normalisation (`ppo.norm_adv`) |
| Value loss | 0.5·(V − R)², coefficient 0.5. `ppo.value_clip` (default off) clips around the behaviour value |
| Entropy | **The joint entropy of the factored policy, which is the SUM of the 6 per-head entropies** (maximum 5.37 nats). The coefficient multiplies the summed entropy and decays linearly from 0.005 to 0.001 over `total_updates`. Each head's entropy is logged as `entropy_head/<name>` |
| Optimiser | Adam (lr 3e-4, eps 1e-5). The global grad norm is clipped to 0.5; the pre-clip norm is logged. lr is constant, with an optional linear decay to `lr_final` over the last `lr_decay_frac` of updates |
| KL early stop | k3 approx-KL per minibatch. If it exceeds `target_kl` = 0.03, the remaining minibatches and epochs are skipped **before** that step is applied |
| Ratio gate | Epoch 1, minibatch 1, before any step: max\|π_new/π_old − 1\| is logged every iteration as `ppo/ratio_check_max_dev`. If it is ≥ 1e-5, the violation is counted and a warning printed, or the trainer raises when `ppo.ratio_check_fail` is set |
| Sampling | Training always samples at T = 1 with a dedicated `torch.Generator`, which is saved in the checkpoint |

Observation normalisation: none in the trainer. The obs pipeline uses fixed normalisation constants
(DECISIONS D9), so there is no running obs statistic to checkpoint.

## 3. Logged metrics (`metrics.jsonl`; TensorBoard under `tb/` when installed)

- **Losses:** `loss/{total,policy,value,entropy}`, `entropy_head/{move,strafe,turn,attack,use,next_weapon}`.
- **PPO:** `ppo/{approx_kl,clip_frac,explained_variance,ratio_check_max_dev,ratio_check_violations_total,early_stop,optimizer_steps}`, `grad/norm_preclip`.
- **Episodes:** `episode/{reward_mean,return_shaped_mean,return_env_mean,length_mean,completion_rate,death_rate,timeout_rate,cells_mean,keys_mean,kills_mean,use_edges_mean,weapon_edges_mean,door_reversals_mean}`, with `door_reversals` logged only when `info["episode"]` provides it (D11).
- **Reward:** `reward/{raw_mean,scaled_mean,return_std,extrinsic_mean,intrinsic_mean}` and `reward_terms/<term>_per_step`. `intrinsic_mean` is 0 in v1, because the cell bonus is part of r_ext (D8).
- **Diagnostics** (RL_RESEARCH §7): `value/at_episode_start`, `value/trunc_value_mean`, `adv/at_truncation` and `adv/mid_episode`. A systematic negative `adv/at_truncation` means V(s_T) was zeroed.
- **Env health:** `env/crashed`, `env/crashed_total`, `env/episodes_missing_info`, `episodes/count`, `episodes/total`.
- **Throughput:** `sps/{total,env_only,collect,learner_only}`, `time/{env_s,policy_inference_s,collect_s,learner_s,env_frac,learner_frac}`.
- **Schedules:** `sched/lr`, `sched/ent_coef`.

**TensorBoard status:** `tensorboard` is **not installed** in the local dev venv. The logger prints one
warning, records `"tensorboard": "unavailable: …"` in `run.json` and keeps writing JSONL.

## 4. Checkpoints (ckpt-1) and exact resume

`training/ppo/checkpoint.py` writes the files atomically (tmp file, then rename). Each checkpoint holds every
section 8.1 field: the policy, optimiser, scheduler, `update`, `global_step`, the RNG states (python, numpy,
torch global, trainer generator), the return-norm statistics **and** the per-env return accumulators, the env
carry (last obs, episode_start, hidden state), `obs_pipeline_hash`, `obs_schema_version`, `obs_config`,
`action_space_version`, `train_manifest_ids`, `train_seeds` (all < 10000, enforced), the ppo, reward and env
configs, the full train config, and `git_commit`/`git_dirty`.

- ckpt-1 contains numpy RNG tuples, so it loads with `torch.load(weights_only=False)`. Only load your own files. The export loads with `weights_only=True`.
- **Bit-identical resume** is tested on the fake env, which can be snapshotted (`get_state`/`set_state`).
- **A real `SubprocVecEnv` cannot be snapshotted.** On resume, the envs reset, the hidden state starts at zero and the return accumulators reset. The learner state is exact; the data stream after the resume is not.
- `training/train.py --resume` refuses a checkpoint whose `obs_pipeline_hash` differs from the current tree.

## 5. Export (export-1)

`training/ppo/export.py` writes `policy.pt`, which holds only `DoomPolicy.state_dict()` as CPU float32 and loads with
`weights_only=True`, and `policy.json`, which holds every section 8.2 key. It fails hard on:
- an `obs_pipeline_hash` mismatch (`--allow-hash-mismatch` overrides it; each override goes to stderr, `policy.json` and `export_overrides.log`);
- a schema or action-space version mismatch;
- key sets differing from `DoomPolicy(cfg)` (strict load);
- any `aux`/`priv` key;
- `ObsConfig.enable_*` disagreeing with `PolicyConfig.use_*`.

Aux heads and the privileged critic live outside `DoomPolicy` (§4.5), so they are never in `policy_state`. They are stored separately as `aux_state` and `privileged_critic_state`, and the tests check that they are stripped. Exports default to `default_inference: "sample"` with T = 1.0 (D15, provisional until H13).

## 6. Device and threads (MEASURED)

Command: `uv run --no-sync python -m training.benchmarks.train_throughput --envs 16 64 --threads 1 2 4 --devices cpu mps --iters 2`
(2026-11-26, M4, load average 2.96, real `DoomPolicy` with 624,744 params, T = 128, 3 epochs × 4 minibatches, FakeVecEnv).

| Device | Threads | N | Rollout-inference SPS | Learner SPS | Learner s/iter |
| --- | --- | --- | --- | --- | --- |
| cpu | 1 | 16 | 6.7k | 2.1k | 0.98 |
| cpu | 1 | 64 | 29.5k | 3.3k | 2.48 |
| cpu | 2 | 16 | 7.2k | 2.2k | 0.95 |
| cpu | **2** | **64** | **29.4k** | **3.8k** | **2.14** |
| cpu | 4 | 64 | 22.6k | 2.8k | 2.94 |
| mps | – | 16 | 2.4k | 1.3k | 1.60 |
| mps | – | 64 | 11.8k | 3.7k | 2.22 |

**Decision: learner and rollout inference on CPU with `torch.set_num_threads(2)`** (the `run.torch_threads` default).
- MPS is not faster for the learner (3.7k vs 3.8k) and is 2.5× slower for batched rollout inference.
- 4 threads is slower than 2, which leaves the remaining cores to the ViZDoom workers.

Implication: one 8,192-decision iteration costs about 2.1 s of learner time plus about 0.3 s of inference. Against the MEASURED
env rates (RL_RESEARCH §1: 3–9k decisions/s at 8 processes on small maps), the synchronous loop spends roughly
40–70 % of wall time in the learner on small maps. ESTIMATE, to be re-measured with the real SubprocVecEnv. If confirmed, the
options are fewer epochs, overlapping the update with collection (APPO-lite, RL_RESEARCH §2) or a bptt_chunk of 32.
These are 2-iteration measurements on a shared machine, so treat them as ±20 %.

### 6.1 Real env (MEASURED, 2026-11-26, shared machine)

`real-two-rooms-smoke` (command in §1): 8 envs (2 workers × 4) on the smoke manifest, 6 updates, 6,144 decisions.
- **Correctness:** 0 crashes and ratio deviation ≤ 9.5e-07 on every update. Episodes ended by exit, death and timeout; on the timeout update, `trunc_value_mean` was 0.76 and `adv/at_truncation` was −0.009. Explained variance rose from −0.01 to 0.69.
- **Speed:** env-only SPS was **310–620**, total 270–450, with the learner taking 10–19 % of wall time. That env rate is far below the about 13k/s at 8 processes measured in RL_RESEARCH §1 with the raw wrapper, so the gap is in the A3 pipeline (preprocessing, reward terms, privileged reads, IPC) or in machine load. This is flagged to the environment-engineer and not yet diagnosed.

### 6.2 H1 pilot (MEASURED, 2026-11-27, run `h1-pilot`)

Setup:
- the H1 config (`h1.json`): 8 workers × 8 envs = 64, T = 128, 3 epochs × 4 minibatches, CPU with 2 threads;
- the family-a pool of 1,000 maps, uniform over the 5 stages;
- `--max-minutes 12` and `run.ckpt_every=40`, which gave a checkpoint about every 4 min.

| Quantity | Value |
| --- | --- |
| Decisions | 1,089,536 over 133 updates, in 714 s of iteration time |
| End-to-end SPS | **1,526**. Per quartile: 1,546 / 1,557 / 1,598 / 1,468. First 10 updates 1,580, last 10 1,425 (−10 %) |
| Env-only SPS | per quartile: 2,752 / 2,739 / 2,518 / 2,490 (−9.5 % over the run) |
| Time split | env 59.3 %, policy inference 7.6 %, learner 33.0 % (about 1.8 s per update) |
| Peak RSS, whole tree (75 processes) | **4.62 GB** (root process 0.62 GB) |
| Ratio check | max \|ratio−1\| = 0.0 on all 133 updates, 0 violations |
| Crashes | 0 crashed slot-steps, 0 done steps missing episode info |
| Episodes | 1,968 |
| Completion | 4.4 % (first quarter) → 20.1 % (last quarter) |
| Deaths | 45 % → 39 % |
| Cells per episode | 23.8 → 49.9 |
| Keys per episode | 0.03 → 0.19 |
| use_edges per episode | 66 → 85 |
| weapon_edges per episode | 87 → 68 |
| door_reversals per episode | 0.11 → 0.74 |
| Value | explained variance 0.26 → 0.94; value loss 0.84 → 0.17 |
| KL | approx-KL mean 0.0055, max 0.013. 13 iterations early-stopped. Clip fraction max 0.15 |
| Grad norm | median 1.39. Spikes of 20.8 / 10.5 / 29.9 / 28.1 at updates 28–31 (about 9.5× the running median, about 21× the whole-run median), with healthy KL and explained variance |
| Export → agent path | export of ckpt 133 played through the copied `agent.py`: p50 0.49 ms and p99 0.66 ms per act |

The 8×4 comparison (`h1-pilot-8x4`, 3 min) gave 1,475 end-to-end and 2,623 env-only SPS, against 1,542 and 2,703 for 8×8 over its first
40 updates, so **8×8 = 64 envs is kept**. The SPS decline could be thermal or a shift in the map mix as episodes go deeper; the data cannot separate the two.

## 7. Known limitations

- H1 needs reward terms that aren't registered yet: `cell_visit`, `kill`, `damage_taken`. `train.py` validates `reward.terms` with `RewardComposer` before any worker starts.
- The contract's `exit` term is implemented as `exit_reached`, and locked doors as `locked_door_open` (a separate sector set), so `h1.json` uses those names.
- `extra_value_heads` (H7) and aux heads raise `NotImplementedError` in the trainer. Their checkpoint and export slots exist.
- The Bernoulli (H2) arm has not been run through the trainer yet; the adapter supports it.
- `--set` overrides are ignored on `--resume`, because the config comes from the checkpoint.
- The ratio gate raises by default (`ppo.ratio_check_fail=true`, DECISIONS D17). Set it to false only for debugging.
