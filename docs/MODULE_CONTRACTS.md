# Module Contracts (v1)

Owner: project-architect. Status: **v1 contract, pending main-session approval**. This is the source of truth for class and function names, tensor shapes, dtypes, config keys and the hidden-state layout (`docs/OWNERSHIP.md`, "Interfaces").
Companion contracts: `docs/OBSERVATION_SCHEMA.md` (features), `docs/ACTION_SPACE.md` (heads, codec, inference).

Conventions:
- B = batch (envs), T = time, H = 256 (GRU size), S = state width (= H for a GRU).
- Tensors are float32 unless stated. "(T, B, …)" means time-major.
- Every module below names its owner, target path, inputs and outputs, the state it keeps, its dependencies and its tests.
- Paths are taken from the ownership table (`docs/OWNERSHIP.md:13-29`). Any path not covered there is flagged `PROPOSED PATH`.

---

## 0. Rulings recorded for v1

- **DECISION (main session): a privileged-input critic is ALLOWED as training-only scaffolding.**
  - It must never be imported by `submission/`, and it is stripped from exported weights.
  - It is an **optional slot (§4.5), not built in v1**.
  - Alternative: forbid it. Rejected because `docs/RULES.md` §1 allows "any training-time scaffolding".
  - Kill criterion: H5 (`research/EXPERIMENT_HYPOTHESES.md:110-129`).
- **DECISION (main session): geodesic or path-distance reward shaping is DEFERRED.** It is not in v1. There is no interface for it, and no module may compute map paths.
- **DECISION (main session): ICP scan matching and the ego-map are DEFERRED to v2.** They are gated on the oracle-pose ego-map experiment first (`docs/SPATIAL_OBSERVATION_DESIGN.md` §6 B). v1 keeps only reserved slots (`docs/OBSERVATION_SCHEMA.md` §3).
- **DECISION (main session): the RGB key-colour hint is DEFERRED.** If it is ever needed, it is a learned RGB encoder, never hand rules.
- **DECISION (main session): v1 excludes depth, the ego-map and the ICP flag** (reserved slots only).
- **DECISION (main session): the v1 exploration reward is the episodic true-cell count bonus.** It is a training-only reward term in r_ext, with a single value head (`research/RL_RESEARCH.md` §4.1, §4.3). It is implemented as a reward term in `training/rewards/exploration.py` (§6.2).
- **DECISION: the policy runs on CPU at eval, whatever `device` is passed.**
  - MEASURED: the v1 network (625k params, the encoders of §3, GRUCell-256, 15 logits) takes **0.116 ms p50 and 0.125 ms p99** per batch-1 forward on 1 thread.
  - Measured with torch 2.15.0 and `torch.set_num_threads(1)`, 2,000 iterations, on the M4 at load average 1.6. Command: `.venv/bin/python /private/tmp/claude-501/-Users-raghavsharma-Desktop-DOOMSDAY/8af00418-f4ee-48ce-a8af-028a5b4c7dbc/scratchpad/bench_v1_policy.py`.
  - That is 0.3 % of the 40 ms budget. CUDA launch latency and host-device copies would only add time and VRAM risk.
  - These macOS numbers are not authoritative. Re-measure with `validate.py --docker`.

- **DECISION: the canonical import name is the top-level package `doom_agent` (`submission/doom_agent/`).**
  - The scorer and `validate.py` put `<repo>/submission` and `<repo>` on `sys.path` before loading `agent.py` (`env/doom_env/remote.py:344-346`, `validate.py:210`). So `agent.py` writes `from doom_agent.obs import ObsPreprocessor` with no `sys.path` edits.
  - `training/` must import the **same** name (`import doom_agent…`, never `submission.doom_agent…`), so there is one module identity.
  - PROPOSED to the main session (owner of `pyproject.toml`): add `submission` to the dev path, for example `[tool.pytest.ini_options] pythonpath = ["submission", "."]`, and have the `training/train.py` and harness entry points do the same.
  - `submission/doom_agent/__init__.py` (main session) must exist and stay import-light (no torch at package import).
  - Alternative: `submission.doom_agent` as a namespace package. Rejected, because it gives a different module path than the scorer's loader.
- **DECISION: runtime checks in the scored path warn and never raise; hard checks live upstream.**
  - In `DoomAgent`, these print one stderr warning and continue:
    - a mismatch in `obs_schema_version`, `action_space_version` or `obs_pipeline_hash`;
    - an invalid `DOOM_AGENT_INFERENCE`;
    - a non-int `DOOM_AGENT_SEED` (falls back to 0);
    - unexpected obs keys (`ObsPreprocessor(strict=False)`).
  - The same mismatches **fail hard** in `training/ppo/export.py`, the eval harness and `tests/`, so they can't reach a release silently.
  - Missing weights or a failed `strict` state-dict load still raise in `__init__`: there is no sensible fallback, and `validate.py` catches it.

---

## 1. Build order and parallelism

```
          [A1] obs/schema.py (FEATURE_SPECS, dummy_features)  <- unblocks everyone, build first (env-eng, ~hours)
         /                 |                      \
 [A2] ObsPreprocessor   [B1] action_codec +      [C1] rewards/base.py
     (env-eng)              distributions          (env-eng)
         |                  (model-arch)              |
 [A3] env factory +     [B2] DoomPolicy           [C2] CellVisitBonus (exploration-eng)
      vector env            (model-arch)          [C3] progression.py / combat.py terms
     (env-eng)                 |                        (main session / combat-eng)
         \                     |                      /
          +------> [D1] PPO rollout/GAE/loss/train.py (rl-training-eng) <---+
                               |              (can start now against dummy_features + a fake VecEnv)
                       [D2] checkpoint + export (rl-training-eng)
                               |
                       [E1] submission/agent.py wiring (main session)  -> validate.py gate
                               |
                       [F1] H1 baseline run; eval via training.eval.harness (eval researcher)
```

- **Parallel from day 0, once this contract is approved:**
  - A1 → A2 → A3 (environment-engineer);
  - B1 → B2 (model-architect);
  - C2 and C3 (exploration-engineer, main session, combat-and-survival-engineer), against the C1 interface in §6;
  - D1 (rl-training-engineer) on synthetic data.
- **Sequential gates:**
  - A1 must land first, but it is about 50 lines.
  - D1 integration needs A3 and B2.
  - E1 needs B1, B2, A2 and D2 (an exported random-init policy is enough to pass `validate.py`).
- **Critical path:** A1 → A2 → A3 → D1 integration → D2 → E1 → `validate.py` → H1.
- The model and reward work is off the critical path.

---

## 2. Observation preprocessor (environment-engineer)

| Item | Contract |
| --- | --- |
| Paths | `submission/doom_agent/obs/__init__.py`, `schema.py`, `preprocess.py` |
| Classes / functions | `ObsConfig`, `ObsPreprocessor` (`reset`, `process`, `commit_action`, `state_dict`), `FEATURE_SPECS`, `FeatureSpec`, `dummy_features`, `obs_pipeline_hash`, `OBS_SCHEMA_VERSION` |
| Input | the raw obs dict from `env/doom_env/spaces.py:73-95` (single env, no batch dim); committed action (9,) int8 |
| Output | `dict[str, np.ndarray]`, the v1 keys in `docs/OBSERVATION_SCHEMA.md` §2, with no batch dim |
| State | per-episode history (`docs/OBSERVATION_SCHEMA.md` §1). Reset by `reset()` |
| Depends on | numpy only. **No torch, no `training/`, no `env/`** |
| Used by | training workers (`training/envs/`), `submission/agent.py`. The same class is used in both: the "no eval-only shortcuts" guarantee |
| Tests | `tests/env/test_obs_*.py`, the list in `docs/OBSERVATION_SCHEMA.md` §5 |

---

## 3. Policy network (model-architect)

### 3.1 Files

| Path | Contents |
| --- | --- |
| `submission/doom_agent/models/config.py` | `PolicyConfig` (dataclass; `to_dict()` / `from_dict()`, JSON-serialisable) |
| `submission/doom_agent/models/encoders.py` | `RayEncoder`, `EntityEncoder`, `VitalsEncoder` |
| `submission/doom_agent/models/policy.py` | `DoomPolicy` |
| `submission/doom_agent/models/action_codec.py` | `docs/ACTION_SPACE.md` §2.1 (numpy) |
| `submission/doom_agent/models/distributions.py` | `FactoredCategorical`, `BernoulliHeads` |
| `submission/doom_agent/models/io.py` | `features_to_torch(feats: dict[str, np.ndarray], device) -> dict[str, Tensor]`. It keeps leading dims and converts dtypes: float32 → float32, int64 → int64, bool → bool |

Depends on torch 2.15.0 and numpy only. Nothing under `submission/doom_agent/models/` imports `training/`.

### 3.2 `PolicyConfig` keys (v1 defaults)

```
hidden_size: 256              core: "gru"                 # "lstm" reserved (state = concat(h, c), S = 2H)
ray_encoder: "circconv"       # alt "mlp" (Proposal A)
entity_encoder: "deepsets"    # alt "attnpool" (ablation 1), "transformer" (ablation 2)
fusion: "concat"              # alt "film" (cheap ablation, SPATIAL_OBSERVATION_DESIGN §5)
action_head: "factored"       # alt "bernoulli" (H2)
head_bias_init: {use: -2.0, next_weapon: -2.0}
extra_value_heads: []         # e.g. ["int"] from H7 on
use_egomap: false             use_depth_band: false       # v2 slots; true raises if the feature key is absent
obs_schema_version: "1.0.0"   action_space_version: "1.0.0"
```

**Flag authority.** `ObsConfig.enable_*` decides which keys the preprocessor emits. `PolicyConfig.use_*` decides which encoders exist. They must agree:
- `ObsConfig.enable_depth_band == PolicyConfig.use_depth_band`, and the same for the ego-map.
- `training/train.py` derives `PolicyConfig.use_*` from `ObsConfig`, so there is one source in the run config, `obs.*`. The export asserts that the two agree.

### 3.3 Architecture (v1, from `docs/SPATIAL_OBSERVATION_DESIGN.md` §2)

| Block | Input | Output | Notes |
| --- | --- | --- | --- |
| `RayEncoder` | `ray_dist`, `ray_near`, `ray_ddist` (B, 64) and `ray_hit` (B, 64) int64 → Embedding(6, 4) | `z_ray` (B, 128) | 7 channels × 64. Circular Conv1d 7→32 k5 s1, 32→32 k5 s2, 32→64 k3 s2, flatten (1024), Linear → 128, ReLU. Circular padding |
| `EntityEncoder` | `ent_feat` (B, 32, 7), `ent_type` (B, 32) → Embedding(12, 8), `ent_mask` (B, 32) bool | `z_ent` (B, 64) | φ: 15→64→64 ReLU per slot. Masked mean ‖ masked max ‖ count/32 → ρ 129→64. **Finite for n = 0** (the max is replaced by 0 when n = 0; no softmax over an all-masked row) |
| `VitalsEncoder` | `vitals_feat` (27) ‖ `prev_action` (9) ‖ `act_aux` (6) = (B, 42) | `z_vit` (B, 64) | Linear 42→64, ReLU |
| Fusion | `z_ray` ‖ `z_ent` ‖ `z_vit` = (B, 256) | `z` (B, 256) | Linear 256→256, ReLU |
| Core | `z`, state (B, 256) | `core` (B, 256) = new state | single-layer GRU |
| Actor | `core` | `logits` (B, 15) | Linear, orthogonal gain 0.01. Biases per `docs/ACTION_SPACE.md` §2.3 |
| Critic | `core` | `value` (B,) | Linear, gain 1.0 |
| Extra value heads | `core` | `values[name]` (B,) | one Linear per name in `extra_value_heads`. Empty in v1 |

Hidden layers use orthogonal init with gain √2 (`research/RL_RESEARCH.md` §3).

### 3.4 `DoomPolicy` API

```python
class DoomPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig): ...
    state_size: int                                    # S (= hidden_size for GRU)

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        """zeros (B, S) float32."""

    def step(self, obs: dict[str, Tensor], state: Tensor, episode_start: Tensor
             ) -> tuple[dict[str, Tensor | dict[str, Tensor]], Tensor]:
        """obs: v1 feature keys, each (B, ...). state: (B, S). episode_start: (B,) bool.
        Semantics: state = state * (~episode_start)[:, None]  BEFORE consuming obs; then one GRU step.
        Returns (out, new_state) with out tensors shaped (B, ...)."""

    def forward_sequence(self, obs: dict[str, Tensor], state0: Tensor, episode_start: Tensor
                         ) -> tuple[dict[str, Tensor | dict[str, Tensor]], Tensor]:
        """obs: each (T, B, ...). state0: (B, S) = state BEFORE step 0 of the chunk.
        episode_start: (T, B) bool. At every t: state = state * (~episode_start[t]); state = core(z_t, state).
        Encoders run batched over T*B; the core loops over T (or uses nn.GRU on segments).
        Returns (out with tensors (T, B, ...), final_state (B, S))."""
```

**Output dict `out` (frozen keys):**

| Key | Shape (step / sequence) | Dtype | Meaning |
| --- | --- | --- | --- |
| `"logits"` | (B, 15) / (T, B, 15) | f32 | actor logits, sliced by `action_codec.LOGIT_SLICES`. (B, 9) for the Bernoulli arm |
| `"value"` | (B,) / (T, B) | f32 | V_ext (normalised-return scale) |
| `"values"` | dict name → (B,) / (T, B) | f32 | extra value heads (`extra_value_heads`). `{}` in v1 |
| `"core"` | (B, 256) / (T, B, 256) | f32 | GRU output. This is the **aux-head and privileged-critic attachment point**. Gradients flow into the trunk |

**DECISION: the hidden state is one opaque tensor, (B, S).** There are no tuples. Rollout storage, masking and export are the same for GRU and LSTM: an LSTM stores `cat(h, c)`.
- Alternative: an `(h, c)` tuple. Rejected, because every consumer would need to branch on the core type.

**DECISION: done-mask convention = `episode_start[t]`.** It is True iff `obs[t]` is the first observation of an episode, so `episode_start[t] = done[t-1]`. The state is zeroed **before** step t is consumed.
- `step` and `forward_sequence` use the same convention, so a rollout can be replayed exactly.
- Alternative: a `done[t]` mask applied after the step. Rejected, because it moves the reset by one step and is the classic leak.

### 3.5 Tests (`tests/model/`)

1. Shapes and dtypes for B ∈ {1, 7}, using `dummy_features`.
2. **Step and sequence equivalence.** Running T = 32 steps one at a time with `step` must match one `forward_sequence` call, max abs diff < 1e-5 on `logits` and `value`. Use random `episode_start`, including resets mid-chunk.
3. **Reset isolation.** With `episode_start[t] = 1`, outputs from t on do not depend on anything before t.
4. **Permutation invariance** over entity slots. Garbage in masked slots does not change outputs. n = 0 gives finite outputs.
5. **Circular shift equivariance** of the ray conv stack before the flatten.
6. Param count, plus 1-thread batch-1 p50/p99 latency in `training/benchmarks/model_latency.py`. Reported in `docs/MODEL.md`. Budget: p99 < 15 ms (RL_RESEARCH §7).
7. `state_dict` round-trip through the export format (§8) with `strict=True`.

---

## 4. PPO-facing interface (rl-training-engineer)

Paths: `training/ppo/rollout.py` (`RolloutBuffer`), `training/ppo/gae.py` (`compute_gae`), `training/ppo/loss.py`, `training/ppo/trainer.py` (`PPOTrainer`), `training/train.py` (CLI), `training/configs/*.yaml|json`. Seeds used in training must be **< 10000** (`docs/EVALUATION_PROTOCOL.md` §5).

### 4.1 Rollout batch layout (`RolloutBuffer`, time-major, numpy or CPU torch)

N = num_envs (v1 default 64), T = rollout length (v1 default 128).

| Field | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `obs[k]` for each v1 feature key | (T, N, …) | as `FEATURE_SPECS` | features of `obs_t` |
| `episode_start` | (T, N) | bool | §3.4 convention |
| `state0` | (N, S) | f32 | **hidden state at chunk start**: the state passed into `step` at t = 0, *before* masking by `episode_start[0]` |
| `action_idx` | (T, N, 6) | int64 | factor indices ((T, N, 9) for the Bernoulli arm) |
| `action_mb` | (T, N, 9) | int8 | the codec output actually sent to the env |
| `logp` | (T, N) | f32 | joint log-prob under the behaviour policy |
| `value` | (T, N) | f32 | V_ext(obs_t) at rollout time |
| `values_extra[name]` | (T, N) | f32 | only if `extra_value_heads` is set |
| `reward` | (T, N) | f32 | total shaped training reward (sum of the weighted reward terms, §6) |
| `terminated`, `truncated` | (T, N) | bool | from the env for the transition t → t+1 |
| `trunc_value` | (T, N) | f32 | V_ext(final_obs_t, h_T) where `truncated[t]`, else 0 |
| `next_obs[k]`, `next_episode_start`, `next_state` | (N, …), (N,), (N, S) | | for the end-of-rollout bootstrap. `next_state` becomes `state0` of the next rollout |

### 4.2 Truncation bootstrapping (from `research/RL_RESEARCH.md` §3.3)

- When `truncated[t]`, the vector env returns the post-reset first obs as `obs_{t+1}`, and `final_feats` in `info` (§5.2).
  1. The trainer immediately runs `policy.step(final_feats, h_t_post, episode_start=False)` under `no_grad`. Here `h_t_post` is the state returned by step t, before any reset.
  2. It stores the value as `trunc_value[t]`.
  3. The final obs is stale (`env/doom_env/wrapper.py:142`, `docs/OBSERVATION_SCHEMA.md` §1), and this is accepted.
- **GAE**, with `done = terminated | truncated`:
  - δ_t = r_t + γ·[(1 − done_t)·V_{t+1} + truncated_t·trunc_value_t] − V_t
  - A_t = δ_t + γλ·(1 − done_t)·A_{t+1}
  - V_T is the bootstrap `V(next_obs, next_state)`.
- **Termination** (death or exit) does not bootstrap.
- **Required unit test:** a constant-reward fake env with a truncation; the advantage must equal the analytic value.

### 4.3 Sequences and minibatches

- v1 trains on whole-rollout sequences (BPTT = T = 128).
- Minibatches are formed **over envs**: 4 minibatches × 16 envs × 128 steps.
- Each minibatch replays `forward_sequence(obs[:, envs], state0[envs], episode_start[:, envs])`.
- H9's chunk-32 arm stores `state0` per chunk with shape (T/32, N, S). The field name stays `state0`.

**Mandatory check (gate for H1).** On epoch 1, minibatch 1, before any optimiser step, |π_new/π_old − 1| < 1e-5 on every sample (RL_RESEARCH §7). Log it every iteration.

### 4.4 Losses and config keys (names frozen; values in RL_RESEARCH §3)

| Group | Keys |
| --- | --- |
| `ppo` | `gamma` 0.995, `gae_lambda` 0.95, `clip_eps` 0.2, `lr` 3e-4, `adam_eps` 1e-5, `max_grad_norm` 0.5, `num_envs` 64, `rollout_len` 128, `num_minibatches` 4, `epochs` 3, `vf_coef` 0.5, `ent_coef` 0.005→0.001, `target_kl` 0.03, `value_clip` false, `return_norm` true |
| `aux` | `weight` (global, 0.25), `heads: []` (empty in v1) |
| `env` | §5.1 |
| `reward` | §6.3 |

### 4.5 Training-only extension slots (not built in v1)

| Slot | Path | Interface | Export |
| --- | --- | --- | --- |
| Aux heads | `training/ppo/aux_heads.py` | `class AuxHead(nn.Module)`: `name: str`, `forward(core: Tensor[..., 256]) -> Tensor`, `loss(pred, target, mask) -> Tensor`. Targets come from `info["privileged"]` or future legal obs. Collected by `AuxHeads(nn.ModuleDict)` | never exported |
| Privileged critic | `training/ppo/privileged_critic.py` | `class PrivilegedCritic(nn.Module)`: `forward(core: Tensor, priv_feat: Tensor) -> Tensor (value)`. Privileged inputs enter **only after** the shared torso, at `core` | **never exported. Never imported by `submission/`** (grep test) |

**DECISION: aux heads and the privileged critic live outside `DoomPolicy`, in `training/`, and attach to `out["core"]`.**
- Rationale: export is then stripping by construction, because `DoomPolicy.state_dict()` never contains them. `submission/` cannot import them even by accident.
- Alternative: heads inside `DoomPolicy`, filtered out at export. Rejected, because a key filter is easy to get wrong.

---

## 5. Env factory and vector env (environment-engineer)

### 5.1 Factory (`training/envs/factory.py`)

```python
@dataclass(frozen=True)
class EnvSpec:
    map_id: str; wad: str; map: str; episode_ticks: int     # from a maps/splits manifest entry
    render_buffers: bool = False                            # must be True if obs.enable_depth_band
    mods: tuple[str, ...] = ()                              # training-time engine mods; () = vanilla

def make_train_env(spec: EnvSpec, *, seed: int, reward_cfg: dict, obs_cfg: ObsConfig) -> "TrainEnv"
```

`TrainEnv` wraps `doom_env.make_env` and owns:
- one `ObsPreprocessor`;
- one `RewardComposer` (§6);
- the per-episode privileged bookkeeping.

**Map registry and sampler.** `training/envs/maps.py: MapSampler(manifest_paths, weights, rng)`:
- `sample() -> EnvSpec`;
- loads `train` and `smoke` manifests only. It **refuses `val.json` and `test.json` entries**, plus any WAD whose sha256 appears in val or test (`docs/EVALUATION_PROTOCOL.md` §7, test-set access policy). Today `train.json`, `val.json` and `test.json` are empty and only `smoke.json` has maps (`docs/EVALUATION_PROTOCOL.md` §1.5), so the first runs train on smoke maps. **Populating `train.json` is on the H1 critical path**;
- randomises `episode_ticks` ∈ [2100, 4200] when `env.randomize_ticks` is set.

Config keys:
- `env.manifests`, `env.randomize_ticks` (true), `env.render_buffers` (false), `env.mods` ([]), `env.return_privileged` (false);
- `env.num_workers` (8), `env.envs_per_worker` (8);
- `env.seed_base` (< 10000).

### 5.2 Vector env (`training/envs/vector.py`)

```python
class SubprocVecEnv:
    def __init__(self, sampler_cfg, *, num_workers: int, envs_per_worker: int, seed_base: int,
                 reward_cfg: dict, obs_cfg: ObsConfig): ...
    num_envs: int
    def reset(self) -> dict[str, np.ndarray]                          # feats (N, ...); all episode_start = True
    def step(self, actions_mb: np.ndarray) -> StepResult              # actions_mb (N, 9) int8
    def close(self) -> None

@dataclass
class StepResult:
    obs: dict[str, np.ndarray]      # (N, ...) features of the NEXT obs (post auto-reset where done)
    reward: np.ndarray              # (N,) f32 total shaped reward
    terminated: np.ndarray          # (N,) bool
    truncated: np.ndarray           # (N,) bool
    episode_start: np.ndarray       # (N,) bool == terminated | truncated (obs is a fresh episode)
    infos: list[dict]               # per env, see below; NEVER merged into obs
```

**Worker loop, per env:**
1. `commit_action(a)`.
2. `env.step(a)`.
3. Reward terms.
4. `feats = pre.process(obs)`.
5. If the episode is done: stash `final_feats = feats`, then `env.reset(seed=next)`, `pre.reset()`, `feats = pre.process(obs0)`.

**Seeding.** Every `env.reset(seed=…)` call in training uses a seed in **[0, 9999]**, because seeds ≥ 10000 are reserved for val and test (`docs/EVALUATION_PROTOCOL.md` §5).
- Per env slot i: `rng_i = np.random.default_rng([seed_base, i])`, and each episode's seed is `int(rng_i.integers(0, 10000))`. Map sampling uses `default_rng([seed_base, i, 1])`.
- This is deterministic given (`seed_base`, i, episode index).
- The seed pool is reused across episodes. That is accepted: map, spawn and `episode_ticks` randomisation provide the diversity.
- The realised (map_id, seed) of every episode goes into `info["episode"]`.

**Info keys** (training-only; the obs path never sees them; the policy never reads them):

| Key | When | Content |
| --- | --- | --- |
| `info["privileged"]` | every step, **opt-in** (`env.return_privileged: false` by default) | a slim subset of `env.privileged_state()` (`wrapper.py:203-227`): `x, y, angle, health, kills, damage_taken, keys, dead, exit_reached`. No sector arrays. Reward terms run **inside the worker** on the full dict, so the cross-process copy is only for aux labels and diagnostics. **Training-only** |
| `info["reward_terms"]` | every step | `dict[str, float]`, the unweighted value of each reward term |
| `info["final_feats"]` | done steps only, **except crash steps** (see below) | features of the terminal obs, for truncation bootstrapping (§4.2) |
| `info["episode"]` | done steps only | `{map_id, seed, length, return_shaped, return_env, outcome ∈ {exit, death, timeout}, cells, keys, kills, use_edges, weapon_edges}`. The default env return is logged too (`docs/EVALUATION_PROTOCOL.md` §9.1) |
| `info["map_id"]`, `info["tic"]`, `info["exit_reached"]` | every step | from `EnvSpec` and the wrapper info (`wrapper.py:199-200`) |

**Crash handling.** If a worker's ViZDoom process dies:
- the env is recreated and the slot is reported as `truncated` with `info["crashed"] = True`;
- there is no `final_feats`, and `trunc_value` is 0 (do not bootstrap from a garbage state);
- the crash is counted in logs.

**Tests (`tests/env/`):**
- auto-reset ordering;
- `episode_start` alignment;
- `final_feats` present exactly on non-crash done steps;
- no `privileged` key inside `obs`;
- the test-split refusal;
- the throughput benchmark `training/benchmarks/env_throughput.py`.

---

## 6. Reward-term plug-ins (`training/rewards/base.py`, environment-engineer)

### 6.1 Interface

```python
@dataclass(frozen=True)
class StepContext:
    priv_prev: dict          # privileged_state() before the step
    priv: dict               # privileged_state() after the step
    obs: dict                # raw official obs after the step (stale on the terminal step, wrapper.py:142)
    action_mb: np.ndarray    # (9,) int8 committed action
    terminated: bool
    truncated: bool
    env_reward: float        # wrapper default reward
    info: dict               # wrapper info (tic, exit_reached)
    step: int                # decision index in episode

class RewardTerm(Protocol):
    name: str                                     # unique, snake_case; key in info["reward_terms"]
    def reset(self, priv0: dict, obs0: dict, inventory: dict, spec: "EnvSpec") -> None: ...
    def __call__(self, ctx: StepContext) -> float: ...   # UNWEIGHTED value

REWARD_TERMS: dict[str, type[RewardTerm]]          # registry
def register_term(cls): ...                        # decorator, keyed by cls.name

class RewardComposer:
    def __init__(self, cfg: list[dict]): ...       # [{"name": ..., "weight": ..., "params": {...}}]
    def reset(self, priv0, obs0, inventory, spec) -> None
    def __call__(self, ctx) -> tuple[float, dict[str, float]]   # (weighted total, unweighted per-term)
```

**Rules**
- Terms are pure functions of `StepContext` plus their own per-episode state. They never read engine internals beyond `privileged_state()` and the inventory, and they never change the env.
- **Known wrapper quirks a term must handle:**
  - After an exit, `privileged_state()["tic"]` reads 0 (`docs/EVALUATION_PROTOCOL.md` §4).
  - On the terminal step, x, y and the game variables are the final ones (`wrapper.py:144`), but `obs` is stale.
- `inventory` comes from the PROPOSED `training/envs/inventory.py: map_inventory(env) -> dict` (`docs/EVALUATION_PROTOCOL.md` §9.1). The keys are `monsters`, `keys`, `door_sectors`, `sector_area` and `cells_total`.
- No term computes geodesic or path distances (deferred ruling, §0).

### 6.2 v1 terms and owners

| Term `name` | Path | Owner | Signal | Start weight (RESEARCH_NOTES §6) |
| --- | --- | --- | --- | --- |
| `cell_visit` | `training/rewards/exploration.py` | exploration-engineer | a new 64 × 64 true-(x, y) cell this episode (`priv`), once per cell | 0.05 |
| `exit`, `key_pickup`, `door_open` | `training/rewards/progression.py` | main session (until assigned) | `exit_reached`; a new bit in `keys`; door sector opening ≥ 56, once per sector (`door_lock`, floor/ceiling) | +10; +3; +1 (locked +3) |
| `kill`, `damage_taken`, `death` | `training/rewards/combat.py` | combat-and-survival-engineer | Δ`kills`; Δ`damage_taken`; `dead` | +0.5; −0.01 per HP; −3 |

### 6.3 Config

```
reward:
  terms:
    - {name: cell_visit, weight: 0.05, params: {cell: 64}}
    - {name: exit, weight: 10.0}
    - ...
```

**Tests (each owner, in their own test dir):**
- a hand-built `StepContext` sequence gives the expected values;
- `cell_visit` gives 0 for spinning in place;
- `door_open` fires once per sector;
- the composer total equals the weighted sum of the per-term values.

---

## 7. Metric plug-ins (evaluation-and-generalization-researcher; plug-ins by exploration and combat)

This adopts `docs/EVALUATION_PROTOCOL.md` §9.3 and `training/eval/metrics/core.py:1-10,62` as is:

```python
MetricFn = Callable[[EpisodeTrace, MapInventory], dict[str, float | int | str]]
```

- `EpisodeTrace` is defined in `training/eval/metrics/core.py:29-59`, and `MapInventory` in `training/eval/inventory.py:31`.
- Plug-ins are **pure** and never touch the env. Their keys are prefixed `expl.` (`training/eval/metrics/exploration.py`, exploration-engineer) or `combat.` (`training/eval/metrics/combat.py`, combat-and-survival-engineer). Values are JSON scalars.
- **PROPOSED CONTRACT (registry; eval researcher to implement):** each plug-in module exports `METRICS: list[MetricFn]`. `training/eval/metrics/__init__.py` collects them into `ALL_METRICS`, and `harness.py` calls every one per episode.
- A plug-in that needs more per-step fields requests them from the eval researcher, who adds them to `EpisodeTrace.record`.
- Evaluation always uses vanilla `doom_env.make_env`, never `training/envs` (`docs/EVALUATION_PROTOCOL.md` §5).

---

## 8. Checkpoint format and weight export (rl-training-engineer)

### 8.1 Training checkpoint

Written to `runs/rl-training-engineer/<run-id>/checkpoints/ckpt_<update:06d>.pt` with `torch.save`. Contents:

```
format_version: "ckpt-1"
policy_state: DoomPolicy.state_dict()      policy_config: PolicyConfig.to_dict()
aux_state: {...} | None                    privileged_critic_state: {...} | None
optimizer_state, scheduler_state, rng_states (python, numpy, torch)
return_norm_state                          # running std of returns
update, global_step (decisions), wall_time_s
obs_schema_version, obs_pipeline_hash, obs_config      # docs/EVALUATION_PROTOCOL.md §9.4
action_space_version
train_manifest_ids: [...], train_seeds: {seed_base, ...}   # all seeds < 10000
ppo_config, reward_config, env_config, git_commit, git_dirty
```

The full run config and metrics go to `runs/<owner>/<run-id>/run.json` and to logs (CLAUDE.md rule 10).

### 8.2 Export (`training/ppo/export.py`, CLI `python -m training.ppo.export --ckpt … --out DIR`)

| Output file | Content |
| --- | --- |
| `DIR/policy.pt` | `DoomPolicy.state_dict()` only, as CPU float32 tensors. Saved with `torch.save` and loadable with `torch.load(weights_only=True)`. **No optimiser, no aux heads, no privileged critic, no return-norm state** |
| `DIR/policy.json` | `{format_version: "export-1", policy_config, obs_schema_version, obs_pipeline_hash, obs_config, action_space_version, default_inference, inference_temperature, source_ckpt_sha256, git_commit, global_step}` |

- The export refuses a checkpoint whose `obs_pipeline_hash` differs from the current tree. An override flag exists for deliberate migrations, and every override is logged.
- The export verifies that `policy.pt` keys == `DoomPolicy(cfg).state_dict()` keys (strict).
- The **main session** copies `DIR/*` into `submission/weights/` (main-session owned) and commits it with Git LFS (`*.pt` is already tracked; `docs/SUBMISSION.md`).
- Tests: export → load in a fresh process with `strict=True` → identical logits on `dummy_features`. A grep shows no `aux`/`priv` keys.

---

## 9. `submission/agent.py` wiring (main session)

```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        torch.set_num_threads(1)
        meta = json.load(open(WEIGHTS_DIR / "policy.json"))
        # version / obs_pipeline_hash mismatch -> one stderr WARNING, continue (section 0); export + harness fail hard
        self.cfg = PolicyConfig.from_dict(meta["policy_config"])
        self.policy = DoomPolicy(self.cfg).eval()                                         # CPU (section 0)
        self.policy.load_state_dict(torch.load(WEIGHTS_DIR / "policy.pt", weights_only=True), strict=True)
        self.pre = ObsPreprocessor(ObsConfig(**meta["obs_config"]), strict=False)
        self.mode = os.environ.get("DOOM_AGENT_INFERENCE") or meta["default_inference"]   # validated, see ACTION_SPACE §4
        self.temperature = float(meta.get("inference_temperature", 1.0))
        self.seed = int(os.environ.get("DOOM_AGENT_SEED", "0")); self._reset_count = 0
        # warm-up: one dummy step through pre + policy, so the first act() has no lazy-init cost
    def reset(self):
        self.pre.reset(); self.state = self.policy.initial_state(1); self.start = True
        self._reset_count += 1; self.rng = np.random.default_rng([self.seed, self._reset_count])
    def act(self, observation):
        feats = self.pre.process(observation)
        x = features_to_torch({k: v[None] for k, v in feats.items()}, "cpu")
        with torch.inference_mode():
            out, self.state = self.policy.step(x, self.state, torch.tensor([self.start]))
        self.start = False
        a = select_action(out["logits"][0].numpy(), self.mode, self.temperature, self.rng,
                          self.cfg.action_head)                                       # (9,) int8, ACTION_SPACE §4
        self.pre.commit_action(a)
        return a
```

- `select_action` lives in `submission/doom_agent/models/distributions.py` (numpy path, `docs/ACTION_SPACE.md` §4), so that training eval and the submission share it.
- Imports follow the `doom_agent` convention in §0. There must be no import of `training/` or `env/`.
- **Gates:**
  - `uv run --no-sync python validate.py` passes;
  - the Mac-safe pytest command (`docs/OWNERSHIP.md` rule 7) passes;
  - `validate.py --docker` before any release candidate.

---

## 10. Ownership and interface summary

| # | Module | Owner | Path(s) | Provides | Consumes | Parallel group |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Feature spec | environment-engineer | `submission/doom_agent/obs/schema.py` | `FEATURE_SPECS`, `dummy_features`, `obs_pipeline_hash` | OBSERVATION_SCHEMA | first, blocks B2/D1 tests (stub from this doc meanwhile) |
| A2 | Preprocessor | environment-engineer | `submission/doom_agent/obs/preprocess.py` | `ObsPreprocessor` | raw obs | A |
| A3 | Env factory, vec env, map sampler, inventory | environment-engineer | `training/envs/{factory,vector,maps,inventory}.py` | `make_train_env`, `SubprocVecEnv`, `StepResult`, `MapSampler`, `map_inventory` | A2, C1, manifests | A |
| B1 | Action codec, distributions | model-architect | `submission/doom_agent/models/{action_codec,distributions}.py` | codec, `FactoredCategorical`, `select_action` | ACTION_SPACE | B |
| B2 | Policy | model-architect | `submission/doom_agent/models/{config,encoders,policy,io}.py` | `DoomPolicy`, `PolicyConfig`, `features_to_torch` | A1, B1 | B |
| C1 | Reward framework | environment-engineer | `training/rewards/base.py` | `RewardTerm`, `RewardComposer`, `StepContext`, registry | privileged_state | A |
| C2 | Cell-visit bonus (+ `expl.` metrics) | exploration-engineer | `training/rewards/exploration.py`, `training/eval/metrics/exploration.py` | `cell_visit` term, `METRICS` | C1, §7 | C |
| C3 | Progression / combat terms (+ `combat.` metrics) | main session / combat-and-survival-engineer | `training/rewards/progression.py`, `training/rewards/combat.py`, `training/eval/metrics/combat.py` | terms, `METRICS` | C1, §7 | C |
| D1 | PPO | rl-training-engineer | `training/ppo/{rollout,gae,loss,trainer}.py`, `training/train.py`, `training/configs/` | `RolloutBuffer`, `compute_gae`, `PPOTrainer` | A3, B2 (stubs until then) | D (starts now on stubs) |
| D2 | Checkpoint and export | rl-training-engineer | `training/ppo/export.py` | `ckpt-1`, `export-1` | B2 | after B2 |
| E1 | Agent wiring | main session | `submission/agent.py`, `submission/weights/` | `DoomAgent` | A2, B1, B2, D2 | after D2 |
| F1 | Eval harness, splits, registry | evaluation-and-generalization-researcher | `training/eval/`, `maps/splits/` | harness, `ALL_METRICS` | E1 | exists; registry is a small change |

---

## 11. Open questions

- **OPEN QUESTION: the H13 frozen inference default** (`docs/ACTION_SPACE.md` §4). It is provisionally `sample` with T = 1.0.
- **OPEN QUESTION: USE-debounce mask legality** (`docs/ACTION_SPACE.md` §5). It is not built in v1.
- **OPEN QUESTION: crash safety in `act()`.** If inference raises, the agent stops and stands still (`docs/EVALUATION.md`). Should `act()` catch the exception, log it to stderr and return NOOP, keeping the GRU state? That is functionally the same as stopping, but it lets later steps recover. Main-session call. It must never hide errors in tests.
- **OPEN QUESTION: `map_inventory` location.** The eval protocol proposes `training/envs/inventory.py` (environment-engineer). `training/eval/inventory.py` exists today. Both reward terms and metrics need it, so there should be one implementation and the other path should import it.
