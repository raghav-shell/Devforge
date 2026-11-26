# Policy Model (v1)

Owner: model-architect. Implements `docs/MODULE_CONTRACTS.md` §3 (B1 and B2) and `docs/ACTION_SPACE.md` §2 and §4, under the main-session rulings in `docs/DECISIONS.md` (D2, D9, D14).
Code: `submission/doom_agent/models/`. Tests: `tests/model/`. Benchmark: `training/benchmarks/model_latency.py`.

## 1. Files and public API

| File | Public names | Dependencies |
| --- | --- | --- |
| `action_codec.py` | `ACTION_SPACE_VERSION`, `HEADS`, `HEAD_NAMES`, `HEAD_SIZES`, `NUM_HEADS`, `NUM_LOGITS`, `LOGIT_SLICES`, `NUM_JOINT_ACTIONS`, `BUTTON_OF_BINARY_HEAD`, `factors_to_buttons`, `buttons_to_factors`, `all_factor_tuples` | **numpy only** (env workers import it without torch; tested in a subprocess) |
| `distributions.py` | `FactoredCategorical`, `BernoulliHeads`, `make_distribution`, `actions_to_buttons`, `select_action`, `INFERENCE_MODES`, `ACTION_HEADS` | torch, numpy |
| `config.py` | `PolicyConfig` (`to_dict`, `from_dict`, `validate`), `CHOICES` | stdlib |
| `encoders.py` | `RayEncoder`, `EntityEncoder`, `VitalsEncoder`, `FiLMGenerator`, `CircularConv1d`, `init_linear` | torch, `doom_agent.obs.schema` |
| `policy.py` | `DoomPolicy`, `REQUIRED_KEYS` | the above |
| `io.py` | `features_to_torch(feats, device)` | torch, numpy |

`models/__init__.py` imports nothing, so `import doom_agent.models.action_codec` stays torch-free.

### `DoomPolicy` (the contract, §3.4)
```python
p = DoomPolicy(PolicyConfig())
s = p.initial_state(B, device)                              # zeros (B, 256) f32
out, s = p.step(obs_B, s, episode_start_B)                  # obs: (B, ...), episode_start: (B,) bool
out, sT = p.forward_sequence(obs_TB, s0, episode_start_TB)  # obs: (T, B, ...), episode_start: (T, B)
out["logits"]   # (…, 15) factored | (…, 9) bernoulli
out["value"]    # (…,)  V_ext
out["values"]   # {name: (…,)} extra value heads, {} by default
out["core"]     # (…, 256) GRU output (attachment point for training-only aux heads / privileged critic)
p.distribution(out["logits"], temperature=1.0)  # FactoredCategorical | BernoulliHeads
```
Extra helpers that are not in the contract: `encode`, `distribution` and `num_params`. `forward` is an alias of `step`.

**Done mask.** `episode_start[t]` is True iff `obs[t]` is the first observation of an episode. At every t, including t = 0, the state is set to zero with `torch.where` **before** `obs[t]` is consumed. `step` and `forward_sequence` both call the same `_core_step`, so there is exactly one masking implementation. `torch.where` is used instead of multiplying by the mask, so a NaN state is also cleared by a reset (tested).

**The hidden state is the explicit tensor that is passed in.** The module stores no per-episode state, and train and eval mode give the same outputs: there is no dropout and no batch normalisation (tested).

## 2. Architecture (v1 default)

All sizes are derived from `doom_agent.obs.schema.FEATURE_SPECS`: number of rays, slot count, feature widths, the embedding vocabularies (`high + 1`) and the vitals input width (27 + 9 + 6 = 42).

| Block | Layers | Output |
| --- | --- | --- |
| `RayEncoder` (`circconv`) | channels = dist, near, ddist ‖ Emb(6, 4)(hit) → 7 × 64. CircularConv1d 7→32 k5 s1 → 32→32 k5 s2 → 32→64 k3 s2 (ReLU each) → flatten 1024 → Linear 128, ReLU | z_ray 128 |
| `EntityEncoder` (`deepsets`) | masked slots sanitised with `torch.where` (features → 0, type → 0) **before** φ. φ: [feat 7 ‖ Emb(12, 8)] → 64 → 64 ReLU. Pool: masked mean ‖ masked max ‖ n/32 → ρ Linear 129→64, ReLU | z_ent 64 |
| `VitalsEncoder` | [vitals 27 ‖ prev_action 9 ‖ act_aux 6] → Linear 64, ReLU | z_vit 64 |
| Fusion (`concat`) | z_ray ‖ z_ent ‖ z_vit (256) → Linear 256, ReLU | z 256 |
| Core | `nn.GRUCell(256, 256)` | core = state 256 |
| Actor | Linear 256→15 (orthogonal, gain 0.01) | logits 15 |
| Critic | Linear 256→1 (gain 1.0) | value |
| Extra values | one Linear 256→1 per name in `extra_value_heads` | values[name] |

**MEASURED: 624,744 parameters** (contract estimate: about 625k).

### Initialisation (`research/RL_RESEARCH.md` §3)
- Hidden Linear and conv layers: orthogonal weights with gain √2, zero bias.
- GRU: an orthogonal block for each gate (r, z, n) of `weight_ih` and `weight_hh`, gain 1, zero biases.
- Actor: gain 0.01. Critic and extra value heads: gain 1.
- Embeddings: torch default N(0, 1).
- FiLM generator: zero weights and bias, so it starts at the identity (γ = 1, β = 0).
- **Head biases** (`head_bias_init`, default `{use: -2.0, next_weapon: -2.0}`):
  - The factored arm sets the class-1 logit of the named binary head, giving p(1) = σ(−2) = 0.1192.
  - The Bernoulli arm sets the bias of button 7 or 8.
  - Only binary heads are accepted; anything else raises `ValueError`. The D14 ablation is `{"next_weapon": 0.0}`.
  - Tested: exactly 0.1192 at zero core input for both arms, and the other heads stay uniform.

### Entity masking: why there is no NaN at n = 0
- φ ends in a ReLU, so every slot embedding is ≥ 0, and masked slots are set to exactly 0.
- The plain max over slots is therefore exactly the masked max when n > 0, and exactly 0 when n = 0. No `-inf` or softmax is involved.
- The mean divides by `clamp(n, 1)`.
- Masked inputs are replaced *before* φ (a multiply afterwards fails, because NaN × 0 = NaN, which also poisons the weight gradients). Garbage in padded slots, including NaN, inf and 1e30, gives bit-identical outputs, and the gradients stay finite (tested).

### `CircularConv1d`: an implementation detail with the same parameters
- `nn.Conv1d` subclass with the same `weight`/`bias` and `state_dict` keys as `nn.Conv1d(padding_mode="circular")`. Forward is circular pad (cat) + `unfold` + `matmul`.
- MEASURED on this macOS arm64 torch 2.15 CPU build (`torch.backends.mkldnn.is_available() == False`): the native conv1d took **124 ms** for one 2048 × 7 × 64 layer, against **8.4 ms** for unfold + matmul (1 thread, no grad).
- Before the change, the ray encoder was 212 ms of a 240 ms 16 × 128 sequence forward. After it, the encoder is 35 ms.
- At B = 1 the native kernel is about 5 µs per layer faster, and the crossover is at B ≈ 4. I kept **one code path for all batch sizes**:
  - a two-path version gave a step-vs-sequence state difference of 1.0e-5, which failed the 1e-5 test;
  - the PPO ratio gate (§4.3) needs rollout and replay numerics to agree.
- A/B at B = 1 in the same process: unfold 0.231–0.233 ms p50, native 0.212–0.213 ms p50 per `step`.
- Tested to equal `F.conv1d` on a circularly padded input, forward and input gradient, at B = 5 and B = 1.

## 3. Ablation flags (`PolicyConfig`)

| Key | Values (implemented = **bold**) | Params (MEASURED) | Notes |
| --- | --- | --- | --- |
| `core` | **gru**, lstm (reserved → `NotImplementedError`) | | LSTM state would be cat(h, c), S = 512 |
| `ray_encoder` | **circconv**, **mlp** (448 → 256 → 128) | 628,872 for mlp | Proposal A |
| `entity_encoder` | **deepsets**, **attnpool**, transformer (reserved → `NotImplementedError`) | 624,874 for attnpool | attnpool: the mean is replaced by a softmax attention pool with a learned, always-valid null slot, so it is finite at n = 0 |
| `fusion` | **concat**, **film** | 641,384 for film | FiLM: z_vit → (γ, β) on the last ray-conv channels (64) and on z_ent (64). Identity at init |
| `action_head` | **factored**, **bernoulli** | 623,202 for bernoulli | H2 |
| `head_bias_init` | dict of binary head → class-1 bias | | D14 |
| `extra_value_heads` | list of names, e.g. `["int"]` | +257 each | H7 |
| `use_egomap`, `use_depth_band` | `True` → `NotImplementedError` | | D7: v2 slots |

Every implemented variant runs through the same parametrised tests: shapes, step-vs-sequence equivalence and state-dict round trip. The masking, permutation and n = 0 tests run on the four encoder variants.

`from_dict` rejects unknown keys, and `validate` rejects unknown enum values. The default config serialises to exactly the §3.2 key set (tested).

## 4. Action distributions and selection

- `FactoredCategorical(logits, temperature=1.0)`:
  - `sample(generator)` and `mode()` return (…, 6) int64. Mode takes the first maximum, so ties go to "none".
  - `log_prob(idx)` and `entropy()` are sums over heads. `entropy_per_head()` is (…, 6). `probs()` and `head_log_probs()` are lists per head.
  - Sampling inverts the CDF with uniforms drawn on the generator's device, then moved to the logits' device. This works with a CPU generator and MPS logits, and `torch.multinomial` is not used.
- `BernoulliHeads`: the same method names, with (…, 9) logits and actions.
- `select_action(logits, mode, temperature, rng, action_head)`:
  - numpy only, float64, following ACTION_SPACE §4 literally: per head, in order 0..5, `u = rng.random()`, `searchsorted(cumsum(p), u, side="right")`, clipped. The Bernoulli arm uses one `rng.random(9)`.
  - Tested against an independent reimplementation of the contract text, and checked that torch and numpy give the same mode and matching sample frequencies.
- Training rollouts sample with the torch distribution. Eval-mode rollouts and `DoomAgent` use `select_action`.

## 5. Benchmarks (MEASURED)

Command: `.venv/bin/python training/benchmarks/model_latency.py --json runs/model-architect/bench-final/model_latency.json`.
Setup: torch 2.15.0, Python 3.13.18, Apple M4, torch default of 4 threads. Other agents were running throughout, at load average 3.1–3.3, and the numbers move with load: the same `step` measured 0.120 ms p50 at load 1.9 with native conv. These are macOS numbers; re-measure with `validate.py --docker`.

**1-thread, batch-1 latency** (`torch.set_num_threads(1)`, 2,000 iterations after 200 warm-up, inputs cycled from 64 `dummy_features`):

| Variant | Params | `step` p50 / p99 (ms) | act path p50 / p99 (ms) |
| --- | --- | --- | --- |
| v1 default | 624,744 | 0.238 / 0.363 | 0.367 / 0.539 |
| film | 641,384 | 0.277 / 0.476 | 0.418 / 0.643 |
| mlp rays | 628,872 | 0.152 / 0.210 | 0.296 / 0.610 |
| attnpool | 624,874 | 0.279 / 0.440 | 0.433 / 0.707 |
| bernoulli | 623,202 | 0.231 / 0.305 | 0.263 / 0.369 |

- The act path is `features_to_torch`, then `step`, then `select_action("sample")`, i.e. `DoomAgent.act` without `ObsPreprocessor`.
- Worst case p99 is 0.71 ms: **1.8 % of the 40 ms budget, and well under the 15 ms p99 target** of MODULE_CONTRACTS §3.5.

**Rollout inference** (batched `step`, no grad, CPU): B = 16 takes 0.66 ms p50 on 1 thread. B = 64 takes 2.0 ms on 1 thread and 2.2 ms on 4 threads.

**Training minibatch** (16 envs × 128 steps = 2,048 samples; forward_sequence + factored log-prob, entropy and value loss + backward + Adam; median of 10):

| Device | fwd + bwd + Adam | samples/s | fwd only (no grad) |
| --- | --- | --- | --- |
| CPU, 1 thread | 253 ms | 8,087 | 80 ms |
| CPU, 4 threads | 166 ms | 12,306 | 47 ms |
| MPS | 207 ms | 9,875 | 68 ms |

- Across the three runs of the current code (unfold conv) under changing load, CPU at 4 threads took 132–222 ms and MPS took 166–244 ms. One earlier run used the native conv (the slow CPU kernel), and there MPS was 151 ms (forward 39 ms) against CPU at 4 threads 469 ms. **There is no consistent MPS win with the current code, but this is provisional**: the CPU numbers roughly doubled under the other agents' load, and MPS was not A/B-tested native vs unfold in one process.
- The 128-step GRUCell loop is launch-bound on MPS. Recommendation: train on CPU at 4 or more threads, or choose per run after re-measuring on a quiet machine.
- ESTIMATE: one PPO update at the v1 defaults (64 envs × 128 steps, 4 minibatches × 3 epochs = 12 minibatches at about 166 ms each) takes about 2 s of compute.
- **CPU vs MPS drift:** max |logits_cpu − logits_mps| = **3.9e-6** on a 128 × 16 forward_sequence with perturbed weights. That is below the 1e-5 ratio gate, but close. If the trainer collects on CPU and updates on MPS, the epoch-1 ratio check (§4.3) has to be watched.
- **Memory:** the MPS current allocation for model, Adam and one minibatch was **14.8 MB** (driver 1.1 GB, which is the Metal pool, not model state). Peak RSS of the whole benchmark process was 892 MB (start 199 MB), mostly autograd activations of the CPU minibatches and the torch/MPS runtime. At eval, the weights are 2.5 MB in float32. The eval-relevant RSS: **211 MB** after all the 1-thread batch-1 latency runs (199 MB at start, which is mostly the torch import).

**Step vs sequence margin** (no grad, torch with 4 threads, random resets at p = 0.05):
- With weights perturbed off-init (+0.1·N(0, 1)), T = 32 and B = 4: max |Δlogits| was 3.3e-6 for the default, 5.2e-6 for film and 2.3e-6 for bernoulli, and max |ratio − 1| was up to 6.7e-6 for film.
- **At the trainer shapes** (rollout `step` at B = 64 for 128 steps, replayed as `forward_sequence` on 16-env slices): **exactly 0.0**, both at init and with perturbed weights.
- So on CPU, the §4.3 gate (< 1e-5) has margin at the shapes the trainer uses. The small-shape test (1e-5) has about 2–3× margin, and less for film.

**Where the training time goes** (1 thread, torch profiler, fwd + bwd): `unfold_backward` about 51 ms, matmuls about 70 ms, and the 128-step GRU loop about 14 ms forward. I tried a strided-slice column build instead of unfold. Forward + backward at 2048 samples, it was slower on the first two layers (22.5 vs 17.9 ms and 64.9 vs 43.8 ms) and about equal on the third (19.0–21.5 vs 19.9 ms). If the update ever dominates wall time, the next step is a fused `nn.GRU` over the reset-free segments (it needs a state_dict key migration, so it is deferred).

## 6. Tests (`tests/model/`, 78 tests)

| File | Covers |
| --- | --- |
| `test_action_codec.py` | frozen heads and slices; 216-action round trip, single and batched, int8 (9,) {0, 1}, injective; never opposing buttons; exact mapping table; `ValueError` on bad index, width or float; opposing buttons decode to none |
| `test_distributions.py` | entropy 5.37 nats at zero logits; log_prob against a manual computation; joint probabilities sum to 1 over 216 actions and entropy = −E[log p]; sample frequencies; mode and ties; temperature; Bernoulli log-prob against BCE; `select_action` greedy against torch mode, the sample stream against an independent contract reimplementation, numpy frequencies against torch probabilities, the Bernoulli draw order, validation |
| `test_policy.py` | shapes for B ∈ {1, 7} × 6 variants (step and sequence); spec-driven inputs and a missing key raises; `features_to_torch` dtypes; no module state, train == eval; permutation invariance; NaN/inf garbage in masked slots gives identical outputs; n = 0 finite outputs and gradients (all-empty and mixed batch); circular shift equivariance by multiples of the stride 4; **step vs forward_sequence < 1e-5 with mid-chunk resets, for 6 variants**; chunked sequence (16 + 16) == 32; reset isolation (a sequence prefix, a NaN state with start=True, a per-env reset); no future leakage; USE and next_weapon p = 0.1192 for both arms; near-uniform policy at init; reserved configs raise; greedy is deterministic and sampling is reproducible; strict state-dict round trip through `torch.save`, `torch.load(weights_only=True)` and a JSON config for 6 variants, with no aux/priv keys; config key set; param count; `CircularConv1d` == `F.conv1d` |
| `test_model_imports.py` | AST scan: allowed roots only (stdlib subset, torch, numpy, `doom_agent`), with no `training`, `env`, `doom_env` or `vizdoom`; `action_codec` and `__init__` have no torch; a subprocess check that the codec import leaves torch out of `sys.modules`; no `privileged_state` |

Mutation check (scratch script, files restored afterwards): each of the following made the suite fail:
- skipping the mask in `forward_sequence`;
- removing the pre-φ sanitising;
- zero instead of circular padding: run first against `nn.Conv1d(padding_mode=...)`, and re-run after the `CircularConv1d` rewrite by zeroing the wrap-around `cat`, which fails `test_ray_conv_circular_shift_equivariance` and `test_circular_conv_matches_native_conv1d`;
- a bias on the wrong class.

## 7. Decisions

- **DECISION: GRUCell loop core** rather than fused `nn.GRU`. Equivalence and reset isolation hold by construction, there is no fused-RNN-on-MPS risk, and the measured cost of the GRU loop is about 14 ms **forward only** (1 thread, B = 16 × 128 steps; its backward is not separated out) of a 166–253 ms fwd + bwd + Adam minibatch. Parameter names (`core.weight_ih`, …) are now frozen into checkpoints.
- **DECISION: `torch.where` masking** for state resets and entity slots, instead of multiplying, so NaN garbage can't leak.
- **DECISION: always-unfold circular conv** (§2).
- **DECISION: alt encoders `mlp`, `attnpool` and `film` are implemented and tested. `lstm`, `transformer`, `use_egomap` and `use_depth_band` raise `NotImplementedError`.**
- **Aux heads and the privileged critic are not in `DoomPolicy`** (D9). They attach to `out["core"]` from `training/`.
