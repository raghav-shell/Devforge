# Action Space (v1)

Owner: project-architect. Status: **v1 contract, pending main-session approval**. `ACTION_SPACE_VERSION = "1.0.0"`.
Implementer: model-architect, in `submission/doom_agent/models/action_codec.py` and `submission/doom_agent/models/distributions.py`.
Consumers: rl-training-engineer, `submission/agent.py` (main session), `ObsPreprocessor.commit_action`.

---

## 1. Button contract (fixed, never changed)

`gymnasium.spaces.MultiBinary(9)` (`env/doom_env/spaces.py:58-68,98-99`, `docs/CONTRACT.md` §3).

| Index | Button | Edge semantics |
| --- | --- | --- |
| 0 | MOVE_FORWARD | held |
| 1 | MOVE_BACKWARD | held |
| 2 | MOVE_LEFT (strafe) | held |
| 3 | MOVE_RIGHT (strafe) | held |
| 4 | TURN_LEFT | held. The Doom turn ramp means a slower first tic, and the first step after reset ignores turning (R§4.1, R§9) |
| 5 | TURN_RIGHT | held |
| 6 | ATTACK | held (auto-fire while held) |
| 7 | USE | **fires on the rising edge**. Holding it presses once. Pressing it on a moving door **reverses** the door |
| 8 | SELECT_NEXT_WEAPON | **fires on the rising edge** |

Each button is held for all 4 tics of a step (`spaces.py:70`).

**Validity.** `act()` must return shape (9,) with values in {0, 1}. Anything else ends the map (`env/doom_env/remote.py:209-217`, `docs/CONTRACT.md` §3).

**DECISION: actions are always returned as `np.ndarray`, shape (9,), dtype `np.int8`.** This is the dtype the server validates to (`remote.py:217`) and the one `training/eval/harness.py:40` uses for NOOP.

---

## 2. Policy head: factored categorical (DECISION, main session)

**DECISION (main session, recorded): six independent categorical factor heads, mapped one-to-one onto MultiBinary(9).**
- Rationale (`research/RL_RESEARCH.md` §3.1):
  - Independent Bernoulli heads at p = 0.5 hold FORWARD and BACK together 25 % of the time, and TURN_LEFT and TURN_RIGHT together 25 % of the time. Opposing turns give a measured 0° turn.
  - The same initialisation presses USE about half the time, so a new rising edge, and a door reversal, comes about every 4 steps.
  - Factor heads remove the cancelling combinations without losing expressivity, because pressing both opposing buttons only cancels.
  - Sample Factory used this kind of factored head for VizDoom [V].
- Alternatives considered:
  1. 9 Bernoulli heads. This is the H2 control arm, kept behind `model.action_head = "bernoulli"`.
  2. A discrete macro set (for example 12–20 hand-picked combinations). Rejected: it loses combinations such as strafe + turn + fire, and choosing the set is hand design.
  3. A full joint softmax over 216 actions. Rejected: it does not factor, and the logit count grows multiplicatively.
- Metric and ablation: H2 (`research/EXPERIMENT_HYPOTHESES.md:61-76`) compares decisions-to-`cells`, USE rising edges per episode and door reversals.

### 2.1 Heads and the exact mapping table

Head order and logit layout are frozen. The policy emits one tensor, `logits`, of shape (…, 15), sliced as follows:

| Head (k) | Name | n choices | Logit slice | Choice → buttons set to 1 |
| --- | --- | --- | --- | --- |
| 0 | `move` | 3 | `[0:3]` | 0 none → ∅ · 1 fwd → {0} · 2 back → {1} |
| 1 | `strafe` | 3 | `[3:6]` | 0 none → ∅ · 1 left → {2} · 2 right → {3} |
| 2 | `turn` | 3 | `[6:9]` | 0 none → ∅ · 1 left → {4} · 2 right → {5} |
| 3 | `attack` | 2 | `[9:11]` | 0 → ∅ · 1 → {6} |
| 4 | `use` | 2 | `[11:13]` | 0 → ∅ · 1 → {7} |
| 5 | `next_weapon` | 2 | `[13:15]` | 0 → ∅ · 1 → {8} |

The joint space has 3·3·3·2·2·2 = 216 actions. Maximum entropy is 3 ln 3 + 3 ln 2 = 5.37 nats.

`action_codec.py` (numpy only; torch versions of the same functions may live alongside):

```python
ACTION_SPACE_VERSION = "1.0.0"
HEADS: tuple[tuple[str, int], ...] = (("move", 3), ("strafe", 3), ("turn", 3),
                                      ("attack", 2), ("use", 2), ("next_weapon", 2))
NUM_LOGITS = 15
LOGIT_SLICES: tuple[slice, ...]          # derived from HEADS, as in the table above
# _TABLE[k][choice] -> button index, or -1 for "none"
def factors_to_buttons(idx: np.ndarray) -> np.ndarray   # (..., 6) int64 -> (..., 9) int8
def buttons_to_factors(buttons: np.ndarray) -> np.ndarray  # (..., 9) {0,1} -> (..., 6) int64
```

- `factors_to_buttons` is a pure table lookup. It must reject any index that is out of range (`ValueError`).
- `buttons_to_factors` is only for tests and for logging Bernoulli-arm actions. When both opposing buttons are 1, it returns choice 0 (none), which is the engine's net effect for opposing turns (RESEARCH_NOTES §2).
- Tests: for all 216 index tuples, `buttons_to_factors(factors_to_buttons(i)) == i`; every output is a valid MultiBinary(9).

### 2.2 Distribution (`distributions.py`)

```python
class FactoredCategorical:
    def __init__(self, logits: Tensor): ...                  # (..., 15)
    def sample(self, generator: torch.Generator | None = None) -> Tensor  # (..., 6) int64
    def mode(self) -> Tensor                                 # (..., 6) per-head argmax, ties -> lowest index
    def log_prob(self, idx: Tensor) -> Tensor                # (...,) = sum over heads
    def entropy(self) -> Tensor                              # (...,) = sum over heads
    def entropy_per_head(self) -> Tensor                     # (..., 6), logged per head (RL_RESEARCH §7)
```

- The joint log-prob and entropy are **sums over heads**. The PPO entropy coefficient applies to the summed entropy (`research/RL_RESEARCH.md` §3.1: start at 0.005). This must be documented in `docs/TRAINING.md`.
- The H2 Bernoulli arm uses `BernoulliHeads` with the same method names. Its `logits` are (…, 9), its actions are (…, 9), and the codec is the identity.

### 2.3 Initialisation

| Parameter | Init | Rationale |
| --- | --- | --- |
| Policy-head weight (15 × 256) | orthogonal, gain 0.01 | near-uniform start (RL_RESEARCH §3, "Initialisation") |
| `move`, `strafe`, `turn`, `attack` biases | 0 | uniform start |
| **`use` bias** | **class 0: 0.0, class 1: −2.0** → p(USE) = σ(−2) ≈ 0.12 | fewer random rising edges, so fewer door reversals early (RL_RESEARCH §3.1). It is a learnable initial parameter, not a rule |
| **`next_weapon` bias** | **class 0: 0.0, class 1: −2.0** → p ≈ 0.12 | DECISION, see below |

Config keys are `model.head_bias_init.use = -2.0` and `model.head_bias_init.next_weapon = -2.0`.

**DECISION: also bias `next_weapon` to −2.**
- Rationale: it has the same rising-edge rule as USE. At p = 0.5, a new switch edge comes about every 4 steps. A switch takes the weapon down and up again and interrupts fire, which disrupts early combat learning.
- Alternative: 0 bias. This is the cheap ablation if the main session disagrees.
- Metric: weapon switches per episode and kills at 10M decisions.

---

## 3. Edge-trigger handling (USE, SELECT_NEXT_WEAPON)

- **The learned policy handles both triggers.** There is no hand-written debounce, toggle or "press once then wait" logic. That would be a scripted door sequence (`docs/RULES.md` §1, OWNERSHIP rule 2).
- **The inputs make the edge rule learnable:**
  - `prev_action[7]` and `prev_action[8]` (last committed USE and NEXT_WEAPON);
  - `act_aux[0]` `steps_since_use_edge` (`docs/OBSERVATION_SCHEMA.md` §2.3);
  - the GRU state.
  With these, the policy can tell "USE held" (no new press) from "USE just went down".
- **Training-time observability.** Log per episode:
  - USE rising edges;
  - door reversals (privileged, from door sector heights);
  - weapon-switch edges.
  The H2 metric and the door-bouncing alarm (> about 5 reversals of one door per episode, RL_RESEARCH §7) read these logs.
- **No masking in v1.** Every head can choose any value at every step.

---

## 4. Inference at evaluation

`submission/agent.py` (main session) selects actions from the policy's `logits` in one of two modes:

| Mode | Rule | RNG use |
| --- | --- | --- |
| `greedy` | per-head argmax of `logits` (ties go to the lowest index, i.e. "none" / 0) | none |
| `sample` | per-head categorical sample from `softmax(logits_k / T)`, with temperature T | numpy `Generator` |

**Selection mechanism** (`docs/EVALUATION_PROTOCOL.md` §9.2):
- `DOOM_AGENT_INFERENCE` ∈ {`sample`, `greedy`} is read once in `DoomAgent.__init__`.
  - If it is unset or empty, the mode is the **frozen default** stored in the export metadata `submission/weights/policy.json`, key `"default_inference"`.
  - Any other value prints one warning to stderr and falls back to the frozen default. It never raises, because the scorer never sets the variable and an exception would cost a map.
- The temperature T is **not** an env var. It comes from `policy.json` key `"inference_temperature"`, which defaults to 1.0. H13 sets it before the export is frozen.
- `DOOM_AGENT_SEED` (int, default 0) seeds sampling. In `reset()`, `self._reset_count += 1`, then `self.rng = np.random.default_rng([seed, self._reset_count])`. So map k of a pass always gets the same RNG stream for a given seed, whatever the earlier episodes' lengths were. This fixes the order dependence the eval researcher measured for the random baseline (`docs/EVALUATION_PROTOCOL.md` §5, "Lifecycle").
- **Sampling is done in numpy, on CPU, from float64 probabilities.** For each head: `u = rng.random()`, then `choice = searchsorted(cumsum(p_k), u, side="right")`, clipped to n_k − 1. The draw order is fixed as head 0 … 5, so the stream does not depend on torch or the device.
- **Shared selector** (`submission/doom_agent/models/distributions.py`, numpy only):
  ```python
  def select_action(logits: np.ndarray, mode: str, temperature: float,
                    rng: np.random.Generator | None, action_head: str = "factored") -> np.ndarray:
      """logits: (15,) or (9,) float32 for ONE agent (no batch dim). Returns MultiBinary (9,) int8.
      factored : per-head argmax (greedy) or the searchsorted draw above (sample), then factors_to_buttons.
      bernoulli: greedy = (logits > 0); sample = (rng.random(9) < sigmoid(logits / T)), drawn in index order 0..8.
      mode must be 'greedy' or 'sample' (validated by the caller); rng may be None only for 'greedy'."""
  ```
  It returns the final button vector, so `DoomAgent` never calls the codec directly. The same function is used by eval-mode training rollouts.
- The mode changes **only** the action selection. Preprocessing, the model forward and state updates are identical in both modes.

**OPEN QUESTION (H13): the frozen default.** H13 (`research/EXPERIMENT_HYPOTHESES.md:251-261`) decides `sample` vs `greedy` and T ∈ {0.7, 1.0} on **val** only.
- Until H13 runs, exports set `"default_inference": "sample"` and `"inference_temperature": 1.0`. This is provisional, on the "sampling breaks deterministic loops" hypothesis (RL_RESEARCH §6).
- The main session freezes the winner before the release candidate.

**Training rollouts always sample (T = 1).** Greedy is used only in evaluation runs.

---

## 5. Legal masking (none in v1)

`docs/RULES.md` §1 allows "action masking from observation fields".

**DECISION: v1 applies no masks.** Every module needs a hypothesis, a metric and an ablation, and no mask has evidence behind it yet.

Candidates if one is needed later:
- **Mask ATTACK when `selected_weapon_ammo == 0`.** Not useful: the fist and chainsaw need no ammo.
- **Force USE = 0 on the step after USE = 1** (a debounce from own action history). See the open question below.

**OPEN QUESTION: is a USE-debounce mask legal and useful?**
- It reads only our own previous action, and it guarantees that every USE = 1 is a rising edge. That halves the effective action semantics the policy has to learn.
- It is arguably "action masking" (allowed), but it is also close to a hand-written door rule. It comes from own actions, not from an observation field.
- Do not build it in v1. If H2 shows door bouncing persisting past 10M decisions, bring it to the main session and Fable as a ruled ablation.

---

## 6. Tests (model-architect, `tests/model/`)

1. Codec round-trip over all 216 tuples. Every output has dtype int8, shape (9,) and values in {0, 1}.
2. `FactoredCategorical.log_prob(sample())` is finite. `entropy()` at zero logits equals 5.37 nats ± 1e-5.
3. After init with the head biases, p(USE = 1) ≈ 0.119 and p(NEXT_WEAPON = 1) ≈ 0.119 at zero trunk input.
4. `greedy` is deterministic. `sample` with the same `(seed, reset_count)` gives an identical action stream across two `DoomAgent` instances.
5. Logit slices match `HEADS`, which guards against silent re-ordering.
