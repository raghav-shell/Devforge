# Ground Rules

Breaking these rules can lead to disqualification, even with a high automated score.
Organizers manually review the code and replays of the top 4–5 teams.

## 1. Decisions must come from trained parameters

Final agents must choose actions through **trained neural network parameters**. Not allowed:

- Hand-written `if/else` rule trees that choose actions.
- Classical pathfinding or planning (`A*`, `Dijkstra`, BFS over a map, etc.).
- Trigonometric aimbots (e.g. `atan2` on entity angles to turn toward enemies).
- Unmodified, off-the-shelf `Stable-Baselines3` (or similar) policies with default settings.
  You may use libraries, but the core architecture and training pipeline must be your own.

Allowed: preprocessing and encoding observations, action masking from observation fields,
frame stacking, learned memory modules, and any training-time scaffolding.

## 2. Training vs evaluation

- **During training** you may change anything: the engine, map scripts (ACS), game variables,
  auto-opening doors, infinite ammo, curricula, separate combat/navigation phases, and so on.
- **During evaluation** the scorer runs the **standard, unmodified** engine with vanilla rules, the
  official `doom_env` wrapper, and **unseen maps**. Your repository cannot change any of this. The
  scorer ignores the `env/` folder in your repo.

## 3. Original work

- Create your repository **during the hackathon** from this template.
- You may use existing open-source libraries. Your core agent architecture and training pipeline
  must be built during the event.
- Keep your training code in the repository (e.g. under `training/`). Judges read it.

## 4. Fair play at evaluation

Any of these leads to disqualification:

- Trying to reach the game engine, the evaluation maps, or the host from your agent.
- Trying to reach the network or to escape the sandbox.
- Deliberately using up shared resources (GPU memory, CPU, disk) to hurt other teams.

## 5. Judging flow

1. The automated scorer ranks every team on the hidden maps. See [EVALUATION.md](EVALUATION.md).
2. Organizers review the code and replays of the top 4–5 for rule violations.
3. The top 2 are shortlisted for final judging. Innovative, custom-engineered architectures
   (custom spatial memory, curiosity, hierarchical control, etc.) are rewarded.
