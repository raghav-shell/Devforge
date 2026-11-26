"""Probe 6: value ranges of vitals and entity velocity features under the probe policies (for normalisation)."""
import json

import numpy as np

from _common import MAPS, POLICIES
from doom_env import make_env

V, E = [], []
for g in ("two_rooms", "freedoom2"):
    for wad, mp in MAPS[g]:
        for pol in POLICIES:
            env = make_env(wad, mp, episode_ticks=2400, render_buffers=False)
            obs, _ = env.reset(seed=0)
            p = POLICIES[pol](0)
            for t in range(600):
                V.append(obs["vitals"].copy())
                m = obs["visible_entities"]["mask"].astype(bool)
                E.extend(obs["visible_entities"]["features"][m])
                obs, _, te, tr, _ = env.step(p(obs))
                if te or tr:
                    break
            env.close()
V, E = np.array(V), np.array(E)
q = lambda x: np.round(np.quantile(x, [0, .01, .5, .99, 1]), 3).tolist()
from doom_env.spaces import VITALS_FIELDS, ENTITY_FEATURES
out = {"n_vitals": len(V), "n_entities": len(E)}
out.update({f"vit_{n}": q(V[:, i]) for i, n in enumerate(VITALS_FIELDS) if n in ("health", "armor", "selected_weapon", "selected_weapon_ammo", "ammo_bullets", "ammo_shells", "vel_forward", "vel_right", "vel_up")})
out["speed_xy"] = q(np.hypot(V[:, 14], V[:, 15]))
out.update({f"ent_{n}": q(E[:, i]) for i, n in enumerate(ENTITY_FEATURES)})
print(json.dumps(out, indent=1))
