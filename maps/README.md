# Maps

`two_rooms.wad` is a single small test map (one door, one monster, an exit). It's here so you can
check your setup and run `validate.py`. It is **not** an evaluation map and won't teach an agent
much.

Evaluation uses hidden, unseen maps across four tiers (see [docs/EVALUATION.md](../docs/EVALUATION.md)).
Finding or building a varied set of training maps is part of the challenge. Any WAD that ViZDoom can
load works with `doom_env`:

```python
from doom_env import make_env
env = make_env("path/to/your.wad", "MAP01", episode_ticks=2100)
```

Evaluation runs with the `freedoom2.wad` IWAD (bundled with ViZDoom), skill 3 and the vanilla engine.

To dry-run your agent on your own maps: `python validate.py --map path/to/your.wad:MAP05`.
