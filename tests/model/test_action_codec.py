"""B1 codec tests (docs/ACTION_SPACE.md 2.1, 6)."""

import numpy as np
import pytest

from doom_agent.models import action_codec as ac


def test_heads_and_slices_frozen():
    assert ac.HEADS == (("move", 3), ("strafe", 3), ("turn", 3), ("attack", 2), ("use", 2), ("next_weapon", 2))
    assert ac.NUM_LOGITS == 15
    assert ac.LOGIT_SLICES == (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 11), slice(11, 13), slice(13, 15))
    for (name, n), s in zip(ac.HEADS, ac.LOGIT_SLICES):
        assert s.stop - s.start == n, name
    assert ac.NUM_JOINT_ACTIONS == 216
    assert ac.ACTION_SPACE_VERSION == "1.0.0"


def test_roundtrip_all_216():
    idx = ac.all_factor_tuples()
    assert idx.shape == (216, 6)
    assert len({tuple(r) for r in idx}) == 216
    for row in idx:
        b = ac.factors_to_buttons(row)
        assert b.shape == (9,) and b.dtype == np.int8
        assert set(np.unique(b)) <= {0, 1}
        np.testing.assert_array_equal(ac.buttons_to_factors(b), row)
    # batched as well
    B = ac.factors_to_buttons(idx)
    assert B.shape == (216, 9) and B.dtype == np.int8
    np.testing.assert_array_equal(ac.buttons_to_factors(B), idx)
    # no two joint actions map to the same buttons
    assert len({tuple(r) for r in B}) == 216


def test_never_opposing_buttons():
    B = ac.factors_to_buttons(ac.all_factor_tuples())
    for a, b in ((0, 1), (2, 3), (4, 5)):
        assert not np.any((B[:, a] == 1) & (B[:, b] == 1))


def test_exact_mapping_table():
    expect = {
        (0, 1): 0, (0, 2): 1, (1, 1): 2, (1, 2): 3, (2, 1): 4, (2, 2): 5, (3, 1): 6, (4, 1): 7, (5, 1): 8,
    }
    for (k, choice), button in expect.items():
        idx = np.zeros(6, dtype=np.int64)
        idx[k] = choice
        b = ac.factors_to_buttons(idx)
        assert b.sum() == 1 and b[button] == 1
    assert ac.factors_to_buttons(np.zeros(6, dtype=np.int64)).sum() == 0
    assert ac.BUTTON_OF_BINARY_HEAD == {"attack": 6, "use": 7, "next_weapon": 8}


@pytest.mark.parametrize(
    "bad",
    [
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0],  # wrong width
        [0.5, 0, 0, 0, 0, 0],  # non-integer
    ],
)
def test_bad_indices_raise(bad):
    with pytest.raises(ValueError):
        ac.factors_to_buttons(np.asarray(bad))


def test_opposing_buttons_decode_to_none():
    b = np.zeros(9, dtype=np.int8)
    b[[0, 1, 4, 5]] = 1
    f = ac.buttons_to_factors(b)
    assert f[0] == 0 and f[2] == 0
    with pytest.raises(ValueError):
        ac.buttons_to_factors(np.full(9, 2))
