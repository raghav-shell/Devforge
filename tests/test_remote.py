"""Protocol tests with real agent-server subprocesses and misbehaving agents."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from doom_env import make_env, remote
from fixtures import fixture_wad

GOOD = """
import numpy as np
class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        self.n = action_space.shape[0]
        self.calls = 0
    def reset(self):
        self.calls = 0
    def act(self, obs):
        assert set(obs) == {"rgb", "depth", "vitals", "rays", "visible_entities"}
        assert obs["rgb"].shape == (120, 160, 3) and obs["rgb"].dtype == np.uint8
        obs["rgb"][0, 0, 0] = 1  # arrays must be writable
        self.calls += 1
        print("spam " * 100)  # stdout noise must not break the protocol
        a = np.zeros(self.n, np.int64); a[0] = 1
        return a
"""

VARIANTS = {
    "good": GOOD,
    "init_raises": "class DoomAgent:\n    def __init__(self, o, a, d):\n        raise RuntimeError('weights missing')\n",
    "act_raises": GOOD.replace("self.calls += 1", "raise ValueError('boom')"),
    "act_hangs": GOOD.replace("self.calls += 1", "import time; time.sleep(30)"),
    "bad_shape": GOOD.replace("a = np.zeros(self.n, np.int64)", "a = np.zeros(3, np.int64)"),
    "bad_values": GOOD.replace("a[0] = 1", "a[0] = 7"),
    "returns_str": GOOD.replace("return a", "return 'forward'"),
    "sys_exit": "import sys\nclass DoomAgent:\n    def __init__(self, o, a, d):\n        sys.exit(3)\n",
    "hard_crash": GOOD.replace("self.calls += 1", "import os; os._exit(9)"),
    "no_class": "x = 1\n",
    "gc_snoop": GOOD.replace(
        "self.calls += 1",
        "import gc\n        found = [o for o in gc.get_objects() if type(o).__name__ == 'DoomGame']\n"
        "        if found: raise RuntimeError('found DoomGame in agent process')",
    ),
}


@pytest.fixture
def env(tmp_path):
    e = make_env(fixture_wad("#######\n#P...E#\n#######"), episode_ticks=200)
    yield e
    e.close()


def start(tmp_path, variant):
    repo = tmp_path / variant
    (repo / "submission").mkdir(parents=True)
    (repo / "submission" / "agent.py").write_text(textwrap.dedent(VARIANTS[variant]))
    sock_path = tmp_path / f"{variant}.sock"
    srv = remote.listen(sock_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "doom_env.remote", "serve", "--socket", remote.address(srv), "--repo", str(repo), "--device", "cpu"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    conn = remote.AgentConnection.accept(srv, timeout=20, alive=lambda: proc.poll() is None)
    return proc, conn, srv


def stop(proc, conn, srv):
    conn.close()
    srv.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def test_good_agent_full_episode(tmp_path, env):
    proc, conn, srv = start(tmp_path, "good")
    info = conn.init("cpu", timeout=20)
    assert info["device"] == "cpu"
    conn.reset(timeout=5)
    obs, _ = env.reset(seed=0)
    for _ in range(30):
        action = conn.act(obs, timeout=5)
        assert action.dtype == np.int8 and action.tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 0]
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    stop(proc, conn, srv)
    assert proc.returncode == 0


def test_loopback_tcp_fallback(tmp_path, env, monkeypatch):
    """Native Windows Python has no AF_UNIX: local tools then talk to the agent over loopback TCP."""
    monkeypatch.setattr(remote, "HAS_UNIX_SOCKETS", False)
    test_good_agent_full_episode(tmp_path, env)


@pytest.mark.parametrize(
    "variant,where,kind",
    [
        ("init_raises", "init", "error"),
        ("sys_exit", "init", "error"),
        ("no_class", "init", "error"),
        ("act_raises", "act", "error"),
        ("act_hangs", "act", "timeout"),
        ("bad_shape", "act", "invalid_action"),
        ("bad_values", "act", "invalid_action"),
        ("returns_str", "act", "error"),
        ("hard_crash", "act", "crash"),
        ("gc_snoop", None, None),  # must NOT find the engine: no failure expected
    ],
)
def test_misbehaving_agents(tmp_path, env, variant, where, kind):
    proc, conn, srv = start(tmp_path, variant)
    obs, _ = env.reset(seed=0)
    try:
        conn.init("cpu", timeout=20)
        conn.reset(timeout=5)
        for _ in range(3):
            obs, *_ = env.step(conn.act(obs, timeout=2))
    except remote.AgentFailure as e:
        assert (e.where, e.kind) == (where, kind), str(e)
        if kind == "error":
            assert e.message
    else:
        assert where is None, f"expected a failure in {where}"
    finally:
        proc.kill()
        proc.wait()
        srv.close()


def test_startup_crash_detected_quickly(tmp_path):
    sock_path = tmp_path / "x.sock"
    srv = remote.listen(sock_path)
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
    with pytest.raises(remote.AgentFailure) as ei:
        remote.AgentConnection.accept(srv, timeout=10, alive=lambda: proc.poll() is None)
    assert ei.value.kind == "crash"
    srv.close()


def test_encode_rejects_object_arrays():
    with pytest.raises(remote.ProtocolError):
        remote.encode({"type": "x"}, {"a": np.array([object()])})


def test_roundtrip_nested():
    import socket
    a, b = socket.socketpair()
    obs = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "n": {"m": np.array([1, 0], np.int8)}}
    a.sendall(remote.encode({"type": "act"}, obs))
    header, data = remote.recv_message(b, None, 1 << 20)
    assert header["type"] == "act"
    np.testing.assert_array_equal(data["x"], obs["x"])
    np.testing.assert_array_equal(data["n"]["m"], obs["n"]["m"])


def _raw(header: dict | str, payload: bytes = b"") -> bytes:
    import json
    import struct
    h = header if isinstance(header, str) else json.dumps(header)
    return struct.pack(">I", len(h.encode())) + h.encode() + payload


HOSTILE = {
    "deep_json": "[" * 50000 + "]" * 50000,
    "huge_int": '{"type": "action", "seq": 1, "x": ' + "9" * 5000 + "}",
    "conflicting_paths": {"type": "action", "seq": 1, "arrays": [
        {"path": "a", "dtype": "int8", "shape": [1]}, {"path": "a/b", "dtype": "int8", "shape": [1]}]},
    "int64_wrap_shape": {"type": "action", "seq": 1, "arrays": [{"path": "action", "dtype": "int8", "shape": [2**24] * 4}]},
    "negative_size_bomb": {"type": "action", "seq": 1, "arrays": [
        {"path": "a", "dtype": "float64", "shape": [2**24, 2**24, 2**16]},
        {"path": "b", "dtype": "int8", "shape": [2**24, 2**24]}]},
    "not_an_object": "[1, 2, 3]",
    "wrong_seq": {"type": "action", "seq": 999, "arrays": []},
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_replies_become_protocol_failures(name):
    import resource
    import socket
    host, agent = socket.socketpair()
    conn = remote.AgentConnection(host)
    agent.sendall(_raw(HOSTILE[name]))  # pre-sent before the request even arrives
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with pytest.raises(remote.AgentFailure) as ei:
        conn.act({"x": np.zeros(3, np.int8)}, timeout=1.0)
    assert ei.value.kind in ("protocol", "timeout")
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before < 200_000  # < ~200 MB growth
    host.close()
    agent.close()


def test_presented_valid_reply_is_rejected():
    """A reply sent before the observation (to fake low latency) carries the wrong sequence number."""
    import socket
    host, agent = socket.socketpair()
    conn = remote.AgentConnection(host)
    early = remote.encode({"type": "action", "seq": 0}, {"action": np.zeros(9, np.int8)})
    agent.sendall(early)
    with pytest.raises(remote.AgentFailure, match="sequence"):
        conn.act({"x": np.zeros(3, np.int8)}, timeout=1.0)


def test_guessed_sequence_numbers_are_rejected():
    """Replies pre-sent with guessed counters (1, 2, 3, ...) never match the random request nonce."""
    import socket
    host, agent = socket.socketpair()
    conn = remote.AgentConnection(host)
    for guess in (1, 2, 3, "1"):
        agent.sendall(remote.encode({"type": "action", "seq": guess}, {"action": np.zeros(9, np.int8)}))
    with pytest.raises(remote.AgentFailure, match="sequence"):
        conn.act({"x": np.zeros(3, np.int8)}, timeout=1.0)
