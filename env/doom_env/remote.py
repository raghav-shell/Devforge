"""Process boundary between the environment (host) and the agent (sandboxed process).

Used identically by `validate.py` (template) and the private scorer:

    host: env + metrics  --obs-->  agent server (subprocess / container)  --action-->  host

Transport: a unix domain socket. It works with `docker --network none` through a bind-mounted
directory, and since it's separate from stdout/stderr, team prints can't corrupt it.
Framing: [uint32 big-endian header length][JSON header][raw array bytes...]. The header lists
each array's key path, dtype and shape. **No pickle in either direction.**

Agent server entry point (runs inside the sandbox):

    python -m doom_env.remote serve --socket /ipc/agent.sock --repo /submission [--device cuda]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import secrets
import socket
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import spaces as S

PROTOCOL_VERSION = 1
MAX_HEADER = 64 * 1024
MAX_PAYLOAD_TO_AGENT = 8 * 1024 * 1024
MAX_PAYLOAD_FROM_AGENT = 64 * 1024
MAX_ERROR_TEXT = 16 * 1024
ALLOWED_DTYPES = {"uint8", "int8", "int16", "int32", "int64", "uint16", "uint32", "float16", "float32", "float64", "bool"}


class ProtocolError(Exception):
    """Malformed or oversized message."""


class AgentFailure(Exception):
    """The agent failed. kind: 'timeout' | 'crash' (connection lost) | 'error' (exception in agent code)
    | 'invalid_action' | 'protocol'."""

    def __init__(self, kind: str, message: str, where: str = "", tb: str = ""):
        super().__init__(f"{kind} during {where or '?'}: {message}")
        self.kind, self.message, self.where, self.traceback = kind, message, where, tb


# --------------------------------------------------------------------------- framing
def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from _flatten(obj[k], f"{prefix}{k}/")
    else:
        yield prefix[:-1], np.ascontiguousarray(obj)


def _unflatten(items):
    out: dict = {}
    for path, arr in items:
        node = out
        parts = path.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ProtocolError(f"conflicting array paths at {path!r}")
        if parts[-1] in node:
            raise ProtocolError(f"duplicate or conflicting array path {path!r}")
        node[parts[-1]] = arr
    return out


def encode(msg: dict, arrays: dict | None = None) -> bytes:
    specs, blobs = [], []
    if arrays is not None:
        for path, arr in _flatten(arrays):
            if arr.dtype.name not in ALLOWED_DTYPES:
                raise ProtocolError(f"dtype {arr.dtype} not allowed")
            specs.append({"path": path, "dtype": arr.dtype.name, "shape": list(arr.shape)})
            blobs.append(arr.tobytes())
    header = json.dumps({**msg, "arrays": specs}).encode()
    return struct.pack(">I", len(header)) + header + b"".join(blobs)


def _recv_exact(sock: socket.socket, n: int, deadline: float | None) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            sock.settimeout(remaining)
        else:
            sock.settimeout(None)
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def recv_message(sock: socket.socket, deadline: float | None, max_payload: int) -> tuple[dict, dict]:
    """Read one message. Malformed input of any kind raises ProtocolError (never anything else),
    except for TimeoutError / ConnectionError / OSError from the socket itself."""
    try:
        return _recv_message(sock, deadline, max_payload)
    except (ProtocolError, TimeoutError, ConnectionError, OSError):
        raise
    except Exception as e:  # noqa: BLE001 - e.g. RecursionError/ValueError from hostile JSON
        raise ProtocolError(f"malformed message ({type(e).__name__})") from None


def _recv_message(sock: socket.socket, deadline: float | None, max_payload: int) -> tuple[dict, dict]:
    (hlen,) = struct.unpack(">I", _recv_exact(sock, 4, deadline))
    if hlen > MAX_HEADER:
        raise ProtocolError("header too large")
    try:
        header = json.loads(_recv_exact(sock, hlen, deadline))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad header: {e}") from None
    if not isinstance(header, dict) or not isinstance(header.get("type"), str):
        raise ProtocolError("header must be an object with a 'type'")
    specs = header.pop("arrays", [])
    if not isinstance(specs, list) or len(specs) > 64:
        raise ProtocolError("bad array list")
    sizes = []
    for spec in specs:
        if not isinstance(spec, dict) or spec.get("dtype") not in ALLOWED_DTYPES:
            raise ProtocolError("bad array spec")
        shape = spec.get("shape")
        if not isinstance(shape, list) or len(shape) > 8 or not all(isinstance(d, int) and 0 <= d <= 1 << 24 for d in shape):
            raise ProtocolError("bad array shape")
        if not isinstance(spec.get("path"), str) or len(spec["path"]) > 256:
            raise ProtocolError("bad array path")
        size = math.prod(shape) * np.dtype(spec["dtype"]).itemsize  # Python ints: no overflow
        if size > max_payload:
            raise ProtocolError("payload too large")
        sizes.append(size)
    if sum(sizes) > max_payload:
        raise ProtocolError("payload too large")
    items = []
    for spec, size in zip(specs, sizes):
        raw = _recv_exact(sock, size, deadline)
        items.append((spec["path"], np.frombuffer(raw, dtype=spec["dtype"]).reshape(spec["shape"]).copy()))
    return header, _unflatten(items)


# --------------------------------------------------------------------------- host side
# Unix sockets everywhere except native Windows Python, which lacks AF_UNIX. There, local tools
# (validate.py) fall back to loopback TCP. The scorer always runs on Linux with Unix sockets.
HAS_UNIX_SOCKETS = hasattr(socket, "AF_UNIX")


def listen(path: str | os.PathLike) -> socket.socket:
    """Listening socket for one agent. Pass `address(server)` to `serve --socket`."""
    if not HAS_UNIX_SOCKETS:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        return srv
    path = str(path)
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o666)  # the sandboxed agent may run as a different uid
    srv.listen(1)
    return srv


def address(server: socket.socket) -> str:
    """What the agent process connects to: a socket path, or tcp:127.0.0.1:<port>."""
    if server.family == socket.AF_INET:
        host, port = server.getsockname()[:2]
        return f"tcp:{host}:{port}"
    return str(server.getsockname())


def _connect(addr: str) -> socket.socket:
    if addr.startswith("tcp:"):
        host, _, port = addr[4:].rpartition(":")
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("agent sockets are local only")
        sock = socket.create_connection((host, int(port)), timeout=5)
        sock.settimeout(None)
        return sock
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(addr)
    except OSError:
        sock.close()
        raise
    return sock


def _nodelay(sock: socket.socket) -> None:
    if sock.family == socket.AF_INET:  # no Nagle delays on small replies
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def validate_action(action, action_space=None) -> np.ndarray:
    """Return the action as int8 (9,) or raise AgentFailure('invalid_action')."""
    shape = (len(S.BUTTONS),) if action_space is None else action_space.shape
    a = np.asarray(action) if not isinstance(action, np.ndarray) else action
    if a.dtype.kind not in "biuf" or a.shape != shape:
        raise AgentFailure("invalid_action", f"expected shape {shape} of 0/1 values, got dtype {a.dtype} shape {a.shape}", "act")
    if not np.all((a == 0) | (a == 1)):
        raise AgentFailure("invalid_action", "action values must be 0 or 1", "act")
    return a.astype(np.int8)


class AgentConnection:
    """Host-side handle to a connected agent server. All calls take a timeout in seconds."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.seq: str | None = None

    @classmethod
    def accept(cls, server: socket.socket, timeout: float, alive=lambda: True) -> "AgentConnection":
        """Wait for the agent server to connect. `alive()` is polled so a crashed process fails fast."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentFailure("timeout", f"agent process did not connect within {timeout:.0f}s", "startup")
            server.settimeout(min(0.2, remaining))
            try:
                conn, _ = server.accept()
                conn.settimeout(None)
                _nodelay(conn)
                return cls(conn)
            except socket.timeout:
                if not alive():
                    raise AgentFailure("crash", "agent process exited before connecting", "startup") from None

    def _call(self, where: str, msg: dict, arrays: dict | None, timeout: float) -> tuple[dict, dict]:
        deadline = time.monotonic() + timeout
        # Unpredictable per-request nonce: a reply can't be sent before its request arrives.
        self.seq = secrets.token_hex(8)
        try:
            self.sock.settimeout(timeout)
            self.sock.sendall(encode({**msg, "seq": self.seq}, arrays))
            header, data = recv_message(self.sock, deadline, MAX_PAYLOAD_FROM_AGENT)
        except (TimeoutError, socket.timeout):
            raise AgentFailure("timeout", f"no reply within {timeout:.1f}s", where) from None
        except ProtocolError as e:
            raise AgentFailure("protocol", str(e), where) from None
        except (ConnectionError, OSError) as e:
            raise AgentFailure("crash", f"agent process connection lost ({e.__class__.__name__})", where) from None
        except Exception as e:  # noqa: BLE001 - never let agent input crash the host
            raise AgentFailure("protocol", f"malformed reply ({type(e).__name__})", where) from None
        if header.get("seq") != self.seq:
            # Replies must answer the request just sent (no pre-sent or stale replies).
            raise AgentFailure("protocol", "reply does not match the request (sequence mismatch)", where)
        if header["type"] == "error":
            raise AgentFailure(
                "error",
                str(header.get("error", ""))[:MAX_ERROR_TEXT],
                where,
                str(header.get("traceback", ""))[:MAX_ERROR_TEXT],
            )
        return header, data

    def init(self, device: str, timeout: float) -> dict:
        from . import CONTRACT_VERSION

        header, _ = self._call("init", {"type": "init", "device": device, "protocol": PROTOCOL_VERSION}, None, timeout)
        if header.get("type") != "ready":
            raise AgentFailure("protocol", f"expected 'ready', got {header.get('type')!r}", "init")
        if header.get("contract_version") != CONTRACT_VERSION:
            raise AgentFailure(
                "protocol",
                f"contract version mismatch: agent side {header.get('contract_version')!r}, host {CONTRACT_VERSION!r}",
                "init",
            )
        return header

    def reset(self, timeout: float) -> None:
        header, _ = self._call("reset", {"type": "reset"}, None, timeout)
        if header.get("type") != "ok":
            raise AgentFailure("protocol", f"expected 'ok', got {header.get('type')!r}", "reset")

    def act(self, obs: dict, timeout: float) -> np.ndarray:
        header, data = self._call("act", {"type": "act"}, obs, timeout)
        if header.get("type") != "action" or "action" not in data:
            raise AgentFailure("protocol", "expected an 'action' message", "act")
        return validate_action(data["action"])

    def close(self) -> None:
        try:
            self.sock.settimeout(1.0)
            self.sock.sendall(encode({"type": "close"}))
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------- agent side
def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _cap_torch_vram() -> None:
    """Enforce the per-agent VRAM budget for PyTorch (JAX is capped via XLA_PYTHON_CLIENT_MEM_FRACTION).
    Applied once DoomAgent.__init__ has returned; allocations beyond the budget then raise CUDA OOM."""
    torch = sys.modules.get("torch")
    try:
        if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        from .limits import MAX_VRAM_MB

        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, MAX_VRAM_MB * 1024**2 / total), i)
    except Exception:  # noqa: BLE001 - never fail an agent because of the cap itself
        pass


def _load_agent_class(repo: Path):
    path = repo / "submission" / "agent.py"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    for p in (str(repo / "submission"), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("submission_agent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["submission_agent"] = module
    spec.loader.exec_module(module)
    cls = getattr(module, "DoomAgent", None)
    if cls is None:
        raise AttributeError("submission/agent.py does not define DoomAgent")
    return cls


def _error(where: str, exc: BaseException) -> tuple[dict, None]:
    tb = traceback.format_exc()
    sys.stderr.write(tb)
    sys.stderr.flush()
    return {"type": "error", "where": where, "error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_TEXT], "traceback": tb[-MAX_ERROR_TEXT:]}, None


def serve(socket_path: str, repo: str, device: str = "auto") -> int:
    from . import CONTRACT_VERSION
    from .limits import THREAD_ENV

    # One CPU per agent: libraries loaded from here on (torch, jax, ...) start a single compute
    # thread. The scorer and validate.py also set these before this process starts (for numpy).
    os.environ.update(THREAD_ENV)

    for _ in range(100):
        try:
            sock = _connect(socket_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.05)
    else:
        print(f"could not connect to {socket_path}", file=sys.stderr)
        return 2
    _nodelay(sock)

    agent = None
    repo_path = Path(repo).resolve()
    while True:
        try:
            header, data = recv_message(sock, None, MAX_PAYLOAD_TO_AGENT)
        except (ConnectionError, OSError):
            return 0
        kind = header["type"]
        if kind == "close":
            return 0
        if kind == "init":
            try:
                t0 = time.perf_counter()
                cls = _load_agent_class(repo_path)
                os.chdir(repo_path)
                dev = _resolve_device(str(header.get("device", "auto")))
                agent = cls(S.observation_space(), S.action_space(), dev)
                _cap_torch_vram()
                reply = ({"type": "ready", "contract_version": CONTRACT_VERSION, "device": dev,
                          "init_seconds": time.perf_counter() - t0}, None)
            except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit
                reply = _error("init", e)
        elif agent is None:
            reply = ({"type": "error", "where": kind, "error": "agent not initialised"}, None)
        elif kind == "reset":
            try:
                agent.reset()
                reply = ({"type": "ok"}, None)
            except BaseException as e:  # noqa: BLE001
                reply = _error("reset", e)
        elif kind == "act":
            try:
                action = np.asarray(agent.act(data))
                if action.dtype.kind not in "biuf":
                    raise TypeError(f"act() returned dtype {action.dtype}; expected 0/1 integers")
                if action.dtype.name not in ALLOWED_DTYPES:
                    action = action.astype(np.float64)
                reply = ({"type": "action"}, {"action": action})
            except BaseException as e:  # noqa: BLE001
                reply = _error("act", e)
        else:
            reply = ({"type": "error", "where": kind, "error": f"unknown message {kind!r}"}, None)
        try:
            sock.sendall(encode({**reply[0], "seq": header.get("seq")}, reply[1]))
        except OSError:
            return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m doom_env.remote")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve", help="run the agent server (inside the sandbox)")
    p.add_argument("--socket", required=True)
    p.add_argument("--repo", required=True, help="team repository root (contains submission/agent.py)")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    args = parser.parse_args(argv)
    return serve(args.socket, args.repo, args.device)


if __name__ == "__main__":
    sys.exit(main())
