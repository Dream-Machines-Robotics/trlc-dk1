"""Detailed teleop logs: the follower's per-cycle RT telemetry, written to disk.

Behind the session dev toggle ``teleop-telemetry`` (dreamhub/dev-toggles.json),
and only where the hub asked for it: the hub sets ``DM_TELEOP_LOG_DIR`` for a
quick teleop session (`make teleop`) whose starter has the toggle on. Both must
be set, so recordings and rollouts never log, whatever toggles they carry.

When on, each follower arm's RT loop keeps one telemetry slot per control cycle
(csrc/telemetry.h, 250 Hz) and a :class:`FollowerTelemetry` thread drains it ten
times a second into numbered ``.npy`` chunks. Plain numpy on purpose: the VR
teleop process runs without pyarrow, so the hub converts the chunks to Parquet
when the session ends (dreamhub/backend/dm_hub/teleop_telemetry.py). A chunk is
written to a temporary name and renamed, so a crash loses at most the rows of
the chunk being filled (about 10 s).

When off, nothing here runs: no ring in the RT loop, no thread, no file.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

TOGGLE = "teleop-telemetry"
DIR_ENV = "DM_TELEOP_LOG_DIR"
# About 10 s at 250 Hz per chunk: what a crash can lose at most.
ROWS_PER_CHUNK = 2500
DRAIN_PERIOD_S = 0.1


def toggle_on(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return TOGGLE in {n.strip() for n in env.get("DM_DEV_TOGGLES", "").split(",") if n.strip()}


def log_dir(environ: dict[str, str] | None = None) -> Path | None:
    """The session's log folder when detailed logging is on for this process, else None.

    On needs the toggle in ``DM_DEV_TOGGLES`` and an existing folder in
    ``DM_TELEOP_LOG_DIR`` (the hub makes it). Never raises."""
    env = os.environ if environ is None else environ
    raw = env.get(DIR_ENV, "").strip()
    if not raw or not toggle_on(env):
        return None
    path = Path(raw)
    if not path.is_dir():
        logger.warning(
            "dev toggle %s is on, but %s=%s is not a folder: no detailed log", TOGGLE, DIR_ENV, raw
        )
        return None
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON through a temporary file, so a reader never sees half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default))
    os.replace(tmp, path)


class ChunkWriter:
    """Rows of one stream as ``<dir>/<stream>.<pid>.<seq>.npy`` chunks of a fixed dtype.

    The pid keeps two processes of one session apart (the teleop, then the park).
    Not thread-safe: one writer thread per stream."""

    def __init__(self, directory: Path, stream: str, dtype: np.dtype, rows_per_chunk: int = ROWS_PER_CHUNK):
        self.directory = Path(directory)
        self.stream = stream
        self.dtype = np.dtype(dtype)
        self.rows_per_chunk = max(1, int(rows_per_chunk))
        self._pid = os.getpid()
        self._seq = 0
        self._parts: list[np.ndarray] = []
        self._buffered = 0
        self.rows = 0

    def append(self, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        if rows.dtype != self.dtype:
            raise ValueError(f"{self.stream}: rows of dtype {rows.dtype}, expected {self.dtype}")
        self._parts.append(np.array(rows, copy=True))
        self._buffered += rows.size
        self.rows += rows.size
        if self._buffered >= self.rows_per_chunk:
            self.flush()

    def flush(self) -> None:
        if not self._buffered:
            return
        data = np.concatenate(self._parts) if len(self._parts) > 1 else self._parts[0]
        self._parts, self._buffered = [], 0
        name = f"{self.stream}.{self._pid}.{self._seq:06d}.npy"
        self._seq += 1
        tmp = self.directory / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, data, allow_pickle=False)
        os.replace(tmp, self.directory / name)


def slot_dtype() -> np.dtype:
    """The numpy dtype of one RT telemetry slot, built from the native layout table."""
    from trlc_dk1_control._trlc_dk1_rt import telemetry_fields, telemetry_slot_size

    return dtype_from_fields(telemetry_fields(), telemetry_slot_size())


def dtype_from_fields(fields, itemsize: int) -> np.dtype:
    names, formats, offsets = [], [], []
    for name, offset, kind, size, count in fields:
        base = np.dtype(f"<{kind}{size}")
        names.append(name)
        formats.append(base if count == 1 else (base, (count,)))
        offsets.append(offset)
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": itemsize})


class FollowerTelemetry(threading.Thread):
    """Drains one arm's RT telemetry ring into chunks until :meth:`close`.

    ``rt_robot`` needs ``read_telemetry(from_idx, max_count)``. It must never hurt
    the arm: any error stops the logging, says so once, and leaves the arm alone."""

    def __init__(
        self,
        rt_robot,
        directory: Path,
        stream: str,
        dtype: np.dtype,
        period_s: float = DRAIN_PERIOD_S,
        rows_per_chunk: int = ROWS_PER_CHUNK,
    ):
        super().__init__(name=f"telemetry-{stream}", daemon=True)
        self._robot = rt_robot
        self._dtype = np.dtype(dtype)
        self._writer = ChunkWriter(directory, stream, self._dtype, rows_per_chunk)
        self._period_s = period_s
        self._halt = threading.Event()
        self._next = 0
        self.dropped = 0
        self.error: str | None = None
        self.directory = Path(directory)
        self.stream = stream

    def drain(self) -> int:
        """Copy whatever is new; returns the rows taken."""
        taken = 0
        while True:
            raw, nxt, dropped = self._robot.read_telemetry(self._next, 4096)
            self.dropped += int(dropped)
            self._next = int(nxt)
            n = int(raw.shape[0]) if raw.ndim == 2 else 0
            if n == 0:
                return taken
            if raw.shape[1] != self._dtype.itemsize:
                raise ValueError(f"telemetry slot is {raw.shape[1]} bytes, expected {self._dtype.itemsize}")
            self._writer.append(np.ascontiguousarray(raw).view(self._dtype).reshape(n))
            taken += n
            if n < 4096:
                return taken

    def run(self) -> None:
        while not self._halt.wait(self._period_s):
            if not self._step():
                return

    def _step(self) -> bool:
        try:
            self.drain()
            return True
        except Exception as e:  # noqa: BLE001 - logging must never touch the arm
            self.error = f"{type(e).__name__}: {e}"
            logger.error("detailed teleop log %s stopped: %s", self.stream, self.error)
            return False

    def close(self, timeout: float = 2.0) -> None:
        """Stop, take the last rows, write the open chunk and the stream summary."""
        self._halt.set()
        if self.is_alive():
            self.join(timeout)
        if self.error is None:
            self._step()
        try:
            self._writer.flush()
        except Exception as e:  # noqa: BLE001
            self.error = self.error or f"{type(e).__name__}: {e}"
        try:
            write_json(
                self.directory / f"{self.stream}.{os.getpid()}.summary.json",
                {
                    "stream": self.stream,
                    "rows": self._writer.rows,
                    "dropped": self.dropped,
                    "error": self.error,
                    "closed_unix_ns": time.time_ns(),
                },
            )
        except OSError as e:
            logger.error("detailed teleop log %s: could not write its summary: %s", self.stream, e)


# RtLoopConfig fields worth keeping with the data (constant for the session).
_NATIVE_KEYS = (
    "default_kp",
    "default_kd",
    "joint_pos_limits",
    "joint_torque_limits",
    "joint_velocity_limits",
    "limit_buffer",
    "max_pos_delta_per_cycle",
    "max_accel_per_cycle2",
    "gravity_comp_scale",
    "model_path",
    "command_timeout_s",
    "overcurrent_threshold",
    "overspeed_threshold",
    "gripper_open_pos",
    "gripper_closed_pos",
    "max_gripper_torque_nm",
    "strong_gripper_torque_nm",
    "torque_constant",
    "loop_hz",
    "serial_port",
    "rt_priority",
    "rt_cpu_affinity",
    "telemetry_ring_size",
)


def _plain(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    try:
        return np.asarray(v).tolist()
    except Exception:  # noqa: BLE001
        return str(v)


def follower_stream(label: str) -> str:
    """``follower_left`` / ``follower_right``, or ``follower`` for a single arm."""
    return f"follower_{label}" if label in ("left", "right") else "follower"


def start_follower(rt_robot, directory: Path, label: str, rt_config: Any = None) -> FollowerTelemetry | None:
    """Start draining one arm, after writing its settings next to the data. Returns
    None (and logs why) when it cannot start; the arm runs on either way."""
    stream = follower_stream(label)
    try:
        if not rt_robot.telemetry_enabled():
            logger.warning("detailed teleop log %s: the RT loop has no telemetry (old native build?)", stream)
            return None
        meta: dict[str, Any] = {
            "stream": stream,
            "label": label,
            "rate_hz": None,
            "pid": os.getpid(),
            "started_unix_ns": time.time_ns(),
            "started_monotonic_ns": time.monotonic_ns(),
        }
        if rt_config is not None and dataclasses.is_dataclass(rt_config):
            meta["rt_config"] = dataclasses.asdict(rt_config)
            meta["rate_hz"] = getattr(rt_config, "motor_thread_hz", None)
        native = getattr(rt_robot, "_rt_cfg", None)
        if native is not None:
            # What the native loop really got (after joint_velocity_scaling etc.).
            meta["rt_loop"] = {k: _plain(getattr(native, k)) for k in _NATIVE_KEYS if hasattr(native, k)}
        loop = getattr(rt_robot, "_loop", None)
        if loop is not None and hasattr(loop, "is_rt_active"):
            meta["rt_active"] = bool(loop.is_rt_active())
        write_json(Path(directory) / f"{stream}.{os.getpid()}.meta.json", meta)
        t = FollowerTelemetry(rt_robot, Path(directory), stream, slot_dtype())
        t.start()
        logger.info("detailed teleop log: %s at every RT cycle into %s", stream, directory)
        return t
    except Exception as e:  # noqa: BLE001 - never stop the arm for a log
        logger.error("detailed teleop log %s could not start: %s", stream, e)
        return None
