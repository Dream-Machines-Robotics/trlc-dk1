"""Detailed teleop logs (lerobot_robot_trlc_dk1/telemetry.py): when they are on, the
chunk files, and the follower drain against a fake RT ring. No hardware. The RT
ring itself is tested in csrc/tests/test_telemetry.cpp."""

import json
import os
import time

import numpy as np
import pytest

from lerobot_robot_trlc_dk1 import telemetry as tm

# A small stand-in for the native slot layout (name, offset, kind, itemsize, count).
FIELDS = [
    ("t_ns", 0, "u", 8, 1),
    ("loop_count", 8, "u", 8, 1),
    ("pos", 16, "f", 4, 7),
    ("err", 44, "u", 1, 7),
]
ITEMSIZE = 56
DTYPE = tm.dtype_from_fields(FIELDS, ITEMSIZE)


def test_on_needs_the_toggle_and_the_hub_folder(tmp_path):
    on = {"DM_DEV_TOGGLES": f"strong-grip,{tm.TOGGLE}", tm.DIR_ENV: str(tmp_path)}
    assert tm.log_dir(on) == tmp_path
    # A recording or rollout gets no folder from the hub: off, whatever its toggles.
    assert tm.log_dir({"DM_DEV_TOGGLES": tm.TOGGLE}) is None
    assert tm.log_dir({**on, tm.DIR_ENV: ""}) is None
    # The folder without the toggle: off.
    assert tm.log_dir({"DM_DEV_TOGGLES": "strong-grip", tm.DIR_ENV: str(tmp_path)}) is None
    # A folder that is not there: off, with a warning, never an error.
    assert tm.log_dir({**on, tm.DIR_ENV: str(tmp_path / "gone")}) is None
    assert tm.log_dir({}) is None


def test_dtype_from_fields_matches_the_layout():
    assert DTYPE.itemsize == ITEMSIZE
    assert DTYPE.fields["pos"][0].shape == (7,)
    assert DTYPE.fields["err"][1] == 44


def _rows(start, n):
    a = np.zeros(n, dtype=DTYPE)
    a["loop_count"] = np.arange(start, start + n)
    a["t_ns"] = a["loop_count"] * 4_000_000
    a["pos"][:, 0] = a["loop_count"] / 1000.0
    a["err"][:] = 1
    return a


def test_chunk_writer_writes_numbered_chunks(tmp_path):
    w = tm.ChunkWriter(tmp_path, "follower_left", DTYPE, rows_per_chunk=100)
    w.append(_rows(0, 60))
    assert not list(tmp_path.glob("*.npy"))  # still buffered
    w.append(_rows(60, 60))  # 120 >= 100: one chunk
    w.append(_rows(120, 10))
    w.flush()
    files = sorted(tmp_path.glob("follower_left.*.npy"))
    assert [f.name for f in files] == [
        f"follower_left.{os.getpid()}.000000.npy",
        f"follower_left.{os.getpid()}.000001.npy",
    ]
    data = np.concatenate([np.load(f, allow_pickle=False) for f in files])
    # Saved packed (no padding), same fields.
    assert data.dtype.names == DTYPE.names and data.dtype.fields["pos"][0].shape == (7,)
    assert list(data["loop_count"]) == list(range(130))
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        w.append(np.zeros(3, dtype=np.float32))


class FakeRing:
    """read_telemetry() like DK1RobotRT's: raw bytes, the next index, rows lost."""

    def __init__(self):
        self.rows = _rows(0, 0)
        self.lose_next = 0

    def add(self, n):
        # Field by field: np.concatenate would pack the padded dtype.
        old, new = self.rows, _rows(len(self.rows), n)
        self.rows = np.zeros(len(old) + n, dtype=DTYPE)
        for name in DTYPE.names:
            self.rows[name][: len(old)] = old[name]
            self.rows[name][len(old) :] = new[name]

    def telemetry_enabled(self):
        return True

    def read_telemetry(self, from_idx, max_count):
        start = from_idx + self.lose_next
        dropped, self.lose_next = self.lose_next, 0
        part = self.rows[start : start + max_count]
        raw = np.frombuffer(part.tobytes(), dtype=np.uint8).reshape(len(part), ITEMSIZE)
        return raw, start + len(part), dropped


def test_follower_drain_keeps_every_row_and_counts_losses(tmp_path):
    ring = FakeRing()
    t = tm.FollowerTelemetry(ring, tmp_path, "follower_right", DTYPE, period_s=60.0, rows_per_chunk=1000)
    ring.add(5000)  # more than one read (4096) at once
    assert t.drain() == 5000
    ring.add(10)
    ring.lose_next = 3  # the writer lapped the reader by 3 rows
    assert t.drain() == 7
    t.close()  # never started: close still takes the rest and writes all
    data = np.concatenate([np.load(f) for f in sorted(tmp_path.glob("follower_right.*.npy"))])
    assert len(data) == 5007
    assert list(data["loop_count"][:5000]) == list(range(5000))
    assert list(data["loop_count"][5000:]) == list(range(5003, 5010))
    summary = json.loads((tmp_path / f"follower_right.{os.getpid()}.summary.json").read_text())
    assert summary["rows"] == 5007 and summary["dropped"] == 3 and summary["error"] is None


def test_follower_thread_drains_while_running(tmp_path):
    ring = FakeRing()
    t = tm.FollowerTelemetry(ring, tmp_path, "follower", DTYPE, period_s=0.01)
    t.start()
    ring.add(50)
    for _ in range(200):
        if t._writer.rows >= 50:
            break
        time.sleep(0.01)
    t.close()
    assert t._writer.rows == 50


def test_an_error_stops_the_log_not_the_arm(tmp_path):
    class Broken(FakeRing):
        def read_telemetry(self, from_idx, max_count):
            raise RuntimeError("ring gone")

    t = tm.FollowerTelemetry(Broken(), tmp_path, "follower", DTYPE, period_s=0.01)
    t.start()
    t.join(2.0)
    assert not t.is_alive() and "ring gone" in t.error
    t.close()  # no raise
    summary = json.loads((tmp_path / f"follower.{os.getpid()}.summary.json").read_text())
    assert "ring gone" in summary["error"]


def test_start_follower_writes_its_settings(tmp_path, monkeypatch):
    ring = FakeRing()
    monkeypatch.setattr(tm, "slot_dtype", lambda: DTYPE)
    t = tm.start_follower(ring, tmp_path, "left")
    assert t is not None and t.stream == "follower_left"
    t.close()
    meta = json.loads((tmp_path / f"follower_left.{os.getpid()}.meta.json").read_text())
    assert meta["label"] == "left" and meta["started_monotonic_ns"] > 0
    # A loop without telemetry (an old native build): no log, no error.
    ring.telemetry_enabled = lambda: False
    assert tm.start_follower(ring, tmp_path, "right") is None


def test_follower_stream_names():
    assert tm.follower_stream("left") == "follower_left"
    assert tm.follower_stream("right") == "follower_right"
    assert tm.follower_stream("") == "follower"


def test_native_layout_when_built():
    native = pytest.importorskip("trlc_dk1_control._trlc_dk1_rt")
    if not hasattr(native, "telemetry_fields"):
        pytest.skip("native extension built before telemetry")
    dt = tm.slot_dtype()
    assert dt.itemsize == native.telemetry_slot_size() == 288
    assert dt.fields["t_mos"][0].shape == (7,) and dt.fields["q_sent"][0].shape == (6,)
    cfg = native.RtLoopConfig()
    assert cfg.telemetry is False  # off by default
    loop = native.RtControlLoop(cfg)  # not started: no port opened
    assert not loop.telemetry_enabled()
    raw, nxt, dropped = loop.read_telemetry(5, 16)
    assert raw.shape == (0, 288) and nxt == 5 and dropped == 0
