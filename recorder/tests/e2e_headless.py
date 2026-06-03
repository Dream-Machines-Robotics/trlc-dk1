#!/usr/bin/env python3
"""Headless end-to-end test of the pub/sub recorder core (no motors).

Spawns the real C++ camera daemon on 2 cameras, drives a synthetic state ring +
episode events from Python, runs the Rust recorder against them, and asserts the
.rrd + manifest (episode kept w/ grade, discarded episode absent). Run with the
lerobot venv python from the repo root.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/dominique/dreammachines/ledream")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks/pubsub_recorder/bench"))  # _shmcam
from lerobot.recording.state_publisher import StateRingWriter  # noqa: E402
from lerobot.recording.events import EventSender  # noqa: E402
from _shmcam import spawn_daemon  # noqa: E402

DAEMON = str(REPO / "scripts/lerobot_cam_capture")
RECORDER = str(REPO / "trlc-dk1/recorder/target/release/dm_recorder")
CAMS = {
    "top": "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:2:1.0-video-index0",
    "wrist": "/dev/v4l/by-path/pci-0000:0d:00.0-usb-0:3.3:1.0-video-index0",
}
FPS = 50
DIM = 14


def mono_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def main():
    out_rrd, out_manifest, sock = "/tmp/e2e.rrd", "/tmp/e2e.json", "/tmp/dm_ev_test.sock"
    for p in (out_rrd, out_manifest, "/dev/shm/dm_state_test"):
        Path(p).unlink(missing_ok=True)

    # 1. spawn camera daemons (fixed shm names)
    daemons = {n: spawn_daemon(DAEMON, dev, 1280, 720, FPS, f"/dm_cam_{n}") for n, dev in CAMS.items()}
    time.sleep(1.5)

    # 2. state ring writer
    writer = StateRingWriter("/dm_state_test", state_dim=DIM, action_dim=DIM, ring_size=256)

    # 3. spawn recorder
    cam_args = []
    for n in CAMS:
        cam_args += ["--cam", f"name={n},shm=/dm_cam_{n}"]
    rec = subprocess.Popen(
        [RECORDER, *cam_args, "--state-shm", "/dm_state_test", "--event-socket", sock,
         "--fps", str(FPS), "--out", out_rrd, "--manifest", out_manifest, "--task", "e2e test"],
        stderr=subprocess.PIPE, text=True)
    time.sleep(0.8)  # let it attach + bind socket
    ev = EventSender(sock)

    # 4. drive: episode 0 (~2s) -> commit grade 4; episode 1 (~2s) -> discard; session_end
    def drive_episode(ep, dur_s):
        n = int(dur_s * FPS)
        period = 1.0 / FPS
        nxt = time.perf_counter()
        for i in range(n):
            st = np.sin(np.arange(DIM) * 0.3 + i * 0.05).astype(np.float32)
            act = (st + 0.01).astype(np.float32)
            writer.write(mono_ns(), ep, st, act)
            nxt += period
            sl = nxt - time.perf_counter()
            if sl > 0:
                time.sleep(sl)

    drive_episode(0, 2.0)
    ev.commit(0, 4.0)
    drive_episode(1, 2.0)
    ev.discard(1)
    time.sleep(0.2)
    ev.session_end()

    # 5. wait for recorder to finalize
    try:
        rec.wait(timeout=8)
    except subprocess.TimeoutExpired:
        rec.terminate()
        rec.wait(timeout=3)
    print("--- recorder stderr ---")
    print(rec.stderr.read())

    writer.close()
    for d in daemons.values():
        d.terminate()

    # 6. assertions
    man = json.loads(Path(out_manifest).read_text())
    eps = {e["episode_index"]: e for e in man["episodes"]}
    print("manifest episodes:", man["episodes"])
    assert 0 in eps, "episode 0 missing"
    assert 1 not in eps, "episode 1 should have been discarded"
    assert eps[0]["quality"] == 4.0, f"episode 0 grade wrong: {eps[0]}"
    assert 90 <= eps[0]["num_frames"] <= 110, f"ep0 frames {eps[0]['num_frames']} not ~100"
    assert man["cameras"] and len(man["cameras"]) == 2, "expected 2 cameras"
    assert Path(out_rrd).stat().st_size > 100_000, "rrd too small"
    print("\n✓ E2E PASS: ep0 kept (grade 4, %d frames, 2 cams), ep1 discarded, rrd=%.1f MB"
          % (eps[0]["num_frames"], Path(out_rrd).stat().st_size / 1e6))


if __name__ == "__main__":
    main()
