# dm_recorder — out-of-process pub/sub recorder

The control loop (`lerobot-record`) only **publishes**; this separate Rust process
**consumes** and writes the `.rrd`, so the operator never feels recorder work
(NVENC cold-start, disk, `.rrd` flush) and a recorder crash leaves the arms
untouched. `.rrd` is viewable in the Rerun viewer and converted to LeRobot v3 by
`src/lerobot/recording/rrd_to_lerobot.py` (`make export`). Design + rationale:
`~/.claude/plans/starry-sleeping-map.md`.

## Data planes (all CLOCK_MONOTONIC)
| plane | producer | transport |
|---|---|---|
| camera frames | C++ daemon (existing) | POSIX shm rings — recorder attaches read-only by name |
| state + action + ref_ts + episode_index | **`StatePublisher` thread @ the control rate (~250 Hz)** | **state shm ring** `/dm_state_*` (`src/lerobot/recording/state_publisher.py` ↔ `src/ring.rs::StateRingReader`) |
| episode events (discard / commit grade / session_end) | control loop SM callbacks | Unix datagram socket (`events.py` → recorder) |

**Full resolution (motor data at the control rate, not the camera rate):** a
`StatePublisher` thread samples proprioception (`bi_follower.get_proprioception`
→ the RT loop's 250 Hz state ring) + the teleop snapshot (the canonical
`act_processed`) at ~250 Hz and writes the state ring — so the `.rrd` carries
motor data at full resolution. It uses the SAME representation as the 50 Hz
record path, so the data stays consistent with inference. (We benchmarked a
Python publisher vs the C++ RT loop writing the ring directly — equal rate
fidelity + zero teleop impact, but Python keeps the canonical *action*
representation and never touches the safety loop; see `benchmarks/pubsub_recorder/bench_state_publisher.py`.)

The recorder logs each **camera at its own kernel capture timestamp, deduped**
(skip when the at-or-before frame hasn't advanced) so 250 Hz state doesn't 5×
the JPEGs; state/action are per-motor **named scalar** timelines (unfold
`observation/state` in the viewer). Episodes are segmented by the slot's
`episode_index` (a **monotonic attempt index** — see below).

**Export is camera-anchored + `align=1`:** `rrd_to_lerobot.py` reads the
`real_time` timeline and emits one v3 frame per reference-camera frame, sampling
joints/action at-or-before that frame's capture ts (deterministic, reproducible,
matching `get_observation` under `LEROBOT_OBS_ALIGN=1`). The alignment mode is
stamped into `info.json` as `obs_align` so eval can match a policy's training
alignment. Because the `.rrd` stores the raw full-res streams, the alignment is a
re-exportable parameter (re-export with a different rule, no re-recording).

## Build / test
```bash
make build-recorder                 # cargo build --release (system libturbojpeg)
# headless end-to-end (real cameras, no motors): record -> .rrd + manifest
.venv/bin/python trlc-dk1/recorder/tests/e2e_headless.py
# export a session -> LeRobot v3 (auto-creates a rerun-0.33 venv)
make export RRD=/tmp/e2e.rrd MANIFEST=/tmp/e2e.json DATASET_REPO_ID=you/name NEW_W=320 NEW_H=240
```

## Status
**Validated end-to-end, headless, on real cameras** (`tests/e2e_headless.py`):
2-camera shm attach, state-ring round-trip (0 drops), timestamp alignment,
episode segmentation, grade + discard, `.rrd` write, and `make export` → v3 that
**loads in the real `LeRobotDataset`** and decodes both camera videos (discard
honored: kept episodes only).

Built + in place:
- `src/{ring.rs,tj.rs,main.rs}`, `Cargo.toml`, `build.rs` — the recorder.
- `src/lerobot/recording/{state_publisher,events,recorder_proc,rrd_to_lerobot}.py`.
- `src/lerobot/cameras/cpp/camera_cpp.py` → `CppCamera.shm_name` accessor.
- `src/lerobot/scripts/lerobot_record.py` → `record_backend` config field +
  `record_loop_decoupled(pubsub_recorder=…)` hook + `_flatten_numeric` (inline path unchanged).
- `Makefile` → `build-recorder`, `export`.

## `main()` wiring — DONE (pending rig test)
`make record-pubsub` is fully wired in `lerobot_record.py` (compile- + config- +
Makefile-verified; the **inline path is untouched**, regression-checked). What's in:
- `record_backend` config field; `make record-pubsub` (= `record RECORD_BACKEND=pubsub`).
- `LeRobotDataset.create/resume` gated to inline; in pubsub, after `robot.connect()` +
  teleop start, a `PubSubRecorder` is built (cameras = `[(name, cam.shm_name) …]`, dims
  probed via `_flatten_numeric` of one obs + a teleop snapshot) and `dataset` becomes a
  `_PubSubDatasetShim` routing save/discard/finalize to it.
- **Episode indexing:** the shim uses a **monotonic attempt counter** as `episode_index`
  (discarded attempts don't collide); the exporter re-contiguates kept attempts → v3 0..N.
- `_countdown_prep` warm-up + `VideoEncodingManager` skipped for pubsub.
- `_finalize_episode`: realized-fps gate kept (loop still fills `episode_timing_log`);
  on save → `recorder.commit(attempt, grade)`; discard/fps-reject → `recorder.discard`.
- Crash policy: `recorder.crashed()` in the loop → TTS + stop accepting new episodes.
- Session end → shim `finalize()` → `recorder.shutdown_and_finalize()`.

One `.rrd` per episode (`episode_<i>.rrd`) + the manifest are written to
`<dataset.root>/_rrd/` (`session.json`); convert with
`make export RRD=<…>/_rrd DATASET_REPO_ID=… NEW_W= NEW_H=`. View per-episode with
`make view RRD=<…>/_rrd` (Rerun's Recordings panel is the episode selector).

The manifest is rewritten **incrementally** (atomic tmp+rename) on every episode
rollover and on every grade/discard event, not only at exit — the pedal grades
live nowhere else, so a recorder that dies without a final write (SIGKILL after
a stuck flush, panic, power) loses at most the in-flight episode's entry, never
the whole session's grades. (A 243-episode session lost all its grades this way
on 2026-06-29, before incremental writes.) The exit pass folds in the final
episode + any grade racing `session_end` and deletes a last-episode discard.

### Rig test (the remaining validation)
`make build-recorder && make record-pubsub ARM=both` → record 3 episodes (commit /
discard / grade), Q out. Confirm: arm feel hitch-free at episode start vs `make
record` (no frame-1 spike); `make view RRD=<…>/_rrd` shows one clean recording
per episode (no cross-episode bleed, cameras load); `make export …` → v3 loads
+ dashboard shows kept episodes w/ grade. Drills: `kill` the recorder mid-episode
(arms keep following, banner fires); disk-stall (operator feels nothing).

## Known v1 limits
- State/action published as flattened numeric vectors (semantic feature names not
  preserved end-to-end yet; the exporter writes `observation.state`/`action` with
  `names: null`). Fine for training; add names via the manifest later.
- `--append`/RESUME not implemented (manual export per session for now).
- Export needs an isolated rerun-0.33 venv (`.rrd` is N-reads-N-1; training never
  imports rerun). `make export` auto-creates it at `~/.cache/dm-export-venv`.
