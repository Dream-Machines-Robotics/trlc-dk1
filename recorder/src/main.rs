//! dm_recorder — production out-of-process recorder for `make record-pubsub`.
//!
//! Subscribes to data planes the control process publishes and writes one
//! `.rrd` (Rerun 0.33) PER EPISODE into an output dir + a `session.json`
//! manifest, fully isolated from the control loop (so encoder cold-start / disk /
//! flush never hitch the operator, and a recorder crash leaves the arms
//! untouched). One recording per episode is Rerun's intended pattern for
//! episodic data — the viewer scopes its timeline/charts to a single take:
//!   * camera frames  — existing C++ daemon POSIX shm rings (attach by name, read-only)
//!   * state/action   — the Python state ring (`/dm_state_*`); each slot carries the
//!                      reference timestamp + the episode index it belongs to
//!   * episode events — a Unix datagram socket: `discard <ep>`, `commit <ep> <grade>`,
//!                      `session_end`
//!
//! The `.rrd` is converted to LeRobot v3 later by `rrd_to_lerobot.py` (`make export`).

mod ring;
mod tj;
use ring::{CameraRingReader, StateRingReader};
use tj::TjCompressor;

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::os::unix::net::UnixDatagram;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static SESSION_END: AtomicBool = AtomicBool::new(false);
extern "C" fn on_term(_: libc::c_int) {
    SESSION_END.store(true, Ordering::SeqCst);
}

struct CamCfg {
    name: String,
    shm: String,
}

struct Args {
    cams: Vec<CamCfg>,
    state_shm: String,
    event_socket: String,
    fps: u32,
    out: String,
    manifest: String,
    quality: u8,
    task: String,
    // Per-element labels for the state/action vectors. Empty → filled with
    // index strings ("0".."N-1") once the ring dims are known. Each becomes its
    // own scalar entity (/observation/state/<name>, /action/<name>) so the
    // viewer shows one named timeline per motor, and the names are written to
    // the manifest so the exporter can reassemble the vectors in order.
    state_names: Vec<String>,
    action_names: Vec<String>,
    // Record-time observation-alignment mode (0/1), passed through to the
    // manifest for provenance (the export applies its own deterministic rule).
    obs_align: u8,
}

fn parse_args() -> Args {
    let mut a = Args {
        cams: vec![],
        state_shm: "/dm_state_0".into(),
        event_socket: "/tmp/dm_record_events.sock".into(),
        fps: 50,
        out: "/tmp/dm_session".into(), // output DIR for per-episode episode_<i>.rrd files
        manifest: "/tmp/dm_session/session.json".into(),
        quality: 85,
        task: "unnamed task".into(),
        state_names: vec![],
        action_names: vec![],
        obs_align: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().expect("missing value");
        match k.as_str() {
            "--cam" => {
                // name=top,shm=/lerobot_cam_123_abc
                let spec = val();
                let mut name = String::from("cam");
                let mut shm = String::new();
                for kv in spec.split(',') {
                    let (k, v) = kv.split_once('=').expect("--cam key=val");
                    match k {
                        "name" => name = v.into(),
                        "shm" => shm = v.into(),
                        _ => panic!("unknown cam key {k}"),
                    }
                }
                assert!(!shm.is_empty(), "--cam needs shm=");
                a.cams.push(CamCfg { name, shm });
            }
            "--state-shm" => a.state_shm = val(),
            "--event-socket" => a.event_socket = val(),
            "--fps" => a.fps = val().parse().unwrap(),
            "--out" => a.out = val(),
            "--manifest" => a.manifest = val(),
            "--quality" => a.quality = val().parse().unwrap(),
            "--task" => a.task = val(),
            "--state-names" => a.state_names = val().split(',').map(|s| s.to_string()).collect(),
            "--action-names" => a.action_names = val().split(',').map(|s| s.to_string()).collect(),
            "--obs-align" => a.obs_align = val().parse().unwrap_or(0),
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(!a.cams.is_empty(), "need at least one --cam");
    a
}

#[derive(Default)]
struct EventState {
    discarded: HashSet<i64>,
    grades: HashMap<i64, f64>,
}

/// Bind the Unix datagram socket and consume episode events into shared state.
fn spawn_event_thread(path: &str, ev: Arc<Mutex<EventState>>) {
    let _ = std::fs::remove_file(path);
    let sock = match UnixDatagram::bind(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[recorder] WARN: cannot bind event socket {path}: {e} (no grade/discard events)");
            return;
        }
    };
    let _ = sock.set_read_timeout(Some(Duration::from_millis(200)));
    std::thread::spawn(move || {
        let mut buf = [0u8; 512];
        loop {
            if SESSION_END.load(Ordering::SeqCst) {
                break;
            }
            match sock.recv(&mut buf) {
                Ok(n) => {
                    let line = String::from_utf8_lossy(&buf[..n]);
                    for msg in line.split('\n') {
                        let parts: Vec<&str> = msg.split_whitespace().collect();
                        match parts.as_slice() {
                            ["discard", ep] => {
                                if let Ok(ep) = ep.parse::<i64>() {
                                    ev.lock().unwrap().discarded.insert(ep);
                                }
                            }
                            ["commit", ep, grade] => {
                                if let (Ok(ep), Ok(g)) = (ep.parse::<i64>(), grade.parse::<f64>()) {
                                    ev.lock().unwrap().grades.insert(ep, g);
                                }
                            }
                            ["session_end"] => SESSION_END.store(true, Ordering::SeqCst),
                            _ => {}
                        }
                    }
                }
                Err(_) => {} // timeout — loop to re-check SESSION_END
            }
        }
    });
}

struct EpisodeRange {
    index: i64,
    num_frames: i64,
    // real_time bounds (ref_ts of first/last sample) — recorded in the manifest.
    start_ts_ns: u64,
    end_ts_ns: u64,
    // This episode's own .rrd file (relative to the output dir). One recording per
    // episode is Rerun's intended pattern for episodic data: the viewer scopes its
    // timeline + charts to a single take, with no cross-episode bleed.
    file: String,
}

/// Per-series static styling, applied to EACH per-episode recording so every
/// episode file is self-contained: observation (FOLLOWER) draws as a solid line,
/// action (LEADER) as points (Rerun's closest to dotted), and the matching
/// obs/action of a joint share a color (paired by index).
fn apply_styling(
    rec: &rerun::RecordingStream,
    state_names: &[String],
    action_names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    const PALETTE: [[u8; 3]; 7] = [
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194],
    ];
    for (i, name) in state_names.iter().enumerate() {
        let c = PALETTE[i % PALETTE.len()];
        rec.log_static(
            format!("/observation/state/{}", name.replace('/', "_")),
            &rerun::SeriesLines::new().with_colors([c]).with_names([name.clone()]),
        )?;
    }
    for (i, name) in action_names.iter().enumerate() {
        let c = PALETTE[i % PALETTE.len()];
        rec.log_static(
            format!("/action/{}", name.replace('/', "_")),
            &rerun::SeriesPoints::new()
                .with_colors([c])
                .with_names([name.clone()])
                .with_marker_sizes([2.0]),
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = parse_args();
    unsafe { libc::signal(libc::SIGTERM, on_term as *const () as libc::sighandler_t); }
    unsafe { libc::signal(libc::SIGINT, on_term as *const () as libc::sighandler_t); }
    // Self-terminate if the parent (the control process) dies — so a crashed or
    // killed recording session can't leave an orphaned recorder writing to the
    // .rrd (concurrent writes with the next run corrupt it). On parent death the
    // kernel raises SIGTERM, handled above → clean finalize. The getppid check
    // closes the race where the parent already exited before the prctl call.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM as libc::c_ulong,
                    0 as libc::c_ulong, 0 as libc::c_ulong, 0 as libc::c_ulong);
        if libc::getppid() == 1 {
            SESSION_END.store(true, Ordering::SeqCst);
        }
    }

    // Attach to existing camera rings (read-only; the control process owns the daemons).
    let mut cams: Vec<(String, CameraRingReader, Vec<u8>)> = Vec::new();
    for c in &a.cams {
        let r = CameraRingReader::open(&c.shm)?;
        eprintln!("[recorder] cam '{}' attached: {}x{} ({} B/frame)", c.name, r.width, r.height, r.frame_bytes);
        let buf = vec![0u8; r.frame_bytes];
        cams.push((c.name.clone(), r, buf));
    }
    // Attach to the state ring (driver of the recording cadence).
    let state = StateRingReader::open(&a.state_shm)?;
    eprintln!("[recorder] state ring: dim(state={}, action={}) ring={}", state.state_dim, state.action_dim, state.ring_size);
    // Fall back to index labels if names were absent or don't match the ring
    // dims (keeps per-component logging + the manifest self-consistent).
    if a.state_names.len() != state.state_dim {
        a.state_names = (0..state.state_dim).map(|i| i.to_string()).collect();
    }
    if a.action_names.len() != state.action_dim {
        a.action_names = (0..state.action_dim).map(|i| i.to_string()).collect();
    }

    let ev = Arc::new(Mutex::new(EventState::default()));
    spawn_event_thread(&a.event_socket, ev.clone());

    let tjc = TjCompressor::new();
    std::fs::create_dir_all(&a.out)?;  // --out is now a DIRECTORY of per-episode .rrd files

    let mut eps: Vec<EpisodeRange> = Vec::new();
    let mut cur: Option<EpisodeRange> = None;
    let mut rec: Option<rerun::RecordingStream> = None; // the current episode's stream
    let mut frame: i64 = 0; // per-episode frame counter (resets each episode)
    let mut drops: u64 = 0;
    let mut last_cam_ts = vec![0u64; cams.len()]; // per-cam last-logged capture ts (dedup)
    let mut consumed = state.write_idx(); // skip pre-existing slots

    while !SESSION_END.load(Ordering::SeqCst) {
        let w = state.write_idx();
        if w < consumed {
            // write_idx went backwards → the ring was re-created under us (a
            // different session re-using the shm name). Resync rather than
            // underflowing the unsigned subtraction (and don't emit garbage).
            consumed = w;
            continue;
        }
        if w == consumed {
            std::thread::sleep(Duration::from_micros(300));
            continue;
        }
        if w - consumed > state.ring_size {
            drops += (w - consumed) - state.ring_size;
            consumed = w - state.ring_size;
        }
        for idx in consumed..w {
            let s = match state.read_index(idx) {
                Some(s) => s,
                None => {
                    drops += 1;
                    continue;
                }
            };
            // New episode index → close the prior recording, open a fresh one
            // (one .rrd per episode). Each gets its own recording-id + static
            // styling, so the file is self-contained and the viewer can scope to it.
            if cur.as_ref().map(|c| c.index) != Some(s.episode_index) {
                if let Some(r) = rec.take() {
                    let _ = r.flush_blocking(); // close the prior episode's file
                }
                if let Some(c) = cur.take() {
                    // A discarded take's .rrd would otherwise linger until session-end
                    // cleanup — long enough for the hub's file-existence watcher to copy,
                    // register, and upload it the instant THIS next episode's file appears
                    // (it ingests episode_<N> once any later-indexed file exists). Delete
                    // it now, before that next file is created, so the watcher never sees
                    // it. The session-end manifest pass still covers a discard of the very
                    // last episode (no next-episode transition to trigger this).
                    if ev.lock().unwrap().discarded.contains(&c.index) {
                        let _ = std::fs::remove_file(format!("{}/{}", a.out, c.file));
                    }
                    eps.push(c);
                }
                let file = format!("episode_{:03}.rrd", s.episode_index);
                let r = rerun::RecordingStreamBuilder::new("dm_record")
                    .recording_id(format!("episode-{}", s.episode_index))
                    .save(format!("{}/{}", a.out, file))?;
                apply_styling(&r, &a.state_names, &a.action_names)?;
                rec = Some(r);
                frame = 0;
                last_cam_ts.iter_mut().for_each(|t| *t = 0); // re-log the first frame of each episode
                cur = Some(EpisodeRange {
                    index: s.episode_index,
                    num_frames: 0,
                    start_ts_ns: s.ref_ts_ns,
                    end_ts_ns: s.ref_ts_ns,
                    file,
                });
            }
            let rec = rec.as_ref().expect("episode stream open");
            rec.set_time_sequence("frame", frame);

            // Cameras: log each at its OWN kernel capture timestamp, deduped —
            // skip when the at-or-before frame hasn't advanced since we last
            // logged it. State arrives at the control rate (~250 Hz) but cameras
            // only update at ~their fps, so without dedup every JPEG would be
            // logged ~5×. Logging at the true capture ts (not the slot ref_ts)
            // keeps the .rrd faithful so the exporter can do a deterministic
            // camera-anchored align=1 join.
            let mut cap_ts: Vec<u64> = Vec::with_capacity(cams.len());
            for (ci, (name, reader, buf)) in cams.iter_mut().enumerate() {
                if let Some(meta) = reader.read_index_at_ts(s.ref_ts_ns, buf) {
                    cap_ts.push(meta.capture_mono_ns);
                    // The image, at its true capture time, deduped (skip if unchanged).
                    if meta.capture_mono_ns != last_cam_ts[ci] {
                        last_cam_ts[ci] = meta.capture_mono_ns;
                        rec.set_duration_secs("real_time", meta.capture_mono_ns as f64 / 1e9);
                        let jpg = tjc.compress_bgr(buf, reader.width as i32, reader.height as i32, a.quality as i32);
                        rec.log(format!("/cam/{name}"), &rerun::EncodedImage::from_file_contents(jpg))?;
                    }
                    // Latency diagnostics, stamped at the sampling tick (ref_ts) so
                    // they line up with the joints: how old the frame was when we
                    // sampled it, and how long the daemon took to decode+publish it.
                    rec.set_duration_secs("real_time", s.ref_ts_ns as f64 / 1e9);
                    let age_ms = s.ref_ts_ns.saturating_sub(meta.capture_mono_ns) as f64 / 1e6;
                    let decode_ms = meta.published_mono_ns.saturating_sub(meta.capture_mono_ns) as f64 / 1e6;
                    rec.log(format!("/diagnostics/image_age_ms/{name}"), &rerun::Scalars::new([age_ms]))?;
                    rec.log(format!("/diagnostics/decode_latency_ms/{name}"), &rerun::Scalars::new([decode_ms]))?;
                }
            }
            // Inter-camera skew = FULL SPREAD (newest − oldest capture) across all
            // cameras at this sample — the worst-case misalignment of the
            // free-running/unsynced cameras, not a pairwise or anchor-relative delta.
            if cap_ts.len() >= 2 {
                let skew_ms = (cap_ts.iter().max().unwrap() - cap_ts.iter().min().unwrap()) as f64 / 1e6;
                rec.set_duration_secs("real_time", s.ref_ts_ns as f64 / 1e9);
                rec.log("/diagnostics/inter_camera_skew_ms", &rerun::Scalars::new([skew_ms]))?;
            }
            // State/action at the joint sample's own timestamp. One named scalar
            // entity per motor → one timeline per motor in the viewer (unfold
            // /observation/state). The exporter reassembles the vectors from
            // these columns in manifest (state_names) order.
            rec.set_duration_secs("real_time", s.ref_ts_ns as f64 / 1e9);
            for (i, &v) in s.state.iter().enumerate() {
                rec.log(
                    format!("/observation/state/{}", a.state_names[i].replace('/', "_")),
                    &rerun::Scalars::new([v as f64]),
                )?;
            }
            for (i, &v) in s.action.iter().enumerate() {
                rec.log(
                    format!("/action/{}", a.action_names[i].replace('/', "_")),
                    &rerun::Scalars::new([v as f64]),
                )?;
            }
            // Rollout phase + intervention, stamped at the joint sample's time so
            // they line up with state/action. Only logged for policy rollouts
            // (control_phase >= 0); plain teleop recordings publish PHASE_NONE and
            // therefore carry no /intervention or /control_phase timeline, so the
            // exporter adds those columns only when they really exist.
            if s.control_phase >= 0 {
                rec.log("/control_phase", &rerun::Scalars::new([s.control_phase as f64]))?;
                let intervening = (s.control_phase == ring::PHASE_CORRECTING) as i32;
                rec.log("/intervention", &rerun::Scalars::new([intervening as f64]))?;
            }

            if let Some(c) = cur.as_mut() {
                c.num_frames += 1;
                c.end_ts_ns = s.ref_ts_ns;
            }
            frame += 1;
        }
        consumed = w;
    }
    if let Some(c) = cur.take() {
        eps.push(c);
    }
    if let Some(r) = rec.take() {
        let tf = Instant::now();
        r.flush_blocking()?; // close the final episode's file
        eprintln!("[recorder] flush_blocking {:.1} ms", tf.elapsed().as_secs_f64() * 1e3);
    }

    // ── manifest (drop + delete discarded episodes' files, attach grades) ──
    let evs = ev.lock().unwrap();
    let cams_json: Vec<String> = cams.iter().map(|(name, r, _)|
        format!("{{\"name\": {:?}, \"width\": {}, \"height\": {}}}", name, r.width, r.height)).collect();
    let mut eps_json: Vec<String> = Vec::new();
    let mut kept = 0;
    for e in &eps {
        if evs.discarded.contains(&e.index) {
            let _ = std::fs::remove_file(format!("{}/{}", a.out, e.file)); // discarded → delete its file
            continue;
        }
        kept += 1;
        let q = evs.grades.get(&e.index).copied();
        let qual = q.map(|g| format!("{g}")).unwrap_or_else(|| "null".into());
        eps_json.push(format!(
            "{{\"episode_index\": {}, \"file\": {:?}, \"num_frames\": {}, \"start_ts_ns\": {}, \"end_ts_ns\": {}, \"quality\": {}}}",
            e.index, e.file, e.num_frames, e.start_ts_ns, e.end_ts_ns, qual
        ));
    }
    let snames_json: String = a.state_names.iter().map(|n| format!("{n:?}")).collect::<Vec<_>>().join(", ");
    let anames_json: String = a.action_names.iter().map(|n| format!("{n:?}")).collect::<Vec<_>>().join(", ");
    let m = format!(
        "{{\n  \"fps\": {},\n  \"task\": {:?},\n  \"state_dim\": {},\n  \"action_dim\": {},\n  \"obs_align\": {},\n  \"state_names\": [{}],\n  \"action_names\": [{}],\n  \"dir\": {:?},\n  \"cameras\": [{}],\n  \"episodes\": [{}]\n}}\n",
        a.fps, a.task, state.state_dim, state.action_dim, a.obs_align, snames_json, anames_json, a.out, cams_json.join(", "), eps_json.join(", ")
    );
    std::fs::File::create(&a.manifest)?.write_all(m.as_bytes())?;

    let total: i64 = eps.iter().map(|e| e.num_frames).sum();
    eprintln!(
        "[recorder] done: {} episodes ({} kept, {} discarded), {} frames, {} cam(s), state-drops={} -> {}/ (manifest {})",
        eps.len(), kept, eps.len() - kept, total, cams.len(), drops, a.out, a.manifest
    );
    Ok(())
}
