//! Lock-free shm readers for the pub/sub recorder.
//!
//! Two ring types, both POSIX shm written by a producer and read by this
//! recorder process, x86-64 little-endian, seqlock-on-write_idx discipline:
//!   * CameraRingReader  — the C++ camera daemon's frame ring (lerobot_cam_capture.cpp)
//!   * StateRingReader   — the Python control process's state/action ring
//!                         (src/lerobot/recording/state_publisher.py). The publisher
//!                         stamps each slot with the reference timestamp the recorder
//!                         uses to align camera frames.

use memmap2::Mmap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};

// ───────────────────────── camera ring ─────────────────────────
pub const SHM_MAGIC: u32 = 0x4C52_434D; // 'LRCM'
pub const SHM_VERSION: u32 = 1;
const WRITE_IDX_OFFSET: usize = 48;
const FRAME_META_SIZE: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct FrameMeta {
    pub seq: u64,
    /// CLOCK_MONOTONIC INSTANT the frame was captured by the kernel — a point in
    /// time (≈ end-of-frame for these UVC cameras), NOT a duration/exposure window.
    pub capture_mono_ns: u64,
    /// CLOCK_MONOTONIC instant the daemon finished MJPEG→BGR decode + published to shm.
    pub published_mono_ns: u64,
}

pub struct CameraRingReader {
    mmap: Mmap,
    pub ring_size: usize,
    pub width: u32,
    pub height: u32,
    pub frame_bytes: usize,
    metadata_offset: usize,
    frames_offset: usize,
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
#[inline]
fn rd_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}
#[inline]
fn acquire_u64(mmap: &Mmap, off: usize) -> u64 {
    let ptr = unsafe { mmap.as_ptr().add(off) } as *const AtomicU64;
    unsafe { (*ptr).load(Ordering::Acquire) }
}

impl CameraRingReader {
    pub fn open(shm_name: &str) -> std::io::Result<Self> {
        let path = format!("/dev/shm/{}", shm_name.trim_start_matches('/'));
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let invalid = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        if rd_u32(&mmap, 0) != SHM_MAGIC {
            return Err(invalid(format!("bad camera magic 0x{:08x}", rd_u32(&mmap, 0))));
        }
        if rd_u32(&mmap, 4) != SHM_VERSION {
            return Err(invalid(format!("bad camera version {}", rd_u32(&mmap, 4))));
        }
        let ring_size = rd_u32(&mmap, 8) as usize;
        let width = rd_u32(&mmap, 12);
        let height = rd_u32(&mmap, 16);
        let channels = rd_u32(&mmap, 20);
        let frame_bytes = rd_u32(&mmap, 24) as usize;
        let metadata_offset = rd_u32(&mmap, 28) as usize;
        let frames_offset = rd_u32(&mmap, 32) as usize;
        if channels != 3 {
            return Err(invalid(format!("expected 3 channels, got {channels}")));
        }
        Ok(Self { mmap, ring_size, width, height, frame_bytes, metadata_offset, frames_offset })
    }

    #[inline]
    pub fn write_idx(&self) -> u64 {
        acquire_u64(&self.mmap, WRITE_IDX_OFFSET)
    }

    #[inline]
    fn meta_at(&self, slot: usize) -> FrameMeta {
        // On-wire FRAME_META_SIZE (32 B) layout: seq u64 @0, capture u64 @8,
        // published u64 @16, bytes_used u32 @24, flags u32 @28. (bytes_used/flags
        // skipped — frames are fixed-size raw BGR.)
        let o = self.metadata_offset + slot * FRAME_META_SIZE;
        FrameMeta {
            seq: rd_u64(&self.mmap, o),
            capture_mono_ns: rd_u64(&self.mmap, o + 8),
            published_mono_ns: rd_u64(&self.mmap, o + 16),
        }
    }

    #[inline]
    fn copy_frame_into(&self, slot: usize, dst: &mut [u8]) {
        let o = self.frames_offset + slot * self.frame_bytes;
        dst.copy_from_slice(&self.mmap[o..o + self.frame_bytes]);
    }

    pub fn read_index_into(&self, idx: u64, dst: &mut [u8]) -> Option<FrameMeta> {
        let slot = (idx % self.ring_size as u64) as usize;
        let meta_before = self.meta_at(slot);
        self.copy_frame_into(slot, dst);
        let now = self.write_idx();
        if now.saturating_sub(idx) > self.ring_size as u64 {
            return None; // lapped during copy
        }
        if self.meta_at(slot).seq != meta_before.seq {
            return None; // slot rewritten during copy
        }
        Some(meta_before)
    }

    /// Timestamp-aligned read: copy the newest frame whose kernel capture time
    /// is at-or-before `ref_ts_ns` (same selection as the in-process
    /// `get_state_at`/`get_snapshot_at`). Returns None if none in the ring.
    pub fn read_index_at_ts(&self, ref_ts_ns: u64, dst: &mut [u8]) -> Option<FrameMeta> {
        let w = self.write_idx();
        if w == 0 {
            return None;
        }
        let n = self.ring_size as u64;
        let scan = n.min(w);
        for off in 0..scan {
            let idx = w - 1 - off;
            let slot = (idx % n) as usize;
            if self.meta_at(slot).capture_mono_ns <= ref_ts_ns {
                return self.read_index_into(idx, dst);
            }
        }
        None
    }
}

// ───────────────────────── state ring ─────────────────────────
// Header (little-endian), 64-byte aligned. Written by state_publisher.py.
//   0  magic u32 = 'DMST'        12 state_dim u32     24 meta_offset u32 (64-aligned)
//   4  version u32 = 1           16 action_dim u32    28 _pad u32
//   8  ring_size u32             20 slot_stride u32   32 write_idx u64 (atomic)
// Per slot @ meta_offset + idx*slot_stride (publisher writes a slot only while
// recording, tagging it with the episode it belongs to — so the recorder groups
// frames by episode without racing the event channel):
//   0 seq u64 | 8 ref_ts_ns u64 | 16 episode_index i64 | 24 state[sdim] f32 | .. action[adim] f32
pub const STATE_MAGIC: u32 = 0x444D_5354; // 'DMST'
pub const STATE_VERSION: u32 = 1;
const STATE_WRITE_IDX_OFFSET: usize = 32;

#[derive(Clone, Debug)]
pub struct StateSample {
    pub seq: u64,
    pub ref_ts_ns: u64,
    pub episode_index: i64,
    pub state: Vec<f32>,
    pub action: Vec<f32>,
}

pub struct StateRingReader {
    mmap: Mmap,
    pub ring_size: u64,
    pub state_dim: usize,
    pub action_dim: usize,
    slot_stride: usize,
    meta_offset: usize,
}

impl StateRingReader {
    pub fn open(shm_name: &str) -> std::io::Result<Self> {
        let path = format!("/dev/shm/{}", shm_name.trim_start_matches('/'));
        let file = File::open(&path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let invalid = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        if rd_u32(&mmap, 0) != STATE_MAGIC {
            return Err(invalid(format!("bad state magic 0x{:08x}", rd_u32(&mmap, 0))));
        }
        if rd_u32(&mmap, 4) != STATE_VERSION {
            return Err(invalid(format!("bad state version {}", rd_u32(&mmap, 4))));
        }
        Ok(Self {
            ring_size: rd_u32(&mmap, 8) as u64,
            state_dim: rd_u32(&mmap, 12) as usize,
            action_dim: rd_u32(&mmap, 16) as usize,
            slot_stride: rd_u32(&mmap, 20) as usize,
            meta_offset: rd_u32(&mmap, 24) as usize,
            mmap,
        })
    }

    #[inline]
    pub fn write_idx(&self) -> u64 {
        acquire_u64(&self.mmap, STATE_WRITE_IDX_OFFSET)
    }

    fn read_slot(&self, slot: usize) -> StateSample {
        let base = self.meta_offset + slot * self.slot_stride;
        let rd_f32 = |o: usize| f32::from_le_bytes([self.mmap[o], self.mmap[o + 1], self.mmap[o + 2], self.mmap[o + 3]]);
        let mut o = base + 24;
        let mut state = Vec::with_capacity(self.state_dim);
        for _ in 0..self.state_dim {
            state.push(rd_f32(o));
            o += 4;
        }
        let mut action = Vec::with_capacity(self.action_dim);
        for _ in 0..self.action_dim {
            action.push(rd_f32(o));
            o += 4;
        }
        StateSample {
            seq: rd_u64(&self.mmap, base),
            ref_ts_ns: rd_u64(&self.mmap, base + 8),
            episode_index: rd_u64(&self.mmap, base + 16) as i64,
            state,
            action,
        }
    }

    /// Read absolute index `idx`, verifying the producer didn't lap us mid-read.
    pub fn read_index(&self, idx: u64) -> Option<StateSample> {
        let slot = (idx % self.ring_size) as usize;
        let s = self.read_slot(slot);
        if self.write_idx().saturating_sub(idx) > self.ring_size {
            return None;
        }
        if rd_u64(&self.mmap, self.meta_offset + slot * self.slot_stride) != s.seq {
            return None;
        }
        Some(s)
    }
}
