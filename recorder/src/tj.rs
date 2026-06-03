//! Minimal FFI to the system libturbojpeg (2.1.x) for JPEG compression — same
//! encoder as the C++ recorder, so the Rust-vs-C++ comparison is apples-to-apples.

use std::os::raw::{c_int, c_uchar, c_ulong, c_void};

const TJPF_BGR: c_int = 1; // enum TJPF: RGB=0, BGR=1, ...
const TJSAMP_420: c_int = 2; // enum TJSAMP: 444=0, 422=1, 420=2

extern "C" {
    fn tjInitCompress() -> *mut c_void;
    fn tjCompress2(
        handle: *mut c_void,
        src: *const c_uchar,
        width: c_int,
        pitch: c_int,
        height: c_int,
        pixel_format: c_int,
        jpeg_buf: *mut *mut c_uchar,
        jpeg_size: *mut c_ulong,
        jpeg_subsamp: c_int,
        jpeg_qual: c_int,
        flags: c_int,
    ) -> c_int;
    fn tjFree(buf: *mut c_uchar);
    fn tjDestroy(handle: *mut c_void) -> c_int;
}

pub struct TjCompressor {
    handle: *mut c_void,
}

impl TjCompressor {
    pub fn new() -> Self {
        let handle = unsafe { tjInitCompress() };
        assert!(!handle.is_null(), "tjInitCompress failed");
        Self { handle }
    }

    /// JPEG-compress a BGR8 buffer (w*h*3) into a fresh Vec.
    pub fn compress_bgr(&self, bgr: &[u8], w: i32, h: i32, quality: i32) -> Vec<u8> {
        let mut out: *mut c_uchar = std::ptr::null_mut();
        let mut out_sz: c_ulong = 0;
        let rc = unsafe {
            tjCompress2(
                self.handle, bgr.as_ptr(), w, 0, h, TJPF_BGR,
                &mut out, &mut out_sz, TJSAMP_420, quality, 0,
            )
        };
        assert_eq!(rc, 0, "tjCompress2 failed");
        let v = unsafe { std::slice::from_raw_parts(out, out_sz as usize) }.to_vec();
        unsafe { tjFree(out) };
        v
    }
}

impl Drop for TjCompressor {
    fn drop(&mut self) {
        unsafe { tjDestroy(self.handle) };
    }
}
