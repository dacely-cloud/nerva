//! DPDK mbuf mempool wrapper.

use std::ffi::CString;
use std::os::raw::c_uint;
use std::ptr::NonNull;

use crate::{Error, Result, ffi};

/// Pool of mbufs all sharing one underlying memzone.
///
/// Mempools are reference-counted by DPDK; the safe wrapper just owns
/// the pointer and frees on drop.
pub struct Mempool {
    raw: NonNull<ffi::rte_mempool>,
}

unsafe impl Send for Mempool {}
unsafe impl Sync for Mempool {}

impl Mempool {
    /// Create a packet-buffer mempool sized for `n` mbufs with the
    /// default DPDK buffer size (2 KiB room + 128 B headroom).
    pub fn new_packet(name: &str, n: u32) -> Result<Self> {
        Self::new_packet_with_buf_size(name, n, ffi::RTE_MBUF_DEFAULT_BUF_SIZE as u16)
    }

    /// Create a packet-buffer mempool with caller-chosen mbuf data
    /// room. Use this for jumbo-frame configurations where the
    /// default ~2 KiB buffer is too small to hold a single frame.
    ///
    /// `buf_size` is the full mbuf size including the 128 B
    /// `RTE_PKTMBUF_HEADROOM`. For a 9100-byte MTU plus 14-byte
    /// Ethernet header, pick at least 9242; rounding up to a power
    /// of 2 or cache-friendly size (10 240) is conventional.
    pub fn new_packet_with_buf_size(name: &str, n: u32, buf_size: u16) -> Result<Self> {
        let cname = CString::new(name).map_err(|_| Error::other("mempool name has NUL"))?;
        let raw = unsafe {
            ffi::rte_pktmbuf_pool_create(
                cname.as_ptr(),
                n as c_uint,
                256, /* per-lcore cache */
                0,   /* private data size */
                buf_size,
                0, /* socket_id */
            )
        };
        match NonNull::new(raw) {
            Some(p) => Ok(Self { raw: p }),
            None => Err(Error::from_rte("rte_pktmbuf_pool_create")),
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::rte_mempool {
        self.raw.as_ptr()
    }
}

impl Drop for Mempool {
    fn drop(&mut self) {
        // SAFETY: `raw` was obtained from `rte_pktmbuf_pool_create`
        // and is non-null. `rte_mempool_free` accepts the pointer and
        // is safe to call once per pool.
        unsafe { ffi::rte_mempool_free(self.raw.as_ptr()) };
    }
}
