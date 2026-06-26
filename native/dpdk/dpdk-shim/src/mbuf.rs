//! Owned mbuf handle. Dropping returns the buffer to its mempool.

use std::ptr::NonNull;
use std::slice;

use crate::ffi;

/// Owned single-segment mbuf. We don't expose multi-segment chains
/// because our HTTP path never builds them — every response is one
/// contiguous write.
pub struct Mbuf {
    raw: NonNull<ffi::rte_mbuf>,
}

unsafe impl Send for Mbuf {}

impl Mbuf {
    /// Wrap a raw mbuf pointer obtained from `rx_burst` or `alloc`.
    /// Returns None for null.
    pub(crate) fn from_raw(raw: *mut ffi::rte_mbuf) -> Option<Self> {
        NonNull::new(raw).map(|p| Self { raw: p })
    }

    /// Pointer to the first segment's data region. Empty slice if the
    /// mbuf has zero-length data.
    pub fn data(&self) -> &[u8] {
        unsafe {
            let ptr = ffi::shim_pktmbuf_mtod(self.raw.as_ptr());
            let len = ffi::shim_pktmbuf_data_len(self.raw.as_ptr());
            if ptr.is_null() || len == 0 {
                &[]
            } else {
                slice::from_raw_parts(ptr, len as usize)
            }
        }
    }

    /// Writable view of the data region. Capacity is the segment's
    /// current data_len plus its tailroom: i.e. the bytes we can
    /// safely write at the mtod pointer without overflowing the mbuf.
    pub fn data_mut(&mut self) -> Option<&mut [u8]> {
        unsafe {
            let ptr = ffi::shim_pktmbuf_mtod(self.raw.as_ptr());
            if ptr.is_null() {
                return None;
            }
            let cur = ffi::shim_pktmbuf_data_len(self.raw.as_ptr()) as usize;
            let tail = ffi::shim_pktmbuf_tailroom(self.raw.as_ptr()) as usize;
            Some(slice::from_raw_parts_mut(ptr, cur + tail))
        }
    }

    pub fn set_data_len(&mut self, len: u16) {
        unsafe { ffi::shim_pktmbuf_set_data_len(self.raw.as_ptr(), len) };
    }

    pub fn set_pkt_len(&mut self, len: u32) {
        unsafe { ffi::shim_pktmbuf_set_pkt_len(self.raw.as_ptr(), len) };
    }

    /// Annotate this mbuf so the NIC computes the IPv4 and TCP
    /// checksums on tx. `l2_len` is the Ethernet header size (14 for
    /// untagged), `l3_len` is the IP header size (20 for IPv4 no
    /// options). Must be called BEFORE the mbuf is handed to
    /// `rte_eth_tx_burst`.
    pub fn set_tx_offload_v4tcp(&mut self, l2_len: u16, l3_len: u16) {
        unsafe {
            ffi::shim_mbuf_set_tx_offload_v4tcp(self.raw.as_ptr(), l2_len, l3_len);
        }
    }

    /// Annotate this mbuf for IPv4-header-checksum-only tx offload (no L4
    /// offload), for non-TCP IPv4 packets (ICMP, UDP). smoltcp computes
    /// their L4 checksum in software; using [`Self::set_tx_offload_v4tcp`]
    /// on a non-TCP packet makes the NIC scribble a bogus TCP checksum into
    /// the payload and the peer drops it. Must be called BEFORE the mbuf is
    /// handed to `rte_eth_tx_burst`.
    pub fn set_tx_offload_v4(&mut self, l2_len: u16, l3_len: u16) {
        unsafe {
            ffi::shim_mbuf_set_tx_offload_v4(self.raw.as_ptr(), l2_len, l3_len);
        }
    }

    /// IPv4 + UDP checksum tx offload, for UDP/IPv4 packets.
    pub fn set_tx_offload_v4udp(&mut self, l2_len: u16, l3_len: u16) {
        unsafe {
            ffi::shim_mbuf_set_tx_offload_v4udp(self.raw.as_ptr(), l2_len, l3_len);
        }
    }

    /// Prepare this IPv4/TCP mbuf for hardware TCP segmentation offload
    /// (TSO): sets header lengths, per-segment size `mss`, the offload
    /// flags, and the IPv4 pseudo-header checksum the mlx5 TSO contract
    /// requires in the TCP checksum field (IPv4 checksum left 0). `mss` is
    /// the connection's negotiated MSS (on-wire per-segment payload size).
    pub fn prepare_tso_v4tcp(&mut self, l2_len: u16, l3_len: u16, l4_len: u16, mss: u16) {
        unsafe {
            ffi::shim_mbuf_prepare_tso_v4tcp(self.raw.as_ptr(), l2_len, l3_len, l4_len, mss);
        }
    }

    /// Read the mbuf's offload flags. On RX these carry the NIC's
    /// checksum verdicts (`RTE_MBUF_F_RX_{IP,L4}_CKSUM_*`).
    pub fn ol_flags(&self) -> u64 {
        unsafe { ffi::shim_mbuf_ol_flags(self.raw.as_ptr()) }
    }

    /// TSO variant. Annotates the mbuf so the NIC computes per-segment
    /// IPv4 + TCP checksums **and** slices the payload into
    /// `tso_segsz`-byte TCP segments on the wire.
    ///
    ///   * `l2_len`  = Ethernet header bytes (14)
    ///   * `l3_len`  = IPv4 header bytes (20)
    ///   * `l4_len`  = TCP header bytes (20)
    ///   * `tso_segsz` = per-on-wire-segment payload size
    ///
    /// The port must have been configured with
    /// `RTE_ETH_TX_OFFLOAD_TCP_TSO` enabled, otherwise `tx_burst` will
    /// either drop the mbuf or send malformed frames depending on
    /// PMD behaviour.
    pub fn set_tx_offload_v4tcp_tso(
        &mut self,
        l2_len: u16,
        l3_len: u16,
        l4_len: u16,
        tso_segsz: u16,
    ) {
        unsafe {
            ffi::shim_mbuf_set_tx_offload_v4tcp_tso(
                self.raw.as_ptr(),
                l2_len,
                l3_len,
                l4_len,
                tso_segsz,
            );
        }
    }

    pub(crate) fn into_raw(self) -> *mut ffi::rte_mbuf {
        let p = self.raw.as_ptr();
        std::mem::forget(self);
        p
    }

    pub(crate) fn as_raw(&self) -> *mut ffi::rte_mbuf {
        self.raw.as_ptr()
    }
}

impl Drop for Mbuf {
    fn drop(&mut self) {
        // SAFETY: `raw` is non-null and owned. shim_pktmbuf_free
        // forwards to rte_pktmbuf_free.
        unsafe { ffi::shim_pktmbuf_free(self.raw.as_ptr()) };
    }
}
