//! Per-port rx/tx queue wrappers.

use crate::mbuf::Mbuf;
use crate::mempool::Mempool;
use crate::{Error, Result, ffi};

/// Reference to one rx queue on a started port. Cheap to copy/move;
/// the queue itself is owned by the Port.
#[derive(Clone, Copy)]
pub struct RxQueue {
    port_id: u16,
    queue_id: u16,
}

impl RxQueue {
    pub fn new(port_id: u16, queue_id: u16) -> Self {
        Self { port_id, queue_id }
    }

    pub fn queue_id(&self) -> u16 {
        self.queue_id
    }

    /// Largest single rx burst [`RxQueue::receive_burst_into`] will
    /// issue. Bounds the stack scratch array (256 ptrs = 2 KiB).
    pub const MAX_BURST: usize = 256;

    /// Poll up to `nb_max` packets (clamped to [`Self::MAX_BURST`])
    /// and append them to `out`. Returns how many were received.
    ///
    /// Allocation-free on the hot path: the raw pointer scratch lives
    /// on the stack and `out` keeps its capacity across calls, so a
    /// steady-state poll loop never touches the heap here. Each Mbuf
    /// frees itself to its source mempool on drop.
    pub fn receive_burst_into(
        &self,
        out: &mut std::collections::VecDeque<Mbuf>,
        nb_max: u16,
    ) -> u16 {
        let nb = nb_max.min(Self::MAX_BURST as u16);
        let mut raw: [*mut ffi::rte_mbuf; Self::MAX_BURST] =
            [std::ptr::null_mut(); Self::MAX_BURST];
        let n =
            unsafe { ffi::shim_eth_rx_burst(self.port_id, self.queue_id, raw.as_mut_ptr(), nb) };
        for &p in &raw[..n as usize] {
            if let Some(m) = Mbuf::from_raw(p) {
                out.push_back(m);
            }
        }
        n
    }

    /// Poll up to `nb_max` packets. Caller can immediately consume
    /// the returned Mbufs. Each Mbuf will free itself to its source
    /// mempool on drop.
    ///
    /// Allocates per call; the server hot path uses
    /// [`Self::receive_burst_into`] instead. Kept for examples/tools.
    pub fn receive_burst(&self, nb_max: u16) -> Vec<Mbuf> {
        let mut raw: Vec<*mut ffi::rte_mbuf> = vec![std::ptr::null_mut(); nb_max as usize];
        let n = unsafe {
            ffi::shim_eth_rx_burst(self.port_id, self.queue_id, raw.as_mut_ptr(), nb_max)
        };
        let mut out = Vec::with_capacity(n as usize);
        for p in raw.into_iter().take(n as usize) {
            if let Some(m) = Mbuf::from_raw(p) {
                out.push(m);
            }
        }
        out
    }
}

/// Tx queue paired with the mempool used to allocate outgoing mbufs.
pub struct TxQueue {
    port_id: u16,
    queue_id: u16,
    pool: Mempool,
}

impl TxQueue {
    pub fn new(port_id: u16, queue_id: u16, pool: Mempool) -> Self {
        Self {
            port_id,
            queue_id,
            pool,
        }
    }

    /// Allocate one fresh mbuf from the queue's pool. Returns an error
    /// when the pool is exhausted (NIC backpressure).
    pub fn alloc(&self) -> Result<Mbuf> {
        let raw = unsafe { ffi::shim_pktmbuf_alloc(self.pool.as_ptr()) };
        Mbuf::from_raw(raw).ok_or_else(|| Error::other("mempool exhausted"))
    }

    /// Send a single mbuf. Returns Err if the NIC tx ring is full.
    pub fn send_one(&self, mbuf: Mbuf) -> Result<()> {
        let mut p = [mbuf.into_raw()];
        let sent =
            unsafe { ffi::shim_eth_tx_burst(self.port_id, self.queue_id, p.as_mut_ptr(), 1) };
        if sent != 1 {
            // Drop the mbuf we couldn't send to avoid a leak: re-wrap
            // it so its Drop returns it to the pool.
            // SAFETY: we own p[0] and it's the one we passed in.
            let _ = Mbuf::from_raw(p[0]);
            Err(Error::other("rte_eth_tx_burst returned 0"))
        } else {
            Ok(())
        }
    }

    /// Send a burst of mbufs in a single `rte_eth_tx_burst` call.
    ///
    /// Drains `mbufs` (it's left empty on return) and returns the
    /// number actually queued by the NIC. Any mbufs the NIC did not
    /// accept (typically because its tx ring is momentarily full) are
    /// freed back to their pool — the caller does not need to handle
    /// them.
    ///
    /// One C call per batch instead of one per frame. For 1 MiB
    /// responses (~700 mbufs) that's a 30×+ reduction in PMD/syscall
    /// crossover overhead vs `send_one` in a loop.
    pub fn send_burst(&self, mbufs: &mut Vec<Mbuf>) -> usize {
        if mbufs.is_empty() {
            return 0;
        }
        let mut raw: Vec<*mut ffi::rte_mbuf> = mbufs.drain(..).map(|m| m.into_raw()).collect();
        let to_send = raw.len() as u16;
        let sent = unsafe {
            ffi::shim_eth_tx_burst(self.port_id, self.queue_id, raw.as_mut_ptr(), to_send)
        };
        // Anything past `sent` was not accepted by the NIC — free those
        // mbufs by re-wrapping so their Drop returns them to the pool.
        for &p in raw.iter().skip(sent as usize) {
            let _ = Mbuf::from_raw(p);
        }
        sent as usize
    }
}
