//! Raw DPDK FFI for NERVA transport experiments.
//!
//! The crate is intentionally thin: it exposes the bindgen-generated
//! FFI under [`ffi`] and a few safe wrappers (EAL init, mempool,
//! rx/tx queues, flow rule). Higher-level transport protocol logic
//! lives outside this shim.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

/// Raw bindgen-generated FFI bindings.
///
/// Contents are emitted at build time by `build.rs` so we can't
/// directly annotate items inside. The `unsafe_op_in_unsafe_fn`
/// allow silences Rust 2024 edition warnings on bindgen's
/// `unsafe fn` wrappers that internally call other unsafe FFI without
/// an explicit `unsafe { ... }` block.
#[allow(unsafe_op_in_unsafe_fn)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

mod eal;
mod flow;
mod mbuf;
mod mempool;
mod port;
mod queue;

pub use eal::{Eal, EalArgs};
pub use flow::{
    install_arp_to_queue_rule, install_icmp_to_queue_rule, install_tcp_port_flow_rule,
    install_tcp_rss_flow_rule, install_udp_rss_flow_rule,
};
pub use mbuf::Mbuf;
pub use mempool::Mempool;
pub use port::{
    DESIRED_RX_OFFLOADS, DESIRED_TX_OFFLOADS, OffloadCaps, Port, PortConfig, RSS_HF_TCP, RSS_HF_TCP_UDP, RSS_HF_IP, rss_hf_from_alias,
    RTE_ETH_RSS_IPV4, RTE_ETH_RSS_IPV6, RTE_ETH_RSS_NONFRAG_IPV4_TCP, RTE_ETH_RSS_NONFRAG_IPV4_UDP,
    RTE_ETH_RSS_NONFRAG_IPV6_TCP, RTE_ETH_RSS_NONFRAG_IPV6_UDP, RTE_ETH_RX_OFFLOAD_IPV4_CKSUM,
    RTE_ETH_RX_OFFLOAD_TCP_CKSUM, RTE_ETH_RX_OFFLOAD_UDP_CKSUM, RTE_ETH_TX_OFFLOAD_IPV4_CKSUM,
    RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE, RTE_ETH_TX_OFFLOAD_TCP_CKSUM, RTE_ETH_TX_OFFLOAD_TCP_TSO,
    RTE_ETH_TX_OFFLOAD_UDP_CKSUM,
};
pub use queue::{RxQueue, TxQueue};

/// Register the current thread with DPDK so its calls into mempool /
/// ring / PMD code see a real `rte_lcore_id` instead of
/// `LCORE_ID_ANY`. Without this, the per-lcore mempool cache is
/// bypassed and every `rte_pktmbuf_free` round-trips through the
/// shared MP/MC ring with a CAS, burning ~50–70 % of CPU under
/// packet-heavy load.
///
/// Call ONCE per worker thread, after `pin_to_core` and BEFORE the
/// first DPDK call from that thread (in practice: before
/// the worker's packet loop). Idempotent: calling twice is harmless;
/// DPDK returns 0 if already registered.
pub fn register_thread() -> Result<()> {
    // SAFETY: rte_thread_register takes no arguments and is safe to
    // call from any thread. Returns 0 on success, -1 with rte_errno
    // set on failure.
    let rc = unsafe { ffi::rte_thread_register() };
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::from_rte("rte_thread_register"))
    }
}

use std::ffi::CStr;
use std::fmt;

/// All errors from this crate surface through `Error`. Each variant
/// carries the negative `rte_errno` value (or 0 if the failure didn't
/// come from a DPDK call) and a human-readable message.
#[derive(Debug)]
pub struct Error {
    pub rte_errno: i32,
    pub msg: String,
}

impl Error {
    pub(crate) fn from_rte(prefix: &str) -> Self {
        // SAFETY: shim_rte_errno reads the per-lcore int set by DPDK
        // after every failed call. Reading is sound from any thread
        // that just made such a call (the value is per-lcore TLS).
        let errno = unsafe { ffi::shim_rte_errno() };
        let msg_ptr = unsafe { ffi::rte_strerror(errno) };
        let detail = if msg_ptr.is_null() {
            String::from("(null)")
        } else {
            // SAFETY: DPDK guarantees rte_strerror returns a NUL-
            // terminated static string.
            unsafe { CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        Self {
            rte_errno: errno,
            msg: format!("{prefix}: {detail} (rte_errno {errno})"),
        }
    }

    pub(crate) fn other(msg: impl Into<String>) -> Self {
        Self {
            rte_errno: 0,
            msg: msg.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.msg)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
