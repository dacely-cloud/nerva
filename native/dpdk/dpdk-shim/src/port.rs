//! Ethernet port (NIC) configuration.

use std::os::raw::c_uint;

use crate::mempool::Mempool;
use crate::{Error, Result, ffi};

/// RSS hash flags (DPDK 22+ definitions). We hardcode the few we use
/// instead of pulling them via bindgen because they expand from a
/// macro (`RTE_BIT64(n)`) bindgen can't reach.
pub const RTE_ETH_RSS_IPV4: u64 = 1 << 2;
pub const RTE_ETH_RSS_NONFRAG_IPV4_TCP: u64 = 1 << 4;
pub const RTE_ETH_RSS_NONFRAG_IPV4_UDP: u64 = 1 << 5;
pub const RTE_ETH_RSS_IPV6: u64 = 1 << 8;
pub const RTE_ETH_RSS_NONFRAG_IPV6_TCP: u64 = 1 << 10;
pub const RTE_ETH_RSS_NONFRAG_IPV6_UDP: u64 = 1 << 11;

/// `tcp` RSS: hash the TCP 4-tuple (src/dst IP + src/dst port) for IPv4 + IPv6, with an
/// IP-only fallback so non-TCP traffic still lands on a queue. The default - best for an
/// HTTP/TCP edge (per-connection flows spread evenly across RX queues).
pub const RSS_HF_TCP: u64 = RTE_ETH_RSS_NONFRAG_IPV4_TCP
    | RTE_ETH_RSS_NONFRAG_IPV6_TCP
    | RTE_ETH_RSS_IPV4
    | RTE_ETH_RSS_IPV6;

/// `tcp_udp` RSS: the TCP 4-tuple plus the UDP 4-tuple (also spread QUIC/UDP flows across
/// queues - useful once WebTransport/QUIC ingress carries real load).
pub const RSS_HF_TCP_UDP: u64 =
    RSS_HF_TCP | RTE_ETH_RSS_NONFRAG_IPV4_UDP | RTE_ETH_RSS_NONFRAG_IPV6_UDP;

/// `ip` RSS: hash the IP 2-tuple only (src/dst IP). Coarser - all flows between a pair of
/// hosts land on one queue - but stable if L4 hashing misbehaves on a given NIC.
pub const RSS_HF_IP: u64 = RTE_ETH_RSS_IPV4 | RTE_ETH_RSS_IPV6;

/// Map a config RSS alias (`dpdk { rss_hf ... }`) to its hash bitmask, or `None` for an
/// unknown alias. RSS is an L3/L4 mechanism - these aliases name which packet fields the
/// NIC hashes; there is no application-layer ("HTTP") RSS.
pub fn rss_hf_from_alias(alias: &str) -> Option<u64> {
    Some(match alias {
        "tcp" => RSS_HF_TCP,
        "tcp_udp" => RSS_HF_TCP_UDP,
        "ip" => RSS_HF_IP,
        _ => return None,
    })
}

// NIC hardware-offload capability bits (`RTE_ETH_{TX,RX}_OFFLOAD_*`,
// each `RTE_BIT64(n)`). bindgen can't expand the `RTE_BIT64` macro, so
// — like the RSS flags above — we hardcode the few we use. Values are
// stable across DPDK 21.11+ (the post-rename ABI).
pub const RTE_ETH_TX_OFFLOAD_IPV4_CKSUM: u64 = 1 << 1;
pub const RTE_ETH_TX_OFFLOAD_UDP_CKSUM: u64 = 1 << 2;
pub const RTE_ETH_TX_OFFLOAD_TCP_CKSUM: u64 = 1 << 3;
pub const RTE_ETH_TX_OFFLOAD_TCP_TSO: u64 = 1 << 5;
pub const RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE: u64 = 1 << 16;

pub const RTE_ETH_RX_OFFLOAD_IPV4_CKSUM: u64 = 1 << 1;
pub const RTE_ETH_RX_OFFLOAD_UDP_CKSUM: u64 = 1 << 2;
pub const RTE_ETH_RX_OFFLOAD_TCP_CKSUM: u64 = 1 << 3;

/// The TX/RX offloads we know how to drive, requested at port config.
/// Only the subset the PMD actually advertises is enabled (see
/// [`Port::configure_and_start`]); the rest fall back to software.
/// TSO is added by the caller (`boot.rs`) only when the full TSO data
/// path is live, because advertising it without ever emitting a
/// TSO-flagged mbuf crashes the mlx5 PMD (see the long note in
/// `configure_and_start`).
pub const DESIRED_TX_OFFLOADS: u64 = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM
    | RTE_ETH_TX_OFFLOAD_UDP_CKSUM
    | RTE_ETH_TX_OFFLOAD_TCP_CKSUM
    | RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
pub const DESIRED_RX_OFFLOADS: u64 = RTE_ETH_RX_OFFLOAD_IPV4_CKSUM
    | RTE_ETH_RX_OFFLOAD_UDP_CKSUM
    | RTE_ETH_RX_OFFLOAD_TCP_CKSUM;

/// The offloads actually enabled on a port after intersecting what we
/// requested with what the PMD advertises. Copied into the worker's
/// `DpdkDevice` so the smoltcp checksum capabilities and the per-packet
/// TX path match what the hardware is doing.
#[derive(Clone, Copy, Debug, Default)]
pub struct OffloadCaps {
    pub tx: u64,
    pub rx: u64,
}

impl OffloadCaps {
    #[inline]
    pub fn tx_has(&self, bit: u64) -> bool {
        self.tx & bit != 0
    }
    #[inline]
    pub fn rx_has(&self, bit: u64) -> bool {
        self.rx & bit != 0
    }
    #[inline]
    pub fn tx_ipv4_cksum(&self) -> bool {
        self.tx_has(RTE_ETH_TX_OFFLOAD_IPV4_CKSUM)
    }
    #[inline]
    pub fn tx_tcp_cksum(&self) -> bool {
        self.tx_has(RTE_ETH_TX_OFFLOAD_TCP_CKSUM)
    }
    #[inline]
    pub fn tx_udp_cksum(&self) -> bool {
        self.tx_has(RTE_ETH_TX_OFFLOAD_UDP_CKSUM)
    }
    #[inline]
    pub fn tx_tso(&self) -> bool {
        self.tx_has(RTE_ETH_TX_OFFLOAD_TCP_TSO)
    }
    #[inline]
    pub fn rx_ipv4_cksum(&self) -> bool {
        self.rx_has(RTE_ETH_RX_OFFLOAD_IPV4_CKSUM)
    }
    #[inline]
    pub fn rx_tcp_cksum(&self) -> bool {
        self.rx_has(RTE_ETH_RX_OFFLOAD_TCP_CKSUM)
    }
    #[inline]
    pub fn rx_udp_cksum(&self) -> bool {
        self.rx_has(RTE_ETH_RX_OFFLOAD_UDP_CKSUM)
    }
}

#[derive(Clone, Copy)]
pub struct PortConfig {
    pub rx_queues: u16,
    pub tx_queues: u16,
    pub rx_descriptors: u16,
    pub tx_descriptors: u16,
    /// RSS hash function bitmask. Set to 0 to leave RSS disabled
    /// (single-queue mode). Defaults to TCP+IP 5-tuple.
    pub rss_hf: u64,
    /// Max transmission unit (payload bytes per frame, not including
    /// the 14-byte Ethernet header). 1500 = classic Ethernet, 9000–
    /// 9100 = jumbo frames. Required end-to-end LAN MTU support to
    /// be useful. The mempool's mbuf buf_size must be ≥ MTU + 14 +
    /// headroom (128).
    pub mtu: u16,
    /// TX offloads to request. Intersected with the PMD's advertised
    /// `tx_offload_capa` at configure time; only the supported subset
    /// is enabled at both port and per-queue level.
    pub desired_tx_offloads: u64,
    /// RX offloads to request. Intersected with `rx_offload_capa`.
    pub desired_rx_offloads: u64,
}

impl Default for PortConfig {
    fn default() -> Self {
        Self {
            rx_queues: 1,
            tx_queues: 1,
            // 4096 — the sweet spot. Tried bumping to mlx5's max
            // (32768) on a -c50000 wrk run; throughput dropped from
            // 2580 to 1605 r/s. The larger ring + mempool needed to
            // back it (918K mbufs ≈ 1.8 GB hugepages) blow out L2/L3
            // cache for the workers, so per-packet cost goes up
            // everywhere and that swamps the rare burst-absorption
            // win. 4096 absorbs typical SYN bursts cleanly and keeps
            // the worker's hot path resident in cache.
            rx_descriptors: 4096,
            tx_descriptors: 4096,
            rss_hf: RSS_HF_TCP,
            mtu: 1500,
            desired_tx_offloads: DESIRED_TX_OFFLOADS,
            desired_rx_offloads: DESIRED_RX_OFFLOADS,
        }
    }
}

/// Owns the configured & started state of one NIC port. Drop stops &
/// closes the port.
pub struct Port {
    id: u16,
    cfg: PortConfig,
    offloads: OffloadCaps,
}

impl Port {
    /// Configure + start `port_id` with the given queue counts. Each
    /// rx queue is wired to `rx_pool` for fresh-buffer DMA.
    pub fn configure_and_start(port_id: u16, cfg: PortConfig, rx_pool: &Mempool) -> Result<Self> {
        // Most fields default to zero; we only need to flip on RSS
        // when the caller asked for multiple rx queues so packets
        // spread across them.
        let mut port_conf: ffi::rte_eth_conf = unsafe { std::mem::zeroed() };
        if cfg.rx_queues > 1 && cfg.rss_hf != 0 {
            port_conf.rxmode.mq_mode = ffi::rte_eth_rx_mq_mode_RTE_ETH_MQ_RX_RSS;
            // mlx5 ignores rss_key when len is 0, using a built-in
            // Toeplitz key. That's the right default — no custom key.
            port_conf.rx_adv_conf.rss_conf.rss_hf = cfg.rss_hf;
            port_conf.rx_adv_conf.rss_conf.rss_key = std::ptr::null_mut();
            port_conf.rx_adv_conf.rss_conf.rss_key_len = 0;
        }
        // MTU: rte_eth_dev_configure picks this up from rxmode in
        // DPDK 22+. Bigger frames = fewer packets per byte = lower
        // per-packet processing cost. Caller is responsible for
        // ensuring the mempool's mbuf buf_size can hold the frame.
        port_conf.rxmode.mtu = cfg.mtu as u32;

        // Auto-detect hardware offloads: read what the PMD advertises
        // and enable only the intersection with what we requested. This
        // is the foundation for moving checksum + segmentation work off
        // the CPU — anything the NIC can do, we let it.
        //
        // CRITICAL (mlx5): an offload bit must be set BOTH at port level
        // (`txmode/rxmode.offloads`) AND per queue (`txconf/rxconf.offloads`)
        // or the queue won't honor it. Earlier this code passed a NULL
        // queue config, so per-queue offloads were never requested. We
        // copy the PMD's recommended `default_{tx,rx}conf` and OR in the
        // enabled set.
        //
        // TSO note: `desired_tx_offloads` only includes
        // RTE_ETH_TX_OFFLOAD_TCP_TSO when the caller (boot.rs) has the
        // full TSO data path live and has primed the mlx5 quota context.
        // Advertising TSO while never emitting a TSO-flagged mbuf
        // SIGSEGVs the PMD in `mlx5_flow_quota_init` (NULL deref at
        // +0x44 on the never-populated quota context).
        let mut info: ffi::rte_eth_dev_info = unsafe { std::mem::zeroed() };
        let rc = unsafe { ffi::rte_eth_dev_info_get(port_id, &mut info as *mut _) };
        if rc < 0 {
            return Err(Error::from_rte("rte_eth_dev_info_get"));
        }
        let enabled_tx = cfg.desired_tx_offloads & info.tx_offload_capa;
        let enabled_rx = cfg.desired_rx_offloads & info.rx_offload_capa;
        let offloads = OffloadCaps {
            tx: enabled_tx,
            rx: enabled_rx,
        };
        port_conf.txmode.offloads = enabled_tx;
        port_conf.rxmode.offloads |= enabled_rx;

        let rc = unsafe {
            ffi::rte_eth_dev_configure(
                port_id,
                cfg.rx_queues,
                cfg.tx_queues,
                &port_conf as *const _,
            )
        };
        if rc < 0 {
            return Err(Error::from_rte("rte_eth_dev_configure"));
        }

        for q in 0..cfg.rx_queues {
            // Copy the PMD's recommended rxconf and request our offloads
            // on this queue (per-queue is mandatory; see above).
            let mut rx_conf = info.default_rxconf;
            rx_conf.offloads = enabled_rx;
            let rc = unsafe {
                ffi::rte_eth_rx_queue_setup(
                    port_id,
                    q,
                    cfg.rx_descriptors,
                    ffi::rte_eth_dev_socket_id(port_id) as c_uint,
                    &rx_conf as *const _,
                    rx_pool.as_ptr(),
                )
            };
            if rc < 0 {
                return Err(Error::from_rte("rte_eth_rx_queue_setup"));
            }
        }

        for q in 0..cfg.tx_queues {
            let mut tx_conf = info.default_txconf;
            tx_conf.offloads = enabled_tx;
            let rc = unsafe {
                ffi::rte_eth_tx_queue_setup(
                    port_id,
                    q,
                    cfg.tx_descriptors,
                    ffi::rte_eth_dev_socket_id(port_id) as c_uint,
                    &tx_conf as *const _,
                )
            };
            if rc < 0 {
                return Err(Error::from_rte("rte_eth_tx_queue_setup"));
            }
        }

        let rc = unsafe { ffi::rte_eth_dev_start(port_id) };
        if rc < 0 {
            return Err(Error::from_rte("rte_eth_dev_start"));
        }
        // Promiscuous mode so we receive frames not addressed to our
        // MAC — useful while developing on a host that's not the
        // primary owner of the NIC.
        unsafe { ffi::rte_eth_promiscuous_enable(port_id) };

        Ok(Self {
            id: port_id,
            cfg,
            offloads,
        })
    }

    pub fn id(&self) -> u16 {
        self.id
    }

    pub fn config(&self) -> PortConfig {
        self.cfg
    }

    /// The hardware offloads actually enabled on this port (the
    /// intersection of what we requested and what the PMD advertises).
    pub fn offloads(&self) -> OffloadCaps {
        self.offloads
    }

    /// Whether the PMD reports the given `RTE_ETH_TX_OFFLOAD_*` bit
    /// in its `tx_offload_capa`. Caller passes the bit value (e.g.
    /// 0x20 for RTE_ETH_TX_OFFLOAD_TCP_TSO).
    pub fn tx_offload_supported(&self, offload_bit: u64) -> crate::Result<bool> {
        let mut info: ffi::rte_eth_dev_info = unsafe { std::mem::zeroed() };
        let rc = unsafe { ffi::rte_eth_dev_info_get(self.id, &mut info as *mut _) };
        if rc < 0 {
            return Err(crate::Error::from_rte("rte_eth_dev_info_get"));
        }
        Ok((info.tx_offload_capa & offload_bit) != 0)
    }

    /// Read the hardware MAC address as configured on the bound port.
    /// Required for smoltcp's `HardwareAddress::Ethernet` construction.
    pub fn mac_address(&self) -> crate::Result<[u8; 6]> {
        let mut addr: ffi::rte_ether_addr = unsafe { std::mem::zeroed() };
        let rc = unsafe { ffi::rte_eth_macaddr_get(self.id, &mut addr as *mut _) };
        if rc < 0 {
            return Err(crate::Error::from_rte("rte_eth_macaddr_get"));
        }
        Ok(addr.addr_bytes)
    }
}

impl Drop for Port {
    fn drop(&mut self) {
        unsafe {
            ffi::rte_eth_dev_stop(self.id);
            ffi::rte_eth_dev_close(self.id);
        }
    }
}
