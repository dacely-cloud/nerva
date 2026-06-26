//! NIC checksum and segmentation offload declarations.

pub const RTE_ETH_TX_OFFLOAD_IPV4_CKSUM: u64 = 1 << 1;
pub const RTE_ETH_TX_OFFLOAD_UDP_CKSUM: u64 = 1 << 2;
pub const RTE_ETH_TX_OFFLOAD_TCP_CKSUM: u64 = 1 << 3;
pub const RTE_ETH_TX_OFFLOAD_TCP_TSO: u64 = 1 << 5;
pub const RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE: u64 = 1 << 16;

pub const RTE_ETH_RX_OFFLOAD_IPV4_CKSUM: u64 = 1 << 1;
pub const RTE_ETH_RX_OFFLOAD_UDP_CKSUM: u64 = 1 << 2;
pub const RTE_ETH_RX_OFFLOAD_TCP_CKSUM: u64 = 1 << 3;

pub const DESIRED_TX_OFFLOADS: u64 = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM
    | RTE_ETH_TX_OFFLOAD_UDP_CKSUM
    | RTE_ETH_TX_OFFLOAD_TCP_CKSUM
    | RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE;
pub const DESIRED_RX_OFFLOADS: u64 =
    RTE_ETH_RX_OFFLOAD_IPV4_CKSUM | RTE_ETH_RX_OFFLOAD_UDP_CKSUM | RTE_ETH_RX_OFFLOAD_TCP_CKSUM;

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
