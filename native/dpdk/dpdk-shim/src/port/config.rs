//! User-facing NIC port configuration.

use crate::port::offload::{DESIRED_RX_OFFLOADS, DESIRED_TX_OFFLOADS};
use crate::port::rss::RSS_HF_TCP;

#[derive(Clone, Copy)]
pub struct PortConfig {
    pub rx_queues: u16,
    pub tx_queues: u16,
    pub rx_descriptors: u16,
    pub tx_descriptors: u16,
    pub rss_hf: u64,
    pub mtu: u16,
    pub desired_tx_offloads: u64,
    pub desired_rx_offloads: u64,
}

impl Default for PortConfig {
    fn default() -> Self {
        Self {
            rx_queues: 1,
            tx_queues: 1,
            rx_descriptors: 4096,
            tx_descriptors: 4096,
            rss_hf: RSS_HF_TCP,
            mtu: 1500,
            desired_tx_offloads: DESIRED_TX_OFFLOADS,
            desired_rx_offloads: DESIRED_RX_OFFLOADS,
        }
    }
}
