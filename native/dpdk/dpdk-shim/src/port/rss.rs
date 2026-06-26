//! RSS hash selection for NIC receive queues.

pub const RTE_ETH_RSS_IPV4: u64 = 1 << 2;
pub const RTE_ETH_RSS_NONFRAG_IPV4_TCP: u64 = 1 << 4;
pub const RTE_ETH_RSS_NONFRAG_IPV4_UDP: u64 = 1 << 5;
pub const RTE_ETH_RSS_IPV6: u64 = 1 << 8;
pub const RTE_ETH_RSS_NONFRAG_IPV6_TCP: u64 = 1 << 10;
pub const RTE_ETH_RSS_NONFRAG_IPV6_UDP: u64 = 1 << 11;

pub const RSS_HF_TCP: u64 = RTE_ETH_RSS_NONFRAG_IPV4_TCP
    | RTE_ETH_RSS_NONFRAG_IPV6_TCP
    | RTE_ETH_RSS_IPV4
    | RTE_ETH_RSS_IPV6;
pub const RSS_HF_TCP_UDP: u64 =
    RSS_HF_TCP | RTE_ETH_RSS_NONFRAG_IPV4_UDP | RTE_ETH_RSS_NONFRAG_IPV6_UDP;
pub const RSS_HF_IP: u64 = RTE_ETH_RSS_IPV4 | RTE_ETH_RSS_IPV6;

pub fn rss_hf_from_alias(alias: &str) -> Option<u64> {
    Some(match alias {
        "tcp" => RSS_HF_TCP,
        "tcp_udp" => RSS_HF_TCP_UDP,
        "ip" => RSS_HF_IP,
        _ => return None,
    })
}
