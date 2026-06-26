/*
 * Headers we want Rust FFI for. Order matters: DPDK assumes
 * rte_config.h is force-included (we do that via -include in build.rs).
 *
 * We deliberately keep this list short. Each entry pulls in a bunch of
 * transitive headers and bindgen's allowlist filters everything down to
 * the `rte_*` symbols we actually use.
 */

#ifndef DPDK_SHIM_WRAPPER_H
#define DPDK_SHIM_WRAPPER_H

#include <rte_eal.h>
#include <rte_errno.h>
#include <rte_lcore.h>

#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
#include <rte_mbuf_core.h>

#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_tcp.h>
#include <rte_udp.h>

#include <rte_flow.h>

/*
 * The hot-path inline accessors live in shim.c so bindgen can see them
 * as ordinary extern "C" functions. Declarations here so Rust callers
 * (via this same wrapper.h) see consistent signatures.
 */

/* Burst rx into `mbufs[0..nb]`. Returns number actually received. */
uint16_t shim_eth_rx_burst(uint16_t port_id, uint16_t queue_id,
                           struct rte_mbuf **mbufs, uint16_t nb);

/* Burst tx of `mbufs[0..nb]`. Returns number actually queued. */
uint16_t shim_eth_tx_burst(uint16_t port_id, uint16_t queue_id,
                           struct rte_mbuf **mbufs, uint16_t nb);

/* Alloc one mbuf from `pool`. NULL on exhaustion. */
struct rte_mbuf *shim_pktmbuf_alloc(struct rte_mempool *pool);

/* Free one mbuf back to its pool. */
void shim_pktmbuf_free(struct rte_mbuf *m);

/* Pointer to the writable data area of `m`, or NULL if zero-len. */
uint8_t *shim_pktmbuf_mtod(struct rte_mbuf *m);

/* Length of the first segment's data region. */
uint16_t shim_pktmbuf_data_len(struct rte_mbuf *m);

/* Set data_len / pkt_len after a write. */
void shim_pktmbuf_set_data_len(struct rte_mbuf *m, uint16_t len);
void shim_pktmbuf_set_pkt_len(struct rte_mbuf *m, uint32_t len);

/* Writable tailroom on the first segment (buf_len - data_off - data_len).
 * Useful when the caller wants to fill a fresh mbuf from offset zero. */
uint16_t shim_pktmbuf_tailroom(struct rte_mbuf *m);

/* Per-lcore rte_errno value. DPDK exposes this as a macro that
 * bindgen can't reach. Returns the value at call time. */
int shim_rte_errno(void);

/* Annotate an outgoing mbuf for IPv4+TCP checksum offload.
 *
 *   * mbuf->l2_len = Ethernet header bytes (14 for plain frames)
 *   * mbuf->l3_len = IP header bytes (20 for IPv4 with no options)
 *   * mbuf->ol_flags |= RTE_MBUF_F_TX_IP_CKSUM
 *                    | RTE_MBUF_F_TX_TCP_CKSUM
 *                    | RTE_MBUF_F_TX_IPV4
 *
 * The NIC will compute and write the IPv4 + TCP checksums on the
 * wire, overriding whatever bytes were in the packet at the relevant
 * offsets. Combined with telling smoltcp the iface trusts the
 * device for tx checksums, this skips ~1 MiB of CPU-side checksum
 * work per 1 MiB response.
 */
void shim_mbuf_set_tx_offload_v4tcp(struct rte_mbuf *m,
                                    uint16_t l2_len,
                                    uint16_t l3_len);

/* IPv4-header-checksum-only TX offload (no L4 offload), for non-TCP IPv4
 * packets (ICMP, UDP). smoltcp computes the L4 checksum in software for
 * these; requesting a TCP checksum offload on a non-TCP packet corrupts
 * it. Use this instead of `shim_mbuf_set_tx_offload_v4tcp` for ICMP/UDP.
 */
void shim_mbuf_set_tx_offload_v4(struct rte_mbuf *m,
                                 uint16_t l2_len,
                                 uint16_t l3_len);

/* TSO variant. In addition to IP+TCP checksum offload, asks the NIC to
 * split the mbuf's payload into per-`tso_segsz` TCP segments on the
 * wire and regenerate the TCP/IP headers for each one. The mbuf's
 * single TCP header is treated as the template; the device fixes up
 * seq, IP id, and checksums per slice.
 *
 *   * l2_len: Ethernet header bytes (14 for plain Ethernet)
 *   * l3_len: IP header bytes (20 for IPv4 with no options)
 *   * l4_len: TCP header bytes (20 typical, no TCP options used)
 *   * tso_segsz: per-segment payload size in bytes (e.g. 9046 for an
 *     MTU 9100 setup: 9100 - 14 - 20 - 20)
 *
 * Sets ol_flags: RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM |
 * RTE_MBUF_F_TX_TCP_SEG | RTE_MBUF_F_TX_IPV4.
 *
 * MUST be called before the mbuf is queued for tx_burst, and the port
 * MUST have RTE_ETH_TX_OFFLOAD_TCP_TSO enabled at configure time.
 */
void shim_mbuf_set_tx_offload_v4tcp_tso(struct rte_mbuf *m,
                                        uint16_t l2_len,
                                        uint16_t l3_len,
                                        uint16_t l4_len,
                                        uint16_t tso_segsz);

/* Read an mbuf's ol_flags (RX checksum verdicts live here). */
uint64_t shim_mbuf_ol_flags(struct rte_mbuf *m);

/* IPv4 + UDP checksum TX offload for UDP/IPv4 packets. */
void shim_mbuf_set_tx_offload_v4udp(struct rte_mbuf *m,
                                    uint16_t l2_len,
                                    uint16_t l3_len);

/* Prepare an IPv4/TCP mbuf for hardware TSO with the correct
 * pseudo-header checksum (via rte_ipv4_phdr_cksum). `mss` is the
 * on-wire per-segment payload size (the connection's negotiated MSS). */
void shim_mbuf_prepare_tso_v4tcp(struct rte_mbuf *m,
                                 uint16_t l2_len,
                                 uint16_t l3_len,
                                 uint16_t l4_len,
                                 uint16_t mss);

/*
 * Install a flow rule: TCP destination port == tcp_dst -> rx queue
 * `queue_id`. Returns 0 on success, negative rte_errno on failure.
 * Writes a textual error into err_buf (capacity err_cap) on failure.
 */
int shim_install_tcp_port_flow_rule(uint16_t port_id, uint16_t queue_id,
                                    uint16_t tcp_dst,
                                    char *err_buf, size_t err_cap);

/*
 * Install a flow rule: any frame whose Ethernet type is 0x0806 (ARP)
 * lands in rx queue `queue_id`. Required in mlx5 bifurcated mode so
 * smoltcp can answer ARP for the IP it owns.
 */
int shim_install_arp_to_queue_rule(uint16_t port_id, uint16_t queue_id,
                                   char *err_buf, size_t err_cap);

/*
 * Install a flow rule: any IPv4 packet whose `next_proto_id` is 1
 * (ICMP) lands in rx queue `queue_id`. Convenience for `ping` probes;
 * not load-bearing for HTTP.
 */
int shim_install_icmp_to_queue_rule(uint16_t port_id, uint16_t queue_id,
                                    char *err_buf, size_t err_cap);

/*
 * Install a flow rule: any IPv4 TCP packet is hashed across the queue
 * list `queues[0..queue_count]` using the PMD's default Toeplitz key
 * and `RTE_ETH_RSS_NONFRAG_IPV4_TCP` types.
 *
 * Required in mlx5 bifurcated mode for TCP traffic to reach DPDK at
 * all. The PortConfig-level RSS setup decides what hashing is used
 * for packets that arrive on DPDK queues; this rule decides which
 * packets arrive on DPDK queues vs. the kernel.
 */
int shim_install_tcp_rss_flow_rule(uint16_t port_id,
                                   uint16_t queue_count,
                                   const uint16_t *queues,
                                   char *err_buf, size_t err_cap);

/*
 * Install a flow rule: IPv4 UDP packets whose destination port is
 * `udp_dst` are hashed across `queues[0..queue_count]` using
 * `RTE_ETH_RSS_NONFRAG_IPV4_UDP`.
 *
 * This is intentionally destination-port scoped for WebTransport
 * (UDP/443), so enabling WT does not steal unrelated UDP traffic from
 * the kernel side of a bifurcated mlx5 setup.
 */
int shim_install_udp_rss_flow_rule(uint16_t port_id,
                                   uint16_t udp_dst,
                                   uint16_t queue_count,
                                   const uint16_t *queues,
                                   char *err_buf, size_t err_cap);

#endif /* DPDK_SHIM_WRAPPER_H */
