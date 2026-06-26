/*
 * Thin C shim over DPDK's static-inline hot path and rte_flow setup.
 *
 * We expose three classes of symbol:
 *
 *   1. `shim_eth_{rx,tx}_burst` and `shim_pktmbuf_*` — wrappers for
 *      static-inline functions that bindgen can't see. Implemented as
 *      direct one-line forwards; LTO eliminates the call.
 *
 *   2. `shim_install_tcp_port_flow_rule` — installs an rte_flow rule
 *      that steers TCP/<dst_port> traffic to a specific RX queue.
 *      Encapsulated here because the rte_flow_item/action layout is
 *      a union-heavy mess that's painful to construct from Rust.
 */

#include "wrapper.h"
#include <string.h>
#include <stdio.h>

uint16_t shim_eth_rx_burst(uint16_t port_id, uint16_t queue_id,
                           struct rte_mbuf **mbufs, uint16_t nb)
{
    return rte_eth_rx_burst(port_id, queue_id, mbufs, nb);
}

uint16_t shim_eth_tx_burst(uint16_t port_id, uint16_t queue_id,
                           struct rte_mbuf **mbufs, uint16_t nb)
{
    return rte_eth_tx_burst(port_id, queue_id, mbufs, nb);
}

struct rte_mbuf *shim_pktmbuf_alloc(struct rte_mempool *pool)
{
    return rte_pktmbuf_alloc(pool);
}

void shim_pktmbuf_free(struct rte_mbuf *m)
{
    rte_pktmbuf_free(m);
}

uint8_t *shim_pktmbuf_mtod(struct rte_mbuf *m)
{
    if (!m) return NULL;
    return rte_pktmbuf_mtod(m, uint8_t *);
}

uint16_t shim_pktmbuf_data_len(struct rte_mbuf *m)
{
    if (!m) return 0;
    return m->data_len;
}

void shim_pktmbuf_set_data_len(struct rte_mbuf *m, uint16_t len)
{
    if (!m) return;
    m->data_len = len;
}

void shim_pktmbuf_set_pkt_len(struct rte_mbuf *m, uint32_t len)
{
    if (!m) return;
    m->pkt_len = len;
}

uint16_t shim_pktmbuf_tailroom(struct rte_mbuf *m)
{
    if (!m) return 0;
    return rte_pktmbuf_tailroom(m);
}

int shim_rte_errno(void)
{
    return rte_errno;
}

void shim_mbuf_set_tx_offload_v4tcp(struct rte_mbuf *m,
                                    uint16_t l2_len,
                                    uint16_t l3_len)
{
    if (!m) return;
    m->l2_len = l2_len;
    m->l3_len = l3_len;
    m->ol_flags |= (RTE_MBUF_F_TX_IP_CKSUM
                    | RTE_MBUF_F_TX_TCP_CKSUM
                    | RTE_MBUF_F_TX_IPV4);
}

void shim_mbuf_set_tx_offload_v4(struct rte_mbuf *m,
                                 uint16_t l2_len,
                                 uint16_t l3_len)
{
    if (!m) return;
    m->l2_len = l2_len;
    m->l3_len = l3_len;
    /* IPv4 header checksum only, NO L4 offload. For non-TCP IPv4 packets
       (ICMP, UDP) whose L4 checksum smoltcp computes in software. Setting
       RTE_MBUF_F_TX_TCP_CKSUM on a non-TCP packet makes the NIC write a
       bogus "TCP checksum" into the payload at the TCP-checksum offset,
       corrupting the frame so the peer drops it (this is why ping failed
       while TCP worked). */
    m->ol_flags |= (RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_IPV4);
}

void shim_mbuf_set_tx_offload_v4tcp_tso(struct rte_mbuf *m,
                                        uint16_t l2_len,
                                        uint16_t l3_len,
                                        uint16_t l4_len,
                                        uint16_t tso_segsz)
{
    if (!m) return;
    m->l2_len = l2_len;
    m->l3_len = l3_len;
    m->l4_len = l4_len;
    m->tso_segsz = tso_segsz;
    m->ol_flags |= (RTE_MBUF_F_TX_IP_CKSUM
                    | RTE_MBUF_F_TX_TCP_CKSUM
                    | RTE_MBUF_F_TX_TCP_SEG
                    | RTE_MBUF_F_TX_IPV4);
}

/* Read an mbuf's offload flags. For RX, these carry the NIC's checksum
 * verdicts (RTE_MBUF_F_RX_IP_CKSUM_*, RTE_MBUF_F_RX_L4_CKSUM_*) so the
 * caller can drop frames the hardware flagged as corrupt. */
uint64_t shim_mbuf_ol_flags(struct rte_mbuf *m)
{
    if (!m) return 0;
    return m->ol_flags;
}

/* IPv4 + UDP checksum TX offload, for UDP/IPv4 packets. */
void shim_mbuf_set_tx_offload_v4udp(struct rte_mbuf *m,
                                    uint16_t l2_len,
                                    uint16_t l3_len)
{
    if (!m) return;
    m->l2_len = l2_len;
    m->l3_len = l3_len;
    m->ol_flags |= (RTE_MBUF_F_TX_IP_CKSUM
                    | RTE_MBUF_F_TX_UDP_CKSUM
                    | RTE_MBUF_F_TX_IPV4);
}

/* Prepare an IPv4/TCP mbuf for hardware TCP segmentation offload (TSO),
 * the correct way (the parked hand-rolled patch in device.rs got the
 * pseudo-header wrong). In addition to the offload bits + header lengths
 * + per-segment size, the mlx5 TSO contract requires:
 *   - the IPv4 header checksum field = 0 (NIC fills it per segment), and
 *   - the TCP checksum field = the IPv4 PSEUDO-HEADER checksum WITHOUT
 *     the length term (the NIC adds each segment's length + payload sum).
 * `rte_ipv4_phdr_cksum(ip, ol_flags)` produces exactly that when the
 * TCP_SEG flag is set in ol_flags. Getting this wrong is the classic
 * "TSO silently does nothing / frames dropped downstream" failure.
 *
 * `l2_len`/`l3_len`/`l4_len` are the Ethernet/IPv4/TCP header byte
 * lengths (14/20/20 for our frames); `mss` is the on-wire per-segment
 * payload size (the connection's negotiated MSS). */
void shim_mbuf_prepare_tso_v4tcp(struct rte_mbuf *m,
                                 uint16_t l2_len,
                                 uint16_t l3_len,
                                 uint16_t l4_len,
                                 uint16_t mss)
{
    if (!m) return;
    m->l2_len = l2_len;
    m->l3_len = l3_len;
    m->l4_len = l4_len;
    m->tso_segsz = mss;
    m->ol_flags |= (RTE_MBUF_F_TX_IP_CKSUM
                    | RTE_MBUF_F_TX_TCP_CKSUM
                    | RTE_MBUF_F_TX_TCP_SEG
                    | RTE_MBUF_F_TX_IPV4);

    uint8_t *data = rte_pktmbuf_mtod(m, uint8_t *);
    struct rte_ipv4_hdr *ip = (struct rte_ipv4_hdr *)(data + l2_len);
    struct rte_tcp_hdr *tcp = (struct rte_tcp_hdr *)(data + l2_len + l3_len);
    ip->hdr_checksum = 0;
    tcp->cksum = 0;
    tcp->cksum = rte_ipv4_phdr_cksum(ip, m->ol_flags);
}

/*
 * Build and install an rte_flow rule:
 *
 *   pattern = ETH / IPv4 / TCP(dst=tcp_dst) / END
 *   action  = QUEUE(index=queue_id) / END
 *   attr    = ingress
 *
 * On mlx5 (ConnectX-4/5/6) this gets pushed into the NIC's flow
 * tables. Packets matching the TCP destination port are DMA'd directly
 * into our DPDK rx queue; everything else stays on the kernel TCP
 * stack (PMD bifurcates via the bonded interface).
 */
int shim_install_tcp_port_flow_rule(uint16_t port_id, uint16_t queue_id,
                                    uint16_t tcp_dst,
                                    char *err_buf, size_t err_cap)
{
    struct rte_flow_attr attr;
    struct rte_flow_item pattern[4];
    struct rte_flow_action actions[2];
    struct rte_flow_action_queue queue;
    struct rte_flow_item_tcp tcp_spec;
    struct rte_flow_item_tcp tcp_mask;
    struct rte_flow_error error;
    struct rte_flow *flow;

    memset(&attr, 0, sizeof(attr));
    attr.ingress = 1;

    memset(pattern, 0, sizeof(pattern));
    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;

    memset(&tcp_spec, 0, sizeof(tcp_spec));
    memset(&tcp_mask, 0, sizeof(tcp_mask));
    tcp_spec.hdr.dst_port = rte_cpu_to_be_16(tcp_dst);
    tcp_mask.hdr.dst_port = 0xFFFF;

    pattern[2].type = RTE_FLOW_ITEM_TYPE_TCP;
    pattern[2].spec = &tcp_spec;
    pattern[2].mask = &tcp_mask;

    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

    memset(actions, 0, sizeof(actions));
    memset(&queue, 0, sizeof(queue));
    queue.index = queue_id;
    actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
    actions[0].conf = &queue;
    actions[1].type = RTE_FLOW_ACTION_TYPE_END;

    memset(&error, 0, sizeof(error));

    /* Validate before create so a bad rule comes back with a useful
     * diagnostic, not just -EIO. */
    int rc = rte_flow_validate(port_id, &attr, pattern, actions, &error);
    if (rc != 0) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "rte_flow_validate: %s",
                     error.message ? error.message : "(no message)");
        }
        return rc;
    }

    flow = rte_flow_create(port_id, &attr, pattern, actions, &error);
    if (flow == NULL) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "rte_flow_create: %s",
                     error.message ? error.message : "(no message)");
        }
        return -1;
    }

    return 0;
}

/*
 * Generic single-queue installer used by the ARP and ICMP rules.
 * `pattern` must already be a zero-terminated rte_flow_item array
 * ending in RTE_FLOW_ITEM_TYPE_END.
 */
static int install_pattern_to_queue(uint16_t port_id, uint16_t queue_id,
                                    struct rte_flow_item *pattern,
                                    char *err_buf, size_t err_cap)
{
    struct rte_flow_attr attr;
    struct rte_flow_action actions[2];
    struct rte_flow_action_queue queue;
    struct rte_flow_error error;
    struct rte_flow *flow;

    memset(&attr, 0, sizeof(attr));
    attr.ingress = 1;

    memset(actions, 0, sizeof(actions));
    memset(&queue, 0, sizeof(queue));
    queue.index = queue_id;
    actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
    actions[0].conf = &queue;
    actions[1].type = RTE_FLOW_ACTION_TYPE_END;

    memset(&error, 0, sizeof(error));

    int rc = rte_flow_validate(port_id, &attr, pattern, actions, &error);
    if (rc != 0) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "rte_flow_validate: %s",
                     error.message ? error.message : "(no message)");
        }
        return rc;
    }

    flow = rte_flow_create(port_id, &attr, pattern, actions, &error);
    if (flow == NULL) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "rte_flow_create: %s",
                     error.message ? error.message : "(no message)");
        }
        return -1;
    }
    return 0;
}

int shim_install_arp_to_queue_rule(uint16_t port_id, uint16_t queue_id,
                                   char *err_buf, size_t err_cap)
{
    struct rte_flow_item pattern[3];
    struct rte_flow_item_eth eth_spec;
    struct rte_flow_item_eth eth_mask;

    memset(pattern, 0, sizeof(pattern));
    memset(&eth_spec, 0, sizeof(eth_spec));
    memset(&eth_mask, 0, sizeof(eth_mask));

    /* Match EtherType 0x0806 (ARP). MAC addresses left zero -> wildcard. */
    eth_spec.hdr.ether_type = rte_cpu_to_be_16(0x0806);
    eth_mask.hdr.ether_type = 0xFFFF;

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[0].spec = &eth_spec;
    pattern[0].mask = &eth_mask;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_END;

    return install_pattern_to_queue(port_id, queue_id, pattern,
                                    err_buf, err_cap);
}

int shim_install_icmp_to_queue_rule(uint16_t port_id, uint16_t queue_id,
                                    char *err_buf, size_t err_cap)
{
    struct rte_flow_item pattern[3];
    struct rte_flow_item_ipv4 ipv4_spec;
    struct rte_flow_item_ipv4 ipv4_mask;

    memset(pattern, 0, sizeof(pattern));
    memset(&ipv4_spec, 0, sizeof(ipv4_spec));
    memset(&ipv4_mask, 0, sizeof(ipv4_mask));

    /* Match IPv4 protocol 1 (ICMP). All other IPv4 fields wildcarded. */
    ipv4_spec.hdr.next_proto_id = 1;
    ipv4_mask.hdr.next_proto_id = 0xFF;

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[1].spec = &ipv4_spec;
    pattern[1].mask = &ipv4_mask;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_END;

    return install_pattern_to_queue(port_id, queue_id, pattern,
                                    err_buf, err_cap);
}

static int install_pattern_to_rss(uint16_t port_id,
                                  uint16_t queue_count,
                                  const uint16_t *queues,
                                  struct rte_flow_item *pattern,
                                  uint64_t rss_types,
                                  const char *label,
                                  char *err_buf, size_t err_cap)
{
    struct rte_flow_attr attr;
    struct rte_flow_action actions[2];
    struct rte_flow_action_rss rss;
    struct rte_flow_error error;
    struct rte_flow *flow;

    if (queue_count == 0 || queues == NULL) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap,
                     "tcp_rss: queue_count==0 or queues==NULL");
        }
        return -1;
    }

    memset(&attr, 0, sizeof(attr));
    attr.ingress = 1;

    memset(&actions, 0, sizeof(actions));
    memset(&rss, 0, sizeof(rss));
    rss.func = RTE_ETH_HASH_FUNCTION_DEFAULT;
    rss.level = 0;
    rss.types = rss_types;
    rss.queue_num = queue_count;
    rss.queue = queues;       /* caller keeps this alive across the call */
    rss.key = NULL;
    rss.key_len = 0;

    actions[0].type = RTE_FLOW_ACTION_TYPE_RSS;
    actions[0].conf = &rss;
    actions[1].type = RTE_FLOW_ACTION_TYPE_END;

    memset(&error, 0, sizeof(error));

    int rc = rte_flow_validate(port_id, &attr, pattern, actions, &error);
    if (rc != 0) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "rte_flow_validate(%s): %s",
                     label, error.message ? error.message : "(no message)");
        }
        return rc;
    }

    flow = rte_flow_create(port_id, &attr, pattern, actions, &error);
    if (flow == NULL) {
        if (err_buf && err_cap > 0) {
            snprintf(err_buf, err_cap, "rte_flow_create(%s): %s",
                     label, error.message ? error.message : "(no message)");
        }
        return -1;
    }
    return 0;
}

/*
 * TCP + RSS rule: pattern matches any TCP/IPv4 packet, action spreads
 * across the caller-provided queue list using the PMD's default key
 * and the standard 5-tuple hash for TCP. mlx5 supports this shape
 * since ConnectX-4.
 */
int shim_install_tcp_rss_flow_rule(uint16_t port_id,
                                   uint16_t queue_count,
                                   const uint16_t *queues,
                                   char *err_buf, size_t err_cap)
{
    struct rte_flow_item pattern[4];

    memset(pattern, 0, sizeof(pattern));
    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_TCP;
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

    return install_pattern_to_rss(port_id, queue_count, queues, pattern,
                                  RTE_ETH_RSS_NONFRAG_IPV4_TCP,
                                  "tcp_rss", err_buf, err_cap);
}

/*
 * UDP + RSS rule for WebTransport: pattern matches IPv4 UDP packets with
 * destination port `udp_dst`, action spreads across the caller-provided queue
 * list using the PMD's default key and the standard UDP 4-tuple hash.
 */
int shim_install_udp_rss_flow_rule(uint16_t port_id,
                                   uint16_t udp_dst,
                                   uint16_t queue_count,
                                   const uint16_t *queues,
                                   char *err_buf, size_t err_cap)
{
    struct rte_flow_item pattern[4];
    struct rte_flow_item_udp udp_spec;
    struct rte_flow_item_udp udp_mask;

    memset(pattern, 0, sizeof(pattern));
    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;

    memset(&udp_spec, 0, sizeof(udp_spec));
    memset(&udp_mask, 0, sizeof(udp_mask));
    udp_spec.hdr.dst_port = rte_cpu_to_be_16(udp_dst);
    udp_mask.hdr.dst_port = 0xFFFF;

    pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
    pattern[2].spec = &udp_spec;
    pattern[2].mask = &udp_mask;
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

    return install_pattern_to_rss(port_id, queue_count, queues, pattern,
                                  RTE_ETH_RSS_NONFRAG_IPV4_UDP,
                                  "udp_rss", err_buf, err_cap);
}
