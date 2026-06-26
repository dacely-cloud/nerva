//! rte_flow rule installation.
//!
//! Currently we only need a single shape: steer TCP packets with a
//! given destination port to a specific rx queue. The shim's C side
//! builds the rte_flow_pattern/action arrays (their union-heavy
//! layout is painful from Rust) and reports failures via a textual
//! error buffer.

use std::os::raw::c_char;

use crate::{Error, Result, ffi};

/// Install a rule: `tcp/dst == tcp_dst -> queue_id`. The rule sits on
/// the NIC's flow tables until the port is closed.
///
/// Returns `Ok(())` on success. On failure the message includes both
/// the textual error from DPDK and the rte_errno value.
pub fn install_tcp_port_flow_rule(port_id: u16, queue_id: u16, tcp_dst: u16) -> Result<()> {
    let mut buf = [0u8; 256];
    let rc = unsafe {
        ffi::shim_install_tcp_port_flow_rule(
            port_id,
            queue_id,
            tcp_dst,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as libc::size_t,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(decode_err(
        format_args!(
            "install_tcp_port_flow_rule(port {port_id}, queue {queue_id}, tcp_dst {tcp_dst})"
        ),
        &buf,
    ))
}

/// Install a rule: every ARP frame (EtherType 0x0806) lands in
/// `queue_id`. mlx5 in bifurcated mode needs this so smoltcp can
/// answer ARP for the IP it owns — without it, ARP goes to the
/// kernel which has no interface claiming the IP.
pub fn install_arp_to_queue_rule(port_id: u16, queue_id: u16) -> Result<()> {
    let mut buf = [0u8; 256];
    let rc = unsafe {
        ffi::shim_install_arp_to_queue_rule(
            port_id,
            queue_id,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as libc::size_t,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(decode_err(
        format_args!("install_arp_to_queue_rule(port {port_id}, queue {queue_id})"),
        &buf,
    ))
}

/// Install a rule: every IPv4 ICMP packet lands in `queue_id`.
/// Convenience for external `ping` probes; not required for HTTP.
pub fn install_icmp_to_queue_rule(port_id: u16, queue_id: u16) -> Result<()> {
    let mut buf = [0u8; 256];
    let rc = unsafe {
        ffi::shim_install_icmp_to_queue_rule(
            port_id,
            queue_id,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as libc::size_t,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(decode_err(
        format_args!("install_icmp_to_queue_rule(port {port_id}, queue {queue_id})"),
        &buf,
    ))
}

/// Install a rule: every TCP/IPv4 packet is RSS-hashed across
/// `queues` using the PMD's default Toeplitz key and the standard
/// 5-tuple TCP type. The pointed-to queue list is read by DPDK
/// during validate/create only; we own it across the call.
pub fn install_tcp_rss_flow_rule(port_id: u16, queues: &[u16]) -> Result<()> {
    ensure_queues("install_tcp_rss_flow_rule", queues)?;
    let mut buf = [0u8; 256];
    let rc = unsafe {
        ffi::shim_install_tcp_rss_flow_rule(
            port_id,
            queues.len() as u16,
            queues.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as libc::size_t,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(decode_err(
        format_args!(
            "install_tcp_rss_flow_rule(port {port_id}, queues=[{}..{}])",
            queues.first().copied().unwrap_or(0),
            queues.last().copied().unwrap_or(0)
        ),
        &buf,
    ))
}

/// Install a rule: IPv4 UDP packets with destination port `udp_dst` are
/// RSS-hashed across `queues` using the PMD's default Toeplitz key and the
/// standard UDP 4-tuple hash. Destination-port scoped so WebTransport
/// UDP/443 does not capture unrelated UDP traffic.
pub fn install_udp_rss_flow_rule(port_id: u16, udp_dst: u16, queues: &[u16]) -> Result<()> {
    ensure_queues("install_udp_rss_flow_rule", queues)?;
    let mut buf = [0u8; 256];
    let rc = unsafe {
        ffi::shim_install_udp_rss_flow_rule(
            port_id,
            udp_dst,
            queues.len() as u16,
            queues.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as libc::size_t,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(decode_err(
        format_args!(
            "install_udp_rss_flow_rule(port {port_id}, udp_dst {udp_dst}, queues=[{}..{}])",
            queues.first().copied().unwrap_or(0),
            queues.last().copied().unwrap_or(0)
        ),
        &buf,
    ))
}

fn ensure_queues(label: &str, queues: &[u16]) -> Result<()> {
    if queues.is_empty() {
        return Err(Error::other(format!("{label}: queues is empty")));
    }
    if queues.len() > u16::MAX as usize {
        return Err(Error::other(format!(
            "{label}: queue list length {} exceeds u16::MAX",
            queues.len()
        )));
    }
    Ok(())
}

fn decode_err(label: std::fmt::Arguments<'_>, buf: &[u8]) -> Error {
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let msg = String::from_utf8_lossy(&buf[..nul]).into_owned();
    Error::other(format!("{label}: {msg}"))
}
