use std::net::UdpSocket;
use std::time::Instant;

use nerva_core::types::error::{NervaError, Result};

pub(super) fn bind_loopback_socket() -> Result<UdpSocket> {
    UdpSocket::bind("127.0.0.1:0").map_err(io_error)
}

pub(super) fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

pub(super) fn io_error(err: std::io::Error) -> NervaError {
    NervaError::BackendUnavailable {
        backend: "kernel_udp_test",
        reason: err.to_string(),
    }
}
