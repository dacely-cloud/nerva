//! Minimal DPDK rx loop. Validates that:
//!
//!   1. EAL initialises against the system DPDK 25.11
//!   2. The mlx5 PMD attaches to the bound ConnectX-4 PCI device
//!   3. A single rx queue can be created and polled in a loop
//!   4. Packet bytes flow from the NIC DMA region into a Rust slice
//!      without an intermediate copy
//!
//! Not a server. Receives every packet that the NIC's flow rules steer
//! to queue 0 of the bound port, prints packets/sec + bytes/sec every
//! second, and exits on Ctrl+C.
//!
//! Run (requires root for hugepages + RDMA verbs):
//!
//!   sudo ./target/release/examples/dpdk_rx_demo -a 0000:17:00.0
//!
//! By default we allowlist the inactive Mellanox port so the active
//! port keeps serving HTTP. Pass `-a 0000:17:00.1` to attach to the
//! active port; DPDK will not steal traffic from the kernel unless an
//! RTE_FLOW rule is installed (see `--flow-port`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use dpdk_shim::eal::{Eal, EalArgs};
use dpdk_shim::flow::install_tcp_port_flow_rule;
use dpdk_shim::mempool::Mempool;
use dpdk_shim::port::config::PortConfig;
use dpdk_shim::port::device::Port;
use dpdk_shim::queue::RxQueue;

fn main() -> anyhow::Result<()> {
    // Pull off our own --flow-port=<u16> flag before handing the rest
    // to EAL. `--flow-port` installs a TCP-dst-port -> queue-0 rule.
    let raw_args: Vec<String> = std::env::args().collect();
    let mut flow_port: Option<u16> = None;
    let mut eal_args: Vec<String> = Vec::with_capacity(raw_args.len());
    let mut it = raw_args.into_iter();
    if let Some(a0) = it.next() {
        eal_args.push(a0);
    }
    while let Some(arg) = it.next() {
        if let Some(v) = arg.strip_prefix("--flow-port=") {
            flow_port = Some(v.parse().expect("--flow-port=<u16>"));
            continue;
        }
        if arg == "--flow-port" {
            let v = it.next().expect("--flow-port needs a value");
            flow_port = Some(v.parse().expect("--flow-port <u16>"));
            continue;
        }
        eal_args.push(arg);
    }

    // Defaults if no EAL args were given.
    if eal_args.len() == 1 {
        for d in ["-l", "0", "-n", "4", "-a", "0000:17:00.0", "--in-memory"] {
            eal_args.push(d.to_string());
        }
    }

    let _eal = Eal::init(EalArgs::new(&eal_args[0]).extend(eal_args[1..].iter().cloned()))?;
    println!("EAL initialised ({} port(s) probed)", _eal.port_count());

    // Build a single 8k mbuf pool shared by rx and tx, configure port
    // 0 with one queue pair, start it.
    let rx_pool = Mempool::new_packet("rx_pool", 8192)?;
    let _port = Port::configure_and_start(0, PortConfig::default(), &rx_pool)?;
    let rx = RxQueue::new(0, 0);
    println!("port 0 started, rx queue 0 ready");

    if let Some(p) = flow_port {
        install_tcp_port_flow_rule(0, 0, p)?;
        println!("installed flow rule: tcp/dst={p} -> rx queue 0");
    }

    let running = Arc::new(AtomicBool::new(true));
    let rfc = Arc::clone(&running);
    ctrlc::set_handler(move || {
        rfc.store(false, Ordering::Relaxed);
    })
    .ok();

    let mut last_print = Instant::now();
    let mut pkts_window: u64 = 0;
    let mut bytes_window: u64 = 0;
    let mut total_pkts: u64 = 0;
    let mut total_bytes: u64 = 0;

    println!("polling rx queue (Ctrl+C to stop)");
    while running.load(Ordering::Relaxed) {
        let batch = rx.receive_burst(32);
        for mbuf in batch {
            let len = mbuf.data().len() as u64;
            pkts_window += 1;
            bytes_window += len;
        }
        if last_print.elapsed().as_secs_f64() >= 1.0 {
            let secs = last_print.elapsed().as_secs_f64();
            total_pkts += pkts_window;
            total_bytes += bytes_window;
            println!(
                "rx: {:>10.0} pkt/s   {:>10.2} Mbit/s   (total: {} pkts, {} bytes)",
                pkts_window as f64 / secs,
                (bytes_window as f64 * 8.0 / 1e6) / secs,
                total_pkts,
                total_bytes,
            );
            pkts_window = 0;
            bytes_window = 0;
            last_print = Instant::now();
        }
    }
    println!("stopped, totals: {total_pkts} pkts, {total_bytes} bytes");
    Ok(())
}
