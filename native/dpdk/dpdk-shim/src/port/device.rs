//! Started NIC port ownership and lifecycle.

use std::os::raw::c_uint;

use crate::mempool::Mempool;
use crate::port::config::PortConfig;
use crate::port::offload::OffloadCaps;
use crate::{Error, Result, ffi};

pub struct Port {
    id: u16,
    cfg: PortConfig,
    offloads: OffloadCaps,
}

impl Port {
    pub fn configure_and_start(port_id: u16, cfg: PortConfig, rx_pool: &Mempool) -> Result<Self> {
        let mut port_conf: ffi::rte_eth_conf = unsafe { std::mem::zeroed() };
        configure_rss(&mut port_conf, &cfg);
        port_conf.rxmode.mtu = cfg.mtu as u32;

        let info = device_info(port_id)?;
        let offloads = enabled_offloads(&cfg, &info);
        port_conf.txmode.offloads = offloads.tx;
        port_conf.rxmode.offloads |= offloads.rx;

        configure_device(port_id, &cfg, &port_conf)?;
        setup_rx_queues(port_id, &cfg, rx_pool, &info, offloads.rx)?;
        setup_tx_queues(port_id, &cfg, &info, offloads.tx)?;
        start_device(port_id)?;

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

    pub fn offloads(&self) -> OffloadCaps {
        self.offloads
    }

    pub fn tx_offload_supported(&self, offload_bit: u64) -> Result<bool> {
        let info = device_info(self.id)?;
        Ok((info.tx_offload_capa & offload_bit) != 0)
    }

    pub fn mac_address(&self) -> Result<[u8; 6]> {
        let mut addr: ffi::rte_ether_addr = unsafe { std::mem::zeroed() };
        let rc = unsafe { ffi::rte_eth_macaddr_get(self.id, &mut addr as *mut _) };
        if rc < 0 {
            return Err(Error::from_rte("rte_eth_macaddr_get"));
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

fn configure_rss(port_conf: &mut ffi::rte_eth_conf, cfg: &PortConfig) {
    if cfg.rx_queues <= 1 || cfg.rss_hf == 0 {
        return;
    }
    port_conf.rxmode.mq_mode = ffi::rte_eth_rx_mq_mode_RTE_ETH_MQ_RX_RSS;
    port_conf.rx_adv_conf.rss_conf.rss_hf = cfg.rss_hf;
    port_conf.rx_adv_conf.rss_conf.rss_key = std::ptr::null_mut();
    port_conf.rx_adv_conf.rss_conf.rss_key_len = 0;
}

fn device_info(port_id: u16) -> Result<ffi::rte_eth_dev_info> {
    let mut info: ffi::rte_eth_dev_info = unsafe { std::mem::zeroed() };
    let rc = unsafe { ffi::rte_eth_dev_info_get(port_id, &mut info as *mut _) };
    if rc < 0 {
        return Err(Error::from_rte("rte_eth_dev_info_get"));
    }
    Ok(info)
}

fn enabled_offloads(cfg: &PortConfig, info: &ffi::rte_eth_dev_info) -> OffloadCaps {
    OffloadCaps {
        tx: cfg.desired_tx_offloads & info.tx_offload_capa,
        rx: cfg.desired_rx_offloads & info.rx_offload_capa,
    }
}

fn configure_device(port_id: u16, cfg: &PortConfig, port_conf: &ffi::rte_eth_conf) -> Result<()> {
    let rc = unsafe {
        ffi::rte_eth_dev_configure(port_id, cfg.rx_queues, cfg.tx_queues, port_conf as *const _)
    };
    if rc < 0 {
        return Err(Error::from_rte("rte_eth_dev_configure"));
    }
    Ok(())
}

fn setup_rx_queues(
    port_id: u16,
    cfg: &PortConfig,
    rx_pool: &Mempool,
    info: &ffi::rte_eth_dev_info,
    enabled_rx: u64,
) -> Result<()> {
    for queue in 0..cfg.rx_queues {
        let mut rx_conf = info.default_rxconf;
        rx_conf.offloads = enabled_rx;
        let rc = unsafe {
            ffi::rte_eth_rx_queue_setup(
                port_id,
                queue,
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
    Ok(())
}

fn setup_tx_queues(
    port_id: u16,
    cfg: &PortConfig,
    info: &ffi::rte_eth_dev_info,
    enabled_tx: u64,
) -> Result<()> {
    for queue in 0..cfg.tx_queues {
        let mut tx_conf = info.default_txconf;
        tx_conf.offloads = enabled_tx;
        let rc = unsafe {
            ffi::rte_eth_tx_queue_setup(
                port_id,
                queue,
                cfg.tx_descriptors,
                ffi::rte_eth_dev_socket_id(port_id) as c_uint,
                &tx_conf as *const _,
            )
        };
        if rc < 0 {
            return Err(Error::from_rte("rte_eth_tx_queue_setup"));
        }
    }
    Ok(())
}

fn start_device(port_id: u16) -> Result<()> {
    let rc = unsafe { ffi::rte_eth_dev_start(port_id) };
    if rc < 0 {
        return Err(Error::from_rte("rte_eth_dev_start"));
    }
    unsafe { ffi::rte_eth_promiscuous_enable(port_id) };
    Ok(())
}
