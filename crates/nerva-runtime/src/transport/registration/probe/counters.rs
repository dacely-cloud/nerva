use crate::transport::registration::types::TransportRegistrationBackend;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RegistrationProbeCounters {
    pub(crate) bootstrap_registrations: u64,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) stale_address_rejections: u64,
    pub(crate) stale_version_rejections: u64,
    pub(crate) hot_path_registration_attempts: u64,
    pub(crate) hot_path_registration_rejections: u64,
    pub(crate) per_token_registrations: u64,
    pub(crate) pinned_host_registrations: u64,
    pub(crate) gpu_direct_registrations: u64,
    pub(crate) gpu_direct_registration_skips: u64,
    pub(crate) false_gpu_direct_registrations: u64,
}

impl RegistrationProbeCounters {
    pub(crate) const fn new() -> Self {
        Self {
            bootstrap_registrations: 0,
            cache_hits: 0,
            cache_misses: 0,
            stale_address_rejections: 0,
            stale_version_rejections: 0,
            hot_path_registration_attempts: 0,
            hot_path_registration_rejections: 0,
            per_token_registrations: 0,
            pinned_host_registrations: 0,
            gpu_direct_registrations: 0,
            gpu_direct_registration_skips: 0,
            false_gpu_direct_registrations: 0,
        }
    }

    pub(crate) fn record_bootstrap_registration(
        &mut self,
        backend: TransportRegistrationBackend,
        direct_verified: bool,
    ) {
        self.bootstrap_registrations += 1;
        match backend {
            TransportRegistrationBackend::RdmaPinnedHost
            | TransportRegistrationBackend::DpdkPinnedHost => self.pinned_host_registrations += 1,
            TransportRegistrationBackend::RdmaGpuDirect | TransportRegistrationBackend::DpdkGpu => {
                self.gpu_direct_registrations += 1;
                if !direct_verified {
                    self.false_gpu_direct_registrations += 1;
                }
            }
        }
    }

    pub(crate) fn record_gpu_direct_registration_skip(&mut self) {
        self.gpu_direct_registration_skips += 1;
    }

    pub(crate) fn record_hot_path_registration_rejection(&mut self) {
        self.hot_path_registration_attempts += 1;
        self.hot_path_registration_rejections += 1;
    }

    pub(crate) fn lookup_count(&self) -> u64 {
        self.cache_hits
            + self.cache_misses
            + self.stale_address_rejections
            + self.stale_version_rejections
    }
}
