use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::registration::lifetime::run::run_transport_registration_lifecycle_probe;
use crate::transport::registration::lifetime::summary::TransportRegistrationLifecycleSummary;
use crate::transport::registration::probe::run::run_transport_registration_probe;
use crate::transport::registration::summary::TransportRegistrationSummary;

impl Runtime {
    pub fn run_transport_registration_probe(&self) -> Result<TransportRegistrationSummary> {
        let _ = self.config;
        run_transport_registration_probe()
    }

    pub fn run_transport_registration_lifecycle_probe(
        &self,
    ) -> Result<TransportRegistrationLifecycleSummary> {
        let _ = self.config;
        run_transport_registration_lifecycle_probe()
    }
}
