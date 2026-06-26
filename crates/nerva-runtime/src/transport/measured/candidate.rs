use nerva_ledger::types::decision::CandidateCost;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MeasuredTransportCandidate {
    pub label: &'static str,
    pub bucket_payload_bytes: usize,
    pub measured_p95_ns: u64,
    pub effective_payload_bandwidth_bps: u64,
    pub visible_ns: u64,
}

impl MeasuredTransportCandidate {
    pub(crate) const fn as_cost(self) -> CandidateCost {
        CandidateCost::measured(self.label, self.visible_ns)
    }
}
