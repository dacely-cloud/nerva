#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MetricSource {
    RuntimeTimestamp,
    GpuEvent,
    HardwareCounter,
    Profiler,
    TransportCompletion,
    EstimatedModel,
}

impl MetricSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeTimestamp => "runtime_timestamp",
            Self::GpuEvent => "gpu_event",
            Self::HardwareCounter => "hardware_counter",
            Self::Profiler => "profiler",
            Self::TransportCompletion => "transport_completion",
            Self::EstimatedModel => "estimated_model",
        }
    }
}
