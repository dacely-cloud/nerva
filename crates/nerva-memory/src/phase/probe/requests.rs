use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::transport::TransportDeviceId;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::phase::probe::fixture::PhaseProbeFixture;
use crate::phase::types::PhaseHandoffRequest;

pub(super) fn phase_handoff_requests(fixture: &PhaseProbeFixture) -> [PhaseHandoffRequest; 7] {
    [
        PhaseHandoffRequest {
            block_id: fixture.cpu_activation,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 1,
            reason: "phase_cpu_to_gpu_activation",
        },
        PhaseHandoffRequest {
            block_id: fixture.gpu_logits,
            from: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            to: ExecutionOwner::Cpu,
            required_version: 2,
            reason: "phase_gpu_to_cpu_logits",
        },
        PhaseHandoffRequest {
            block_id: fixture.gpu_transport,
            from: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            to: ExecutionOwner::Nic(TransportDeviceId(0)),
            required_version: 3,
            reason: "phase_gpu_to_nic_transport",
        },
        PhaseHandoffRequest {
            block_id: fixture.wrong_owner,
            from: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            to: ExecutionOwner::Cpu,
            required_version: 1,
            reason: "phase_reject_uncoordinated_writer",
        },
        PhaseHandoffRequest {
            block_id: fixture.stale,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 99,
            reason: "phase_reject_stale_version",
        },
        PhaseHandoffRequest {
            block_id: fixture.unready,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 1,
            reason: "phase_reject_unready_block",
        },
        PhaseHandoffRequest {
            block_id: fixture.shared_read_only,
            from: ExecutionOwner::SharedReadOnly,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 1,
            reason: "phase_reject_shared_read_only_mutation",
        },
    ]
}
