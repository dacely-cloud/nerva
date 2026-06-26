use nerva_core::types::backend::contract::DeviceBackend;
use nerva_core::types::backend::operation::BackendTransactionDescriptor;
use nerva_core::types::backend::validation::{
    BackendContractValidation, validate_backend_contract,
};
use nerva_core::types::id::transaction::TransactionId;
use nerva_cuda::smoke::status::SmokeStatus;

use crate::backend::contract::adapter::CudaBackendContractAdapter;
use crate::backend::contract::capabilities::cuda_capabilities_from_probe;
use crate::backend::contract::status::BackendContractProbeStatus;
use crate::backend::contract::summary::RuntimeBackendContractSummary;
use crate::engine::runtime::Runtime;

const PROBE_DEVICE_BYTES: usize = 4096;
const PROBE_PINNED_BYTES: usize = 4096;
const PROBE_GRAPH_STEPS: u32 = 16;
const PROBE_GRAPH_RING_CAPACITY: u32 = 4;
const PROBE_GRAPH_SEED_TOKEN: u32 = 1;

impl Runtime {
    pub fn run_backend_contract_probe(&self) -> RuntimeBackendContractSummary {
        let backend = crate::engine::cuda::cuda_backend_contract_smoke(
            PROBE_DEVICE_BYTES,
            PROBE_PINNED_BYTES,
        );
        let graph = crate::engine::cuda::cuda_synthetic_graph_smoke(
            PROBE_GRAPH_STEPS,
            PROBE_GRAPH_RING_CAPACITY,
            PROBE_GRAPH_SEED_TOKEN,
        );
        let sampler = crate::engine::cuda::cuda_greedy_sampler_smoke();
        let capabilities = cuda_capabilities_from_probe(
            &backend,
            graph.status == SmokeStatus::Ok && graph.hot_path_allocations == 0,
            sampler.status == SmokeStatus::Ok && sampler.hot_path_allocations == 0,
        );
        let mut summary = RuntimeBackendContractSummary::from_parts(
            probe_status(&backend, &graph, &sampler),
            &capabilities,
            BackendContractValidation {
                bootstrap_decode_ready: false,
                device_allocation_ready: false,
                pinned_allocation_ready: false,
                queue_ready: false,
                graph_ready: false,
            },
        );
        summary.requested_device_bytes = backend.requested_device_bytes;
        summary.requested_pinned_bytes = backend.requested_pinned_bytes;
        summary.allocated_device_bytes = backend.allocated_device_bytes;
        summary.allocated_pinned_bytes = backend.allocated_pinned_bytes;
        summary.graph_replays = graph.graph_replays;
        summary.graph_nodes = graph.graph_nodes;
        summary.sampler_tokens = (sampler.status == SmokeStatus::Ok) as u64;
        summary.hot_path_allocations = backend.hot_path_allocations
            + graph.hot_path_allocations
            + sampler.hot_path_allocations;

        match CudaBackendContractAdapter::from_probe(&backend, &graph, &sampler)
            .and_then(exercise_backend_contract)
        {
            Ok((validation, submission_id)) => {
                summary.status = BackendContractProbeStatus::Ok;
                summary.queue_ready = validation.queue_ready;
                summary.event_ready = true;
                summary.graph_ready = validation.graph_ready;
                summary.submission_id = Some(submission_id.0);
                summary.validation = validation;
            }
            Err(err) => {
                if summary.status == BackendContractProbeStatus::Ok {
                    summary.status = BackendContractProbeStatus::Failed;
                }
                summary.error = Some(format!("{err:?}"));
            }
        }

        summary
    }
}

fn exercise_backend_contract(
    mut adapter: CudaBackendContractAdapter,
) -> nerva_core::types::error::Result<(
    BackendContractValidation,
    nerva_core::types::backend::operation::BackendSubmissionId,
)> {
    let device = adapter.create_device(adapter.capabilities().device)?;
    let queue = adapter.create_queue(&device)?;
    let event = adapter.create_event(&device)?;
    let device_allocation = adapter.allocate_device(&device, PROBE_DEVICE_BYTES, 256)?;
    let pinned_allocation = adapter.allocate_pinned(PROBE_PINNED_BYTES, 64)?;
    let transaction = BackendTransactionDescriptor {
        id: TransactionId(1),
        operation_count: PROBE_GRAPH_STEPS as usize,
        block_use_count: PROBE_GRAPH_RING_CAPACITY as usize,
        graph_capturable: true,
    };
    let graph = adapter.capture(&transaction)?;
    let submission = adapter.submit(&graph)?;
    let validation = validate_backend_contract(
        adapter.capabilities(),
        device_allocation,
        pinned_allocation,
        queue,
        graph,
    );
    if !event.preallocated {
        return Err(nerva_core::types::error::NervaError::BackendUnavailable {
            backend: "cuda",
            reason: "CUDA event contract was not preallocated".to_string(),
        });
    }
    Ok((validation, submission))
}

fn probe_status(
    backend: &nerva_cuda::backend::summary::CudaBackendContractSummary,
    graph: &nerva_cuda::graph::summary::CudaSyntheticGraphSummary,
    sampler: &nerva_cuda::sampler::summary::CudaGreedySamplerSummary,
) -> BackendContractProbeStatus {
    if backend.status == SmokeStatus::Unavailable {
        BackendContractProbeStatus::Unavailable
    } else if backend.status == SmokeStatus::Ok
        && graph.status == SmokeStatus::Ok
        && sampler.status == SmokeStatus::Ok
    {
        BackendContractProbeStatus::Ok
    } else {
        BackendContractProbeStatus::Failed
    }
}
