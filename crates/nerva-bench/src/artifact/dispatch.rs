use crate::{
    acceptance::runner::run_acceptance_probe,
    cli::model::precision_model_pair_json,
    model_io::{
        config::{run_layout_probe, run_manifest_probe, run_metadata_probe},
        resident::{
            run_hotset_probe, run_resident_shard_probe, run_resident_weight_probe,
            run_weight_execution_probe,
        },
        safetensors::{run_safetensors_probe, run_safetensors_shard_probe},
    },
    parity::run::run_vllm_token_identity_parity,
    parse::{parse_optional_u32, parse_optional_u64, parse_optional_usize},
    probes::{
        kv, memory_loop, mgpu, phase, queue, runtime, synthetic, token, transaction, transport,
    },
};

pub(crate) fn run_artifact_probe(command: &str, args: &[String]) -> Result<String, String> {
    match command {
        "smoke" => Ok(nerva_runtime::capabilities::discovery::cuda_smoke().to_json()),
        "cuda-backend" => {
            let device_bytes = parse_optional_usize(args.first().cloned(), 4096, "device_bytes")?;
            let pinned_bytes = parse_optional_usize(args.get(1).cloned(), 4096, "pinned_bytes")?;
            Ok(
                nerva_runtime::engine::cuda::cuda_backend_contract_smoke(
                    device_bytes,
                    pinned_bytes,
                )
                .to_json(),
            )
        }
        "cuda-graph" => {
            let steps = parse_optional_u32(args.first().cloned(), 1024, "steps")?;
            let ring_capacity = parse_optional_u32(args.get(1).cloned(), 64, "ring_capacity")?;
            let seed_token = parse_optional_u32(args.get(2).cloned(), 1, "seed_token")?;
            Ok(nerva_runtime::engine::cuda::cuda_synthetic_graph_smoke(
                steps,
                ring_capacity,
                seed_token,
            )
            .to_json())
        }
        "cuda-block" => Ok(nerva_runtime::engine::cuda::cuda_tiny_block_smoke().to_json()),
        "cuda-loaded-block" => {
            Ok(nerva_runtime::engine::cuda::cuda_loaded_tiny_block_smoke().to_json())
        }
        "cuda-attention" => {
            Ok(nerva_runtime::engine::cuda::cuda_tiered_attention_smoke().to_json())
        }
        "cuda-sampler" => Ok(nerva_runtime::engine::cuda::cuda_greedy_sampler_smoke().to_json()),
        "cuda-tiny-decode" => {
            let steps = parse_optional_u32(args.first().cloned(), 8, "steps")?;
            let ring_capacity = parse_optional_u32(args.get(1).cloned(), 4, "ring_capacity")?;
            let seed_token = parse_optional_u32(args.get(2).cloned(), 0, "seed_token")?;
            Ok(nerva_runtime::engine::cuda::cuda_tiny_decode_smoke(
                steps,
                ring_capacity,
                seed_token,
            )
            .to_json())
        }
        "capabilities" => runtime::run_capabilities(),
        "topology" => runtime::run_topology_probe(),
        "synthetic" => {
            let steps = parse_optional_u64(args.first().cloned(), 1024, "steps")?;
            let ring_capacity = parse_optional_usize(args.get(1).cloned(), 64, "ring_capacity")?;
            synthetic::run_synthetic(steps, ring_capacity)
        }
        "ledger" => synthetic::run_synthetic_ledger_probe(),
        "token-policy" => token::run_token_policy_probe(),
        "phase-handoff" => phase::run_phase_handoff_probe(),
        "shared-queue" => queue::run_shared_queue_probe(),
        "transaction" => transaction::run_transaction_probe(),
        "memory-loop" => memory_loop::run_memory_loop_probe(),
        "block" => nerva_model::reference::smoke::reference_block_smoke()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("reference block failed: {err:?}")),
        "precision" => nerva_model::precision::smoke::precision_block_smoke()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("precision block failed: {err:?}")),
        "safetensors-block" => {
            nerva_model::precision::file_smoke::run::precision_block_from_safetensors_smoke()
                .map(|summary| summary.to_json())
                .map_err(|err| format!("safetensors precision block failed: {err:?}"))
        }
        "model" => {
            let steps = parse_optional_usize(args.first().cloned(), 8, "steps")?;
            nerva_model::tiny::smoke::tiny_greedy_decode_smoke(steps)
                .map(|summary| summary.to_json())
                .map_err(|err| format!("tiny greedy model failed: {err:?}"))
        }
        "precision-model" => {
            let steps = parse_optional_usize(args.first().cloned(), 8, "steps")?;
            precision_model_pair_json(steps)
        }
        "vllm-parity" => {
            let steps = parse_optional_usize(args.get(1).cloned(), 8, "steps")?;
            run_vllm_token_identity_parity(args.first().cloned(), steps)
        }
        "metadata" => run_metadata_probe(args.first().cloned()),
        "layout" => run_layout_probe(args.first().cloned()),
        "manifest" => run_manifest_probe(args.first().cloned()),
        "safetensors" => run_safetensors_probe(args.first().cloned(), args.get(1).cloned()),
        "safetensors-shards" => run_safetensors_shard_probe(
            args.first().cloned(),
            args.get(1).cloned(),
            args.get(2).cloned(),
        ),
        "resident-shards" => {
            let max_task_bytes =
                parse_optional_usize(args.get(3).cloned(), 16 * 1024 * 1024, "max_task_bytes")?;
            run_resident_shard_probe(
                args.first().cloned(),
                args.get(1).cloned(),
                args.get(2).cloned(),
                max_task_bytes,
            )
        }
        "resident-weights" => run_resident_weight_probe(args.first().cloned()),
        "hotset" => {
            let vram_bytes =
                parse_optional_usize(args.get(1).cloned(), 512 * 1024 * 1024, "vram_bytes")?;
            let max_promote_bytes =
                parse_optional_usize(args.get(2).cloned(), vram_bytes, "max_promote_bytes")?;
            run_hotset_probe(args.first().cloned(), vram_bytes, max_promote_bytes)
        }
        "weight-exec" => {
            let vram_bytes =
                parse_optional_usize(args.get(1).cloned(), 512 * 1024 * 1024, "vram_bytes")?;
            let max_promote_bytes =
                parse_optional_usize(args.get(2).cloned(), vram_bytes, "max_promote_bytes")?;
            let max_steps = parse_optional_usize(args.get(3).cloned(), 32, "max_steps")?;
            let compute_capability =
                parse_optional_u64(args.get(4).cloned(), 89, "compute_capability")?;
            run_weight_execution_probe(
                args.first().cloned(),
                vram_bytes,
                max_promote_bytes,
                max_steps,
                Some(compute_capability as u32),
            )
        }
        "attention" => nerva_model::attention::smoke::blockwise_attention_smoke()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("blockwise attention failed: {err:?}")),
        "warm" => nerva_model::warm_compute::probe::warm_compute_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("warm compute probe failed: {err:?}")),
        "contracts" => nerva_kernel_contracts::registry::probe::kernel_registry_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("kernel contract probe failed: {err:?}")),
        "kv" => kv::run_kv_probe(),
        "fabric-topology" => transport::run_fabric_topology_probe(),
        "fabric-backends" => transport::run_fabric_backend_probe(),
        "dpdk-udp" => transport::run_dpdk_udp_probe(),
        "transport" => transport::run_transport_probe(),
        "transport-matrix" => transport::run_transport_matrix_probe(),
        "transport-registration" => transport::run_transport_registration_probe(),
        "stage-pipeline" => transport::run_stage_pipeline_probe(),
        "multi-gpu" => mgpu::run_multi_gpu_probe(),
        "acceptance" => run_acceptance_probe(),
        _ => Err(format!("unknown artifact probe '{command}'")),
    }
}
