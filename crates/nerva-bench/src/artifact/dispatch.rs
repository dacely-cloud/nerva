use crate::{
    acceptance::runner::run_acceptance_probe,
    artifact::model,
    cli::model::precision::precision_model_pair_json,
    model_io::{
        config::{run_layout_probe, run_manifest_probe, run_metadata_probe},
        deepseek::{
            run_deepseek_cuda_primitive_bench, run_deepseek_cuda_readiness,
            run_deepseek_runtime_plan, run_deepseek_vllm_parity_gate,
            run_deepseek_vllm_reference_audit,
        },
        resident::{
            run_hotset_probe, run_resident_shard_probe, run_resident_weight_probe,
            run_weight_execution_probe,
        },
        safetensors::{run_safetensors_probe, run_safetensors_shard_probe},
    },
    parity::run::{run_token_identity_artifact_parity, run_vllm_token_identity_parity},
    parse::{parse_optional_u32, parse_optional_u64, parse_optional_usize},
    perf::{external::external_baseline_json_from_args, run::perf_baseline_json_from_args},
    probes::{
        backend, compute, kv, measurements, memory_loop, mgpu, phase, projection, queue, runtime,
        synthetic, token, transaction, transport,
    },
};

pub(crate) fn run_artifact_probe(command: &str, args: &[String]) -> Result<String, String> {
    if let Some(result) = model::run_model_artifact(command, args) {
        return result;
    }

    match command {
        "smoke" => Ok(nerva_runtime::capabilities::discovery::cuda_smoke().to_json()),
        "cuda-backend" => {
            let device_bytes = parse_optional_usize(args.first().cloned(), 4096, "device_bytes")?;
            let pinned_bytes = parse_optional_usize(args.get(1).cloned(), 4096, "pinned_bytes")?;
            Ok(
                nerva_cuda::backend::probe::backend_contract_smoke(device_bytes, pinned_bytes)
                    .to_json(),
            )
        }
        "cuda-graph" => {
            let steps = parse_optional_u32(args.first().cloned(), 1024, "steps")?;
            let ring_capacity = parse_optional_u32(args.get(1).cloned(), 64, "ring_capacity")?;
            let seed_token = parse_optional_u32(args.get(2).cloned(), 1, "seed_token")?;
            Ok(
                nerva_cuda::graph::probe::synthetic_graph_smoke(steps, ring_capacity, seed_token)
                    .to_json(),
            )
        }
        "cuda-block" => Ok(nerva_cuda::block::probe::tiny_block_smoke().to_json()),
        "cuda-loaded-block" => Ok(nerva_cuda::block::probe::loaded_tiny_block_smoke().to_json()),
        "cuda-attention" => Ok(nerva_cuda::attention::probe::tiered_attention_smoke().to_json()),
        "cuda-deepseek-mla" => Ok(nerva_cuda::deepseek_mla::probe::deepseek_mla_smoke().to_json()),
        "cuda-deepseek-moe" => Ok(nerva_cuda::deepseek_moe::probe::deepseek_moe_smoke().to_json()),
        "cuda-deepseek-quant" => {
            Ok(nerva_cuda::deepseek_quant::probe::deepseek_quant_smoke().to_json())
        }
        "cuda-deepseek-inv-rope-fp8-quant" => Ok(
            nerva_cuda::deepseek_quant::probe::deepseek_fused_inv_rope_fp8_quant_smoke().to_json(),
        ),
        "cuda-deepseek-router" => {
            Ok(nerva_cuda::deepseek_router::probe::deepseek_router_smoke().to_json())
        }
        "cuda-deepseek-qkv-rmsnorm" => {
            Ok(nerva_cuda::deepseek_mla::probe::deepseek_qkv_rmsnorm_smoke().to_json())
        }
        "cuda-deepseek-kv" => Ok(nerva_cuda::deepseek_kv::probe::deepseek_kv_smoke().to_json()),
        "cuda-deepseek-compressed-slots" => {
            Ok(nerva_cuda::deepseek_kv::probe::deepseek_compressed_slot_mapping_smoke().to_json())
        }
        "cuda-deepseek-c128-topk" => {
            Ok(nerva_cuda::deepseek_kv::probe::deepseek_c128_topk_metadata_smoke().to_json())
        }
        "cuda-deepseek-save-partial-states" => {
            Ok(nerva_cuda::deepseek_kv::probe::deepseek_save_partial_states_smoke().to_json())
        }
        "cuda-deepseek-compress-cache" => Ok(
            nerva_cuda::deepseek_kv::probe::deepseek_compress_norm_rope_fp8_cache_smoke().to_json(),
        ),
        "cuda-sampler" => Ok(nerva_cuda::sampler::probe::greedy_sampler_smoke().to_json()),
        "cuda-tiny-decode" => {
            let steps = parse_optional_u32(args.first().cloned(), 8, "steps")?;
            let ring_capacity = parse_optional_u32(args.get(1).cloned(), 4, "ring_capacity")?;
            let seed_token = parse_optional_u32(args.get(2).cloned(), 0, "seed_token")?;
            Ok(
                nerva_cuda::decode::probe::tiny_decode_smoke(steps, ring_capacity, seed_token)
                    .to_json(),
            )
        }
        "capabilities" => runtime::run_capabilities(),
        "backend-contract" => backend::run_backend_contract_probe(),
        "hot-path-guard" => runtime::run_hot_path_guard_probe(),
        "security-isolation" => runtime::run_security_isolation_probe(),
        "correctness" => runtime::run_correctness_validation_probe(),
        "production-invariants" => runtime::run_production_invariant_probe(),
        "request-state" => runtime::run_request_state_probe(),
        "request-scheduler" => runtime::run_request_scheduler_probe(),
        "topology" => runtime::run_topology_probe(),
        "synthetic" => {
            let steps = parse_optional_u64(args.first().cloned(), 1024, "steps")?;
            let ring_capacity = parse_optional_usize(args.get(1).cloned(), 64, "ring_capacity")?;
            synthetic::run_synthetic(steps, ring_capacity)
        }
        "ledger" => synthetic::run_synthetic_ledger_probe(),
        "critical-path" => synthetic::run_critical_path_probe(),
        "token-policy" => token::run_token_policy_probe(),
        "phase-handoff" => phase::run_phase_handoff_probe(),
        "shared-queue" => queue::run_shared_queue_probe(),
        "transaction" => transaction::run_transaction_probe(),
        "compute-near-data" => compute::run_compute_near_data_probe(),
        "measurements" => measurements::run_measurement_table_probe(),
        "measured-planner" => measurements::run_measured_planner_probe(),
        "memory-loop" => memory_loop::run_memory_loop_probe(),
        "projection-bench" => projection::run_projection_bench_from_args(args),
        "block" => nerva_model::reference::smoke::run::reference_block_smoke()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("reference block failed: {err:?}")),
        "precision" => nerva_model::precision::smoke::run::precision_block_smoke()
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
        "prompt-model" => {
            let prompt = args.first().map_or("zero", String::as_str);
            let steps = parse_optional_usize(args.get(1).cloned(), 8, "steps")?;
            nerva_model::prompt::decode::tiny_prompt_decode_smoke(prompt, steps)
                .map(|summary| summary.to_json())
                .map_err(|err| format!("tiny prompt model failed: {err:?}"))
        }
        "precision-model" => {
            let steps = parse_optional_usize(args.first().cloned(), 8, "steps")?;
            precision_model_pair_json(steps)
        }
        "vllm-parity" => {
            let steps = parse_optional_usize(args.get(1).cloned(), 8, "steps")?;
            run_vllm_token_identity_parity(args.first().cloned(), steps)
        }
        "token-parity" => {
            run_token_identity_artifact_parity(args.first().cloned(), args.get(1).cloned())
        }
        "perf-baseline" => perf_baseline_json_from_args(args),
        "external-baseline" => external_baseline_json_from_args(args),
        "metadata" => run_metadata_probe(args.first().cloned()),
        "layout" => run_layout_probe(args.first().cloned()),
        "manifest" => run_manifest_probe(args.first().cloned()),
        "deepseek-runtime-plan" => run_deepseek_runtime_plan(args.first().cloned()),
        "deepseek-cuda-readiness" => run_deepseek_cuda_readiness(args.first().cloned()),
        "deepseek-cuda-primitive-bench" => {
            let iterations = parse_optional_usize(args.first().cloned(), 16, "iterations")?;
            run_deepseek_cuda_primitive_bench(iterations)
        }
        "deepseek-vllm-reference-audit" => run_deepseek_vllm_reference_audit(args.first().cloned()),
        "deepseek-vllm-parity-gate" => {
            run_deepseek_vllm_parity_gate(args.first().cloned(), args.get(1).cloned())
        }
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
        "warm" => nerva_model::warm_compute::probe::run::warm_compute_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("warm compute probe failed: {err:?}")),
        "contracts" => nerva_kernel_contracts::registry::probe::kernel_registry_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("kernel contract probe failed: {err:?}")),
        "kv" => kv::run_kv_probe(),
        "tiered-kv" => kv::run_tiered_kv_attention_probe(),
        "fabric-topology" => transport::run_fabric_topology_probe(),
        "fabric-backends" => transport::run_fabric_backend_probe(),
        "dpdk-udp" => transport::run_dpdk_udp_probe(),
        "kernel-udp" => transport::run_kernel_udp_probe(),
        "kernel-udp-matrix" => transport::run_kernel_udp_matrix_probe(),
        "tcp-control" => transport::run_tcp_control_probe(),
        "measured-transport" => transport::run_measured_transport_selector_probe(),
        "transport-provenance" => transport::run_transport_metric_provenance_probe(),
        "transport" => transport::run_transport_probe(),
        "transport-contract" => transport::run_transport_contract_probe(),
        "transport-matrix" => transport::run_transport_matrix_probe(),
        "transport-registration" => transport::run_transport_registration_probe(),
        "transport-registration-lifecycle" => {
            transport::run_transport_registration_lifecycle_probe()
        }
        "stage-pipeline" => transport::run_stage_pipeline_probe(),
        "multi-gpu" => mgpu::run_multi_gpu_probe(),
        "acceptance" => run_acceptance_probe(),
        _ => Err(format!("unknown artifact probe '{command}'")),
    }
}
