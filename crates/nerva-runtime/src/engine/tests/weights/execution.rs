use nerva_core::types::block::ResidencyState;
use nerva_ledger::types::fallback::FallbackClass;

use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::support::tiny_llama_manifest;
use crate::weights::execution::ResidentWeightExecutionStrategy;

#[test]
fn resident_weight_execution_plans_gpu_staged_for_dram_fp16_weights() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 3, Some(89))
        .unwrap();

    assert_eq!(plan.steps.len(), 3);
    assert_eq!(plan.cpu_steps, 0);
    assert_eq!(plan.gpu_resident_steps, 0);
    assert_eq!(plan.gpu_staged_steps, 3);
    assert_eq!(plan.fallback_steps, 0);
    assert_eq!(plan.fallback_decisions, 0);
    assert_eq!(plan.block_version_dependencies, 3);
    assert_eq!(plan.ledger.execution_decisions.len(), 3);
    assert!(plan.ledger.require_satisfied_block_versions().is_ok());
    assert_eq!(
        plan.steps[0].strategy,
        ResidentWeightExecutionStrategy::GpuStaged
    );
    assert_eq!(
        plan.steps[0].kernel_name,
        "cuda_decode_dense_matvec_fp16_bf16"
    );
    assert!(plan.to_json().contains("\"gpu_staged_steps\":3"));
}

#[test]
fn resident_weight_execution_uses_gpu_resident_hotset_blocks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(128, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    runtime
        .promote_resident_weight_hotset(&mut table, 100)
        .unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 3, Some(89))
        .unwrap();

    assert_eq!(plan.gpu_resident_steps, 2);
    assert_eq!(plan.gpu_staged_steps, 1);
    assert_eq!(
        plan.steps[0].strategy,
        ResidentWeightExecutionStrategy::GpuResident
    );
    assert_eq!(
        plan.steps[2].strategy,
        ResidentWeightExecutionStrategy::GpuStaged
    );
}

#[test]
fn resident_weight_execution_uses_exact_cpu_fallback_for_f32_cuda() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float32"
            }"#,
    )
    .unwrap();
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();

    assert_eq!(plan.cpu_steps, 2);
    assert_eq!(plan.fallback_steps, 2);
    assert_eq!(plan.fallback_decisions, 2);
    assert_eq!(plan.ledger.fallback_count_for(FallbackClass::ExactNamed), 2);
    assert!(plan.steps.iter().all(|step| step.fallback));
    assert!(
        plan.steps
            .iter()
            .all(|step| step.kernel_name == "cpu_reference_dense_matvec_f32")
    );
}

#[test]
fn resident_weight_execution_rejects_non_ready_blocks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let first = table.entries.first().unwrap().block_id;
    table
        .registry
        .transition(first, ResidencyState::Prefetching)
        .unwrap();

    assert!(
        runtime
            .plan_resident_weight_execution(&table, 1, Some(89))
            .is_err()
    );
}

#[test]
fn resident_weight_execution_run_ledgers_gpu_resident_and_staged_work() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(128, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    runtime
        .promote_resident_weight_hotset(&mut table, 100)
        .unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 3, Some(89))
        .unwrap();
    let summary = runtime
        .execute_resident_weight_execution_plan(&table, &plan)
        .unwrap();

    assert_eq!(summary.steps, 3);
    assert_eq!(summary.gpu_resident_steps, 2);
    assert_eq!(summary.gpu_staged_steps, 1);
    assert_eq!(summary.fallback_steps, 0);
    assert_eq!(summary.fallback_decisions, 0);
    assert_eq!(summary.block_version_dependencies, 3);
    assert_eq!(summary.cpu_events, 0);
    assert_eq!(summary.device_events, 3);
    assert_eq!(summary.copy_events, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.total_latency_ns > 0);
    assert!(summary.to_json().contains("\"device_events\":3"));
}

#[test]
fn resident_weight_execution_run_ledgers_exact_cpu_fallback() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float32"
            }"#,
    )
    .unwrap();
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata).unwrap();
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout).unwrap();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();
    let summary = runtime
        .execute_resident_weight_execution_plan(&table, &plan)
        .unwrap();

    assert_eq!(summary.steps, 2);
    assert_eq!(summary.cpu_events, 2);
    assert_eq!(summary.device_events, 0);
    assert_eq!(summary.copy_events, 0);
    assert_eq!(summary.fallback_steps, 2);
    assert_eq!(summary.fallback_decisions, 2);
    assert_eq!(
        summary.ledger.fallback_count_for(FallbackClass::ExactNamed),
        2
    );
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn resident_weight_execution_run_rejects_unsatisfied_block_version() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();
    let mut plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();
    plan.steps[0].block_version = plan.steps[0].block_version.saturating_add(1);

    assert!(
        runtime
            .execute_resident_weight_execution_plan(&table, &plan)
            .is_err()
    );
}

#[test]
fn resident_weight_execution_run_rejects_stale_plan_after_tier_change() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(128, 0, manifest.total_weight_bytes),
        )
        .unwrap();
    let plan = runtime
        .plan_resident_weight_execution(&table, 2, Some(89))
        .unwrap();
    runtime
        .promote_resident_weight_hotset(&mut table, 100)
        .unwrap();

    assert!(
        runtime
            .execute_resident_weight_execution_plan(&table, &plan)
            .is_err()
    );
}
