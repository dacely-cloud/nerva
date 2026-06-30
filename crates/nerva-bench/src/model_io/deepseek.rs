use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use nerva_cuda::deepseek_kv::c128_topk::deepseek_c128_topk_metadata;
use nerva_cuda::deepseek_kv::c4_indexer_topk::deepseek_c4_indexer_topk;
use nerva_cuda::deepseek_kv::pack::deepseek_fp8_ds_mla_pack;
use nerva_cuda::deepseek_kv::partial_states::deepseek_save_partial_states;
use nerva_cuda::deepseek_kv::slot_mapping::deepseek_compressed_slot_mapping;
use nerva_cuda::deepseek_mla::decode::{deepseek_mla_decode, CudaDeepSeekMlaDecodeInput};
use nerva_cuda::deepseek_mla::qkv_norm::deepseek_qkv_rmsnorm;
use nerva_cuda::deepseek_moe::experts::{
    deepseek_megamoe_experts, CudaDeepSeekMegaMoeExpertsInput,
};
use nerva_cuda::deepseek_moe::forward::{deepseek_moe_forward, CudaDeepSeekMoeForwardInput};
use nerva_cuda::deepseek_moe::prepare::{
    deepseek_megamoe_prepare, CudaDeepSeekMegaMoeEplbMapping, CudaDeepSeekMegaMoePrepareInput,
};
use nerva_cuda::deepseek_quant::dequant::{
    deepseek_fp8_e4m3fn_e8m0_dequant, deepseek_mxfp4_e2m1_e8m0_dequant,
};
use nerva_cuda::deepseek_quant::inv_rope::deepseek_fused_inv_rope_fp8_quant;
use nerva_cuda::deepseek_router::route::{
    deepseek_router_route_v3_grouped_sigmoid, deepseek_router_route_v4_hash,
    deepseek_router_route_v4_sqrtsoftplus,
};
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::hf::architecture::HfArchitectureKind;
use nerva_model::hf::contract::validate_exact_runtime_contract;
use nerva_model::hf::deepseek::plan_deepseek_vllm_kv_cache;
use nerva_model::hf::deepseek_runtime::{
    deepseek_execution_unit_coverage as execution_unit_coverage,
    deepseek_implemented_primitives as implemented_primitives,
    deepseek_layer_report as layer_report,
    deepseek_required_execution_units as required_execution_units,
    deepseek_v4_mhc_warmup_token_sizes, DeepSeekExecutionUnitCoverage,
    DEEPSEEK_V4_MHC_AUTO_WARMUP_MAX_TOKENS,
};
use nerva_model::hf::metadata::HfModelMetadata;
use nerva_model::hf::parser::parse_hf_config_metadata;

use crate::json::{json_escape, json_string_array};
use crate::parity::compare::compare_token_slices;
use crate::parity::hash::hash_tokens;
use crate::parity::parser::{parse_token_ids_for_key, parse_vllm_token_ids};

pub(crate) struct DeepSeekCudaPrimitiveReport<'a> {
    pub(crate) name: &'a str,
    pub(crate) status: &'a str,
    pub(crate) summary_json: &'a str,
}

struct DeepSeekVllmReferenceSpec {
    architecture: &'static str,
    execution_unit: &'static str,
    relative_path: &'static str,
    required_symbols: &'static [&'static str],
}

struct DeepSeekVllmReferenceUnit {
    architecture: &'static str,
    execution_unit: &'static str,
    relative_path: &'static str,
    absolute_path: String,
    status: &'static str,
    size_bytes: u64,
    fnv1a64: Option<u64>,
    required_symbols: Vec<String>,
    missing_symbols: Vec<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
struct DeepSeekArtifactMetric {
    value: f64,
    source: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
struct DeepSeekSamplerArtifact {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<f64>,
    seed: Option<f64>,
}

impl DeepSeekSamplerArtifact {
    fn comparable_with(&self, other: &Self) -> bool {
        self.temperature.is_some()
            && self.top_p.is_some()
            && self.top_k.is_some()
            && self.seed.is_some()
            && other.temperature.is_some()
            && other.top_p.is_some()
            && other.top_k.is_some()
            && other.seed.is_some()
    }

    fn matches(&self, other: &Self) -> bool {
        same_json_number(self.temperature, other.temperature)
            && same_json_number(self.top_p, other.top_p)
            && same_json_number(self.top_k, other.top_k)
            && same_json_number(self.seed, other.seed)
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"temperature\":{},\"top_p\":{},\"top_k\":{},\"seed\":{}}}",
            json_opt_f64(self.temperature),
            json_opt_f64(self.top_p),
            json_opt_f64(self.top_k),
            json_opt_f64(self.seed),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeepSeekCudaPrimitiveBenchSample {
    pub(crate) name: String,
    pub(crate) status: &'static str,
    pub(crate) requested_iterations: usize,
    pub(crate) executed_iterations: usize,
    pub(crate) total_wall_ns: u128,
    pub(crate) avg_wall_ns: u128,
    pub(crate) output_hash: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes_per_iter: u64,
    pub(crate) d2h_bytes_per_iter: u64,
    pub(crate) kernel_launches_per_iter: u64,
    pub(crate) sync_calls_per_iter: u64,
    pub(crate) hot_path_allocations_per_iter: u64,
    pub(crate) error: Option<String>,
}

#[derive(Clone, Debug)]
struct DeepSeekPrimitiveMetrics {
    status: SmokeStatus,
    output_hash: u64,
    device_arena_bytes: u64,
    pinned_host_bytes: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    kernel_launches: u64,
    sync_calls: u64,
    hot_path_allocations: u64,
    error: Option<String>,
}

pub(crate) fn run_deepseek_runtime_plan(config_path: Option<String>) -> Result<String, String> {
    let path =
        config_path.ok_or_else(|| "deepseek-runtime-plan requires config.json".to_string())?;
    let config =
        std::fs::read_to_string(&path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let metadata = parse_hf_config_metadata(&config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    deepseek_runtime_plan_json(&metadata)
}

pub(crate) fn run_deepseek_vllm_reference_audit(
    vllm_root_arg: Option<String>,
) -> Result<String, String> {
    let vllm_root = PathBuf::from(vllm_root_arg.unwrap_or_else(|| "/root/vllm".to_string()));
    let specs = deepseek_vllm_reference_specs();
    let units = specs
        .iter()
        .map(|spec| read_vllm_reference_unit(&vllm_root, spec))
        .collect::<Vec<_>>();
    Ok(deepseek_vllm_reference_audit_json(&vllm_root, &units))
}

pub(crate) fn run_deepseek_vllm_parity_gate(
    config_path: Option<String>,
    vllm_root_arg: Option<String>,
) -> Result<String, String> {
    let config_path =
        config_path.ok_or_else(|| "deepseek-vllm-parity-gate requires config.json".to_string())?;
    let config = std::fs::read_to_string(&config_path)
        .map_err(|err| format!("failed to read {config_path}: {err}"))?;
    let metadata = parse_hf_config_metadata(&config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    if !metadata.architecture.is_deepseek() {
        return Err(format!(
            "deepseek-vllm-parity-gate requires a DeepSeek architecture, got {}",
            metadata.architecture.as_str()
        ));
    }

    let vllm_root = PathBuf::from(vllm_root_arg.unwrap_or_else(|| "/root/vllm".to_string()));
    let specs = deepseek_vllm_reference_specs();
    let vllm_units = specs
        .iter()
        .map(|spec| read_vllm_reference_unit(&vllm_root, spec))
        .collect::<Vec<_>>();
    Ok(deepseek_vllm_parity_gate_json(
        &config_path,
        &vllm_root,
        &metadata,
        &vllm_units,
    ))
}

pub(crate) fn run_deepseek_vllm_benchmark_plan(
    checkpoint_dir: Option<String>,
    prompt_spec: Option<String>,
    max_context_tokens: usize,
    max_new_tokens: usize,
    vllm_root_arg: Option<String>,
) -> Result<String, String> {
    let checkpoint_dir = checkpoint_dir
        .ok_or_else(|| "deepseek-vllm-benchmark-plan requires checkpoint_dir".to_string())?;
    let prompt_spec = prompt_spec.ok_or_else(|| {
        "deepseek-vllm-benchmark-plan requires prompt_text|@prompt.txt".to_string()
    })?;
    if max_context_tokens == 0 || max_new_tokens == 0 {
        return Err(
            "deepseek-vllm-benchmark-plan requires non-zero context and output tokens".to_string(),
        );
    }

    let checkpoint_path = PathBuf::from(&checkpoint_dir);
    let config_path = checkpoint_path.join("config.json");
    let checkpoint_exists = checkpoint_path.is_dir();
    let weights_present = checkpoint_path
        .join("model.safetensors.index.json")
        .exists()
        || checkpoint_path.join("model.safetensors").exists();
    let prompt_present = prompt_path_status(&prompt_spec);
    let vllm_root = PathBuf::from(vllm_root_arg.unwrap_or_else(|| "/root/vllm".to_string()));
    let vllm_units = deepseek_vllm_reference_specs()
        .iter()
        .map(|spec| read_vllm_reference_unit(&vllm_root, spec))
        .collect::<Vec<_>>();
    let vllm_reference_status = vllm_reference_status(&vllm_units);

    let (architecture, config_status, config_error, runtime_units, runtime_blockers) =
        match std::fs::read_to_string(&config_path) {
            Ok(config) => match parse_hf_config_metadata(&config) {
                Ok(metadata) if metadata.architecture.is_deepseek() => {
                    let execution_units = execution_unit_coverage(&metadata);
                    let blockers = execution_units
                        .iter()
                        .filter(|unit| {
                            unit.status != "complete" && unit.status != "optional_missing"
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    (
                        Some(metadata.architecture.as_str().to_string()),
                        "ok",
                        None,
                        execution_units,
                        blockers,
                    )
                }
                Ok(metadata) => (
                    Some(metadata.architecture.as_str().to_string()),
                    "non_deepseek",
                    Some(format!(
                        "checkpoint config architecture is {}, expected DeepSeek V3/V3.2/V4",
                        metadata.architecture.as_str()
                    )),
                    Vec::new(),
                    Vec::new(),
                ),
                Err(err) => (
                    None,
                    "parse_failed",
                    Some(format!("{err:?}")),
                    Vec::new(),
                    Vec::new(),
                ),
            },
            Err(err) => (
                None,
                "missing_config",
                Some(format!("failed to read {}: {err}", config_path.display())),
                Vec::new(),
                Vec::new(),
            ),
        };

    let status = if !checkpoint_exists {
        "missing_checkpoint"
    } else if config_status != "ok" {
        "config_blocked"
    } else if !weights_present {
        "missing_weights"
    } else if prompt_present != "ok" {
        "missing_prompt"
    } else if vllm_reference_status != "ok" {
        "vllm_reference_blocked"
    } else {
        "ready"
    };
    let benchmark_allowed = status == "ready";

    let nerva_generate = vec![
        "cargo".to_string(),
        "run".to_string(),
        "--release".to_string(),
        "-p".to_string(),
        "nerva".to_string(),
        "--".to_string(),
        "--json".to_string(),
        "--raw".to_string(),
        "-m".to_string(),
        checkpoint_dir.clone(),
        "-p".to_string(),
        prompt_spec.clone(),
        "-c".to_string(),
        max_context_tokens.to_string(),
        "-o".to_string(),
        max_new_tokens.to_string(),
        "--temperature".to_string(),
        "0".to_string(),
        "--top-p".to_string(),
        "1".to_string(),
        "--top-k".to_string(),
        "0".to_string(),
        "--seed".to_string(),
        "0".to_string(),
    ];
    let nerva_bench = vec![
        "cargo".to_string(),
        "run".to_string(),
        "--release".to_string(),
        "-p".to_string(),
        "nerva-bench".to_string(),
        "--".to_string(),
        "hf-cuda-generate".to_string(),
        checkpoint_dir.clone(),
        max_context_tokens.to_string(),
        max_new_tokens.to_string(),
        "1024".to_string(),
        prompt_spec.clone(),
    ];
    let vllm_generate = vec![
        "python3".to_string(),
        "tools/deepseek_vllm_generate.py".to_string(),
        "--vllm-root".to_string(),
        vllm_root.display().to_string(),
        "--model".to_string(),
        checkpoint_dir.clone(),
        "--prompt".to_string(),
        prompt_spec.clone(),
        "--max-model-len".to_string(),
        max_context_tokens.to_string(),
        "--max-tokens".to_string(),
        max_new_tokens.to_string(),
        "--temperature".to_string(),
        "0".to_string(),
        "--top-p".to_string(),
        "1".to_string(),
        "--top-k".to_string(),
        "0".to_string(),
        "--seed".to_string(),
        "0".to_string(),
        "--dtype".to_string(),
        "bfloat16".to_string(),
        "--runs".to_string(),
        "3".to_string(),
        "--warmup-runs".to_string(),
        "1".to_string(),
    ];
    let compare = vec![
        "cargo".to_string(),
        "run".to_string(),
        "-p".to_string(),
        "nerva-bench".to_string(),
        "--".to_string(),
        "deepseek-vllm-compare".to_string(),
        "vllm.json".to_string(),
        "nerva.json".to_string(),
    ];

    Ok(format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-vllm-benchmark-plan-v1\",\"architecture\":{},\"checkpoint_dir\":\"{}\",\"checkpoint_exists\":{},\"config_path\":\"{}\",\"config_status\":\"{}\",\"config_error\":{},\"weights_present\":{},\"prompt_spec\":\"{}\",\"prompt_status\":\"{}\",\"max_context_tokens\":{},\"max_new_tokens\":{},\"sampler\":{{\"temperature\":0,\"top_p\":1,\"top_k\":0,\"seed\":0}},\"vllm_root\":\"{}\",\"vllm_reference_status\":\"{}\",\"vllm_reference_units_total\":{},\"vllm_reference_units_ok\":{},\"runtime_units_total\":{},\"runtime_blocking_units_total\":{},\"runtime_blocking_units\":{},\"commands\":{{\"nerva_generate\":{},\"nerva_bench_generate\":{},\"vllm_generate\":{},\"compare\":{}}},\"required_comparison\":[\"same checkpoint directory\",\"same literal prompt text with NERVA --raw and vLLM tokenizer.encode(prompt)\",\"same prompt_token_ids in both JSON artifacts\",\"same greedy sampler temperature=0 top_p=1 top_k=0 seed=0\",\"discard vLLM warmup run and keep prefix caching disabled unless explicitly requested\",\"compare generated token ids and generated text\",\"compare post-load/decode tokens_per_second and p99 latency\",\"run deepseek-vllm-compare on the two JSON artifacts\"],\"runtime_parity_status\":\"{}\",\"performance_status\":\"{}\",\"benchmark_allowed\":{},\"claim_allowed\":false}}",
        status,
        json_opt_string(architecture.as_deref()),
        json_escape(&checkpoint_dir),
        checkpoint_exists,
        json_escape(&config_path.display().to_string()),
        config_status,
        json_opt_string(config_error.as_deref()),
        weights_present,
        json_escape(&prompt_spec),
        prompt_present,
        max_context_tokens,
        max_new_tokens,
        json_escape(&vllm_root.display().to_string()),
        vllm_reference_status,
        vllm_units.len(),
        vllm_units.iter().filter(|unit| unit.status == "ok").count(),
        runtime_units.len(),
        runtime_blockers.len(),
        execution_unit_coverage_json(&runtime_blockers),
        json_string_array(&nerva_generate),
        json_string_array(&nerva_bench),
        json_string_array(&vllm_generate),
        json_string_array(&compare),
        if benchmark_allowed {
            "ready_for_same_checkpoint_run"
        } else {
            "blocked_before_same_checkpoint_run"
        },
        if benchmark_allowed {
            "ready_for_vllm_runtime_benchmark"
        } else {
            "not_benchmarked"
        },
        benchmark_allowed,
    ))
}

pub(crate) fn run_deepseek_vllm_benchmark_run(
    checkpoint_dir: Option<String>,
    prompt_spec: Option<String>,
    max_context_tokens: usize,
    max_new_tokens: usize,
    vllm_root_arg: Option<String>,
    artifact_dir_arg: Option<String>,
) -> Result<String, String> {
    let checkpoint_dir = checkpoint_dir
        .ok_or_else(|| "deepseek-vllm-benchmark-run requires checkpoint_dir".to_string())?;
    let prompt_spec = prompt_spec.ok_or_else(|| {
        "deepseek-vllm-benchmark-run requires prompt_text|@prompt.txt".to_string()
    })?;
    if max_context_tokens == 0 || max_new_tokens == 0 {
        return Err(
            "deepseek-vllm-benchmark-run requires non-zero context and output tokens".to_string(),
        );
    }

    let checkpoint_path = PathBuf::from(&checkpoint_dir);
    let config_path = checkpoint_path.join("config.json");
    if !checkpoint_path.is_dir() {
        return Err(format!(
            "checkpoint directory does not exist: {checkpoint_dir}"
        ));
    }
    if !config_path.is_file() {
        return Err(format!(
            "checkpoint config.json does not exist: {}",
            config_path.display()
        ));
    }
    if !checkpoint_path
        .join("model.safetensors.index.json")
        .exists()
        && !checkpoint_path.join("model.safetensors").exists()
    {
        return Err(format!(
            "checkpoint has no model.safetensors or model.safetensors.index.json: {checkpoint_dir}"
        ));
    }
    if prompt_path_status(&prompt_spec) != "ok" {
        return Err(format!("prompt is not readable: {prompt_spec}"));
    }
    let config = std::fs::read_to_string(&config_path)
        .map_err(|err| format!("failed to read {}: {err}", config_path.display()))?;
    let metadata = parse_hf_config_metadata(&config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    if !metadata.architecture.is_deepseek() {
        return Err(format!(
            "deepseek-vllm-benchmark-run requires DeepSeek V3/V3.2/V4, got {}",
            metadata.architecture.as_str()
        ));
    }

    let repo_root = repo_root();
    let vllm_root = PathBuf::from(vllm_root_arg.unwrap_or_else(|| "/root/vllm".to_string()));
    if !vllm_root.is_dir() {
        return Err(format!("vLLM root does not exist: {}", vllm_root.display()));
    }
    let artifact_dir = artifact_dir_arg.map_or_else(default_deepseek_artifact_dir, PathBuf::from);
    std::fs::create_dir_all(&artifact_dir).map_err(|err| {
        format!(
            "failed to create artifact dir {}: {err}",
            artifact_dir.display()
        )
    })?;
    let vllm_artifact = artifact_dir.join("vllm.json");
    let nerva_artifact = artifact_dir.join("nerva.json");
    let compare_artifact = artifact_dir.join("compare.json");

    let vllm_command = deepseek_vllm_generate_command(
        &repo_root,
        &vllm_root,
        &checkpoint_dir,
        &prompt_spec,
        max_context_tokens,
        max_new_tokens,
    );
    let nerva_command = deepseek_nerva_generate_command(
        &checkpoint_dir,
        &prompt_spec,
        max_context_tokens,
        max_new_tokens,
    );

    let vllm_run = run_json_command(&vllm_command, &repo_root)?;
    std::fs::write(&vllm_artifact, &vllm_run.json).map_err(|err| {
        format!(
            "failed to write vLLM artifact {}: {err}",
            vllm_artifact.display()
        )
    })?;
    if vllm_run.status != 0 {
        return Ok(deepseek_benchmark_run_json(
            "vllm_failed",
            &checkpoint_dir,
            &prompt_spec,
            max_context_tokens,
            max_new_tokens,
            &artifact_dir,
            &vllm_artifact,
            &nerva_artifact,
            &compare_artifact,
            &vllm_command,
            &nerva_command,
            &vllm_run,
            None,
            None,
        ));
    }

    let nerva_run = run_json_command(&nerva_command, &repo_root)?;
    std::fs::write(&nerva_artifact, &nerva_run.json).map_err(|err| {
        format!(
            "failed to write NERVA artifact {}: {err}",
            nerva_artifact.display()
        )
    })?;
    if nerva_run.status != 0 {
        return Ok(deepseek_benchmark_run_json(
            "nerva_failed",
            &checkpoint_dir,
            &prompt_spec,
            max_context_tokens,
            max_new_tokens,
            &artifact_dir,
            &vllm_artifact,
            &nerva_artifact,
            &compare_artifact,
            &vllm_command,
            &nerva_command,
            &vllm_run,
            Some(&nerva_run),
            None,
        ));
    }

    let compare_json = deepseek_vllm_compare_json(
        &vllm_artifact.display().to_string(),
        &nerva_artifact.display().to_string(),
        &vllm_run.json,
        &nerva_run.json,
    )?;
    std::fs::write(&compare_artifact, &compare_json).map_err(|err| {
        format!(
            "failed to write compare artifact {}: {err}",
            compare_artifact.display()
        )
    })?;
    let compare_status = find_first_json_string_field(&compare_json, "status")?
        .unwrap_or_else(|| "unknown".to_string());
    let status = if compare_status == "ok" {
        "ok"
    } else {
        "compare_failed"
    };
    Ok(deepseek_benchmark_run_json(
        status,
        &checkpoint_dir,
        &prompt_spec,
        max_context_tokens,
        max_new_tokens,
        &artifact_dir,
        &vllm_artifact,
        &nerva_artifact,
        &compare_artifact,
        &vllm_command,
        &nerva_command,
        &vllm_run,
        Some(&nerva_run),
        Some(&compare_json),
    ))
}

pub(crate) fn run_deepseek_vllm_compare(
    vllm_artifact_path: Option<String>,
    nerva_artifact_path: Option<String>,
) -> Result<String, String> {
    let vllm_artifact_path =
        vllm_artifact_path.ok_or_else(|| "deepseek-vllm-compare requires vllm_json".to_string())?;
    let nerva_artifact_path = nerva_artifact_path
        .ok_or_else(|| "deepseek-vllm-compare requires nerva_json".to_string())?;
    let vllm_json = std::fs::read_to_string(&vllm_artifact_path)
        .map_err(|err| format!("failed to read {vllm_artifact_path}: {err}"))?;
    let nerva_json = std::fs::read_to_string(&nerva_artifact_path)
        .map_err(|err| format!("failed to read {nerva_artifact_path}: {err}"))?;
    deepseek_vllm_compare_json(
        &vllm_artifact_path,
        &nerva_artifact_path,
        &vllm_json,
        &nerva_json,
    )
}

fn deepseek_vllm_compare_json(
    vllm_artifact_path: &str,
    nerva_artifact_path: &str,
    vllm_json: &str,
    nerva_json: &str,
) -> Result<String, String> {
    let (vllm_source_format, vllm_tokens) = parse_vllm_token_ids(vllm_json)?;
    let (nerva_source_format, nerva_tokens) = parse_vllm_token_ids(nerva_json)?;
    let vllm_model = find_deepseek_artifact_identity(vllm_json, &["model", "path"])?;
    let nerva_model = find_deepseek_artifact_identity(nerva_json, &["path", "model"])?;
    let model_comparable = vllm_model.is_some() && nerva_model.is_some();
    let model_parity = match (&vllm_model, &nerva_model) {
        (Some(vllm), Some(nerva)) => {
            normalized_artifact_identity(vllm) == normalized_artifact_identity(nerva)
        }
        _ => false,
    };
    let comparison = compare_token_slices(&vllm_tokens, &nerva_tokens);
    let vllm_token_hash = hash_tokens(&vllm_tokens);
    let nerva_token_hash = hash_tokens(&nerva_tokens);
    let token_parity = comparison.mismatched_tokens == 0
        && comparison.missing_tokens == 0
        && comparison.extra_tokens == 0
        && vllm_token_hash == nerva_token_hash;

    let vllm_prompt_tokens = parse_token_ids_for_key(vllm_json, "prompt_token_ids")?;
    let nerva_prompt_tokens = parse_token_ids_for_key(nerva_json, "prompt_token_ids")?;
    let prompt_comparison = match (&vllm_prompt_tokens, &nerva_prompt_tokens) {
        (Some(vllm), Some(nerva)) => Some(compare_token_slices(vllm, nerva)),
        _ => None,
    };
    let vllm_prompt_hash = vllm_prompt_tokens
        .as_ref()
        .map(|tokens| hash_tokens(tokens));
    let nerva_prompt_hash = nerva_prompt_tokens
        .as_ref()
        .map(|tokens| hash_tokens(tokens));
    let prompt_comparable = prompt_comparison.is_some();
    let prompt_token_parity = prompt_comparison.as_ref().is_some_and(|comparison| {
        comparison.mismatched_tokens == 0
            && comparison.missing_tokens == 0
            && comparison.extra_tokens == 0
            && vllm_prompt_hash == nerva_prompt_hash
    });

    let vllm_sampler = find_deepseek_sampler_artifact(vllm_json)?;
    let nerva_sampler = find_deepseek_sampler_artifact(nerva_json)?;
    let sampler_comparable = vllm_sampler.comparable_with(&nerva_sampler);
    let sampler_parity = sampler_comparable && vllm_sampler.matches(&nerva_sampler);

    let vllm_text = find_first_json_string_field(vllm_json, "generated_text")?;
    let nerva_text = find_first_json_string_field(nerva_json, "generated_text")?;
    let text_parity = match (&vllm_text, &nerva_text) {
        (Some(vllm), Some(nerva)) => vllm == nerva,
        _ => false,
    };

    let vllm_tps = find_deepseek_throughput_metric(vllm_json)?;
    let nerva_tps = find_deepseek_throughput_metric(nerva_json)?;
    let throughput_speedup = match (&nerva_tps, &vllm_tps) {
        (Some(nerva), Some(vllm)) if vllm.value > 0.0 => Some(nerva.value / vllm.value),
        _ => None,
    };
    let throughput_comparable = throughput_speedup.is_some();
    let throughput_ok = throughput_speedup.is_some_and(|speedup| speedup >= 1.0);

    let vllm_p99 = find_deepseek_p99_metric(vllm_json)?;
    let nerva_p99 = find_deepseek_p99_metric(nerva_json)?;
    let p99_ratio = match (&nerva_p99, &vllm_p99) {
        (Some(nerva), Some(vllm)) if vllm.value > 0.0 => Some(nerva.value / vllm.value),
        _ => None,
    };
    let latency_comparable = p99_ratio.is_some();
    let latency_ok = p99_ratio.is_some_and(|ratio| ratio <= 1.0);

    let status = if !model_comparable || !prompt_comparable || !sampler_comparable {
        "not_comparable"
    } else if !model_parity
        || !prompt_token_parity
        || !sampler_parity
        || !token_parity
        || !text_parity
    {
        "mismatch"
    } else if !throughput_comparable {
        "not_comparable"
    } else if !throughput_ok {
        "performance_regression"
    } else if !latency_comparable {
        "throughput_ok_latency_missing"
    } else if !latency_ok {
        "latency_regression"
    } else {
        "ok"
    };
    let claim_allowed = status == "ok";
    let throughput_claim_allowed = model_parity
        && prompt_token_parity
        && sampler_parity
        && token_parity
        && text_parity
        && throughput_ok;

    Ok(format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-vllm-compare-v1\",\"vllm_artifact\":\"{}\",\"nerva_artifact\":\"{}\",\"source_formats\":{{\"vllm\":\"{}\",\"nerva\":\"{}\"}},\"model_comparable\":{},\"model_parity\":{},\"vllm_model\":{},\"nerva_model\":{},\"prompt_comparable\":{},\"prompt_token_parity\":{},\"matched_prompt_tokens\":{},\"mismatched_prompt_tokens\":{},\"missing_prompt_tokens\":{},\"extra_prompt_tokens\":{},\"first_prompt_mismatch_index\":{},\"vllm_prompt_hash\":{},\"nerva_prompt_hash\":{},\"vllm_prompt_tokens\":{},\"nerva_prompt_tokens\":{},\"sampler_comparable\":{},\"sampler_parity\":{},\"vllm_sampler\":{},\"nerva_sampler\":{},\"token_parity\":{},\"text_parity\":{},\"matched_tokens\":{},\"mismatched_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"first_mismatch_index\":{},\"vllm_token_hash\":{},\"nerva_token_hash\":{},\"vllm_generated_tokens\":{},\"nerva_generated_tokens\":{},\"vllm_tokens_per_second\":{},\"vllm_throughput_source\":{},\"nerva_tokens_per_second\":{},\"nerva_throughput_source\":{},\"throughput_speedup_vs_vllm\":{},\"throughput_comparable\":{},\"throughput_ok\":{},\"vllm_p99_ms\":{},\"vllm_p99_source\":{},\"nerva_p99_ms\":{},\"nerva_p99_source\":{},\"p99_ratio_vs_vllm\":{},\"latency_comparable\":{},\"latency_ok\":{},\"throughput_claim_allowed\":{},\"claim_allowed\":{},\"blocking_reasons\":{}}}",
        status,
        json_escape(vllm_artifact_path),
        json_escape(nerva_artifact_path),
        json_escape(vllm_source_format),
        json_escape(nerva_source_format),
        model_comparable,
        model_parity,
        json_opt_string(vllm_model.as_deref()),
        json_opt_string(nerva_model.as_deref()),
        prompt_comparable,
        prompt_token_parity,
        json_opt_usize(prompt_comparison.as_ref().map(|comparison| comparison.matched_tokens)),
        json_opt_usize(prompt_comparison.as_ref().map(|comparison| comparison.mismatched_tokens)),
        json_opt_usize(prompt_comparison.as_ref().map(|comparison| comparison.missing_tokens)),
        json_opt_usize(prompt_comparison.as_ref().map(|comparison| comparison.extra_tokens)),
        json_opt_usize(
            prompt_comparison
                .as_ref()
                .and_then(|comparison| comparison.first_mismatch_index),
        ),
        json_opt_u64(vllm_prompt_hash),
        json_opt_u64(nerva_prompt_hash),
        json_opt_usize(vllm_prompt_tokens.as_ref().map(Vec::len)),
        json_opt_usize(nerva_prompt_tokens.as_ref().map(Vec::len)),
        sampler_comparable,
        sampler_parity,
        vllm_sampler.to_json(),
        nerva_sampler.to_json(),
        token_parity,
        text_parity,
        comparison.matched_tokens,
        comparison.mismatched_tokens,
        comparison.missing_tokens,
        comparison.extra_tokens,
        json_opt_usize(comparison.first_mismatch_index),
        vllm_token_hash,
        nerva_token_hash,
        vllm_tokens.len(),
        nerva_tokens.len(),
        json_opt_metric_value(vllm_tps.as_ref()),
        json_opt_metric_source(vllm_tps.as_ref()),
        json_opt_metric_value(nerva_tps.as_ref()),
        json_opt_metric_source(nerva_tps.as_ref()),
        json_opt_f64(throughput_speedup),
        throughput_comparable,
        throughput_ok,
        json_opt_metric_value(vllm_p99.as_ref()),
        json_opt_metric_source(vllm_p99.as_ref()),
        json_opt_metric_value(nerva_p99.as_ref()),
        json_opt_metric_source(nerva_p99.as_ref()),
        json_opt_f64(p99_ratio),
        latency_comparable,
        latency_ok,
        throughput_claim_allowed,
        claim_allowed,
        json_string_array(&deepseek_compare_blocking_reasons(
            model_comparable,
            model_parity,
            prompt_comparable,
            prompt_token_parity,
            sampler_comparable,
            sampler_parity,
            token_parity,
            text_parity,
            throughput_comparable,
            throughput_ok,
            latency_comparable,
            latency_ok,
        )),
    ))
}

fn deepseek_vllm_parity_gate_json(
    config_path: &str,
    vllm_root: &Path,
    metadata: &HfModelMetadata,
    vllm_units: &[DeepSeekVllmReferenceUnit],
) -> String {
    let runtime_contract = match validate_exact_runtime_contract(metadata) {
        Ok(()) => RuntimeContractReport {
            status: "supported",
            reason: "exact runtime contract accepts this DeepSeek config".to_string(),
        },
        Err(err) => RuntimeContractReport {
            status: "unsupported",
            reason: format!("{err:?}"),
        },
    };
    let execution_units = execution_unit_coverage(metadata);
    let runtime_blockers = execution_units
        .iter()
        .filter(|unit| unit.status != "complete" && unit.status != "optional_missing")
        .cloned()
        .collect::<Vec<_>>();
    let runtime_missing = execution_units
        .iter()
        .filter(|unit| unit.status == "missing")
        .count();
    let runtime_partial = execution_units
        .iter()
        .filter(|unit| unit.status == "partial")
        .count();
    let vllm_ok = vllm_units.iter().filter(|unit| unit.status == "ok").count();
    let vllm_missing = vllm_units
        .iter()
        .filter(|unit| unit.status == "missing_file")
        .count();
    let vllm_symbol_gap = vllm_units
        .iter()
        .filter(|unit| unit.status == "symbol_gap")
        .count();
    let vllm_failed = vllm_units
        .iter()
        .filter(|unit| unit.status == "failed")
        .count();
    let vllm_reference_status = vllm_reference_status(vllm_units);
    let claim_allowed = runtime_contract.status == "supported"
        && runtime_blockers.is_empty()
        && vllm_reference_status == "ok";
    let status = if claim_allowed {
        "ready"
    } else if vllm_reference_status != "ok" {
        "reference_blocked"
    } else {
        "runtime_blocked"
    };
    let blocking_reasons = deepseek_parity_blocking_reasons(
        &runtime_contract,
        &runtime_blockers,
        vllm_reference_status,
    );
    let next_runtime_units = runtime_blockers
        .iter()
        .map(|unit| unit.unit.clone())
        .collect::<Vec<_>>();

    format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-vllm-parity-gate-v1\",\"architecture\":\"{}\",\"config_path\":\"{}\",\"vllm_root\":\"{}\",\"runtime_contract_status\":\"{}\",\"runtime_contract_reason\":\"{}\",\"runtime_units_total\":{},\"runtime_units_partial\":{},\"runtime_units_missing\":{},\"runtime_blocking_units_total\":{},\"required_execution_units\":{},\"implemented_primitives\":{},\"execution_unit_status\":{},\"runtime_blocking_units\":{},\"next_runtime_units\":{},\"vllm_reference_status\":\"{}\",\"vllm_reference_units_total\":{},\"vllm_reference_units_ok\":{},\"vllm_reference_units_missing_file\":{},\"vllm_reference_units_symbol_gap\":{},\"vllm_reference_units_failed\":{},\"vllm_reference_units\":{},\"runtime_parity_status\":\"{}\",\"performance_status\":\"{}\",\"blocking_reasons\":{},\"claim_allowed\":{},\"performance_comparison_allowed\":{}}}",
        status,
        metadata.architecture.as_str(),
        json_escape(config_path),
        json_escape(&vllm_root.display().to_string()),
        runtime_contract.status,
        json_escape(&runtime_contract.reason),
        execution_units.len(),
        runtime_partial,
        runtime_missing,
        runtime_blockers.len(),
        json_string_array(&required_execution_units(metadata)),
        json_string_array(&implemented_primitives(metadata)),
        execution_unit_coverage_json(&execution_units),
        execution_unit_coverage_json(&runtime_blockers),
        json_string_array(&next_runtime_units),
        vllm_reference_status,
        vllm_units.len(),
        vllm_ok,
        vllm_missing,
        vllm_symbol_gap,
        vllm_failed,
        deepseek_vllm_reference_units_json(vllm_units),
        if claim_allowed {
            "verified_ready_for_end_to_end_parity"
        } else {
            "blocked_before_end_to_end_parity"
        },
        if claim_allowed {
            "ready_for_vllm_runtime_benchmark"
        } else {
            "blocked_until_runtime_units_complete"
        },
        json_string_array(&blocking_reasons),
        claim_allowed,
        claim_allowed,
    )
}

fn deepseek_parity_blocking_reasons(
    runtime_contract: &RuntimeContractReport,
    runtime_blockers: &[DeepSeekExecutionUnitCoverage],
    vllm_reference_status: &str,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if runtime_contract.status != "supported" {
        reasons.push(runtime_contract.reason.clone());
    }
    if vllm_reference_status != "ok" {
        reasons.push(format!(
            "vLLM reference audit status is {vllm_reference_status}"
        ));
    }
    for unit in runtime_blockers {
        reasons.push(format!(
            "{} is {}; remaining gaps: {}",
            unit.unit,
            unit.status,
            unit.remaining_gaps.join("; ")
        ));
    }
    reasons
}

fn deepseek_vllm_reference_audit_json(
    vllm_root: &Path,
    units: &[DeepSeekVllmReferenceUnit],
) -> String {
    let ok = units.iter().filter(|unit| unit.status == "ok").count();
    let missing_file = units
        .iter()
        .filter(|unit| unit.status == "missing_file")
        .count();
    let symbol_gap = units
        .iter()
        .filter(|unit| unit.status == "symbol_gap")
        .count();
    let failed = units.iter().filter(|unit| unit.status == "failed").count();
    let status = vllm_reference_status(units);

    format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-vllm-reference-audit-v1\",\"vllm_root\":\"{}\",\"reference_units_total\":{},\"reference_units_ok\":{},\"reference_units_missing_file\":{},\"reference_units_symbol_gap\":{},\"reference_units_failed\":{},\"runtime_parity_status\":\"vllm_reference_sources_pinned\",\"performance_status\":\"source_audit_only_not_runtime_benchmark\",\"claim_allowed\":false,\"units\":{}}}",
        status,
        json_escape(&vllm_root.display().to_string()),
        units.len(),
        ok,
        missing_file,
        symbol_gap,
        failed,
        deepseek_vllm_reference_units_json(units),
    )
}

pub(crate) fn run_deepseek_cuda_primitive_bench(iterations: usize) -> Result<String, String> {
    if iterations == 0 {
        return Err("iterations must be greater than zero".to_string());
    }

    let samples = vec![
        bench_primitive("router_v3_grouped_sigmoid", iterations, bench_router_v3),
        bench_primitive("router_v4_sqrtsoftplus", iterations, bench_router_v4),
        bench_primitive("router_v4_hash", iterations, bench_router_v4_hash),
        bench_primitive("quant_fp8_e4m3fn_e8m0", iterations, bench_quant_fp8),
        bench_primitive("quant_mxfp4_e2m1_e8m0", iterations, bench_quant_mxfp4),
        bench_primitive(
            "fused_inv_rope_fp8_quant",
            iterations,
            bench_inv_rope_fp8_quant,
        ),
        bench_primitive("mla_decode_mqa", iterations, bench_mla_decode),
        bench_primitive("qkv_rmsnorm", iterations, bench_qkv_rmsnorm),
        bench_primitive("kv_fp8_ds_mla_pack", iterations, bench_kv_fp8_ds_mla_pack),
        bench_primitive(
            "compressed_slot_mapping",
            iterations,
            bench_compressed_slot_mapping,
        ),
        bench_primitive("c128_topk_metadata", iterations, bench_c128_topk_metadata),
        bench_primitive("c4_indexer_topk", iterations, bench_c4_indexer_topk),
        bench_primitive("save_partial_states", iterations, bench_save_partial_states),
        bench_primitive(
            "compress_norm_rope_fp8_cache",
            iterations,
            bench_compress_norm_rope_fp8_cache,
        ),
        bench_primitive(
            "compress_norm_rope_mxfp4_cache",
            iterations,
            bench_compress_norm_rope_mxfp4_cache,
        ),
        bench_primitive("routed_moe_forward", iterations, bench_moe_forward),
        bench_primitive("megamoe_prepare", iterations, bench_megamoe_prepare),
        bench_primitive(
            "megamoe_prepare_eplb",
            iterations,
            bench_megamoe_prepare_eplb,
        ),
        bench_primitive("megamoe_experts", iterations, bench_megamoe_experts),
    ];
    Ok(deepseek_cuda_primitive_bench_report_json(
        iterations, &samples,
    ))
}

pub(crate) fn deepseek_cuda_primitive_bench_report_json(
    iterations: usize,
    samples: &[DeepSeekCudaPrimitiveBenchSample],
) -> String {
    let ok = samples
        .iter()
        .filter(|sample| sample.status == "ok")
        .count();
    let failed = samples
        .iter()
        .filter(|sample| sample.status == "failed")
        .count();
    let unavailable = samples
        .iter()
        .filter(|sample| sample.status == "unavailable")
        .count();
    let status = if failed > 0 {
        "failed"
    } else if unavailable > 0 {
        "unavailable"
    } else if ok == samples.len() {
        "ok"
    } else {
        "incomplete"
    };
    let refs = deepseek_vllm_reference_units();

    format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-cuda-primitive-bench-v1\",\"iterations\":{},\"primitive_samples_total\":{},\"primitive_samples_ok\":{},\"primitive_samples_unavailable\":{},\"primitive_samples_failed\":{},\"runtime_parity_status\":\"primitive_microbench_only\",\"performance_status\":\"not_vllm_end_to_end_comparable\",\"claim_allowed\":false,\"vllm_reference_units\":{},\"samples\":{}}}",
        status,
        iterations,
        samples.len(),
        ok,
        unavailable,
        failed,
        json_string_array(&refs),
        deepseek_cuda_primitive_bench_samples_json(samples),
    )
}

pub(crate) fn run_deepseek_cuda_readiness(config_path: Option<String>) -> Result<String, String> {
    let mla = nerva_cuda::deepseek_mla::probe::deepseek_mla_smoke();
    let moe = nerva_cuda::deepseek_moe::probe::deepseek_moe_smoke();
    let quant = nerva_cuda::deepseek_quant::probe::deepseek_quant_smoke();
    let inv_rope_quant =
        nerva_cuda::deepseek_quant::probe::deepseek_fused_inv_rope_fp8_quant_smoke();
    let router = nerva_cuda::deepseek_router::probe::deepseek_router_smoke();
    let qkv_norm = nerva_cuda::deepseek_mla::probe::deepseek_qkv_rmsnorm_smoke();
    let kv = nerva_cuda::deepseek_kv::probe::deepseek_kv_smoke();
    let compressed_slots = nerva_cuda::deepseek_kv::probe::deepseek_compressed_slot_mapping_smoke();
    let c128_topk = nerva_cuda::deepseek_kv::probe::deepseek_c128_topk_metadata_smoke();
    let c4_indexer_topk = nerva_cuda::deepseek_kv::probe::deepseek_c4_indexer_topk_smoke();
    let partial_states = nerva_cuda::deepseek_kv::probe::deepseek_save_partial_states_smoke();
    let compress_cache =
        nerva_cuda::deepseek_kv::probe::deepseek_compress_norm_rope_fp8_cache_smoke();
    let mxfp4_compress_cache =
        nerva_cuda::deepseek_kv::probe::deepseek_compress_norm_rope_mxfp4_cache_smoke();
    let mla_json = mla.to_json();
    let moe_json = moe.to_json();
    let quant_json = quant.to_json();
    let inv_rope_quant_json = inv_rope_quant.to_json();
    let router_json = router.to_json();
    let qkv_norm_json = qkv_norm.to_json();
    let kv_json = kv.to_json();
    let compressed_slots_json = compressed_slots.to_json();
    let c128_topk_json = c128_topk.to_json();
    let c4_indexer_topk_json = c4_indexer_topk.to_json();
    let partial_states_json = partial_states.to_json();
    let compress_cache_json = compress_cache.to_json();
    let mxfp4_compress_cache_json = mxfp4_compress_cache.to_json();
    let primitives = [
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_mla_decode_mqa_smoke",
            status: smoke_status_label(&mla.status),
            summary_json: &mla_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_routed_moe_smoke",
            status: smoke_status_label(&moe.status),
            summary_json: &moe_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_quant_block_dequant_smoke",
            status: smoke_status_label(&quant.status),
            summary_json: &quant_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_fused_inv_rope_fp8_quant_smoke",
            status: smoke_status_label(&inv_rope_quant.status),
            summary_json: &inv_rope_quant_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_router_smoke",
            status: smoke_status_label(&router.status),
            summary_json: &router_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_qkv_rmsnorm_smoke",
            status: smoke_status_label(&qkv_norm.status),
            summary_json: &qkv_norm_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_fp8_ds_mla_kv_pack_smoke",
            status: smoke_status_label(&kv.status),
            summary_json: &kv_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_compressed_slot_mapping_smoke",
            status: smoke_status_label(&compressed_slots.status),
            summary_json: &compressed_slots_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_c128_topk_metadata_smoke",
            status: smoke_status_label(&c128_topk.status),
            summary_json: &c128_topk_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_c4_indexer_topk_smoke",
            status: smoke_status_label(&c4_indexer_topk.status),
            summary_json: &c4_indexer_topk_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_save_partial_states_smoke",
            status: smoke_status_label(&partial_states.status),
            summary_json: &partial_states_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_compress_norm_rope_fp8_cache_smoke",
            status: smoke_status_label(&compress_cache.status),
            summary_json: &compress_cache_json,
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke",
            status: smoke_status_label(&mxfp4_compress_cache.status),
            summary_json: &mxfp4_compress_cache_json,
        },
    ];
    deepseek_cuda_readiness_report_json(config_path, &primitives)
}

pub(crate) fn deepseek_cuda_readiness_report_json(
    config_path: Option<String>,
    primitives: &[DeepSeekCudaPrimitiveReport<'_>],
) -> Result<String, String> {
    let metadata = match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            if !metadata.architecture.is_deepseek() {
                return Err(format!(
                    "deepseek-cuda-readiness requires a DeepSeek architecture, got {}",
                    metadata.architecture.as_str()
                ));
            }
            Some(metadata)
        }
        None => None,
    };
    let metadata_ref = metadata.as_ref();
    let architecture = metadata_ref.map(|metadata| metadata.architecture.as_str());
    let implemented = metadata_ref.map_or_else(Vec::new, implemented_primitives);
    let required = metadata_ref.map_or_else(Vec::new, required_execution_units);
    let execution_unit_status = metadata_ref
        .map(execution_unit_coverage)
        .unwrap_or_default();
    let vllm_refs = metadata_ref.map_or_else(Vec::new, |metadata| {
        vllm_reference_units(metadata.architecture)
    });
    let vllm_kv_cache_plan = metadata_ref
        .map(|metadata| {
            let cache_dtype = match metadata.architecture {
                HfArchitectureKind::DeepSeekV32 | HfArchitectureKind::DeepSeekV4 => "fp8_ds_mla",
                _ => "bfloat16",
            };
            plan_deepseek_vllm_kv_cache(metadata, cache_dtype).map(|plan| plan.to_json())
        })
        .transpose()
        .map_err(|err| format!("DeepSeek vLLM KV cache plan failed: {err:?}"))?;
    let passed = primitives
        .iter()
        .filter(|primitive| primitive.status == "ok")
        .count();
    let failed = primitives
        .iter()
        .filter(|primitive| primitive.status == "failed")
        .count();
    let unavailable = primitives
        .iter()
        .filter(|primitive| primitive.status == "unavailable")
        .count();
    let primitive_status = if failed > 0 {
        "failed"
    } else if unavailable > 0 {
        "unavailable"
    } else if passed == primitives.len() {
        "ok"
    } else {
        "incomplete"
    };
    let readiness_status = if primitive_status == "ok" {
        "primitive_smokes_ok"
    } else {
        "primitive_smokes_incomplete"
    };

    Ok(format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-cuda-readiness-v1\",\"architecture\":{},\"primitive_status\":\"{}\",\"primitive_smokes_passed\":{},\"primitive_smokes_total\":{},\"cuda_primitives\":{},\"implemented_primitives\":{},\"required_execution_units\":{},\"remaining_required_execution_units\":{},\"execution_unit_status\":{},\"vllm_reference_units\":{},\"vllm_kv_cache_plan\":{},\"runtime_parity_status\":\"not_verified\",\"performance_status\":\"not_benchmarked\",\"claim_allowed\":false}}",
        readiness_status,
        json_opt_architecture(architecture),
        primitive_status,
        passed,
        primitives.len(),
        cuda_primitives_json(primitives),
        json_string_array(&implemented),
        json_string_array(&required),
        json_string_array(&required),
        execution_unit_coverage_json(&execution_unit_status),
        json_string_array(&vllm_refs),
        vllm_kv_cache_plan.unwrap_or_else(|| "null".to_string()),
    ))
}

pub(crate) fn deepseek_runtime_plan_json(metadata: &HfModelMetadata) -> Result<String, String> {
    if !metadata.architecture.is_deepseek() {
        return Err(format!(
            "deepseek-runtime-plan requires a DeepSeek architecture, got {}",
            metadata.architecture.as_str()
        ));
    }

    let runtime_contract = match validate_exact_runtime_contract(metadata) {
        Ok(()) => RuntimeContractReport {
            status: "supported",
            reason: "exact runtime contract accepts this DeepSeek config".to_string(),
        },
        Err(err) => RuntimeContractReport {
            status: "unsupported",
            reason: format!("{err:?}"),
        },
    };
    let implemented = implemented_primitives(metadata);
    let units = required_execution_units(metadata);
    let execution_unit_status = execution_unit_coverage(metadata);
    let vllm_refs = vllm_reference_units(metadata.architecture);
    let layer_report = layer_report(metadata);
    let claim_allowed = runtime_contract.status == "supported";
    let v4_mhc_warmup_max_tokens = if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        DEEPSEEK_V4_MHC_AUTO_WARMUP_MAX_TOKENS.to_string()
    } else {
        "null".to_string()
    };
    let v4_mhc_warmup_token_sizes = if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        json_usize_array(&deepseek_v4_mhc_warmup_token_sizes(
            DEEPSEEK_V4_MHC_AUTO_WARMUP_MAX_TOKENS,
            &[],
        ))
    } else {
        "null".to_string()
    };

    Ok(format!(
        "{{\"status\":\"ok\",\"schema\":\"nerva-deepseek-runtime-plan-v1\",\"architecture\":\"{}\",\"layers\":{},\"hidden_size\":{},\"heads\":{},\"head_dim\":{},\"moe_layers\":{},\"dense_mlp_layers\":{},\"mla_layers\":{},\"v4_swa_layers\":{},\"v4_c4_layers\":{},\"v4_c128_layers\":{},\"v4_indexer_layers\":{},\"v4_hash_router_layers\":{},\"v4_mhc_warmup_max_tokens\":{},\"v4_mhc_warmup_token_sizes\":{},\"runtime_status\":\"{}\",\"runtime_reason\":\"{}\",\"implemented_primitives\":{},\"required_execution_units\":{},\"execution_unit_status\":{},\"vllm_reference_units\":{},\"claim_allowed\":{}}}",
        metadata.architecture.as_str(),
        metadata.num_hidden_layers,
        metadata.hidden_size,
        metadata.num_attention_heads,
        metadata.head_dim(),
        layer_report.moe_layers,
        layer_report.dense_mlp_layers,
        metadata.num_hidden_layers,
        layer_report.v4_swa_layers,
        layer_report.v4_c4_layers,
        layer_report.v4_c128_layers,
        layer_report.v4_indexer_layers,
        layer_report.v4_hash_router_layers,
        v4_mhc_warmup_max_tokens,
        v4_mhc_warmup_token_sizes,
        runtime_contract.status,
        json_escape(&runtime_contract.reason),
        json_string_array(&implemented),
        json_string_array(&units),
        execution_unit_coverage_json(&execution_unit_status),
        json_string_array(&vllm_refs),
        claim_allowed,
    ))
}

fn json_usize_array(values: &[usize]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

fn prompt_path_status(prompt_spec: &str) -> &'static str {
    let Some(path) = prompt_spec.strip_prefix('@') else {
        return "ok";
    };
    if path.is_empty() {
        return "missing";
    }
    if Path::new(path).is_file() {
        "ok"
    } else {
        "missing"
    }
}

fn vllm_reference_status(units: &[DeepSeekVllmReferenceUnit]) -> &'static str {
    let ok = units.iter().filter(|unit| unit.status == "ok").count();
    if units.iter().any(|unit| unit.status == "failed") {
        "failed"
    } else if units.iter().any(|unit| unit.status == "missing_file") {
        "missing_reference"
    } else if units.iter().any(|unit| unit.status == "symbol_gap") {
        "symbol_gap"
    } else if ok == units.len() {
        "ok"
    } else {
        "incomplete"
    }
}

struct RuntimeContractReport {
    status: &'static str,
    reason: String,
}

fn execution_unit_coverage_json(units: &[DeepSeekExecutionUnitCoverage]) -> String {
    let mut out = String::from("[");
    for (index, unit) in units.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"unit\":\"{}\",\"status\":\"{}\",\"validated_primitives\":{},\"remaining_gaps\":{}}}",
            json_escape(&unit.unit),
            unit.status,
            json_string_array(&unit.validated_primitives),
            json_string_array(&unit.remaining_gaps),
        ));
    }
    out.push(']');
    out
}

fn smoke_status_label(status: &nerva_cuda::smoke::status::SmokeStatus) -> &'static str {
    match status {
        nerva_cuda::smoke::status::SmokeStatus::Ok => "ok",
        nerva_cuda::smoke::status::SmokeStatus::Unavailable => "unavailable",
        nerva_cuda::smoke::status::SmokeStatus::Failed => "failed",
    }
}

fn deepseek_vllm_reference_specs() -> Vec<DeepSeekVllmReferenceSpec> {
    vec![
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v3_v32",
            execution_unit: "v3_mla_moe_model",
            relative_path: "vllm/model_executor/models/deepseek_v2.py",
            required_symbols: &[
                "class DeepseekV2MLAAttention",
                "class DeepseekV2MoE",
                "FusedMoE(",
                "MultiHeadLatentAttentionWrapper",
                "MLAAttentionSpec",
                "DeepseekV32IndexerBackend",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v3.2_v4",
            execution_unit: "sparse_indexer_metadata",
            relative_path: "vllm/v1/attention/backends/mla/indexer.py",
            required_symbols: &[
                "class DeepseekV32IndexerBackend",
                "class DeepseekV4IndexerBackend",
                "compress_ratio",
                "get_compressed_slot_mapping",
                "DeepseekV32IndexerMetadataBuilder",
                "get_supported_kernel_block_sizes",
                "return [1, 64] if current_platform.is_rocm() else [64]",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v3.2_v4",
            execution_unit: "mla_kv_cache_specs",
            relative_path: "vllm/v1/kv_cache_interface.py",
            required_symbols: &[
                "class MLAAttentionSpec",
                "class SlidingWindowMLASpec",
                "fp8_ds_mla",
                "compress_ratio",
                "real_page_size_bytes",
                "return self.block_size // self.compress_ratio",
                "return self.storage_block_size * 584",
                "return self.block_size * 656",
                "_apply_alignment_padding",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_packed_kv_cache_config",
            relative_path: "vllm/v1/core/kv_cache_utils.py",
            required_symbols: &[
                "def group_and_unify_kv_cache_specs",
                "def _get_kv_cache_groups_uniform_groups",
                "def _bucket_layers_by_page_size",
                "def _get_kv_cache_config_packed",
                "total_num_bytes_per_block",
                "block_stride=total_num_bytes_per_block",
                "DeepSeek V4 uses the packed layout by default",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_mhc_torch_reference",
            relative_path: "vllm/model_executor/kernels/mhc/torch.py",
            required_symbols: &[
                "def mhc_pre_torch",
                "torch.matmul",
                "torch.softmax",
                "sinkhorn_repeat",
                "def mhc_post_torch",
                "torch.einsum",
                "post_layer_mix.to(torch.float32)",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_mhc_tilelang_ops",
            relative_path: "vllm/model_executor/kernels/mhc/tilelang.py",
            required_symbols: &[
                "def mhc_pre_tilelang",
                "def mhc_post_tilelang",
                "def mhc_fused_post_pre_tilelang",
                "def hc_head_fused_kernel_tilelang",
                "hc_head_fuse_tilelang",
                "direct_register_custom_op",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_mhc_tilelang_warmup",
            relative_path: "vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py",
            required_symbols: &[
                "_AUTO_WARMUP_MAX_TOKENS = 16_384",
                "_DEFAULT_TOKEN_SIZE_CANDIDATES",
                "def _compute_mhc_pre_num_split",
                "block_k = 64",
                "block_m = 64",
                "split_k = min(split_k, num_block_k // 4)",
                "return max(split_k, 1)",
                "def _select_mhc_warmup_token_sizes",
                "def deepseek_v4_mhc_warmup",
                "_warmup_layer_mhc",
                "_warmup_hc_head",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_attention_graph",
            relative_path: "vllm/models/deepseek_v4/attention.py",
            required_symbols: &[
                "class DeepseekV4Attention",
                "_resolve_dsv4_kv_cache_dtype",
                "DeepseekCompressor",
                "execute_in_parallel",
                "MLAAttentionSpec",
                "compress_ratio",
                "fp8_ds_mla",
                "DeepseekV4SWACache",
                "alignment=576 if uses_fp8_ds_mla_layout else None",
                "self.quant_block_size = 128",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_multi_stream_attention_overlap",
            relative_path: "vllm/utils/multi_stream_utils.py",
            required_symbols: &[
                "def maybe_execute_in_parallel",
                "def execute_in_parallel",
                "start_event.record",
                "done_events[i].record",
                "ev.wait",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_compressor_cache",
            relative_path: "vllm/models/deepseek_v4/compressor.py",
            required_symbols: &[
                "class CompressorStateCache",
                "class DeepseekCompressor",
                "save_partial_states",
                "compress_norm_rope_store_triton",
                "compress_ratio",
                "SlidingWindowMLASpec",
                "alignment=576",
                "self.block_size = 4",
                "self.block_size = 8",
                "state_dim=2 * self.coff * self.head_dim",
                "CompressorMetadataBuilder",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_swa_cache_spec",
            relative_path: "vllm/v1/attention/backends/mla/sparse_swa.py",
            required_symbols: &[
                "class DeepseekV4SWACache",
                "self.block_size = 64",
                "SlidingWindowMLASpec",
                "alignment=576 if uses_fp8_ds_mla_layout else None",
                "model_version=\"deepseek_v4\"",
                "return (num_blocks, block_size, 584)",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_save_partial_states",
            relative_path: "vllm/models/deepseek_v4/common/ops/save_partial_states.py",
            required_symbols: &[
                "def save_partial_states",
                "_save_partial_states_kernel",
                "slot_id < 0",
                "score + ape",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_fused_qkv_rmsnorm",
            relative_path: "vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py",
            required_symbols: &[
                "def fused_q_kv_rmsnorm",
                "_fused_q_kv_rmsnorm_kernel",
                "num_tokens",
                "pid_task",
                "RMSNorm in fp32",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_sparse_mla_backend",
            relative_path: "vllm/models/deepseek_v4/sparse_mla.py",
            required_symbols: &[
                "class DeepseekV4FlashMLABackend",
                "FLASHMLA_SPARSE_DSV4",
                "fp8_ds_mla",
                "584",
                "return [256]",
                "return (num_blocks, block_size, 584)",
                "DeepseekV4FlashMLAMetadataBuilder",
                "build_c128a_topk_metadata",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_nvidia_megamoe_router_model",
            relative_path: "vllm/models/deepseek_v4/nvidia/model.py",
            required_symbols: &[
                "class DeepseekV4MegaMoEExperts",
                "prepare_megamoe_inputs",
                "fused_topk_bias",
                "class DeepseekV4MoE",
                "class DeepseekV4ForCausalLM",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_megamoe_prepare_kernel",
            relative_path: "vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py",
            required_symbols: &[
                "def prepare_megamoe_inputs",
                "_prepare_megamoe_inputs_kernel",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_fp8_o_projection",
            relative_path: "vllm/models/deepseek_v4/nvidia/ops/o_proj.py",
            required_symbols: &[
                "def compute_fp8_einsum_recipe",
                "def deep_gemm_fp8_o_proj",
                "fused_inv_rope_fp8_quant",
                "fp8_einsum",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_fused_inv_rope_fp8_quant",
            relative_path: "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
            required_symbols: &[
                "def fused_inv_rope_fp8_quant",
                "_fused_inv_rope_fp8_quant_per_head",
                "TMA_ALIGNED_SCALES",
                "float8e4nv",
                "packed_val",
            ],
        },
        DeepSeekVllmReferenceSpec {
            architecture: "deepseek_v4",
            execution_unit: "v4_fused_compress_quant_cache",
            relative_path: "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
            required_symbols: &[
                "compress_norm_rope_store_triton",
                "_fused_kv_compress_norm_rope_insert_sparse_attn",
                "_fused_kv_compress_norm_rope_insert_indexer_attn",
                "_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn",
                "COMPRESS_RATIO",
            ],
        },
    ]
}

fn read_vllm_reference_unit(
    vllm_root: &Path,
    spec: &DeepSeekVllmReferenceSpec,
) -> DeepSeekVllmReferenceUnit {
    let absolute = vllm_root.join(spec.relative_path);
    let required_symbols = spec
        .required_symbols
        .iter()
        .map(|symbol| (*symbol).to_string())
        .collect::<Vec<_>>();
    match std::fs::read(&absolute) {
        Ok(bytes) => {
            let source = String::from_utf8_lossy(&bytes);
            let missing_symbols = spec
                .required_symbols
                .iter()
                .filter(|symbol| !source.contains(**symbol))
                .map(|symbol| (*symbol).to_string())
                .collect::<Vec<_>>();
            let status = if missing_symbols.is_empty() {
                "ok"
            } else {
                "symbol_gap"
            };
            DeepSeekVllmReferenceUnit {
                architecture: spec.architecture,
                execution_unit: spec.execution_unit,
                relative_path: spec.relative_path,
                absolute_path: absolute.display().to_string(),
                status,
                size_bytes: bytes.len() as u64,
                fnv1a64: Some(fnv1a64(&bytes)),
                required_symbols,
                missing_symbols,
                error: None,
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => DeepSeekVllmReferenceUnit {
            architecture: spec.architecture,
            execution_unit: spec.execution_unit,
            relative_path: spec.relative_path,
            absolute_path: absolute.display().to_string(),
            status: "missing_file",
            size_bytes: 0,
            fnv1a64: None,
            required_symbols: required_symbols.clone(),
            missing_symbols: required_symbols,
            error: Some("vLLM reference file is missing".to_string()),
        },
        Err(err) => DeepSeekVllmReferenceUnit {
            architecture: spec.architecture,
            execution_unit: spec.execution_unit,
            relative_path: spec.relative_path,
            absolute_path: absolute.display().to_string(),
            status: "failed",
            size_bytes: 0,
            fnv1a64: None,
            required_symbols,
            missing_symbols: Vec::new(),
            error: Some(format!("failed to read vLLM reference file: {err}")),
        },
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn bench_primitive(
    name: &'static str,
    iterations: usize,
    mut op: impl FnMut() -> DeepSeekPrimitiveMetrics,
) -> DeepSeekCudaPrimitiveBenchSample {
    let mut last = None;
    let mut executed = 0usize;
    let start = Instant::now();
    for _ in 0..iterations {
        let metrics = op();
        executed += 1;
        let terminal = metrics.status != SmokeStatus::Ok;
        last = Some(metrics);
        if terminal {
            break;
        }
    }
    let total_wall_ns = start.elapsed().as_nanos();
    let avg_wall_ns = total_wall_ns / executed.max(1) as u128;
    let metrics = last.expect("bench primitive should execute at least once");

    DeepSeekCudaPrimitiveBenchSample {
        name: name.to_string(),
        status: smoke_status_label(&metrics.status),
        requested_iterations: iterations,
        executed_iterations: executed,
        total_wall_ns,
        avg_wall_ns,
        output_hash: metrics.output_hash,
        device_arena_bytes: metrics.device_arena_bytes,
        pinned_host_bytes: metrics.pinned_host_bytes,
        h2d_bytes_per_iter: metrics.h2d_bytes,
        d2h_bytes_per_iter: metrics.d2h_bytes,
        kernel_launches_per_iter: metrics.kernel_launches,
        sync_calls_per_iter: metrics.sync_calls,
        hot_path_allocations_per_iter: metrics.hot_path_allocations,
        error: metrics.error,
    }
}

fn bench_router_v3() -> DeepSeekPrimitiveMetrics {
    let summary = deepseek_router_route_v3_grouped_sigmoid(
        &[-2.0, 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0],
        Some(&[0.0, 0.0, 0.0, 4.0, 0.0, 0.0, -4.0, 0.0]),
        2,
        1,
        2,
        true,
        2.5,
    );
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_router_v4() -> DeepSeekPrimitiveMetrics {
    let summary = deepseek_router_route_v4_sqrtsoftplus(
        &[-2.0, 0.0, 1.0, 3.0],
        Some(&[0.0, 3.0, 0.0, -3.0]),
        2,
        true,
        1.5,
    );
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_router_v4_hash() -> DeepSeekPrimitiveMetrics {
    let hash_table = [
        0u32, 1, 3, // token 0
        2, 1, 3, // token 1
        3, 0, 2, // token 2
    ];
    let summary =
        deepseek_router_route_v4_hash(&[4.0, -1.0, 0.0, 2.0], &hash_table, 1, 3, true, 1.0);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_quant_fp8() -> DeepSeekPrimitiveMetrics {
    let weights = [
        0x38, 0x40, 0x30, 0xb8, 0x70, 0x77, 0x78, 0x7e, 0x20, 0x28, 0x30, 0x38,
    ];
    let scales = [0x7f, 0x80, 0x7e, 0x81];
    let summary = deepseek_fp8_e4m3fn_e8m0_dequant(&weights, &scales, 3, 4, 2, 2);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_quant_mxfp4() -> DeepSeekPrimitiveMetrics {
    let packed = [0x21, 0x76, 0xa9, 0xfe, 0x10, 0x54, 0x98, 0xdc];
    let scales = [0x7f, 0x80, 0x7e, 0x81];
    let summary = deepseek_mxfp4_e2m1_e8m0_dequant(&packed, &scales, 2, 4, 2);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_inv_rope_fp8_quant() -> DeepSeekPrimitiveMetrics {
    let input = [
        1.0, -2.0, 3.0, -4.0, -0.5, 1.5, -2.5, 3.5, 0.25, -0.75, 1.25, -1.5, -2.0, 2.25, -2.5, 2.75,
    ];
    let positions = [0i64, 1i64];
    let cos_sin_cache = [1.0, 0.0, 0.6, 0.8];
    let summary = deepseek_fused_inv_rope_fp8_quant(
        &input,
        &positions,
        &cos_sin_cache,
        2,
        1,
        2,
        4,
        2,
        2,
        2,
        448.0,
        1e-10,
    );
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.fp8_output_hash
            ^ summary.scale_output_hash
            ^ summary.packed_scale_output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_mla_decode() -> DeepSeekPrimitiveMetrics {
    let q_nope = [0.2, -0.3, 0.4, 0.1];
    let q_pe = [0.15, -0.25];
    let kv_c = [0.3, -0.1, 0.2, -0.4, 0.5, 0.1, 0.2, 0.4, -0.3];
    let k_pe = [0.05, -0.2, 0.3];
    let w_uk = [
        0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.6, -0.1, 0.7, 0.3, -0.2, 0.5,
    ];
    let w_uv = [
        0.2, -0.4, 0.5, 0.1, -0.3, 0.6, 0.4, -0.2, 0.7, 0.2, -0.1, 0.3,
    ];
    let summary = deepseek_mla_decode(CudaDeepSeekMlaDecodeInput {
        heads: 2,
        tokens: 3,
        kv_lora_rank: 3,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 1,
        v_head_dim: 2,
        softmax_scale: 0.7,
        q_nope: &q_nope,
        q_pe: &q_pe,
        kv_c: &kv_c,
        k_pe: &k_pe,
        w_uk: &w_uk,
        w_uv: &w_uv,
    });
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_qkv_rmsnorm() -> DeepSeekPrimitiveMetrics {
    let q = [
        1.0, -2.0, 3.0, -4.0, // token 0
        -0.5, 1.5, -2.5, 3.5, // token 1
    ];
    let kv = [
        0.25, -0.75, 1.25, // token 0
        -1.5, 2.0, -2.5, // token 1
    ];
    let q_weight = [0.5, 1.0, -1.5, 2.0];
    let kv_weight = [1.25, -0.5, 0.75];
    let summary = deepseek_qkv_rmsnorm(&q, &kv, &q_weight, &kv_weight, 2, 4, 3, 1e-5);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_kv_fp8_ds_mla_pack() -> DeepSeekPrimitiveMetrics {
    let nope = (0..448)
        .map(|idx| (idx as u8).wrapping_mul(5).wrapping_add(3))
        .collect::<Vec<_>>();
    let rope = (0..64)
        .map(|idx| 0x3f80u16.wrapping_add(idx as u16))
        .collect::<Vec<_>>();
    let scales = [0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x00];
    let summary = deepseek_fp8_ds_mla_pack(4, 2, &nope, &rope, &scales);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_compressed_slot_mapping() -> DeepSeekPrimitiveMetrics {
    let query_start_loc = [0, 5, 9];
    let seq_lens = [10, 7];
    let block_table = [
        20, 21, 22, 23, // request 0
        30, 31, 32, 33, // request 1
    ];
    let summary =
        deepseek_compressed_slot_mapping(&query_start_loc, &seq_lens, &block_table, 4, 4, 4);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_c128_topk_metadata() -> DeepSeekPrimitiveMetrics {
    let positions = [127, 255, 383, 511];
    let token_to_req = [0, 1, 0, 1];
    let block_table = [
        40, 41, 42, 43, // request 0
        50, 51, 52, 53, // request 1
    ];
    let slot_mapping = [10, -1, 12, 13];
    let summary = deepseek_c128_topk_metadata(
        &positions,
        2,
        &token_to_req,
        &block_table,
        4,
        &slot_mapping,
        2,
        128,
        4,
    );
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_c4_indexer_topk() -> DeepSeekPrimitiveMetrics {
    let query = [
        1.0, 0.0, // token 0, head 0
        0.0, 1.0, // token 0, head 1
        0.0, 2.0, // token 1, head 0
        1.0, 0.0, // token 1, head 1
    ];
    let key_cache = [
        1.0, 0.0, // slot 0
        0.0, 1.0, // slot 1
        1.0, 1.0, // slot 2
        -1.0, 0.5, // slot 3
    ];
    let weights = [
        1.0, 0.5, // token 0
        0.25, 2.0, // token 1
    ];
    let context_lens = [4, 2];
    let summary = deepseek_c4_indexer_topk(&query, &key_cache, &weights, &context_lens, 2, 2, 2, 2);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_save_partial_states() -> DeepSeekPrimitiveMetrics {
    let kv = [
        1.0, 2.0, 3.0, // token 0
        4.0, 5.0, 6.0, // token 1 skipped
        7.0, 8.0, 9.0, // token 2
    ];
    let score = [
        0.1, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1 skipped
        0.7, 0.8, 0.9, // token 2
    ];
    let ape = [
        10.0, 20.0, 30.0, // row 0
        40.0, 50.0, 60.0, // row 1
        70.0, 80.0, 90.0, // row 2
        100.0, 110.0, 120.0, // row 3
    ];
    let positions = [5, 6, 7];
    let slot_mapping = [1, -1, 4];
    let summary =
        deepseek_save_partial_states(&kv, &score, &ape, &positions, &slot_mapping, 4, 3, 4, 4, 2);
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_compress_norm_rope_fp8_cache() -> DeepSeekPrimitiveMetrics {
    let summary = nerva_cuda::deepseek_kv::probe::deepseek_compress_norm_rope_fp8_cache_smoke();
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_compress_norm_rope_mxfp4_cache() -> DeepSeekPrimitiveMetrics {
    let summary = nerva_cuda::deepseek_kv::probe::deepseek_compress_norm_rope_mxfp4_cache_smoke();
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_moe_forward() -> DeepSeekPrimitiveMetrics {
    let input = [1.2, -0.7, 0.3];
    let expert_ids = [1, 0];
    let expert_weights = [0.75, 0.25];
    let w_gate = [
        1.0, -0.5, 0.25, -0.25, 0.75, 1.25, 0.5, 0.2, -0.1, -1.0, 0.4, 0.3,
    ];
    let w_up = [
        -0.2, 0.4, 1.1, 0.8, -0.6, 0.2, 1.5, -0.3, 0.1, 0.7, 0.6, -0.4,
    ];
    let w_down = [
        0.3, -0.2, 0.4, 0.1, -0.5, 0.2, -0.7, 0.6, -0.1, 0.25, 0.35, -0.45,
    ];
    let summary = deepseek_moe_forward(CudaDeepSeekMoeForwardInput {
        hidden_size: 3,
        intermediate_size: 2,
        num_experts: 2,
        top_k: 2,
        clamp_swiglu: true,
        swiglu_limit: 1.0,
        input: &input,
        expert_ids: &expert_ids,
        expert_weights: &expert_weights,
        w_gate: &w_gate,
        w_up: &w_up,
        w_down: &w_down,
    });
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_megamoe_prepare() -> DeepSeekPrimitiveMetrics {
    let num_tokens = 2usize;
    let hidden_size = 128usize;
    let top_k = 3usize;
    let mut hidden_states = vec![0.0f32; num_tokens * hidden_size];
    for hidden in 0..hidden_size {
        hidden_states[hidden] = ((hidden % 11) as f32 - 5.0) * 0.125;
        hidden_states[hidden_size + hidden] = ((hidden % 7) as f32 - 3.0) * -0.25;
    }
    let topk_ids = [5i64, 2, 1, 7, 4, 3];
    let topk_weights = [0.5f32, 0.25, 0.125, 0.75, 0.125, 0.0625];
    let is_padding = [0u8, 1u8];
    let summary = deepseek_megamoe_prepare(CudaDeepSeekMegaMoePrepareInput {
        num_tokens: num_tokens as u32,
        hidden_size: hidden_size as u32,
        top_k: top_k as u32,
        hidden_states: &hidden_states,
        topk_ids: &topk_ids,
        topk_weights: &topk_weights,
        is_padding: Some(&is_padding),
        eplb_mapping: None,
    });
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.x_fp8_hash ^ summary.x_scales_hash ^ summary.topk_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_megamoe_prepare_eplb() -> DeepSeekPrimitiveMetrics {
    let num_tokens = 2usize;
    let hidden_size = 128usize;
    let top_k = 3usize;
    let mut hidden_states = vec![0.0f32; num_tokens * hidden_size];
    for token in 0..num_tokens {
        for hidden in 0..hidden_size {
            hidden_states[token * hidden_size + hidden] =
                (token as f32 + 1.0) * ((hidden % 13) as f32 - 6.0) * 0.0625;
        }
    }
    let topk_ids = [0i64, 1, 2, 0, 2, 9];
    let topk_weights = [0.5f32, 0.25, 0.125, 0.75, 0.125, 0.0625];
    let logical_to_physical_map = [
        10i64, 11, -1, //
        12, -1, -1, //
        13, 14, 15,
    ];
    let logical_replica_count = [2u32, 1, 3];
    let summary = deepseek_megamoe_prepare(CudaDeepSeekMegaMoePrepareInput {
        num_tokens: num_tokens as u32,
        hidden_size: hidden_size as u32,
        top_k: top_k as u32,
        hidden_states: &hidden_states,
        topk_ids: &topk_ids,
        topk_weights: &topk_weights,
        is_padding: None,
        eplb_mapping: Some(CudaDeepSeekMegaMoeEplbMapping {
            num_logical_experts: 3,
            map_slots: 3,
            expert_load_size: 16,
            record_expert_load: true,
            logical_to_physical_map: &logical_to_physical_map,
            logical_replica_count: &logical_replica_count,
        }),
    });
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.x_fp8_hash
            ^ summary.x_scales_hash
            ^ summary.topk_hash
            ^ summary.expert_load_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn bench_megamoe_experts() -> DeepSeekPrimitiveMetrics {
    let num_tokens = 2usize;
    let hidden_size = 128usize;
    let intermediate_size = 32usize;
    let num_experts = 2usize;
    let top_k = 2usize;
    let mut x_fp8 = vec![0u8; num_tokens * hidden_size];
    for token in 0..num_tokens {
        for hidden in 0..8usize {
            x_fp8[token * hidden_size + hidden] =
                [0x38u8, 0x30, 0x28, 0x40, 0x34, 0x3c, 0x2c, 0x44][hidden];
        }
    }
    let x_scales = vec![0x7f7f7f7fu32; num_tokens * (hidden_size / 128)];
    let topk_ids = [0i64, 1, 1, 0];
    let topk_weights = [0.75f32, 0.25, 0.5, 0.5];
    let (w13_packed, w13_scales, w2_packed, w2_scales) =
        megamoe_expert_weights(hidden_size, intermediate_size, num_experts);
    let summary = deepseek_megamoe_experts(CudaDeepSeekMegaMoeExpertsInput {
        num_tokens: num_tokens as u32,
        hidden_size: hidden_size as u32,
        intermediate_size: intermediate_size as u32,
        num_experts: num_experts as u32,
        top_k: top_k as u32,
        swiglu_limit: 7.0,
        x_fp8: &x_fp8,
        x_scales: &x_scales,
        topk_ids: &topk_ids,
        topk_weights: &topk_weights,
        w13_packed: &w13_packed,
        w13_scales: &w13_scales,
        w2_packed: &w2_packed,
        w2_scales: &w2_scales,
    });
    DeepSeekPrimitiveMetrics {
        status: summary.status,
        output_hash: summary.output_hash,
        device_arena_bytes: summary.device_arena_bytes,
        pinned_host_bytes: summary.pinned_host_bytes,
        h2d_bytes: summary.h2d_bytes,
        d2h_bytes: summary.d2h_bytes,
        kernel_launches: summary.kernel_launches,
        sync_calls: summary.sync_calls,
        hot_path_allocations: summary.hot_path_allocations,
        error: summary.error,
    }
}

fn megamoe_expert_weights(
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let w13_rows_per_expert = intermediate_size * 2;
    let w13_packed_cols = hidden_size / 2;
    let w13_scale_cols = hidden_size / 32;
    let w2_packed_cols = intermediate_size / 2;
    let w2_scale_cols = intermediate_size / 32;
    let mut w13_packed = vec![0u8; num_experts * w13_rows_per_expert * w13_packed_cols];
    let w13_scales = vec![127u8; num_experts * w13_rows_per_expert * w13_scale_cols];
    let mut w2_packed = vec![0u8; num_experts * hidden_size * w2_packed_cols];
    let w2_scales = vec![127u8; num_experts * hidden_size * w2_scale_cols];

    set_packed_nibble(&mut w13_packed, 0, 0, w13_packed_cols, 2);
    set_packed_nibble(&mut w13_packed, 0, 1, w13_packed_cols, 1);
    set_packed_nibble(&mut w13_packed, intermediate_size, 2, w13_packed_cols, 2);
    set_packed_nibble(&mut w2_packed, 0, 0, w2_packed_cols, 3);
    set_packed_nibble(&mut w2_packed, 1, 0, w2_packed_cols, 10);

    let expert_1_base = w13_rows_per_expert;
    set_packed_nibble(&mut w13_packed, expert_1_base + 1, 3, w13_packed_cols, 3);
    set_packed_nibble(
        &mut w13_packed,
        expert_1_base + intermediate_size + 1,
        4,
        w13_packed_cols,
        1,
    );
    let expert_1_w2_base = hidden_size;
    set_packed_nibble(&mut w2_packed, expert_1_w2_base + 2, 1, w2_packed_cols, 4);
    set_packed_nibble(&mut w2_packed, expert_1_w2_base + 3, 1, w2_packed_cols, 9);

    (w13_packed, w13_scales, w2_packed, w2_scales)
}

fn set_packed_nibble(packed: &mut [u8], row: usize, col: usize, packed_cols: usize, nibble: u8) {
    let index = row * packed_cols + col / 2;
    if col & 1 == 0 {
        packed[index] = (packed[index] & 0xf0) | (nibble & 0x0f);
    } else {
        packed[index] = (packed[index] & 0x0f) | ((nibble & 0x0f) << 4);
    }
}

fn cuda_primitives_json(primitives: &[DeepSeekCudaPrimitiveReport<'_>]) -> String {
    let mut out = String::from("[");
    for (index, primitive) in primitives.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"name\":\"{}\",\"status\":\"{}\",\"summary\":{}}}",
            json_escape(primitive.name),
            json_escape(primitive.status),
            primitive.summary_json,
        ));
    }
    out.push(']');
    out
}

fn deepseek_vllm_reference_units_json(units: &[DeepSeekVllmReferenceUnit]) -> String {
    let mut out = String::from("[");
    for (index, unit) in units.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"architecture\":\"{}\",\"execution_unit\":\"{}\",\"relative_path\":\"{}\",\"absolute_path\":\"{}\",\"status\":\"{}\",\"size_bytes\":{},\"fnv1a64\":{},\"required_symbols\":{},\"missing_symbols\":{},\"error\":{}}}",
            unit.architecture,
            json_escape(unit.execution_unit),
            json_escape(unit.relative_path),
            json_escape(&unit.absolute_path),
            unit.status,
            unit.size_bytes,
            json_opt_hash(unit.fnv1a64),
            json_string_array(&unit.required_symbols),
            json_string_array(&unit.missing_symbols),
            json_opt_string(unit.error.as_deref()),
        ));
    }
    out.push(']');
    out
}

fn deepseek_cuda_primitive_bench_samples_json(
    samples: &[DeepSeekCudaPrimitiveBenchSample],
) -> String {
    let mut out = String::from("[");
    for (index, sample) in samples.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"name\":\"{}\",\"status\":\"{}\",\"requested_iterations\":{},\"executed_iterations\":{},\"total_wall_ns\":{},\"avg_wall_ns\":{},\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes_per_iter\":{},\"D2H_bytes_per_iter\":{},\"kernel_launches_per_iter\":{},\"sync_calls_per_iter\":{},\"hot_path_allocations_per_iter\":{},\"error\":{}}}",
            json_escape(&sample.name),
            sample.status,
            sample.requested_iterations,
            sample.executed_iterations,
            sample.total_wall_ns,
            sample.avg_wall_ns,
            sample.output_hash,
            sample.device_arena_bytes,
            sample.pinned_host_bytes,
            sample.h2d_bytes_per_iter,
            sample.d2h_bytes_per_iter,
            sample.kernel_launches_per_iter,
            sample.sync_calls_per_iter,
            sample.hot_path_allocations_per_iter,
            json_opt_string(sample.error.as_deref()),
        ));
    }
    out.push(']');
    out
}

struct DeepSeekCommandRun {
    status: i32,
    json: String,
    stderr_tail: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn default_deepseek_artifact_dir() -> PathBuf {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "nerva-deepseek-vllm-benchmark-{}-{millis}",
        std::process::id()
    ))
}

fn deepseek_vllm_generate_command(
    repo_root: &Path,
    vllm_root: &Path,
    checkpoint_dir: &str,
    prompt_spec: &str,
    max_context_tokens: usize,
    max_new_tokens: usize,
) -> Vec<String> {
    let venv_python = vllm_root.join(".venv/bin/python");
    let python = if venv_python.is_file() {
        venv_python.display().to_string()
    } else {
        "python3".to_string()
    };
    vec![
        python,
        repo_root
            .join("tools/deepseek_vllm_generate.py")
            .display()
            .to_string(),
        "--vllm-root".to_string(),
        vllm_root.display().to_string(),
        "--model".to_string(),
        checkpoint_dir.to_string(),
        "--prompt".to_string(),
        prompt_spec.to_string(),
        "--max-model-len".to_string(),
        max_context_tokens.to_string(),
        "--max-tokens".to_string(),
        max_new_tokens.to_string(),
        "--temperature".to_string(),
        "0".to_string(),
        "--top-p".to_string(),
        "1".to_string(),
        "--top-k".to_string(),
        "0".to_string(),
        "--seed".to_string(),
        "0".to_string(),
        "--dtype".to_string(),
        "bfloat16".to_string(),
        "--runs".to_string(),
        "3".to_string(),
        "--warmup-runs".to_string(),
        "1".to_string(),
    ]
}

fn deepseek_nerva_generate_command(
    checkpoint_dir: &str,
    prompt_spec: &str,
    max_context_tokens: usize,
    max_new_tokens: usize,
) -> Vec<String> {
    vec![
        "cargo".to_string(),
        "run".to_string(),
        "--release".to_string(),
        "-p".to_string(),
        "nerva".to_string(),
        "--".to_string(),
        "--json".to_string(),
        "--raw".to_string(),
        "-m".to_string(),
        checkpoint_dir.to_string(),
        "-p".to_string(),
        prompt_spec.to_string(),
        "-c".to_string(),
        max_context_tokens.to_string(),
        "-o".to_string(),
        max_new_tokens.to_string(),
        "--temperature".to_string(),
        "0".to_string(),
        "--top-p".to_string(),
        "1".to_string(),
        "--top-k".to_string(),
        "0".to_string(),
        "--seed".to_string(),
        "0".to_string(),
    ]
}

fn run_json_command(command: &[String], cwd: &Path) -> Result<DeepSeekCommandRun, String> {
    let executable = command
        .first()
        .ok_or_else(|| "cannot run empty command".to_string())?;
    let output = Command::new(executable)
        .args(&command[1..])
        .current_dir(cwd)
        .output()
        .map_err(|err| format!("failed to run {}: {err}", command.join(" ")))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let json = extract_json_payload(&stdout).unwrap_or_else(|_| {
        format!(
            "{{\"status\":\"command_failed\",\"stdout_tail\":\"{}\"}}",
            json_escape(&tail_chars(&stdout, 4096))
        )
    });
    Ok(DeepSeekCommandRun {
        status: output.status.code().unwrap_or(-1),
        json,
        stderr_tail: tail_chars(&stderr, 4096),
    })
}

fn extract_json_payload(stdout: &str) -> Result<String, String> {
    let start = stdout
        .find('{')
        .ok_or_else(|| "command stdout did not contain a JSON object".to_string())?;
    let end = stdout
        .rfind('}')
        .ok_or_else(|| "command stdout did not contain a complete JSON object".to_string())?;
    if end < start {
        return Err("command stdout JSON bounds are invalid".to_string());
    }
    Ok(stdout[start..=end].trim().to_string())
}

fn tail_chars(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars {
        return value.to_string();
    }
    value.chars().skip(len - max_chars).collect()
}

fn command_run_json(run: Option<&DeepSeekCommandRun>) -> String {
    match run {
        Some(run) => format!(
            "{{\"exit_status\":{},\"stderr_tail\":\"{}\"}}",
            run.status,
            json_escape(&run.stderr_tail)
        ),
        None => "null".to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
fn deepseek_benchmark_run_json(
    status: &str,
    checkpoint_dir: &str,
    prompt_spec: &str,
    max_context_tokens: usize,
    max_new_tokens: usize,
    artifact_dir: &Path,
    vllm_artifact: &Path,
    nerva_artifact: &Path,
    compare_artifact: &Path,
    vllm_command: &[String],
    nerva_command: &[String],
    vllm_run: &DeepSeekCommandRun,
    nerva_run: Option<&DeepSeekCommandRun>,
    compare_json: Option<&str>,
) -> String {
    let claim_allowed = compare_json
        .and_then(|json| find_first_json_string_field(json, "status").ok().flatten())
        .is_some_and(|status| status == "ok");
    format!(
        "{{\"status\":\"{}\",\"schema\":\"nerva-deepseek-vllm-benchmark-run-v1\",\"checkpoint_dir\":\"{}\",\"prompt_spec\":\"{}\",\"max_context_tokens\":{},\"max_new_tokens\":{},\"sampler\":{{\"temperature\":0,\"top_p\":1,\"top_k\":0,\"seed\":0}},\"artifact_dir\":\"{}\",\"artifacts\":{{\"vllm\":\"{}\",\"nerva\":\"{}\",\"compare\":\"{}\"}},\"commands\":{{\"vllm_generate\":{},\"nerva_generate\":{}}},\"runs\":{{\"vllm\":{},\"nerva\":{}}},\"compare\":{},\"claim_allowed\":{}}}",
        status,
        json_escape(checkpoint_dir),
        json_escape(prompt_spec),
        max_context_tokens,
        max_new_tokens,
        json_escape(&artifact_dir.display().to_string()),
        json_escape(&vllm_artifact.display().to_string()),
        json_escape(&nerva_artifact.display().to_string()),
        json_escape(&compare_artifact.display().to_string()),
        json_string_array(vllm_command),
        json_string_array(nerva_command),
        command_run_json(Some(vllm_run)),
        command_run_json(nerva_run),
        compare_json.unwrap_or("null"),
        claim_allowed,
    )
}

fn json_opt_hash(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"0x{value:016x}\""))
}

fn json_opt_string(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_f64(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() && value.fract() == 0.0 => format!("{value:.1}"),
        Some(value) if value.is_finite() => value.to_string(),
        _ => "null".to_string(),
    }
}

fn deepseek_compare_blocking_reasons(
    model_comparable: bool,
    model_parity: bool,
    prompt_comparable: bool,
    prompt_token_parity: bool,
    sampler_comparable: bool,
    sampler_parity: bool,
    token_parity: bool,
    text_parity: bool,
    throughput_comparable: bool,
    throughput_ok: bool,
    latency_comparable: bool,
    latency_ok: bool,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if !model_comparable {
        reasons.push("model/path identity is missing from one or both artifacts".to_string());
    } else if !model_parity {
        reasons.push("model/path identity differs".to_string());
    }
    if !prompt_comparable {
        reasons.push("prompt_token_ids is missing from one or both artifacts".to_string());
    } else if !prompt_token_parity {
        reasons.push("prompt token IDs differ".to_string());
    }
    if !sampler_comparable {
        reasons.push(
            "sampler temperature/top_p/top_k/seed is missing from one or both artifacts"
                .to_string(),
        );
    } else if !sampler_parity {
        reasons.push("sampler temperature/top_p/top_k/seed differs".to_string());
    }
    if !token_parity {
        reasons.push("generated token IDs differ".to_string());
    }
    if !text_parity {
        reasons.push("generated text differs or is missing".to_string());
    }
    if !throughput_comparable {
        reasons.push("tokens_per_second is missing from one or both artifacts".to_string());
    } else if !throughput_ok {
        reasons.push("NERVA throughput is below vLLM throughput".to_string());
    }
    if !latency_comparable {
        reasons.push("p99 latency is missing from one or both artifacts".to_string());
    } else if !latency_ok {
        reasons.push("NERVA p99 latency is above vLLM p99 latency".to_string());
    }
    reasons
}

fn find_deepseek_artifact_identity(
    source: &str,
    keys: &[&'static str],
) -> Result<Option<String>, String> {
    for key in keys {
        if let Some(value) = find_first_json_string_field(source, key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn normalized_artifact_identity(value: &str) -> String {
    let trimmed = value.trim();
    if let Ok(canonical) = std::fs::canonicalize(trimmed) {
        return canonical.display().to_string();
    }
    if trimmed.len() > 1 {
        trimmed.trim_end_matches('/').to_string()
    } else {
        trimmed.to_string()
    }
}

fn find_deepseek_sampler_artifact(source: &str) -> Result<DeepSeekSamplerArtifact, String> {
    Ok(DeepSeekSamplerArtifact {
        temperature: find_first_json_number_field(source, "temperature")?,
        top_p: find_first_json_number_field(source, "top_p")?,
        top_k: find_first_json_number_field(source, "top_k")?,
        seed: find_first_json_number_field(source, "seed")?,
    })
}

fn same_json_number(left: Option<f64>, right: Option<f64>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => (left - right).abs() <= 1e-9,
        (None, None) => true,
        _ => false,
    }
}

fn find_deepseek_throughput_metric(source: &str) -> Result<Option<DeepSeekArtifactMetric>, String> {
    for key in [
        "post_load_tokens_per_second",
        "critical_path_tokens_per_second",
        "decode_tokens_per_second",
        "tokens_per_second",
        "end_to_end_tokens_per_second",
    ] {
        if let Some(value) = find_first_json_number_field(source, key)? {
            return Ok(Some(DeepSeekArtifactMetric { value, source: key }));
        }
    }
    Ok(None)
}

fn find_deepseek_p99_metric(source: &str) -> Result<Option<DeepSeekArtifactMetric>, String> {
    for key in ["token_p99_ms", "p99_ms", "decode_p99_ms"] {
        if let Some(value) = find_first_json_number_field(source, key)? {
            return Ok(Some(DeepSeekArtifactMetric { value, source: key }));
        }
    }
    if let Some(value) = token_wall_latency_p99_ms(source)? {
        return Ok(Some(DeepSeekArtifactMetric {
            value,
            source: "token_critical_paths.wall_latency_ns",
        }));
    }
    Ok(None)
}

fn token_wall_latency_p99_ms(source: &str) -> Result<Option<f64>, String> {
    let mut latencies = find_all_json_number_fields(source, "wall_latency_ns")?;
    if latencies.is_empty() {
        return Ok(None);
    }
    latencies.sort_by(|left, right| left.total_cmp(right));
    let rank = ((latencies.len() as f64) * 0.99).ceil().max(1.0) as usize;
    let index = rank.saturating_sub(1).min(latencies.len() - 1);
    Ok(Some(latencies[index] / 1_000_000.0))
}

fn json_opt_metric_value(metric: Option<&DeepSeekArtifactMetric>) -> String {
    json_opt_f64(metric.map(|metric| metric.value))
}

fn json_opt_metric_source(metric: Option<&DeepSeekArtifactMetric>) -> String {
    metric
        .map(|metric| format!("\"{}\"", json_escape(metric.source)))
        .unwrap_or_else(|| "null".to_string())
}

fn find_all_json_number_fields(source: &str, key: &str) -> Result<Vec<f64>, String> {
    let bytes = source.as_bytes();
    let mut index = 0usize;
    let mut values = Vec::new();
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let (field, after_field) = parse_json_string_at(source, index)?;
        index = after_field;
        if field != key {
            continue;
        }
        let colon = skip_json_ws(bytes, after_field);
        if bytes.get(colon) != Some(&b':') {
            return Err(format!("JSON key {key} is missing ':'"));
        }
        let start = skip_json_ws(bytes, colon + 1);
        if let Some(value) = parse_json_number_at(source, key, start)? {
            values.push(value);
        }
        index = start.saturating_add(1);
    }
    Ok(values)
}

fn parse_json_number_at(source: &str, key: &str, start: usize) -> Result<Option<f64>, String> {
    let bytes = source.as_bytes();
    if bytes.get(start..start + 4) == Some(b"null") {
        return Ok(None);
    }
    let mut end = start;
    while matches!(
        bytes.get(end),
        Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
    ) {
        end += 1;
    }
    if end == start {
        return Err(format!("JSON key {key} is not numeric"));
    }
    source[start..end]
        .parse::<f64>()
        .map(Some)
        .map_err(|_| format!("JSON key {key} is not a valid f64"))
}

fn find_first_json_number_field(source: &str, key: &str) -> Result<Option<f64>, String> {
    let Some(start) = find_json_value_start_for_key(source, key)? else {
        return Ok(None);
    };
    parse_json_number_at(source, key, start)
}

fn find_first_json_string_field(source: &str, key: &str) -> Result<Option<String>, String> {
    let Some(start) = find_json_value_start_for_key(source, key)? else {
        return Ok(None);
    };
    if source.as_bytes().get(start) != Some(&b'"') {
        return Ok(None);
    }
    parse_json_string_at(source, start).map(|(value, _)| Some(value))
}

fn find_json_value_start_for_key(source: &str, key: &str) -> Result<Option<usize>, String> {
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let (field, after_field) = parse_json_string_at(source, index)?;
        index = after_field;
        if field != key {
            continue;
        }
        let colon = skip_json_ws(bytes, after_field);
        if bytes.get(colon) != Some(&b':') {
            return Err(format!("JSON key {key} is missing ':'"));
        }
        return Ok(Some(skip_json_ws(bytes, colon + 1)));
    }
    Ok(None)
}

fn parse_json_string_at(source: &str, start: usize) -> Result<(String, usize), String> {
    if source.as_bytes().get(start) != Some(&b'"') {
        return Err("expected JSON string".to_string());
    }
    let mut out = String::new();
    let mut chars = source[start + 1..].char_indices();
    while let Some((offset, ch)) = chars.next() {
        let index = start + 1 + offset;
        match ch {
            '"' => return Ok((out, index + 1)),
            '\\' => {
                let Some((_, escaped)) = chars.next() else {
                    return Err("JSON string escape is incomplete".to_string());
                };
                match escaped {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let mut codepoint = 0u32;
                        for _ in 0..4 {
                            let Some((_, hex)) = chars.next() else {
                                return Err("JSON unicode escape is incomplete".to_string());
                            };
                            let Some(value) = hex.to_digit(16) else {
                                return Err("JSON unicode escape has non-hex digit".to_string());
                            };
                            codepoint = (codepoint << 4) | value;
                        }
                        let Some(decoded) = char::from_u32(codepoint) else {
                            return Err("JSON unicode escape is invalid".to_string());
                        };
                        out.push(decoded);
                    }
                    _ => return Err("unsupported JSON string escape".to_string()),
                }
            }
            ch => out.push(ch),
        }
    }
    Err("JSON string is not closed".to_string())
}

fn skip_json_ws(bytes: &[u8], mut index: usize) -> usize {
    while matches!(bytes.get(index), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        index += 1;
    }
    index
}

fn json_opt_architecture(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn vllm_reference_units(architecture: HfArchitectureKind) -> Vec<String> {
    match architecture {
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32 => vec![
            "/root/vllm/vllm/model_executor/models/deepseek_v2.py".to_string(),
            "/root/vllm/vllm/v1/attention/backends/mla/indexer.py".to_string(),
            "/root/vllm/vllm/model_executor/layers/fused_moe".to_string(),
        ],
        HfArchitectureKind::DeepSeekV4 => vec![
            "/root/vllm/vllm/models/deepseek_v4/nvidia/model.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/attention.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/compressor.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/sparse_mla.py".to_string(),
            "/root/vllm/vllm/model_executor/kernels/mhc/torch.py".to_string(),
            "/root/vllm/vllm/model_executor/kernels/mhc/tilelang.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/common/ops/save_partial_states.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py".to_string(),
            "/root/vllm/vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py".to_string(),
            "/root/vllm/vllm/v1/kv_cache_interface.py".to_string(),
        ],
        _ => Vec::new(),
    }
}

fn deepseek_vllm_reference_units() -> Vec<String> {
    let mut refs = vllm_reference_units(HfArchitectureKind::DeepSeekV3);
    refs.extend(vllm_reference_units(HfArchitectureKind::DeepSeekV4));
    refs.push("/root/vllm/vllm/models/deepseek_v4/nvidia/ops/o_proj.py".to_string());
    refs.push("/root/vllm/vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py".to_string());
    refs.push(
        "/root/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py".to_string(),
    );
    refs.sort();
    refs.dedup();
    refs
}
