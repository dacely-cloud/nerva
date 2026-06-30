use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::dtype::dtype_to_str;
use crate::common::json::format::{json_escape, json_opt_str, json_opt_usize};
use crate::hf::architecture::HfArchitectureKind;
use crate::hf::metadata::HfModelMetadata;

const VLLM_DEEPSEEK_V3_BLOCK_SIZE: usize = 64;
const VLLM_DEEPSEEK_V4_BLOCK_SIZE: usize = 256;
const VLLM_DEEPSEEK_V4_SWA_BLOCK_SIZE: usize = 64;
const VLLM_DEEPSEEK_V4_FP8_DS_MLA_BYTES_PER_TOKEN: usize = 584;
const VLLM_DEEPSEEK_V32_FP8_DS_MLA_BYTES_PER_TOKEN: usize = 656;
const VLLM_DEEPSEEK_INDEXER_QUANT_BLOCK: usize = 128;
const VLLM_DEEPSEEK_V4_ALIGNMENT: usize = 576;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekMlaDimensions {
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub semantic_head_size: usize,
}

impl DeepSeekMlaDimensions {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"q_lora_rank\":{},\"kv_lora_rank\":{},\"qk_nope_head_dim\":{},\"qk_rope_head_dim\":{},\"v_head_dim\":{},\"semantic_head_size\":{}}}",
            json_opt_usize(self.q_lora_rank),
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.semantic_head_size,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekVllmKvCacheSpec {
    pub kind: &'static str,
    pub block_size: usize,
    pub storage_block_size: usize,
    pub num_kv_heads: usize,
    pub head_size: usize,
    pub dtype: DType,
    pub cache_dtype_str: String,
    pub compress_ratio: usize,
    pub sliding_window: Option<usize>,
    pub alignment: Option<usize>,
    pub model_version: Option<&'static str>,
    pub real_page_size_bytes: usize,
    pub page_size_bytes: usize,
}

impl DeepSeekVllmKvCacheSpec {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"kind\":\"{}\",\"block_size\":{},\"storage_block_size\":{},\"num_kv_heads\":{},\"head_size\":{},\"dtype\":\"{}\",\"cache_dtype_str\":\"{}\",\"compress_ratio\":{},\"sliding_window\":{},\"alignment\":{},\"model_version\":{},\"real_page_size_bytes\":{},\"page_size_bytes\":{}}}",
            self.kind,
            self.block_size,
            self.storage_block_size,
            self.num_kv_heads,
            self.head_size,
            dtype_to_str(self.dtype),
            json_escape(&self.cache_dtype_str),
            self.compress_ratio,
            json_opt_usize(self.sliding_window),
            json_opt_usize(self.alignment),
            json_opt_str(self.model_version),
            self.real_page_size_bytes,
            self.page_size_bytes,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekVllmKvCacheGroup {
    pub name: String,
    pub layers: usize,
    pub spec: DeepSeekVllmKvCacheSpec,
}

impl DeepSeekVllmKvCacheGroup {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"layers\":{},\"spec\":{}}}",
            json_escape(&self.name),
            self.layers,
            self.spec.to_json(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekVllmKvCachePlan {
    pub architecture: HfArchitectureKind,
    pub default_block_size: usize,
    pub cache_dtype_str: String,
    pub mla_dimensions: DeepSeekMlaDimensions,
    pub groups: Vec<DeepSeekVllmKvCacheGroup>,
    pub vllm_reference_units: Vec<&'static str>,
}

impl DeepSeekVllmKvCachePlan {
    pub fn to_json(&self) -> String {
        let groups = json_groups(&self.groups);
        let refs = json_static_str_array(&self.vllm_reference_units);
        format!(
            "{{\"architecture\":\"{}\",\"default_block_size\":{},\"cache_dtype_str\":\"{}\",\"mla_dimensions\":{},\"groups\":{},\"vllm_reference_units\":{}}}",
            self.architecture.as_str(),
            self.default_block_size,
            json_escape(&self.cache_dtype_str),
            self.mla_dimensions.to_json(),
            groups,
            refs,
        )
    }
}

pub fn default_vllm_deepseek_block_size(metadata: &HfModelMetadata) -> Result<usize> {
    match metadata.architecture {
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32 => {
            Ok(VLLM_DEEPSEEK_V3_BLOCK_SIZE)
        }
        HfArchitectureKind::DeepSeekV4 => Ok(VLLM_DEEPSEEK_V4_BLOCK_SIZE),
        _ => Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek vLLM KV cache planning requires a DeepSeek architecture, got {}",
                metadata.architecture.as_str()
            ),
        }),
    }
}

pub fn plan_deepseek_vllm_kv_cache(
    metadata: &HfModelMetadata,
    cache_dtype_str: &str,
) -> Result<DeepSeekVllmKvCachePlan> {
    let block_size = default_vllm_deepseek_block_size(metadata)?;
    plan_deepseek_vllm_kv_cache_with_block_size(metadata, cache_dtype_str, block_size)
}

pub fn plan_deepseek_vllm_kv_cache_with_block_size(
    metadata: &HfModelMetadata,
    cache_dtype_str: &str,
    block_size: usize,
) -> Result<DeepSeekVllmKvCachePlan> {
    if block_size == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek vLLM KV cache block size must be non-zero".to_string(),
        });
    }
    let mla_dimensions = deepseek_mla_dimensions(metadata)?;
    let normalized_cache_dtype = normalize_cache_dtype(metadata.architecture, cache_dtype_str)?;
    let mut groups = Vec::new();

    match metadata.architecture {
        HfArchitectureKind::DeepSeekV3 => {
            groups.push(DeepSeekVllmKvCacheGroup {
                name: "v3_main_mla".to_string(),
                layers: metadata.num_hidden_layers,
                spec: mla_spec(
                    "mla_attention",
                    block_size,
                    mla_dimensions.semantic_head_size,
                    normalized_cache_dtype,
                    1,
                    None,
                    None,
                    None,
                    metadata.architecture,
                )?,
            });
        }
        HfArchitectureKind::DeepSeekV32 => {
            groups.push(DeepSeekVllmKvCacheGroup {
                name: "v3_2_main_mla".to_string(),
                layers: metadata.num_hidden_layers,
                spec: mla_spec(
                    "mla_attention",
                    block_size,
                    mla_dimensions.semantic_head_size,
                    normalized_cache_dtype,
                    1,
                    None,
                    None,
                    None,
                    metadata.architecture,
                )?,
            });
            groups.push(DeepSeekVllmKvCacheGroup {
                name: "v3_2_sparse_indexer".to_string(),
                layers: metadata.num_hidden_layers,
                spec: indexer_spec(
                    "mla_indexer",
                    block_size,
                    required(metadata.index_head_dim, "index_head_dim")?,
                    normalized_cache_dtype,
                    1,
                    None,
                    None,
                    metadata.architecture,
                )?,
            });
        }
        HfArchitectureKind::DeepSeekV4 => {
            let mut swa_layers = 0usize;
            let mut c4_layers = 0usize;
            let mut c128_layers = 0usize;
            for ratio in v4_layer_compress_ratios(metadata)? {
                match ratio {
                    0 | 1 => swa_layers += 1,
                    4 => c4_layers += 1,
                    128 => c128_layers += 1,
                    other => {
                        return Err(NervaError::InvalidArgument {
                            reason: format!(
                                "DeepSeek V4 unsupported compress_ratio {other}; vLLM accepts 0/1, 4, or 128"
                            ),
                        });
                    }
                }
            }
            if swa_layers > 0 {
                groups.push(DeepSeekVllmKvCacheGroup {
                    name: "v4_swa".to_string(),
                    layers: swa_layers,
                    spec: mla_spec(
                        "sliding_window_mla",
                        VLLM_DEEPSEEK_V4_SWA_BLOCK_SIZE,
                        mla_dimensions.semantic_head_size,
                        normalized_cache_dtype,
                        1,
                        metadata.sliding_window,
                        v4_alignment(normalized_cache_dtype),
                        Some("deepseek_v4"),
                        metadata.architecture,
                    )?,
                });
            }
            for (name, ratio, layers) in [
                ("v4_c4_mla", 4usize, c4_layers),
                ("v4_c128_mla", 128, c128_layers),
            ] {
                if layers == 0 {
                    continue;
                }
                groups.push(DeepSeekVllmKvCacheGroup {
                    name: name.to_string(),
                    layers,
                    spec: mla_spec(
                        "mla_attention",
                        block_size,
                        mla_dimensions.semantic_head_size,
                        normalized_cache_dtype,
                        ratio,
                        None,
                        v4_alignment(normalized_cache_dtype),
                        Some("deepseek_v4"),
                        metadata.architecture,
                    )?,
                });
                groups.push(DeepSeekVllmKvCacheGroup {
                    name: format!("{name}_indexer"),
                    layers,
                    spec: indexer_spec(
                        "mla_indexer",
                        block_size,
                        required(metadata.index_head_dim, "index_head_dim")?,
                        normalized_cache_dtype,
                        ratio,
                        Some(VLLM_DEEPSEEK_V4_ALIGNMENT),
                        Some("deepseek_v4"),
                        metadata.architecture,
                    )?,
                });
            }
        }
        _ => {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "DeepSeek vLLM KV cache planning requires a DeepSeek architecture, got {}",
                    metadata.architecture.as_str()
                ),
            });
        }
    }

    Ok(DeepSeekVllmKvCachePlan {
        architecture: metadata.architecture,
        default_block_size: block_size,
        cache_dtype_str: normalized_cache_dtype.to_string(),
        mla_dimensions,
        groups,
        vllm_reference_units: vllm_kv_reference_units(metadata.architecture),
    })
}

pub fn deepseek_mla_dimensions(metadata: &HfModelMetadata) -> Result<DeepSeekMlaDimensions> {
    match metadata.architecture {
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32 => {
            let kv_lora_rank = required(metadata.kv_lora_rank, "kv_lora_rank")?;
            let qk_nope_head_dim = required(metadata.qk_nope_head_dim, "qk_nope_head_dim")?;
            let qk_rope_head_dim = required(metadata.qk_rope_head_dim, "qk_rope_head_dim")?;
            let v_head_dim = required(metadata.v_head_dim, "v_head_dim")?;
            Ok(DeepSeekMlaDimensions {
                q_lora_rank: metadata.q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                semantic_head_size: checked_add(
                    kv_lora_rank,
                    qk_rope_head_dim,
                    "DeepSeek V3 MLA head size",
                )?,
            })
        }
        HfArchitectureKind::DeepSeekV4 => {
            let qk_rope_head_dim = required(metadata.qk_rope_head_dim, "qk_rope_head_dim")?;
            let qk_nope_head_dim =
                metadata
                    .head_dim
                    .checked_sub(qk_rope_head_dim)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: "DeepSeek V4 head_dim must be at least qk_rope_head_dim"
                            .to_string(),
                    })?;
            Ok(DeepSeekMlaDimensions {
                q_lora_rank: metadata.q_lora_rank,
                kv_lora_rank: metadata.head_dim,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim: metadata.head_dim,
                semantic_head_size: metadata.head_dim,
            })
        }
        _ => Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek MLA dimensions require a DeepSeek architecture, got {}",
                metadata.architecture.as_str()
            ),
        }),
    }
}

fn mla_spec(
    kind: &'static str,
    block_size: usize,
    head_size: usize,
    cache_dtype_str: &'static str,
    compress_ratio: usize,
    sliding_window: Option<usize>,
    alignment: Option<usize>,
    model_version: Option<&'static str>,
    architecture: HfArchitectureKind,
) -> Result<DeepSeekVllmKvCacheSpec> {
    let storage_block_size = storage_block_size(block_size, compress_ratio)?;
    let dtype = cache_dtype(cache_dtype_str);
    let real_page_size_bytes = if cache_dtype_str == "fp8_ds_mla" {
        match architecture {
            HfArchitectureKind::DeepSeekV4 => checked_mul(
                storage_block_size,
                VLLM_DEEPSEEK_V4_FP8_DS_MLA_BYTES_PER_TOKEN,
                "DeepSeek V4 fp8_ds_mla page bytes",
            )?,
            HfArchitectureKind::DeepSeekV32 => checked_mul(
                block_size,
                VLLM_DEEPSEEK_V32_FP8_DS_MLA_BYTES_PER_TOKEN,
                "DeepSeek V3.2 fp8_ds_mla page bytes",
            )?,
            _ => checked_mul3(
                storage_block_size,
                1,
                head_size,
                dtype_size_bytes(dtype),
                "DeepSeek MLA page bytes",
            )?,
        }
    } else {
        checked_mul3(
            storage_block_size,
            1,
            head_size,
            dtype_size_bytes(dtype),
            "DeepSeek MLA page bytes",
        )?
    };
    Ok(spec_from_parts(
        kind,
        block_size,
        storage_block_size,
        head_size,
        dtype,
        cache_dtype_str,
        compress_ratio,
        sliding_window,
        alignment,
        model_version,
        real_page_size_bytes,
    )?)
}

fn indexer_spec(
    kind: &'static str,
    block_size: usize,
    index_head_dim: usize,
    _cache_dtype_str: &'static str,
    compress_ratio: usize,
    alignment: Option<usize>,
    model_version: Option<&'static str>,
    architecture: HfArchitectureKind,
) -> Result<DeepSeekVllmKvCacheSpec> {
    let storage_block_size = storage_block_size(block_size, compress_ratio)?;
    let scale_bytes = checked_mul(
        index_head_dim / VLLM_DEEPSEEK_INDEXER_QUANT_BLOCK,
        4,
        "DeepSeek indexer scale bytes",
    )?;
    let head_size = checked_add(index_head_dim, scale_bytes, "DeepSeek indexer head size")?;
    let real_page_size_bytes =
        checked_mul(storage_block_size, head_size, "DeepSeek indexer page bytes")?;
    let cache_dtype_str = if architecture == HfArchitectureKind::DeepSeekV4 {
        "fp8_indexer"
    } else {
        "fp8_naive"
    };
    Ok(spec_from_parts(
        kind,
        block_size,
        storage_block_size,
        head_size,
        DType::U8,
        cache_dtype_str,
        compress_ratio,
        None,
        alignment,
        model_version,
        real_page_size_bytes,
    )?)
}

fn spec_from_parts(
    kind: &'static str,
    block_size: usize,
    storage_block_size: usize,
    head_size: usize,
    dtype: DType,
    cache_dtype_str: &str,
    compress_ratio: usize,
    sliding_window: Option<usize>,
    alignment: Option<usize>,
    model_version: Option<&'static str>,
    real_page_size_bytes: usize,
) -> Result<DeepSeekVllmKvCacheSpec> {
    let page_size_bytes = match alignment {
        Some(alignment) => round_up(real_page_size_bytes, alignment)?,
        None => real_page_size_bytes,
    };
    Ok(DeepSeekVllmKvCacheSpec {
        kind,
        block_size,
        storage_block_size,
        num_kv_heads: 1,
        head_size,
        dtype,
        cache_dtype_str: cache_dtype_str.to_string(),
        compress_ratio,
        sliding_window,
        alignment,
        model_version,
        real_page_size_bytes,
        page_size_bytes,
    })
}

fn normalize_cache_dtype(
    architecture: HfArchitectureKind,
    cache_dtype_str: &str,
) -> Result<&'static str> {
    match cache_dtype_str {
        "auto" | "bfloat16" | "bf16" => Ok("bfloat16"),
        "float16" | "fp16" => Ok("float16"),
        "fp8" | "fp8_e4m3" | "float8_e4m3" => Ok("fp8_e4m3"),
        "fp8_ds_mla" => {
            if matches!(
                architecture,
                HfArchitectureKind::DeepSeekV32 | HfArchitectureKind::DeepSeekV4
            ) {
                Ok("fp8_ds_mla")
            } else {
                Err(NervaError::InvalidArgument {
                    reason: "vLLM fp8_ds_mla KV cache is only valid for DeepSeek V3.2 sparse MLA or DeepSeek V4".to_string(),
                })
            }
        }
        other => Err(NervaError::InvalidArgument {
            reason: format!("unsupported DeepSeek vLLM KV cache dtype {other}"),
        }),
    }
}

fn cache_dtype(cache_dtype_str: &str) -> DType {
    match cache_dtype_str {
        "float16" => DType::F16,
        "fp8_e4m3" => DType::F8E4M3,
        "fp8_ds_mla" => DType::U8,
        _ => DType::BF16,
    }
}

fn v4_alignment(cache_dtype_str: &str) -> Option<usize> {
    (cache_dtype_str == "fp8_ds_mla").then_some(VLLM_DEEPSEEK_V4_ALIGNMENT)
}

fn v4_layer_compress_ratios(metadata: &HfModelMetadata) -> Result<Vec<usize>> {
    if metadata.compress_ratios.is_empty() {
        return Ok(vec![1; metadata.num_hidden_layers]);
    }
    if metadata.compress_ratios.len() != metadata.num_hidden_layers {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek V4 compress_ratios length {} does not match num_hidden_layers {}",
                metadata.compress_ratios.len(),
                metadata.num_hidden_layers
            ),
        });
    }
    Ok(metadata.compress_ratios.clone())
}

fn vllm_kv_reference_units(architecture: HfArchitectureKind) -> Vec<&'static str> {
    match architecture {
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32 => vec![
            "/root/vllm/vllm/model_executor/layers/attention/mla_attention.py",
            "/root/vllm/vllm/model_executor/models/deepseek_v2.py",
            "/root/vllm/vllm/v1/kv_cache_interface.py",
            "/root/vllm/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py",
        ],
        HfArchitectureKind::DeepSeekV4 => vec![
            "/root/vllm/vllm/models/deepseek_v4/attention.py",
            "/root/vllm/vllm/models/deepseek_v4/sparse_mla.py",
            "/root/vllm/vllm/v1/attention/backends/mla/sparse_swa.py",
            "/root/vllm/vllm/v1/kv_cache_interface.py",
        ],
        _ => Vec::new(),
    }
}

fn required(value: Option<usize>, name: &'static str) -> Result<usize> {
    value.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("DeepSeek metadata is missing {name}"),
    })
}

fn storage_block_size(block_size: usize, compress_ratio: usize) -> Result<usize> {
    if compress_ratio == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek compress_ratio must be non-zero for a cache spec".to_string(),
        });
    }
    if !block_size.is_multiple_of(compress_ratio) {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek vLLM cache block_size {block_size} must be divisible by compress_ratio {compress_ratio}"
            ),
        });
    }
    Ok(block_size / compress_ratio)
}

fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E8M0 => 1,
        DType::U16 | DType::F16 | DType::BF16 => 2,
        DType::U32 | DType::I32 | DType::F32 => 4,
        DType::I64 => 8,
    }
}

fn round_up(value: usize, alignment: usize) -> Result<usize> {
    if alignment == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek vLLM KV cache alignment must be non-zero".to_string(),
        });
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        checked_add(value, alignment - remainder, "DeepSeek aligned page bytes")
    }
}

fn checked_add(left: usize, right: usize, label: &'static str) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: left,
            reason: format!("{label} overflow"),
        })
}

fn checked_mul(left: usize, right: usize, label: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: left,
            reason: format!("{label} overflow"),
        })
}

fn checked_mul3(
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
    label: &'static str,
) -> Result<usize> {
    checked_mul(first, second, label)
        .and_then(|value| checked_mul(value, third, label))
        .and_then(|value| checked_mul(value, fourth, label))
}

fn json_groups(groups: &[DeepSeekVllmKvCacheGroup]) -> String {
    let mut out = String::from("[");
    for (index, group) in groups.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&group.to_json());
    }
    out.push(']');
    out
}

fn json_static_str_array(values: &[&'static str]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(value));
        out.push('"');
    }
    out.push(']');
    out
}
