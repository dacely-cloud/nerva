#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{
    BlockKind, DType, DeviceOrdinal, ExecutionOwner, MemoryTier, NervaError, Result, TokenId,
};
use nerva_ledger::{
    CandidateCost, ExecutionDecision, LedgerEvent, LedgerEventKind, MetricSource, TokenLedger,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransformerBlockShape {
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
}

impl TransformerBlockShape {
    pub const fn new(hidden: usize, heads: usize, intermediate: usize) -> Self {
        Self {
            hidden,
            heads,
            intermediate,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.hidden == 0 || self.heads == 0 || self.intermediate == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transformer block dimensions must be non-zero".to_string(),
            });
        }
        if !self.hidden.is_multiple_of(self.heads) {
            return Err(NervaError::InvalidArgument {
                reason: "hidden size must be divisible by head count".to_string(),
            });
        }
        Ok(())
    }

    pub const fn head_dim(self) -> usize {
        self.hidden / self.heads
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfArchitectureKind {
    Llama,
    Mistral,
    Gemma,
    Qwen2,
    Unknown,
}

impl HfArchitectureKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Gemma => "gemma",
            Self::Qwen2 => "qwen2",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfModelMetadata {
    pub architecture: HfArchitectureKind,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: Option<usize>,
    pub rope_theta: Option<f32>,
    pub rms_norm_eps: Option<f32>,
    pub torch_dtype: Option<DType>,
}

impl HfModelMetadata {
    pub fn block_shape(&self) -> TransformerBlockShape {
        TransformerBlockShape::new(
            self.hidden_size,
            self.num_attention_heads,
            self.intermediate_size,
        )
    }

    pub const fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub const fn kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"architecture\":\"{}\",\"hidden_size\":{},\"num_hidden_layers\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},\"head_dim\":{},\"kv_groups\":{},\"intermediate_size\":{},\"vocab_size\":{},\"max_position_embeddings\":{},\"rope_theta\":{},\"rms_norm_eps\":{},\"torch_dtype\":{}}}",
            self.architecture.as_str(),
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim(),
            self.kv_groups(),
            self.intermediate_size,
            self.vocab_size,
            json_opt_usize(self.max_position_embeddings),
            json_opt_f32(self.rope_theta),
            json_opt_f32(self.rms_norm_eps),
            json_opt_dtype(self.torch_dtype),
        )
    }
}

pub fn parse_hf_config_metadata(config_json: &str) -> Result<HfModelMetadata> {
    let architecture = architecture_from_config(config_json)?;
    let hidden_size = required_usize(config_json, "hidden_size")?;
    let num_hidden_layers = required_usize(config_json, "num_hidden_layers")?;
    let num_attention_heads = required_usize(config_json, "num_attention_heads")?;
    let num_key_value_heads =
        optional_usize(config_json, "num_key_value_heads")?.unwrap_or(num_attention_heads);
    let intermediate_size = required_usize(config_json, "intermediate_size")?;
    let vocab_size = required_usize(config_json, "vocab_size")?;
    let max_position_embeddings = optional_usize(config_json, "max_position_embeddings")?;
    let rope_theta = optional_f32(config_json, "rope_theta")?;
    let rms_norm_eps = match optional_f32(config_json, "rms_norm_eps")? {
        Some(value) => Some(value),
        None => optional_f32(config_json, "layer_norm_eps")?,
    };
    let torch_dtype = optional_string(config_json, "torch_dtype")?
        .as_deref()
        .map(dtype_from_hf_string)
        .transpose()?;

    validate_hf_metadata(
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
    )?;

    Ok(HfModelMetadata {
        architecture,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rope_theta,
        rms_norm_eps,
        torch_dtype,
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfMetadataProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfMetadataProbeSummary {
    pub status: HfMetadataProbeStatus,
    pub metadata: HfModelMetadata,
    pub metadata_hash: u64,
}

impl HfMetadataProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            HfMetadataProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"metadata\":{},\"metadata_hash\":{}}}",
            status,
            self.metadata.to_json(),
            self.metadata_hash,
        )
    }
}

pub fn hf_metadata_probe() -> Result<HfMetadataProbeSummary> {
    let config = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 32000,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 0.000001,
        "rope_theta": 10000.0,
        "torch_dtype": "bfloat16"
    }"#;
    let metadata = parse_hf_config_metadata(config)?;
    Ok(HfMetadataProbeSummary {
        metadata_hash: hash_metadata(&metadata),
        status: HfMetadataProbeStatus::Ok,
        metadata,
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WeightBlockRole {
    TokenEmbedding,
    AttentionNorm,
    QueryProjection,
    KeyProjection,
    ValueProjection,
    OutputProjection,
    MlpNorm,
    GateProjection,
    UpProjection,
    DownProjection,
    LmHead,
}

impl WeightBlockRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::AttentionNorm => "attention_norm",
            Self::QueryProjection => "q_proj",
            Self::KeyProjection => "k_proj",
            Self::ValueProjection => "v_proj",
            Self::OutputProjection => "o_proj",
            Self::MlpNorm => "mlp_norm",
            Self::GateProjection => "gate_proj",
            Self::UpProjection => "up_proj",
            Self::DownProjection => "down_proj",
            Self::LmHead => "lm_head",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct WeightBlockSpec {
    pub role: WeightBlockRole,
    pub layer: Option<u32>,
    pub rows: usize,
    pub cols: usize,
    pub elements: usize,
    pub bytes: usize,
    pub dtype: DType,
    pub tier: MemoryTier,
}

impl WeightBlockSpec {
    fn new(
        role: WeightBlockRole,
        layer: Option<u32>,
        rows: usize,
        cols: usize,
        dtype: DType,
        tier: MemoryTier,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!("weight block {} shape must be non-zero", role.as_str()),
            });
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: 0,
                reason: format!("weight block {} element count overflow", role.as_str()),
            })?;
        let bytes = elements
            .checked_mul(dtype_size_bytes(dtype)?)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: elements,
                reason: format!("weight block {} byte count overflow", role.as_str()),
            })?;
        Ok(Self {
            role,
            layer,
            rows,
            cols,
            elements,
            bytes,
            dtype,
            tier,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfWeightLayoutPlan {
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub blocks: Vec<WeightBlockSpec>,
    pub total_weight_bytes: usize,
    pub per_layer_weight_bytes: usize,
    pub static_weight_bytes: usize,
}

impl HfWeightLayoutPlan {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"architecture\":\"{}\",\"dtype\":\"{}\",\"blocks\":{},\"layers\":{},\"total_weight_bytes\":{},\"per_layer_weight_bytes\":{},\"static_weight_bytes\":{},\"hidden_size\":{},\"head_dim\":{},\"kv_hidden_size\":{}}}",
            self.metadata.architecture.as_str(),
            dtype_to_str(self.dtype),
            self.blocks.len(),
            self.metadata.num_hidden_layers,
            self.total_weight_bytes,
            self.per_layer_weight_bytes,
            self.static_weight_bytes,
            self.metadata.hidden_size,
            self.metadata.head_dim(),
            self.metadata.num_key_value_heads * self.metadata.head_dim(),
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfWeightLayoutProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfWeightLayoutProbeSummary {
    pub status: HfWeightLayoutProbeStatus,
    pub plan: HfWeightLayoutPlan,
    pub layout_hash: u64,
}

impl HfWeightLayoutProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            HfWeightLayoutProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"plan\":{},\"layout_hash\":{}}}",
            status,
            self.plan.to_json(),
            self.layout_hash,
        )
    }
}

pub fn plan_hf_weight_layout(metadata: &HfModelMetadata) -> Result<HfWeightLayoutPlan> {
    metadata.block_shape().validate()?;
    validate_hf_metadata(
        metadata.hidden_size,
        metadata.num_hidden_layers,
        metadata.num_attention_heads,
        metadata.num_key_value_heads,
        metadata.intermediate_size,
        metadata.vocab_size,
    )?;
    let dtype = metadata
        .torch_dtype
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "HF weight layout requires torch_dtype".to_string(),
        })?;
    let kv_hidden = metadata
        .num_key_value_heads
        .checked_mul(metadata.head_dim())
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: 0,
            reason: "KV hidden size overflow".to_string(),
        })?;

    let mut blocks = Vec::with_capacity(metadata.num_hidden_layers.saturating_mul(9) + 2);
    blocks.push(WeightBlockSpec::new(
        WeightBlockRole::TokenEmbedding,
        None,
        metadata.vocab_size,
        metadata.hidden_size,
        dtype,
        MemoryTier::Dram,
    )?);

    for layer in 0..metadata.num_hidden_layers {
        let layer = u32::try_from(layer).map_err(|_| NervaError::InvalidArgument {
            reason: "layer index does not fit u32".to_string(),
        })?;
        push_layer_weight_blocks(&mut blocks, metadata, kv_hidden, dtype, layer)?;
    }

    blocks.push(WeightBlockSpec::new(
        WeightBlockRole::LmHead,
        None,
        metadata.vocab_size,
        metadata.hidden_size,
        dtype,
        MemoryTier::Dram,
    )?);

    let total_weight_bytes = sum_weight_bytes(&blocks)?;
    let static_weight_bytes = blocks
        .iter()
        .filter(|block| block.layer.is_none())
        .map(|block| block.bytes)
        .try_fold(0usize, |acc, bytes| {
            acc.checked_add(bytes)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes,
                    reason: "static weight byte count overflow".to_string(),
                })
        })?;
    let per_layer_weight_bytes = total_weight_bytes
        .checked_sub(static_weight_bytes)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: total_weight_bytes,
            reason: "weight byte accounting underflow".to_string(),
        })?
        / metadata.num_hidden_layers;

    Ok(HfWeightLayoutPlan {
        metadata: metadata.clone(),
        dtype,
        blocks,
        total_weight_bytes,
        per_layer_weight_bytes,
        static_weight_bytes,
    })
}

pub fn hf_weight_layout_probe() -> Result<HfWeightLayoutProbeSummary> {
    let metadata = hf_metadata_probe()?.metadata;
    let plan = plan_hf_weight_layout(&metadata)?;
    Ok(HfWeightLayoutProbeSummary {
        layout_hash: hash_weight_layout(&plan),
        status: HfWeightLayoutProbeStatus::Ok,
        plan,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfTensorManifestEntry {
    pub name: String,
    pub role: WeightBlockRole,
    pub layer: Option<u32>,
    pub rows: usize,
    pub cols: usize,
    pub rank: u8,
    pub elements: usize,
    pub bytes: usize,
    pub dtype: DType,
    pub tier: MemoryTier,
}

impl HfTensorManifestEntry {
    fn from_block(block: WeightBlockSpec, name: String) -> Self {
        Self {
            name,
            role: block.role,
            layer: block.layer,
            rows: block.rows,
            cols: block.cols,
            rank: weight_block_rank(block.role),
            elements: block.elements,
            bytes: block.bytes,
            dtype: block.dtype,
            tier: block.tier,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfTensorManifest {
    pub architecture: HfArchitectureKind,
    pub entries: Vec<HfTensorManifestEntry>,
    pub total_weight_bytes: usize,
    pub manifest_hash: u64,
}

impl HfTensorManifest {
    pub fn to_json(&self) -> String {
        let first = self.entries.first().map(|entry| entry.name.as_str());
        let last = self.entries.last().map(|entry| entry.name.as_str());
        format!(
            "{{\"architecture\":\"{}\",\"entries\":{},\"total_weight_bytes\":{},\"first_tensor\":{},\"last_tensor\":{},\"manifest_hash\":{}}}",
            self.architecture.as_str(),
            self.entries.len(),
            self.total_weight_bytes,
            json_opt_str(first),
            json_opt_str(last),
            self.manifest_hash,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfTensorManifestProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfTensorManifestProbeSummary {
    pub status: HfTensorManifestProbeStatus,
    pub manifest: HfTensorManifest,
}

impl HfTensorManifestProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            HfTensorManifestProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"manifest\":{}}}",
            status,
            self.manifest.to_json(),
        )
    }
}

pub fn build_hf_tensor_manifest(plan: &HfWeightLayoutPlan) -> Result<HfTensorManifest> {
    ensure_supported_hf_tensor_names(plan.metadata.architecture)?;
    let mut entries = Vec::with_capacity(plan.blocks.len());
    for block in plan.blocks.iter().copied() {
        let name = hf_tensor_name(plan.metadata.architecture, block.role, block.layer)?;
        entries.push(HfTensorManifestEntry::from_block(block, name));
    }
    let total_weight_bytes = entries.iter().try_fold(0usize, |acc, entry| {
        acc.checked_add(entry.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: entry.bytes,
                reason: "HF tensor manifest byte count overflow".to_string(),
            })
    })?;
    if total_weight_bytes != plan.total_weight_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "HF tensor manifest byte count does not match layout plan".to_string(),
        });
    }

    let mut manifest = HfTensorManifest {
        architecture: plan.metadata.architecture,
        entries,
        total_weight_bytes,
        manifest_hash: 0,
    };
    manifest.manifest_hash = hash_tensor_manifest(&manifest);
    Ok(manifest)
}

pub fn hf_tensor_manifest_probe() -> Result<HfTensorManifestProbeSummary> {
    let plan = hf_weight_layout_probe()?.plan;
    let manifest = build_hf_tensor_manifest(&plan)?;
    Ok(HfTensorManifestProbeSummary {
        status: HfTensorManifestProbeStatus::Ok,
        manifest,
    })
}

#[derive(Clone, Debug)]
pub struct ReferenceTransformerBlock {
    shape: TransformerBlockShape,
    rms_attn_weight: Vec<f32>,
    rms_mlp_weight: Vec<f32>,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
    w_o: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
    rms_eps: f32,
}

impl ReferenceTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shape: TransformerBlockShape,
        rms_attn_weight: Vec<f32>,
        rms_mlp_weight: Vec<f32>,
        w_q: Vec<f32>,
        w_k: Vec<f32>,
        w_v: Vec<f32>,
        w_o: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
        rms_eps: f32,
    ) -> Result<Self> {
        shape.validate()?;
        require_len("rms_attn_weight", rms_attn_weight.len(), shape.hidden)?;
        require_len("rms_mlp_weight", rms_mlp_weight.len(), shape.hidden)?;
        require_len("w_q", w_q.len(), shape.hidden * shape.hidden)?;
        require_len("w_k", w_k.len(), shape.hidden * shape.hidden)?;
        require_len("w_v", w_v.len(), shape.hidden * shape.hidden)?;
        require_len("w_o", w_o.len(), shape.hidden * shape.hidden)?;
        require_len("w_gate", w_gate.len(), shape.intermediate * shape.hidden)?;
        require_len("w_up", w_up.len(), shape.intermediate * shape.hidden)?;
        require_len("w_down", w_down.len(), shape.hidden * shape.intermediate)?;
        if rms_eps <= 0.0 || !rms_eps.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "rms epsilon must be positive and finite".to_string(),
            });
        }
        Ok(Self {
            shape,
            rms_attn_weight,
            rms_mlp_weight,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            rms_eps,
        })
    }

    pub fn zero_for_shape(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Self::new(
            shape,
            vec![1.0; shape.hidden],
            vec![1.0; shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.hidden * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.intermediate * shape.hidden],
            vec![0.0; shape.hidden * shape.intermediate],
            1e-5,
        )
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub fn forward_into(
        &self,
        input: &[f32],
        scratch: &mut TransformerBlockScratch,
        output: &mut [f32],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let _ = ledger;
        let shape = self.shape;
        require_len("input", input.len(), shape.hidden)?;
        require_len("output", output.len(), shape.hidden)?;
        scratch.require_shape(shape)?;

        rms_norm_into(
            input,
            &self.rms_attn_weight,
            self.rms_eps,
            &mut scratch.attn_norm,
        );
        mat_vec_row_major(&self.w_q, &scratch.attn_norm, &mut scratch.q);
        mat_vec_row_major(&self.w_k, &scratch.attn_norm, &mut scratch.k);
        mat_vec_row_major(&self.w_v, &scratch.attn_norm, &mut scratch.v);

        single_token_attention(shape, &scratch.q, &scratch.k, &scratch.v, &mut scratch.attn);
        mat_vec_row_major(&self.w_o, &scratch.attn, output);
        for (out, residual) in output.iter_mut().zip(input.iter().copied()) {
            *out += residual;
        }

        rms_norm_into(
            output,
            &self.rms_mlp_weight,
            self.rms_eps,
            &mut scratch.mlp_norm,
        );
        mat_vec_row_major(&self.w_gate, &scratch.mlp_norm, &mut scratch.gate);
        mat_vec_row_major(&self.w_up, &scratch.mlp_norm, &mut scratch.up);
        for ((ff, gate), up) in scratch
            .ff
            .iter_mut()
            .zip(scratch.gate.iter().copied())
            .zip(scratch.up.iter().copied())
        {
            *ff = silu(gate) * up;
        }
        mat_vec_row_major(&self.w_down, &scratch.ff, &mut scratch.down);
        for (out, mlp) in output.iter_mut().zip(scratch.down.iter().copied()) {
            *out += mlp;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TransformerBlockScratch {
    shape: TransformerBlockShape,
    attn_norm: Vec<f32>,
    mlp_norm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ff: Vec<f32>,
    down: Vec<f32>,
}

impl TransformerBlockScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            attn_norm: vec![0.0; shape.hidden],
            mlp_norm: vec![0.0; shape.hidden],
            q: vec![0.0; shape.hidden],
            k: vec![0.0; shape.hidden],
            v: vec![0.0; shape.hidden],
            attn: vec![0.0; shape.hidden],
            gate: vec![0.0; shape.intermediate],
            up: vec![0.0; shape.intermediate],
            ff: vec![0.0; shape.intermediate],
            down: vec![0.0; shape.hidden],
        })
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    fn require_shape(&self, shape: TransformerBlockShape) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "transformer block scratch shape does not match block shape".to_string(),
            })
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KvAttentionBlock<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub token_count: usize,
    pub tier: MemoryTier,
}

impl<'a> KvAttentionBlock<'a> {
    pub const fn new(
        keys: &'a [f32],
        values: &'a [f32],
        token_count: usize,
        tier: MemoryTier,
    ) -> Self {
        Self {
            keys,
            values,
            token_count,
            tier,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlockwiseAttentionScratch {
    shape: TransformerBlockShape,
    local_output: Vec<f32>,
    global_m: Vec<f32>,
    global_l: Vec<f32>,
}

impl BlockwiseAttentionScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            local_output: vec![0.0; shape.hidden],
            global_m: vec![f32::NEG_INFINITY; shape.heads],
            global_l: vec![0.0; shape.heads],
        })
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    fn require_shape(&self, shape: TransformerBlockShape) -> Result<()> {
        if self.shape == shape {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "blockwise attention scratch shape does not match".to_string(),
            })
        }
    }
}

pub fn exact_blockwise_attention_into(
    shape: TransformerBlockShape,
    query: &[f32],
    blocks: &[KvAttentionBlock<'_>],
    scratch: &mut BlockwiseAttentionScratch,
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<()> {
    shape.validate()?;
    scratch.require_shape(shape)?;
    require_len("attention query", query.len(), shape.hidden)?;
    require_len("attention output", output.len(), shape.hidden)?;

    scratch.local_output.fill(0.0);
    scratch.global_m.fill(f32::NEG_INFINITY);
    scratch.global_l.fill(0.0);
    output.fill(0.0);

    let head_dim = shape.head_dim();
    let scale = (head_dim as f32).sqrt().recip();
    let mut total_tokens = 0usize;

    for block in blocks {
        let values = block.token_count.checked_mul(shape.hidden).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "KV attention block token count overflow".to_string(),
            }
        })?;
        require_len("KV block keys", block.keys.len(), values)?;
        require_len("KV block values", block.values.len(), values)?;
        if block.token_count == 0 {
            continue;
        }
        total_tokens = total_tokens.checked_add(block.token_count).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "KV attention total token count overflow".to_string(),
            }
        })?;
        record_attention_block_event(shape, block, ledger);

        for head in 0..shape.heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            scratch.local_output[head_start..head_end].fill(0.0);
            let mut local_m = f32::NEG_INFINITY;
            let mut local_l = 0.0f32;

            for token_index in 0..block.token_count {
                let token_start = token_index * shape.hidden + head_start;
                let token_end = token_start + head_dim;
                let score = dot(
                    &query[head_start..head_end],
                    &block.keys[token_start..token_end],
                ) * scale;
                let next_m = local_m.max(score);
                let old_scale = if local_l == 0.0 {
                    0.0
                } else {
                    (local_m - next_m).exp()
                };
                let new_scale = (score - next_m).exp();
                for (local, value) in scratch.local_output[head_start..head_end]
                    .iter_mut()
                    .zip(block.values[token_start..token_end].iter().copied())
                {
                    *local = *local * old_scale + value * new_scale;
                }
                local_l = local_l * old_scale + new_scale;
                local_m = next_m;
            }

            let global_m = scratch.global_m[head];
            let global_l = scratch.global_l[head];
            let next_m = global_m.max(local_m);
            let global_scale = if global_l == 0.0 {
                0.0
            } else {
                (global_m - next_m).exp()
            };
            let local_scale = (local_m - next_m).exp();
            for (global, local) in output[head_start..head_end]
                .iter_mut()
                .zip(scratch.local_output[head_start..head_end].iter().copied())
            {
                *global = *global * global_scale + local * local_scale;
            }
            scratch.global_l[head] = global_l * global_scale + local_l * local_scale;
            scratch.global_m[head] = next_m;
        }
    }

    if total_tokens == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "blockwise attention requires at least one KV token".to_string(),
        });
    }

    for head in 0..shape.heads {
        let normalizer = scratch.global_l[head];
        if normalizer == 0.0 || !normalizer.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "blockwise attention produced invalid normalizer".to_string(),
            });
        }
        let head_start = head * head_dim;
        let head_end = head_start + head_dim;
        for value in &mut output[head_start..head_end] {
            *value /= normalizer;
        }
    }

    Ok(())
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ModelBlockContract {
    pub block_kind: BlockKind,
    pub weight_dtype: DType,
    pub activation_dtype: DType,
    pub weight_tier: MemoryTier,
    pub activation_tier: MemoryTier,
}

impl ModelBlockContract {
    pub const fn reference_f32() -> Self {
        Self {
            block_kind: BlockKind::Weight,
            weight_dtype: DType::F32,
            activation_dtype: DType::F32,
            weight_tier: MemoryTier::Dram,
            activation_tier: MemoryTier::Dram,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReferenceBlockSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReferenceBlockSmokeSummary {
    pub status: ReferenceBlockSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub hot_path_allocations: u64,
}

impl ReferenceBlockSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            ReferenceBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"output\":[{},{}],\"output_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.hot_path_allocations,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockwiseAttentionSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockwiseAttentionSmokeSummary {
    pub status: BlockwiseAttentionSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub blocks: usize,
    pub tokens: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub cpu_block_events: u64,
    pub device_block_events: u64,
    pub hot_path_allocations: u64,
}

impl BlockwiseAttentionSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            BlockwiseAttentionSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"blocks\":{},\"tokens\":{},\"output\":[{},{}],\"output_hash\":{},\"cpu_block_events\":{},\"device_block_events\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.blocks,
            self.tokens,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.cpu_block_events,
            self.device_block_events,
            self.hot_path_allocations,
        )
    }
}

pub fn reference_block_smoke() -> Result<ReferenceBlockSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::new(
        shape,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        1e-5,
    )?;
    let input = [1.0, 2.0];
    let mut scratch = TransformerBlockScratch::new(shape)?;
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    block.forward_into(&input, &mut scratch, &mut output, &mut ledger)?;
    Ok(ReferenceBlockSmokeSummary {
        status: ReferenceBlockSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        intermediate: shape.intermediate,
        output,
        output_hash: hash_f32s(&output),
        hot_path_allocations: ledger.hot_path_allocations,
    })
}

pub fn blockwise_attention_smoke() -> Result<BlockwiseAttentionSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let query = [1.0, 0.25];
    let dram_keys = [0.2, 0.0, 0.0, 0.4];
    let dram_values = [1.0, 0.0, 0.5, 0.5];
    let vram_keys = [0.5, 0.1, -0.2, 0.3];
    let vram_values = [0.0, 1.0, 2.0, -1.0];
    let blocks = [
        KvAttentionBlock::new(&dram_keys, &dram_values, 2, MemoryTier::Dram),
        KvAttentionBlock::new(&vram_keys, &vram_values, 2, MemoryTier::Vram),
    ];
    let mut scratch = BlockwiseAttentionScratch::new(shape)?;
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    exact_blockwise_attention_into(
        shape,
        &query,
        &blocks,
        &mut scratch,
        &mut output,
        &mut ledger,
    )?;

    Ok(BlockwiseAttentionSmokeSummary {
        status: BlockwiseAttentionSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        blocks: blocks.len(),
        tokens: blocks.iter().map(|block| block.token_count).sum(),
        output,
        output_hash: hash_f32s(&output),
        cpu_block_events: ledger.event_count(LedgerEventKind::CpuActivity),
        device_block_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        hot_path_allocations: ledger.hot_path_allocations,
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WarmComputeStrategy {
    CpuDram,
    GpuResident,
    GpuStaged,
    HybridSplit,
}

impl WarmComputeStrategy {
    pub const fn label(self) -> &'static str {
        match self {
            Self::CpuDram => "cpu-dram",
            Self::GpuResident => "gpu-resident",
            Self::GpuStaged => "gpu-staged",
            Self::HybridSplit => "hybrid-split",
        }
    }

    pub const fn executor(self) -> ExecutionOwner {
        match self {
            Self::CpuDram => ExecutionOwner::Cpu,
            Self::GpuResident | Self::GpuStaged | Self::HybridSplit => {
                ExecutionOwner::Gpu(DeviceOrdinal(0))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WarmComputeCandidate {
    pub strategy: WarmComputeStrategy,
    pub visible_ns: u64,
    pub output_hash: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WarmComputeProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WarmComputeProbeSummary {
    pub status: WarmComputeProbeStatus,
    pub rows: usize,
    pub cols: usize,
    pub candidates: Vec<WarmComputeCandidate>,
    pub selected_strategy: WarmComputeStrategy,
    pub parity: bool,
    pub cpu_beats_staged: bool,
    pub execution_decisions: u64,
    pub cpu_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub copy_bytes: usize,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl WarmComputeProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            WarmComputeProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"rows\":{},\"cols\":{},\"selected_strategy\":\"{}\",\"parity\":{},\"cpu_beats_staged\":{},\"candidate_count\":{},\"execution_decisions\":{},\"cpu_events\":{},\"device_events\":{},\"copy_events\":{},\"copy_bytes\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            self.rows,
            self.cols,
            self.selected_strategy.label(),
            self.parity,
            self.cpu_beats_staged,
            self.candidates.len(),
            self.execution_decisions,
            self.cpu_events,
            self.device_events,
            self.copy_events,
            self.copy_bytes,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}

pub fn warm_compute_probe() -> Result<WarmComputeProbeSummary> {
    const ROWS: usize = 4;
    const COLS: usize = 4;
    let matrix = [
        1.0, 0.0, 0.0, 1.0, 0.5, -1.0, 2.0, 0.0, -1.0, 0.0, 1.0, 0.5, 0.0, 2.0, 0.25, -0.5,
    ];
    let input = [1.0, -2.0, 0.5, 3.0];
    let mut ledger = TokenLedger::new(0);
    let mut candidates = Vec::new();

    for strategy in [
        WarmComputeStrategy::CpuDram,
        WarmComputeStrategy::GpuResident,
        WarmComputeStrategy::GpuStaged,
        WarmComputeStrategy::HybridSplit,
    ] {
        candidates.push(run_warm_compute_candidate(
            strategy,
            ROWS,
            COLS,
            &matrix,
            &input,
            &mut ledger,
        )?);
    }

    let output_hash = candidates
        .first()
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "warm compute probe produced no candidates".to_string(),
        })?
        .output_hash;
    let parity = candidates
        .iter()
        .all(|candidate| candidate.output_hash == output_hash);
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "warm compute candidate parity failed".to_string(),
        });
    }

    let selected = candidates
        .iter()
        .min_by_key(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "warm compute candidate selection failed".to_string(),
        })?;
    let selected_strategy = selected.strategy;
    let selected_visible_ns = selected.visible_ns;
    let cpu_visible = candidate_visible_ns(&candidates, WarmComputeStrategy::CpuDram)?;
    let staged_visible = candidate_visible_ns(&candidates, WarmComputeStrategy::GpuStaged)?;

    ledger.record_execution_decision(ExecutionDecision {
        operation: "dense_matvec",
        executor_selected: selected_strategy.executor(),
        candidate_costs: candidates
            .iter()
            .map(|candidate| {
                CandidateCost::estimated(candidate.strategy.label(), candidate.visible_ns)
            })
            .collect(),
        reason: "select exact candidate with lowest visible critical-path cost",
        predicted_visible_ns: selected_visible_ns,
        actual_visible_ns: Some(selected_visible_ns),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.require_zero_hot_path_allocations()?;

    let copy_bytes = ledger
        .events
        .iter()
        .filter(|event| event.kind == LedgerEventKind::Copy)
        .map(|event| event.bytes)
        .sum();

    Ok(WarmComputeProbeSummary {
        status: WarmComputeProbeStatus::Ok,
        rows: ROWS,
        cols: COLS,
        candidates,
        selected_strategy,
        parity,
        cpu_beats_staged: cpu_visible < staged_visible,
        execution_decisions: ledger.execution_decisions.len() as u64,
        cpu_events: ledger.event_count(LedgerEventKind::CpuActivity),
        device_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        copy_events: ledger.event_count(LedgerEventKind::Copy),
        copy_bytes,
        total_latency_ns: ledger.total_latency_ns(),
        hot_path_allocations: ledger.hot_path_allocations,
        output_hash,
    })
}

#[derive(Clone, Debug)]
pub struct TinyGreedyModel {
    vocab_size: usize,
    shape: TransformerBlockShape,
    block: ReferenceTransformerBlock,
    embeddings: Vec<f32>,
    lm_head: Vec<f32>,
}

impl TinyGreedyModel {
    pub fn new(
        vocab_size: usize,
        block: ReferenceTransformerBlock,
        embeddings: Vec<f32>,
        lm_head: Vec<f32>,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny model vocabulary must be non-zero".to_string(),
            });
        }
        let shape = block.shape();
        require_len("embeddings", embeddings.len(), vocab_size * shape.hidden)?;
        require_len("lm_head", lm_head.len(), vocab_size * shape.hidden)?;
        Ok(Self {
            vocab_size,
            shape,
            block,
            embeddings,
            lm_head,
        })
    }

    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub const fn shape(&self) -> TransformerBlockShape {
        self.shape
    }

    pub fn decode_greedy(
        &self,
        seed_token: TokenId,
        steps: usize,
        scratch: &mut TinyGreedyDecodeScratch,
    ) -> Result<TinyGreedyDecodeOutput> {
        if steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny greedy decode steps must be non-zero".to_string(),
            });
        }
        scratch.require_shape(self.shape, self.vocab_size)?;
        require_token_in_vocab(seed_token, self.vocab_size)?;

        let mut current_token = seed_token;
        let mut tokens = Vec::with_capacity(steps);
        let mut ledgers = Vec::with_capacity(steps);
        for step in 0..steps {
            copy_embedding_row(
                &self.embeddings,
                self.shape.hidden,
                current_token,
                &mut scratch.hidden,
            )?;
            let mut ledger = TokenLedger::new(step as u64);
            self.block.forward_into(
                &scratch.hidden,
                &mut scratch.block_scratch,
                &mut scratch.block_output,
                &mut ledger,
            )?;
            mat_vec_row_major(&self.lm_head, &scratch.block_output, &mut scratch.logits);
            let next_token = greedy_argmax(&scratch.logits)?;
            ledger.record_execution_decision(ExecutionDecision {
                operation: "tiny_greedy_decode",
                executor_selected: ExecutionOwner::Cpu,
                candidate_costs: vec![
                    CandidateCost::estimated("cpu-resident-reference", 1),
                    CandidateCost::estimated("gpu-staged-reference", 3),
                ],
                reason: "tiny reference model is already resident in DRAM",
                predicted_visible_ns: 1,
                actual_visible_ns: Some(1),
                metric_source: MetricSource::EstimatedModel,
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Dram),
                bytes: (self.shape.hidden + self.vocab_size) * core::mem::size_of::<f32>(),
                latency_ns: 1,
                label: "tiny_greedy_decode_reference",
            });
            ledger.require_zero_hot_path_allocations()?;
            tokens.push(next_token);
            ledgers.push(ledger);
            current_token = next_token;
        }

        Ok(TinyGreedyDecodeOutput { tokens, ledgers })
    }
}

#[derive(Clone, Debug)]
pub struct TinyGreedyDecodeScratch {
    shape: TransformerBlockShape,
    vocab_size: usize,
    block_scratch: TransformerBlockScratch,
    hidden: Vec<f32>,
    block_output: Vec<f32>,
    logits: Vec<f32>,
}

impl TinyGreedyDecodeScratch {
    pub fn new(shape: TransformerBlockShape, vocab_size: usize) -> Result<Self> {
        shape.validate()?;
        if vocab_size == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "tiny greedy scratch vocabulary must be non-zero".to_string(),
            });
        }
        Ok(Self {
            shape,
            vocab_size,
            block_scratch: TransformerBlockScratch::new(shape)?,
            hidden: vec![0.0; shape.hidden],
            block_output: vec![0.0; shape.hidden],
            logits: vec![0.0; vocab_size],
        })
    }

    fn require_shape(&self, shape: TransformerBlockShape, vocab_size: usize) -> Result<()> {
        if self.shape == shape && self.vocab_size == vocab_size {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: "tiny greedy scratch shape does not match model shape".to_string(),
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyGreedyDecodeOutput {
    pub tokens: Vec<TokenId>,
    pub ledgers: Vec<TokenLedger>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TinyGreedyDecodeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TinyGreedyDecodeSummary {
    pub status: TinyGreedyDecodeStatus,
    pub seed_token: TokenId,
    pub steps: usize,
    pub vocab_size: usize,
    pub tokens: Vec<TokenId>,
    pub expected_tokens: Vec<TokenId>,
    pub parity: bool,
    pub ledger_count: u64,
    pub device_events: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl TinyGreedyDecodeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            TinyGreedyDecodeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"seed_token\":{},\"steps\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            self.seed_token.0,
            self.steps,
            self.vocab_size,
            token_ids_to_json(&self.tokens),
            token_ids_to_json(&self.expected_tokens),
            self.parity,
            self.ledger_count,
            self.device_events,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}

pub fn tiny_greedy_decode_smoke(steps: usize) -> Result<TinyGreedyDecodeSummary> {
    let model = tiny_cycle_model()?;
    let seed_token = TokenId(0);
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size())?;
    let output = model.decode_greedy(seed_token, steps, &mut scratch)?;
    let expected_tokens = expected_cycle(seed_token, steps, model.vocab_size());
    let parity = output.tokens == expected_tokens;
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "tiny greedy decode token parity failed".to_string(),
        });
    }
    let hot_path_allocations = output
        .ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();
    let device_events = output
        .ledgers
        .iter()
        .map(|ledger| ledger.event_count(LedgerEventKind::DeviceActivity))
        .sum();
    let total_latency_ns = output
        .ledgers
        .iter()
        .map(TokenLedger::total_latency_ns)
        .sum();

    Ok(TinyGreedyDecodeSummary {
        status: TinyGreedyDecodeStatus::Ok,
        seed_token,
        steps,
        vocab_size: model.vocab_size(),
        output_hash: hash_tokens(&output.tokens),
        tokens: output.tokens,
        expected_tokens,
        parity,
        ledger_count: output.ledgers.len() as u64,
        device_events,
        total_latency_ns,
        hot_path_allocations,
    })
}

fn tiny_cycle_model() -> Result<TinyGreedyModel> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::zero_for_shape(shape)?;
    TinyGreedyModel::new(
        4,
        block,
        vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        vec![0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
    )
}

fn architecture_from_config(config_json: &str) -> Result<HfArchitectureKind> {
    if let Some(architecture) = optional_first_string(config_json, "architectures")? {
        return Ok(architecture_kind_from_str(&architecture));
    }
    if let Some(model_type) = optional_string(config_json, "model_type")? {
        return Ok(architecture_kind_from_str(&model_type));
    }
    Ok(HfArchitectureKind::Unknown)
}

fn architecture_kind_from_str(value: &str) -> HfArchitectureKind {
    let lower = value.to_ascii_lowercase();
    if lower.contains("llama") {
        HfArchitectureKind::Llama
    } else if lower.contains("mistral") {
        HfArchitectureKind::Mistral
    } else if lower.contains("gemma") {
        HfArchitectureKind::Gemma
    } else if lower.contains("qwen2") {
        HfArchitectureKind::Qwen2
    } else {
        HfArchitectureKind::Unknown
    }
}

fn validate_hf_metadata(
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
) -> Result<()> {
    if hidden_size == 0
        || num_hidden_layers == 0
        || num_attention_heads == 0
        || num_key_value_heads == 0
        || intermediate_size == 0
        || vocab_size == 0
    {
        return Err(NervaError::InvalidArgument {
            reason: "HF model metadata dimensions must be non-zero".to_string(),
        });
    }
    if !hidden_size.is_multiple_of(num_attention_heads) {
        return Err(NervaError::InvalidArgument {
            reason: "HF hidden size must be divisible by attention head count".to_string(),
        });
    }
    if num_key_value_heads > num_attention_heads {
        return Err(NervaError::InvalidArgument {
            reason: "HF KV head count cannot exceed attention head count".to_string(),
        });
    }
    if !num_attention_heads.is_multiple_of(num_key_value_heads) {
        return Err(NervaError::InvalidArgument {
            reason: "HF attention head count must be divisible by KV head count".to_string(),
        });
    }
    Ok(())
}

fn required_usize(config_json: &str, key: &'static str) -> Result<usize> {
    optional_usize(config_json, key)?.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("HF config is missing required field {key}"),
    })
}

fn optional_usize(config_json: &str, key: &'static str) -> Result<Option<usize>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    if value.starts_with('-') {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be unsigned"),
        });
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be an integer"),
        })?;
    usize::try_from(parsed)
        .map(Some)
        .map_err(|_| NervaError::InvalidArgument {
            reason: format!("HF config field {key} does not fit usize"),
        })
}

fn optional_f32(config_json: &str, key: &'static str) -> Result<Option<f32>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    let parsed = value
        .parse::<f32>()
        .map_err(|_| NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be a number"),
        })?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be positive and finite"),
        });
    }
    Ok(Some(parsed))
}

fn optional_string(config_json: &str, key: &'static str) -> Result<Option<String>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    parse_json_string_value(value).map(Some)
}

fn optional_first_string(config_json: &str, key: &'static str) -> Result<Option<String>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    let value = value.trim();
    if value.starts_with('"') {
        return parse_json_string_value(value).map(Some);
    }
    if !value.starts_with('[') {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be a string array"),
        });
    }
    let mut index = skip_json_ws(value.as_bytes(), 1);
    if index < value.len() && value.as_bytes()[index] == b']' {
        return Ok(None);
    }
    if index >= value.len() || value.as_bytes()[index] != b'"' {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must contain string values"),
        });
    }
    let (parsed, after) = parse_json_string_at(value, index)?;
    index = skip_json_ws(value.as_bytes(), after);
    if index >= value.len() || !matches!(value.as_bytes()[index], b',' | b']') {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} has malformed string array"),
        });
    }
    Ok(Some(parsed))
}

fn dtype_from_hf_string(value: &str) -> Result<DType> {
    match value.to_ascii_lowercase().as_str() {
        "float16" | "fp16" | "f16" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        "float32" | "fp32" | "f32" => Ok(DType::F32),
        other => Err(NervaError::InvalidArgument {
            reason: format!("unsupported HF torch_dtype {other}"),
        }),
    }
}

fn dtype_size_bytes(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F16 | DType::BF16 => Ok(2),
        DType::F32 => Ok(4),
        _ => Err(NervaError::InvalidArgument {
            reason: format!(
                "dtype {} is not a supported exact weight dtype",
                dtype_to_str(dtype)
            ),
        }),
    }
}

fn push_layer_weight_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    kv_hidden: usize,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    let hidden = metadata.hidden_size;
    let intermediate = metadata.intermediate_size;
    for (role, rows, cols) in [
        (WeightBlockRole::AttentionNorm, hidden, 1),
        (WeightBlockRole::QueryProjection, hidden, hidden),
        (WeightBlockRole::KeyProjection, kv_hidden, hidden),
        (WeightBlockRole::ValueProjection, kv_hidden, hidden),
        (WeightBlockRole::OutputProjection, hidden, hidden),
        (WeightBlockRole::MlpNorm, hidden, 1),
        (WeightBlockRole::GateProjection, intermediate, hidden),
        (WeightBlockRole::UpProjection, intermediate, hidden),
        (WeightBlockRole::DownProjection, hidden, intermediate),
    ] {
        blocks.push(WeightBlockSpec::new(
            role,
            Some(layer),
            rows,
            cols,
            dtype,
            MemoryTier::Dram,
        )?);
    }
    Ok(())
}

fn sum_weight_bytes(blocks: &[WeightBlockSpec]) -> Result<usize> {
    blocks.iter().try_fold(0usize, |acc, block| {
        acc.checked_add(block.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: block.bytes,
                reason: "total weight byte count overflow".to_string(),
            })
    })
}

fn ensure_supported_hf_tensor_names(architecture: HfArchitectureKind) -> Result<()> {
    match architecture {
        HfArchitectureKind::Llama | HfArchitectureKind::Mistral | HfArchitectureKind::Qwen2 => {
            Ok(())
        }
        HfArchitectureKind::Gemma | HfArchitectureKind::Unknown => {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "HF tensor names for architecture {} are not implemented",
                    architecture.as_str()
                ),
            })
        }
    }
}

fn hf_tensor_name(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
) -> Result<String> {
    ensure_supported_hf_tensor_names(architecture)?;
    match role {
        WeightBlockRole::TokenEmbedding => {
            require_static_tensor(role, layer).map(|()| "model.embed_tokens.weight".to_string())
        }
        WeightBlockRole::LmHead => {
            require_static_tensor(role, layer).map(|()| "lm_head.weight".to_string())
        }
        WeightBlockRole::AttentionNorm => layer_name(role, layer, "input_layernorm.weight"),
        WeightBlockRole::MlpNorm => layer_name(role, layer, "post_attention_layernorm.weight"),
        WeightBlockRole::QueryProjection => layer_name(role, layer, "self_attn.q_proj.weight"),
        WeightBlockRole::KeyProjection => layer_name(role, layer, "self_attn.k_proj.weight"),
        WeightBlockRole::ValueProjection => layer_name(role, layer, "self_attn.v_proj.weight"),
        WeightBlockRole::OutputProjection => layer_name(role, layer, "self_attn.o_proj.weight"),
        WeightBlockRole::GateProjection => layer_name(role, layer, "mlp.gate_proj.weight"),
        WeightBlockRole::UpProjection => layer_name(role, layer, "mlp.up_proj.weight"),
        WeightBlockRole::DownProjection => layer_name(role, layer, "mlp.down_proj.weight"),
    }
}

fn require_static_tensor(role: WeightBlockRole, layer: Option<u32>) -> Result<()> {
    if layer.is_none() {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("weight block {} must not have a layer", role.as_str()),
        })
    }
}

fn layer_name(role: WeightBlockRole, layer: Option<u32>, suffix: &'static str) -> Result<String> {
    let layer = layer.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("weight block {} must have a layer", role.as_str()),
    })?;
    Ok(format!("model.layers.{layer}.{suffix}"))
}

fn weight_block_rank(role: WeightBlockRole) -> u8 {
    match role {
        WeightBlockRole::AttentionNorm | WeightBlockRole::MlpNorm => 1,
        WeightBlockRole::TokenEmbedding
        | WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection
        | WeightBlockRole::LmHead => 2,
    }
}

fn find_top_level_json_value<'a>(source: &'a str, key: &str) -> Result<Option<&'a str>> {
    let bytes = source.as_bytes();
    let mut index = skip_json_ws(bytes, 0);
    if index >= bytes.len() || bytes[index] != b'{' {
        return Err(NervaError::InvalidArgument {
            reason: "HF config must be a JSON object".to_string(),
        });
    }
    index += 1;

    loop {
        index = skip_json_ws(bytes, index);
        if index >= bytes.len() {
            return Err(NervaError::InvalidArgument {
                reason: "HF config object is not closed".to_string(),
            });
        }
        if bytes[index] == b'}' {
            return Ok(None);
        }
        if bytes[index] == b',' {
            index += 1;
            continue;
        }
        if bytes[index] != b'"' {
            return Err(NervaError::InvalidArgument {
                reason: "HF config object key must be a JSON string".to_string(),
            });
        }

        let (field, after_key) = parse_json_string_at(source, index)?;
        index = skip_json_ws(bytes, after_key);
        if index >= bytes.len() || bytes[index] != b':' {
            return Err(NervaError::InvalidArgument {
                reason: "HF config object key is missing ':'".to_string(),
            });
        }
        index = skip_json_ws(bytes, index + 1);
        let value_start = index;
        let value_end = find_json_value_end(source, value_start)?;
        if field == key {
            return Ok(Some(source[value_start..value_end].trim()));
        }
        index = value_end;
    }
}

fn find_json_value_end(source: &str, start: usize) -> Result<usize> {
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in source[start..].char_indices() {
        let index = start + offset;
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' | '{' => depth = depth.saturating_add(1),
            ']' => {
                if depth == 0 {
                    return Err(NervaError::InvalidArgument {
                        reason: "HF config has unmatched ']'".to_string(),
                    });
                }
                depth -= 1;
            }
            '}' => {
                if depth == 0 {
                    return Ok(index);
                }
                depth -= 1;
            }
            ',' if depth == 0 => return Ok(index),
            _ => {}
        }
    }
    if depth == 0 && !in_string {
        Ok(source.len())
    } else {
        Err(NervaError::InvalidArgument {
            reason: "HF config value is not closed".to_string(),
        })
    }
}

fn parse_json_string_value(value: &str) -> Result<String> {
    let value = value.trim();
    if !value.starts_with('"') {
        return Err(NervaError::InvalidArgument {
            reason: "HF config field must be a JSON string".to_string(),
        });
    }
    let (parsed, after) = parse_json_string_at(value, 0)?;
    if !value[after..].trim().is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "HF config string field has trailing data".to_string(),
        });
    }
    Ok(parsed)
}

fn parse_json_string_at(source: &str, start: usize) -> Result<(String, usize)> {
    if source.as_bytes().get(start) != Some(&b'"') {
        return Err(NervaError::InvalidArgument {
            reason: "expected JSON string".to_string(),
        });
    }
    let mut out = String::new();
    let mut chars = source[start + 1..].char_indices();
    while let Some((offset, ch)) = chars.next() {
        let index = start + 1 + offset;
        match ch {
            '"' => return Ok((out, index + 1)),
            '\\' => {
                let Some((_, escaped)) = chars.next() else {
                    return Err(NervaError::InvalidArgument {
                        reason: "JSON string escape is incomplete".to_string(),
                    });
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
                                return Err(NervaError::InvalidArgument {
                                    reason: "JSON unicode escape is incomplete".to_string(),
                                });
                            };
                            let Some(value) = hex.to_digit(16) else {
                                return Err(NervaError::InvalidArgument {
                                    reason: "JSON unicode escape has non-hex digit".to_string(),
                                });
                            };
                            codepoint = (codepoint << 4) | value;
                        }
                        let Some(decoded) = char::from_u32(codepoint) else {
                            return Err(NervaError::InvalidArgument {
                                reason: "JSON unicode escape is invalid".to_string(),
                            });
                        };
                        out.push(decoded);
                    }
                    _ => {
                        return Err(NervaError::InvalidArgument {
                            reason: "unsupported JSON string escape".to_string(),
                        });
                    }
                }
            }
            ch => out.push(ch),
        }
    }
    Err(NervaError::InvalidArgument {
        reason: "JSON string is not closed".to_string(),
    })
}

fn skip_json_ws(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}

fn require_len(label: &'static str, got: usize, expected: usize) -> Result<()> {
    if got == expected {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("{label} length {got} does not match expected {expected}"),
        })
    }
}

fn require_token_in_vocab(token: TokenId, vocab_size: usize) -> Result<()> {
    if token.0 as usize >= vocab_size {
        Err(NervaError::InvalidArgument {
            reason: format!(
                "token id {} is outside tiny model vocabulary {}",
                token.0, vocab_size
            ),
        })
    } else {
        Ok(())
    }
}

fn copy_embedding_row(
    embeddings: &[f32],
    hidden: usize,
    token: TokenId,
    output: &mut [f32],
) -> Result<()> {
    require_token_in_vocab(token, embeddings.len() / hidden)?;
    let start = token.0 as usize * hidden;
    let end = start + hidden;
    output.copy_from_slice(&embeddings[start..end]);
    Ok(())
}

fn run_warm_compute_candidate(
    strategy: WarmComputeStrategy,
    rows: usize,
    cols: usize,
    matrix: &[f32],
    input: &[f32],
    ledger: &mut TokenLedger,
) -> Result<WarmComputeCandidate> {
    require_len("warm compute matrix", matrix.len(), rows * cols)?;
    require_len("warm compute input", input.len(), cols)?;
    let mut output = vec![0.0; rows];
    let matrix_bytes = matrix.len() * core::mem::size_of::<f32>();
    let input_bytes = input.len() * core::mem::size_of::<f32>();
    let output_bytes = output.len() * core::mem::size_of::<f32>();

    let visible_ns = match strategy {
        WarmComputeStrategy::CpuDram => {
            mat_vec_row_major(matrix, input, &mut output);
            let compute_ns = (rows * cols) as u64;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::CpuActivity,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Dram),
                bytes: matrix_bytes + input_bytes + output_bytes,
                latency_ns: compute_ns,
                label: "warm_matvec_cpu_dram",
            });
            compute_ns
        }
        WarmComputeStrategy::GpuResident => {
            mat_vec_row_major(matrix, input, &mut output);
            let compute_ns = rows as u64;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                block_id: None,
                from_tier: Some(MemoryTier::Vram),
                to_tier: Some(MemoryTier::Vram),
                bytes: matrix_bytes + input_bytes + output_bytes,
                latency_ns: compute_ns,
                label: "warm_matvec_gpu_resident",
            });
            compute_ns
        }
        WarmComputeStrategy::GpuStaged => {
            let copy_in_ns = (matrix_bytes + input_bytes) as u64;
            let compute_ns = rows as u64;
            let copy_out_ns = output_bytes as u64;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Vram),
                bytes: matrix_bytes + input_bytes,
                latency_ns: copy_in_ns,
                label: "warm_matvec_stage_to_gpu",
            });
            mat_vec_row_major(matrix, input, &mut output);
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                block_id: None,
                from_tier: Some(MemoryTier::Vram),
                to_tier: Some(MemoryTier::Vram),
                bytes: matrix_bytes + input_bytes + output_bytes,
                latency_ns: compute_ns,
                label: "warm_matvec_gpu_staged_compute",
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                block_id: None,
                from_tier: Some(MemoryTier::Vram),
                to_tier: Some(MemoryTier::Dram),
                bytes: output_bytes,
                latency_ns: copy_out_ns,
                label: "warm_matvec_stage_from_gpu",
            });
            copy_in_ns + compute_ns + copy_out_ns
        }
        WarmComputeStrategy::HybridSplit => {
            let split = rows / 2;
            mat_vec_row_range(matrix, input, cols, 0, split, &mut output)?;
            mat_vec_row_range(matrix, input, cols, split, rows, &mut output)?;
            let cpu_ns = (split * cols) as u64;
            let gpu_ns = rows.saturating_sub(split) as u64;
            let merge_bytes = rows.saturating_sub(split) * core::mem::size_of::<f32>();
            let merge_ns = merge_bytes as u64;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::CpuActivity,
                block_id: None,
                from_tier: Some(MemoryTier::Dram),
                to_tier: Some(MemoryTier::Dram),
                bytes: split * cols * core::mem::size_of::<f32>(),
                latency_ns: cpu_ns,
                label: "warm_matvec_hybrid_cpu_rows",
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                block_id: None,
                from_tier: Some(MemoryTier::Vram),
                to_tier: Some(MemoryTier::Vram),
                bytes: rows.saturating_sub(split) * cols * core::mem::size_of::<f32>(),
                latency_ns: gpu_ns,
                label: "warm_matvec_hybrid_gpu_rows",
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                block_id: None,
                from_tier: Some(MemoryTier::Vram),
                to_tier: Some(MemoryTier::Dram),
                bytes: merge_bytes,
                latency_ns: merge_ns,
                label: "warm_matvec_hybrid_merge",
            });
            cpu_ns.max(gpu_ns) + merge_ns
        }
    };

    Ok(WarmComputeCandidate {
        strategy,
        visible_ns,
        output_hash: hash_f32s(&output),
    })
}

fn mat_vec_row_range(
    matrix: &[f32],
    input: &[f32],
    cols: usize,
    row_start: usize,
    row_end: usize,
    output: &mut [f32],
) -> Result<()> {
    if row_start > row_end || row_end > output.len() {
        return Err(NervaError::InvalidArgument {
            reason: "matvec row range is invalid".to_string(),
        });
    }
    for row_index in row_start..row_end {
        let start = row_index * cols;
        let end = start + cols;
        output[row_index] = matrix[start..end]
            .iter()
            .zip(input.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }
    Ok(())
}

fn candidate_visible_ns(
    candidates: &[WarmComputeCandidate],
    strategy: WarmComputeStrategy,
) -> Result<u64> {
    candidates
        .iter()
        .find(|candidate| candidate.strategy == strategy)
        .map(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("missing warm compute candidate {}", strategy.label()),
        })
}

fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + eps).sqrt().recip();
    for ((out, value), weight) in output
        .iter_mut()
        .zip(input.iter().copied())
        .zip(weight.iter().copied())
    {
        *out = value * scale * weight;
    }
}

fn mat_vec_row_major(matrix: &[f32], input: &[f32], output: &mut [f32]) {
    let cols = input.len();
    for (row, out) in matrix.chunks_exact(cols).zip(output.iter_mut()) {
        *out = row
            .iter()
            .zip(input.iter())
            .map(|(weight, value)| weight * value)
            .sum();
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn single_token_attention(
    shape: TransformerBlockShape,
    _q: &[f32],
    _k: &[f32],
    v: &[f32],
    output: &mut [f32],
) {
    let head_dim = shape.head_dim();
    for head in 0..shape.heads {
        let start = head * head_dim;
        let end = start + head_dim;
        output[start..end].copy_from_slice(&v[start..end]);
    }
}

fn record_attention_block_event(
    shape: TransformerBlockShape,
    block: &KvAttentionBlock<'_>,
    ledger: &mut TokenLedger,
) {
    let (kind, executor_selected, reason) = match block.tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => (
            LedgerEventKind::DeviceActivity,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            "hot KV block is already device resident",
        ),
        MemoryTier::PinnedDram | MemoryTier::Dram | MemoryTier::Cxl | MemoryTier::Disk => (
            LedgerEventKind::CpuActivity,
            ExecutionOwner::Cpu,
            "warm KV block is cheaper to compute near than stage",
        ),
    };
    let label = match kind {
        LedgerEventKind::DeviceActivity => "attention_hot_kv_block",
        LedgerEventKind::CpuActivity => "attention_warm_kv_block",
        _ => "attention_kv_block",
    };
    let latency_ns = block.token_count as u64;
    ledger.record_execution_decision(ExecutionDecision {
        operation: "blockwise_attention",
        executor_selected,
        candidate_costs: vec![
            CandidateCost::estimated("compute-near-current-tier", latency_ns),
            CandidateCost::estimated("stage-to-gpu", latency_ns + 2),
        ],
        reason,
        predicted_visible_ns: latency_ns,
        actual_visible_ns: Some(latency_ns),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record(LedgerEvent {
        kind,
        block_id: None,
        from_tier: Some(block.tier),
        to_tier: Some(block.tier),
        bytes: block.token_count * shape.hidden * core::mem::size_of::<f32>() * 2,
        latency_ns,
        label,
    });
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn greedy_argmax(logits: &[f32]) -> Result<TokenId> {
    if logits.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "greedy argmax requires non-empty logits".to_string(),
        });
    }
    let mut best_index = 0usize;
    let mut best_value = logits[0];
    if !best_value.is_finite() {
        return Err(NervaError::InvalidArgument {
            reason: "greedy argmax saw non-finite logit".to_string(),
        });
    }
    for (index, value) in logits.iter().copied().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "greedy argmax saw non-finite logit".to_string(),
            });
        }
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    Ok(TokenId(best_index as u32))
}

fn expected_cycle(seed_token: TokenId, steps: usize, vocab_size: usize) -> Vec<TokenId> {
    (0..steps)
        .map(|step| TokenId((seed_token.0 + step as u32 + 1) % vocab_size as u32))
        .collect()
}

fn hash_f32s(values: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn hash_tokens(values: &[TokenId]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.0.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn hash_metadata(metadata: &HfModelMetadata) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in [
        metadata.hidden_size as u64,
        metadata.num_hidden_layers as u64,
        metadata.num_attention_heads as u64,
        metadata.num_key_value_heads as u64,
        metadata.intermediate_size as u64,
        metadata.vocab_size as u64,
        metadata.max_position_embeddings.unwrap_or_default() as u64,
        metadata.head_dim() as u64,
        metadata.kv_groups() as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for byte in metadata.architecture.as_str().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(dtype) = metadata.torch_dtype {
        for byte in dtype_to_str(dtype).as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn hash_weight_layout(plan: &HfWeightLayoutPlan) -> u64 {
    let mut hash = hash_metadata(&plan.metadata);
    for block in &plan.blocks {
        for byte in block.role.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for value in [
            block.layer.map(u64::from).unwrap_or(u64::MAX),
            block.rows as u64,
            block.cols as u64,
            block.elements as u64,
            block.bytes as u64,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    for value in [
        plan.total_weight_bytes as u64,
        plan.per_layer_weight_bytes as u64,
        plan.static_weight_bytes as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn hash_tensor_manifest(manifest: &HfTensorManifest) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for entry in &manifest.entries {
        for byte in entry.name.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in entry.role.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for value in [
            entry.layer.map(u64::from).unwrap_or(u64::MAX),
            entry.rows as u64,
            entry.cols as u64,
            u64::from(entry.rank),
            entry.elements as u64,
            entry.bytes as u64,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    for value in [
        manifest.entries.len() as u64,
        manifest.total_weight_bytes as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn token_ids_to_json(tokens: &[TokenId]) -> String {
    let mut out = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&token.0.to_string());
    }
    out.push(']');
    out
}

fn json_opt_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_f32(value: Option<f32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_dtype(value: Option<DType>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", dtype_to_str(value)),
    )
}

fn dtype_to_str(value: DType) -> &'static str {
    match value {
        DType::U8 => "u8",
        DType::U16 => "u16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
    }
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_hf_config_metadata() {
        let metadata = parse_hf_config_metadata(
            r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0,
                "torch_dtype": "bfloat16"
            }"#,
        )
        .unwrap();

        assert_eq!(metadata.architecture, HfArchitectureKind::Llama);
        assert_eq!(
            metadata.block_shape(),
            TransformerBlockShape::new(4096, 32, 11008)
        );
        assert_eq!(metadata.head_dim(), 128);
        assert_eq!(metadata.kv_groups(), 4);
        assert_eq!(metadata.torch_dtype, Some(DType::BF16));
        assert!(metadata.to_json().contains("\"architecture\":\"llama\""));
    }

    #[test]
    fn parses_model_type_and_defaults_kv_heads_to_attention_heads() {
        let metadata = parse_hf_config_metadata(
            r#"{
                "model_type": "mistral",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "vocab_size": 32000,
                "torch_dtype": "float16"
            }"#,
        )
        .unwrap();

        assert_eq!(metadata.architecture, HfArchitectureKind::Mistral);
        assert_eq!(metadata.num_key_value_heads, 32);
        assert_eq!(metadata.kv_groups(), 1);
        assert_eq!(metadata.torch_dtype, Some(DType::F16));
    }

    #[test]
    fn rejects_invalid_hf_metadata_shapes_and_dtypes() {
        let bad_heads = parse_hf_config_metadata(
            r#"{
                "model_type": "llama",
                "hidden_size": 4097,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000
            }"#,
        );
        let bad_dtype = parse_hf_config_metadata(
            r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "torch_dtype": "int4"
            }"#,
        );

        assert!(bad_heads.is_err());
        assert!(bad_dtype.is_err());
    }

    #[test]
    fn hf_metadata_probe_reports_valid_shape() {
        let summary = hf_metadata_probe().unwrap();

        assert_eq!(summary.status, HfMetadataProbeStatus::Ok);
        assert_eq!(summary.metadata.architecture, HfArchitectureKind::Llama);
        assert_eq!(summary.metadata.hidden_size, 4096);
        assert_eq!(summary.metadata.num_attention_heads, 32);
        assert_eq!(summary.metadata.num_key_value_heads, 8);
        assert_eq!(summary.metadata.head_dim(), 128);
        assert_eq!(summary.metadata.kv_groups(), 4);
        assert_ne!(summary.metadata_hash, 0);
        assert!(summary.to_json().contains("\"metadata\""));
    }

    #[test]
    fn plans_hf_weight_layout_from_metadata() {
        let metadata = parse_hf_config_metadata(
            r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
        )
        .unwrap();
        let plan = plan_hf_weight_layout(&metadata).unwrap();

        assert_eq!(plan.blocks.len(), 20);
        assert_eq!(plan.static_weight_bytes, 160);
        assert_eq!(plan.per_layer_weight_bytes, 304);
        assert_eq!(plan.total_weight_bytes, 768);
        assert_eq!(plan.blocks[0].role, WeightBlockRole::TokenEmbedding);
        assert_eq!(plan.blocks[2].role, WeightBlockRole::QueryProjection);
        assert_eq!(plan.blocks[2].rows, 4);
        assert_eq!(plan.blocks[2].cols, 4);
        assert_eq!(plan.blocks[3].role, WeightBlockRole::KeyProjection);
        assert_eq!(plan.blocks[3].rows, 2);
        assert_eq!(plan.blocks[3].cols, 4);
    }

    #[test]
    fn hf_weight_layout_probe_reports_llama_scale_counts() {
        let summary = hf_weight_layout_probe().unwrap();

        assert_eq!(summary.status, HfWeightLayoutProbeStatus::Ok);
        assert_eq!(summary.plan.blocks.len(), 290);
        assert_eq!(summary.plan.static_weight_bytes, 524_288_000);
        assert_eq!(summary.plan.per_layer_weight_bytes, 354_435_072);
        assert_eq!(summary.plan.total_weight_bytes, 11_866_210_304);
        assert_eq!(summary.plan.dtype, DType::BF16);
        assert_ne!(summary.layout_hash, 0);
        assert!(summary.to_json().contains("\"blocks\":290"));
    }

    #[test]
    fn weight_layout_requires_exact_declared_dtype() {
        let mut metadata = hf_metadata_probe().unwrap().metadata;
        metadata.torch_dtype = None;
        assert!(plan_hf_weight_layout(&metadata).is_err());

        metadata.torch_dtype = Some(DType::U8);
        assert!(plan_hf_weight_layout(&metadata).is_err());
    }

    #[test]
    fn builds_canonical_hf_tensor_manifest_names() {
        let metadata = parse_hf_config_metadata(
            r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 10,
                "torch_dtype": "float16"
            }"#,
        )
        .unwrap();
        let plan = plan_hf_weight_layout(&metadata).unwrap();
        let manifest = build_hf_tensor_manifest(&plan).unwrap();

        assert_eq!(manifest.entries.len(), plan.blocks.len());
        assert_eq!(manifest.total_weight_bytes, plan.total_weight_bytes);
        assert_eq!(manifest.entries[0].name, "model.embed_tokens.weight");
        assert_eq!(
            manifest.entries[1].name,
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(manifest.entries[1].rank, 1);
        assert_eq!(
            manifest.entries[2].name,
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(manifest.entries[2].rank, 2);
        assert_eq!(
            manifest.entries[9].name,
            "model.layers.0.mlp.down_proj.weight"
        );
        assert_eq!(
            manifest.entries[10].name,
            "model.layers.1.input_layernorm.weight"
        );
        assert_eq!(manifest.entries.last().unwrap().name, "lm_head.weight");
        assert_ne!(manifest.manifest_hash, 0);
    }

    #[test]
    fn tensor_manifest_rejects_unsupported_architecture_names() {
        let mut metadata = hf_metadata_probe().unwrap().metadata;
        metadata.architecture = HfArchitectureKind::Gemma;
        let plan = plan_hf_weight_layout(&metadata).unwrap();

        assert!(build_hf_tensor_manifest(&plan).is_err());
    }

    #[test]
    fn hf_tensor_manifest_probe_reports_llama_manifest() {
        let summary = hf_tensor_manifest_probe().unwrap();

        assert_eq!(summary.status, HfTensorManifestProbeStatus::Ok);
        assert_eq!(summary.manifest.entries.len(), 290);
        assert_eq!(summary.manifest.total_weight_bytes, 11_866_210_304);
        assert_eq!(
            summary.manifest.entries.first().unwrap().name,
            "model.embed_tokens.weight"
        );
        assert_eq!(
            summary.manifest.entries.last().unwrap().name,
            "lm_head.weight"
        );
        assert_ne!(summary.manifest.manifest_hash, 0);
        assert!(summary.to_json().contains("\"entries\":290"));
    }

    #[test]
    fn zero_block_preserves_residual() {
        let shape = TransformerBlockShape::new(4, 2, 8);
        let block = ReferenceTransformerBlock::zero_for_shape(shape).unwrap();
        let mut scratch = TransformerBlockScratch::new(shape).unwrap();
        let mut output = [0.0; 4];
        let input = [1.0, -2.0, 3.0, -4.0];
        let mut ledger = TokenLedger::new(0);

        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .unwrap();

        assert_eq!(output, input);
        assert_eq!(ledger.hot_path_allocations, 0);
        assert!(ledger.require_zero_hot_path_allocations().is_ok());
    }

    #[test]
    fn nontrivial_block_matches_hand_reference() {
        let shape = TransformerBlockShape::new(2, 1, 2);
        let block = ReferenceTransformerBlock::new(
            shape,
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.5, 0.0, 0.0, 0.5],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            1e-5,
        )
        .unwrap();
        let mut scratch = TransformerBlockScratch::new(shape).unwrap();
        let mut output = [0.0; 2];
        let input = [1.0, 2.0];
        let mut ledger = TokenLedger::new(7);

        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .unwrap();

        let attn_norm_scale = ((1.0_f32 + 4.0) / 2.0 + 1e-5).sqrt().recip();
        let attn = [input[0] * attn_norm_scale, input[1] * attn_norm_scale];
        let residual = [input[0] + attn[0], input[1] + attn[1]];
        let mlp_norm_scale = ((residual[0] * residual[0] + residual[1] * residual[1]) / 2.0 + 1e-5)
            .sqrt()
            .recip();
        let mlp_norm = [residual[0] * mlp_norm_scale, residual[1] * mlp_norm_scale];
        let expected = [
            residual[0] + silu(0.5 * mlp_norm[0]) * mlp_norm[0],
            residual[1] + silu(0.5 * mlp_norm[1]) * mlp_norm[1],
        ];

        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(ledger.hot_path_allocations, 0);
    }

    #[test]
    fn rejects_bad_shapes_and_scratch_mismatch() {
        assert!(TransformerBlockShape::new(3, 2, 4).validate().is_err());
        let block =
            ReferenceTransformerBlock::zero_for_shape(TransformerBlockShape::new(4, 2, 8)).unwrap();
        let mut scratch =
            TransformerBlockScratch::new(TransformerBlockShape::new(2, 1, 2)).unwrap();
        let mut ledger = TokenLedger::new(0);
        let mut output = [0.0; 4];
        assert!(
            block
                .forward_into(&[0.0; 4], &mut scratch, &mut output, &mut ledger)
                .is_err()
        );
    }

    #[test]
    fn reference_block_smoke_reports_hash_and_no_allocations() {
        let summary = reference_block_smoke().unwrap();
        assert_eq!(summary.status, ReferenceBlockSmokeStatus::Ok);
        assert_eq!(summary.hidden, 2);
        assert_eq!(summary.heads, 1);
        assert_eq!(summary.intermediate, 2);
        assert_eq!(summary.hot_path_allocations, 0);
        assert_eq!(summary.output_hash, 3_850_145_622_605_741_247);
        assert!(summary.to_json().contains("\"status\":\"ok\""));
    }

    #[test]
    fn blockwise_attention_matches_dense_reference_across_tiers() {
        let shape = TransformerBlockShape::new(4, 2, 4);
        let query = [0.5, -1.0, 0.25, 0.75];
        let keys = [0.1, 0.2, 0.3, 0.4, 0.0, -0.5, 0.6, 0.2, 0.7, 0.1, -0.2, 0.3];
        let values = [
            1.0, 0.0, 0.5, -0.5, -1.0, 2.0, 0.25, 0.75, 0.3, -0.8, 1.5, 0.2,
        ];
        let blocks = [
            KvAttentionBlock::new(&keys[..4], &values[..4], 1, MemoryTier::Dram),
            KvAttentionBlock::new(&keys[4..], &values[4..], 2, MemoryTier::Vram),
        ];
        let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
        let mut output = [0.0; 4];
        let mut ledger = TokenLedger::new(11);

        exact_blockwise_attention_into(
            shape,
            &query,
            &blocks,
            &mut scratch,
            &mut output,
            &mut ledger,
        )
        .unwrap();

        let expected = dense_attention_reference(shape, &query, &keys, &values, 3);
        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(ledger.event_count(LedgerEventKind::CpuActivity), 1);
        assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
        assert_eq!(ledger.total_latency_ns(), 3);
        assert_eq!(ledger.hot_path_allocations, 0);
    }

    #[test]
    fn blockwise_attention_rejects_empty_and_malformed_blocks() {
        let shape = TransformerBlockShape::new(2, 1, 2);
        let query = [1.0, 0.0];
        let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
        let mut output = [0.0; 2];
        let mut ledger = TokenLedger::new(0);

        assert!(
            exact_blockwise_attention_into(
                shape,
                &query,
                &[],
                &mut scratch,
                &mut output,
                &mut ledger
            )
            .is_err()
        );

        let bad_block = [KvAttentionBlock::new(
            &[1.0],
            &[1.0, 0.0],
            1,
            MemoryTier::Dram,
        )];
        assert!(
            exact_blockwise_attention_into(
                shape,
                &query,
                &bad_block,
                &mut scratch,
                &mut output,
                &mut ledger,
            )
            .is_err()
        );
    }

    #[test]
    fn blockwise_attention_smoke_reports_tier_events() {
        let summary = blockwise_attention_smoke().unwrap();
        assert_eq!(summary.status, BlockwiseAttentionSmokeStatus::Ok);
        assert_eq!(summary.hidden, 2);
        assert_eq!(summary.heads, 1);
        assert_eq!(summary.blocks, 2);
        assert_eq!(summary.tokens, 4);
        assert_eq!(summary.cpu_block_events, 1);
        assert_eq!(summary.device_block_events, 1);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(summary.to_json().contains("\"device_block_events\":1"));
    }

    #[test]
    fn warm_compute_probe_compares_all_exact_strategies() {
        let summary = warm_compute_probe().unwrap();

        assert_eq!(summary.status, WarmComputeProbeStatus::Ok);
        assert_eq!(summary.rows, 4);
        assert_eq!(summary.cols, 4);
        assert_eq!(summary.candidates.len(), 4);
        assert_eq!(summary.selected_strategy, WarmComputeStrategy::GpuResident);
        assert!(summary.parity);
        assert!(summary.cpu_beats_staged);
        assert_eq!(summary.execution_decisions, 1);
        assert_eq!(summary.cpu_events, 2);
        assert_eq!(summary.device_events, 3);
        assert_eq!(summary.copy_events, 3);
        assert_eq!(summary.copy_bytes, 104);
        assert_eq!(summary.total_latency_ns, 138);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(
            summary
                .candidates
                .iter()
                .all(|candidate| candidate.output_hash == summary.output_hash)
        );
        assert!(
            summary
                .to_json()
                .contains("\"selected_strategy\":\"gpu-resident\"")
        );
    }

    #[test]
    fn tiny_greedy_model_matches_expected_token_cycle() {
        let model = tiny_cycle_model().unwrap();
        let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();
        let output = model.decode_greedy(TokenId(0), 8, &mut scratch).unwrap();

        assert_eq!(
            output.tokens,
            vec![
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(0),
                TokenId(1),
                TokenId(2),
                TokenId(3),
                TokenId(0),
            ]
        );
        assert_eq!(output.ledgers.len(), 8);
        assert_eq!(
            output
                .ledgers
                .iter()
                .map(|ledger| ledger.event_count(LedgerEventKind::DeviceActivity))
                .sum::<u64>(),
            8
        );
        assert_eq!(
            output
                .ledgers
                .iter()
                .map(|ledger| ledger.hot_path_allocations)
                .sum::<u64>(),
            0
        );
    }

    #[test]
    fn tiny_greedy_model_rejects_bad_decode_inputs() {
        let model = tiny_cycle_model().unwrap();
        let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size()).unwrap();

        assert!(model.decode_greedy(TokenId(0), 0, &mut scratch).is_err());
        assert!(model.decode_greedy(TokenId(99), 1, &mut scratch).is_err());

        let mut wrong_scratch =
            TinyGreedyDecodeScratch::new(TransformerBlockShape::new(4, 2, 4), model.vocab_size())
                .unwrap();
        assert!(
            model
                .decode_greedy(TokenId(0), 1, &mut wrong_scratch)
                .is_err()
        );
    }

    #[test]
    fn tiny_greedy_decode_smoke_reports_parity_and_ledger() {
        let summary = tiny_greedy_decode_smoke(8).unwrap();

        assert_eq!(summary.status, TinyGreedyDecodeStatus::Ok);
        assert_eq!(summary.seed_token, TokenId(0));
        assert_eq!(summary.steps, 8);
        assert_eq!(summary.vocab_size, 4);
        assert!(summary.parity);
        assert_eq!(summary.tokens, summary.expected_tokens);
        assert_eq!(summary.ledger_count, 8);
        assert_eq!(summary.device_events, 8);
        assert_eq!(summary.total_latency_ns, 8);
        assert_eq!(summary.hot_path_allocations, 0);
        assert!(summary.to_json().contains("\"parity\":true"));
    }

    fn dense_attention_reference(
        shape: TransformerBlockShape,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        token_count: usize,
    ) -> Vec<f32> {
        let head_dim = shape.head_dim();
        let scale = (head_dim as f32).sqrt().recip();
        let mut output = vec![0.0; shape.hidden];
        for head in 0..shape.heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            let mut max_score = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(token_count);
            for token_index in 0..token_count {
                let token_start = token_index * shape.hidden + head_start;
                let token_end = token_start + head_dim;
                let score =
                    dot(&query[head_start..head_end], &keys[token_start..token_end]) * scale;
                max_score = max_score.max(score);
                scores.push(score);
            }
            let mut denom = 0.0f32;
            for (token_index, score) in scores.iter().copied().enumerate() {
                let weight = (score - max_score).exp();
                denom += weight;
                let token_start = token_index * shape.hidden + head_start;
                let token_end = token_start + head_dim;
                for (out, value) in output[head_start..head_end]
                    .iter_mut()
                    .zip(values[token_start..token_end].iter().copied())
                {
                    *out += weight * value;
                }
            }
            for out in &mut output[head_start..head_end] {
                *out /= denom;
            }
        }
        output
    }
}
