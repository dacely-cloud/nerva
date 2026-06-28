use std::path::Path;
use std::time::{Duration, Instant};

use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput;
use nerva_runtime::engine::hf_cuda_decode::file_backed::progress::{
    HfCudaDeviceProgressPhase, HfCudaDeviceSessionChunkProgress,
};

use crate::cli::ui::format;
use crate::cli::ui::logo_image::TerminalLogo;
use crate::cli::ui::stats::DecodeStats;

pub(crate) struct UiState {
    pub(crate) logo: Option<TerminalLogo>,
    pub(crate) phase: &'static str,
    pub(crate) title: String,
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) context: String,
    pub(crate) output_cap: String,
    pub(crate) queue: String,
    pub(crate) compute: String,
    pub(crate) stop_tokens: String,
    pub(crate) progress: Option<HfCudaDeviceSessionChunkProgress>,
    pub(crate) decode_samples: Vec<u64>,
    pub(crate) prompt_tokens: usize,
    pub(crate) decode_elapsed_ns: u64,
    pub(crate) logs: Vec<String>,
    pub(crate) summary: Vec<(String, String)>,
    pub(crate) generated_text: String,
    pub(crate) spinner: usize,
    pub(crate) boot: Instant,
}

impl UiState {
    pub(crate) fn new(logo: Option<TerminalLogo>) -> Self {
        Self {
            logo,
            phase: "boot",
            title: "initializing console".to_string(),
            model: String::new(),
            prompt: String::new(),
            context: String::new(),
            output_cap: String::new(),
            queue: String::new(),
            compute: String::new(),
            stop_tokens: String::new(),
            progress: None,
            decode_samples: Vec::new(),
            prompt_tokens: 0,
            decode_elapsed_ns: 0,
            logs: Vec::new(),
            summary: Vec::new(),
            generated_text: String::new(),
            spinner: 0,
            boot: Instant::now(),
        }
    }

    pub(crate) fn configure(&mut self, input: ConfigureInput<'_>) {
        self.phase = "configure";
        self.title = "request and model plan".to_string();
        self.model = input.model_path.display().to_string();
        self.prompt = format!("{}, {} tokens", input.prompt_mode, input.prompt_tokens);
        self.context = format!("{} tokens", input.context_tokens);
        self.output_cap = format!("{} tokens", input.output_tokens);
        self.queue = format!("{} host-visible slots", input.queue_capacity);
        self.compute = input
            .compute_capability
            .map(|value| format!("sm_{value}"))
            .unwrap_or_else(|| "auto-discovery".to_string());
        self.stop_tokens = input.stop_token_count.to_string();
        self.prompt_tokens = input.prompt_tokens;
        self.push_log("request configured");
        self.push_log(format!("model {}", input.model_path.display()));
        self.push_log(format!(
            "prompt {} tokens, context {} tokens, output cap {} tokens",
            input.prompt_tokens, input.context_tokens, input.output_tokens
        ));
        self.push_log(format!(
            "queue {} slots, stop policy {} ids",
            input.queue_capacity, input.stop_token_count
        ));
    }

    pub(crate) fn set_phase(&mut self, phase: &'static str, title: impl Into<String>) {
        self.phase = phase;
        self.title = title.into();
        self.push_log(format!("{phase}: {}", self.title));
    }

    pub(crate) fn update_progress(&mut self, progress: HfCudaDeviceSessionChunkProgress) {
        self.phase = progress.phase.as_str();
        if progress.phase == HfCudaDeviceProgressPhase::Decode {
            self.decode_elapsed_ns = self.decode_elapsed_ns.saturating_add(progress.wall_ns);
            let rate = if progress.wall_ns == 0 {
                0
            } else {
                1_000_000_000u64 / progress.wall_ns
            };
            self.decode_samples.push(rate);
            if self.decode_samples.len() > 48 {
                self.decode_samples.remove(0);
            }
        }
        self.progress = Some(progress);
    }

    pub(crate) fn tick(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
    }

    pub(crate) fn finish(&mut self, output: &HfCudaDeviceGenerateOutput, elapsed: Duration) {
        self.phase = "complete";
        self.title = "run complete".to_string();
        let create = &output.stream.create;
        let start = &output.stream.start;
        let stats = DecodeStats::from_output(output);
        self.summary = vec![
            ("elapsed".into(), format::duration(elapsed)),
            ("stop".into(), output.stop_reason().as_str().into()),
            (
                "generated".into(),
                format!("{} tokens", output.tokens().len()),
            ),
            (
                "model".into(),
                format!("{} layers, hidden {}", create.layer_count, create.hidden),
            ),
            (
                "weights".into(),
                format::bytes(create.resident_weight_bytes),
            ),
            ("weight H2D".into(), format::bytes(create.h2d_bytes)),
            (
                "load window".into(),
                format::gb_per_s(create.h2d_bytes, elapsed),
            ),
            ("KV arena".into(), format::bytes(create.resident_kv_bytes)),
            (
                "device arena".into(),
                format::bytes(create.device_arena_bytes),
            ),
            ("prompt H2D".into(), format::bytes(start.h2d_bytes)),
            ("decode tok/s".into(), decode_rate(&stats)),
            ("latency".into(), latency_line(&stats)),
            ("kernels".into(), kernel_line(&stats)),
            ("ledger".into(), ledger_line(&stats)),
            ("time split".into(), time_split_line(&stats)),
            ("boot clock".into(), format::duration(self.boot.elapsed())),
        ];
        self.push_log("completed generation and ledger aggregation");
    }

    pub(crate) fn set_generated_text(&mut self, generated_text: impl Into<String>) {
        self.generated_text = generated_text.into();
        if !self.generated_text.is_empty() {
            self.push_log("decoded generated text");
        }
    }

    fn push_log(&mut self, value: impl Into<String>) {
        self.logs.push(format!(
            "{:>7}  {}",
            format::duration(self.boot.elapsed()),
            value.into()
        ));
        if self.logs.len() > 64 {
            self.logs.remove(0);
        }
    }
}

pub(crate) struct ConfigureInput<'a> {
    pub(crate) model_path: &'a Path,
    pub(crate) prompt_mode: &'a str,
    pub(crate) prompt_tokens: usize,
    pub(crate) context_tokens: usize,
    pub(crate) output_tokens: usize,
    pub(crate) queue_capacity: usize,
    pub(crate) compute_capability: Option<u32>,
    pub(crate) stop_token_count: usize,
}

fn decode_rate(stats: &DecodeStats) -> String {
    if stats.wall_ns == 0 {
        "n/a".to_string()
    } else {
        format::tokens_per_s(stats.tokens, Duration::from_nanos(stats.wall_ns))
    }
}

fn latency_line(stats: &DecodeStats) -> String {
    format!(
        "mean {} p50 {} p95 {} p99 {}",
        format::ms_from_ns(stats.mean_ns()),
        format::ms_from_ns(stats.p50_ns),
        format::ms_from_ns(stats.p95_ns),
        format::ms_from_ns(stats.p99_ns)
    )
}

fn kernel_line(stats: &DecodeStats) -> String {
    format!(
        "{} launches, {} graph nodes, {} replays, {} cache hits",
        stats.kernel_launches, stats.graph_nodes, stats.graph_replays, stats.graph_cache_hits
    )
}

fn ledger_line(stats: &DecodeStats) -> String {
    format!(
        "H2D {} D2H {} sync {} host_edges {} hot_alloc {}",
        format::bytes(stats.h2d_bytes),
        format::bytes(stats.d2h_bytes),
        stats.sync_calls,
        stats.host_causality_edges,
        stats.hot_path_allocations
    )
}

fn time_split_line(stats: &DecodeStats) -> String {
    format!(
        "proj {} attn {} mlp {} norm {} sample {}",
        format::ms_from_ns(stats.projection_ns),
        format::ms_from_ns(stats.attention_ns),
        format::ms_from_ns(stats.mlp_ns),
        format::ms_from_ns(stats.norm_ns),
        format::ms_from_ns(stats.sampling_ns)
    )
}
