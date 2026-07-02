use std::ffi::OsString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use nerva_runtime::engine::hf_cuda_decode::file_backed::progress::{
    HfCudaDeviceProgressPhase, HfCudaDeviceSessionChunkProgress,
};

use crate::cli::ui::color::{ColorMode, Tone, code, paint, reset, stderr_color_mode};
use crate::cli::ui::format;
use crate::cli::ui::logo_image::TerminalLogo;
use crate::cli::ui::state::{ConfigureInput, UiState};
use crate::cli::ui::stats::DecodeStats;
use crate::cli::ui::terminal::TuiSession;

const PLAIN_PROGRESS_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone)]
pub(crate) struct NervaCliLogger {
    inner: Arc<Mutex<NervaCliLoggerInner>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LoggerMode {
    Quiet,
    Plain,
    DebugTui,
}

struct NervaCliLoggerInner {
    mode: LoggerMode,
    tui: Option<TuiSession>,
    state: UiState,
    last_plain_emit: Instant,
    banner_printed: bool,
    color: ColorMode,
}

impl NervaCliLogger {
    pub(crate) fn new(json: bool, debug: bool) -> Self {
        let requested_mode = if json {
            LoggerMode::Quiet
        } else if debug {
            LoggerMode::DebugTui
        } else {
            LoggerMode::Plain
        };
        let tui = (requested_mode == LoggerMode::DebugTui)
            .then(TuiSession::start)
            .flatten();
        let mode = if requested_mode == LoggerMode::DebugTui && tui.is_none() {
            LoggerMode::Plain
        } else {
            requested_mode
        };
        let logo = tui
            .as_ref()
            .and_then(|_| TerminalLogo::load((terminal_width_hint() / 2).clamp(32, 72)));
        Self {
            inner: Arc::new(Mutex::new(NervaCliLoggerInner {
                mode,
                tui,
                state: UiState::new(logo),
                last_plain_emit: Instant::now() - PLAIN_PROGRESS_INTERVAL,
                banner_printed: false,
                color: stderr_color_mode(),
            })),
        }
    }

    pub(crate) fn is_tui_active(&self) -> bool {
        self.inner
            .lock()
            .map(|inner| inner.tui.is_some())
            .unwrap_or(false)
    }

    pub(crate) fn start_progress_guards(&self) -> (NativeLoadProgressGuard, UiTickerGuard) {
        (
            NativeLoadProgressGuard::install(self.native_load_progress_mode()),
            UiTickerGuard::start(Arc::clone(&self.inner)),
        )
    }

    pub(crate) fn banner(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.banner();
        }
    }

    pub(crate) fn configure(
        &mut self,
        model_path: &Path,
        prompt_mode: &str,
        prompt_tokens: usize,
        context_tokens: usize,
        output_tokens: usize,
        queue_capacity: usize,
        compute_capability: Option<u32>,
        stop_token_count: usize,
    ) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.state.configure(ConfigureInput {
                model_path,
                prompt_mode,
                prompt_tokens,
                context_tokens,
                output_tokens,
                queue_capacity,
                compute_capability,
                stop_token_count,
            });
            inner.configured();
        }
    }

    pub(crate) fn runtime_init(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner
                .state
                .set_phase("runtime", "initializing scheduler and CUDA backend");
            inner.phase_changed();
        }
    }

    pub(crate) fn load_start(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner
                .state
                .set_phase("load", "resident weight plan, arenas, H2D, warmup");
            inner.phase_changed();
        }
    }

    pub(crate) fn decode_progress(&mut self, progress: HfCudaDeviceSessionChunkProgress) {
        if let Ok(mut inner) = self.inner.lock() {
            if inner.mode == LoggerMode::Quiet {
                return;
            }
            inner.state.update_progress(progress);
            inner.progress_changed();
        }
    }

    pub(crate) fn finish(
        &mut self,
        output: &nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput,
        elapsed: std::time::Duration,
    ) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.state.finish(output, elapsed);
            inner.finished(output, elapsed);
        }
    }

    pub(crate) fn generated_text(&mut self, generated_text: impl Into<String>) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.state.set_generated_text(generated_text);
            inner.generated_text_ready();
        }
    }

    fn native_load_progress_mode(&self) -> &'static str {
        self.inner
            .lock()
            .map(|inner| match inner.mode {
                LoggerMode::Plain if inner.color.truecolor() => "color",
                LoggerMode::Plain if inner.color == ColorMode::Ansi16 => "ansi",
                LoggerMode::Plain => "plain",
                LoggerMode::Quiet | LoggerMode::DebugTui => "quiet",
            })
            .unwrap_or("quiet")
    }
}

impl NervaCliLoggerInner {
    fn banner(&mut self) {
        match self.mode {
            LoggerMode::Quiet => {}
            LoggerMode::Plain => self.print_plain_banner(),
            LoggerMode::DebugTui => self.draw(),
        }
    }

    fn configured(&mut self) {
        match self.mode {
            LoggerMode::Quiet => {}
            LoggerMode::Plain => {
                self.print_plain_line("request", format!("model {}", self.state.model));
                self.print_plain_line(
                    "request",
                    format!(
                        "prompt {} context {} output {} queue {} device {} stop_ids {}",
                        self.state.prompt,
                        self.state.context,
                        self.state.output_cap,
                        self.state.queue,
                        self.state.compute,
                        self.state.stop_tokens
                    ),
                );
            }
            LoggerMode::DebugTui => self.draw(),
        }
    }

    fn phase_changed(&mut self) {
        match self.mode {
            LoggerMode::Quiet => {}
            LoggerMode::Plain => self.print_plain_line(self.state.phase, self.state.title.clone()),
            LoggerMode::DebugTui => self.draw(),
        }
    }

    fn progress_changed(&mut self) {
        match self.mode {
            LoggerMode::Quiet => {}
            LoggerMode::Plain => {
                let final_progress = self
                    .state
                    .progress
                    .as_ref()
                    .map(|progress| progress.hit_stop || progress.generated == progress.requested)
                    .unwrap_or(false);
                if self.should_emit_plain_progress() || final_progress {
                    self.print_plain_progress();
                }
            }
            LoggerMode::DebugTui => {
                if self
                    .state
                    .progress
                    .as_ref()
                    .map(should_draw_debug_progress)
                    .unwrap_or(false)
                {
                    self.draw();
                }
            }
        }
    }

    fn finished(
        &mut self,
        output: &nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput,
        elapsed: std::time::Duration,
    ) {
        match self.mode {
            LoggerMode::Quiet => {}
            LoggerMode::Plain => self.print_plain_summary(output, elapsed),
            LoggerMode::DebugTui => self.draw(),
        }
    }

    fn generated_text_ready(&mut self) {
        match self.mode {
            LoggerMode::Quiet | LoggerMode::Plain => {}
            LoggerMode::DebugTui => self.draw(),
        }
    }

    fn draw(&mut self) {
        self.state.tick();
        if let Some(tui) = &mut self.tui {
            tui.draw(&mut self.state);
        }
    }

    fn tick_or_log(&mut self) {
        match self.mode {
            LoggerMode::Quiet => {}
            LoggerMode::Plain => {
                if self.should_emit_plain_progress() {
                    if self.state.progress.is_some() {
                        let final_progress = self
                            .state
                            .progress
                            .as_ref()
                            .map(|progress| {
                                progress.hit_stop || progress.generated == progress.requested
                            })
                            .unwrap_or(false);
                        if !final_progress {
                            self.print_plain_progress();
                        }
                    } else if self.state.phase != "load" {
                        self.print_plain_line(self.state.phase, self.state.title.clone());
                    }
                }
            }
            LoggerMode::DebugTui => self.draw(),
        }
    }

    fn print_plain_banner(&mut self) {
        if self.banner_printed {
            return;
        }
        self.banner_printed = true;
        for line in crate::cli::ui::logo_image::plain_brand(terminal_width_hint(), self.color) {
            eprintln!("{line}");
        }
        self.print_plain_line("version", env!("CARGO_PKG_VERSION"));
        self.print_plain_line("boot", "starting NERVA");
    }

    fn print_plain_progress(&mut self) {
        let Some(progress) = self.state.progress.clone() else {
            return;
        };
        if progress.phase == HfCudaDeviceProgressPhase::Prefill {
            self.print_plain_prefill_progress(progress);
            return;
        }
        let elapsed = Duration::from_nanos(self.state.decode_elapsed_ns.max(1));
        let avg_rate = format::tokens_per_s(progress.generated, elapsed);
        let inst_rate = format::tokens_per_s(
            progress.observed.max(1),
            Duration::from_nanos(progress.wall_ns.max(1)),
        );
        let percent = if progress.requested == 0 {
            0.0
        } else {
            progress.generated as f64 * 100.0 / progress.requested as f64
        };
        let profiled_ns = progress
            .projection_ns
            .saturating_add(progress.attention_ns)
            .saturating_add(progress.mlp_ns)
            .saturating_add(progress.norm_ns)
            .saturating_add(progress.sampling_ns);
        let untracked_ns = progress.wall_ns.saturating_sub(profiled_ns);
        let mut fields = vec![
            paint(
                self.color,
                Tone::Green,
                format!(
                    "{}/{} ({percent:.1}%)",
                    progress.generated, progress.requested
                ),
            ),
            metric(self.color, "avg", avg_rate, Tone::Cyan),
            metric(self.color, "inst", inst_rate, Tone::Green),
            metric(
                self.color,
                "bw",
                format::weight_bandwidth(
                    progress.resident_weight_bytes,
                    progress.observed.max(1) as u64,
                    Duration::from_nanos(progress.wall_ns.max(1)),
                    progress.device_memory_bandwidth_bps,
                ),
                Tone::Green,
            ),
        ];
        if progress.chunk_requested > 1 {
            fields.push(metric(
                self.color,
                "chunk",
                format!("{}/{}", progress.observed, progress.chunk_requested),
                Tone::Green,
            ));
        }
        fields.extend([
            metric(
                self.color,
                "last",
                format::ms_from_ns(progress.wall_ns),
                Tone::Yellow,
            ),
            metric(
                self.color,
                "gpu",
                format::ms_from_ns(progress.device_ns),
                Tone::Yellow,
            ),
            metric(
                self.color,
                "kv",
                progress.kv_tokens.to_string(),
                Tone::Magenta,
            ),
            metric(
                self.color,
                "profile",
                format::ms_from_ns(profiled_ns),
                Tone::Blue,
            ),
            metric(
                self.color,
                "untracked",
                format::ms_from_ns(untracked_ns),
                if untracked_ns > progress.wall_ns / 4 {
                    Tone::Red
                } else {
                    Tone::Dim
                },
            ),
            metric(
                self.color,
                "attn",
                format::ms_from_ns(progress.attention_ns),
                Tone::Cyan,
            ),
            metric(
                self.color,
                "kernels",
                progress.kernel_launches.to_string(),
                Tone::Magenta,
            ),
            metric(
                self.color,
                "graph",
                progress.graph_nodes.to_string(),
                Tone::Magenta,
            ),
            metric(
                self.color,
                "replay",
                progress.graph_replays.to_string(),
                Tone::Magenta,
            ),
            metric(
                self.color,
                "cache",
                progress.graph_cache_hits.to_string(),
                Tone::Blue,
            ),
            metric(
                self.color,
                "hot",
                progress.hot_path_allocations.to_string(),
                if progress.hot_path_allocations == 0 {
                    Tone::Green
                } else {
                    Tone::Red
                },
            ),
        ]);
        self.print_plain_line(progress.phase.as_str(), fields.join("  "));
    }

    fn print_plain_prefill_progress(&mut self, progress: HfCudaDeviceSessionChunkProgress) {
        if progress.wall_ns == 0 {
            self.print_plain_line(
                progress.phase.as_str(),
                vec![
                    paint(
                        self.color,
                        Tone::Green,
                        format!("prompt {} tokens", progress.observed),
                    ),
                    metric(self.color, "wall", "running", Tone::Yellow),
                    metric(self.color, "kv", "building", Tone::Magenta),
                ]
                .join("  "),
            );
            return;
        }
        let wall = Duration::from_nanos(progress.wall_ns.max(1));
        let rate = format::tokens_per_s(progress.observed, wall);
        self.print_plain_line(
            progress.phase.as_str(),
            vec![
                paint(
                    self.color,
                    Tone::Green,
                    format!("prompt {} tokens", progress.observed),
                ),
                metric(
                    self.color,
                    "wall",
                    format::duration(Duration::from_nanos(progress.wall_ns)),
                    Tone::Yellow,
                ),
                metric(
                    self.color,
                    "gpu",
                    format::ms_from_ns(progress.device_ns),
                    Tone::Yellow,
                ),
                metric(self.color, "rate", rate, Tone::Cyan),
                metric(
                    self.color,
                    "kv",
                    progress.kv_tokens.to_string(),
                    Tone::Magenta,
                ),
                metric(
                    self.color,
                    "kernels",
                    progress.kernel_launches.to_string(),
                    Tone::Magenta,
                ),
                metric(
                    self.color,
                    "graph",
                    progress.graph_replays.to_string(),
                    Tone::Magenta,
                ),
                metric(
                    self.color,
                    "cache",
                    progress.graph_cache_hits.to_string(),
                    Tone::Blue,
                ),
            ]
            .join("  "),
        );
    }

    fn print_plain_summary(
        &mut self,
        output: &nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput,
        elapsed: std::time::Duration,
    ) {
        let stats = DecodeStats::from_output(output);
        let prefill_profile_total_ns = output
            .stream
            .start
            .projection_ns
            .saturating_add(output.stream.start.attention_ns)
            .saturating_add(output.stream.start.mlp_ns)
            .saturating_add(output.stream.start.norm_ns)
            .saturating_add(output.stream.start.sampling_ns);
        let profile_total_ns = stats
            .projection_ns
            .saturating_add(stats.attention_ns)
            .saturating_add(stats.mlp_ns)
            .saturating_add(stats.norm_ns)
            .saturating_add(stats.sampling_ns);
        self.print_plain_line(
            "report",
            paint(self.color, Tone::Green, "final performance report"),
        );
        self.print_plain_report_block_line(paint(
            self.color,
            Tone::Green,
            "NERVA PERFORMANCE REPORT",
        ));
        self.print_plain_report_block_line(paint(self.color, Tone::Green, "RUN"));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "generated",
            format!("{} tokens", output.tokens().len()),
            Tone::Green,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "stop reason",
            output.stop_reason().as_str(),
            Tone::Yellow,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "elapsed",
            format::duration(elapsed),
            Tone::Cyan,
        ));
        self.print_plain_report_block_line("");

        self.print_plain_report_block_line(paint(self.color, Tone::Orange, "LOAD"));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "resident weights",
            format::bytes(output.stream.create.resident_weight_bytes),
            Tone::Orange,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "H2D copied",
            format::bytes(output.stream.create.h2d_bytes),
            Tone::Yellow,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "load wall",
            format::duration(Duration::from_nanos(output.stream.load_wall_ns)),
            Tone::Cyan,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "load bandwidth",
            format::gb_per_s(
                output.stream.create.h2d_bytes,
                Duration::from_nanos(output.stream.load_wall_ns.max(1)),
            ),
            Tone::Green,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "prefill chunk",
            format!("{} tokens", output.stream.create.prefill_chunk_tokens),
            Tone::Cyan,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "head threads",
            output.stream.create.head_threads.to_string(),
            Tone::Cyan,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "tensors",
            output.stream.tensors_loaded.to_string(),
            Tone::Dim,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "descriptors",
            output
                .stream
                .create
                .planned_weight_descriptor_count
                .to_string(),
            Tone::Dim,
        ));
        self.print_plain_report_block_line("");

        if prefill_profile_total_ns != 0 {
            self.print_plain_report_block_line(paint(self.color, Tone::Magenta, "PREFILL PROFILE"));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "profiled total",
                format::ms_from_ns(prefill_profile_total_ns),
                Tone::Dim,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "projection",
                output.stream.start.projection_ns,
                prefill_profile_total_ns,
                Tone::Blue,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "attention",
                output.stream.start.attention_ns,
                prefill_profile_total_ns,
                Tone::Cyan,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "norm/cache",
                output.stream.start.norm_ns,
                prefill_profile_total_ns,
                Tone::Yellow,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "mlp",
                output.stream.start.mlp_ns,
                prefill_profile_total_ns,
                Tone::Magenta,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "sample",
                output.stream.start.sampling_ns,
                prefill_profile_total_ns,
                Tone::Green,
            ));
            self.print_plain_report_block_line("");
        }

        self.print_plain_report_block_line(paint(self.color, Tone::Cyan, "DECODE"));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "throughput",
            format::tokens_per_s(stats.tokens, Duration::from_nanos(stats.wall_ns.max(1))),
            Tone::Green,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "memory bandwidth",
            format::weight_bandwidth(
                output.stream.create.resident_weight_bytes,
                stats.tokens as u64,
                Duration::from_nanos(stats.wall_ns.max(1)),
                output.stream.create.device_memory_bandwidth_bps,
            ),
            Tone::Green,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "decode chunks",
            output.stream.chunks.len().to_string(),
            Tone::Dim,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "mean latency",
            format::ms_from_ns(stats.mean_ns()),
            Tone::Cyan,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "p50 latency",
            format::ms_from_ns(stats.p50_ns),
            Tone::Green,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "p95 latency",
            format::ms_from_ns(stats.p95_ns),
            Tone::Yellow,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "p99 latency",
            format::ms_from_ns(stats.p99_ns),
            Tone::Red,
        ));
        if let Some(drift) = decode_drift(output) {
            self.print_plain_report_block_line(report_drift_line(self.color, drift));
        }
        self.print_plain_report_block_line("");

        self.print_plain_report_block_line(paint(self.color, Tone::Magenta, "TIME PROFILE"));
        if profile_total_ns == 0 {
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "detail profiler",
                "off",
                Tone::Dim,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "enable",
                "--profiling",
                Tone::Yellow,
            ));
        } else {
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "profiled total",
                format::ms_from_ns(profile_total_ns),
                Tone::Dim,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "projection",
                stats.projection_ns,
                profile_total_ns,
                Tone::Blue,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "attention",
                stats.attention_ns,
                profile_total_ns,
                Tone::Cyan,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "norm",
                stats.norm_ns,
                profile_total_ns,
                Tone::Yellow,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "mlp",
                stats.mlp_ns,
                profile_total_ns,
                Tone::Magenta,
            ));
            self.print_plain_report_block_line(report_timing_line(
                self.color,
                "sample",
                stats.sampling_ns,
                profile_total_ns,
                Tone::Green,
            ));
        }
        self.print_plain_report_block_line("");

        if stats.has_deepseek_activity() || deepseek_prefill_activity(output) {
            self.print_plain_report_block_line(paint(self.color, Tone::Orange, "DEEPSEEK"));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "prefill raw scan",
                format!(
                    "{} tokens",
                    output.stream.start.deepseek_raw_attention_tokens_scanned
                ),
                Tone::Cyan,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "decode raw scan",
                format!("{} tokens", stats.deepseek_raw_attention_tokens_scanned),
                Tone::Cyan,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "top-k selections",
                stats.deepseek_sparse_topk_selections.to_string(),
                Tone::Green,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "top-k candidates",
                stats.deepseek_sparse_topk_candidates_scored.to_string(),
                Tone::Yellow,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "top-k slots",
                stats.deepseek_sparse_topk_slots_selected.to_string(),
                Tone::Green,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "compressed reads",
                format!(
                    "{} reads / {} slots",
                    stats.deepseek_compressed_kv_attention_reads,
                    stats.deepseek_compressed_kv_attention_slots_scanned
                ),
                Tone::Blue,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "state writes",
                format!(
                    "compressor {}  indexer {}  indexer-kv {}",
                    stats.deepseek_compressor_state_writes,
                    stats.deepseek_indexer_state_writes,
                    stats.deepseek_indexer_kv_writes
                ),
                Tone::Magenta,
            ));
            self.print_plain_report_block_line(report_kv_line(
                self.color,
                "router",
                format!(
                    "v3-grouped {}  v4-bias {}  v4-hash {}",
                    stats.deepseek_v3_grouped_router_selections,
                    stats.deepseek_v4_bias_router_selections,
                    stats.deepseek_v4_hash_router_selections
                ),
                Tone::Orange,
            ));
            self.print_plain_report_block_line("");
        }

        self.print_plain_report_block_line(paint(self.color, Tone::Magenta, "CUDA GRAPH"));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "replays",
            stats.graph_replays.to_string(),
            Tone::Blue,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "cache hits",
            stats.graph_cache_hits.to_string(),
            Tone::Green,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "kernels",
            stats.kernel_launches.to_string(),
            Tone::Magenta,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "graph nodes",
            stats.graph_nodes.to_string(),
            Tone::Magenta,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "sync calls",
            stats.sync_calls.to_string(),
            Tone::Yellow,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "hot allocations",
            stats.hot_path_allocations.to_string(),
            if stats.hot_path_allocations == 0 {
                Tone::Green
            } else {
                Tone::Red
            },
        ));
        self.print_plain_report_block_line("");

        self.print_plain_report_block_line(paint(self.color, Tone::Blue, "MEMORY"));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "weights",
            format::bytes(output.stream.create.resident_weight_bytes),
            Tone::Orange,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "KV cache",
            format::bytes(output.stream.create.resident_kv_bytes),
            Tone::Blue,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "decode H2D",
            format::bytes(stats.h2d_bytes),
            Tone::Yellow,
        ));
        self.print_plain_report_block_line(report_kv_line(
            self.color,
            "decode D2H",
            format::bytes(stats.d2h_bytes),
            Tone::Cyan,
        ));
    }

    fn print_plain_line(&mut self, phase: &str, message: impl AsRef<str>) {
        self.last_plain_emit = Instant::now();
        if self.color.enabled() {
            eprintln!(
                "{}[{}]{} {}{:<8}{} {}",
                code(self.color, Tone::Dim),
                format::duration(self.state.boot.elapsed()),
                reset(self.color),
                code(self.color, phase_tone(phase)),
                phase,
                reset(self.color),
                message.as_ref()
            );
            return;
        }
        eprintln!(
            "[{}] {:<8} {}",
            format::duration(self.state.boot.elapsed()),
            phase,
            message.as_ref()
        );
    }

    fn print_plain_report_block_line(&mut self, message: impl AsRef<str>) {
        self.last_plain_emit = Instant::now();
        eprintln!("{}", message.as_ref());
    }

    fn should_emit_plain_progress(&self) -> bool {
        self.last_plain_emit.elapsed() >= PLAIN_PROGRESS_INTERVAL
    }
}

pub(crate) struct UiTickerGuard {
    active: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl UiTickerGuard {
    fn start(inner: Arc<Mutex<NervaCliLoggerInner>>) -> Self {
        let active = Arc::new(AtomicBool::new(
            inner
                .lock()
                .map(|inner| inner.mode != LoggerMode::Quiet)
                .unwrap_or(false),
        ));
        let handle = if active.load(Ordering::Relaxed) {
            let active_thread = Arc::clone(&active);
            Some(thread::spawn(move || {
                while active_thread.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(250));
                    let Ok(mut inner) = inner.lock() else {
                        break;
                    };
                    if inner.state.phase == "complete" {
                        continue;
                    }
                    inner.tick_or_log();
                }
            }))
        } else {
            None
        };
        Self { active, handle }
    }
}

impl Drop for UiTickerGuard {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn terminal_width_hint() -> u16 {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(120)
}

fn should_draw_debug_progress(progress: &HfCudaDeviceSessionChunkProgress) -> bool {
    progress.phase == HfCudaDeviceProgressPhase::Prefill
        || progress.hit_stop
        || progress.generated == progress.requested
        || progress.generated <= 8
        || progress.generated.is_multiple_of(16)
}

pub(crate) struct NativeLoadProgressGuard {
    previous: Option<OsString>,
    installed: bool,
}

impl NativeLoadProgressGuard {
    fn install(mode: &'static str) -> Self {
        let previous = std::env::var_os("NERVA_NATIVE_LOAD_PROGRESS");
        unsafe {
            std::env::set_var("NERVA_NATIVE_LOAD_PROGRESS", mode);
        }
        Self {
            previous,
            installed: true,
        }
    }
}

impl Drop for NativeLoadProgressGuard {
    fn drop(&mut self) {
        if !self.installed {
            return;
        }
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var("NERVA_NATIVE_LOAD_PROGRESS", previous);
            } else {
                std::env::remove_var("NERVA_NATIVE_LOAD_PROGRESS");
            }
        }
    }
}

fn phase_tone(phase: &str) -> Tone {
    match phase {
        "decode" | "prefill" | "complete" | "perf" | "report" => Tone::Green,
        "load" | "boot" | "version" => Tone::Orange,
        "request" | "runtime" => Tone::Cyan,
        "time" | "drift" => Tone::Yellow,
        "graph" => Tone::Magenta,
        "memory" => Tone::Blue,
        _ => Tone::Dim,
    }
}

const REPORT_LABEL_WIDTH: usize = 18;
const REPORT_BAR_WIDTH: usize = 32;

#[derive(Clone, Copy, Debug)]
struct DecodeDrift {
    first_ns: u64,
    last_ns: u64,
    delta: f64,
    tone: Tone,
}

fn metric(color: ColorMode, label: &str, value: impl AsRef<str>, value_tone: Tone) -> String {
    if color.enabled() {
        format!(
            "{}{}{} {}{}{}",
            code(color, Tone::Dim),
            label,
            reset(color),
            code(color, value_tone),
            value.as_ref(),
            reset(color)
        )
    } else {
        format!("{} {}", label, value.as_ref())
    }
}

fn report_kv_line(
    color: ColorMode,
    label: &str,
    value: impl AsRef<str>,
    value_tone: Tone,
) -> String {
    let label = format!("    {:<width$}", label, width = REPORT_LABEL_WIDTH);
    format!(
        "{} {}",
        paint(color, Tone::Dim, label),
        paint(color, value_tone, value.as_ref())
    )
}

fn report_timing_line(color: ColorMode, label: &str, ns: u64, total_ns: u64, tone: Tone) -> String {
    let ratio = ratio(ns, total_ns);
    let label = format!("    {:<width$}", label, width = REPORT_LABEL_WIDTH);
    let value = format!("{:>12}", format::ms_from_ns(ns));
    let pct = format!("{:>6.1}%", ratio * 100.0);
    format!(
        "{} {}  {}  {}",
        paint(color, Tone::Dim, label),
        paint(color, tone, value),
        paint(color, tone, pct),
        report_bar(color, ratio, tone)
    )
}

fn report_drift_line(color: ColorMode, drift: DecodeDrift) -> String {
    let label = format!("    {:<width$}", "decode drift", width = REPORT_LABEL_WIDTH);
    let delta = format!("{:+.2}%", drift.delta);
    let span = format!(
        "{} -> {}",
        format::ms_from_ns(drift.first_ns),
        format::ms_from_ns(drift.last_ns)
    );
    let status = if drift.delta > 10.0 {
        "critical"
    } else if drift.delta > 2.0 {
        "watch"
    } else {
        "stable"
    };
    format!(
        "{} {}  {}  {}",
        paint(color, Tone::Dim, label),
        paint(color, drift.tone, delta),
        paint(color, Tone::Dim, span),
        paint(color, drift.tone, status)
    )
}

fn report_bar(color: ColorMode, ratio: f64, tone: Tone) -> String {
    let ratio = if ratio.is_finite() {
        ratio.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = if ratio > 0.0 {
        ((ratio * REPORT_BAR_WIDTH as f64).round() as usize)
            .max(1)
            .min(REPORT_BAR_WIDTH)
    } else {
        0
    };
    let empty = REPORT_BAR_WIDTH.saturating_sub(filled);
    let filled_bar = "#".repeat(filled);
    let empty_bar = ".".repeat(empty);
    if color.enabled() {
        format!(
            "{}{}{}{}{}{}",
            code(color, tone),
            filled_bar,
            reset(color),
            code(color, Tone::Dim),
            empty_bar,
            reset(color)
        )
    } else {
        format!("{filled_bar}{empty_bar}")
    }
}

fn ratio(value: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64 / total as f64
    }
}

fn decode_drift(
    output: &nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput,
) -> Option<DecodeDrift> {
    let latencies = output
        .stream
        .chunks
        .iter()
        .flat_map(|chunk| chunk.critical_paths.iter())
        .map(|path| path.wall_latency_ns)
        .collect::<Vec<_>>();
    if latencies.len() < 8 {
        return None;
    }
    let steady = if latencies.len() > 32 {
        &latencies[1..]
    } else {
        latencies.as_slice()
    };
    if steady.is_empty() {
        return None;
    }
    let window = steady.len().min(16);
    let first = mean_ns(&steady[..window]);
    let last = mean_ns(&steady[steady.len() - window..]);
    let delta = if first == 0 {
        0.0
    } else {
        (last as f64 - first as f64) * 100.0 / first as f64
    };
    let tone_for_delta = if delta > 10.0 {
        Tone::Red
    } else if delta > 2.0 {
        Tone::Yellow
    } else {
        Tone::Green
    };
    Some(DecodeDrift {
        first_ns: first,
        last_ns: last,
        delta,
        tone: tone_for_delta,
    })
}

fn deepseek_prefill_activity(
    output: &nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput,
) -> bool {
    let summary = &output.stream.start;
    summary.deepseek_compressor_state_writes != 0
        || summary.deepseek_compressed_kv_writes != 0
        || summary.deepseek_indexer_state_writes != 0
        || summary.deepseek_indexer_kv_writes != 0
        || summary.deepseek_compressed_kv_attention_reads != 0
        || summary.deepseek_compressed_kv_attention_slots_scanned != 0
        || summary.deepseek_sparse_topk_selections != 0
        || summary.deepseek_sparse_topk_slots_selected != 0
        || summary.deepseek_sparse_topk_candidates_scored != 0
        || summary.deepseek_v3_grouped_router_selections != 0
        || summary.deepseek_v4_bias_router_selections != 0
        || summary.deepseek_v4_hash_router_selections != 0
        || summary.deepseek_raw_attention_tokens_scanned != 0
}

fn mean_ns(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.iter().sum::<u64>() / values.len() as u64
}
