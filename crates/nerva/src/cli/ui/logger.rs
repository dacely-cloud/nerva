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

    pub(crate) fn native_load_progress_guard(&self) -> NativeLoadProgressGuard {
        NativeLoadProgressGuard::install(self.native_load_progress_mode())
    }

    pub(crate) fn ticker_guard(&self) -> UiTickerGuard {
        UiTickerGuard::start(Arc::clone(&self.inner))
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
        self.print_plain_line(
            progress.phase.as_str(),
            vec![
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
            ]
            .join(" "),
        );
    }

    fn print_plain_prefill_progress(&mut self, progress: HfCudaDeviceSessionChunkProgress) {
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
            .join(" "),
        );
    }

    fn print_plain_summary(
        &mut self,
        output: &nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput,
        elapsed: std::time::Duration,
    ) {
        let stats = DecodeStats::from_output(output);
        self.print_plain_line(
            "load",
            format!(
                "weights {} h2d {} load {} {} tensors {} descriptors {}",
                format::bytes(output.stream.create.resident_weight_bytes),
                format::bytes(output.stream.create.h2d_bytes),
                format::duration(Duration::from_nanos(output.stream.load_wall_ns)),
                format::gb_per_s(
                    output.stream.create.h2d_bytes,
                    Duration::from_nanos(output.stream.load_wall_ns.max(1))
                ),
                output.stream.tensors_loaded,
                output.stream.create.planned_weight_descriptor_count
            ),
        );
        self.print_plain_line(
            "complete",
            format!(
                "generated {} tokens stop {} elapsed {}",
                output.tokens().len(),
                output.stop_reason().as_str(),
                format::duration(elapsed)
            ),
        );
        self.print_plain_line(
            "perf",
            format!(
                "decode {} mean {} p50 {} p95 {} p99 {}",
                format::tokens_per_s(stats.tokens, Duration::from_nanos(stats.wall_ns.max(1))),
                format::ms_from_ns(stats.mean_ns()),
                format::ms_from_ns(stats.p50_ns),
                format::ms_from_ns(stats.p95_ns),
                format::ms_from_ns(stats.p99_ns)
            ),
        );
        if let Some(drift) = decode_drift_line(output, self.color) {
            self.print_plain_line("drift", drift);
        }
        self.print_plain_line(
            "time",
            vec![
                metric(
                    self.color,
                    "proj",
                    format::ms_from_ns(stats.projection_ns),
                    Tone::Blue,
                ),
                metric(
                    self.color,
                    "attn",
                    format::ms_from_ns(stats.attention_ns),
                    Tone::Cyan,
                ),
                metric(
                    self.color,
                    "mlp",
                    format::ms_from_ns(stats.mlp_ns),
                    Tone::Magenta,
                ),
                metric(
                    self.color,
                    "norm",
                    format::ms_from_ns(stats.norm_ns),
                    Tone::Yellow,
                ),
                metric(
                    self.color,
                    "sample",
                    format::ms_from_ns(stats.sampling_ns),
                    Tone::Green,
                ),
            ]
            .join(" "),
        );
        self.print_plain_line(
            "graph",
            format!(
                "kernels {} nodes {} replays {} cache_hits {} sync_calls {} hot_alloc {}",
                stats.kernel_launches,
                stats.graph_nodes,
                stats.graph_replays,
                stats.graph_cache_hits,
                stats.sync_calls,
                stats.hot_path_allocations
            ),
        );
        self.print_plain_line(
            "memory",
            format!(
                "weights {} kv {} decode_h2d {} d2h {}",
                format::bytes(output.stream.create.resident_weight_bytes),
                format::bytes(output.stream.create.resident_kv_bytes),
                format::bytes(stats.h2d_bytes),
                format::bytes(stats.d2h_bytes)
            ),
        );
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
        "decode" | "prefill" | "complete" | "perf" => Tone::Green,
        "load" | "boot" | "version" => Tone::Orange,
        "request" | "runtime" => Tone::Cyan,
        "time" | "drift" => Tone::Yellow,
        "graph" => Tone::Magenta,
        "memory" => Tone::Blue,
        _ => Tone::Dim,
    }
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

fn decode_drift_line(
    output: &nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput,
    color: ColorMode,
) -> Option<String> {
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
    Some(
        vec![
            metric(color, "first", format::ms_from_ns(first), Tone::Green),
            metric(color, "last", format::ms_from_ns(last), tone_for_delta),
            metric(color, "delta", format!("{delta:+.2}%"), tone_for_delta),
        ]
        .join(" "),
    )
}

fn mean_ns(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    values.iter().sum::<u64>() / values.len() as u64
}
