use std::time::Duration;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

use crate::cli::ui::format;
use crate::cli::ui::state::UiState;

const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

pub(crate) fn performance_lines(state: &UiState) -> Vec<Line<'static>> {
    let Some(progress) = &state.progress else {
        return vec![
            kv("load", "resident weight upload and warmup"),
            kv("prefill", "waiting for prompt transaction"),
            kv("decode", "waiting for first token"),
            kv("policy", "device-first token causality"),
            kv("hot alloc", current_hot_alloc(state)),
        ];
    };
    vec![
        kv("last token", format::ms_from_ns(progress.wall_ns)),
        kv("gpu", format::ms_from_ns(progress.device_ns)),
        kv("projection", format::ms_from_ns(progress.projection_ns)),
        kv("attention", format::ms_from_ns(progress.attention_ns)),
        kv("mlp", format::ms_from_ns(progress.mlp_ns)),
        kv("kernels", progress.kernel_launches.to_string()),
        kv("graph nodes", progress.graph_nodes.to_string()),
        kv("sync calls", progress.sync_calls.to_string()),
        kv("hot alloc", current_hot_alloc(state)),
    ]
}

pub(crate) fn summary_lines(state: &UiState) -> Vec<Line<'static>> {
    state
        .summary
        .iter()
        .map(|(label, value)| kv(label, value.clone()))
        .collect()
}

pub(crate) fn log_lines(state: &UiState) -> Vec<Line<'static>> {
    state
        .logs
        .iter()
        .map(|value| Line::from(Span::raw(value.clone())))
        .collect()
}

pub(crate) fn output_lines(state: &UiState) -> Vec<Line<'static>> {
    if state.generated_text.is_empty() {
        return vec![
            Line::from(Span::styled("waiting for generated text...", muted())),
            Line::from(""),
        ];
    }
    state
        .generated_text
        .lines()
        .map(|value| Line::from(Span::raw(value.to_string())))
        .collect()
}

pub(crate) fn progress_label(state: &UiState) -> (String, f64) {
    let Some(progress) = &state.progress else {
        return (
            format!("{} loading resident model", spinner(state)),
            loading_ratio(state),
        );
    };
    let ratio = if progress.requested == 0 {
        0.0
    } else {
        progress.generated as f64 / progress.requested as f64
    };
    let elapsed = Duration::from_nanos(state.decode_elapsed_ns.max(1));
    let rate = format::tokens_per_s(progress.generated, elapsed);
    (
        format!("{} / {}   {}", progress.generated, progress.requested, rate),
        ratio.clamp(0.0, 1.0),
    )
}

pub(crate) fn load_bar_lines(state: &UiState) -> Vec<Line<'static>> {
    let width = 24usize;
    let head = state.spinner % width;
    let mut spans = Vec::with_capacity(width + 1);
    for index in 0..width {
        let bright = index == head || index == (head + width - 1) % width;
        spans.push(Span::styled(
            if bright { "█" } else { "░" },
            if bright { accent() } else { dim() },
        ));
    }
    vec![
        Line::from(vec![
            Span::styled(spinner(state), accent()),
            Span::raw("  loading model into resident arenas"),
        ]),
        Line::from(spans),
        Line::from(Span::styled(
            "waiting for first decode progress event",
            muted(),
        )),
    ]
}

pub(crate) fn current_hot_alloc(state: &UiState) -> String {
    state
        .progress
        .as_ref()
        .map(|progress| progress.hot_path_allocations.to_string())
        .unwrap_or_else(|| "0".to_string())
}

pub(crate) fn panel(title: &'static str) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(format!(" {title} "))
        .border_style(border())
        .style(Style::default().fg(Color::Rgb(207, 214, 222)))
}

pub(crate) fn accent() -> Style {
    Style::default()
        .fg(Color::Rgb(255, 106, 42))
        .add_modifier(Modifier::BOLD)
}

pub(crate) fn title() -> Style {
    Style::default()
        .fg(Color::Rgb(245, 247, 250))
        .add_modifier(Modifier::BOLD)
}

pub(crate) fn label() -> Style {
    Style::default().fg(Color::Rgb(122, 132, 142))
}

pub(crate) fn ok() -> Style {
    Style::default().fg(Color::Rgb(112, 223, 158))
}

pub(crate) fn muted() -> Style {
    Style::default().fg(Color::Rgb(150, 158, 166))
}

pub(crate) fn dim() -> Style {
    Style::default().fg(Color::Rgb(58, 66, 75))
}

pub(crate) fn border() -> Style {
    Style::default().fg(Color::Rgb(66, 74, 84))
}

fn kv(label_text: impl Into<String>, value: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{:<10}", label_text.into()), label()),
        Span::raw(" "),
        Span::raw(value.into()),
    ])
}

fn spinner(state: &UiState) -> &'static str {
    if state.phase == "complete" {
        "✓"
    } else {
        SPINNER[state.spinner % SPINNER.len()]
    }
}

fn loading_ratio(state: &UiState) -> f64 {
    0.08 + ((state.spinner % 20) as f64 / 20.0) * 0.84
}
