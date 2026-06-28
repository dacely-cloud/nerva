use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Wrap};

use crate::cli::ui::logo_image::{TerminalLogo, block_logo, tagline_lines, wordmark};
use crate::cli::ui::render_text::{
    accent, border, dim, load_bar_lines, log_lines, muted, ok, output_lines, panel,
    performance_lines, progress_label, summary_lines, title,
};
use crate::cli::ui::state::UiState;

pub(crate) fn render(frame: &mut Frame<'_>, state: &mut UiState) {
    let area = frame.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
            Constraint::Length(2),
            Constraint::Length(7),
            Constraint::Min(10),
            Constraint::Length(2),
        ])
        .split(area);

    render_brand(frame, rows[0], state.logo.as_ref());
    render_status_strip(frame, rows[1], state);
    render_generation(frame, rows[2], state);
    render_workspace(frame, rows[3], state);
    render_footer(frame, rows[4], state);
}

fn render_brand(frame: &mut Frame<'_>, area: Rect, logo: Option<&TerminalLogo>) {
    let mut lines = logo
        .map(|logo| logo.lines.clone())
        .unwrap_or_else(block_logo);
    let display = if area.width >= 52 && area.height >= 10 {
        lines.push(Line::from(""));
        lines.extend(tagline_lines());
        lines
    } else {
        vec![wordmark()]
    };
    frame.render_widget(
        Paragraph::new(display)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Rgb(255, 106, 42))),
        inset(area, 0, 0),
    );
}

fn render_status_strip(frame: &mut Frame<'_>, area: Rect, state: &UiState) {
    let line = Line::from(vec![
        Span::styled("phase ", muted()),
        Span::styled(state.phase, accent()),
        Span::raw("  "),
        Span::styled(state.title.clone(), title()),
        Span::raw("  "),
        Span::styled("model ", muted()),
        Span::raw(compact(&state.model, 42)),
        Span::raw("  "),
        Span::styled("device ", muted()),
        Span::raw(if state.compute.is_empty() {
            "pending".to_string()
        } else {
            state.compute.clone()
        }),
        Span::raw("  "),
        Span::styled("runtime ", muted()),
        Span::raw(crate::cli::ui::format::duration(state.boot.elapsed())),
    ]);
    frame.render_widget(
        Paragraph::new(line)
            .alignment(Alignment::Left)
            .block(top_rule()),
        inset(area, 1, 0),
    );
}

fn render_generation(frame: &mut Frame<'_>, area: Rect, state: &UiState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .spacing(1)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(inset(area, 1, 0));

    if state.progress.is_none() {
        render_panel_lines(frame, cols[0], "loading", load_bar_lines(state), false);
    } else {
        let (label, ratio) = progress_label(state);
        frame.render_widget(
            Gauge::default()
                .block(panel("generation"))
                .gauge_style(
                    Style::default()
                        .fg(Color::Rgb(255, 106, 42))
                        .bg(Color::Rgb(31, 36, 42))
                        .add_modifier(Modifier::BOLD),
                )
                .label(label)
                .ratio(ratio),
            cols[0],
        );
    }
    render_token_graph(frame, cols[1], state);
}

fn render_token_graph(frame: &mut Frame<'_>, area: Rect, state: &UiState) {
    let max = state.decode_samples.iter().copied().max().unwrap_or(1);
    if state.decode_samples.is_empty() {
        render_panel_lines(
            frame,
            area,
            "tokens / second",
            vec![
                Line::from(Span::styled("waiting for decode samples", muted())),
                Line::from(Span::styled("graph starts on first token", dim())),
            ],
            false,
        );
        return;
    }
    frame.render_widget(
        Sparkline::default()
            .block(panel("tokens / second"))
            .data(&state.decode_samples)
            .style(ok())
            .max(max),
        area,
    );
}

fn render_workspace(frame: &mut Frame<'_>, area: Rect, state: &UiState) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .spacing(1)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(inset(area, 1, 0));

    render_panel_lines(frame, cols[0], "logs", log_lines(state), true);

    let mut perf = performance_lines(state);
    if !state.summary.is_empty() {
        perf.push(Line::from(""));
        perf.push(Line::from(Span::styled("summary", accent())));
        perf.extend(summary_lines(state));
    }
    render_panel_lines(frame, cols[1], "performance", perf, false);
    render_panel_lines(frame, cols[2], "output", output_lines(state), false);
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, state: &UiState) {
    let line = Line::from(vec![
        Span::styled("NERVA", accent()),
        Span::raw("  "),
        Span::styled(state.phase, muted()),
        Span::raw("  "),
        Span::raw(state.title.clone()),
    ]);
    frame.render_widget(
        Paragraph::new(line)
            .alignment(Alignment::Left)
            .block(bottom_rule()),
        inset(area, 1, 0),
    );
}

fn render_panel_lines(
    frame: &mut Frame<'_>,
    area: Rect,
    panel_title: &'static str,
    lines: Vec<Line<'static>>,
    tail: bool,
) {
    let block = panel(panel_title);
    let max_lines = block.inner(area).height as usize;
    let lines = if tail {
        tail_lines(lines, max_lines)
    } else {
        head_lines(lines, max_lines)
    };
    frame.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        area,
    );
}

fn top_rule() -> Block<'static> {
    Block::default()
        .borders(Borders::TOP)
        .border_style(border())
}

fn bottom_rule() -> Block<'static> {
    Block::default()
        .borders(Borders::TOP)
        .border_style(border())
}

fn head_lines(mut lines: Vec<Line<'static>>, max_lines: usize) -> Vec<Line<'static>> {
    if max_lines == 0 {
        return Vec::new();
    }
    lines.truncate(max_lines);
    lines
}

fn tail_lines(mut lines: Vec<Line<'static>>, max_lines: usize) -> Vec<Line<'static>> {
    if max_lines == 0 {
        return Vec::new();
    }
    if lines.len() > max_lines {
        lines.drain(0..lines.len() - max_lines);
    }
    lines
}

fn compact(value: &str, max: usize) -> String {
    if value.chars().count() <= max {
        return value.to_string();
    }
    let keep = max.saturating_sub(3);
    let tail = value
        .chars()
        .rev()
        .take(keep)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("...{tail}")
}

fn inset(area: Rect, horizontal: u16, vertical: u16) -> Rect {
    Rect {
        x: area.x.saturating_add(horizontal),
        y: area.y.saturating_add(vertical),
        width: area.width.saturating_sub(horizontal.saturating_mul(2)),
        height: area.height.saturating_sub(vertical.saturating_mul(2)),
    }
}
