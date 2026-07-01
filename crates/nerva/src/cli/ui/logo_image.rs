use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use crate::cli::ui::color::ColorMode;

const BIG_N_LINES: [&str; 7] = [
    "███╗   ██╗",
    "████╗  ██║",
    "██╔██╗ ██║",
    "██║╚██╗██║",
    "██║ ╚████║",
    "██║  ╚███║",
    "╚═╝   ╚══╝",
];
const SMALL_ERVA_LINES: [&str; 5] = [
    "███████╗██████╗ ██╗   ██╗ █████╗",
    "██╔════╝██╔══██╗██║   ██║██╔══██╗",
    "█████╗  ██████╔╝██║   ██║███████║",
    "██╔══╝  ██╔══██╗╚██╗ ██╔╝██╔══██║",
    "███████╗██║  ██║ ╚████╔╝ ██║  ██║",
];
const TAGLINE_PRIMARY: &str = "An inference operating system for large models";
const TAGLINE_SECONDARY: &str = "AI inference beyond the VRAM wall";
const ANSI_RESET: &str = "\x1b[0m";
const LOGO_GAP: &str = " ";
const LOGO_ORANGE: (u8, u8, u8) = (255, 106, 42);

#[derive(Clone, Debug)]
pub(crate) struct TerminalLogo {
    pub(crate) lines: Vec<Line<'static>>,
}

impl TerminalLogo {
    pub(crate) fn load(_target_width_cells: u16) -> Option<Self> {
        Some(Self {
            lines: block_logo(),
        })
    }
}

pub(crate) fn block_logo() -> Vec<Line<'static>> {
    logo_rows()
        .into_iter()
        .map(|(n, erva)| logo_line(&n, &erva))
        .collect()
}

pub(crate) fn tagline_lines() -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(TAGLINE_PRIMARY, muted_style())),
        Line::from(Span::styled(TAGLINE_SECONDARY, muted_style())),
    ]
}

pub(crate) fn plain_brand(width: u16, color: ColorMode) -> Vec<String> {
    let mut lines = logo_rows()
        .into_iter()
        .map(|(n, erva)| {
            let visible_width = n.chars().count() + LOGO_GAP.chars().count() + erva.chars().count();
            center_line(&plain_logo_line(&n, &erva, color), visible_width, width)
        })
        .collect::<Vec<_>>();
    lines.push(String::new());
    lines.push(center_line(
        TAGLINE_PRIMARY,
        TAGLINE_PRIMARY.chars().count(),
        width,
    ));
    lines.push(center_line(
        TAGLINE_SECONDARY,
        TAGLINE_SECONDARY.chars().count(),
        width,
    ));
    lines
}

pub(crate) fn wordmark() -> Line<'static> {
    Line::from(vec![
        Span::styled("NERVA", gradient_style(0, 5)),
        Span::raw("  "),
        Span::styled("Neural Execution & Residency VM", muted_style()),
    ])
}

fn logo_line(n: &str, erva: &str) -> Line<'static> {
    let mut spans = Vec::new();
    spans.extend(flat_orange_spans(n));
    spans.push(Span::raw(LOGO_GAP));
    spans.extend(gradient_spans(erva));
    Line::from(spans)
}

fn logo_rows() -> Vec<(String, String)> {
    let n_width = max_width(&BIG_N_LINES);
    let erva_width = max_width(&SMALL_ERVA_LINES);
    BIG_N_LINES
        .iter()
        .enumerate()
        .map(|(row, n)| {
            let erva = row
                .checked_sub(1)
                .and_then(|index| SMALL_ERVA_LINES.get(index))
                .copied()
                .unwrap_or("");
            (
                format!("{n:<n_width$}", n = *n, n_width = n_width),
                format!("{erva:<erva_width$}", erva = erva, erva_width = erva_width),
            )
        })
        .collect()
}

fn max_width<const N: usize>(lines: &[&str; N]) -> usize {
    lines
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0)
}

fn gradient_style(index: usize, width: usize) -> Style {
    let (red, green, blue) = gradient_rgb(index, width);
    Style::default()
        .fg(Color::Rgb(red, green, blue))
        .add_modifier(Modifier::BOLD)
}

fn flat_orange_spans(value: &str) -> Vec<Span<'static>> {
    value
        .chars()
        .map(|character| {
            if character.is_whitespace() {
                Span::raw(character.to_string())
            } else {
                Span::styled(
                    character.to_string(),
                    Style::default()
                        .fg(Color::Rgb(LOGO_ORANGE.0, LOGO_ORANGE.1, LOGO_ORANGE.2))
                        .add_modifier(Modifier::BOLD),
                )
            }
        })
        .collect()
}

fn gradient_spans(value: &str) -> Vec<Span<'static>> {
    let width = value
        .chars()
        .filter(|character| !character.is_whitespace())
        .count()
        .max(1);
    let mut index = 0;
    value
        .chars()
        .map(|character| {
            if character.is_whitespace() {
                Span::raw(character.to_string())
            } else {
                let span = Span::styled(character.to_string(), gradient_style(index, width));
                index += 1;
                span
            }
        })
        .collect()
}

fn plain_logo_line(n: &str, erva: &str, color: ColorMode) -> String {
    let mut output = String::new();
    output.push_str(&plain_orange_segment(n, color));
    output.push_str(LOGO_GAP);
    output.push_str(&plain_gradient_segment(erva.trim_end(), color));
    output
}

fn plain_orange_segment(value: &str, color: ColorMode) -> String {
    if !color.enabled() {
        return value.to_string();
    }
    let mut output = String::new();
    let mut colored = false;
    for character in value.chars() {
        if character.is_whitespace() {
            if colored {
                output.push_str(ANSI_RESET);
                colored = false;
            }
            output.push(character);
            continue;
        }
        if color.truecolor() {
            output.push_str(&format!(
                "\x1b[38;2;{};{};{}m{}",
                LOGO_ORANGE.0, LOGO_ORANGE.1, LOGO_ORANGE.2, character
            ));
        } else {
            output.push_str("\x1b[93m");
            output.push(character);
        }
        colored = true;
    }
    if colored {
        output.push_str(ANSI_RESET);
    }
    output
}

fn plain_gradient_segment(value: &str, color: ColorMode) -> String {
    if !color.enabled() {
        return value.to_string();
    }
    let width = value
        .chars()
        .filter(|character| !character.is_whitespace())
        .count()
        .max(1);
    let mut output = String::new();
    let mut colored = false;
    let mut index = 0;
    for character in value.chars() {
        if character.is_whitespace() {
            if colored {
                output.push_str(ANSI_RESET);
                colored = false;
            }
            output.push(character);
            continue;
        }
        if color.truecolor() {
            let (red, green, blue) = gradient_rgb(index, width);
            output.push_str(&format!("\x1b[38;2;{red};{green};{blue}m{character}"));
        } else {
            output.push_str(ansi_gradient_code(index, width));
            output.push(character);
        }
        index += 1;
        colored = true;
    }
    if colored {
        output.push_str(ANSI_RESET);
    }
    output
}

fn ansi_gradient_code(index: usize, width: usize) -> &'static str {
    let denominator = width.saturating_sub(1).max(1) as f32;
    let ratio = index as f32 / denominator;
    if ratio < 0.34 {
        "\x1b[93m"
    } else if ratio < 0.67 {
        "\x1b[97m"
    } else {
        "\x1b[96m"
    }
}

fn gradient_rgb(index: usize, width: usize) -> (u8, u8, u8) {
    let denominator = width.saturating_sub(1).max(1) as f32;
    let ratio = index as f32 / denominator;
    let lerp = |start: u8, end: u8| start as f32 + (end as f32 - start as f32) * ratio;
    (
        lerp(255, 255) as u8,
        lerp(106, 255) as u8,
        lerp(42, 255) as u8,
    )
}

fn center_line(value: &str, visible_width: usize, terminal_width: u16) -> String {
    let padding = (terminal_width as usize).saturating_sub(visible_width) / 2;
    format!("{}{}", " ".repeat(padding), value)
}

fn muted_style() -> Style {
    Style::default().fg(Color::Rgb(150, 158, 166))
}
