use std::io::IsTerminal;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ColorMode {
    None,
    Ansi16,
    Truecolor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Tone {
    Dim,
    Orange,
    Green,
    Cyan,
    Yellow,
    Magenta,
    Blue,
    Red,
}

impl ColorMode {
    pub(crate) const fn enabled(self) -> bool {
        !matches!(self, Self::None)
    }

    pub(crate) const fn truecolor(self) -> bool {
        matches!(self, Self::Truecolor)
    }
}

pub(crate) fn stderr_color_mode() -> ColorMode {
    detect_color_mode(std::io::stderr().is_terminal())
}

pub(crate) fn paint(mode: ColorMode, tone: Tone, value: impl AsRef<str>) -> String {
    if !mode.enabled() {
        return value.as_ref().to_string();
    }
    format!("{}{}{}", code(mode, tone), value.as_ref(), reset(mode))
}

pub(crate) fn code(mode: ColorMode, tone: Tone) -> &'static str {
    match mode {
        ColorMode::None => "",
        ColorMode::Ansi16 => match tone {
            Tone::Dim => "\x1b[2m",
            Tone::Orange => "\x1b[93m",
            Tone::Green => "\x1b[92m",
            Tone::Cyan => "\x1b[96m",
            Tone::Yellow => "\x1b[93m",
            Tone::Magenta => "\x1b[95m",
            Tone::Blue => "\x1b[94m",
            Tone::Red => "\x1b[91m",
        },
        ColorMode::Truecolor => match tone {
            Tone::Dim => "\x1b[2m",
            Tone::Orange => "\x1b[38;2;255;106;42m",
            Tone::Green => "\x1b[38;2;112;223;158m",
            Tone::Cyan => "\x1b[38;2;87;190;255m",
            Tone::Yellow => "\x1b[38;2;255;212;102m",
            Tone::Magenta => "\x1b[38;2;203;146;255m",
            Tone::Blue => "\x1b[38;2;132;170;255m",
            Tone::Red => "\x1b[38;2;255;92;92m",
        },
    }
}

pub(crate) fn reset(mode: ColorMode) -> &'static str {
    if mode.enabled() { "\x1b[0m" } else { "" }
}

fn detect_color_mode(is_terminal: bool) -> ColorMode {
    match std::env::var("NERVA_COLOR").ok().as_deref() {
        Some("never" | "none" | "off" | "0") => return ColorMode::None,
        Some("ansi" | "16" | "basic") => return ColorMode::Ansi16,
        Some("truecolor" | "24bit" | "full") => return ColorMode::Truecolor,
        Some("always" | "on" | "1") => return forced_color_mode(),
        _ => {}
    }

    if std::env::var_os("NO_COLOR").is_some() {
        return ColorMode::None;
    }
    if force_color_requested() {
        return forced_color_mode();
    }
    if !is_terminal {
        return ColorMode::None;
    }
    terminal_color_mode()
}

fn force_color_requested() -> bool {
    std::env::var("FORCE_COLOR")
        .map(|value| value != "0" && !value.is_empty())
        .unwrap_or(false)
        || std::env::var("CLICOLOR_FORCE")
            .map(|value| value != "0" && !value.is_empty())
            .unwrap_or(false)
}

fn forced_color_mode() -> ColorMode {
    ColorMode::Truecolor
}

fn terminal_color_mode() -> ColorMode {
    if std::env::var("TERM")
        .map(|term| term == "dumb")
        .unwrap_or(false)
    {
        return ColorMode::None;
    }
    ColorMode::Truecolor
}
