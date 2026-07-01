use std::io::{self, IsTerminal, Stderr};
#[cfg(unix)]
use std::os::raw::{c_int, c_void};
#[cfg(unix)]
use std::sync::Once;

use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::execute;
use crossterm::terminal::size as terminal_size;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::{Terminal, TerminalOptions, Viewport};

use crate::cli::ui::render;
use crate::cli::ui::state::UiState;

#[cfg(unix)]
const TERMINAL_CLEANUP: &[u8] = b"\x1b[0m\x1b[?25h\r\n";

pub(crate) struct TuiSession {
    terminal: Terminal<CrosstermBackend<Stderr>>,
}

impl TuiSession {
    pub(crate) fn start() -> Option<Self> {
        if !io::stderr().is_terminal() {
            return None;
        }
        let (width, height) = current_terminal_size();
        let viewport = Viewport::Fixed(Rect::new(0, 0, width, height));
        let backend = CrosstermBackend::new(io::stderr());
        let mut terminal = Terminal::with_options(backend, TerminalOptions { viewport }).ok()?;
        execute!(terminal.backend_mut(), Hide, MoveTo(0, 0)).ok()?;
        Some(Self { terminal })
    }

    pub(crate) fn draw(&mut self, state: &mut UiState) {
        self.resize_to_terminal();
        let _ = self.terminal.draw(|frame| render::render(frame, state));
    }

    fn resize_to_terminal(&mut self) {
        let (width, height) = current_terminal_size();
        let _ = self.terminal.resize(Rect::new(0, 0, width, height));
    }
}

impl Drop for TuiSession {
    fn drop(&mut self) {
        let backend = self.terminal.backend_mut();
        let _ = execute!(backend, Show);
        eprintln!();
    }
}

#[cfg(unix)]
pub(crate) fn install_signal_cleanup() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| unsafe {
        let _ = signal(SIGINT, restore_terminal_and_exit);
        let _ = signal(SIGTERM, restore_terminal_and_exit);
    });
}

fn current_terminal_size() -> (u16, u16) {
    terminal_size().unwrap_or_else(|_| (terminal_width_hint(), terminal_height_hint()))
}

fn terminal_width_hint() -> u16 {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(160)
}

fn terminal_height_hint() -> u16 {
    std::env::var("LINES")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(48)
}

#[cfg(unix)]
const SIGINT: c_int = 2;
#[cfg(unix)]
const SIGTERM: c_int = 15;
#[cfg(unix)]
const STDERR_FILENO: c_int = 2;

#[cfg(unix)]
extern "C" fn restore_terminal_and_exit(signal_number: c_int) {
    unsafe {
        let _ = write(
            STDERR_FILENO,
            TERMINAL_CLEANUP.as_ptr().cast::<c_void>(),
            TERMINAL_CLEANUP.len(),
        );
        _exit(128 + signal_number);
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn signal(signum: c_int, handler: extern "C" fn(c_int)) -> extern "C" fn(c_int);
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
    fn _exit(status: c_int) -> !;
}
