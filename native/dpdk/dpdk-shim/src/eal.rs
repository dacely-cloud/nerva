//! DPDK Environment Abstraction Layer init/teardown.
//!
//! `Eal::init` is called exactly once per process. We use a `Once`
//! guard to make repeat calls a programming error: DPDK itself panics
//! the process if it sees a second `rte_eal_init`.

use std::ffi::{CString, c_char};
use std::os::raw::c_int;
use std::ptr;
use std::sync::Once;

use crate::{Error, Result, ffi};

/// Arguments to pass to `rte_eal_init`. The first element is treated
/// as `argv[0]` (the program name); the rest are EAL flags. Build with
/// [`EalArgs::new`] then [`EalArgs::push`].
pub struct EalArgs {
    args: Vec<CString>,
}

impl EalArgs {
    pub fn new(program: &str) -> Self {
        Self {
            args: vec![CString::new(program).expect("program name has no NUL")],
        }
    }

    pub fn push(mut self, arg: impl Into<Vec<u8>>) -> Self {
        let c = CString::new(arg).expect("EAL arg has no NUL");
        self.args.push(c);
        self
    }

    pub fn extend<I>(mut self, args: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Vec<u8>>,
    {
        for a in args {
            self.args.push(CString::new(a).expect("EAL arg has no NUL"));
        }
        self
    }
}

/// Process-wide EAL handle. Drop calls `rte_eal_cleanup`; you usually
/// want exactly one of these live for the program's lifetime.
pub struct Eal {
    _private: (),
}

static EAL_INIT: Once = Once::new();
static mut EAL_INIT_RESULT: i32 = 0;

impl Eal {
    /// Initialise DPDK. Returns `Err` if a previous init failed, the
    /// second call returns the same result without re-entering DPDK.
    pub fn init(args: EalArgs) -> Result<Self> {
        EAL_INIT.call_once(|| {
            let mut ptrs: Vec<*mut c_char> = args
                .args
                .iter()
                .map(|c| c.as_ptr() as *mut c_char)
                .collect();
            let argc = ptrs.len() as c_int;
            // SAFETY: ptrs lives until call_once exits; DPDK is allowed
            // to permute argv but not free it. rte_eal_init may modify
            // the global env and spawn lcore threads; first-call only.
            let rc = unsafe { ffi::rte_eal_init(argc, ptrs.as_mut_ptr()) };
            if rc < 0 {
                // SAFETY: single-threaded init, only writer.
                unsafe { EAL_INIT_RESULT = -1 };
            }
        });

        // SAFETY: written exactly once inside the call_once above;
        // every subsequent read is post-fence.
        let result = unsafe { EAL_INIT_RESULT };
        if result < 0 {
            return Err(Error::from_rte("rte_eal_init"));
        }
        Ok(Self { _private: () })
    }

    /// How many physical ports DPDK detected.
    pub fn port_count(&self) -> u16 {
        // SAFETY: read-only DPDK getter.
        unsafe { ffi::rte_eth_dev_count_avail() }
    }
}

impl Drop for Eal {
    fn drop(&mut self) {
        // Best-effort cleanup. Some DPDK PMDs (mlx5) hold OS resources
        // we want freed at process exit; ignore the return code.
        // SAFETY: called at most once per process because Eal cannot
        // be cloned and `init`'s call_once gives us exactly one
        // handle in the program.
        let _ = unsafe { ffi::rte_eal_cleanup() };
    }
}

/// Convenience: build EAL args from a `[(flag, value)]` list and call
/// init. Equivalent to:
///
/// ```ignore
/// Eal::init(EalArgs::new("toil").push("-l").push("0").push("-n").push("4"))
/// ```
pub fn init_default(program: &str, flags: &[&str]) -> Result<Eal> {
    let mut a = EalArgs::new(program);
    for f in flags {
        a = a.push(*f);
    }
    Eal::init(a)
}

#[allow(dead_code)]
fn _silence_unused_ptr<T>() -> *mut T {
    ptr::null_mut()
}
