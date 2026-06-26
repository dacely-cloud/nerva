use crate::types::arch::HostArch;
use crate::types::id::ResidentBlockId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NervaError {
    UnsupportedHost {
        arch: HostArch,
    },
    BackendUnavailable {
        backend: &'static str,
        reason: String,
    },
    AllocationFailed {
        bytes: usize,
        reason: String,
    },
    InvalidArgument {
        reason: String,
    },
    ResidencyViolation {
        block_id: ResidentBlockId,
        reason: String,
    },
}

pub type Result<T> = std::result::Result<T, NervaError>;
