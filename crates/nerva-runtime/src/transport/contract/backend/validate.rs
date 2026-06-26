use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::MemoryTier;

use crate::transport::contract::types::{ReceiveDescriptor, TransferDescriptor};
use crate::transport::registration::types::{TransportRegistration, TransportRegistrationBackend};

pub(super) fn validate_receive_descriptor(
    backend: TransportRegistrationBackend,
    receive: ReceiveDescriptor,
) -> Result<()> {
    validate_registration("receive", backend, receive.destination)?;
    validate_registered_range(
        "receive",
        receive.destination.bytes,
        receive.destination_offset,
        receive.bytes,
    )?;
    if receive.destination.key.replica != receive.destination_replica {
        return Err(NervaError::InvalidArgument {
            reason: "receive descriptor replica does not match registration".to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_transfer_descriptor(
    backend: TransportRegistrationBackend,
    transfer: TransferDescriptor,
) -> Result<()> {
    validate_registration("send", backend, transfer.source)?;
    validate_registered_range(
        "send",
        transfer.source.bytes,
        transfer.source_offset,
        transfer.bytes,
    )?;
    if transfer.source.key.replica != transfer.source_replica {
        return Err(NervaError::InvalidArgument {
            reason: "transfer descriptor replica does not match registration".to_string(),
        });
    }
    if transfer.block_version < transfer.source.registered_min_version {
        return Err(NervaError::ResidencyViolation {
            block_id: transfer.source.key.block_id,
            reason: "transport send requires a registered block version".to_string(),
        });
    }
    Ok(())
}

fn validate_registration(
    direction: &'static str,
    backend: TransportRegistrationBackend,
    registration: TransportRegistration,
) -> Result<()> {
    if registration.key.backend != backend {
        return Err(NervaError::InvalidArgument {
            reason: format!("{direction} registration backend does not match transport backend"),
        });
    }
    if registration.tier != MemoryTier::PinnedDram {
        return Err(NervaError::InvalidArgument {
            reason: format!("{direction} registration must use pinned host memory"),
        });
    }
    Ok(())
}

fn validate_registered_range(
    direction: &'static str,
    registered_bytes: usize,
    offset: usize,
    bytes: usize,
) -> Result<()> {
    let Some(end) = offset.checked_add(bytes) else {
        return Err(NervaError::InvalidArgument {
            reason: format!("{direction} descriptor byte range overflows"),
        });
    };
    if bytes == 0 || end > registered_bytes {
        return Err(NervaError::InvalidArgument {
            reason: format!("{direction} descriptor byte range is outside registration"),
        });
    }
    Ok(())
}
