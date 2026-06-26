use nerva_core::types::error::{NervaError, Result};

use crate::arena::region::ArenaReservation;

#[derive(Clone, Debug)]
pub struct HostArena {
    bytes: Vec<u8>,
    used: usize,
}

impl HostArena {
    pub fn new(capacity: usize) -> Self {
        Self {
            bytes: vec![0; capacity],
            used: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.used
    }

    pub fn reserve(&mut self, bytes: usize, align: usize) -> Result<ArenaReservation> {
        let align = align.max(1);
        let offset = self.used.next_multiple_of(align);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes,
                reason: "arena offset overflow".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: "host arena exhausted".to_string(),
            });
        }
        self.used = end;
        Ok(ArenaReservation {
            offset,
            bytes,
            align,
        })
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}
