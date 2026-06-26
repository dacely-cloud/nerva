#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{
    MemoryTier, NervaError, ResidentBlock, ResidentBlockId, ResidentBlockKind, Result,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArenaKind {
    Device,
    PinnedHost,
    Host,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaReservation {
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}

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
}

pub fn resident_block_for_reservation(
    id: ResidentBlockId,
    kind: ResidentBlockKind,
    reservation: ArenaReservation,
) -> ResidentBlock {
    ResidentBlock::new(id, kind, MemoryTier::Dram, reservation.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_arena_respects_alignment() {
        let mut arena = HostArena::new(1024);
        let a = arena.reserve(3, 1).unwrap();
        let b = arena.reserve(8, 64).unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset % 64, 0);
        assert!(arena.used() >= b.offset + 8);
    }
}
