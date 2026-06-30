use crate::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DType {
    U8,
    U4,
    I4,
    I8,
    U16,
    U32,
    I32,
    I64,
    F16,
    BF16,
    F4E2M1,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    F32,
    TF32,
}

impl DType {
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U4 => "u4",
            Self::I4 => "i4",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F4E2M1 => "float4_e2m1",
            Self::F8E4M3 => "float8_e4m3",
            Self::F8E5M2 => "float8_e5m2",
            Self::F8E8M0 => "float8_e8m0",
            Self::F32 => "float32",
            Self::TF32 => "tensorfloat32",
        }
    }

    pub const fn storage_bits(self) -> usize {
        match self {
            Self::U4 | Self::I4 | Self::F4E2M1 => 4,
            Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 => 8,
            Self::U16 | Self::F16 | Self::BF16 => 16,
            Self::U32 | Self::I32 | Self::F32 | Self::TF32 => 32,
            Self::I64 => 64,
        }
    }

    pub const fn is_subbyte_packed(self) -> bool {
        matches!(self, Self::U4 | Self::I4 | Self::F4E2M1)
    }

    pub const fn whole_bytes_per_element(self) -> Option<usize> {
        match self.storage_bits() {
            8 => Some(1),
            16 => Some(2),
            32 => Some(4),
            64 => Some(8),
            _ => None,
        }
    }

    pub fn packed_storage_bytes(self, elements: usize) -> Result<usize> {
        let bits = elements.checked_mul(self.storage_bits()).ok_or_else(|| {
            NervaError::AllocationFailed {
                bytes: elements,
                reason: format!("{} packed bit count overflow", self.name()),
            }
        })?;
        bits.checked_add(7)
            .map(|bits| bits / 8)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: elements,
                reason: format!("{} packed byte count overflow", self.name()),
            })
    }
}
