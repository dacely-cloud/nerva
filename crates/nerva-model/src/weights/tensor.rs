use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::hash::hash_bytes;
use crate::common::json::format::json_escape;
use crate::weights::safetensors::shard::SafetensorsShardPlanEntry;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedSafetensorsTensorU16 {
    pub name: String,
    pub dtype: DType,
    pub values: Vec<u16>,
    pub bytes_read: usize,
    pub data_hash: u64,
}

impl LoadedSafetensorsTensorU16 {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"dtype\":\"{}\",\"values\":{},\"bytes_read\":{},\"data_hash\":{}}}",
            json_escape(&self.name),
            dtype_json_label(self.dtype),
            self.values.len(),
            self.bytes_read,
            self.data_hash,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LoadedSafetensorsTensorF32 {
    pub name: String,
    pub values: Vec<f32>,
    pub bytes_read: usize,
    pub data_hash: u64,
}

pub fn read_safetensors_tensor_u16(
    shard_path: impl AsRef<Path>,
    entry: &SafetensorsShardPlanEntry,
) -> Result<LoadedSafetensorsTensorU16> {
    read_safetensors_tensor_u16_with_hash(shard_path, entry, true)
}

pub fn read_safetensors_tensor_u16_with_hash(
    shard_path: impl AsRef<Path>,
    entry: &SafetensorsShardPlanEntry,
    compute_hash: bool,
) -> Result<LoadedSafetensorsTensorU16> {
    match entry.dtype {
        DType::F16 | DType::BF16 => {}
        dtype => {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors tensor {} has dtype {:?}; u16 tensor loading supports only FP16/BF16",
                    entry.tensor_name, dtype
                ),
            });
        }
    }
    if entry.bytes % 2 != 0 {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} byte count {} is not divisible by 2",
                entry.tensor_name, entry.bytes
            ),
        });
    }
    let bytes = read_safetensors_tensor_bytes(shard_path, entry)?;
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    Ok(LoadedSafetensorsTensorU16 {
        name: entry.tensor_name.clone(),
        dtype: entry.dtype,
        values,
        bytes_read: entry.bytes,
        data_hash: if compute_hash { hash_bytes(&bytes) } else { 0 },
    })
}

pub fn read_safetensors_tensor_f32_with_hash(
    shard_path: impl AsRef<Path>,
    entry: &SafetensorsShardPlanEntry,
    compute_hash: bool,
) -> Result<LoadedSafetensorsTensorF32> {
    if entry.dtype != DType::F32 {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} has dtype {:?}; f32 tensor loading supports only F32",
                entry.tensor_name, entry.dtype
            ),
        });
    }
    if entry.bytes % 4 != 0 {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} byte count {} is not divisible by 4",
                entry.tensor_name, entry.bytes
            ),
        });
    }
    let bytes = read_safetensors_tensor_bytes(shard_path, entry)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok(LoadedSafetensorsTensorF32 {
        name: entry.tensor_name.clone(),
        values,
        bytes_read: entry.bytes,
        data_hash: if compute_hash { hash_bytes(&bytes) } else { 0 },
    })
}

fn read_safetensors_tensor_bytes(
    shard_path: impl AsRef<Path>,
    entry: &SafetensorsShardPlanEntry,
) -> Result<Vec<u8>> {
    if entry.file_offset_end < entry.file_offset_begin
        || entry.file_offset_end - entry.file_offset_begin != entry.bytes
    {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} has invalid source file span",
                entry.tensor_name
            ),
        });
    }

    let shard_path = shard_path.as_ref();
    let mut file = File::open(shard_path).map_err(|err| NervaError::InvalidArgument {
        reason: format!(
            "failed to open safetensors shard {}: {err}",
            shard_path.display()
        ),
    })?;
    file.seek(SeekFrom::Start(
        u64::try_from(entry.file_offset_begin).map_err(|_| NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} file offset does not fit u64",
                entry.tensor_name
            ),
        })?,
    ))
    .map_err(|err| NervaError::InvalidArgument {
        reason: format!(
            "failed to seek safetensors shard {}: {err}",
            shard_path.display()
        ),
    })?;
    let mut bytes = vec![0u8; entry.bytes];
    file.read_exact(&mut bytes)
        .map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "failed to read safetensors tensor {} from {}: {err}",
                entry.tensor_name,
                shard_path.display()
            ),
        })?;
    Ok(bytes)
}

fn dtype_json_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F32 => "float32",
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::F8E4M3 => "float8_e4m3",
        DType::F8E8M0 => "float8_e8m0",
    }
}
