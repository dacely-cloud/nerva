use nerva_core::types::error::{NervaError, Result};

use crate::common::json::json_escape;
use crate::weights::manifest::HfTensorManifest;
use crate::weights::safetensors::tensor;

pub fn safetensors_header_from_bytes(bytes: &[u8]) -> Result<&str> {
    if bytes.len() < 8 {
        return Err(NervaError::InvalidArgument {
            reason: "safetensors file is too small to contain a header length".to_string(),
        });
    }
    let mut header_len_bytes = [0u8; 8];
    header_len_bytes.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(header_len_bytes);
    let header_len = usize::try_from(header_len).map_err(|_| NervaError::InvalidArgument {
        reason: "safetensors header length does not fit usize".to_string(),
    })?;
    let header_end = 8usize
        .checked_add(header_len)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "safetensors header length overflows file offset".to_string(),
        })?;
    if header_end > bytes.len() {
        return Err(NervaError::InvalidArgument {
            reason: "safetensors header length exceeds available bytes".to_string(),
        });
    }
    core::str::from_utf8(&bytes[8..header_end]).map_err(|_| NervaError::InvalidArgument {
        reason: "safetensors header is not valid UTF-8".to_string(),
    })
}

pub fn synthetic_safetensors_header_for_manifest(manifest: &HfTensorManifest) -> Result<String> {
    let mut header = String::from("{");
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        let begin = manifest.entries[..index]
            .iter()
            .try_fold(0usize, |acc, item| {
                acc.checked_add(item.bytes)
                    .ok_or_else(|| NervaError::AllocationFailed {
                        bytes: item.bytes,
                        reason: "synthetic safetensors offset overflow".to_string(),
                    })
            })?;
        let end = begin
            .checked_add(entry.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: entry.bytes,
                reason: "synthetic safetensors tensor end overflow".to_string(),
            })?;
        header.push('"');
        header.push_str(&json_escape(&entry.name));
        header.push_str("\":{\"dtype\":\"");
        header.push_str(tensor::safetensors_dtype(entry.dtype)?);
        header.push_str("\",\"shape\":");
        tensor::push_safetensors_shape_json(entry, &mut header);
        header.push_str(",\"data_offsets\":[");
        header.push_str(&begin.to_string());
        header.push(',');
        header.push_str(&end.to_string());
        header.push_str("]}");
    }
    header.push('}');
    Ok(header)
}
