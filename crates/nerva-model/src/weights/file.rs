use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use nerva_core::types::{NervaError, Result};

use crate::common::json::json_escape;

pub const DEFAULT_MAX_SAFETENSORS_HEADER_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsFileHeader {
    pub path: PathBuf,
    pub header_json: String,
    pub header_bytes: usize,
    pub data_start: usize,
    pub file_bytes: u64,
    pub payload_bytes: u64,
}

impl SafetensorsFileHeader {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"path\":\"{}\",\"header_bytes\":{},\"data_start\":{},\"file_bytes\":{},\"payload_bytes\":{}}}",
            json_escape(&self.path.display().to_string()),
            self.header_bytes,
            self.data_start,
            self.file_bytes,
            self.payload_bytes,
        )
    }

    pub fn require_payload_bytes(&self, required_payload_bytes: usize) -> Result<()> {
        let required_payload_bytes =
            u64::try_from(required_payload_bytes).map_err(|_| NervaError::InvalidArgument {
                reason: format!(
                    "required safetensors payload byte count does not fit u64 for {}",
                    self.path.display()
                ),
            })?;
        if self.payload_bytes >= required_payload_bytes {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors file {} has payload bytes {} but requires {}",
                    self.path.display(),
                    self.payload_bytes,
                    required_payload_bytes,
                ),
            })
        }
    }

    pub fn require_file_offset_end(&self, file_offset_end: usize) -> Result<()> {
        let file_offset_end =
            u64::try_from(file_offset_end).map_err(|_| NervaError::InvalidArgument {
                reason: format!(
                    "required safetensors file offset does not fit u64 for {}",
                    self.path.display()
                ),
            })?;
        if self.file_bytes >= file_offset_end {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "safetensors file {} has bytes {} but requires offset {}",
                    self.path.display(),
                    self.file_bytes,
                    file_offset_end,
                ),
            })
        }
    }
}

pub fn read_safetensors_header_file(path: impl AsRef<Path>) -> Result<SafetensorsFileHeader> {
    read_safetensors_header_file_with_limit(path, DEFAULT_MAX_SAFETENSORS_HEADER_BYTES)
}

pub fn read_safetensors_header_file_with_limit(
    path: impl AsRef<Path>,
    max_header_bytes: usize,
) -> Result<SafetensorsFileHeader> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|err| NervaError::InvalidArgument {
        reason: format!("failed to open safetensors file {}: {err}", path.display()),
    })?;
    let file_bytes = file
        .metadata()
        .map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "failed to read safetensors file metadata for {}: {err}",
                path.display()
            ),
        })?
        .len();
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "failed to read safetensors header length from {}: {err}",
                path.display()
            ),
        })?;
    let header_bytes = usize::try_from(u64::from_le_bytes(header_len_bytes)).map_err(|_| {
        NervaError::InvalidArgument {
            reason: format!(
                "safetensors header length in {} does not fit usize",
                path.display()
            ),
        }
    })?;
    if header_bytes > max_header_bytes {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors header length {} exceeds configured limit {} for {}",
                header_bytes,
                max_header_bytes,
                path.display()
            ),
        });
    }
    let data_start =
        8usize
            .checked_add(header_bytes)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("safetensors header length overflows for {}", path.display()),
            })?;
    let data_start_u64 = u64::try_from(data_start).map_err(|_| NervaError::InvalidArgument {
        reason: format!(
            "safetensors header length overflows u64 for {}",
            path.display()
        ),
    })?;
    if data_start_u64 > file_bytes {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors header length {} exceeds file size {} for {}",
                header_bytes,
                file_bytes,
                path.display()
            ),
        });
    }
    let mut header = vec![0u8; header_bytes];
    file.read_exact(&mut header)
        .map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "failed to read safetensors header from {}: {err}",
                path.display()
            ),
        })?;
    let header_json = String::from_utf8(header).map_err(|err| NervaError::InvalidArgument {
        reason: format!(
            "safetensors header in {} is not valid UTF-8: {err}",
            path.display()
        ),
    })?;
    Ok(SafetensorsFileHeader {
        path: path.to_path_buf(),
        header_json,
        header_bytes,
        data_start,
        file_bytes,
        payload_bytes: file_bytes - data_start_u64,
    })
}
