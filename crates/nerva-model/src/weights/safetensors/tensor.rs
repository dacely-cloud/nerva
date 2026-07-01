use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::json::fields::optional_string;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::manifest::HfTensorManifestEntry;
use crate::weights::safetensors::parse::required_usize_array;

pub(crate) fn validate_safetensors_tensor_header(
    tensor_json: &str,
    entry: &HfTensorManifestEntry,
) -> Result<()> {
    let _ = safetensors_tensor_data_offsets(tensor_json, entry)?;
    Ok(())
}

pub(crate) fn safetensors_tensor_data_offsets(
    tensor_json: &str,
    entry: &HfTensorManifestEntry,
) -> Result<(usize, usize)> {
    let dtype =
        optional_string(tensor_json, "dtype")?.ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("safetensors tensor {} is missing dtype", entry.name),
        })?;
    if dtype != safetensors_dtype(entry.dtype)? {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} dtype {} does not match expected {}",
                entry.name,
                dtype,
                safetensors_dtype(entry.dtype)?
            ),
        });
    }

    let shape = required_usize_array(tensor_json, "shape")?;
    if shape != expected_safetensors_shape(entry) {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} shape {:?} does not match expected {:?}",
                entry.name,
                shape,
                expected_safetensors_shape(entry)
            ),
        });
    }

    let offsets = required_usize_array(tensor_json, "data_offsets")?;
    if offsets.len() != 2 || offsets[1] < offsets[0] {
        return Err(NervaError::InvalidArgument {
            reason: format!("safetensors tensor {} has invalid offsets", entry.name),
        });
    }
    let span = offsets[1] - offsets[0];
    if span != entry.bytes {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "safetensors tensor {} offset span {} does not match expected bytes {}",
                entry.name, span, entry.bytes
            ),
        });
    }
    Ok((offsets[0], offsets[1]))
}

pub(crate) fn safetensors_file_data_start(header_json: &str) -> Result<usize> {
    8usize
        .checked_add(header_json.len())
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: header_json.len(),
            reason: "safetensors header file offset overflow".to_string(),
        })
}

pub(crate) fn safetensors_dtype(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::U8 => Ok("U8"),
        DType::I8 => Ok("I8"),
        DType::U32 => Ok("U32"),
        DType::I32 => Ok("I32"),
        DType::I64 => Ok("I64"),
        DType::F16 => Ok("F16"),
        DType::BF16 => Ok("BF16"),
        DType::F4E2M1 => Ok("F4"),
        DType::F8E4M3 => Ok("F8_E4M3"),
        DType::F8E5M2 => Ok("F8_E5M2"),
        DType::F8E8M0 => Ok("F8_E8M0"),
        DType::F32 => Ok("F32"),
        _ => Err(NervaError::InvalidArgument {
            reason: format!(
                "dtype {} is not supported in exact safetensors manifest validation",
                dtype.name()
            ),
        }),
    }
}

fn expected_safetensors_shape(entry: &HfTensorManifestEntry) -> Vec<usize> {
    if entry.role == WeightBlockRole::LinearConvProjection {
        return vec![entry.rows, 1, entry.cols];
    }
    match entry.rank {
        1 => vec![entry.rows],
        2 => vec![entry.rows, entry.cols],
        3 => entry
            .depth
            .map(|depth| vec![depth, entry.rows, entry.cols])
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

pub(crate) fn push_safetensors_shape_json(entry: &HfTensorManifestEntry, out: &mut String) {
    if entry.role == WeightBlockRole::LinearConvProjection {
        out.push('[');
        out.push_str(&entry.rows.to_string());
        out.push_str(",1,");
        out.push_str(&entry.cols.to_string());
        out.push(']');
        return;
    }
    match entry.rank {
        1 => {
            out.push('[');
            out.push_str(&entry.rows.to_string());
            out.push(']');
        }
        2 => {
            out.push('[');
            out.push_str(&entry.rows.to_string());
            out.push(',');
            out.push_str(&entry.cols.to_string());
            out.push(']');
        }
        3 => {
            out.push('[');
            out.push_str(&entry.depth.unwrap_or(0).to_string());
            out.push(',');
            out.push_str(&entry.rows.to_string());
            out.push(',');
            out.push_str(&entry.cols.to_string());
            out.push(']');
        }
        _ => out.push_str("[]"),
    }
}

#[cfg(test)]
mod tests {
    use nerva_core::types::dtype::DType;
    use nerva_core::types::memory::tier::MemoryTier;

    use super::{safetensors_dtype, validate_safetensors_tensor_header};
    use crate::weights::layout::entry::WeightBlockRole;
    use crate::weights::manifest::HfTensorManifestEntry;

    #[test]
    fn deepseek_storage_dtypes_map_to_safetensors_headers() {
        for (dtype, header_dtype) in [
            (DType::I8, "I8"),
            (DType::I64, "I64"),
            (DType::F4E2M1, "F4"),
            (DType::F8E4M3, "F8_E4M3"),
            (DType::F8E5M2, "F8_E5M2"),
            (DType::F8E8M0, "F8_E8M0"),
        ] {
            assert_eq!(safetensors_dtype(dtype).unwrap(), header_dtype);
            let entry = HfTensorManifestEntry {
                name: "layers.0.ffn.experts.0.w1.weight".to_string(),
                role: WeightBlockRole::ExpertGateProjection,
                layer: Some(0),
                expert: Some(0),
                rows: 2,
                cols: 3,
                depth: None,
                rank: 2,
                elements: 6,
                bytes: 6,
                dtype,
                tier: MemoryTier::Dram,
            };
            let header =
                format!(r#"{{"dtype":"{header_dtype}","shape":[2,3],"data_offsets":[0,6]}}"#);

            validate_safetensors_tensor_header(&header, &entry).unwrap();
        }
    }
}
