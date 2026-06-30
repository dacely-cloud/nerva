use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

pub(crate) fn dtype_size_bytes(dtype: DType) -> Result<usize> {
    match dtype {
        DType::I8 | DType::F8E4M3 | DType::F8E8M0 => Ok(1),
        DType::F16 | DType::BF16 => Ok(2),
        DType::F32 => Ok(4),
        DType::I64 => Ok(8),
        _ => Err(NervaError::InvalidArgument {
            reason: format!(
                "dtype {} is not a supported exact weight dtype",
                dtype_to_str(dtype)
            ),
        }),
    }
}

pub(crate) fn dtype_to_str(value: DType) -> &'static str {
    match value {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F8E4M3 => "float8_e4m3",
        DType::F8E8M0 => "float8_e8m0",
        DType::F32 => "float32",
    }
}

pub(crate) fn json_opt_dtype(value: Option<DType>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", dtype_to_str(value)),
    )
}
