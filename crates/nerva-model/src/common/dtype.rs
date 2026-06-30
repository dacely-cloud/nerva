use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

pub(crate) fn dtype_size_bytes(dtype: DType) -> Result<usize> {
    dtype
        .whole_bytes_per_element()
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "dtype {} is packed below one byte; use packed_storage_bytes with an element count",
                dtype.name()
            ),
        })
}

pub(crate) fn json_opt_dtype(value: Option<DType>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", value.name()),
    )
}
