use nerva_core::types::backend::capabilities::DeviceBackendKind;
use nerva_core::types::dtype::DType;

pub(crate) fn backend_kind_to_str(kind: DeviceBackendKind) -> &'static str {
    kind.as_str()
}

pub(crate) fn dtype_to_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "u8",
        DType::I8 => "i8",
        DType::U16 => "u16",
        DType::U32 => "u32",
        DType::I32 => "i32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F8E4M3 => "float8_e4m3",
        DType::F8E8M0 => "float8_e8m0",
        DType::F32 => "float32",
    }
}

pub(crate) fn dtype_array(values: &[DType]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(dtype_to_str(*value));
        out.push('"');
    }
    out.push(']');
    out
}
