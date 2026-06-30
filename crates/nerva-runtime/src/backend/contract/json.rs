use nerva_core::types::backend::capabilities::DeviceBackendKind;
use nerva_core::types::dtype::DType;

pub(crate) fn backend_kind_to_str(kind: DeviceBackendKind) -> &'static str {
    kind.as_str()
}

pub(crate) fn dtype_array(values: &[DType]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(value.name());
        out.push('"');
    }
    out.push(']');
    out
}
