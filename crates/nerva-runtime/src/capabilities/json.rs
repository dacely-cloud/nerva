use nerva_core::types::arch::HostArch;
use nerva_core::types::memory::MemoryFabricKind;

pub(crate) fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn host_arch_to_str(value: HostArch) -> &'static str {
    match value {
        HostArch::X86_64 => "x86_64",
        HostArch::Aarch64 => "aarch64",
        HostArch::Other => "other",
    }
}

pub(crate) fn memory_fabric_to_str(value: MemoryFabricKind) -> &'static str {
    match value {
        MemoryFabricKind::DiscreteExplicit => "DiscreteExplicit",
        MemoryFabricKind::UnifiedVirtualManaged => "UnifiedVirtualManaged",
        MemoryFabricKind::CoherentSharedPhysical => "CoherentSharedPhysical",
        MemoryFabricKind::CxlCoherentFabric => "CxlCoherentFabric",
    }
}

pub(crate) fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

pub(crate) fn json_opt_string(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

pub(crate) fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(value));
        out.push('"');
    }
    out.push(']');
    out
}
