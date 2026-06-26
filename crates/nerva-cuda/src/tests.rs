use std::os::raw::c_char;

use crate::CudaSmokeSummary;
use crate::smoke::{c_char_array_to_string, escape_json};

#[test]
fn json_escapes_control_chars() {
    assert_eq!(escape_json("a\"b\\c\n"), "a\\\"b\\\\c\\n");
}

#[test]
fn unavailable_summary_is_valid_shape() {
    let summary = CudaSmokeSummary::unavailable("no cuda", Some(13_010));
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"unavailable\""));
    assert!(json.contains("\"runtime_version\":13010"));
    assert!(json.contains("\"compute_capability_major\":null"));
    assert!(json.contains("\"compute_capability_minor\":null"));
    assert!(json.contains("\"device_total_memory_bytes\":null"));
    assert!(json.contains("\"pci_bus_id\":null"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn c_char_array_conversion_handles_empty_and_terminated_values() {
    let empty = [0 as c_char; 8];
    assert_eq!(c_char_array_to_string(&empty), None);

    let mut value = [0 as c_char; 8];
    value[0] = b'R' as c_char;
    value[1] = b'T' as c_char;
    value[2] = b'X' as c_char;
    assert_eq!(c_char_array_to_string(&value).as_deref(), Some("RTX"));
}
