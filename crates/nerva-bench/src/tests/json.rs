use crate::json::json_string_array;

#[test]
fn json_string_array_escapes_probe_args() {
    let args = vec!["quote\"".to_string(), "line\nbreak".to_string()];
    assert_eq!(json_string_array(&args), "[\"quote\\\"\",\"line\\nbreak\"]");
}
