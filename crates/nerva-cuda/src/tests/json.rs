use crate::json::{escape_json, json_opt_bool};

#[test]
fn json_escapes_control_chars() {
    assert_eq!(escape_json("a\"b\\c\n"), "a\\\"b\\\\c\\n");
}

#[test]
fn json_optional_bool_serializes_known_and_unknown_values() {
    assert_eq!(json_opt_bool(Some(true)), "true");
    assert_eq!(json_opt_bool(Some(false)), "false");
    assert_eq!(json_opt_bool(None), "null");
}
