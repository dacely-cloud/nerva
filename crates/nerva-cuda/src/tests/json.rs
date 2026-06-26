use crate::json::escape_json;

#[test]
fn json_escapes_control_chars() {
    assert_eq!(escape_json("a\"b\\c\n"), "a\\\"b\\\\c\\n");
}
