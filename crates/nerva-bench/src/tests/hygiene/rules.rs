pub(crate) const MAX_RUST_MODULE_LINES: usize = 200;

pub(crate) fn is_forbidden_import(trimmed: &str) -> bool {
    is_public_use(trimmed) || is_super_glob_import(trimmed)
}

pub(crate) fn is_allowed_mod_rs_line(trimmed: &str) -> bool {
    trimmed.is_empty()
        || trimmed.starts_with("//")
        || trimmed.starts_with("#[")
        || trimmed.starts_with("#![")
        || trimmed == "mod tests;"
        || trimmed.starts_with("mod ") && trimmed.ends_with(';')
        || trimmed.starts_with("pub mod ") && trimmed.ends_with(';')
        || trimmed.starts_with("pub(crate) mod ") && trimmed.ends_with(';')
}

fn is_public_use(trimmed: &str) -> bool {
    let Some(rest) = trimmed.strip_prefix("pub") else {
        return false;
    };
    let rest = rest.trim_start();
    if rest.starts_with("use ") {
        return true;
    }
    if let Some(scoped) = rest.strip_prefix('(') {
        let Some(end) = scoped.find(')') else {
            return false;
        };
        return scoped[end + 1..].trim_start().starts_with("use ");
    }
    false
}

fn is_super_glob_import(trimmed: &str) -> bool {
    let compact: String = trimmed.chars().filter(|ch| !ch.is_whitespace()).collect();
    compact == "usesuper::*;"
}

#[test]
fn public_use_rule_rejects_reexport_forms() {
    let grouped_reexport = ["pub", "use crate::model::{"].join(" ");
    assert!(is_forbidden_import(&grouped_reexport));
    assert!(is_forbidden_import("pub(crate) use crate::model::Thing;"));
    assert!(is_forbidden_import("pub(super) use crate::model::Thing;"));
    assert!(is_forbidden_import(
        "pub(in crate::model) use crate::model::Thing;"
    ));
}

#[test]
fn import_rule_allows_direct_private_imports() {
    assert!(!is_forbidden_import("use crate::model::reference::Thing;"));
    assert!(!is_forbidden_import("pub mod model;"));
    assert!(!is_forbidden_import("pub(crate) mod model;"));
}

#[test]
fn super_glob_rule_rejects_whitespace_variants() {
    assert!(is_forbidden_import("use super::*;"));
    assert!(is_forbidden_import("use super :: * ;"));
    assert!(!is_forbidden_import("use super::thing;"));
}
