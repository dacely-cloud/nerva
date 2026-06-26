use std::fs;

use crate::tests::hygiene::files::{repo_root, rust_code_files};
use crate::tests::hygiene::rules::MAX_RUST_MODULE_LINES;

#[test]
fn rust_modules_stay_split_by_responsibility() {
    let repo_root = repo_root();
    let files = rust_code_files(&repo_root);
    let mut violations = Vec::new();

    for path in files {
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let lines = content.lines().count();
        if lines > MAX_RUST_MODULE_LINES {
            violations.push(format!(
                "{}: {lines} lines",
                path.strip_prefix(&repo_root).unwrap_or(&path).display()
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "Rust modules over {MAX_RUST_MODULE_LINES} lines must be split:\n{}",
        violations.join("\n")
    );
}
