use std::fs;

use crate::tests::hygiene::files::{repo_root, rust_code_files};
use crate::tests::hygiene::rules::is_forbidden_import;

#[test]
fn rust_modules_do_not_use_reexport_shims() {
    let repo_root = repo_root();
    let files = rust_code_files(&repo_root);
    let mut violations = Vec::new();

    for path in files {
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        for (line_index, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            if is_forbidden_import(trimmed) {
                violations.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(&repo_root).unwrap_or(&path).display(),
                    line_index + 1,
                    trimmed
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "module hygiene violations:\n{}",
        violations.join("\n")
    );
}
