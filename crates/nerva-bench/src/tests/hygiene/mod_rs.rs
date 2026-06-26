use std::fs;

use crate::tests::hygiene::files::{repo_root, rust_code_files};
use crate::tests::hygiene::rules::is_allowed_mod_rs_line;

#[test]
fn mod_rs_files_only_declare_modules() {
    let repo_root = repo_root();
    let files = rust_code_files(&repo_root);
    let mut violations = Vec::new();

    for path in files {
        if path.file_name().and_then(|name| name.to_str()) != Some("mod.rs") {
            continue;
        }
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        for (line_index, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if !is_allowed_mod_rs_line(trimmed) {
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
        "mod.rs files must only declare child modules:\n{}",
        violations.join("\n")
    );
}
