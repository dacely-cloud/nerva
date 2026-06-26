use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn rust_modules_do_not_use_reexport_shims() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate should live under repo_root/crates");
    let mut files = Vec::new();
    collect_rust_files(&repo_root.join("crates"), &mut files);

    let mut violations = Vec::new();
    for path in files {
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        for (line_index, line) in content.lines().enumerate() {
            let trimmed = line.trim_start();
            if is_forbidden_import(trimmed) {
                violations.push(format!(
                    "{}:{}: {}",
                    path.strip_prefix(repo_root).unwrap_or(&path).display(),
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

#[test]
fn mod_rs_files_only_declare_modules() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate should live under repo_root/crates");
    let mut files = Vec::new();
    collect_rust_files(&repo_root.join("crates"), &mut files);

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
                    path.strip_prefix(repo_root).unwrap_or(&path).display(),
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

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|err| panic!("failed to read {}: {err}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("failed to inspect {}: {err}", dir.display()))
            .path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

fn is_forbidden_import(trimmed: &str) -> bool {
    trimmed.starts_with("pub use ")
        || trimmed.starts_with("pub(crate) use ")
        || trimmed.starts_with("use super::*")
}

fn is_allowed_mod_rs_line(trimmed: &str) -> bool {
    trimmed.is_empty()
        || trimmed.starts_with("//")
        || trimmed.starts_with("#[")
        || trimmed == "mod tests;"
        || trimmed.starts_with("mod ") && trimmed.ends_with(';')
        || trimmed.starts_with("pub mod ") && trimmed.ends_with(';')
        || trimmed.starts_with("pub(crate) mod ") && trimmed.ends_with(';')
}
