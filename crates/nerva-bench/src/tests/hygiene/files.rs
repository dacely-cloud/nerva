use std::fs;
use std::path::{Path, PathBuf};

pub(crate) fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate should live under repo_root/crates")
        .to_path_buf()
}

pub(crate) fn rust_code_files(repo_root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_rust_files(&repo_root.join("crates"), &mut files);
    collect_rust_files(&repo_root.join("native"), &mut files);
    files
}

fn collect_rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|err| panic!("failed to read {}: {err}", dir.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|err| panic!("failed to inspect {}: {err}", dir.display()))
            .path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) == Some("target") {
                continue;
            }
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}
