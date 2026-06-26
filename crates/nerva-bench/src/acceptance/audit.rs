use std::{fs, path::Path};

const AUDIT_PATH: &str = "docs/audits/VLLM_RVLLM_ARCHITECTURE_AUDIT.md";
const REQUIRED_AUDIT_ROWS: &[&str] = &[
    "runtime language",
    "hot path owner",
    "request scheduler",
    "GPU context ownership",
    "graph capture/replay",
    "static arenas",
    "hot-path allocation",
    "token source of truth",
    "sampling",
    "host output handoff",
    "KV cache",
    "weight loading",
    "kernel contracts",
    "silent fallback behavior",
    "CUDA portability",
    "AMD/HIP portability",
    "model coverage",
    "old hardware viability",
    "exact FP16/BF16 viability",
    "DRAM warm-tier compute",
    "transport assumptions",
    "ResidentBlock compatibility",
];

pub(crate) fn audit_acceptance() -> (bool, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(AUDIT_PATH);
    let Ok(contents) = fs::read_to_string(&path) else {
        return (false, format!("missing audit file at {}", path.display()));
    };
    let required_sections = [
        "## vLLM Summary",
        "## rvLLM Summary",
        "| Area | vLLM | rvLLM | NERVA decision |",
        "## Required Questions",
    ];
    let section_hits = required_sections
        .iter()
        .filter(|section| contents.contains(**section))
        .count();
    let missing_rows = REQUIRED_AUDIT_ROWS
        .iter()
        .filter(|row| !audit_has_table_row(&contents, row))
        .copied()
        .collect::<Vec<_>>();
    let passed = section_hits == required_sections.len() && missing_rows.is_empty();
    let missing = if missing_rows.is_empty() {
        "none".to_string()
    } else {
        missing_rows.join("|")
    };
    (
        passed,
        format!(
            "path={} sections={}/{} required_rows={} missing_rows={}",
            AUDIT_PATH,
            section_hits,
            required_sections.len(),
            REQUIRED_AUDIT_ROWS.len(),
            missing,
        ),
    )
}

fn audit_has_table_row(contents: &str, row: &str) -> bool {
    contents
        .lines()
        .any(|line| line.trim_start().starts_with(&format!("| {row} |")))
}
