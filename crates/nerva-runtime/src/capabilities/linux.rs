use std::fs;

pub(crate) fn extract_iommu_kernel_args(cmdline: &str) -> Option<String> {
    let args = cmdline
        .split_whitespace()
        .filter(|arg| arg.contains("iommu"))
        .collect::<Vec<_>>();
    (!args.is_empty()).then(|| args.join(" "))
}

pub(crate) fn discover_iommu_mode(
    iommu_group_count: usize,
    iommu_kernel_args: Option<&str>,
) -> String {
    let args = iommu_kernel_args.unwrap_or_default();
    if has_kernel_arg(args, &["iommu=off", "intel_iommu=off", "amd_iommu=off"]) {
        return "disabled_by_kernel_arg".to_string();
    }
    if iommu_group_count > 0 && has_kernel_arg(args, &["iommu=pt"]) {
        return "passthrough_groups_present".to_string();
    }
    if iommu_group_count > 0 {
        return "enabled_groups_present".to_string();
    }
    if has_kernel_arg(args, &["iommu=pt"]) {
        return "passthrough_requested".to_string();
    }
    if has_kernel_arg(args, &["iommu=on", "intel_iommu=on", "amd_iommu=on"]) {
        return "enabled_requested".to_string();
    }
    "not_detected".to_string()
}

fn has_kernel_arg(args: &str, candidates: &[&str]) -> bool {
    args.split_whitespace()
        .any(|arg| candidates.iter().any(|candidate| arg == *candidate))
}

pub(crate) fn parse_pci_class(value: &str) -> Option<u32> {
    u32::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok()
}

pub(crate) fn count_linux_id_list(value: &str) -> Option<usize> {
    let mut total = 0usize;
    for part in value.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start.trim().parse::<usize>().ok()?;
            let end = end.trim().parse::<usize>().ok()?;
            if end < start {
                return None;
            }
            total = total.checked_add(end - start + 1)?;
        } else {
            part.parse::<usize>().ok()?;
            total = total.checked_add(1)?;
        }
    }
    Some(total)
}

pub(crate) fn count_entries(path: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries.flatten().count()
}

pub(crate) fn count_dirs(path: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .count()
}

pub(crate) fn count_prefixed_entries(path: &str, prefix: &str) -> usize {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
        .count()
}

pub(crate) fn list_entry_names(path: &str) -> Vec<String> {
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut names = entries
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    names.sort();
    names
}

pub(crate) fn read_trimmed_first_line(path: &str) -> Option<String> {
    let contents = fs::read_to_string(path).ok()?;
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToOwned::to_owned)
}
