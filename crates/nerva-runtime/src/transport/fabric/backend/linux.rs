use std::path::Path;

pub(crate) fn module_loaded(name: &str) -> bool {
    Path::new("/sys/module").join(name).is_dir()
}

pub(crate) fn dpdk_shim_sources_present() -> bool {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shim_root = manifest.join("../../native/dpdk/dpdk-shim");
    shim_root.join("Cargo.toml").is_file()
        && shim_root.join("shim.c").is_file()
        && shim_root.join("wrapper.h").is_file()
        && shim_root.join("src/lib.rs").is_file()
}

pub(crate) fn hugepages_total() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    contents.lines().find_map(|line| {
        let value = line.strip_prefix("HugePages_Total:")?.trim();
        value.parse::<u64>().ok()
    })
}
