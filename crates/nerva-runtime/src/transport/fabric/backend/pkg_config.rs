use std::process::Command;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DpdkPkgConfig {
    pub(crate) present: bool,
    pub(crate) version: Option<String>,
    pub(crate) mlx5_pmd_linked: bool,
    pub(crate) gpudev_linked: bool,
}

pub(crate) fn read_dpdk_pkg_config() -> DpdkPkgConfig {
    if !command_success("pkg-config", &["--exists", "libdpdk"]) {
        return DpdkPkgConfig::default();
    }
    let version = command_stdout("pkg-config", &["--modversion", "libdpdk"]);
    let libs = command_stdout("pkg-config", &["--libs", "libdpdk"]).unwrap_or_default();
    let static_libs =
        command_stdout("pkg-config", &["--libs", "--static", "libdpdk"]).unwrap_or_default();
    let link_flags = format!("{libs} {static_libs}");
    DpdkPkgConfig {
        present: true,
        version,
        mlx5_pmd_linked: link_flags.contains("mlx5") || link_flags.contains("rte_net_mlx5"),
        gpudev_linked: link_flags.contains("gpudev")
            || link_flags.contains("rte_gpudev")
            || link_flags.contains("rte_gpu"),
    }
}

fn command_success(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}
