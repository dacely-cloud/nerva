//! Build the in-tree DPDK shim.
//!
//! Pipeline:
//!
//!   1. Locate DPDK via `pkg-config libdpdk`. We require it because the
//!      shim's whole job is to wrap real DPDK; missing DPDK is a build
//!      failure, not a simulated data path.
//!   2. Compile `shim.c` with `cc`, feeding it the same CFLAGS DPDK
//!      asked for (include paths, `-march=native -mrtm`,
//!      `-include rte_config.h`).
//!   3. Run bindgen on `wrapper.h` using the same flags so the C macros
//!      (RTE_ATOMIC, __rte_capability, RTE_FLOW_ITEM_TYPE_*) resolve.
//!   4. Emit `cargo:rustc-link-*` for the DPDK libraries pkg-config
//!      reports — DPDK splits into ~60 .so's, so let pkg-config handle
//!      it instead of hand-listing.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=shim.c");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    // 1. pkg-config: include paths, link libs, extra CFLAGS.
    let dpdk = pkg_config::Config::new()
        .statik(false)
        .cargo_metadata(true)
        .probe("libdpdk")
        .expect(
            "pkg-config could not find libdpdk. Install DPDK 22+ and \
             ensure its `libdpdk.pc` is on PKG_CONFIG_PATH.",
        );

    // pkg-config already emitted -l flags for us via cargo_metadata.

    // Reassemble pkg-config CFLAGS as a Vec<String> so both `cc` and
    // bindgen see the same set. We can't read them off the `Library`
    // value directly (the struct exposes include_paths / defines /
    // ldflags but not raw cflags), so we shell out once.
    let cflags_out = std::process::Command::new("pkg-config")
        .args(["--cflags", "libdpdk"])
        .output()
        .expect("pkg-config --cflags libdpdk failed");
    if !cflags_out.status.success() {
        panic!(
            "pkg-config --cflags libdpdk exited {}: {}",
            cflags_out.status,
            String::from_utf8_lossy(&cflags_out.stderr)
        );
    }
    let raw = String::from_utf8(cflags_out.stdout).unwrap();
    let cflags: Vec<String> = raw.split_whitespace().map(str::to_string).collect();

    // 2. Compile shim.c with the DPDK CFLAGS. We add -fno-strict-aliasing
    //    because rte_mbuf macro arithmetic relies on type-punned access
    //    through the mbuf head pointer.
    let mut cc_build = cc::Build::new();
    cc_build
        .file("shim.c")
        .include(".") // for wrapper.h
        .flag("-fno-strict-aliasing")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-deprecated-declarations");
    for inc in dpdk.include_paths.iter() {
        cc_build.include(inc);
    }
    for flag in &cflags {
        // pkg-config returns -I dirs (already added above), -include, and
        // -march/-m flags. Forward everything that cc::Build will accept.
        if flag.starts_with("-I") {
            continue; // already handled
        }
        cc_build.flag(flag);
    }
    cc_build.compile("dpdk-shim");

    // 3. Run bindgen on the same headers.
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("rte_.*")
        .allowlist_function("shim_.*")
        .allowlist_type("rte_.*")
        .allowlist_var("RTE_.*")
        .allowlist_var("rte_errno")
        // Some DPDK types embed packed structs that contain types with
        // `#[repr(align)]`. bindgen rejects the combo by default. We
        // shadow the offending containers as opaque byte arrays — they
        // are only used by DPDK internals and never read by us.
        .opaque_type("rte_arp_ipv4")
        .opaque_type("rte_arp_hdr")
        .opaque_type("rte_l2tpv2_combined_msg_hdr")
        .opaque_type("rte_gtp_psc_generic_hdr")
        .blocklist_type("max_align_t")
        .derive_default(true)
        .derive_debug(true)
        .layout_tests(false)
        .use_core()
        .ctypes_prefix("libc");

    for inc in dpdk.include_paths.iter() {
        builder = builder.clang_arg(format!("-I{}", inc.display()));
    }
    for flag in &cflags {
        // Bindgen wants -I and -include + -march flags forwarded;
        // skip dup -I (already added above).
        if flag.starts_with("-I") {
            continue;
        }
        builder = builder.clang_arg(flag);
    }
    // Also make sure clang sees our own dir so wrapper.h's shim_*
    // declarations resolve when bindgen parses it.
    builder = builder.clang_arg("-I.");

    let bindings = builder
        .generate()
        .expect("bindgen failed to generate DPDK bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("could not write bindings.rs");
}
