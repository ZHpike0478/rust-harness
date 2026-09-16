// build.rs -- link the cxx staticlib produced by llama-harness.
//
// llama-harness's build.rs compiles the C++ sources into
// target/<profile>/deps/llama_harness_inline.{lib,a}. We tell the
// linker where to find it. (Cargo doesn't propagate staticlib link
// directives transitively across build scripts, so each binary needs
// this.)

use std::env;

fn main() {
    // target/ directory is either CARGO_TARGET_DIR or workspace_root/target.
    let target_dir = env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| "target".into());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let deps = format!("{target_dir}/{profile}/deps");

    println!("cargo:rustc-link-search=native={deps}");
    println!("cargo:rustc-link-lib=static=llama_harness_inline");
}
