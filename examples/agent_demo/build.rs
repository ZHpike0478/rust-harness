// build.rs -- link the cxx staticlib produced by llama-harness.
//
// (See ../chat_cli/build.rs for explanation.)

use std::env;

fn main() {
    let target_dir = env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| "target".into());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let deps = format!("{target_dir}/{profile}/deps");

    println!("cargo:rustc-link-search=native={deps}");
    println!("cargo:rustc-link-lib=static=llama_harness_inline");
}
