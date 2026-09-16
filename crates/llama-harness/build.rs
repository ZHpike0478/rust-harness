// build.rs -- compile the C++ side of the FFI bridge directly into this
// crate, and link it from any downstream crate via the `links` mechanism.
//
// Steps:
//   1. Get the cxx DTO header directory from llama-harness-ffi's build.rs.
//   2. Compile the C++ sources using cxx_build into a static archive.
//   3. Copy the static archive into target/<profile>/deps/ so downstream
//      crates can find it (Cargo's standard search path).
//   4. Emit link directives to rustc.

use std::env;
use std::path::PathBuf;

fn main() {
    let workspace_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent().unwrap()
        .parent().unwrap()
        .to_path_buf();
    let ffi_dir = workspace_root.join("crates").join("llama-harness-ffi");
    let cpp_dir = ffi_dir.join("cpp");

    let cxx_dto_dir = env::var("DEP_LLAMA_HARNESS_FFI_CXX_DTO_DIR")
        .expect("DEP_LLAMA_HARNESS_FFI_CXX_DTO_DIR must be set by llama-harness-ffi");
    let cxx_dto_dir = PathBuf::from(cxx_dto_dir);

    let mut build = cxx_build::bridge(ffi_dir.join("src").join("lib.rs"));

    build
        .file(cpp_dir.join("HarnessError.cpp"))
        .file(cpp_dir.join("StubEngine.cpp"))
        .include(&cpp_dir)
        .include(&cxx_dto_dir)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .flag_if_supported("/utf-8")
        .flag_if_supported("-Wno-unused-parameter")
        .warnings(false);

    // Optionally include llama.cpp when vendored.
    let llama_dir = ffi_dir.join("vendor").join("llama.cpp");
    if llama_dir.exists() {
        let llama_inc = llama_dir.join("include");
        let llama_src = llama_dir.join("ggml").join("src");
        build
            .include(&llama_inc)
            .include(&llama_src)
            .define("LLAMA_HARNESS_REAL_BACKEND", None);
        for entry in walkdir(&llama_src) {
            if entry.extension().and_then(|s| s.to_str()) == Some("cpp") {
                build.file(entry);
            }
        }
    }

    build.compile("llama_harness_inline");

    // ----- Propagate the link directive to downstream crates -----
    // Copy the static archive into target/<profile>/deps/ so any
    // downstream linker invocation that searches there can find it.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
    let target_dir = env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| workspace_root.join("target").to_string_lossy().to_string());
    let deps_dir = PathBuf::from(&target_dir).join(&profile).join("deps");
    let _ = std::fs::create_dir_all(&deps_dir);

    // The archive's actual filename has a hash suffix added by cc-rs.
    // Find it.
    eprintln!("OUT_DIR: {}", out_dir.display());
    if let Ok(rd) = std::fs::read_dir(&out_dir) {
        for e in rd.flatten() {
            eprintln!("  entry: {}", e.file_name().to_string_lossy());
        }
    }
    let archive = std::fs::read_dir(&out_dir)
        .ok()
        .and_then(|rd| {
            rd.flatten()
                .find(|e| {
                    let n = e.file_name().to_string_lossy().to_string();
                    let base = n.strip_prefix("lib").unwrap_or(&n);
                    base.starts_with("llama_harness_inline-")
                        || n == "libllama_harness_inline.a"
                        || n == "llama_harness_inline.lib"
                })
                .map(|e| e.path())
        })
        .expect("libllama_harness_inline.{a,lib} not found in build out dir");

    // Copy to deps/ under the canonical name (without hash, which Cargo
    // uses only for incremental rebuilds; the staticlib link directive
    // refers to the bare name).
    let canonical = if cfg!(windows) {
        deps_dir.join("llama_harness_inline.lib")
    } else {
        deps_dir.join("libllama_harness_inline.a")
    };
    let _ = std::fs::copy(&archive, &canonical);

    // Emit link directives at the build.rs level so downstream crates
    // (e.g. examples/chat_cli) pick them up via cargo's transitive
    // rustc-link mechanism.
    println!("cargo:rustc-link-search=native={}", deps_dir.display());
    println!("cargo:rustc-link-lib=static=llama_harness_inline");

    println!("cargo:rerun-if-changed={}", ffi_dir.join("src").join("lib.rs").display());
    println!("cargo:rerun-if-changed={}", cpp_dir.join("harness.h").display());
    for entry in walkdir(&cpp_dir) {
        if entry.extension().and_then(|s| s.to_str()) == Some("cpp") {
            println!("cargo:rerun-if-changed={}", entry.display());
        }
    }
}

fn walkdir(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}
