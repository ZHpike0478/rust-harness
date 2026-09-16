use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp_dir = manifest_dir.join("cpp");
    let inc_dir = cpp_dir.clone();

    // cxx_build::bridge points at the Rust file holding #[cxx::bridge].
    let mut build = cxx_build::bridge("src/lib.rs");

    // cxx-generated DTO header is under
    // <out>/cxxbridge/include/llama-harness-ffi/src/. Add it explicitly
    // so StubEngine.cpp can #include "bridge.rs.h".
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cxx_dto_dir = out_dir
        .join("cxxbridge")
        .join("include")
        .join("llama-harness-ffi")
        .join("src");

    build
        .file(cpp_dir.join("HarnessError.cpp"))
        .file(cpp_dir.join("StubEngine.cpp"))
        .include(&inc_dir)
        .include(&cxx_dto_dir)
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .flag_if_supported("/utf-8")
        .flag_if_supported("-Wno-unused-parameter")
        .warnings(false);

    // ----- Optional: link llama.cpp when vendored -----
    let llama_dir = manifest_dir.join("vendor").join("llama.cpp");
    if llama_dir.exists() {
        let llama_inc = llama_dir.join("include");
        let llama_src = llama_dir.join("ggml").join("src");
        println!("cargo:rerun-if-changed={}", llama_inc.display());
        println!("cargo:rerun-if-changed={}", llama_src.display());
        build
            .include(&llama_inc)
            .include(&llama_src)
            .define("LLAMA_HARNESS_REAL_BACKEND", None);
        for entry in walkdir(&llama_src) {
            if entry.extension().and_then(|s| s.to_str()) == Some("cpp") {
                build.file(entry);
            }
        }
        #[cfg(target_os = "linux")]
        {
            println!("cargo:rustc-link-lib=pthread");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
        }
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-lib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    } else {
        println!(
            "cargo:warning=llama.cpp not vendored at vendor/llama.cpp -- \
             building stub engine (no real inference). See docs/VENDORING.md"
        );
    }

    build.compile("llama_harness_ffi");

    // Emit the cxx-generated header directory so downstream build scripts
    // (llama-harness) can compile the same C++ sources against the
    // DTOs cxx generated here. Cargo exposes this as
    // DEP_<links>_<KEY>.
    let cxx_dto_dir = out_dir
        .join("cxxbridge")
        .join("include")
        .join("llama-harness-ffi")
        .join("src");
    println!("cargo:CXX_DTO_DIR={}", cxx_dto_dir.display());

    // Staticlib name + path for downstream linking (fallback).
    println!("cargo:STATICLIB_NAME=llama_harness_ffi");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=llama_harness_ffi");

    // Re-run triggers.
    println!("cargo:rerun-if-changed=src/lib.rs");
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
