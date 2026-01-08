//! Build script for caslo-planner
//!
//! This script:
//! 1. Compiles the generated ACADOS C code
//! 2. Generates Rust FFI bindings using bindgen
//!
//! Prerequisites:
//! - ACADOS must be installed and ACADOS_SOURCE_DIR environment variable set
//! - Python code generation script must have been run first:
//!   cd codegen && python caslo_ocp.py --output-dir ../generated

use std::env;
use std::path::PathBuf;

fn main() {
    // Check if ACADOS feature is enabled
    #[cfg(not(feature = "acados"))]
    {
        println!("cargo:warning=ACADOS feature not enabled, skipping solver compilation");
        return;
    }

    #[cfg(feature = "acados")]
    build_acados();
}

#[cfg(feature = "acados")]
fn build_acados() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Get ACADOS installation path
    let acados_dir = env::var("ACADOS_SOURCE_DIR").unwrap_or_else(|_| {
        // Try common locations
        if PathBuf::from("/opt/acados").exists() {
            "/opt/acados".to_string()
        } else if PathBuf::from("/usr/local/acados").exists() {
            "/usr/local/acados".to_string()
        } else {
            let home = env::var("HOME").unwrap_or_default();
            format!("{}/acados", home)
        }
    });

    let acados_path = PathBuf::from(&acados_dir);

    if !acados_path.exists() {
        println!("cargo:warning=ACADOS not found at {}", acados_dir);
        println!("cargo:warning=Set ACADOS_SOURCE_DIR environment variable");
        println!("cargo:warning=Skipping ACADOS solver compilation");
        return;
    }

    // Determine number of quadrotors from environment or default
    let num_quads: usize = env::var("CASLO_NUM_QUADS")
        .unwrap_or_else(|_| "3".to_string())
        .parse()
        .unwrap_or(3);

    let generated_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("generated")
        .join(format!("caslo_{}quad", num_quads));

    if !generated_dir.exists() {
        println!("cargo:warning=Generated ACADOS code not found at {:?}", generated_dir);
        println!("cargo:warning=Run: cd codegen && python caslo_ocp.py --num-quads {}", num_quads);
        println!("cargo:warning=Skipping ACADOS solver compilation");
        return;
    }

    println!("cargo:rerun-if-changed=generated/");
    println!("cargo:rerun-if-env-changed=ACADOS_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=CASLO_NUM_QUADS");

    // Compile generated ACADOS code
    let mut build = cc::Build::new();

    // Add ACADOS include paths
    build.include(acados_path.join("include"));
    build.include(acados_path.join("include/blasfeo/include"));
    build.include(acados_path.join("include/hpipm/include"));
    build.include(&generated_dir);

    // Find and compile all C files in generated directory and subdirectories
    let mut c_files: Vec<PathBuf> = Vec::new();

    // Main directory
    if let Ok(entries) = std::fs::read_dir(&generated_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().map(|e| e == "c").unwrap_or(false) {
                c_files.push(path);
            }
        }
    }

    // Model subdirectory
    let model_dir = generated_dir.join(format!("caslo_{}quad_model", num_quads));
    if model_dir.exists() {
        build.include(&model_dir);
        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "c").unwrap_or(false) {
                    c_files.push(path);
                }
            }
        }
    }

    // Cost subdirectory
    let cost_dir = generated_dir.join(format!("caslo_{}quad_cost", num_quads));
    if cost_dir.exists() {
        build.include(&cost_dir);
        if let Ok(entries) = std::fs::read_dir(&cost_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().map(|e| e == "c").unwrap_or(false) {
                    c_files.push(path);
                }
            }
        }
    }

    if c_files.is_empty() {
        println!("cargo:warning=No C files found in {:?}", generated_dir);
        return;
    }

    for c_file in &c_files {
        build.file(c_file);
        println!("cargo:rerun-if-changed={}", c_file.display());
    }

    build.compile("caslo_acados");

    // Link ACADOS libraries
    println!("cargo:rustc-link-search=native={}/lib", acados_dir);
    println!("cargo:rustc-link-lib=acados");
    println!("cargo:rustc-link-lib=blasfeo");
    println!("cargo:rustc-link-lib=hpipm");

    // Also link math library
    println!("cargo:rustc-link-lib=m");

    // Generate Rust bindings using bindgen
    generate_bindings(&generated_dir, &acados_path, &out_dir, num_quads);
}

#[cfg(feature = "acados")]
fn generate_bindings(
    generated_dir: &PathBuf,
    acados_path: &PathBuf,
    out_dir: &PathBuf,
    num_quads: usize,
) {
    // Find the main header file
    let header_name = format!("caslo_{}quad_model.h", num_quads);
    let main_header = generated_dir.join(&header_name);

    if !main_header.exists() {
        // Try alternative header name
        let alt_header = generated_dir.join("acados_solver_caslo.h");
        if !alt_header.exists() {
            println!("cargo:warning=Header file not found: {:?}", main_header);
            return;
        }
    }

    // Create a wrapper header that includes all necessary headers
    let wrapper_header = out_dir.join("wrapper.h");
    std::fs::write(
        &wrapper_header,
        format!(
            r#"
#include <acados/ocp_nlp/ocp_nlp_common.h>
#include <acados/ocp_nlp/ocp_nlp_solver_common.h>
#include <acados_c/ocp_nlp_interface.h>

// Include generated solver header
#include "acados_solver_caslo_{}quad.h"
"#,
            num_quads
        ),
    )
    .expect("Failed to write wrapper header");

    let bindings = bindgen::Builder::default()
        .header(wrapper_header.to_string_lossy())
        .clang_arg(format!("-I{}", acados_path.join("include").display()))
        .clang_arg(format!(
            "-I{}",
            acados_path.join("include/blasfeo/include").display()
        ))
        .clang_arg(format!(
            "-I{}",
            acados_path.join("include/hpipm/include").display()
        ))
        .clang_arg(format!("-I{}", generated_dir.display()))
        // Whitelist ACADOS types and functions
        .allowlist_type("ocp_nlp_.*")
        .allowlist_function("ocp_nlp_.*")
        .allowlist_function("caslo_.*")
        .allowlist_var("CASLO_.*")
        // Generate proper Rust types
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    let bindings_path = out_dir.join("acados_bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Couldn't write bindings!");

    println!("cargo:warning=Generated bindings at {:?}", bindings_path);
}
