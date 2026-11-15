//! Tier C cross-check pipeline (roadmap §6), the developer-side half:
//!
//! ```text
//! cargo run --example tier_c -- export <dir>
//!     writes <dir>/<case>.xml (MJCF) and <dir>/cases.txt (name end hout)
//! uv run --with mujoco python tests/tier_c/run_mujoco.py <dir> tests/tier_c
//!     MuJoCo (RK4) writes tests/tier_c/<case>.json
//! ```
//!
//! The JSON references are committed; `cargo test` never needs Python.

#[path = "../tests/tier_c/cases.rs"]
mod cases;

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("export") if args.len() == 2 => export(Path::new(&args[1])),
        _ => Err("usage: tier_c export <dir>".to_string()),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn export(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut list = String::new();
    for case in cases::all() {
        let xml = rcad_mbd::write_mjcf(&case.system)
            .map_err(|e| format!("{}: {e}", case.name))?;
        let path = dir.join(format!("{}.xml", case.name));
        std::fs::write(&path, xml).map_err(|e| e.to_string())?;
        list.push_str(&format!("{} {} {}\n", case.name, case.end, case.hout));
        println!("wrote {}", path.display());
    }
    std::fs::write(dir.join("cases.txt"), list).map_err(|e| e.to_string())
}
