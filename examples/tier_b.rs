//! Tier B reference pipeline (roadmap §6), the developer-side half:
//!
//! ```text
//! cargo run --example tier_b -- export <dir>
//!     writes <dir>/<case>.asmt for every case in tests/tier_b/cases.rs
//! ondsel_run <dir>/<case>.asmt <dir>/<case>.out.asmt ...
//!     an independent OndselSolver build (see tests/tier_b/README.md)
//! cargo run --example tier_b -- reference <dir> [<source label>]
//!     parses each <case>.out.asmt time series into tests/tier_b/<case>.json
//! ```
//!
//! The JSON files are committed; `cargo test` compares against them and
//! never needs OndselSolver.

#[path = "../tests/tier_b/cases.rs"]
mod cases;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cases::{Reference, PART_SERIES};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("export") if args.len() == 2 => export(Path::new(&args[1])),
        Some("reference") if args.len() >= 2 => reference(
            Path::new(&args[1]),
            args.get(2).map(String::as_str).unwrap_or("OndselSolver"),
        ),
        _ => Err("usage: tier_b export <dir> | tier_b reference <dir> [<source label>]".into()),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn export(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    for case in cases::all() {
        let text = rcad_mbd::write_asmt(&case.system, &case.sim).map_err(|e| e.to_string())?;
        let path = dir.join(format!("{}.asmt", case.name));
        std::fs::write(&path, text).map_err(|e| e.to_string())?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn reference(dir: &Path, source: &str) -> Result<(), String> {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/tier_b");
    for case in cases::all() {
        let path = dir.join(format!("{}.out.asmt", case.name));
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let reference =
            parse_time_series(&text, source).map_err(|e| format!("{}: {e}", path.display()))?;
        let json = serde_json::to_string(&reference).map_err(|e| e.to_string())?;
        let target = out_dir.join(format!("{}.json", case.name));
        std::fs::write(&target, json + "\n").map_err(|e| e.to_string())?;
        println!(
            "wrote {} ({} frames)",
            target.display(),
            reference.times.len()
        );
    }
    Ok(())
}

/// Parse the `TimeSeries` section OndselSolver appends to its output: the
/// `Time` row and every `PartSeries` block (the first value after each label
/// is the authored input, skipped).
fn parse_time_series(text: &str, source: &str) -> Result<Reference, String> {
    let mut lines = text.lines().skip_while(|l| l.trim() != "TimeSeries");
    if lines.next().is_none() {
        return Err("no TimeSeries section (did the kinematic run fail?)".into());
    }
    let row = |line: &str| -> Result<(String, Vec<f64>), String> {
        let mut cells = line.split_whitespace();
        let label = cells.next().unwrap_or_default().to_string();
        let _input = cells.next();
        let values = cells
            .map(|c| c.parse::<f64>().map_err(|e| format!("{label}: {c}: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((label, values))
    };
    let mut times = Vec::new();
    let mut parts: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    let mut current: Option<(String, BTreeMap<String, Vec<f64>>)> = None;
    let flush = |current: &mut Option<(String, BTreeMap<String, Vec<f64>>)>,
                 parts: &mut BTreeMap<String, Vec<Vec<f64>>>|
     -> Result<(), String> {
        if let Some((name, series)) = current.take() {
            let n = series.get("X").map_or(0, Vec::len);
            let mut samples = vec![Vec::with_capacity(18); n];
            for label in PART_SERIES {
                let values = series
                    .get(label)
                    .ok_or(format!("{name}: missing {label}"))?;
                if values.len() != n {
                    return Err(format!(
                        "{name}: {label} has {} values, X has {n}",
                        values.len()
                    ));
                }
                for (sample, v) in samples.iter_mut().zip(values) {
                    sample.push(*v);
                }
            }
            parts.insert(name, samples);
        }
        Ok(())
    };
    for line in lines {
        if line.starts_with("Number") {
            continue;
        }
        if line.starts_with("Time") {
            times = row(line)?.1;
            continue;
        }
        if line.ends_with("Series") || line.contains("Series\t") || line.contains("Series ") {
            flush(&mut current, &mut parts)?;
            if let Some(rest) = line.strip_prefix("PartSeries") {
                let full = rest.trim();
                let name = full.rsplit('/').next().unwrap_or(full).to_string();
                current = Some((name, BTreeMap::new()));
            }
            continue;
        }
        if let Some((_, series)) = current.as_mut() {
            let (label, values) = row(line)?;
            series.insert(label, values);
        }
    }
    flush(&mut current, &mut parts)?;
    if times.is_empty() || parts.is_empty() {
        return Err("TimeSeries without times or parts".into());
    }
    Ok(Reference {
        source: source.to_string(),
        times,
        parts,
    })
}
