//! CLI: evaluate a Decision-Maker request JSON file against a running service and
//! print the response as JSON. Used by scripts/parity_golden.py as the Rust
//! producer in cross-language parity runs.
//!
//! Usage: cargo run -p decisionmaker --example cli -- <baseURL> <request.json>

use std::process;

fn main() {
    let mut args = std::env::args().skip(1);
    let base = match args.next() {
        Some(a) => a,
        None => {
            eprintln!("usage: cli <baseURL> <request.json>");
            process::exit(2);
        }
    };
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: cli <baseURL> <request.json>");
            process::exit(2);
        }
    };

    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("read request {path}: {e}");
        process::exit(1);
    });
    let req: decisionmaker::Request = serde_json::from_slice(&raw).unwrap_or_else(|e| {
        eprintln!("decode request {path}: {e}");
        process::exit(1);
    });

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let resp = rt.block_on(async move { decisionmaker::Client::new(&base).evaluate(&req).await });
    match resp {
        Ok(r) => {
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
        }
        Err(e) => {
            eprintln!("evaluate: {e}");
            process::exit(1);
        }
    }
}
