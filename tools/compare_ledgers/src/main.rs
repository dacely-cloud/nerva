#![forbid(unsafe_code)]

use std::{env, fs, process::ExitCode};

use nerva_compare_ledgers::{compare_ledgers, json_escape};

fn main() -> ExitCode {
    match run() {
        Ok((json, success)) => {
            println!("{json}");
            if success {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(reason) => {
            eprintln!("{reason}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(String, bool), String> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(usage());
    }

    let mut latency_tolerance_ns = 0i128;
    let mut index = 0usize;
    while index < args.len() {
        if args[index] != "--tolerance-ns" {
            index += 1;
            continue;
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| "--tolerance-ns requires a value".to_string())?;
        latency_tolerance_ns = value
            .parse::<i128>()
            .map_err(|_| "--tolerance-ns must be an integer".to_string())?;
        args.drain(index..=index + 1);
    }

    if args.len() != 2 {
        return Err(usage());
    }

    let baseline_path = &args[0];
    let candidate_path = &args[1];
    let baseline = fs::read_to_string(baseline_path)
        .map_err(|err| format!("failed to read {baseline_path}: {err}"))?;
    let candidate = fs::read_to_string(candidate_path)
        .map_err(|err| format!("failed to read {candidate_path}: {err}"))?;
    let comparison = compare_ledgers(&baseline, &candidate, latency_tolerance_ns);
    let success = comparison.status() == "ok";
    Ok((
        format!(
            "{{\"status\":\"{}\",\"baseline\":\"{}\",\"candidate\":\"{}\",\"latency_tolerance_ns\":{},\"comparison\":{}}}",
            comparison.status(),
            json_escape(baseline_path),
            json_escape(candidate_path),
            latency_tolerance_ns.max(0),
            comparison.to_json(),
        ),
        success,
    ))
}

fn usage() -> String {
    "usage: cargo run -p nerva-compare-ledgers -- [--tolerance-ns N] baseline.json candidate.json"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_mentions_tolerance() {
        assert!(usage().contains("--tolerance-ns"));
    }
}
