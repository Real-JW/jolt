//! Shout lookup proof for boolean gate circuits (`.czbc`) using 4-bit truth-table masks.
//!
//! This example proves that every gate evaluation in a `.czbc` trace is a
//! correct lookup into one of the gate truth tables (AND/OR/XOR/NOT, etc.)
//! using a single Shout mega-table instance.

pub mod czbc;
pub mod shout_gate;

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_bn254::Bn254;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::transcripts::{KeccakTranscript, Transcript};

type PCS = HyperKZG<Bn254>;

fn parse_bit_string(s: &str) -> io::Result<Vec<bool>> {
    let mut bits = Vec::new();
    for ch in s.chars() {
        match ch {
            '0' => bits.push(false),
            '1' => bits.push(true),
            ' ' | ',' | '_' | '-' => {}
            c => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("invalid bit char '{c}'"),
                ))
            }
        }
    }
    Ok(bits)
}

fn print_usage(bin: &str) {
    eprintln!("Usage:");
    eprintln!("  {bin} <circuit.czbc> [input-bits] [--cycles N] [--bench-csv <file>]");
    eprintln!();
    eprintln!("Notes:");
    eprintln!("  - Proves gate truth-table lookups with Shout (no LUT merging).");
    eprintln!("  - input-bits is a 0/1 string matching the circuit's primary inputs.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(2);
    }

    let mut bytecode: Option<PathBuf> = None;
    let mut input_bits_raw: Option<String> = None;
    let mut cycles_override: Option<u32> = None;
    let mut bench_csv: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--cycles" => {
                cycles_override =
                    Some(args[i + 1].parse().unwrap_or_else(|_| panic!("bad --cycles")));
                i += 2;
            }
            "--bench-csv" => {
                if i + 1 >= args.len() {
                    eprintln!("--bench-csv requires a file argument");
                    print_usage(&args[0]);
                    std::process::exit(2);
                }
                bench_csv = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "-h" | "--help" => {
                print_usage(&args[0]);
                std::process::exit(0);
            }
            token if token.starts_with('-') => {
                eprintln!("unknown flag: {token}");
                print_usage(&args[0]);
                std::process::exit(2);
            }
            token => {
                if bytecode.is_none() {
                    bytecode = Some(PathBuf::from(token));
                } else if input_bits_raw.is_none() {
                    input_bits_raw = Some(token.to_string());
                } else {
                    eprintln!("unexpected extra argument: {token}");
                    print_usage(&args[0]);
                    std::process::exit(2);
                }
                i += 1;
            }
        }
    }

    let bytecode = bytecode.expect("circuit path required");
    let circ = czbc::load_circuit(&bytecode).unwrap_or_else(|e| panic!("load {}: {e}", bytecode.display()));
    let cycles = cycles_override.unwrap_or(circ.default_cycles).max(1);

    let inputs = if let Some(s) = input_bits_raw.as_deref() {
        parse_bit_string(s).unwrap_or_else(|e| panic!("bad input bits: {e}"))
    } else {
        vec![false; circ.primary_inputs.len()]
    };

    println!("Circuit: {}", bytecode.display());
    println!("  primary_inputs: {}", circ.primary_inputs.len());
    println!("  cycles        : {cycles}");

    let type_order = czbc::gate_type_order(&circ);
    let n_types = type_order.len();
    if n_types == 0 {
        eprintln!("no gates found in circuit");
        std::process::exit(2);
    }
    println!("  distinct gate masks: {n_types}");

    let type_index_of: HashMap<u8, usize> = type_order
        .iter()
        .enumerate()
        .map(|(i, &m)| (m, i))
        .collect();

    let (trace, final_outputs) = czbc::evaluate_circuit(&circ, &inputs, cycles);
    let out_bits: String = final_outputs.iter().map(|&b| if b { '1' } else { '0' }).collect();
    println!("Circuit outputs (cycle {cycles}): {out_bits}");

    let k = 2usize;
    let t_total = trace.len().next_power_of_two().max(1);
    let max_num_vars = shout_gate::shout_max_num_vars(n_types, k, cycles, trace.len() / cycles as usize);

    println!("\nShout prover + verifier:");
    println!("  trace rows : {} (t_total={t_total})", trace.len());
    println!("  k          : {k}");
    println!("  SRS vars   : {max_num_vars}  (2^{max_num_vars} G1 points)");

    let t_srs = Instant::now();
    let pk = <PCS as CommitmentScheme>::setup_prover(max_num_vars);
    let vk = <PCS as CommitmentScheme>::setup_verifier(&pk);
    let srs_ms = t_srs.elapsed().as_millis();
    println!("  SRS time   : {srs_ms} ms");

    let mut prove_transcript = KeccakTranscript::new(b"bool-shout");
    prove_transcript.append_u64(trace.len() as u64);
    prove_transcript.append_u64(cycles as u64);
    for &b in &inputs {
        prove_transcript.append_u64(b as u64);
    }
    for &m in &type_order {
        prove_transcript.append_u64(m as u64);
    }

    println!("\n  Proving…");
    let t_prove = Instant::now();
    let proof = shout_gate::prove_shout_gate(
        &type_order,
        &trace,
        &type_index_of,
        k,
        t_total,
        &pk,
        &mut prove_transcript,
    );
    let prove_ms = t_prove.elapsed().as_millis();
    println!("  Prover time: {prove_ms} ms");

    let mut verify_transcript = KeccakTranscript::new(b"bool-shout");
    verify_transcript.append_u64(trace.len() as u64);
    verify_transcript.append_u64(cycles as u64);
    for &b in &inputs {
        verify_transcript.append_u64(b as u64);
    }
    for &m in &type_order {
        verify_transcript.append_u64(m as u64);
    }

    println!("\n  Verifying…");
    let t_verify = Instant::now();
    let ok = shout_gate::verify_shout_gate(&proof, &type_order, &vk, &mut verify_transcript);
    let verify_ms = t_verify.elapsed().as_millis();
    println!("  Verifier time: {verify_ms} ms");

    if ok {
        println!("\n✓  Shout proof VALID.");
    } else {
        eprintln!("\n✗  Shout proof INVALID.");
        std::process::exit(2);
    }

    if let Some(ref csv_path) = bench_csv {
        let proof_size = shout_gate::compute_shout_proof_size_bytes(&proof);
        let srs_size = 1usize << max_num_vars;

        let write_header = !csv_path.exists();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(csv_path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", csv_path.display()));

        if write_header {
            writeln!(
                file,
                "circuit,cycles,total_evals,max_sumcheck_vars,srs_g1_points,srs_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,num_gate_types"
            )
            .expect("write CSV header");
        }
        writeln!(
            file,
            "{},{cycles},{},{},{},{srs_ms},{prove_ms},{verify_ms},{proof_size},{n_types}",
            path_label(&bytecode),
            trace.len(),
            max_num_vars,
            srs_size,
        )
        .expect("write CSV row");

        println!("\nBench CSV row appended to: {}", csv_path.display());
        println!("  proof_size_bytes : {proof_size}");
        println!("  srs_g1_points    : {srs_size}");
        println!("  total_evals      : {}", trace.len());
    }
}

fn path_label(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>")
        .to_string()
}
