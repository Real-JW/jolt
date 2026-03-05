//! Shout-based LUT prover for boolean gate circuits (.czbc / .lczbc).
//! --shout-gate: greedy LUT merging + Shout prover on raw .czbc
//! Default path: load pre-processed .lczbc and run Shout prover directly.

pub mod lut_czbc;
pub mod shout_lut;

use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_bn254::Bn254;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::poly::commitment::hyperkzg::HyperKZG;
use jolt_core::transcripts::{KeccakTranscript, Transcript};

/// Type alias for the PCS we use throughout.
type PCS = HyperKZG<Bn254>;

// ─── bytecode format constants ──────────────────────────────────────────────
const MAGIC: u32 = 0x43425A43;
const VERSION: u16 = 1;
const NOT_SENTINEL: u32 = 0xFFFF_FFFF;

// ─── bytecode parsing helpers ───────────────────────────────────────────────
fn ru8(d: &[u8], o: &mut usize) -> io::Result<u8> {
    let v = *d
        .get(*o)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "eof"))?;
    *o += 1;
    Ok(v)
}
fn ru16(d: &[u8], o: &mut usize) -> io::Result<u16> {
    if *o + 2 > d.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
    }
    let v = u16::from_le_bytes([d[*o], d[*o + 1]]);
    *o += 2;
    Ok(v)
}
fn ru32(d: &[u8], o: &mut usize) -> io::Result<u32> {
    if *o + 4 > d.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "eof"));
    }
    let v = u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    Ok(v)
}

// ─── circuit representation ─────────────────────────────────────────────────
#[derive(Clone, Copy)]
enum OpCode {
    And,
    Or,
    Xor,
    Not,
}

#[derive(Clone, Copy)]
struct Op {
    opcode: OpCode,
    dst: u32,
    a: u32,
    b: u32,
}

pub(crate) struct Circ {
    _num_wires: u32,
    primary_inputs: Vec<u32>,
    registers: Vec<(u32, u32)>,
    outputs: Vec<u32>,
    ops: Vec<Op>,
    default_cycles: u32,
}

fn load_circuit(path: &Path) -> io::Result<Circ> {
    let mut raw = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut raw)?;
    let mut o = 0usize;
    assert_eq!(ru32(&raw, &mut o)?, MAGIC, "bad magic");
    assert_eq!(ru16(&raw, &mut o)?, VERSION, "bad version");
    let _f = ru16(&raw, &mut o)?;
    let num_wires = ru32(&raw, &mut o)?;
    let n_in = ru32(&raw, &mut o)?;
    let n_reg = ru32(&raw, &mut o)?;
    let n_out = ru32(&raw, &mut o)?;
    let n_ops = ru32(&raw, &mut o)?;
    let default_cyc = ru32(&raw, &mut o)?;
    let mut primary_inputs = Vec::with_capacity(n_in as usize);
    for _ in 0..n_in {
        primary_inputs.push(ru32(&raw, &mut o)?);
    }
    let mut registers = Vec::with_capacity(n_reg as usize);
    for _ in 0..n_reg {
        let ro = ru32(&raw, &mut o)?;
        let ri = ru32(&raw, &mut o)?;
        registers.push((ro, ri));
    }
    let mut outputs = Vec::with_capacity(n_out as usize);
    for _ in 0..n_out {
        outputs.push(ru32(&raw, &mut o)?);
    }
    let mut ops = Vec::with_capacity(n_ops as usize);
    for _ in 0..n_ops {
        let op = ru8(&raw, &mut o)?;
        o += 3;
        let dst = ru32(&raw, &mut o)?;
        let a = ru32(&raw, &mut o)?;
        let b = ru32(&raw, &mut o)?;
        let opc = match op {
            1 => OpCode::And,
            2 => OpCode::Or,
            3 => OpCode::Xor,
            4 => OpCode::Not,
            _ => panic!("opcode {op}"),
        };
        ops.push(Op { opcode: opc, dst, a, b });
    }
    Ok(Circ {
        _num_wires: num_wires,
        primary_inputs,
        registers,
        outputs,
        ops,
        default_cycles: default_cyc,
    })
}

/// Convert a raw gate [`Circ`] into a [`lut_czbc::LutCirc`] by **merging**
/// gates via greedy cone-growing (same algorithm as `circuitToLut.py`).
///
/// This reduces the number of LUT ops (and therefore the Shout trace) by
/// ~2× for a typical circuit, matching the performance of the pre-processed
/// `.lczbc` path.
fn gate_circ_to_lut_circ(circ: &Circ, k: usize) -> lut_czbc::LutCirc {
    use std::collections::{BTreeSet, HashMap, HashSet};

    let gates = &circ.ops;
    let n_gates = gates.len();

    // gate_by_dst: wire → gate index that produces it
    let mut gate_by_dst: HashMap<u32, usize> = HashMap::new();
    for (i, g) in gates.iter().enumerate() {
        gate_by_dst.insert(g.dst, i);
    }

    // Boundary wires: primary inputs + register out/in + circuit outputs.
    let mut boundary_wires: HashSet<u32> = HashSet::new();
    for &w in &circ.primary_inputs {
        boundary_wires.insert(w);
    }
    for &(reg_out, reg_in) in &circ.registers {
        boundary_wires.insert(reg_out);
        boundary_wires.insert(reg_in);
    }
    for &w in &circ.outputs {
        boundary_wires.insert(w);
    }

    // Fan-out map: gate_idx → set of gate_idx consuming its output.
    let mut fanout: Vec<Vec<usize>> = vec![Vec::new(); n_gates];
    for (i, g) in gates.iter().enumerate() {
        let fanin_wires: Vec<u32> = if g.b == NOT_SENTINEL {
            vec![g.a]
        } else {
            vec![g.a, g.b]
        };
        for w in fanin_wires {
            if let Some(&prod_idx) = gate_by_dst.get(&w) {
                fanout[prod_idx].push(i);
            }
        }
    }

    // Helper: compute external inputs for a cone.
    let cone_inputs = |cone: &BTreeSet<usize>| -> BTreeSet<u32> {
        let cone_dsts: HashSet<u32> = cone.iter().map(|&gi| gates[gi].dst).collect();
        let mut ext = BTreeSet::new();
        for &gi in cone {
            let g = &gates[gi];
            let fanin: Vec<u32> = if g.b == NOT_SENTINEL {
                vec![g.a]
            } else {
                vec![g.a, g.b]
            };
            for w in fanin {
                if !cone_dsts.contains(&w) {
                    ext.insert(w);
                }
            }
        }
        ext
    };

    // Helper: evaluate a cone's truth table for all 2^k input combos.
    let eval_truth_table =
        |cone: &BTreeSet<usize>, output_wire: u32, inputs: &[u32]| -> Vec<u8> {
            let k_actual = inputs.len();
            let n_entries = 1usize << k_actual;
            // Sort cone by gate index (preserves topo order).
            let cone_sorted: Vec<usize> = cone.iter().copied().collect();

            let mut results = Vec::with_capacity(n_entries);
            for combo in 0..n_entries {
                let mut wire_val: HashMap<u32, bool> = HashMap::new();
                for (bit_pos, &w) in inputs.iter().enumerate() {
                    wire_val.insert(w, (combo >> bit_pos) & 1 == 1);
                }
                for &gi in &cone_sorted {
                    let g = &gates[gi];
                    let a_val = wire_val.get(&g.a).copied().unwrap_or(false);
                    let b_val = if g.b == NOT_SENTINEL {
                        a_val
                    } else {
                        wire_val.get(&g.b).copied().unwrap_or(false)
                    };
                    let out = match g.opcode {
                        OpCode::And => a_val & b_val,
                        OpCode::Or  => a_val | b_val,
                        OpCode::Xor => a_val ^ b_val,
                        OpCode::Not => !a_val,
                    };
                    wire_val.insert(g.dst, out);
                }
                results.push(wire_val.get(&output_wire).copied().unwrap_or(false));
            }

            // Pack into bytes, LSB-first.
            let num_bytes = (n_entries + 7) / 8;
            let mut table = vec![0u8; num_bytes];
            for (i, &r) in results.iter().enumerate() {
                if r {
                    table[i / 8] |= 1 << (i % 8);
                }
            }
            table
        };

    // ── Greedy covering: process gates in reverse topo order ─────────────
    let mut assigned: HashSet<usize> = HashSet::new();
    // (lut_id, k, truth_table_bytes, input_wires, output_wire)
    let mut lut_nodes: Vec<(u32, usize, Vec<u8>, Vec<u32>, u32)> = Vec::new();
    // truth-table key → lut_id
    let mut table_to_id: HashMap<Vec<u8>, u32> = HashMap::new();

    for idx_rev in (0..n_gates).rev() {
        if assigned.contains(&idx_rev) {
            continue;
        }
        let g_root = &gates[idx_rev];
        let mut cone: BTreeSet<usize> = BTreeSet::new();
        cone.insert(idx_rev);

        // Grow the cone greedily.
        let mut changed = true;
        while changed {
            changed = false;
            let ext = cone_inputs(&cone);
            if ext.len() > k {
                break;
            }
            for bw in ext.iter() {
                if boundary_wires.contains(bw) {
                    continue;
                }
                let h_idx = match gate_by_dst.get(bw) {
                    Some(&i) => i,
                    None => continue,
                };
                if assigned.contains(&h_idx) || cone.contains(&h_idx) {
                    continue;
                }
                // Tentatively absorb.
                let mut trial_cone = cone.clone();
                trial_cone.insert(h_idx);
                let trial_ext = cone_inputs(&trial_cone);
                if trial_ext.len() > k {
                    continue;
                }
                // Fan-out safety: all consumers of h must be inside the cone.
                let h_consumers = &fanout[h_idx];
                if h_consumers.iter().any(|ci| !trial_cone.contains(ci)) {
                    continue;
                }
                cone = trial_cone;
                changed = true;
                break; // restart with updated cone
            }
        }

        let ext = cone_inputs(&cone);
        let mut inputs_list: Vec<u32> = ext.into_iter().collect();
        // Pad to exactly k inputs by repeating the last wire.
        if inputs_list.is_empty() {
            inputs_list.push(0);
        }
        while inputs_list.len() < k {
            inputs_list.push(*inputs_list.last().unwrap());
        }

        let truth_bytes = eval_truth_table(&cone, g_root.dst, &inputs_list);

        // Assign stable lut_id based on (truth_table, k).
        let mut key = truth_bytes.clone();
        key.push(inputs_list.len() as u8);
        let next_id = table_to_id.len() as u32;
        let lut_id = *table_to_id.entry(key).or_insert(next_id);

        lut_nodes.push((lut_id, inputs_list.len(), truth_bytes, inputs_list, g_root.dst));
        for &gi in &cone {
            assigned.insert(gi);
        }
    }

    // Reverse to topological order (we built in reverse-topo).
    lut_nodes.reverse();

    // ── Build LutCirc ─────────────────────────────────────────────────────
    let mut lut_types: HashMap<u32, lut_czbc::LutDesc> = HashMap::new();
    let mut ops: Vec<lut_czbc::LutOp> = Vec::with_capacity(lut_nodes.len());
    for &(lut_id, k_val, ref tt, ref inputs, dst) in &lut_nodes {
        lut_types.entry(lut_id).or_insert_with(|| lut_czbc::LutDesc {
            lut_id,
            k: k_val,
            m: 1,
            truth_table: tt.clone(),
        });
        ops.push(lut_czbc::LutOp {
            lut_id,
            dst_wire: dst,
            src_wires: inputs.clone(),
        });
    }

    println!("  gate_circ_to_lut_circ: {} gates → {} LUTs ({} types, k={})",
        n_gates, ops.len(), lut_types.len(), k);

    lut_czbc::LutCirc {
        num_wires: circ._num_wires as usize,
        primary_inputs: circ.primary_inputs.clone(),
        registers: circ.registers.clone(),
        outputs: circ.outputs.clone(),
        lut_types,
        ops,
        default_cycles: circ.default_cycles,
    }
}

fn tiny_circuit() -> Circ {
    Circ {
        _num_wires: 4,
        primary_inputs: vec![0, 1],
        registers: vec![],
        outputs: vec![3],
        ops: vec![
            Op { opcode: OpCode::Not, dst: 2, a: 0, b: NOT_SENTINEL },
            Op { opcode: OpCode::Xor, dst: 3, a: 2, b: 1 },
        ],
        default_cycles: 1,
    }
}

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

enum Mode {
    Tiny,
    Bytecode(PathBuf),
    /// Raw gate circuit (.czbc); auto-converted to LutCirc then run through Shout.
    ShoutGate(PathBuf),
}

fn print_usage(bin: &str) {
    eprintln!("Usage:");
    eprintln!("  {bin} --tiny [input-bits] [--cycles N] [--show-pages] [--bench-csv <file>]");
    eprintln!("  {bin} <bytecode.czbc>  [input-bits] [--cycles N] [--show-pages] [--bench-csv <file>]");
    eprintln!("  {bin} --shout-gate   <circuit.czbc>  [input-bits] [--cycles N] [--bench-csv <file>]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --shout-gate        Phase S* Shout prover on a raw gate circuit (.czbc);");
    eprintln!("                      auto-converts AND/OR/XOR/NOT to LUT types internally.");
    eprintln!("  --bench-csv <file>  Append a CSV row with timing/size metrics to <file>.");
    eprintln!("                      Creates the file (with header) if it does not exist.");
    eprintln!("  --show-pages        Print the gate / LUT trace (gate mode only).");
    eprintln!("  --show-pages        Print the gate trace.");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(2);
    }
    let mut mode: Option<Mode> = None;
    let mut input_bits_raw: Option<String> = None;
    let mut cycles_override: Option<u32> = None;
    let mut show_pages = false;
    let mut bench_csv: Option<PathBuf> = None;
    let mut force_shout_gate = false;
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--shout-gate" => {
                force_shout_gate = true;
                i += 1;
            }
            "--tiny" => {
                mode = Some(Mode::Tiny);
                i += 1;
            }
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
            "--show-pages" => {
                show_pages = true;
                i += 1;
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
                if mode.is_none() {
                    let p = PathBuf::from(token);
                    let is_czbc = force_shout_gate
                        || p.extension().map(|e| e == "czbc").unwrap_or(false);
                    mode = Some(if is_czbc {
                        Mode::ShoutGate(p)
                    } else {
                        Mode::Bytecode(p)
                    });
                    force_shout_gate = false;
                } else if input_bits_raw.is_none() {
                    input_bits_raw = Some(token.to_string());
                } else {
                    eprintln!("unexpected arg: {token}");
                    print_usage(&args[0]);
                    std::process::exit(2);
                }
                i += 1;
            }
        }
    }
    let _ = show_pages; // reserved for future use

    // ── Phase S* (gate): Shout prover directly on a raw .czbc circuit ───────
    // Merges individual gates into k-input LUTs first (greedy cone-growing).
    if let Some(Mode::ShoutGate(ref p)) = mode {
        let label = p.display().to_string();
        let raw_circ = load_circuit(p)
            .unwrap_or_else(|e| panic!("load {}: {e}", p.display()));

        let input_bits: Vec<bool> = input_bits_raw
            .as_deref()
            .map(parse_bit_string)
            .transpose()
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();

        let default_cycles = if raw_circ.default_cycles == 0 { 1 } else { raw_circ.default_cycles };
        let cycles = cycles_override.unwrap_or(default_cycles).max(1);

        let mut inputs = vec![false; raw_circ.primary_inputs.len()];
        for (j, &b) in input_bits.iter().enumerate() {
            if j < inputs.len() { inputs[j] = b; }
        }

        println!("Circuit (Shout on gate circuit — Phase S*, with LUT merging) : {label}");
        println!("  wires      : {}", raw_circ._num_wires);
        println!("  inputs     : {}", raw_circ.primary_inputs.len());
        println!("  regs       : {}", raw_circ.registers.len());
        println!("  gate ops   : {}", raw_circ.ops.len());
        println!("  outputs    : {}", raw_circ.outputs.len());
        println!("  cycles     : {cycles}");

        // ── Gate merging: convert raw Circ → LutCirc (greedy cone-growing) ───
        let t_merge = Instant::now();
        let circ = gate_circ_to_lut_circ(&raw_circ, 4);
        let merge_ms = t_merge.elapsed().as_millis();
        println!("  merge time : {merge_ms} ms");
        println!("  lut ops    : {}", circ.ops.len());
        println!("  lut types  : {}", circ.lut_types.len());

        let mut type_order: Vec<u32> = circ.lut_types.keys().copied().collect();
        type_order.sort();
        let type_index_of: std::collections::HashMap<u32, usize> = type_order
            .iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Validate LUT wrappers.
        println!("\nLUT wrapper validation:");
        for (tid, &lut_id) in type_order.iter().enumerate() {
            let desc = &circ.lut_types[&lut_id];
            for out_bit in 0..desc.m {
                let shout_table = shout_lut::LutShoutTable::from_lut_desc(desc, out_bit);
                let ok = (0..(1usize << desc.k)).all(|idx| {
                    let bit_pos = idx * desc.m + out_bit;
                    shout_table.entry(idx)
                        == ((desc.truth_table[bit_pos / 8] >> (bit_pos % 8)) as u64 & 1)
                });
                if tid < 10 || !ok {
                    println!("  lut_id={lut_id:>4}  type_idx={tid}  k={}  MLE-wrapper-ok={ok}", desc.k);
                }
            }
        }
        println!("✓  LUT wrappers valid ({} types).", type_order.len());

        // Simulate merged LUT circuit.
        use lut_czbc::evaluate_lut_circuit;
        let (trace, final_outputs) = evaluate_lut_circuit(&circ, &inputs, cycles);

        let out_bits: String = final_outputs.iter().map(|&b| if b { '1' } else { '0' }).collect();
        println!("\nCircuit outputs (cycle {cycles}): {out_bits}");
        let all_one  = final_outputs.iter().all(|&b| b);
        let all_zero = final_outputs.iter().all(|&b| !b);
        if all_one  { println!("  WARNING: all outputs are 1 — check gate inputs."); }
        if all_zero { println!("  WARNING: all outputs are 0 — check gate inputs."); }

        let k = circ.lut_types.values().map(|d| d.k).max().unwrap_or(0);
        let n_types = type_order.len();
        let log_k_chunk: usize = 4;
        let params = shout_lut::OneHotParams::new(n_types, k, log_k_chunk);
        let t_total = trace.len().next_power_of_two().max(1);

        println!("\nTrace simulation:");
        println!("  k (uniform input bits) : {k}");
        println!("  n_types                : {n_types}");
        println!("  total_address_bits     : {}", params.total_address_bits);
        println!("  trace rows             : {}  (t_total={t_total})", trace.len());

        let _ = DoryGlobals::initialize(params.k_chunk, t_total);

        let t_s2 = Instant::now();
        let witnesses = shout_lut::build_shout_witnesses(
            &trace, &type_index_of, k, &params, t_total,
        );
        let s2_elapsed = t_s2.elapsed();
        println!("  witnesses built: {} (time: {:.3?})", witnesses.len(), s2_elapsed);
        println!("✓  Trace simulation complete.");

        // Shout prover + verifier.
        let max_num_vars = shout_lut::shout_max_num_vars(n_types, k, cycles, circ.ops.len());
        println!("\nShout prover + verifier:");
        println!("  SRS size   : 2^{max_num_vars} = {} G1 points", 1usize << max_num_vars);

        let t_srs = Instant::now();
        let pk = <PCS as CommitmentScheme>::setup_prover(max_num_vars);
        let vk = <PCS as CommitmentScheme>::setup_verifier(&pk);
        let srs_ms = t_srs.elapsed().as_millis();
        println!("  SRS time   : {srs_ms} ms");

        let mut prove_transcript = KeccakTranscript::new(b"shout-gate");
        prove_transcript.append_u64(circ.ops.len() as u64);
        prove_transcript.append_u64(cycles as u64);
        for &b in &inputs { prove_transcript.append_u64(b as u64); }

        println!("\n  Proving…");
        let t_prove = Instant::now();
        let shout_proof = shout_lut::prove_shout_lut(
            &circ.lut_types, &trace, &type_index_of, k, t_total, &pk, &mut prove_transcript,
        );
        let prove_ms = t_prove.elapsed().as_millis();
        println!("  Prover time: {prove_ms} ms");

        let mut verify_transcript = KeccakTranscript::new(b"shout-gate");
        verify_transcript.append_u64(circ.ops.len() as u64);
        verify_transcript.append_u64(cycles as u64);
        for &b in &inputs { verify_transcript.append_u64(b as u64); }

        println!("\n  Verifying…");
        let t_verify = Instant::now();
        let ok = shout_lut::verify_shout_lut(
            &shout_proof, &circ.lut_types, &vk, &mut verify_transcript,
        );
        let verify_ms = t_verify.elapsed().as_millis();
        println!("  Verifier time: {verify_ms} ms");

        if ok {
            println!("\n✓  Shout proof VALID.");
        } else {
            eprintln!("\n✗  Shout proof INVALID.");
            std::process::exit(2);
        }

        if let Some(ref csv_path) = bench_csv {
            let proof_size  = shout_lut::compute_shout_proof_size_bytes(&shout_proof);
            let srs_size    = 1usize << max_num_vars;
            let total_evals = trace.len();

            let write_header = !csv_path.exists();
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(csv_path)
                .unwrap_or_else(|e| panic!("cannot open {}: {e}", csv_path.display()));

            if write_header {
                writeln!(
                    file,
                    "circuit,gates,cycles,total_evals,max_sumcheck_vars,srs_g1_points,\
                     srs_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,num_lut_types"
                ).expect("write CSV header");
            }
            writeln!(
                file,
                "{label},{},{cycles},{total_evals},{max_num_vars},{srs_size},{srs_ms},{prove_ms},{verify_ms},{proof_size},{n_types}",
                circ.ops.len(),
            ).expect("write CSV row");

            println!("\nBench CSV row appended to: {}", csv_path.display());
            println!("  proof_size_bytes : {proof_size}");
            println!("  srs_g1_points    : {srs_size}");
            println!("  total_evals      : {total_evals}");
        }

        return;
    }


    // ── Bytecode (.lczbc) / Tiny path ───────────────────────────────────────
    let (label, lut_circ) = match mode {
        Some(Mode::Tiny) => {
            let raw = tiny_circuit();
            (String::from("tiny"), gate_circ_to_lut_circ(&raw, 4))
        }
        Some(Mode::Bytecode(ref p)) => {
            let c = lut_czbc::load_lut_circuit(p)
                .unwrap_or_else(|e| panic!("load {}: {e}", p.display()));
            (p.display().to_string(), c)
        }
        Some(Mode::ShoutGate(_)) => unreachable!("handled above"),
        None => {
            eprintln!("missing --tiny or <bytecode>");
            print_usage(&args[0]);
            std::process::exit(2);
        }
    };

    let input_bits: Vec<bool> = input_bits_raw
        .as_deref()
        .map(parse_bit_string)
        .transpose()
        .unwrap_or_else(|e| panic!("{e}"))
        .unwrap_or_default();

    let default_cycles = if lut_circ.default_cycles == 0 { 1 } else { lut_circ.default_cycles };
    let cycles = cycles_override.unwrap_or(default_cycles).max(1);
    let mut inputs = vec![false; lut_circ.primary_inputs.len()];
    for (j, &b) in input_bits.iter().enumerate() {
        if j < inputs.len() { inputs[j] = b; }
    }

    println!("Circuit (Shout LUT prover): {label}");
    println!("  wires     : {}", lut_circ.num_wires);
    println!("  inputs    : {}", lut_circ.primary_inputs.len());
    println!("  regs      : {}", lut_circ.registers.len());
    println!("  lut ops   : {}", lut_circ.ops.len());
    println!("  lut types : {}", lut_circ.lut_types.len());
    println!("  outputs   : {}", lut_circ.outputs.len());
    println!("  cycles    : {cycles}");

    let mut type_order: Vec<u32> = lut_circ.lut_types.keys().copied().collect();
    type_order.sort();
    let type_index_of: std::collections::HashMap<u32, usize> = type_order
        .iter().enumerate().map(|(i, &id)| (id, i)).collect();

    use lut_czbc::evaluate_lut_circuit;
    let (trace, final_outputs) = evaluate_lut_circuit(&lut_circ, &inputs, cycles);

    let out_bits: String = final_outputs.iter().map(|&b| if b { '1' } else { '0' }).collect();
    println!("\nCircuit outputs (cycle {cycles}): {out_bits}");

    let k = lut_circ.lut_types.values().map(|d| d.k).max().unwrap_or(0);
    let n_types = type_order.len();
    let log_k_chunk: usize = 4;
    let params = shout_lut::OneHotParams::new(n_types, k, log_k_chunk);
    let t_total = trace.len().next_power_of_two().max(1);
    let _ = DoryGlobals::initialize(params.k_chunk, t_total);

    let max_num_vars = shout_lut::shout_max_num_vars(n_types, k, cycles, lut_circ.ops.len());
    let pk = <PCS as CommitmentScheme>::setup_prover(max_num_vars);
    let vk = <PCS as CommitmentScheme>::setup_verifier(&pk);

    let mut prove_transcript = KeccakTranscript::new(b"bool-lut");
    prove_transcript.append_u64(lut_circ.ops.len() as u64);
    prove_transcript.append_u64(cycles as u64);
    for &b in &inputs { prove_transcript.append_u64(b as u64); }

    println!("\n  Proving…");
    let t_prove = Instant::now();
    let shout_proof = shout_lut::prove_shout_lut(
        &lut_circ.lut_types, &trace, &type_index_of, k, t_total, &pk, &mut prove_transcript,
    );
    let prove_ms = t_prove.elapsed().as_millis();
    println!("  Prover time: {prove_ms} ms");

    let mut verify_transcript = KeccakTranscript::new(b"bool-lut");
    verify_transcript.append_u64(lut_circ.ops.len() as u64);
    verify_transcript.append_u64(cycles as u64);
    for &b in &inputs { verify_transcript.append_u64(b as u64); }

    println!("\n  Verifying…");
    let t_verify = Instant::now();
    let ok = shout_lut::verify_shout_lut(
        &shout_proof, &lut_circ.lut_types, &vk, &mut verify_transcript,
    );
    let verify_ms = t_verify.elapsed().as_millis();
    println!("  Verifier time: {verify_ms} ms");

    if ok {
        println!("\n✓  Shout proof VALID.");
    } else {
        eprintln!("\n✗  Shout proof INVALID.");
        std::process::exit(2);
    }

    if let Some(ref csv_path) = bench_csv {
        let proof_size = shout_lut::compute_shout_proof_size_bytes(&shout_proof);
        let srs_size = 1usize << max_num_vars;
        let total_evals = trace.len();
        let write_header = !csv_path.exists();
        let mut file = OpenOptions::new()
            .create(true).append(true)
            .open(csv_path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", csv_path.display()));
        if write_header {
            writeln!(
                file,
                "circuit,lut_ops,cycles,total_evals,max_sumcheck_vars,srs_g1_points,\
                 prove_time_ms,verify_time_ms,proof_size_bytes,num_lut_types"
            ).expect("write CSV header");
        }
        writeln!(
            file,
            "{label},{},{cycles},{total_evals},{max_num_vars},{srs_size},{prove_ms},{verify_ms},{proof_size},{n_types}",
            lut_circ.ops.len(),
        ).expect("write CSV row");
        println!("\nBench CSV row appended to: {}", csv_path.display());
    }
}
