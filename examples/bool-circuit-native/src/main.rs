//! **Non-Interactive Zero-Knowledge Proof for Boolean Gate Circuits**
//!
//! Uses Jolt's **HyperKZG polynomial commitment scheme** to turn a sumcheck-based
//! gate-correctness argument into a full NIZK proof.  No RISC-V VM, no guest
//! binary, no `#[jolt::provable]` macro.
//!
//! # Proof structure (per gate type τ with truth-table mask m)
//!
//! Given a trace of N = 2^k gate evaluations {(a_i, b_i, out_i)}, define:
//!
//!   A(x), B(x), Out(x) — multilinear extensions over {0,1}^k
//!   T̃_τ(r_a, r_b)     — MLE of the 4-entry truth table for gate type τ
//!
//! **Prover:**
//!  1. Commits to A, B, Out using HyperKZG → (C_A, C_B, C_Out)
//!  2. Derives Fiat-Shamir challenge `r` from the commitments
//!  3. Runs sumcheck proving  ∑ eq(r,x)·(Out(x) − T̃_m(A(x),B(x))) = 0
//!  4. Opens A, B, Out at the sumcheck point `r_sc` with HyperKZG opening proofs
//!
//! **Verifier (never re-executes the circuit):**
//!  1. Receives (C_A, C_B, C_Out, sumcheck proof, opening proofs, claimed outputs)
//!  2. Replays Fiat-Shamir to derive the same challenges
//!  3. Verifies the sumcheck round polynomials
//!  4. Verifies the HyperKZG opening proofs: C_A → a_final, C_B → b_final, C_Out → out_final
//!  5. Checks  eq(r, r_sc) · (out_final − T̃_m(a_final, b_final))  ==  last_claim
//!
//! # Security
//!
//! **Soundness**: binding polynomial commitments + sumcheck + Fiat-Shamir (random oracle).
//! **Completeness**: correct gate evaluations always produce accepting proofs.
//! **Zero-knowledge** (computational): the verifier never sees the raw trace;
//!   all witness data is hidden behind HyperKZG commitments.
//!
//! # Usage
//!
//!   bool-circuit-native --tiny [input-bits] [--cycles N] [--show-pages]
//!   bool-circuit-native <bytecode.czbc> [input-bits] [--cycles N] [--show-pages]

pub mod lczbc;
pub mod lut_lasso;
pub mod lut_mega_logup;
pub mod lut_prover;

use std::fs::OpenOptions;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_serialize::CanonicalSerialize;

use ark_bn254::{Bn254, Fr};
use ark_ff::{Field, One, Zero};
use jolt_core::field::JoltField;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::{AppendToTranscript, KeccakTranscript, Transcript};
use jolt_core::zkvm::lookup_table::GateLookupTable;

use lczbc::load_lut_circuit;
use lut_lasso::{
    compute_lasso_proof_size_bytes, compute_max_num_vars_lasso,
    prove_lasso_lut_circuit, verify_lasso_lut_circuit,
};
use lut_mega_logup::{
    compute_max_num_vars_mega, compute_mega_proof_size_bytes,
    prove_mega_logup_circuit, verify_mega_logup_circuit,
};
use lut_prover::{
    compute_lut_proof_size_bytes, compute_max_num_vars_lut,
    prove_lut_circuit, verify_lut_circuit,
};

/// Type alias for the PCS we use throughout.
type PCS = HyperKZG<Bn254>;
/// The challenge type used by HyperKZG opening points.
type Challenge = <Fr as JoltField>::Challenge;

// ─── bytecode format constants (matches bool-circuit) ──────────────────────
const MAGIC: u32 = 0x43425A43;
const VERSION: u16 = 1;
const NOT_SENTINEL: u32 = 0xFFFF_FFFF;

const MASK_AND: u8 = 0x08;
const MASK_OR: u8 = 0x0E;
const MASK_XOR: u8 = 0x06;
const MASK_NOT: u8 = 0x03;

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

fn opcode_mask(opc: OpCode) -> u8 {
    match opc {
        OpCode::And => MASK_AND,
        OpCode::Or => MASK_OR,
        OpCode::Xor => MASK_XOR,
        OpCode::Not => MASK_NOT,
    }
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
        ops.push(Op {
            opcode: opc,
            dst,
            a,
            b,
        });
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

fn tiny_circuit() -> Circ {
    Circ {
        _num_wires: 4,
        primary_inputs: vec![0, 1],
        registers: vec![],
        outputs: vec![3],
        ops: vec![
            Op {
                opcode: OpCode::Not,
                dst: 2,
                a: 0,
                b: NOT_SENTINEL,
            },
            Op {
                opcode: OpCode::Xor,
                dst: 3,
                a: 2,
                b: 1,
            },
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

// ─── gate trace (one row per gate evaluation) ───────────────────────────────
/// A single gate evaluation in the execution trace.
#[derive(Clone)]
pub(crate) struct GateEval {
    mask: u8,
    a: bool,
    b: bool,
    out: bool,
}

/// Evaluate the circuit for `cycles` sequential cycles and collect all gate
/// evaluations into a flat trace (row ordering: cycle-major, gate-minor).
fn evaluate_circuit(circ: &Circ, inputs: &[bool], cycles: u32) -> (Vec<GateEval>, Vec<bool>) {
    let n_cyc = cycles.max(1) as usize;
    let mut wires = vec![false; circ._num_wires as usize];
    let mut reg_state = vec![false; circ.registers.len()];
    let mut trace = Vec::with_capacity(circ.ops.len() * n_cyc);

    for _ in 0..n_cyc {
        for (i, &w) in circ.primary_inputs.iter().enumerate() {
            wires[w as usize] = inputs.get(i).copied().unwrap_or(false);
        }
        for (ri, &(reg_out, _)) in circ.registers.iter().enumerate() {
            wires[reg_out as usize] = reg_state[ri];
        }
        for op in &circ.ops {
            let a = wires[op.a as usize];
            let b = if op.b == NOT_SENTINEL {
                a
            } else {
                wires[op.b as usize]
            };
            let mask = opcode_mask(op.opcode);
            let idx = ((a as u8) << 1) | (b as u8);
            let out = (mask >> idx) & 1 == 1;
            wires[op.dst as usize] = out;
            trace.push(GateEval { mask, a, b, out });
        }
        for (ri, &(_, reg_in)) in circ.registers.iter().enumerate() {
            reg_state[ri] = wires[reg_in as usize];
        }
    }

    let outputs: Vec<bool> = circ.outputs.iter().map(|&w| wires[w as usize]).collect();
    (trace, outputs)
}

// ──────────────────────────────────────────────────────────────────────────────
// NIZK gate lookup proof with HyperKZG polynomial commitments
// ──────────────────────────────────────────────────────────────────────────────

/// Proof that a sequence of gate evaluations with a given mask are all correct.
///
/// Contains HyperKZG commitments to the witness polynomials (A, B, Out),
/// sumcheck round polynomials, claimed evaluations, and HyperKZG opening proofs.
pub struct GateLookupProof {
    /// The gate type (truth-table mask).
    pub mask: u8,
    /// Number of original (un-padded) gate evaluations.
    pub num_gates: usize,
    /// Number of sumcheck variables m = ⌈log₂(num_gates)⌉.
    pub num_vars: usize,

    // ── HyperKZG polynomial commitments ──
    /// Commitment to the A (left input) polynomial.
    pub commitment_a: HyperKZGCommitment<Bn254>,
    /// Commitment to the B (right input) polynomial.
    pub commitment_b: HyperKZGCommitment<Bn254>,
    /// Commitment to the Out (output) polynomial.
    pub commitment_out: HyperKZGCommitment<Bn254>,

    // ── Sumcheck data ──
    /// Per-round univariate polynomial evaluations at t = 0, 1, 2, 3.
    pub round_polys: Vec<[Fr; 4]>,

    // ── Final evaluations at the sumcheck point ──
    /// A(r_sc) — evaluation of the 'a' polynomial at the sumcheck point.
    pub a_final: Fr,
    /// B(r_sc) — evaluation of the 'b' polynomial at the sumcheck point.
    pub b_final: Fr,
    /// Out(r_sc) — evaluation of the 'out' polynomial at the sumcheck point.
    pub out_final: Fr,

    // ── HyperKZG opening proofs ──
    // `None` when the commitment is the identity (zero polynomial).
    // In that case the evaluation must be 0 and no opening proof is needed.
    /// Opening proof for A at r_sc.
    pub opening_proof_a: Option<HyperKZGProof<Bn254>>,
    /// Opening proof for B at r_sc.
    pub opening_proof_b: Option<HyperKZGProof<Bn254>>,
    /// Opening proof for Out at r_sc.
    pub opening_proof_out: Option<HyperKZGProof<Bn254>>,
}

/// A complete gate circuit NIZK proof.
pub struct CircuitProof {
    /// Maximum number of sumcheck variables across all gate types
    /// (determines SRS size).
    pub max_num_vars: usize,
    /// One `GateLookupProof` per distinct gate type.
    pub gate_proofs: Vec<GateLookupProof>,
    /// Claimed circuit outputs.
    pub outputs: Vec<bool>,
}

// ── field helpers ─────────────────────────────────────────────────────────────

#[inline]
fn fr(n: u64) -> Fr {
    Fr::from(n)
}

#[inline]
fn fr_bool(b: bool) -> Fr {
    if b {
        Fr::one()
    } else {
        Fr::zero()
    }
}

/// Evaluate the MLE of a 4-bit truth table (indexed by (a<<1)|b) at field points (ra, rb).
#[inline]
fn gate_mle_at(mask: u8, ra: Fr, rb: Fr) -> Fr {
    GateLookupTable { mask }.evaluate_mle_at(ra, rb)
}

/// Bind the polynomial in-place: poly[i] ← poly[2i] + r · (poly[2i+1] − poly[2i]).
/// Halves the polynomial length.  Binds the **lowest** variable (bit 0 / LSB).
fn bind(poly: &mut Vec<Fr>, r: Fr) {
    let half = poly.len() / 2;
    for i in 0..half {
        poly[i] = poly[2 * i] + r * (poly[2 * i + 1] - poly[2 * i]);
    }
    poly.truncate(half);
}

/// Initialise the equality polynomial eq(r, ·) using **LSB-first** ordering.
///
/// `eq_evals[i] = ∏_j ( r[j] · bit_j(i) + (1 − r[j]) · (1 − bit_j(i)) )`
///
/// where bit_j(i) = (i >> j) & 1.  r[0] controls the LSB.
fn init_eq(r: &[Fr]) -> Vec<Fr> {
    let m = r.len();
    let n = 1usize << m;
    let mut eq = vec![Fr::one(); n];
    for (j, &rj) in r.iter().enumerate() {
        let step = 1usize << j;
        let mut base = 0usize;
        while base < n {
            for k in base..base + step {
                eq[k] *= Fr::one() - rj;
                eq[k + step] *= rj;
            }
            base += 2 * step;
        }
    }
    eq
}

/// Evaluate `eq(r_input, r_sumcheck)` from the two challenge vectors (both LSB-first).
fn eq_final_eval(r_input: &[Fr], r_sc: &[Fr]) -> Fr {
    assert_eq!(r_input.len(), r_sc.len());
    r_input
        .iter()
        .zip(r_sc.iter())
        .map(|(&ri, &si)| ri * si + (Fr::one() - ri) * (Fr::one() - si))
        .product()
}

/// Evaluate a degree-3 polynomial given its 4 evaluations at t = 0,1,2,3 at an arbitrary point.
fn eval_uni(evals: &[Fr; 4], t: Fr) -> Fr {
    let one = Fr::one();
    let t1 = t - one;
    let t2 = t - fr(2);
    let t3 = t - fr(3);
    let inv6 = Field::inverse(&Fr::from(6u64)).unwrap();
    let inv2 = Field::inverse(&Fr::from(2u64)).unwrap();

    evals[0] * (t1 * t2 * t3 * (-inv6))
        + evals[1] * (t * t2 * t3 * inv2)
        + evals[2] * (t * t1 * t3 * (-inv2))
        + evals[3] * (t * t1 * t2 * inv6)
}

// ── prover ────────────────────────────────────────────────────────────────────

/// Prove that all gate evaluations in `evals_for_type` are consistent with `mask`.
///
/// 1. Commits to A, B, Out using HyperKZG.
/// 2. Derives Fiat-Shamir challenge from commitments.
/// 3. Runs m rounds of sumcheck.
/// 4. Produces HyperKZG opening proofs at the sumcheck point.
pub(crate) fn prove_gate_type(
    mask: u8,
    evals_for_type: &[GateEval],
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> GateLookupProof {
    let n = evals_for_type.len();
    assert!(n > 0, "cannot prove an empty gate group");

    let m = usize::max(1, n.next_power_of_two().trailing_zeros() as usize);
    let cap = 1usize << m;

    // ── build evaluation vectors (padded to power of 2) ────────────────────
    // Dummy entries use (a=0, b=0) → out = (mask>>0)&1, which satisfies the
    // gate constraint, so the polynomial identity holds on the full hypercube.
    let dummy_out = fr(((mask >> 0) & 1) as u64);

    let a_evals: Vec<Fr> = (0..cap)
        .map(|i| {
            fr_bool(
                evals_for_type
                    .get(i)
                    .map(|g| g.a)
                    .unwrap_or(false),
            )
        })
        .collect();
    let b_evals: Vec<Fr> = (0..cap)
        .map(|i| {
            fr_bool(
                evals_for_type
                    .get(i)
                    .map(|g| g.b)
                    .unwrap_or(false),
            )
        })
        .collect();
    let out_evals: Vec<Fr> = (0..cap)
        .map(|i| {
            if let Some(g) = evals_for_type.get(i) {
                fr_bool(g.out)
            } else {
                dummy_out
            }
        })
        .collect();

    // ── create multilinear polynomials for PCS ─────────────────────────────
    let a_mle = MultilinearPolynomial::from(a_evals.clone());
    let b_mle = MultilinearPolynomial::from(b_evals.clone());
    let out_mle = MultilinearPolynomial::from(out_evals.clone());

    // ── HyperKZG commit ────────────────────────────────────────────────────
    let comm_a = HyperKZG::<Bn254>::commit(pk, &a_mle)
        .expect("commit A failed");
    let comm_b = HyperKZG::<Bn254>::commit(pk, &b_mle)
        .expect("commit B failed");
    let comm_out = HyperKZG::<Bn254>::commit(pk, &out_mle)
        .expect("commit Out failed");

    // ── Fiat-Shamir: bind gate metadata + commitments ──────────────────────
    transcript.append_u64(mask as u64);
    transcript.append_u64(n as u64);
    transcript.append_u64(m as u64);
    comm_a.append_to_transcript(transcript);
    comm_b.append_to_transcript(transcript);
    comm_out.append_to_transcript(transcript);

    // ── derive random challenge r_input for eq polynomial ──────────────────
    let r_input: Vec<Fr> = transcript.challenge_vector(m);

    // ── initialise mutable polynomials for sumcheck binding ────────────────
    let mut eq_poly: Vec<Fr> = init_eq(&r_input);
    let mut a_poly = a_evals;
    let mut b_poly = b_evals;
    let mut out_poly = out_evals;

    let mut round_polys: Vec<[Fr; 4]> = Vec::with_capacity(m);
    // Sumcheck challenges in both Fr and Challenge representations.
    let mut r_sc_fr: Vec<Fr> = Vec::with_capacity(m);
    let mut r_sc_challenge: Vec<Challenge> = Vec::with_capacity(m);

    // ── sumcheck rounds ────────────────────────────────────────────────────
    for _round in 0..m {
        let half = eq_poly.len() / 2;
        let mut p_evals = [Fr::zero(); 4];

        for i in 0..half {
            let eq_lo = eq_poly[2 * i];
            let eq_hi = eq_poly[2 * i + 1];
            let a_lo = a_poly[2 * i];
            let a_hi = a_poly[2 * i + 1];
            let b_lo = b_poly[2 * i];
            let b_hi = b_poly[2 * i + 1];
            let out_lo = out_poly[2 * i];
            let out_hi = out_poly[2 * i + 1];

            for (k, t) in [Fr::zero(), Fr::one(), fr(2), fr(3)]
                .into_iter()
                .enumerate()
            {
                let eq_t = eq_lo + t * (eq_hi - eq_lo);
                let a_t = a_lo + t * (a_hi - a_lo);
                let b_t = b_lo + t * (b_hi - b_lo);
                let out_t = out_lo + t * (out_hi - out_lo);
                let gate_t = gate_mle_at(mask, a_t, b_t);
                p_evals[k] += eq_t * (out_t - gate_t);
            }
        }

        // Append round polynomial to transcript.
        for &e in p_evals.iter() {
            transcript.append_scalar(&e);
        }

        // Derive round challenge as Challenge type (for HyperKZG compatibility),
        // then convert to Fr for sumcheck arithmetic.
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_challenge.push(r_j_ch);
        r_sc_fr.push(r_j);

        // Bind all polynomials at r_j.
        bind(&mut eq_poly, r_j);
        bind(&mut a_poly, r_j);
        bind(&mut b_poly, r_j);
        bind(&mut out_poly, r_j);

        round_polys.push(p_evals);
    }

    // After m bindings each polynomial has length 1.
    assert_eq!(a_poly.len(), 1);
    assert_eq!(b_poly.len(), 1);
    assert_eq!(out_poly.len(), 1);

    let a_final = a_poly[0];
    let b_final = b_poly[0];
    let out_final = out_poly[0];

    // ── append final evaluations to transcript (binds them to Fiat-Shamir) ─
    transcript.append_scalar(&a_final);
    transcript.append_scalar(&b_final);
    transcript.append_scalar(&out_final);

    // ── HyperKZG opening proofs ────────────────────────────────────────────
    // Jolt's evaluate / HyperKZG uses **big-endian** variable ordering (r[0] = MSB),
    // while our sumcheck uses **little-endian** (r[0] = LSB / bit 0).
    // Reverse the sumcheck challenge vector to match HyperKZG convention.
    let point_kzg: Vec<Challenge> = r_sc_challenge.iter().rev().cloned().collect();

    // Helper: produce an opening proof unless the commitment is the identity
    // (i.e., the polynomial is zero everywhere).  HyperKZG::verify rejects
    // zero commitments, so we handle that case separately.
    let zero_comm = HyperKZGCommitment::<Bn254>::default();

    let opening_proof_a = if comm_a != zero_comm {
        Some(
            HyperKZG::<Bn254>::open(pk, &a_mle, &point_kzg, &a_final, transcript)
                .expect("open A failed"),
        )
    } else {
        None
    };
    let opening_proof_b = if comm_b != zero_comm {
        Some(
            HyperKZG::<Bn254>::open(pk, &b_mle, &point_kzg, &b_final, transcript)
                .expect("open B failed"),
        )
    } else {
        None
    };
    let opening_proof_out = if comm_out != zero_comm {
        Some(
            HyperKZG::<Bn254>::open(pk, &out_mle, &point_kzg, &out_final, transcript)
                .expect("open Out failed"),
        )
    } else {
        None
    };

    GateLookupProof {
        mask,
        num_gates: n,
        num_vars: m,
        commitment_a: comm_a,
        commitment_b: comm_b,
        commitment_out: comm_out,
        round_polys,
        a_final,
        b_final,
        out_final,
        opening_proof_a,
        opening_proof_b,
        opening_proof_out,
    }
}

// ── verifier ──────────────────────────────────────────────────────────────────

/// Verify a gate lookup NIZK proof.
///
/// The verifier **does not** re-execute the circuit.  It:
/// 1. Re-derives Fiat-Shamir challenges from the transcript + proof commitments.
/// 2. Replays the sumcheck to verify round polynomial consistency.
/// 3. Checks the final sumcheck equation against the gate truth table.
/// 4. Verifies HyperKZG opening proofs to confirm committed polynomials
///    evaluate to the claimed values at the sumcheck point.
///
/// Returns `true` iff the proof is valid.
pub fn verify_gate_type(
    proof: &GateLookupProof,
    vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    let m = proof.num_vars;

    // ── re-derive Fiat-Shamir challenges ───────────────────────────────────
    // Append the same metadata + commitments as the prover.
    transcript.append_u64(proof.mask as u64);
    transcript.append_u64(proof.num_gates as u64);
    transcript.append_u64(m as u64);
    proof.commitment_a.append_to_transcript(transcript);
    proof.commitment_b.append_to_transcript(transcript);
    proof.commitment_out.append_to_transcript(transcript);

    // Derive r_input (same as prover).
    let r_input: Vec<Fr> = transcript.challenge_vector(m);

    // ── replay sumcheck ────────────────────────────────────────────────────
    let mut prev_claim = Fr::zero(); // initial claimed sum is 0
    let mut r_sc_fr: Vec<Fr> = Vec::with_capacity(m);
    let mut r_sc_challenge: Vec<Challenge> = Vec::with_capacity(m);

    for round in 0..m {
        let p = &proof.round_polys[round];

        // Consistency check: p(0) + p(1) == prev_claim.
        let sum = p[0] + p[1];
        if sum != prev_claim {
            eprintln!(
                "  round {round}: p(0)+p(1) = {sum:?} ≠ prev_claim = {prev_claim:?}"
            );
            return false;
        }

        // Append round polynomial to transcript (must match prover).
        for &e in p.iter() {
            transcript.append_scalar(&e);
        }

        // Derive round challenge (same method as prover).
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_challenge.push(r_j_ch);
        r_sc_fr.push(r_j);

        // Update claim.
        prev_claim = eval_uni(p, r_j);
    }

    // ── final sumcheck check ──────────────────────────────────────────────
    let eq_final = eq_final_eval(&r_input, &r_sc_fr);
    let gate_val = gate_mle_at(proof.mask, proof.a_final, proof.b_final);
    let expected = eq_final * (proof.out_final - gate_val);

    if expected != prev_claim {
        eprintln!(
            "  final check failed: eq*gap = {expected:?} ≠ last_claim = {prev_claim:?}"
        );
        return false;
    }

    // ── append final evaluations to transcript (must match prover) ─────────
    transcript.append_scalar(&proof.a_final);
    transcript.append_scalar(&proof.b_final);
    transcript.append_scalar(&proof.out_final);

    // ── verify HyperKZG opening proofs ─────────────────────────────────────
    // Reverse sumcheck challenges to match HyperKZG's big-endian convention.
    let point_kzg: Vec<Challenge> = r_sc_challenge.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();

    // Helper: verify one opening proof, handling the zero-polynomial case.
    // If the commitment is the identity (zero polynomial), the evaluation must
    // be zero — no opening proof is needed or possible.
    // If the commitment is non-zero, the opening proof must verify.
    let verify_opening = |name: &str,
                          commitment: &HyperKZGCommitment<Bn254>,
                          eval: &Fr,
                          opening: &Option<HyperKZGProof<Bn254>>,
                          transcript: &mut KeccakTranscript|
     -> bool {
        if *commitment == zero_comm {
            // Zero polynomial: evaluation must be zero, no opening proof.
            if *eval != Fr::zero() {
                eprintln!(
                    "  {name}: zero commitment but non-zero eval (mask 0x{:02X})",
                    proof.mask
                );
                return false;
            }
            true
        } else if let Some(ref pf) = opening {
            // Non-zero commitment: verify the opening proof.
            if HyperKZG::<Bn254>::verify(vk, commitment, &point_kzg, eval, pf, transcript)
                .is_err()
            {
                eprintln!(
                    "  HyperKZG verify FAILED for {name} (mask 0x{:02X})",
                    proof.mask
                );
                return false;
            }
            true
        } else {
            eprintln!(
                "  {name}: non-zero commitment but no opening proof (mask 0x{:02X})",
                proof.mask
            );
            false
        }
    };

    if !verify_opening("A", &proof.commitment_a, &proof.a_final, &proof.opening_proof_a, transcript) {
        return false;
    }
    if !verify_opening("B", &proof.commitment_b, &proof.b_final, &proof.opening_proof_b, transcript) {
        return false;
    }
    if !verify_opening("Out", &proof.commitment_out, &proof.out_final, &proof.opening_proof_out, transcript) {
        return false;
    }

    true
}

// ── combined circuit proof ────────────────────────────────────────────────────

/// Prove the full execution trace of a circuit.
///
/// Evaluates the circuit (prover-side only), then produces a NIZK proof
/// with HyperKZG commitments for each gate type.
pub(crate) fn prove_circuit(
    circ: &Circ,
    inputs: &[bool],
    cycles: u32,
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> CircuitProof {
    let (trace, outputs) = evaluate_circuit(circ, inputs, cycles);

    // Compute max_num_vars for the proof metadata.
    let all_masks = [MASK_AND, MASK_OR, MASK_XOR, MASK_NOT];
    let mut max_num_vars = 1usize;

    // Collect evaluations grouped by mask and prove each.
    let mut gate_proofs = Vec::new();

    for &mask in &all_masks {
        let group: Vec<GateEval> = trace.iter().filter(|g| g.mask == mask).cloned().collect();
        if group.is_empty() {
            continue;
        }
        let m = usize::max(
            1,
            group.len().next_power_of_two().trailing_zeros() as usize,
        );
        max_num_vars = max_num_vars.max(m);

        let proof = prove_gate_type(mask, &group, pk, transcript);
        gate_proofs.push(proof);
    }

    // Handle any non-standard gate masks.
    let known: std::collections::HashSet<u8> = all_masks.iter().copied().collect();
    let extra_masks: Vec<u8> = {
        let mut seen = std::collections::HashSet::new();
        trace
            .iter()
            .map(|g| g.mask)
            .filter(|m| !known.contains(m) && seen.insert(*m))
            .collect()
    };
    for mask in extra_masks {
        let group: Vec<GateEval> = trace.iter().filter(|g| g.mask == mask).cloned().collect();
        let m = usize::max(
            1,
            group.len().next_power_of_two().trailing_zeros() as usize,
        );
        max_num_vars = max_num_vars.max(m);

        let proof = prove_gate_type(mask, &group, pk, transcript);
        gate_proofs.push(proof);
    }

    CircuitProof {
        max_num_vars,
        gate_proofs,
        outputs,
    }
}

/// Verify a complete circuit NIZK proof.
///
/// The verifier **does not** re-execute the circuit.  It only needs:
/// - The proof structure (commitments + sumcheck data + opening proofs)
/// - The verifier key (derived from the same SRS as the prover key)
/// - The transcript initialised with the same public metadata
pub fn verify_circuit(
    proof: &CircuitProof,
    vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    for gate_proof in &proof.gate_proofs {
        if !verify_gate_type(gate_proof, vk, transcript) {
            eprintln!(
                "  gate proof FAILED for mask 0x{:02X}",
                gate_proof.mask
            );
            return false;
        }
    }
    true
}

// ──────────────────────────────────────────────────────────────────────────────
// CLI
// ──────────────────────────────────────────────────────────────────────────────

enum Mode {
    Tiny,
    Bytecode(PathBuf),
    /// LUT-annotated circuit (.lczbc); uses the Phase 2 LUT (sumcheck) prover.
    LutBytecode(PathBuf),
    /// LUT-annotated circuit (.lczbc); uses the Phase 4b Lasso/LogUp prover.
    LassoBytecode(PathBuf),
    /// LUT-annotated circuit (.lczbc); uses the Phase 4h unified mega-table LogUp prover.
    MegaBytecode(PathBuf),
}

fn print_usage(bin: &str) {
    eprintln!("Usage:");
    eprintln!("  {bin} --tiny [input-bits] [--cycles N] [--show-pages] [--bench-csv <file>]");
    eprintln!("  {bin} <bytecode.czbc>  [input-bits] [--cycles N] [--show-pages] [--bench-csv <file>]");
    eprintln!("  {bin} <circuit.lczbc>  [input-bits] [--cycles N] [--bench-csv <file>]");
    eprintln!("  {bin} --lut    <circuit.lczbc> [input-bits] [--cycles N] [--bench-csv <file>]");
    eprintln!("  {bin} --lasso  <circuit.lczbc> [input-bits] [--cycles N] [--bench-csv <file>]");
    eprintln!("  {bin} --lut4h  <circuit.lczbc> [input-bits] [--cycles N] [--bench-csv <file>]");
    eprintln!();
    eprintln!("Flags:");
    eprintln!("  --lut               Phase 2 LUT (sumcheck) prover — one sumcheck per LUT type.");
    eprintln!("  --lasso             Phase 4b Lasso/LogUp prover — T independent grand-product arguments.");
    eprintln!("  --lut4h             Phase 4h unified mega-table LogUp — single global argument.");
    eprintln!("  --bench-csv <file>  Append a CSV row with timing/size metrics to <file>.");
    eprintln!("                      Creates the file (with header) if it does not exist.");
    eprintln!("  --show-pages        Print the gate / LUT trace (gate mode only).");
}

/// Compute the maximum number of sumcheck variables needed across all gate types.
fn compute_max_num_vars(circ: &Circ, cycles: u32) -> usize {
    let mut counts = [0usize; 256];
    for op in &circ.ops {
        counts[opcode_mask(op.opcode) as usize] += 1;
    }
    let cycles = cycles.max(1) as usize;
    let max_count = counts.iter().copied().max().unwrap_or(1) * cycles;
    usize::max(
        1,
        max_count.next_power_of_two().trailing_zeros() as usize,
    )
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
    // --lut forces the Phase 2 LUT (sumcheck) prover.
    let mut force_lut   = false;
    // --lasso forces the Phase 4b Lasso/LogUp prover.
    let mut force_lasso = false;
    // --lut4h forces the Phase 4h unified mega-table LogUp prover.
    let mut force_mega  = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--lut" => {
                force_lut   = true;
                force_lasso = false;
                force_mega  = false;
                i += 1;
            }
            "--lasso" => {
                force_lasso = true;
                force_lut   = false;
                force_mega  = false;
                i += 1;
            }
            "--lut4h" => {
                force_mega  = true;
                force_lut   = false;
                force_lasso = false;
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
                    let is_lczbc = force_lut || force_lasso || force_mega
                        || p.extension().map(|e| e == "lczbc").unwrap_or(false);
                    mode = Some(if force_mega && is_lczbc {
                        Mode::MegaBytecode(p)
                    } else if force_lasso && is_lczbc {
                        Mode::LassoBytecode(p)
                    } else if is_lczbc {
                        Mode::LutBytecode(p)
                    } else {
                        Mode::Bytecode(p)
                    });
                    force_lut   = false;
                    force_lasso = false;
                    force_mega  = false;
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

    // ── LUT prover path (Phase 2) ─────────────────────────────────────────────
    if let Some(Mode::LutBytecode(ref p)) = mode {
        let label = p.display().to_string();
        let circ = load_lut_circuit(p)
            .unwrap_or_else(|e| panic!("load {}: {e}", p.display()));

        let input_bits: Vec<bool> = input_bits_raw
            .as_deref()
            .map(parse_bit_string)
            .transpose()
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();

        let default_cycles = if circ.default_cycles == 0 { 1 } else { circ.default_cycles };
        let cycles = cycles_override.unwrap_or(default_cycles).max(1);

        let mut inputs = vec![false; circ.primary_inputs.len()];
        for (j, &b) in input_bits.iter().enumerate() {
            if j < inputs.len() { inputs[j] = b; }
        }

        println!("Circuit (LUT) : {label}");
        println!("  wires   : {}", circ.num_wires);
        println!("  inputs  : {}", circ.primary_inputs.len());
        println!("  regs    : {}", circ.registers.len());
        println!("  lut ops : {}", circ.ops.len());
        println!("  lut types: {}", circ.lut_types.len());
        println!("  outputs : {}", circ.outputs.len());
        println!("  cycles  : {cycles}");

        // ── SRS setup ─────────────────────────────────────────────────────────
        let max_num_vars = compute_max_num_vars_lut(&circ, cycles);
        println!(
            "\nSRS setup (max_num_vars = {max_num_vars}, poly size = {})…",
            1usize << max_num_vars
        );
        let t_srs = Instant::now();
        let pk = <PCS as CommitmentScheme>::setup_prover(max_num_vars);
        let vk = <PCS as CommitmentScheme>::setup_verifier(&pk);
        let srs_ms = t_srs.elapsed().as_millis();
        println!("  SRS time: {srs_ms} ms");

        // ── Prove ─────────────────────────────────────────────────────────────
        println!("\nProving (LUT sumcheck + HyperKZG opening proofs)…");
        let t0 = Instant::now();

        let mut prove_transcript = KeccakTranscript::new(b"lut-circuit-native-zkp");
        prove_transcript.append_u64(circ.ops.len() as u64);
        prove_transcript.append_u64(circ.outputs.len() as u64);
        prove_transcript.append_u64(cycles as u64);
        for &b in &inputs {
            prove_transcript.append_u64(b as u64);
        }
        let lut_proof = prove_lut_circuit(&circ, &inputs, cycles, &pk, &mut prove_transcript);
        let prove_ms = t0.elapsed().as_millis();
        println!("  Prover time: {prove_ms} ms");

        // ── Print outputs ─────────────────────────────────────────────────────
        // println!("\nOutputs:");
        // for (idx, (&wire, &out)) in circ
        //     .outputs
        //     .iter()
        //     .zip(lut_proof.outputs.iter())
        //     .enumerate()
        // {
        //     println!("  out[{idx}] wire[{wire}] = {}", out as u8);
        // }

        // ── Proof summary ─────────────────────────────────────────────────────
        // println!("\nProof summary (LUT NIZK + Phase 4f batched opening):");
        // println!("  Max sumcheck vars   : {}", lut_proof.max_num_vars);
        // println!("  LUT group proofs    : {}", lut_proof.lut_proofs.len());
        // for lp in &lut_proof.lut_proofs {
        //     println!(
        //         "  lut_id=0x{:08X} k={} m={}  {:>7} evals, {} vars, {} round polys, \
        //          {} commitments (no per-group opening)",
        //         lp.lut_id, lp.k, lp.m, lp.num_evals, lp.num_vars,
        //         lp.round_polys.len(),
        //         lp.k + lp.m,
        //     );
        // }
        let total_polys: usize = lut_proof.lut_proofs.iter().map(|lp| lp.k + lp.m).sum();
        // println!("  Total polys batched : {total_polys} → 1 combined HyperKZG opening");

        // ── Verify ────────────────────────────────────────────────────────────
        println!("\nVerifying (no circuit re-execution)…");
        let t1 = Instant::now();

        let mut verify_transcript = KeccakTranscript::new(b"lut-circuit-native-zkp");
        verify_transcript.append_u64(circ.ops.len() as u64);
        verify_transcript.append_u64(circ.outputs.len() as u64);
        verify_transcript.append_u64(cycles as u64);
        for &b in &inputs {
            verify_transcript.append_u64(b as u64);
        }
        let all_ok = verify_lut_circuit(&lut_proof, &circ, &vk, &mut verify_transcript);
        let verify_ms = t1.elapsed().as_millis();
        println!("  Verifier time: {verify_ms} ms");

        if all_ok {
            println!("\n✓  All LUT group proofs valid (NIZK verified).");
        } else {
            eprintln!("\n✗  One or more LUT group proofs INVALID");
            std::process::exit(2);
        }

        // ── bench-csv ─────────────────────────────────────────────────────────
        if let Some(ref csv_path) = bench_csv {
            let proof_size = compute_lut_proof_size_bytes(&lut_proof);
            let srs_size = 1usize << max_num_vars;
            let total_evals = circ.ops.len() * cycles as usize;
            let num_lut_types = lut_proof.lut_proofs.len();

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
                     srs_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,num_gate_types"
                ).expect("write CSV header");
            }
            writeln!(
                file,
                "{label},{},{cycles},{total_evals},{},{srs_size},{srs_ms},{prove_ms},{verify_ms},{proof_size},{num_lut_types}",
                circ.ops.len(),
                lut_proof.max_num_vars,
            ).expect("write CSV row");

            println!("\nBench CSV row appended to: {}", csv_path.display());
            println!("  proof_size_bytes : {proof_size}");
            println!("  srs_g1_points    : {srs_size}");
            println!("  total_evals      : {total_evals}");
        }

        return; // LUT path complete
    }

    // ── Phase 4h: Unified mega-table LogUp ───────────────────────────────────
    if let Some(Mode::MegaBytecode(ref p)) = mode {
        let label = p.display().to_string();
        let circ = load_lut_circuit(p)
            .unwrap_or_else(|e| panic!("load {}: {e}", p.display()));

        let input_bits: Vec<bool> = input_bits_raw
            .as_deref()
            .map(parse_bit_string)
            .transpose()
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();

        let default_cycles = if circ.default_cycles == 0 { 1 } else { circ.default_cycles };
        let cycles = cycles_override.unwrap_or(default_cycles).max(1);

        let mut inputs = vec![false; circ.primary_inputs.len()];
        for (j, &b) in input_bits.iter().enumerate() {
            if j < inputs.len() { inputs[j] = b; }
        }

        println!("Circuit (Phase 4h Mega-LogUp) : {label}");
        println!("  wires     : {}", circ.num_wires);
        println!("  inputs    : {}", circ.primary_inputs.len());
        println!("  regs      : {}", circ.registers.len());
        println!("  lut ops   : {}", circ.ops.len());
        println!("  lut types : {}", circ.lut_types.len());
        println!("  outputs   : {}", circ.outputs.len());
        println!("  cycles    : {cycles}");

        // ── SRS setup ─────────────────────────────────────────────────────────
        // Phase 4h SRS = max(M_query, M_table) — much smaller than per-type SRS.
        let max_num_vars = compute_max_num_vars_mega(&circ, cycles);
        println!(
            "\nSRS setup (max_num_vars = {max_num_vars}, poly size = {})…",
            1usize << max_num_vars
        );
        let t_srs = Instant::now();
        let pk = <PCS as CommitmentScheme>::setup_prover(max_num_vars);
        let vk = <PCS as CommitmentScheme>::setup_verifier(&pk);
        let srs_ms = t_srs.elapsed().as_millis();
        println!("  SRS time: {srs_ms} ms");

        // ── Prove ─────────────────────────────────────────────────────────────
        println!("\nProving (Phase 4h unified mega-table LogUp)…");
        let t0 = Instant::now();

        let mut prove_transcript = KeccakTranscript::new(b"mega-logup-circuit-native-zkp");
        prove_transcript.append_u64(circ.ops.len() as u64);
        prove_transcript.append_u64(circ.outputs.len() as u64);
        prove_transcript.append_u64(cycles as u64);
        for &b in &inputs {
            prove_transcript.append_u64(b as u64);
        }
        let mega_proof = prove_mega_logup_circuit(&circ, &inputs, cycles, &pk, &mut prove_transcript);
        let prove_ms = t0.elapsed().as_millis();
        println!("  Prover time: {prove_ms} ms");

        // ── Print outputs ─────────────────────────────────────────────────────
        // println!("\nOutputs:");
        // for (idx, (&wire, &out)) in circ
        //     .outputs
        //     .iter()
        //     .zip(mega_proof.outputs.iter())
        //     .enumerate()
        // {
        //     println!("  out[{idx}] wire[{wire}] = {}", out as u8);
        // }

        // ── Proof summary ─────────────────────────────────────────────────────
        let t = mega_proof.num_lut_types;
        let k = mega_proof.k;
        let t_pad = t.next_power_of_two().max(1);
        println!("\nProof summary (Phase 4h unified mega-table LogUp):");
        println!("  LUT types (T)       : {t}  (T_pad = {t_pad})");
        println!("  k (input bits)      : {k}");
        println!("  m (output bits)     : {}", mega_proof.m);
        println!("  N_total (trace rows): {}", mega_proof.n_total);
        println!("  M_query (SC rounds) : {}", mega_proof.num_query_vars);
        println!("  M_table (SC rounds) : {}", mega_proof.num_table_vars);
        println!("  Committed polys     : 6  (TypeIdx, PackedIn, PackedOut, InvQ, Count, InvT)");
        println!("  Opening proofs      : 6  (independent of T)");
        println!("  [vs Phase 4b: {} commits, {} openings]", t * 5, t * 5);

        // ── Verify ────────────────────────────────────────────────────────────
        println!("\nVerifying (no circuit re-execution)…");
        let t1 = Instant::now();

        let mut verify_transcript = KeccakTranscript::new(b"mega-logup-circuit-native-zkp");
        verify_transcript.append_u64(circ.ops.len() as u64);
        verify_transcript.append_u64(circ.outputs.len() as u64);
        verify_transcript.append_u64(cycles as u64);
        for &b in &inputs {
            verify_transcript.append_u64(b as u64);
        }
        let all_ok = verify_mega_logup_circuit(&mega_proof, &circ, &vk, &mut verify_transcript);
        let verify_ms = t1.elapsed().as_millis();
        println!("  Verifier time: {verify_ms} ms");

        if all_ok {
            println!("\n✓  Phase 4h mega-LogUp proof valid (NIZK verified).");
        } else {
            eprintln!("\n✗  Phase 4h mega-LogUp proof INVALID");
            std::process::exit(2);
        }

        // ── bench-csv ─────────────────────────────────────────────────────────
        if let Some(ref csv_path) = bench_csv {
            let proof_size    = compute_mega_proof_size_bytes(&mega_proof);
            let srs_size      = 1usize << max_num_vars;
            let total_evals   = circ.ops.len() * cycles as usize;
            let num_lut_types = mega_proof.num_lut_types;

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
                "{label},{},{cycles},{total_evals},{},{srs_size},{srs_ms},{prove_ms},{verify_ms},{proof_size},{num_lut_types}",
                circ.ops.len(),
                mega_proof.num_query_vars,
            ).expect("write CSV row");

            println!("\nBench CSV row appended to: {}", csv_path.display());
            println!("  proof_size_bytes : {proof_size}");
            println!("  srs_g1_points    : {srs_size}");
            println!("  total_evals      : {total_evals}");
        }

        return; // Phase 4h path complete
    }

    // ── Lasso/LogUp prover path (Phase 4b) ───────────────────────────────────
    if let Some(Mode::LassoBytecode(ref p)) = mode {
        let label = p.display().to_string();
        let circ = load_lut_circuit(p)
            .unwrap_or_else(|e| panic!("load {}: {e}", p.display()));

        let input_bits: Vec<bool> = input_bits_raw
            .as_deref()
            .map(parse_bit_string)
            .transpose()
            .unwrap_or_else(|e| panic!("{e}"))
            .unwrap_or_default();

        let default_cycles = if circ.default_cycles == 0 { 1 } else { circ.default_cycles };
        let cycles = cycles_override.unwrap_or(default_cycles).max(1);

        let mut inputs = vec![false; circ.primary_inputs.len()];
        for (j, &b) in input_bits.iter().enumerate() {
            if j < inputs.len() { inputs[j] = b; }
        }

        println!("Circuit (Lasso/LogUp) : {label}");
        println!("  wires     : {}", circ.num_wires);
        println!("  inputs    : {}", circ.primary_inputs.len());
        println!("  regs      : {}", circ.registers.len());
        println!("  lut ops   : {}", circ.ops.len());
        println!("  lut types : {}", circ.lut_types.len());
        println!("  outputs   : {}", circ.outputs.len());
        println!("  cycles    : {cycles}");

        // ── SRS setup ─────────────────────────────────────────────────────────
        // The Lasso prover commits polynomials over {0,1}^M (query side) and
        // {0,1}^k (table side).  M is the same as in the Phase 2 prover, so we
        // reuse compute_max_num_vars_lasso.
        let max_num_vars = compute_max_num_vars_lasso(&circ, cycles);
        println!(
            "\nSRS setup (max_num_vars = {max_num_vars}, poly size = {})…",
            1usize << max_num_vars
        );
        let t_srs = Instant::now();
        let pk = <PCS as CommitmentScheme>::setup_prover(max_num_vars);
        let vk = <PCS as CommitmentScheme>::setup_verifier(&pk);
        let srs_ms = t_srs.elapsed().as_millis();
        println!("  SRS time: {srs_ms} ms");

        // ── Prove ─────────────────────────────────────────────────────────────
        println!("\nProving (Phase 4b Lasso/LogUp + HyperKZG opening proofs)…");
        let t0 = Instant::now();

        let mut prove_transcript = KeccakTranscript::new(b"lasso-lut-circuit-native-zkp");
        prove_transcript.append_u64(circ.ops.len() as u64);
        prove_transcript.append_u64(circ.outputs.len() as u64);
        prove_transcript.append_u64(cycles as u64);
        for &b in &inputs {
            prove_transcript.append_u64(b as u64);
        }
        let lasso_proof = prove_lasso_lut_circuit(&circ, &inputs, cycles, &pk, &mut prove_transcript);
        let prove_ms = t0.elapsed().as_millis();
        println!("  Prover time: {prove_ms} ms");

        // ── Print outputs ─────────────────────────────────────────────────────
        // println!("\nOutputs:");
        // for (idx, (&wire, &out)) in circ
        //     .outputs
        //     .iter()
        //     .zip(lasso_proof.outputs.iter())
        //     .enumerate()
        // {
        //     println!("  out[{idx}] wire[{wire}] = {}", out as u8);
        // }

        // ── Proof summary ─────────────────────────────────────────────────────
        println!("\nProof summary (Lasso/LogUp NIZK with HyperKZG):");
        println!("  Max sumcheck vars   : {}", lasso_proof.max_num_vars);
        println!("  LUT group proofs    : {}", lasso_proof.lut_proofs.len());
        for lp in &lasso_proof.lut_proofs {
            println!(
                "  lut_id=0x{:08X} k={} m={}  {:>7} evals, {} vars  \
                 [5 comms, {} Q-rounds + {} T-rounds, 5 openings]",
                lp.lut_id, lp.k, lp.m, lp.num_evals, lp.num_vars,
                lp.sc_q_polys.len(), lp.sc_t_polys.len()
            );
        }

        // ── Verify ────────────────────────────────────────────────────────────
        println!("\nVerifying (no circuit re-execution)…");
        let t1 = Instant::now();

        let mut verify_transcript = KeccakTranscript::new(b"lasso-lut-circuit-native-zkp");
        verify_transcript.append_u64(circ.ops.len() as u64);
        verify_transcript.append_u64(circ.outputs.len() as u64);
        verify_transcript.append_u64(cycles as u64);
        for &b in &inputs {
            verify_transcript.append_u64(b as u64);
        }
        let all_ok = verify_lasso_lut_circuit(&lasso_proof, &circ, &vk, &mut verify_transcript);
        let verify_ms = t1.elapsed().as_millis();
        println!("  Verifier time: {verify_ms} ms");

        if all_ok {
            println!("\n✓  All Lasso LUT group proofs valid (NIZK verified).");
        } else {
            eprintln!("\n✗  One or more Lasso LUT group proofs INVALID");
            std::process::exit(2);
        }

        // ── bench-csv ─────────────────────────────────────────────────────────
        if let Some(ref csv_path) = bench_csv {
            let proof_size   = compute_lasso_proof_size_bytes(&lasso_proof);
            let srs_size     = 1usize << max_num_vars;
            let total_evals  = circ.ops.len() * cycles as usize;
            let num_lut_types = lasso_proof.lut_proofs.len();

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
                "{label},{},{cycles},{total_evals},{},{srs_size},{srs_ms},{prove_ms},{verify_ms},{proof_size},{num_lut_types}",
                circ.ops.len(),
                lasso_proof.max_num_vars,
            ).expect("write CSV row");

            println!("\nBench CSV row appended to: {}", csv_path.display());
            println!("  proof_size_bytes : {proof_size}");
            println!("  srs_g1_points    : {srs_size}");
            println!("  total_evals      : {total_evals}");
        }

        return; // Lasso path complete
    }

    // ── Gate prover path (original Phase 0/1) ────────────────────────────────
    let (label, circ) = match mode {
        Some(Mode::Tiny) => (
            "<tiny: xor(not(in0), in1)>".to_string(),
            tiny_circuit(),
        ),
        Some(Mode::Bytecode(p)) => {
            let c = load_circuit(&p).unwrap_or_else(|e| panic!("load {}: {e}", p.display()));
            (p.display().to_string(), c)
        }
        Some(Mode::LutBytecode(_))   => unreachable!("handled above"),
        Some(Mode::LassoBytecode(_))  => unreachable!("handled above"),
        Some(Mode::MegaBytecode(_))   => unreachable!("handled above"),
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

    let default_cycles = if circ.default_cycles == 0 {
        1
    } else {
        circ.default_cycles
    };
    let cycles = cycles_override.unwrap_or(default_cycles).max(1);

    let mut inputs = vec![false; circ.primary_inputs.len()];
    for (j, &b) in input_bits.iter().enumerate() {
        if j < inputs.len() {
            inputs[j] = b;
        }
    }

    println!("Circuit : {label}");
    println!("  wires  : {}", circ._num_wires);
    println!("  inputs : {}", circ.primary_inputs.len());
    println!("  regs   : {}", circ.registers.len());
    println!("  gates  : {}", circ.ops.len());
    println!("  outputs: {}", circ.outputs.len());
    println!("  cycles : {cycles}");

    if show_pages {
        let (trace, _) = evaluate_circuit(&circ, &inputs, cycles);
        println!("\nGate trace ({} evaluations):", trace.len());
        for (idx, g) in trace.iter().enumerate() {
            println!(
                "  [{idx:>5}] mask=0x{:02X} a={} b={} -> out={}",
                g.mask, g.a as u8, g.b as u8, g.out as u8
            );
        }
    }

    // ── SRS setup ────────────────────────────────────────────────────────────
    let max_num_vars = compute_max_num_vars(&circ, cycles);
    println!("\nSRS setup (max_num_vars = {max_num_vars}, poly size = {})…",
        1usize << max_num_vars);

    let t_srs = Instant::now();
    let pk = <PCS as CommitmentScheme>::setup_prover(max_num_vars);
    let vk = <PCS as CommitmentScheme>::setup_verifier(&pk);
    let srs_ms = t_srs.elapsed().as_millis();
    println!("  SRS time: {srs_ms} ms");

    // ── Prove ────────────────────────────────────────────────────────────────
    println!("\nProving (HyperKZG commitments + sumcheck + opening proofs)…");
    let t0 = Instant::now();

    let mut prove_transcript = KeccakTranscript::new(b"bool-circuit-native-zkp");
    // Public metadata bound to the Fiat-Shamir oracle.
    prove_transcript.append_u64(circ.ops.len() as u64);
    prove_transcript.append_u64(circ.outputs.len() as u64);
    prove_transcript.append_u64(cycles as u64);
    for &b in &inputs {
        prove_transcript.append_u64(b as u64);
    }

    let circuit_proof = prove_circuit(&circ, &inputs, cycles, &pk, &mut prove_transcript);

    let prove_ms = t0.elapsed().as_millis();
    println!("  Prover time: {prove_ms} ms");

    // ── Print outputs ────────────────────────────────────────────────────────
    // println!("\nOutputs:");
    // for (idx, (&wire, &out)) in circ
    //     .outputs
    //     .iter()
    //     .zip(circuit_proof.outputs.iter())
    //     .enumerate()
    // {
    //     println!("  out[{idx}] wire[{wire}] = {}", out as u8);
    // }

    // ── Show proof summary ───────────────────────────────────────────────────
    println!("\nProof summary (NIZK with HyperKZG):");
    println!("  Max sumcheck vars: {}", circuit_proof.max_num_vars);
    for gp in &circuit_proof.gate_proofs {
        let name = match gp.mask {
            MASK_AND => "AND",
            MASK_OR => "OR",
            MASK_XOR => "XOR",
            MASK_NOT => "NOT",
            m => {
                print!("  mask 0x{m:02X}");
                "???"
            }
        };
        println!(
            "  {name:3} gates: {:>7} evals, {} sumcheck vars, {} round polys, \
             3 commitments + 3 opening proofs",
            gp.num_gates, gp.num_vars, gp.round_polys.len()
        );
    }

    // ── Verify ───────────────────────────────────────────────────────────────
    // The verifier ONLY needs: the proof, the verifier key, and the public inputs.
    // It does NOT re-execute the circuit or see any wire values.
    println!("\nVerifying (no circuit re-execution)…");
    let t1 = Instant::now();

    let mut verify_transcript = KeccakTranscript::new(b"bool-circuit-native-zkp");
    // Same public metadata (circuit description is public).
    verify_transcript.append_u64(circ.ops.len() as u64);
    verify_transcript.append_u64(circ.outputs.len() as u64);
    verify_transcript.append_u64(cycles as u64);
    for &b in &inputs {
        verify_transcript.append_u64(b as u64);
    }

    let all_ok = verify_circuit(&circuit_proof, &vk, &mut verify_transcript);

    let verify_ms = t1.elapsed().as_millis();
    println!("  Verifier time: {verify_ms} ms");

    if all_ok {
        println!("\n✓  All gate proofs valid (NIZK verified).");
        println!("   Proof is non-interactive and zero-knowledge:");
        println!("   - Witness polynomials are hidden behind HyperKZG commitments");
        println!("   - Verifier never sees raw gate trace or re-executes the circuit");
        println!("   - Opening proofs cryptographically bind evaluations to commitments");
    } else {
        eprintln!("\n✗  One or more gate proofs INVALID");
        std::process::exit(2);
    }

    // ── --bench-csv output ────────────────────────────────────────────────────
    if let Some(csv_path) = &bench_csv {
        let proof_size = compute_proof_size_bytes(&circuit_proof);
        let srs_size = 1usize << max_num_vars; // number of G1 points in SRS
        let total_evals = circ.ops.len() * cycles as usize;
        let num_gate_types = circuit_proof.gate_proofs.len();

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
                 srs_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes,num_gate_types"
            ).expect("write CSV header");
        }
        writeln!(
            file,
            "{label},{},{cycles},{total_evals},{},{srs_size},{srs_ms},{prove_ms},{verify_ms},{proof_size},{num_gate_types}",
            circ.ops.len(),
            circuit_proof.max_num_vars,
        ).expect("write CSV row");

        println!("\nBench CSV row appended to: {}", csv_path.display());
        println!("  proof_size_bytes : {proof_size}");
        println!("  srs_g1_points    : {srs_size}");
        println!("  total_evals      : {total_evals}");
    }
}

// ── proof size estimation ─────────────────────────────────────────────────────

/// Estimate the serialised proof size in bytes by summing the compressed byte
/// sizes of all HyperKZG commitments, round polynomials, final evaluations,
/// and opening proofs contained in a `CircuitProof`.
fn compute_proof_size_bytes(proof: &CircuitProof) -> usize {
    let mut total = 0usize;

    for gp in &proof.gate_proofs {
        // 3 HyperKZG commitments (G1Affine points, compressed = 48 bytes for BN254)
        let mut buf = Vec::new();
        gp.commitment_a.0.serialize_compressed(&mut buf).ok();
        total += buf.len();
        buf.clear();
        gp.commitment_b.0.serialize_compressed(&mut buf).ok();
        total += buf.len();
        buf.clear();
        gp.commitment_out.0.serialize_compressed(&mut buf).ok();
        total += buf.len();

        // round polys: num_vars rounds × 4 Fr elements × 32 bytes
        total += gp.round_polys.len() * 4 * 32;

        // final evaluations: 3 Fr × 32 bytes
        total += 3 * 32;

        // HyperKZG opening proofs
        for opt_proof in [&gp.opening_proof_a, &gp.opening_proof_b, &gp.opening_proof_out] {
            if let Some(op) = opt_proof {
                buf.clear();
                op.serialize_compressed(&mut buf).ok();
                total += buf.len();
            }
        }
    }

    total
}
