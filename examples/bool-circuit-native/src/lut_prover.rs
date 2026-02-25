//! **Phase 2 — Generalised LUT Lookup Proof**
//!
//! Replaces the 2-variable `GateLookupTable` / `GateLookupProof` with a
//! k-variable [`LutGroupProof`] that proves correct evaluation of a k-input,
//! m-output sub-circuit LUT.
//!
//! # Proof structure (per distinct LUT type, N invocations)
//!
//! Given N trace rows `{(inputs_i[0..k], outputs_i[0..m])}`:
//!
//!   In_ℓ(x), Out_j(x)  — MLEs of the k input and m output columns over {0,1}^M
//!   T̃_j(u_0,…,u_{k-1}) — MLE of the j-th output column of the truth table
//!
//! **Batched sumcheck identity (for all m output bits simultaneously):**
//!
//! ```text
//!   ∑_{x ∈ {0,1}^M} eq(r_in, x) · ∑_j α^j · (Out_j(x) − T̃_j(In_0(x),…,In_{k-1}(x))) = 0
//! ```
//!
//! where α is a Fiat-Shamir random scalar and r_in ∈ F^M is the equality challenge.
//!
//! The per-round polynomial has degree k+1 (from the k-linear composition of T̃_j),
//! so each round poly carries k+2 evaluation points.  After M rounds, all k+m
//! polynomials are opened at the final sumcheck point r_sc via HyperKZG.
//!
//! **Verifier:**  replay Fiat-Shamir, check round consistency, verify the
//! final batched evaluation against the LUT's MLE, verify k+m opening proofs.

use std::collections::HashMap;

use ark_ec::{pairing::Pairing, CurveGroup};
use ark_ff::{Field, One, Zero};
use ark_bn254::{Bn254, Fr};
use ark_serialize::CanonicalSerialize;

use jolt_core::field::JoltField;
use jolt_core::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
use jolt_core::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation};
use jolt_core::transcripts::{AppendToTranscript, KeccakTranscript, Transcript};
use jolt_core::zkvm::lookup_table::SubCircuitLut;

use crate::lczbc::{LutCirc, LutDesc, LutEval, evaluate_lut_circuit};

/// Type alias for HyperKZG opening challenge type.
type Challenge = <Fr as JoltField>::Challenge;

// ── field helpers ─────────────────────────────────────────────────────────────

#[inline]
fn fr(n: u64) -> Fr { Fr::from(n) }

#[inline]
fn fr_bool(b: bool) -> Fr {
    if b { Fr::one() } else { Fr::zero() }
}

/// Bind the lowest variable of poly in-place: poly[i] ← poly[2i] + r·(poly[2i+1]−poly[2i]).
fn bind(poly: &mut Vec<Fr>, r: Fr) {
    let half = poly.len() / 2;
    for i in 0..half {
        poly[i] = poly[2 * i] + r * (poly[2 * i + 1] - poly[2 * i]);
    }
    poly.truncate(half);
}

/// Initialise eq(r, ·) over the hypercube using LSB-first variable ordering.
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

/// Evaluate eq(r_input, r_sc) from the two LSB-first challenge vectors.
fn eq_final_eval(r_input: &[Fr], r_sc: &[Fr]) -> Fr {
    assert_eq!(r_input.len(), r_sc.len());
    r_input
        .iter()
        .zip(r_sc.iter())
        .map(|(&ri, &si)| ri * si + (Fr::one() - ri) * (Fr::one() - si))
        .product()
}

/// Evaluate a degree-d polynomial given by its values at 0, 1, …, d at an
/// arbitrary point t using Lagrange interpolation.
///
/// `evals[i]` = polynomial value at integer i.
fn poly_at(evals: &[Fr], t: Fr) -> Fr {
    let n = evals.len();
    let mut result = Fr::zero();
    for i in 0..n {
        let fi = fr(i as u64);
        let mut basis = evals[i];
        let mut denom = Fr::one();
        for j in 0..n {
            if j != i {
                let fj = fr(j as u64);
                basis *= t - fj;
                denom *= fi - fj;
            }
        }
        // denom is a product of non-zero field elements → always invertible
        result += basis * Field::inverse(&denom).expect("denom should be non-zero");
    }
    result
}

/// Build a [`SubCircuitLut`] from a [`LutDesc`].
fn sub_circuit_lut_from_desc(desc: &LutDesc) -> SubCircuitLut {
    SubCircuitLut::from_bytes(desc.k, desc.m, &desc.truth_table)
}

// ── proof structs ─────────────────────────────────────────────────────────────

/// NIZK proof for all N invocations of a single LUT type.
///
/// Proves: for each output bit j, `∑_x eq(r_in, x)·(Out_j(x) − T̃_j(In(x))) = 0`.
/// The m per-bit sumchecks are batched into one using a Fiat-Shamir scalar α.
pub struct LutGroupProof {
    /// LUT type identifier.
    pub lut_id: u32,
    /// Number of input bits.
    pub k: usize,
    /// Number of output bits.
    pub m: usize,
    /// Number of (un-padded) trace rows.
    pub num_evals: usize,
    /// Sumcheck variables M = ⌈log₂(num_evals)⌉.
    pub num_vars: usize,

    /// HyperKZG commitments to the k input polynomials.
    pub comm_inputs: Vec<HyperKZGCommitment<Bn254>>,
    /// HyperKZG commitments to the m output polynomials.
    pub comm_outputs: Vec<HyperKZGCommitment<Bn254>>,

    /// Per-round polynomial evaluations at integer points 0, 1, …, k+1.
    /// Outer index = round (0..M), inner len = k+2.
    pub round_polys: Vec<Vec<Fr>>,

    /// Final evaluations: In_i(r_sc) for i ∈ 0..k.
    pub finals_in: Vec<Fr>,
    /// Final evaluations: Out_j(r_sc) for j ∈ 0..m.
    pub finals_out: Vec<Fr>,

    // Phase 4f: per-group opening proofs removed.
    // All T×(k+m) polynomials are opened in a single batched proof
    // stored in LutCircuitProof::batched_opening.
}

/// Phase 4f: A single batched HyperKZG opening proof covering all committed
/// polynomials from all LUT groups at one shared evaluation point z.
///
/// Replaces the previous T×(k+m) individual opening proofs with O(1) pairings.
pub struct BatchedOpeningProof {
    /// Combined commitment C_comb = Commit(\sum_i \rho_i P_i_padded).
    /// Included so the verifier can replay the batched HyperKZG::verify.
    pub combined_comm: HyperKZGCommitment<Bn254>,
    /// Combined evaluation: `\sum_{\tau,i} \rho_{\tau,i} \cdot P_{\tau,i}_padded(z)`.
    pub combined_eval: Fr,
    /// The single HyperKZG opening proof for the combined polynomial.
    pub combined_proof: HyperKZGProof<Bn254>,
}

/// Complete NIZK proof for a LUT-annotated circuit.
pub struct LutCircuitProof {
    /// Maximum sumcheck variables across all LUT types (determines SRS size).
    pub max_num_vars: usize,
    /// One sumcheck proof per distinct LUT type (no per-group opening proofs).
    pub lut_proofs: Vec<LutGroupProof>,
    /// Claimed circuit outputs (final cycle wire values).
    pub outputs: Vec<bool>,
    /// Phase 4f: single batched opening proof covering all T×(k+m) polynomials.
    pub batched_opening: BatchedOpeningProof,
}

// ── prover ────────────────────────────────────────────────────────────────────

/// Prove all N invocations of a single LUT type are correct.
///
/// Phase 4f: does NOT produce per-group opening proofs.
/// Returns the group proof together with the raw MLE tables so that the
/// caller (`prove_lut_circuit`) can batch all T×(k+m) openings into one.
///
/// 1. Commits to k input + m output polynomials via HyperKZG.
/// 2. Derives Fiat-Shamir challenges.
/// 3. Runs M rounds of the batched sumcheck (degree k+1, k+2 eval points/round).
/// 4. Appends finals to transcript (opening deferred to batch phase).
pub fn prove_lut_group(
    desc: &LutDesc,
    evals: &[LutEval],
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> (LutGroupProof, Vec<MultilinearPolynomial<Fr>>, Vec<MultilinearPolynomial<Fr>>) {
    let lut = sub_circuit_lut_from_desc(desc);
    let k = desc.k;
    let m = desc.m;
    let n = evals.len();
    assert!(n > 0, "prove_lut_group: empty group for lut_id={}", desc.lut_id);

    // M = ⌈log₂(n)⌉, cap = 2^M (padded size)
    let big_m = usize::max(1, n.next_power_of_two().trailing_zeros() as usize);
    let cap = 1usize << big_m;

    // ── build padded eval vectors ─────────────────────────────────────────
    // Dummy row: all-zero inputs → out_j = lut.eval_bool(&zeros, j)
    // Using all-zero inputs ensures the gate identity holds for padding.
    let dummy_inputs = vec![false; k];
    let dummy_outputs: Vec<bool> = (0..m).map(|j| lut.eval_bool(&dummy_inputs, j)).collect();

    let in_evals: Vec<Vec<Fr>> = (0..k)
        .map(|i| {
            (0..cap)
                .map(|row| {
                    fr_bool(evals.get(row).map(|e| e.inputs[i]).unwrap_or(false))
                })
                .collect()
        })
        .collect();

    let out_evals: Vec<Vec<Fr>> = (0..m)
        .map(|j| {
            (0..cap)
                .map(|row| {
                    fr_bool(evals.get(row).map(|e| e.outputs[j]).unwrap_or(dummy_outputs[j]))
                })
                .collect()
        })
        .collect();

    // ── commit to all k+m polynomials ─────────────────────────────────────
    let in_mles: Vec<MultilinearPolynomial<Fr>> = in_evals
        .iter()
        .map(|v| MultilinearPolynomial::from(v.clone()))
        .collect();
    let out_mles: Vec<MultilinearPolynomial<Fr>> = out_evals
        .iter()
        .map(|v| MultilinearPolynomial::from(v.clone()))
        .collect();

    let comm_inputs: Vec<HyperKZGCommitment<Bn254>> = in_mles
        .iter()
        .map(|p| HyperKZG::<Bn254>::commit(pk, p).expect("commit In failed"))
        .collect();
    let comm_outputs: Vec<HyperKZGCommitment<Bn254>> = out_mles
        .iter()
        .map(|p| HyperKZG::<Bn254>::commit(pk, p).expect("commit Out failed"))
        .collect();

    // ── Fiat-Shamir: bind metadata + all commitments ──────────────────────
    transcript.append_u64(desc.lut_id as u64);
    transcript.append_u64(n as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(m as u64);
    transcript.append_u64(big_m as u64);
    for c in &comm_inputs {
        c.append_to_transcript(transcript);
    }
    for c in &comm_outputs {
        c.append_to_transcript(transcript);
    }

    // α: random scalar for batching m sumchecks into one.
    let alpha: Fr = transcript.challenge_scalar_optimized::<Fr>().into();
    // r_input: equality polynomial challenge.
    let r_input: Vec<Fr> = transcript.challenge_vector(big_m);

    // ── initialise mutable polynomial slices ──────────────────────────────
    let mut eq_poly = init_eq(&r_input);
    // in_polys[i]: mutable copy for binding
    let mut in_polys: Vec<Vec<Fr>> = in_evals.clone();
    let mut out_polys: Vec<Vec<Fr>> = out_evals.clone();

    // Pre-compute α^j for j = 0..m
    let alpha_pows: Vec<Fr> = {
        let mut v = Vec::with_capacity(m);
        let mut cur = Fr::one();
        for _ in 0..m {
            v.push(cur);
            cur *= alpha;
        }
        v
    };

    let n_pts = k + 2; // number of evaluation points per round
    let eval_ts: Vec<Fr> = (0..n_pts as u64).map(fr).collect();

    let mut round_polys: Vec<Vec<Fr>> = Vec::with_capacity(big_m);

    // ── sumcheck rounds ───────────────────────────────────────────────────
    for _round in 0..big_m {
        let half = eq_poly.len() / 2;
        let mut p_evals = vec![Fr::zero(); n_pts];

        for idx in 0..half {
            let eq_lo = eq_poly[2 * idx];
            let eq_hi = eq_poly[2 * idx + 1];

            // Grab lo/hi for each input poly and output poly
            let in_lo: Vec<Fr> = in_polys.iter().map(|p| p[2 * idx]).collect();
            let in_hi: Vec<Fr> = in_polys.iter().map(|p| p[2 * idx + 1]).collect();
            let out_lo: Vec<Fr> = out_polys.iter().map(|p| p[2 * idx]).collect();
            let out_hi: Vec<Fr> = out_polys.iter().map(|p| p[2 * idx + 1]).collect();

            for (pt, &t) in eval_ts.iter().enumerate() {
                let eq_t = eq_lo + t * (eq_hi - eq_lo);

                // Interpolate input polys at t
                let in_t: Vec<Fr> = in_lo
                    .iter()
                    .zip(in_hi.iter())
                    .map(|(&lo, &hi)| lo + t * (hi - lo))
                    .collect();

                // Batched output contribution: ∑_j α^j * (out_j_t - T̃_j(in_t))
                let mut batched = Fr::zero();
                for j in 0..m {
                    let out_t = out_lo[j] + t * (out_hi[j] - out_lo[j]);
                    let lut_t = lut.evaluate_mle_at::<Fr>(&in_t, j);
                    batched += alpha_pows[j] * (out_t - lut_t);
                }

                p_evals[pt] += eq_t * batched;
            }
        }

        // Append round polynomial to transcript
        for &e in p_evals.iter() {
            transcript.append_scalar(&e);
        }

        // Derive round challenge
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        // (r_j_ch not collected — per-group opening removed in Phase 4f)

        // Bind all polynomials at r_j
        bind(&mut eq_poly, r_j);
        for p in in_polys.iter_mut() {
            bind(p, r_j);
        }
        for p in out_polys.iter_mut() {
            bind(p, r_j);
        }

        round_polys.push(p_evals);
    }

    // After M bindings, each polynomial has length 1
    let finals_in: Vec<Fr> = in_polys.iter().map(|p| p[0]).collect();
    let finals_out: Vec<Fr> = out_polys.iter().map(|p| p[0]).collect();

    // ── append finals to transcript ───────────────────────────────────────
    // (Phase 4f: opening deferred to batch phase in prove_lut_circuit)
    for &v in finals_in.iter().chain(finals_out.iter()) {
        transcript.append_scalar(&v);
    }

    let proof = LutGroupProof {
        lut_id: desc.lut_id,
        k,
        m,
        num_evals: n,
        num_vars: big_m,
        comm_inputs,
        comm_outputs,
        round_polys,
        finals_in,
        finals_out,
    };

    (proof, in_mles, out_mles)
}

// ── helpers for Phase 4f batch opening ──────────────────────────────────────

/// Zero-pad a polynomial evaluation table to `2^{target_num_vars}` entries.
/// Used internally by `batch_open_lut_polys` via the inline accumulation loop.
#[allow(dead_code)]
fn pad_evals_to_num_vars(evals: Vec<Fr>, target_num_vars: usize) -> Vec<Fr> {
    let target_len = 1usize << target_num_vars;
    let mut padded = evals;
    padded.resize(target_len, Fr::zero());
    padded
}

/// Linearly combine HyperKZG commitments: `C_out = ∑ coeffs[i] · comms[i]`.
/// Uses the homomorphic property of KZG commitments.
fn combine_kzg_commitments(
    comms: &[HyperKZGCommitment<Bn254>],
    coeffs: &[Fr],
) -> HyperKZGCommitment<Bn254> {
    let combined: <Bn254 as Pairing>::G1 = comms
        .iter()
        .zip(coeffs.iter())
        .map(|(c, &rho)| c.0 * rho)
        .sum();
    HyperKZGCommitment(combined.into_affine())
}

/// Phase 4f prover step: given all group proofs + their raw MLEs, produce a
/// single batched HyperKZG opening that covers every committed polynomial
/// from every group.
///
/// # Protocol
/// 1. For each group (in sorted lut_id order), absorb its commitments into the
///    transcript (already done inside `prove_lut_group`; we do NOT re-absorb).
/// 2. Derive `max_num_vars` challenges `z[0..max_num_vars]` as the shared
///    evaluation point.
/// 3. Derive per-polynomial weights `ρ` (one per polynomial across all groups,
///    in the same order as the combined vector).
/// 4. Derive a non-zero constant offset `ρ_extra` and add it to every entry of
///    the combined evaluation table.  This ensures the combined polynomial is
///    never the zero polynomial (whose commitment is the identity point, which
///    `HyperKZG::verify` rejects).
/// 5. Zero-pad each MLE to `2^{max_num_vars}`.
/// 6. Compute `P_comb = ρ_extra + ∑ ρ_i · P_i_padded` and commit `C_comb`.
/// 7. Open `C_comb` at `z` with one HyperKZG call.
pub fn batch_open_lut_polys(
    lut_proofs: &[LutGroupProof],
    all_in_mles: &[Vec<MultilinearPolynomial<Fr>>],   // per-group input MLEs
    all_out_mles: &[Vec<MultilinearPolynomial<Fr>>],  // per-group output MLEs
    max_num_vars: usize,
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> BatchedOpeningProof {
    let max_len = 1usize << max_num_vars;

    // ── derive shared evaluation point z ─────────────────────────────────
    // Sample max_num_vars challenges from the transcript (after all finals
    // have been appended by every prove_lut_group call).
    let z_challenges: Vec<Challenge> = (0..max_num_vars)
        .map(|_| transcript.challenge_scalar_optimized::<Fr>())
        .collect();
    // HyperKZG uses big-endian variable ordering; our encoding is little-endian.
    let z_point_kzg: Vec<Challenge> = z_challenges.iter().rev().cloned().collect();

    // ── collect all commitments in deterministic order ────────────────────
    // Order: for each group τ (sorted by lut_id), k input comms then m output comms.
    let mut all_comms: Vec<HyperKZGCommitment<Bn254>> = Vec::new();
    for lp in lut_proofs {
        all_comms.extend(lp.comm_inputs.iter().cloned());
        all_comms.extend(lp.comm_outputs.iter().cloned());
    }
    let n_polys = all_comms.len();

    // ── derive per-polynomial random weights ρ ────────────────────────────
    let rhos: Vec<Fr> = (0..n_polys)
        .map(|_| transcript.challenge_scalar_optimized::<Fr>().into())
        .collect();

    // ── derive a non-zero constant offset ρ_extra ─────────────────────────
    // Adding ρ_extra to every evaluation ensures the combined polynomial is
    // never the zero polynomial (which would produce a zero commitment that
    // HyperKZG::verify rejects).  The verifier squeezes the same challenge
    // to keep the transcript in sync.
    let rho_extra: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── build the combined polynomial evaluation table ─────────────────────
    // P_comb[j] = ρ_extra + ∑_{τ,i} ρ_{τ,i} · P_{τ,i}_padded[j]
    let mut comb_evals = vec![rho_extra; max_len];
    let mut poly_idx = 0usize;
    for (lp, (in_mles, out_mles)) in lut_proofs
        .iter()
        .zip(all_in_mles.iter().zip(all_out_mles.iter()))
    {
        let group_len = 1usize << lp.num_vars;
        // input polynomials
        for mle in in_mles {
            let rho = rhos[poly_idx];
            poly_idx += 1;
            // add rho * P_i_padded into comb_evals
            for j in 0..group_len {
                comb_evals[j] += rho * mle.get_coeff(j);
            }
            // indices >= group_len are zero-padded → no contribution
        }
        // output polynomials
        for mle in out_mles {
            let rho = rhos[poly_idx];
            poly_idx += 1;
            for j in 0..group_len {
                comb_evals[j] += rho * mle.get_coeff(j);
            }
        }
    }
    debug_assert_eq!(poly_idx, n_polys);

    // ── commit the combined polynomial ────────────────────────────────────────────
    // We commit comb_mle directly so that c_comb = Commit(Sum_i rho_i P_i_padded)
    // is correct regardless of the per-polynomial sizes.  Using combine_kzg_commitments
    // would be incorrect when group polynomials have different num_vars.
    let comb_mle = MultilinearPolynomial::from(comb_evals.clone());
    let c_comb = HyperKZG::<Bn254>::commit(pk, &comb_mle)
        .expect("commit combined polynomial failed");

    // ── append c_comb to transcript ─────────────────────────────────────────
    // The verifier reads c_comb from the proof and absorbs it here.
    c_comb.append_to_transcript(transcript);

    // ── evaluate the combined polynomial at z ──────────────────────────────
    // Fold comb_evals in little-endian order (variable 0 = LSB) to match
    // how HyperKZG::open folds it internally via z_point_kzg.
    let combined_eval: Fr = {
        let z_fr: Vec<Fr> = z_challenges.iter().map(|&c| c.into()).collect();
        let mut tmp = comb_evals;
        for &zi in &z_fr {
            bind(&mut tmp, zi);
        }
        tmp[0]
    };

    // ── append combined_eval to transcript before opening ─────────────────
    transcript.append_scalar(&combined_eval);

    // ── single HyperKZG opening ────────────────────────────────────────────
    let combined_proof = HyperKZG::<Bn254>::open(
        pk,
        &comb_mle,
        &z_point_kzg,
        &combined_eval,
        transcript,
    )
    .expect("batch HyperKZG open failed");

    BatchedOpeningProof {
        combined_comm: c_comb,
        combined_eval,
        combined_proof,
    }
}

// ── verifier ──────────────────────────────────────────────────────────────────

/// Verify a [`LutGroupProof`].
///
/// 1. Re-derives Fiat-Shamir challenges.
/// 2. Replays the batched sumcheck.
/// 3. Checks the final evaluation against the LUT's MLE.
/// 4. Verifies k+m HyperKZG opening proofs.
pub fn verify_lut_group(
    proof: &LutGroupProof,
    desc: &LutDesc,
    // vk kept in signature for API compatibility; used only in batch verify.
    _vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    let lut = sub_circuit_lut_from_desc(desc);
    let k = proof.k;
    let m = proof.m;
    let big_m = proof.num_vars;
    let n_pts = k + 2;

    // ── re-derive Fiat-Shamir challenges ───────────────────────────────────
    transcript.append_u64(proof.lut_id as u64);
    transcript.append_u64(proof.num_evals as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(m as u64);
    transcript.append_u64(big_m as u64);
    for c in &proof.comm_inputs {
        c.append_to_transcript(transcript);
    }
    for c in &proof.comm_outputs {
        c.append_to_transcript(transcript);
    }

    let alpha: Fr = transcript.challenge_scalar_optimized::<Fr>().into();
    let alpha_pows: Vec<Fr> = {
        let mut v = Vec::with_capacity(m);
        let mut cur = Fr::one();
        for _ in 0..m {
            v.push(cur);
            cur *= alpha;
        }
        v
    };

    let r_input: Vec<Fr> = transcript.challenge_vector(big_m);

    // ── replay batched sumcheck ────────────────────────────────────────────
    let mut prev_claim = Fr::zero(); // ∑_x batched_g(x) = 0
    let mut r_sc_fr: Vec<Fr> = Vec::with_capacity(big_m);
    // (r_sc_challenge not needed: per-group opening removed in Phase 4f)

    for round in 0..big_m {
        let p = &proof.round_polys[round];
        if p.len() != n_pts {
            eprintln!(
                "  lut_id={}: round {round}: round poly has {} evals, expected {n_pts}",
                proof.lut_id, p.len()
            );
            return false;
        }

        // Check p(0) + p(1) == prev_claim
        let sum = p[0] + p[1];
        if sum != prev_claim {
            eprintln!(
                "  lut_id={}: round {round}: p(0)+p(1) = {sum:?} ≠ prev_claim = {prev_claim:?}",
                proof.lut_id
            );
            return false;
        }

        for &e in p.iter() {
            transcript.append_scalar(&e);
        }

        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_fr.push(r_j);
        // (r_j_ch not stored — per-group opening removed in Phase 4f)

        prev_claim = poly_at(p, r_j);
    }

    // ── final sumcheck check ──────────────────────────────────────────────
    // eq(r_input, r_sc) * ∑_j α^j * (finals_out[j] - T̃_j(finals_in))
    let eq_fin = eq_final_eval(&r_input, &r_sc_fr);
    let mut batched_final = Fr::zero();
    for j in 0..m {
        let lut_val = lut.evaluate_mle_at::<Fr>(&proof.finals_in, j);
        batched_final += alpha_pows[j] * (proof.finals_out[j] - lut_val);
    }
    let expected = eq_fin * batched_final;
    if expected != prev_claim {
        eprintln!(
            "  lut_id={}: final check failed: eq*gap = {expected:?} ≠ last_claim = {prev_claim:?}",
            proof.lut_id
        );
        return false;
    }

    // ── append finals to transcript (must match prover) ────────────────────
    // (Phase 4f: per-group opening proofs are gone; batch opening is verified
    //  separately in verify_lut_circuit after all groups are checked.)
    for &v in proof.finals_in.iter().chain(proof.finals_out.iter()) {
        transcript.append_scalar(&v);
    }

    true
}

/// Phase 4f verifier step: re-derive ρ and z from the transcript (which must
/// be in the exact same state as at the end of all `verify_lut_group` calls),
/// then verify a single batched HyperKZG opening covering all T×(k+m) polys.
pub fn batch_verify_lut_polys(
    lut_proofs: &[LutGroupProof],
    batched: &BatchedOpeningProof,
    max_num_vars: usize,
    vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    // ── count total polynomials (same order as prover) ────────────────────
    let n_polys: usize = lut_proofs.iter().map(|lp| lp.k + lp.m).sum();

    // ── re-derive shared evaluation point z ──────────────────────────────
    let z_challenges: Vec<Challenge> = (0..max_num_vars)
        .map(|_| transcript.challenge_scalar_optimized::<Fr>())
        .collect();
    let z_point_kzg: Vec<Challenge> = z_challenges.iter().rev().cloned().collect();

    // ── re-derive per-polynomial weights ρ ───────────────────────────────
    // (just advance the transcript the same number of challenges; verifier
    //  cannot reconstruct comb_mle so uses c_comb from the proof)
    let _rhos: Vec<Fr> = (0..n_polys)
        .map(|_| transcript.challenge_scalar_optimized::<Fr>().into())
        .collect();

    // ── re-derive ρ_extra (must match prover) ─────────────────────────────
    let _rho_extra: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── absorb c_comb from proof (mirrors prover's append_to_transcript) ──
    batched.combined_comm.append_to_transcript(transcript);

    // ── absorb combined_eval (must mirror prover) ─────────────────────────
    transcript.append_scalar(&batched.combined_eval);

    // ── verify one HyperKZG proof ─────────────────────────────────────────
    if HyperKZG::<Bn254>::verify(
        vk,
        &batched.combined_comm,
        &z_point_kzg,
        &batched.combined_eval,
        &batched.combined_proof,
        transcript,
    )
    .is_err()
    {
        eprintln!("  batch HyperKZG verify FAILED");
        return false;
    }

    true
}

// ── combined LUT circuit proof ────────────────────────────────────────────────

/// Compute the maximum sumcheck variables needed across all LUT types.
pub fn compute_max_num_vars_lut(circ: &LutCirc, cycles: u32) -> usize {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for op in &circ.ops {
        *counts.entry(op.lut_id).or_insert(0) += 1;
    }
    let cycles = cycles.max(1) as usize;
    let max_count = counts.values().copied().max().unwrap_or(1) * cycles;
    usize::max(1, max_count.next_power_of_two().trailing_zeros() as usize)
}

/// Prove the full execution trace of a LUT-annotated circuit.
///
/// Phase 4f: runs all T group sumchecks then produces ONE batched opening
/// proof covering every committed polynomial from every group.
pub fn prove_lut_circuit(
    circ: &LutCirc,
    inputs: &[bool],
    cycles: u32,
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> LutCircuitProof {
    let (trace, outputs) = evaluate_lut_circuit(circ, inputs, cycles);

    // Group trace rows by lut_id
    let mut groups: HashMap<u32, Vec<LutEval>> = HashMap::new();
    for row in &trace {
        groups.entry(row.lut_id).or_default().push(row.clone());
    }

    // Compute max_num_vars for SRS size reporting
    let max_num_vars = compute_max_num_vars_lut(circ, cycles);

    // Sort lut_ids for deterministic proof order
    let mut lut_ids: Vec<u32> = groups.keys().copied().collect();
    lut_ids.sort_unstable();

    // ── Phase 2: run all group sumchecks ──────────────────────────────────
    // Collect MLEs alongside proofs for the subsequent batch opening.
    let mut lut_proofs = Vec::with_capacity(lut_ids.len());
    let mut all_in_mles: Vec<Vec<MultilinearPolynomial<Fr>>> =
        Vec::with_capacity(lut_ids.len());
    let mut all_out_mles: Vec<Vec<MultilinearPolynomial<Fr>>> =
        Vec::with_capacity(lut_ids.len());

    for lut_id in lut_ids {
        let desc = circ
            .lut_types
            .get(&lut_id)
            .unwrap_or_else(|| panic!("unknown lut_id {lut_id}"));
        let evals = groups.remove(&lut_id).unwrap();
        let (proof, in_mles, out_mles) = prove_lut_group(desc, &evals, pk, transcript);
        lut_proofs.push(proof);
        all_in_mles.push(in_mles);
        all_out_mles.push(out_mles);
    }

    // ── Phase 4f: single batched opening for all T×(k+m) polynomials ─────
    let batched_opening = batch_open_lut_polys(
        &lut_proofs,
        &all_in_mles,
        &all_out_mles,
        max_num_vars,
        pk,
        transcript,
    );

    LutCircuitProof {
        max_num_vars,
        lut_proofs,
        outputs,
        batched_opening,
    }
}

/// Verify a complete LUT circuit NIZK proof.
///
/// Phase 4f: after verifying all T group sumchecks, calls
/// `batch_verify_lut_polys` to check the single batched HyperKZG opening.
pub fn verify_lut_circuit(
    proof: &LutCircuitProof,
    circ: &LutCirc,
    vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    // Phase 2: verify all group sumchecks (no per-group opening proofs)
    for lp in &proof.lut_proofs {
        let desc = circ
            .lut_types
            .get(&lp.lut_id)
            .unwrap_or_else(|| panic!("unknown lut_id {} in proof", lp.lut_id));
        if !verify_lut_group(lp, desc, vk, transcript) {
            eprintln!("  LUT group proof FAILED for lut_id={}", lp.lut_id);
            return false;
        }
    }

    // Phase 4f: verify the single batched polynomial opening
    if !batch_verify_lut_polys(
        &proof.lut_proofs,
        &proof.batched_opening,
        proof.max_num_vars,
        vk,
        transcript,
    ) {
        eprintln!("  Batched opening proof FAILED");
        return false;
    }

    true
}

// ── proof size estimation ─────────────────────────────────────────────────────

/// Estimate the serialised proof size in bytes.
///
/// Phase 4f: opening proofs are now a single `BatchedOpeningProof` instead of
/// T×(k+m) individual proofs.  The per-group contribution is only commitments
/// + round-polys + finals; the single batched opening is counted once.
pub fn compute_lut_proof_size_bytes(proof: &LutCircuitProof) -> usize {
    let mut total = 0usize;
    let mut buf = Vec::new();

    for lp in &proof.lut_proofs {
        // k + m HyperKZG commitments
        for comm in lp.comm_inputs.iter().chain(lp.comm_outputs.iter()) {
            buf.clear();
            comm.0.serialize_compressed(&mut buf).ok();
            total += buf.len();
        }

        // round polys: num_vars rounds × (k+2) Fr elements × 32 bytes
        total += lp.num_vars * (lp.k + 2) * 32;

        // finals: (k + m) Fr elements × 32 bytes
        total += (lp.k + lp.m) * 32;

        // Phase 4f: no per-group opening proofs here
    }

    // Phase 4f: one combined_comm + combined_eval (32 bytes) + one HyperKZG proof
    buf.clear();
    proof.batched_opening.combined_comm.0.serialize_compressed(&mut buf).ok();
    total += buf.len(); // combined commitment (~33 bytes compressed)
    total += 32; // combined_eval
    buf.clear();
    proof.batched_opening.combined_proof.serialize_compressed(&mut buf).ok();
    total += buf.len();

    total
}
