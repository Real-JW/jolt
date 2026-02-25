//! **Phase 4b — Lasso / LogUp lookup argument for LUT groups**
//!
//! Replaces the per-evaluation sumcheck of Phase 2 ([`lut_prover`]) with an
//! offline-memory **LogUp** (logarithmic derivative) argument.
//!
//! # Asymptotic improvement
//!
//! | Protocol | Sumcheck work | Commitments |
//! |---|---|---|
//! | Phase 2 (sumcheck) | O(N × k) — k rounds, each sweeping N entries | k + m |
//! | Phase 4b (LogUp)   | O(N + 2^k) — one M-round + one k-round check  | 5 fixed |
//!
//! For k = 6, N = 65 536: Phase 2 ≈ 6 × 65 K = 390 K inner-loop ops;
//! Phase 4b ≈ 65 K + 64 ≈ 65 K — roughly 6× cheaper on ops alone,
//! plus the k-round evaluation cost for the MLE of T̃ disappears entirely.
//!
//! # Protocol (one LUT type, k inputs, m outputs, N invocations)
//!
//! **Step 1 — Encode rows and table entries**
//! ```text
//!   pack[i]      = sum_{ℓ=0}^{k-1} in_ℓ[i] × 2^ℓ         (k-bit query index)
//!   pack_out[i]  = sum_{j=0}^{m-1}  out_j[i] × 2^j         (m-bit output word)
//!   A[i]         = fr(pack[i]) + γ × fr(pack_out[i])        (query field encoding)
//!   B[x]         = fr(x)       + γ × fr(pack_out_T[x])      (table field encoding, x=0..2^k-1)
//! ```
//! where `γ` is a Fiat-Shamir challenge and `pack_out_T[x]` is the m output bits
//! of the truth table at index x.
//!
//! **Step 2 — Count**
//! `count[x] = #{i : pack[i] = x}` (multiplicity of each table entry in queries).
//!
//! **Step 3 — LogUp identity**
//! For a random field element λ:
//! ```text
//!   ∑_{i=0}^{N-1} 1/(λ − A[i])  =  ∑_{x=0}^{2^k-1} count[x]/(λ − B[x])
//! ```
//! This is proven by committing the inverse polynomials and running two
//! combined (correction + sum) sumchecks.
//!
//! **Step 4 — Commit**
//! Five polynomials are committed via HyperKZG:
//! ```text
//!   PackedIdx  over {0,1}^M  (pack[i], padded to 2^M)
//!   PackedOut  over {0,1}^M  (pack_out[i])
//!   Count      over {0,1}^k  (count[x])
//!   InvQ       over {0,1}^M  (1/(λ − A[i]))
//!   InvT       over {0,1}^k  (1/(λ − B[x]))
//! ```
//!
//! **Step 5 — Two combined sumchecks (degree 3, 4 eval pts/round)**
//!
//! Let `ξ` be a Fiat-Shamir challenge (batches correction and sum).
//! Each sumcheck proves a combined zero-check + sum identity:
//!
//! *Query sumcheck* (M rounds over {0,1}^M):
//! ```text
//!   ∑_i eq(r_q, i) × (InvQ[i] × (λ − A[i]) − 1)  +  ξ × InvQ[i]  =  ξ × S
//! ```
//! where `S` = claimed LogUp sum (sent in proof).
//!
//! *Table sumcheck* (k rounds over {0,1}^k):
//! ```text
//!   ∑_x eq(r_t, x) × (InvT[x] × (λ − B[x]) − Count[x])  +  ξ × InvT[x]  =  ξ × S
//! ```
//!
//! Both RHS equal `ξ × S` because the zero-check term sums to 0 (for an
//! honest prover) and `∑InvQ = ∑ Count×InvT = S`.
//!
//! **Step 6 — HyperKZG openings**
//! All five committed polynomials are opened at the respective sumcheck
//! evaluation points.
//!
//! # Security
//! The LogUp identity holds with overwhelming probability over the random
//! challenges γ, λ iff the query multiset equals the table multiset (with
//! multiplicities).  Soundness follows from Schwartz–Zippel + binding of
//! HyperKZG.

use std::collections::HashMap;

use ark_bn254::{Bn254, Fr};
use ark_ff::{Field, One, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;

use jolt_core::field::JoltField;
use jolt_core::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::transcripts::{AppendToTranscript, KeccakTranscript, Transcript};

use crate::lczbc::{evaluate_lut_circuit, LutCirc, LutDesc, LutEval};

type Challenge = <Fr as JoltField>::Challenge;

// ── field helpers ─────────────────────────────────────────────────────────────

#[inline]
fn fr(n: u64) -> Fr {
    Fr::from(n)
}

/// Bind the lowest unfixed variable of `poly` at point `r` in-place.
/// poly[i] ← poly[2i] + r·(poly[2i+1] − poly[2i])
fn bind(poly: &mut Vec<Fr>, r: Fr) {
    let half = poly.len() / 2;
    for i in 0..half {
        poly[i] = poly[2 * i] + r * (poly[2 * i + 1] - poly[2 * i]);
    }
    poly.truncate(half);
}

/// Initialise the equality polynomial eq(r, ·) over {0,1}^m in LSB-first order.
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

/// eq(r_input, r_sc) using LSB-first variable ordering.
fn eq_final_eval(r_input: &[Fr], r_sc: &[Fr]) -> Fr {
    assert_eq!(r_input.len(), r_sc.len());
    r_input
        .iter()
        .zip(r_sc.iter())
        .map(|(&ri, &si)| ri * si + (Fr::one() - ri) * (Fr::one() - si))
        .product()
}

/// Evaluate a polynomial of degree d−1, given its values at integers 0..d,
/// at an arbitrary field point t using Lagrange interpolation.
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
        result += basis * Field::inverse(&denom).expect("interpolation denom != 0");
    }
    result
}

// ── Table encoding helper ─────────────────────────────────────────────────────

/// Precompute the table field encoding `B[x] = fr(x) + γ × fr(pack_out_T[x])`
/// for x = 0..2^k, given the LUT truth table.
fn table_encoding(desc: &LutDesc, gamma: Fr) -> Vec<Fr> {
    let n = 1usize << desc.k;
    (0..n)
        .map(|x| {
            // Reconstruct the integer packing of all m output bits for table entry x.
            let mut pack_out = 0u64;
            for j in 0..desc.m {
                let bit_pos = x * desc.m + j;
                let bit = (desc.truth_table[bit_pos / 8] >> (bit_pos % 8)) & 1;
                if bit != 0 {
                    pack_out |= 1u64 << j;
                }
            }
            fr(x as u64) + gamma * fr(pack_out)
        })
        .collect()
}

// ── Proof structs ─────────────────────────────────────────────────────────────

/// LogUp / Lasso NIZK for N invocations of a single LUT type.
pub struct LassoLutGroupProof {
    pub lut_id:    u32,
    pub k:         usize,
    pub m:         usize,
    /// N (un-padded invocation count).
    pub num_evals: usize,
    /// M = ⌈log₂ N⌉ (query polynomial variables).
    pub num_vars:  usize,

    // ── Commitments ──────────────────────────────────────────────────────────
    /// Pack[i] = ∑ℓ in_ℓ[i]·2^ℓ (k-bit integer, over {0,1}^M).
    pub comm_packed_idx: HyperKZGCommitment<Bn254>,
    /// PackOut[i] = ∑_j out_j[i]·2^j (m-bit integer, over {0,1}^M).
    pub comm_packed_out: HyperKZGCommitment<Bn254>,
    /// Count[x] = multiplicity of entry x in queries (over {0,1}^k).
    pub comm_count:      HyperKZGCommitment<Bn254>,
    /// InvQ[i] = 1/(λ − A[i]) (over {0,1}^M).
    pub comm_inv_q:      HyperKZGCommitment<Bn254>,
    /// InvT[x] = 1/(λ − B[x]) (over {0,1}^k).
    pub comm_inv_t:      HyperKZGCommitment<Bn254>,

    // ── Claimed LogUp sum: ∑InvQ = ∑(Count·InvT) ─────────────────────────
    pub claimed_sum: Fr,

    // ── Query sumcheck (M rounds, degree 3, 4 pts/round) ─────────────────
    /// Round polys for ∑_i [eq(r_q,i)·(InvQ[i]·(λ−A[i])−1) + ξ·InvQ[i]] = ξ·S.
    pub sc_q_polys: Vec<Vec<Fr>>,
    /// InvQ(r_sc_q), PackedIdx(r_sc_q), PackedOut(r_sc_q).
    pub finals_q:   [Fr; 3],
    pub opening_inv_q:      Option<HyperKZGProof<Bn254>>,
    pub opening_packed_idx: Option<HyperKZGProof<Bn254>>,
    pub opening_packed_out: Option<HyperKZGProof<Bn254>>,

    // ── Table sumcheck (k rounds, degree 3, 4 pts/round) ─────────────────
    /// Round polys for ∑_x [eq(r_t,x)·(InvT[x]·(λ−B[x])−Count[x]) + ξ·InvT[x]] = ξ·S.
    pub sc_t_polys: Vec<Vec<Fr>>,
    /// InvT(r_sc_t), Count(r_sc_t).
    pub finals_t:   [Fr; 2],
    pub opening_inv_t: Option<HyperKZGProof<Bn254>>,
    pub opening_count: Option<HyperKZGProof<Bn254>>,
}

/// Complete Lasso-mode NIZK for a full LUT-annotated circuit execution.
pub struct LassoLutCircuitProof {
    pub max_num_vars: usize,
    pub lut_proofs:   Vec<LassoLutGroupProof>,
    pub outputs:      Vec<bool>,
}

// ── Prover ────────────────────────────────────────────────────────────────────

/// Prove all N invocations of a single LUT type using the LogUp argument.
pub fn prove_lasso_lut_group(
    desc:       &LutDesc,
    evals:      &[LutEval],
    pk:         &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> LassoLutGroupProof {
    let k   = desc.k;
    let m   = desc.m;
    let n   = evals.len();
    assert!(n > 0, "prove_lasso_lut_group: empty group for lut_id={}", desc.lut_id);

    // M = ⌈log₂ n⌉, padded query domain size cap = 2^M.
    let big_m = usize::max(1, n.next_power_of_two().trailing_zeros() as usize);
    let cap   = 1usize << big_m;
    // Table domain size.
    let table_size = 1usize << k;

    // ── Step 1: Build packed-index and packed-output vectors (padded) ────────
    // pack[i]     = integer index into truth table (LSB-first)
    // pack_out[i] = integer packing of m output bits
    let (pack_idx_vec, pack_out_vec): (Vec<Fr>, Vec<Fr>) = {
        let mut idx_v = Vec::with_capacity(cap);
        let mut out_v = Vec::with_capacity(cap);
        for i in 0..cap {
            if i < n {
                let e = &evals[i];
                let mut idx: u64 = 0;
                for (wire_l, &b) in e.inputs.iter().enumerate() {
                    if b { idx |= 1u64 << wire_l; }
                }
                let mut out: u64 = 0;
                for (j, &b) in e.outputs.iter().enumerate() {
                    if b { out |= 1u64 << j; }
                }
                idx_v.push(fr(idx));
                out_v.push(fr(out));
            } else {
                // Padding rows: use index 0 (a valid table entry).
                idx_v.push(Fr::zero());
                out_v.push(fr({
                    // output for table entry 0
                    let mut out: u64 = 0;
                    for j in 0..m {
                        let bit_pos = j; // x=0 → bit_pos = 0*m + j = j
                        let bit = (desc.truth_table[bit_pos / 8] >> (bit_pos % 8)) & 1;
                        if bit != 0 { out |= 1u64 << j; }
                    }
                    out
                }));
            }
        }
        (idx_v, out_v)
    };

    // ── Step 2: Compute multiplicity / count table ───────────────────────────
    let mut count_vec: Vec<u64> = vec![0u64; table_size];
    for i in 0..cap {
        let idx = pack_idx_vec[i].into_bigint().as_ref()[0] as usize;
        if idx < table_size {
            count_vec[idx] += 1;
        }
    }
    let count_fr: Vec<Fr> = count_vec.iter().map(|&c| fr(c)).collect();

    // ── Commit PackedIdx, PackedOut, Count ───────────────────────────────────
    let mle_packed_idx = MultilinearPolynomial::from(pack_idx_vec.clone());
    let mle_packed_out = MultilinearPolynomial::from(pack_out_vec.clone());
    let mle_count      = MultilinearPolynomial::from(count_fr.clone());

    let comm_packed_idx =
        HyperKZG::<Bn254>::commit(pk, &mle_packed_idx).expect("commit PackedIdx");
    let comm_packed_out =
        HyperKZG::<Bn254>::commit(pk, &mle_packed_out).expect("commit PackedOut");
    let comm_count =
        HyperKZG::<Bn254>::commit(pk, &mle_count).expect("commit Count");

    // ── Fiat-Shamir: metadata + early commitments ────────────────────────────
    transcript.append_u64(desc.lut_id as u64);
    transcript.append_u64(n as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(m as u64);
    transcript.append_u64(big_m as u64);
    comm_packed_idx.append_to_transcript(transcript);
    comm_packed_out.append_to_transcript(transcript);
    comm_count.append_to_transcript(transcript);

    // γ: mixes input index and output bits into a single field encoding.
    let gamma: Fr = transcript.challenge_scalar_optimized::<Fr>().into();
    // λ: denominator challenge for LogUp.
    let lambda: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── Step 3: Compute table encoding B[x] (public, derived from truth table) ─
    let b_enc: Vec<Fr> = table_encoding(desc, gamma);

    // ── Step 4: Compute query encoding A[i] and inverse polynomials ─────────
    // A[i] = pack_idx[i] + γ × pack_out[i]
    let a_enc: Vec<Fr> = pack_idx_vec
        .iter()
        .zip(pack_out_vec.iter())
        .map(|(&p, &o)| p + gamma * o)
        .collect();

    // InvQ[i] = 1/(λ − A[i])
    let inv_q_vec: Vec<Fr> = a_enc
        .iter()
        .map(|&ai| {
            let denom = lambda - ai;
            Field::inverse(&denom).expect("λ - A[i] = 0: extremely unlikely for random λ")
        })
        .collect();

    // InvT[x] = 1/(λ − B[x])
    let inv_t_vec: Vec<Fr> = b_enc
        .iter()
        .map(|&bx| {
            let denom = lambda - bx;
            Field::inverse(&denom).expect("λ - B[x] = 0: extremely unlikely for random λ")
        })
        .collect();

    // ── Claimed LogUp sums ────────────────────────────────────────────────────
    let sum_inv_q: Fr = inv_q_vec.iter().copied().sum();
    let _sum_inv_t: Fr = (0..table_size).map(|x| count_fr[x] * inv_t_vec[x]).sum();
    // For an honest prover, sum_inv_q == sum_inv_t.
    // We use sum_inv_q as the canonical claimed sum S.
    let claimed_sum = sum_inv_q;

    // ── Commit InvQ, InvT ─────────────────────────────────────────────────────
    let mle_inv_q = MultilinearPolynomial::from(inv_q_vec.clone());
    let mle_inv_t = MultilinearPolynomial::from(inv_t_vec.clone());

    let comm_inv_q = HyperKZG::<Bn254>::commit(pk, &mle_inv_q).expect("commit InvQ");
    let comm_inv_t = HyperKZG::<Bn254>::commit(pk, &mle_inv_t).expect("commit InvT");

    comm_inv_q.append_to_transcript(transcript);
    comm_inv_t.append_to_transcript(transcript);
    transcript.append_scalar(&claimed_sum);

    // ξ: challenge that batches the zero-check term and the sum claim.
    let xi: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── Query sumcheck (M rounds over {0,1}^M) ────────────────────────────────
    // Proves: ∑_i [eq(r_q, i)·(InvQ[i]·(λ−A[i])−1) + ξ·InvQ[i]] = ξ·S
    //       = ∑_i [eq(r_q, i)·(InvQ[i]·(λ−(Pack[i]+γ·POut[i]))−1) + ξ·InvQ[i]]
    //
    // Degree per round: 3 (eq linear, InvQ linear, (λ−A) linear → product degree 2,
    //                      with eq head factor → degree 3).  Need 4 eval points.

    let r_q: Vec<Fr> = transcript.challenge_vector(big_m);

    let mut sc_q_polys: Vec<Vec<Fr>>          = Vec::with_capacity(big_m);
    let mut r_sc_q_fr:  Vec<Fr>               = Vec::with_capacity(big_m);
    let mut r_sc_q_ch:  Vec<Challenge>        = Vec::with_capacity(big_m);

    // Mutable working copies
    let mut eq_q        = init_eq(&r_q);
    let mut wk_inv_q    = inv_q_vec.clone();
    let mut wk_pack_idx = pack_idx_vec.clone();
    let mut wk_pack_out = pack_out_vec.clone();

    let eval_pts: Vec<Fr> = (0..4u64).map(fr).collect();

    for _round in 0..big_m {
        let half = eq_q.len() / 2;
        let mut p_evals = [Fr::zero(); 4];

        for idx in 0..half {
            let eq_lo = eq_q[2 * idx];
            let eq_hi = eq_q[2 * idx + 1];
            let iq_lo = wk_inv_q[2 * idx];
            let iq_hi = wk_inv_q[2 * idx + 1];
            let pi_lo = wk_pack_idx[2 * idx];
            let pi_hi = wk_pack_idx[2 * idx + 1];
            let po_lo = wk_pack_out[2 * idx];
            let po_hi = wk_pack_out[2 * idx + 1];

            for (pt, &t) in eval_pts.iter().enumerate() {
                let eq_t  = eq_lo + t * (eq_hi - eq_lo);
                let iq_t  = iq_lo + t * (iq_hi - iq_lo);
                let pi_t  = pi_lo + t * (pi_hi - pi_lo);
                let po_t  = po_lo + t * (po_hi - po_lo);
                let a_t   = pi_t + gamma * po_t;
                // zero-check term: eq·(InvQ·(λ−A)−1)
                let zero_check = eq_t * (iq_t * (lambda - a_t) - Fr::one());
                // sum term: ξ·InvQ
                p_evals[pt] += zero_check + xi * iq_t;
            }
        }

        for &e in p_evals.iter() {
            transcript.append_scalar(&e);
        }

        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_q_ch.push(r_j_ch);
        r_sc_q_fr.push(r_j);

        bind(&mut eq_q, r_j);
        bind(&mut wk_inv_q, r_j);
        bind(&mut wk_pack_idx, r_j);
        bind(&mut wk_pack_out, r_j);

        sc_q_polys.push(p_evals.to_vec());
    }

    let finals_q = [wk_inv_q[0], wk_pack_idx[0], wk_pack_out[0]];
    for &v in finals_q.iter() {
        transcript.append_scalar(&v);
    }

    // ── Table sumcheck (k rounds over {0,1}^k) ────────────────────────────────
    // Proves: ∑_x [eq(r_t, x)·(InvT[x]·(λ−B[x])−Count[x]) + ξ·InvT[x]] = ξ·S
    //
    // B[x] is a precomputed constant vector (not a committed polynomial):
    //   - In each sumcheck round, b_enc is bound like a regular polynomial.
    // Degree per round: 3 (same analysis; B[x] is linear per variable as it's
    //                      a multilinear polynomial).  Need 4 eval points.

    let r_t: Vec<Fr> = transcript.challenge_vector(k);

    let mut sc_t_polys: Vec<Vec<Fr>>  = Vec::with_capacity(k);
    let mut r_sc_t_fr:  Vec<Fr>       = Vec::with_capacity(k);
    let mut r_sc_t_ch:  Vec<Challenge> = Vec::with_capacity(k);

    let mut eq_t       = init_eq(&r_t);
    let mut wk_inv_t   = inv_t_vec.clone();
    let mut wk_count   = count_fr.clone();
    let mut wk_b_enc   = b_enc.clone();

    for _round in 0..k {
        let half = eq_t.len() / 2;
        let mut p_evals = [Fr::zero(); 4];

        for idx in 0..half {
            let eq_lo = eq_t[2 * idx];
            let eq_hi = eq_t[2 * idx + 1];
            let it_lo = wk_inv_t[2 * idx];
            let it_hi = wk_inv_t[2 * idx + 1];
            let ct_lo = wk_count[2 * idx];
            let ct_hi = wk_count[2 * idx + 1];
            let b_lo  = wk_b_enc[2 * idx];
            let b_hi  = wk_b_enc[2 * idx + 1];

            for (pt, &t) in eval_pts.iter().enumerate() {
                let eq_t_v = eq_lo + t * (eq_hi - eq_lo);
                let it_t   = it_lo + t * (it_hi - it_lo);
                let ct_t   = ct_lo + t * (ct_hi - ct_lo);
                let b_t    = b_lo  + t * (b_hi  - b_lo);
                // zero-check: InvT[x]·(λ−B[x]) = 1  (table inverse correctness)
                let zero_check = eq_t_v * (it_t * (lambda - b_t) - Fr::one());
                // sum term: ξ·Count[x]·InvT[x]  (weighted LogUp sum)
                p_evals[pt] += zero_check + xi * ct_t * it_t;
            }
        }

        for &e in p_evals.iter() {
            transcript.append_scalar(&e);
        }

        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_t_ch.push(r_j_ch);
        r_sc_t_fr.push(r_j);

        bind(&mut eq_t, r_j);
        bind(&mut wk_inv_t, r_j);
        bind(&mut wk_count, r_j);
        bind(&mut wk_b_enc, r_j);

        sc_t_polys.push(p_evals.to_vec());
    }

    let finals_t = [wk_inv_t[0], wk_count[0]];
    for &v in finals_t.iter() {
        transcript.append_scalar(&v);
    }

    // ── HyperKZG opening proofs ──────────────────────────────────────────────
    // HyperKZG uses big-endian (MSB-first) variable ordering; our sumcheck used
    // LSB-first, so we reverse the challenge vectors.
    let point_q_kzg: Vec<Challenge> = r_sc_q_ch.iter().rev().cloned().collect();
    let point_t_kzg: Vec<Challenge> = r_sc_t_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();

    let open = |mle: &MultilinearPolynomial<Fr>,
                comm: &HyperKZGCommitment<Bn254>,
                eval: &Fr,
                point: &[Challenge],
                t:    &mut KeccakTranscript|
     -> Option<HyperKZGProof<Bn254>> {
        if *comm == zero_comm {
            None
        } else {
            Some(HyperKZG::<Bn254>::open(pk, mle, point, eval, t).expect("HyperKZG open failed"))
        }
    };

    let opening_inv_q      = open(&mle_inv_q,      &comm_inv_q,      &finals_q[0], &point_q_kzg, transcript);
    let opening_packed_idx = open(&mle_packed_idx,  &comm_packed_idx,  &finals_q[1], &point_q_kzg, transcript);
    let opening_packed_out = open(&mle_packed_out,  &comm_packed_out,  &finals_q[2], &point_q_kzg, transcript);
    let opening_inv_t      = open(&mle_inv_t,      &comm_inv_t,      &finals_t[0], &point_t_kzg, transcript);
    let opening_count      = open(&mle_count,      &comm_count,      &finals_t[1], &point_t_kzg, transcript);

    LassoLutGroupProof {
        lut_id: desc.lut_id,
        k,
        m,
        num_evals: n,
        num_vars: big_m,
        comm_packed_idx,
        comm_packed_out,
        comm_count,
        comm_inv_q,
        comm_inv_t,
        claimed_sum,
        sc_q_polys,
        finals_q,
        opening_inv_q,
        opening_packed_idx,
        opening_packed_out,
        sc_t_polys,
        finals_t,
        opening_inv_t,
        opening_count,
    }
}

// ── Verifier ──────────────────────────────────────────────────────────────────

/// Verify a [`LassoLutGroupProof`].
///
/// Returns `true` iff all checks pass.
pub fn verify_lasso_lut_group(
    proof:      &LassoLutGroupProof,
    desc:       &LutDesc,
    vk:         &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    let k     = proof.k;
    let m     = proof.m;
    let big_m = proof.num_vars;
    let _table_size = 1usize << k;

    // ── Re-derive Fiat-Shamir challenges ─────────────────────────────────────
    transcript.append_u64(proof.lut_id as u64);
    transcript.append_u64(proof.num_evals as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(m as u64);
    transcript.append_u64(big_m as u64);
    proof.comm_packed_idx.append_to_transcript(transcript);
    proof.comm_packed_out.append_to_transcript(transcript);
    proof.comm_count.append_to_transcript(transcript);

    let gamma: Fr  = transcript.challenge_scalar_optimized::<Fr>().into();
    let lambda: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // Recompute public table encoding B[x].
    let b_enc = table_encoding(desc, gamma);

    proof.comm_inv_q.append_to_transcript(transcript);
    proof.comm_inv_t.append_to_transcript(transcript);
    transcript.append_scalar(&proof.claimed_sum);

    let xi: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── Replay query sumcheck ─────────────────────────────────────────────────
    let r_q: Vec<Fr> = transcript.challenge_vector(big_m);
    let mut prev_claim_q = xi * proof.claimed_sum;
    let mut r_sc_q_fr:  Vec<Fr>    = Vec::with_capacity(big_m);
    let mut r_sc_q_ch:  Vec<Challenge> = Vec::with_capacity(big_m);

    for round in 0..big_m {
        let p = &proof.sc_q_polys[round];
        if p.len() != 4 {
            eprintln!("lut_id={}: query sc round {round}: wrong poly len {}", proof.lut_id, p.len());
            return false;
        }
        // p(0) + p(1) must equal the previous claim.
        let sum = p[0] + p[1];
        if sum != prev_claim_q {
            eprintln!("lut_id={}: query sc round {round}: p(0)+p(1)={sum:?} ≠ {prev_claim_q:?}", proof.lut_id);
            return false;
        }
        for &e in p.iter() { transcript.append_scalar(&e); }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_q_ch.push(r_j_ch);
        r_sc_q_fr.push(r_j);
        prev_claim_q = poly_at(p, r_j);
    }

    // Final query sumcheck check:
    //   eq(r_q, r_sc_q) × (InvQ_final × (λ − A_final) − 1) + ξ × InvQ_final = prev_claim_q
    let eq_q_final = eq_final_eval(&r_q, &r_sc_q_fr);
    let inv_q_fin  = proof.finals_q[0];
    let pack_idx_fin = proof.finals_q[1];
    let pack_out_fin = proof.finals_q[2];
    let a_final = pack_idx_fin + gamma * pack_out_fin;
    let expected_q = eq_q_final * (inv_q_fin * (lambda - a_final) - Fr::one()) + xi * inv_q_fin;
    if expected_q != prev_claim_q {
        eprintln!("lut_id={}: query sc final check failed: {expected_q:?} ≠ {prev_claim_q:?}", proof.lut_id);
        return false;
    }

    for &v in proof.finals_q.iter() { transcript.append_scalar(&v); }

    // ── Replay table sumcheck ─────────────────────────────────────────────────
    let r_t: Vec<Fr> = transcript.challenge_vector(k);
    let mut prev_claim_t = xi * proof.claimed_sum;
    let mut r_sc_t_fr:  Vec<Fr>    = Vec::with_capacity(k);
    let mut r_sc_t_ch:  Vec<Challenge> = Vec::with_capacity(k);

    for round in 0..k {
        let p = &proof.sc_t_polys[round];
        if p.len() != 4 {
            eprintln!("lut_id={}: table sc round {round}: wrong poly len {}", proof.lut_id, p.len());
            return false;
        }
        let sum = p[0] + p[1];
        if sum != prev_claim_t {
            eprintln!("lut_id={}: table sc round {round}: p(0)+p(1)={sum:?} ≠ {prev_claim_t:?}", proof.lut_id);
            return false;
        }
        for &e in p.iter() { transcript.append_scalar(&e); }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_t_ch.push(r_j_ch);
        r_sc_t_fr.push(r_j);
        prev_claim_t = poly_at(p, r_j);
    }

    // Final table sumcheck check:
    //   eq(r_t, r_sc_t) × (InvT_final × (λ − B_final) − Count_final) + ξ × InvT_final = prev_claim_t
    // B_final = MLE of b_enc evaluated at r_sc_t.
    let eq_t_final   = eq_final_eval(&r_t, &r_sc_t_fr);
    let inv_t_fin    = proof.finals_t[0];
    let count_fin    = proof.finals_t[1];

    // Evaluate public B(r_sc_t) using the standard bind algorithm on b_enc.
    let b_final: Fr = {
        let mut bv = b_enc.clone();
        for &rj in &r_sc_t_fr {
            bind(&mut bv, rj);
        }
        bv[0]
    };

    let expected_t = eq_t_final * (inv_t_fin * (lambda - b_final) - Fr::one()) + xi * count_fin * inv_t_fin;
    if expected_t != prev_claim_t {
        eprintln!("lut_id={}: table sc final check failed: {expected_t:?} ≠ {prev_claim_t:?}", proof.lut_id);
        return false;
    }

    for &v in proof.finals_t.iter() { transcript.append_scalar(&v); }

    // ── Check LogUp sum equality ──────────────────────────────────────────────
    // The verifier checks claimed_sum was consistently used in both sumchecks
    // (already enforced by FS transcript binding).  The additional soundness
    // guarantee comes from: if sum_inv_q ≠ sum_inv_t, then ξ × sum_inv_q ≠
    // ξ × sum_inv_t, and at least one sumcheck starting claim will be wrong.
    // (This is implicit in the transcript binding above.)

    // ── Verify HyperKZG openings ──────────────────────────────────────────────
    let point_q_kzg: Vec<Challenge> = r_sc_q_ch.iter().rev().cloned().collect();
    let point_t_kzg: Vec<Challenge> = r_sc_t_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();

    let verify_one = |name: &str,
                      lut_id: u32,
                      comm: &HyperKZGCommitment<Bn254>,
                      eval: &Fr,
                      opening: &Option<HyperKZGProof<Bn254>>,
                      point: &[Challenge],
                      t:     &mut KeccakTranscript|
     -> bool {
        if *comm == zero_comm {
            if *eval != Fr::zero() {
                eprintln!("lut_id={lut_id}: {name}: zero comm but non-zero eval");
                return false;
            }
            true
        } else if let Some(ref pf) = opening {
            if HyperKZG::<Bn254>::verify(vk, comm, point, eval, pf, t).is_err() {
                eprintln!("lut_id={lut_id}: {name}: HyperKZG verify FAILED");
                return false;
            }
            true
        } else {
            eprintln!("lut_id={lut_id}: {name}: non-zero comm but no opening proof");
            false
        }
    };

    let ok = verify_one("InvQ",      proof.lut_id, &proof.comm_inv_q,      &proof.finals_q[0], &proof.opening_inv_q,      &point_q_kzg, transcript)
          && verify_one("PackedIdx", proof.lut_id, &proof.comm_packed_idx,  &proof.finals_q[1], &proof.opening_packed_idx, &point_q_kzg, transcript)
          && verify_one("PackedOut", proof.lut_id, &proof.comm_packed_out,  &proof.finals_q[2], &proof.opening_packed_out, &point_q_kzg, transcript)
          && verify_one("InvT",      proof.lut_id, &proof.comm_inv_t,      &proof.finals_t[0], &proof.opening_inv_t,      &point_t_kzg, transcript)
          && verify_one("Count",     proof.lut_id, &proof.comm_count,      &proof.finals_t[1], &proof.opening_count,      &point_t_kzg, transcript);

    ok
}

// ── Circuit-level wrappers ────────────────────────────────────────────────────

/// Estimate the maximum sumcheck variables needed for the Lasso prover.
/// (Query side: M = ⌈log₂(max_invocations_per_type)⌉, same as Phase 2.)
pub fn compute_max_num_vars_lasso(circ: &LutCirc, cycles: u32) -> usize {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for op in &circ.ops {
        *counts.entry(op.lut_id).or_insert(0) += 1;
    }
    let cycles = cycles.max(1) as usize;
    let max_count = counts.values().copied().max().unwrap_or(1) * cycles;
    let query_vars = usize::max(1, max_count.next_power_of_two().trailing_zeros() as usize);
    // Table-side polynomials are committed over {0,1}^k.
    // The SRS must be at least k variables, so take the max over all LUT types.
    let max_k = circ.lut_types.values().map(|d| d.k).max().unwrap_or(0);
    usize::max(query_vars, max_k)
}

/// Prove a full LUT-annotated circuit execution using the Lasso/LogUp prover.
pub fn prove_lasso_lut_circuit(
    circ:       &LutCirc,
    inputs:     &[bool],
    cycles:     u32,
    pk:         &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> LassoLutCircuitProof {
    let (trace, outputs) = evaluate_lut_circuit(circ, inputs, cycles);

    // Group trace rows by lut_id.
    let mut groups: HashMap<u32, Vec<LutEval>> = HashMap::new();
    for row in &trace {
        groups.entry(row.lut_id).or_default().push(row.clone());
    }

    let max_num_vars = compute_max_num_vars_lasso(circ, cycles);

    let mut lut_ids: Vec<u32> = groups.keys().copied().collect();
    lut_ids.sort_unstable();

    let mut lut_proofs = Vec::with_capacity(lut_ids.len());
    for lut_id in lut_ids {
        let desc  = circ.lut_types.get(&lut_id)
            .unwrap_or_else(|| panic!("unknown lut_id {lut_id}"));
        let evals = groups.remove(&lut_id).unwrap();
        let proof = prove_lasso_lut_group(desc, &evals, pk, transcript);
        lut_proofs.push(proof);
    }

    LassoLutCircuitProof { max_num_vars, lut_proofs, outputs }
}

/// Verify a complete Lasso LUT circuit proof.
pub fn verify_lasso_lut_circuit(
    proof:      &LassoLutCircuitProof,
    circ:       &LutCirc,
    vk:         &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    for lp in &proof.lut_proofs {
        let desc = circ.lut_types.get(&lp.lut_id)
            .unwrap_or_else(|| panic!("unknown lut_id {} in proof", lp.lut_id));
        if !verify_lasso_lut_group(lp, desc, vk, transcript) {
            eprintln!("Lasso LUT group proof FAILED for lut_id={}", lp.lut_id);
            return false;
        }
    }
    true
}

/// Estimate serialised proof size in bytes.
pub fn compute_lasso_proof_size_bytes(proof: &LassoLutCircuitProof) -> usize {
    let mut total = 0usize;
    let mut buf   = Vec::new();

    for lp in &proof.lut_proofs {
        // 5 HyperKZG commitments.
        for comm in [
            &lp.comm_packed_idx,
            &lp.comm_packed_out,
            &lp.comm_count,
            &lp.comm_inv_q,
            &lp.comm_inv_t,
        ] {
            buf.clear();
            comm.0.serialize_compressed(&mut buf).ok();
            total += buf.len();
        }
        // Claimed sum: 1 Fr (32 bytes).
        total += 32;

        // Query sumcheck: big_m rounds × 4 Fr.
        total += lp.num_vars * 4 * 32;
        // Query finals: 3 Fr.
        total += 3 * 32;

        // Table sumcheck: k rounds × 4 Fr.
        total += lp.k * 4 * 32;
        // Table finals: 2 Fr.
        total += 2 * 32;

        // Opening proofs.
        for opt_pf in [
            &lp.opening_inv_q,
            &lp.opening_packed_idx,
            &lp.opening_packed_out,
            &lp.opening_inv_t,
            &lp.opening_count,
        ] {
            if let Some(pf) = opt_pf {
                buf.clear();
                pf.serialize_compressed(&mut buf).ok();
                total += buf.len();
            }
        }
    }
    total
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lczbc::{LutCirc, LutDesc, LutEval, LutOp};
    use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
    use jolt_core::poly::commitment::hyperkzg::HyperKZG;
    use std::collections::HashMap;

    type PCS = HyperKZG<Bn254>;

    /// Build a minimal test LUT desc.
    fn xor_desc() -> LutDesc {
        LutDesc {
            lut_id:      1,
            k:           2,
            m:           1,
            // XOR: table = [0,1,1,0] packed LSB-first = 0x06
            truth_table: vec![0x06],
        }
    }

    fn and_desc() -> LutDesc {
        LutDesc {
            lut_id:      2,
            k:           2,
            m:           1,
            // AND: table = [0,0,0,1] packed = 0x08
            truth_table: vec![0x08],
        }
    }

    fn make_xor_evals(n: usize) -> Vec<LutEval> {
        let pairs = [(false, false), (true, false), (false, true), (true, true)];
        (0..n)
            .map(|i| {
                let (a, b) = pairs[i % 4];
                LutEval {
                    lut_id:  1,
                    inputs:  vec![a, b],
                    outputs: vec![a ^ b],
                }
            })
            .collect()
    }

    #[test]
    fn lasso_xor_proves_and_verifies() {
        let desc  = xor_desc();
        let evals = make_xor_evals(8);
        let num_vars = 3usize; // 2^3 = 8 >= n=8

        let pk = PCS::setup_prover(num_vars);
        let vk = PCS::setup_verifier(&pk);

        let mut pt = KeccakTranscript::new(b"lasso-test");
        let proof = prove_lasso_lut_group(&desc, &evals, &pk, &mut pt);

        let mut vt = KeccakTranscript::new(b"lasso-test");
        assert!(
            verify_lasso_lut_group(&proof, &desc, &vk, &mut vt),
            "XOR Lasso proof should verify"
        );
    }

    #[test]
    fn lasso_and_proves_and_verifies() {
        let desc  = and_desc();
        let evals: Vec<LutEval> = vec![
            LutEval { lut_id: 2, inputs: vec![true, true],   outputs: vec![true]  },
            LutEval { lut_id: 2, inputs: vec![false, true],  outputs: vec![false] },
            LutEval { lut_id: 2, inputs: vec![true, false],  outputs: vec![false] },
            LutEval { lut_id: 2, inputs: vec![false, false], outputs: vec![false] },
        ];

        let pk = PCS::setup_prover(2usize);
        let vk = PCS::setup_verifier(&pk);

        let mut pt = KeccakTranscript::new(b"lasso-and");
        let proof = prove_lasso_lut_group(&desc, &evals, &pk, &mut pt);

        let mut vt = KeccakTranscript::new(b"lasso-and");
        assert!(
            verify_lasso_lut_group(&proof, &desc, &vk, &mut vt),
            "AND Lasso proof should verify"
        );
    }

    /// Tampered proof: wrong output bit — should fail.
    #[test]
    fn lasso_tampered_output_fails() {
        let desc = xor_desc();
        let mut evals = make_xor_evals(4);
        // Flip the output of the first row.
        evals[0].outputs[0] = !evals[0].outputs[0];

        let pk = PCS::setup_prover(2usize);
        let vk = PCS::setup_verifier(&pk);

        let mut pt = KeccakTranscript::new(b"lasso-tamper");
        let proof = prove_lasso_lut_group(&desc, &evals, &pk, &mut pt);

        let mut vt = KeccakTranscript::new(b"lasso-tamper");
        // The proof commits the wrong values; verification should fail.
        assert!(
            !verify_lasso_lut_group(&proof, &desc, &vk, &mut vt),
            "Tampered output should fail verification"
        );
    }

    /// Full circuit test: single-XOR circuit, 1 cycle.
    #[test]
    fn lasso_circuit_single_xor() {
        let xor_lut = xor_desc();
        let mut lut_types = HashMap::new();
        lut_types.insert(1u32, xor_lut);

        let circ = LutCirc {
            num_wires:      3,
            primary_inputs: vec![0, 1],
            registers:      vec![],
            outputs:        vec![2],
            lut_types,
            ops: vec![LutOp {
                lut_id:    1,
                dst_wire:  2,
                src_wires: vec![0, 1],
            }],
            default_cycles: 1,
        };

        let max_vars = compute_max_num_vars_lasso(&circ, 1);
        let pk = PCS::setup_prover(max_vars);
        let vk = PCS::setup_verifier(&pk);

        let inputs = vec![true, false]; // 1 XOR 0 = 1
        let mut pt = KeccakTranscript::new(b"circuit-lasso");
        let proof = prove_lasso_lut_circuit(&circ, &inputs, 1, &pk, &mut pt);
        assert_eq!(proof.outputs[0], true);

        let mut vt = KeccakTranscript::new(b"circuit-lasso");
        assert!(
            verify_lasso_lut_circuit(&proof, &circ, &vk, &mut vt),
            "circuit Lasso proof should verify"
        );
    }
}
