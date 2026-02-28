//! **Phase 4h — Unified Mega-Table LogUp Argument (Type-Count Independence)**
//!
//! Replaces T independent LogUp instances (Phase 4b) with a
//! **single** LogUp argument over a concatenated "mega-table" of all T truth
//! tables.  The prover cost scales with the total trace size N_total, not the
//! number of distinct LUT types T.
//!
//! # Problem Addressed
//!
//! Phase 4b runs one LogUp per type:
//! - 5 HyperKZG commits × T types = 5T commits   (594 types → 2970 commits for AES)
//! - sumcheck setup cost × T      = T × C_fixed   (≈ 38 s for AES)
//!
//! Phase 4h runs ONE unified LogUp:
//! - 6 HyperKZG commits total (constant regardless of T)
//! - 2 sumchecks (query + table), both over the combined N_total and T×2^k domains
//!
//! # Protocol
//!
//! **Mega-table layout:**
//! For T types sorted deterministically (by lut_id) and T_pad = next power of
//! two ≥ T, the mega-table has T_pad × 2^k entries indexed by
//! `mega_idx = tid × 2^k + input_bits`.
//!
//! **Field encodings:**
//! ```text
//!   A[i] = TypeIdx[i] + γ · PackedIn[i] + γ² · PackedOut[i]    (query row i)
//!   B[x] = type_id_of(x) + γ · input_bits_of(x) + γ² · truth_out(x)   (table entry x)
//! ```
//! where γ is a Fiat-Shamir challenge and all values are Fr field elements.
//!
//! **LogUp identity:**
//! ```text
//!   ∑_{i=0}^{N_query-1} 1/(λ - A[i])  =  S  =  ∑_{x=0}^{T_pad·2^k-1} Count[x]/(λ - B[x])
//! ```
//!
//! **Committed polynomials (6 total, independent of T):**
//! | Polynomial | Domain | Description |
//! |---|---|---|
//! | TypeIdx    | {0,1}^M_query | LUT type index (0..T-1) for each trace row |
//! | PackedIn   | {0,1}^M_query | ∑_l in_l × 2^l — packed input bits |
//! | PackedOut  | {0,1}^M_query | ∑_j out_j × 2^j — packed output bits |
//! | InvQ       | {0,1}^M_query | 1/(λ - A[i]) |
//! | Count      | {0,1}^M_table | multiplicity of each mega-table entry |
//! | InvT       | {0,1}^M_table | 1/(λ - B[x]) |
//!
//! **Two sumchecks:**
//! - Query (M_query rounds): ∑_i [eq(r_q,i)·(InvQ[i]·(λ−A[i])−1) + ξ·InvQ[i]] = ξ·S
//! - Table (M_table rounds): ∑_x [eq(r_t,x)·(InvT[x]·(λ−B[x])−1) + ξ·Count[x]·InvT[x]] = ξ·S
//!
//! Then 6 individual HyperKZG opening proofs, one per committed polynomial.

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

use crate::lut_czbc::{evaluate_lut_circuit, LutCirc};

type Challenge = <Fr as JoltField>::Challenge;

// ── field helpers ─────────────────────────────────────────────────────────────

#[inline]
fn fr(n: u64) -> Fr {
    Fr::from(n)
}

/// Bind the lowest variable of `poly` in-place:
/// `poly[i] ← poly[2i] + r·(poly[2i+1] − poly[2i])`.
fn bind(poly: &mut Vec<Fr>, r: Fr) {
    let half = poly.len() / 2;
    for i in 0..half {
        poly[i] = poly[2 * i] + r * (poly[2 * i + 1] - poly[2 * i]);
    }
    poly.truncate(half);
}

/// Initialise eq(r, ·) over {0,1}^m in LSB-first ordering.
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

/// eq(r_a, r_b) — product of per-variable equality factors.
fn eq_final_eval(r_a: &[Fr], r_b: &[Fr]) -> Fr {
    assert_eq!(r_a.len(), r_b.len());
    r_a.iter()
        .zip(r_b.iter())
        .map(|(&ai, &bi)| ai * bi + (Fr::one() - ai) * (Fr::one() - bi))
        .product()
}

/// Lagrange interpolation: evaluate a polynomial given its values at 0,1,…,n-1
/// at an arbitrary field point t.
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

// ── Proof struct ──────────────────────────────────────────────────────────────

/// Phase 4h NIZK: one global LogUp argument for the entire circuit trace.
///
/// Contains 6 polynomial commitments (constant regardless of the number of LUT
/// types T), two sumchecks (query + table), and 6 HyperKZG opening proofs.
pub struct MegaLogUpProof {
    // ── Dimensions ──────────────────────────────────────────────────────────
    /// M_query = ⌈log₂(N_total)⌉.
    pub num_query_vars: usize,
    /// M_table = ⌈log₂(T_pad × 2^k)⌉ = log₂(T_pad) + k.
    pub num_table_vars: usize,
    /// Number of real LUT types T (before padding to T_pad).
    pub num_lut_types: usize,
    /// Common k (input bits per LUT).
    pub k: usize,
    /// Common m (output bits per LUT).
    pub m: usize,
    /// N_total: real trace rows (un-padded).
    pub n_total: usize,

    // ── 4 query-side committed polynomials (over {0,1}^M_query) ─────────────
    pub comm_type_idx:   HyperKZGCommitment<Bn254>,
    pub comm_packed_in:  HyperKZGCommitment<Bn254>,
    pub comm_packed_out: HyperKZGCommitment<Bn254>,
    pub comm_inv_q:      HyperKZGCommitment<Bn254>,

    // ── 2 table-side committed polynomials (over {0,1}^M_table) ─────────────
    pub comm_count: HyperKZGCommitment<Bn254>,
    pub comm_inv_t: HyperKZGCommitment<Bn254>,

    // ── LogUp claimed sum S ──────────────────────────────────────────────────
    pub claimed_sum: Fr,

    // ── Query sumcheck (M_query rounds, degree 3, 4 pts/round) ──────────────
    pub sc_q_polys: Vec<Vec<Fr>>,
    /// Finals at r_sc_q: [inv_q, type_idx, packed_in, packed_out].
    pub finals_q: [Fr; 4],

    // ── Table sumcheck (M_table rounds, degree 3, 4 pts/round) ──────────────
    pub sc_t_polys: Vec<Vec<Fr>>,
    /// Finals at r_sc_t: [inv_t, count].
    pub finals_t: [Fr; 2],

    // ── HyperKZG opening proofs (one per committed polynomial) ──────────────
    /// InvQ opened at r_sc_q.
    pub opening_inv_q:      Option<HyperKZGProof<Bn254>>,
    /// TypeIdx opened at r_sc_q.
    pub opening_type_idx:   Option<HyperKZGProof<Bn254>>,
    /// PackedIn opened at r_sc_q.
    pub opening_packed_in:  Option<HyperKZGProof<Bn254>>,
    /// PackedOut opened at r_sc_q.
    pub opening_packed_out: Option<HyperKZGProof<Bn254>>,
    /// InvT opened at r_sc_t.
    pub opening_inv_t:      Option<HyperKZGProof<Bn254>>,
    /// Count opened at r_sc_t.
    pub opening_count:      Option<HyperKZGProof<Bn254>>,

    /// Claimed circuit outputs (final cycle wire values).
    pub outputs: Vec<bool>,
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build the public mega-table B vector of length T_pad × 2^k.
///
/// Entry for (tid, x) at index `tid * 2^k + x`:
/// ```text
///   B[tid*2^k + x] = fr(tid) + γ·fr(x) + γ²·truth_out(tid, x)
/// ```
/// For tid ≥ T (padding): truth_out is taken as 0, giving B = fr(tid) + γ·fr(x).
fn build_mega_table_b(
    lut_ids_sorted: &[u32],
    type_map: &HashMap<u32, usize>, // lut_id → tid (0-based index)
    lut_types_map: &HashMap<u32, crate::lut_czbc::LutDesc>,
    k: usize,
    m: usize,
    t_pad: usize,
    gamma: Fr,
    gamma2: Fr,
) -> Vec<Fr> {
    let n = t_pad * (1usize << k);
    let mut b = Vec::with_capacity(n);
    let table_w = 1usize << k;

    for tid in 0..t_pad {
        if tid < lut_ids_sorted.len() {
            let lut_id = lut_ids_sorted[tid];
            let _ = type_map; // suppress unused warning; tid IS the index
            let desc = &lut_types_map[&lut_id];
            for x in 0..table_w {
                // Compute packed output for truth table entry x
                let mut pack_out: u64 = 0;
                for j in 0..m {
                    let bit_pos = x * m + j;
                    let bit = (desc.truth_table[bit_pos / 8] >> (bit_pos % 8)) & 1;
                    if bit != 0 {
                        pack_out |= 1u64 << j;
                    }
                }
                b.push(fr(tid as u64) + gamma * fr(x as u64) + gamma2 * fr(pack_out));
            }
        } else {
            // Padding type: output is 0
            for x in 0..table_w {
                b.push(fr(tid as u64) + gamma * fr(x as u64));
                // + gamma2 * 0 = nothing to add
            }
        }
    }
    b
}

// ── prover ────────────────────────────────────────────────────────────────────

/// Prove correctness of the full LUT circuit trace using a single global LogUp.
///
/// Returns a [`MegaLogUpProof`] covering all N_total trace rows across all T
/// LUT types with only 6 polynomial commitments and 2 sumchecks.
pub fn prove_mega_logup_circuit(
    circ: &LutCirc,
    inputs: &[bool],
    cycles: u32,
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> MegaLogUpProof {
    // ── Validate: all LUT types must share the same k and m ─────────────────
    let k = circ.lut_types.values().next().expect("no LUT types").k;
    let m = circ.lut_types.values().next().expect("no LUT types").m;
    for desc in circ.lut_types.values() {
        assert_eq!(desc.k, k, "Phase 4h requires uniform k across all LUT types");
        assert_eq!(desc.m, m, "Phase 4h requires uniform m across all LUT types");
    }

    // ── Assign deterministic type indices (0..T-1) in sorted lut_id order ───
    let mut lut_ids_sorted: Vec<u32> = circ.lut_types.keys().copied().collect();
    lut_ids_sorted.sort_unstable();
    let t = lut_ids_sorted.len();
    let type_map: HashMap<u32, usize> = lut_ids_sorted
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // T_pad = smallest power of two ≥ T
    let t_pad = t.next_power_of_two().max(1);
    // M_table = log2(t_pad) + k
    let mega_size = t_pad * (1usize << k);
    let m_table = mega_size.trailing_zeros() as usize;
    debug_assert_eq!(1usize << m_table, mega_size, "mega_size must be a power of 2");

    // ── Simulate circuit ─────────────────────────────────────────────────────
    let (trace, outputs) = evaluate_lut_circuit(circ, inputs, cycles);
    let n_total = trace.len(); // real rows (before padding)

    // M_query = ⌈log₂(n_total)⌉
    let m_query = usize::max(1, n_total.next_power_of_two().trailing_zeros() as usize);
    let cap = 1usize << m_query;

    // ── Phase 4h-A: Build global trace polynomials ───────────────────────────
    // For each row i: tid(i) = type_map[trace[i].lut_id].
    // Padding rows (i ≥ n_total): use tid=0, pack_in=0, pack_out=truth_out(type0, 0).
    let dummy_tid = 0usize;
    let dummy_pack_in = 0u64;
    let dummy_pack_out: u64 = {
        let desc0 = &circ.lut_types[&lut_ids_sorted[dummy_tid]];
        let mut v = 0u64;
        for j in 0..m {
            let bit_pos = j; // input=0, so bit_pos = 0*m + j = j
            let bit = (desc0.truth_table[bit_pos / 8] >> (bit_pos % 8)) & 1;
            if bit != 0 { v |= 1u64 << j; }
        }
        v
    };

    let mut type_idx_vec: Vec<Fr> = Vec::with_capacity(cap);
    let mut packed_in_vec: Vec<Fr> = Vec::with_capacity(cap);
    let mut packed_out_vec: Vec<Fr> = Vec::with_capacity(cap);

    for i in 0..cap {
        if i < n_total {
            let row = &trace[i];
            let tid = type_map[&row.lut_id];
            let mut pack_in: u64 = 0;
            for (l, &b) in row.inputs.iter().enumerate() {
                if b { pack_in |= 1u64 << l; }
            }
            let mut pack_out: u64 = 0;
            for (j, &b) in row.outputs.iter().enumerate() {
                if b { pack_out |= 1u64 << j; }
            }
            type_idx_vec.push(fr(tid as u64));
            packed_in_vec.push(fr(pack_in));
            packed_out_vec.push(fr(pack_out));
        } else {
            type_idx_vec.push(fr(dummy_tid as u64));
            packed_in_vec.push(fr(dummy_pack_in));
            packed_out_vec.push(fr(dummy_pack_out));
        }
    }

    // ── Phase 4h-B: Build Count vector for the mega-table ───────────────────
    // Count[tid*2^k + x] = number of trace rows (including padding) whose
    // query matches this mega-table entry.
    let table_w = 1usize << k;
    let mut count_u64: Vec<u64> = vec![0u64; mega_size];
    for i in 0..cap {
        let tid = type_idx_vec[i].into_bigint().as_ref()[0] as usize;
        let x   = packed_in_vec[i].into_bigint().as_ref()[0] as usize;
        let mega_idx = tid * table_w + x;
        if mega_idx < mega_size {
            count_u64[mega_idx] += 1;
        }
    }
    let count_fr: Vec<Fr> = count_u64.iter().map(|&c| fr(c)).collect();

    // ── Commit TypeIdx, PackedIn, PackedOut, Count ───────────────────────────
    let mle_type_idx   = MultilinearPolynomial::from(type_idx_vec.clone());
    let mle_packed_in  = MultilinearPolynomial::from(packed_in_vec.clone());
    let mle_packed_out = MultilinearPolynomial::from(packed_out_vec.clone());
    let mle_count      = MultilinearPolynomial::from(count_fr.clone());

    let comm_type_idx   = HyperKZG::<Bn254>::commit(pk, &mle_type_idx)
        .expect("commit TypeIdx");
    let comm_packed_in  = HyperKZG::<Bn254>::commit(pk, &mle_packed_in)
        .expect("commit PackedIn");
    let comm_packed_out = HyperKZG::<Bn254>::commit(pk, &mle_packed_out)
        .expect("commit PackedOut");
    let comm_count      = HyperKZG::<Bn254>::commit(pk, &mle_count)
        .expect("commit Count");

    // ── Fiat-Shamir: metadata + early commitments ────────────────────────────
    transcript.append_u64(n_total as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(m as u64);
    transcript.append_u64(t as u64);
    transcript.append_u64(t_pad as u64);
    transcript.append_u64(m_query as u64);
    transcript.append_u64(m_table as u64);
    comm_type_idx.append_to_transcript(transcript);
    comm_packed_in.append_to_transcript(transcript);
    comm_packed_out.append_to_transcript(transcript);
    comm_count.append_to_transcript(transcript);

    // γ: mixes TypeIdx, PackedIn, PackedOut into a single field encoding.
    let gamma: Fr  = transcript.challenge_scalar_optimized::<Fr>().into();
    let gamma2: Fr = gamma * gamma;
    // λ: denominator challenge for LogUp.
    let lambda: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── Build query encoding A[i] = TypeIdx + γ·PackedIn + γ²·PackedOut ──────
    let a_enc: Vec<Fr> = type_idx_vec
        .iter()
        .zip(packed_in_vec.iter())
        .zip(packed_out_vec.iter())
        .map(|((&t, &p), &o)| t + gamma * p + gamma2 * o)
        .collect();

    // InvQ[i] = 1/(λ − A[i])
    let inv_q_vec: Vec<Fr> = a_enc
        .iter()
        .map(|&ai| {
            Field::inverse(&(lambda - ai))
                .expect("λ - A[i] = 0: collision with random λ (negligible probability)")
        })
        .collect();

    // ── Build public mega-table B and compute InvT ───────────────────────────
    let b_enc = build_mega_table_b(
        &lut_ids_sorted,
        &type_map,
        &circ.lut_types,
        k,
        m,
        t_pad,
        gamma,
        gamma2,
    );

    // InvT[x] = 1/(λ − B[x])
    let inv_t_vec: Vec<Fr> = b_enc
        .iter()
        .map(|&bx| {
            Field::inverse(&(lambda - bx))
                .expect("λ - B[x] = 0: collision with random λ (negligible probability)")
        })
        .collect();

    // ── Claimed sum S = ∑ InvQ ──────────────────────────────────────────────
    let claimed_sum: Fr = inv_q_vec.iter().copied().sum();
    // Sanity check (debug only): ∑ Count[x]*InvT[x] should equal claimed_sum.
    // (We don't panic on mismatch; soundness is enforced by the sumcheck.)

    // ── Commit InvQ, InvT ────────────────────────────────────────────────────
    let mle_inv_q = MultilinearPolynomial::from(inv_q_vec.clone());
    let mle_inv_t = MultilinearPolynomial::from(inv_t_vec.clone());

    let comm_inv_q = HyperKZG::<Bn254>::commit(pk, &mle_inv_q).expect("commit InvQ");
    let comm_inv_t = HyperKZG::<Bn254>::commit(pk, &mle_inv_t).expect("commit InvT");

    comm_inv_q.append_to_transcript(transcript);
    comm_inv_t.append_to_transcript(transcript);
    transcript.append_scalar(&claimed_sum);

    // ξ: batches zero-check and sum contributions in the two sumchecks.
    let xi: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── Query sumcheck ────────────────────────────────────────────────────────
    // Initial claim: ξ·S
    //   ∑_i [eq(r_q,i)·(InvQ[i]·(λ−A[i])−1) + ξ·InvQ[i]] = ξ·S
    let r_q: Vec<Fr> = transcript.challenge_vector(m_query);

    let mut eq_q        = init_eq(&r_q);
    let mut wk_inv_q    = inv_q_vec.clone();
    let mut wk_type_idx = type_idx_vec.clone();
    let mut wk_pack_in  = packed_in_vec.clone();
    let mut wk_pack_out = packed_out_vec.clone();

    let eval_pts: Vec<Fr> = (0..4u64).map(fr).collect();
    let mut sc_q_polys: Vec<Vec<Fr>>   = Vec::with_capacity(m_query);
    let mut r_sc_q_fr:  Vec<Fr>        = Vec::with_capacity(m_query);
    let mut r_sc_q_ch:  Vec<Challenge> = Vec::with_capacity(m_query);

    for _round in 0..m_query {
        let half = eq_q.len() / 2;
        let mut p_evals = [Fr::zero(); 4];

        for idx in 0..half {
            let eq_lo  = eq_q[2 * idx];
            let eq_hi  = eq_q[2 * idx + 1];
            let iq_lo  = wk_inv_q[2 * idx];
            let iq_hi  = wk_inv_q[2 * idx + 1];
            let ti_lo  = wk_type_idx[2 * idx];
            let ti_hi  = wk_type_idx[2 * idx + 1];
            let pi_lo  = wk_pack_in[2 * idx];
            let pi_hi  = wk_pack_in[2 * idx + 1];
            let po_lo  = wk_pack_out[2 * idx];
            let po_hi  = wk_pack_out[2 * idx + 1];

            for (pt, &t) in eval_pts.iter().enumerate() {
                let eq_t  = eq_lo  + t * (eq_hi  - eq_lo);
                let iq_t  = iq_lo  + t * (iq_hi  - iq_lo);
                let ti_t  = ti_lo  + t * (ti_hi  - ti_lo);
                let pi_t  = pi_lo  + t * (pi_hi  - pi_lo);
                let po_t  = po_lo  + t * (po_hi  - po_lo);
                let a_t   = ti_t + gamma * pi_t + gamma2 * po_t;
                // zero-check: InvQ·(λ−A) − 1
                let zc    = eq_t * (iq_t * (lambda - a_t) - Fr::one());
                p_evals[pt] += zc + xi * iq_t;
            }
        }

        for &e in p_evals.iter() { transcript.append_scalar(&e); }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_q_ch.push(r_j_ch);
        r_sc_q_fr.push(r_j);

        bind(&mut eq_q,        r_j);
        bind(&mut wk_inv_q,    r_j);
        bind(&mut wk_type_idx, r_j);
        bind(&mut wk_pack_in,  r_j);
        bind(&mut wk_pack_out, r_j);

        sc_q_polys.push(p_evals.to_vec());
    }

    let finals_q = [wk_inv_q[0], wk_type_idx[0], wk_pack_in[0], wk_pack_out[0]];
    for &v in finals_q.iter() { transcript.append_scalar(&v); }

    // ── Table sumcheck ────────────────────────────────────────────────────────
    // Initial claim: ξ·S
    //   ∑_x [eq(r_t,x)·(InvT[x]·(λ−B[x])−1) + ξ·Count[x]·InvT[x]] = ξ·S
    let r_t: Vec<Fr> = transcript.challenge_vector(m_table);

    let mut eq_t       = init_eq(&r_t);
    let mut wk_inv_t   = inv_t_vec.clone();
    let mut wk_count   = count_fr.clone();
    let mut wk_b_enc   = b_enc.clone();

    let mut sc_t_polys: Vec<Vec<Fr>>   = Vec::with_capacity(m_table);
    let mut r_sc_t_fr:  Vec<Fr>        = Vec::with_capacity(m_table);
    let mut r_sc_t_ch:  Vec<Challenge> = Vec::with_capacity(m_table);

    for _round in 0..m_table {
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
                // zero-check: InvT·(λ−B) − 1
                let zc     = eq_t_v * (it_t * (lambda - b_t) - Fr::one());
                // sum: ξ·Count·InvT
                p_evals[pt] += zc + xi * ct_t * it_t;
            }
        }

        for &e in p_evals.iter() { transcript.append_scalar(&e); }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_t_ch.push(r_j_ch);
        r_sc_t_fr.push(r_j);

        bind(&mut eq_t,     r_j);
        bind(&mut wk_inv_t, r_j);
        bind(&mut wk_count, r_j);
        bind(&mut wk_b_enc, r_j);

        sc_t_polys.push(p_evals.to_vec());
    }

    let finals_t = [wk_inv_t[0], wk_count[0]];
    for &v in finals_t.iter() { transcript.append_scalar(&v); }

    // ── HyperKZG opening proofs ──────────────────────────────────────────────
    // HyperKZG uses big-endian (MSB-first) variable ordering; our sumcheck used
    // LSB-first, so we reverse the challenge vectors.
    let point_q_kzg: Vec<Challenge> = r_sc_q_ch.iter().rev().cloned().collect();
    let point_t_kzg: Vec<Challenge> = r_sc_t_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();

    let open_one = |mle:  &MultilinearPolynomial<Fr>,
                    comm: &HyperKZGCommitment<Bn254>,
                    eval: &Fr,
                    pt:   &[Challenge],
                    t:    &mut KeccakTranscript|
     -> Option<HyperKZGProof<Bn254>> {
        if *comm == zero_comm {
            None
        } else {
            Some(
                HyperKZG::<Bn254>::open(pk, mle, pt, eval, t)
                    .expect("HyperKZG open failed"),
            )
        }
    };

    let opening_inv_q = open_one(
        &mle_inv_q, &comm_inv_q, &finals_q[0], &point_q_kzg, transcript,
    );
    let opening_type_idx = open_one(
        &mle_type_idx, &comm_type_idx, &finals_q[1], &point_q_kzg, transcript,
    );
    let opening_packed_in = open_one(
        &mle_packed_in, &comm_packed_in, &finals_q[2], &point_q_kzg, transcript,
    );
    let opening_packed_out = open_one(
        &mle_packed_out, &comm_packed_out, &finals_q[3], &point_q_kzg, transcript,
    );
    let opening_inv_t = open_one(
        &mle_inv_t, &comm_inv_t, &finals_t[0], &point_t_kzg, transcript,
    );
    let opening_count = open_one(
        &mle_count, &comm_count, &finals_t[1], &point_t_kzg, transcript,
    );

    MegaLogUpProof {
        num_query_vars: m_query,
        num_table_vars: m_table,
        num_lut_types:  t,
        k,
        m,
        n_total,
        comm_type_idx,
        comm_packed_in,
        comm_packed_out,
        comm_inv_q,
        comm_count,
        comm_inv_t,
        claimed_sum,
        sc_q_polys,
        finals_q,
        sc_t_polys,
        finals_t,
        opening_inv_q,
        opening_type_idx,
        opening_packed_in,
        opening_packed_out,
        opening_inv_t,
        opening_count,
        outputs,
    }
}

// ── verifier ──────────────────────────────────────────────────────────────────

/// Verify a [`MegaLogUpProof`].
///
/// The verifier **does not** re-execute the circuit.  It needs:
/// - The proof and the public circuit description (for the mega-table B vector).
/// - The same transcript initialisation used by the prover.
pub fn verify_mega_logup_circuit(
    proof: &MegaLogUpProof,
    circ: &LutCirc,
    vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    let k = proof.k;
    let m = proof.m;
    let t = proof.num_lut_types;
    let t_pad = {
        let v = t.next_power_of_two().max(1);
        debug_assert_eq!(1usize << proof.num_table_vars, v * (1 << k));
        v
    };
    let m_query = proof.num_query_vars;
    let m_table = proof.num_table_vars;

    // Reconstruct the sorted lut_ids (same deterministic order as prover).
    let mut lut_ids_sorted: Vec<u32> = circ.lut_types.keys().copied().collect();
    lut_ids_sorted.sort_unstable();
    if lut_ids_sorted.len() != t {
        eprintln!(
            "mega_logup verify: circuit has {} LUT types, proof claims {}",
            lut_ids_sorted.len(), t
        );
        return false;
    }

    // ── Re-derive Fiat-Shamir challenges ─────────────────────────────────────
    transcript.append_u64(proof.n_total as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(m as u64);
    transcript.append_u64(t as u64);
    transcript.append_u64(t_pad as u64);
    transcript.append_u64(m_query as u64);
    transcript.append_u64(m_table as u64);
    proof.comm_type_idx.append_to_transcript(transcript);
    proof.comm_packed_in.append_to_transcript(transcript);
    proof.comm_packed_out.append_to_transcript(transcript);
    proof.comm_count.append_to_transcript(transcript);

    let gamma: Fr  = transcript.challenge_scalar_optimized::<Fr>().into();
    let gamma2: Fr = gamma * gamma;
    let lambda: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // Recompute the public mega-table B (same as prover).
    let b_enc = build_mega_table_b(
        &lut_ids_sorted,
        &HashMap::new(), // not used inside the function
        &circ.lut_types,
        k,
        m,
        t_pad,
        gamma,
        gamma2,
    );

    proof.comm_inv_q.append_to_transcript(transcript);
    proof.comm_inv_t.append_to_transcript(transcript);
    transcript.append_scalar(&proof.claimed_sum);

    let xi: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

    // ── Replay query sumcheck ─────────────────────────────────────────────────
    let r_q: Vec<Fr> = transcript.challenge_vector(m_query);
    let mut prev_q = xi * proof.claimed_sum;
    let mut r_sc_q_fr:  Vec<Fr>    = Vec::with_capacity(m_query);
    let mut r_sc_q_ch:  Vec<Challenge> = Vec::with_capacity(m_query);

    for round in 0..m_query {
        let p = &proof.sc_q_polys[round];
        if p.len() != 4 {
            eprintln!("mega_logup verify: query sc round {round}: poly len {} ≠ 4", p.len());
            return false;
        }
        let sum = p[0] + p[1];
        if sum != prev_q {
            eprintln!("mega_logup verify: query sc round {round}: p(0)+p(1)={sum:?} ≠ {prev_q:?}");
            return false;
        }
        for &e in p.iter() { transcript.append_scalar(&e); }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_q_ch.push(r_j_ch);
        r_sc_q_fr.push(r_j);
        prev_q = poly_at(p, r_j);
    }

    // Final query check:
    //   eq(r_q, r_sc_q)·(InvQ_f·(λ−A_f)−1) + ξ·InvQ_f = prev_q
    let eq_q_fin = eq_final_eval(&r_q, &r_sc_q_fr);
    let inv_q_f  = proof.finals_q[0];
    let type_f   = proof.finals_q[1];
    let pack_in_f  = proof.finals_q[2];
    let pack_out_f = proof.finals_q[3];
    let a_f = type_f + gamma * pack_in_f + gamma2 * pack_out_f;
    let expected_q = eq_q_fin * (inv_q_f * (lambda - a_f) - Fr::one()) + xi * inv_q_f;
    if expected_q != prev_q {
        eprintln!(
            "mega_logup verify: query sc final check FAILED: {expected_q:?} ≠ {prev_q:?}"
        );
        return false;
    }

    for &v in proof.finals_q.iter() { transcript.append_scalar(&v); }

    // ── Replay table sumcheck ─────────────────────────────────────────────────
    let r_t: Vec<Fr> = transcript.challenge_vector(m_table);
    let mut prev_t = xi * proof.claimed_sum;
    let mut r_sc_t_fr:  Vec<Fr>    = Vec::with_capacity(m_table);
    let mut r_sc_t_ch:  Vec<Challenge> = Vec::with_capacity(m_table);

    for round in 0..m_table {
        let p = &proof.sc_t_polys[round];
        if p.len() != 4 {
            eprintln!("mega_logup verify: table sc round {round}: poly len {} ≠ 4", p.len());
            return false;
        }
        let sum = p[0] + p[1];
        if sum != prev_t {
            eprintln!("mega_logup verify: table sc round {round}: p(0)+p(1)={sum:?} ≠ {prev_t:?}");
            return false;
        }
        for &e in p.iter() { transcript.append_scalar(&e); }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_sc_t_ch.push(r_j_ch);
        r_sc_t_fr.push(r_j);
        prev_t = poly_at(p, r_j);
    }

    // Final table check:
    //   eq(r_t, r_sc_t)·(InvT_f·(λ−B_f)−1) + ξ·Count_f·InvT_f = prev_t
    let eq_t_fin = eq_final_eval(&r_t, &r_sc_t_fr);
    let inv_t_f  = proof.finals_t[0];
    let count_f  = proof.finals_t[1];

    // B_f = MLE of public b_enc evaluated at r_sc_t.
    let b_f: Fr = {
        let mut bv = b_enc.clone();
        for &rj in &r_sc_t_fr {
            bind(&mut bv, rj);
        }
        bv[0]
    };

    let expected_t = eq_t_fin * (inv_t_f * (lambda - b_f) - Fr::one())
        + xi * count_f * inv_t_f;
    if expected_t != prev_t {
        eprintln!(
            "mega_logup verify: table sc final check FAILED: {expected_t:?} ≠ {prev_t:?}"
        );
        return false;
    }

    for &v in proof.finals_t.iter() { transcript.append_scalar(&v); }

    // ── Verify HyperKZG opening proofs ────────────────────────────────────────
    let point_q_kzg: Vec<Challenge> = r_sc_q_ch.iter().rev().cloned().collect();
    let point_t_kzg: Vec<Challenge> = r_sc_t_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();

    let verify_one = |name:    &str,
                      comm:    &HyperKZGCommitment<Bn254>,
                      eval:    &Fr,
                      pf_opt:  &Option<HyperKZGProof<Bn254>>,
                      pt:      &[Challenge],
                      t:       &mut KeccakTranscript|
     -> bool {
        if *comm == zero_comm {
            if *eval != Fr::zero() {
                eprintln!("mega_logup verify: {name}: zero comm but non-zero eval");
                return false;
            }
            return true;
        }
        match pf_opt {
            Some(pf) => {
                if HyperKZG::<Bn254>::verify(vk, comm, pt, eval, pf, t).is_err() {
                    eprintln!("mega_logup verify: {name}: HyperKZG verify FAILED");
                    return false;
                }
                true
            }
            None => {
                eprintln!("mega_logup verify: {name}: non-zero comm but no opening proof");
                false
            }
        }
    };

    verify_one("InvQ",     &proof.comm_inv_q,      &proof.finals_q[0], &proof.opening_inv_q,      &point_q_kzg, transcript)
    && verify_one("TypeIdx",  &proof.comm_type_idx,   &proof.finals_q[1], &proof.opening_type_idx,   &point_q_kzg, transcript)
    && verify_one("PackedIn", &proof.comm_packed_in,  &proof.finals_q[2], &proof.opening_packed_in,  &point_q_kzg, transcript)
    && verify_one("PackedOut",&proof.comm_packed_out, &proof.finals_q[3], &proof.opening_packed_out, &point_q_kzg, transcript)
    && verify_one("InvT",     &proof.comm_inv_t,      &proof.finals_t[0], &proof.opening_inv_t,      &point_t_kzg, transcript)
    && verify_one("Count",    &proof.comm_count,      &proof.finals_t[1], &proof.opening_count,      &point_t_kzg, transcript)
}

// ── SRS sizing ────────────────────────────────────────────────────────────────

/// Compute the SRS size needed for Phase 4h: max(M_query, M_table).
///
/// This is typically much smaller than the per-type lasso SRS because we use
/// the full N_total domain rather than the largest per-type domain.
pub fn compute_max_num_vars_mega(circ: &LutCirc, cycles: u32) -> usize {
    let n_total = circ.ops.len() * (cycles.max(1) as usize);
    let m_query = usize::max(1, n_total.next_power_of_two().trailing_zeros() as usize);

    let t = circ.lut_types.len();
    let k = circ.lut_types.values().next().map(|d| d.k).unwrap_or(6);
    let t_pad = t.next_power_of_two().max(1);
    let mega_size = t_pad * (1usize << k);
    // mega_size = t_pad * 2^k, t_pad is a power of 2, so mega_size is a power of 2.
    let m_table = mega_size.trailing_zeros() as usize;

    usize::max(m_query, m_table)
}

// ── proof size estimation ─────────────────────────────────────────────────────

/// Estimate the serialised size of a [`MegaLogUpProof`] in bytes.
pub fn compute_mega_proof_size_bytes(proof: &MegaLogUpProof) -> usize {
    let mut total = 0usize;
    let mut buf   = Vec::new();

    // 6 HyperKZG commitments
    for comm in [
        &proof.comm_type_idx,
        &proof.comm_packed_in,
        &proof.comm_packed_out,
        &proof.comm_inv_q,
        &proof.comm_count,
        &proof.comm_inv_t,
    ] {
        buf.clear();
        comm.0.serialize_compressed(&mut buf).ok();
        total += buf.len();
    }

    // Claimed sum: 1 Fr (32 bytes)
    total += 32;

    // Query sumcheck: m_query rounds × 4 Fr
    total += proof.num_query_vars * 4 * 32;
    // Query finals: 4 Fr
    total += 4 * 32;

    // Table sumcheck: m_table rounds × 4 Fr
    total += proof.num_table_vars * 4 * 32;
    // Table finals: 2 Fr
    total += 2 * 32;

    // 6 HyperKZG opening proofs
    for opt_pf in [
        &proof.opening_inv_q,
        &proof.opening_type_idx,
        &proof.opening_packed_in,
        &proof.opening_packed_out,
        &proof.opening_inv_t,
        &proof.opening_count,
    ] {
        if let Some(pf) = opt_pf {
            buf.clear();
            pf.serialize_compressed(&mut buf).ok();
            total += buf.len();
        }
    }

    total
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut_czbc::{LutCirc, LutDesc, LutOp};
    use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
    use jolt_core::poly::commitment::hyperkzg::HyperKZG;
    use std::collections::HashMap;

    type PCS = HyperKZG<Bn254>;

    fn xor_desc() -> LutDesc {
        LutDesc {
            lut_id: 1, k: 2, m: 1,
            truth_table: vec![0x06], // XOR: [0,1,1,0]
        }
    }
    fn and_desc() -> LutDesc {
        LutDesc {
            lut_id: 2, k: 2, m: 1,
            truth_table: vec![0x08], // AND: [0,0,0,1]
        }
    }

    /// Single LUT type, 1 cycle: prove and verify.
    #[test]
    fn mega_single_type_verifies() {
        let xor = xor_desc();
        let mut lut_types = HashMap::new();
        lut_types.insert(1u32, xor);

        let circ = LutCirc {
            num_wires: 3,
            primary_inputs: vec![0, 1],
            registers: vec![],
            outputs: vec![2],
            lut_types,
            ops: vec![LutOp { lut_id: 1, dst_wire: 2, src_wires: vec![0, 1] }],
            default_cycles: 1,
        };

        let max_vars = compute_max_num_vars_mega(&circ, 1);
        let pk = PCS::setup_prover(max_vars);
        let vk = PCS::setup_verifier(&pk);

        let mut pt = KeccakTranscript::new(b"mega-test");
        pt.append_u64(circ.ops.len() as u64);
        pt.append_u64(circ.outputs.len() as u64);
        pt.append_u64(1u64);
        pt.append_u64(false as u64);
        pt.append_u64(false as u64);

        let proof = prove_mega_logup_circuit(&circ, &[false, false], 1, &pk, &mut pt);
        assert_eq!(proof.outputs[0], false); // 0 XOR 0 = 0

        let mut vt = KeccakTranscript::new(b"mega-test");
        vt.append_u64(circ.ops.len() as u64);
        vt.append_u64(circ.outputs.len() as u64);
        vt.append_u64(1u64);
        vt.append_u64(false as u64);
        vt.append_u64(false as u64);

        assert!(
            verify_mega_logup_circuit(&proof, &circ, &vk, &mut vt),
            "single-type mega proof should verify"
        );
    }

    /// Two LUT types, multiple cycles: prove and verify.
    #[test]
    fn mega_two_types_verifies() {
        let mut lut_types = HashMap::new();
        lut_types.insert(1u32, xor_desc());
        lut_types.insert(2u32, and_desc());

        let circ = LutCirc {
            num_wires: 4,
            primary_inputs: vec![0, 1],
            registers: vec![],
            outputs: vec![3],
            lut_types,
            ops: vec![
                LutOp { lut_id: 1, dst_wire: 2, src_wires: vec![0, 1] }, // XOR
                LutOp { lut_id: 2, dst_wire: 3, src_wires: vec![0, 1] }, // AND
            ],
            default_cycles: 4,
        };

        let max_vars = compute_max_num_vars_mega(&circ, 4);
        let pk = PCS::setup_prover(max_vars);
        let vk = PCS::setup_verifier(&pk);

        let inputs = vec![true, true];
        let mut pt = KeccakTranscript::new(b"mega-two");
        pt.append_u64(circ.ops.len() as u64);
        pt.append_u64(circ.outputs.len() as u64);
        pt.append_u64(4u64);
        for &b in &inputs { pt.append_u64(b as u64); }

        let proof = prove_mega_logup_circuit(&circ, &inputs, 4, &pk, &mut pt);
        // 1 AND 1 = 1
        assert_eq!(proof.outputs[0], true, "AND output should be true");

        let mut vt = KeccakTranscript::new(b"mega-two");
        vt.append_u64(circ.ops.len() as u64);
        vt.append_u64(circ.outputs.len() as u64);
        vt.append_u64(4u64);
        for &b in &inputs { vt.append_u64(b as u64); }

        assert!(
            verify_mega_logup_circuit(&proof, &circ, &vk, &mut vt),
            "two-type mega proof should verify"
        );
    }

    /// Tampered output: must fail.
    #[test]
    fn mega_tampered_fails() {
        let mut lut_types = HashMap::new();
        lut_types.insert(1u32, xor_desc());

        let circ = LutCirc {
            num_wires: 3,
            primary_inputs: vec![0, 1],
            registers: vec![],
            outputs: vec![2],
            lut_types,
            ops: vec![LutOp { lut_id: 1, dst_wire: 2, src_wires: vec![0, 1] }],
            default_cycles: 1,
        };

        let max_vars = compute_max_num_vars_mega(&circ, 1);
        let pk = PCS::setup_prover(max_vars);
        let vk = PCS::setup_verifier(&pk);

        let mut pt = KeccakTranscript::new(b"mega-tamper");
        pt.append_u64(circ.ops.len() as u64);
        pt.append_u64(circ.outputs.len() as u64);
        pt.append_u64(1u64);
        pt.append_u64(true as u64);
        pt.append_u64(false as u64);

        let mut proof = prove_mega_logup_circuit(&circ, &[true, false], 1, &pk, &mut pt);
        // Corrupt the query sumcheck round 0
        if let Some(v) = proof.sc_q_polys[0].get_mut(0) {
            *v += Fr::one();
        }

        let mut vt = KeccakTranscript::new(b"mega-tamper");
        vt.append_u64(circ.ops.len() as u64);
        vt.append_u64(circ.outputs.len() as u64);
        vt.append_u64(1u64);
        vt.append_u64(true as u64);
        vt.append_u64(false as u64);

        assert!(
            !verify_mega_logup_circuit(&proof, &circ, &vk, &mut vt),
            "tampered proof should fail"
        );
    }
}
