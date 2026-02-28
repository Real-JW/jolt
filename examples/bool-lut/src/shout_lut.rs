//! **Phase S1 — LUT Table → JoltLookupTable**
//!
//! This module wraps each [`LutDesc`] (a k-input, m-output Boolean truth table
//! loaded from a `.lczbc` file) as a Jolt-compatible lookup table.
//!
//! # Overview
//!
//! | Item | Description |
//! |------|-------------|
//! | [`LutShoutTable`] | Wraps one `LutDesc` + one output-bit selection; implements [`JoltLookupTable`] |
//! | [`lut_desc_to_sub_circuit`] | Thin converter: `LutDesc` → [`SubCircuitLut`] |
//! | [`alpha_batch_table_entries`] | Computes `Σ_j α^j T_j[x]` for multi-output alpha-batching |
//! | [`ShoutAddress`] helpers | Encode a `LutEval` trace row into a Shout lookup address |
//!
//! # MLE convention
//!
//! All MLE evaluations use **LSB-first** variable ordering, matching
//! [`SubCircuitLut::evaluate_mle_at`] and `lut_czbc.rs`.  The truth-table
//! index is:
//! ```text
//! idx = inputs[0] | (inputs[1] << 1) | … | (inputs[k-1] << (k-1))
//! ```
//!
//! # Multi-output batching
//!
//! Each LUT has m output bits.  `JoltLookupTable::evaluate_mle` returns a
//! single scalar.  To handle m > 1 outputs in a single Shout instance, the
//! prover uses a random linear combination:
//! ```text
//! Val(x) = Σ_{j=0}^{m-1} α^j · T_j[x]
//! ```
//! Use [`alpha_batch_table_entries`] to materialise the combined table, then
//! feed it into sumcheck directly (see Phase S3).

use std::collections::HashMap;

use ark_bn254::{Bn254, Fr};
use ark_ff::{One, Zero};
use serde::{Deserialize, Serialize};

use jolt_core::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use jolt_core::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::transcripts::{AppendToTranscript, KeccakTranscript, Transcript};
use jolt_core::zkvm::lookup_table::{JoltLookupTable, SubCircuitLut};

use crate::lut_czbc::{LutCirc, LutDesc, LutEval};

// ─────────────────────────────────────────────────────────────────────────────
// LutShoutTable
// ─────────────────────────────────────────────────────────────────────────────

/// A single-output view of a `LutDesc` truth table, implementing
/// [`JoltLookupTable`] for use in the Shout lookup argument.
///
/// For a LUT with m output bits, create m instances with `out_bit` ∈ 0..m.
/// Use [`alpha_batch_table_entries`] when you want a single combined entry for
/// all outputs.
///
/// # Table layout
/// `table` has `2^k` entries.  `(table[idx] >> out_bit) & 1` is the selected
/// output bit for input combination `idx` (LSB-first encoding).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LutShoutTable {
    /// Stable identifier matching `LutDesc::lut_id`.
    pub lut_id: u32,
    /// Number of input bits (k).
    pub k: usize,
    /// Total number of output bits (m) in the original LUT.
    pub m: usize,
    /// Which output bit (0..m) this instance exposes.
    pub out_bit: usize,
    /// Packed truth table: `2^k` u64 entries; bit j of `table[i]` is output
    /// bit j for input combination i (copied from `SubCircuitLut::table`).
    table: Vec<u64>,
}

impl LutShoutTable {
    /// Build a `LutShoutTable` from a [`LutDesc`].
    ///
    /// Parses the raw truth-table bytes and expands them into a `Vec<u64>` of
    /// `2^k` entries (same format as [`SubCircuitLut`]).
    ///
    /// # Panics
    /// Panics if `out_bit >= m`.
    pub fn from_lut_desc(desc: &LutDesc, out_bit: usize) -> Self {
        assert!(out_bit < desc.m, "out_bit {out_bit} >= m {}", desc.m);
        // Reuse SubCircuitLut's byte parser.
        let sub = SubCircuitLut::from_bytes(desc.k, desc.m, &desc.truth_table);
        LutShoutTable {
            lut_id: desc.lut_id,
            k: desc.k,
            m: desc.m,
            out_bit,
            table: sub.table,
        }
    }

    /// Number of truth-table entries (2^k).
    #[inline]
    pub fn table_size(&self) -> usize {
        1usize << self.k
    }

    /// Raw truth-table value (all m bits packed) for input combination `idx`.
    #[inline]
    pub fn raw_entry(&self, idx: usize) -> u64 {
        self.table[idx]
    }

    /// Value of the selected output bit for input combination `idx`.
    #[inline]
    pub fn entry(&self, idx: usize) -> u64 {
        (self.table[idx] >> self.out_bit) & 1u64
    }
}

impl JoltLookupTable for LutShoutTable {
    /// Return the selected output bit for the given input combination `index`.
    ///
    /// # Note
    /// `index` is cast to `usize`; it must be in range `0..2^k`.
    fn materialize_entry(&self, index: u128) -> u64 {
        let idx = index as usize;
        debug_assert!(idx < self.table_size(), "LutShoutTable: index {idx} out of range (2^k={})", self.table_size());
        self.entry(idx)
    }

    /// Evaluate the MLE of the selected output bit at the field point `r ∈ F^k`.
    ///
    /// Uses the standard O(2^k) fold algorithm (LSB-first variable ordering).
    ///
    /// # Algorithm
    /// ```text
    /// T̃_{out_bit}(r) = Σ_{x ∈ {0,1}^k} T_{out_bit}[x] · eq_lsb(r, x)
    /// ```
    /// where `eq_lsb(r, x) = Π_i (r_i·x_i + (1-r_i)·(1-x_i))`.
    ///
    /// The fold is: for each variable i (0..k), halve the table by binding
    /// variable i to `r[i]`:
    /// ```text
    /// vals[i] ← vals[2i] + r_i · (vals[2i+1] − vals[2i])
    /// ```
    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        assert_eq!(
            r.len(), self.k,
            "LutShoutTable::evaluate_mle: expected {} vars, got {}",
            self.k,
            r.len()
        );
        let n = self.table_size();

        // Initialise values from the selected output column.
        let mut vals: Vec<F> = (0..n)
            .map(|idx| F::from_u64(self.entry(idx)))
            .collect();

        // Fold each variable in LSB-first order.
        // After binding variable i at r[i], the slice halves.
        for &ri in r.iter() {
            let half = vals.len() / 2;
            for j in 0..half {
                let lo = vals[2 * j];
                let hi = vals[2 * j + 1];
                // lo + ri * (hi - lo)   [C * F = F via ChallengeFieldOps]
                vals[j] = lo + ri * (hi - lo);
            }
            vals.truncate(half);
        }

        debug_assert_eq!(vals.len(), 1, "LutShoutTable: fold did not converge");
        vals[0]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Converter: LutDesc → SubCircuitLut
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a [`LutDesc`] into a [`SubCircuitLut`].
///
/// This is a thin wrapper around [`SubCircuitLut::from_bytes`], extracting the
/// byte-format truth table from the `LutDesc`.
///
/// # Example
/// ```rust,ignore
/// let sub = lut_desc_to_sub_circuit(&desc);
/// let val = sub.evaluate_mle_at::<Fr>(&r, 0);
/// ```
pub fn lut_desc_to_sub_circuit(desc: &LutDesc) -> SubCircuitLut {
    SubCircuitLut::from_bytes(desc.k, desc.m, &desc.truth_table)
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-output alpha batching
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the alpha-batched table entries for a multi-output [`LutDesc`].
///
/// For a LUT with m output bits and a random challenge `α`, returns a vector of
/// `2^k` field elements:
/// ```text
/// batched[idx] = Σ_{j=0}^{m-1}  α^j · T_j[idx]
/// ```
/// where `T_j[idx] = (table[idx] >> j) & 1`.
///
/// This allows reducing m output columns to a single scalar for use in the
/// Shout `Val` polynomial during the Read sumcheck.
///
/// # Usage in Phase S3
/// ```rust,ignore
/// let alpha = transcript.challenge_scalar::<Fr>();
/// let batched = alpha_batch_table_entries(&desc, alpha);
/// // Use `batched` as the Val(k) entries for the mega-table MLE.
/// ```
pub fn alpha_batch_table_entries(desc: &LutDesc, alpha: Fr) -> Vec<Fr> {
    let sub = SubCircuitLut::from_bytes(desc.k, desc.m, &desc.truth_table);
    let n = 1usize << desc.k;
    let mut result = vec![Fr::zero(); n];

    // Accumulate: result[idx] += alpha^j * T_j[idx], for j in 0..m
    let mut alpha_pow = Fr::one(); // α^0 = 1
    for j in 0..desc.m {
        for idx in 0..n {
            let bit = Fr::from_u64((sub.table[idx] >> j) & 1u64);
            result[idx] += alpha_pow * bit;
        }
        alpha_pow *= alpha; // advance to α^{j+1}
    }

    result
}

/// Evaluate the MLE of the alpha-batched combined table at point `r ∈ F^k`.
///
/// Equivalent to `Σ_j α^j · T̃_j(r)`, but computed in a single fold pass over
/// the combined entries.
///
/// # Arguments
/// * `batched` — output of [`alpha_batch_table_entries`], length `2^k`
/// * `r`       — evaluation point in `F^k` (LSB-first ordering)
pub fn evaluate_batched_table_mle(batched: &[Fr], r: &[Fr]) -> Fr {
    let k = r.len();
    assert_eq!(
        batched.len(),
        1usize << k,
        "evaluate_batched_table_mle: batched.len()={} != 2^k={}",
        batched.len(),
        1usize << k,
    );

    let mut vals = batched.to_vec();

    for &ri in r.iter() {
        let half = vals.len() / 2;
        for j in 0..half {
            let lo = vals[2 * j];
            let hi = vals[2 * j + 1];
            vals[j] = lo + ri * (hi - lo);
        }
        vals.truncate(half);
    }

    debug_assert_eq!(vals.len(), 1);
    vals[0]
}

// ─────────────────────────────────────────────────────────────────────────────
// Shout address encoding  (Phase S2 preparation)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **mega-table lookup address** for a trace row in the Shout scheme.
///
/// In the mega-table layout (single Shout instance covering all LUT types),
/// the address encodes both *which* LUT type was used and *which* input
/// combination was applied:
/// ```text
/// address = type_index * 2^k + packed_input
/// ```
/// where `type_index` is the 0-based index of the LUT type in the sorted type
/// map, and `packed_input = Σ_i input[i] * 2^i` (LSB-first).
///
/// # Arguments
/// * `type_index`   — canonical index of the LUT type (from `type_order_map`)
/// * `k`            — number of input bits (same k for all LUT types in mega-table)
/// * `packed_input` — inputs packed as a bit integer (`PackedInput` encoding)
///
/// # Returns
/// The integer lookup address, suitable for constructing `OneHotPolynomial`
/// witnesses in Phase S2.
#[inline]
pub fn shout_address(type_index: usize, k: usize, packed_input: usize) -> usize {
    (type_index << k) | packed_input
}

/// Pack boolean inputs into an integer (LSB-first: `inputs[0]` = bit 0).
#[inline]
pub fn pack_inputs(inputs: &[bool]) -> usize {
    inputs
        .iter()
        .enumerate()
        .fold(0usize, |acc, (i, &b)| if b { acc | (1 << i) } else { acc })
}

/// Total Shout address-space size for a mega-table with `t_pad` padded LUT
/// types and `k` input bits.
///
/// The address space is `t_pad * 2^k`, requiring
/// `ceil(log2(t_pad)) + k` address bits in total.
#[inline]
pub fn mega_table_address_bits(t_pad: usize, k: usize) -> usize {
    // ceil(log2(t_pad)) = trailing_zeros(t_pad.next_power_of_two())
    let type_bits = if t_pad <= 1 {
        0
    } else {
        t_pad.next_power_of_two().trailing_zeros() as usize
    };
    type_bits + k
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase S2: Trace → OneHotPolynomial witnesses
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Shout witness construction: how the mega-table address is
/// chunked into segments for the One-Hot RA polynomials.
///
/// # Address space layout
///
/// For a mega-table with `t_pad` padded LUT types and uniform `k` input bits,
/// every trace row is encoded as:
/// ```text
/// address = type_index * 2^k + packed_input
/// ```
/// The address has `total_address_bits = ceil(log2(t_pad)) + k` bits.
/// These bits are split into `d = ceil(total_address_bits / log_k_chunk)` chunks
/// of `log_k_chunk` bits each, from LSB to MSB.
///
/// Each chunk gives one [`OneHotPolynomial<Fr>`] with `K = 2^log_k_chunk` entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneHotParams {
    /// Bits per chunk. Must satisfy `1 ≤ log_k_chunk ≤ 8` (K ≤ 256 fits in `u8`).
    pub log_k_chunk: usize,
    /// Number of chunks: `d = ceil(total_address_bits / log_k_chunk)`.
    pub d: usize,
    /// Total address bits: `ceil(log2(t_pad)) + k`.
    pub total_address_bits: usize,
    /// Chunk entry count (K per chunk): `1 << log_k_chunk`.
    pub k_chunk: usize,
}

impl OneHotParams {
    /// Build [`OneHotParams`] from circuit parameters.
    ///
    /// # Arguments
    /// * `n_types`     — number of distinct LUT types in the circuit.
    /// * `k`           — uniform number of input bits (max k across all LUT types).
    /// * `log_k_chunk` — bits per chunk (must be `1..=8`).
    ///
    /// # Panics
    /// Panics if `log_k_chunk == 0` or `log_k_chunk > 8`.
    pub fn new(n_types: usize, k: usize, log_k_chunk: usize) -> Self {
        assert!(
            log_k_chunk >= 1 && log_k_chunk <= 8,
            "log_k_chunk must be 1..=8, got {log_k_chunk}"
        );
        let t_pad = n_types.next_power_of_two().max(1);
        let total_address_bits = mega_table_address_bits(t_pad, k);
        let d = total_address_bits.div_ceil(log_k_chunk);
        let k_chunk = 1usize << log_k_chunk;
        OneHotParams { log_k_chunk, d, total_address_bits, k_chunk }
    }
}

/// Extract chunk `chunk_idx` from a mega-table `address`.
///
/// Returns the `log_k_chunk`-bit slice starting at bit `chunk_idx * log_k_chunk`
/// as a `u8`.  Bits above `total_address_bits` are zero.
///
/// # Example
/// ```text
/// address = 0b_1011_0101, log_k_chunk = 4
/// chunk 0 → bits [0..4)  = 0b0101 = 5
/// chunk 1 → bits [4..8)  = 0b1011 = 11
/// ```
#[inline]
pub fn address_chunk(address: usize, chunk_idx: usize, log_k_chunk: usize) -> u8 {
    let shift = chunk_idx * log_k_chunk;
    let mask = (1usize << log_k_chunk) - 1;
    ((address >> shift) & mask) as u8
}

/// Convert a [`LutEval`] trace into `d` [`OneHotPolynomial`] witnesses.
///
/// For each trace row `j`:
/// 1. Compute the mega-table address: `address[j] = type_index * 2^k + packed_input`
/// 2. Split address into `d = params.d` chunks of `params.log_k_chunk` bits.
/// 3. Set `ra_i.nonzero_indices[j] = Some(chunk_i(address[j]))` for each chunk `i`.
///
/// Rows `j ≥ trace.len()` are padded with `None` (no lookup at those timesteps).
///
/// # Precondition
/// `DoryGlobals::initialize(params.k_chunk, t_total)` **must** have been called
/// before this function.  Each `OneHotPolynomial::from_indices` `debug_assert`s
/// that `DoryGlobals::get_T() == t_total`.
///
/// # Arguments
/// * `trace`         — flat evaluation trace from [`evaluate_lut_circuit`]
/// * `type_index_of` — maps `lut_id` → canonical 0-based type index (sorted order)
/// * `k`             — uniform input-bit count (max k across all LUT types)
/// * `params`        — chunking configuration (from [`OneHotParams::new`])
/// * `t_total`       — padded trace length (must be a power of two ≥ `trace.len()`)
///
/// # Returns
/// `params.d` `OneHotPolynomial<Fr>` witnesses, one per address chunk.
///
/// [`evaluate_lut_circuit`]: crate::lut_czbc::evaluate_lut_circuit
pub fn build_shout_witnesses(
    trace: &[LutEval],
    type_index_of: &HashMap<u32, usize>,
    k: usize,
    params: &OneHotParams,
    t_total: usize,
) -> Vec<OneHotPolynomial<Fr>> {
    assert!(
        t_total >= trace.len(),
        "t_total ({t_total}) must be ≥ trace.len() ({})",
        trace.len()
    );
    assert!(
        t_total == 0 || t_total.is_power_of_two(),
        "t_total must be a power of two, got {t_total}"
    );

    // Pre-compute the mega-table address for every trace row.
    let addresses: Vec<usize> = trace
        .iter()
        .map(|ev| {
            let tid = *type_index_of
                .get(&ev.lut_id)
                .unwrap_or_else(|| panic!(
                    "build_shout_witnesses: unknown lut_id={}",
                    ev.lut_id
                ));
            ev.address_for_shout(tid, k)
        })
        .collect();

    // Build one OneHotPolynomial per address chunk.
    (0..params.d)
        .map(|chunk_i| {
            let indices: Vec<Option<u8>> = (0..t_total)
                .map(|j| {
                    if j < addresses.len() {
                        // Active trace row: record which chunk-entry is hot.
                        Some(address_chunk(addresses[j], chunk_i, params.log_k_chunk))
                    } else {
                        // Padded row: no lookup.
                        None
                    }
                })
                .collect();
            OneHotPolynomial::from_indices(indices, params.k_chunk)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase S3: Shout Prover        (simplified, self-contained for LUTs)
// Phase S4: Shout Verifier
// ─────────────────────────────────────────────────────────────────────────────
//
// # Protocol overview
//
// The prover convinces the verifier that every trace row j executed a LUT
// evaluation `batch_val[j] = mega_table[address[j]]` correctly, using two
// chained sumchecks and two HyperKZG opening proofs.
//
// **Committed polynomials:**
// - `batch_val` (dense, length t_total): batch_val[j] = α-batched table value
//   for cycle j's lookup address.
// - `G_agg`     (dense, length mega_size): G_agg[addr] = Σ_j eq(r_cy, j) · 1[address[j]=addr]
//   = weight of address `addr` in the cycle-sumcheck reduction.
//
// **Phase 1 — Cycle sumcheck** (log_T rounds, degree 2):
//   Σ_{j ∈ {0,1}^{log_T}} eq(r_T, j) · batch_val[j]  =  output_claim
//   → challenges r_cycle; final claim batch_val(r_cycle) = bv_eval.
//
// **Phase 2 — Address sumcheck** (total_address_bits rounds, degree 2):
//   Σ_{addr ∈ {0,1}^M} G_agg(addr) · mega_table(addr)  =  bv_eval
//   → challenges r_addr; final claim G_agg(r_addr) · mega_table_mle(r_addr) = addr_final.
//
// **Soundness:** batch_val is committed before r_T is sampled (binding the
// prover); G_agg is committed before r_addr is sampled.

type Challenge = <Fr as JoltField>::Challenge;

// ── Internal field helpers (equiv. to lut_construct.rs helpers, kept private) ──

/// Bind the LSB variable of `poly` in-place to `r`:
///   `poly[i] ← poly[2i]·(1−r) + poly[2i+1]·r`.
#[inline]
fn bind_poly(poly: &mut Vec<Fr>, r: Fr) {
    let half = poly.len() / 2;
    for i in 0..half {
        poly[i] = poly[2 * i] + r * (poly[2 * i + 1] - poly[2 * i]);
    }
    poly.truncate(half);
}

/// Build eq(r, ·) over {0,1}^m in LSB-first ordering (same as `lut_construct::init_eq`).
fn init_eq_fr(r: &[Fr]) -> Vec<Fr> {
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

/// eq(r_a, r_b): product of per-variable equality factors (LSB-first).
fn eq_final_eval(r_a: &[Fr], r_b: &[Fr]) -> Fr {
    debug_assert_eq!(r_a.len(), r_b.len());
    r_a.iter()
        .zip(r_b.iter())
        .map(|(&ai, &bi)| ai * bi + (Fr::one() - ai) * (Fr::one() - bi))
        .product()
}

/// Evaluate a degree-2 univariate (given as evaluations at t=0,1,2) at `t` via
/// Lagrange interpolation.
fn lagrange_deg2(evals: &[Fr; 3], t: Fr) -> Fr {
    let [e0, e1, e2] = *evals;
    let two = Fr::from(2u64);
    let half = two.inverse().expect("2 is invertible in BN254 Fr");
    // L_0(t) = (t-1)(t-2)/2,  L_1(t) = -t(t-2),  L_2(t) = t(t-1)/2
    let l0 = (t - Fr::one()) * (t - two) * half;
    let l1 = -t * (t - two);
    let l2 = t * (t - Fr::one()) * half;
    e0 * l0 + e1 * l1 + e2 * l2
}

/// Evaluate the MLE of `entries` (LSB-first, dense) at field point `point`.
///
/// Requires `entries.len() == 2^(point.len())`.
pub fn mle_eval_fr(entries: &[Fr], point: &[Fr]) -> Fr {
    debug_assert_eq!(
        entries.len(),
        1usize << point.len(),
        "mle_eval_fr: length mismatch"
    );
    let mut vals = entries.to_vec();
    for &ri in point.iter() {
        let half = vals.len() / 2;
        for j in 0..half {
            vals[j] = vals[2 * j] + ri * (vals[2 * j + 1] - vals[2 * j]);
        }
        vals.truncate(half);
    }
    vals[0]
}

// ── Alpha-batched mega-table ─────────────────────────────────────────────────

/// Build the alpha-batched mega-table for `circ`.
///
/// For a circuit with `n_types` LUT types sorted by `lut_id`, the mega-table
/// has `t_pad * 2^k` entries, where `t_pad` is the smallest power of two ≥
/// `n_types`.  Entry at index `tid * 2^k + input_bits` equals
/// `Σ_{j=0}^{m-1} α^j · T_j[input_bits]` for the truth table of LUT type `tid`.
/// LUT types with fewer than `k` input bits are replicated over the MSBs.
///
/// # Arguments
/// * `circ`       — loaded LUT circuit (used for truth tables)
/// * `type_order` — sorted list of `lut_id` values (determines `tid` mapping)
/// * `k`          — uniform (padded) input-bit count
/// * `alpha`      — Fiat-Shamir alpha challenge for multi-output batching
pub fn build_mega_table_batched(
    circ: &LutCirc,
    type_order: &[u32],
    k: usize,
    alpha: Fr,
) -> Vec<Fr> {
    let n_types = type_order.len();
    let t_pad = n_types.next_power_of_two().max(1);
    let table_size = 1usize << k;
    let mega_size = t_pad * table_size;

    let mut mega = vec![Fr::zero(); mega_size];
    for (tid, &lut_id) in type_order.iter().enumerate() {
        let desc = &circ.lut_types[&lut_id];
        let batched = alpha_batch_table_entries(desc, alpha);
        let k_lut = desc.k;
        // If k_lut < k, replicate the truth table for the extra MSBs (they are
        // don't-care bits in the mega-table layout).
        let reps = 1usize << (k - k_lut);
        for rep in 0..reps {
            for input_lo in 0usize..(1 << k_lut) {
                let mega_idx = tid * table_size + rep * (1 << k_lut) + input_lo;
                mega[mega_idx] = batched[input_lo];
            }
        }
    }
    mega
}

/// SRS size (number of multilinear variables) required for the Shout prover.
///
/// The prover commits two polynomials:
/// - `batch_val` of length `t_total = 2^{log_T}`        → needs `log_T` vars  
/// - `G_agg`     of length `2^{total_address_bits}`      → needs `total_address_bits` vars
///
/// Returns the larger of the two.
pub fn shout_max_num_vars(n_types: usize, k: usize, cycles: u32, n_ops: usize) -> usize {
    let n_total = n_ops * cycles.max(1) as usize;
    let t_total = n_total.next_power_of_two().max(1);
    let log_t = t_total.trailing_zeros() as usize;

    let t_pad = n_types.next_power_of_two().max(1);
    let total_address_bits = mega_table_address_bits(t_pad, k);

    log_t.max(total_address_bits).max(1)
}

// ── Proof types ──────────────────────────────────────────────────────────────

/// The complete Shout-based LUT proof (Phases S3+S4).
///
/// Contains two chained sumcheck transcripts and two HyperKZG commitment +
/// opening proof pairs, plus the scalar links between the phases.
#[derive(Clone, Debug)]
pub struct ShoutLutProof {
    // ── public metadata (replicated so verifier can check them) ──
    /// Number of distinct LUT types in the circuit.
    pub n_types: usize,
    /// Uniform (padded) input-bit count across all LUT types.
    pub k: usize,
    /// Padded trace length (power of two).
    pub t_total: usize,

    // ── Fiat-Shamir: alpha challenge ──
    /// Random challenge for multi-output alpha-batching.
    pub alpha: Fr,

    // ── Phase 1: Cycle sumcheck ──
    /// HyperKZG commitment to `batch_val` (length `t_total`).
    pub comm_bv: HyperKZGCommitment<Bn254>,
    /// Round polynomials for the cycle sumcheck (log_T rounds, degree 2,
    /// each stored as evaluations at 0, 1, 2).
    pub cycle_sc_polys: Vec<[Fr; 3]>,
    /// `batch_val(r_cycle)` — the cycle-sumcheck's final claimed evaluation.
    pub bv_eval: Fr,
    /// HyperKZG opening proof for `batch_val` at `r_cycle`.
    pub opening_bv: Option<HyperKZGProof<Bn254>>,

    // ── Phase 2: Address sumcheck ──
    /// HyperKZG commitment to `G_agg` (length `2^{total_address_bits}`).
    pub comm_g: HyperKZGCommitment<Bn254>,
    /// Round polynomials for the address sumcheck (`total_address_bits` rounds).
    pub addr_sc_polys: Vec<[Fr; 3]>,
    /// `G_agg(r_addr)` — the address-sumcheck's final claimed G evaluation.
    pub final_g_eval: Fr,
    /// `mega_table_mle(r_addr)` — prover's claimed table MLE at `r_addr`
    /// (verifier independently recomputes and checks this).
    pub final_table_eval: Fr,
    /// HyperKZG opening proof for `G_agg` at `r_addr`.
    pub opening_g: Option<HyperKZGProof<Bn254>>,
}

// ── Prover ───────────────────────────────────────────────────────────────────

/// Prove that the LUT circuit evaluation trace is correct using the Shout
/// protocol (two chained sumchecks).
///
/// # Arguments
/// * `circ`          — loaded LUT circuit (truth tables, not the trace)
/// * `trace`         — evaluation trace from [`evaluate_lut_circuit`]
/// * `type_index_of` — maps `lut_id` → canonical type index (sorted order)
/// * `k`             — uniform input-bit count (max across all LUT types)
/// * `t_total`       — padded trace length (must be a power of two)
/// * `pk`            — HyperKZG prover key (must support `shout_max_num_vars` vars)
/// * `transcript`    — Keccak Fiat-Shamir transcript
///
/// # Returns
/// A [`ShoutLutProof`] whose validity can be checked with [`verify_shout_lut`].
pub fn prove_shout_lut(
    circ: &LutCirc,
    trace: &[LutEval],
    type_index_of: &HashMap<u32, usize>,
    k: usize,
    t_total: usize,
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> ShoutLutProof {
    let log_t = (t_total.trailing_zeros() as usize).max(0);
    let n_types = type_index_of.len();
    let t_pad = n_types.next_power_of_two().max(1);
    let mega_size = t_pad * (1usize << k);
    let total_address_bits = mega_table_address_bits(t_pad, k);

    // Sorted type order (deterministic, used by both prover and verifier).
    let mut type_order: Vec<u32> = type_index_of.keys().copied().collect();
    type_order.sort_unstable();

    // ── Fiat-Shamir: bind public circuit parameters ──────────────────────────
    transcript.append_u64(n_types as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(t_total as u64);

    // ── 1. Alpha challenge for multi-output batching ─────────────────────────
    let alpha: Fr = transcript.challenge_scalar();

    // ── 2. Build alpha-batched mega-table ────────────────────────────────────
    let mega_table = build_mega_table_batched(circ, &type_order, k, alpha);

    // ── 3. Compute batch_val[j] for each trace row ───────────────────────────
    // batch_val[j] = mega_table[type_index[j] * 2^k + packed_input[j]]
    let mut batch_val = vec![Fr::zero(); t_total];
    for (j, ev) in trace.iter().enumerate() {
        let tid = *type_index_of
            .get(&ev.lut_id)
            .expect("prove_shout_lut: unknown lut_id");
        let addr = ev.address_for_shout(tid, k);
        batch_val[j] = mega_table[addr];
    }

    // ── 4. Commit batch_val ──────────────────────────────────────────────────
    let mle_bv = MultilinearPolynomial::from(batch_val.clone());
    let comm_bv = HyperKZG::<Bn254>::commit(pk, &mle_bv).expect("commit batch_val");
    comm_bv.append_to_transcript(transcript);

    // ── 5. Cycle sumcheck challenge r_T ──────────────────────────────────────
    // The verifier supplies r_T AFTER seeing comm_bv.
    let r_t_fr: Vec<Fr> = transcript.challenge_vector(log_t);

    // ── 6. Compute and commit output_claim ──────────────────────────────────
    let eq_t = init_eq_fr(&r_t_fr);
    let output_claim: Fr = eq_t
        .iter()
        .zip(batch_val.iter())
        .map(|(e, v)| *e * *v)
        .sum();
    transcript.append_scalar(&output_claim);

    // ── 7. Phase 1: Cycle sumcheck ───────────────────────────────────────────
    // Proves: Σ_{j ∈ {0,1}^{log_T}} eq(r_T, j) · batch_val[j] = output_claim
    // Round i (degree 2 univariate): h_i(X) = Σ_{j_rest} eq_i(X,j_rest) · bv_i(X,j_rest)
    let mut eq_work = eq_t;
    let mut bv_work = batch_val;

    let mut cycle_sc_polys: Vec<[Fr; 3]> = Vec::with_capacity(log_t);
    let mut r_cycle_fr: Vec<Fr> = Vec::with_capacity(log_t);
    let mut r_cycle_ch: Vec<Challenge> = Vec::with_capacity(log_t);

    for _round in 0..log_t {
        let half = eq_work.len() / 2;
        let two = Fr::from(2u64);
        let mut p = [Fr::zero(); 3];
        for idx in 0..half {
            let (eq_lo, eq_hi) = (eq_work[2 * idx], eq_work[2 * idx + 1]);
            let (bv_lo, bv_hi) = (bv_work[2 * idx], bv_work[2 * idx + 1]);
            // Evaluations at X = 0, 1, 2 (linear extension in X):
            let eq_2 = eq_lo + two * (eq_hi - eq_lo);
            let bv_2 = bv_lo + two * (bv_hi - bv_lo);
            p[0] += eq_lo * bv_lo;
            p[1] += eq_hi * bv_hi;
            p[2] += eq_2 * bv_2;
        }
        for &e in p.iter() {
            transcript.append_scalar(&e);
        }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_cycle_ch.push(r_j_ch);
        r_cycle_fr.push(r_j);
        bind_poly(&mut eq_work, r_j);
        bind_poly(&mut bv_work, r_j);
        cycle_sc_polys.push(p);
    }

    // Final cycle-sumcheck claim: bv_work[0] = batch_val(r_cycle).
    let bv_eval = bv_work[0];
    transcript.append_scalar(&bv_eval);

    // ── 8. HyperKZG: open batch_val at r_cycle ──────────────────────────────
    // HyperKZG uses MSB-first (big-endian) variable ordering; sumcheck used LSB-first.
    let point_bv_kzg: Vec<Challenge> = r_cycle_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();
    let opening_bv = if comm_bv != zero_comm {
        Some(
            HyperKZG::<Bn254>::open(pk, &mle_bv, &point_bv_kzg, &bv_eval, transcript)
                .expect("HyperKZG open batch_val"),
        )
    } else {
        None
    };

    // ── 9. Build G_agg ───────────────────────────────────────────────────────
    // G_agg[addr] = Σ_j eq(r_cycle, j) · 1[address[j] = addr]
    // Satisfies: Σ_{addr} G_agg[addr] · mega_table[addr] = batch_val(r_cycle) = bv_eval.
    let eq_cycle = init_eq_fr(&r_cycle_fr);
    let mut g_agg = vec![Fr::zero(); mega_size];
    for (j, ev) in trace.iter().enumerate() {
        let tid = *type_index_of.get(&ev.lut_id).unwrap();
        let addr = ev.address_for_shout(tid, k);
        g_agg[addr] += eq_cycle[j];
    }

    // ── 10. Commit G_agg ─────────────────────────────────────────────────────
    let mle_g = MultilinearPolynomial::from(g_agg.clone());
    let comm_g = HyperKZG::<Bn254>::commit(pk, &mle_g).expect("commit G_agg");
    comm_g.append_to_transcript(transcript);

    // ── 11. Phase 2: Address sumcheck ────────────────────────────────────────
    // Proves: Σ_{addr ∈ {0,1}^M} G_agg(addr) · mega_table(addr) = bv_eval
    let mut g_work = g_agg;
    let mut t_work = mega_table.clone();

    let mut addr_sc_polys: Vec<[Fr; 3]> = Vec::with_capacity(total_address_bits);
    let mut r_addr_fr: Vec<Fr> = Vec::with_capacity(total_address_bits);
    let mut r_addr_ch: Vec<Challenge> = Vec::with_capacity(total_address_bits);

    for _round in 0..total_address_bits {
        let half = g_work.len() / 2;
        let two = Fr::from(2u64);
        let mut p = [Fr::zero(); 3];
        for idx in 0..half {
            let (g_lo, g_hi) = (g_work[2 * idx], g_work[2 * idx + 1]);
            let (t_lo, t_hi) = (t_work[2 * idx], t_work[2 * idx + 1]);
            let g_2 = g_lo + two * (g_hi - g_lo);
            let t_2 = t_lo + two * (t_hi - t_lo);
            p[0] += g_lo * t_lo;
            p[1] += g_hi * t_hi;
            p[2] += g_2 * t_2;
        }
        for &e in p.iter() {
            transcript.append_scalar(&e);
        }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_addr_ch.push(r_j_ch);
        r_addr_fr.push(r_j);
        bind_poly(&mut g_work, r_j);
        bind_poly(&mut t_work, r_j);
        addr_sc_polys.push(p);
    }

    let final_g_eval = g_work[0];
    let final_table_eval = t_work[0];
    transcript.append_scalar(&final_g_eval);
    transcript.append_scalar(&final_table_eval);

    // ── 12. HyperKZG: open G_agg at r_addr ──────────────────────────────────
    let point_g_kzg: Vec<Challenge> = r_addr_ch.iter().rev().cloned().collect();
    let opening_g = if comm_g != zero_comm {
        Some(
            HyperKZG::<Bn254>::open(pk, &mle_g, &point_g_kzg, &final_g_eval, transcript)
                .expect("HyperKZG open G_agg"),
        )
    } else {
        None
    };

    ShoutLutProof {
        n_types,
        k,
        t_total,
        alpha,
        comm_bv,
        cycle_sc_polys,
        bv_eval,
        opening_bv,
        comm_g: comm_g,
        addr_sc_polys,
        final_g_eval: final_g_eval,
        final_table_eval,
        opening_g: opening_g,
    }
}

// ── Verifier ─────────────────────────────────────────────────────────────────

/// Verify a [`ShoutLutProof`].
///
/// The verifier does **not** re-execute the circuit.  It needs only:
/// - The proof.
/// - The public circuit description (truth tables) — to reconstruct the
///   alpha-batched mega-table MLE.
/// - The HyperKZG verifier key.
/// - The same transcript initialisation used by the prover.
///
/// Returns `true` iff all checks pass.
pub fn verify_shout_lut(
    proof: &ShoutLutProof,
    circ: &LutCirc,
    vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    let ShoutLutProof {
        n_types,
        k,
        t_total,
        alpha,
        comm_bv,
        cycle_sc_polys,
        bv_eval,
        opening_bv,
        comm_g,
        addr_sc_polys,
        final_g_eval,
        final_table_eval,
        opening_g,
    } = proof;

    let n_types = *n_types;
    let k = *k;
    let t_total = *t_total;
    let log_t = (t_total.trailing_zeros() as usize).max(0);
    let t_pad = n_types.next_power_of_two().max(1);
    let total_address_bits = mega_table_address_bits(t_pad, k);

    // ── Re-bind public circuit parameters ───────────────────────────────────
    transcript.append_u64(n_types as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(t_total as u64);

    // ── Re-derive alpha and check it matches the proof ──────────────────────
    let alpha_check: Fr = transcript.challenge_scalar();
    if alpha_check != *alpha {
        eprintln!("verify_shout_lut: alpha mismatch");
        return false;
    }

    // ── Reconstruct mega-table (public computation) ──────────────────────────
    let mut type_order: Vec<u32> = circ.lut_types.keys().copied().collect();
    type_order.sort_unstable();
    let mega_table = build_mega_table_batched(circ, &type_order, k, *alpha);

    // ── Absorb comm_bv → re-derive r_T ──────────────────────────────────────
    comm_bv.append_to_transcript(transcript);
    let r_t_fr: Vec<Fr> = transcript.challenge_vector(log_t);

    // Absorb output_claim (prover sends it; verifier checks via the sumcheck).
    // The verifier can independently compute the claimed sum from the sumcheck
    // consistency — it just needs to absorb the value the prover sent.
    let output_claim: Fr = {
        // Reconstruct: equals p_0(0) + p_0(1) from the first cycle-sc round.
        // If there are no rounds (log_t=0), the prover just sends bv_eval directly
        // as the output_claim (single-element trace).
        if cycle_sc_polys.is_empty() {
            *bv_eval
        } else {
            cycle_sc_polys[0][0] + cycle_sc_polys[0][1]
        }
    };
    transcript.append_scalar(&output_claim);

    // ── Phase 1: Verify cycle sumcheck ───────────────────────────────────────
    let mut prev_claim = output_claim;
    let mut r_cycle_fr: Vec<Fr> = Vec::with_capacity(log_t);
    let mut r_cycle_ch: Vec<Challenge> = Vec::with_capacity(log_t);

    for (round, poly) in cycle_sc_polys.iter().enumerate() {
        let sum = poly[0] + poly[1];
        if sum != prev_claim {
            eprintln!(
                "verify_shout_lut: cycle sc round {round}: p(0)+p(1)={sum:?} ≠ {prev_claim:?}"
            );
            return false;
        }
        for &e in poly.iter() {
            transcript.append_scalar(&e);
        }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_cycle_ch.push(r_j_ch);
        r_cycle_fr.push(r_j);
        prev_claim = lagrange_deg2(poly, r_j);
    }

    // Final cycle-sumcheck claim: prev_claim = eq(r_T, r_cycle) · bv(r_cycle).
    // The verifier independently checks the eq factor.
    let eq_t_at_rcycle = eq_final_eval(&r_t_fr, &r_cycle_fr);
    let expected_cycle_final = eq_t_at_rcycle * bv_eval;
    if expected_cycle_final != prev_claim {
        eprintln!(
            "verify_shout_lut: cycle sc final mismatch: eq·bv_eval={expected_cycle_final:?} ≠ prev_claim={prev_claim:?}"
        );
        return false;
    }

    // Absorb bv_eval into transcript (matches prover step 7 append).
    transcript.append_scalar(bv_eval);

    // ── Verify HyperKZG opening of batch_val at r_cycle ──────────────────────
    let point_bv_kzg: Vec<Challenge> = r_cycle_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();
    if *comm_bv != zero_comm {
        match opening_bv {
            Some(pf) => {
                if HyperKZG::<Bn254>::verify(vk, comm_bv, &point_bv_kzg, bv_eval, pf, transcript)
                    .is_err()
                {
                    eprintln!("verify_shout_lut: batch_val HyperKZG verify FAILED");
                    return false;
                }
            }
            None => {
                eprintln!("verify_shout_lut: non-zero comm_bv but no opening proof");
                return false;
            }
        }
    }

    // ── Absorb comm_g → Phase 2 address sumcheck ─────────────────────────────
    comm_g.append_to_transcript(transcript);

    // Phase 2 initial claim = bv_eval = Σ_{addr} G_agg(addr) · mega_table(addr).
    // (This follows from the derivation: bv(r_cycle) = Σ_addr G_agg[addr]*mega[addr].)
    let mut addr_prev_claim = *bv_eval;
    let mut r_addr_fr: Vec<Fr> = Vec::with_capacity(total_address_bits);
    let mut r_addr_ch: Vec<Challenge> = Vec::with_capacity(total_address_bits);

    for (round, poly) in addr_sc_polys.iter().enumerate() {
        let sum = poly[0] + poly[1];
        if sum != addr_prev_claim {
            eprintln!(
                "verify_shout_lut: addr sc round {round}: p(0)+p(1)={sum:?} ≠ {addr_prev_claim:?}"
            );
            return false;
        }
        for &e in poly.iter() {
            transcript.append_scalar(&e);
        }
        let r_j_ch: Challenge = transcript.challenge_scalar_optimized::<Fr>();
        let r_j: Fr = r_j_ch.into();
        r_addr_ch.push(r_j_ch);
        r_addr_fr.push(r_j);
        addr_prev_claim = lagrange_deg2(poly, r_j);
    }

    // Final address-sumcheck claim: G_agg(r_addr) · mega_table_mle(r_addr) = addr_prev_claim.
    let addr_final_expected = *final_g_eval * *final_table_eval;
    if addr_final_expected != addr_prev_claim {
        eprintln!(
            "verify_shout_lut: addr sc final: G·T={addr_final_expected:?} ≠ {addr_prev_claim:?}"
        );
        return false;
    }

    // Verifier independently recomputes mega_table_mle at r_addr and checks it.
    let table_eval_check = mle_eval_fr(&mega_table, &r_addr_fr);
    if table_eval_check != *final_table_eval {
        eprintln!(
            "verify_shout_lut: mega_table_mle mismatch: {table_eval_check:?} ≠ {final_table_eval:?}"
        );
        return false;
    }

    transcript.append_scalar(final_g_eval);
    transcript.append_scalar(final_table_eval);

    // ── Verify HyperKZG opening of G_agg at r_addr ───────────────────────────
    let point_g_kzg: Vec<Challenge> = r_addr_ch.iter().rev().cloned().collect();
    if *comm_g != zero_comm {
        match opening_g {
            Some(pf) => {
                if HyperKZG::<Bn254>::verify(
                    vk,
                    comm_g,
                    &point_g_kzg,
                    final_g_eval,
                    pf,
                    transcript,
                )
                .is_err()
                {
                    eprintln!("verify_shout_lut: G_agg HyperKZG verify FAILED");
                    return false;
                }
            }
            None => {
                eprintln!("verify_shout_lut: non-zero comm_g but no opening proof");
                return false;
            }
        }
    }

    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{One, Zero};
    use jolt_core::poly::commitment::dory::DoryGlobals;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a 2-input AND LutDesc.
    /// truth table (LSB-first index): AND[0b00]=0, AND[0b01]=0, AND[0b10]=0, AND[0b11]=1
    /// packed byte: bit 3 = 1 → 0b0000_1000 = 0x08
    fn and_desc() -> LutDesc {
        LutDesc { lut_id: 0, k: 2, m: 1, truth_table: vec![0x08] }
    }

    /// Build a 2-input XOR LutDesc.
    /// XOR[0b00]=0, XOR[0b01]=1, XOR[0b10]=1, XOR[0b11]=0
    /// packed byte: bits 1,2 = 1 → 0b0000_0110 = 0x06
    fn xor_desc() -> LutDesc {
        LutDesc { lut_id: 1, k: 2, m: 1, truth_table: vec![0x06] }
    }

    /// Build a 2-input, 2-output LutDesc where out0 = AND(a,b), out1 = OR(a,b).
    /// Entry layout (m=2 output bits per entry, LSB = out0):
    ///   idx=0 (0,0): out0=0, out1=0 → bits: 0b00
    ///   idx=1 (1,0): out0=0, out1=1 → bits: 0b10
    ///   idx=2 (0,1): out0=0, out1=1 → bits: 0b10
    ///   idx=3 (1,1): out0=1, out1=1 → bits: 0b11
    /// Packed into bytes (8 values × 2 bits = 8 bits = 1 byte), bit-position = idx*2+out:
    ///   bit 0 (idx=0,out=0) = AND(0,0) = 0
    ///   bit 1 (idx=0,out=1) = OR(0,0)  = 0
    ///   bit 2 (idx=1,out=0) = AND(1,0) = 0
    ///   bit 3 (idx=1,out=1) = OR(1,0)  = 1
    ///   bit 4 (idx=2,out=0) = AND(0,1) = 0
    ///   bit 5 (idx=2,out=1) = OR(0,1)  = 1
    ///   bit 6 (idx=3,out=0) = AND(1,1) = 1
    ///   bit 7 (idx=3,out=1) = OR(1,1)  = 1
    /// → byte = 0b1110_1000 = 0xE8
    fn and_or_desc() -> LutDesc {
        LutDesc { lut_id: 2, k: 2, m: 2, truth_table: vec![0xE8] }
    }

    fn fr(v: u64) -> Fr { Fr::from_u64(v) }

    // ── LutShoutTable construction ─────────────────────────────────────────────

    #[test]
    fn materialize_and() {
        let t = LutShoutTable::from_lut_desc(&and_desc(), 0);
        assert_eq!(t.table_size(), 4);
        assert_eq!(t.materialize_entry(0), 0); // AND(0,0)=0
        assert_eq!(t.materialize_entry(1), 0); // AND(1,0)=0
        assert_eq!(t.materialize_entry(2), 0); // AND(0,1)=0
        assert_eq!(t.materialize_entry(3), 1); // AND(1,1)=1
    }

    #[test]
    fn materialize_xor() {
        let t = LutShoutTable::from_lut_desc(&xor_desc(), 0);
        assert_eq!(t.materialize_entry(0), 0); // XOR(0,0)=0
        assert_eq!(t.materialize_entry(1), 1); // XOR(1,0)=1
        assert_eq!(t.materialize_entry(2), 1); // XOR(0,1)=1
        assert_eq!(t.materialize_entry(3), 0); // XOR(1,1)=0
    }

    #[test]
    fn materialize_two_output_lut() {
        let t0 = LutShoutTable::from_lut_desc(&and_or_desc(), 0); // AND column
        let t1 = LutShoutTable::from_lut_desc(&and_or_desc(), 1); // OR  column

        // AND column: [0, 0, 0, 1]
        assert_eq!(t0.materialize_entry(0), 0);
        assert_eq!(t0.materialize_entry(1), 0);
        assert_eq!(t0.materialize_entry(2), 0);
        assert_eq!(t0.materialize_entry(3), 1);

        // OR column: [0, 1, 1, 1]
        assert_eq!(t1.materialize_entry(0), 0);
        assert_eq!(t1.materialize_entry(1), 1);
        assert_eq!(t1.materialize_entry(2), 1);
        assert_eq!(t1.materialize_entry(3), 1);
    }

    // ── MLE correctness on Boolean hypercube ──────────────────────────────────

    /// On Boolean inputs {0,1}^k, the MLE should reproduce the truth table.
    fn mle_on_hypercube(t: &LutShoutTable) {
        for idx in 0usize..(1 << t.k) {
            let r: Vec<Fr> = (0..t.k)
                .map(|i| if (idx >> i) & 1 == 1 { Fr::one() } else { Fr::zero() })
                .collect();
            let got: Fr = t.evaluate_mle(&r);
            let expected = fr(t.materialize_entry(idx as u128));
            assert_eq!(
                got, expected,
                "MLE mismatch at idx={idx}: got={got:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn mle_and_on_hypercube() {
        mle_on_hypercube(&LutShoutTable::from_lut_desc(&and_desc(), 0));
    }

    #[test]
    fn mle_xor_on_hypercube() {
        mle_on_hypercube(&LutShoutTable::from_lut_desc(&xor_desc(), 0));
    }

    #[test]
    fn mle_and_or_out0_on_hypercube() {
        mle_on_hypercube(&LutShoutTable::from_lut_desc(&and_or_desc(), 0));
    }

    #[test]
    fn mle_and_or_out1_on_hypercube() {
        mle_on_hypercube(&LutShoutTable::from_lut_desc(&and_or_desc(), 1));
    }

    // ── MLE linearity check ───────────────────────────────────────────────────

    /// MLE is multilinear: T̃(λ·a + (1-λ)·b) for λ from the field traces
    /// a straight line between T̃(a) and T̃(b).
    #[test]
    fn mle_linearity_and() {
        let t = LutShoutTable::from_lut_desc(&and_desc(), 0);
        // Fix r[1] = 1/2, vary r[0]
        let half = Fr::from_u64(2).inverse().unwrap();

        let r0 = vec![Fr::zero(), half];
        let r1 = vec![Fr::one(), half];
        let r_half = vec![half, half];

        let v0: Fr = t.evaluate_mle(&r0);
        let v1: Fr = t.evaluate_mle(&r1);
        let v_half: Fr = t.evaluate_mle(&r_half);

        let expected = (v0 + v1) * half;
        assert_eq!(v_half, expected, "MLE linearity check failed");
    }

    // ── lut_desc_to_sub_circuit ───────────────────────────────────────────────

    #[test]
    fn sub_circuit_matches_shout_table() {
        let desc = xor_desc();
        let sub = lut_desc_to_sub_circuit(&desc);
        let shout = LutShoutTable::from_lut_desc(&desc, 0);
        for idx in 0..4 {
            assert_eq!(
                sub.table[idx] & 1, shout.table[idx] & 1,
                "SubCircuitLut vs LutShoutTable mismatch at idx={idx}"
            );
        }
    }

    // ── alpha batching ────────────────────────────────────────────────────────

    #[test]
    fn alpha_batch_single_output_is_identity() {
        let desc = and_desc();
        let alpha = fr(5); // arbitrary, doesn't matter for m=1
        let batched = alpha_batch_table_entries(&desc, alpha);
        // With m=1, batched[idx] = α^0 * T_0[idx] = T_0[idx]
        for idx in 0..4 {
            let expected = fr((desc.truth_table[0] >> (idx * desc.m)) as u64 & 1);
            assert_eq!(batched[idx], expected, "alpha-batch identity failed at idx={idx}");
        }
    }

    #[test]
    fn alpha_batch_two_outputs() {
        let desc = and_or_desc();
        let alpha = fr(3);
        let batched = alpha_batch_table_entries(&desc, alpha);

        // AND column: [0, 0, 0, 1]
        // OR column:  [0, 1, 1, 1]
        // batched = 1 * AND + alpha * OR
        let expected = [
            fr(0) + alpha * fr(0), // idx=0: AND=0, OR=0
            fr(0) + alpha * fr(1), // idx=1: AND=0, OR=1
            fr(0) + alpha * fr(1), // idx=2: AND=0, OR=1
            fr(1) + alpha * fr(1), // idx=3: AND=1, OR=1
        ];
        for idx in 0..4 {
            assert_eq!(batched[idx], expected[idx], "alpha-batch two-output mismatch at idx={idx}");
        }
    }

    #[test]
    fn evaluate_batched_mle_on_hypercube() {
        let desc = and_or_desc();
        let alpha = fr(7);
        let batched = alpha_batch_table_entries(&desc, alpha);

        for idx in 0usize..4 {
            let r: Vec<Fr> = (0..2)
                .map(|i| if (idx >> i) & 1 == 1 { Fr::one() } else { Fr::zero() })
                .collect();
            let got = evaluate_batched_table_mle(&batched, &r);
            assert_eq!(
                got, batched[idx],
                "batched MLE mismatch at hypercube point idx={idx}"
            );
        }
    }

    // ── address encoding ──────────────────────────────────────────────────────

    #[test]
    fn shout_address_encoding() {
        // type_index=2, k=3, inputs=(1,0,1) → packed=0b101=5
        // address = 2*8 + 5 = 21
        let packed = pack_inputs(&[true, false, true]);
        assert_eq!(packed, 0b101);
        let addr = shout_address(2, 3, packed);
        assert_eq!(addr, 2 * 8 + 5);
    }

    #[test]
    fn mega_table_address_bits_computation() {
        // T=4 types (t_pad=4), k=3 → 2 type bits + 3 input bits = 5
        assert_eq!(mega_table_address_bits(4, 3), 5);
        // T=1 types (t_pad=1), k=4 → 0 type bits + 4 input bits = 4
        assert_eq!(mega_table_address_bits(1, 4), 4);
        // T=8 types (t_pad=8), k=6 → 3 type bits + 6 = 9
        assert_eq!(mega_table_address_bits(8, 6), 9);
        // T=5 types → t_pad=8, k=4 → 3+4=7
        assert_eq!(mega_table_address_bits(8, 4), 7);
    }

    // ── OneHotParams ───────────────────────────────────────────────────────

    #[test]
    fn one_hot_params_single_chunk() {
        // n_types=2 (t_pad=2), k=2, log_k_chunk=4:
        //   type_bits = ceil(log2(2)) = 1
        //   total_address_bits = 1 + 2 = 3
        //   d = ceil(3/4) = 1  (one chunk covers all 3 bits)
        //   k_chunk = 16
        let p = OneHotParams::new(2, 2, 4);
        assert_eq!(p.log_k_chunk, 4);
        assert_eq!(p.total_address_bits, 3);
        assert_eq!(p.d, 1);
        assert_eq!(p.k_chunk, 16);
    }

    #[test]
    fn one_hot_params_multiple_chunks() {
        // n_types=8 (t_pad=8), k=6, log_k_chunk=4:
        //   type_bits = 3, total_address_bits = 9
        //   d = ceil(9/4) = 3
        //   k_chunk = 16
        let p = OneHotParams::new(8, 6, 4);
        assert_eq!(p.total_address_bits, 9);
        assert_eq!(p.d, 3);
        assert_eq!(p.k_chunk, 16);
    }

    #[test]
    fn address_chunk_extraction() {
        // address = 0b_1011_0101 = 181, log_k_chunk = 4
        let addr = 0b_1011_0101_usize;
        assert_eq!(address_chunk(addr, 0, 4), 0b0101_u8); // bits [0..4)
        assert_eq!(address_chunk(addr, 1, 4), 0b1011_u8); // bits [4..8)

        // address = 0b_001_100_011 = 0x063 = 99, log_k_chunk = 3
        let addr2 = 0b_001_100_011_usize;
        assert_eq!(address_chunk(addr2, 0, 3), 0b011_u8); // bits [0..3)
        assert_eq!(address_chunk(addr2, 1, 3), 0b100_u8); // bits [3..6)
        assert_eq!(address_chunk(addr2, 2, 3), 0b001_u8); // bits [6..9)
    }

    // ── build_shout_witnesses ────────────────────────────────────────────

    /// Build a minimal type_index_of map for basic tests:
    /// AND  → lut_id=0 → type_index=0
    /// XOR  → lut_id=1 → type_index=1
    fn basic_type_map() -> std::collections::HashMap<u32, usize> {
        [(0, 0), (1, 1)].into_iter().collect()
    }

    /// Build a two-row trace: AND(1,1) then XOR(1,0).
    /// Inputs stored as Vec<bool> in LSB-first order.
    fn basic_trace() -> Vec<LutEval> {
        vec![
            LutEval {
                lut_id: 0,
                inputs: vec![true, true],   // AND(1,1)=1
                outputs: vec![true],
            },
            LutEval {
                lut_id: 1,
                inputs: vec![true, false],  // XOR(1,0)=1
                outputs: vec![true],
            },
        ]
    }

    #[test]
    #[serial_test::serial]
    fn single_chunk_nonzero_indices() {
        // Setup: n_types=2, k=2, log_k_chunk=4, t_total=16
        // type AND (id=0, idx=0), k=2:
        //   addr(AND, 1,1) = 0*4 + packed(1,1) = 3  → chunk0 = 3
        // type XOR (id=1, idx=1), k=2:
        //   addr(XOR, 1,0) = 1*4 + packed(1,0) = 5  → chunk0 = 5
        // t_total=16 → indices[0]=Some(3), indices[1]=Some(5), rest=None
        let k_chunk = 16usize;
        let t_total = 16usize;
        let _init = DoryGlobals::initialize(k_chunk, t_total);

        let params = OneHotParams::new(2, 2, 4);
        assert_eq!(params.d, 1, "expected 1 chunk for 3-bit address with log_k_chunk=4");

        let trace = basic_trace();
        let type_map = basic_type_map();
        let witnesses = build_shout_witnesses(&trace, &type_map, 2, &params, t_total);

        assert_eq!(witnesses.len(), 1, "expected 1 OneHotPolynomial");
        let ra = &witnesses[0];
        assert_eq!(ra.K, k_chunk);
        assert_eq!(ra.nonzero_indices.len(), t_total);

        // Row 0: AND(1,1) → type_index=0, packed=(1,1)=3, addr=0*4+3=3, chunk0=3
        assert_eq!(ra.nonzero_indices[0], Some(3u8));
        // Row 1: XOR(1,0) → type_index=1, packed=(1,0)=1, addr=1*4+1=5, chunk0=5
        assert_eq!(ra.nonzero_indices[1], Some(5u8));
        // Rows 2..t_total: padding → None
        for j in 2..t_total {
            assert_eq!(ra.nonzero_indices[j], None, "padded row {j} should be None");
        }
    }

    #[test]
    #[serial_test::serial]
    fn multiple_chunks_correct_extraction() {
        // n_types=8 (t_pad=8), k=6, log_k_chunk=4:
        //   total_address_bits = 3+6 = 9, d = 3 chunks
        //   k_chunk = 16, t_total = 16
        // Build a single trace row: type_index=5, packed_input=0b101010=42
        //   address = 5 * 64 + 42 = 362 = 0b_1_0110_1010
        //   chunk0 → bits [0..4)  = 0b1010 = 10
        //   chunk1 → bits [4..8)  = 0b0110 = 6
        //   chunk2 → bits [8..12) = 0b0001 = 1
        let k_chunk = 16usize;
        let t_total = 16usize;
        let _init = DoryGlobals::initialize(k_chunk, t_total);

        let n_types = 8usize;
        let k = 6usize;
        let params = OneHotParams::new(n_types, k, 4);
        assert_eq!(params.d, 3);
        assert_eq!(params.k_chunk, 16);

        // Manufacture a trace row for type_index=5, packed_input=0b101010=42
        // lut_id=5 (using type_map: lut_id=i → type_index=i for i in 0..8)
        let trace = vec![LutEval {
            lut_id: 5,
            inputs: vec![false, true, false, true, false, true], // LSB-first: 0b101010=42
            outputs: vec![false],
        }];
        let type_map: std::collections::HashMap<u32, usize> =
            (0u32..8).map(|i| (i, i as usize)).collect();

        let witnesses = build_shout_witnesses(&trace, &type_map, k, &params, t_total);
        assert_eq!(witnesses.len(), 3);

        let expected_addr = 5 * (1 << k) + 42; // = 5*64+42 = 362
        assert_eq!(expected_addr, 362);

        // Verify each chunk
        assert_eq!(witnesses[0].nonzero_indices[0], Some(address_chunk(362, 0, 4))); // =10
        assert_eq!(witnesses[1].nonzero_indices[0], Some(address_chunk(362, 1, 4))); // =6
        assert_eq!(witnesses[2].nonzero_indices[0], Some(address_chunk(362, 2, 4))); // =1

        // Explicit values
        assert_eq!(witnesses[0].nonzero_indices[0], Some(10u8)); // 0b1010
        assert_eq!(witnesses[1].nonzero_indices[0], Some(6u8));  // 0b0110
        assert_eq!(witnesses[2].nonzero_indices[0], Some(1u8));  // 0b0001

        // Padded rows
        for j in 1..t_total {
            assert_eq!(witnesses[0].nonzero_indices[j], None);
        }
    }

    #[test]
    #[serial_test::serial]
    fn empty_trace_is_all_none() {
        // An empty trace with t_total=16 should produce d polys with all-None indices.
        let k_chunk = 16usize;
        let t_total = 16usize;
        let _init = DoryGlobals::initialize(k_chunk, t_total);

        let params = OneHotParams::new(2, 2, 4);
        let witnesses = build_shout_witnesses(&[], &std::collections::HashMap::new(), 2, &params, t_total);
        assert_eq!(witnesses.len(), 1);
        assert!(witnesses[0].nonzero_indices.iter().all(|x| x.is_none()));
    }

    #[test]
    fn address_chunk_round_trip() {
        // For a range of addresses and chunk configs, verify round-trip:
        // reconstruct address from chunks → matches original.
        for log_k_chunk in 1usize..=6 {
            let _mask = (1usize << log_k_chunk) - 1;
            for addr in 0usize..256 {
                let bits_needed = if addr == 0 { 1 } else { usize::BITS as usize - addr.leading_zeros() as usize };
                let d = bits_needed.div_ceil(log_k_chunk);
                let mut reconstructed = 0usize;
                for i in 0..d {
                    let chunk = address_chunk(addr, i, log_k_chunk) as usize;
                    reconstructed |= chunk << (i * log_k_chunk);
                }
                assert_eq!(
                    reconstructed & mask_usize(bits_needed),
                    addr & mask_usize(bits_needed),
                    "round-trip failed for addr={addr} log_k_chunk={log_k_chunk}"
                );
            }
        }
    }

    // Helper: mask for n bits.
    fn mask_usize(n: usize) -> usize {
        if n >= usize::BITS as usize { usize::MAX } else { (1 << n) - 1 }
    }

    // ── Phase S3 / S4 tests ───────────────────────────────────────────────────

    use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
    use jolt_core::poly::commitment::hyperkzg::HyperKZG;
    use jolt_core::transcripts::KeccakTranscript;
    use crate::lut_czbc::{LutCirc, LutOp};
    use ark_bn254::Bn254;
    use std::collections::HashMap;

    type PCS = HyperKZG<Bn254>;

    /// Build a minimal two-type circuit: AND (lut_id=0) and XOR (lut_id=1).
    fn two_type_circ() -> LutCirc {
        let mut lut_types = HashMap::new();
        lut_types.insert(0u32, and_desc());
        lut_types.insert(1u32, xor_desc());
        LutCirc {
            num_wires: 5,
            primary_inputs: vec![0, 1],
            registers: vec![],
            outputs: vec![3, 4],
            lut_types,
            ops: vec![
                LutOp { lut_id: 0, dst_wire: 2, src_wires: vec![0, 1] }, // AND
                LutOp { lut_id: 1, dst_wire: 3, src_wires: vec![0, 1] }, // XOR
                LutOp { lut_id: 0, dst_wire: 4, src_wires: vec![0, 1] }, // AND again
            ],
            default_cycles: 1,
        }
    }

    /// Build a small synthetic trace for `two_type_circ()`.
    /// Three rows: AND(1,1), XOR(1,0), AND(0,1).
    fn two_type_trace() -> Vec<LutEval> {
        vec![
            LutEval { lut_id: 0, inputs: vec![true, true],  outputs: vec![true]  }, // AND(1,1)=1
            LutEval { lut_id: 1, inputs: vec![true, false], outputs: vec![true]  }, // XOR(1,0)=1
            LutEval { lut_id: 0, inputs: vec![false, true], outputs: vec![false] }, // AND(0,1)=0
        ]
    }

    /// type_index_of map for two_type_circ: lut_id 0 → tid 0, lut_id 1 → tid 1.
    fn two_type_index_of() -> HashMap<u32, usize> {
        [(0, 0), (1, 1)].into_iter().collect()
    }

    // ── build_mega_table_batched ─────────────────────────────────────────────

    #[test]
    fn mega_table_single_type_alpha_1() {
        // Single AND LUT, alpha = 1.  Batched table = AND truth table.
        let mut lut_types = HashMap::new();
        lut_types.insert(0u32, and_desc());
        let circ = LutCirc {
            num_wires: 3, primary_inputs: vec![0, 1], registers: vec![],
            outputs: vec![2], lut_types,
            ops: vec![LutOp { lut_id: 0, dst_wire: 2, src_wires: vec![0, 1] }],
            default_cycles: 1,
        };
        let type_order = vec![0u32];
        let alpha = Fr::one();
        let mega = build_mega_table_batched(&circ, &type_order, 2, alpha);
        // t_pad=1, table_size=4, mega_size=4
        assert_eq!(mega.len(), 4);
        // AND truth table (LSB-first): [0, 0, 0, 1]
        assert_eq!(mega[0], Fr::zero()); // AND(0,0)=0
        assert_eq!(mega[1], Fr::zero()); // AND(1,0)=0
        assert_eq!(mega[2], Fr::zero()); // AND(0,1)=0
        assert_eq!(mega[3], Fr::one());  // AND(1,1)=1
    }

    #[test]
    fn mega_table_two_types() {
        let circ = two_type_circ();
        let type_order = vec![0u32, 1u32];
        let alpha = fr(2); // alpha=2 for multi-output batching (m=1 so just identity)
        let mega = build_mega_table_batched(&circ, &type_order, 2, alpha);
        // t_pad=2, table_size=4, mega_size=8
        assert_eq!(mega.len(), 8);
        // AND entries at [0..4]: [0, 0, 0, 1]
        assert_eq!(mega[0], Fr::zero()); assert_eq!(mega[3], Fr::one());
        // XOR entries at [4..8]: [0, 1, 1, 0]
        assert_eq!(mega[4], Fr::zero()); assert_eq!(mega[5], Fr::one());
        assert_eq!(mega[6], Fr::one());  assert_eq!(mega[7], Fr::zero());
    }

    // ── mle_eval_fr ─────────────────────────────────────────────────────────

    #[test]
    fn mle_eval_fr_on_hypercube() {
        // For a 2-entry vector [a, b], MLE at (0)=a and (1)=b.
        let v = vec![fr(3), fr(7)];
        assert_eq!(mle_eval_fr(&v, &[Fr::zero()]), fr(3));
        assert_eq!(mle_eval_fr(&v, &[Fr::one()]),  fr(7));
    }

    #[test]
    fn mle_eval_fr_linear() {
        // MLE of [0, 1]: equals the identity polynomial.
        let v = vec![Fr::zero(), Fr::one()];
        let half = Fr::from(2u64).inverse().unwrap();
        assert_eq!(mle_eval_fr(&v, &[half]), half);
    }

    // ── End-to-end prover + verifier ─────────────────────────────────────────

    fn make_shout_srs(circ: &LutCirc, trace_len: usize) -> (
        jolt_core::poly::commitment::hyperkzg::HyperKZGProverKey<Bn254>,
        jolt_core::poly::commitment::hyperkzg::HyperKZGVerifierKey<Bn254>,
    ) {
        let k = circ.lut_types.values().map(|d| d.k).max().unwrap_or(0);
        let n_types = circ.lut_types.len();
        let max_vars = shout_max_num_vars(n_types, k, 1, trace_len);
        let pk = PCS::setup_prover(max_vars.max(1));
        let vk = PCS::setup_verifier(&pk);
        (pk, vk)
    }

    #[test]
    fn shout_prove_verify_two_type() {
        let circ = two_type_circ();
        let trace = two_type_trace();
        let type_index_of = two_type_index_of();
        let k = 2usize;
        let t_total = trace.len().next_power_of_two().max(1); // = 4

        let (pk, vk) = make_shout_srs(&circ, trace.len());

        // Prove.
        let mut pt = KeccakTranscript::new(b"shout-lut-test");
        let proof = prove_shout_lut(&circ, &trace, &type_index_of, k, t_total, &pk, &mut pt);

        // Verify with a fresh transcript (same initialization).
        let mut vt = KeccakTranscript::new(b"shout-lut-test");
        let ok = verify_shout_lut(&proof, &circ, &vk, &mut vt);
        assert!(ok, "shout prove+verify should succeed for valid two-type trace");
    }

    #[test]
    fn shout_prove_verify_single_type() {
        // Single AND LUT, 2-row trace: AND(1,1), AND(0,1).
        let mut lut_types = HashMap::new();
        lut_types.insert(0u32, and_desc());
        let circ = LutCirc {
            num_wires: 3, primary_inputs: vec![0, 1], registers: vec![],
            outputs: vec![2], lut_types,
            ops: vec![LutOp { lut_id: 0, dst_wire: 2, src_wires: vec![0, 1] }],
            default_cycles: 2,
        };
        let trace = vec![
            LutEval { lut_id: 0, inputs: vec![true, true],  outputs: vec![true]  },
            LutEval { lut_id: 0, inputs: vec![false, true], outputs: vec![false] },
        ];
        let type_index_of: HashMap<u32, usize> = [(0, 0)].into_iter().collect();
        let k = 2usize;
        let t_total = trace.len().next_power_of_two().max(1); // = 2

        let (pk, vk) = make_shout_srs(&circ, trace.len());

        let mut pt = KeccakTranscript::new(b"shout-single");
        let proof = prove_shout_lut(&circ, &trace, &type_index_of, k, t_total, &pk, &mut pt);

        let mut vt = KeccakTranscript::new(b"shout-single");
        assert!(verify_shout_lut(&proof, &circ, &vk, &mut vt),
            "single-type shout proof should verify");
    }

    #[test]
    fn shout_tampered_cycle_poly_fails() {
        let circ = two_type_circ();
        let trace = two_type_trace();
        let type_index_of = two_type_index_of();
        let k = 2usize;
        let t_total = trace.len().next_power_of_two().max(1);

        let (pk, vk) = make_shout_srs(&circ, trace.len());

        let mut pt = KeccakTranscript::new(b"shout-tamper-cy");
        let mut proof = prove_shout_lut(&circ, &trace, &type_index_of, k, t_total, &pk, &mut pt);

        // Corrupt the first cycle-sumcheck round polynomial.
        if let Some(p) = proof.cycle_sc_polys.first_mut() {
            p[0] += Fr::one(); // breaks p(0) + p(1) = prev_claim
        }

        let mut vt = KeccakTranscript::new(b"shout-tamper-cy");
        assert!(!verify_shout_lut(&proof, &circ, &vk, &mut vt),
            "tampered cycle sumcheck should fail verification");
    }

    #[test]
    fn shout_tampered_addr_poly_fails() {
        let circ = two_type_circ();
        let trace = two_type_trace();
        let type_index_of = two_type_index_of();
        let k = 2usize;
        let t_total = trace.len().next_power_of_two().max(1);

        let (pk, vk) = make_shout_srs(&circ, trace.len());

        let mut pt = KeccakTranscript::new(b"shout-tamper-ad");
        let mut proof = prove_shout_lut(&circ, &trace, &type_index_of, k, t_total, &pk, &mut pt);

        // Corrupt the first address-sumcheck round polynomial.
        if let Some(p) = proof.addr_sc_polys.first_mut() {
            p[1] += Fr::one();
        }

        let mut vt = KeccakTranscript::new(b"shout-tamper-ad");
        assert!(!verify_shout_lut(&proof, &circ, &vk, &mut vt),
            "tampered addr sumcheck should fail verification");
    }
}
