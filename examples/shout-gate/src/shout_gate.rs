//! Shout lookup argument specialised to 2-input boolean gates with 4-bit masks.

use std::collections::HashMap;
use std::time::Instant;

use ark_bn254::{Bn254, Fr};
use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;

use jolt_core::field::JoltField;
use jolt_core::poly::commitment::hyperkzg::{
    HyperKZG, HyperKZGCommitment, HyperKZGProof, HyperKZGProverKey, HyperKZGVerifierKey,
};
use jolt_core::poly::multilinear_polynomial::MultilinearPolynomial;
use jolt_core::poly::one_hot_polynomial::OneHotPolynomial;
use jolt_core::transcripts::{AppendToTranscript, KeccakTranscript, Transcript};

use crate::czbc::GateEval;

/// Type alias for the PCS we use throughout.
type PCS = HyperKZG<Bn254>;
type Challenge = <Fr as JoltField>::Challenge;

#[inline]
fn bind_poly(poly: &mut Vec<Fr>, r: Fr) {
    let half = poly.len() / 2;
    for i in 0..half {
        poly[i] = poly[2 * i] + r * (poly[2 * i + 1] - poly[2 * i]);
    }
    poly.truncate(half);
}

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

fn eq_final_eval(r: &[Fr], x: &[Fr]) -> Fr {
    assert_eq!(r.len(), x.len());
    r.iter()
        .zip(x.iter())
        .map(|(&ri, &xi)| ri * xi + (Fr::one() - ri) * (Fr::one() - xi))
        .product()
}

#[inline]
fn lagrange_deg2(poly: &[Fr; 3], r: Fr) -> Fr {
    let [e0, e1, e2] = *poly;
    let two = Fr::from(2u64);
    let half = two.inverse().expect("2 is invertible in BN254 Fr");
    // L_0(t) = (t-1)(t-2)/2,  L_1(t) = -t(t-2),  L_2(t) = t(t-1)/2
    let l0 = (r - Fr::one()) * (r - two) * half;
    let l1 = -r * (r - two);
    let l2 = r * (r - Fr::one()) * half;
    e0 * l0 + e1 * l1 + e2 * l2
}

/// Total address bits for a mega-table with `t_pad` types and `k` input bits.
#[inline]
pub fn mega_table_address_bits(t_pad: usize, k: usize) -> usize {
    let type_bits = if t_pad <= 1 {
        0
    } else {
        t_pad.next_power_of_two().trailing_zeros() as usize
    };
    type_bits + k
}

pub fn mle_eval_fr(entries: &[Fr], point: &[Fr]) -> Fr {
    debug_assert_eq!(entries.len(), 1usize << point.len());
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

pub fn build_mega_table(type_order: &[u8], k: usize) -> Vec<Fr> {
    assert!(k >= 2, "build_mega_table: expected k>=2 for 2-input gates");
    let n_types = type_order.len();
    let t_pad = n_types.next_power_of_two().max(1);
    let table_size = 1usize << k;
    let mega_size = t_pad * table_size;

    let mut mega = vec![Fr::zero(); mega_size];
    let reps = 1usize << (k - 2);
    for (tid, &mask) in type_order.iter().enumerate() {
        for rep in 0..reps {
            for idx in 0usize..4 {
                let bit = (mask >> idx) & 1;
                let mega_idx = tid * table_size + rep * 4 + idx;
                mega[mega_idx] = Fr::from_u64(bit as u64);
            }
        }
    }
    mega
}

#[inline]
fn eq_eval_bool_point(point: &[Fr], idx: usize) -> Fr {
    point
        .iter()
        .enumerate()
        .map(|(bit, &r)| {
            if ((idx >> bit) & 1) == 1 {
                r
            } else {
                Fr::one() - r
            }
        })
        .product()
}

pub fn gate_mask_mle(mask: u8, input_point: &[Fr]) -> Fr {
    debug_assert_eq!(
        input_point.len(),
        2,
        "gate_mask_mle: expected 2 input variables"
    );
    (0usize..4)
        .map(|idx| Fr::from_bool(((mask >> idx) & 1) == 1) * eq_eval_bool_point(input_point, idx))
        .sum()
}

pub fn mega_table_mle_symbolic(type_order: &[u8], k: usize, point: &[Fr]) -> Fr {
    assert_eq!(
        k, 2,
        "mega_table_mle_symbolic: expected k=2 for raw gate lookups"
    );
    let t_pad = type_order.len().next_power_of_two().max(1);
    let type_bits = mega_table_address_bits(t_pad, k) - k;
    assert_eq!(
        point.len(),
        type_bits + k,
        "mega_table_mle_symbolic: point length mismatch"
    );

    let (input_point, type_point) = point.split_at(k);
    (0usize..t_pad)
        .map(|tid| {
            let table_val = type_order
                .get(tid)
                .map(|&mask| gate_mask_mle(mask, input_point))
                .unwrap_or_else(Fr::zero);
            eq_eval_bool_point(type_point, tid) * table_val
        })
        .sum()
}

pub fn shout_max_num_vars(n_types: usize, k: usize, cycles: u32, n_ops: usize) -> usize {
    let n_total = n_ops * cycles.max(1) as usize;
    let t_total = n_total.next_power_of_two().max(1);
    let log_t = t_total.trailing_zeros() as usize;

    let t_pad = n_types.next_power_of_two().max(1);
    let total_address_bits = mega_table_address_bits(t_pad, k);

    log_t.max(total_address_bits).max(1)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateWitnessParams {
    pub log_k_chunk: usize,
    pub d: usize,
    pub total_address_bits: usize,
    pub k_chunk: usize,
}

impl GateWitnessParams {
    pub fn new(n_types: usize, k: usize, log_k_chunk: usize) -> Self {
        assert!(
            (1..=8).contains(&log_k_chunk),
            "log_k_chunk must be 1..=8, got {log_k_chunk}"
        );
        let t_pad = n_types.next_power_of_two().max(1);
        let total_address_bits = mega_table_address_bits(t_pad, k);
        let d = total_address_bits.div_ceil(log_k_chunk);
        let k_chunk = 1usize << log_k_chunk;
        Self {
            log_k_chunk,
            d,
            total_address_bits,
            k_chunk,
        }
    }
}

#[inline]
pub fn address_chunk(address: usize, chunk_idx: usize, log_k_chunk: usize) -> u8 {
    let shift = chunk_idx * log_k_chunk;
    let mask = (1usize << log_k_chunk) - 1;
    ((address >> shift) & mask) as u8
}

pub fn build_gate_shout_witnesses(
    trace: &[GateEval],
    type_index_of: &HashMap<u8, usize>,
    k: usize,
    params: &GateWitnessParams,
    t_total: usize,
) -> Vec<OneHotPolynomial<Fr>> {
    assert!(
        t_total >= trace.len(),
        "t_total ({t_total}) must be >= trace.len() ({})",
        trace.len()
    );
    assert!(
        t_total == 0 || t_total.is_power_of_two(),
        "t_total must be a power of two, got {t_total}"
    );

    let addresses: Vec<usize> = trace
        .iter()
        .map(|ev| {
            let tid = *type_index_of.get(&ev.mask).unwrap_or_else(|| {
                panic!("build_gate_shout_witnesses: unknown mask=0x{:02X}", ev.mask)
            });
            ev.address_for_shout(tid, k)
        })
        .collect();

    (0..params.d)
        .map(|chunk_i| {
            let indices: Vec<Option<u8>> = (0..t_total)
                .map(|j| {
                    if j < addresses.len() {
                        Some(address_chunk(addresses[j], chunk_i, params.log_k_chunk))
                    } else {
                        None
                    }
                })
                .collect();
            OneHotPolynomial::from_indices(indices, params.k_chunk)
        })
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct ShoutGateProverStats {
    pub dense_table_build_ms: u128,
    pub batch_val_build_ms: u128,
    pub comm_bv_ms: u128,
    pub cycle_sumcheck_ms: u128,
    pub g_agg_build_ms: u128,
    pub comm_g_ms: u128,
    pub address_sumcheck_ms: u128,
    pub opening_generation_ms: u128,
    pub t_pad: usize,
    pub mega_table_len: usize,
    pub batch_val_len: usize,
    pub batch_val_compact_bytes: usize,
    pub g_agg_len: usize,
    pub selector_vec_len: usize,
    pub peak_dense_elems_estimate: usize,
}

#[derive(Clone, Debug)]
pub struct ShoutGateProverOutput {
    pub proof: ShoutGateProof,
    pub stats: ShoutGateProverStats,
}

#[inline]
fn update_peak_dense_elems(peak: &mut usize, live_vec_lens: &[usize]) {
    *peak = (*peak).max(live_vec_lens.iter().sum());
}

#[derive(Clone, Debug)]
pub struct ShoutGateProof {
    pub n_types: usize,
    pub k: usize,
    pub t_total: usize,

    pub alpha: Fr,

    pub comm_bv: HyperKZGCommitment<Bn254>,
    pub cycle_sc_polys: Vec<[Fr; 3]>,
    pub bv_eval: Fr,
    pub opening_bv: Option<HyperKZGProof<Bn254>>,

    pub comm_g: HyperKZGCommitment<Bn254>,
    pub addr_sc_polys: Vec<[Fr; 3]>,
    pub final_g_eval: Fr,
    pub final_table_eval: Fr,
    pub opening_g: Option<HyperKZGProof<Bn254>>,
}

pub fn prove_shout_gate(
    type_order: &[u8],
    trace: &[GateEval],
    type_index_of: &HashMap<u8, usize>,
    k: usize,
    t_total: usize,
    pk: &HyperKZGProverKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> ShoutGateProverOutput {
    assert_eq!(k, 2, "prove_shout_gate: expected k=2 for gate lookups");
    let log_t = (t_total.trailing_zeros() as usize).max(0);
    let n_types = type_order.len();
    let t_pad = n_types.next_power_of_two().max(1);
    let mega_size = t_pad * (1usize << k);
    let total_address_bits = mega_table_address_bits(t_pad, k);
    let mut stats = ShoutGateProverStats {
        t_pad,
        mega_table_len: mega_size,
        batch_val_len: t_total,
        batch_val_compact_bytes: t_total,
        g_agg_len: mega_size,
        selector_vec_len: mega_size,
        ..Default::default()
    };

    transcript.append_u64(n_types as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(t_total as u64);

    // Kept for transcript compatibility with the LUT variant (m=1, so unused).
    let alpha: Fr = transcript.challenge_scalar();

    let t_batch_val = Instant::now();
    let mut batch_val = vec![0u8; t_total];
    for (j, ev) in trace.iter().enumerate() {
        batch_val[j] = ev.out as u8;
    }
    stats.batch_val_build_ms = t_batch_val.elapsed().as_millis();

    let t_comm_bv = Instant::now();
    let mle_bv = MultilinearPolynomial::from(batch_val);
    let comm_bv = PCS::commit(pk, &mle_bv).expect("commit batch_val");
    stats.comm_bv_ms = t_comm_bv.elapsed().as_millis();
    comm_bv.append_to_transcript(transcript);

    let r_t_fr: Vec<Fr> = transcript.challenge_vector(log_t);

    let eq_t = init_eq_fr(&r_t_fr);
    let output_claim: Fr = mle_bv.dot_product(&eq_t);
    transcript.append_scalar(&output_claim);

    let mut eq_work = eq_t;
    let mut batch_val_dense = vec![Fr::zero(); t_total];
    for (j, ev) in trace.iter().enumerate() {
        batch_val_dense[j] = Fr::from_bool(ev.out);
    }
    update_peak_dense_elems(
        &mut stats.peak_dense_elems_estimate,
        &[batch_val_dense.len()],
    );
    let mut bv_work = batch_val_dense.clone();

    let mut cycle_sc_polys: Vec<[Fr; 3]> = Vec::with_capacity(log_t);
    let mut r_cycle_fr: Vec<Fr> = Vec::with_capacity(log_t);
    let mut r_cycle_ch: Vec<Challenge> = Vec::with_capacity(log_t);

    let t_cycle_sc = Instant::now();
    for _round in 0..log_t {
        let half = eq_work.len() / 2;
        let two = Fr::from(2u64);
        let mut p = [Fr::zero(); 3];
        for idx in 0..half {
            let (eq_lo, eq_hi) = (eq_work[2 * idx], eq_work[2 * idx + 1]);
            let (bv_lo, bv_hi) = (bv_work[2 * idx], bv_work[2 * idx + 1]);
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
    stats.cycle_sumcheck_ms = t_cycle_sc.elapsed().as_millis();

    let bv_eval = bv_work[0];
    transcript.append_scalar(&bv_eval);

    let point_bv_kzg: Vec<Challenge> = r_cycle_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();
    let t_openings = Instant::now();
    let opening_bv = if comm_bv != zero_comm {
        let mle_bv_dense = MultilinearPolynomial::from(batch_val_dense);
        Some(
            PCS::open(pk, &mle_bv_dense, &point_bv_kzg, &bv_eval, transcript)
                .expect("HyperKZG open batch_val"),
        )
    } else {
        None
    };

    let t_dense_table = Instant::now();
    let mega_table = build_mega_table(type_order, k);
    stats.dense_table_build_ms = t_dense_table.elapsed().as_millis();
    update_peak_dense_elems(&mut stats.peak_dense_elems_estimate, &[mega_table.len()]);

    let t_g_agg = Instant::now();
    let eq_cycle = init_eq_fr(&r_cycle_fr);
    let mut g_agg = vec![Fr::zero(); mega_size];
    for (j, ev) in trace.iter().enumerate() {
        let tid = *type_index_of.get(&ev.mask).unwrap();
        let addr = ev.address_for_shout(tid, k);
        g_agg[addr] += eq_cycle[j];
    }
    stats.g_agg_build_ms = t_g_agg.elapsed().as_millis();
    update_peak_dense_elems(
        &mut stats.peak_dense_elems_estimate,
        &[mega_table.len(), g_agg.len(), eq_cycle.len()],
    );

    let t_comm_g = Instant::now();
    let mle_g = MultilinearPolynomial::from(g_agg.clone());
    let comm_g = PCS::commit(pk, &mle_g).expect("commit G_agg");
    stats.comm_g_ms = t_comm_g.elapsed().as_millis();
    comm_g.append_to_transcript(transcript);

    let mut g_work = g_agg;
    let mut t_work = mega_table.clone();

    let mut addr_sc_polys: Vec<[Fr; 3]> = Vec::with_capacity(total_address_bits);
    let mut r_addr_fr: Vec<Fr> = Vec::with_capacity(total_address_bits);
    let mut r_addr_ch: Vec<Challenge> = Vec::with_capacity(total_address_bits);

    let t_addr_sc = Instant::now();
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
    stats.address_sumcheck_ms = t_addr_sc.elapsed().as_millis();
    update_peak_dense_elems(
        &mut stats.peak_dense_elems_estimate,
        &[g_work.len(), t_work.len(), mega_size],
    );

    let final_g_eval = g_work[0];
    let final_table_eval = t_work[0];
    transcript.append_scalar(&final_g_eval);
    transcript.append_scalar(&final_table_eval);

    let point_g_kzg: Vec<Challenge> = r_addr_ch.iter().rev().cloned().collect();
    let opening_g = if comm_g != zero_comm {
        Some(
            PCS::open(pk, &mle_g, &point_g_kzg, &final_g_eval, transcript)
                .expect("HyperKZG open G_agg"),
        )
    } else {
        None
    };
    stats.opening_generation_ms = t_openings.elapsed().as_millis();

    ShoutGateProverOutput {
        proof: ShoutGateProof {
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
        },
        stats,
    }
}

pub fn verify_shout_gate(
    proof: &ShoutGateProof,
    type_order: &[u8],
    vk: &HyperKZGVerifierKey<Bn254>,
    transcript: &mut KeccakTranscript,
) -> bool {
    let ShoutGateProof {
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

    if type_order.len() != n_types {
        eprintln!(
            "verify_shout_gate: type_order length mismatch: got {}, expected {}",
            type_order.len(),
            n_types
        );
        return false;
    }
    if k != 2 {
        eprintln!("verify_shout_gate: expected k=2, got {k}");
        return false;
    }

    transcript.append_u64(n_types as u64);
    transcript.append_u64(k as u64);
    transcript.append_u64(t_total as u64);

    let alpha_check: Fr = transcript.challenge_scalar();
    if alpha_check != *alpha {
        eprintln!("verify_shout_gate: alpha mismatch");
        return false;
    }

    comm_bv.append_to_transcript(transcript);
    let r_t_fr: Vec<Fr> = transcript.challenge_vector(log_t);

    let output_claim: Fr = if cycle_sc_polys.is_empty() {
        *bv_eval
    } else {
        cycle_sc_polys[0][0] + cycle_sc_polys[0][1]
    };
    transcript.append_scalar(&output_claim);

    let mut prev_claim = output_claim;
    let mut r_cycle_fr: Vec<Fr> = Vec::with_capacity(log_t);
    let mut r_cycle_ch: Vec<Challenge> = Vec::with_capacity(log_t);

    for (round, poly) in cycle_sc_polys.iter().enumerate() {
        let sum = poly[0] + poly[1];
        if sum != prev_claim {
            eprintln!(
                "verify_shout_gate: cycle sc round {round}: p(0)+p(1)={sum:?} ≠ {prev_claim:?}"
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

    let eq_t_at_rcycle = eq_final_eval(&r_t_fr, &r_cycle_fr);
    let expected_cycle_final = eq_t_at_rcycle * bv_eval;
    if expected_cycle_final != prev_claim {
        eprintln!(
            "verify_shout_gate: cycle sc final mismatch: eq·bv_eval={expected_cycle_final:?} ≠ prev_claim={prev_claim:?}"
        );
        return false;
    }

    transcript.append_scalar(bv_eval);

    let point_bv_kzg: Vec<Challenge> = r_cycle_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();
    if *comm_bv != zero_comm {
        match opening_bv {
            Some(pf) => {
                if PCS::verify(vk, comm_bv, &point_bv_kzg, bv_eval, pf, transcript).is_err() {
                    eprintln!("verify_shout_gate: batch_val HyperKZG verify FAILED");
                    return false;
                }
            }
            None => {
                eprintln!("verify_shout_gate: non-zero comm_bv but no opening proof");
                return false;
            }
        }
    }

    comm_g.append_to_transcript(transcript);

    let mut addr_prev_claim = *bv_eval;
    let mut r_addr_fr: Vec<Fr> = Vec::with_capacity(total_address_bits);
    let mut r_addr_ch: Vec<Challenge> = Vec::with_capacity(total_address_bits);

    for (round, poly) in addr_sc_polys.iter().enumerate() {
        let sum = poly[0] + poly[1];
        if sum != addr_prev_claim {
            eprintln!(
                "verify_shout_gate: addr sc round {round}: p(0)+p(1)={sum:?} ≠ {addr_prev_claim:?}"
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

    let addr_final_expected = *final_g_eval * *final_table_eval;
    if addr_final_expected != addr_prev_claim {
        eprintln!(
            "verify_shout_gate: addr sc final: G·T={addr_final_expected:?} ≠ {addr_prev_claim:?}"
        );
        return false;
    }

    let table_eval_check = mega_table_mle_symbolic(type_order, k, &r_addr_fr);
    if table_eval_check != *final_table_eval {
        eprintln!(
            "verify_shout_gate: mega_table_mle mismatch: {table_eval_check:?} ≠ {final_table_eval:?}"
        );
        return false;
    }

    transcript.append_scalar(final_g_eval);
    transcript.append_scalar(final_table_eval);

    let point_g_kzg: Vec<Challenge> = r_addr_ch.iter().rev().cloned().collect();
    let zero_comm = HyperKZGCommitment::<Bn254>::default();
    if *comm_g != zero_comm {
        match opening_g {
            Some(pf) => {
                if PCS::verify(vk, comm_g, &point_g_kzg, final_g_eval, pf, transcript).is_err() {
                    eprintln!("verify_shout_gate: G_agg HyperKZG verify FAILED");
                    return false;
                }
            }
            None => {
                eprintln!("verify_shout_gate: non-zero comm_g but no opening proof");
                return false;
            }
        }
    }

    true
}

pub fn compute_shout_proof_size_bytes(proof: &ShoutGateProof) -> usize {
    let mut total = 0usize;
    let mut buf = Vec::new();

    for comm in [&proof.comm_bv] {
        buf.clear();
        comm.0.serialize_compressed(&mut buf).ok();
        total += buf.len();
    }
    buf.clear();
    proof.comm_g.0.serialize_compressed(&mut buf).ok();
    total += buf.len();

    total += 4 * 32;
    total += proof.cycle_sc_polys.len() * 3 * 32;
    total += proof.addr_sc_polys.len() * 3 * 32;

    if let Some(pf) = &proof.opening_bv {
        buf.clear();
        pf.serialize_compressed(&mut buf).ok();
        total += buf.len();
    }
    if let Some(pf) = &proof.opening_g {
        buf.clear();
        pf.serialize_compressed(&mut buf).ok();
        total += buf.len();
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbolic_gate_mle_matches_dense_mask_table() {
        let point = vec![Fr::from(3u64), Fr::from(7u64)];
        for mask in [0x03u8, 0x06u8, 0x08u8, 0x0Eu8] {
            let dense = mle_eval_fr(
                &(0usize..4)
                    .map(|idx| Fr::from_bool(((mask >> idx) & 1) == 1))
                    .collect::<Vec<_>>(),
                &point,
            );
            assert_eq!(gate_mask_mle(mask, &point), dense);
        }
    }

    #[test]
    fn symbolic_mega_table_matches_dense_materialization() {
        let type_order = vec![0x03u8, 0x06u8, 0x08u8];
        let dense = build_mega_table(&type_order, 2);
        let point = vec![
            Fr::from(2u64),
            Fr::from(5u64),
            Fr::from(11u64),
            Fr::from(13u64),
        ];
        assert_eq!(
            mega_table_mle_symbolic(&type_order, 2, &point),
            mle_eval_fr(&dense, &point)
        );
    }
}
