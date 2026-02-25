//! k-input, m-output sub-circuit LUT for zkFPGA Phase 2.
//!
//! [`SubCircuitLut`] stores a Boolean function on k input bits and m output
//! bits as a packed truth table and provides:
//! - Boolean evaluation (`eval_bool`)
//! - Multilinear extension (MLE) evaluation at arbitrary field points
//!   (`evaluate_mle_at`), using the standard O(2^k) fold algorithm.
//!
//! The truth-table index convention matches `lczbc.rs`:
//!   `idx = input[0] | (input[1] << 1) | … | (input[k-1] << (k-1))`   (LSB-first)
//! Output bit j is stored at bit position j of `table[idx]`.
//!
//! MLE variable ordering is also LSB-first, consistent with the `init_eq` /
//! `bind` helpers in `bool-circuit-native`.

use crate::field::JoltField;

/// A k-input, m-output sub-circuit truth-table LUT.
///
/// # Truth-table layout
/// `table` has `2^k` entries.  Entry at `idx` stores all m output bits:
/// `(table[idx] >> j) & 1` is output bit j for that input combination.
///
/// # MLE convention
/// `evaluate_mle_at(r, j)` computes
/// ```text
/// T̃_j(r) = ∑_{x ∈ {0,1}^k} ((table[x] >> j) & 1) · eq_lsb(r, x)
/// ```
/// where `eq_lsb(r, x) = ∏_i (r[i]·bit_i(x) + (1-r[i])·(1-bit_i(x)))`.
#[derive(Clone, Debug)]
pub struct SubCircuitLut {
    /// Number of input bits.
    pub k: usize,
    /// Number of output bits.
    pub m: usize,
    /// Packed truth table: `2^k` entries, each `u64`; bit j of `table[i]` =
    /// output bit j for input combination i.
    pub table: Vec<u64>,
}

impl SubCircuitLut {
    /// Construct from the raw truth-table bytes produced by `circuitToLut.py`.
    ///
    /// The byte format is LSB-first: bit `idx*m + out_bit` of the byte array
    /// is output bit `out_bit` for input combination `idx`.
    pub fn from_bytes(k: usize, m: usize, bytes: &[u8]) -> Self {
        let n = 1usize << k;
        let mut table = vec![0u64; n];
        for idx in 0..n {
            for out_bit in 0..m {
                let bit_pos = idx * m + out_bit;
                let val = (bytes[bit_pos / 8] >> (bit_pos % 8)) & 1;
                if val != 0 {
                    table[idx] |= 1u64 << out_bit;
                }
            }
        }
        SubCircuitLut { k, m, table }
    }

    /// Evaluate the LUT on concrete boolean inputs.
    ///
    /// `inputs` must have `k` elements; `out_bit` selects which of the m
    /// output columns to return.
    #[inline]
    pub fn eval_bool(&self, inputs: &[bool], out_bit: usize) -> bool {
        debug_assert_eq!(inputs.len(), self.k, "SubCircuitLut: wrong input count");
        debug_assert!(out_bit < self.m, "SubCircuitLut: out_bit out of range");
        let mut idx = 0usize;
        for (i, &b) in inputs.iter().enumerate() {
            if b {
                idx |= 1 << i;
            }
        }
        (self.table[idx] >> out_bit) & 1 == 1
    }

    /// Evaluate the MLE of output bit `out_bit` at arbitrary field points
    /// `r ∈ F^k` using the standard O(2^k) fold algorithm (LSB-first).
    ///
    /// The algorithm folds variable 0 first (LSB), matching the convention in
    /// `init_eq` / `bind` in `bool-circuit-native`.
    pub fn evaluate_mle_at<F: JoltField>(&self, r: &[F], out_bit: usize) -> F {
        assert_eq!(r.len(), self.k, "SubCircuitLut::evaluate_mle_at: wrong r length");
        let n = 1usize << self.k;

        // Initialise table values as field elements for the selected output bit.
        let mut vals: Vec<F> = (0..n)
            .map(|idx| F::from_u64((self.table[idx] >> out_bit) & 1))
            .collect();

        // Fold each variable in LSB-first order.
        // After folding variable i at r[i], the vector halves and the remaining
        // entries represent partial sums over the already-bound dimensions.
        for &ri in r.iter() {
            let half = vals.len() / 2;
            for i in 0..half {
                let lo = vals[2 * i];
                let hi = vals[2 * i + 1];
                // lo * (1 - ri) + hi * ri  =  lo + ri * (hi - lo)
                vals[i] = lo + ri * (hi - lo);
            }
            vals.truncate(half);
        }

        debug_assert_eq!(vals.len(), 1);
        vals[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{One, Zero};

    /// 2-input AND: truth table 0b0001 using the AND mask 0x08.
    /// idx = (a<<0)|(b<<1): idx=0 → (0,0) → 0, idx=1 → (1,0) → 0,
    ///                       idx=2 → (0,1) → 0, idx=3 → (1,1) → 1.
    fn make_and_lut() -> SubCircuitLut {
        SubCircuitLut {
            k: 2,
            m: 1,
            table: vec![0, 0, 0, 1], // AND
        }
    }

    /// 2-input XOR.
    fn make_xor_lut() -> SubCircuitLut {
        SubCircuitLut {
            k: 2,
            m: 1,
            table: vec![0, 1, 1, 0], // XOR
        }
    }

    #[test]
    fn eval_bool_and() {
        let lut = make_and_lut();
        assert!(!lut.eval_bool(&[false, false], 0));
        assert!(!lut.eval_bool(&[true, false], 0));
        assert!(!lut.eval_bool(&[false, true], 0));
        assert!(lut.eval_bool(&[true, true], 0));
    }

    #[test]
    fn eval_bool_xor() {
        let lut = make_xor_lut();
        assert!(!lut.eval_bool(&[false, false], 0));
        assert!(lut.eval_bool(&[true, false], 0));
        assert!(lut.eval_bool(&[false, true], 0));
        assert!(!lut.eval_bool(&[true, true], 0));
    }

    #[test]
    fn mle_at_boolean_points_and() {
        let lut = make_and_lut();
        let zero = Fr::zero();
        let one = Fr::one();
        // MLE must agree with the truth table at all boolean points.
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[zero, zero], 0), zero);
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[one, zero], 0), zero);
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[zero, one], 0), zero);
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[one, one], 0), one);
    }

    #[test]
    fn mle_at_boolean_points_xor() {
        let lut = make_xor_lut();
        let zero = Fr::zero();
        let one = Fr::one();
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[zero, zero], 0), zero);
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[one, zero], 0), one);
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[zero, one], 0), one);
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[one, one], 0), zero);
    }

    #[test]
    fn mle_linearity_and() {
        use ark_ff::UniformRand;
        let lut = make_and_lut();
        let mut rng = ark_std::test_rng();
        // AND MLE = r[0] * r[1].
        // Check at random points.
        for _ in 0..10 {
            let r0 = Fr::rand(&mut rng);
            let r1 = Fr::rand(&mut rng);
            let expected = r0 * r1;
            let got = lut.evaluate_mle_at::<Fr>(&[r0, r1], 0);
            assert_eq!(got, expected, "AND MLE(r0,r1) should equal r0*r1");
        }
    }

    #[test]
    fn from_bytes_and() {
        // AND mask 0x08 = 0b00001000; bit i = output for idx i.
        // idx=0 → bit0 = 0, idx=1 → bit1 = 0, idx=2 → bit2 = 0, idx=3 → bit3 = 1
        // LSB-first byte: 0x08
        let lut = SubCircuitLut::from_bytes(2, 1, &[0x08]);
        let zero = Fr::zero();
        let one = Fr::one();
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[zero, zero], 0), zero);
        assert_eq!(lut.evaluate_mle_at::<Fr>(&[one, one], 0), one);
    }

    #[test]
    fn k3_eval_bool() {
        // 3-input majority: output = 1 iff ≥ 2 inputs are 1.
        // idx: 0b000=0→0, 0b001=1→0, 0b010=2→0, 0b011=3→1
        //      0b100=4→0, 0b101=5→1, 0b110=6→1, 0b111=7→1
        let lut = SubCircuitLut {
            k: 3,
            m: 1,
            table: vec![0, 0, 0, 1, 0, 1, 1, 1],
        };
        assert!(!lut.eval_bool(&[false, false, false], 0));
        assert!(!lut.eval_bool(&[true, false, false], 0));
        assert!(lut.eval_bool(&[true, true, false], 0));
        assert!(lut.eval_bool(&[true, true, true], 0));
    }
}
