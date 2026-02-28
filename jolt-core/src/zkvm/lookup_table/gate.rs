//! Edited0730
//! Generic 4-bit truth-table lookup table for boolean gate circuits.
//!
//! Key layout: `index = (a << 1) | b` where bit `i` of `mask` is `f(a=(i>>1), b=(i&1))`.
//! Standard masks: AND=0x08, OR=0x0E, XOR=0x06, NOT=0x03.

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use serde::{Deserialize, Serialize};
use super::JoltLookupTable;

/// A generic boolean gate parameterised by a 4-bit truth-table mask.
///
/// The lookup key is 2 bits wide: `index = (a << 1) | b`.
#[derive(Copy, Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateLookupTable {
    /// 4-bit truth table: bit `i` is the gate output for input combination `i`.
    pub mask: u8,
}

impl GateLookupTable {
    pub const AND: Self = Self { mask: 0x08 };
    pub const OR:  Self = Self { mask: 0x0E };
    pub const XOR: Self = Self { mask: 0x06 };
    pub const NOT: Self = Self { mask: 0x03 };

    /// Evaluate the gate at boolean inputs `(a, b)`.
    #[inline]
    pub fn eval_bool(&self, a: bool, b: bool) -> bool {
        let idx = ((a as u8) << 1) | (b as u8);
        (self.mask >> idx) & 1 == 1
    }

    /// Evaluate the MLE of this gate's truth table at arbitrary field points `(ra, rb)`.
    ///
    /// Both `ra` and `rb` are plain `F` field elements (no Challenge wrapper).
    /// This is the helper used by the standalone sumcheck in `bool-lut`.
    pub fn evaluate_mle_at<F: JoltField>(&self, ra: F, rb: F) -> F {
        let t = |i: u32| F::from_u64(((self.mask >> i) & 1) as u64);
        let one = F::one();
        t(0) * ((one - ra) * (one - rb))
            + t(1) * ((one - ra) * rb)
            + t(2) * (ra * (one - rb))
            + t(3) * (ra * rb)
    }
}

impl JoltLookupTable for GateLookupTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        ((self.mask >> (index as u32)) & 1) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        assert_eq!(r.len(), 2, "GateLookupTable: expected 2 variables [r_a, r_b]");
        let (ra, rb) = (r[0], r[1]);
        let t = |i: u32| F::from_u64(((self.mask >> i) & 1) as u64);
        let one_m_ra: F = F::one() - ra;
        let one_m_rb: F = F::one() - rb;
        t(0) * (one_m_ra * one_m_rb)
            + t(1) * (one_m_ra * rb)
            + t(2) * (ra * one_m_rb)
            + t(3) * (ra * rb)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{One, Zero};
    use super::GateLookupTable;
    use crate::zkvm::lookup_table::JoltLookupTable;

    fn f(b: bool) -> Fr {
        if b { Fr::one() } else { Fr::zero() }
    }

    fn test_gate_boolean(table: GateLookupTable) {
        for a in [false, true] {
            for b in [false, true] {
                let expected = table.eval_bool(a, b);
                let got = table.evaluate_mle_at(f(a), f(b));
                assert_eq!(got, f(expected), "gate 0x{:02X} a={} b={}", table.mask, a as u8, b as u8);
            }
        }
    }

    #[test]
    fn and_gate_boolean_inputs() { test_gate_boolean(GateLookupTable::AND); }
    #[test]
    fn or_gate_boolean_inputs()  { test_gate_boolean(GateLookupTable::OR); }
    #[test]
    fn xor_gate_boolean_inputs() { test_gate_boolean(GateLookupTable::XOR); }
    #[test]
    fn not_gate_boolean_inputs() { test_gate_boolean(GateLookupTable::NOT); }

    #[test]
    fn materialize_xor() {
        let t = GateLookupTable::XOR;
        assert_eq!(t.materialize_entry(0), 0); // 0 XOR 0
        assert_eq!(t.materialize_entry(1), 1); // 0 XOR 1
        assert_eq!(t.materialize_entry(2), 1); // 1 XOR 0
        assert_eq!(t.materialize_entry(3), 0); // 1 XOR 1
    }
}
