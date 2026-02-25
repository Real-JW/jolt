//! **`.lczbc` (LUT-annotated czbc) binary format loader**
//!
//! Reads the `.lczbc` files produced by `data/circuitCompiler/circuitToLut.py`
//! (Phase 1) and provides:
//!
//! * [`LutDesc`] — description of one unique LUT type (truth table + dimensions)
//! * [`LutOp`]   — one LUT evaluation in the circuit (which LUT, dst wire, src wires)
//! * [`LutCirc`] — the complete LUT circuit
//! * [`load_lut_circuit`] — parse a `.lczbc` file into a [`LutCirc`]
//! * [`LutEval`] — a single simulation trace row (k inputs + m outputs)
//! * [`evaluate_lut_circuit`] — runs the circuit and collects the trace
//!
//! # Binary format
//!
//! ```text
//! Header (40 bytes):
//!   u32  magic          = 0x4C435A43  ('LCZC')
//!   u16  version        = 2
//!   u16  flags
//!   u32  num_wires
//!   u32  n_primary_inputs
//!   u32  n_registers
//!   u32  n_outputs
//!   u32  n_lut_types
//!   u32  n_ops
//!   u32  default_cycles
//!
//! Primary input wires:  n_primary_inputs × u32
//! Register wires:       n_registers × (u32 out_wire, u32 in_wire)
//! Output wires:         n_outputs × u32
//!
//! LUT type table (n_lut_types entries):
//!   u32  lut_id
//!   u8   k_inputs
//!   u8   m_outputs
//!   i16  reserved       (signed in Python `struct.pack("<IBBh", …)`)
//!   u8[] truth_table    (ceil(2^k_inputs × m_outputs / 8) bytes, LSB-first)
//!
//! Op stream (n_ops entries, variable-length per op):
//!   u32  lut_id
//!   u32  dst_wire
//!   u32  src_wire[0..k)  (k_inputs words; k looked up from lut_type)
//! ```

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;

// ── format constants ────────────────────────────────────────────────────────
pub const MAGIC: u32   = 0x4C435A43; // 'LCZC'
pub const VERSION: u16 = 2;

// ── binary helpers ───────────────────────────────────────────────────────────
fn ru8(d: &[u8], o: &mut usize) -> io::Result<u8> {
    let v = *d
        .get(*o)
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected eof"))?;
    *o += 1;
    Ok(v)
}

fn ru16(d: &[u8], o: &mut usize) -> io::Result<u16> {
    if *o + 2 > d.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected eof"));
    }
    let v = u16::from_le_bytes([d[*o], d[*o + 1]]);
    *o += 2;
    Ok(v)
}

fn ru32(d: &[u8], o: &mut usize) -> io::Result<u32> {
    if *o + 4 > d.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "unexpected eof"));
    }
    let v = u32::from_le_bytes([d[*o], d[*o + 1], d[*o + 2], d[*o + 3]]);
    *o += 4;
    Ok(v)
}

// ── data types ───────────────────────────────────────────────────────────────

/// Unique LUT type: specifies the Boolean function over k inputs and m outputs.
#[derive(Clone, Debug)]
pub struct LutDesc {
    /// Stable identifier derived from the truth table hash (assigned by the
    /// Python extractor).
    pub lut_id: u32,
    /// Number of input bits.
    pub k: usize,
    /// Number of output bits (always 1 in Phase 1).
    pub m: usize,
    /// Truth-table bytes: `ceil(2^k * m / 8)` bytes, LSB-first.
    /// Bit `i` of `table[0]` (when m=1) is the output for input combination `i`.
    pub truth_table: Vec<u8>,
}

impl LutDesc {
    /// Evaluate the LUT on concrete boolean inputs.
    ///
    /// `inputs` must have exactly `k` elements; `out_bit` selects which of the
    /// `m` output bits to return.
    pub fn eval_bool(&self, inputs: &[bool], out_bit: usize) -> bool {
        debug_assert_eq!(inputs.len(), self.k, "wrong number of inputs");
        debug_assert!(out_bit < self.m, "out_bit out of range");

        // Encode inputs as an integer index into the truth table.
        let mut idx: usize = 0;
        for (i, &b) in inputs.iter().enumerate() {
            if b {
                idx |= 1 << i;
            }
        }
        // idx selects the (idx * m + out_bit)-th bit in truth_table.
        let bit_pos = idx * self.m + out_bit;
        (self.truth_table[bit_pos / 8] >> (bit_pos % 8)) & 1 == 1
    }
}

/// One LUT operation in the circuit (a single LUT invocation).
#[derive(Clone, Debug)]
pub struct LutOp {
    /// Which LUT type is used (index into `LutCirc::lut_types`).
    pub lut_id: u32,
    /// Output wire (first of `m` consecutive wires; m=1 in Phase 1).
    pub dst_wire: u32,
    /// Source wires: exactly `k` entries (order matches truth-table bit positions).
    pub src_wires: Vec<u32>,
}

/// Complete LUT-covered circuit loaded from a `.lczbc` file.
#[derive(Debug)]
pub struct LutCirc {
    pub num_wires: usize,
    /// Primary input wire indices (driven externally each cycle).
    pub primary_inputs: Vec<u32>,
    /// Register (DFF) pairs: `(output_wire, input_wire)`.
    pub registers: Vec<(u32, u32)>,
    /// Observable output wire indices.
    pub outputs: Vec<u32>,
    /// All unique LUT types, keyed by `lut_id`.
    pub lut_types: HashMap<u32, LutDesc>,
    /// LUT ops in topological order (safe to evaluate sequentially).
    pub ops: Vec<LutOp>,
    /// Default number of simulation cycles (from the bytecode header).
    pub default_cycles: u32,
}

// ── loader ───────────────────────────────────────────────────────────────────

/// Read a `.lczbc` file and return the parsed [`LutCirc`].
pub fn load_lut_circuit(path: &Path) -> io::Result<LutCirc> {
    let mut raw = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut raw)?;

    let mut o = 0usize;

    // --- header ---
    let magic = ru32(&raw, &mut o)?;
    if magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad magic 0x{magic:08X} (expected 0x{MAGIC:08X})"),
        ));
    }
    let version = ru16(&raw, &mut o)?;
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported .lczbc version {version} (expected {VERSION})"),
        ));
    }
    let _flags       = ru16(&raw, &mut o)?;
    let num_wires    = ru32(&raw, &mut o)? as usize;
    let n_in         = ru32(&raw, &mut o)? as usize;
    let n_reg        = ru32(&raw, &mut o)? as usize;
    let n_out        = ru32(&raw, &mut o)? as usize;
    let n_lut_types  = ru32(&raw, &mut o)? as usize;
    let n_ops        = ru32(&raw, &mut o)? as usize;
    let default_cycles = ru32(&raw, &mut o)?;

    // --- primary inputs ---
    let mut primary_inputs = Vec::with_capacity(n_in);
    for _ in 0..n_in {
        primary_inputs.push(ru32(&raw, &mut o)?);
    }

    // --- registers ---
    let mut registers = Vec::with_capacity(n_reg);
    for _ in 0..n_reg {
        let rout = ru32(&raw, &mut o)?;
        let rin  = ru32(&raw, &mut o)?;
        registers.push((rout, rin));
    }

    // --- output wires ---
    let mut outputs = Vec::with_capacity(n_out);
    for _ in 0..n_out {
        outputs.push(ru32(&raw, &mut o)?);
    }

    // --- LUT type table ---
    let mut lut_types: HashMap<u32, LutDesc> = HashMap::with_capacity(n_lut_types);
    for _ in 0..n_lut_types {
        let lut_id = ru32(&raw, &mut o)?;
        let k      = ru8(&raw, &mut o)? as usize;
        let m      = ru8(&raw, &mut o)? as usize;
        let _reserved = ru16(&raw, &mut o)?; // skip 2 reserved bytes

        let table_bits  = (1usize << k) * m;
        let table_bytes = (table_bits + 7) / 8;
        if o + table_bytes > raw.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("truth table for lut_id={lut_id} truncated"),
            ));
        }
        let truth_table = raw[o..o + table_bytes].to_vec();
        o += table_bytes;

        lut_types.insert(
            lut_id,
            LutDesc { lut_id, k, m, truth_table },
        );
    }

    // --- op stream ---
    let mut ops = Vec::with_capacity(n_ops);
    for op_idx in 0..n_ops {
        let lut_id   = ru32(&raw, &mut o)?;
        let dst_wire = ru32(&raw, &mut o)?;

        let k = lut_types
            .get(&lut_id)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("op #{op_idx}: unknown lut_id={lut_id}"),
                )
            })?
            .k;

        let mut src_wires = Vec::with_capacity(k);
        for _ in 0..k {
            src_wires.push(ru32(&raw, &mut o)?);
        }

        ops.push(LutOp { lut_id, dst_wire, src_wires });
    }

    Ok(LutCirc {
        num_wires,
        primary_inputs,
        registers,
        outputs,
        lut_types,
        ops,
        default_cycles,
    })
}

// ── simulation ───────────────────────────────────────────────────────────────

/// A single LUT evaluation in the simulation trace.
#[derive(Clone, Debug)]
pub struct LutEval {
    /// Which LUT type was invoked.
    pub lut_id: u32,
    /// Input bit values (k bits).
    pub inputs: Vec<bool>,
    /// Output bit values (m bits; m=1 in Phase 1).
    pub outputs: Vec<bool>,
}

/// Simulate [`LutCirc`] for `cycles` clock cycles, collecting every LUT
/// invocation into a flat trace (cycle-major, op-minor ordering).
///
/// Returns `(trace, final_output_values)`.
pub fn evaluate_lut_circuit(
    circ: &LutCirc,
    inputs: &[bool],
    cycles: u32,
) -> (Vec<LutEval>, Vec<bool>) {
    let n_cyc = cycles.max(1) as usize;
    let mut wires = vec![false; circ.num_wires];
    let mut reg_state = vec![false; circ.registers.len()];

    let mut trace: Vec<LutEval> = Vec::with_capacity(circ.ops.len() * n_cyc);

    for _cycle in 0..n_cyc {
        // Drive primary inputs
        for (i, &w) in circ.primary_inputs.iter().enumerate() {
            wires[w as usize] = inputs.get(i).copied().unwrap_or(false);
        }
        // Apply register state (from previous cycle)
        for (ri, &(reg_out, _)) in circ.registers.iter().enumerate() {
            wires[reg_out as usize] = reg_state[ri];
        }

        // Evaluate LUT ops in topological order
        for op in &circ.ops {
            let desc = &circ.lut_types[&op.lut_id];
            let in_vals: Vec<bool> = op
                .src_wires
                .iter()
                .map(|&w| wires[w as usize])
                .collect();

            let out_vals: Vec<bool> = (0..desc.m)
                .map(|j| desc.eval_bool(&in_vals, j))
                .collect();

            // Write output bits to destination wires
            for (j, &v) in out_vals.iter().enumerate() {
                wires[(op.dst_wire as usize) + j] = v;
            }

            trace.push(LutEval {
                lut_id:  op.lut_id,
                inputs:  in_vals,
                outputs: out_vals,
            });
        }

        // Capture next register state
        for (ri, &(_, reg_in)) in circ.registers.iter().enumerate() {
            reg_state[ri] = wires[reg_in as usize];
        }
    }

    let final_outputs: Vec<bool> = circ
        .outputs
        .iter()
        .map(|&w| wires[w as usize])
        .collect();

    (trace, final_outputs)
}

// ── unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_and_lut() -> LutDesc {
        // 2-input AND: truth table = 0b0001 = 0x01
        LutDesc {
            lut_id:      0,
            k:           2,
            m:           1,
            truth_table: vec![0x08], // AND mask matching czbc: (a=1,b=1)→1
        }
    }

    fn make_xor_lut() -> LutDesc {
        // 2-input XOR: truth table = 0b0110 = 0x06
        LutDesc {
            lut_id:      1,
            k:           2,
            m:           1,
            truth_table: vec![0x06],
        }
    }

    #[test]
    fn lut_eval_bool_and() {
        let lut = make_and_lut();
        assert!(!lut.eval_bool(&[false, false], 0));
        assert!(!lut.eval_bool(&[true,  false], 0));
        assert!(!lut.eval_bool(&[false, true],  0));
        assert!( lut.eval_bool(&[true,  true],  0));
    }

    #[test]
    fn lut_eval_bool_xor() {
        let lut = make_xor_lut();
        assert!(!lut.eval_bool(&[false, false], 0));
        assert!( lut.eval_bool(&[true,  false], 0));
        assert!( lut.eval_bool(&[false, true],  0));
        assert!(!lut.eval_bool(&[true,  true],  0));
    }

    /// Simulate a tiny circuit:   out = (a XOR b)
    #[test]
    fn simulate_single_xor_lut() {
        let xor_lut = make_xor_lut();
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

        let (trace, outs) = evaluate_lut_circuit(&circ, &[true, false], 1);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].outputs[0], true);  // 1 XOR 0 = 1
        assert_eq!(outs[0], true);

        let (trace2, outs2) = evaluate_lut_circuit(&circ, &[true, true], 1);
        assert_eq!(trace2[0].outputs[0], false); // 1 XOR 1 = 0
        assert!(!outs2[0]);
    }
}
