use std::io::{self, Read};
use std::path::Path;

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

pub struct Circ {
    _num_wires: u32,
    pub primary_inputs: Vec<u32>,
    pub registers: Vec<(u32, u32)>,
    pub outputs: Vec<u32>,
    ops: Vec<Op>,
    pub default_cycles: u32,
}

fn opcode_mask(opc: OpCode) -> u8 {
    match opc {
        OpCode::And => MASK_AND,
        OpCode::Or => MASK_OR,
        OpCode::Xor => MASK_XOR,
        OpCode::Not => MASK_NOT,
    }
}

pub fn load_circuit(path: &Path) -> io::Result<Circ> {
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

/// A single gate evaluation in the execution trace.
#[derive(Clone, Debug)]
pub struct GateEval {
    pub mask: u8,
    pub a: bool,
    pub b: bool,
    pub out: bool,
}

impl GateEval {
    /// 2-bit packed input consistent with `mask` indexing: `idx = (a << 1) | b`.
    #[inline]
    pub fn packed_input(&self) -> usize {
        ((self.a as usize) << 1) | (self.b as usize)
    }

    /// Mega-table address encoding used by the Shout scheme:
    /// `address = type_index * 2^k + packed_input`.
    #[inline]
    pub fn address_for_shout(&self, type_index: usize, k: usize) -> usize {
        debug_assert_eq!(k, 2, "GateEval::address_for_shout: expected k=2");
        (type_index << k) | self.packed_input()
    }
}

/// Evaluate the circuit for `cycles` sequential cycles and collect all gate
/// evaluations into a flat trace (row ordering: cycle-major, gate-minor).
pub fn evaluate_circuit(circ: &Circ, inputs: &[bool], cycles: u32) -> (Vec<GateEval>, Vec<bool>) {
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
            let b = if op.b == NOT_SENTINEL { a } else { wires[op.b as usize] };
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

/// Deterministic ordering of distinct gate masks present in the circuit.
pub fn gate_type_order(circ: &Circ) -> Vec<u8> {
    let mut seen = [false; 256];
    for op in &circ.ops {
        seen[opcode_mask(op.opcode) as usize] = true;
    }
    let mut out = Vec::new();
    for (m, present) in seen.iter().enumerate() {
        if *present {
            out.push(m as u8);
        }
    }
    out
}

