#![cfg_attr(feature = "guest", no_std)]
use jolt::{end_cycle_tracking, start_cycle_tracking};

// Support up to 2^20 boolean AND gate operations in a chain
// Each AND gate takes 2 boolean inputs and produces 1 boolean output
#[jolt::provable(memory_size = 8388608, max_trace_length = 4194304)]
fn and_chain(input: u32, chain_length: u32) -> u32 {
    // Start with a simple boolean
    let mut result: bool = input != 0;
    
    start_cycle_tracking("and_chain_loop");
    
    // Chain of 2-input boolean AND gates
    // Simple pattern: alternate between AND with 1 and AND with 0
    for i in 0..chain_length {
        // Simple alternating bit to prevent optimization
        let bit: bool = (i % 2) == 0;
        
        // 2-input boolean AND gate
        result = result & bit;
    }
    
    end_cycle_tracking("and_chain_loop");
    
    // Convert boolean result back to u32
    result as u32
}
