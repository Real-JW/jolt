use jolt_sdk::serialize_and_print_size;
use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let save_to_disk = std::env::args().any(|arg| arg == "--save");
    
    // Default to 2^20 (1,048,576 AND gates)
    let chain_length: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1 << 20); // 2^20 = 1,048,576
    
    println!("Running AND chain with {} operations (2^{:.1})", chain_length, (chain_length as f64).log2());
    info!("Running AND chain with {} operations", chain_length);

    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_and_chain(target_dir);

    let shared_preprocessing = guest::preprocess_shared_and_chain(&mut program);

    let prover_preprocessing = guest::preprocess_prover_and_chain(shared_preprocessing.clone());
    let verifier_setup = prover_preprocessing.generators.to_verifier_setup();
    let verifier_preprocessing =
        guest::preprocess_verifier_and_chain(shared_preprocessing, verifier_setup);

    if save_to_disk {
        serialize_and_print_size(
            "Verifier Preprocessing",
            "/tmp/jolt_verifier_preprocessing.dat",
            &verifier_preprocessing,
        )
        .expect("Could not serialize preprocessing.");
    }

    let prove_and_chain = guest::build_prover_and_chain(program, prover_preprocessing);
    let verify_and_chain = guest::build_verifier_and_chain(verifier_preprocessing);

    let input = 0xFFFFFFFF;
    
    let program_summary = guest::analyze_and_chain(input, chain_length);
    program_summary
        .write_to_file("and_chain.txt".into())
        .expect("should write");

    let trace_file = "/tmp/and_chain_trace.bin";
    guest::trace_and_chain_to_file(trace_file, input, chain_length);
    info!("Trace file written to: {trace_file}.");

    println!("Starting proof generation...");
    let now = Instant::now();
    let (output, proof, io_device) = prove_and_chain(input, chain_length);
    let prover_time = now.elapsed().as_secs_f64();
    println!("Prover runtime: {:.2} s", prover_time);
    info!("Prover runtime: {:.2} s", prover_time);
    info!("Output result: {}", output);

    if save_to_disk {
        serialize_and_print_size("Proof", "/tmp/and_chain_proof.bin", &proof)
            .expect("Could not serialize proof.");
        serialize_and_print_size("io_device", "/tmp/and_chain_io_device.bin", &io_device)
            .expect("Could not serialize io_device.");
    }

    println!("\nStarting verification...");
    let now = Instant::now();
    let is_valid = verify_and_chain(input, chain_length, output, io_device.panic, proof);
    let verifier_time = now.elapsed().as_secs_f64();
    
    println!("\n=== Results ===");
    println!("Input: 0x{:08X}", input);
    println!("Chain length: {} (2^{:.1})", chain_length, (chain_length as f64).log2());
    println!("Output: 0x{:08X}", output);
    println!("Prover time: {:.2} s", prover_time);
    println!("Verifier time: {:.2} s", verifier_time);
    println!("Valid: {}", is_valid);
    
    info!("Verification complete. Valid: {}", is_valid);
    
    if !is_valid {
        panic!("Verification failed!");
    }
}
