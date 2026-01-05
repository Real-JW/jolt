use std::time::Instant;
use tracing::info;

pub fn main() {
    tracing_subscriber::fmt::init();

    info!("Compiling AES guest program...");
    let target_dir = "/tmp/jolt-guest-targets";
    let mut program = guest::compile_aes_encrypt(target_dir);

    info!("Running preprocessing...");
    let shared_preprocessing = guest::preprocess_shared_aes_encrypt(&mut program);
    let prover_preprocessing = guest::preprocess_prover_aes_encrypt(shared_preprocessing.clone());
    let verifier_preprocessing = guest::preprocess_verifier_aes_encrypt(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    let prove_aes_encrypt = guest::build_prover_aes_encrypt(program, prover_preprocessing);
    let verify_aes_encrypt = guest::build_verifier_aes_encrypt(verifier_preprocessing);

    // Example plaintext and key
    let plaintext: [u8; 16] = [
        0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d,
        0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07, 0x34
    ];
    let key: [u8; 16] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c
    ];

    info!("Input plaintext: {}", hex::encode(plaintext));
    info!("Key: {}", hex::encode(key));
    info!("");

    // Proving phase
    info!("Starting proof generation...");
    let prove_start = Instant::now();
    let (ciphertext, proof, program_io) = prove_aes_encrypt(plaintext, key);
    let prove_time = prove_start.elapsed();
    info!("✓ Prover time: {:.3} seconds", prove_time.as_secs_f64());
    info!("");

    // Verification phase
    info!("Starting proof verification...");
    let verify_start = Instant::now();
    let is_valid = verify_aes_encrypt(plaintext, key, ciphertext, program_io.panic, proof);
    let verify_time = verify_start.elapsed();
    info!("✓ Verifier time: {:.3} seconds", verify_time.as_secs_f64());
    info!("");

    // Results
    info!("Output ciphertext: {}", hex::encode(ciphertext));
    info!("Proof is valid: {}", is_valid);
    info!("");
    info!("=== Performance Summary ===");
    info!("Prover time:   {:.3}s", prove_time.as_secs_f64());
    info!("Verifier time: {:.3}s", verify_time.as_secs_f64());
    info!("Total time:    {:.3}s", (prove_time + verify_time).as_secs_f64());
}
