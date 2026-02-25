#!/bin/bash

# Run AES-vm example with single thread
echo "=========================================="
echo "Running AES-vm with single thread"
echo "RAYON_NUM_THREADS=1"
echo "=========================================="
echo ""

cd examples/aes-vm
RAYON_NUM_THREADS=1 cargo run --release