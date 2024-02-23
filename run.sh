#!/bin/sh

i=${1:-0}
zi=$(printf "%04d" $i)
src=${2:-"main"}

# Rust
cargo build --release --bin "${src}"
cd tools/
cargo run -r --bin tester "../target/release/${src}" < "in/${zi}.txt" | clip.exe

# Python
# cd tools/
# cargo run -r --bin tester python3 "../src/${src}.py" < "in/${zi}.txt" | clip.exe
