//! Dependent-load latency by working-set size. Run with:
//!   cargo run --release --example pointer_chase
//! Fixes vs the Criterion version: Sattolo's algorithm (one single cycle),
//! one 64-byte line per node, explicit cycle validation, sizes that straddle
//! the real cache boundaries (Ryzen 5 7430U: 32 KB L1D, 512 KB L2, 16 MB L3 per core / shared).

use std::hint::black_box;
use std::time::Instant;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line {
    next: u32,
}

/// Sattolo's algorithm: uniformly random permutation that is a single n-cycle.
fn sattolo(n: usize, mut s: u64) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        // splitmix64 (good high bits, unlike a raw LCG)
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let j = (z % i as u64) as usize; // 0..i-1  => single cycle
        p.swap(i, j);
    }
    p
}

fn assert_single_cycle(lines: &[Line]) {
    let mut cur = 0usize;
    let mut steps = 0usize;
    loop {
        cur = lines[cur].next as usize;
        steps += 1;
        if cur == 0 {
            break;
        }
        assert!(steps <= lines.len(), "chain is not a cycle through node 0");
    }
    assert_eq!(steps, lines.len(), "chain is NOT a single full cycle");
}

fn chase_ns(bytes: usize, iters: usize) -> f64 {
    let n = bytes / 64;
    let lines: Vec<Line> = sattolo(n, 0x1234_5678_9ABC_DEF0)
        .into_iter()
        .map(|x| Line { next: x })
        .collect();
    assert_single_cycle(&lines);

    let mut cur = 0usize;
    for _ in 0..n.min(2_000_000) {
        cur = lines[cur].next as usize; // warm up / fault in pages
    }
    let t = Instant::now();
    for _ in 0..iters {
        cur = lines[cur].next as usize; // each address depends on the previous load
    }
    let ns = t.elapsed().as_nanos() as f64 / iters as f64;
    black_box(cur);
    ns
}

fn main() {
    let kb = 1024usize;
    let sizes: &[(&str, usize)] = &[
        ("16KB", 16 * kb),
        ("64KB", 64 * kb),
        ("256KB", 256 * kb),
        ("1MB", 1024 * kb),
        ("4MB", 4 * 1024 * kb),
        ("8MB", 8 * 1024 * kb),
        ("12MB", 12 * 1024 * kb),
        ("24MB", 24 * 1024 * kb),
        ("64MB", 64 * 1024 * kb),
        ("256MB", 256 * 1024 * kb),
    ];
    for &(label, bytes) in sizes {
        let best = (0..3).map(|_| chase_ns(bytes, 20_000_000)).fold(f64::MAX, f64::min);
        println!("{label:>6}: {best:7.2} ns/hop");
    }
}