//! Multi-Session Clean Rerun & Raw Output Archiver (Step 1).
//! Runs 3 isolated benchmark sessions, saves raw session outputs to disk,
//! and computes statistical variance (mean, median, sample stddev) across sessions.

use std::fs::File;
use std::io::Write;
use std::hint::black_box;
use std::time::Instant;

use state_search_engine::engine::{EngineScratch, SearchEngine};
use state_search_engine::graph::generate_uniform_graph;

const SESSIONS: usize = 3;
const WARMUP: usize = 3;
const RUNS_PER_SESSION: usize = 10;

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Computes Sample Standard Deviation (N - 1 denominator)
fn sample_std_dev(v: &[f64], mean_val: f64) -> f64 {
    if v.len() <= 1 {
        return 0.0;
    }
    let var = v.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> std::io::Result<()> {
    println!("=========================================================================================");
    println!("             MULTI-SESSION CLEAN BENCHMARK RERUN & RAW ARCHIVER (STEP 1)                 ");
    println!("=========================================================================================\n");

    let sizes = [100_000usize, 500_000, 1_000_000, 4_000_000];
    let mut session_medians: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); sizes.len()]; SESSIONS];

    for session in 0..SESSIONS {
        let log_name = format!("raw_session_{}.log", session + 1);
        let mut file = File::create(&log_name)?;

        writeln!(file, "--- SESSION {} RAW TIMINGS (ms) ---", session + 1)?;
        println!("Executing Session {} of {} (Archiving to {})...", session + 1, SESSIONS, log_name);

        for (s_idx, &n) in sizes.iter().enumerate() {
            let nodes = generate_uniform_graph(n, 2, 0x1234_5678 ^ (session as u64 * 100) ^ n as u64);
            let engine = SearchEngine::new(nodes);
            let mut scratch = EngineScratch::new(n);

            let mut run_times = Vec::new();
            for r in 0..(WARMUP + RUNS_PER_SESSION) {
                let start = Instant::now();
                let edges = engine.traverse_bfs_with_scratch(black_box(0), &mut scratch);
                let ms = start.elapsed().as_secs_f64() * 1e3;

                writeln!(file, "Size={n} | Run={r} | Edges={edges} | Time={ms:.4} ms")?;
                if r >= WARMUP {
                    run_times.push(ms);
                }
            }

            let med = median(&mut run_times);
            session_medians[session][s_idx].push(med);
            writeln!(file, "Size={n} | SESSION {} MEDIAN = {:.4} ms\n", session + 1, med)?;
        }
    }

    println!("\n=========================================================================================");
    println!("                MULTI-SESSION STATISTICAL VARIANCE SUMMARY TABLE                          ");
    println!("=========================================================================================");
    println!("Node Count | Session 1 (ms) | Session 2 (ms) | Session 3 (ms) | Mean (ms) | Sample SD (%)");
    println!("-----------------------------------------------------------------------------------------");

    for (s_idx, &n) in sizes.iter().enumerate() {
        let s1 = session_medians[0][s_idx][0];
        let s2 = session_medians[1][s_idx][0];
        let s3 = session_medians[2][s_idx][0];

        let vals = [s1, s2, s3];
        let m = mean(&vals);
        let sd = sample_std_dev(&vals, m);
        let pct = (sd / m) * 100.0;

        println!(
            "{:10} | {:14.3} | {:14.3} | {:14.3} | {:9.3} | {:12.2}%",
            n, s1, s2, s3, m, pct
        );
    }
    println!("=========================================================================================\n");

    Ok(())
}
