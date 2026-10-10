//! Unified Level-Synchronous BFS Traversal Driver.
//! Parameterized over any GraphLayout design and configured via TraversalConfig.

use crate::variants::traits::{GraphLayout, PrefetchMode, TraversalConfig};
use crate::variants::visited::{SharedParallelScratch, SharedSingleScratch};
use rayon::prelude::*;
use std::sync::atomic::Ordering;

/// Prepares scratch and graph visited state prior to traversal.
/// MUST be called outside the timed region!
#[inline(always)]
pub fn prepare_scratch_single<G: GraphLayout>(
    graph: &G,
    scratch: &mut SharedSingleScratch,
) {
    scratch.reset();
    if graph.has_internal_visited() {
        graph.internal_reset_visited();
    }
}

/// Prepares parallel scratch and graph visited state prior to traversal.
/// MUST be called outside the timed region!
#[inline(always)]
pub fn prepare_scratch_parallel<G: GraphLayout>(graph: &G, scratch: &mut SharedParallelScratch) {
    scratch.reset();
    if graph.has_internal_visited() {
        graph.internal_reset_visited();
    }
}

/// Prepares reusable parallel frontiers for the selected worker count.
/// MUST be called outside the timed region!
#[inline(always)]
pub fn prepare_scratch_parallel_for_threads<G: GraphLayout>(
    graph: &G,
    scratch: &mut SharedParallelScratch,
    worker_count: usize,
) {
    scratch.prepare_frontiers(worker_count);
    prepare_scratch_parallel(graph, scratch);
}

/// Single-threaded level-synchronous BFS using pre-allocated scratch buffer.
/// Prepares scratch internally if not pre-prepared.
pub fn traverse_bfs_single<G: GraphLayout>(
    graph: &G,
    start_node: u32,
    config: &TraversalConfig,
    scratch: &mut SharedSingleScratch,
) -> (u64, usize) {
    prepare_scratch_single(graph, scratch);
    traverse_bfs_single_prepared(graph, start_node, config, scratch)
}

/// Zero-allocation single-threaded BFS where scratch and graph have ALREADY been reset.
/// Guarantees that zero allocation and zero resetting occurs inside the timed region.
pub fn traverse_bfs_single_prepared<G: GraphLayout>(
    graph: &G,
    start_node: u32,
    config: &TraversalConfig,
    scratch: &mut SharedSingleScratch,
) -> (u64, usize) {
    let n = graph.num_nodes();
    if (start_node as usize) >= n {
        return (0, 0);
    }

    let start_already = if graph.has_internal_visited() {
        graph.internal_test_and_set(start_node)
    } else {
        scratch.test_and_set(start_node)
    };
    let _ = start_already;

    scratch.cur.push(start_node);
    let mut total_edges: u64 = 0;
    let mut reached_nodes: usize = 1;

    while !scratch.cur.is_empty() {
        let cur_len = scratch.cur.len();

        match config.prefetch {
            PrefetchMode::None => {
                for idx in 0..cur_len {
                    let u = scratch.cur[idx];
                    graph.for_each_neighbor(u, |v| {
                        total_edges += 1;
                        let already = if graph.has_internal_visited() {
                            graph.internal_test_and_set(v)
                        } else {
                            scratch.test_and_set(v)
                        };
                        if !already {
                            scratch.next.push(v);
                        }
                    });
                }
            }
            PrefetchMode::KAhead(k) => {
                for idx in 0..cur_len {
                    if idx + k < cur_len {
                        let lookahead = scratch.cur[idx + k];
                        graph.prefetch_stage1(lookahead);
                        graph.prefetch_stage2(lookahead);
                    }
                    let u = scratch.cur[idx];
                    graph.for_each_neighbor(u, |v| {
                        total_edges += 1;
                        let already = if graph.has_internal_visited() {
                            graph.internal_test_and_set(v)
                        } else {
                            scratch.test_and_set(v)
                        };
                        if !already {
                            scratch.next.push(v);
                        }
                    });
                }
            }
            PrefetchMode::TwoStage(k) => {
                for idx in 0..cur_len {
                    if idx + 2 * k < cur_len {
                        graph.prefetch_stage1(scratch.cur[idx + 2 * k]);
                    }
                    if idx + k < cur_len {
                        graph.prefetch_stage2(scratch.cur[idx + k]);
                    }
                    let u = scratch.cur[idx];
                    graph.for_each_neighbor(u, |v| {
                        total_edges += 1;
                        let already = if graph.has_internal_visited() {
                            graph.internal_test_and_set(v)
                        } else {
                            scratch.test_and_set(v)
                        };
                        if !already {
                            scratch.next.push(v);
                        }
                    });
                }
            }
            PrefetchMode::Batched(batch_size) => {
                let mut idx = 0;
                while idx < cur_len {
                    let batch_end = (idx + batch_size).min(cur_len);
                    for b in idx..batch_end {
                        graph.prefetch_stage1(scratch.cur[b]);
                    }
                    for b in idx..batch_end {
                        graph.prefetch_stage2(scratch.cur[b]);
                    }
                    for b in idx..batch_end {
                        let u = scratch.cur[b];
                        graph.for_each_neighbor(u, |v| {
                            total_edges += 1;
                            let already = if graph.has_internal_visited() {
                                graph.internal_test_and_set(v)
                            } else {
                                scratch.test_and_set(v)
                            };
                            if !already {
                                scratch.next.push(v);
                            }
                        });
                    }
                    idx = batch_end;
                }
            }
        }

        reached_nodes += scratch.next.len();
        scratch.cur.clear();
        std::mem::swap(&mut scratch.cur, &mut scratch.next);
    }

    (total_edges, reached_nodes)
}

/// Multi-threaded level-synchronous BFS using pre-instantiated thread pool & visited buffer.
/// Prepares scratch internally if not pre-prepared.
pub fn traverse_bfs_parallel<G: GraphLayout>(
    graph: &G,
    start_node: u32,
    pool: &rayon::ThreadPool,
    config: &TraversalConfig,
    scratch: &mut SharedParallelScratch,
) -> (u64, usize) {
    prepare_scratch_parallel_for_threads(graph, scratch, pool.current_num_threads());
    traverse_bfs_parallel_prepared(graph, start_node, pool, config, scratch)
}

/// Zero-allocation multi-threaded level-synchronous BFS where scratch and graph are already reset.
pub fn traverse_bfs_parallel_prepared<G: GraphLayout>(
    graph: &G,
    start_node: u32,
    pool: &rayon::ThreadPool,
    config: &TraversalConfig,
    scratch: &mut SharedParallelScratch,
) -> (u64, usize) {
    let n = graph.num_nodes();
    if n == 0 || (start_node as usize) >= n {
        return (0, 0);
    }

    let num_threads = pool.current_num_threads();
    assert_eq!(scratch.worker_next.len(), num_threads, "parallel scratch must be prepared for this thread count");
    let test_before_atomic = config.test_before_atomic;
    let prefetch = config.prefetch;

    let visited = &scratch.visited;
    let current_frontier = &mut scratch.current_frontier;
    let next_frontier = &mut scratch.next_frontier;
    let worker_next = &mut scratch.worker_next;

    pool.install(|| {
        let mut total_edges = 0u64;

        if graph.has_internal_visited() {
            graph.internal_test_and_set_parallel(start_node, false);
        } else {
            visited[start_node as usize].store(true, Ordering::Relaxed);
        }

        current_frontier.push(start_node);
        let mut reached_nodes = 1usize;

        while !current_frontier.is_empty() {
            let chunk_size = (current_frontier.len() + num_threads - 1) / num_threads;
            let chunk_size = chunk_size.max(1);
            let chunk_count = (current_frontier.len() + chunk_size - 1) / chunk_size;

            let level_edges: u64 = worker_next[..chunk_count]
                .par_iter_mut()
                .zip(current_frontier.par_chunks(chunk_size))
                .map(|(local_next, chunk)| {
                    local_next.clear();
                    let mut local_edges = 0u64;
                    let chunk_len = chunk.len();

                    match prefetch {
                        PrefetchMode::None => {
                            if test_before_atomic {
                                for &u in chunk {
                                    graph.for_each_neighbor(u, |v| {
                                        local_edges += 1;
                                        let already = if graph.has_internal_visited() {
                                            graph.internal_test_and_set_parallel(v, true)
                                        } else {
                                            let idx = v as usize;
                                            visited[idx].load(Ordering::Relaxed)
                                                || visited[idx].swap(true, Ordering::AcqRel)
                                        };
                                        if !already {
                                            local_next.push(v);
                                        }
                                    });
                                }
                            } else {
                                for &u in chunk {
                                    graph.for_each_neighbor(u, |v| {
                                        local_edges += 1;
                                        let already = if graph.has_internal_visited() {
                                            graph.internal_test_and_set_parallel(v, false)
                                        } else {
                                            visited[v as usize].swap(true, Ordering::AcqRel)
                                        };
                                        if !already {
                                            local_next.push(v);
                                        }
                                    });
                                }
                            }
                        }
                        PrefetchMode::KAhead(k) => {
                            for idx in 0..chunk_len {
                                if idx + k < chunk_len {
                                    let lookahead = chunk[idx + k];
                                    graph.prefetch_stage1(lookahead);
                                    graph.prefetch_stage2(lookahead);
                                }
                                let u = chunk[idx];
                                graph.for_each_neighbor(u, |v| {
                                    local_edges += 1;
                                    let already = if graph.has_internal_visited() {
                                        graph.internal_test_and_set_parallel(v, test_before_atomic)
                                    } else {
                                        let idx = v as usize;
                                        if test_before_atomic {
                                            visited[idx].load(Ordering::Relaxed)
                                                || visited[idx].swap(true, Ordering::AcqRel)
                                        } else {
                                            visited[idx].swap(true, Ordering::AcqRel)
                                        }
                                    };
                                    if !already {
                                        local_next.push(v);
                                    }
                                });
                            }
                        }
                        PrefetchMode::TwoStage(k) => {
                            for idx in 0..chunk_len {
                                if idx + 2 * k < chunk_len {
                                    graph.prefetch_stage1(chunk[idx + 2 * k]);
                                }
                                if idx + k < chunk_len {
                                    graph.prefetch_stage2(chunk[idx + k]);
                                }
                                let u = chunk[idx];
                                graph.for_each_neighbor(u, |v| {
                                    local_edges += 1;
                                    let already = if graph.has_internal_visited() {
                                        graph.internal_test_and_set_parallel(v, test_before_atomic)
                                    } else {
                                        let idx = v as usize;
                                        if test_before_atomic {
                                            visited[idx].load(Ordering::Relaxed)
                                                || visited[idx].swap(true, Ordering::AcqRel)
                                        } else {
                                            visited[idx].swap(true, Ordering::AcqRel)
                                        }
                                    };
                                    if !already {
                                        local_next.push(v);
                                    }
                                });
                            }
                        }
                        PrefetchMode::Batched(batch_size) => {
                            let mut idx = 0;
                            while idx < chunk_len {
                                let batch_end = (idx + batch_size).min(chunk_len);
                                for b in idx..batch_end {
                                    graph.prefetch_stage1(chunk[b]);
                                }
                                for b in idx..batch_end {
                                    graph.prefetch_stage2(chunk[b]);
                                }
                                for b in idx..batch_end {
                                    let u = chunk[b];
                                    graph.for_each_neighbor(u, |v| {
                                        local_edges += 1;
                                        let already = if graph.has_internal_visited() {
                                            graph.internal_test_and_set_parallel(v, test_before_atomic)
                                        } else {
                                            let idx = v as usize;
                                            if test_before_atomic {
                                                visited[idx].load(Ordering::Relaxed)
                                                    || visited[idx].swap(true, Ordering::AcqRel)
                                            } else {
                                                visited[idx].swap(true, Ordering::AcqRel)
                                            }
                                        };
                                        if !already {
                                            local_next.push(v);
                                        }
                                    });
                                }
                                idx = batch_end;
                            }
                        }
                    }

                    local_edges
                })
                .sum();

            total_edges += level_edges;
            next_frontier.clear();
            for local_next in &worker_next[..chunk_count] {
                next_frontier.extend_from_slice(local_next);
            }
            reached_nodes += next_frontier.len();
            current_frontier.clear();
            std::mem::swap(current_frontier, next_frontier);
        }

        (total_edges, reached_nodes)
    })
}

/// Diagnostic traversal that records exact BFS levels for every reached node.
/// Used to verify Hard Rule 5 (exact matching of reached-node set and per-node levels).
pub fn traverse_and_record_levels<G: GraphLayout>(
    graph: &G,
    start_node: u32,
) -> (u64, usize, Vec<u32>) {
    let n = graph.num_nodes();
    let mut levels = vec![u32::MAX; n];
    if (start_node as usize) >= n {
        return (0, 0, levels);
    }

    let mut current_frontier = vec![start_node];
    let mut next_frontier = Vec::new();
    levels[start_node as usize] = 0;

    let mut total_edges = 0u64;
    let mut reached = 1usize;
    let mut current_level = 0u32;

    while !current_frontier.is_empty() {
        current_level += 1;
        for &u in &current_frontier {
            graph.for_each_neighbor(u, |v| {
                total_edges += 1;
                let v_idx = v as usize;
                if v_idx < n && levels[v_idx] == u32::MAX {
                    levels[v_idx] = current_level;
                    reached += 1;
                    next_frontier.push(v);
                }
            });
        }
        current_frontier.clear();
        std::mem::swap(&mut current_frontier, &mut next_frontier);
    }

    (total_edges, reached, levels)
}
