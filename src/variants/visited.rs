//! Pre-allocated, zero-allocation visited buffers shared across variants.

use std::sync::atomic::{AtomicBool, Ordering};

/// Reusable scratch space for single-threaded level-synchronous BFS.
pub struct SharedSingleScratch {
    pub visited: Vec<u64>,
    pub cur: Vec<u32>,
    pub next: Vec<u32>,
}

impl SharedSingleScratch {
    pub fn new(capacity: usize) -> Self {
        Self {
            visited: vec![0; (capacity + 63) / 64],
            cur: Vec::with_capacity(capacity),
            next: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.visited.fill(0);
        self.cur.clear();
        self.next.clear();
    }

    #[inline(always)]
    pub fn test_and_set(&mut self, v: u32) -> bool {
        let w = (v >> 6) as usize;
        let m = 1u64 << (v & 63);
        let old = self.visited[w];
        if (old & m) == 0 {
            self.visited[w] = old | m;
            false
        } else {
            true
        }
    }

    #[inline(always)]
    pub fn is_visited(&self, v: u32) -> bool {
        let w = (v >> 6) as usize;
        let m = 1u64 << (v & 63);
        (self.visited[w] & m) != 0
    }
}

/// Reusable scratch space for multi-threaded parallel level-synchronous BFS.
pub struct SharedParallelScratch {
    pub visited: Vec<AtomicBool>,
    pub current_frontier: Vec<u32>,
    pub next_frontier: Vec<u32>,
    pub worker_next: Vec<Vec<u32>>,
}

impl SharedParallelScratch {
    pub fn new(capacity: usize) -> Self {
        let visited = (0..capacity).map(|_| AtomicBool::new(false)).collect();
        Self {
            visited,
            current_frontier: Vec::with_capacity(capacity),
            next_frontier: Vec::with_capacity(capacity),
            worker_next: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn prepare_frontiers(&mut self, worker_count: usize) {
        if self.worker_next.len() != worker_count {
            self.worker_next = (0..worker_count)
                .map(|_| Vec::with_capacity(self.visited.len()))
                .collect();
        }
        self.current_frontier.clear();
        self.next_frontier.clear();
        for worker_frontier in &mut self.worker_next {
            worker_frontier.clear();
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        for v in &self.visited {
            v.store(false, Ordering::Relaxed);
        }
        self.current_frontier.clear();
        self.next_frontier.clear();
        for worker_frontier in &mut self.worker_next {
            worker_frontier.clear();
        }
    }

    #[inline(always)]
    pub fn test_and_set(&self, v: u32, test_before_atomic: bool) -> bool {
        let idx = v as usize;
        if test_before_atomic {
            if self.visited[idx].load(Ordering::Relaxed) {
                return true;
            }
        }
        self.visited[idx].swap(true, Ordering::AcqRel)
    }

    #[inline(always)]
    pub fn is_visited(&self, v: u32) -> bool {
        self.visited[v as usize].load(Ordering::Relaxed)
    }
}
