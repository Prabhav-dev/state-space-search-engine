// src/frontier.rs
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

#[derive(Debug, Clone)]
pub struct PartitionedFrontier {
    queue: Vec<u32>,
}

impl PartitionedFrontier {
    #[inline]
    pub fn new(capacity: usize) -> Self {
        Self { queue: Vec::with_capacity(capacity) }
    }

    #[inline(always)]
    pub fn push(&mut self, node_id: u32) { self.queue.push(node_id); }

    #[inline(always)]
    pub fn pop(&mut self) -> Option<u32> {
        self.queue.pop()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.queue.is_empty() }

    #[inline]
    pub fn clear(&mut self) { self.queue.clear(); }

    pub fn inner_vec(&self) -> &[u32] { &self.queue }
}

/// Shared lock-free atomic CAS frontier for testing hub-node contention (H2)[cite: 2, 3]
pub struct SharedCasFrontier {
    queue: Vec<AtomicU32>,
    head: AtomicUsize,
    tail: AtomicUsize,
    capacity: usize,
}

impl SharedCasFrontier {
    pub fn new(capacity: usize) -> Self {
        let mut queue = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            queue.push(AtomicU32::new(u32::MAX));
        }
        Self {
            queue,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            capacity,
        }
    }

    #[inline]
    pub fn push(&self, node_id: u32) -> bool {
        let tail = self.tail.fetch_add(1, Ordering::AcqRel);
        if tail < self.capacity {
            self.queue[tail].store(node_id, Ordering::Release);
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn pop(&self) -> Option<u32> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            if head >= tail { return None; }
            if self.head.compare_exchange_weak(head, head + 1, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                // Spin until value store has completed to fix the sentinel race
                loop {
                    let val = self.queue[head].load(Ordering::Acquire);
                    if val != u32::MAX {
                        return Some(val);
                    }
                    std::hint::spin_loop();
                }
            }
        }
    }

    pub fn clear(&self) {
        let tail = self.tail.load(Ordering::Acquire);
        for i in 0..tail.min(self.capacity) {
            self.queue[i].store(u32::MAX, Ordering::Release);
        }
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
    }
}