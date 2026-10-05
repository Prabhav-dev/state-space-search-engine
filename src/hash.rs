/// A 64-bit register tracking path state via Zobrist-style hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PathHasher {
    state: u64,
}

impl PathHasher {
    /// Initialize the hasher with a starting seed or zero.
    #[inline(always)]
    pub fn new(initial_state: u64) -> Self {
        Self { state: initial_state }
    }

    /// Updates the path hash state by combining a node's random key.
    /// An optional bit rotation helps preserve partial order information[cite: 2].
    #[inline(always)]
    pub fn update(&mut self, node_key: u64, rotate_bits: u32) {
        if rotate_bits > 0 {
            self.state = self.state.rotate_left(rotate_bits);
        }
        self.state ^= node_key;
    }

    /// Retrieve the final 64-bit hash for deduplication/transposition table lookup.
    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.state
    }
}

/// Fixed-size probabilistic bitstate visited set for Zobrist-style path deduplication (H3).
pub struct ZobristVisitedSet {
    bits: Vec<u64>,
    mask: usize,
    bits_count: usize,
}

impl ZobristVisitedSet {
    /// Creates a bitstate table of size 2^power_of_two_bits.
    pub fn new(power_of_two_bits: usize) -> Self {
        let bits_count = 1usize << power_of_two_bits;
        let u64_count = (bits_count + 63) / 64;
        Self {
            bits: vec![0; u64_count],
            mask: bits_count - 1,
            bits_count,
        }
    }

    #[inline(always)]
    pub fn test_and_set(&mut self, hash_val: u64) -> bool {
        let idx = (hash_val as usize) & self.mask;
        let word_idx = idx >> 6;
        let bit_mask = 1u64 << (idx & 63);
        let old = self.bits[word_idx];
        self.bits[word_idx] = old | bit_mask;
        (old & bit_mask) != 0
    }

    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u64>()
    }

    pub fn bit_capacity(&self) -> usize {
        self.bits_count
    }

    pub fn clear(&mut self) {
        self.bits.fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_hasher_properties() {
        let mut hasher = PathHasher::new(0);
        let key = 0x9e3779b97f4a7c15;

        hasher.update(key, 0);
        assert_eq!(hasher.hash(), key);

        // Self-inverse check for plain XOR (without rotation)
        hasher.update(key, 0);
        assert_eq!(hasher.hash(), 0, "Plain XOR of the same key twice should cancel out");
    }

    #[test]
    fn test_zobrist_visited_set() {
        let mut set = ZobristVisitedSet::new(16); // 64K bits
        let key = 0x9e3779b97f4a7c15;

        assert!(!set.test_and_set(key), "First test_and_set should return false (unvisited)");
        assert!(set.test_and_set(key), "Second test_and_set should return true (already visited)");
    }
}