use std::mem::size_of;

/// Maximum number of outbound edges stored inline inside a StateNode record.
pub const MAX_INLINE_EDGES: usize = 3;

/// Fixed-size node record designed to fit multiple items cleanly into cache lines.
/// Aligned to 32 bytes to match hardware cache assumptions.
#[repr(C)]
#[repr(align(32))]
#[derive(Debug, Clone, Copy)]
pub struct StateNode {
    pub node_id: u32,
    pub structural_mask: u32,
    pub outbound_count: u32,
    pub metadata_flags: u32,
    pub outbound_links: [u32; 4], // 3 inline edges + 1 overflow index (when outbound_count > 3)
}

impl StateNode {
    /// Compile-time verification that the node size matches the 32-byte target.
    pub const SIZE: usize = size_of::<Self>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_node_layout() {
        assert_eq!(size_of::<StateNode>(), 32, "StateNode must be exactly 32 bytes");
        assert_eq!(std::mem::align_of::<StateNode>(), 32, "StateNode must be 32-byte aligned");
        assert_eq!(MAX_INLINE_EDGES, 3, "MAX_INLINE_EDGES must be 3");
    }
}