use crate::types::memory::tier::MemoryTier;
use crate::types::ownership::mutation::MutationSemantics;

use super::hotness::Hotness;
use super::kind::BlockKind;
use super::lifetime::Lifetime;

pub const fn default_mutation_semantics(kind: BlockKind) -> MutationSemantics {
    match kind {
        BlockKind::Weight => MutationSemantics::Immutable,
        BlockKind::KvPage => MutationSemantics::AppendOnly,
        BlockKind::Activation | BlockKind::Logits | BlockKind::Workspace => {
            MutationSemantics::Ephemeral
        }
        BlockKind::TokenState
        | BlockKind::SamplerState
        | BlockKind::Queue
        | BlockKind::Ledger
        | BlockKind::Metadata
        | BlockKind::TransportBuffer => MutationSemantics::SingleWriter,
    }
}

pub const fn default_lifetime(kind: BlockKind) -> Lifetime {
    match kind {
        BlockKind::Weight => Lifetime::Static,
        BlockKind::KvPage
        | BlockKind::TokenState
        | BlockKind::SamplerState
        | BlockKind::Queue
        | BlockKind::Ledger
        | BlockKind::Metadata
        | BlockKind::TransportBuffer => Lifetime::Request,
        BlockKind::Activation | BlockKind::Logits => Lifetime::Token,
        BlockKind::Workspace => Lifetime::Scratch,
    }
}

pub const fn default_hotness(tier: MemoryTier) -> Hotness {
    match tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => Hotness::Hot,
        MemoryTier::PinnedDram | MemoryTier::Dram | MemoryTier::Cxl => Hotness::Warm,
        MemoryTier::Disk => Hotness::Cold,
    }
}
