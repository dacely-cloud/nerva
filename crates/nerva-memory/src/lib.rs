#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::collections::{BTreeMap, VecDeque};

use nerva_core::{
    AllocationId, BlockKind, DType, GlobalBlockAddress, LayoutId, MemoryDomainId, MemoryTier,
    NervaError, ResidencyState, ResidentBlock, ResidentBlockId, ResidentBlockKind, Result,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArenaKind {
    Device,
    PinnedHost,
    Host,
}

impl ArenaKind {
    pub const fn tier(self) -> MemoryTier {
        match self {
            Self::Device => MemoryTier::Vram,
            Self::PinnedHost => MemoryTier::PinnedDram,
            Self::Host => MemoryTier::Dram,
        }
    }

    pub const fn domain(self) -> MemoryDomainId {
        MemoryDomainId::for_tier(self.tier())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AllocationPhase {
    Initialization,
    HotPath,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaReservation {
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}

#[derive(Clone, Debug)]
pub struct HostArena {
    bytes: Vec<u8>,
    used: usize,
}

impl HostArena {
    pub fn new(capacity: usize) -> Self {
        Self {
            bytes: vec![0; capacity],
            used: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.bytes.len()
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.bytes.len() - self.used
    }

    pub fn reserve(&mut self, bytes: usize, align: usize) -> Result<ArenaReservation> {
        let align = align.max(1);
        let offset = self.used.next_multiple_of(align);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes,
                reason: "arena offset overflow".to_string(),
            })?;
        if end > self.bytes.len() {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: "host arena exhausted".to_string(),
            });
        }
        self.used = end;
        Ok(ArenaReservation {
            offset,
            bytes,
            align,
        })
    }

    pub fn reset(&mut self) {
        self.used = 0;
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaCheckpoint {
    used: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaRegion {
    pub name: &'static str,
    pub kind: ArenaKind,
    pub tier: MemoryTier,
    pub address: GlobalBlockAddress,
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArena {
    kind: ArenaKind,
    allocation: AllocationId,
    capacity_bytes: usize,
    used_bytes: usize,
}

impl StaticArena {
    pub const fn new(kind: ArenaKind, allocation: AllocationId, capacity_bytes: usize) -> Self {
        Self {
            kind,
            allocation,
            capacity_bytes,
            used_bytes: 0,
        }
    }

    pub const fn kind(&self) -> ArenaKind {
        self.kind
    }

    pub const fn tier(&self) -> MemoryTier {
        self.kind.tier()
    }

    pub const fn domain(&self) -> MemoryDomainId {
        self.kind.domain()
    }

    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    pub const fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    pub const fn used(&self) -> usize {
        self.used_bytes
    }

    pub const fn remaining(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    pub const fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint {
            used: self.used_bytes,
        }
    }

    pub fn restore(&mut self, checkpoint: ArenaCheckpoint) -> Result<()> {
        if checkpoint.used > self.used_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "arena checkpoint is ahead of current usage".to_string(),
            });
        }
        self.used_bytes = checkpoint.used;
        Ok(())
    }

    pub fn reserve(
        &mut self,
        name: &'static str,
        bytes: usize,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ArenaRegion> {
        if phase == AllocationPhase::HotPath {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: "static arena allocation attempted during hot path".to_string(),
            });
        }
        let align = align.max(1);
        let offset = self.used_bytes.next_multiple_of(align);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes,
                reason: "static arena offset overflow".to_string(),
            })?;
        if end > self.capacity_bytes {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!("static {:?} arena exhausted", self.kind),
            });
        }
        self.used_bytes = end;
        Ok(ArenaRegion {
            name,
            kind: self.kind,
            tier: self.tier(),
            address: GlobalBlockAddress {
                domain: self.domain(),
                allocation: self.allocation,
                offset: offset as u64,
            },
            offset,
            bytes,
            align,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaSet {
    device: StaticArena,
    pinned_host: StaticArena,
    host: StaticArena,
}

impl StaticArenaSet {
    pub const fn new(device_bytes: usize, pinned_host_bytes: usize, host_bytes: usize) -> Self {
        Self {
            device: StaticArena::new(ArenaKind::Device, AllocationId(1), device_bytes),
            pinned_host: StaticArena::new(
                ArenaKind::PinnedHost,
                AllocationId(2),
                pinned_host_bytes,
            ),
            host: StaticArena::new(ArenaKind::Host, AllocationId(3), host_bytes),
        }
    }

    pub const fn device(&self) -> &StaticArena {
        &self.device
    }

    pub const fn pinned_host(&self) -> &StaticArena {
        &self.pinned_host
    }

    pub const fn host(&self) -> &StaticArena {
        &self.host
    }

    pub fn arena_mut(&mut self, kind: ArenaKind) -> &mut StaticArena {
        match kind {
            ArenaKind::Device => &mut self.device,
            ArenaKind::PinnedHost => &mut self.pinned_host,
            ArenaKind::Host => &mut self.host,
        }
    }

    pub fn reserve(
        &mut self,
        kind: ArenaKind,
        name: &'static str,
        bytes: usize,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ArenaRegion> {
        self.arena_mut(kind).reserve(name, bytes, align, phase)
    }

    pub fn reserve_resident_block(
        &mut self,
        registry: &mut BlockRegistry,
        kind: ArenaKind,
        name: &'static str,
        request: BlockAllocationRequest,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ResidentBlockId> {
        if request.tier != kind.tier() {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "arena kind {:?} cannot reserve block requested for {:?}",
                    kind, request.tier
                ),
            });
        }

        let checkpoint = self.arena_mut(kind).checkpoint();
        let region = self.reserve(kind, name, request.bytes, align, phase)?;
        match registry.allocate(request) {
            Ok(id) => {
                registry.bind_address(id, region.address)?;
                Ok(id)
            }
            Err(err) => {
                let _ = self.arena_mut(kind).restore(checkpoint);
                Err(err)
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvPageSpec {
    pub layer_id: u32,
    pub head_group_id: u32,
    pub block_size_tokens: u32,
    pub page_bytes: usize,
    pub tier: MemoryTier,
    pub arena_kind: ArenaKind,
    pub align: usize,
}

impl KvPageSpec {
    pub const fn new(
        layer_id: u32,
        head_group_id: u32,
        block_size_tokens: u32,
        page_bytes: usize,
        tier: MemoryTier,
        arena_kind: ArenaKind,
        align: usize,
    ) -> Self {
        Self {
            layer_id,
            head_group_id,
            block_size_tokens,
            page_bytes,
            tier,
            arena_kind,
            align,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct KvPrefixKey {
    pub hash: [u8; 32],
    pub group_id: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvPageHandle {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPageDescriptor {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
    pub layer_id: u32,
    pub head_group_id: u32,
    pub token_start: u32,
    pub token_count: u32,
    pub block_size_tokens: u32,
    pub ref_count: u32,
    pub prefix_key: Option<KvPrefixKey>,
    pub prefix_tokens: Option<u32>,
    pub last_use: u64,
    pub next_use: Option<u64>,
    is_free: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPagePool {
    pages: Vec<KvPageDescriptor>,
    free_pages: VecDeque<u32>,
    prefix_cache: BTreeMap<KvPrefixKey, u32>,
}

impl KvPagePool {
    pub fn preallocate(
        arenas: &mut StaticArenaSet,
        registry: &mut BlockRegistry,
        num_pages: u32,
        spec: KvPageSpec,
    ) -> Result<Self> {
        if spec.tier != spec.arena_kind.tier() {
            return Err(NervaError::InvalidArgument {
                reason: "KV page spec tier and arena kind do not match".to_string(),
            });
        }
        if spec.block_size_tokens == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "KV page block size must be non-zero".to_string(),
            });
        }

        let mut pages = Vec::with_capacity(num_pages as usize);
        let mut free_pages = VecDeque::with_capacity(num_pages as usize);
        for page_index in 0..num_pages {
            let block_id = arenas.reserve_resident_block(
                registry,
                spec.arena_kind,
                "kv-page",
                BlockAllocationRequest::new(BlockKind::KvPage, spec.tier, spec.page_bytes),
                spec.align,
                AllocationPhase::Initialization,
            )?;
            registry.mark_ready(block_id)?;
            pages.push(KvPageDescriptor {
                page_index,
                block_id,
                layer_id: spec.layer_id,
                head_group_id: spec.head_group_id,
                token_start: 0,
                token_count: 0,
                block_size_tokens: spec.block_size_tokens,
                ref_count: 0,
                prefix_key: None,
                prefix_tokens: None,
                last_use: 0,
                next_use: None,
                is_free: true,
            });
            free_pages.push_back(page_index);
        }

        Ok(Self {
            pages,
            free_pages,
            prefix_cache: BTreeMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    pub fn num_free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn usage(&self) -> f32 {
        if self.pages.is_empty() {
            0.0
        } else {
            1.0 - (self.num_free_pages() as f32 / self.pages.len() as f32)
        }
    }

    pub fn page(&self, page_index: u32) -> Option<&KvPageDescriptor> {
        self.pages.get(page_index as usize)
    }

    pub fn lookup_cached(&self, key: KvPrefixKey) -> Option<KvPageHandle> {
        let page_index = *self.prefix_cache.get(&key)?;
        let page = self.page(page_index)?;
        Some(KvPageHandle {
            page_index,
            block_id: page.block_id,
        })
    }

    pub fn allocate_page(
        &mut self,
        token_start: u32,
        token_count: u32,
        step: u64,
    ) -> Result<KvPageHandle> {
        let page_index =
            self.free_pages
                .pop_front()
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: 0,
                    reason: "KV page pool exhausted".to_string(),
                })?;
        let page = self.page_mut(page_index)?;
        if token_count > page.block_size_tokens {
            self.free_pages.push_front(page_index);
            return Err(NervaError::InvalidArgument {
                reason: "KV page token count exceeds page block size".to_string(),
            });
        }
        page.token_start = token_start;
        page.token_count = token_count;
        page.ref_count = 1;
        page.last_use = step;
        page.next_use = None;
        page.is_free = false;
        Ok(KvPageHandle {
            page_index,
            block_id: page.block_id,
        })
    }

    pub fn retain_page(&mut self, page_index: u32, step: u64) -> Result<KvPageHandle> {
        let was_free = self
            .page(page_index)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown KV page index {page_index}"),
            })?
            .is_free;
        if was_free {
            self.free_pages.retain(|free| *free != page_index);
        }
        let page = self.page_mut(page_index)?;
        page.is_free = false;
        page.ref_count =
            page.ref_count
                .checked_add(1)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: "KV page reference count overflow".to_string(),
                })?;
        page.last_use = step;
        Ok(KvPageHandle {
            page_index,
            block_id: page.block_id,
        })
    }

    pub fn retain_cached(&mut self, key: KvPrefixKey, step: u64) -> Result<Option<KvPageHandle>> {
        let Some(page_index) = self.prefix_cache.get(&key).copied() else {
            return Ok(None);
        };
        self.retain_page(page_index, step).map(Some)
    }

    pub fn release_page(&mut self, page_index: u32, step: u64) -> Result<()> {
        let page = self.page_mut(page_index)?;
        if page.ref_count == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "KV page released with zero references".to_string(),
            });
        }
        page.ref_count -= 1;
        page.last_use = step;
        if page.ref_count == 0 && !page.is_free {
            page.is_free = true;
            page.token_count = 0;
            self.free_pages.push_back(page_index);
        }
        Ok(())
    }

    pub fn cache_page(
        &mut self,
        page_index: u32,
        key: KvPrefixKey,
        prefix_tokens: u32,
    ) -> Result<()> {
        let old_key = {
            let page = self.page_mut(page_index)?;
            if prefix_tokens == 0 {
                return Err(NervaError::InvalidArgument {
                    reason: "cached KV prefix must cover at least one token".to_string(),
                });
            }
            let old_key = page.prefix_key;
            page.prefix_key = Some(key);
            page.prefix_tokens = Some(prefix_tokens);
            old_key
        };
        if let Some(old_key) = old_key {
            self.prefix_cache.remove(&old_key);
        }
        self.prefix_cache.insert(key, page_index);
        Ok(())
    }

    pub fn evict_cached_page(&mut self, page_index: u32) -> Result<Option<KvPrefixKey>> {
        let old_key = {
            let page = self.page_mut(page_index)?;
            let old_key = page.prefix_key.take();
            page.prefix_tokens = None;
            old_key
        };
        if let Some(old_key) = old_key {
            self.prefix_cache.remove(&old_key);
        }
        Ok(old_key)
    }

    fn page_mut(&mut self, page_index: u32) -> Result<&mut KvPageDescriptor> {
        self.pages
            .get_mut(page_index as usize)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown KV page index {page_index}"),
            })
    }
}

pub fn resident_block_for_reservation(
    id: ResidentBlockId,
    kind: ResidentBlockKind,
    reservation: ArenaReservation,
) -> ResidentBlock {
    ResidentBlock::new(id, kind, MemoryTier::Dram, reservation.bytes).with_address(
        GlobalBlockAddress {
            domain: MemoryDomainId::CPU_DRAM,
            allocation: AllocationId(id.0),
            offset: reservation.offset as u64,
        },
    )
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TierAccount {
    pub tier: MemoryTier,
    pub capacity_bytes: usize,
    pub used_bytes: usize,
}

impl TierAccount {
    pub const fn remaining_bytes(self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockAllocationRequest {
    pub kind: BlockKind,
    pub tier: MemoryTier,
    pub bytes: usize,
    pub dtype: DType,
    pub layout: LayoutId,
}

impl BlockAllocationRequest {
    pub const fn new(kind: BlockKind, tier: MemoryTier, bytes: usize) -> Self {
        Self {
            kind,
            tier,
            bytes,
            dtype: DType::U8,
            layout: LayoutId(0),
        }
    }

    pub const fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub const fn with_layout(mut self, layout: LayoutId) -> Self {
        self.layout = layout;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockRegistry {
    next_id: u64,
    accounts: BTreeMap<MemoryTier, TierAccount>,
    blocks: BTreeMap<ResidentBlockId, ResidentBlock>,
}

impl BlockRegistry {
    pub fn new(accounts: impl IntoIterator<Item = (MemoryTier, usize)>) -> Self {
        let mut registry = Self {
            next_id: 1,
            accounts: BTreeMap::new(),
            blocks: BTreeMap::new(),
        };
        for (tier, capacity_bytes) in accounts {
            registry.accounts.insert(
                tier,
                TierAccount {
                    tier,
                    capacity_bytes,
                    used_bytes: 0,
                },
            );
        }
        registry
    }

    pub fn account(&self, tier: MemoryTier) -> Option<TierAccount> {
        self.accounts.get(&tier).copied()
    }

    pub fn used_bytes(&self, tier: MemoryTier) -> usize {
        self.account(tier).map_or(0, |account| account.used_bytes)
    }

    pub fn remaining_bytes(&self, tier: MemoryTier) -> Option<usize> {
        self.account(tier).map(|account| account.remaining_bytes())
    }

    pub fn block(&self, id: ResidentBlockId) -> Option<&ResidentBlock> {
        self.blocks.get(&id)
    }

    pub fn block_mut(&mut self, id: ResidentBlockId) -> Option<&mut ResidentBlock> {
        self.blocks.get_mut(&id)
    }

    pub fn allocate(&mut self, request: BlockAllocationRequest) -> Result<ResidentBlockId> {
        self.reserve_tier(request.tier, request.bytes)?;
        let id = ResidentBlockId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: request.bytes,
                reason: "resident block id overflow".to_string(),
            })?;

        let block = ResidentBlock::new(id, request.kind, request.tier, request.bytes).with_shape(
            request.dtype,
            nerva_core::BlockShape::scalar(),
            request.layout,
        );
        self.blocks.insert(id, block);
        Ok(id)
    }

    pub fn mark_ready(&mut self, id: ResidentBlockId) -> Result<()> {
        let block = self.require_block_mut(id)?;
        block.mark_ready();
        Ok(())
    }

    pub fn transition(&mut self, id: ResidentBlockId, state: ResidencyState) -> Result<()> {
        let block = self.require_block_mut(id)?;
        block.state = state;
        Ok(())
    }

    pub fn move_block(
        &mut self,
        id: ResidentBlockId,
        to_tier: MemoryTier,
        allocation: AllocationId,
        offset: u64,
    ) -> Result<()> {
        let (from_tier, bytes) = {
            let block = self.require_block(id)?;
            (block.tier, block.bytes)
        };

        if from_tier == to_tier {
            let block = self.require_block_mut(id)?;
            block.address = GlobalBlockAddress {
                domain: MemoryDomainId::for_tier(to_tier),
                allocation,
                offset,
            };
            block.memory_domain = MemoryDomainId::for_tier(to_tier);
            return Ok(());
        }

        self.reserve_tier(to_tier, bytes)?;
        self.release_tier(from_tier, bytes);

        let block = self.require_block_mut(id)?;
        block.tier = to_tier;
        block.address = GlobalBlockAddress {
            domain: MemoryDomainId::for_tier(to_tier),
            allocation,
            offset,
        };
        block.memory_domain = MemoryDomainId::for_tier(to_tier);
        block.state = ResidencyState::Prefetching;
        Ok(())
    }

    pub fn bind_address(&mut self, id: ResidentBlockId, address: GlobalBlockAddress) -> Result<()> {
        let block = self.require_block_mut(id)?;
        let address_tier =
            memory_tier_for_domain(address.domain).ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown memory domain {}", address.domain.0),
            })?;
        if block.tier != address_tier {
            return Err(NervaError::InvalidArgument {
                reason: "block tier and arena address domain do not match".to_string(),
            });
        }
        block.address = address;
        block.memory_domain = address.domain;
        Ok(())
    }

    fn require_block(&self, id: ResidentBlockId) -> Result<&ResidentBlock> {
        self.blocks
            .get(&id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown resident block id {}", id.0),
            })
    }

    fn require_block_mut(&mut self, id: ResidentBlockId) -> Result<&mut ResidentBlock> {
        self.blocks
            .get_mut(&id)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("unknown resident block id {}", id.0),
            })
    }

    fn reserve_tier(&mut self, tier: MemoryTier, bytes: usize) -> Result<()> {
        let account = self
            .accounts
            .get_mut(&tier)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("memory tier {tier:?} is not configured"),
            })?;
        let new_used =
            account
                .used_bytes
                .checked_add(bytes)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes,
                    reason: "tier accounting overflow".to_string(),
                })?;
        if new_used > account.capacity_bytes {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!("memory tier {tier:?} exhausted"),
            });
        }
        account.used_bytes = new_used;
        Ok(())
    }

    fn release_tier(&mut self, tier: MemoryTier, bytes: usize) {
        if let Some(account) = self.accounts.get_mut(&tier) {
            account.used_bytes = account.used_bytes.saturating_sub(bytes);
        }
    }
}

fn memory_tier_for_domain(domain: MemoryDomainId) -> Option<MemoryTier> {
    Some(match domain {
        MemoryDomainId::GPU_VRAM => MemoryTier::Vram,
        MemoryDomainId::PINNED_DRAM => MemoryTier::PinnedDram,
        MemoryDomainId::CPU_DRAM => MemoryTier::Dram,
        MemoryDomainId::SHARED_HBM_OR_LPDDR => MemoryTier::SharedHbmOrLpddr,
        MemoryDomainId::CXL => MemoryTier::Cxl,
        MemoryDomainId::DISK => MemoryTier::Disk,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_arena_respects_alignment() {
        let mut arena = HostArena::new(1024);
        let a = arena.reserve(3, 1).unwrap();
        let b = arena.reserve(8, 64).unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset % 64, 0);
        assert!(arena.used() >= b.offset + 8);
    }

    #[test]
    fn registry_tracks_tier_capacity() {
        let mut registry = BlockRegistry::new([(MemoryTier::Dram, 128), (MemoryTier::Vram, 64)]);
        let first = registry
            .allocate(BlockAllocationRequest::new(
                BlockKind::Weight,
                MemoryTier::Dram,
                96,
            ))
            .unwrap();
        assert_eq!(first, ResidentBlockId(1));
        assert_eq!(registry.used_bytes(MemoryTier::Dram), 96);
        assert_eq!(registry.remaining_bytes(MemoryTier::Dram), Some(32));

        let err = registry
            .allocate(BlockAllocationRequest::new(
                BlockKind::Activation,
                MemoryTier::Dram,
                64,
            ))
            .unwrap_err();
        assert!(matches!(err, NervaError::AllocationFailed { .. }));
        assert_eq!(registry.used_bytes(MemoryTier::Dram), 96);
    }

    #[test]
    fn registry_moves_blocks_between_tiers_with_accounting() {
        let mut registry = BlockRegistry::new([(MemoryTier::Dram, 128), (MemoryTier::Vram, 128)]);
        let id = registry
            .allocate(BlockAllocationRequest::new(
                BlockKind::KvPage,
                MemoryTier::Dram,
                64,
            ))
            .unwrap();
        registry.mark_ready(id).unwrap();
        registry
            .move_block(id, MemoryTier::Vram, AllocationId(99), 256)
            .unwrap();

        let block = registry.block(id).unwrap();
        assert_eq!(block.tier, MemoryTier::Vram);
        assert_eq!(block.state, ResidencyState::Prefetching);
        assert_eq!(block.address.domain, MemoryDomainId::GPU_VRAM);
        assert_eq!(block.address.allocation, AllocationId(99));
        assert_eq!(block.address.offset, 256);
        assert_eq!(registry.used_bytes(MemoryTier::Dram), 0);
        assert_eq!(registry.used_bytes(MemoryTier::Vram), 64);
    }

    #[test]
    fn host_reservation_becomes_dram_block_address() {
        let reservation = ArenaReservation {
            offset: 32,
            bytes: 16,
            align: 8,
        };
        let block =
            resident_block_for_reservation(ResidentBlockId(77), BlockKind::Metadata, reservation);
        assert_eq!(block.tier, MemoryTier::Dram);
        assert_eq!(block.address.domain, MemoryDomainId::CPU_DRAM);
        assert_eq!(block.address.offset, 32);
    }

    #[test]
    fn static_arena_reserves_stable_aligned_regions() {
        let mut arena = StaticArena::new(ArenaKind::Device, AllocationId(10), 1024);
        let first = arena
            .reserve("weights", 33, 1, AllocationPhase::Initialization)
            .unwrap();
        let second = arena
            .reserve("workspace", 64, 128, AllocationPhase::Initialization)
            .unwrap();

        assert_eq!(first.address.domain, MemoryDomainId::GPU_VRAM);
        assert_eq!(first.address.allocation, AllocationId(10));
        assert_eq!(first.offset, 0);
        assert_eq!(second.offset % 128, 0);
        assert_eq!(second.address.offset, second.offset as u64);
        assert!(arena.used() >= second.offset + second.bytes);
    }

    #[test]
    fn static_arena_rejects_hot_path_reservation() {
        let mut arena = StaticArena::new(ArenaKind::PinnedHost, AllocationId(22), 1024);
        let err = arena
            .reserve("token-ring", 64, 64, AllocationPhase::HotPath)
            .unwrap_err();
        assert!(matches!(err, NervaError::AllocationFailed { .. }));
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn static_arena_checkpoint_restore_rewinds_scratch() {
        let mut arena = StaticArena::new(ArenaKind::Host, AllocationId(33), 256);
        let _metadata = arena
            .reserve("metadata", 32, 8, AllocationPhase::Initialization)
            .unwrap();
        let checkpoint = arena.checkpoint();
        let _scratch = arena
            .reserve("scratch", 128, 16, AllocationPhase::Initialization)
            .unwrap();
        assert!(arena.used() > checkpoint.used);
        arena.restore(checkpoint).unwrap();
        assert_eq!(arena.used(), checkpoint.used);
    }

    #[test]
    fn arena_set_reserves_blocks_and_binds_addresses() {
        let mut arenas = StaticArenaSet::new(512, 512, 512);
        let mut registry = BlockRegistry::new([
            (MemoryTier::Vram, 512),
            (MemoryTier::PinnedDram, 512),
            (MemoryTier::Dram, 512),
        ]);

        let id = arenas
            .reserve_resident_block(
                &mut registry,
                ArenaKind::Device,
                "kv-page",
                BlockAllocationRequest::new(BlockKind::KvPage, MemoryTier::Vram, 128),
                128,
                AllocationPhase::Initialization,
            )
            .unwrap();
        let block = registry.block(id).unwrap();
        assert_eq!(block.tier, MemoryTier::Vram);
        assert_eq!(block.address.domain, MemoryDomainId::GPU_VRAM);
        assert_eq!(block.address.allocation, AllocationId(1));
        assert_eq!(block.address.offset, 0);
        assert_eq!(arenas.device().used(), 128);
    }

    #[test]
    fn arena_set_rewinds_if_registry_rejects_block() {
        let mut arenas = StaticArenaSet::new(512, 0, 0);
        let mut registry = BlockRegistry::new([(MemoryTier::Vram, 64)]);

        let err = arenas
            .reserve_resident_block(
                &mut registry,
                ArenaKind::Device,
                "too-large",
                BlockAllocationRequest::new(BlockKind::Activation, MemoryTier::Vram, 128),
                1,
                AllocationPhase::Initialization,
            )
            .unwrap_err();
        assert!(matches!(err, NervaError::AllocationFailed { .. }));
        assert_eq!(arenas.device().used(), 0);
        assert_eq!(registry.used_bytes(MemoryTier::Vram), 0);
    }

    #[test]
    fn kv_page_pool_preallocates_resident_blocks() {
        let mut arenas = StaticArenaSet::new(1024, 0, 0);
        let mut registry = BlockRegistry::new([(MemoryTier::Vram, 1024)]);
        let pool = KvPagePool::preallocate(
            &mut arenas,
            &mut registry,
            4,
            KvPageSpec::new(2, 1, 16, 128, MemoryTier::Vram, ArenaKind::Device, 128),
        )
        .unwrap();

        assert_eq!(pool.len(), 4);
        assert_eq!(pool.num_free_pages(), 4);
        assert_eq!(registry.used_bytes(MemoryTier::Vram), 512);
        assert_eq!(arenas.device().used(), 512);
        assert!(
            pool.page(0)
                .and_then(|page| registry.block(page.block_id))
                .is_some_and(|block| block.state == ResidencyState::Ready)
        );
    }

    #[test]
    fn kv_page_pool_allocates_and_releases_pages() {
        let mut arenas = StaticArenaSet::new(512, 0, 0);
        let mut registry = BlockRegistry::new([(MemoryTier::Vram, 512)]);
        let mut pool = KvPagePool::preallocate(
            &mut arenas,
            &mut registry,
            2,
            KvPageSpec::new(0, 0, 16, 128, MemoryTier::Vram, ArenaKind::Device, 64),
        )
        .unwrap();

        let handle = pool.allocate_page(32, 8, 7).unwrap();
        assert_eq!(pool.num_free_pages(), 1);
        let page = pool.page(handle.page_index).unwrap();
        assert_eq!(page.token_start, 32);
        assert_eq!(page.token_count, 8);
        assert_eq!(page.ref_count, 1);
        assert_eq!(page.last_use, 7);

        pool.release_page(handle.page_index, 8).unwrap();
        assert_eq!(pool.num_free_pages(), 2);
        assert_eq!(pool.page(handle.page_index).unwrap().ref_count, 0);
    }

    #[test]
    fn kv_page_pool_caches_prefix_keys_and_retain_hits() {
        let mut arenas = StaticArenaSet::new(512, 0, 0);
        let mut registry = BlockRegistry::new([(MemoryTier::Vram, 512)]);
        let mut pool = KvPagePool::preallocate(
            &mut arenas,
            &mut registry,
            2,
            KvPageSpec::new(0, 3, 16, 128, MemoryTier::Vram, ArenaKind::Device, 64),
        )
        .unwrap();

        let key = KvPrefixKey {
            hash: [7; 32],
            group_id: 3,
        };
        let handle = pool.allocate_page(0, 16, 1).unwrap();
        pool.cache_page(handle.page_index, key, 16).unwrap();
        assert_eq!(pool.lookup_cached(key), Some(handle));

        pool.release_page(handle.page_index, 2).unwrap();
        assert_eq!(pool.num_free_pages(), 2);
        let retained = pool.retain_cached(key, 3).unwrap().unwrap();
        assert_eq!(retained, handle);
        assert_eq!(pool.num_free_pages(), 1);
        assert_eq!(pool.page(handle.page_index).unwrap().ref_count, 1);

        assert_eq!(
            pool.evict_cached_page(handle.page_index).unwrap(),
            Some(key)
        );
        assert_eq!(pool.lookup_cached(key), None);
    }
}
