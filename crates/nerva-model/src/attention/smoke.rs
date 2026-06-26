use nerva_core::types::error::Result;
use nerva_core::types::memory::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::attention::block::KvAttentionBlock;
use crate::attention::exact::run::exact_blockwise_attention_into;
use crate::attention::scratch::BlockwiseAttentionScratch;
use crate::common::hash::hash_f32s;
use crate::common::shape::TransformerBlockShape;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockwiseAttentionSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockwiseAttentionSmokeSummary {
    pub status: BlockwiseAttentionSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub blocks: usize,
    pub tokens: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub cpu_block_events: u64,
    pub device_block_events: u64,
    pub hot_path_allocations: u64,
}

impl BlockwiseAttentionSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            BlockwiseAttentionSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"blocks\":{},\"tokens\":{},\"output\":[{},{}],\"output_hash\":{},\"cpu_block_events\":{},\"device_block_events\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.blocks,
            self.tokens,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.cpu_block_events,
            self.device_block_events,
            self.hot_path_allocations,
        )
    }
}

pub fn blockwise_attention_smoke() -> Result<BlockwiseAttentionSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let query = [1.0, 0.25];
    let dram_keys = [0.2, 0.0, 0.0, 0.4];
    let dram_values = [1.0, 0.0, 0.5, 0.5];
    let vram_keys = [0.5, 0.1, -0.2, 0.3];
    let vram_values = [0.0, 1.0, 2.0, -1.0];
    let blocks = [
        KvAttentionBlock::new(&dram_keys, &dram_values, 2, MemoryTier::Dram),
        KvAttentionBlock::new(&vram_keys, &vram_values, 2, MemoryTier::Vram),
    ];
    let mut scratch = BlockwiseAttentionScratch::new(shape)?;
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    exact_blockwise_attention_into(
        shape,
        &query,
        &blocks,
        &mut scratch,
        &mut output,
        &mut ledger,
    )?;

    Ok(BlockwiseAttentionSmokeSummary {
        status: BlockwiseAttentionSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        blocks: blocks.len(),
        tokens: blocks.iter().map(|block| block.token_count).sum(),
        output,
        output_hash: hash_f32s(&output),
        cpu_block_events: ledger.event_count(LedgerEventKind::CpuActivity),
        device_block_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        hot_path_allocations: ledger.hot_path_allocations,
    })
}
