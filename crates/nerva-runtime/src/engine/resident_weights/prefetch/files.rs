use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::resident_weights::helpers::update_prefetch_data_hash;
use crate::engine::runtime::Runtime;
use crate::weights::block::ResidentWeightTable;
use crate::weights::prefetch::{ResidentWeightPrefetchIoSummary, ResidentWeightPrefetchPlan};

impl Runtime {
    pub fn execute_resident_weight_prefetch_plan_from_files(
        &self,
        table: &mut ResidentWeightTable,
        plan: &ResidentWeightPrefetchPlan,
        checkpoint_dir: impl AsRef<Path>,
    ) -> Result<ResidentWeightPrefetchIoSummary> {
        if plan.total_bytes != table.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "resident weight file prefetch plan bytes do not match table".to_string(),
            });
        }

        let checkpoint_dir = checkpoint_dir.as_ref();
        let mut ledger = TokenLedger::new(0);
        let mut bytes_by_block = BTreeMap::new();
        let mut total_bytes = 0usize;
        let mut data_hash = 0xcbf2_9ce4_8422_2325u64;
        let mut read_buffer = Vec::new();

        for task in &plan.tasks {
            let block =
                table
                    .registry
                    .block(task.block_id)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "file prefetch task references unknown block {}",
                            task.block_id.0
                        ),
                    })?;
            if block.kind != BlockKind::Weight || block.tier != task.target_tier {
                return Err(NervaError::InvalidArgument {
                    reason: format!(
                        "file prefetch task block {} does not match resident weight",
                        task.block_id.0
                    ),
                });
            }
            if task.file_offset_end < task.file_offset_begin
                || task.file_offset_end - task.file_offset_begin != task.bytes
            {
                return Err(NervaError::InvalidArgument {
                    reason: format!(
                        "file prefetch task {} has invalid file span",
                        task.task_index
                    ),
                });
            }

            let first_task_for_block = !bytes_by_block.contains_key(&task.block_id);
            if first_task_for_block {
                table
                    .registry
                    .transition(task.block_id, ResidencyState::Prefetching)?;
            }

            if task.bytes > read_buffer.len() {
                read_buffer.resize(task.bytes, 0);
            }
            let shard_path = checkpoint_dir.join(&task.source_shard);
            let mut shard = File::open(&shard_path).map_err(|err| NervaError::InvalidArgument {
                reason: format!(
                    "failed to open safetensors shard {}: {err}",
                    shard_path.display()
                ),
            })?;
            shard
                .seek(SeekFrom::Start(
                    u64::try_from(task.file_offset_begin).map_err(|_| {
                        NervaError::InvalidArgument {
                            reason: format!(
                                "file prefetch task {} offset does not fit u64",
                                task.task_index
                            ),
                        }
                    })?,
                ))
                .map_err(|err| NervaError::InvalidArgument {
                    reason: format!(
                        "failed to seek safetensors shard {}: {err}",
                        shard_path.display()
                    ),
                })?;
            shard
                .read_exact(&mut read_buffer[..task.bytes])
                .map_err(|err| NervaError::InvalidArgument {
                    reason: format!(
                        "failed to read safetensors shard {} span {}..{}: {err}",
                        shard_path.display(),
                        task.file_offset_begin,
                        task.file_offset_end,
                    ),
                })?;
            data_hash = update_prefetch_data_hash(data_hash, &read_buffer[..task.bytes]);

            let block_bytes = bytes_by_block.entry(task.block_id).or_insert(0usize);
            *block_bytes = block_bytes.checked_add(task.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: task.bytes,
                    reason: "file prefetch task byte accounting overflow".to_string(),
                }
            })?;
            total_bytes = total_bytes.checked_add(task.bytes).ok_or_else(|| {
                NervaError::AllocationFailed {
                    bytes: task.bytes,
                    reason: "file prefetch execution byte accounting overflow".to_string(),
                }
            })?;
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Prefetch,
                sync_class: None,
                metric_source: MetricSource::RuntimeTimestamp,
                block_id: Some(task.block_id),
                from_tier: Some(MemoryTier::Disk),
                to_tier: Some(MemoryTier::PinnedDram),
                bytes: task.bytes,
                latency_ns: 0,
                label: "weight_prefetch_file_read",
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Copy,
                sync_class: None,
                metric_source: MetricSource::RuntimeTimestamp,
                block_id: Some(task.block_id),
                from_tier: Some(MemoryTier::PinnedDram),
                to_tier: Some(task.target_tier),
                bytes: task.bytes,
                latency_ns: 0,
                label: "weight_prefetch_file_commit",
            });
        }

        for (block_id, bytes) in &bytes_by_block {
            let block =
                table
                    .registry
                    .block(*block_id)
                    .ok_or_else(|| NervaError::InvalidArgument {
                        reason: format!(
                            "file prefetch completion references unknown block {}",
                            block_id.0
                        ),
                    })?;
            if *bytes != block.bytes {
                return Err(NervaError::InvalidArgument {
                    reason: format!(
                        "file prefetch completion for block {} is incomplete",
                        block_id.0
                    ),
                });
            }
            table.registry.mark_ready(*block_id)?;
        }

        if total_bytes != table.total_weight_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "file prefetch execution bytes do not match table".to_string(),
            });
        }
        ledger.require_zero_hot_path_allocations()?;

        let ready_blocks = table
            .entries
            .iter()
            .filter(|entry| {
                table
                    .registry
                    .block(entry.block_id)
                    .is_some_and(|block| block.state == ResidencyState::Ready)
            })
            .count();

        Ok(ResidentWeightPrefetchIoSummary {
            tasks: plan.tasks.len(),
            completed_blocks: bytes_by_block.len(),
            total_bytes,
            shard_count: plan.shard_count,
            disk_read_events: ledger.event_count(LedgerEventKind::Prefetch),
            copy_events: ledger.event_count(LedgerEventKind::Copy),
            ready_blocks,
            data_hash,
            hot_path_allocations: ledger.hot_path_allocations,
            ledger,
        })
    }
}
