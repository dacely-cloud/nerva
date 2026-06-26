use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::memory::MemoryTier;

use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::support::{
    SHARD_ONE, tiny_llama_manifest, tiny_shard_plan, tiny_shard_plan_with_header,
    write_tiny_shard_file,
};

#[test]
fn resident_weight_prefetch_plan_records_bounded_tasks() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header_len) = tiny_shard_plan();
    let table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 128).unwrap();

    assert_eq!(prefetch.tasks.len(), table.entries.len());
    assert_eq!(prefetch.total_bytes, table.total_weight_bytes);
    assert_eq!(prefetch.shard_count, 1);
    assert_eq!(prefetch.prefetch_events, prefetch.tasks.len() as u64);
    assert_eq!(prefetch.copy_events, prefetch.tasks.len() as u64);
    assert_eq!(prefetch.ledger.hot_path_allocations, 0);
    assert_eq!(prefetch.first_source_shard.as_deref(), Some(SHARD_ONE));
    assert_eq!(prefetch.last_source_shard.as_deref(), Some(SHARD_ONE));

    let first = prefetch.tasks.first().unwrap();
    assert_eq!(first.task_index, 0);
    assert_eq!(first.name, "model.embed_tokens.weight");
    assert_eq!(first.file_offset_begin, 8 + header_len);
    assert_eq!(first.file_offset_end, 8 + header_len + first.bytes);
    assert_eq!(first.target_tier, MemoryTier::Dram);
    assert!(prefetch.to_json().contains("\"tasks\":11"));
}

#[test]
fn resident_weight_prefetch_plan_splits_large_source_spans() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header_len) = tiny_shard_plan();
    let table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();

    assert!(prefetch.tasks.len() > table.entries.len());
    assert_eq!(prefetch.total_bytes, table.total_weight_bytes);
    assert_eq!(prefetch.tasks[0].bytes, 16);
    assert_eq!(prefetch.tasks[0].file_offset_begin, 8 + header_len);
    assert_eq!(prefetch.tasks[1].file_offset_begin, 8 + header_len + 16);
    assert_eq!(prefetch.tasks[4].file_offset_end, 8 + header_len + 80);
    assert_eq!(
        prefetch.tasks[5].name,
        "model.layers.0.input_layernorm.weight"
    );
}

#[test]
fn resident_weight_prefetch_requires_source_offsets() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let manifest = tiny_llama_manifest();
    let table = runtime.materialize_hf_weight_manifest(&manifest).unwrap();

    assert!(runtime.plan_resident_weight_prefetch(&table, 128).is_err());
    assert!(runtime.plan_resident_weight_prefetch(&table, 0).is_err());
}

#[test]
fn resident_weight_prefetch_execution_marks_blocks_ready() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, _) = tiny_shard_plan();
    let mut table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();
    let summary = runtime
        .execute_resident_weight_prefetch_plan(&mut table, &prefetch)
        .unwrap();

    assert_eq!(summary.tasks, prefetch.tasks.len());
    assert_eq!(summary.completed_blocks, table.entries.len());
    assert_eq!(summary.total_bytes, table.total_weight_bytes);
    assert_eq!(summary.prefetch_events, prefetch.tasks.len() as u64);
    assert_eq!(summary.copy_events, prefetch.tasks.len() as u64);
    assert_eq!(summary.ready_blocks, table.entries.len());
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"ready_blocks\":11"));
    assert!(table.entries.iter().all(|entry| {
        table
            .registry
            .block(entry.block_id)
            .is_some_and(|block| block.state == ResidencyState::Ready)
    }));
}

#[test]
fn resident_weight_file_prefetch_reads_shard_ranges() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header) = tiny_shard_plan_with_header();
    let dir = std::env::temp_dir().join(format!("nerva-runtime-prefetch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    write_tiny_shard_file(&dir, &header, plan.total_weight_bytes);

    let mut table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();
    let summary = runtime
        .execute_resident_weight_prefetch_plan_from_files(&mut table, &prefetch, &dir)
        .unwrap();

    assert_eq!(summary.tasks, prefetch.tasks.len());
    assert_eq!(summary.completed_blocks, table.entries.len());
    assert_eq!(summary.total_bytes, table.total_weight_bytes);
    assert_eq!(summary.shard_count, 1);
    assert_eq!(summary.disk_read_events, prefetch.tasks.len() as u64);
    assert_eq!(summary.copy_events, prefetch.tasks.len() as u64);
    assert_eq!(summary.ready_blocks, table.entries.len());
    assert_ne!(summary.data_hash, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"data_hash\""));
    assert!(table.entries.iter().all(|entry| {
        table
            .registry
            .block(entry.block_id)
            .is_some_and(|block| block.state == ResidencyState::Ready)
    }));

    let _ = std::fs::remove_file(dir.join(SHARD_ONE));
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn resident_weight_file_prefetch_rejects_short_shard() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, header) = tiny_shard_plan_with_header();
    let dir = std::env::temp_dir().join(format!(
        "nerva-runtime-prefetch-short-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    write_tiny_shard_file(&dir, &header, plan.total_weight_bytes - 1);

    let mut table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();

    assert!(
        runtime
            .execute_resident_weight_prefetch_plan_from_files(&mut table, &prefetch, &dir)
            .is_err()
    );

    let _ = std::fs::remove_file(dir.join(SHARD_ONE));
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn resident_weight_prefetch_execution_rejects_incomplete_plan() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let (plan, _) = tiny_shard_plan();
    let mut table = runtime.materialize_safetensors_shard_plan(&plan).unwrap();
    let mut prefetch = runtime.plan_resident_weight_prefetch(&table, 16).unwrap();
    prefetch.tasks.pop();

    assert!(
        runtime
            .execute_resident_weight_prefetch_plan(&mut table, &prefetch)
            .is_err()
    );
}
