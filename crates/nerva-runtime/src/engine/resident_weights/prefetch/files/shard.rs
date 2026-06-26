use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use nerva_core::types::error::{NervaError, Result};

use crate::weights::prefetch::ResidentWeightPrefetchTask;

pub(super) fn read_prefetch_task_span<'a>(
    checkpoint_dir: &Path,
    task: &ResidentWeightPrefetchTask,
    read_buffer: &'a mut Vec<u8>,
) -> Result<&'a [u8]> {
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
            u64::try_from(task.file_offset_begin).map_err(|_| NervaError::InvalidArgument {
                reason: format!(
                    "file prefetch task {} offset does not fit u64",
                    task.task_index
                ),
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
    Ok(&read_buffer[..task.bytes])
}
