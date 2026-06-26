use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::DeviceOrdinal;

use crate::types::event::DeviceTimelineSpan;

pub(crate) fn device_timeline_totals(
    timeline: &[DeviceTimelineSpan],
    device: DeviceOrdinal,
) -> Result<(u64, u64)> {
    let mut spans = timeline
        .iter()
        .filter(|span| span.device == device)
        .collect::<Vec<_>>();
    spans.sort_by_key(|span| (span.start_ns, span.end_ns));

    let mut active_ns = 0u64;
    let mut idle_ns = 0u64;
    let mut merged_end: Option<u64> = None;

    for span in spans {
        validate_device_span(span)?;
        match merged_end {
            None => {
                active_ns = active_ns.saturating_add(span.end_ns - span.start_ns);
                merged_end = Some(span.end_ns);
            }
            Some(end) if span.end_ns <= end => {}
            Some(end) if span.start_ns <= end => {
                active_ns = active_ns.saturating_add(span.end_ns - end);
                merged_end = Some(span.end_ns);
            }
            Some(end) => {
                idle_ns = idle_ns.saturating_add(span.start_ns - end);
                active_ns = active_ns.saturating_add(span.end_ns - span.start_ns);
                merged_end = Some(span.end_ns);
            }
        }
    }

    Ok((active_ns, idle_ns))
}

pub(crate) fn validate_device_span(span: &DeviceTimelineSpan) -> Result<()> {
    if span.end_ns < span.start_ns {
        Err(NervaError::InvalidArgument {
            reason: format!("device span '{}' ends before it starts", span.label),
        })
    } else {
        Ok(())
    }
}
