pub(super) struct ResidentKvAttentionPayload<'a> {
    pub(super) page_index: u32,
    pub(super) keys: &'a [f32],
    pub(super) values: &'a [f32],
}
