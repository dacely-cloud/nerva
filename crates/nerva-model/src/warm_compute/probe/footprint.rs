#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct WarmComputeFootprint {
    pub(crate) matrix_bytes: usize,
    pub(crate) input_bytes: usize,
    pub(crate) output_bytes: usize,
}

impl WarmComputeFootprint {
    pub(crate) fn new(matrix_len: usize, input_len: usize, output_len: usize) -> Self {
        Self {
            matrix_bytes: matrix_len * core::mem::size_of::<f32>(),
            input_bytes: input_len * core::mem::size_of::<f32>(),
            output_bytes: output_len * core::mem::size_of::<f32>(),
        }
    }
}
