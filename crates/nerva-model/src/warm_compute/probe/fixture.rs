pub(crate) struct WarmComputeFixture {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) matrix: [f32; 16],
    pub(crate) input: [f32; 4],
}

impl Default for WarmComputeFixture {
    fn default() -> Self {
        Self {
            rows: 4,
            cols: 4,
            matrix: [
                1.0, 0.0, 0.0, 1.0, 0.5, -1.0, 2.0, 0.0, -1.0, 0.0, 1.0, 0.5, 0.0, 2.0, 0.25, -0.5,
            ],
            input: [1.0, -2.0, 0.5, 3.0],
        }
    }
}
