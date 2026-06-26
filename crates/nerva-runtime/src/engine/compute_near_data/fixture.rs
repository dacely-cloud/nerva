use crate::engine::compute_near_data::config::ComputeNearDataProbeConfig;

pub(crate) struct ComputeNearDataFixture {
    pub(crate) input: [f32; 3],
    pub(crate) matrix: [f32; 12],
}

impl ComputeNearDataFixture {
    pub(crate) fn new() -> Self {
        Self {
            input: [2.0, -1.0, 0.5],
            matrix: [
                1.0, 0.0, 2.0, -1.0, 3.0, 0.0, 0.25, 0.5, 1.0, 2.0, -2.0, 1.0,
            ],
        }
    }

    pub(crate) fn cpu_weights(&self, config: ComputeNearDataProbeConfig) -> &[f32] {
        let split = config.split_row * config.cols;
        &self.matrix[..split]
    }

    pub(crate) fn gpu_weights(&self, config: ComputeNearDataProbeConfig) -> &[f32] {
        let split = config.split_row * config.cols;
        &self.matrix[split..]
    }
}
