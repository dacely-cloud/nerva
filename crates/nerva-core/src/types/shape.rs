use crate::types::error::{NervaError, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockShape {
    dims: Vec<u64>,
}

impl BlockShape {
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    pub fn from_dims(dims: impl Into<Vec<u64>>) -> Result<Self> {
        let dims = dims.into();
        if dims.iter().any(|dim| *dim == 0) {
            return Err(NervaError::InvalidArgument {
                reason: "block shape dimensions must be non-zero".to_string(),
            });
        }
        Ok(Self { dims })
    }

    pub fn dims(&self) -> &[u64] {
        &self.dims
    }
}
