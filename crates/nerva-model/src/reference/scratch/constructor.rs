use nerva_core::types::error::Result;

use crate::common::shape::TransformerBlockShape;
use crate::reference::scratch::types::TransformerBlockScratch;

impl TransformerBlockScratch {
    pub fn new(shape: TransformerBlockShape) -> Result<Self> {
        shape.validate()?;
        Ok(Self {
            shape,
            attn_norm: vec![0.0; shape.hidden],
            mlp_norm: vec![0.0; shape.hidden],
            q: vec![0.0; shape.hidden],
            k: vec![0.0; shape.kv_hidden()],
            v: vec![0.0; shape.kv_hidden()],
            attn: vec![0.0; shape.hidden],
            gate: vec![0.0; shape.intermediate],
            up: vec![0.0; shape.intermediate],
            ff: vec![0.0; shape.intermediate],
            down: vec![0.0; shape.hidden],
        })
    }
}
