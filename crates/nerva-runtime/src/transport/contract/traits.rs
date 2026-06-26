use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::error::Result;
use nerva_core::types::id::replica::ReplicaId;

use crate::transport::contract::types::{
    ReceiveDescriptor, TransferCompletion, TransferDescriptor, TransferId,
};

pub trait TensorTransportContract {
    type Endpoint;
    type Registration;

    fn register(&mut self, block: &ResidentBlock, replica: ReplicaId)
    -> Result<Self::Registration>;

    fn post_receive(
        &mut self,
        src: &Self::Endpoint,
        receive: ReceiveDescriptor,
    ) -> Result<TransferId>;

    fn send(&mut self, dst: &Self::Endpoint, transfer: TransferDescriptor) -> Result<TransferId>;

    fn poll(&mut self, completions: &mut [TransferCompletion]) -> Result<usize>;
}
