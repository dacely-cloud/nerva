use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransactionAccess {
    Read,
    Write,
    ReadWrite,
}

impl TransactionAccess {
    pub const fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransactionOperationKind {
    TensorCompute,
    KvAppend,
    DeviceSampling,
    HostObservation,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransactionBlockUse {
    pub block_id: ResidentBlockId,
    pub access: TransactionAccess,
    pub owner: ExecutionOwner,
    pub expected_tier: MemoryTier,
    pub required_version: u64,
    pub label: &'static str,
}

impl TransactionBlockUse {
    pub const fn read(
        block_id: ResidentBlockId,
        owner: ExecutionOwner,
        expected_tier: MemoryTier,
        required_version: u64,
        label: &'static str,
    ) -> Self {
        Self {
            block_id,
            access: TransactionAccess::Read,
            owner,
            expected_tier,
            required_version,
            label,
        }
    }

    pub const fn write(
        block_id: ResidentBlockId,
        owner: ExecutionOwner,
        expected_tier: MemoryTier,
        required_version: u64,
        label: &'static str,
    ) -> Self {
        Self {
            block_id,
            access: TransactionAccess::Write,
            owner,
            expected_tier,
            required_version,
            label,
        }
    }

    pub const fn read_write(
        block_id: ResidentBlockId,
        owner: ExecutionOwner,
        expected_tier: MemoryTier,
        required_version: u64,
        label: &'static str,
    ) -> Self {
        Self {
            block_id,
            access: TransactionAccess::ReadWrite,
            owner,
            expected_tier,
            required_version,
            label,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransactionOperation {
    pub name: &'static str,
    pub kind: TransactionOperationKind,
    pub executor: ExecutionOwner,
    pub block_uses: Vec<TransactionBlockUse>,
    pub graph_capturable: bool,
    pub predicted_visible_ns: u64,
}

impl TransactionOperation {
    pub fn new(
        name: &'static str,
        kind: TransactionOperationKind,
        executor: ExecutionOwner,
        predicted_visible_ns: u64,
    ) -> Self {
        Self {
            name,
            kind,
            executor,
            block_uses: Vec::new(),
            graph_capturable: false,
            predicted_visible_ns,
        }
    }

    pub fn with_block_use(mut self, block_use: TransactionBlockUse) -> Self {
        self.block_uses.push(block_use);
        self
    }

    pub const fn graph_capturable(mut self, graph_capturable: bool) -> Self {
        self.graph_capturable = graph_capturable;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionTransactionSpec {
    pub name: &'static str,
    pub token_index: u64,
    pub operations: Vec<TransactionOperation>,
}

impl ExecutionTransactionSpec {
    pub fn new(name: &'static str, token_index: u64) -> Self {
        Self {
            name,
            token_index,
            operations: Vec::new(),
        }
    }

    pub fn with_operation(mut self, operation: TransactionOperation) -> Self {
        self.operations.push(operation);
        self
    }
}
