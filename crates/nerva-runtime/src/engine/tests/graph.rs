use crate::graph::layout::GraphLayout;
use crate::graph::pool::GraphPool;

#[test]
fn graph_pool_rejects_layout_drift() {
    let captured = GraphLayout::new(1, 8, 16, 4);
    let replay = GraphLayout::new(1, 8, 32, 4);
    let mut pool = GraphPool::new();
    pool.capture_synthetic(captured);

    assert!(pool.check_before_replay(captured).is_ok());
    assert!(pool.check_before_replay(replay).is_err());
}
