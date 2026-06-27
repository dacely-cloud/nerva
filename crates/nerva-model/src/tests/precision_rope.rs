use crate::common::rope::apply_rotary_to_query_key;
use crate::common::shape::TransformerBlockShape;

#[test]
fn rotary_embedding_accepts_explicit_attention_width() {
    let shape = TransformerBlockShape::new_with_kv_heads_and_head_dim(5, 2, 1, 4, 7);
    let mut query = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
    let mut key = [0.5, -1.0, -0.25, 0.75];

    apply_rotary_to_query_key(shape, 1, 10_000.0, &mut query, &mut key).unwrap();

    assert_eq!(query.len(), shape.attention_hidden());
    assert_eq!(key.len(), shape.kv_hidden());
    assert_ne!(query[0], 1.0);
    assert_ne!(key[0], 0.5);
}
