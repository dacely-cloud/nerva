use nerva_core::types::id::TokenId;

pub(crate) const TOKEN_STREAM_HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

pub(crate) fn hash_observed_token(current: u64, token_index: u64, token: TokenId) -> u64 {
    let mut hash = current ^ token_index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash = hash.rotate_left(13) ^ u64::from(token.0);
    hash.wrapping_mul(0xff51_afd7_ed55_8ccd)
}
