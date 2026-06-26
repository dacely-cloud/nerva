use nerva_core::types::id::token::TokenId;

use crate::prompt::decode::tiny_prompt_decode_smoke;
use crate::prompt::summary::TinyPromptDecodeStatus;
use crate::prompt::vocabulary::TinyPromptVocabulary;

#[test]
fn tiny_prompt_vocabulary_encodes_and_decodes_words() {
    let vocabulary = TinyPromptVocabulary::cycle_vocab();
    let encoded = vocabulary.encode("zero ONE two").unwrap();

    assert_eq!(encoded.tokens, vec![TokenId(0), TokenId(1), TokenId(2)]);
    assert_eq!(vocabulary.decode(&encoded.tokens).unwrap(), "zero one two");
    assert!(vocabulary.contains_all(&encoded.tokens));
}

#[test]
fn tiny_prompt_vocabulary_rejects_empty_or_unknown_words() {
    let vocabulary = TinyPromptVocabulary::cycle_vocab();

    assert!(vocabulary.encode(" \t\n ").is_err());
    assert!(vocabulary.encode("zero four").is_err());
    assert!(vocabulary.decode(&[TokenId(99)]).is_err());
}

#[test]
fn tiny_prompt_decode_uses_last_prompt_token_as_seed() {
    let summary = tiny_prompt_decode_smoke("zero one", 4).unwrap();

    assert_eq!(summary.status, TinyPromptDecodeStatus::Ok);
    assert_eq!(summary.prompt_tokens, vec![TokenId(0), TokenId(1)]);
    assert_eq!(summary.seed_token, TokenId(1));
    assert_eq!(
        summary.generated_tokens,
        vec![TokenId(2), TokenId(3), TokenId(0), TokenId(1)]
    );
    assert_eq!(
        summary.full_sequence,
        vec![
            TokenId(0),
            TokenId(1),
            TokenId(2),
            TokenId(3),
            TokenId(0),
            TokenId(1)
        ]
    );
    assert_eq!(summary.prompt_text_roundtrip, "zero one");
    assert_eq!(summary.generated_text, "two three zero one");
    assert_eq!(summary.ledger_count, 4);
    assert_eq!(summary.device_events, 4);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.passed());
    assert!(summary.to_json().contains("\"seed_from_prompt\":true"));
}

#[test]
fn tiny_prompt_decode_rejects_zero_steps() {
    assert!(tiny_prompt_decode_smoke("zero", 0).is_err());
}
