use std::path::Path;

use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::normalizers::unicode::NFC;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;
use tokenizers::tokenizer::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior};

use crate::hf::tokenizer::json::{read_json_file, token_content};

const BYTE_BPE_SPLIT_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub(super) fn load_byte_bpe_tokenizer(dir: &Path) -> Result<Tokenizer, String> {
    let vocab = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    if !vocab.is_file() || !merges.is_file() {
        return Err("expected tokenizer.json or vocab.json plus merges.txt".to_string());
    }
    let vocab = vocab
        .to_str()
        .ok_or_else(|| "vocab.json path is not valid UTF-8".to_string())?;
    let merges = merges
        .to_str()
        .ok_or_else(|| "merges.txt path is not valid UTF-8".to_string())?;
    let bpe = BPE::from_file(vocab, merges)
        .continuing_subword_prefix(String::new())
        .end_of_word_suffix(String::new())
        .fuse_unk(false)
        .byte_fallback(false)
        .ignore_merges(false)
        .build()
        .map_err(|err| err.to_string())?;
    let split = Split::new(
        SplitPattern::Regex(BYTE_BPE_SPLIT_REGEX.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|err| err.to_string())?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer
        .with_normalizer(Some(NFC))
        .with_pre_tokenizer(Some(Sequence::new(vec![
            PreTokenizerWrapper::Split(split),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])))
        .with_post_processor(Some(ByteLevelProcessor::new(false, false, false)))
        .with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer.add_special_tokens(&added_tokens_from_config(dir)?);
    Ok(tokenizer)
}

fn added_tokens_from_config(dir: &Path) -> Result<Vec<AddedToken>, String> {
    let Some(config) = read_json_file(&dir.join("tokenizer_config.json"))? else {
        return Ok(Vec::new());
    };
    let Some(decoder) = config
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
    else {
        return Ok(Vec::new());
    };
    let mut contents = decoder
        .values()
        .filter_map(token_content)
        .map(str::to_string)
        .collect::<Vec<_>>();
    contents.sort();
    contents.dedup();
    Ok(contents
        .into_iter()
        .map(|token| AddedToken::from(token, true).normalized(false))
        .collect())
}
