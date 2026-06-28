use clap::Parser;
use nerva_model::hf::tokenizer::PromptFormat;
use nerva_runtime::engine::hf_cuda_decode::file_backed::projection_mode::{
    DEFAULT_PROJECTION_BLOCK_TOKENS, HfCudaProjectionMode, MAX_PROJECTION_BLOCK_TOKENS,
    block_verify_mode,
};

pub(crate) const AUTO_CONTEXT_MARGIN: usize = 16;
pub(crate) const DEFAULT_OUTPUT_TOKENS: usize = 256;
pub(crate) const DEFAULT_QUEUE_CAPACITY: usize = 128;

#[derive(Debug)]
pub(crate) struct GenerateArgs {
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub context_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
    pub queue_capacity: Option<usize>,
    pub compute_capability: Option<u32>,
    pub prompt_format: PromptFormat,
    pub projection_mode: HfCudaProjectionMode,
    pub json: bool,
    pub debug: bool,
}

impl Default for GenerateArgs {
    fn default() -> Self {
        Self {
            model: None,
            prompt: None,
            context_tokens: None,
            output_tokens: None,
            queue_capacity: None,
            compute_capability: None,
            prompt_format: PromptFormat::Auto,
            projection_mode: HfCudaProjectionMode::Token,
            json: false,
            debug: false,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "nerva",
    disable_help_flag = true,
    about = "Run a Hugging Face causal LM through NERVA"
)]
struct ClapGenerateArgs {
    #[arg(short = 'm', long = "model")]
    model: Option<String>,
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,
    #[arg(short = 'c', long = "context", value_parser = parse_token_count)]
    context_tokens: Option<usize>,
    #[arg(
        short = 'o',
        long = "output",
        alias = "max-new-tokens",
        value_parser = parse_token_count
    )]
    output_tokens: Option<usize>,
    #[arg(short = 'q', long = "queue", value_parser = parse_token_count)]
    queue_capacity: Option<usize>,
    #[arg(long = "compute-cap", alias = "compute-capability")]
    compute_capability: Option<u32>,
    #[arg(long = "json")]
    json: bool,
    #[arg(long = "debug")]
    debug: bool,
    #[arg(long = "projection-mode", default_value = "token")]
    projection_mode: String,
    #[arg(
        long = "projection-block-tokens",
        default_value_t = DEFAULT_PROJECTION_BLOCK_TOKENS,
        value_parser = parse_token_count
    )]
    projection_block_tokens: usize,
    #[arg(long = "raw", conflicts_with = "chat")]
    raw: bool,
    #[arg(long = "chat")]
    chat: bool,
    #[arg(short = 'h', long = "help")]
    help: bool,
}

pub(crate) fn parse_args(args: &[String]) -> Result<GenerateArgs, String> {
    let argv = std::iter::once("nerva".to_string())
        .chain(args.iter().cloned())
        .collect::<Vec<_>>();
    let parsed = ClapGenerateArgs::try_parse_from(argv).map_err(|err| err.to_string())?;
    if parsed.help {
        return Err(
            "usage: cargo run -p nerva -- -m model -p prompt [-c context] [-o output] [--projection-mode token|block-verify] [--chat|--raw] [--json] [--debug]"
                .to_string(),
        );
    }
    let projection_mode =
        parse_projection_mode(&parsed.projection_mode, parsed.projection_block_tokens)?;
    Ok(GenerateArgs {
        model: parsed.model,
        prompt: parsed.prompt,
        context_tokens: parsed.context_tokens,
        output_tokens: parsed.output_tokens,
        queue_capacity: parsed.queue_capacity,
        compute_capability: parsed.compute_capability,
        prompt_format: if parsed.raw {
            PromptFormat::Raw
        } else if parsed.chat {
            PromptFormat::Chat
        } else {
            PromptFormat::Auto
        },
        projection_mode,
        json: parsed.json,
        debug: parsed.debug,
    })
}

fn parse_projection_mode(value: &str, block_tokens: usize) -> Result<HfCudaProjectionMode, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "token" | "single" | "single-token" | "gemv" => Ok(HfCudaProjectionMode::Token),
        "block" | "block-verify" | "verify" | "speculative" | "speculative-greedy" => {
            if block_tokens < 2 {
                return Err(
                    "--projection-block-tokens must be at least 2 for block-verify".to_string(),
                );
            }
            if block_tokens > MAX_PROJECTION_BLOCK_TOKENS {
                return Err(format!(
                    "--projection-block-tokens supports at most {MAX_PROJECTION_BLOCK_TOKENS}"
                ));
            }
            block_verify_mode(block_tokens)
                .map_err(|_| "invalid --projection-block-tokens".to_string())
        }
        _ => Err(format!(
            "invalid --projection-mode: {value}; expected token or block-verify"
        )),
    }
}

pub(crate) fn parse_token_count(value: &str) -> Result<usize, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err("token count cannot be empty".to_string());
    }
    let (digits, multiplier) = match trimmed.as_bytes().last().copied() {
        Some(b'k') | Some(b'K') => (&trimmed[..trimmed.len() - 1], 1024usize),
        Some(b'm') | Some(b'M') => (&trimmed[..trimmed.len() - 1], 1024usize * 1024usize),
        _ => (trimmed, 1usize),
    };
    let count = digits
        .parse::<usize>()
        .map_err(|_| format!("invalid token count: {value}"))?;
    count
        .checked_mul(multiplier)
        .ok_or_else(|| format!("token count is too large: {value}"))
}

#[cfg(test)]
mod tests {
    use super::{HfCudaProjectionMode, PromptFormat, parse_args, parse_token_count};

    #[test]
    fn parses_k_token_counts() {
        assert_eq!(parse_token_count("32k").unwrap(), 32 * 1024);
        assert_eq!(parse_token_count("16K").unwrap(), 16 * 1024);
        assert_eq!(parse_token_count("128").unwrap(), 128);
    }

    #[test]
    fn parses_generate_flags() {
        let args = [
            "-m", "qwen3-8b", "-p", "hello", "-c", "32k", "-o", "16k", "--raw", "--debug",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        let parsed = parse_args(&args).unwrap();
        assert_eq!(parsed.model.as_deref(), Some("qwen3-8b"));
        assert_eq!(parsed.prompt.as_deref(), Some("hello"));
        assert_eq!(parsed.context_tokens, Some(32 * 1024));
        assert_eq!(parsed.output_tokens, Some(16 * 1024));
        assert_eq!(parsed.prompt_format, PromptFormat::Raw);
        assert_eq!(parsed.projection_mode, HfCudaProjectionMode::Token);
        assert!(parsed.debug);
    }

    #[test]
    fn parses_projection_mode() {
        let args = [
            "-m",
            "qwen3-8b",
            "-p",
            "hello",
            "--projection-mode",
            "block-verify",
            "--projection-block-tokens",
            "8",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        let parsed = parse_args(&args).unwrap();
        assert_eq!(
            parsed.projection_mode,
            HfCudaProjectionMode::BlockVerify { block_tokens: 8 }
        );
    }
}
