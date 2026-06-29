use clap::Parser;
use nerva_model::hf::tokenizer::PromptFormat;

pub(crate) const AUTO_CONTEXT_MARGIN: usize = 16;
pub(crate) const DEFAULT_OUTPUT_TOKENS: usize = 256;
pub(crate) const DEFAULT_QUEUE_CAPACITY: usize = 1024;

#[derive(Debug)]
pub(crate) struct GenerateArgs {
    pub model: Option<String>,
    pub prompt: Option<String>,
    pub context_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
    pub queue_capacity: Option<usize>,
    pub compute_capability: Option<u32>,
    pub prompt_format: PromptFormat,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: u64,
    pub rt: bool,
    pub profiling: bool,
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
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            rt: false,
            profiling: false,
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
    #[arg(long = "profiling")]
    profiling: bool,
    #[arg(long = "temperature", default_value_t = 0.0)]
    temperature: f32,
    #[arg(long = "top-p", default_value_t = 1.0)]
    top_p: f32,
    #[arg(long = "top-k", default_value_t = 0)]
    top_k: u32,
    #[arg(long = "seed", default_value_t = 0)]
    seed: u64,
    #[arg(long = "rt")]
    rt: bool,
    #[arg(long = "raw", conflicts_with = "chat")]
    raw: bool,
    #[arg(long = "chat")]
    chat: bool,
    #[arg(short = 'h', long = "help")]
    help: bool,
}

pub(crate) fn parse_args(args: &[String]) -> Result<GenerateArgs, String> {
    let argv = std::iter::once("nerva".to_string())
        .chain(args.iter().map(|arg| {
            if arg == "-rt" {
                "--rt".to_string()
            } else {
                arg.clone()
            }
        }))
        .collect::<Vec<_>>();
    let parsed = ClapGenerateArgs::try_parse_from(argv).map_err(|err| err.to_string())?;
    if parsed.help {
        return Err(
            "usage: cargo run -p nerva -- -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [-rt|--rt] [--profiling] [--chat|--raw] [--json] [--debug]"
                .to_string(),
        );
    }
    validate_sampling(parsed.temperature, parsed.top_p)?;
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
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        seed: parsed.seed,
        rt: parsed.rt,
        profiling: parsed.profiling,
        json: parsed.json,
        debug: parsed.debug,
    })
}

fn validate_sampling(temperature: f32, top_p: f32) -> Result<(), String> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("--temperature must be finite and >= 0".to_string());
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err("--top-p must be finite and in (0, 1]".to_string());
    }
    Ok(())
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
    use super::{PromptFormat, parse_args, parse_token_count};

    #[test]
    fn parses_k_token_counts() {
        assert_eq!(parse_token_count("32k").unwrap(), 32 * 1024);
        assert_eq!(parse_token_count("16K").unwrap(), 16 * 1024);
        assert_eq!(parse_token_count("128").unwrap(), 128);
    }

    #[test]
    fn parses_generate_flags() {
        let args = [
            "-m",
            "qwen3-8b",
            "-p",
            "hello",
            "-c",
            "32k",
            "-o",
            "16k",
            "--raw",
            "--debug",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--top-k",
            "40",
            "--seed",
            "123",
            "-rt",
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
        assert_eq!(parsed.temperature, 0.7);
        assert_eq!(parsed.top_p, 0.9);
        assert_eq!(parsed.top_k, 40);
        assert_eq!(parsed.seed, 123);
        assert!(parsed.rt);
        assert!(parsed.debug);
        assert!(!parsed.profiling);
    }

    #[test]
    fn rejects_bad_sampling_flags() {
        let args = ["-m", "qwen3-8b", "-p", "hello", "--top-p", "0"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert!(parse_args(&args).is_err());
    }
}
