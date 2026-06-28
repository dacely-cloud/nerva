use std::path::Path;

use serde_json::Value;

use crate::hf::tokenizer::json::read_json_file;
use crate::hf::tokenizer::{FormattedPrompt, PromptFormat};

pub(super) fn format_prompt_for_model(
    path: &str,
    prompt: &str,
    format: PromptFormat,
) -> Result<FormattedPrompt, String> {
    match format {
        PromptFormat::Raw => Ok(FormattedPrompt {
            text: prompt.to_string(),
            mode: "raw",
        }),
        PromptFormat::Auto => match chat_template_kind(Path::new(path))? {
            Some(kind) => Ok(FormattedPrompt {
                text: kind.render(prompt),
                mode: kind.mode(),
            }),
            None => Ok(FormattedPrompt {
                text: prompt.to_string(),
                mode: "raw",
            }),
        },
        PromptFormat::Chat => {
            let kind = chat_template_kind(Path::new(path))?
                .ok_or_else(|| "model does not declare a supported chat template".to_string())?;
            Ok(FormattedPrompt {
                text: kind.render(prompt),
                mode: kind.mode(),
            })
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ChatTemplateKind {
    ChatMl,
    Llama3,
    Gemma,
    MistralInst,
}

impl ChatTemplateKind {
    const fn mode(self) -> &'static str {
        match self {
            Self::ChatMl => "chatml",
            Self::Llama3 => "llama3_chat",
            Self::Gemma => "gemma_chat",
            Self::MistralInst => "mistral_inst",
        }
    }

    fn render(self, user_prompt: &str) -> String {
        match self {
            Self::ChatMl => format!(
                "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
            Self::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::Gemma => {
                format!("<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n")
            }
            Self::MistralInst => format!("<s>[INST] {user_prompt} [/INST]"),
        }
    }
}

fn chat_template_kind(dir: &Path) -> Result<Option<ChatTemplateKind>, String> {
    let Some(config) = read_json_file(&dir.join("tokenizer_config.json"))? else {
        return Ok(None);
    };
    let Some(template) = config.get("chat_template").and_then(Value::as_str) else {
        return Ok(None);
    };
    if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
        return Ok(Some(ChatTemplateKind::ChatMl));
    }
    if template.contains("<|start_header_id|>") && template.contains("<|eot_id|>") {
        return Ok(Some(ChatTemplateKind::Llama3));
    }
    if template.contains("<start_of_turn>") && template.contains("<end_of_turn>") {
        return Ok(Some(ChatTemplateKind::Gemma));
    }
    if template.contains("[INST]") && template.contains("[/INST]") {
        return Ok(Some(ChatTemplateKind::MistralInst));
    }
    Err("model has a chat_template, but NERVA cannot render that template yet; use --raw for raw completion mode".to_string())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::hf::tokenizer::{PromptFormat, format_prompt_for_model};

    #[test]
    fn formats_qwen_chat_template_in_auto_mode() {
        let dir = temp_dir("qwen-chat");
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"<|im_start|>user\n{{ content }}<|im_end|>\n<|im_start|>assistant\n"}"#,
        )
        .unwrap();
        let formatted =
            format_prompt_for_model(dir.to_str().unwrap(), "hello", PromptFormat::Auto).unwrap();
        assert_eq!(formatted.mode, "chatml");
        assert!(formatted.text.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(formatted.text.ends_with("</think>\n\n"));
        let _ = fs::remove_dir_all(dir);
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("nerva-tokenizer-{name}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
