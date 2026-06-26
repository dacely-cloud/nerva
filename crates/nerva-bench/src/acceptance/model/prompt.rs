use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_prompt_model(report: &mut AcceptanceReport) {
    match nerva_model::prompt::decode::tiny_prompt_decode_smoke("zero one", 4) {
        Ok(summary) => report.push(
            "prompt_tokenization_decode",
            summary.passed()
                && summary.prompt_tokens.len() == 2
                && summary.generated_tokens.len() == 4
                && summary.full_sequence.len() == 6
                && summary.generated_text == "two three zero one",
            format!(
                "prompt_tokens={} seed={} steps={} generated_tokens={} generated_text={} seed_from_prompt={} vocabulary_covered={} ledger_count={} hot_path_allocations={} output_hash={}",
                summary.prompt_tokens.len(),
                summary.seed_token.0,
                summary.steps,
                summary.generated_tokens.len(),
                summary.generated_text,
                summary.seed_from_prompt,
                summary.vocabulary_covered,
                summary.ledger_count,
                summary.hot_path_allocations,
                summary.output_hash,
            ),
        ),
        Err(err) => report.push("prompt_tokenization_decode", false, format!("{err:?}")),
    }
}
