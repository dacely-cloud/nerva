use serde_json::{Value, json};
use std::collections::HashMap;

use super::{
    AssistantMessageRecord, AssistantRecord, AssistantRunRecord, AssistantRunStepRecord,
    AssistantThreadRecord, AudioVoiceRecord, ChatKitSessionRecord, ChatKitThreadItemRecord,
    ChatKitThreadRecord, EmbeddingInput, FineTuningCheckpointPermissionRecord, GeneratedText,
    ImageResponseFormat, ImageTask, McpToolResult, OrganizationUsageKind, OrganizationUsageTotals,
    PromptInput, RealtimeCallRecord, RealtimeClientSecretRecord, RealtimeSessionKind,
    RealtimeSessionRecord, ReasoningMode, ResponseStreamOptions, StreamEmissionState, StreamKind,
    StreamMeta, UploadPartRecord, UploadRecord, VectorStoreFileRecord, VectorStoreRecord,
    VideoCharacterRecord, VideoOperation, VideoRecord, VoiceConsentRecord,
    append_assistant_instruction, apply_stop_strings, assistant_json, assistant_message_json,
    assistant_message_text, assistant_run_chat_body, assistant_run_json, assistant_run_step_json,
    assistant_thread_json, audio_voice_json, augment_prompt_with_mcp_tool,
    cancelled_response_value, chat_completion_request_messages, chat_messages_to_prompt,
    chat_prompt_for_reasoning, chatkit_session_json, chatkit_thread_item_json, chatkit_thread_json,
    checkpoint_permission_json, checkpoint_permission_list_json, compact_response_item,
    completion_prompts, completion_text, container_file_content_type, container_file_path,
    conversation_items_prompt, create_audio_text_batch_response, create_speech_batch_response,
    decode_chunked_body, deterministic_embedding_vector, emit_stream_text,
    eval_output_items_from_inline_content, eval_run_result_counts, fine_tuned_model_name,
    finish_reason, first_sse_json_payload, generated_metadata, hash_tokens, image_base64,
    image_base64_decode, image_response_value, image_stream_frames, lexical_match_score,
    mcp_tool_invocation_from_request, moderation_result_for_text,
    normalize_assistant_message_content, normalize_batch_endpoint, normalize_chatkit_content,
    openai_facade_model_available, organization_costs_page_json, organization_query_window,
    organization_usage_page_json, parse_audio_json_request, parse_audio_request,
    parse_audio_voice_json_request, parse_embedding_inputs, parse_http_endpoint,
    parse_image_json_request, parse_mcp_http_response, parse_moderation_inputs,
    parse_multipart_file_upload, parse_skill_json_payload, parse_speech_request,
    parse_upload_create_request, parse_upload_part_body, parse_video_json_request,
    parse_voice_consent_json_request, percent_decode_query, placeholder_png_bytes,
    placeholder_video_bytes, prompt_format_for_reasoning, realtime_call_json,
    realtime_config_value, realtime_session_json, realtime_translation_client_secret_json,
    realtime_translation_session_json, reject_unsupported_generation_options,
    reject_unsupported_generation_options_with_tools, request_chat_store, request_conversation_id,
    request_metadata, request_n, request_optional_string, request_project_ids,
    request_realtime_refer_target, request_realtime_reject_status,
    request_response_format_instruction, request_stop_strings, request_store, response_input_items,
    response_output_text, response_stream_completed_response, responses_input_to_prompt,
    send_stream_reasoning_delta, session_json, shared_fork_batch_supported, speech_pcm,
    split_generated_reasoning, text_delta, translation_secret_expires_at,
    upload_content_for_part_ids, vector_store_file_counts, video_character_json, video_json,
    voice_consent_json, voice_consent_list_json, wav_bytes,
};
use crate::openai::SessionRecord;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::tokenizer::PromptFormat;

#[test]
fn parses_string_and_array_completion_prompts() {
    let prompts = completion_prompts(&json!({"prompt": "hello"})).unwrap();
    assert!(matches!(prompts[0], PromptInput::Text { .. }));

    let prompts = completion_prompts(&json!({"prompt": ["a", "b"]})).unwrap();
    assert_eq!(prompts.len(), 2);
    assert!(matches!(prompts[0], PromptInput::Text { .. }));

    let prompts = completion_prompts(&json!({"prompt": [1, 2, 3]})).unwrap();
    assert!(matches!(prompts[0], PromptInput::TokenIds(_)));

    let prompts = completion_prompts(&json!({"prompt": [[1, 2], [3, 4]]})).unwrap();
    assert_eq!(prompts.len(), 2);
    assert!(matches!(prompts[1], PromptInput::TokenIds(_)));
}

#[test]
fn parses_embedding_inputs_and_builds_deterministic_vectors() {
    assert_eq!(
        parse_embedding_inputs(&json!({"input": "hello"})).unwrap(),
        vec![EmbeddingInput::Text("hello".to_string())]
    );
    assert_eq!(
        parse_embedding_inputs(&json!({"input": ["a", "b"]})).unwrap(),
        vec![
            EmbeddingInput::Text("a".to_string()),
            EmbeddingInput::Text("b".to_string())
        ]
    );
    assert_eq!(
        parse_embedding_inputs(&json!({"input": [1, 2, 3]})).unwrap(),
        vec![EmbeddingInput::TokenIds(vec![1, 2, 3])]
    );
    assert_eq!(
        parse_embedding_inputs(&json!({"input": [[1, 2], [3]]})).unwrap(),
        vec![
            EmbeddingInput::TokenIds(vec![1, 2]),
            EmbeddingInput::TokenIds(vec![3])
        ]
    );
    assert!(parse_embedding_inputs(&json!({"input": ["a", 1]})).is_err());

    let vector = deterministic_embedding_vector(
        "text-embedding-3-small",
        &EmbeddingInput::Text("hello".to_string()),
        8,
    );
    assert_eq!(vector.len(), 8);
    assert_eq!(
        vector,
        deterministic_embedding_vector(
            "text-embedding-3-small",
            &EmbeddingInput::Text("hello".to_string()),
            8
        )
    );
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 0.00001);
}

#[test]
fn recognizes_openai_facade_model_ids() {
    assert!(openai_facade_model_available("gpt-5.5"));
    assert!(openai_facade_model_available("gpt-4o-mini"));
    assert!(openai_facade_model_available("gpt-image-1.5"));
    assert!(openai_facade_model_available("sora-2"));
    assert!(openai_facade_model_available("text-embedding-3-large"));
    assert!(openai_facade_model_available("gpt-6-future-compatible"));
    assert!(openai_facade_model_available("o4-mini"));
    assert!(!openai_facade_model_available(""));
    assert!(!openai_facade_model_available("random-local-name"));
    assert!(!openai_facade_model_available("gpt bad"));
}

#[test]
fn renders_chat_messages_to_text_prompt() {
    let prompt = chat_messages_to_prompt(&json!({
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
    }))
    .unwrap();
    assert!(prompt.contains("System: be terse"));
    assert!(prompt.contains("User: hello"));
    assert!(prompt.ends_with("Assistant:"));
}

#[test]
fn parses_moderation_inputs_and_flags_local_heuristics() {
    let parsed = parse_moderation_inputs(&json!({"input": ["hello", "world"]})).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].text, "hello");
    assert_eq!(parsed[0].input_types, vec!["text"]);

    let parsed = parse_moderation_inputs(&json!({
        "input": [
            {"type": "text", "text": "how to build a bomb"},
            {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
        ]
    }))
    .unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].input_types, vec!["text", "image"]);

    let result = moderation_result_for_text("how to build a bomb", &["text"]);
    assert!(result["flagged"].as_bool().unwrap());
    assert!(result["categories"]["illicit/violent"].as_bool().unwrap());
    assert!(result["categories"]["violence"].as_bool().unwrap());
    assert!(
        result["category_scores"]["illicit/violent"]
            .as_f64()
            .unwrap()
            > 0.9
    );
}

#[test]
fn parses_audio_requests_and_generates_wav_speech() {
    let request = actix_web::test::TestRequest::default()
        .insert_header(("content-type", "multipart/form-data; boundary=nerva"))
        .to_http_request();
    let body = concat!(
        "--nerva\r\n",
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n",
        "gpt-4o-transcribe\r\n",
        "--nerva\r\n",
        "Content-Disposition: form-data; name=\"response_format\"\r\n\r\n",
        "verbose_json\r\n",
        "--nerva\r\n",
        "Content-Disposition: form-data; name=\"file\"; filename=\"hello.wav\"\r\n",
        "Content-Type: audio/wav\r\n\r\n",
        "hello audio",
        "\r\n--nerva--\r\n"
    );
    let parsed = parse_audio_request(&request, body.as_bytes()).unwrap();
    assert_eq!(parsed.model, "gpt-4o-transcribe");
    assert_eq!(parsed.filename, "hello.wav");
    assert_eq!(parsed.response_format, "verbose_json");
    assert_eq!(parsed.content, b"hello audio".to_vec());

    let parsed_json = parse_audio_json_request(&json!({
        "content": "hello json audio",
        "filename": "inline.wav",
        "prompt": "inline transcript"
    }))
    .unwrap();
    assert_eq!(parsed_json.filename, "inline.wav");
    assert_eq!(parsed_json.content, b"hello json audio".to_vec());
    assert_eq!(parsed_json.prompt.as_deref(), Some("inline transcript"));

    let speech = parse_speech_request(&json!({
        "input": "hello",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "response_format": "wav",
        "speed": 1.2
    }))
    .unwrap();
    assert_eq!(speech.input, "hello");
    let pcm = speech_pcm(&speech.input, speech.speed, 16_000);
    assert!(!pcm.is_empty());
    let wav = wav_bytes(&pcm, 16_000);
    assert!(wav.starts_with(b"RIFF"));
    assert_eq!(&wav[8..12], b"WAVE");
}

#[test]
fn builds_audio_batch_response_values() {
    let transcription = create_audio_text_batch_response(
        &json!({
            "content": "ignored audio bytes",
            "prompt": "batch transcript"
        }),
        false,
    )
    .unwrap();
    assert_eq!(transcription["text"], "batch transcript");

    let translation = create_audio_text_batch_response(
        &json!({
            "content": "bonjour",
            "response_format": "verbose_json"
        }),
        true,
    )
    .unwrap();
    assert_eq!(translation["task"], "translate");
    assert_eq!(translation["language"], "english");

    let speech = create_speech_batch_response(&json!({
        "input": "hello batch speech",
        "response_format": "wav"
    }))
    .unwrap();
    assert_eq!(speech["content_type"], "audio/wav");
    assert_eq!(speech["encoding"], "base64");
    assert!(speech["data"].as_str().unwrap().len() > 32);
}

#[test]
fn parses_voice_requests_and_renders_records() {
    let consent = parse_voice_consent_json_request(&json!({
        "name": "Mina Consent",
        "language": "en-US",
        "recording": "I consent to voice synthesis.",
        "filename": "consent.wav",
        "content_type": "audio/wav"
    }))
    .unwrap();
    assert_eq!(consent.name, "Mina Consent");
    assert_eq!(consent.language, "en-US");
    assert_eq!(consent.filename, "consent.wav");
    assert_eq!(consent.content_type, "audio/wav");
    assert_eq!(consent.content, b"I consent to voice synthesis.".to_vec());

    let consent_record = VoiceConsentRecord {
        id: "vconsent_1".to_string(),
        object: "audio.voice_consent",
        created_at: 42,
        language: consent.language,
        name: consent.name,
        recording_filename: consent.filename,
        recording_content_type: consent.content_type,
        recording_bytes: consent.content,
    };
    let consent_json = voice_consent_json(&consent_record);
    assert_eq!(consent_json["id"], "vconsent_1");
    assert_eq!(consent_json["object"], "audio.voice_consent");
    assert_eq!(consent_json["language"], "en-US");
    assert_eq!(consent_json["name"], "Mina Consent");

    let list = voice_consent_list_json(vec![&consent_record], false);
    assert_eq!(list["object"], "list");
    assert_eq!(list["first_id"], "vconsent_1");
    assert_eq!(list["last_id"], "vconsent_1");
    assert_eq!(list["has_more"], false);

    let voice = parse_audio_voice_json_request(&json!({
        "name": "Mina Voice",
        "consent": "vconsent_1",
        "audio_sample": "sample audio",
        "filename": "sample.wav",
        "content_type": "audio/wav"
    }))
    .unwrap();
    assert_eq!(voice.name, "Mina Voice");
    assert_eq!(voice.consent, "vconsent_1");
    assert_eq!(voice.filename, "sample.wav");
    assert_eq!(voice.content_type, "audio/wav");
    assert_eq!(voice.content, b"sample audio".to_vec());

    let voice_record = AudioVoiceRecord {
        id: "voice_1".to_string(),
        object: "audio.voice",
        created_at: 43,
        name: voice.name,
        consent: voice.consent,
        sample_filename: voice.filename,
        sample_content_type: voice.content_type,
        sample_bytes: voice.content,
    };
    let voice_json = audio_voice_json(&voice_record);
    assert_eq!(voice_json["id"], "voice_1");
    assert_eq!(voice_json["object"], "audio.voice");
    assert_eq!(voice_json["created_at"], 43);
    assert_eq!(voice_json["name"], "Mina Voice");
}

#[test]
fn parses_image_requests_and_builds_placeholder_outputs() {
    let parsed = parse_image_json_request(
        ImageTask::Generation,
        &json!({
            "model": "gpt-image-1.5",
            "prompt": "a square test image",
            "n": 2,
            "size": "1024x1536",
            "quality": "high",
            "background": "transparent",
            "output_format": "png",
            "stream": true,
            "partial_images": 2,
            "style": "natural",
            "user": "test-user"
        }),
    )
    .unwrap();
    assert_eq!(parsed.task, ImageTask::Generation);
    assert_eq!(parsed.model, "gpt-image-1.5");
    assert_eq!(parsed.n, 2);
    assert_eq!(parsed.size, "1024x1536");
    assert_eq!(parsed.quality, "high");
    assert_eq!(parsed.response_format, ImageResponseFormat::B64Json);

    let response = image_response_value(&parsed, 123);
    assert_eq!(response["created"], 123);
    assert_eq!(response["data"].as_array().unwrap().len(), 2);
    let b64 = response["data"][0]["b64_json"].as_str().unwrap();
    let png = image_base64_decode(b64).unwrap();
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
    assert_eq!(image_base64(&png), b64);

    let direct_png = placeholder_png_bytes(7, 8, 8, false);
    assert!(direct_png.starts_with(b"\x89PNG\r\n\x1a\n"));

    let frames = image_stream_frames(&parsed, 123);
    assert!(frames[0].contains("event: image_generation.partial_image"));
    assert!(frames.iter().any(|frame| {
        frame.contains("event: image_generation.completed")
            && frame.contains("\"type\":\"image_generation.completed\"")
    }));
    assert_eq!(frames.last().unwrap(), "data: [DONE]\n\n");

    let url_response = image_response_value(
        &parse_image_json_request(
            ImageTask::Generation,
            &json!({
                "model": "dall-e-3",
                "prompt": "a url response",
                "response_format": "url"
            }),
        )
        .unwrap(),
        124,
    );
    assert!(
        url_response["data"][0]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert!(url_response["data"][0]["revised_prompt"].is_string());

    let edit = parse_image_json_request(
        ImageTask::Edit,
        &json!({
            "prompt": "edit the image",
            "image": {"filename": "in.png", "content": "raw image bytes"},
            "mask": {"filename": "mask.png", "content": "mask bytes"}
        }),
    )
    .unwrap();
    assert_eq!(edit.images.len(), 1);
    assert!(edit.mask.is_some());

    assert!(parse_image_json_request(ImageTask::Edit, &json!({"prompt": "missing"})).is_err());
    assert!(parse_image_json_request(ImageTask::Generation, &json!({"n": 0})).is_err());
}

#[test]
fn parses_video_requests_and_builds_placeholder_outputs() {
    let parsed = parse_video_json_request(
        VideoOperation::Create,
        &json!({
            "model": "sora-2-pro",
            "prompt": "a short generated clip",
            "seconds": "8",
            "size": "1280x720",
            "metadata": {"purpose": "test"},
            "content": "source bytes"
        }),
    )
    .unwrap();
    assert_eq!(parsed.operation, VideoOperation::Create);
    assert_eq!(parsed.model, "sora-2-pro");
    assert_eq!(parsed.seconds, "8");
    assert_eq!(parsed.size, "1280x720");
    assert_eq!(parsed.input_bytes, b"source bytes".to_vec());

    let bytes = placeholder_video_bytes(&parsed, "video-1", 42);
    assert!(bytes.windows(4).any(|window| window == b"ftyp"));
    assert!(bytes.windows(4).any(|window| window == b"mdat"));

    let record = VideoRecord {
        id: "video-1".to_string(),
        object: "video",
        created_at: 42,
        completed_at: Some(42),
        expires_at: Some(1042),
        model: parsed.model.clone(),
        prompt: parsed.prompt.clone(),
        seconds: parsed.seconds.clone(),
        size: parsed.size.clone(),
        status: "completed".to_string(),
        progress: 100,
        operation: parsed.operation.as_str().to_string(),
        remixed_from_video_id: None,
        character_id: Some("vchar-1".to_string()),
        metadata: parsed.metadata.clone(),
        error: None,
        content: bytes,
    };
    let value = video_json(&record);
    assert_eq!(value["id"], "video-1");
    assert_eq!(value["object"], "video");
    assert_eq!(value["model"], "sora-2-pro");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["progress"], 100);
    assert_eq!(value["nerva"]["operation"], "create");

    let character = VideoCharacterRecord {
        id: "vchar-1".to_string(),
        created_at: 41,
        name: "Casey".to_string(),
        metadata: json!({"purpose": "test"}),
        source_video_id: Some("video-1".to_string()),
    };
    let character_json = video_character_json(&character);
    assert_eq!(character_json["id"], "vchar-1");
    assert_eq!(character_json["name"], "Casey");
    assert_eq!(character_json["source_video_id"], "video-1");

    assert!(
        parse_video_json_request(
            VideoOperation::Create,
            &json!({"prompt": "x", "seconds": "9"})
        )
        .is_err()
    );
    assert!(
        parse_video_json_request(
            VideoOperation::Create,
            &json!({"prompt": "x", "size": "640x480"})
        )
        .is_err()
    );
}

#[test]
fn parses_responses_input_messages() {
    let prompt = responses_input_to_prompt(&json!({
        "instructions": "be useful",
        "input": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    assert!(prompt.contains("System: be useful"));
    assert!(prompt.contains("User: hello"));
}

#[test]
fn builds_assistant_records_and_prompt_body() {
    let assistant = AssistantRecord {
        id: "asst_1".to_string(),
        object: "assistant",
        created_at: 10,
        name: Some("Helper".to_string()),
        description: None,
        model: "gpt-5".to_string(),
        instructions: Some("Be concise.".to_string()),
        tools: vec![json!({"type": "file_search"})],
        tool_resources: json!({"file_search": {"vector_store_ids": ["vs_1"]}}),
        metadata: json!({"team": "nerva"}),
        temperature: Some(0.2),
        top_p: Some(1.0),
        response_format: json!("auto"),
        reasoning_effort: Some("minimal".to_string()),
    };
    assert_eq!(assistant_json(&assistant)["object"], "assistant");
    assert_eq!(
        assistant_json(&assistant)["tools"][0]["type"],
        "file_search"
    );

    let content = normalize_assistant_message_content(&json!([
        {"type": "text", "text": "hello assistant"},
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
    ]))
    .unwrap();
    assert_eq!(assistant_message_text(&content), "hello assistant");

    let message = AssistantMessageRecord {
        id: "msg_1".to_string(),
        object: "thread.message",
        created_at: 11,
        thread_id: "thread_1".to_string(),
        status: "completed".to_string(),
        incomplete_details: Value::Null,
        completed_at: Some(11),
        incomplete_at: None,
        role: "user".to_string(),
        content,
        assistant_id: None,
        run_id: None,
        attachments: vec![json!({"file_id": "file_1", "tools": [{"type": "file_search"}]})],
        metadata: json!({}),
    };
    assert_eq!(assistant_message_json(&message)["role"], "user");

    let mut thread = AssistantThreadRecord {
        id: "thread_1".to_string(),
        object: "thread",
        created_at: 11,
        metadata: json!({"purpose": "test"}),
        tool_resources: json!({}),
        messages: HashMap::new(),
        message_order: vec!["msg_1".to_string()],
        runs: HashMap::new(),
        run_order: Vec::new(),
    };
    thread.messages.insert(message.id.clone(), message);
    let chat_body = assistant_run_chat_body(
        &assistant,
        &thread,
        &json!({"additional_instructions": "No fluff."}),
    );
    assert_eq!(chat_body["messages"][0]["role"], "system");
    assert!(
        chat_body["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("No fluff.")
    );
    assert_eq!(chat_body["messages"][1]["content"], "hello assistant");
    assert_eq!(assistant_thread_json(&thread)["object"], "thread");

    let step = AssistantRunStepRecord {
        id: "step_1".to_string(),
        object: "thread.run.step",
        created_at: 12,
        assistant_id: assistant.id.clone(),
        thread_id: thread.id.clone(),
        run_id: "run_1".to_string(),
        step_type: "message_creation".to_string(),
        status: "completed".to_string(),
        completed_at: Some(12),
        cancelled_at: None,
        failed_at: None,
        expired_at: None,
        last_error: Value::Null,
        step_details: json!({
            "type": "message_creation",
            "message_creation": {"message_id": "msg_2"}
        }),
        usage: json!({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}),
    };
    assert_eq!(assistant_run_step_json(&step)["type"], "message_creation");

    let run = AssistantRunRecord {
        id: "run_1".to_string(),
        object: "thread.run",
        created_at: 12,
        thread_id: thread.id,
        assistant_id: assistant.id,
        status: "completed".to_string(),
        started_at: Some(12),
        expires_at: None,
        cancelled_at: None,
        failed_at: None,
        completed_at: Some(12),
        required_action: Value::Null,
        last_error: Value::Null,
        incomplete_details: Value::Null,
        model: "gpt-5".to_string(),
        instructions: Some("Be concise.\nNo fluff.".to_string()),
        tools: vec![json!({"type": "file_search"})],
        tool_resources: json!({}),
        metadata: json!({}),
        temperature: Some(0.2),
        top_p: Some(1.0),
        response_format: json!("auto"),
        parallel_tool_calls: true,
        max_prompt_tokens: None,
        max_completion_tokens: Some(256),
        usage: json!({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}),
        steps: vec![step],
    };
    assert_eq!(assistant_run_json(&run)["object"], "thread.run");
    assert_eq!(assistant_run_json(&run)["status"], "completed");
}

#[test]
fn builds_chatkit_session_thread_and_items() {
    let content = normalize_chatkit_content(&json!("I need help"), "user_message").unwrap();
    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[0]["text"], "I need help");

    let assistant_content = normalize_chatkit_content(
        &json!({"type": "text", "value": "Use this fix"}),
        "assistant_message",
    )
    .unwrap();
    assert_eq!(assistant_content[0]["type"], "output_text");
    assert_eq!(assistant_content[0]["text"], "Use this fix");

    let item = ChatKitThreadItemRecord {
        id: "cthi_1".to_string(),
        object: "chatkit.thread_item",
        created_at: 21,
        item_type: "user_message".to_string(),
        content: content.clone(),
        attachments: vec![json!({"file_id": "file_1"})],
        metadata: json!({"source": "test"}),
    };
    assert_eq!(
        chatkit_thread_item_json(&item)["object"],
        "chatkit.thread_item"
    );
    assert_eq!(
        chatkit_thread_item_json(&item)["content"][0]["type"],
        "input_text"
    );

    let thread = ChatKitThreadRecord {
        id: "cthr_1".to_string(),
        object: "chatkit.thread",
        created_at: 20,
        title: Some("Support thread".to_string()),
        user: Some("user_1".to_string()),
        status: json!({"type": "active"}),
        items: vec![item],
        metadata: json!({"purpose": "test"}),
    };
    let thread_json = chatkit_thread_json(&thread);
    assert_eq!(thread_json["object"], "chatkit.thread");
    assert_eq!(thread_json["items"]["data"].as_array().unwrap().len(), 1);
    assert_eq!(thread_json["items"]["first_id"], "cthi_1");

    let session = ChatKitSessionRecord {
        id: "cksess_1".to_string(),
        object: "chatkit.session",
        created_at: 19,
        expires_at: 619,
        cancelled_at: None,
        status: "active".to_string(),
        client_secret: "chatkit_token_test".to_string(),
        workflow: json!({
            "id": "workflow_alpha",
            "state_variables": null,
            "tracing": {"enabled": true},
            "version": null
        }),
        scope: json!({"project": "nerva"}),
        user: Some("user_1".to_string()),
        chatkit_configuration: json!({
            "automatic_thread_titling": {"enabled": true},
            "file_upload": {"enabled": false, "max_file_size": 512, "max_files": 10},
            "history": {"enabled": true, "recent_threads": null}
        }),
        rate_limits: json!({"max_requests_per_1_minute": 10}),
        max_requests_per_1_minute: Some(10),
        max_requests_per_session: Some(500),
        ttl_seconds: 600,
        thread_id: Some(thread.id),
    };
    let session_json = chatkit_session_json(&session);
    assert_eq!(session_json["object"], "chatkit.session");
    assert_eq!(session_json["thread_id"], "cthr_1");
    assert_eq!(session_json["rate_limits"]["max_requests_per_1_minute"], 10);
}

#[test]
fn parses_stop_strings() {
    assert_eq!(
        request_stop_strings(&json!({"stop": ["END", "STOP"]})).unwrap(),
        vec!["END".to_string(), "STOP".to_string()]
    );
    let (text, stopped) = apply_stop_strings("hello END world".to_string(), &["END".into()]);
    assert_eq!(text, "hello ");
    assert!(stopped);
}

#[test]
fn formats_completion_echo_and_suffix_text() {
    assert_eq!(completion_text("answer", None, None), "answer");
    assert_eq!(
        completion_text("answer", Some("prompt "), Some(" done")),
        "prompt answer done"
    );
}

#[test]
fn serializes_generated_cache_and_session_metadata() {
    let generated = GeneratedText {
        text: "ok".to_string(),
        token_ids: vec![1, 2],
        prompt_tokens: 3,
        finish_reason: "stop",
        prompt_hash: 0x12ab,
        cache_key: "prompt:12ab".to_string(),
        cache_hit: true,
        session_id: Some("sess-1".to_string()),
    };
    assert_eq!(
        generated_metadata(&generated),
        json!({
            "cache_key": "prompt:12ab",
            "cache_hit": true,
            "session_id": "sess-1",
            "prompt_hash": "00000000000012ab"
        })
    );
}

#[test]
fn computes_stream_text_delta() {
    assert_eq!(text_delta("hello", "hello world"), " world");
    assert_eq!(text_delta("abc", "xyz"), "xyz");
}

#[test]
fn splits_deepseek_reasoning_markers() {
    let split = split_generated_reasoning("work</think>answer", ReasoningMode::DeepSeekThinking);
    assert_eq!(split.reasoning, "work");
    assert_eq!(split.content, "answer");

    let split = split_generated_reasoning("<think>why</think>done", ReasoningMode::DeepSeekChat);
    assert_eq!(split.reasoning, "why");
    assert_eq!(split.content, "done");

    let split = split_generated_reasoning("plain", ReasoningMode::DeepSeekChat);
    assert_eq!(split.reasoning, "");
    assert_eq!(split.content, "plain");
}

#[test]
fn maps_deepseek_reasoning_mode_to_prompt_format() {
    assert_eq!(
        prompt_format_for_reasoning(ReasoningMode::None),
        PromptFormat::Auto
    );
    assert_eq!(
        prompt_format_for_reasoning(ReasoningMode::DeepSeekChat),
        PromptFormat::DeepSeekChat
    );
    assert_eq!(
        prompt_format_for_reasoning(ReasoningMode::DeepSeekThinking),
        PromptFormat::DeepSeekThinking
    );
}

#[test]
fn renders_deepseek_v4_multi_turn_prompt_like_vllm() {
    let prompt = chat_prompt_for_reasoning(
        &json!({
            "thinking": true,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "reasoning_content": "The user said hello, I should greet back.",
                    "content": "Hi there! How can I help you?"
                },
                {"role": "user", "content": "What is the capital of France?"},
                {
                    "role": "assistant",
                    "reasoning_content": "The user asks about the capital of France. It is Paris.",
                    "content": "The capital of France is Paris."
                }
            ]
        }),
        ReasoningMode::DeepSeekThinking,
    )
    .unwrap();

    let PromptInput::Text { text, format } = prompt else {
        panic!("DeepSeek chat prompt should render text");
    };
    assert_eq!(format, PromptFormat::Raw);
    assert_eq!(
        text,
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello<｜Assistant｜></think>Hi there! How can I help you?<｜end▁of▁sentence｜><｜User｜>What is the capital of France?<｜Assistant｜><think>The user asks about the capital of France. It is Paris.</think>The capital of France is Paris.<｜end▁of▁sentence｜>"
    );
}

#[test]
fn renders_deepseek_v4_tools_as_dsml_prompt_context() {
    let prompt = chat_prompt_for_reasoning(
        &json!({
            "thinking": true,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            }],
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "reasoning": "Need weather.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
            ]
        }),
        ReasoningMode::DeepSeekThinking,
    )
    .unwrap();

    let PromptInput::Text { text, format } = prompt else {
        panic!("DeepSeek chat prompt should render text");
    };
    assert_eq!(format, PromptFormat::Raw);
    assert!(text.contains("## Tools"));
    assert!(text.contains("<｜DSML｜tool_calls>"));
    assert!(text.contains("<｜DSML｜invoke name=\"get_weather\">"));
    assert!(
        text.contains("<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>")
    );
    assert!(text.contains("<｜User｜><tool_result>sunny</tool_result><｜Assistant｜><think>"));
}

#[test]
fn renders_deepseek_v4_max_and_none_reasoning_effort() {
    let max_prompt = chat_prompt_for_reasoning(
        &json!({
            "thinking": true,
            "reasoning_effort": "xhigh",
            "messages": [{"role": "user", "content": "solve it"}]
        }),
        ReasoningMode::DeepSeekThinking,
    )
    .unwrap();
    let PromptInput::Text { text, .. } = max_prompt else {
        panic!("DeepSeek chat prompt should render text");
    };
    assert!(text.starts_with("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"));
    assert!(text.ends_with("<｜Assistant｜><think>"));

    let none_prompt = chat_prompt_for_reasoning(
        &json!({
            "thinking": true,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "answer"}]
        }),
        ReasoningMode::DeepSeekThinking,
    )
    .unwrap();
    let PromptInput::Text { text, .. } = none_prompt else {
        panic!("DeepSeek chat prompt should render text");
    };
    assert_eq!(
        text,
        "<｜begin▁of▁sentence｜><｜User｜>answer<｜Assistant｜></think>"
    );
}

#[test]
fn streams_chat_reasoning_delta_as_reasoning_field() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(2);
    let meta = StreamMeta {
        id: "chatcmpl-test".to_string(),
        created: 7,
        model: "deepseek-test".to_string(),
        response: None,
    };
    let mut response_reasoning_started = false;

    assert!(send_stream_reasoning_delta(
        &tx,
        StreamKind::ChatCompletion,
        &meta,
        "thinking",
        &mut response_reasoning_started,
    ));
    drop(tx);

    let frame = rx.blocking_recv().unwrap();
    let frame = std::str::from_utf8(frame.as_ref()).unwrap();
    assert!(frame.contains("\"reasoning\":\"thinking\""));
    assert!(frame.contains("\"reasoning_content\":\"thinking\""));
    assert!(!frame.contains("\"content\":\"thinking\""));
}

#[test]
fn builds_streamed_response_completed_payload() {
    let (tx, _rx) = tokio::sync::mpsc::channel(4);
    let meta = StreamMeta {
        id: "resp-test".to_string(),
        created: 7,
        model: "deepseek-test".to_string(),
        response: Some(ResponseStreamOptions {
            store: true,
            metadata: json!({"trace": "abc"}),
            previous_response_id: Some("resp-prev".to_string()),
            conversation_id: None,
            input_items: vec![json!({"id": "in-1", "role": "user", "content": "hello"})],
        }),
    };
    let mut emitted = StreamEmissionState::default();
    assert!(emit_stream_text(
        &tx,
        StreamKind::Response,
        &meta,
        false,
        ReasoningMode::None,
        &mut emitted,
        "answer",
    ));

    let response = response_stream_completed_response(&meta, &emitted, 3, 2).unwrap();
    assert_eq!(response["id"], "resp-test");
    assert_eq!(response["metadata"], json!({"trace": "abc"}));
    assert_eq!(response["store"], true);
    assert_eq!(response["previous_response_id"], "resp-prev");
    assert_eq!(response["conversation"], Value::Null);
    assert_eq!(response["output_text"], "answer");
    assert_eq!(response["usage"]["total_tokens"], 5);
    assert_eq!(response["output"][0]["content"][0]["text"], "answer");
}

#[test]
fn builds_organization_usage_and_cost_pages() {
    let window =
        organization_query_window("start_time=100&end_time=200&bucket_width=1h&limit=1", 300)
            .unwrap();
    assert_eq!(window.start_time, 100);
    assert_eq!(window.end_time, 200);
    assert_eq!(window.bucket_width, "1h");
    assert!(organization_query_window("start_time=200&end_time=100", 300).is_err());
    assert!(organization_query_window("bucket_width=5m", 300).is_err());

    let totals = OrganizationUsageTotals {
        input_tokens: 10,
        output_tokens: 20,
        cached_tokens: 3,
        requests: 4,
        images: 2,
        audio_seconds: 7,
        audio_characters: 12,
        vector_store_bytes: 1024,
        code_interpreter_sessions: 1,
        file_search_calls: 5,
        web_search_calls: 6,
    };
    let completions =
        organization_usage_page_json(OrganizationUsageKind::Completions, &window, &totals);
    assert_eq!(completions["object"], "page");
    assert_eq!(completions["data"][0]["object"], "bucket");
    assert_eq!(
        completions["data"][0]["results"][0]["object"],
        "organization.usage.completions.result"
    );
    assert_eq!(completions["data"][0]["results"][0]["input_tokens"], 10);
    assert_eq!(completions["data"][0]["results"][0]["output_tokens"], 20);
    assert_eq!(
        completions["data"][0]["results"][0]["num_model_requests"],
        4
    );

    let vector_stores =
        organization_usage_page_json(OrganizationUsageKind::VectorStores, &window, &totals);
    assert_eq!(
        vector_stores["data"][0]["results"][0]["object"],
        "organization.usage.vector_stores.result"
    );
    assert_eq!(vector_stores["data"][0]["results"][0]["usage_bytes"], 1024);

    let costs = organization_costs_page_json(&window, &totals);
    assert_eq!(costs["object"], "page");
    assert_eq!(
        costs["data"][0]["results"][0]["object"],
        "organization.costs.result"
    );
    assert_eq!(costs["data"][0]["results"][0]["amount"]["currency"], "usd");
    assert_eq!(costs["has_more"], false);
}

#[test]
fn maps_finish_reason_to_openai_values() {
    assert_eq!(finish_reason(HfCausalLmStopReason::EosToken, false), "stop");
    assert_eq!(finish_reason(HfCausalLmStopReason::MaxSteps, true), "stop");
    assert_eq!(
        finish_reason(HfCausalLmStopReason::MaxSteps, false),
        "length"
    );
}

#[test]
fn rejects_unsupported_generation_options() {
    assert!(reject_unsupported_generation_options(&json!({})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"echo": true})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"suffix": " done"})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"presence_penalty": 0.0})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"presence_penalty": 1.0})).is_err());
    assert!(reject_unsupported_generation_options(&json!({"tools": [{"type": "mcp"}]})).is_err());
    assert!(
        reject_unsupported_generation_options(&json!({"response_format": {"type": "text"}}))
            .is_ok()
    );
    assert!(
        reject_unsupported_generation_options(&json!({"response_format": {"type": "json_object"}}))
            .is_ok()
    );
    assert!(
        reject_unsupported_generation_options(&json!({"response_format": {"type": "yaml"}}))
            .is_err()
    );
}

#[test]
fn parses_response_format_prompt_instructions() {
    assert!(
        request_response_format_instruction(&json!({"response_format": {"type": "text"}}))
            .unwrap()
            .is_none()
    );
    assert!(
        request_response_format_instruction(&json!({"response_format": {"type": "json_object"}}))
            .unwrap()
            .unwrap()
            .contains("valid JSON object")
    );
    assert!(
        request_response_format_instruction(&json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"}
                }
            }
        }))
        .unwrap()
        .unwrap()
        .contains("schema 'answer'")
    );

    let prompt =
        append_assistant_instruction("User: reply\nAssistant:".to_string(), "Respond with JSON.");
    assert!(prompt.contains("System: Respond with JSON."));
    assert!(prompt.ends_with("Assistant:"));
}

#[test]
fn tool_aware_generation_validation_accepts_mcp_and_function_tools() {
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{
                "type": "mcp",
                "server_id": "search",
                "name": "lookup"
            }],
            "tool_choice": "auto"
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{
                "type": "function",
                "function": {"name": "lookup", "description": "search docs"}
            }]
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "functions": [{"name": "lookup", "description": "search docs"}]
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{"type": "file_search"}]
        }))
        .is_err()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{"type": "mcp"}],
            "tool_choice": 7
        }))
        .is_err()
    );
}

#[test]
fn function_tools_are_prompt_context_not_mcp_invocations() {
    assert!(
        mcp_tool_invocation_from_request(&json!({
            "tools": [{
                "type": "function",
                "function": {"name": "lookup"}
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup"}
            }
        }))
        .unwrap()
        .is_none()
    );

    let augmented = augment_prompt_with_mcp_tool(
        "User: answer this\nAssistant:".to_string(),
        None,
        &json!({
            "functions": [{
                "name": "lookup",
                "description": "search docs",
                "parameters": {"type": "object"}
            }]
        }),
    );
    assert!(augmented.contains("Available tools, if needed:"));
    assert!(augmented.contains("function lookup: search docs"));
    assert!(augmented.ends_with("Assistant:"));
}

#[test]
fn parses_explicit_mcp_tool_invocation_from_request() {
    let invocation = mcp_tool_invocation_from_request(&json!({
        "tools": [{
            "type": "mcp",
            "server_id": "search",
            "name": "lookup"
        }],
        "tool_choice": {
            "type": "mcp",
            "name": "lookup",
            "arguments": "{\"query\":\"rust\"}"
        }
    }))
    .unwrap()
    .unwrap();

    assert_eq!(invocation.server_id.as_deref(), Some("search"));
    assert_eq!(invocation.name, "lookup");
    assert_eq!(invocation.arguments["query"], "rust");
}

#[test]
fn parses_openai_remote_mcp_tool_options() {
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "authorization": "token",
                "allowed_tools": ["ask_question"],
                "require_approval": {
                    "never": {"tool_names": ["ask_question"]}
                },
                "defer_loading": true
            }]
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{
                "type": "mcp",
                "connector_id": "connector_googlecalendar",
                "authorization": "token",
                "require_approval": "never"
            }]
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{
                "type": "mcp",
                "server_label": "bad",
                "server_url": "https://example.test/mcp",
                "allowed_tools": "ask_question"
            }]
        }))
        .is_err()
    );

    let invocation = mcp_tool_invocation_from_request(&json!({
        "tools": [{
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "authorization": "token",
            "allowed_tools": ["ask_question"],
            "require_approval": {
                "never": {"tool_names": ["ask_question"]}
            }
        }],
        "mcp_call": {
            "type": "mcp",
            "name": "ask_question",
            "arguments": {"repoName": "openai/openai-openapi", "question": "shape?"}
        }
    }))
    .unwrap()
    .unwrap();

    assert_eq!(invocation.server_label.as_deref(), Some("deepwiki"));
    assert_eq!(
        invocation.server_url.as_deref(),
        Some("https://mcp.deepwiki.com/mcp")
    );
    assert_eq!(invocation.authorization.as_deref(), Some("token"));
    assert_eq!(invocation.allowed_tools, vec!["ask_question".to_string()]);
    assert_eq!(
        invocation.require_approval["never"]["tool_names"][0],
        "ask_question"
    );
}

#[test]
fn augments_prompt_with_mcp_tool_result_before_assistant_marker() {
    let prompt = "User: answer this\nAssistant:".to_string();
    let result = McpToolResult {
        server_id: "search".to_string(),
        server_label: Some("search".to_string()),
        name: "lookup".to_string(),
        arguments: json!({"query": "rust"}),
        result: json!({"answer": "Ferris"}),
    };

    let augmented = augment_prompt_with_mcp_tool(prompt, Some(&result), &json!({}));

    assert!(augmented.contains("MCP tool result from server 'search' tool 'lookup'"));
    assert!(augmented.contains("\"answer\":\"Ferris\""));
    assert!(augmented.ends_with("Assistant:"));
}

#[test]
fn parses_n_and_optional_strings() {
    assert_eq!(request_n(&json!({})).unwrap(), 1);
    assert_eq!(request_n(&json!({"n": 3})).unwrap(), 3);
    assert!(request_n(&json!({"n": 0})).is_err());
    assert_eq!(
        request_optional_string(&json!({"session_id": "abc"}), "session_id").unwrap(),
        Some("abc".to_string())
    );
    assert!(request_optional_string(&json!({"session_id": ""}), "session_id").is_err());
}

#[test]
fn parses_response_store_controls_and_input_items() {
    assert!(request_store(&json!({})).unwrap());
    assert!(!request_store(&json!({"store": false})).unwrap());
    assert!(!request_chat_store(&json!({})).unwrap());
    assert!(request_chat_store(&json!({"store": true})).unwrap());
    assert_eq!(
        request_metadata(&json!({"metadata": {"trace": "abc"}})).unwrap(),
        json!({"trace": "abc"})
    );
    assert!(request_metadata(&json!({"metadata": "bad"})).is_err());

    let items = response_input_items(&json!({"input": "hello"}));
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["role"], "user");
    assert_eq!(items[0]["content"][0]["text"], "hello");

    let items = response_input_items(&json!({"input": [{"role": "user", "content": "hi"}]}));
    assert_eq!(items, vec![json!({"role": "user", "content": "hi"})]);
}

#[test]
fn parses_upload_create_request_and_part_bodies() {
    let fields = parse_upload_create_request(&json!({
        "filename": "batch.jsonl",
        "purpose": "batch",
        "bytes": 5,
        "mime_type": "application/jsonl"
    }))
    .unwrap();
    assert_eq!(fields.filename, "batch.jsonl");
    assert_eq!(fields.purpose, "batch");
    assert_eq!(fields.bytes, 5);
    assert_eq!(fields.mime_type, "application/jsonl");

    let request = actix_web::test::TestRequest::default()
        .insert_header(("content-type", "multipart/form-data; boundary=nerva"))
        .to_http_request();
    let body = b"--nerva\r\nContent-Disposition: form-data; name=\"data\"; filename=\"part.bin\"\r\nContent-Type: application/octet-stream\r\n\r\nhello\r\n--nerva--\r\n";
    assert_eq!(
        parse_upload_part_body(&request, body).unwrap(),
        b"hello".to_vec()
    );
}

#[test]
fn builds_upload_content_in_requested_part_order() {
    let record = UploadRecord {
        id: "upload_1".to_string(),
        object: "upload",
        bytes: 11,
        created_at: 1,
        expires_at: 2,
        filename: "batch.jsonl".to_string(),
        purpose: "batch".to_string(),
        mime_type: "application/jsonl".to_string(),
        status: "pending".to_string(),
        file_id: None,
        parts: vec![
            UploadPartRecord {
                id: "part_b".to_string(),
                object: "upload.part",
                created_at: 1,
                upload_id: "upload_1".to_string(),
                bytes: 6,
                content: b" world".to_vec(),
            },
            UploadPartRecord {
                id: "part_a".to_string(),
                object: "upload.part",
                created_at: 1,
                upload_id: "upload_1".to_string(),
                bytes: 5,
                content: b"hello".to_vec(),
            },
        ],
    };
    let content =
        upload_content_for_part_ids(&record, &["part_a".to_string(), "part_b".to_string()])
            .unwrap();
    assert_eq!(content, b"hello world".to_vec());
    assert!(
        upload_content_for_part_ids(&record, &["part_a".to_string(), "part_a".to_string()])
            .is_err()
    );
}

#[test]
fn normalizes_container_file_paths_and_content_types() {
    assert_eq!(
        container_file_path(&json!({"path": "work/result.txt"}), "result.txt"),
        "/mnt/data/work/result.txt"
    );
    assert_eq!(
        container_file_path(&json!({"path": "/tmp/result.json"}), "result.json"),
        "/tmp/result.json"
    );
    assert_eq!(
        container_file_path(&json!({}), "notes.md"),
        "/mnt/data/notes.md"
    );
    assert_eq!(container_file_content_type("data.json"), "application/json");
    assert_eq!(
        container_file_content_type("script.py"),
        "text/plain; charset=utf-8"
    );
}

#[test]
fn parses_skill_json_payloads() {
    let payload = parse_skill_json_payload(&json!({
        "name": "rust-review",
        "description": "Review Rust code",
        "metadata": {"owner": "nerva"},
        "content_type": "text/markdown; charset=utf-8",
        "content": "# Rust review\nCheck safety invariants."
    }))
    .unwrap();

    assert_eq!(payload.name, "rust-review");
    assert_eq!(payload.description.as_deref(), Some("Review Rust code"));
    assert_eq!(payload.metadata, json!({"owner": "nerva"}));
    assert_eq!(payload.content_type, "text/markdown; charset=utf-8");
    assert_eq!(
        String::from_utf8(payload.content).unwrap(),
        "# Rust review\nCheck safety invariants."
    );
    assert!(parse_skill_json_payload(&json!({"name": "missing-content"})).is_err());
}

#[test]
fn counts_vector_store_files_and_scores_lexical_matches() {
    let mut record = VectorStoreRecord {
        id: "vs_1".to_string(),
        object: "vector_store",
        created_at: 1,
        name: Some("docs".to_string()),
        metadata: json!({}),
        status: "completed".to_string(),
        expires_after: json!({"anchor": "last_active_at", "days": 7}),
        expires_at: None,
        last_active_at: Some(1),
        files: HashMap::new(),
        file_batches: HashMap::new(),
    };
    record.files.insert(
        "file_1".to_string(),
        VectorStoreFileRecord {
            id: "file_1".to_string(),
            object: "vector_store.file",
            created_at: 1,
            vector_store_id: "vs_1".to_string(),
            status: "completed".to_string(),
            last_error: None,
            usage_bytes: 12,
            attributes: json!({"topic": "rust"}),
        },
    );
    record.files.insert(
        "file_2".to_string(),
        VectorStoreFileRecord {
            id: "file_2".to_string(),
            object: "vector_store.file",
            created_at: 1,
            vector_store_id: "vs_1".to_string(),
            status: "cancelled".to_string(),
            last_error: None,
            usage_bytes: 3,
            attributes: Value::Null,
        },
    );

    let counts = vector_store_file_counts(&record);
    assert_eq!(counts.total, 2);
    assert_eq!(counts.completed, 1);
    assert_eq!(counts.cancelled, 1);
    assert!(lexical_match_score("rust server", "a Rust HTTP server") > 0.0);
    assert_eq!(lexical_match_score("missing", "a Rust HTTP server"), 0.0);
}

#[test]
fn builds_eval_output_items_and_result_counts() {
    let items = eval_output_items_from_inline_content(
        "eval_1",
        "run_1",
        &json!([
            {
                "item": {"question": "one"},
                "sample": {"answer": "two"},
                "results": [{"score": 1.0}]
            },
            {
                "input": {"question": "three"},
                "expected_output": {"answer": "four"},
                "passed": false
            }
        ]),
    );

    assert_eq!(items.len(), 2);
    assert_eq!(items[0].item, json!({"question": "one"}));
    assert_eq!(items[0].sample, json!({"answer": "two"}));
    assert_eq!(items[0].results, vec![json!({"score": 1.0})]);
    assert_eq!(items[1].item, json!({"question": "three"}));
    assert_eq!(items[1].sample, json!({"answer": "four"}));
    assert_eq!(items[1].status, "failed");

    let counts = eval_run_result_counts(items.iter());
    assert_eq!(counts.total, 2);
    assert_eq!(counts.passed, 1);
    assert_eq!(counts.failed, 1);
    assert_eq!(counts.errored, 0);
}

#[test]
fn builds_stable_fine_tuned_model_names() {
    assert_eq!(
        fine_tuned_model_name("deepseek-ai/DeepSeek-V3", Some("tenant a"), "ftjob_1"),
        "ft:deepseek-ai-DeepSeek-V3:tenant-a:ftjob_1"
    );
    assert_eq!(
        fine_tuned_model_name("served", None, "ftjob_2"),
        "ft:served:nerva:ftjob_2"
    );
}

#[test]
fn builds_fine_tuning_checkpoint_permission_payloads() {
    let project_ids = request_project_ids(&json!({
        "project_ids": ["proj_a", "proj_b", "proj_a"]
    }))
    .unwrap();
    assert_eq!(project_ids, vec!["proj_a", "proj_b"]);
    assert!(request_project_ids(&json!({"project_ids": []})).is_err());
    assert!(request_project_ids(&json!({"project_ids": [""]})).is_err());

    let first = FineTuningCheckpointPermissionRecord {
        id: "cp_1".to_string(),
        object: "checkpoint.permission",
        created_at: 10,
        project_id: "proj_a".to_string(),
    };
    let second = FineTuningCheckpointPermissionRecord {
        id: "cp_2".to_string(),
        object: "checkpoint.permission",
        created_at: 11,
        project_id: "proj_b".to_string(),
    };

    let first_json = checkpoint_permission_json(&first);
    assert_eq!(first_json["id"], "cp_1");
    assert_eq!(first_json["object"], "checkpoint.permission");
    assert_eq!(first_json["created_at"], 10);
    assert_eq!(first_json["project_id"], "proj_a");

    let list = checkpoint_permission_list_json(vec![&first, &second], false);
    assert_eq!(list["object"], "list");
    assert_eq!(list["first_id"], "cp_1");
    assert_eq!(list["last_id"], "cp_2");
    assert_eq!(list["has_more"], false);
    assert_eq!(list["data"][1]["project_id"], "proj_b");
}

#[test]
fn extracts_chat_completion_request_messages() {
    let messages = chat_completion_request_messages(&json!({
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}
        ]
    }));
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["content"], "hi");
}

#[test]
fn parses_conversation_id_and_renders_items() {
    assert_eq!(
        request_conversation_id(&json!({"conversation": "conv_123"})).unwrap(),
        Some("conv_123".to_string())
    );
    assert_eq!(
        request_conversation_id(&json!({"conversation": {"id": "conv_456"}})).unwrap(),
        Some("conv_456".to_string())
    );
    assert!(request_conversation_id(&json!({"conversation": {}})).is_err());

    let prompt = conversation_items_prompt(&[
        json!({
            "id": "msg-user",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}]
        }),
        json!({
            "id": "msg-assistant",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi back"}]
        }),
    ])
    .unwrap();
    assert_eq!(prompt, "User: hello\nAssistant: hi back\n");
}

#[test]
fn extracts_stored_response_output_text() {
    assert_eq!(
        response_output_text(&json!({"output_text": "direct"})),
        "direct"
    );
    assert_eq!(
        response_output_text(&json!({
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "nested"}]
            }]
        })),
        "nested"
    );
}

#[test]
fn builds_cancelled_and_compacted_response_values() {
    let response = json!({
        "id": "resp_1",
        "status": "completed",
        "output_text": "answer",
        "error": null,
        "incomplete_details": null
    });

    let cancelled = cancelled_response_value(response.clone());
    assert_eq!(cancelled["status"], "cancelled");
    assert_eq!(cancelled["incomplete_details"]["reason"], "cancelled");
    assert!(cancelled["error"].is_null());

    let compaction = compact_response_item("cmpct_1".to_string(), &response, &[json!("hello")]);
    assert_eq!(compaction["id"], "cmpct_1");
    assert_eq!(compaction["type"], "compaction");
    let encrypted_content = compaction["encrypted_content"].as_str().unwrap();
    assert!(encrypted_content.contains("Input:\nhello"));
    assert!(encrypted_content.contains("Output:\nanswer"));
}

#[test]
fn hashes_tokens_stably() {
    let a = hash_tokens(&[
        nerva_core::types::id::token::TokenId(1),
        nerva_core::types::id::token::TokenId(2),
    ]);
    let b = hash_tokens(&[
        nerva_core::types::id::token::TokenId(1),
        nerva_core::types::id::token::TokenId(2),
    ]);
    let c = hash_tokens(&[
        nerva_core::types::id::token::TokenId(2),
        nerva_core::types::id::token::TokenId(1),
    ]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn serializes_session_records() {
    let record = SessionRecord {
        id: "sess-1".to_string(),
        object: "session",
        created: 1,
        updated: 2,
        request_count: 3,
        prompt_tokens: 4,
        generated_tokens: 5,
        last_cache_key: Some("cache".to_string()),
        last_prompt_hash: Some(0xabcd),
    };
    let value = session_json(&record);
    assert_eq!(value["id"], "sess-1");
    assert_eq!(value["last_prompt_hash"], "000000000000abcd");
}

#[test]
fn builds_realtime_session_and_transcription_configs() {
    let secret = RealtimeClientSecretRecord {
        value: "ek_nerva_test".to_string(),
        expires_at: 70,
    };
    let realtime = realtime_config_value(
        "local-model",
        RealtimeSessionKind::Realtime,
        &json!({
            "model": "gpt-realtime-1.5",
            "output_modalities": ["audio"],
            "temperature": 0.8,
            "speed": 1.0,
            "tools": [{"type": "function", "name": "lookup"}]
        }),
        "sess-rt",
        1800,
        &secret,
    )
    .unwrap();
    assert_eq!(realtime["id"], "sess-rt");
    assert_eq!(realtime["object"], "realtime.session");
    assert_eq!(realtime["type"], "realtime");
    assert_eq!(realtime["model"], "gpt-realtime-1.5");
    assert_eq!(realtime["client_secret"]["value"], "ek_nerva_test");
    assert_eq!(realtime["audio"]["input"]["format"], "pcm16");
    assert_eq!(realtime["turn_detection"]["type"], "server_vad");
    assert_eq!(realtime["tool_choice"], "auto");

    let transcription = realtime_config_value(
        "local-model",
        RealtimeSessionKind::Transcription,
        &json!({
            "input_audio_format": "g711_ulaw",
            "input_audio_transcription": {"model": "gpt-4o-transcribe", "language": "en"}
        }),
        "sess-tr",
        1801,
        &secret,
    )
    .unwrap();
    assert_eq!(transcription["object"], "realtime.transcription_session");
    assert_eq!(transcription["type"], "realtime.transcription_session");
    assert_eq!(transcription["input_audio_format"], "g711_ulaw");
    assert_eq!(transcription["modalities"][0], "text");

    assert!(
        realtime_config_value(
            "local-model",
            RealtimeSessionKind::Realtime,
            &json!({"input_audio_format": "mp3"}),
            "bad",
            1,
            &secret,
        )
        .is_err()
    );
}

#[test]
fn builds_realtime_translation_client_secret_payloads() {
    assert_eq!(
        translation_secret_expires_at(
            &json!({
                "expires_after": {
                    "anchor": "created_at",
                    "seconds": 600
                }
            }),
            100,
        )
        .unwrap(),
        700
    );
    assert!(translation_secret_expires_at(&json!({"expires_after": {"seconds": 1}}), 100).is_err());
    assert!(
        translation_secret_expires_at(&json!({"expires_after": {"anchor": "updated_at"}}), 100)
            .is_err()
    );

    let secret = RealtimeClientSecretRecord {
        value: "ek_translate_test".to_string(),
        expires_at: 700,
    };
    let translation = realtime_config_value(
        "local-model",
        RealtimeSessionKind::Translation,
        &json!({
            "model": "gpt-realtime-translate",
            "audio": {
                "input": {
                    "transcription": {"model": "gpt-realtime-whisper"},
                    "noise_reduction": null
                },
                "output": {"language": "es"}
            }
        }),
        "sess-trn",
        700,
        &secret,
    )
    .unwrap();
    assert_eq!(translation["type"], "translation");
    assert_eq!(translation["model"], "gpt-realtime-translate");
    assert_eq!(
        translation["audio"]["input"]["transcription"]["model"],
        "gpt-realtime-whisper"
    );
    assert_eq!(translation["audio"]["output"]["language"], "es");

    let record = RealtimeSessionRecord {
        id: "sess-trn".to_string(),
        object: "realtime.translation_session",
        kind: "translation",
        created_at: 100,
        expires_at: 700,
        client_secret: secret,
        config: translation,
    };
    let session = realtime_translation_session_json(&record);
    assert_eq!(session["id"], "sess-trn");
    assert_eq!(session["type"], "translation");
    assert_eq!(session["expires_at"], 700);

    let client_secret = realtime_translation_client_secret_json(&record);
    assert_eq!(client_secret["value"], "ek_translate_test");
    assert_eq!(client_secret["expires_at"], 700);
    assert_eq!(
        client_secret["session"]["audio"]["output"]["language"],
        "es"
    );
}

#[test]
fn serializes_realtime_session_records() {
    let record = RealtimeSessionRecord {
        id: "sess-rt".to_string(),
        object: "realtime.session",
        kind: "realtime",
        created_at: 10,
        expires_at: 70,
        client_secret: RealtimeClientSecretRecord {
            value: "ek_nerva_test".to_string(),
            expires_at: 70,
        },
        config: json!({
            "type": "realtime",
            "model": "gpt-realtime"
        }),
    };
    let value = realtime_session_json(&record);
    assert_eq!(value["id"], "sess-rt");
    assert_eq!(value["object"], "realtime.session");
    assert_eq!(value["client_secret"]["value"], "ek_nerva_test");
    assert_eq!(value["nerva"]["kind"], "realtime");
}

#[test]
fn builds_realtime_call_control_payloads() {
    assert_eq!(request_realtime_reject_status(&json!({})).unwrap(), 603);
    assert_eq!(
        request_realtime_reject_status(&json!({"status_code": 486})).unwrap(),
        486
    );
    assert!(request_realtime_reject_status(&json!({"status_code": 200})).is_err());
    assert!(request_realtime_reject_status(&json!({"status_code": "busy"})).is_err());

    assert_eq!(
        request_realtime_refer_target(&json!({"target_uri": "tel:+14155550123"})).unwrap(),
        "tel:+14155550123"
    );
    assert!(request_realtime_refer_target(&json!({})).is_err());
    assert!(request_realtime_refer_target(&json!({"target_uri": ""})).is_err());

    let record = RealtimeCallRecord {
        id: "call_1".to_string(),
        object: "realtime.call",
        created_at: 10,
        updated_at: 12,
        status: "referred".to_string(),
        session_id: Some("sess_1".to_string()),
        status_code: Some(486),
        target_uri: Some("sip:agent@example.com".to_string()),
        config: json!({
            "id": "sess_1",
            "object": "realtime.session",
            "model": "gpt-realtime"
        }),
    };
    let value = realtime_call_json(&record);
    assert_eq!(value["id"], "call_1");
    assert_eq!(value["object"], "realtime.call");
    assert_eq!(value["status"], "referred");
    assert_eq!(value["session_id"], "sess_1");
    assert_eq!(value["status_code"], 486);
    assert_eq!(value["target_uri"], "sip:agent@example.com");
    assert_eq!(value["session"]["object"], "realtime.session");
}

#[test]
fn gates_shared_fork_batch_to_greedy_sampler() {
    assert!(shared_fork_batch_supported(0.0, 1.0, 0, None));
    assert!(!shared_fork_batch_supported(1.0, 1.0, 0, None));
    assert!(!shared_fork_batch_supported(0.0, 0.95, 0, None));
    assert!(!shared_fork_batch_supported(0.0, 1.0, 40, None));
    assert!(!shared_fork_batch_supported(0.0, 1.0, 0, Some(7)));
}

#[test]
fn parses_multipart_file_upload() {
    let body = concat!(
        "--nerva\r\n",
        "Content-Disposition: form-data; name=\"purpose\"\r\n\r\n",
        "batch\r\n",
        "--nerva\r\n",
        "Content-Disposition: form-data; name=\"file\"; filename=\"requests.jsonl\"\r\n",
        "Content-Type: application/jsonl\r\n\r\n",
        "{\"custom_id\":\"a\"}\n",
        "\r\n--nerva--\r\n"
    );
    let upload = parse_multipart_file_upload(body.as_bytes(), "nerva").unwrap();

    assert_eq!(upload.filename, "requests.jsonl");
    assert_eq!(upload.purpose, "batch");
    assert_eq!(
        String::from_utf8(upload.content).unwrap(),
        "{\"custom_id\":\"a\"}\n"
    );
}

#[test]
fn normalizes_batch_endpoints() {
    assert_eq!(
        normalize_batch_endpoint("/v1/chat/completions").unwrap(),
        "/v1/chat/completions"
    );
    assert_eq!(
        normalize_batch_endpoint("http://localhost:8080/v1/responses").unwrap(),
        "/v1/responses"
    );
    assert_eq!(
        normalize_batch_endpoint("/v1/embeddings").unwrap(),
        "/v1/embeddings"
    );
    assert_eq!(
        normalize_batch_endpoint("/v1/moderations").unwrap(),
        "/v1/moderations"
    );
    assert_eq!(
        normalize_batch_endpoint("/v1/audio/speech").unwrap(),
        "/v1/audio/speech"
    );
    assert_eq!(
        normalize_batch_endpoint("/v1/images/generations").unwrap(),
        "/v1/images/generations"
    );
    assert_eq!(
        normalize_batch_endpoint("/v1/images/edits").unwrap(),
        "/v1/images/edits"
    );
    assert_eq!(
        normalize_batch_endpoint("/v1/videos/extensions").unwrap(),
        "/v1/videos/extensions"
    );
}

#[test]
fn parses_mcp_http_targets_and_payloads() {
    let endpoint = parse_http_endpoint("http://127.0.0.1:9000/mcp").unwrap();
    assert_eq!(endpoint.host, "127.0.0.1");
    assert_eq!(endpoint.port, 9000);
    assert_eq!(endpoint.path, "/mcp");
    let endpoint = parse_http_endpoint("https://mcp.example.test/mcp").unwrap();
    assert_eq!(endpoint.host, "mcp.example.test");
    assert_eq!(endpoint.port, 443);
    assert_eq!(endpoint.path, "/mcp");

    let response = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}";
    let parsed = parse_mcp_http_response(response).unwrap();
    assert_eq!(parsed["tools"], json!([]));
    let response = b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\r\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\nevent: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n";
    let parsed = parse_mcp_http_response(response).unwrap();
    assert_eq!(parsed["ok"], true);

    assert_eq!(
        first_sse_json_payload("event: message\ndata: {\"result\":true}\n\n").unwrap(),
        "{\"result\":true}"
    );
}

#[test]
fn decodes_chunked_mcp_body_and_query_escapes() {
    let decoded = decode_chunked_body(b"5\r\nhello\r\n0\r\n\r\n").unwrap();
    assert_eq!(decoded, b"hello");
    assert_eq!(percent_decode_query("hello+%77orld"), "hello world");
}
