# Qwen context biasing for issue #4

Research date: 2026-09-25

## Bottom line

There are two different, model-specific mechanisms that should not be conflated:

1. **Direct `qwen3-asr-flash-realtime` ASR** supports free-form contextual
   biasing through the raw WebSocket field
   `session.input_audio_transcription.corpus.text`. In the Python DashScope SDK,
   the corresponding argument is `TranscriptionParams(corpus_text=...)`.
2. **`qwen3.8-livetranslate-flash-realtime` LiveTranslate** officially
   documents terminology biasing through
   `session.translation.corpus.phrases`, a mapping of source-language terms to
   their desired target-language translations. Its LiveTranslate schema does
   not document `input_audio_transcription.corpus.text`.

Therefore, adding `input_audio_transcription.corpus.text` to this repository's
current Qwen3.8 LiveTranslate request is an **experiment with an undocumented
pass-through**, not a documented LiveTranslate feature. If the objective is to
test the supported `corpus_text` feature itself, call
`qwen3-asr-flash-realtime` directly and compare its source transcripts. If the
objective is to improve Qwen3.8 translated output, the officially supported
mechanism is `translation.corpus.phrases`, which tests a different hypothesis
and requires source-to-target term mappings.

In the 2026-09-25 experiment against the Singapore endpoint, the
`qwen3.8-livetranslate-flash-realtime` server accepted this undocumented
pass-through and its `session.updated` event echoed the full nested
`input_audio_transcription.corpus.text` value. That proves the field was not
rejected or stripped from the session configuration, although it does not by
itself prove that the ASR decoder used the context.

## Exact supported ASR interface

Alibaba Cloud's Qwen-ASR-Realtime client-events reference defines this optional
field on a `session.update` request:

```json
{
  "type": "session.update",
  "event_id": "event_unique",
  "session": {
    "input_audio_format": "pcm",
    "sample_rate": 16000,
    "input_audio_transcription": {
      "language": "ms",
      "corpus": {
        "text": "source-language medical terms and background text"
      }
    }
  }
}
```

The documented path is
`input_audio_transcription.corpus.text`. The value may contain background text,
entity vocabulary, or other reference material and has a maximum size of
10,000 tokens. The field is configured in `session.update`, before audio is
streamed, so a caller can create/update it independently for each evaluation
request. [Qwen-ASR-Realtime client events](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-client-events)

The Python SDK exposes the same feature as follows:

```python
from dashscope.audio.qwen_omni import MultiModality, TranscriptionParams

transcription_params = TranscriptionParams(
    language="ms",
    sample_rate=16000,
    input_audio_format="pcm",
    corpus_text="source-language medical terms and background text",
)

conversation.update_session(
    output_modalities=[MultiModality.TEXT],
    enable_input_audio_transcription=True,
    transcription_params=transcription_params,
)
```

The official Python reference requires DashScope SDK 1.25.6 or newer and lists
`corpus_text` as optional, with the same 10,000-token maximum. It also confirms
that Malay (`ms`) is a supported source language. For this model, documented
audio formats are PCM and Opus and documented sample rates are 16 kHz and 8
kHz. [Qwen-ASR-Realtime Python SDK reference](https://help.aliyun.com/en/model-studio/qwen-asr-realtime-python-sdk)

The official SDK source shows the wire conversion directly: its
`TranscriptionParams` dataclass accepts `corpus_text`, and
`_apply_transcription_params()` converts it to `{"text": corpus_text}` under
`input_audio_transcription["corpus"]`.
[SDK `TranscriptionParams` source](https://github.com/dashscope/dashscope-sdk-python/blob/2cd356a499e7d70dc28035b34fd9ee1ad2d12572/dashscope/audio/qwen_omni/omni_realtime.py#L51-L60),
[SDK request-building source](https://github.com/dashscope/dashscope-sdk-python/blob/2cd356a499e7d70dc28035b34fd9ee1ad2d12572/dashscope/audio/qwen_omni/omni_realtime.py#L363-L386)

The currently documented stable `qwen3-asr-flash-realtime` alias maps to the
`2025-10-27` snapshot; `2026-02-10` is also listed as a newer explicit snapshot
in both Beijing and Singapore. Pinning the exact snapshot is preferable for a
repeatable before/after experiment.
[Realtime ASR model/region list](https://help.aliyun.com/en/model-studio/real-time-speech-recognition-user-guide)

## Exact supported LiveTranslate interface

For `qwen3.8-livetranslate-flash-realtime`, Alibaba Cloud documents a different
field:

```json
{
  "type": "session.update",
  "session": {
    "output_modalities": ["text"],
    "translation": {
      "language": "en",
      "corpus": {
        "phrases": {
          "source medical term": "desired English medical term"
        }
      }
    }
  }
}
```

`translation.corpus.phrases` is a source-term to target-translation map and is
documented as improving translation accuracy for specific terms. The
LiveTranslate docs explicitly say it applies to
`qwen3.8-livetranslate-flash-realtime`. They also say Qwen3.8 ASR is always
enabled, but the LiveTranslate `input_audio_transcription` schema documents only
the ASR model and source language—not an ASR corpus field.
[LiveTranslate client events](https://help.aliyun.com/en/model-studio/live-translator-client-events)

This matters for issue #4: a phrase map may legitimately improve the final
translation even if source-word recognition did not improve. Conversely, a
source-only `corpus_text` test should be scored against the returned source
transcript, not inferred solely from the translation.

## Related but incompatible mechanism: weighted `vocabulary`

The newer Qwen-Audio-3.x ASR families support inline weighted hotwords through
a request-level `vocabulary` object such as `{"John": 5, "Speech Lab": 50}`.
The documented weights are 1–5 or 50, with 3–4 recommended as a starting point;
up to 2,000 hotwords can be passed per request, including at most 50 weight-50
"super hotwords." This feature is documented only for the
Qwen-Audio-3.x-ASR-Flash Streaming/Filetrans/Flash families, not
`qwen3-asr-flash-realtime` or Qwen LiveTranslate. It should not be substituted
for `corpus_text` in this experiment.
[Improve recognition accuracy](https://help.aliyun.com/en/model-studio/improve-asr-accuracy)

## Practical evaluation guidance

- Keep the audio, source-language code, model snapshot, streaming pace, VAD,
  and all other settings identical between baseline and biased conditions.
- Supply only source-language domain terms/background text to `corpus_text`.
  Do not include the target-language reference; that would leak the expected
  output.
- Record both the source transcription events and translated text. Score term
  recognition on the source transcript, and translation quality separately.
- Verify the server's `session.updated` response. If a Qwen3.8 LiveTranslate
  session does not echo the nested ASR corpus or returns an invalid-parameter
  error, treat `corpus_text` as unsupported on that endpoint rather than as a
  failed efficacy result.
- Watch for corpus-driven false insertions. The corpus is a bias, not a
  constraint, and unrelated terms appearing in the output count as regressions.
- The issue itself asks for per-row source-language context, identical paired
  runs, term-level recall, and explicit regression/hallucination reporting.
  [Issue #4](https://github.com/mcxraider/live-translate-diarize-poc/issues/4)

## Source reconciliation

The general "Improve recognition accuracy" page's current support tables focus
on the Qwen-Audio-3.x families and do not list Qwen3-ASR for generic context
enhancement. The model-specific Qwen-ASR-Realtime API and SDK references are
more precise for `qwen3-asr-flash-realtime` and explicitly document
`corpus.text` / `corpus_text`. The LiveTranslate reference remains the
authoritative schema for `qwen3.8-livetranslate-flash-realtime`, where only
`translation.corpus.phrases` is documented.
