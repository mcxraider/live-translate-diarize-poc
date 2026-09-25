# Qwen corpus-bias experiment: golden rows 1, 3, and 4

Run date: 2026-09-25

Models:

- Qwen LiveTranslate: `qwen3.8-livetranslate-flash-realtime`
- Qwen input ASR: `qwen3-asr-flash-realtime`
- Gemini comparator: `gemini-3.5-live-translate-preview`

The experiment used the same WAV for each paired Qwen run. The baseline omitted
`input_audio_transcription.corpus`; the biased condition added only
`input_audio_transcription.corpus.text`, using a pre-curated source-language
medical corpus for that row and direction. The existing `evals/eval.py` was not
edited.

## Result

Corpus text did not improve these samples.

| Metric across 6 directions | Qwen baseline | Qwen + corpus | Gemini |
|---|---:|---:|---:|
| Source medical-term recall | 95.1% | 95.1% | n/a |
| Source transcript WER (lower is better) | 4.2% | 4.2% | n/a |
| Translated medical-term recall | 86.8% | 86.8% | 86.8% |
| Literal reference WER (lower is better) | 32.4% | 32.4% | 30.2% |

Translation term-recall outcomes: **0 improved, 6 unchanged, 0 regressed**.
There were no corpus-driven medical-term substitutions or hallucinations.

Five of six Qwen translations were byte-for-byte identical before and after
corpus injection. Row 1 English-to-Malay had one non-clinical synonym change:
`kira-kira` became `sekitar` (both mean approximately). Its normalized source
transcript, term recall, source WER, translated-term recall, and literal
reference WER were all unchanged.

## Per-direction results

| Row | Direction | Qwen source recall before → after | Qwen translation recall before → after | Gemini translation recall | Qwen output effect |
|---:|---|---:|---:|---:|---|
| 1 | English → Malay | 100% → 100% | 83.3% → 83.3% | 83.3% | Synonym-only change (`kira-kira` → `sekitar`) |
| 1 | Malay → English | 83.3% → 83.3% | 50.0% → 50.0% | 50.0% | Identical |
| 3 | English → Malay | 100% → 100% | 100% → 100% | 100% | Identical |
| 3 | Malay → English | 100% → 100% | 100% → 100% | 100% | Identical |
| 4 | English → Malay | 100% → 100% | 87.5% → 87.5% | 87.5% | Identical |
| 4 | Malay → English | 87.5% → 87.5% | 100% → 100% | 100% | Identical |

The sub-100% exact term-recall values include conservative surface-form misses.
For example, row 1 Malay-to-English translated `hidung saya berair dan
tersumbat` correctly as “my nose is runny, and it's blocked,” but the scorer
requires the contiguous phrases “runny nose” and “blocked nose.” This means the
term metric understates semantic accuracy equally in both Qwen conditions and
for Gemini; it does not change the before/after conclusion.

The corpus also did not correct the one notable source spelling miss: row 4's
Malay `platlet` was transcribed as `platelet` in both conditions. Clinically the
meaning remained clear. The most visible model difference was general fluency,
not terminology: on row 4 Malay-to-English, Qwen began “By because you have a
high fever,” whereas Gemini produced a fluent “Because you have a high fever.”

## Interpretation

These three synthetic recordings already have a strong baseline and correctly
recognize almost every selected medical term, leaving little headroom for
context biasing. On this narrow set there is no evidence that adding
source-language `corpus.text` improves either Qwen ASR recognition or Qwen's
translated output. This is a negative result for these recordings, not proof
that context biasing never helps: a useful follow-up would use noisy, accented,
or deliberately ambiguous audio containing baseline term errors.

Alibaba documents `corpus.text` for direct Qwen Realtime ASR but not in the
LiveTranslate schema. The tested LiveTranslate endpoint nevertheless accepted
and echoed the exact corpus field in `session.updated`, confirming the field was
present in the active session. LiveTranslate's separately documented
translation terminology feature is `translation.corpus.phrases`; it was not
used here because it requires source-to-target mappings and tests translation
terminology rather than source-word recognition.

## Artifacts and rerun

- Input/corpus rows: `evals/corpus_experiment_input_rows_1_3_4.csv`
- Full outputs and metrics: `evals/corpus_experiment_results_20260925.csv`
- Aggregate machine-readable summary:
  `evals/corpus_experiment_results_20260925.summary.json`
- Documentation research: `docs/research/qwen-context-biasing.md`

Rerun with:

```bash
uv run evals/corpus_bias_experiment.py \
  --output evals/corpus_experiment_results_<run-id>.csv
```
