# Plan: name and orchestrate the evaluation pipeline

## Goal

Give each existing evaluation stage a precise name and a standalone command,
then add one command that runs the same stages end to end. Keep the recovery
behavior already implemented in `evals/eval.py`; this work is primarily about
clear stage interfaces and orchestration, with a small number of remaining
validation fixes.

## Current baseline to preserve

The current script already has the important reliability mechanics. Do not
replace or weaken them during the refactor:

- Manifest, translation, and judgment CSVs are written atomically with flush,
  `fsync`, and rename.
- TTS stops on the first failed file, cleans temporary files, and leaves prior
  valid WAVs in place so a retry skips them.
- Translation and judging checkpoint every completed unit.
- Translation and judging stop scheduling new work after the first remote
  failure while allowing in-flight work to finish and be checkpointed.
- Resume restores only mutable result/error columns after validating columns,
  row order, `pair_id`, and immutable source values.
- Completed translation cells and judgment rows are skipped on resume; failed
  or empty work is retried.
- Printed resume commands retain `--language` and `--limit`.
- Judging rejects incomplete inference before making judge calls.
- `--output` cannot overwrite the judge input itself.
- `--self-check` covers WAV validation, deterministic audio mapping, judgment
  parsing, atomic CSV I/O, and checkpoint restoration.

## Stage names and commands

Replace the generic mode flags with verb-based subcommands:

| Stage | Command | Reads | Produces |
|---|---|---|---|
| 1. Build evaluation manifest | `build-manifest` | golden-pairs CSV | `evals/eval_input.csv` |
| 2. Synthesize evaluation audio | `synthesize-audio` | manifest text and audio paths | validated WAVs in `audio_files/` |
| 3. Run translation models | `run-translations` | manifest and WAVs | `evals/results_<run-id>.csv` |
| 4. Judge translation results | `judge-results` | translation results CSV | `evals/judged_results_<run-id>.csv` |
| Orchestrate all four | `run-pipeline` | golden-pairs CSV | all artifacts above |

`self-check` remains a diagnostic command, not a pipeline stage.

Target CLI:

```bash
uv run evals/eval.py build-manifest
uv run evals/eval.py synthesize-audio --language ms --limit 2
uv run evals/eval.py run-translations --language ms --limit 2
uv run evals/eval.py judge-results evals/results_20260925_120000.csv

uv run evals/eval.py run-pipeline --language ms --limit 2
uv run evals/eval.py self-check
```

Use `argparse` subparsers so each command shows only relevant options. TTS
options belong only to `synthesize-audio` and `run-pipeline`; websocket pacing
options belong only to `run-translations` and `run-pipeline`; judge input and
output options belong only to `judge-results`.

## Stage interface

Rename the existing stage functions and make their artifact handoffs explicit:

```python
build_manifest(...) -> Path
synthesize_audio(manifest_path, ...) -> StageSummary
run_translations(manifest_path, output_path, ...) -> Path
judge_results(results_path, output_path, ...) -> Path
run_pipeline(...) -> PipelineArtifacts
```

The important seam is the returned artifact path. `run-pipeline` must pass the
exact path returned by one stage into the next; it must never glob for or guess
the newest results file.

Keep this lightweight. One small `StageSummary` record for generated/skipped
counts and one `PipelineArtifacts` record for final paths are sufficient. Do
not introduce provider classes or a general workflow framework; Qwen, Gemini,
OpenAI TTS, and Router logic can remain where they are.

Stage functions should not call `sys.exit()`, because that makes them awkward
to compose and test. Raise a small `EvalError` for expected input/configuration
failures and `StageFailed` when remote work fails after a checkpoint is saved.
Only the CLI adapter converts those errors into messages and exit codes.

## Full-pipeline behavior

`run-pipeline` is thin orchestration over the same four public stage functions:

```text
golden CSV
    -> build-manifest
    -> synthesize-audio
    -> run-translations
    -> judge-results
```

At startup, allocate one `run-id` using the existing timestamp format. Use it
for both result artifacts:

```text
evals/results_<run-id>.csv
evals/judged_results_<run-id>.csv
```

Print the run id and planned artifact paths before any network call. Apply
`--language` and `--limit` consistently to synthesis, translation, and judging.
The manifest can continue to contain the full golden dataset; the selected rows
in later stages must remain identical.

If any stage fails, do not start the next stage. The raised error identifies
the failed stage and, when applicable, the checkpoint and exact resume command.

Support deterministic pipeline resume:

```bash
uv run evals/eval.py run-pipeline --run-id 20260925_120000 --resume \
  --language ms --limit 2
```

On resume:

- rebuild the deterministic manifest;
- let audio synthesis skip valid existing WAVs;
- resume the named translation checkpoint if it exists, otherwise start it;
- run or resume the derived judged-results checkpoint only after translation
  is complete.

Never infer a run id from “the latest” file.

## Remaining error-handling work

### Preflight validation

Finish validation before making paid or remote calls:

- All commands: reject `--limit <= 0`.
- Audio synthesis: validate `OPENAI_API_KEY`, TTS speed, `ffmpeg`, selected rows,
  and output directory before creating the client.
- Translation: reject `--workers <= 0`, negative `--sleep`, negative `--pace`,
  and non-positive provider timeouts; validate manifest columns and unique
  `pair_id` values; validate every selected WAV before starting either provider;
  validate Qwen and Gemini credentials/configuration up front.
- Judging: keep the current schema, duplicate-id, completeness, and input/output
  checks; validate Router configuration before creating the output checkpoint.
- Manifest: add an explicit duplicate-`pair_id` check and reject empty required
  English/translation text rather than emitting blank audio paths.

This prevents a malformed late row or missing credential from consuming earlier
paid calls before the run fails.

### Output safety

- A fresh `run-translations --output PATH` must reject an existing path unless
  `--resume` or an explicit `--overwrite` is supplied.
- `judge-results` must likewise avoid silently replacing an existing judgment
  checkpoint unless resuming or explicitly overwriting it.
- Preserve the current atomic temporary-file cleanup behavior.

### Exit behavior

- Exit `0` when requested work completes or the checkpoint is already complete.
- Exit `2` for invalid CLI usage, configuration, input, or checkpoint mismatch.
- Exit `1` for a remote stage failure after durable work may have been saved.
- On `KeyboardInterrupt`, stop scheduling work, allow safe checkpointing of any
  completed result already in hand, print the checkpoint path, and exit `130`.
- Expected errors get concise `ERROR [stage]: ...` messages without tracebacks.
  Unexpected programming errors retain their tracebacks.

Keep per-cell and per-row error columns as the durable error record. Continue
truncating unbounded judge errors, and do not print credentials, authorization
headers, or raw provider events containing transcript/audio data.

## Implementation order

1. Add `EvalError`/`StageFailed` and convert current stage-level `sys.exit()`
   calls to raised errors. Preserve the existing worker-level error capture.
2. Rename `prepare`, `synthesize`, `run`, and `judge` to `build_manifest`,
   `synthesize_audio`, `run_translations`, and `judge_results`. Return artifact
   paths/summaries without changing provider request logic.
3. Add the remaining preflight and output-collision checks listed above.
4. Replace the mutually exclusive flags with stage-specific subparsers and
   update resume-command formatting to use the new command names.
5. Add `run_pipeline` with one run id and explicit path passing. Reuse the stage
   functions; do not duplicate their implementations.
6. Update the module docstring and `README.md` to match the final commands,
   artifact names, credentials, and resume behavior.

## Verification

Keep the current self-check assertions and extend them with focused, offline
tests using temporary files and fake provider callables:

1. CLI dispatch:
   - every subcommand invokes the correct stage;
   - options appear only on relevant commands;
   - invalid numeric values fail before a stage starts.
2. Manifest:
   - shared English and unique target paths remain deterministic;
   - duplicate ids, unsafe labels, inconsistent English, empty required text,
     and missing columns fail cleanly.
3. Synthesis:
   - valid WAVs are skipped and invalid WAVs are regenerated;
   - temporary files are removed after failure;
   - a mocked TTS failure preserves earlier files and stops later calls.
4. Translation:
   - four cells are scheduled per selected row;
   - malformed audio or missing credentials cause zero provider calls;
   - a mocked failure stops new scheduling and leaves a resumable checkpoint;
   - resume retries only failed or empty cells and retains filters.
5. Judging:
   - blind ordering and winner mapping remain stable;
   - incomplete inference causes zero Router calls;
   - malformed judge JSON is checkpointed as an error;
   - resume retries only failed or empty rows and retains filters.
6. Pipeline:
   - fake stages run once in the required order;
   - each stage receives the exact previous artifact path;
   - one run id names both result files;
   - a stage failure prevents every downstream stage;
   - `--resume --run-id ...` uses the correct existing checkpoints.

Local acceptance commands after implementation:

```bash
uv run evals/eval.py self-check
uv run evals/eval.py build-manifest
uv run evals/eval.py run-pipeline --language ms --limit 1
```

The final live smoke test succeeds only when it produces one judged row, prints
all artifact paths, exits zero, and leaves no `.tmp` files behind.

## Acceptance criteria

- Each of the four accurately named stages runs independently.
- `run-pipeline` runs the same four functions in order with one command.
- Existing atomic checkpoint, fail-fast, and resume behavior is preserved.
- Pipeline artifact handoff is explicit; no “latest file” lookup exists.
- Invalid input/configuration causes no remote calls.
- Partial remote failure returns non-zero with a valid checkpoint and exact
  resume command.
- Filters remain identical across a full run and its resume.
- CLI help, module documentation, and README examples agree with behavior.
