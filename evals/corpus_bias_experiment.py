# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets", "google-genai", "openai", "tiktoken"]
# ///
"""Focused Qwen corpus-bias experiment for golden-set rows 1, 3, and 4.

This deliberately lives outside eval.py. It reuses eval.py's audio validation,
TTS, and Gemini helpers, while sending Qwen requests itself so the only Qwen
difference between paired runs is input_audio_transcription.corpus.text.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import websockets

import eval as core


SELECTED_ZERO_BASED_ROWS = (0, 2, 3)
INPUT_CSV = core.EVALS / "corpus_experiment_input_rows_1_3_4.csv"

# Corpus is curated up front rather than extracted from references at request
# time. Each value is in the language of the audio sent to Qwen.
CORPUS_BY_PAIR_AND_LANGUAGE = {
    ("en-ms-001", "en"): (
        "runny nose; blocked nose; sneezing; scratchy throat; fever; body aches"
    ),
    ("en-ms-001", "ms"): (
        "hidung berair; hidung tersumbat; bersin; tekak gatal; demam; "
        "sakit-sakit badan"
    ),
    ("en-ms-003", "en"): (
        "temperature; thirty-nine point two degrees; Panadol; fever; chills; "
        "night sweats"
    ),
    ("en-ms-003", "ms"): (
        "suhu badan; tiga puluh sembilan perpuluhan dua darjah; Panadol; demam; "
        "menggigil; berpeluh pada waktu malam"
    ),
    ("en-ms-004", "en"): (
        "high fever; pain behind the eyes; dengue; blood test; platelet count; "
        "ibuprofen; aspirin; paracetamol"
    ),
    ("en-ms-004", "ms"): (
        "demam tinggi; sakit di belakang mata; denggi; ujian darah; bilangan "
        "platlet; ibuprofen; aspirin; paracetamol"
    ),
}

# A term is counted when any listed surface form is present after normalization.
TERM_GROUPS = {
    ("en-ms-001", "en"): [
        ["runny nose"], ["blocked nose"], ["sneezing"], ["scratchy"],
        ["fever"], ["aching", "body aches"],
    ],
    ("en-ms-001", "ms"): [
        ["hidung berair"], ["tersumbat"], ["bersin"], ["gatal"],
        ["demam"], ["sakit sakit", "lenguh lenguh"],
    ],
    ("en-ms-003", "en"): [
        ["temperature"], ["thirty nine point two", "39 point 2", "39 2"],
        ["panadol"], ["fever"], ["chills", "shivering"],
        ["sweating", "sweat"],
    ],
    ("en-ms-003", "ms"): [
        ["suhu badan", "suhu"],
        ["tiga puluh sembilan perpuluhan dua", "39 perpuluhan 2", "39 2"],
        ["panadol"], ["demam"], ["menggigil"], ["berpeluh"],
    ],
    ("en-ms-004", "en"): [
        ["high fever"], ["pain behind your eyes", "pain behind the eyes"],
        ["dengue"], ["blood test"], ["platelet count", "platelets"],
        ["ibuprofen"], ["aspirin"], ["paracetamol", "acetaminophen"],
    ],
    ("en-ms-004", "ms"): [
        ["demam tinggi"], ["sakit di belakang mata", "sakit belakang mata"],
        ["denggi", "dengue"], ["ujian darah"],
        ["bilangan platlet", "kiraan platlet", "platlet"], ["ibuprofen"],
        ["aspirin"], ["paracetamol", "parasetamol"],
    ],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=\d)[.,](?=\d)", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def term_hits(text: str, groups: list[list[str]]) -> tuple[int, list[str]]:
    haystack = f" {normalize(text)} "
    hits = []
    for alternatives in groups:
        matched = next(
            (
                term
                for term in alternatives
                if f" {normalize(term)} " in haystack
            ),
            None,
        )
        if matched:
            hits.append(alternatives[0])
    return len(hits), hits


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_rows() -> list[dict]:
    _, all_rows = core._read_manifest(core.EVAL_INPUT, stage="corpus-experiment")
    return [dict(all_rows[index]) for index in SELECTED_ZERO_BASED_ROWS]


def prepare_audio(rows: list[dict]) -> None:
    # A temporary selected-row manifest lets the existing synthesizer reuse
    # valid WAVs and create only the four missing row-3/row-4 files.
    fields = list(rows[0])
    manifest = core.EVALS / ".corpus_experiment_audio_manifest.csv"
    core._write_csv_atomic(manifest, fields, rows)
    try:
        core.synthesize_audio(manifest, core.EvalConfig())
    finally:
        manifest.unlink(missing_ok=True)


def build_jobs(rows: list[dict]) -> list[dict]:
    jobs = []
    for row_number, row in zip((1, 3, 4), rows, strict=True):
        directions = (
            (
                "en_to_ms", "en", "ms", row["english"], row["translation"],
                row["english_audio_path"],
            ),
            (
                "ms_to_en", "ms", "en", row["translation"], row["english"],
                row["target_audio_path"],
            ),
        )
        for direction, source, target, source_ref, target_ref, audio_rel in directions:
            jobs.append(
                {
                    "row_number": row_number,
                    "pair_id": row["pair_id"],
                    "direction": direction,
                    "source_language": source,
                    "target_language": target,
                    "source_reference": source_ref,
                    "target_reference": target_ref,
                    "audio_path": audio_rel,
                    "corpus_text": CORPUS_BY_PAIR_AND_LANGUAGE[(row["pair_id"], source)],
                }
            )
    return jobs


def write_input_csv(jobs: list[dict]) -> None:
    core._write_csv_atomic(INPUT_CSV, list(jobs[0]), jobs)


async def qwen_translate(
    pcm: bytes,
    source: str,
    target: str,
    *,
    corpus_text: str | None,
    pace: float,
    timeout: float,
) -> tuple[str, str]:
    key = core.os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")

    transcription = {
        "model": core.DEFAULT_QWEN_ASR_MODEL,
        "language": source,
    }
    if corpus_text:
        transcription["corpus"] = {"text": corpus_text}

    session_update = {
        "event_id": f"event_{time.time_ns()}",
        "type": "session.update",
        "session": {
            "output_modalities": ["text"],
            "voice": core.os.environ.get(
                "DASHSCOPE_VOICE", core.DEFAULT_DASHSCOPE_VOICE
            ),
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "input_audio_transcription": transcription,
            "translation": {"language": target},
            "turn_detection": {"type": "speaker_detection"},
        },
    }

    async with websockets.connect(
        core.dashscope_url(),
        additional_headers={"Authorization": f"Bearer {key}"},
    ) as websocket:
        await websocket.send(json.dumps(session_update, ensure_ascii=False))
        for chunk in core._chunks(pcm):
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            )
            await asyncio.sleep(core.CHUNK_MS / 1000 * pace)

        silence_frames = round(
            core.SAMPLE_RATE * core.QWEN_TRAILING_SILENCE_SECONDS
        )
        silence = b"\x00" * silence_frames * core.SAMPLE_WIDTH_BYTES
        for chunk in core._chunks(silence):
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            )
            await asyncio.sleep(core.CHUNK_MS / 1000 * pace)
        await websocket.send(
            json.dumps({"type": "session.finish", "event_id": f"event_{time.time_ns()}"})
        )

        translation_parts: list[str] = []
        transcript_parts: list[str] = []
        completed_transcript = ""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        done_deadline: float | None = None
        while loop.time() < deadline:
            receive_deadline = min(deadline, done_deadline or deadline)
            try:
                raw = await asyncio.wait_for(
                    websocket.recv(), timeout=max(0.01, receive_deadline - loop.time())
                )
            except asyncio.TimeoutError:
                break
            event = json.loads(raw)
            event_type = event.get("type", "")
            if event_type in ("response.text.delta", "response.audio_transcript.delta"):
                translation_parts.append(event.get("delta", ""))
            elif event_type == "conversation.item.input_audio_transcription.delta":
                transcript_parts.append(event.get("delta", ""))
            elif event_type in (
                "conversation.item.input_audio_transcription.completed",
                "conversation.item.input_audio_transcription.text",
            ):
                completed_transcript = (
                    event.get("transcript") or event.get("text") or completed_transcript
                )
            elif event_type == "response.done":
                # Source transcription can finish just after the translation.
                done_deadline = min(deadline, loop.time() + 3.0)
            elif event_type == "session.finished":
                break
            elif event_type == "error":
                raise RuntimeError(f"Qwen error event: {event}")

    transcript = completed_transcript or "".join(transcript_parts)
    return transcript.strip(), "".join(translation_parts).strip()


def qwen_call(job: dict, corpus: bool, args: argparse.Namespace) -> tuple[str, str, str]:
    path = (core.REPO / job["audio_path"]).resolve()
    pcm = core.load_pcm16_mono_16k(path)
    try:
        transcript, translation = asyncio.run(
            qwen_translate(
                pcm,
                job["source_language"],
                job["target_language"],
                corpus_text=job["corpus_text"] if corpus else None,
                pace=args.pace,
                timeout=args.qwen_timeout,
            )
        )
        if not translation:
            return transcript, "", "empty translation"
        return transcript, translation, ""
    except Exception as exc:  # Preserve the full paired experiment artifact.
        return "", "", core._error_text(exc)


def gemini_call(job: dict, args: argparse.Namespace) -> tuple[str, str]:
    path = (core.REPO / job["audio_path"]).resolve()
    pcm = core.load_pcm16_mono_16k(path)
    try:
        translation, _ = asyncio.run(
            core.translate_gemini(
                pcm,
                job["target_language"],
                pace=args.pace,
                drain=args.gemini_drain,
                trailing_silence=args.gemini_trailing_silence,
                want_audio=False,
            )
        )
        return translation, "" if translation else "empty translation"
    except Exception as exc:
        return "", core._error_text(exc)


def run_round(label: str, jobs: list[dict], worker, workers: int) -> list[tuple]:
    print(f"Running {label} ({len(jobs)} calls)...", flush=True)
    results: list[tuple | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index = {pool.submit(worker, job): i for i, job in enumerate(jobs)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
            job = jobs[index]
            print(
                f"  {label}: row {job['row_number']} {job['direction']} done",
                flush=True,
            )
    return list(results)


def score_rows(
    jobs: list[dict],
    baseline: list[tuple[str, str, str]],
    biased: list[tuple[str, str, str]],
    gemini: list[tuple[str, str]],
) -> list[dict]:
    results = []
    for job, base, bias, gem in zip(jobs, baseline, biased, gemini, strict=True):
        base_transcript, base_translation, base_error = base
        bias_transcript, bias_translation, bias_error = bias
        gem_translation, gem_error = gem
        source_groups = TERM_GROUPS[(job["pair_id"], job["source_language"])]
        target_groups = TERM_GROUPS[(job["pair_id"], job["target_language"])]
        base_source_count, base_source_hits = term_hits(base_transcript, source_groups)
        bias_source_count, bias_source_hits = term_hits(bias_transcript, source_groups)
        base_target_count, base_target_hits = term_hits(base_translation, target_groups)
        bias_target_count, bias_target_hits = term_hits(bias_translation, target_groups)
        gem_target_count, gem_target_hits = term_hits(gem_translation, target_groups)
        audio_path = (core.REPO / job["audio_path"]).resolve()
        total_source = len(source_groups)
        total_target = len(target_groups)
        results.append(
            {
                **job,
                "audio_sha256": sha256(audio_path),
                "source_terms": " | ".join(group[0] for group in source_groups),
                "target_terms": " | ".join(group[0] for group in target_groups),
                "qwen_baseline_source_transcript": base_transcript,
                "qwen_corpus_source_transcript": bias_transcript,
                "qwen_baseline_translation": base_translation,
                "qwen_corpus_translation": bias_translation,
                "gemini_translation": gem_translation,
                "qwen_baseline_source_term_hits": " | ".join(base_source_hits),
                "qwen_corpus_source_term_hits": " | ".join(bias_source_hits),
                "qwen_baseline_source_term_recall": round(base_source_count / total_source, 4),
                "qwen_corpus_source_term_recall": round(bias_source_count / total_source, 4),
                "qwen_baseline_source_wer": round(
                    word_error_rate(job["source_reference"], base_transcript), 4
                ),
                "qwen_corpus_source_wer": round(
                    word_error_rate(job["source_reference"], bias_transcript), 4
                ),
                "qwen_baseline_translation_term_hits": " | ".join(base_target_hits),
                "qwen_corpus_translation_term_hits": " | ".join(bias_target_hits),
                "gemini_translation_term_hits": " | ".join(gem_target_hits),
                "qwen_baseline_translation_term_recall": round(
                    base_target_count / total_target, 4
                ),
                "qwen_corpus_translation_term_recall": round(
                    bias_target_count / total_target, 4
                ),
                "gemini_translation_term_recall": round(
                    gem_target_count / total_target, 4
                ),
                "qwen_baseline_reference_wer": round(
                    word_error_rate(job["target_reference"], base_translation), 4
                ),
                "qwen_corpus_reference_wer": round(
                    word_error_rate(job["target_reference"], bias_translation), 4
                ),
                "gemini_reference_wer": round(
                    word_error_rate(job["target_reference"], gem_translation), 4
                ),
                "qwen_baseline_error": base_error,
                "qwen_corpus_error": bias_error,
                "gemini_error": gem_error,
            }
        )
    return results


def mean(rows: list[dict], field: str) -> float:
    return round(sum(float(row[field]) for row in rows) / len(rows), 4)


def aggregate(rows: list[dict]) -> dict:
    improved = unchanged = regressed = 0
    for row in rows:
        before = float(row["qwen_baseline_translation_term_recall"])
        after = float(row["qwen_corpus_translation_term_recall"])
        if after > before:
            improved += 1
        elif after < before:
            regressed += 1
        else:
            unchanged += 1
    return {
        "sample_directions": len(rows),
        "qwen_source_term_recall_before": mean(
            rows, "qwen_baseline_source_term_recall"
        ),
        "qwen_source_term_recall_after": mean(rows, "qwen_corpus_source_term_recall"),
        "qwen_source_wer_before": mean(rows, "qwen_baseline_source_wer"),
        "qwen_source_wer_after": mean(rows, "qwen_corpus_source_wer"),
        "qwen_translation_term_recall_before": mean(
            rows, "qwen_baseline_translation_term_recall"
        ),
        "qwen_translation_term_recall_after": mean(
            rows, "qwen_corpus_translation_term_recall"
        ),
        "gemini_translation_term_recall": mean(rows, "gemini_translation_term_recall"),
        "qwen_reference_wer_before": mean(rows, "qwen_baseline_reference_wer"),
        "qwen_reference_wer_after": mean(rows, "qwen_corpus_reference_wer"),
        "gemini_reference_wer": mean(rows, "gemini_reference_wer"),
        "translation_term_recall_outcomes": {
            "improved": improved,
            "unchanged": unchanged,
            "regressed": regressed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--qwen-timeout", type=float, default=40.0)
    parser.add_argument("--gemini-drain", type=float, default=30.0)
    parser.add_argument("--gemini-trailing-silence", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be greater than zero")
    rows = selected_rows()
    prepare_audio(rows)
    jobs = build_jobs(rows)
    write_input_csv(jobs)

    baseline = run_round(
        "Qwen baseline",
        jobs,
        lambda job: qwen_call(job, False, args),
        args.workers,
    )
    biased = run_round(
        "Qwen corpus",
        jobs,
        lambda job: qwen_call(job, True, args),
        args.workers,
    )
    gemini = run_round(
        "Gemini",
        jobs,
        lambda job: gemini_call(job, args),
        args.workers,
    )
    results = score_rows(jobs, baseline, biased, gemini)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or core.EVALS / f"corpus_experiment_results_{run_id}.csv"
    if not output.is_absolute():
        output = (core.REPO / output).resolve()
    core._write_csv_atomic(output, list(results[0]), results)
    summary_path = output.with_suffix(".summary.json")
    summary = {
        "run_id": run_id,
        "qwen_model": core.QWEN_MODEL,
        "qwen_asr_model": core.DEFAULT_QWEN_ASR_MODEL,
        "gemini_model": core.GEMINI_MODEL,
        "input_csv": str(INPUT_CSV.relative_to(core.REPO)),
        "results_csv": str(output.relative_to(core.REPO)),
        "aggregate": aggregate(results),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
