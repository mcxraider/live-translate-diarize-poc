# /// script
# requires-python = ">=3.11"
# dependencies = ["websockets", "google-genai"]
# ///
"""Offline back-translation eval for the live-translate models.

Feeds pre-recorded WAV audio (that you synthesize separately) through the
qwen and gemini *live* translate sessions and records the text they emit.
For each golden row we run 4 calls:
    qwen   English audio -> target text     (qwen_en2tgt)
    qwen   target  audio -> English text    (qwen_tgt2en)
    gemini English audio -> target text     (gemini_en2tgt)
    gemini target  audio -> English text    (gemini_tgt2en)

Two phases:
  uv run evals/eval.py --prepare
      Reads data/polyclinic_golden_pairs.csv (never mutated), writes
      evals/eval_input.csv with two extra columns holding deterministic
      audio paths, and creates the audio_files/ folder. You then synthesize
      16 kHz / mono / 16-bit PCM WAVs into those exact paths.

  uv run evals/eval.py --run [--limit N] [--language ms] ...
      Streams each audio file through both models and writes
      evals/results_<timestamp>.csv.

  uv run evals/eval.py --self-check      (no API key, no network)
      Exercises the WAV loader's accept/reject logic.

These are the audio-only realtime models — there is no text-in path — so the
CSV text must first be turned into audio; that is your job, this consumes it.
"""
import argparse
import asyncio
import base64
import csv
import os
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parent.parent
EVALS = REPO / "evals"
GOLDEN = REPO / "data" / "polyclinic_golden_pairs.csv"
EVAL_INPUT = EVALS / "eval_input.csv"
AUDIO_DIR = REPO / "audio_files"

SAMPLE_RATE = 16000          # both models want 16 kHz mono PCM16 in
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2   # 100ms of PCM16 = 3200 bytes

# Extra columns added by --prepare, then filled by --run.
AUDIO_COLS = ["english_audio_path", "target_audio_path"]
RESULT_COLS = ["qwen_en2tgt", "qwen_tgt2en", "gemini_en2tgt", "gemini_tgt2en"]
ERROR_COLS = [c + "_error" for c in RESULT_COLS]


# --- .env + provider config (copied from server.py; kept tiny so this script's
# only runtime deps are websockets + google-genai, not fastapi/uvicorn) --------
def _load_dotenv():
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()
QWEN_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.8-livetranslate-flash-realtime")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-live-translate-preview")


def dashscope_url() -> str:
    return os.environ.get(
        "DASHSCOPE_WS_URL",
        f"wss://maas.qwencloudapi.com/api-ws/v1/realtime?model={QWEN_MODEL}",
    )


def gemini_lang(code: str) -> str:
    # Our codes are valid BCP-47 except Mandarin, which needs a script tag.
    return {"zh": "zh-Hans"}.get(code, code)


_gemini_client = None


def gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            _gemini_client = genai.Client(api_key=api_key)
        else:
            _gemini_client = genai.Client(
                enterprise=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_REGION", "global"),
            )
    return _gemini_client


# --- WAV loading with strict format checks ------------------------------------
def load_pcm16_mono_16k(path: Path) -> bytes:
    """Read a WAV as raw PCM bytes, asserting it is exactly 16 kHz / mono /
    16-bit. Fails loudly rather than silently mistranslating wrong-rate audio."""
    with wave.open(str(path), "rb") as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if ch != 1:
            raise ValueError(f"expected mono, got {ch} channels")
        if width != 2:
            raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")
        if rate != SAMPLE_RATE:
            raise ValueError(f"expected {SAMPLE_RATE} Hz, got {rate} Hz")
        return w.readframes(w.getnframes())


def _chunks(pcm: bytes):
    for i in range(0, len(pcm), CHUNK_BYTES):
        yield pcm[i:i + CHUNK_BYTES]


# --- Qwen (DashScope raw websocket) -------------------------------------------
def _qwen_session_update(target: str, source: str | None, want_audio: bool) -> dict:
    transcription = {"model": "qwen3-asr-flash-realtime"}
    if source:
        transcription["language"] = source
    return {
        "event_id": f"event_{int(time.time() * 1000)}",
        "type": "session.update",
        "session": {
            # text-only by default; voice stays required even without audio out.
            "output_modalities": ["text", "audio"] if want_audio else ["text"],
            "voice": os.environ.get("DASHSCOPE_VOICE", "Tina"),
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "input_audio_transcription": transcription,
            "translation": {"language": target},
            "turn_detection": {"type": "speaker_detection"},
        },
    }


async def translate_qwen(pcm: bytes, source: str | None, target: str,
                         *, pace: float, timeout: float, want_audio: bool) -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    async with websockets.connect(
        dashscope_url(), additional_headers={"Authorization": f"Bearer {key}"}
    ) as ws:
        await ws.send(_json(_qwen_session_update(target, source, want_audio)))
        # Stream paced ~realtime so speaker_detection VAD segments the turn.
        for chunk in _chunks(pcm):
            await ws.send(_json({"type": "input_audio_buffer.append",
                                 "audio": base64.b64encode(chunk).decode()}))
            await asyncio.sleep(CHUNK_MS / 1000 * pace)
        # 0.5s trailing silence nudges the VAD to close the turn.
        for chunk in _chunks(b"\x00" * (SAMPLE_RATE // 2 * 2)):
            await ws.send(_json({"type": "input_audio_buffer.append",
                                 "audio": base64.b64encode(chunk).decode()}))
            await asyncio.sleep(CHUNK_MS / 1000 * pace)
        await ws.send(_json({"type": "session.finish",
                             "event_id": f"event_{int(time.time() * 1000)}"}))

        parts: list[str] = []
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            ev = _loads(raw)
            et = ev.get("type", "")
            if et in ("response.text.delta", "response.audio_transcript.delta"):
                parts.append(ev.get("delta", ""))
            elif et == "response.done":
                break
            elif et == "error":
                raise RuntimeError(f"qwen error event: {ev}")
        return "".join(parts).strip()


# --- Gemini (google-genai live session) ---------------------------------------
async def translate_gemini(pcm: bytes, target: str,
                           *, pace: float, drain: float, want_audio: bool) -> str:
    from google.genai import types
    modalities = ["AUDIO", "TEXT"] if want_audio else ["TEXT"]
    cfg = types.LiveConnectConfig(
        response_modalities=modalities,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=gemini_lang(target),
            echo_target_language=False,
        ),
    )
    async with gemini_client().aio.live.connect(model=GEMINI_MODEL, config=cfg) as session:
        for chunk in _chunks(pcm):
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
            await asyncio.sleep(CHUNK_MS / 1000 * pace)
        await session.send_realtime_input(audio_stream_end=True)

        # This model never sends turn_complete, so receive() never ends on its
        # own — read for `drain` seconds and take whatever text accumulated.
        out_trans: list[str] = []   # translated text in audio mode
        model_text: list[str] = []  # translated text in text-only mode

        async def reader():
            async for message in session.receive():
                sc = message.server_content
                if not sc:
                    continue
                if sc.output_transcription and sc.output_transcription.text:
                    out_trans.append(sc.output_transcription.text)
                if sc.model_turn:
                    for part in sc.model_turn.parts:
                        t = getattr(part, "text", None)
                        if t:
                            model_text.append(t)
                if getattr(sc, "turn_complete", False):
                    return

        try:
            await asyncio.wait_for(reader(), timeout=drain)
        except asyncio.TimeoutError:
            pass
        # Prefer text-part output (text-only mode); fall back to transcription.
        return ("".join(model_text) or "".join(out_trans)).strip()


# small json helpers (avoid importing json name-shadow confusion)
import json as _json_mod
def _json(obj) -> str: return _json_mod.dumps(obj)
def _loads(raw): return _json_mod.loads(raw)


# --- phase 1: prepare ---------------------------------------------------------
def prepare() -> None:
    if not GOLDEN.exists():
        sys.exit(f"golden dataset not found: {GOLDEN}")
    AUDIO_DIR.mkdir(exist_ok=True)
    with GOLDEN.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    out_fields = fields + AUDIO_COLS
    for r in rows:
        pid, code = r["pair_id"], r["language_code"]
        r["english_audio_path"] = f"audio_files/{pid}_en.wav"
        r["target_audio_path"] = f"audio_files/{pid}_{code}.wav"
    with EVAL_INPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {EVAL_INPUT}  ({len(rows)} rows)")
    print(f"created {AUDIO_DIR}/  — synthesize 16kHz/mono/16-bit WAVs into the "
          f"paths in columns {AUDIO_COLS}, then run with --run")


# --- phase 2: run -------------------------------------------------------------
def _translate_cell(args, model: str, audio_rel: str, source: str | None, target: str):
    """Blocking wrapper for one (model, direction) cell. Returns (text, error)."""
    path = REPO / audio_rel
    try:
        pcm = load_pcm16_mono_16k(path)
    except FileNotFoundError:
        return "", "missing audio"
    except (wave.Error, ValueError, EOFError) as e:
        return "", f"bad audio: {e}"
    try:
        if model == "qwen":
            text = asyncio.run(translate_qwen(
                pcm, source, target,
                pace=args.pace, timeout=args.qwen_timeout, want_audio=args.qwen_audio))
        else:
            text = asyncio.run(translate_gemini(
                pcm, target, pace=args.pace, drain=args.gemini_drain,
                want_audio=args.gemini_audio))
    except Exception as e:  # noqa: BLE001 — record, never abort the whole run
        return "", f"{type(e).__name__}: {e}"
    finally:
        time.sleep(args.sleep)  # pace each worker to be gentle on rate limits
    if not text:
        return "", "empty output"
    return text, ""


def run(args) -> None:
    if not EVAL_INPUT.exists():
        sys.exit(f"{EVAL_INPUT} not found — run --prepare first")
    with EVAL_INPUT.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    if args.language:
        rows = [r for r in rows if r["language_code"] == args.language]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no rows to run (check --language / --limit)")

    # Init output cells and build the flat task list. Each cell is independent.
    for r in rows:
        for c in RESULT_COLS + ERROR_COLS:
            r[c] = ""
    by_id = {r["pair_id"]: r for r in rows}

    # (pair_id, result_col, error_col, model, audio_rel, source, target)
    tasks = []
    for r in rows:
        pid, code = r["pair_id"], r["language_code"]
        en_audio, tgt_audio = r["english_audio_path"], r["target_audio_path"]
        for model in ("qwen", "gemini"):
            tasks.append((pid, f"{model}_en2tgt", f"{model}_en2tgt_error",
                          model, en_audio, "en", code))
            tasks.append((pid, f"{model}_tgt2en", f"{model}_tgt2en_error",
                          model, tgt_audio, code, "en"))

    print(f"running {len(tasks)} cells across {len(rows)} rows "
          f"({args.workers} workers)...")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_translate_cell, args, model, audio, src, tgt):
                (pid, rcol, ecol)
            for (pid, rcol, ecol, model, audio, src, tgt) in tasks
        }
        for fut in as_completed(futs):
            pid, rcol, ecol = futs[fut]
            text, err = fut.result()
            by_id[pid][rcol] = text
            by_id[pid][ecol] = err
            done += 1
            print(f"  [{done}/{len(tasks)}] {pid} {rcol}: "
                  f"{err or (text[:40] + ('...' if len(text) > 40 else ''))}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = EVALS / f"results_{ts}.csv"
    out_fields = fields + RESULT_COLS + ERROR_COLS
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


# --- self-check: WAV loader accept/reject (no key, no network) -----------------
def _self_check() -> None:
    import tempfile
    d = Path(tempfile.mkdtemp())

    def make_wav(p, ch, width, rate, frames=1600):
        with wave.open(str(p), "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(b"\x00" * frames * ch * width)

    ok = d / "ok.wav"
    make_wav(ok, 1, 2, 16000)
    assert load_pcm16_mono_16k(ok) == b"\x00" * 1600 * 2, "valid 16k mono not read intact"

    for name, ch, width, rate in [
        ("stereo.wav", 2, 2, 16000),
        ("8bit.wav", 1, 1, 16000),
        ("44k.wav", 1, 2, 44100),
    ]:
        p = d / name
        make_wav(p, ch, width, rate)
        try:
            load_pcm16_mono_16k(p)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} should have been rejected")
    print("self-check OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true",
                      help="write evals/eval_input.csv + create audio_files/")
    mode.add_argument("--run", action="store_true", help="run the eval")
    mode.add_argument("--self-check", action="store_true",
                      help="test the WAV loader (no key/network)")
    ap.add_argument("--limit", type=int, help="only the first N rows")
    ap.add_argument("--language", help="only this language_code (e.g. ms)")
    ap.add_argument("--workers", type=int, default=3, help="concurrent sessions")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="seconds a worker sleeps after each call")
    ap.add_argument("--pace", type=float, default=1.0,
                    help="audio send pacing (1.0 = realtime; lower = faster)")
    ap.add_argument("--gemini-drain", type=float, default=7.0,
                    help="seconds to read gemini output after end-of-input")
    ap.add_argument("--qwen-timeout", type=float, default=30.0,
                    help="max seconds to wait for qwen response.done")
    ap.add_argument("--qwen-audio", action="store_true",
                    help="request audio+text from qwen (fallback if text empty)")
    ap.add_argument("--gemini-audio", action="store_true",
                    help="request audio+text from gemini (fallback if text empty)")
    args = ap.parse_args()

    if args.self_check:
        _self_check()
    elif args.prepare:
        prepare()
    else:
        run(args)


if __name__ == "__main__":
    main()
