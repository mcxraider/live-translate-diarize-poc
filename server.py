# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn[standard]", "websockets", "google-genai"]
# ///
"""Qwen3.8 LiveTranslate POC — browser <-> DashScope WebSocket bridge.

Run:      uv run server.py           (needs DASHSCOPE_API_KEY, DASHSCOPE_WORKSPACE_ID)
Selfcheck: uv run server.py --self-check   (no key needed)

The browser owns mic capture + display; this backend owns the DashScope
connection and the API key. It's a thin relay + a normalize_event() parser.
"""
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def _load_dotenv():
    """Minimal .env loader (no dep). Doesn't override already-set env vars."""
    env = STATIC_DIR.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()
# 3.8 is required for diarisation; the older qwen3-livetranslate has no speaker_id.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.8-livetranslate-flash-realtime")


def dashscope_url() -> str:
    """Full DashScope realtime WS URL. Single global endpoint (no workspace id).
    Override the whole URL with DASHSCOPE_WS_URL if the docs change."""
    return os.environ.get(
        "DASHSCOPE_WS_URL",
        f"wss://maas.qwencloudapi.com/api-ws/v1/realtime?model={MODEL}",
    )


OPENAI_MODEL = os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-realtime-translate")


def openai_url() -> str:
    """OpenAI realtime translations WS URL. Override with OPENAI_WS_URL."""
    return os.environ.get(
        "OPENAI_WS_URL",
        f"wss://api.openai.com/v1/realtime/translations?model={OPENAI_MODEL}",
    )


# ISO 639-1 mostly matches our LANGS codes; only Filipino differs.
# ponytail: single hardcoded override; make a per-provider map if more appear.
def openai_lang(code: str) -> str:
    return {"fil": "tl"}.get(code, code)


# --- Gemini 3.5 Live Translate (Agent Platform / Vertex, google-genai SDK) ---
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-live-translate-preview")

_gemini_client = None


def gemini_client():
    """Gemini client. Prefers a Developer API key (GEMINI_API_KEY — the Live
    Translate quickstart path); falls back to Vertex/Agent-Platform via ADC
    (enterprise=True + project/location) when no key is set. Lazy so --self-check
    + the other providers need no Google creds or SDK import at module load."""
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


# Our LANGS codes are valid Gemini BCP-47 codes except Mandarin, which needs a script tag.
def gemini_lang(code: str) -> str:
    return {"zh": "zh-Hans"}.get(code, code)


def normalize_event(event: dict) -> dict | None:
    """Map a raw DashScope server event to a flat message for the frontend.
    Returns None for events we don't forward. Pure function — the one bit of
    real logic here, so it has the self-check below. Schema verified live
    against qwen3.8-livetranslate-flash-realtime."""
    et = event.get("type", "")

    # Verified live: speaker_id rides on speech_started, which also marks each
    # utterance/turn boundary. Frontend uses it to start a new speaker card.
    if et == "input_audio_buffer.speech_started":
        return {"kind": "speaker", "speaker": event.get("speaker_id")}
    # Translation: response.text.delta (text mode) / response.audio_transcript.delta (audio mode)
    if et in ("response.audio_transcript.delta", "response.text.delta"):
        return {"kind": "translation", "text": event.get("delta", "")}
    if et == "conversation.item.input_audio_transcription.delta":
        return {"kind": "source", "text": event.get("delta", ""), "final": False}
    if et == "conversation.item.input_audio_transcription.completed":
        return {"kind": "source", "text": event.get("transcript", ""), "final": True}
    if et == "response.audio.delta":
        return {"kind": "audio", "b64": event.get("delta", "")}
    if et in ("response.done", "session.finished", "error"):
        return {"kind": "status", "type": et, "raw": event}
    return None


def normalize_openai(event: dict) -> dict | None:
    """Map a raw OpenAI translations event to the same flat frontend contract.
    OpenAI has no diarisation, so no 'speaker' kind. Event names per docs;
    unmapped events fall through to the [ds->] raw logger for live discovery."""
    et = event.get("type", "")
    if et == "session.output_transcript.delta":
        return {"kind": "translation", "text": event.get("delta", "")}
    if et == "session.input_transcript.delta":
        return {"kind": "source", "text": event.get("delta", ""), "final": False}
    if et == "session.output_audio.delta":
        return {"kind": "audio", "b64": event.get("delta", "")}
    if et in ("session.closed", "error"):
        return {"kind": "status", "type": et, "raw": event}
    return None


def normalize_gemini(sc: dict) -> list[dict]:
    """Map extracted Gemini server_content fields to the same flat frontend
    contract. One server_content can carry several frontend messages, so this
    returns a list. Dict-based (not SDK-object-based) so the keyless self-check
    can exercise it without google-genai."""
    out = []
    if sc.get("input_text"):
        out.append({"kind": "source", "text": sc["input_text"], "final": False})
    if sc.get("output_text"):
        out.append({"kind": "translation", "text": sc["output_text"]})
    for b64 in sc.get("audio_b64", []):
        out.append({"kind": "audio", "b64": b64})
    if sc.get("turn_complete"):  # closes the card so the next utterance starts fresh
        out.append({"kind": "status", "type": "response.done", "raw": {}})
    return out


app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return HTMLResponse("<h1>LiveTranslate POC backend up.</h1><p>No frontend yet — drop index.html into static/.</p>")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _session_update(target_language: str, source_language: str | None) -> dict:
    transcription = {"model": "qwen3-asr-flash-realtime"}
    if source_language:
        transcription["language"] = source_language
    return {
        "event_id": f"event_{int(time.time() * 1000)}",
        "type": "session.update",
        "session": {
            "output_modalities": ["text", "audio"],
            "voice": os.environ.get("DASHSCOPE_VOICE", "Tina"),  # required even in text-only
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "input_audio_transcription": transcription,
            "translation": {"language": target_language},
            # verified: speaker_detection emits speaker_id on speech_started.
            # ~2.5s of silence ends a turn / can switch speaker.
            "turn_detection": {"type": "speaker_detection"},
        },
    }


def _openai_session_update(target_language: str) -> dict:
    # Source is auto-detected; only the output language is configurable.
    return {
        "type": "session.update",
        "session": {"audio": {"output": {"language": openai_lang(target_language)}}},
    }


def _gemini_config(target_language: str):
    from google.genai import types
    return types.LiveConnectConfig(
        response_modalities=["AUDIO", "TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        translation_config=types.TranslationConfig(
            target_language_code=gemini_lang(target_language),
            # False: don't re-emit input already in the target language. =True created
            # an acoustic feedback loop — translated TTS was picked up by the mic,
            # re-fed, and rebroadcast verbatim, repeating one utterance forever.
            echo_target_language=False,
        ),
    )


async def _pump_bridge(send_coro, recv_coro, *, drain_timeout=0.0):
    """Shared spine of every provider bridge: run the two browser<->upstream pump
    coroutines concurrently until one finishes, give the receiver `drain_timeout`
    seconds to flush trailing output, then cancel both. Transport connect/close
    and the graceful-close signal stay at the call site (the send-pump sends it)."""
    send_task = asyncio.create_task(send_coro())
    recv_task = asyncio.create_task(recv_coro())
    try:
        await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        if drain_timeout and not recv_task.done():
            # ponytail: fixed drain window; raise it only if real output truncates.
            try:
                await asyncio.wait_for(asyncio.shield(recv_task), timeout=drain_timeout)
            except Exception:  # noqa: BLE001
                pass
        send_task.cancel()
        recv_task.cancel()


async def gemini_bridge(browser: WebSocket, target: str):
    """Gemini Live path. Unlike qwen/openai (raw WS + Bearer key), Gemini uses the
    google-genai SDK's live session (Vertex/Agent-Platform auth via ADC). Source
    language is auto-detected; 16 kHz PCM in, 24 kHz PCM out.
    ponytail: no session rotation — Live sessions have a ~150s practical cap; on drop
    the frontend shows "disconnected". Rotate with ~0.5s overlap if long sessions matter."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_CLOUD_PROJECT")):
        await browser.send_json({"kind": "status", "type": "error",
                                 "raw": "gemini not configured: set GEMINI_API_KEY (or GOOGLE_CLOUD_PROJECT for Vertex/ADC)"})
        await browser.close()
        return
    from google.genai import types
    print(f"[bridge] opening gemini (target={target} -> {gemini_lang(target)})")
    try:
        async with gemini_client().aio.live.connect(
            model=GEMINI_MODEL, config=_gemini_config(target)
        ) as session:

            async def browser_to_gemini():
                try:
                    while True:
                        msg = await browser.receive_json()
                        t = msg.get("type")
                        if t == "input_audio_buffer.append":
                            pcm = base64.b64decode(msg["audio"])
                            await session.send_realtime_input(
                                audio=types.Blob(data=pcm, mime_type="audio/pcm;rate=16000"))
                        elif t in ("session.finish", "session.close"):
                            await session.send_realtime_input(audio_stream_end=True)
                            return  # let _pump_bridge drain trailing output
                except WebSocketDisconnect:
                    pass

            async def gemini_to_browser():
                async for message in session.receive():
                    sc = message.server_content
                    if not sc:
                        continue
                    d = {"audio_b64": []}
                    if sc.input_transcription and sc.input_transcription.text:
                        d["input_text"] = sc.input_transcription.text
                    if sc.output_transcription and sc.output_transcription.text:
                        d["output_text"] = sc.output_transcription.text
                    if sc.model_turn:
                        for part in sc.model_turn.parts:
                            ind = part.inline_data
                            if ind and (ind.mime_type or "").startswith("audio"):
                                d["audio_b64"].append(base64.b64encode(ind.data).decode())
                    if getattr(sc, "turn_complete", False):
                        d["turn_complete"] = True
                    # Raw-event log for live schema discovery (parity with [ds->]).
                    print(f"[gemini->] in={bool(d.get('input_text'))} out={bool(d.get('output_text'))} "
                          f"audio={len(d['audio_b64'])} turn={d.get('turn_complete', False)}")
                    for out in normalize_gemini(d):
                        await browser.send_json(out)

            # Spike finding: translated audio arrives ~1-3s behind input and the
            # model never sends turn_complete, so recv() never ends on its own —
            # _pump_bridge always waits the full drain then cancels. 3s captures
            # the final utterance's audio tail without a long idle-silence wait.
            await _pump_bridge(browser_to_gemini, gemini_to_browser, drain_timeout=3)
    except Exception as e:  # noqa: BLE001 — surface connect/auth failure to the browser
        print(f"[bridge] gemini FAILED: {e}")
        try:
            await browser.send_json({"kind": "status", "type": "error", "raw": f"gemini failed: {e}"})
        except Exception:  # noqa: BLE001
            pass
    try:
        await browser.close()
    except Exception:  # noqa: BLE001
        pass
    print("[bridge] gemini closed")


@app.websocket("/ws")
async def ws_bridge(browser: WebSocket):
    await browser.accept()
    provider = browser.query_params.get("provider", "qwen")

    # First message = config from browser.
    cfg = await browser.receive_json()
    target = cfg.get("target_language", "en")
    source = cfg.get("source_language")

    # Gemini has its own transport (SDK live session, no Bearer key) — handle
    # it before the shared raw-websocket path below.
    if provider == "gemini":
        return await gemini_bridge(browser, target)

    # Per-provider config; everything after the branch is shared.
    # finish_frame = graceful upstream close sent when the browser stops; drain =
    # seconds to keep forwarding trailing audio after it (openai emits a tail).
    if provider == "openai":
        url, api_key = openai_url(), os.environ.get("OPENAI_API_KEY")
        session_update = _openai_session_update(target)  # no source — auto-detected
        normalize = normalize_openai
        keyname = "OPENAI_API_KEY"
        finish_frame = json.dumps({"type": "session.close"})
        drain = 5
    else:
        url, api_key = dashscope_url(), os.environ.get("DASHSCOPE_API_KEY")
        session_update = _session_update(target, source)
        normalize = normalize_event
        keyname = "DASHSCOPE_API_KEY"
        finish_frame = json.dumps({"type": "session.finish", "event_id": f"event_{int(time.time() * 1000)}"})
        drain = 0

    if not api_key:
        await browser.send_json({"kind": "status", "type": "error", "raw": f"{keyname} not set on server"})
        await browser.close()
        return

    print(f"[bridge] opening {provider}: {url}  (target={target}, source={source})")
    try:
        ds = await websockets.connect(url, additional_headers={"Authorization": f"Bearer {api_key}"})
    except Exception as e:  # noqa: BLE001 — surface any connect failure to the browser
        print(f"[bridge] {provider} connect FAILED: {e}")
        await browser.send_json({"kind": "status", "type": "error", "raw": f"{provider} connect failed: {e}"})
        await browser.close()
        return

    await ds.send(json.dumps(session_update))

    async def browser_to_ds():
        try:
            while True:
                msg = await browser.receive_json()
                t = msg.get("type")
                if t == "input_audio_buffer.append":
                    await ds.send(json.dumps(msg))
                elif t in ("session.finish", "session.close"):
                    # Browser is done but keeps its socket open: send the graceful
                    # upstream close, then return so _pump_bridge drains the tail.
                    await ds.send(finish_frame)
                    return
        except WebSocketDisconnect:
            pass

    async def ds_to_browser():
        async for raw in ds:
            event = json.loads(raw)
            et = event.get("type", "")
            # Raw-event log: this is how we discover each provider's schema live.
            print(f"[ds->] {et}  keys={sorted(event.keys())}")
            out = normalize(event)
            if out is not None:
                await browser.send_json(out)

    try:
        await _pump_bridge(browser_to_ds, ds_to_browser, drain_timeout=drain)
    finally:
        await ds.close()
        print("[bridge] closed")


def _self_check():
    assert normalize_event({"type": "input_audio_buffer.speech_started", "speaker_id": 2}) == {
        "kind": "speaker", "speaker": 2}
    assert normalize_event({"type": "response.text.delta", "delta": "hi"}) == {"kind": "translation", "text": "hi"}
    assert normalize_event({"type": "response.audio_transcript.delta", "delta": "hi"}) == {"kind": "translation", "text": "hi"}
    assert normalize_event({"type": "conversation.item.input_audio_transcription.delta", "delta": "he"}) == {
        "kind": "source", "text": "he", "final": False}
    assert normalize_event({"type": "conversation.item.input_audio_transcription.completed", "transcript": "你好"}) == {
        "kind": "source", "text": "你好", "final": True}
    assert normalize_event({"type": "response.audio.delta", "delta": "AAA="}) == {"kind": "audio", "b64": "AAA="}
    assert normalize_event({"type": "some.unknown.event"}) is None

    assert normalize_openai({"type": "session.output_transcript.delta", "delta": "hi"}) == {
        "kind": "translation", "text": "hi"}
    assert normalize_openai({"type": "session.input_transcript.delta", "delta": "ni"}) == {
        "kind": "source", "text": "ni", "final": False}
    assert normalize_openai({"type": "session.output_audio.delta", "delta": "AAA="}) == {"kind": "audio", "b64": "AAA="}
    assert normalize_openai({"type": "session.closed"}) == {
        "kind": "status", "type": "session.closed", "raw": {"type": "session.closed"}}
    assert normalize_openai({"type": "session.some.unknown"}) is None
    assert openai_lang("fil") == "tl" and openai_lang("en") == "en"

    assert gemini_lang("zh") == "zh-Hans" and gemini_lang("en") == "en" and gemini_lang("fil") == "fil"
    assert normalize_gemini({"input_text": "hi"}) == [{"kind": "source", "text": "hi", "final": False}]
    assert normalize_gemini({"output_text": "bonjour"}) == [{"kind": "translation", "text": "bonjour"}]
    assert normalize_gemini({"audio_b64": ["AAA="]}) == [{"kind": "audio", "b64": "AAA="}]
    assert normalize_gemini({"turn_complete": True}) == [{"kind": "status", "type": "response.done", "raw": {}}]
    assert normalize_gemini({}) == []
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        print(f"[bridge] resolved DashScope URL: {dashscope_url()}")
        print("[bridge] listening on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
