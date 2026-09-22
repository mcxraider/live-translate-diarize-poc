# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn[standard]", "websockets"]
# ///
"""Qwen3.8 LiveTranslate POC — browser <-> DashScope WebSocket bridge.

Run:      uv run server.py           (needs DASHSCOPE_API_KEY, DASHSCOPE_WORKSPACE_ID)
Selfcheck: uv run server.py --self-check   (no key needed)

The browser owns mic capture + display; this backend owns the DashScope
connection and the API key. It's a thin relay + a normalize_event() parser.
"""
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


@app.websocket("/ws")
async def ws_bridge(browser: WebSocket):
    await browser.accept()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        await browser.send_json({"kind": "status", "type": "error", "raw": "DASHSCOPE_API_KEY not set on server"})
        await browser.close()
        return

    # First message = config from browser.
    cfg = await browser.receive_json()
    target = cfg.get("target_language", "en")
    source = cfg.get("source_language")

    url = dashscope_url()
    print(f"[bridge] opening DashScope: {url}  (target={target}, source={source})")
    try:
        ds = await websockets.connect(url, additional_headers={"Authorization": f"Bearer {api_key}"})
    except Exception as e:  # noqa: BLE001 — surface any connect failure to the browser
        print(f"[bridge] DashScope connect FAILED: {e}")
        await browser.send_json({"kind": "status", "type": "error", "raw": f"DashScope connect failed: {e}"})
        await browser.close()
        return

    await ds.send(json.dumps(_session_update(target, source)))

    async def browser_to_ds():
        try:
            while True:
                msg = await browser.receive_json()
                if msg.get("type") == "input_audio_buffer.append":
                    await ds.send(json.dumps(msg))
                elif msg.get("type") == "session.finish":
                    await ds.send(json.dumps(msg))
        except WebSocketDisconnect:
            pass

    async def ds_to_browser():
        async for raw in ds:
            event = json.loads(raw)
            et = event.get("type", "")
            # Raw-event log: this is how we discover the speaker-id schema live.
            print(f"[ds->] {et}  keys={sorted(event.keys())}")
            out = normalize_event(event)
            if out is not None:
                await browser.send_json(out)

    import asyncio
    b2d = asyncio.create_task(browser_to_ds())
    d2b = asyncio.create_task(ds_to_browser())
    try:
        await asyncio.wait({b2d, d2b}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        try:
            await ds.send(json.dumps({"type": "session.finish", "event_id": f"event_{int(time.time() * 1000)}"}))
        except Exception:  # noqa: BLE001
            pass
        b2d.cancel()
        d2b.cancel()
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
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        print(f"[bridge] resolved DashScope URL: {dashscope_url()}")
        print("[bridge] listening on http://localhost:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
