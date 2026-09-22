# Live Translate + Diarize POC

Real-time speech translation with speaker diarization, powered by Alibaba's
Qwen3.8 LiveTranslate model over DashScope's realtime WebSocket API.

The browser captures mic audio and shows results; a thin FastAPI backend
relays audio to DashScope and holds the API key. Each utterance is tagged with
a speaker id and shown as its own card, source text above the translation.

![Demo](assets/demo.png)

## Run

```bash
cp .env.example .env        # fill in DASHSCOPE_API_KEY
uv run server.py            # http://localhost:8000
```

Self-check the event parser without a key:

```bash
uv run server.py --self-check
```

## How it works

- `server.py` — FastAPI app. Serves `static/index.html`, bridges the browser
  `/ws` socket to DashScope, and normalizes DashScope events into flat messages
  (`normalize_event`).
- `static/index.html` — mic capture, language selection, speaker cards.
- Diarization needs the 3.8 model (`qwen3.8-livetranslate-flash-realtime`);
  speaker ids ride on `speech_started` events with `speaker_detection` turn
  detection.

## Config

See `.env.example`. `DASHSCOPE_API_KEY` is required; model, voice, and the full
WS URL are overridable via env vars.
