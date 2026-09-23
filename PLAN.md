# Plan: add GPT realtime-translate alongside Qwen

Add OpenAI's `gpt-realtime-translate` as a second provider behind the existing
browser↔server WebSocket bridge, selectable from the UI. Build the seams for a
future side-by-side (both providers at once) without building the split screen.

Decisions locked in the grilling session (Q1–Q8) drive everything below.

---

## 0. Facts this plan relies on

- **Qwen (current):** browser sends 16kHz PCM16 base64 → `/ws` → DashScope.
  Diarisation via `speaker_id` on `input_audio_buffer.speech_started`. Turn end
  = `response.done`. Finish = `session.finish`. Output audio 24kHz.
- **OpenAI translate:** dedicated endpoint
  `wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate`,
  header `Authorization: Bearer $OPENAI_API_KEY`. Input/output **24kHz** PCM16.
  Config via `session.update` sets **only** `audio.output.language` (ISO 639-1
  two-letter code) — source is auto-detected, no source language. **No
  diarisation.** Close protocol: send `session.close`, keep reading until
  `session.closed`, then close socket (early close drops draining audio).
- **OpenAI event names we forward** (from docs):
  `session.output_transcript.delta` (translated text),
  `session.input_transcript.delta` (source text),
  `session.output_audio.delta` (audio), `session.closed`, `error`.
- **UNVERIFIED — discover live:** OpenAI has no documented per-utterance
  *boundary* event and no `*.completed` events for translation sessions. The
  existing code discovered Qwen's schema live via the `[ds->]` raw logger; do
  the same here (§2.5). Until confirmed, OpenAI renders as **one continuous
  card** (§3.4), which is also the honest "no diarisation" comparison point.
- OpenAI's supported target-language list is **not published**; unsupported
  codes surface as `session.error`. Filipino is `fil` (Qwen) vs `tl` (ISO).

---

## 1. `.env.example`

Add:
```
# OpenAI realtime translate (provider=openai)
OPENAI_API_KEY=your_openai_key_here
# Optional override, default below:
# OPENAI_TRANSLATE_MODEL=gpt-realtime-translate
```

## 2. `server.py`

### 2.1 Provider constants
- `OPENAI_MODEL = os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-realtime-translate")`
- `openai_url()` mirroring `dashscope_url()`:
  `wss://api.openai.com/v1/realtime/translations?model={OPENAI_MODEL}`
  (override via `OPENAI_WS_URL`).

### 2.2 Language code override
```python
# ISO 639-1 mostly matches our LANGS codes; only Filipino differs.
# ponytail: single hardcoded override; make a per-provider map if more appear.
def openai_lang(code: str) -> str:
    return {"fil": "tl"}.get(code, code)
```

### 2.3 Per-provider config builders
- Keep `_session_update(...)` as the **Qwen** builder (unchanged).
- Add `_openai_session_update(target_language)`:
```python
{"type": "session.update",
 "session": {"audio": {"output": {"language": openai_lang(target_language)}}}}
```
  (No source, no voice, no turn_detection.)

### 2.4 Per-provider normalize
- Keep `normalize_event` (Qwen) unchanged.
- Add `normalize_openai(event)` → same flat `{kind:...}` contract:
  - `session.output_transcript.delta` → `{"kind":"translation","text":delta}`
  - `session.input_transcript.delta`  → `{"kind":"source","text":delta,"final":False}`
  - `session.output_audio.delta`      → `{"kind":"audio","b64":delta}`
  - `error` / `session.closed`        → `{"kind":"status","type":et,"raw":event}`
  - else `None`
  - (No `speaker` kind — OpenAI has no diarisation.)

### 2.5 `/ws` dispatch
Read provider from query: `browser: WebSocket` handler reads
`browser.query_params.get("provider", "qwen")`. Then branch a small config:
```python
if provider == "openai":
    url, key = openai_url(), os.environ.get("OPENAI_API_KEY")
    session_update = _openai_session_update(target)          # no source
    normalize = normalize_openai
    keyname = "OPENAI_API_KEY"
else:
    url, key = dashscope_url(), os.environ.get("DASHSCOPE_API_KEY")
    session_update = _session_update(target, source)
    normalize = normalize_event
    keyname = "DASHSCOPE_API_KEY"
```
Everything after (accept, missing-key error using `keyname`, connect, send
`session_update`, the two pump tasks) stays shared. `ds_to_browser` keeps the
raw `[ds->]` logger — this is how we discover OpenAI's real event schema live.

### 2.6 Close/drain (Q7)
Replace the current `finally` block with provider-aware drain:
- **qwen:** unchanged — best-effort `session.finish`, cancel tasks, close.
- **openai:** send `{"type":"session.close"}`, then **keep reading upstream and
  forwarding to the browser until a `session.closed` event arrives** (with a
  timeout guard, e.g. 5s), then close. Do not cancel `ds_to_browser` before the
  drain completes.
  - Implementation: when `browser_to_ds` ends (browser gone/finished), trigger
    the drain in `ds_to_browser` rather than cancelling it immediately.
  - `ponytail: 5s drain timeout; raise only if real output gets truncated.`

### 2.7 Self-check
Extend `_self_check()` with `normalize_openai` assertions for each mapped event
+ one `None` case. Keep it runnable via `--self-check` (no key needed).

## 3. `static/index.html`

### 3.1 Session object refactor (Q2 seam)
Today `ws / cur / curSpeaker / nextTime / speakerColors / playCtx` are module
globals. Wrap per-connection state in a `Session`:
```js
function createSession({ provider, mount }) {
  // owns: ws, cur, curSpeaker, nextTime, playCtx, speakerColors
  // methods: start(cfg), stop(), handle(msg), and card rendering into `mount`
}
```
- Mic capture (`getUserMedia` + ScriptProcessor) stays **outside** the session
  (one mic) and pushes PCM into whichever session(s) are active. For now a
  single active session; the fan-out to N sessions is the future seam.
- `#log` becomes the mount for the single current session.
- `ponytail: one session today; side-by-side = two sessions into two mounts.`

### 3.2 Provider dropdown
Header gets `<label>Provider <select id="provider"><option>Qwen</option>
<option>OpenAI</option></select></label>`. Changing it restarts like language
changes do (`applyLangChange` → generalize to `applyChange`).

### 3.3 "From" selector gating (Q3)
When provider = OpenAI: `$("src").disabled = true`, greyed, and show an
"auto-detected" hint near it. Re-enable for Qwen. The swap button is disabled
too when source is meaningless.

### 3.4 Card rendering per provider (Q4)
- **qwen:** unchanged — `speaker` msg starts a new card, label `Speaker N`,
  color per speaker.
- **openai:** no `speaker` msgs arrive. Start with **one running card** labeled
  plain `Speaker` (no number, neutral color); source/translation deltas append
  to it. Reset the card on `response`/utterance-boundary **once we confirm the
  real boundary event from the live log** (§2.5) — until then, one card.
  - `ponytail: single running card until OpenAI's boundary event is verified.`

### 3.5 WS URL + config send
- `wsUrl` gains `?provider=${provider}` (preserve existing `?backend=` override;
  append provider to whichever URL is built).
- `config` message still sends `source_language` + `target_language`; backend
  ignores source for OpenAI. No frontend change to the config payload.

### 3.6 Capture sample rate (Q6)
`startMic()` sets `AudioContext({ sampleRate: provider === "openai" ? 24000 : 16000 })`.
- `ponytail: per-provider capture rate; side-by-side needs one-rate capture +
  a JS resampler for the non-native provider.`

### 3.7 Playback
No change — OpenAI output is 24kHz PCM16, same as the current `playCtx`
(`sampleRate: 24000`) and `playChunk`. Reused as-is.

## 4. Explicitly deferred (seams left, code not written)
- Two-column simultaneous Qwen+OpenAI view.
- JS resampler so one mic feeds both providers at once.
- Generalized per-provider language-code map (only `fil→tl` today).
- WebRTC transport for OpenAI (staying on server WebSocket relay).

## 5. Verification
1. `uv run server.py --self-check` — passes (Qwen + OpenAI normalize).
2. Manual: `provider=qwen` still works end-to-end (no regression).
3. Manual: `provider=openai` with a real key — source auto-detected, translated
   text + audio play, `From` greyed, one card. Watch server `[ds->]` log to
   confirm OpenAI's real event names and any utterance-boundary event; fold
   findings back into §2.4 / §3.4.
4. Try an unsupported OpenAI language (e.g. `ms`) — confirm it surfaces as a
   status-line error, not a silent hang.

## Open risk
OpenAI's per-utterance boundary + `*.completed` events are unconfirmed. Plan
ships the honest single-card fallback and discovers the real schema from the
live log, exactly as the Qwen path was built. No architectural change expected
either way — only §3.4 card-reset logic.
