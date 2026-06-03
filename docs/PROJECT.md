# RIDI — Project Reference

RIDI is a local, real-time AI conversation system. It listens through a microphone, generates responses with a locally-running LLM, speaks back with a cloned voice via GPT-SoVITS TTS, and optionally relays conversations through Discord. A lighter "Lite" mode (`conversation.py`) replaces the local LLM with the Google Gemini API.

---

## File Structure

```
RIDI/
├── infer_v2.py              # Main entry point — Ridi class + ConversationManager
├── conversation.py          # Lite entry point — Gemini API version
├── install.py               # Package build helper
├── req.bat                  # One-click Windows installer (conda + pip + torch CUDA)
├── environment.yml          # Conda environment definition (env name: mini, python 3.10)
├── requirements.txt         # pip dependencies
│
├── configs/
│   ├── __init__.py          # Re-exports SystemConfig, QueueManager, EventManager
│   ├── system_config.py     # All paths, model names, feature flags
│   ├── AppConfig.py         # Initializes every subsystem; holds all component references
│   ├── event_manager.py     # Singleton: exit / interrupt / llm_tts_idle events
│   └── queue_manager.py     # Singleton: user_input, audio, llm_task, discord queues
│
├── core/
│   ├── audio/
│   │   ├── AudioProcess.py  # Microphone capture → VAD → Whisper STT + audio playback
│   │   ├── AudioThreads.py  # Thread entry-point wrappers
│   │   ├── SilenceDetector.py  # Detects prolonged silence to trigger auto-continue
│   │   └── SpeechBuffer.py  # Two-stage text buffering between transcription fragments
│   │
│   ├── engine/
│   │   ├── ModelChat.py     # LLM wrapper: streaming, single-shot, tool-decision calls
│   │   ├── prompt_builder.py # Assembles system prompts (final response, summary, memory)
│   │   └── Summary.py       # Saves conversation turns to a CSV dataset file
│   │
│   ├── memory/
│   │   ├── MemoryItem.py    # Dataclass: id, content, tags, metadata, timestamp
│   │   ├── stm/             # Short-Term Memory (Redis)
│   │   │   ├── STM_Manager.py     # Orchestrates turns, summarization, eviction
│   │   │   └── RedisControl.py    # Low-level Redis list/string helpers
│   │   └── ltm/             # Long-Term Memory (SQLite + FAISS)
│   │       ├── LTM_Manager.py           # MemoryOrchestrator: hybrid RAG search
│   │       ├── keyword_search_sqlite.py # Full-text + tag search (SQLite FTS5)
│   │       ├── semantic_search_faiss.py # Vector search (FAISS + SentenceTransformer)
│   │       ├── extract_key.py           # Keyword extraction utilities
│   │       ├── korean_text_analyzer.py  # Korean morpheme analysis
│   │       ├── MiniKoNLP.py             # Lightweight Korean NLP
│   │       ├── MemGraph.py              # Memory graph (WIP)
│   │       └── Reranker.py              # Reranking (WIP)
│   │
│   ├── tools/
│   │   ├── discord_bot.py   # Discord client; runs in a separate process
│   │   ├── ex_tools.py      # Example tool implementations for function calling
│   │   ├── extract_subject.py
│   │   ├── gpu_usage.py
│   │   ├── ltm_score_analyzer.py
│   │   ├── setup_data.py    # CLI helper to initialize LTM data files
│   │   └── modeltools/
│   │       ├── FunctionSchema.py  # TOOL_REGISTRY + TOOLS_SCHEMA (JSON schema for LLM)
│   │       └── FunctionCall.py    # Executes a tool call returned by the LLM
│   │
│   └── tts/                 # GPT-SoVITS TTS engine (forked from RVC-Boss/GPT-SoVITS)
│       ├── tts.py                  # run_sovits_tts(): yields (sample_rate, audio_chunk)
│       ├── config.py               # TTS runtime config
│       ├── TTS_infer_pack/
│       │   ├── TTS.py              # TTS + TTS_Config: loads GPT and SoVITS weights
│       │   └── TextPreprocessor.py # Cleans text before synthesis
│       ├── AR/                     # Autoregressive token predictor (GPT half)
│       │   └── models/t2s_model.py
│       ├── module/                 # VITS decoder (SoVITS half)
│       │   └── models.py
│       ├── feature_extractor/
│       │   ├── cnhubert.py         # Chinese HuBERT for semantic tokens
│       │   └── whisper_enc.py      # Whisper encoder for reference audio
│       └── text/                   # Multi-language G2P (Chinese, Korean, Japanese, English)
│           ├── korean.py
│           ├── chinese.py
│           └── english.py
│
├── data/                    # Runtime data (not tracked in git by default)
│   ├── model/               # GGUF LLM file + Korean embedding model directory
│   ├── voice/               # Reference WAV + text for TTS voice cloning
│   ├── voice_model/         # GPT-SoVITS .ckpt and .pth weight files
│   ├── ltm_data/            # ltm_storage.sqlite, faiss_ltm_index.idx, faiss_ltm_meta.json
│   └── logs/                # Conversation CSV, summary text, debug logs
│
└── utils2/
    ├── debug_logger.py      # Configurable LLM I/O logger (llm_only / all modes)
    └── Keydata_Creator.py
```

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `llama-cpp-python` | ≥0.3.9 | Local GGUF LLM inference (GPU offloaded via `n_gpu_layers=-1`) |
| `torch` + CUDA | ≥2.6.0+cu124 | GPU tensor operations for TTS, VAD, embeddings |
| `faster-whisper` | ≥1.1.1 | Speech-to-text (Whisper large-v3 on CUDA, float16) |
| `silero-vad` | via torch.hub | Voice Activity Detection — detects speech start/end frames |
| `sounddevice` | ≥0.5.2 | Microphone capture and audio playback (InputStream / OutputStream) |
| `sentence-transformers` | ≥4.1.0 | Korean semantic embedding (`jhgan/ko-sroberta-multitask`, 768-dim) |
| `faiss-cpu` | ≥1.11.0 | Approximate-nearest-neighbor vector search for LTM |
| `redis` | ≥6.2.0 | Short-term memory storage (port 6849 by default) |
| `pytorch-lightning` | latest | Training scaffolding used by GPT-SoVITS AR model |
| `einops` | ≥0.8.1 | Tensor reshaping in TTS attention layers |
| `librosa` | ≥0.11.0 | Audio feature extraction in TTS pipeline |
| `noisereduce` | ≥3.0.3 | Optional audio noise suppression |
| `LangSegment` | 0.3.5 (manual) | Multi-language text segmentation for TTS |
| `pypinyin` | ≥0.54.0 | Chinese pinyin for TTS G2P |
| `cn2an` | ≥0.5.23 | Chinese numeral normalization |
| `jieba_fast` | ≥0.53 | Chinese word segmentation |
| `g2pk2` / `ko_pron` / `jamo` | — | Korean G2P and phoneme tools |
| `g2p_en` | ≥2.1.0 | English G2P |
| `discord.py` | ≥2.10.2 | Discord bot client |
| `openai` | ≥1.88.0 | OpenAI-compatible API client (optional alternative LLM) |
| `google-genai` | — | Google Gemini API (Lite mode only) |
| `pydantic` | ≥2.11.7 | Data validation |
| `pandas` | ≥2.3.0 | Conversation log CSV handling |
| `ffmpeg-python` | ≥0.2.0 | Audio format conversion |

---

## Architecture & Pipeline

### Full Mode (`python infer_v2.py`)

```
Microphone
    │
    ▼ sounddevice InputStream (16 kHz, mono, 512-frame chunks)
AudioProcessor (AudioProcess.py)
    │  Silero VAD — detects speech start/end
    │  Collects audio frames while speaking
    │  On speech end → faster-whisper transcription (Korean, beam=5)
    │  SpeechBuffer assembles fragments into complete utterances
    │
    ▼ user_input_queue (multiprocessing.Queue — shared with Discord process)
ConversationManager._loop_step() (infer_v2.py — runs in main thread, ~10 Hz poll)
    │  Priority: pending_input > queue > auto-continue (silence timeout)
    │  On interrupt: clears audio queue, merges pending inputs, rolls back STM turn
    │
    ▼ llm_task_queue (queue.Queue)
LLM Worker Thread (ConversationManager.llm_worker_target)
    │
    ├─ _gather_prompt_context(user_utter)
    │       └─ LTMloader → MemoryOrchestrator.search_memories_for_rag()
    │              ├─ KeywordSearchSQLite.search() (SQLite FTS5)
    │              ├─ SemanticSearchFAISS.search() (FAISS cosine similarity)
    │              └─ Hybrid score: semantic_weight * cosine + keyword_weight * rank_norm
    │       └─ STM.get_summary_for_prompt() → Redis summary window
    │
    ├─ PromptBuilder.build_final_response_prompt(memory, summary, user_name)
    ├─ STM.add_turn(role="user", ...)  ──► Redis list
    ├─ ModelChat.generate_stream(messages) ──► llama-cpp streaming (Llama-3 chat format)
    │       Interrupts stop the generator mid-stream via EventManager.interrupt_event
    │
    ├─ STM.add_turn(role="assistant", ...)
    │       └─ manage_memory_flow():
    │              If STM full: evict oldest turns → summarize → add to summary window
    │              If summary window full: re-summarize → archive to LTM
    │
    └─ Output handler (keyed by source: "mic" | "discord" | "auto_continue")
           mic → process_tts_for_buffer()
                    └─ run_sovits_tts(tts_pipeline, text, config)
                            ├─ TextPreprocessor: language detection, G2P
                            ├─ AR model: text tokens → semantic tokens (autoregressive)
                            └─ VITS decoder: semantic tokens → waveform (32 kHz)
                    └─ audio_queue.put(chunk) ──► AudioPlayback thread
           discord → discord_response_queue.put(response_data)
```

### Lite Mode (`python conversation.py`)

```
Microphone → AudioProcessor → user_input_queue
    │
SemiConvManager._loop_step()
    │
    ├─ Redis STM: fetch last 20 entries for context
    ├─ Gemini API (gemini-2.5-flash) → text response
    ├─ Save to conversation.txt (JSONL, one entry per turn)
    ├─ Redis STM: push new user + AI entries, trim to 100
    └─ run_sovits_tts() → audio_queue → AudioPlayback thread
```

---

## Memory System

### Short-Term Memory (STM)

Backed by **Redis** (default: `localhost:6849`). Three keys per user:

| Redis key | Type | Contents |
|---|---|---|
| `stm:{user_id}` | list | JSON-encoded `{"role", "content"}` turns |
| `summary_window:{user_id}` | list | LLM-generated summaries of evicted turns |
| `interrupted_utterance:{user_id}` | string | Partial user utterance saved during interrupt |

Three operating modes (set in `STM_Manager.__init__`):

- **original** — evicts oldest turns to summary window when STM exceeds `max_stm_conversations * 2` messages; archives summary window to LTM when it exceeds `max_summary_window_size`.
- **mini** — summarizes each user+assistant turn pair immediately; bypasses full turn history for prompt context.
- **semi** — summarizes and archives the entire STM when it fills up, then resets.

On startup, any leftover STM from a previous session is summarized and archived to LTM before clearing.

### Long-Term Memory (LTM)

Dual-store:

1. **SQLite** (`data/ltm_data/ltm_storage.sqlite`) — keyword search via FTS5. Each `MemoryItem` stores: `item_id`, `content`, `tags`, `metadata` (JSON), `created_at_timestamp`, `source`.
2. **FAISS** (`data/ltm_data/faiss_ltm_index.idx` + `faiss_ltm_meta.json`) — vector search using 768-dim embeddings from `jhgan/ko-sroberta-multitask`.

**Retrieval (RAG):** `MemoryOrchestrator.search_memories_for_rag()` runs both stores in parallel, merges candidates, normalizes scores, and returns top-k by:

```
final_score = semantic_weight * cosine_similarity + keyword_weight * normalized_rank
```

Memories tagged `memory_type = "core"` receive a score boost (`boost_factor`, default 1.5).

**Keyword extraction strategies** (configured in `SystemConfig.ltm_keyword_config`):

| Strategy | Description |
|---|---|
| `simple` | Whitespace tokenization (default, no dependencies) |
| `okt` | Korean Okt morpheme analyzer (requires konlpy) |
| `llm` | LLM extracts keywords via prompt template |

---

## Concurrency Model

The system uses a mix of threads and one subprocess:

| Component | Mechanism | Queue/Event |
|---|---|---|
| Main loop (`_loop_step`) | Main thread | reads `user_input_queue`, writes `llm_task_queue` |
| LLM worker | `threading.Thread` (daemon) | reads `llm_task_queue` |
| Audio playback | `threading.Thread` (daemon) | reads `audio_queue` |
| Microphone / STT | `threading.Thread` (daemon) | writes `user_input_queue` |
| Discord bot | `multiprocessing.Process` (daemon) | reads/writes `user_input_queue`, `discord_queue` (both `multiprocessing.Queue`) |

`EventManager` exposes three events:

- `exit_event` — `multiprocessing.Event`, set on shutdown; checked by all threads and the Discord process.
- `interrupt_event` — `threading.Event`, set when a user speaks while the AI is responding; causes the LLM stream and TTS to abort mid-flight.
- `llm_tts_idle_event` — `threading.Event`, set when the LLM worker is idle; gating condition for auto-continue silence detection.

`QueueManager` and `EventManager` are both **singletons** using double-checked locking with `multiprocessing.RLock`.

---

## Configuration Reference (`configs/system_config.py`)

| Field | Default | Description |
|---|---|---|
| `user_name` | `"재한"` | Username injected into prompts and Redis keys |
| `model_path` | `./data/model/ridi-v0.2-3b-q4_k_m.gguf` | GGUF model file |
| `model_n_ctx` | `4096` | LLM context window |
| `auto_continue_enabled` | `False` | Trigger AI monologue when silence is detected |
| `tts_enabled` | `True` | Enable/disable TTS pipeline |
| `sample_rate` | `32000` | TTS output sample rate (Hz) |
| `gpt_sovits_config_path` | `core/tts/configs/tts_infer.yaml` | TTS config YAML |
| `gpt_sovits_gpt_path` | `data/voice_model/*.ckpt` | AR model checkpoint |
| `gpt_sovits_sovits_path` | `data/voice_model/*.pth` | VITS model weights |
| `ref_audio_path` | `data/voice/neuro-sama-tts-file.wav` | Reference WAV for voice cloning |
| `ref_text` | (transcript of ref audio) | Reference text matching the WAV |
| `embedding_model` | `./data/model/jhgan_ko-sroberta-multitask` | Sentence embedding model |
| `embedding_model_dim` | `768` | Embedding dimension |
| `ltm_rag_similarity_threshold` | `0.7` | Minimum score to include a memory in the prompt |
| `ltm_keyword_config` | `{"strategy": "simple"}` | Keyword extraction strategy |
| `discord_bot_enabled` | `True` | Enable Discord process |
| `discord_token` | `$DISCORD_BOT_TOKEN` env var | Discord bot token |
| `debug_log_path` | `llm_io_debug.log` | LLM I/O debug log path |

---

## Environment Variables

| Variable | Used by | Description |
|---|---|---|
| `DISCORD_BOT_TOKEN` | `system_config.py` | Discord bot token |
| `GEMINI_API_KEY` | `conversation.py` | Google Gemini API key (Lite mode) |

---

## Quick Start

### Automatic (Windows)

```bat
rem 1. Run the installer — creates conda env "mini", installs all deps + torch CUDA
req.bat

rem 2. Manually unpack LangSegment into site-packages (required)
rem    Unzip LangSegment-0.3.5-py3-none-any.whl → copy to anaconda3/envs/mini/Lib/site-packages/

rem 3. Place your GGUF model at:
rem    data/model/ridi-v0.2-3b-q4_k_m.gguf

rem 4. Place GPT-SoVITS weights at:
rem    data/voice_model/<name>-e15.ckpt
rem    data/voice_model/<name>_e8_s248.pth

rem 5. Run
conda activate mini
python infer_v2.py
```

### Manual

```bash
conda create -n ridi python=3.10
conda activate ridi
pip install --upgrade pip setuptools wheel
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# unpack LangSegment-0.3.5-py3-none-any.whl into site-packages manually
python infer_v2.py
```

### Lite Mode (Gemini)

```bash
export GEMINI_API_KEY=your_key_here
python conversation.py
```

---

## Data Files Required at Runtime

| Path | Description |
|---|---|
| `data/model/*.gguf` | GGUF quantized LLM (3B recommended) |
| `data/model/jhgan_ko-sroberta-multitask/` | Korean SentenceTransformer model directory |
| `data/voice_model/*.ckpt` | GPT-SoVITS AR model checkpoint |
| `data/voice_model/*.pth` | GPT-SoVITS VITS model weights |
| `data/voice/neuro-sama-tts-file.wav` | Reference audio for voice cloning |
| Redis server at `localhost:6849` | Required for STM (start before running) |

LTM data files (`data/ltm_data/`) are created automatically on first run. To pre-populate them from a JSONL file, run:

```bash
python core/tools/setup_data.py
```

---

## Credits

TTS engine is derived from [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).
