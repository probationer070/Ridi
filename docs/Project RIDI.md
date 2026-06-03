# Project RIDI — Building a Local AI That Listens, Remembers, and Talks Back

*A deep-dive into architecture, memory design, and a day of refactoring a real-time Korean voice AI.*

---

## What Is RIDI?

RIDI is a local, real-time AI companion that runs entirely on your machine. You talk to it through your microphone. It listens, transcribes your speech, thinks with a locally-running LLM, and answers in a cloned voice — all within a few seconds, with no cloud API required for the core loop.

The name fits the personality: RIDI is sarcastic, opinionated, a little jealous, fond of tiramisu, and built specifically for Korean-language conversation. It was built as a personal project — not as a product demo, but as a working system that a developer actually uses.

The hardware requirement is real: you need a CUDA-capable GPU. The stack uses `llama-cpp-python` for GGUF-quantized local inference, `faster-whisper` for speech-to-text, `silero-vad` for voice activity detection, and a full GPT-SoVITS pipeline for voice cloning. Redis holds the short-term conversation window; SQLite and FAISS back the long-term memory. It is, by any definition, a heavy stack — and it runs on a single Windows machine.

---

## The Two Modes

### Full Mode (`python infer_v2.py`)

The production path. Every component runs locally:

```
Microphone
    ↓ sounddevice (16 kHz, mono)
AudioProcessor — Silero VAD + faster-whisper (Whisper large-v3, Korean)
    ↓ user_input_queue (multiprocessing.Queue)
ConversationManager._loop_step()  (main thread, ~10 Hz poll)
    ↓ llm_task_queue
LLM Worker Thread
    ├─ MemoryContext.context_for(utterance)
    │       ├─ FAISS semantic search + SQLite FTS5 keyword search (hybrid RAG)
    │       └─ Redis summary window
    ├─ PromptBuilder → system prompt with persona + memory + summary
    ├─ ModelChat.generate_stream() → llama-cpp streaming (Llama-3 chat format)
    ├─ STM.add_turn() → Redis → manage_memory_flow()
    └─ process_tts_for_buffer() → GPT-SoVITS → audio_queue → AudioPlayback thread
```

### Lite Mode (`python conversation.py`)

A lighter variant using the Google Gemini API instead of the local LLM. The audio pipeline (VAD, Whisper, TTS) is identical. STM is a raw Redis list instead of the full `ShortTermMemoryManager`. LTM is disabled. This mode exists for devices where you want the voice interface but can't run a 3B GGUF model at useful speed.

---

## The Memory System

RIDI's most interesting engineering is the two-layer memory system. Most chatbot demos run with a flat context window. RIDI runs with three tiers.

### Tier 1 — Active Context (the LLM's token window)

The last N conversation turns, formatted as a `messages` list and passed directly to the LLM. `N` is small (4 turns by default) to keep latency low.

### Tier 2 — Short-Term Memory (Redis)

Three Redis keys per user:

| Key | Type | Contents |
|-----|------|----------|
| `stm:{user_id}` | list | JSON-encoded `{role, content}` turns |
| `summary_window:{user_id}` | list | LLM-generated summaries of evicted turns |
| `interrupted_utterance:{user_id}` | string | Partial utterance saved during mid-response interrupt |

When the STM list exceeds `max_stm_conversations * 2` messages, the oldest turns are evicted — but not deleted. The LLM summarizes them first. That summary goes into the `summary_window`, which feeds into every future prompt as the "previous conversation" context.

When the summary window itself fills up, the LLM re-summarizes the whole window into a single compact memory and archives it to LTM.

Three modes control this flow: `original` (rolling eviction), `mini` (summarize every turn pair immediately), and `semi` (summarize and reset when STM is full).

### Tier 3 — Long-Term Memory (SQLite + FAISS)

Two parallel stores:

- **SQLite with FTS5** — full-text keyword search. Each `MemoryItem` stores: content, tags, metadata (JSON), timestamp, source.
- **FAISS IndexFlatIP** — 768-dimensional semantic vectors from `jhgan/ko-sroberta-multitask`, a Korean sentence embedding model.

At query time, `MemoryOrchestrator.search_memories_for_rag()` runs both stores in parallel, merges candidates, and computes a hybrid score:

```
final_score = (semantic_weight × cosine_similarity) + (keyword_weight × normalized_rank)
```

Memories tagged `memory_type = "core"` receive a configurable score boost (default 1.5×), ensuring that important long-term facts remain relevant even when semantic similarity is moderate. The top-k results above `ltm_rag_similarity_threshold` (default 0.7) are injected into the system prompt.

---

## The Concurrency Model

RIDI is a multi-threaded, single-process application with one subprocess for the Discord bot.

| Component | Mechanism | Communication |
|-----------|-----------|---------------|
| Main loop | Main thread | reads `user_input_queue`, writes `llm_task_queue` |
| LLM worker | `threading.Thread` (daemon) | reads `llm_task_queue` |
| Audio playback | `threading.Thread` (daemon) | reads `audio_queue` |
| Microphone / STT | `threading.Thread` (daemon) | writes `user_input_queue` |
| Discord bot | `multiprocessing.Process` (daemon) | reads/writes `user_input_queue`, `discord_queue` |

Three events coordinate the system:

- `exit_event` — a `multiprocessing.Event`, set on shutdown, checked by all threads and the Discord process
- `interrupt_event` — a `threading.Event`, set when the user speaks mid-response; causes the LLM stream and TTS to abort
- `llm_tts_idle_event` — a `threading.Event`, gates the auto-continue silence detection

The `interrupt_event` wires directly into `ModelChat.generate_stream()`, which checks it after each token chunk. When set, streaming stops immediately, the partial response is discarded, and the next input takes over.

---

## A Day of Refactoring — Breaking the Monolith

Until today, the entire per-turn logic lived in one method: `ConversationManager._process_single_turn()`. That 40-line block did five unrelated things without a seam:

1. LTM retrieval (FAISS + SQLite)
2. STM summary loading (Redis)
3. Prompt assembly
4. LLM streaming
5. STM write-back and CSV logging

To run a test against any part of this, you needed a live GPU, a running Redis server, and an initialized FAISS index. That's not a unit test — that's a full integration run.

Today, three injectable seams were carved out in order.

### Seam 1 — MemoryContext

```python
# core/memory/MemoryContext.py
class MemoryContext:
    def __init__(self, ltm_manager, stm_manager, config): ...
    def context_for(self, query: str) -> dict:
        return {
            "memory": self._load_ltm(query),   # FAISS + SQLite, threshold-filtered
            "summary": self._load_stm_summary() # Redis summary window
        }
```

All knowledge about RAG thresholds, score filtering, and summary formatting lives here. `ConversationManager` no longer imports `MemoryItem`, `List`, or `Tuple`. The `_gather_prompt_context` method collapsed from 12 lines to one:

```python
def _gather_prompt_context(self, user_utter: str) -> dict:
    return self.memory_context.context_for(user_utter)
```

### Seam 2 — Turn

```python
# core/engine/Turn.py
class Turn:
    def __init__(self, backend, stm_manager, prompt_builder,
                 memory_context, user_name: str, log_path: str): ...
    def run(self, user_utter: str) -> Optional[str]: ...
```

The entire per-turn pipeline moved here. `Turn` depends only on its constructor arguments — no `AppConfig`, no `SystemConfig`. A unit test can construct one as:

```python
Turn(
    backend=FakeBackend(),
    stm_manager=None,
    prompt_builder=real_builder,
    memory_context=FakeMemory(),
    user_name="test",
    log_path="/dev/null"
)
```

No GPU. No Redis. No FAISS. `llm_worker_target` in `ConversationManager` now calls `self.turn.run(user_utter)` — one line instead of 40.

Note that `config` was removed from `Turn` entirely in a follow-up step. The full `SystemConfig` was being passed in, but `Turn.run()` only read two fields: `user_name` and `conversation_log_path`. Those are now plain `str` arguments — zero config object dependency.

### Seam 3 — LLMBackend

```python
# core/engine/LLMBackend.py
class LLMBackend(ABC):
    @abstractmethod
    def generate_stream(self, messages: List[Dict]) -> Iterator[str]: ...

class LocalLlamaBackend(LLMBackend):
    def __init__(self, model_chat): ...
    def generate_stream(self, messages): return self.model_chat.generate_stream(messages)

class GeminiBackend(LLMBackend):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"): ...
    def generate_stream(self, messages): ...  # converts message format, yields response.text
```

Before this, swapping Full mode (local llama-cpp) for Lite mode (Gemini) meant overriding `AppConfig.init_components()` in a subclass. Now it is one line in `AppConfig`:

```python
# Full mode
self.llm_backend = LocalLlamaBackend(self.model_chat)

# Lite mode (SemiAppConfig)
self.llm_backend = GeminiBackend(api_key=get_apikey())
```

`Turn` calls `self.backend.generate_stream(final_messages)` and knows nothing about llama-cpp or Gemini.

---

## Bugs Found in Code Review

A code review pass surfaced 10 findings across the codebase. Four are high severity — silent data corruption that runs undetected.

### 1. FAISS `-1` padding injects wrong memories (High)

`SemanticSearchFAISS` (`semantic_search_faiss.py:213`) checks `if idx < len(metadata_list)` to validate FAISS results. When the index has fewer entries than `top_k`, FAISS pads results with `idx = -1`. Python's negative indexing makes `-1 < len(list)` true, so `metadata_list[-1]` — the last stored memory — silently appears in every under-full query.

```python
# Fix: one extra check
if idx != -1 and idx < len(self.metadata_list):
```

### 2. STM eviction deletes the newest turns before reading them (High)

`STM_Manager._evict_turns_to_summary_window()` (`STM_Manager.py:398`) calls `ltrim key 0 -N` to trim the list *before* reading the items to summarize. `ltrim 0 -2` on a list of 10 items deletes the last 1 item. Those turns are gone before the summarizer ever sees them. Every eviction cycle silently drops recent conversation.

```python
# Fix: remove the pre-read trim entirely
# messages_to_summarize = self.context_manager.get_turns(0, num_to_evict - 1)
# self.context_manager.trim(num_to_evict, -1)  ← keep only this, after summarizing
```

### 3. LTM metadata extraction always crashes silently (High)

`MemoryOrchestrator._extract_ltm_metadata()` (`LTM_Manager.py:266`) calls `ModelChat.generate_stream(messages)` as an unbound class method. Python passes `messages` (a list) as `self`. The first line of `generate_stream` runs `self.em.clear_interrupt()`, which raises `AttributeError: 'list' object has no attribute 'em'`. The outer `except Exception` catches it silently. Every memory is stored with empty metadata. The whole extraction pipeline has never run.

### 4. `add_memory_from_summary` reads wrong keys from metadata dict (High)

The same function reads `extracted_data.get("metadata", {})` and `extracted_data.get("tags", [])` from the return value of `_extract_ltm_metadata`, which returns a *flat* dict — no nested `"metadata"` or `"tags"` keys. Even if bug #3 were fixed, every memory would still be stored with empty metadata and no tags.

### Other findings (Medium/Low)

- `SemanticSearchFAISS.delete_item` is not implemented — any memory deletion leaves stale vectors in FAISS indefinitely
- `QueueManager.clear_audio_queue()` does not exist — the interrupt path crashes immediately if ever called
- `handle_interruption` collects merged inputs into a list that is never saved back to `self.pending_input` — interrupted speech is lost
- Singleton `_lock` initialization is not thread-safe — two `EventManager`/`QueueManager` instances can be created under a race
- `_extract_ltm_metadata` calls `self.prompt_builder` without a `None` guard — `setup_data()` creates an orchestrator with no prompt builder, so any call to `add_memory_from_summary` from that path crashes silently
- `discord_bot.process_responses` blocks forever on `Queue.get()` with no timeout — if the main process crashes before the sentinel is sent, the Discord process hangs indefinitely

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         infer_v2.py                         │
│                                                             │
│  Ridi.run()                                                 │
│      └─ AppConfig.init_components()                         │
│              ├─ LocalLlamaBackend(ModelChat(Llama))         │
│              ├─ MemoryOrchestrator (FAISS + SQLite)         │
│              └─ ShortTermMemoryManager (Redis)              │
│                                                             │
│  ConversationManager.__init__(app_context)                  │
│      ├─ MemoryContext(ltm, stm, config)                     │
│      └─ Turn(backend, stm, prompt_builder,                  │
│              memory_context, user_name, log_path)           │
│                                                             │
│  ConversationManager.llm_worker_target()  [thread]          │
│      └─ Turn.run(user_utter)                                │
│              ├─ MemoryContext.context_for(query)            │
│              │       ├─ LTM hybrid RAG search               │
│              │       └─ STM summary window                  │
│              ├─ PromptBuilder.build_final_response_prompt() │
│              ├─ STM.add_turn("user")                        │
│              ├─ LLMBackend.generate_stream(messages)        │
│              └─ STM.add_turn("assistant")                   │
│                      └─ manage_memory_flow()                │
│                              ├─ evict → summarize → LTM     │
│                              └─ archive summary → LTM       │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────┐   ┌──────────────────────────┐
│    AudioProcessor [thread]   │   │  AudioPlayback [thread]  │
│  Silero VAD + Whisper STT    │   │  sounddevice OutputStream│
│  → user_input_queue          │   │  ← audio_queue           │
└──────────────────────────────┘   └──────────────────────────┘

┌──────────────────────────────┐
│    DiscordBot [process]      │
│  discord.py client           │
│  reads user_input_queue      │
│  writes discord_queue        │
└──────────────────────────────┘
```

---

## What's Left

**Immediate (tracked in `docs/todo.md`):**

- Fix the 4 high-severity bugs — the FAISS padding bug and the STM eviction bug are silent data corruption that runs on every session
- Fix `handle_interruption` losing the merged input — interrupted speech should not be discarded
- `MemoryContext` still holds a full `SystemConfig` for just `ltm_rag_similarity_threshold` — reduce it to a plain `float` argument (same pattern as what was done for `Turn`)

**Deferred:**

- `TTSContext.speak(text, audio_queue)` — collapse the three `tts_enabled` guard sites into a single no-op interface
- `SemanticSearchFAISS.delete_item` — implement index rebuild on deletion
- Thread-safe singleton initialization for `EventManager` and `QueueManager`
- `SemiConvManager` still overrides `_loop_step` entirely and never calls `Turn.run()` — migrating Lite mode to use the same turn pipeline requires reconciling its different STM strategy (raw Redis list vs `ShortTermMemoryManager`)

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Speech-to-text | faster-whisper (large-v3, CUDA, float16) |
| Voice activity detection | silero-vad (via torch.hub) |
| Local LLM inference | llama-cpp-python + GGUF (3B, q4_k_m) |
| Cloud LLM (Lite mode) | Google Gemini 2.5 Flash |
| Text-to-speech | GPT-SoVITS (AR token predictor + VITS decoder) |
| Short-term memory | Redis 6.x (localhost:6849) |
| Long-term memory | SQLite FTS5 + FAISS IndexFlatIP |
| Semantic embedding | jhgan/ko-sroberta-multitask (768-dim) |
| Audio I/O | sounddevice |
| Messaging | discord.py |
| Platform | Windows 11, Python 3.10, conda |

---

## Closing Thoughts

RIDI is one of those projects where the ambition of the idea forces you to build systems you didn't originally plan to. A voice-reactive AI that remembers across sessions requires you to solve speech detection, real-time transcription, LLM serving, voice synthesis, interrupt handling, and memory architecture — all at once, on consumer hardware, in a single Python process.

The refactoring session documented here took an untestable 40-line monolith and carved it into three injectable seams in a few hours. The result is `Turn`, `MemoryContext`, and `LLMBackend` — each testable in isolation, each with a clear responsibility. The code review pass that followed identified 4 high-severity silent bugs that had been running undetected.

The most interesting engineering challenge ahead is the memory lifecycle: when and what to evict, how to avoid surfacing irrelevant old memories, and how to keep the hybrid RAG score calibrated as the index grows. That's where the real work is.
