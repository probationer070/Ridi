# RIDI — Refactoring Roadmap

Last updated: 2026-06-03

## Motivation

`ConversationManager._process_single_turn()` is an untestable monolith: it does LTM
retrieval, prompt assembly, LLM streaming, STM writes, and CSV logging in one block.
Every test path requires a live GPU, Redis, and FAISS. The three tasks below carve out
injectable seams in recommended order.

---

## Task 1 — Extract MemoryContext ✅ done

**File:** `core/memory/MemoryContext.py`  
**Seam:** `MemoryContext.context_for(query: str) → dict`  
**Moved:** body of `ConversationManager.LTMloader()` + `stm.get_summary_for_prompt()` call  
**Changelog:** `docs/changelog/26-06-03 [upgrade] Extract MemoryContext Interface.md`

## Task 2 — Extract Turn module ✅ done

**File:** `core/engine/Turn.py`  
**Seam:** `Turn.run(user_utter: str) → str`  
**Moved:** body of `ConversationManager._process_single_turn()`  
**Changelog:** `docs/changelog/26-06-03 [upgrade] Extract Turn Module.md`

## Task 3 — Extract LLMBackend interface ✅ done

**File:** `core/engine/LLMBackend.py`  
**Seam:** `LLMBackend.generate_stream(messages) → Iterator[str]`  
**Implementations:** `LocalLlamaBackend` (wraps `ModelChat`), `GeminiBackend` (wraps google-genai)  
**Changelog:** `docs/changelog/26-06-03 [upgrade] Extract LLMBackend Interface.md`

---

## Task 4 — Drop `config` from `Turn` (unblocked)

**Problem:** `Turn.__init__` currently takes the full `SystemConfig`. `Turn.run()` only
reads two fields from it: `user_name` (passed to `prompt_builder`) and
`conversation_log_path` (passed to `save_conversation_for_dataset`). Any unit test
of `Turn.run()` must construct or mock a full `SystemConfig` to avoid writing to the
real log file — defeating the testability goal of the refactor.

**Decision (from grill session 2026-06-03):** Drop `config` from `Turn` entirely.
Inject `user_name: str` and `log_path: str` as plain strings. `Turn` then has zero
dependency on any config object.

**Call site change in `ConversationManager.__init__`:**
```python
# Before
self.turn = Turn(app_context.llm_backend, self.stm_manager,
                 self.prompt_builder, self.memory_context, self.config)

# After
self.turn = Turn(
    backend=app_context.llm_backend,
    stm_manager=self.stm_manager,
    prompt_builder=self.prompt_builder,
    memory_context=self.memory_context,
    user_name=self.config.user_name,
    log_path=self.config.conversation_log_path,
)
```

**Files to change:** `core/engine/Turn.py`, `infer_v2.py`  
**Status:** ✅ done — changelog: `docs/changelog/26-06-03 [upgrade] Drop Config from Turn.md`

---

## Deferred

- **Candidate 4 — TTSContext:** `tts_enabled` guard is checked in 3 places. A
  `TTSContext.speak(text, audio_queue)` no-op collapses them. Low priority until
  TTS path needs its own tests.

- **`MemoryContext` config dependency:** `MemoryContext` also holds the full
  `SystemConfig` and reads only `ltm_rag_similarity_threshold`. Same problem as
  `Turn` — could be reduced to a plain `float` argument. Tackle after Task 4.
