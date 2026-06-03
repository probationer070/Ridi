# Code Review Findings

Reviewed: 2026-06-03 | Scope: full codebase (initial commit) | Effort: high

---

## 1. FAISS -1 padding returns wrong memory in every under-full RAG query

**File:** `core/memory/ltm/semantic_search_faiss.py:213`
**Severity:** High

```python
# current — WRONG
if idx < len(self.metadata_list):
    results.append({"metadata": self.metadata_list[idx], ...})
```

When the LTM index has fewer entries than the requested `top_k`, FAISS pads the result array with `idx = -1`. Python's negative indexing makes `-1 < len(list)` evaluate to `True`, so `self.metadata_list[-1]` (the last stored memory) is silently injected into every under-full search result.

**Fix:**
```python
if idx != -1 and idx < len(self.metadata_list):
    results.append({"metadata": self.metadata_list[idx], ...})
```

---

## 2. STM eviction permanently loses the newest conversation turn on every cycle

**File:** `core/memory/stm/STM_Manager.py:398–399`
**Severity:** High

```python
# current — WRONG
if self.context_manager.exists():
    self.context_manager.trim(0, -num_to_evict)   # deletes last (N-1) items — never summarized
messages_to_summarize = self.context_manager.get_turns(0, num_to_evict - 1)
...
self.context_manager.trim(num_to_evict, -1)
```

`ltrim key 0 -N` removes the last `N-1` elements from the Redis list. Those items are deleted before they are read, so they are never summarized or archived to LTM. With `num_to_evict = 2`, one recent message is silently dropped on every eviction cycle.

**Fix:** remove the pre-read trim entirely. The post-summarization trim already handles cleanup.

```python
# fixed
messages_to_summarize = self.context_manager.get_turns(0, num_to_evict - 1)
if not messages_to_summarize:
    ...
    return
summary = self._summarize_turns(messages_to_summarize, purpose='context')
...
self._add_summary_to_window(summary)
self.context_manager.trim(num_to_evict, -1)
```

---

## 3. `ModelChat.generate_stream` called as unbound class method — LTM metadata extraction is always broken

**File:** `core/memory/ltm/LTM_Manager.py:266`
**Severity:** High

```python
# current — WRONG
extracted_info_str = ModelChat.generate_stream(messages)
```

`generate_stream(self, messages)` is an instance method. Calling it on the class passes `messages` (a list) as `self`. The first line of the method, `if self.em: self.em.clear_interrupt()`, immediately raises `AttributeError: 'list' object has no attribute 'em'`. The outer `except Exception` catches it silently, so `_extract_ltm_metadata` always returns `{}` and the metadata pipeline never runs.

`MemoryOrchestrator` needs a `ModelChat` instance. Add one in `__init__`:

```python
# in MemoryOrchestrator.__init__
if llm_instance:
    self.model_chat = ModelChat(llm_instance)
else:
    self.model_chat = None
```

```python
# fixed call
if not self.model_chat:
    return {}
response_gen = self.model_chat.generate_stream(messages)
extracted_info_str = "".join(response_gen)
```

---

## 4. `add_memory_from_summary` reads wrong keys from `_extract_ltm_metadata` return value

**File:** `core/memory/ltm/LTM_Manager.py:304–306`
**Severity:** High

```python
# _extract_ltm_metadata returns a flat dict, e.g.:
# {"주요_인물": [...], "감정": "...", "extracted_tags": [...]}

# current — WRONG
metadata = extracted_data.get("metadata", {})  # always {} — no nested "metadata" key
tags     = extracted_data.get("tags", [])       # always [] — no "tags" key
```

The return value of `_extract_ltm_metadata` is a flat metadata dict, not a wrapper with `"metadata"` and `"tags"` sub-keys. Every memory is stored with empty metadata and no tags.

**Fix:** read the flat dict directly, and use the already-stored `extracted_tags`:

```python
metadata = extracted_data  # the whole dict is the metadata
tags = extracted_data.pop("extracted_tags", [])
```

---

## 5. `SemanticSearchFAISS.delete_item` does not exist — FAISS index permanently diverges from SQLite on any deletion

**File:** `core/memory/ltm/LTM_Manager.py:389`
**Severity:** Medium

```python
# current — crashes
self.semantic_store.delete_item(item_id)   # AttributeError — method not implemented
```

The TODO comment in the same file acknowledges this. The error is caught and logged, but the item remains in the FAISS index forever. After any deletion, FAISS returns stale items that no longer exist in SQLite.

**Fix:** implement `delete_item` in `SemanticSearchFAISS`. Since `IndexFlatIP` does not support in-place deletion, the standard approach is to rebuild the index from the surviving metadata:

```python
def delete_item(self, item_id: str):
    new_metadata = [m for m in self.metadata_list if m.get("item_id") != item_id]
    if len(new_metadata) == len(self.metadata_list):
        return  # nothing to delete
    self.metadata_list = new_metadata
    # rebuild index from surviving items — requires re-encoding, or store vectors alongside metadata
    self._save_index_and_metadata()
```

---

## 6. `qm.clear_audio_queue()` raises `AttributeError` — interruption path crashes on activation

**File:** `infer_v2.py:252`
**Severity:** Medium

```python
self.qm.clear_audio_queue()  # QueueManager has no such method
```

`QueueManager.__getattr__` strips the `_queue` suffix, looks up `clear_audio` in `_queues`, finds nothing, and raises `AttributeError`. `handle_interruption` is currently unreachable (never called), but any future wiring of the interrupt path will hit this immediately.

**Fix:** add the method to `QueueManager`, or call the existing queue directly:

```python
# in QueueManager
def clear_audio_queue(self):
    q = self._queues.get('audio')
    if q:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except Exception:
                break
```

---

## 7. `handle_interruption` builds merged input list but never saves it — interrupted speech is lost

**File:** `infer_v2.py:255–273`
**Severity:** Medium

```python
new_inputs = []
while not self.qm.user_input_queue.empty():
    ...
    new_inputs.append(content)
if self.pending_input:
    new_inputs.insert(0, self.pending_input)
# new_inputs is never assigned anywhere — silently abandoned
```

`handle_interruption` collects queued inputs and merges them with `pending_input`, but the merged result is never written back to `self.pending_input`. Every word the user spoke during the AI's response is discarded.

**Fix:**
```python
self.pending_input = " ".join(new_inputs).strip()
```

Add this line at the end of the merge block, before the STM delete step.

---

## 8. Singleton `_lock` initialisation is not thread-safe — two `EventManager` / `QueueManager` instances can be created

**File:** `configs/event_manager.py:10–13` (same pattern in `configs/queue_manager.py:21–24`)
**Severity:** Medium

```python
@staticmethod
def get_instance():
    if EventManager._lock is None:            # Thread A reads: True
        EventManager._lock = multiprocessing.RLock()  # Thread B also reads: True → creates second RLock
```

Two threads can both see `_lock is None`, each create a separate `RLock`, and the second write overwrites the first. Thread A then holds `RLock-A` while all other threads use `RLock-B`, breaking the mutual exclusion guarantee and potentially creating two instances.

**Fix:** use a module-level lock created at import time:

```python
_INIT_LOCK = multiprocessing.RLock()

class EventManager:
    _instance = None

    @staticmethod
    def get_instance():
        with _INIT_LOCK:
            if EventManager._instance is None:
                EventManager._instance = EventManager()
        return EventManager._instance
```

---

## 9. `_extract_ltm_metadata` uses `self.prompt_builder` without a `None` guard

**File:** `core/memory/ltm/LTM_Manager.py:261`
**Severity:** Low

```python
messages = self.prompt_builder.build_ltm_metadata_extraction_prompt(summary)
```

`MemoryOrchestrator.setup_data()` instantiates the orchestrator without a `prompt_builder` (parameter defaults to `None`). Any downstream call to `add_memory_from_summary` then crashes with `AttributeError: 'NoneType' object has no attribute 'build_ltm_metadata_extraction_prompt'`, swallowed by the outer except.

**Fix:** guard at the top of `_extract_ltm_metadata`:

```python
if not self.prompt_builder or not self.llm:
    return {}
```

---

## 10. `discord_bot.process_responses` blocks indefinitely if sentinel is never received

**File:** `core/tools/discord_bot.py:58`
**Severity:** Low

```python
response_data = await loop.run_in_executor(None, self.response_queue.get)
```

`multiprocessing.Queue.get()` with no timeout blocks the executor thread forever. If the main process crashes before `ShutdownAct` puts `None` on the discord response queue, the Discord bot process hangs indefinitely, keeping the WebSocket connection and file handles open.

**Fix:** use a timeout and re-check the exit event:

```python
import queue as _queue
try:
    response_data = await loop.run_in_executor(
        None, lambda: self.response_queue.get(timeout=1)
    )
except _queue.Empty:
    continue
```
