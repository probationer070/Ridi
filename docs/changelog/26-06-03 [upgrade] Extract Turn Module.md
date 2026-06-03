---
Type: upgrade
Date: 2026-06-03
Files Changed:
  - core/engine/Turn.py — new file: Turn class with Turn.run()
  - infer_v2.py — ConversationManager.__init__ (added self.turn), removed ConversationManager._process_single_turn, removed ConversationManager._gather_prompt_context, llm_worker_target now calls self.turn.run()
Why Changed: >
  _process_single_turn did five unrelated things in one 40-line block: LTM/STM context
  assembly, prompt building, STM user-turn write, LLM streaming, STM assistant-turn
  write + CSV log. Every test path through this method required a live GPU, Redis, and
  FAISS. Extracting Turn.run() makes the LLM turn testable via constructor injection
  without AppConfig.
Contents Diff: |
  # Before — llm_worker_target (infer_v2.py)
  ai_response = self._process_single_turn(user_utter)

  # After — llm_worker_target (infer_v2.py)
  ai_response = self.turn.run(user_utter)

  # Removed from infer_v2.py: _process_single_turn (~40 lines), _gather_prompt_context (1 line)
  # Added: core/engine/Turn.py
Improvements: >
  ConversationManager no longer imports save_conversation_for_dataset.
  Turn has no reference to AppConfig — depends only on its five constructor args.
  _gather_prompt_context intermediate method eliminated entirely.
Performance Impact: none
Agents Consulted: refactoring, idea-management
Findings Addressed: none
Findings Deferred: none
---
