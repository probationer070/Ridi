---
Type: upgrade
Date: 2026-06-03
Files Changed:
  - core/engine/Turn.py — Turn.__init__ (replaced config param with user_name/log_path), Turn.run (self.config.user_name→self.user_name, self.config.conversation_log_path→self.log_path)
  - infer_v2.py — ConversationManager.__init__ (expanded Turn constructor call)
Why Changed: >
  Turn.__init__ accepted the full SystemConfig but only read two fields from it:
  user_name and conversation_log_path. Any unit test of Turn.run() had to construct
  or mock a full SystemConfig to avoid writing to the real log file, defeating the
  testability goal of the earlier refactor. Identified during grill session 2026-06-03.
Contents Diff: |
  # Before — Turn.__init__
  def __init__(self, backend, stm_manager, prompt_builder, memory_context, config):
      self.config = config

  # After — Turn.__init__
  def __init__(self, backend, stm_manager, prompt_builder, memory_context,
               user_name: str, log_path: str):
      self.user_name = user_name
      self.log_path = log_path

  # Before — ConversationManager.__init__
  self.turn = Turn(..., self.config)

  # After — ConversationManager.__init__
  self.turn = Turn(..., user_name=self.config.user_name,
                   log_path=self.config.conversation_log_path)
Improvements: >
  Turn now has zero dependency on any config object. A unit test can construct
  Turn(backend=FakeBackend(), ..., user_name="test", log_path="/dev/null") without
  importing SystemConfig at all.
Performance Impact: none
Agents Consulted: refactoring
Findings Addressed: none
Findings Deferred: >
  MemoryContext has the same problem — holds full SystemConfig but only reads
  ltm_rag_similarity_threshold. Tracked in docs/todo.md under Deferred.
---
