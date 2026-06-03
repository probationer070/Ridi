---
Type: upgrade
Date: 2026-06-03
Files Changed:
  - core/engine/LLMBackend.py — new file: LLMBackend ABC, LocalLlamaBackend, GeminiBackend
  - configs/AppConfig.py — AppConfig.__init__ (added llm_backend field), AppConfig.init_components (wires LocalLlamaBackend)
  - conversation.py — SemiAppConfig.init_components (wires GeminiBackend), added GeminiBackend import
  - core/engine/Turn.py — Turn.__init__ (model_chat→backend), Turn.run (self.model_chat→self.backend)
  - infer_v2.py — ConversationManager.__init__ (removed self.model_chat, passes app_context.llm_backend to Turn)
Why Changed: >
  Swapping Full mode (llama-cpp) for Lite mode (Gemini) required overriding a 200-line
  AppConfig.init_components() subclass. With LLMBackend, the swap is a single
  AppConfig.llm_backend assignment. Turn no longer imports or references ModelChat.
Contents Diff: |
  # Before — Turn.__init__
  def __init__(self, model_chat, ...):
      self.model_chat = model_chat

  # After — Turn.__init__
  def __init__(self, backend, ...):
      self.backend = backend

  # Before — Turn.run
  for content_piece in self.model_chat.generate_stream(final_messages):

  # After — Turn.run
  for content_piece in self.backend.generate_stream(final_messages):

  # Before — ConversationManager.__init__
  self.model_chat = self.app_context.model_chat
  self.turn = Turn(self.model_chat, ...)

  # After — ConversationManager.__init__
  self.turn = Turn(self.app_context.llm_backend, ...)
Improvements: >
  Turn has zero knowledge of llama-cpp or Gemini.
  Full↔Lite swap is one AppConfig field; no subclass override needed for the LLM call.
  GeminiBackend implements the same generate_stream contract so SemiConvManager can
  adopt Turn.run() in a future step without changing Turn itself.
Performance Impact: none
Agents Consulted: refactoring, idea-management
Findings Addressed: none
Findings Deferred: >
  SemiConvManager still overrides _loop_step entirely and does not call Turn.run().
  Migrating Lite-mode turn logic into Turn is deferred — it requires reconciling
  the different STM strategy (raw Redis list vs ShortTermMemoryManager).
---
