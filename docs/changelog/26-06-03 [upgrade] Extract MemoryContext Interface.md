---
Type: upgrade
Date: 2026-06-03
Files Changed:
  - core/memory/MemoryContext.py — new file: MemoryContext class
  - infer_v2.py — ConversationManager.__init__, ConversationManager._gather_prompt_context, removed ConversationManager.LTMloader
Why Changed: >
  ConversationManager._gather_prompt_context assembled LTM RAG results and STM summary
  inline, meaning all tests of the conversation turn required live Redis and FAISS.
  Extracting a MemoryContext seam makes memory injectable for future testing and
  removes LTM/STM knowledge from ConversationManager.
Contents Diff: |
  # Before — _gather_prompt_context (infer_v2.py)
  def _gather_prompt_context(self, user_utter):
      context = {"memory": "", "summary": ""}
      context["memory"] = self.LTMloader(user_utter)
      if self.stm_manager:
          context["summary"] = self.stm_manager.get_summary_for_prompt()
      return context

  # After — _gather_prompt_context (infer_v2.py)
  def _gather_prompt_context(self, user_utter):
      return self.memory_context.context_for(user_utter)

  # Before — LTMloader (infer_v2.py, ~30 lines, now deleted)
  # Entire method moved verbatim into MemoryContext._load_ltm()
Improvements: >
  ConversationManager no longer imports MemoryItem, List, or Tuple.
  LTM threshold policy, score filtering, and logging live in one place.
Performance Impact: none
Agents Consulted: refactoring, idea-management
Findings Addressed: none
Findings Deferred: none
---
