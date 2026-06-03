from typing import Optional


class MemoryContext:
    """Assembles the memory dict injected into every LLM turn's prompt.

    Owns all knowledge about RAG thresholds, score filtering, and
    summary formatting so ConversationManager stays ignorant of both
    LTM and STM internals.
    """

    def __init__(self, ltm_manager, stm_manager, config):
        self.ltm_manager = ltm_manager
        self.stm_manager = stm_manager
        self.config = config

    def context_for(self, query: str) -> dict:
        """Return {"memory": str, "summary": str} for the given user utterance."""
        return {
            "memory": self._load_ltm(query),
            "summary": self._load_stm_summary(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_ltm(self, user_utter: str) -> str:
        if not self.ltm_manager:
            return ""

        rag_results = self.ltm_manager.search_memories_for_rag(
            current_user_input=user_utter,
            final_top_k=2,
        )

        relevant = [
            mem.content for mem, score in rag_results
            if score >= self.config.ltm_rag_similarity_threshold
        ]

        if relevant:
            print(f"[Ridi LTM] RAG Result (high similarity {rag_results[0][1]:.4f}).")
            return "\n".join(f"- {content}" for content in relevant)

        print("[Ridi LTM] No available LTM information.")
        return "[]"

    def _load_stm_summary(self) -> str:
        if not self.stm_manager:
            return ""
        return self.stm_manager.get_summary_for_prompt()
