from typing import Optional

from core.engine.Summary import save_conversation_for_dataset


class Turn:
    """Executes a single user→AI conversation turn end-to-end.

    Depends only on its constructor arguments so it can be constructed
    and tested without AppConfig, live GPU, or Redis.
    """

    def __init__(self, backend, stm_manager, prompt_builder, memory_context,
                 user_name: str, log_path: str):
        self.backend = backend
        self.stm_manager = stm_manager
        self.prompt_builder = prompt_builder
        self.memory_context = memory_context
        self.user_name = user_name
        self.log_path = log_path

    def run(self, user_utter: str) -> Optional[str]:
        """Process one user utterance and return the full AI response string."""
        print("[Ridi] Final response generation...")

        prompt_context = self.memory_context.context_for(user_utter)
        print(
            f"[Debug] Prompt Context Memory: {prompt_context['memory']},\n"
            f"[Debug] summary: {prompt_context['summary']}"
        )

        final_system_prompt = self.prompt_builder.build_final_response_prompt(
            memory=prompt_context["memory"],
            previous_summary=prompt_context["summary"],
            user_name=self.user_name,
        )

        if self.stm_manager:
            self.stm_manager.add_turn(role="user", content=user_utter)

        previous_turns = (
            self.stm_manager.get_formatted_conversations() if self.stm_manager else []
        )
        final_messages = [{"role": "system", "content": final_system_prompt}, *previous_turns]
        print("[Ridi] Without tool usage response generation...")

        ai_response_full = ""
        print("\n[Ridi AI] ", end="")
        for content_piece in self.backend.generate_stream(final_messages):
            ai_response_full += content_piece
            print(content_piece, end="", flush=True)

        if ai_response_full and ai_response_full.strip():
            if self.stm_manager:
                self.stm_manager.add_turn(role="assistant", content=ai_response_full)
                self.stm_manager.print_stm_status()
                full_history = self.stm_manager.get_formatted_conversations(get_all=True)
                save_conversation_for_dataset(
                    system_prompt=final_system_prompt,
                    conversation_history=full_history,
                    dataset_file_path=self.log_path,
                )
        else:
            print("[Warning] AI did not generate a final response. Moving to the next turn.")

        return ai_response_full
