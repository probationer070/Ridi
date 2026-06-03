import redis
import json
import logging
from typing import List, Dict, Optional, Union, Tuple
from llama_cpp import Llama
import redis.exceptions
from .RedisControl import RedisListControl, RedisStringControl

# ModelChat 클래스를 가져옵니다.
from ...engine.ModelChat import ModelChat 
from ...memory.ltm.LTM_Manager import MemoryOrchestrator
from ...engine.prompt_builder import PromptBuilder
from configs import SystemConfig

logger = logging.getLogger(__name__)

class ContextManager(RedisListControl):
    """Manages the conversation context (list of turns) in Redis."""
    def add_turn(self, role: str, content: str):
        turn_data = {"role": role, "content": content}
        self.push(json.dumps(turn_data, ensure_ascii=False))

    def get_turns(self, start: int, end: int) -> List[Dict]:
        """Gets a range of turns and decodes them from JSON."""
        turns_json = self.get_range(start, end)
        return [json.loads(turn) for turn in turns_json]

    def get_all_turns(self) -> List[Dict]:
        """Gets all turns and decodes them from JSON."""
        turns_json = self.get_all()
        return [json.loads(turn) for turn in turns_json]

    def delete_last_turn(self) -> Optional[Dict]:
        """Pops the last turn and decodes it from JSON."""
        last_turn_json = self.pop()
        return json.loads(last_turn_json) if last_turn_json else None

    def trim_turns(self, start: int, end: int):
        """Trims the conversation history."""
        self.trim(start, end)

class SummaryManager(RedisListControl):
    """Manages the conversation summary window in Redis."""
    def add_summary(self, summary: str):
        """Adds a summary to the front of the window."""
        self.push(summary, to_front=True)

    def get_summaries(self) -> List[str]:
        """Gets all summaries from the window."""
        return self.get_all()

class InterruptedUtteranceManager(RedisStringControl):
    """Manages the interrupted utterance (simple string) in Redis."""
    pass

class ShortTermMemoryManager:
    """
    Orchestrates short-term memory operations using ContextManager and SummaryManager.
    Handles summarization and archiving to long-term memory.
    """
    
    # --- Constants for Redis Keys ---
    STM_KEY_TEMPLATE = "stm:{user_id}"
    INTERRUPTED_UTTERANCE_KEY_TEMPLATE = "interrupted_utterance:{user_id}"
    SUMMARY_WINDOW_KEY_TEMPLATE = "summary_window:{user_id}"

    def __init__(self,
                 llm_instance: Llama,
                 prompt_builder_instance: PromptBuilder = PromptBuilder(),
                 config: SystemConfig = SystemConfig(),
                 memory_orchestrator: Optional[MemoryOrchestrator] = None,
                 redis_host: str = 'localhost',
                 redis_port: int = 6849,
                 redis_db: int = 0,
                 user_id: str = "default_user",
                 mode: str = "original",  # 'original', 'mini', or 'semi'
                 max_stm_conversations: int = 5,
                 max_summary_window_size: int = 5,
                 ):
        self.llm = llm_instance
        self.prompt_builder = prompt_builder_instance
        self.config = config
        self.memory_orchestrator = memory_orchestrator
        self.model_chat = ModelChat(self.llm)
        self.user_id = user_id
        self.mode = mode
        
        self.max_stm_conversations = max_stm_conversations
        self.max_stm_messages = max_stm_conversations * 2
        self.max_summary_window_size = max_summary_window_size

        self.redis_client = self._initialize_redis(redis_host, redis_port, redis_db)
        
        if self.redis_client:
            # Initialize managers with the redis client and specific keys
            stm_key = self.STM_KEY_TEMPLATE.format(user_id=self.user_id)
            summary_key = self.SUMMARY_WINDOW_KEY_TEMPLATE.format(user_id=self.user_id)
            interrupted_key = self.INTERRUPTED_UTTERANCE_KEY_TEMPLATE.format(user_id=self.user_id)

            self.context_manager = ContextManager(self.redis_client, stm_key)
            self.summary_manager = SummaryManager(self.redis_client, summary_key)
            self.interrupted_utterance_manager = InterruptedUtteranceManager(self.redis_client, interrupted_key)

            logger.info(f"[STM Manager] Redis connected (User: {self.user_id}). Managers initialized.")
            self._process_previous_session()
        else:
            logger.error("[STM Manager Error] Redis connection failed. STM will be disabled.")
            self.context_manager = None
            self.summary_manager = None
            self.interrupted_utterance_manager = None

    def _initialize_redis(self, host: str, port: int, db: int) -> Optional[redis.Redis]:
        """Initializes and pings the Redis connection."""
        try:
            client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            client.ping()
            return client
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
            logger.error(f"[STM Manager Error] Redis connection failed: {e}.")
            return None

    def _process_previous_session(self):
        """
        Processes data from a previous session, if it exists.
        Archives old data to LTM and then clears it from Redis.
        """
        if not self.context_manager or not self.context_manager.exists():
            return

        logger.info(f"[STM Manager] Previous session data found for key '{self.context_manager.redis_key}'. Processing...")
        try:
            archived_successfully = self._reboot_session()
            if not archived_successfully:
                logger.warning("[STM Manager] Failed to archive previous session. Data will be preserved for next attempt.")
        except Exception as e:
            logger.error(f"[STM Manager Error] Failed to process previous session data: {e}. Data will be preserved.", exc_info=True)

    def _reboot_session(self) -> bool:
        """
        Reads previous session data, summarizes it for LTM,
        archives it, and then clears the old data.
        Returns True on success, False on failure.
        """
        old_messages = self.context_manager.get_all_turns()
        if not old_messages:
            logger.info("[STM Manager] No messages to archive from previous session.")
            self._clear_session()
            return True

        summary_result = self._summarize_turns(old_messages, purpose='memory')
        
        if summary_result:
            summary, original_text = summary_result
            source_info = f"previous session Archive (User: {self.user_id})"
            self._archive_summary_to_ltm(summary, original_text, source_info)
            # Add the summary of the previous session to the current summary window
            self._add_summary_to_window(summary)    # TODO: 필요할까?
            self._clear_session()
            return True
        else:
            logger.error("[STM Manager Error] Failed to summarize previous session data for archiving.")
            return False

    def _clear_session(self):
        """Clears all STM-related data for the current user."""
        if not self.redis_client:
            return
        
        deleted_count = 0
        if self.context_manager:
            deleted_count += self.context_manager.clear_all()
        if self.summary_manager:
            deleted_count += self.summary_manager.clear_all()
        if self.interrupted_utterance_manager:
            deleted_count += self.interrupted_utterance_manager.clear_all()

        if deleted_count > 0:
            logger.info(f"[STM Manager] Cleared {deleted_count} keys from previous session. New session initialized.")

    def _archive_summary_to_ltm(self, summary: str, original_text: str, source_description: str):
        """Archives a summary to the Long-Term Memory."""
        if not self.memory_orchestrator:
            logger.warning("[STM Manager] MemoryOrchestrator not available. Skipping LTM archiving.")
            return
        try:
            self.memory_orchestrator.add_memory_from_summary(
                summary=summary, 
                original_text=original_text, 
                source_description=source_description # tags는 MemoryOrchestrator에서 처리
            )
            logger.info(f"[STM Manager] Summary archived to LTM: '{summary[:30]}...'")
        except Exception as e:
            logger.error(f"[STM Manager Error] Failed to archive summary to LTM: {e}", exc_info=True)

    def _summarize_turns(self, turns_to_summarize: List[Dict], purpose: str = 'context') -> Optional[Union[str, Tuple[str, str]]]:
        """Summarizes conversation turns using the LLM."""
        if not turns_to_summarize or not self.llm:
            return None
        
        # Handle both list of dicts (from STM) and a single string (from summary window)
        if isinstance(turns_to_summarize, list):
            conversation_text = "\n".join(
                f"{self.user_id if turn.get('role') == 'user' else '리디'}: {turn.get('content', '')}"
                for turn in turns_to_summarize
            )
        elif isinstance(turns_to_summarize, str):
            conversation_text = turns_to_summarize
        else:
            logger.error(f"Unsupported type for summarization: {type(turns_to_summarize)}")
            return None

        try:
            if purpose == 'memory':
                messages = self.prompt_builder.build_memory_creation_prompt(conversation_text)
                log_msg = "Generating detailed summary for LTM storage."
            else:  # 'context'
                messages = self.prompt_builder.build_summary_prompt(conversation_text)
                log_msg = "Generating concise summary for conversation context."
            logger.info(f"[STM Manager] {log_msg}")

            generation_config = {"max_tokens": 300, "temperature": 0.7}
            summary_text = self.model_chat.generate_once(messages=messages, **generation_config)

            if not summary_text:
                logger.warning("[STM Manager] LLM returned an empty summary.")
                return None

            stripped_summary = summary_text.strip()
            logger.info(f"[STM Manager] Summary generated: '{stripped_summary[:50]}...'")
            
            if purpose == 'memory':
                return (stripped_summary, conversation_text)
            return stripped_summary
        except Exception as e:
            logger.error(f"[STM Manager Error] LLM summary generation failed: {e}", exc_info=True)
            return None

    def _add_summary_to_window(self, summary: str):
        """Adds a summary to the summary window."""
        if not self.summary_manager or not summary:
            return
        try:
            self.summary_manager.add_summary(summary)
            current_size = self.summary_manager.get_len()
            logger.info(f"[STM Manager] Summary window updated. Size: {current_size}/{self.max_summary_window_size}")
        except Exception as e:
            logger.error(f"[STM Manager Error] Summary window update failed: {e}", exc_info=True)

    def add_turn(self, role: str, content: str):
        """Adds a conversation turn to STM."""
        if not self.context_manager:
            return

        try:
            # Merge with interrupted utterance if it exists
            if role == 'user' and self.interrupted_utterance_manager.exists():
                interrupted_content = self.interrupted_utterance_manager.get()
                if interrupted_content:
                    merged_content = f"{interrupted_content} {content}"
                    logger.info(f"[STM Manager] Merging interrupted utterance: '{interrupted_content}' + '{content}' -> '{merged_content}'")
                    content = merged_content
                    self.interrupted_utterance_manager.clear_all()

            self.context_manager.add_turn(role, content)
            logger.info(f"[STM Manager] Turn added to STM: Role={role}, Content='{content[:30]}...'")

            # Clean up any lingering interrupted utterance if an assistant turn is added
            if role == 'assistant' and self.interrupted_utterance_manager.exists():
                self.interrupted_utterance_manager.clear_all()
                logger.warning("[STM Manager] Cleared pending interrupted utterance as assistant responded.")

            # Mode-specific memory management
            if role == 'assistant':
                if self.mode == 'mini':
                    # 'mini' 모드: 매 턴마다 마지막 대화 쌍을 요약하여 요약 창에 추가합니다.
                    self._summarize_last_turn_pair()
                elif self.mode == 'semi':
                    self.manage_memory_flow() # semi 모드는 manage_memory_flow에서만 처리
                elif self.mode == 'original':
                    self.manage_memory_flow()

        except Exception as e:
            logger.error(f"[STM Manager Error] Failed to add turn: {e}", exc_info=True)

    def _summarize_last_turn_pair(self):
        """Fetches the last user-assistant turn pair and summarizes it."""
        if not self.context_manager:
            return
        last_two_turns = self.context_manager.get_turns(-2, -1)
        # 'mini' 모드는 매 턴 요약을 생성하여 요약 창에 누적합니다.
        if len(last_two_turns) == 2:
            # This method is now only for 'mini' mode's turn-by-turn summary.
            summary = self._summarize_turns(last_two_turns, purpose='context')
            if summary:
                self._add_summary_to_window(summary)
                logger.info(f"[STM Manager][Mode: mini] Last turn pair summarized and added to summary window.")
 
    def get_summary_for_prompt(self) -> str:
        """Formats the summary window content for an LLM prompt."""
        if not self.summary_manager:
            return ""
        try:
            summaries = self.summary_manager.get_summaries()
            if not summaries:
                return ""
            
            formatted_summaries = "\n".join(f"- {s}" for s in summaries)
            return f"이전 대화들:\n{formatted_summaries}"
        except Exception as e:
            logger.error(f"[STM Manager Error] Failed to load summaries for prompt: {e}", exc_info=True)
            return ""

    def get_short_term_memory(self, 
                              num_recent_conversations: Optional[int] = None, 
                              get_all: bool = False) -> List[Dict]:
        """Fetches recent conversation history from STM."""
        if not self.context_manager:
            return []
        try:
            if get_all:
                return self.context_manager.get_all_turns()
            
            num_messages = self.max_stm_messages
            if num_recent_conversations is not None:
                num_messages = num_recent_conversations * 2
            
            return self.context_manager.get_turns(-num_messages, -1)
        except Exception as e:
            logger.error(f"[STM Manager Error] Failed to load STM: {e}", exc_info=True)
            return []

    def get_formatted_conversations(self, num_recent_to_include: Optional[int] = None, get_all: bool = False) -> List[Dict[str, str]]:
        """Returns conversation history formatted for LLM prompts."""
        if self.mode in ['mini', 'semi'] and not get_all:
            logger.info("[STM Manager][Mode: mini] Skipping recent conversations for prompt optimization.")
            return []

        stm_data = self.get_short_term_memory(num_recent_conversations=num_recent_to_include, get_all=get_all)
        return [{"role": msg.get("role"), "content": msg.get("content")} for msg in stm_data]

    def calculate_messages_to_evict(self, current_length: int) -> int:
        """Calculates how many messages to evict to stay within the STM size limit."""
        if current_length <= self.max_stm_messages:
            return 0
        
        num_to_evict = current_length - self.max_stm_messages
        if num_to_evict % 2 != 0:
            num_to_evict += 1
        
        return min(num_to_evict, current_length)

    def manage_memory_flow(self):
        """
        Manages the entire memory lifecycle in two steps:
        1. Archives the summary window to LTM if it's full.
        2. Evicts old turns from STM to the summary window if STM is full.
        """
        try:
            if self.mode == 'semi':
                # 'semi' 모드: STM이 꽉 찼는지 확인하고, 꽉 찼을 때만 전체를 요약하고 초기화합니다.
                # 대화가 5턴 미만일 경우 아무 작업도 수행하지 않습니다.
                if self.context_manager and self.context_manager.get_len() >= self.max_stm_messages:
                    self._evict_and_summarize_for_semi_mode()
            elif self.mode == 'mini':
                # 'mini' 모드: 요약 창(Summary Window)이 꽉 찼는지 확인하고, LTM으로 보관할지 결정합니다.
                # Step 1: Archive summary window to LTM if needed.
                self._archive_summary_window_if_needed()
            elif self.mode == 'original':
                # Step 1: Archive summary window to LTM if needed.
                self._archive_summary_window_if_needed()

                # Step 2: Evict old turns from STM to the summary window.
                self._evict_turns_to_summary_window()


        except Exception as e:
            logger.error(f"[STM Manager Error] Failed during memory flow management: {e}", exc_info=True)


    def _evict_turns_to_summary_window(self):
        """
        If STM size is exceeded, evicts the oldest turn, summarizes it,
        and adds the summary to the summary window.
        """
        if not self.context_manager:
            return
        try:
            current_length = self.context_manager.get_len()
            num_to_evict = self.calculate_messages_to_evict(current_length)

            if num_to_evict <= 0:
                return

            logger.info(f"[STM Manager] STM size ({current_length}/{self.max_stm_messages}) exceeded. Evicting {num_to_evict} oldest messages to summary window.")

            # Get the oldest messages to be evicted
            # 시스템 메시지는 요약하지 않도록 제외
            if self.context_manager.exists():
                self.context_manager.trim(0, -num_to_evict)
            messages_to_summarize = self.context_manager.get_turns(0, num_to_evict - 1)
            if not messages_to_summarize:
                logger.warning(f"[STM Manager] No messages found to evict (requested: {num_to_evict}).")
                return

            # Summarize the evicted messages
            summary = self._summarize_turns(messages_to_summarize, purpose='context')
            if not summary:
                logger.warning("[STM Manager] Summary generation for evicted turn failed. Postponing eviction.")
                return

            # Add summary to the window and trim the original messages from STM
            self._add_summary_to_window(summary)
            self.context_manager.trim(num_to_evict, -1)
            logger.info(f"[STM Manager] Evicted {num_to_evict} messages from STM and added summary to window.")

        except Exception as e:
            logger.error(f"[STM Manager Error] Failed during STM eviction process: {e}", exc_info=True)
    
    # TODO: semi 모드관련 로직 체크 필요
    def _evict_and_summarize_for_semi_mode(self):
        """
        For 'semi' mode: if STM is full, summarize ALL turns, archive to LTM,
        replace summary window with this new summary, and clear STM.
        """
        if not self.context_manager or self.mode != 'semi':
            return

        logger.info(f"[STM Manager][Mode: semi] STM is full. Processing all turns for summarization, archiving, and reset.")

        all_turns = self.context_manager.get_all_turns()
        if not all_turns:
            logger.warning("[STM Manager][Mode: semi] STM is full but no turns found to process. Clearing STM.")
            self.context_manager.clear_all()
            return

        # 1. STM에 있는 모든 대화(예: 5턴)를 가져와 하나의 요약문으로 만듭니다.
        summary_result = self._summarize_turns(all_turns, purpose='memory')
        if not summary_result:
            logger.error("[STM Manager][Mode: semi] Failed to summarize all turns. Postponing operation.")
            return

        summary_for_ltm, original_text = summary_result
        
        # 2. 생성된 요약문을 LTM(장기기억)에 보관합니다.
        self._archive_summary_to_ltm(summary_for_ltm, original_text, f"Archived from semi-mode STM (User: {self.user_id})")

        # 3. 다음 5턴을 위해 대화 기록(STM)과 요약 창(Summary Window)을 모두 비웁니다.
        self.context_manager.clear_all()
        self.summary_manager.clear_all()
        # 4. 방금 생성된 단 하나의 요약문을 요약 창에 추가합니다. 이 요약은 다음 5턴이 채워지기 전까지 유일한 맥락이 됩니다.
        self._add_summary_to_window(summary_for_ltm) # Use the detailed summary for the next context
        logger.info("[STM Manager][Mode: semi] STM cleared and summary window updated with new full summary.")

    def _archive_summary_window_if_needed(self):
        """
        (Internal) If the summary window is full, it re-summarizes the entire window
        and archives it to LTM, then clears the summary window.
        """
        if not self.summary_manager:
            return
        try:
            summary_len = self.summary_manager.get_len()
            # Trigger archiving exactly when the summary window is full.
            if summary_len < self.max_summary_window_size:
                return

            logger.info(f"[STM Manager] Summary window is full ({summary_len}/{self.max_summary_window_size}). Archiving to LTM.")

            all_summaries = self.summary_manager.get_summaries()
            if not all_summaries:
                logger.warning("[STM Manager] Summary window is full but contains no summaries. Clearing it.")
                self.summary_manager.clear_all()
                return

            summaries_as_text = "\n".join(all_summaries)
            ltm_summary_result = self._summarize_turns(summaries_as_text, purpose='memory')

            if not ltm_summary_result:
                logger.error("[STM Manager] Failed to re-summarize summary window for archiving. Postponing operation.")
                return

            final_summary, _ = ltm_summary_result
            self._archive_summary_to_ltm(final_summary, summaries_as_text, f"Archived from Summary Window (User: {self.user_id})")

            # Clear only the summary window after successful archiving.
            self.summary_manager.clear_all()
            logger.info("[STM Manager] Successfully archived summary window to LTM and cleared it.")
        except Exception as e:
            logger.error(f"[STM Manager Error] Failed during summary window archiving process: {e}", exc_info=True)

    def delete_last_turn(self) -> Optional[Dict]:
        """Deletes the most recent turn and saves its content if it's a user utterance."""
        if not self.context_manager or not self.interrupted_utterance_manager:
            return None
        try:
            last_turn = self.context_manager.delete_last_turn()
            if not last_turn:
                return None

            if last_turn.get("role") == "user" and last_turn.get("content"):
                previous_interrupted = self.interrupted_utterance_manager.get() or ""
                new_interrupted_content = f"{previous_interrupted} {last_turn['content']}".strip()
                
                self.interrupted_utterance_manager.set(new_interrupted_content)
                logger.info(f"[STM Manager] Interrupted utterance saved: '{new_interrupted_content}'")
            
            return last_turn
        except Exception as e:
            logger.error(f"[STM Manager Error] Error deleting and saving last turn: {e}", exc_info=True)
            return None

    def print_stm_status(self):
        """Prints the current status of the STM for debugging."""
        if not self.redis_client:
            print("[STM Status] Redis client is not connected.")
            return
        try:
            print(f"\n--- STM Status (User: {self.user_id}) ---")
            print(f"  - Mode: {self.mode}")

            if self.context_manager:
                current_length = self.context_manager.get_len()
                print(f"  - Conversation History ({self.context_manager.redis_key}): {current_length} messages / {self.max_stm_messages} max")
                if current_length > 0:
                    messages = self.context_manager.get_all_turns()
                    for i, msg in enumerate(messages):
                        print(f"    [{i + 1}] {msg.get('role')}: '{msg.get('content', '')[:70]}...'")
                else:
                    print("  - STM is empty.")
            
            if self.summary_manager:
                summary_len = self.summary_manager.get_len()
                print(f"\n  - Summary Window ({self.summary_manager.redis_key}): {summary_len} summaries / {self.max_summary_window_size} max")
                if summary_len > 0:
                    summaries = self.summary_manager.get_summaries()
                    for i, summary in enumerate(summaries):
                        print(f"    [{i+1}] '{summary[:70]}...'")
                else:
                    print("  - Summary window is empty.")

            print("--- End STM Status ---\n")
        except Exception as e:
            print(f"[STM Status Error] Could not print status: {e}")
    
    def get_last_turn(self):
        """Returns the most recent turn without removing it."""
        if not self.context_manager:
            return None
        try:
            turns = self.context_manager.get_turns(-1, -1)
            return turns[0] if turns else None
        except Exception as e:
            logger.error(f"[STM Manager Error] Error fetching last turn: {e}", exc_info=True)
            return None