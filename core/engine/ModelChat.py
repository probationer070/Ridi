from llama_cpp import Llama
from typing import List, Dict, Optional, Generator
import traceback
import json

from configs import SystemConfig, EventManager

class ModelChat:
    """
    LLM과의 모든 상호작용을 캡슐화하고 중앙에서 관리하는 클래스.
    생성 파라미터 관리, 스트리밍 응답 처리, 도구 사용 결정을 담당합니다.
    """
    def __init__(self, llm: Llama, config: Optional[SystemConfig] = None, em: Optional[EventManager] = None, debug_logger=None):
        if llm is None:
            raise ValueError("ModelChat initialized with llm=None. Please check if the LLM model loaded successfully.")
        self.llm = llm
        self.config = SystemConfig() if config is None else config
        self.em = em
        self.debug_logger = debug_logger

    def _log_debug(self, title: str, content: object):
        if self.debug_logger and self.config:
            try:
                # JSON 직렬화가 가능한 객체는 예쁘게 출력
                log_content = json.dumps(content, ensure_ascii=False, indent=2)
            except (TypeError, OverflowError):
                # 그렇지 않은 경우 문자열로 변환
                log_content = str(content)
            self.debug_logger.debug(f"--- {title} ---\n{log_content}")

    def _get_default_params(self) -> Dict:
        """SystemConfig에서 기본 생성 파라미터를 가져옵니다."""
        if not self.config:
            raise ValueError("ModelChat requires a SystemConfig instance to determine generation parameters.")

        return {
            "max_tokens": 2048,
            "temperature": 0.85,
            "repeat_penalty": 1.2,
            # "presence_penalty": 0,
            # "frequency_penalty": 0,
            "top_p": 1,
            # "top_k": 3,
            "min_p": 0.07,
            "stop": ["<|eot_id|>"],
        }

    def generate_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """
        메시지 리스트를 받아 스트리밍으로 응답을 생성합니다.
        인터럽트 이벤트를 감지하여 생성을 중단할 수 있습니다.
        """
        # 응답 생성을 시작하기 직전에, 이전 턴에서 발생했을 수 있는 인터럽트 플래그를 초기화합니다.
        # 이렇게 하면 새로운 응답이 의도치 않게 즉시 중단되는 것을 방지할 수 있습니다.
        if self.em:
            # EventManager에 clear_interrupt()와 같은 메서드가 있다고 가정합니다.
            self.em.clear_interrupt()

        params = self._get_default_params()
        params["stream"] = True

        self._log_debug("Streaming Generation Input", messages)

        stream = self.llm.create_chat_completion(
            messages=messages,
            **params
        )

        full_response_for_log = ""
        for chunk in stream:
            if self.em and (self.em.is_interrupt_set() or self.em.is_exit_set()):
                print("\n[ModelChat] Generation stopped by interrupt/exit event.")
                break
            
            content_piece = chunk['choices'][0]['delta'].get("content")
            if content_piece:
                full_response_for_log += content_piece
                yield content_piece
        
        self._log_debug("Streaming Generation Full Output", {"content": full_response_for_log})

    def generate_once(self, messages: List[Dict], **kwargs) -> str:
        """
        메시지 리스트를 받아 단일 응답을 생성합니다. (비-스트리밍)
        요약, 키워드 추출 등 백그라운드 작업에 사용됩니다.
        추가적인 생성 파라미터는 키워드 인자(kwargs)로 전달할 수 있습니다.
        """
        params = self._get_default_params()
        params["stream"] = False
        params.update(kwargs)

        self._log_debug("Single Generation Input", {"messages": messages, "params": params})

        response = self.llm.create_chat_completion(messages=messages, **params)
        
        self._log_debug("Single Generation Output", response)
        return response['choices'][0]['message'].get("content", "")

    def decide_action(self, messages: List[Dict], tools: List[Dict]) -> Optional[Dict]:
        """
        모델의 첫 번째 행동을 결정합니다. 도구를 호출할 수도 있고, 직접 답변할 수도 있습니다.
        LLM 응답의 전체 'message' 객체를 반환하여, 호출 측에서 tool_calls와 content를 모두 확인할 수 있게 합니다.
        """
        # 도구 결정에는 창의성이 필요 없으므로 낮은 온도를 사용합니다.
        params = {
            "temperature": 0.1,
            "max_tokens": 1024, # tool_calls JSON을 담기에 충분한 크기
            "stop": ["<|eot_id|>"],
        }
        self._log_debug("Action Decision Input", {"messages": messages, "tools": tools})
        try:
            response = self.llm.create_chat_completion(messages=messages, tools=tools, tool_choice="auto", **params)
        except Exception as e:
            print(f"[ModelChat Error] Action decision API call failed: {e}")
            self._log_debug("Action Decision Error", {"error": str(e), "traceback": traceback.format_exc()})
            return None
        self._log_debug("Action Decision Output", response)
        message = response['choices'][0]['message']
        return message