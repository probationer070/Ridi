import time
from typing import Optional, Callable, Dict

class SilenceDetector:
    """일정 시간 동안의 침묵을 감지하고 관련 상태를 관리하는 클래스 (Dictionary Dispatch 방식 적용)."""
    
    def __init__(self, silence_threshold: int = 10, auto_response_delay: int = 5):
        self.silence_threshold = silence_threshold      # 침묵 감지 임계값 (초)
        self.auto_response_delay = auto_response_delay  # 침묵 감지 후 실제 응답까지의 지연 시간
        self.start_time: Optional[float] = None
        self.state: str = "IDLE"  # IDLE, DETECTING, PONDERING
        self.last_repeated_input: Optional[str] = None

        # [핵심 변경 사항] 상태(Command)와 실행 함수(Handler)를 매핑하는 Dictionary 정의
        self.state_handlers: Dict[str, Callable[[Callable], Optional[str]]] = {
            "IDLE": self._handle_idle,
            "DETECTING": self._handle_detecting,
            "PONDERING": self._handle_pondering
        }

    def check_silence(self, get_last_utterance_func: Callable[[], Optional[str]]) -> Optional[str]:
        """
        현재 상태에 매핑된 핸들러 함수를 딕셔너리에서 찾아 실행합니다.
        """
        handler = self.state_handlers.get(self.state)
        
        if handler:
            return handler(get_last_utterance_func)
        else:
            print(f"[Error] Unknown state: {self.state}")
            self.reset()
            return None

    # --- 상태별 핸들러 함수 분리 ---

    def _handle_idle(self, _: Callable) -> None:
        """IDLE 상태 처리: 타이머 시작 및 상태 변경"""
        self.start_time = time.time()
        self.state = "DETECTING"
        return None

    def _handle_detecting(self, _: Callable) -> None:
        """DETECTING 상태 처리: 침묵 시간 체크"""
        if time.time() - self.start_time >= self.silence_threshold:
            print(f"\n[Ridi] Silence detected for {self.silence_threshold}s. Pondering for {self.auto_response_delay}s...")
            self.start_time = time.time()  # Pondering 시작 시간으로 타이머 리셋
            self.state = "PONDERING"
        return None

    def _handle_pondering(self, get_last_utterance_func: Callable) -> Optional[str]:
        """PONDERING 상태 처리: 지연 시간 후 입력 생성"""
        if time.time() - self.start_time < self.auto_response_delay:
            return None  # 아직 생각 중

        # 생각하는 시간이 끝났으므로, 입력 생성 시도
        last_turn = get_last_utterance_func()
        if last_turn:
            new_input = f"RIDI: {last_turn}"
            
            # 직전의 자체 입력과 동일하면 반복하지 않음
            if new_input == self.last_repeated_input:
                print("[Ridi] Auto-response loop detected. Halting repetition.")
                self.reset()  # 무한 루프 방지를 위해 리셋
                return "다른 할 말이 있으신가요?"  # 반복을 깨는 새로운 입력 제공

            print(f"\n[Ridi] Auto-continue. Using last response as input: {new_input}")
            self.last_repeated_input = new_input
            self.reset()  # 작업 수행 후 리셋
            return new_input
        else:
            print("[Ridi] Silence detected but no previous response found in memory.")
            self.reset()
            return None

    def reset(self):
        """사용자 입력이 감지되었을 때 호출하여 침묵 감지 상태를 리셋합니다."""
        self.start_time = None
        self.state = "IDLE"
        self.last_repeated_input = None
