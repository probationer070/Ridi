import time
from typing import Optional, Callable, List

class SpeechBuffer:
    """
    실시간 음성-텍스트 변환(STT) 결과를 버퍼링하고,
    사용자의 발화가 끝났다고 판단될 때 완전한 문장을 반환하는 클래스.
    """
    def __init__(self, silence_threshold_seconds: float = 1.5, on_speech_end: Optional[Callable[[str], None]] = None):
        """
        Args:
            silence_threshold_seconds (float): 발화 종료로 판단할 침묵 시간 (초).
            on_speech_end (Optional[Callable[[str], None]]): 발화가 끝나면 완성된 문장을 인자로 호출할 콜백 함수.
        """
        self.buffer: List[str] = []
        self.silence_threshold = silence_threshold_seconds
        self.last_input_time: Optional[float] = None
        self.on_speech_end = on_speech_end
        print(f"[SpeechBuffer] Initialized with {silence_threshold_seconds}s silence threshold.")

    def add_transcript_fragment(self, fragment: str):
        """
        STT 엔진으로부터 받은 텍스트 조각을 버퍼에 추가합니다.

        Args:
            fragment (str): STT가 실시간으로 생성한 텍스트 조각.
        """
        if not fragment or not fragment.strip():
            return

        print(f"[SpeechBuffer] Fragment received: '{fragment}'")
        self.buffer.append(fragment.strip())
        self.last_input_time = time.time()

    def check_for_speech_end(self) -> Optional[str]:
        """
        주기적으로 호출되어 발화 종료 여부를 확인합니다.
        발화가 종료되었다면, 버퍼를 비우고 완성된 문장을 반환합니다.
        콜백(on_speech_end)이 지정된 경우, 콜백을 호출합니다.

        Returns:
            Optional[str]: 발화가 종료되었으면 완성된 문장, 아니면 None.
        """
        if self.buffer and self.last_input_time:
            elapsed_time = time.time() - self.last_input_time
            if elapsed_time > self.silence_threshold:
                full_sentence = " ".join(self.buffer).strip()
                print(f"[SpeechBuffer] End of speech detected. Full sentence: '{full_sentence}'")
                
                # 버퍼와 시간 초기화
                self.buffer = []
                self.last_input_time = None
                
                # 콜백 함수가 있다면 완성된 문장으로 호출
                if self.on_speech_end:
                    self.on_speech_end(full_sentence)

                return full_sentence
        return None

    def flush(self) -> Optional[str]:
        """
        버퍼에 남아있는 내용을 강제로 비우고 문장을 반환합니다.
        (예: 프로그램 종료 시, 강제 전송 필요 시)
        """
        if not self.buffer:
            return None
        
        full_sentence = " ".join(self.buffer).strip()
        print(f"[SpeechBuffer] Flushing buffer. Full sentence: '{full_sentence}'")
        self.buffer = []
        self.last_input_time = None

        if self.on_speech_end:
            self.on_speech_end(full_sentence)
            
        return full_sentence

    def is_listening(self) -> bool:
        """
        현재 버퍼에 내용이 있고, 발화가 진행 중인지 여부를 반환합니다.
        """
        return bool(self.buffer)


# 이 클래스를 실제 애플리케이션의 음성 인식 처리 루프에 통합하여 사용하세요.
# 예를 들어, STT 라이브러리가 텍스트 조각을 반환할 때마다 `add_transcript_fragment`를 호출하고,
# 메인 루프에서 주기적으로 `check_for_speech_end`를 호출하여 완성된 문장을 받아 처리합니다.