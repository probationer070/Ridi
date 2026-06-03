import multiprocessing
import threading

class EventManager:
    # 이벤트를 관리하는 싱글톤 클래스
    _instance = None
    _lock = None # Lock을 나중에 생성하도록 None으로 초기화 (RLock으로 변경)

    @staticmethod
    def get_instance():
        if EventManager._lock is None:
            EventManager._lock = multiprocessing.RLock() # Reentrant Lock 사용
        if EventManager._instance is None:
            with EventManager._lock:
                if EventManager._instance is None:
                    EventManager._instance = EventManager()
        return EventManager._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        with self._lock:
            if hasattr(self, '_initialized'):
                return
            
            # 프로세스 간 공유가 필요한 이벤트만 multiprocessing.Event 사용
            self.exit_event = multiprocessing.Event()
            # 스레드 간 공유만 필요한 이벤트는 threading.Event 사용
            self.interrupt_event = threading.Event()
            self.llm_tts_idle_event = threading.Event()
            
            self.llm_tts_idle_event.set() # 초기 상태는 '유휴'이므로 set()
            self._initialized = True # 초기화 상태 설정
            print("Process-safe EventManager instance created.")

    # Set methods for events
    def set_interrupt(self):
        self.interrupt_event.set()

    def set_exit(self):
        self.exit_event.set()

    # Clear methods for events
    def clear_interrupt(self):
        self.interrupt_event.clear()

    # Checking methods for events
    def is_interrupt_set(self):
        return self.interrupt_event.is_set()

    def is_exit_set(self):
        return self.exit_event.is_set()