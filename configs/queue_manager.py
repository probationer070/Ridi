import queue
import multiprocessing
from typing import Union, Optional

class QueueManager:
    """
    Docstring for QueueManager

    This class manages queues for inter-thread and inter-process communication.

    필요시 프로세스 간 공유가 가능한 큐(multiprocessing.Queue)와
    스레드 간 공유만 가능한 큐(queue.Queue)를 모두 지원합니다.

    싱글톤 패턴으로 구현되어 애플리케이션 전역에서 하나의 인스턴스만 존재합니다.
    
    !!! 중요 !!!
    Queue 용량은 기본적으로 무제한이며, 필요시 register_queue 메서드 호출 시 maxsize 파라미터로 제한할 수 있습니다
    """
    # 큐를 관리하는 싱글톤 클래스
    _instance = None
    _lock = None # Lock을 나중에 생성하도록 None으로 초기화 (RLock으로 변경)
    
    @staticmethod
    def get_instance():
        if QueueManager._lock is None:
            QueueManager._lock = multiprocessing.RLock() # Reentrant Lock 사용
        if QueueManager._instance is None:
            with QueueManager._lock:
                if QueueManager._instance is None:  # 왜 반복함? : 다중 스레드 환경에서 안전하게 싱글톤 생성 보장
                    QueueManager._instance = QueueManager()
        return QueueManager._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        with self._lock:
            if hasattr(self, '_initialized'):
                return
            self._queues = {}
            self._initialized = True
            # 기존 코드 호환성을 위해 기본 큐들을 등록합니다.
            self.register_queue('user_input', process_safe=True) # 프로세스 간 공유 필요 (Discord Bot -> Main)
            self.register_queue('audio', process_safe=False)      # 스레드 간 공유만 필요
            self.register_queue('llm_task', process_safe=False)   # 스레드 간 공유만 필요
            print("[QueueManager] QueueManager instance created and default queues registered.")

    def register_queue(self, name: str, maxsize: int = 0, process_safe: bool = False):
        """
        Registers a new queue with the given name.
        """
        with self._lock:
            if name not in self._queues:
                self._queues[name] = multiprocessing.Queue(maxsize) if process_safe else queue.Queue(maxsize)
                print(f"[QueueManager] Queue '{name}' registered ({'process-safe' if process_safe else 'thread-safe'}).")

    def get_queue(self, name: str) -> Union[queue.Queue, multiprocessing.Queue]:
        q = self._queues.get(name)
        if q is None:
            raise ValueError(f"Queue with name '{name}' is not registered.")
            
        return q

    def __getattr__(self, name: str) -> Union[queue.Queue, multiprocessing.Queue]:
        """Supports attribute access (e.g., qm.user_input_queue)"""
        if name.endswith('_queue'):
            queue_name = name[:-6] # '_queue' 접미사 제거
            if queue_name in self._queues:
                return self.get_queue(queue_name)
        raise AttributeError(f"[QueueManager] '{type(self).__name__}' object has no attribute '{name}'")