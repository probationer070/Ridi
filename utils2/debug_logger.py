import logging
import os
import sys
from typing import Literal, Optional


_DEFAULT_LOGS_DIR = "data/logs"

class StreamDuplicator:
    """
    스트림 출력을 원래 스트림(콘솔)과 파일에 동시에 보냅니다.
    print() 구문을 포함한 모든 콘솔 출력을 가로채기 위해 사용됩니다.
    """
    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.flush() # 즉시 파일에 기록되도록 flush 호출

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name):
        # 다른 스트림 속성(예: isatty)에 대한 접근을 위임합니다.
        return getattr(self.stream, name)

def setup_logger(log_path: Optional[str], 
                 enabled: bool = True, 
                 mode: Literal['all', 'llm_only'] = 'llm_only'):
    """
    Sets up a dedicated logger for LLM input/output debugging.
    U Can find file at `data/logs/debug.log` by default.

    Args:
        log_path (Optional[str]): The file path where logs will be saved.
        enabled (bool): Whether to enable logging. If False, returns None.
        mode (Literal['all', 'llm_only']): 
            'all' - Redirects all console output (stdout and stderr) to the log file.
            'llm_only' - Logs only LLM input/output to the log file.
    """
    if not log_path:
        log_path = "debug.log"

    if not os.path.dirname(log_path):
        log_path = os.path.join(_DEFAULT_LOGS_DIR, log_path)

    if not enabled:
        return None

    try:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        if mode == 'all':
            # --- 모든 CMD 출력 로깅 모드 ---
            # 표준 출력(stdout)과 표준 에러(stderr)를 파일로 리디렉션합니다.
            log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
            sys.stdout = StreamDuplicator(sys.__stdout__, log_file)
            sys.stderr = StreamDuplicator(sys.__stderr__, log_file)
            
            # 로깅 모듈도 이 스트림을 사용하도록 설정합니다.
            # 이렇게 하면 logging.info 등도 파일과 콘솔에 모두 기록됩니다.
            logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s', datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)
            
            logging.info("CMD output logger initialized. All console output will be recorded.")
            # 이 모드에서는 특정 로거 객체 대신 None을 반환하거나 루트 로거를 반환할 수 있습니다.
            # 기존 코드와의 호환성을 위해 루트 로거를 반환합니다.
            return logging.getLogger()
        else: # mode == 'llm_only' (기본값)
            # --- LLM 전용 로깅 모드 (기존 방식) ---
            logger = logging.getLogger('LLM_IO_DEBUG')
            logger.setLevel(logging.DEBUG)
            logger.propagate = False

            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info("LLM I/O Debug Logger initialized successfully.")
            return logger
    except Exception as e:
        print(f"[ERROR] Failed to set up logger at {log_path}: {e}")
        return None
