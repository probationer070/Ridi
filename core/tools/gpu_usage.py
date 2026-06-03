
import logging
import pynvml

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def log_gpu_memory(stage=""):
    """지정된 단계에서 각 GPU의 메모리 사용량을 로깅합니다."""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            logging.info(
                f"[{stage}] GPU {i}: "
                f"Used={info.used/1024**2:.2f}MB, "
                f"Free={info.free/1024**2:.2f}MB, "
                f"Total={info.total/1024**2:.2f}MB"
            )
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as error:
        logging.error(f"Failed to query GPU memory: {error}")
