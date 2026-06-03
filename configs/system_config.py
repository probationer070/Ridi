import os, sys

# 현재 작업 디렉토리 및 모듈 경로 추가
NOW_DIR = os.getcwd()
sys.path.append(NOW_DIR)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def SettingDir(parent_dir: str, sub_dir: str):
    return os.path.join(parent_dir, sub_dir)

# --- Module-Level Default Configurations ---
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # e:\Project_HC_Victor\hyeonseo_ai_v0

# log & data filepath Setting
_DEFAULT_LOGS_DIR = SettingDir(_PROJECT_ROOT, "data/logs")
_DEFAULT_SUMMARY_FILE_PATH = SettingDir(_DEFAULT_LOGS_DIR, "summary_memory.txt")
_DEFAULT_CONVERSATION_LOG_PATH = SettingDir(_DEFAULT_LOGS_DIR, "Conversation_log.csv")

# Long Term Memory (LTM) path
_DEFAULT_LTM_DATA_DIR = SettingDir(_PROJECT_ROOT, "data/ltm_data")
_DEFAULT_SQLITE_DB_PATH = SettingDir(_DEFAULT_LTM_DATA_DIR, "ltm_storage.sqlite")
_DEFAULT_FAISS_INDEX_PATH_LTM = SettingDir(_DEFAULT_LTM_DATA_DIR, "faiss_ltm_index.idx")
_DEFAULT_FAISS_METADATA_PATH_LTM = SettingDir(_DEFAULT_LTM_DATA_DIR, "faiss_ltm_meta.json")

# MemoryOrchestrator 초기 데이터용
_DEFAULT_LTM_INITIAL_DATA_PATH = SettingDir(_PROJECT_ROOT, "csv_data/summary_main.jsonl") # LTM 초기 데이터 JSONL (예시)
_DEFAULT_LTM_LOGIC_LOG_PATH = SettingDir(_DEFAULT_LOGS_DIR, "ltm_scoring_log.csv")

# TTS Settings
_DEFAULT_GPT_SOVITS_CONFIG_PATH = SettingDir(_PROJECT_ROOT, "core/tts/configs/tts_infer.yaml")
_DEFAULT_GPT_SOVITS_GPT_PATH = SettingDir(_PROJECT_ROOT, "data/voice_model/voice_test_v2-e15.ckpt")
_DEFAULT_GPT_SOVITS_SOVITS_PATH = SettingDir(_PROJECT_ROOT, "data/voice_model/voice_test_v2_e8_s248.pth")

# Reference audio and text for TTS voice cloning
_DEFAULT_REF_AUDIO_PATH = SettingDir(_PROJECT_ROOT, "data/voice/neuro-sama-tts-file.wav")
_DEFAULT_REF_TEXT = "Clone your voice in minutes with our free AI voice cloning."

class SystemConfig:
    """
    AI 시스템의 전반적인 설정을 관리합니다.
    경로, 모델 이름, TTS 관련 설정 등을 포함합니다.
    """
    def __init__(self):
        self.project_root = _PROJECT_ROOT
        self.user_name = "재한"

        # LLM Model Path Setting
        # self.model_path = "./model/ridi-v0.2-3b-q8_0.gguf"
        self.model_path = "./data/model/ridi-v0.2-3b-q4_k_m.gguf"
        self.model_n_ctx = 4096

        # auto continue
        self.auto_continue_enabled = False  # 자동 응답 모드 기본값 설정

        # log & data filepath Setting
        self.logs_dir = _DEFAULT_LOGS_DIR
        self.summary_file_path = _DEFAULT_SUMMARY_FILE_PATH
        self.conversation_log_path = _DEFAULT_CONVERSATION_LOG_PATH

        # Long Term Memory (LTM) path
        self.ltm_data_dir = _DEFAULT_LTM_DATA_DIR
        self.sqlite_db_path = _DEFAULT_SQLITE_DB_PATH
        self.faiss_index_path_ltm = _DEFAULT_FAISS_INDEX_PATH_LTM
        self.faiss_metadata_path_ltm = _DEFAULT_FAISS_METADATA_PATH_LTM
        self.embedding_model = "./data/model/jhgan_ko-sroberta-multitask"
        self.embedding_model_dim = 768
        self.ltm_rag_similarity_threshold = 0.7
        self.ltm_scoring_log_path = _DEFAULT_LTM_LOGIC_LOG_PATH

        """LTM 키워드 추출 전략 및 관련 설정을 중앙에서 관리합니다.
        'strategy' 값을 변경하여 키워드 추출 방식을 손쉽게 전환할 수 있습니다.
        self.ltm_keyword_config = {
            # 사용 가능한 전략:
            # - "simple": 가장 기본적인 공백 기반 분리. 추가 라이브러리 필요 없음.
            # - "okt": 한국어 형태소 분석기(Okt)를 사용. (konlpy 필요)
            # - "llm": 현재 로드된 LLM을 사용하여 키워드 추출. (LLM 성능에 의존)
            "strategy": "simple",

            # 'okt' 전략을 사용할 경우의 상세 설정
            # "okt_config": {
            #     "stem": True,
            #     "norm": True,
            #     "min_len": 2
            # },
            # 'llm' 전략을 사용할 경우의 상세 설정
            # "llm_config": {
            #     "prompt_template": "다음 텍스트에서 핵심 키워드를 5개만 쉼표로 구분하여 추출해줘. 다른 설명은 붙이지 마.\n\n텍스트: {text}\n\n키워드:"
            # },
            # KeywordSearchSQLite에 전달될 추가 설정 (현재 사용되지 않음)
            # "sqlite_config": {}
        }"""

        # LTM 키워드 추출 전략 및 관련 설정을 중앙에서 관리합니다.
        # 'strategy' 값을 변경하여 키워드 추출 방식을 손쉽게 전환할 수 있습니다.
        self.ltm_keyword_config = {
            "strategy": "simple",
        }

        # MemoryOrchestrator 초기 데이터용 (선택적, 만약 초기 메모리를 jsonl에서 로드한다면)
        self.ltm_initial_data_path = _DEFAULT_LTM_INITIAL_DATA_PATH
        
        # STM Settings
        # self.stm_max_conversations: int = 2
        
        # TTS Settings
        self.tts_enabled: bool = True
        self.sample_rate: int = 32000
        self.gpt_sovits_config_path = _DEFAULT_GPT_SOVITS_CONFIG_PATH
        self.gpt_sovits_gpt_path = _DEFAULT_GPT_SOVITS_GPT_PATH
        self.gpt_sovits_sovits_path = _DEFAULT_GPT_SOVITS_SOVITS_PATH
        self.ref_audio_path = _DEFAULT_REF_AUDIO_PATH
        self.ref_text = _DEFAULT_REF_TEXT

        # --- LLM I/O Debug Logging ---
        # LLM 입출력 디버그 로그를 저장할 파일 경로
        self.debug_log_path: str = "llm_io_debug.log"

        # Discord Bot Settings
        self.discord_bot_enabled: bool = True  # Discord 봇 활성화 여부
        self.discord_token = os.environ.get("DISCORD_BOT_TOKEN", "")
        self.target_channel_id = 812327056842686464  # 대화할 서버 채널 ID (정수형으로 입력) Based on RIDI Official Server
