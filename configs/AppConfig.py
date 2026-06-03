from llama_cpp import Llama
import os
import traceback

from configs import *
from core.tts.TTS_infer_pack.TTS import TTS
from core.engine.prompt_builder import PromptBuilder
from core.memory.ltm.LTM_Manager import MemoryOrchestrator
from core.memory.stm.STM_Manager import ShortTermMemoryManager
from core.engine.ModelChat import ModelChat
from core.engine.LLMBackend import LocalLlamaBackend
from core.tools.modeltools.FunctionSchema import TOOL_REGISTRY, TOOLS_SCHEMA
from utils2.debug_logger import setup_logger

# SystemConfig 인스턴스를 통해 경로 및 설정값 관리
class AppConfig:
    def __init__(self):
        self.config: SystemConfig = SystemConfig()
        
        # 자동 응답 모드 설정 (기본값: True) - SystemConfig에 없을 경우 대비
        if not hasattr(self.config, 'auto_continue_enabled'):
            self.config.auto_continue_enabled = True

        self.qm: QueueManager = QueueManager.get_instance()
        self.em: EventManager = EventManager.get_instance() # EventManager는 여기서 초기화해도 안전합니다.
        
        self.llm: Llama = None
        self.tts_pipeline: TTS = None
        self.prompt_builder: PromptBuilder = None
        self.embed_model = None
        self.memory_orchestrator: MemoryOrchestrator = None     # LTM_Manager 추가
        self.stm_manager: ShortTermMemoryManager = None         # STM_Manager 추가
        self.tools = None                                       # LLM에 제공할 도구 스키마
        self.model_chat: ModelChat = None                       # ModelChat 인스턴스 추가
        self.llm_backend = None                                 # LLMBackend 인터페이스 (Turn에 주입)
        self.tool_registry = None                               # 실제 실행할 함수 레지스트리
        self.debug_logger = None                                # LLM 입출력 디버그 로거

    def init_components(self):
        """initialze all components"""
        self._init_llm()
        self._init_tools()

        if self.config.tts_enabled:
            self.tts_pipeline = self.load_tts_pipeline(self.config)

        # 임베딩 모델 로드
        self._init_embed()

        # LLM 입출력 디버그 로거 설정
        self.debug_logger = setup_logger(
            log_path=self.config.debug_log_path,
            # log_path="data/logs/total_io_debug.log",
            enabled=True,
            mode="llm_only")

        # ModelChat 인스턴스화 (LLM 로드 후)
        try:
            self.model_chat = ModelChat(self.llm, self.config, self.em, self.debug_logger)
            self.llm_backend = LocalLlamaBackend(self.model_chat)
        except Exception as e:
            print(f"[ModelChat Error] ModelChat initialization failed: {e}")
            self.model_chat = None
            self.llm_backend = None

        self._init_LTMManager()
        self._init_STMManager() # STM Manager 초기화 호출

    def _init_llm(self):
        """initialize LLM model & PromptBuilder"""
        try:
            # --- 필요한 디렉토리 생성 ---
            os.makedirs(self.config.logs_dir, exist_ok=True)
            os.makedirs(self.config.ltm_data_dir, exist_ok=True)

            # PromptBuilder는 의존성이 적으므로 가장 먼저 초기화하여 안전성 확보
            self.prompt_builder = PromptBuilder()

            self.llm = Llama(model_path=self.config.model_path, 
                            n_gpu_layers=-1, 
                            verbose=False, 
                            n_ctx=self.config.model_n_ctx,
                            chat_format="llama-3")        
        except Exception as e:
            print(f"[LLM Error] LLM model loading failed: {e}")
            return
        print("[LLM] LLM model loading complete.")

    def _init_tools(self):
        """initialize tools for function calling"""
        self.tools = TOOLS_SCHEMA
        self.tool_registry = TOOL_REGISTRY
        print(f"[Tools] Function calling tools: {len(self.tools)}")

    def _init_embed(self):
        """initialize shared Embedding Model for LTM"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.config.embedding_model)
            print("[Embedding] Shared EmbeddingModel loaded.")
        except Exception as e:
            print(f"[Embedding Error] Shared EmbeddingModel loading failed: {e}")
            self.embed_model = None

    def _init_STMManager(self):
        """initialize Short Term Memory Manager (STM)"""
        if self.llm and self.memory_orchestrator:
            try:
                self.stm_manager = ShortTermMemoryManager(
                    llm_instance=self.llm,
                    prompt_builder_instance=self.prompt_builder,
                    config=self.config,
                    memory_orchestrator=self.memory_orchestrator,
                    user_id=self.config.user_name,
                    mode="original",
                    max_stm_conversations=4,
                    max_summary_window_size=4
                )
                print("[STM] ShortTermMemoryManager Loaded.")
            except Exception as e:
                print(f"[STM Error] ShortTermMemoryManager reset Failed: {e}")
                self.stm_manager = None
        else:
            print("[STM] LLM or MemoryOrchestrator Load failed: ShortTermMemoryManager Deactivated.")

    def _init_LTMManager(self):
        """initialize Long Term Memory Manager (LTM)"""
        ltm_data_files_exist = os.path.exists(self.config.sqlite_db_path) and \
                               os.path.exists(self.config.faiss_index_path_ltm) and \
                               os.path.exists(self.config.faiss_metadata_path_ltm)
        if not ltm_data_files_exist:
            print(f"[Warning] LTM data files not found. MemoryOrchestrator will attempt to create them. If issues persist, run 'python setup_data.py'. LTM 기능이 제한될 수 있습니다.")
        
        if self.embed_model:
            try:
                # LLM 인스턴스를 키워드 추출에 사용할지 여부에 따라 전달
                # ltm_keyword_config의 'strategy' 설정에 따라 결정합니다.
                llm_for_keywords = None
                if self.config.ltm_keyword_config.get("strategy") == "llm":
                    if self.llm:
                        llm_for_keywords = self.llm
                        print("[Info] LTM keyword Extraction using LLM is enabled.")
                    else:
                        print("[Warning] LTM keyword Extraction using LLM is enabled, but LLM is not loaded.")

                # MemoryOrchestrator 인스턴스화
                self.memory_orchestrator = MemoryOrchestrator(
                    config=self.config,
                    user_id=self.config.user_name,
                    embedding_model_name_or_instance=self.embed_model,
                    embedding_dim=self.embed_model.get_sentence_embedding_dimension(),
                    prompt_builder=self.prompt_builder,
                    llm_instance=llm_for_keywords
                )
                print("[Memory] MemoryOrchestrator reset successfully.")
            except Exception as e:
                print(f"[Memory Error] MemoryOrchestrator reset failed: {e}. MemoryOrchestrator disabled.")
                traceback.print_exc() # 디버깅을 위해 스택 트레이스 출력
        else:
            print("[Memory] failed to load shared EmbeddingModel: MemoryOrchestrator deactivated.")
    
    def check_components_ready(self) -> bool:
        """
        Check if all critical components are properly initialized.
        :return: True if all components are initialized, False otherwise.
        """
        components = {
            "LLM": self.llm,
            "PromptBuilder": self.prompt_builder,
            "LTMManager": self.memory_orchestrator,
            "STMManager": self.stm_manager,
            "ModelChat": self.model_chat
        }
        for name, component in components.items():
            if component is None:
                print(f"[Error] {name} is not properly initialized.")
                return False
        
        print("[Info] All components are properly initialized.")
        return True

    def load_tts_pipeline(self, config: SystemConfig):
        """Load TTS pipeline using GPT_SoVITS."""
        from core.tts.TTS_infer_pack.TTS import TTS_Config
        config_path = getattr(config, 'gpt_sovits_config_path', None)
        gpt_model_path = getattr(config, 'gpt_sovits_gpt_path', None)
        sovits_model_path = getattr(config, 'gpt_sovits_sovits_path', None)

        if not all([config_path, gpt_model_path, sovits_model_path]):
            print("[TTS Error] TTS model or configuration file paths are not defined in SystemConfig. TTS disabled. Check 'gpt_sovits_config_path', 'gpt_sovits_gpt_path', and 'gpt_sovits_sovits_path'.")
            return None
        if not all([os.path.exists(p) for p in [config_path, gpt_model_path, sovits_model_path]]):
            print("[TTS Error] TTS model or configuration file paths are incorrect. TTS disabled. Check 'gpt_sovits_config_path', 'gpt_sovits_gpt_path', and 'gpt_sovits_sovits_path'.")
            return None

        tts_config = TTS_Config(config_path)
        tts_pipeline = TTS(tts_config)
        tts_pipeline.init_t2s_weights(gpt_model_path)
        tts_pipeline.init_vits_weights(sovits_model_path)
        print("[TTS] TTS pipeline loading complete.")
        return tts_pipeline