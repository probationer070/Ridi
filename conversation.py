# 본 코드는 Lite 코드로 작성되었습니다.
# Google Gemini 무료 API를 사용해 마이크로 인식한 데이터를 처리하고 답변합니다.
# 모든 대화 내용 따로 conversation.txt에 저장됩니다.
# 사용된 TTS, STT, LLM 모델은 무료 버전입니다.

import os
import threading
import json
from google import genai
import redis

from configs import *
from configs.AppConfig import AppConfig
from core.engine.LLMBackend import GeminiBackend
from core.tts.tts import run_sovits_tts
from core.audio.AudioThreads import background_listening_thread, audio_playback_thread
from utils2.debug_logger import setup_logger
from infer_v2 import ConversationManager


# ------------------------------------------------------------------
# 스레드 관련 함수들
# ------------------------------------------------------------------
def start_threads(em):
    """오디오 재생 및 백그라운드 음성 인식 스레드를 시작"""
    playback_thread = threading.Thread(
        target=audio_playback_thread,
        args=(32000, em.exit_event),
        daemon=True
    )
    playback_thread.start()

    bg_thread = threading.Thread(
        target=background_listening_thread,
        args=(em.exit_event,),
        daemon=True
    )
    bg_thread.start()

    return playback_thread, bg_thread

def get_apikey():
    return os.environ.get("GEMINI_API_KEY", "")



class SemiAppConfig(AppConfig):
    def __init__(self):
        super().__init__()
        self.redis_client = None
        
    def init_components(self):
        """initialze all components"""
        # self._init_embed() # LTM 미사용으로 주석 처리
        # LLM 입출력 디버그 로거 설정. 'all' 모드로 변경하여 모든 콘솔 출력을 로깅합니다.
        self.debug_logger = setup_logger(log_path="data/logs/talk.log",enabled=True, mode="all")
        self._init_STMManager() # STM Manager 초기화 호출

        # Gemini backend — used by Turn if ConversationManager is exercised in Lite mode
        self.llm_backend = GeminiBackend(api_key=get_apikey())

        # TTS Pipeline 초기화
        try:
            self.tts_pipeline = self.load_tts_pipeline(self.config)
        except Exception as e:
            print(f"[TTS Error] Pipeline init failed: {e}")
            self.tts_pipeline = None
    
    def _init_STMManager(self):
        """Initialize simple Redis connection for STM"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6849, db=0, decode_responses=True)
            self.redis_client.ping()
            print("[STM] Simple Redis STM Connected.")
        except Exception as e:
            print(f"[STM] Redis connection failed: {e}")
            self.redis_client = None

    def check_components_ready(self) -> bool:
        """
        Check if critical components for Lite version are initialized.
        """
        components = {
            "TTS Pipeline": self.tts_pipeline,
            "Redis Client": self.redis_client,
        }
        for name, component in components.items():
            if component is None:
                print(f"[Error] {name} is not properly initialized.")
                return False
        
        print("[Info] All Lite components are properly initialized.")
        return True

class SemiConvManager(ConversationManager):
    def __init__(self, config: SemiAppConfig):
        super().__init__(config)
        self.history_file = "conversation.txt"

    def _loop_step(self):
        """A single step of the conversation loop for Gemini Lite version."""
        # 1. 큐에서 사용자 입력 확인
        if not self.app_context.qm.user_input_queue.empty():
            raw_input = self.app_context.qm.user_input_queue.get()
            
            user_utter = ""
            if isinstance(raw_input, dict):
                user_utter = raw_input.get('content', "")
            else:
                user_utter = str(raw_input)
            
            if user_utter.strip():
                self.process_interaction(user_utter)

    def process_interaction(self, user_input):
        """Process user input with Gemini and TTS."""
        print(f"\n[User] {user_input}")

        # 0. STM: Redis에서 대화 기록 가져오기
        history_context = ""
        if self.app_context.redis_client:
            try:
                # 최근 대화 10개(20줄) 가져오기
                stm_data = self.app_context.redis_client.lrange("stm_history", -20, -1)
                if stm_data:
                    history_context = "\n".join(stm_data)
            except Exception as e:
                print(f"[STM Error] Failed to fetch history: {e}")

        # 프롬프트 구성
        full_prompt = f"{history_context}\nUser: {user_input}" if history_context else user_input

        # 1. Gemini API 호출
        answer = ""  # 변수 초기화
        try:
            api_key = get_apikey()
            if not api_key:
                print("[System] Warning: GEMINI_API_KEY not found. Please set it in environment variables.")
                answer = "API Key Error"
            else:
                # 최신 google-genai SDK 사용
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model="gemini-2.5-flash", contents=full_prompt)
                answer = response.text
        except Exception as e:
            print(f"[Gemini Error] {e}\n")
            print("죄송해요, 오류가 발생했어요.")

        # 2. 대화 내용 저장
        conversation_entry = {
            "system": "",
            "conversations": [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer}
            ]
        }
        with open(self.history_file, "a", encoding="utf-8") as f:
            # 요청하신 JSON 형식으로 한 줄에 저장하며, 한글이 깨지지 않도록 ensure_ascii=False 옵션을 사용합니다.
            f.write(json.dumps(conversation_entry, ensure_ascii=False) + "\n")

        # 3. STM: Redis에 대화 기록 저장
        if self.app_context.redis_client and answer:
            try:
                self.app_context.redis_client.rpush("stm_history", f"User: {user_input}", f"AI: {answer}")
                # 기록이 너무 길어지지 않도록 유지 (예: 최근 50턴) 
                self.app_context.redis_client.ltrim("stm_history", -100, -1)
            except Exception as e:
                print(f"[STM Error] Failed to save history: {e}")

        # 4. TTS 재생
        try:
            if self.app_context.tts_pipeline:
                # run_sovits_tts 호출 (tts_pipeline, text, system_config)
                gen = run_sovits_tts(
                    self.app_context.tts_pipeline,
                    answer,
                    self.config
                )
                # 생성된 오디오 데이터를 큐에 전송
                for sr, audio_chunk in gen:
                    self.app_context.qm.audio_queue.put(audio_chunk)
            else:
                print("[TTS Error] Pipeline is not initialized.")
        except Exception as e:
            print(f"[TTS Error] Failed to play TTS: {e}")
    



if __name__ == "__main__":
    
    app_config = SemiAppConfig()        
    app_config.init_components()

    manager = SemiConvManager(app_config)
    app_config.check_components_ready()
    manager.app_context = app_config
    
    start_threads(app_config.em)
    manager.start()
