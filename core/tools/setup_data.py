import os
import sys
import logging
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import sqlite3
import json

# 프로젝트 루트 경로 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 필요한 모듈 임포트
# StressCalculatorFAISS는 stress_calculator.py에 정의되어 있습니다.
from configs import SystemConfig
from core.memory.ltm.LTM_Manager import MemoryOrchestrator
from core.memory.ltm.korean_text_analyzer import TextRankKeywordExtractor # Keyword Extractor 추가
from core.engine.prompt_builder import PromptBuilder # PromptBuilder 임포트
from core.engine.ModelChat import ModelChat

# --- 로깅 설정 ---
LOG_DIR = os.path.join(project_root, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "setup_data.log")

logging.basicConfig(
    level=logging.DEBUG, # DEBUG 레벨로 변경하여 상세 로그 확인
    format="%(asctime)s: %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8'), # 실행 시마다 로그 파일 새로 작성
        logging.StreamHandler(sys.stdout) # 콘솔에도 출력
    ]
)
logger = logging.getLogger(__name__)

def setup_logging_and_config():
    """로깅을 설정하고 시스템 설정 객체를 반환합니다."""
    # 이 함수는 이미 전역에서 실행되었으므로, 여기서는 config 객체만 반환합니다.
    return SystemConfig()

def load_shared_models(config: SystemConfig):
    """공유 임베딩 모델과 LLM을 로드합니다."""
    embedding_model, llm_instance = None, None
    
    # 임베딩 모델 로드
    if hasattr(config, 'embedding_model') and config.embedding_model:
        logger.info(f"Loading shared Sentence Transformer model: {config.embedding_model}")
        try:
            embedding_model = SentenceTransformer(config.embedding_model)
            logger.info("Shared embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load shared embedding model: {e}", exc_info=True)
    else:
        logger.error("Embedding model name not defined in SystemConfig. Cannot proceed.")

    # LLM 로드
    if hasattr(config, 'model_path') and config.model_path:
        logger.info(f"Loading shared LLM: {config.model_path}")
        try:
            llm_instance = Llama(
                model_path="../" + config.model_path,
                n_ctx=config.model_n_ctx,
                n_gpu_layers=-1,
                verbose=False,
                chat_format="llama-3",
            )
            logger.info("Shared LLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load shared LLM: {e}", exc_info=True)
    else:
        logger.error("LLM model path not defined in SystemConfig. Cannot proceed.")
        
    return embedding_model, llm_instance

def _normalize_llm_source(llm_generated_source: any, line_num: int) -> str:
    """LLM이 생성한 다양한 타입의 source를 단일 문자열로 정규화합니다."""
    if isinstance(llm_generated_source, str) and llm_generated_source:
        return llm_generated_source
    
    if isinstance(llm_generated_source, list) and llm_generated_source:
        try:
            string_elements = [str(e) for e in llm_generated_source if isinstance(e, (str, int, float, bool))]
            if string_elements:
                normalized_source = ", ".join(string_elements)
                logger.info(f"Line {line_num}: Converted list source to string: '{normalized_source}' from {llm_generated_source}")
                return normalized_source
            logger.warning(f"Line {line_num}: LLM generated an empty or non-string list for source. Value: {llm_generated_source}")
        except Exception as e:
            logger.warning(f"Line {line_num}: Error converting list source to string: {e}. Value: {llm_generated_source}")
        return None

    if llm_generated_source is not None:
        try:
            normalized_source = str(llm_generated_source)
            logger.info(f"Line {line_num}: Converted non-string source to string: '{normalized_source}' from {llm_generated_source}")
            return normalized_source
        except Exception as e:
            logger.warning(f"Line {line_num}: Error converting non-string source to string: {e}. Value: {llm_generated_source}")
    
    return None

def _extract_ltm_metadata(content: str, llm: Llama, prompt_builder: PromptBuilder, keyword_extractor: TextRankKeywordExtractor, line_num: int, initial_metadata: dict):
    """LLM을 사용하여 콘텐츠에서 태그, 메타데이터, 소스를 추출합니다. 실패 시 fallback 로직을 사용합니다."""
    default_source = "initial_summary_loader"
    
    try:
        ltm_metadata_extraction_messages = prompt_builder.build_ltm_metadata_extraction_prompt(content)
        extracted_info_str = ModelChat.generate_once(llm, ltm_metadata_extraction_messages, max_tokens=300, temperature=0.8)

        if not extracted_info_str:
            logger.warning(f"Line {line_num}: LLM did not return any string for metadata. Using keyword extractor.")
            tags = keyword_extractor.extract_keywords(content, num_keywords=5)
            return tags, initial_metadata, default_source

        json_start = extracted_info_str.find('{')
        json_end = extracted_info_str.rfind('}')

        if json_start == -1 or json_end == -1 or json_start >= json_end:
            logger.warning(f"Line {line_num}: Could not find valid JSON in LLM response. Using keyword extractor. Response: {extracted_info_str}")
            tags = keyword_extractor.extract_keywords(content, num_keywords=5)
            return tags, initial_metadata, default_source

        cleaned_json_str = extracted_info_str[json_start : json_end + 1]
        parsed_info = json.loads(cleaned_json_str)
        
        tags = parsed_info.get("tags", [])
        if not tags:
            logger.info(f"Line {line_num}: LLM returned empty tags. Using keyword extractor as fallback.")
            tags = keyword_extractor.extract_keywords(content, num_keywords=5)

        final_metadata = initial_metadata.copy()
        final_metadata.update(parsed_info.get("metadata", {}))
        
        normalized_source = _normalize_llm_source(parsed_info.get("source"), line_num)
        source = normalized_source if normalized_source is not None else default_source
        
        return tags, final_metadata, source

    except json.JSONDecodeError as e:
        logger.error(f"Line {line_num}: Failed to parse JSON from LLM: {e}. Using keyword extractor. Response: {extracted_info_str}")
    except Exception as e:
        logger.error(f"Line {line_num}: Error processing LLM response for metadata: {e}. Using keyword extractor.")
    
    # Fallback in case of any exception
    tags = keyword_extractor.extract_keywords(content, num_keywords=5)
    return tags, initial_metadata, default_source

def load_initial_ltm_data(config: SystemConfig, embedding_model: SentenceTransformer, llm: Llama, prompt_builder: PromptBuilder, keyword_extractor: TextRankKeywordExtractor):
    """JSONL 파일에서 초기 LTM 데이터를 로드합니다."""
    logger.info("\n--- Setting up LTM with pre-summarized data ---")
    if not MemoryOrchestrator.setup_data(config, embedding_model_instance=embedding_model):
        logger.error("Failed to create or verify LTM file structure. Aborting LTM data loading.")
        return

    initial_data_path = getattr(config, 'ltm_initial_data_path', None)
    if not initial_data_path or not os.path.exists(initial_data_path):
        logger.warning(f"LTM initial data file not found at '{initial_data_path}'. Skipping initial data loading.")
        return

    ltm_orchestrator = None
    try:
        ltm_orchestrator = MemoryOrchestrator(
            sqlite_db_path=config.sqlite_db_path,
            faiss_index_path=config.faiss_index_path_ltm,
            faiss_metadata_path=config.faiss_metadata_path_ltm,
            embedding_model_name_or_instance=embedding_model,
            embedding_dim=embedding_model.get_sentence_embedding_dimension(),
            use_okt_keyword_extractor=getattr(config, 'ltm_use_okt_keyword_extractor', True)
        )
        
        last_processed_num = ltm_orchestrator.last_processed_data_number()
        with open(initial_data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())

        if last_processed_num >= total_lines:
            logger.info(f"LTM is already up to date. Last processed session {last_processed_num} matches total lines {total_lines}.")
            return
        
        logger.info(f"LTM has data up to session {last_processed_num}. Starting data load from line {last_processed_num + 1}.")
        
        with open(initial_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num <= last_processed_num:
                    continue

                line = line.strip()
                if not line: continue
                
                try:
                    session_data = json.loads(line)
                    conversations = session_data.get("conversations", [])
                    assistant_turn = next((turn for turn in reversed(conversations) if turn.get("role") == "assistant"), None)

                    if not assistant_turn or not assistant_turn.get("content"):
                        logger.warning(f"Line {line_num}: No valid assistant summary content found. Skipping.")
                        continue

                    content_to_store = assistant_turn["content"]
                    initial_metadata = {"original_session_id": session_data.get("id", f"session_{line_num}")}
                    
                    tags, metadata, source = _extract_ltm_metadata(
                        content_to_store, llm, prompt_builder, keyword_extractor, line_num, initial_metadata
                    )

                    ltm_orchestrator.add_memory(content=content_to_store, source=source, tags=tags, metadata=metadata)
                    logger.info(f"  Line {line_num}: Added summary to LTM. Source: '{source}', Tags: {tags}, Metadata: {metadata}")

                except (json.JSONDecodeError, IndexError) as e:
                    logger.warning(f"Line {line_num}: Skipping due to parsing error: {e}")
    finally:
        if ltm_orchestrator:
            ltm_orchestrator.close()
        logger.info("Finished loading initial LTM data.")

def backfill_missing_ltm_tags(config: SystemConfig, keyword_extractor: TextRankKeywordExtractor):
    """
    LTM 데이터베이스에서 tags_json이 비어있는 기존 데이터를 찾아
    korean_text_analyzer를 사용하여 태그를 생성하고 채워넣습니다.
    """
    logger.info("\n--- Backfilling missing LTM tags ---")
    conn = None
    try:
        db_path = config.sqlite_db_path
        if not os.path.exists(db_path):
            logger.warning(f"LTM database not found at '{db_path}'. Skipping tag backfill.")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # tags_json이 비어있는 ('[]'), NULL, 또는 빈 문자열인 경우를 모두 찾습니다.
        cursor.execute("SELECT item_id, content FROM memory_fts WHERE tags_json = '[]' OR tags_json IS NULL OR tags_json = ''")
        memories_to_update = cursor.fetchall()
        # logger.info(f"[memory Need to Update] : {memories_to_update}")

        if not memories_to_update:
            logger.info("No memories with empty tags found. Backfilling is not needed.")
            return

        logger.info(f"Found {len(memories_to_update)} memories with empty tags. Starting backfill process...")
        
        updated_count = 0
        for memory_id, content in memories_to_update:
            if not content or not content.strip():
                logger.warning(f"Memory ID {memory_id} has empty content. Skipping tag generation.")
                continue

            new_tags = keyword_extractor.extract_keywords(content, num_keywords=10)
            # logger.info(new_tags)
            if new_tags:
                tags_json = json.dumps(new_tags, ensure_ascii=False)
                cursor.execute("UPDATE memory_fts SET tags_json = ? WHERE item_id = ?", (tags_json, memory_id))
                logger.info(f"  Updated tags for memory ID {memory_id}: {new_tags}")
                updated_count += 1

        conn.commit()
        logger.info(f"--- Finished backfilling tags. Total updated: {updated_count}/{len(memories_to_update)} ---")
    except Exception as e:
        logger.error(f"An error occurred during the LTM tag backfill process: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def main():
    logger.info("--- Starting Data Setup Script ---")
    
    config = setup_logging_and_config()
    
    # 필요한 기본 디렉토리 생성
    os.makedirs(config.logs_dir, exist_ok=True)
    os.makedirs(config.ltm_data_dir, exist_ok=True)

    embedding_model, llm_instance = load_shared_models(config)
    if not embedding_model or not llm_instance:
        logger.error("One or more required models could not be loaded. Aborting data setup.")
        return

    prompt_builder = PromptBuilder()
    keyword_extractor = TextRankKeywordExtractor(
        pos_filter=["Noun", "Pronoun", "Verb", "Adjective", "Adverb"],
        min_word_length=1,
        textranking=True # TextRank 알고리즘 활성화
    )
    logger.info("Initialized TextRankKeywordExtractor for fallback tag generation.")

    load_initial_ltm_data(config, embedding_model, llm_instance, prompt_builder, keyword_extractor)
    
    # LTM에 태그가 비어있는 기존 데이터를 찾아 채워넣는 과정 추가
    backfill_missing_ltm_tags(config, keyword_extractor)
    
    logger.info("\n--- Data Setup Script Finished ---")
    logger.info(f"Check log file for details: {LOG_FILE_PATH}")

if __name__ == "__main__":
    main()
