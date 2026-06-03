import uuid
import logging
import csv
import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from llama_cpp import Llama

from configs import SystemConfig
from ..MemoryItem import MemoryItem
from ...engine.prompt_builder import PromptBuilder
from ...engine.ModelChat import ModelChat
from .keyword_search_sqlite import KeywordSearchSQLite
from .semantic_search_faiss import SemanticSearchFAISS

from sentence_transformers import SentenceTransformer # Import SentenceTransformer

logger = logging.getLogger(__name__)
# Import Okt keyword extractor (optional loading)
try:
    from .okt_keyword_extractor import OktKeywordExtractor, KONLPY_OKT_AVAILABLE
except ImportError:
    OktKeywordExtractor = None
    KONLPY_OKT_AVAILABLE = False
    logger.warning("OktKeywordExtractor could not be imported. Okt-based keyword extraction will not be available.")

class MemoryOrchestrator:

    @classmethod
    def setup_data(cls, config:SystemConfig, embedding_model_instance=SentenceTransformer, llm_instance=Llama):
        """
        Initializes SQLite DB and FAISS files for MemoryOrchestrator.
        If files already exist, skips initialization.
        If initial data JSONL file is provided, loads data into LTM.

        Args:
            cls: MemoryOrchestrator class itself.
            config: SystemConfig object (including path info).
            embedding_model_instance: Pre-loaded SentenceTransformer model instance (for SemanticSearchFAISS).
                                      If None, attempts to load internally.
            llm_instance: LLM instance (needed for add_memory_with_llm).
        Returns:
            bool: True if data setup was successful, False otherwise.
        """
        db_path = config.sqlite_db_path
        index_path = config.faiss_index_path_ltm
        metadata_path = config.faiss_metadata_path_ltm

        # Ensure LTM data directory exists
        os.makedirs(config.ltm_data_dir, exist_ok=True)

        files_exist = os.path.exists(db_path) and \
                      os.path.exists(index_path) and \
                      os.path.exists(metadata_path)

        if files_exist:
            logger.info("[MemoryOrchestrator Setup] LTM data files already exist. Skipping base file initialization.")
            return True

        logger.info("[MemoryOrchestrator Setup] LTM data files not found or incomplete. Initializing MemoryOrchestrator to create them...")
        try:
            # --- 1. Resolve Embedding Model ---
            embedding_model_to_use = embedding_model_instance or getattr(config, 'embedding_model', None)
            if not embedding_model_to_use:
                logger.error("[Setup Error] 'embedding_model' in SystemConfig or an 'embedding_model_instance' must be provided.")
                return False

            # --- 2. Determine Embedding Dimension (Dynamically) ---
            embedding_dim = None
            if hasattr(embedding_model_to_use, 'get_sentence_embedding_dimension'):
                # If a model instance is passed, use it directly.
                embedding_dim = embedding_model_to_use.get_sentence_embedding_dimension()
                logger.info(f"Using provided embedding model instance. Determined dimension: {embedding_dim}")
            elif isinstance(embedding_model_to_use, str):
                # If a path is given, load the model to find its dimension.
                try:
                    logger.info(f"Loading embedding model from '{embedding_model_to_use}' to determine dimension...")
                    temp_model = SentenceTransformer(embedding_model_to_use)
                    embedding_dim = temp_model.get_sentence_embedding_dimension()
                    logger.info(f"Determined embedding dimension: {embedding_dim}")
                    del temp_model  # Release memory
                except Exception as e:
                    logger.error(f"Failed to load model from path '{embedding_model_to_use}' to get dimension: {e}", exc_info=True)
                    # Fallback to config as a last resort
                    embedding_dim = getattr(config, 'embedding_model_dim', None)
                    if embedding_dim:
                        logger.warning(f"Could not determine dimension from model. Falling back to 'embedding_model_dim' from config: {embedding_dim}. Ensure this is correct.")
                    else:
                        logger.error("Could not determine embedding dimension from model, and 'embedding_model_dim' is not set in SystemConfig.")
                        return False
            
            if not embedding_dim:
                logger.error("Unable to determine embedding dimension. Setup cannot proceed.")
                return False

            # --- 3. Create and Close Temporary Orchestrator to Build Files ---
            logger.info("Creating temporary MemoryOrchestrator to initialize data files...")
            # A temporary orchestrator is created just to initialize the underlying data files.
            # It doesn't require a user_id or prompt_builder.
            orchestrator = cls(
                config=config,
                embedding_model_name_or_instance=embedding_model_to_use,
                embedding_dim=embedding_dim,
                llm_instance=llm_instance
            )
            orchestrator.close()
            logger.info("[MemoryOrchestrator Setup] LTM data files created successfully.")
            return True
        except Exception as e:
            logger.error(f"[MemoryOrchestrator Setup Error] Error during LTM data setup: {e}", exc_info=True)
            # Attempt to clean up partially created files
            return False

    def _create_keyword_extractor(self, config: Dict[str, Any]) -> Optional[callable]:
        """generate Keyword extractor instance based on config"""
        strategy = config.get("strategy", "simple") # Default is 'simple'
        logger.info(f"Attempting to create keyword extractor with strategy: '{strategy}'")

        if strategy == "okt":
            if not (KONLPY_OKT_AVAILABLE and OktKeywordExtractor):
                logger.warning("Okt keyword extraction requested, but OktKeywordExtractor is not available. Falling back to simple extractor.")
                return None
            
            okt_params = config.get("okt_config", {})
            try:
                okt_instance = OktKeywordExtractor(**okt_params)
                logger.info(f"OktKeywordExtractor created successfully with config: {okt_params}")
                return okt_instance.extract_keywords
            except Exception as e:
                logger.error(f"Failed to initialize OktKeywordExtractor with config {okt_params}: {e}. Falling back to simple extractor.")
                return None

        elif strategy == "simple":
            # KeywordSearchSQLite가 내부적으로 기본 추출기(공백 기반)를 사용하도록 None을 반환합니다.
            logger.info("Keyword strategy is 'simple'. KeywordSearchSQLite will use its default whitespace-based extractor.")
            return None

        elif strategy == "llm":
            if not self.llm:
                logger.warning("LLM keyword extraction requested, but LLM instance is not available. Falling back to simple extractor.")
                return None
            
            llm_config = config.get("llm_config", {})
            prompt_template = llm_config.get(
                "prompt_template",
                "Extract only 5 core keywords from the following text, separated by commas. Do not add any other explanation.\n\nText: {text}\n\nKeywords:"
            )
            
            def llm_extractor(text: str) -> List[str]:
                try:
                    prompt = prompt_template.format(text=text)
                    response = self.llm.create_completion(
                        prompt=prompt, max_tokens=100, stop=["<|eot_id|>"], echo=False, temperature=0.7, min_p=0.05
                    )
                    extracted_keywords_str = response['choices'][0]['text'].strip()
                    extracted_keywords = [kw.strip() for kw in extracted_keywords_str.split(',') if kw.strip()]
                    logger.info(f"LLM extracted keywords: {extracted_keywords}")
                    return extracted_keywords
                except Exception as e:
                    logger.error(f"LLM keyword extraction failed: {e}. Returning empty list.")
                    return []

            logger.info("Using LLM-based keyword extractor.")
            return llm_extractor
        else:
            logger.warning(f"Unknown keyword extraction strategy: '{strategy}'. Falling back to simple extractor.")
            return None

    def __init__(self,
                 config: SystemConfig,
                 user_id: Optional[str],
                 embedding_model_name_or_instance: Union[str, SentenceTransformer],
                 embedding_dim: int,
                 prompt_builder: Optional[PromptBuilder],
                 llm_instance: Optional[Llama] = None,
                 semantic_weight: float = 1.0, # Semantic score weight
                 keyword_weight: float = 1.0  # Keyword score weight
                 ):
        logger.info("Initializing MemoryOrchestrator...")

        # Extract paths and other configurations from the SystemConfig object
        sqlite_db_path = config.sqlite_db_path
        faiss_index_path = config.faiss_index_path_ltm
        faiss_metadata_path = config.faiss_metadata_path_ltm
        self.keyword_config = config.ltm_keyword_config or {}
        self.scoring_log_path = config.ltm_scoring_log_path

        if embedding_model_name_or_instance is None:
            raise ValueError("An embedding model (instance or path) must be provided via 'embedding_model_name_or_instance'.")

        self.llm = llm_instance # LLM을 먼저 할당해야 _create_keyword_extractor에서 사용 가능

        # --- Setup Keyword Extractor ---
        # This will now return a callable function for 'okt' or 'llm', or None for 'simple'
        self.keyword_extractor = self._create_keyword_extractor(self.keyword_config)
        
        # Pass keyword_extractor during KeywordSearchSQLite initialization
        sqlite_config = self.keyword_config.get("sqlite_config", {})
        self.keyword_store = KeywordSearchSQLite(db_path=sqlite_db_path, 
                                                 keyword_extractor=self.keyword_extractor, **sqlite_config)

        # Determine arguments for SemanticSearchFAISS based on whether a model path or instance was passed.
        if isinstance(embedding_model_name_or_instance, str):
            # If a model name is given, pass the name and dimension for SemanticSearchFAISS to load.
            semantic_embedding_model_arg = embedding_model_name_or_instance
            semantic_embedding_dim_arg = embedding_dim
        else: # If a SentenceTransformer instance is given, pass it directly.
            semantic_embedding_model_arg = embedding_model_name_or_instance
            semantic_embedding_dim_arg = embedding_dim # Use passed dim, or recalculate

        self.semantic_store = SemanticSearchFAISS(
            embedding_model_name_or_instance=semantic_embedding_model_arg, # Pass the resolved argument
            faiss_index_path=faiss_index_path,
            metadata_path=faiss_metadata_path,
            embedding_dim=semantic_embedding_dim_arg
        )
        self.prompt_builder = prompt_builder
        self.user_id = user_id
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.scoring_log_file = None
        self.csv_writer = None
        self._setup_scoring_log()

        logger.info("MemoryOrchestrator initialized.")



    def _setup_scoring_log(self):
        """Initializes the CSV logger for scoring logs."""
        if not self.scoring_log_path:
            return
        
        try:
            log_dir = os.path.dirname(self.scoring_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_exists = os.path.isfile(self.scoring_log_path)
            self.scoring_log_file = open(self.scoring_log_path, 'a', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.scoring_log_file)

            if not file_exists:
                header = [
                    'log_timestamp', 'search_query', 'item_id', 'content_preview', 
                    'final_score', 'semantic_norm', 'keyword_norm', 
                    'semantic_weight', 'keyword_weight'
                ]
                self.csv_writer.writerow(header)
            logger.info(f"Scoring log will be written to {self.scoring_log_path}")
        except Exception as e:
            logger.error(f"Failed to set up scoring log at {self.scoring_log_path}: {e}")
            self.scoring_log_file = None
            self.csv_writer = None

    def _extract_ltm_metadata(self, summary: str) -> Dict:
        """Extracts metadata for LTM storage from a summary."""
        try:
            messages = self.prompt_builder.build_ltm_metadata_extraction_prompt(summary)
            if not self.llm:
                logger.warning("LLM instance not available for metadata extraction. Returning empty metadata.")
                return {}

            extracted_info_str = ModelChat.generate_stream(messages)
            
            if not extracted_info_str:
                return {}

            json_start = extracted_info_str.find('{')
            json_end = extracted_info_str.rfind('}')
            if json_start == -1 or json_end == -1:
                logger.warning(f"Could not find JSON object in LLM response for metadata: {extracted_info_str}")
                return {}
            
            cleaned_json_str = extracted_info_str[json_start : json_end+1]
            parsed_info = json.loads(cleaned_json_str)
            
            # Extracts and returns metadata and tags.
            # The prompt for _build_ltm_metadata_extraction_prompt should be updated
            # to request a 'tags' field (e.g., a list of strings) in the JSON output.
            metadata = parsed_info.get("metadata", {})
            tags = parsed_info.get("tags", []) # Extract tags as well
            metadata['extracted_tags'] = tags # Optionally store tags in metadata for traceability
            return metadata
        except Exception as e:
            logger.error(f"[LTM Manager Error] Error during LTM metadata extraction: {e}", exc_info=True)
            return {}

    def add_memory_from_summary(self, 
                                summary: str, 
                                original_text: str, 
                                source_description: str, 
                                tags: Optional[List[str]] = None):
        """
        Receives a summary, extracts metadata, and saves it to LTM. Called from STM_Manager.
        If tags are not provided, they will be extracted from the summary.
        """
        logger.info(f"Adding memory from summary. Source: {source_description}")
        try:
            # 1. Use LLM to extract structured metadata from the summary
            # This now can return both metadata and tags if the prompt is designed for it.
            extracted_data = self._extract_ltm_metadata(summary)
            metadata = extracted_data.get("metadata", {})
            # If tags are not passed as an argument, try to get them from metadata extraction.
            if tags is None:
                tags = extracted_data.get("tags", [])
            
            # 2. Add the original conversation text to the metadata
            metadata['original_text'] = original_text
            
            self.add_memory(
                content=summary,
                tags=tags, # Pass the provided or extracted tags
                source=source_description, 
                metadata=metadata
            )
            logger.info(f"[LTM Manager] Successfully saved memory from summary: '{summary[:30]}...'")
        except Exception as e:
            logger.error(f"[LTM Manager Error] Error while saving memory from summary: {e}", exc_info=True)

    def add_memory(self, 
                   content: Union[str, MemoryItem], 
                   source: Optional[str] = None, 
                   tags: Optional[List[str]] = None, 
                   metadata: Optional[Dict[str, Any]] = None, 
                   item_id: Optional[str] = None) -> MemoryItem:
        """
        Adds a new memory to LTM.
        Accepts either a content string (with other details) or a pre-made MemoryItem object.
        Generates UUID if no item_id is provided.
        Saved to both SQLite and FAISS.
        """
        if isinstance(content, MemoryItem):
            memory_item = content
            if not memory_item.item_id:
                memory_item.item_id = str(uuid.uuid4())
            if not memory_item.created_at_timestamp:
                memory_item.created_at_timestamp = int(datetime.now().timestamp())
        else:
            memory_item = MemoryItem(
                item_id=item_id or str(uuid.uuid4()),
                content=content,
                created_at_timestamp=int(datetime.now().timestamp()),
                source=source,
                tags=tags or [], # Initialize with empty list if tags is None
                metadata=metadata or {}
            )

        # Add user_id to metadata to ensure all memories are associated with a user.
        if 'user_id' not in memory_item.metadata:
            memory_item.metadata['user_id'] = self.user_id

        # If tags were not provided and a custom extractor ('llm' or 'okt') is configured, run it.
        # The 'simple' strategy is handled inside KeywordSearchSQLite, so we don't call it here.
        # The self.keyword_extractor is a callable function or None.
        should_run_custom_extractor = (tags is None) and callable(self.keyword_extractor)
        
        if should_run_custom_extractor:
            extracted_keywords = self.keyword_extractor(memory_item.content)
            # Add extracted keywords to the item's tags, avoiding duplicates.
            memory_item.tags.extend([kw for kw in extracted_keywords if kw not in memory_item.tags])

        # 1. Save item to the keyword store (SQLite)
        # extract_keywords=True tells SQLite to use its internal extractor if no tags are present.
        self.keyword_store.add_item(memory_item, extract_keywords=True)
        logger.info(f"Memory item {memory_item.item_id} added/updated in KeywordStore.")

        # 2. Save item to the semantic store (FAISS)
        self.semantic_store.add_item(memory_item)
        logger.info(f"Memory item {memory_item.item_id} added/updated in SemanticStore.")
        
        return memory_item

    def delete_memory(self, item_id: str):
        """
        Deletes a memory item from all underlying stores (SQLite and FAISS).
        This is a critical operation and assumes `delete_item` method exists and works
        correctly in both keyword_store and semantic_store.
        """
        if not item_id:
            logger.warning("delete_memory called with no item_id.")
            return

        try:
            logger.info(f"Attempting to delete memory item {item_id} from all stores.")
            # Attempt to delete from FAISS first.   # TODO: Add delete_item to SemanticSearchFAISS
            self.semantic_store.delete_item(item_id)
            logger.info(f"Item {item_id} deleted from SemanticStore (FAISS).")

            self.keyword_store.delete_item(item_id)
            logger.info(f"Item {item_id} deleted from KeywordStore (SQLite).")

        except Exception as e:
            logger.error(f"Failed to completely delete memory item {item_id}: {e}", exc_info=True)

    def _search_keyword_store(self, 
                              query: Optional[str], 
                              filters: Optional[Dict[str, Any]], 
                              top_k: int) -> Dict[str, Dict[str, Any]]:
        """Performs a keyword/filter search and returns a map of candidates.
        
        Args:
            query: The search query string. If None or empty, only filters are applied.
            filters: A dictionary of metadata filters to apply (e.g., {'tag': 'important'}).
            top_k: The maximum number of results to return.
        Returns:
            A dictionary mapping item_id to a dict with 'item' (MemoryItem) and 'keyword_raw_score'.
        """
        keyword_candidates = {}

        # Add the current user's ID to the search filters by default.
        # This ensures that only the current user's memories are searched.
        search_filters = filters.copy() if filters else {}
        search_filters['user_id'] = self.user_id

        if not query and not search_filters:
            return keyword_candidates

        keyword_results_dicts = self.keyword_store.search(query=query, filters=search_filters, top_k=top_k)
        logger.info(f"Keyword search found {len(keyword_results_dicts)} items.")
        for res_dict in keyword_results_dicts:
            item_id = res_dict.get("item_id")
            if not item_id:
                continue
            
            item_data_for_model = {k: v for k, v in res_dict.items() if k not in ['score', 'fts_content']}
            item = MemoryItem(**item_data_for_model)
            
            keyword_candidates[item_id] = {
                "item": item,
                "keyword_raw_score": res_dict.get("score")
            }
        return keyword_candidates

    def _search_semantic_store(self, 
                               query_text: str, 
                               top_k: int) -> Dict[str, Dict[str, Any]]:
        """Performs a semantic search and returns a map of candidates."""
        semantic_candidates = {}
        semantic_results_raw = self.semantic_store.search(query_text=query_text, top_k=top_k)
        logger.info(f"Semantic search found {len(semantic_results_raw)} raw items.")

        for res in semantic_results_raw:
            item_id = res["metadata"].get("item_id")
            if not item_id:
                continue
            
            # Get full item details from the primary store (SQLite).
            item_detail = self.keyword_store.get_item_by_id(item_id)
            if not item_detail:
                logger.warning(f"Item ID {item_id} from semantic search not found in keyword store. Skipping.")
                continue
            
            # Verify that the semantic search result belongs to the current user (very important).
            if item_detail.metadata.get('user_id') != self.user_id:
                logger.debug(f"Item ID {item_id} from semantic search belongs to another user ('{item_detail.metadata.get('user_id')}'). Skipping.")
                continue

            semantic_candidates[item_id] = {
                "item": item_detail,
                "semantic_raw_score": res.get("score")
            }
        return semantic_candidates

    def _merge_candidates(self, 
                          keyword_results: Dict, 
                          semantic_results: Dict) -> Dict[str, Dict[str, Any]]:
        """Merges results from keyword and semantic searches."""
        merged = semantic_results.copy()
        for item_id, kw_data in keyword_results.items():
            if item_id in merged:
                # If found in both, add the keyword score.
                merged[item_id]["keyword_raw_score"] = kw_data["keyword_raw_score"]
            else:
                # If found only in keyword search.
                merged[item_id] = kw_data
        return merged

    def _calculate_final_scores(self, 
                                candidates: Dict[str, Dict[str, Any]], 
                                query_text: str) -> List[Dict[str, Any]]:
        """Normalizes scores and applies weights to calculate the final score."""
        if not candidates:
            return []

        # --- 키워드 점수 정규화 설정 ---
        keyword_raw_scores = [
            data.get("keyword_raw_score") for data in candidates.values() 
            if data.get("keyword_raw_score") is not None
        ]
        abs_keyword_ranks = [abs(s) for s in keyword_raw_scores]
        min_abs_rank = min(abs_keyword_ranks) if abs_keyword_ranks else 0.0
        max_abs_rank = max(abs_keyword_ranks) if abs_keyword_ranks else 0.0
        rank_range = max_abs_rank - min_abs_rank

        # --- 점수 계산 루프 ---
        scored_candidates = []

        for item_id, data in candidates.items():
            item = data["item"]
            
            # 1. 의미론적 점수 (0 to 1)
            semantic_norm = data.get("semantic_raw_score", 0.0) or 0.0

            # 2. 키워드 점수 (0 to 1)
            keyword_raw = data.get("keyword_raw_score")
            keyword_norm = 0.0
            if keyword_raw is not None:
                if rank_range > 1e-9: # Prevent division by zero.
                    keyword_norm = (abs(keyword_raw) - min_abs_rank) / rank_range
                elif abs_keyword_ranks: # When there is only one item or all ranks are the same.
                    keyword_norm = 0.5

            # 4. 최종 점수 계산
            # The weights balance the scores, so a simple sum is used.
            final_score = (self.semantic_weight * semantic_norm) + (self.keyword_weight * keyword_norm)
            
            data['final_score'] = final_score
            scored_candidates.append(data)

            # Improved log message
            log_msg = (
                f"Scoring ID: {item.item_id} | Final: {final_score:.4f} | Breakdown: "
                f"[Sem: {semantic_norm:.3f} * {self.semantic_weight}] + "
                f"[Key: {keyword_norm:.3f} * {self.keyword_weight}] + "
                f"Content: '{item.content[:30]}...'"
            )
            logger.info(log_msg)

            # Add CSV logging
            if self.csv_writer and self.scoring_log_file:
                try:
                    log_row = [
                        datetime.now().isoformat(),
                        query_text,
                        item.item_id,
                        item.content[:100].replace('\n', ' '),
                        f"{final_score:.6f}",
                        f"{semantic_norm:.6f}",
                        f"{keyword_norm:.6f}",
                        self.semantic_weight,
                        self.keyword_weight,
                    ]
                    self.csv_writer.writerow(log_row)
                    self.scoring_log_file.flush()
                except Exception as e:
                    logger.error(f"Error writing to scoring log file: {e}")

        return sorted(scored_candidates, key=lambda x: x["final_score"], reverse=True)

    def search_memories_for_rag(self, 
                                current_user_input: str,
                                keyword_query: Optional[str] = None,
                                filters: Optional[Dict[str, Any]] = None,
                                top_k_keyword: int = 5,
                                top_k_semantic: int = 3,
                                final_top_k: int = 5,
                                boost_factor: float = 1.5
                               ) -> List[Tuple[MemoryItem, float]]:
        """
        키워드 검색과 의미론적 검색을 결합하여 RAG에 사용할 기억을 찾습니다.
        두 소스에서 후보군을 가져와 병합하고, 관련성과 최신성을 기반으로 하이브리드 점수를 계산하여
        가장 순위가 높은 기억들을 반환합니다.

        Args:
            current_user_input: 현재 사용자 입력(의미론적 검색에 사용).
            keyword_query: 키워드 검색에 사용할 쿼리. None 또는 빈 문자열이면 필터만 적용.
            filters: 메타데이터 필터(예: {'tag': 'important'}). None이면 필터 없음.
            top_k_keyword: 키워드 검색에서 반환할 최대 항목 수.
            top_k_semantic: 의미론적 검색에서 반환할 최대 항목 수.
            final_top_k: 최종적으로 반환할 최대 기억 항목 수.
            boost_factor: 'core' 메모리에 대한 점수 부스트 계수. 1.0이면 부스트 없음.
        Returns:
            A list of tuples containing MemoryItem and its final score, sorted by score descending
        """
        logger.info(f"Searching memories for RAG. User input: '{current_user_input[:50]}...', Keyword: '{keyword_query}', Filters: {filters}")
        
        # 1. Search for memory candidates from each store (including core memories)
        keyword_candidates = self._search_keyword_store(keyword_query, filters, top_k_keyword)
        semantic_candidates = self._search_semantic_store(current_user_input, top_k_semantic)

        # 2. Merge candidates
        merged_candidates = self._merge_candidates(keyword_candidates, semantic_candidates)
        if not merged_candidates:
            logger.info("No candidate items found from keyword or semantic search.")
            return []

        # 3. Calculate final scores and re-rank
        ranked_candidates = self._calculate_final_scores(merged_candidates, current_user_input)

        ranked_regular_results = [
            (data["item"], data["final_score"]) 
            for data in ranked_candidates
        ]

        # 4. Boost score for 'core' memories if they appear in the search results.
        # This makes them more likely to be selected if they are relevant to the query,
        # without forcing their inclusion.
        final_results_with_boost = []
        for item, score in ranked_regular_results:
            # You can adjust the boost_factor
            final_score = score
            if item.metadata.get('memory_type') == 'core':
                final_score *= boost_factor
                logger.info(f"Boosting score for core memory item {item.item_id}. Original: {score:.4f}, Boosted: {final_score:.4f}")
            final_results_with_boost.append((item, final_score))

        # 5. Sort by the final (potentially boosted) score and return the top_k
        final_results = sorted(final_results_with_boost, key=lambda x: x[1], reverse=True)[:final_top_k]
        
        logger.info(f"Final {len(final_results)} memory items selected for RAG prompt (requested top_k={final_top_k}):")
        for i, (item, score) in enumerate(final_results):
            logger.info(f"  Selected {i+1}: ID={item.item_id}, Score={score:.4f}, Content='{item.content[:50]}...'")
            
        return final_results

    def get_all_memories_for_debugging(self) -> List[MemoryItem]:
        """
        Retrieves all memories from the SQLite database for debugging purposes.
        Note: This will retrieve memories for ALL users. For user-specific memories, use search with filters.
        The actual implementation of getting all items should be in KeywordSearchSQLite.
        """
        logger.info("Retrieving all memories from KeywordStore for debugging.")
        try:
            # Assumes that KeywordSearchSQLite has a method (e.g., get_all_items) to fetch all items.
            # This method needs to be added to the KeywordSearchSQLite class if it doesn't exist.
            return self.keyword_store.get_all_items()
        except Exception as e:
            logger.error(f"Error retrieving all memories for debugging from keyword_store: {e}", exc_info=True)
            return []

    def close(self) -> None:
        self.keyword_store.close()
        if self.scoring_log_file:
            self.scoring_log_file.close()
        # self.semantic_store.close() # FAISS may not have explicit close
        logger.info("MemoryOrchestrator closed.")

    def get_memory_item(self, user_id: Optional[str] = None) -> Optional[MemoryItem]:
        """
        Retrieves the last memory item from the LTM based on user_id.
        Returns a dictionary with 'content', 'created_at_timestamp', or None.
        """
        cursor = self.keyword_store.conn.cursor()

        cursor.execute("SELECT content FROM memories ORDER BY created_at_timestamp DESC LIMIT 1")
        data = cursor.fetchone()
        
        if data:
            # 딕셔너리 형태로 반환하여 접근 용이하게 함
            memory_item = {"content": data[0]}
            print(f"--- [Tool] Retrieved memory item: {memory_item} ---")
            return memory_item
        else:
            print(f"--- [Tool] No memory item found for user_id='{user_id or self.user_id}' ---")
            return None

# TODO: plz check MemGraph & Reranker integration, it's not added yet. + not done