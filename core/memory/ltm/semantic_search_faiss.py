import faiss
import numpy as np
import os
import sys
import json
import logging
from sentence_transformers import SentenceTransformer # 또는 llama.cpp 연동 임베딩 함수
from typing import Union
from ..MemoryItem import MemoryItem

logger = logging.getLogger(__name__)


class SemanticSearchFAISS:
    """
    데이터가 많아질 경우, 시스템에 저장될 최대 아이템 개수 (예상치)와 원하는 검색 응답 시간 개선 필요
    """
    def __init__(self, 
                 embedding_model_name_or_instance: Union[str, SentenceTransformer], # 모델 이름 또는 인스턴스
                 faiss_index_path: str = "faiss_index.idx", # 이 매개변수는 이제 사용되지 않을 수 있음 (MemoryOrchestrator에서 경로 직접 지정)
                 metadata_path: str = "faiss_metadata.json",
                 embedding_dim: int = 768): # 모델에 맞는 차원 수
        """
        FAISS를 사용한 의미론적 검색기 초기화

        Args:
            embedding_model_name_or_instance (Union[str, SentenceTransformer]): 사용할 Sentence Transformer 모델 이름, 경로 또는 미리 로드된 인스턴스.
            faiss_index_path (str): FAISS 인덱스 파일 저장 경로.
            metadata_path (str): 각 벡터에 대한 메타데이터(예: 원본 텍스트 ID) 저장 경로.
            embedding_dim (int): 임베딩 벡터의 차원 수.
        """
        self.embedding_model_name = None
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        # self.embedding_dim = embedding_dim # embedding_model에서 가져오도록 변경

        if isinstance(embedding_model_name_or_instance, str):
            self.embedding_model_name = embedding_model_name_or_instance
            # 모델 저장 및 로드 경로 설정
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_script_dir, "../../..")) # hyeonseo_ai_v0 폴더
            models_base_dir = os.path.join(project_root, "models")
            
            model_folder_name = self.embedding_model_name.replace("/", "_")
            local_model_path = os.path.join(models_base_dir, model_folder_name)
            
            model_to_load = self.embedding_model_name
            loaded_from_local = False

            if os.path.isdir(local_model_path):
                logger.info(f"Found local model at {local_model_path}. Attempting to load.")
                model_to_load = local_model_path
                loaded_from_local = True
            else:
                logger.info(f"Local model not found at {local_model_path}. Will download from Hugging Face Hub: {self.embedding_model_name}")

            try:
                self.embedding_model = SentenceTransformer(model_to_load)
                logger.info(f"SentenceTransformer model '{model_to_load}' loaded successfully.")
                
                if not loaded_from_local and self.embedding_model:
                    logger.info(f"Saving model to {local_model_path} for future use.")
                    os.makedirs(local_model_path, exist_ok=True)
                    self.embedding_model.save(local_model_path)
                    logger.info(f"Model saved successfully to {local_model_path}.")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model '{model_to_load}': {e}", exc_info=True)
                if loaded_from_local:
                    logger.info(f"Failed to load from local path. Attempting to download from Hugging Face Hub: {self.embedding_model_name}")
                    try:
                        self.embedding_model = SentenceTransformer(self.embedding_model_name)
                        logger.info(f"Successfully downloaded and loaded model from Hugging Face Hub: {self.embedding_model_name}")
                        logger.info(f"Saving model to {local_model_path} for future use.")
                        os.makedirs(local_model_path, exist_ok=True)
                        self.embedding_model.save(local_model_path)
                        logger.info(f"Model saved successfully to {local_model_path}.")
                    except Exception as e_hub:
                        logger.error(f"Error loading SentenceTransformer model from Hugging Face Hub '{self.embedding_model_name}' after local fail: {e_hub}", exc_info=True)
                        raise
                else:
                    raise
        elif isinstance(embedding_model_name_or_instance, SentenceTransformer):
            self.embedding_model = embedding_model_name_or_instance
            # 미리 로드된 인스턴스의 경우, 모델 이름을 가져오는 더 안전한 방법 사용
            try:
                # SentenceTransformer의 첫 번째 모듈(일반적으로 Transformer 모델)의 config에서 _name_or_path를 가져옴
                if hasattr(self.embedding_model, 'tokenizer') and hasattr(self.embedding_model.tokenizer, 'name_or_path'):
                    self.embedding_model_name = self.embedding_model.tokenizer.name_or_path
                elif hasattr(self.embedding_model, '_first_module') and hasattr(self.embedding_model._first_module(), 'auto_model') and hasattr(self.embedding_model._first_module().auto_model, 'config') and hasattr(self.embedding_model._first_module().auto_model.config, '_name_or_path'):
                    self.embedding_model_name = self.embedding_model._first_module().auto_model.config._name_or_path
                else:
                    self.embedding_model_name = "Preloaded_SentenceTransformer_Instance"
            except Exception:
                self.embedding_model_name = "Preloaded_SentenceTransformer_Instance_Name_Unavailable"
            logger.info(f"Using pre-loaded SentenceTransformer model (identified as: {self.embedding_model_name}).")
        else:
            raise TypeError("embedding_model_name_or_instance must be a string or a SentenceTransformer instance.")

        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension() if self.embedding_model else embedding_dim

        self.index = None
        self.metadata_list = [] # 각 벡터의 인덱스에 해당하는 메타데이터 저장
        self._load_index_and_metadata()

    def _load_index_and_metadata(self):
        """FAISS 인덱스와 메타데이터를 파일에서 로드합니다."""
        if os.path.exists(self.faiss_index_path):
            try:
                self.index = faiss.read_index(self.faiss_index_path)
                logger.info(f"FAISS index loaded from {self.faiss_index_path}. Total vectors: {self.index.ntotal}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}. Creating a new index.")
                # IndexFlatIP: 내적(Inner Product) 기반. 정규화된 벡터의 내적은 코사인 유사도와 같음.
                self.index = faiss.IndexFlatIP(self.embedding_dim) 
        else:
            logger.info(f"FAISS index file not found at {self.faiss_index_path}. Creating a new index.")
            self.index = faiss.IndexFlatIP(self.embedding_dim) # L2에서 IP로 변경

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata_list = json.load(f)
                logger.info(f"Metadata loaded from {self.metadata_path}. Total items: {len(self.metadata_list)}")
                if self.index and self.index.ntotal != len(self.metadata_list) and self.index.ntotal > 0: # self.index None 체크 추가
                    logger.warning(f"Mismatch between FAISS index size ({self.index.ntotal}) and metadata count ({len(self.metadata_list)}). Consider rebuilding.")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}. Initializing empty metadata list.")
                self.metadata_list = []
        else:
            logger.info(f"Metadata file not found at {self.metadata_path}. Initializing empty metadata list.")
            self.metadata_list = []

    def _save_index_and_metadata(self):
        """FAISS 인덱스와 메타데이터를 파일에 저장합니다."""
        try:
            if self.index: # 인덱스가 None이 아닐 때만 저장
                faiss.write_index(self.index, self.faiss_index_path)
                logger.info(f"FAISS index saved to {self.faiss_index_path}")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")

        try:
            if self.metadata_list is not None: # 메타데이터 리스트가 None이 아닐 때만 저장
                with open(self.metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata_list, f, ensure_ascii=False, indent=4)
                logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def add_item(self, memory_item: MemoryItem):
        """
        MemoryItem의 content를 임베딩하여 FAISS 인덱스에 추가하고,
        item_id를 메타데이터로 저장합니다.

        Args:
            memory_item (MemoryItem): 저장할 메모리 아이템 객체.
        """
        if not memory_item.content:
            logger.warning("Cannot add empty text to LTM.")
            return

        try:
            embedding = self.embedding_model.encode([memory_item.content])[0]
            embedding_np = np.array([embedding]).astype('float32')
            
            # IndexFlatIP를 사용하므로 벡터를 L2 정규화합니다.
            # faiss.normalize_L2는 인플레이스 연산입니다.
            faiss.normalize_L2(embedding_np)
            
            self.index.add(embedding_np)
            # FAISS 메타데이터에는 SQLite와 연결될 item_id만 저장하거나, 필요한 최소 정보만 저장
            self.metadata_list.append({"item_id": memory_item.item_id}) 
            
            logger.info(f"Added item to LTM. Index size: {self.index.ntotal}, Metadata count: {len(self.metadata_list)}")
            self._save_index_and_metadata() # 변경 사항 즉시 저장
        except Exception as e:
            logger.error(f"Error adding item to FAISS (item_id: {memory_item.item_id}, text: {memory_item.content[:50]}...): {e}", exc_info=True)

    def search(self, query_text: str, top_k: int = 5) -> list:
        """
        주어진 쿼리 텍스트와 유사한 아이템들을 LTM에서 검색합니다.

        Args:
            query_text (str): 검색할 텍스트.
            top_k (int): 반환할 최대 결과 수.

        Returns:
            list: 유사한 아이템의 메타데이터(item_id 포함)와 거리 점수를 포함하는 딕셔너리 리스트.
                  예: [{"metadata": {...}, "score": 0.85}, ...]
        """
        if self.index.ntotal == 0:
            logger.info("LTM is empty. Cannot perform search.")
            return []

        query_embedding = self.embedding_model.encode([query_text])[0]
        # llama.cpp 사용 시:
        # query_embedding = self.llama_embedder.get_embedding(query_text)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # IndexFlatIP를 사용하므로 쿼리 벡터도 L2 정규화합니다.
        # faiss.normalize_L2는 인플레이스 연산입니다.
        faiss.normalize_L2(query_embedding_np)

        distances, indices = self.index.search(query_embedding_np, top_k)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            # IndexFlatIP는 내적값을 반환하며, 정규화된 벡터의 경우 이것이 코사인 유사도와 같습니다.
            # FAISS는 유사도가 높은 순으로 정렬하여 반환합니다.
            similarity_score = distances[0][i]
            # 임계값 필터링 제거, top_k 만큼의 결과를 모두 반환
            if idx < len(self.metadata_list): # 유효한 인덱스인지 확인
                results.append({
                    "metadata": self.metadata_list[idx],
                    "score": float(similarity_score) # 코사인 유사도 (높을수록 유사)
                })
            else:
                logger.warning(f"Found index {idx} out of bounds for metadata_list (size {len(self.metadata_list)}), or other issue with FAISS result.")
        
        logger.info(f"Search for '{query_text[:50]}...' found {len(results)} items (top_k={top_k}).")
        return results

    def get_faiss_metadata_by_item_id(self, item_id_value: str) -> dict | None:
        """
        FAISS에 저장된 메타데이터 리스트에서 item_id로 해당 메타데이터를 찾습니다.
        이 방법은 LTM 크기가 커지면 비효율적일 수 있습니다.
        주로 FAISS 인덱스와 메TA데이터 리스트의 동기화를 확인하거나,
        FAISS 인덱스 ID를 모를 때 사용합니다.
        """
        for item_metadata in self.metadata_list:
            if item_metadata.get("item_id") == item_id_value:
                return item_metadata
        return None
    
    def get_all_items(self) -> list:
        """LTM에 저장된 모든 아이템의 메타데이터를 반환합니다."""
        return self.metadata_list

    def clear_all(self):
        """LTM의 모든 데이터를 삭제합니다."""
        self.index = faiss.IndexFlatIP(self.embedding_dim) # 인덱스 타입 일치 (L2 -> IP)
        self.metadata_list = []
        self._save_index_and_metadata() # 변경 사항 저장
        logger.info("LTM cleared.")

    @property
    def count(self) -> int:
        """LTM에 저장된 아이템의 수를 반환합니다."""
        return self.index.ntotal
