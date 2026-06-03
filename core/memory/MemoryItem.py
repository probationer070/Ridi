from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

# TODO: Implement MemoryItem model + add how to use memory_type ~~
class MemoryItem(BaseModel):
    """
    메모리 아이템의 데이터 구조를 정의하는 Pydantic 모델.
    NCA의 MemoryItem을 간소화한 형태.
    """
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4())) # 고유 ID (UUID)
    content: str # 메모리 내용 (검색 대상 텍스트)
    memory_type: str = 'generic' # 기억의 유형 ('generic', 'core', 'summary' 등). 검색 필터링에 사용.
    created_at_timestamp: int = Field(default_factory=lambda: int(datetime.now().timestamp())) # 생성 시간 (Unix timestamp)
    source: Optional[str] = None # 출처 (예: 'user_input', 'system_log', 'summary')
    tags: List[str] = Field(default_factory=list) # 태그 및 자동 추출 키워드 리스트
    metadata: Dict[str, Any] = Field(default_factory=dict) # 기타 유연한 메타데이터 (JSON 저장)
    score: Optional[float] = None # 검색 결과의 유사도 점수 (추가)
    
    # NCA MemoryItem에 있을 수 있는 다른 필드 (필요시 추가)
    # summary: Optional[str] = None
    # importance: float = 0.5
    # embedding: Optional[List[float]] = None # 임베딩은 FAISS에서 별도 관리

    class Config:
        validate_assignment = True # 필드 값 변경 시 유효성 검사
        # Pydantic V2에서는 model_config = {"validate_assignment": True} 사용 권장
