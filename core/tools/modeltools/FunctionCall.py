# e:\Project_HC_Victor\hyeonseo_ai_v0\tools\tools.py
import json
import datetime
import pytz  # `pip install pytz`가 필요할 수 있습니다.
from ddgs import DDGS
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 순환 참조를 피하면서 타입 힌팅을 위해 사용
    from infer_script import AppConfig

def web_search(query: str) -> str:
    """
    주어진 쿼리에 대해 웹 검색을 수행하고 상위 2개의 결과를 요약하여 JSON 형식의 문자열로 반환합니다.
    최신 정보, 특정 인물, 사건, 장소에 대한 정보가 필요할 때 사용합니다.
    """
    print(f"--- [Tool] Executing web_search with query: {query} ---")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
            if not results:
                return json.dumps({"result": "검색 결과가 없습니다."})
            
            # 검색 결과를 간단한 JSON 형식으로 포맷팅
            search_summary = {
                "query": query,
                "results": [
                    {"title": r.get("title"), "snippet": r.get("body")} for r in results
                ]
            }
            return json.dumps(search_summary, ensure_ascii=False)
    except Exception as e:
        print(f"--- [Tool Error] Web search failed: {e} ---")
        return json.dumps({"error": str(e)})

def get_current_time() -> str:
    """
    현재 대한민국 서울의 날짜와 시간을 JSON 형식의 문자열로 반환합니다.
    '지금 몇 시야?', '오늘 날짜 알려줘' 와 같은 질문에 사용합니다.
    """
    print("--- [Tool] Executing get_current_time ---")
    try:
        korea_tz = pytz.timezone("Asia/Seoul")
        now = datetime.datetime.now(korea_tz)
        time_info = {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": "Asia/Seoul"
        }
        return json.dumps(time_info, ensure_ascii=False)
    except Exception as e:
        print(f"--- [Tool Error] get_current_time failed: {e} ---")
        return json.dumps({"error": str(e)})

def search_local_documents(query: str, app_context: 'AppConfig') -> str:
    """
    사용자의 장기 기억(LTM) 저장소에서 관련 정보를 검색합니다.
    과거 대화, 저장된 메모, 요약 등 내부 지식에 대한 질문에 사용합니다.
    """
    print(f"--- [Tool] Executing search_local_documents with query: {query} ---")
    if not app_context or not app_context.memory_orchestrator:
        return json.dumps({"error": "로컬 문서 검색(LTM)을 위한 컴포넌트가 초기화되지 않았습니다."}, ensure_ascii=False)

    try:
        # MemoryOrchestrator를 사용하여 RAG 검색 수행
        rag_results = app_context.memory_orchestrator.search_memories_for_rag(
            current_user_input=query,
            final_top_k=3
        )
        if not rag_results:
            return json.dumps({"result": f"'{query}'와 관련된 기억을 찾지 못했습니다."}, ensure_ascii=False)

        found_docs = [
            {"content": mem.content, "source": mem.metadata.get("source", "N/A"), "score": f"{score:.4f}"}
            for mem, score in rag_results
        ]
        
        return json.dumps({"query": query, "found_documents": found_docs}, ensure_ascii=False)

    except Exception as e:
        print(f"--- [Tool Error] search_local_documents failed: {e} ---")
        return json.dumps({"error": str(e)}, ensure_ascii=False)