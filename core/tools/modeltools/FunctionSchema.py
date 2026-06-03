from .FunctionCall import web_search, get_current_time, search_local_documents


# 실제 실행할 함수들을 딕셔너리로 매핑
TOOL_REGISTRY = {
    "web_search": web_search,
    "get_current_time": get_current_time,
    "search_local_documents": search_local_documents,
}

# LLM에게 제공할 도구 명세 (OpenAI Function Calling 형식)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "최신 정보, 시사, 특정 인물/장소/사건에 대한 정확한 정보가 필요할 때 웹에서 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 구체적인 키워드나 질문. 예: '대한민국 수도', 'Llama 3 모델 출시일'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "현재 한국(서울 기준)의 날짜와 시간을 확인합니다. '지금 몇시야', '오늘 날짜' 등의 질문에 사용합니다.",
            "parameters": {
                "type": "object",
                "properties": {}, # blank is true
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_documents",
            "description": "과거 대화, 저장된 메모, 요약 등 시스템의 장기 기억(LTM) 저장소에서 정보를 찾습니다. 시스템의 내부 지식이나 과거에 나눈 대화에 대한 질문일 때 사용합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "로컬 문서에서 찾고 싶은 내용에 대한 구체적인 키워드나 질문.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]