# e:\Project_HC_Victor\hyeonseo_ai_v0\tools.py

import json

def get_current_weather(location: str, unit: str = "celsius"):
    """
    지정된 위치의 현재 날씨 정보를 가져옵니다.
    """
    # 실제로는 날씨 API를 호출하겠지만, 여기서는 예시를 위해 더미 데이터를 반환합니다.
    if "서울" in location:
        return json.dumps({"location": location, "temperature": "15", "unit": unit, "forecast": "맑음"})
    elif "부산" in location:
        return json.dumps({"location": location, "temperature": "18", "unit": unit, "forecast": "구름 조금"})
    else:
        return json.dumps({"location": location, "temperature": "알 수 없음"})

def get_stock_price(symbol: str):
    """
    지정된 주식 심볼의 현재 가격을 가져옵니다.
    """
    # 실제로는 주식 API를 호출합니다.
    if symbol.upper() == "AAPL":
        return json.dumps({"symbol": symbol, "price": "172.25", "currency": "USD"})
    elif symbol.upper() == "GOOGL":
        return json.dumps({"symbol": symbol, "price": "135.50", "currency": "USD"})
    else:
        return json.dumps({"symbol": symbol, "price": "알 수 없음"})

# 1. 실제 함수를 이름과 매핑하는 레지스트리 (코드에서 함수를 실행할 때 사용)
TOOL_REGISTRY = {
    "get_current_weather": get_current_weather,
    "get_stock_price": get_stock_price,
}

# 2. LLM에게 제공할 도구의 명세 (JSON Schema 형식)
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "특정 지역의 현재 날씨 정보를 알려준다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "날씨를 알고 싶은 지역 이름, e.g., 서울",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "특정 회사의 주식 심볼(티커)을 받아서 현재 주가를 알려준다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "주가를 알고 싶은 회사의 주식 심볼, e.g., AAPL, GOOGL",
                    }
                },
                "required": ["symbol"],
            },
        },
    }
]
