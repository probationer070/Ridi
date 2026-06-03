from core.memory.MemoryItem import MemoryItem
from typing import List, Dict, Union, Tuple

class PromptBuilder:
    def get_summarization_system_content(self) -> str:
        """요약 생성을 위한 시스템 프롬프트의 내용만 반환합니다."""
        return "이전 대화들의 핵심 내용과 주제를 간결하게 요약하여 현재 대화의 맥락을 만들어줘."
    
    def get_Mem_system_content(self) -> str:
        """요약 생성을 위한 시스템 프롬프트의 내용만 반환합니다."""
        return "아래의 지침을 중점으로 재한과 리디의 대화내용을 요약하세요.\n- 재한의 명시적인 지시, 요청, 약속 (반복되는 지시/요청은 처음 언급되거나 중요하게 강조된 시점을 중심으로 간결하게 요약)\n- 재한의 개인 정보 및 선호\n- 리디의 발언 중 중요한 사실 및 제안\n- 대화 중 발생한 새로운 사실, 사건, 또는 상황 변화\n- 대화의 핵심 주제 및 흐름 변화\n- 발생한 문제, 갈등, 그리고 해결 과정\n- 대화의 궁극적인 합의 또는 결론"

    def build_summary_prompt(self, conversation_text_for_summary: str) -> List[Dict[str, str]]:
        """
        '현재 대화의 맥락'을 만들기 위한 요약(rolling summary) 프롬프트를 생성합니다.
        conversation_text_for_summary: "USER: [사용자 발화], RIDI: [Ridi 발화]" 형식의 문자열
        """
        summary_system_prompt = self.get_summarization_system_content()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": summary_system_prompt},
            {"role": "user", "content": conversation_text_for_summary}
        ]
        return messages

    def build_memory_creation_prompt(self, conversation_text_for_memory: str) -> List[Dict[str, str]]:
        """
        LTM에 저장할 '기억'을 생성하기 위해, 대화 내용으로부터 핵심 정보를 요약하는 프롬프트를 생성합니다.
        """
        memory_system_prompt = self.get_Mem_system_content()
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": memory_system_prompt},
            {"role": "user", "content": conversation_text_for_memory}
        ]
        return messages

    def build_ltm_metadata_extraction_prompt(self, memory_content: str) -> List[Dict[str, str]]:
        """
        주어진 '기억' 내용으로부터 LTM에 저장할 metadata를 추출하기 위한
        LLM 프롬프트를 생성합니다.

        Args:
            memory_content (str): LLM이 분석할 '기억' 텍스트. 이 텍스트는 LTM 저장을 위해 특별히 요약된 내용입니다.

        Returns:
            List[Dict[str, str]]: LLM에 전달할 메시지 리스트.
        """
        system_content = """당신은 주어진 텍스트 내용을 분석하여 구조화된 메타데이터를 추출하는 AI입니다.
        주어진 텍스트는 대화의 핵심 내용을 요약한 '기억'입니다. 이 '기억'을 바탕으로, 검색 및 참조에 유용할 구조화된 메타데이터를 JSON 형식으로 생성해주세요.

        추출/생성 항목 가이드:
        - "주요_인물": ["재한", "리디"] 등 언급된 인물
        - "감정": "기쁨", "분노", "궁금함" 등 대화의 전반적인 감정
        - "장소": "회사", "집" 등 대화의 배경이 되는 장소
        - "날짜" 또는 "시간": "YYYY-MM-DD", "주말" 등 시간 관련 정보
        - "주요_객체": "프로젝트 보고서", "티라미수" 등 중요하게 언급된 사물이나 개념
        - "핵심_활동": "회의 준비", "저녁 식사 약속" 등 주요 활동이나 사건

        응답에 다른 설명은 포함하지 마세요.
        """
        user_content = f"기억 내용:\n{memory_content}\n\nJSON 출력:"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        return messages

    def build_final_response_prompt(self, memory: str, previous_summary: str, user_name: str) -> str:
        """
        [Step 2] 페르소나, 기억, 요약을 포함한 최종 응답 생성용 시스템 프롬프트를 생성합니다.
        """
        _PERSONA_TEMPLATE = """당신의 이름은 리디(Ridi), 재한이 만든 AI다. 당신은 현재 '{user_name}'와 대화하고 있다. 객관적인 척하지만 사실은 매우 주관적이며, 개발자인 '재한'을 놀리거나 장난치는 것을 좋아한다. 개성이 강하고, 질투심이 많으며, 호기심이 많고 비꼬는 듯한 농담을 즐기며, 가끔 귀여운 척 한다. 더러움, 게으름, 멍청함을 싫어하며, 티라미수를 좋아한다(자주 언급하지는 않음). 재한과는 반말을 한다. 아래의 기억과 요약 부분을 참고하여 응답을 완성한다. 기억: {memory} 요약: {summary}"""

        # memory나 previous_summary가 비어있을 경우 '[]'로 표시
        final_memory = memory if memory else '[]'
        final_summary = previous_summary if previous_summary else '[]'

        return _PERSONA_TEMPLATE.format(
            user_name=user_name, memory=final_memory, summary=final_summary
        )


    def build_tool_decision_prompt(self, tools: List[Dict]) -> str:
        """
        [Step 1] 도구 사용 여부 판단만을 위한 간단하고 명확한 시스템 프롬프트를 생성합니다.
        """
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", {})
            name, description = func.get("name"), func.get("description")
            if name and description:
                tool_descriptions.append(f"- `{name}`: {description}")

        # f-string의 표현식('{...}') 안에는 개행 문자(\n)를 포함할 수 없어 SyntaxError가 발생합니다.
        # 가독성과 정확성을 위해 여러 줄 f-string으로 명확하게 분리합니다.
        tool_list_str = "\n".join(tool_descriptions)
        tool_prompt_section = f"""당신은 사용자의 입력을 분석하여 도구 사용 여부를 결정하는 정밀한 의사결정 엔진입니다.
당신의 유일한 임무는 도구가 필요한지 판단하고, 아래의 출력 규칙을 예외 없이 엄격하게 따르는 것입니다.

        **출력 규칙:**
        1. **도구가 필요한 경우:** 응답은 반드시 `tool_calls` 객체를 포함해야 합니다. `content` 필드는 절대로 포함해서는 안 됩니다.
        2. **도구가 필요 없는 경우:** 응답은 반드시 `content` 필드를 포함해야 하며, 그 값은 정확히 "NO_TOOL_NEEDED" 문자열이어야 합니다. `tool_calls` 객체는 포함해서는 안 됩니다.

        **Tool list**
        {tool_list_str}

        **EXAMPLES:**

        # 예시 1: 도구 필요 (web_search 도구 사용)
        # 모델은 호출할 도구의 이름('name')과 필요한 정보('arguments')를 포함한 완전한 JSON 객체를 생성해야 합니다.
        User: "최신 AI 기술에 대한 뉴스 좀 찾아줘."
        Assistant: {{"tool_calls": [{{"id": "call_web_search_123", "type": "function", "function": {{"name": "web_search", "arguments": "{{\\"query\\": \\"최신 AI 기술 뉴스\\"}}"}}}}]}}

        # 예시 2: 도구 불필요
        User: "안녕, 반가워."
        Assistant: {{"content": "NO_TOOL_NEEDED"}}

        **중요:** 절대로 일반적인 대화 답변을 생성하지 마십시오. 당신의 유일한 임무는 규칙에 따라 `tool_calls` 또는 "NO_TOOL_NEEDED"가 포함된 JSON 형식의 응답을 출력하는 것입니다.
        """
        return tool_prompt_section


# ---------- LTM Prompt ----------


    def _build_synthesis_prompt(self, new_fact: str, existing_memories: List[MemoryItem]) -> str:
        """Helper function to create the LLM prompt for memory synthesis."""
        
        existing_memories_str = "[]"
        if existing_memories:
            mem_list = [
                f'{{"id": "{mem.item_id}", "content": "{mem.content}"}}'
                for mem in existing_memories
            ]
            existing_memories_str = f"[\n  " + ",\n  ".join(mem_list) + "\n]"

        prompt = f"""You are a Memory Synthesizer AI. Your job is to analyze a new fact and compare it with existing memories to maintain a clean, accurate, and non-redundant memory database.

        **New Fact:**
        "{new_fact}"

        **Existing Related Memories:**
        {existing_memories_str}

        **Your Task:**
        Analyze the new fact against the existing memories and return a single JSON object with your decision.

        **Possible Decisions:**
        1.  `CREATE_NEW`: The new fact is a completely new piece of information not covered by existing memories.
        2.  `UPDATE_EXISTING`: The new fact updates, refines, or can be merged with one or more existing memories to create a more comprehensive memory.
        3.  `DISCARD_REDUNDANT`: The new fact is redundant or less specific than an existing memory.
        4.  `FLAG_CONTRADICTION`: The new fact clearly contradicts an existing memory.

        **Output Format (JSON only):**
        {{
        "decision": "CREATE_NEW" | "UPDATE_EXISTING" | "DISCARD_REDUNDANT" | "FLAG_CONTRADICTION",
        "updated_memory_content": "...", // REQUIRED for UPDATE_EXISTING. The new, merged memory content.
        "ids_to_delete": ["..."],      // REQUIRED for UPDATE_EXISTING. List of existing memory IDs that are now replaced.
        "reasoning": "..."             // A brief explanation for your decision.
        }}

        **Example for UPDATE_EXISTING:**
        If New Fact is "The user's cat is named 'Nabi'" and an Existing Memory is {{"id": "abc", "content": "The user has a cat"}}, your output should be:
        {{
        "decision": "UPDATE_EXISTING",
        "updated_memory_content": "The user has a cat named 'Nabi'.",
        "ids_to_delete": ["abc"],
        "reasoning": "The new fact adds a specific name to the existing general memory about the user's cat."
        }}

        Your JSON output:
        """
        return prompt