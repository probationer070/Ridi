import os
import json
from typing import List, Dict

def save_conversation_for_dataset(system_prompt: str, conversation_history: List[Dict[str, str]], dataset_file_path: str):
    """
    향후 파인튜닝 데이터셋으로 활용하기 위해, 현재까지의 전체 대화 턴을 JSONL 형식으로 저장합니다.
    한 줄에 하나의 JSON 객체가 저장되며, 각 객체는 시스템 프롬프트와 대화 목록을 포함합니다.
    """
    try:
        # 파일이 저장될 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(dataset_file_path), exist_ok=True)

        # 데이터셋으로 저장할 객체 생성
        dataset_entry = {
            "system": system_prompt,
            "conversations": conversation_history
        }

        # JSONL 파일에 한 줄로 추가 (ensure_ascii=False로 한글 보존)
        with open(dataset_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")

        print(f"데이터셋 항목이 '{dataset_file_path}' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"Error saving conversation for dataset to {dataset_file_path}: {e}")
