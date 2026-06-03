import csv
import json
import os
import re

def filter_and_classify_keywords_from_csv(csv_file_path, output_json_path):
    """
    CSV 파일에서 단어와 품사를 읽어 명사, 동사, 형용사, 수사만 분류하고
    JSON 파일로 저장합니다. 단어에 포함된 숫자는 제거하고,
    동사와 형용사는 어간 형태로 저장하려고 시도합니다.

    Args:
        csv_file_path (str): 입력 CSV 파일 경로.
        output_json_path (str): 출력 JSON 파일 경로.
    """
    classified_data = {
        "nouns": [],
        "verb_stems": [],
        "adj_stems": [],
        "numerals": [],
        "determiners": [],
        "adverbs": [],
        "pronouns": []
    }

    # CSV의 품사 약어를 JSON 키로 매핑
    pos_abbr_map = {
        "명": "nouns",
        "고": "nouns",
        "동": "verb_stems",
        "형": "adj_stems",
        "수": "numerals",
        "관": "determiners",
        "부": "adverbs",
        "대": "pronouns",
        "불": "pronouns",
    }

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None) # 헤더가 있다면 건너뛰기
            if header:
                print(f"CSV Header: {header}")

            for row_number, row in enumerate(reader, 1):
                if len(row) >= 2:
                    original_word = row[0].strip()
                    pos_abbr = row[1].strip()

                    if not original_word: # 단어가 비어있으면 건너뛰기
                        print(f"Skipping empty original_word at row {row_number}.")
                        continue

                    # 단어에서 숫자 제거 (정규 표현식 사용)
                    # 예: "가격03" -> "가격", "가다01" -> "가다"
                    word = re.sub(r'\d+$', '', original_word) # 단어 끝에 붙은 숫자만 제거
                    # 만약 단어 중간에 있는 숫자도 제거하려면 re.sub(r'\d+', '', original_word) 사용

                    if not word: # 숫자 제거 후 단어가 비어있으면 건너뛰기
                        print(f"Skipping original_word '{original_word}' as it became empty after removing numbers at row {row_number}.")
                        continue


                    json_key = pos_abbr_map.get(pos_abbr)

                    if json_key:
                        # 동사/형용사의 경우 '다'로 끝나면 '다'를 제거하여 어간 형태로 저장 시도
                        if json_key in ["verb_stems", "adj_stems"] and word.endswith("다") and len(word) > 1:
                            stem = word[:-1]
                            if stem not in classified_data[json_key]: # 중복 방지
                                classified_data[json_key].append(stem)
                        else:
                            if word not in classified_data[json_key]: # 중복 방지
                                classified_data[json_key].append(word)
                    # else:
                        # print(f"Ignoring POS tag '{pos_abbr}' for word '{original_word}' (processed as '{word}') at row {row_number}.")
                else:
                    print(f"Skipping row {row_number} due to insufficient columns: {row}")

    except FileNotFoundError:
        print(f"Error: CSV 파일 '{csv_file_path}'을(를) 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"CSV 파일 처리 중 오류 발생: {e}")
        return

    # 각 품사 리스트를 알파벳 순으로 정렬 (선택 사항)
    for key in classified_data:
        classified_data[key].sort()

    try:
        with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(classified_data, jsonfile, ensure_ascii=False, indent=2)
        print(f"숫자 제거 및 필터링, 분류된 키워드가 '{output_json_path}' 파일로 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"JSON 파일 저장 중 오류 발생: {e}")

if __name__ == '__main__':
    # --- 설정 ---
    # 입력 CSV 파일 경로 (실제 key.csv 파일이 있는 경로로 수정해주세요)
    # 예: current_project_dir = os.path.dirname(os.path.abspath(__file__))
    # csv_input_path = os.path.join(current_project_dir, "key.csv")
    csv_input_path = "key.csv"  # CSV 파일이 스크립트와 같은 디렉토리에 있다고 가정

    # 출력 JSON 파일 경로
    json_output_path = "filtered_keywords_no_digits.json" # 출력 파일 이름 변경
    # --- ---

    # key.csv 파일이 현재 디렉토리에 있는지 확인
    if not os.path.exists(csv_input_path):
        print(f"입력 파일 '{csv_input_path}'을 찾을 수 없습니다. 경로를 확인해주세요.")
        print(f"스크립트가 있는 디렉토리에 '{csv_input_path}' 파일을 위치시키거나 경로를 수정해주세요.")
    else:
        filter_and_classify_keywords_from_csv(csv_input_path, json_output_path)
