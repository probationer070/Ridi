import json
import logging
import os
from korean_text_analyzer import MiniKoNLPTokenizer # MiniKoNLPTokenizer 임포트

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

def extract_keywords_from_text_file(input_text_file_path: str, output_json_file_path: str, tokenizer_dict_path: str = None):
    """
    텍스트 파일에서 명사, 동사, 형용사를 추출하여 JSON 파일로 저장합니다.

    Args:
        input_text_file_path (str): 분석할 텍스트 파일 경로.
        output_json_file_path (str): 추출된 키워드를 저장할 JSON 파일 경로.
        tokenizer_dict_path (str, optional): MiniKoNLPTokenizer가 사용할 사전 파일 경로.
    """
    if not os.path.exists(input_text_file_path):
        logger.error(f"입력 텍스트 파일이 존재하지 않습니다: {input_text_file_path}")
        return

    try:
        with open(input_text_file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        logger.error(f"텍스트 파일을 읽는 중 오류 발생: {e}")
        return

    # MiniKoNLPTokenizer 인스턴스 생성
    # 사전 파일 경로를 명시적으로 전달하거나, MiniKoNLPTokenizer의 기본 경로를 사용하도록 할 수 있습니다.
    # 여기서는 korean_text_analyzer.py와 같은 위치에 mini_konlp_dict.json이 있다고 가정합니다.
    if tokenizer_dict_path is None:
        # korean_text_analyzer.py 파일의 위치를 기준으로 사전 파일 경로를 추정
        analyzer_dir = os.path.dirname(os.path.abspath(__file__)) # 현재 스크립트 파일의 디렉토리
        default_dict_name = MiniKoNLPTokenizer.DEFAULT_DICT_FILENAME
        tokenizer_dict_path = os.path.join(analyzer_dir, default_dict_name)


    if not os.path.exists(tokenizer_dict_path):
        logger.warning(f"MiniKoNLPTokenizer 사전 파일({tokenizer_dict_path})을 찾을 수 없습니다. 빈 사전으로 동작합니다.")
        # 이 경우, MiniKoNLPTokenizer는 사전에 의존하지 않는 매우 기본적인 fallback 로직만 수행하게 됩니다.
        # 또는, 여기서 에러를 발생시켜 실행을 중단할 수도 있습니다.

    tokenizer = MiniKoNLPTokenizer(dict_json_path=tokenizer_dict_path)
    
    # 텍스트 토큰화
    tagged_tokens = tokenizer.tokenize(text_content)

    nouns = set()
    verb_stems = set()
    adj_stems = set()

    for item in tagged_tokens:
        token = item.get("token")
        pos = item.get("pos")
        if token:
            if pos == "Noun":
                nouns.add(token)
            elif pos == "Verb":
                verb_stems.add(token) # MiniKoNLPTokenizer는 이미 어간을 반환하려고 시도
            elif pos == "Adjective":
                adj_stems.add(token) # MiniKoNLPTokenizer는 이미 어간을 반환하려고 시도

    output_data = {
        "nouns": sorted(list(nouns)),
        "verb_stems": sorted(list(verb_stems)),
        "adj_stems": sorted(list(adj_stems))
    }

    try:
        output_dir = os.path.dirname(output_json_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"출력 디렉토리가 생성되었습니다: {output_dir}")

        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"추출된 키워드가 {output_json_file_path} 파일에 저장되었습니다.")
        logger.info(f"추출된 명사 수: {len(nouns)}, 동사 어간 수: {len(verb_stems)}, 형용사 어간 수: {len(adj_stems)}")
    except Exception as e:
        logger.error(f"JSON 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    # 입력 텍스트 파일 경로 (니체_차라투스트라.txt)
    # 이 스크립트 파일(extract_nietzsche_keywords.py)과 같은 위치에 있다고 가정
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_script_dir, "니체_차라투스트라.txt")
    
    # 출력 JSON 파일 경로 (예: nietzsche_keywords.json)
    output_file = os.path.join(current_script_dir, "nietzsche_keywords.json")

    # MiniKoNLPTokenizer가 사용할 사전 파일 경로 (선택적, None이면 기본값 사용)
    # korean_text_analyzer.py와 같은 위치에 mini_konlp_dict.json이 있다고 가정
    dict_file = os.path.join(current_script_dir, MiniKoNLPTokenizer.DEFAULT_DICT_FILENAME)

    if not os.path.exists(input_file):
        print(f"오류: 입력 파일 '{input_file}'을 찾을 수 없습니다.")
    elif not os.path.exists(dict_file):
         print(f"경고: MiniKoNLP 사전 파일 '{dict_file}'을 찾을 수 없습니다. 토크나이저 성능이 저하될 수 있습니다.")
         # 사전 파일이 없어도 실행은 되도록 함 (MiniKoNLPTokenizer 내부에서 처리)
         extract_keywords_from_text_file(input_file, output_file, tokenizer_dict_path=dict_file)
    else:
        extract_keywords_from_text_file(input_file, output_file, tokenizer_dict_path=dict_file)

