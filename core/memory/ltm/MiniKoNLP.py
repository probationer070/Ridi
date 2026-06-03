import logging
from collections import Counter
from typing import List, Dict, Tuple, Union, Optional # Union 추가
import os # 파일 경로를 위해 추가
import re # 숫자 판별용
import json # 사전 파일 로드를 위해 추가

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None # networkx가 없을 경우 nx를 None으로 설정

logger = logging.getLogger(__name__)

# --- Mini Korean NLP Tokenizer ---
class MiniKoNLPTokenizer:
    DEFAULT_DICT_FILENAME = "keywords.json"

    def __init__(self, dict_json_path: Optional[str] = None):
        if dict_json_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.dict_path = os.path.join(current_dir, self.DEFAULT_DICT_FILENAME)
        else:
            self.dict_path = dict_json_path

        self.nouns = set()  # 명사
        self.verb_stems = set() # 동사 어간
        self.adj_stems = set()  # 형용사 어간
        self.pronouns = set() # 대명사
        self.numerals = set() # 수사
        self.adverbs = set() # 부사
        self.determiners = set() # 관형사
        self.exception_adverbs = {"가장", "제일", "아주", "너무", "매우", "무척", "정말", "진짜", "빨리", "어서"} # 예외 처리할 부사

        self._load_dictionary()
        
        # 일반적인 어미 및 조사 (길이 역순으로 정렬하여 긴 것부터 매칭 시도)
        # 종결형, 연결형, 시제, 높임, 명령/청유 어미 등 확장
        _common_endings = [
            "습니다", "습니다만", "습니다요", "ㅂ니다", "ㅂ니다만", "ㅂ니다요", "어요", "아요", "여요", "었", "았", "였", "겠다", "겠", "었다", "았다", "였다",
            "가", "고", "니", "면", "서", "며", "네", "다", "자", "지", "죠", "든", "든지", "랴", "마", "ㅂ시오", "습니다까", "ㅂ니까",
            "은가", "는가", "나", "을까", "ㄹ까", "을걸", "ㄹ걸", "는구나", "구나", "더라", "을지", "ㄹ지", "렴", "으리", "리",
            "시오", "소서", "십시오", "으십시오", "ㅂ시다", "읍시다", "세", "소서", "거라", "너라", "아라", "어라", "여라",
            "면서", "거나", "지만", "는데", "은데", "ㄴ데", "다가", "어도", "아도", "여도", "어서", "아서", "여서",
            "으니까", "니까", "으려", "려", "으려고", "려고", "고자", "으러", "러", "게", "도록", "자마자", "을수록", "ㄹ수록",
            "던", "음", "ㅁ", "기", "했다", "했었다", "했었네", "했었지", "했었고", "했었으며", "했었지만", "했었는데", "했었으니", "했었으므로",
            "었네", "었죠", "었다", "했소", "ㅂ다", "운", "ㄴ", "는지", "ㄴ지", "을지", "ㄹ지", "이든지", "든지",
            # 복합 어미 (선어말 어미 + 어말 어미)
            "셨습니다", "셨어요", "셨네", "셨다", "시었다", "시었고", "시겠", "시니", "시면", "시지",
            "었네", "었죠", "었다", "었고", "었으며", "었지만", "었는데", "었으니", "었으므로",
            "겠네", "겠다", "겠고", "겠으며", "겠지만", "겠는데", "겠으니", "겠으므로",
            "시어요", "시옵고", "사옵고", "시옵니다", "시옵니까", "시옵니까요", "시옵니까만", "주세요", "주십시오",
            "주시오", "이죠", "죠", "입니다", "이다", "이였다", "이고요", "거라", "너라", "아라", "어라", "여라",
            # 추가적인 어미 (요청 기반)
            "한테요", "한테서요", "한테는요", "한테도요", "한테만요", "한테서도요", "한테서만요",
            "한테까지요", "한테부터요", "한테로요", "한테로부터요", "한테로서요", "한테로서도요",
        ]
        self.verb_endings = sorted(list(set(_common_endings)), key=len, reverse=True)
        self.adj_endings = self.verb_endings # 형용사 어미도 유사하게 처리 (세부 조정 가능)
        
        # 명사와 동사/형용사를 구분하기 위해 조사와 어미를 재분류합니다.
        # 1. 명사 판별 우선순위가 높은 조사/어미 (주로 서술격 조사 기반)
        # 이들은 선행 어근이 사전에 없더라도 명사로 강하게 추정할 수 있게 합니다.
        noun_exclusive_indicators = [
            "입니다", "입니까", "이었습니다", "이었고", "이었지만", "이었는데",
            "이고", "이며", "이지만", "인데", "이라서", "이므로", "이든지", 
            "이여", "시여", "이시여", "이죠", "이지요", "이네", "이군", "이구나", "이야", "이어요",
            "인가요", "인가", "께서", "께서도", "께서는", "처럼", "만큼", "이야말로", "야말로",
            "이랑", "하고", # '하고'는 동사도 있지만, 조사로 쓰일 때 명사 뒤에 옴.
            "됩니다", "되었어요", "되었네", "되었다", "되었고", "되었으며", "되었지만", "되었는데", "되었으니", "되었으므로",
            "되겠네", "되겠다", "되겠고", "되겠으며", "되겠지만", "되겠는데", "되겠으니", "되겠으므로",
            "되니", "되면", "되자", "되지는", "되며", "되네", "되죠", "됩니다요", "되시오",
        ]
        self.noun_exclusive_indicators = sorted(list(set(noun_exclusive_indicators)), key=len, reverse=True)

        # 2. 명사, 용언 어간 등 다양한 어근에 붙을 수 있는 일반적인 조사 (모호성 존재)
        # 주격(이/가), 목적격(을/를), 보조사(은/는/도/만) 등
        # 이 조사를 제거한 후에는 어근이 사전에 있는지 반드시 확인해야 합니다.
        self.general_particles = sorted(list(set([
            "에게는", "에게도", "에게만", "에게서", "로부터", "에서부터", "까지도", "만이라도", "이라도", "보다는",
            "까지", "부터", "조차", "마저", "밖에", "보다", "같이",
            "에게", "에서", "으로", "의", "에", "와", "과", "도", "만", "뿐", "한테", "로",
            "은", "는", "을", "를", "이", "가", "야", "을게", "를게", "이나", "나",
            "시킬", "이라고", "했던", "해", "겠냐고", "했다", "다며", "질", "킨",
        ])), key=len, reverse=True)


    def _load_dictionary(self):
        """JSON 파일에서 사전 데이터를 로드합니다."""
        if os.path.exists(self.dict_path):
            try:
                with open(self.dict_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.nouns = set(data.get("nouns", []))
                    self.verb_stems = set(data.get("verb_stems", []))
                    self.adj_stems = set(data.get("adj_stems", []))
                    self.pronouns = set(data.get("pronouns", [])) 
                    self.numerals = set(data.get("numerals", []))
                    self.adverbs = set(data.get("adverbs", []))
                    self.determiners = set(data.get("determiners", []))
                # logger.info(f"Dictionary loaded from {self.dict_path}. Nouns: {len(self.nouns)}, Verb Stems: {len(self.verb_stems)}, Adj Stems: {len(self.adj_stems)}, Pronouns: {len(self.pronouns)}, Numerals: {len(self.numerals)}, Adverbs: {len(self.adverbs)}, Determiners: {len(self.determiners)}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from {self.dict_path}. Using empty dictionaries.")
            except Exception as e:
                logger.error(f"Error loading dictionary from {self.dict_path}: {e}. Using empty dictionaries.")
        else:
            logger.warning(f"Dictionary file not found at {self.dict_path}. Tokenizer will operate with empty dictionaries.")

    def _get_stem(self, word: str, endings: List[str], stems_dict: set, pos_tag: str) -> Tuple[str, str]:
        """
        단어에서 어미를 제거하고 어간을 추정합니다.
        성공 시 (어간, 품사태그) 반환, 실패 시 (원본단어, "Unknown") 반환.
        """
        original_word = word
        for ending in endings:
            if word.endswith(ending):
                stem = word[:-len(ending)]
                if not stem: continue # 어미 제거 후 아무것도 남지 않으면 다음 어미 시도
                
                if stem in stems_dict:
                    return stem, pos_tag
                if stem + "다" in stems_dict: 
                    return stem, pos_tag
                # 과거 시제 선어말 어미 '었/았/였' 처리 시도
                if len(stem) > 0 and stem[-1] in ['었', '았', '였']:
                    simpler_stem = stem[:-1]
                    if simpler_stem in stems_dict: return simpler_stem, pos_tag
                    if simpler_stem + "다" in stems_dict: return simpler_stem, pos_tag

        # --- 보조 동사 '주다' 포함 어미 처리 ---
        result = self._handle_auxiliary_juda(original_word, stems_dict, pos_tag)
        if result: return result

        # --- 불규칙 활용 처리 (어미 제거 후 어간이 변형된 경우) ---
        # 각 불규칙 활용 함수는 (어간, 품사태그) 또는 None을 반환합니다.
        # None이 아닌 값을 받으면 해당 어간과 품사를 반환합니다.

        # 'ㅂ' 불규칙 (덥다 -> 더워, 고맙다 -> 고마워)
        result = self._handle_b_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # 'ㄷ' 불규칙 (걷다 -> 걸어, 듣다 -> 들어)
        result = self._handle_d_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # '르' 불규칙 (모르다 -> 몰라, 빠르다 -> 빨라)
        result = self._handle_reu_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # 'ㅅ' 불규칙 (낫다 -> 나아, 붓다 -> 부어)
        result = self._handle_s_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # 'ㅎ' 불규칙 (빨갛다 -> 빨개, 빨간)
        result = self._handle_h_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # 'ㅜ' 불규칙 (푸다 -> 퍼)
        result = self._handle_u_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # 'ㅡ' 탈락 불규칙 (쓰다 -> 써, 예쁘다 -> 예뻐)
        result = self._handle_eu_dropping(original_word, stems_dict, pos_tag)
        if result: return result

        # '러' 불규칙 (이르다 -> 이르러)
        result = self._handle_reo_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # '여' 불규칙 (하다 -> 하여/해)
        result = self._handle_yeo_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # '오' 불규칙 (다오 -> 달라) - 매우 드물지만 예외 처리
        result = self._handle_o_irregular(original_word, stems_dict, pos_tag)
        if result: return result

        # 'ㄹ' 탈락 불규칙 (살다 -> 사니, 사오)
        result = self._handle_l_dropping(original_word, stems_dict, pos_tag)
        if result: return result

        # 어간 추출에 실패했더라도, 원본 단어를 해당 품사로 반환 (선택적: 키워드 추출 시 원형 보존)
        # 이렇게 하면 "웃었다" 같은 단어가 "Verb"로 태깅되고,
        # MiniKoNLPTokenizer의 결과에서 "token" 필드에 원본 단어가 들어가게 됩니다.
        # TextRank에서는 item['original']을 사용하므로, 이 변경은 MiniKoNLPTokenizer의
        # "token" 필드 자체를 원본으로 유지시켜 일관성을 높일 수 있습니다.
        # 어미/불규칙 활용으로 어간을 찾지 못하면 Unknown으로 반환
        return original_word, "Unknown"

    def _handle_auxiliary_juda(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'주다' 보조 동사 포함 어미 처리 (예: 기다려주세요 -> 기다리)"""
        # '주'로 시작하는 일반적인 어미 패턴 (길이 역순 정렬)
        # 더 많은 어미 추가 가능
        juda_endings = sorted([
            "주세요", "주다", "줄게", "주었", "주겠", "주니", "주면", "주자", "주지", "주며", "주네",
            "주십시오", "주소서", "주시오", # 높임
            "주어서", "주어서도", "주어서는", # 연결형
            "주니까", "주므로", # 이유/원인
            "주었다", "주었습니다", # 과거 시제
            "주겠다", "주겠습니다", # 미래 시제/추측
            "주느냐", "주냐", # 의문형
            "주세요", "주십시오", # 명령형
            "주자", "주ㅂ시다", # 청유형
            "주는", "준", "줄", "주던", # 관형사형
            "줌", "주기", # 명사형
        ], key=len, reverse=True)

        for ending in juda_endings:
            if word.endswith(ending):
                # '주' 앞 부분 (예: "기다려", "먹어", "하", "가져", "몰라", "고마워", "들어", "걸어")
                part_before_juda = word[:-len(ending)]
                if not part_before_juda: continue # '주세요' 자체가 단어인 경우 등

                # '주' 앞 부분이 '-아', '-어', '-여' 또는 불규칙 변형으로 끝나는지 확인
                stem_candidate = None
                if part_before_juda.endswith("아"): stem_candidate = part_before_juda[:-1]
                elif part_before_juda.endswith("어"): stem_candidate = part_before_juda[:-1]
                elif part_before_juda.endswith("여"): stem_candidate = part_before_juda[:-1]
                elif part_before_juda.endswith("라"): stem_candidate = part_before_juda[:-1] + "르" # 르 불규칙 (몰라 -> 모르)
                elif part_before_juda.endswith("러"): stem_candidate = part_before_juda[:-1] + "르" # 르 불규칙 (흘러 -> 흐르)
                elif part_before_juda.endswith("와"): stem_candidate = part_before_juda[:-1] + "ㅂ" # ㅂ 불규칙 (고마와 -> 고맙)
                elif part_before_juda.endswith("워"): stem_candidate = part_before_juda[:-1] + "ㅂ" # ㅂ 불규칙 (더워 -> 덥)
                elif part_before_juda.endswith("져"): stem_candidate = part_before_juda[:-1] + "ㅣ" # 가지다 (가져 -> 가지)
                # 'ㄷ' 불규칙 (듣다 -> 들어, 걷다 -> 걸어) - '어'로 끝나는 경우에 포함되므로 별도 처리 불필요

                # 추정된 어간이 사전에 있는지 확인
                if stem_candidate and (stem_candidate in stems_dict or stem_candidate + "다" in stems_dict):
                    return stem_candidate, pos_tag

        return None # '주다' 보조 동사 패턴으로 어간을 찾지 못함

    def _check_endwith(self, word: str, filter_list: List[str]) -> bool:
        """단어가 filter_list에 있는 접미사로 끝나는지 확인합니다."""
        for ending in filter_list:
            if word.endswith(ending):
                return True
        return False
    
    def _clean_word_for_processing(self, text: str) -> str:
        """단어의 양 끝 공백을 제거하고, 후행 구두점을 제거합니다."""
        if not text:
            return ""
        word_to_clean = text.strip()
        trailing_punctuations = '.,!?;:)' # Add more if needed
        cleaned_changed = True
        while cleaned_changed and word_to_clean:
            cleaned_changed = False
            if word_to_clean[-1] in trailing_punctuations:
                word_to_clean = word_to_clean[:-1]
                cleaned_changed = True
        return word_to_clean
    
    def _handle_b_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'ㅂ' 불규칙 처리 (덥다 -> 더워, 고맙다 -> 고마워)"""
        if self._check_endwith(word, ["워", "와", "우니"]) and len(word) > 1:
            # '더워' -> '더' + 'ㅂ' = '덥'
            # '고와' -> '고' + 'ㅂ' = '곱'
            potential_stem = word[:-1] + "ㅂ"
            if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                return potential_stem, pos_tag
        # '아름다운' -> '아름답' 같은 경우 (관형사형 '운' + 'ㅂ' 불규칙)
        if word.endswith("운") and len(word) > 1:
            # "아름다운" -> "아름다" + "ㅂ" = "아름답"
            potential_stem_from_un = word[:-1] + "ㅂ"
            if potential_stem_from_un in stems_dict or potential_stem_from_un + "다" in stems_dict:
                return potential_stem_from_un, pos_tag
        return None

    def _handle_d_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'ㄷ' 불규칙 처리 (걷다 -> 걸어, 듣다 -> 들어)"""
        # '어/아' 계열 어미 앞에서 'ㄷ'이 'ㄹ'로 바뀜
        common_d_endings = ["어", "아", "으니", "어서", "아서", "은들", "을"]
        for ending in sorted(common_d_endings, key=len, reverse=True):
            if self._check_endwith(word, [ending]): 
                stem_cand_l = word[:-len(ending)] # 예: "걸", "들"
                if stem_cand_l and stem_cand_l.endswith("ㄹ"):
                    potential_stem = stem_cand_l[:-1] + "ㄷ" # 예: "걷", "듣"
                    if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                        return potential_stem, pos_tag
        return None

    def _handle_reu_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'르' 불규칙 처리 (모르다 -> 몰라, 빠르다 -> 빨라)"""
        # '아/어' 계열 어미 앞에서 '르'의 'ㅡ'가 탈락하고 'ㄹ'이 덧생김
        common_reu_endings = ["라", "러", "라서", "러서"]
        for ending in sorted(common_reu_endings, key=len, reverse=True):
            if self._check_endwith(word, [ending]):
                stem_cand_l = word[:-len(ending)] # 예: "몰", "빨"
                if stem_cand_l and stem_cand_l.endswith("ㄹ"):
                    potential_stem = stem_cand_l[:-1] + "르" # 예: "모르", "빠르"
                    if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                        return potential_stem, pos_tag
        return None

    def _handle_s_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'ㅅ' 불규칙 처리 (낫다 -> 나아, 붓다 -> 부어)"""
        # 모음 어미 앞에서 'ㅅ'이 탈락
        if self._check_endwith(word, ["아", "어", "여"]) and len(word) > 1:
             # '하' 불규칙 ('하여'/'해')와 구분 필요
            if self._check_endwith(word, ["여"]) and (word[:-1]+"하다" in stems_dict or word[:-1]+"하" in stems_dict) :
                return None # '하' 불규칙으로 넘김
            potential_stem = word[:-1] + "ㅅ" # 예: "나아" -> "나" + "ㅅ" = "낫"
            if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                return potential_stem, pos_tag
        return None

    def _handle_h_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'ㅎ' 불규칙 처리 (빨갛다 -> 빨개, 빨간)"""
        # 어간 끝 'ㅎ'이 어미 '-ㄴ', '-ㅁ', '-ㄹ', '-아/-어' 앞에서 변형/탈락
        # 'ㅐ', 'ㅔ'로 끝나는 경우 (예: 빨개, 파래)
        if self._check_endwith(word, ["개", "애", "게", "예"]) and len(word) > 1:
             # '빨개' -> '빨가' + 'ㅎ' = '빨갛'
             potential_stem = word[:-1] + "하" # 단순화된 처리
             if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                 return potential_stem, pos_tag
        # 'ㄴ', 'ㅁ', 'ㄹ' 앞에서 'ㅎ' 탈락 (예: 빨간, 빨감, 빨갈)
        if self._check_endwith(word, ["ㄴ", "ㅁ", "ㄹ"]) and len(word) > 1:
             potential_stem = word + "ㅎ" # 예: "빨간" -> "빨간" + "ㅎ" = "빨갛"
             if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                 return potential_stem, pos_tag
        return None

    def _handle_u_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'ㅜ' 불규칙 처리 (푸다 -> 퍼)"""
        # 어간 'ㅜ'가 어미 '-어' 앞에서 탈락
        if word.endswith("어") and len(word) > 1: # "퍼"
            potential_stem = word[:-1] + "우" # "퍼" -> "푸"
            if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                return potential_stem, pos_tag
        return None

    def _handle_eu_dropping(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'ㅡ' 탈락 불규칙 (쓰다 -> 써, 예쁘다 -> 예뻐)"""
        # '아/어' 계열 어미 앞에서 어간 끝 '으'가 탈락
        if self._check_endwith(word, ["어", "아", "여"]) and len(word) > 1:
            # '하' 불규칙 ('하여'/'해')와 구분 필요
            if self._check_endwith(word, ["여"]) and (word[:-1]+"하다" in stems_dict or word[:-1]+"하" in stems_dict) :
                 return None # '하' 불규칙으로 넘김
            potential_stem = word[:-1] + "으" # 예: "써" -> "스" + "으" -> "쓰"
            if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                return potential_stem, pos_tag
        return None

    def _handle_reo_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'러' 불규칙 처리 (이르다 -> 이르러)"""
        # 어간 '르' 뒤에 어미 '-어'가 올 때, 어미가 '-러'로 바뀜
        if self._check_endwith(word, ["르러"]) and len(word) > 2 : # "이르러", "푸르러"
            potential_stem = word[:-2] + "르" # "이르러" -> "이르"
            if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                return potential_stem, pos_tag
        return None

    def _handle_yeo_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'여' 불규칙 처리 (하다 -> 하여/해)"""
        if self._check_endwith(word, ["해"]) and len(word) > 1: # 예: 공부해
            potential_stem = word[:-1] + "하" # "공부하"
            if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                return potential_stem, pos_tag
        elif self._check_endwith(word, ["여"]) and len(word) > 1: # 예: 공부하여
            if self._check_endwith(word, ["하여"]) and len(word) > 2:
                 potential_stem = word[:-2] + "하" # "공부하"
                 if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                     return potential_stem, pos_tag
        return None

    def _handle_o_irregular(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'오' 불규칙 처리 (다오 -> 달라) - 매우 드물지만 예외 처리"""
        # '달다' 동사의 명령형 '다오'가 '달라'로 변하는 경우
        if word == "달라":
            potential_stem = "달"
            if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                return potential_stem, pos_tag
        return None

    def _handle_l_dropping(self, word: str, stems_dict: set, pos_tag: str) -> Optional[Tuple[str, str]]:
        """'ㄹ' 탈락 불규칙 (살다 -> 사니, 사오, 사는)"""
        # 어간 끝소리 'ㄹ'이 'ㄴ', 'ㄹ', 'ㅂ', '오', '시' 앞에서 사라지는 활용 형식
        # 이 불규칙은 어미가 특정 자음으로 시작할 때 어간이 변하는 경우입니다.
        # 현재 _get_stem의 어미 제거 로직에서 이미 처리될 가능성이 높지만,
        # 명시적으로 처리할 수도 있습니다. 여기서는 어미 제거 후 어간이 'ㄹ'이 없는 형태로 남았을 때
        # 'ㄹ'을 붙여서 원형을 추정하는 방식으로 구현합니다.
        # 예: '사니' -> 어미 '니' 제거 -> '사'. '사' + 'ㄹ' = '살'. '살다'가 사전에 있는지 확인.
        common_l_dropping_endings = ["니", "오", "는", "ㅂ니다", "ㅂ시다", "세요"] # 'ㄹ' 앞에서 탈락하는 어미 시작 자음 포함
        for ending in sorted(common_l_dropping_endings, key=len, reverse=True):
             if word.endswith(ending):
                 stem_candidate = word[:-len(ending)] # 예: "사"
                 potential_stem = stem_candidate + "ㄹ" # 예: "살"
                 if potential_stem in stems_dict or potential_stem + "다" in stems_dict:
                     return potential_stem, pos_tag
        return None

    def _strip_particles(self, word: str) -> Tuple[str, str]:
        """단어에서 조사를 제거하고 명사를 추정합니다."""
        original_word = word
        # 명사, 용언 등에 모두 붙을 수 있는 일반적인 조사를 확인합니다.
        for particle in self.general_particles:
            if word.endswith(particle):
                noun_candidate = word[:-len(particle)]
                if not noun_candidate: continue # 조사 제거 후 아무것도 남지 않으면 다음 조사 시도

                # Check if the candidate is in any relevant dictionary after stripping a particle
                # Prioritize known word types
                # 1. Pronoun 사전에 있는지 확인
                if noun_candidate in self.pronouns:
                    return noun_candidate, "Pronoun"
                # 2. Numeral 사전에 있는지 확인
                elif noun_candidate in self.numerals:
                    return noun_candidate, "Numeral"
                # 3. 명사 사전에 있는지 확인
                if noun_candidate in self.nouns:
                    return noun_candidate, "Noun"
                # 4. 형용사 어간 사전에 있는지 확인 (조사 뒤에 형용사 어간이 오는 경우는 드물지만 가능성은 있음)
                elif noun_candidate in self.adj_stems:
                    return noun_candidate, "Adjective"
                # 5. 동사 어간 사전에 있는지 확인 (조사 뒤에 동사 어간이 오는 경우는 드물지만 가능성은 있음)
                # 예: "먹는가" -> "먹는" + "가". "먹는"은 동사 어미가 붙은 형태. 이 경우는 _get_stem에서 처리되어야 함.
                elif noun_candidate in self.verb_stems:
                    return noun_candidate, "Verb"
                # 5. 위 사전에 모두 없고 조사가 분리되었다면, 명사로 추정 (기존 fallback 로직 유지)
                return noun_candidate, "Noun"
        return original_word, "Unknown"

    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """텍스트를 토큰화하고 품사를 태깅합니다."""
        results = []
        words = text.split() 
        i = 0
        while i < len(words):
            original_word = words[i]
            word = self._clean_word_for_processing(original_word)

            if not word: # 빈 단어는 건너뜀
                i += 1
                continue

            token, pos = word, "Unknown"
            found = False # 이번 단어의 품사를 찾았는지 여부

            # --- 명사/동사/형용사 구분을 위한 새로운 판별 로직 ---

            # 1. 명사로 강하게 추정되는 패턴 우선 확인 (서술격 조사 등)
            for p in self.noun_exclusive_indicators:
                if word.endswith(p):
                    stem = word[:-len(p)]
                    if stem:
                        token, pos = stem, "Noun"
                        found = True
                        break
            if found:
                results.append({"token": token, "pos": pos, "original": original_word})
                i += 1
                continue

            # 2. 숫자 + 명사 조합 확인 (기존 로직 유지)
            is_current_word_number = re.match(r'^\d+(\.\d+)?$', word) is not None
            if is_current_word_number and i + 1 < len(words):
                next_original_word = words[i+1]
                next_word = self._clean_word_for_processing(next_original_word)
                if next_word:
                    noun_candidate, next_pos_candidate = self._strip_particles(next_word)
                    is_next_word_in_noun_dict = next_word in self.nouns
                    if next_pos_candidate == "Noun" or is_next_word_in_noun_dict:
                        # _strip_particles가 조사를 제거한 어근을 반환하므로, 그것을 사용
                        final_noun_part = noun_candidate if next_pos_candidate == "Noun" else next_word
                        combined_token = word + final_noun_part
                        results.append({"token": combined_token, "pos": "Noun", "original": original_word + " " + next_original_word})
                        logger.debug(f"Combined Number+Noun: '{original_word} {next_original_word}' -> '{combined_token}' (Noun)")
                        i += 2
                        continue

            # 3. 부사/관형사 확인 (단일 단어)
            if word in self.exception_adverbs or word in self.adverbs:
                token, pos = word, "Adverb"
                found = True
            elif word in self.determiners:
                token, pos = word, "Determiner"
                found = True
            if found:
                results.append({"token": token, "pos": pos, "original": original_word})
                i += 1
                continue

            # 4. 동사/형용사 어간 추출 시도
            verb_stem, verb_pos = self._get_stem(word, self.verb_endings, self.verb_stems, "Verb")
            if verb_pos != "Unknown":
                token, pos = verb_stem, verb_pos
                if token and not token.endswith("다"): token += "다"
                found = True
            else: # 동사에서 못 찾았으면 형용사 시도
                adj_stem, adj_pos = self._get_stem(word, self.adj_endings, self.adj_stems, "Adjective")
                if adj_pos != "Unknown":
                    token, pos = adj_stem, adj_pos
                    if token and not token.endswith("다"): token += "다"
                    found = True
            if found:
                results.append({"token": token, "pos": pos, "original": original_word})
                i += 1
                continue

            # 5. 일반 조사(ambiguous) 분리 시도
            stem_candidate, stripped_pos = self._strip_particles(word)
            if stripped_pos != "Unknown":
                token, pos = stem_candidate, stripped_pos
                found = True
            if found:
                results.append({"token": token, "pos": pos, "original": original_word})
                i += 1
                continue

            # 6. 모든 시도 실패 시, 전체 단어가 사전에 있는지 최종 확인
            if word in self.pronouns: pos = "Pronoun"
            elif word in self.nouns: pos = "Noun"
            elif word in self.numerals: pos = "Numeral"
            elif word in self.verb_stems:
                pos = "Verb"
                if not word.endswith("다"): token += "다"
            elif word in self.adj_stems:
                pos = "Adjective"
                if not word.endswith("다"): token += "다"
            elif re.match(r'^\d+(\.\d+)?$', word):
                pos = "Numeral"

            results.append({"token": token, "pos": pos, "original": original_word})
            i += 1

        logger.debug(f"MiniKoNLPTokenizer tokenized: {text} -> {results}")
        return results