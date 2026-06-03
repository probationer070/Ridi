import networkx as nx
from collections import Counter
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TextRankExtractor:
    """
    TextRank 알고리즘을 사용하여 텍스트에서 키워드를 추출하는 클래스입니다.
    초기 버전은 단순 공백 분리 토큰화를 사용합니다.
    """
    def __init__(self,
                 window_size: int = 4,
                 damping_factor: float = 0.85,
                 iterations: int = 100,
                 min_word_length: int = 2):
        """
        TextRank 키워드 추출기 초기화.

        Args:
            window_size (int): 단어 그래프 생성 시 고려할 윈도우 크기.
            damping_factor (float): PageRank 알고리즘의 댐핑 팩터.
            iterations (int): PageRank 알고리즘의 최대 반복 횟수.
            min_word_length (int): 키워드로 고려할 최소 단어 길이.
        """
        self.window_size = window_size
        self.damping_factor = damping_factor
        self.iterations = iterations
        self.min_word_length = min_word_length
        
        if not hasattr(nx, 'pagerank'):
             logger.error("networkx library is required for TextRank but not found or PageRank is missing.")
             raise ImportError("networkx library is required for TextRank.")

    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """
        주어진 텍스트에서 TextRank 알고리즘을 사용하여 주요 키워드를 추출합니다.

        Args:
            text (str): 키워드를 추출할 입력 텍스트.
            num_keywords (int): 반환할 최대 키워드 개수.

        Returns:
            List[str]: 추출된 키워드 리스트 (점수 높은 순).
        """
        if not text:
            return []

        try:
            # 1. 단순 공백 분리 토큰화 및 길이 필터링
            # 필요에 따라 구두점 제거 등 추가 전처리 가능
            tokens = [word.strip() for word in text.split() if len(word.strip()) >= self.min_word_length]
            if not tokens:
                logger.debug("TextRank: No tokens found after splitting and filtering.")
                return []

            # 2. 단어 그래프 생성
            graph = nx.Graph()
            # 중복 없는 노드 추가 (set 사용)
            graph.add_nodes_from(set(tokens)) 

            # 윈도우 내 단어들 간의 엣지 추가
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + self.window_size, len(tokens))):
                    # 같은 단어 간의 루프는 만들지 않음
                    if tokens[i] != tokens[j]: 
                        # 엣지 가중치는 동시 등장 빈도로 할 수도 있으나, 여기서는 단순 연결
                        graph.add_edge(tokens[i], tokens[j])
            
            if not graph.nodes or not graph.edges: 
                logger.debug("TextRank: Graph has no nodes or edges. Returning frequency-based tokens.")
                # 그래프가 유효하지 않으면 빈도 기반으로 상위 몇 개 반환
                word_counts = Counter(tokens)
                return [item[0] for item in word_counts.most_common(num_keywords)]

            # 3. PageRank 알고리즘 적용
            # NetworkX의 PageRank는 딕셔너리 (노드: 점수)를 반환
            scores = nx.pagerank(graph, alpha=self.damping_factor, max_iter=self.iterations, tol=1.0e-6, weight=None)

            # 4. 점수가 높은 순으로 키워드 정렬 및 상위 N개 선택
            sorted_keywords = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            
            keywords = [keyword for keyword, score in sorted_keywords[:num_keywords]]
            logger.debug(f"TextRank extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"TextRank keyword extraction failed: {e}", exc_info=True)
            # TextRank 실패 시, 빈도수 기반으로 추출 (fallback)
            try:
                tokens = [word.strip() for word in text.split() if len(word.strip()) >= self.min_word_length]
                word_counts = Counter(tokens)
                fallback_keywords = [item[0] for item in word_counts.most_common(num_keywords)]
                logger.warning(f"Falling back to frequency-based keywords: {fallback_keywords}")
                return fallback_keywords
            except Exception as fallback_e:
                 logger.error(f"TextRank fallback extraction failed: {fallback_e}", exc_info=True)
                 return []

# --- 예시 사용법 ---
if __name__ == "__main__":
    # 로거 설정 (테스트용)
    if not logger.handlers:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # networkx 설치 확인
    if not hasattr(nx, 'pagerank'): # networkx가 없으면 테스트 실행 안 함
        print("Error: networkx library is not installed. Please install it using 'pip install networkx'.")
    else:
        # TextRankExtractor 인스턴스 생성
        extractor = TextRankExtractor(window_size=3) # num_keywords 인자 제거

        # 테스트 텍스트
        test_texts = [
            "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날씨예요. 공원에 가서 산책하고 싶어요.", # 일상, 긍정
            "프로젝트 마감일이 다가와서 조금 바쁘네요. 기능 구현에 집중해야 해요. 테스트도 해야 하고요.", # 업무, 스트레스
            "새로운 인공지능 모델이 발표되었는데, 성능이 매우 뛰어나다고 합니다. 특히 자연어 처리 분야에서 혁신적인 결과를 보여주고 있습니다.", # 기술, 정보
            "주말에 친구들과 함께 맛있는 음식을 먹으러 가기로 했어요. 어떤 메뉴를 고를지 고민 중이에요. 파스타나 피자도 좋고, 한식도 괜찮을 것 같아요.", # 약속, 음식
            "최근에 읽은 책 내용이 너무 인상 깊어서 추천해주고 싶어요. 주인공의 성장 과정이 감동적이었어요.", # 취미, 감상
            "오늘 회의에서는 다음 분기 전략에 대해 논의했습니다. 시장 분석 자료를 바탕으로 새로운 목표를 설정했어요.", # 업무, 회의
            "요즘 건강 관리에 신경 쓰고 있어요. 매일 아침 조깅을 하고, 식단도 조절하려고 노력 중입니다.", # 건강, 생활습관
            "여행 계획을 세우고 있는데, 어디로 갈지 아직 못 정했어요. 바다도 보고 싶고, 산도 가고 싶네요.", # 여행, 계획
            "이 문제는 생각보다 복잡해서 해결하는 데 시간이 좀 걸릴 것 같아요. 다양한 각도에서 접근해야 할 필요가 있습니다.", # 문제 해결, 분석
            "간단한 문장입니다.", # 짧은 문장
            "", # 빈 문자열
            "키워드 하나만 있는 문장입니다. 키워드", # 특정 단어 반복
        ]

        # 추가된 테스트 텍스트들에 대한 처리
        for i, text in enumerate(test_texts[4:], 5): # text1~4 다음부터 번호 매기기
            print(f"\n--- Text {i} ---")
            print(f"Input: {text}")
            keywords = extractor.extract_keywords(text, num_keywords=3) # 예시로 3개 키워드 추출
            print(f"Extracted Keywords: {keywords}")