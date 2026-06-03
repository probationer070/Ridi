import logging
import networkx as nx
from collections import Counter
from typing import List, Dict, Tuple, Union, Optional # Union 추가
from .MiniKoNLP import MiniKoNLPTokenizer # MiniKoNLPTokenizer 임포트

logger = logging.getLogger(__name__)
class TextRankKeywordExtractor:
    def __init__(self,
                 window_size: int = 4,
                 damping_factor: float = 0.85,
                 iterations: int = 100,
                 min_word_length: int = 1, 
                 pos_filter: Optional[List[str]] = None,
                 textranking: bool = False
                 ):
        self.window_size = window_size
        self.damping_factor = damping_factor
        self.iterations = iterations
        self.min_word_length = min_word_length
        self.pos_filter = set(pos_filter) if pos_filter else None
        self.textranking = textranking

        NETWORKX_AVAILABLE = True
        if self.textranking:
            if not NETWORKX_AVAILABLE:
                logger.error("networkx library is required for TextRank but not found or PageRank is missing.")
                raise ImportError("networkx library is required for TextRank.")
        

    def extract_keywords(self, 
                         text_or_tokens: Union[str, List[Dict[str,str]], List[str]], 
                         num_keywords: int = 10) -> List[str]:
        if not text_or_tokens:
            return []

        logger.debug(f"TextRankKeywordExtractor: pos_filter={self.pos_filter}, min_word_length={self.min_word_length}")

        # --- 입력 타입 정규화: 모든 유효한 입력을 List[Dict[str, str]] 형태로 변환하거나 조기 반환 ---
        tokens_to_filter: List[Dict[str, str]] = []

        if isinstance(text_or_tokens, str):
            # 1. 입력이 문자열인 경우, 토크나이저를 사용해 List[Dict] 형태로 변환합니다.
            temp_tokenizer = MiniKoNLPTokenizer() 
            tokens_to_filter = temp_tokenizer.tokenize(text_or_tokens)
        elif isinstance(text_or_tokens, list):
            if not text_or_tokens:
                return []
            
            if isinstance(text_or_tokens[0], dict):
                # 2. 입력이 이미 List[Dict] 형태인 경우, 그대로 사용합니다.
                tokens_to_filter = text_or_tokens
            elif isinstance(text_or_tokens[0], str):
                # 3. 입력이 List[str]인 경우, 품사 필터링이 불가능하므로 빈도수 기반으로 처리하고 조기 반환합니다.
                if self.pos_filter:
                    logger.warning("pos_filter is set, but input is List[str]. POS filtering cannot be applied.")
                word_counts = Counter(
                    token.strip() for token in text_or_tokens
                    if isinstance(token, str) and len(token.strip()) >= self.min_word_length
                )
                return [item[0] for item in word_counts.most_common(num_keywords)]
            else:
                logger.error("Invalid list item type for TextRank. List items must be dict or str.")
                return []
        else:
            logger.error(f"Invalid input type for TextRank. Must be str or List. Got {type(text_or_tokens)}")
            return []

        # --- 품사(POS) 및 길이 기반 필터링 ---
        processed_tokens: List[Dict[str, str]] = []
        for item in tokens_to_filter:
            if isinstance(item, dict) and "token" in item and "pos" in item and \
               (not self.pos_filter or item["pos"] in self.pos_filter) and \
               len(item["original"].strip()) >= self.min_word_length:
                processed_tokens.append(item)

        if not processed_tokens:
            logger.debug("TextRank: No valid tokens found after filtering.")
            return []
        
        # node_strings는 TextRank와 빈도수 계산 모두에 사용됩니다.
        node_strings = [item['token'] for item in processed_tokens]

        # TextRank 사용이 비활성화된 경우, 빈도수 기반으로 키워드를 추출합니다.
        if not self.textranking:
            logger.debug("TextRank is disabled. Using frequency-based keyword extraction.")
            word_counts = Counter(node_strings)
            return [item[0] for item in word_counts.most_common(num_keywords)]
        
        # TextRank 활성화된 경우, 그래프 기반 키워드 추출 수행
        else:
            try:
                graph = nx.Graph()
                # processed_tokens는 이제 List[Dict[str, str]] 형태 (필터링된 결과)
                # 그래프 노드는 토큰 문자열로 추가
                # 원본 단어를 노드로 사용
                graph.add_nodes_from(set(node_strings))

                # 윈도우 내 단어들 간의 엣지 추가
                for i in range(len(processed_tokens)):
                    for j in range(i + 1, min(i + self.window_size, len(processed_tokens))):
                        # 엣지 생성 시에도 원본 단어 사용
                        token1_data = processed_tokens[i]
                        token2_data = processed_tokens[j]
                        
                        token1_str = token1_data['token'] # 원본 단어 사용
                        token2_str = token2_data['token'] # 원본 단어 사용

                        if token1_str != token2_str:
                            # 수정: pos_filter를 통과한 모든 유효한 키워드 후보 단어들 사이에 엣지를 추가합니다.
                            # processed_tokens 리스트에는 이미 pos_filter와 min_word_length를 통과한
                            # {'token': ..., 'pos': ..., 'original': ...} 형태의 딕셔너리만 포함되어 있습니다.
                            # 따라서 token1_data와 token2_data는 이미 유효한 키워드 후보입니다.
                            graph.add_edge(token1_str, token2_str)
                
                if not graph.nodes or not graph.edges: 
                    logger.debug("TextRank: Graph has no nodes or edges. Returning frequency-based tokens.")
                    word_counts = Counter(node_strings) # node_strings 사용
                    return [item[0] for item in word_counts.most_common(num_keywords)]

                scores = nx.pagerank(graph, alpha=self.damping_factor, max_iter=self.iterations, tol=1.0e-6, weight=None)
                sorted_keywords = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                final_keywords = [keyword for keyword, score in sorted_keywords[:num_keywords]]
                logger.debug(f"TextRank extracted keywords from {len(node_strings)} unique nodes: {final_keywords}")
                return final_keywords
            except Exception as e:
                logger.error(f"TextRank keyword extraction failed: {e}", exc_info=True)
                try:
                    word_counts = Counter(node_strings) # node_strings 사용
                    fallback_keywords = [item[0] for item in word_counts.most_common(num_keywords)]
                    logger.warning(f"Falling back to frequency-based keywords: {fallback_keywords}")
                    return fallback_keywords
                except Exception as fallback_e:
                    logger.error(f"TextRank fallback extraction failed: {fallback_e}", exc_info=True)
                    return []
