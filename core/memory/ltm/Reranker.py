# light_rag_example.py

from sentence_transformers import SentenceTransformer, util, CrossEncoder
from typing import List, Dict, Any

# --- 1. LightRAG 파이프라인 모듈 정의 ---

class LightRAGPipeline:
    """
    경량화된 RAG 파이프라인.
    1. Retriever (Bi-Encoder): 빠른 속도로 후보군을 검색.
    2. Re-ranker (Cross-Encoder): 후보군의 순위를 정밀하게 재조정.
    """
    def __init__(self, bi_encoder_model_name: str, cross_encoder_model_name: str):
        # 1차 검색기: 문장들을 독립적으로 임베딩 (빠름)
        print(f"Loading Bi-Encoder model: {bi_encoder_model_name}")
        self.retriever = SentenceTransformer(bi_encoder_model_name)
        
        # 2차 재순위기: (질문, 문서) 쌍의 관련도를 직접 계산 (정확하지만 느림)
        print(f"Loading Cross-Encoder model: {cross_encoder_model_name}")
        self.reranker = CrossEncoder(cross_encoder_model_name)
        
        self.corpus_embeddings = None
        self.corpus = []

    def build_corpus(self, documents: List[str]):
        """검색 대상이 될 문서들(기억들)의 임베딩을 미리 계산합니다."""
        self.corpus = documents
        print(f"Building corpus with {len(documents)} documents...")
        self.corpus_embeddings = self.retriever.encode(documents, convert_to_tensor=True, show_progress_bar=True)
        print("Corpus built successfully.")

    def search(self, query: str, top_k_retrieval: int = 10, final_top_k: int = 3) -> List[Dict[str, Any]]:
        """
        질문에 대해 2단계 검색(검색 -> 재순위)을 수행합니다.
        """
        if self.corpus_embeddings is None:
            raise RuntimeError("Corpus is not built. Call build_corpus() first.")

        # --- 단계 1: 검색 (Retrieval) ---
        # Bi-Encoder를 사용하여 쿼리와 모든 문서 임베딩 간의 코사인 유사도를 계산
        query_embedding = self.retriever.encode(query, convert_to_tensor=True)
        retrieval_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # 점수가 높은 상위 N개(top_k_retrieval) 후보군을 선택
        retrieval_hits = retrieval_scores.topk(min(top_k_retrieval, len(self.corpus)))
        
        candidate_indices = retrieval_hits.indices.tolist()
        candidate_docs = [self.corpus[i] for i in candidate_indices]
        
        print(f"\n--- [Step 1: Retrieval] Found {len(candidate_docs)} candidates ---")
        for i, doc_idx in enumerate(candidate_indices):
            print(f"  - Candidate {i+1}: '{self.corpus[doc_idx][:50]}...' (Score: {retrieval_scores[doc_idx]:.4f})")

        # --- 단계 2: 재순위 (Re-ranking) ---
        # Cross-Encoder 모델에 (쿼리, 후보 문서) 쌍을 입력하여 관련도 점수를 다시 계산
        reranker_inputs = [(query, doc) for doc in candidate_docs]
        reranker_scores = self.reranker.predict(reranker_inputs, show_progress_bar=False)
        
        # 재계산된 점수와 후보 문서를 결합
        reranked_results = list(zip(reranker_scores, candidate_docs, candidate_indices))
        
        # 새로운 점수를 기준으로 내림차순 정렬
        reranked_results.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\n--- [Step 2: Re-ranking] Re-ranked the candidates ---")

        # --- 최종 결과 ---
        final_results = []
        for score, doc, doc_idx in reranked_results[:final_top_k]:
            final_results.append({
                "document": doc,
                "retrieval_score": retrieval_scores[doc_idx].item(),
                "rerank_score": score,
                "original_index": doc_idx
            })
            
        return final_results

# --- 2. 예시 실행 ---

if __name__ == "__main__":
    # 예시 기억 데이터 (문서 모음)
    documents = [
        "우리 강아지 이름은 '해피'이고, 산책을 정말 좋아해요.",
        "오늘 점심 메뉴는 김치찌개였습니다.",
        "파이썬은 배우기 쉬운 프로그래밍 언어입니다.",
        "저는 매일 아침 공원에서 조깅을 합니다. 강아지와 함께 뛰는 것을 좋아하죠.",
        "해피는 공을 가져오는 놀이를 잘합니다.",
        "김치찌개를 잘하는 식당이 근처에 새로 생겼다.",
        "요즘 날씨가 너무 덥네요. 시원한 아이스크림이 먹고 싶어요.",
        "강아지에게는 초콜릿을 주면 안 됩니다. 건강에 해로워요."
    ]

    # 모델 선택:
    # Bi-Encoder: 한국어 문장 임베딩에 특화된 모델 (e.g., 'jhgan/ko-sroberta-multitask')
    # Cross-Encoder: 한국어 관련도 계산에 특화된 모델 (e.g., 'bongsoo/kpf-cross-encoder-v1')
    # (HuggingFace에서 다운로드가 필요하며, 처음 실행 시 시간이 걸릴 수 있습니다.)
    pipeline = LightRAGPipeline(
        bi_encoder_model_name='jhgan/ko-sroberta-multitask',
        cross_encoder_model_name='bongsoo/kpf-cross-encoder-v1'
    )

    # 검색 대상 문서들로 코퍼스 구축
    pipeline.build_corpus(documents)

    # --- 검색 시나리오 ---
    user_query = "우리 강아지랑 같이 할 수 있는 놀이가 뭐가 있을까?"
    print("\n" + "="*30 + "\n")
    print(f"User Query: '{user_query}'")

    # 검색 실행
    search_results = pipeline.search(user_query, top_k_retrieval=5, final_top_k=3)

    print("\n[Final Top 3 Results after Re-ranking]")
    for i, result in enumerate(search_results):
        print(f"Rank {i+1}: '{result['document']}'")
        print(f"  (Initial Score: {result['retrieval_score']:.4f} -> Re-rank Score: {result['rerank_score']:.4f})\n")

    # 결과 분석:
    # 1차 검색에서는 '강아지', '놀이' 같은 키워드 때문에 여러 문서가 후보로 올라올 수 있습니다.
    # 하지만 2차 재순위 과정에서 Cross-Encoder가 "강아지랑 같이 할 수 있는 놀이"라는 문맥 전체와
    # 각 후보 문서의 관련성을 정밀하게 판단하여, '공을 가져오는 놀이'나 '산책' 관련 문서를
    # 더 높은 순위로 올려주는 것을 확인할 수 있습니다.

