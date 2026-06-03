# graph_rag_example.py

import networkx as nx
from typing import List, Dict, Any, Optional

# --- 1. 지식 그래프 구축 (Data to Graph) ---

class GraphMemoryStore:
    """
    기억들을 지식 그래프로 관리하는 간단한 클래스.
    실제 시스템에서는 Neo4j, Neptune 같은 그래프 DB를 사용합니다.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.memory_nodes = {} # memory_id -> node_data 매핑

    def add_memory(self, memory_id: Optional[str], content: str, entities: List[str], relationships: List[tuple]):
        """
        기억을 그래프에 추가합니다.
        - content는 노드의 속성으로 저장됩니다.
        - entities는 그래프의 개체 노드가 됩니다.
        - relationships는 노드 간의 엣지(관계)가 됩니다.
        """
        # 기억 자체를 나타내는 노드 추가
        self.graph.add_node(memory_id, type='memory', content=content)
        self.memory_nodes[memory_id] = {'content': content, 'entities': entities}

        # 기억에 포함된 개체(entity) 노드들을 추가하고, 기억 노드와 연결
        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type='entity')
            self.graph.add_edge(memory_id, entity, relation='contains_entity')

        # 개체들 간의 관계를 엣지로 추가
        for subj, rel, obj in relationships:
            if self.graph.has_node(subj) and self.graph.has_node(obj):
                self.graph.add_edge(subj, obj, relation=rel)
        
        print(f"Added memory '{memory_id}'. Graph now has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def search_related_memories(self, query_entities: List[str], depth: int = 1) -> List[Dict[str, Any]]:
        """
        주어진 개체와 관련된 기억들을 그래프 탐색을 통해 찾습니다.
        
        Args:
            query_entities: 사용자의 질문에서 추출된 핵심 개체들.
            depth: 관련 기억을 찾기 위해 탐색할 깊이.
        
        Returns:
            관련된 기억들의 내용 리스트.
        """
        related_memory_ids = set()
        
        for entity in query_entities:
            if not self.graph.has_node(entity):
                continue
            
            # 1. 시작 노드(entity)와 직접 연결된 기억들을 찾습니다.
            for neighbor in self.graph.neighbors(entity):
                if self.graph.nodes[neighbor].get('type') == 'memory':
                    related_memory_ids.add(neighbor)
            
            # 2. 더 깊은 탐색 (Ego Graph)
            # 시작 노드로부터 'depth' 거리 내의 모든 노드를 포함하는 하위 그래프를 생성합니다.
            ego_graph = nx.ego_graph(self.graph, entity, radius=depth)
            for node in ego_graph.nodes():
                if self.graph.nodes[node].get('type') == 'memory':
                    related_memory_ids.add(node)

        # 찾은 ID를 기반으로 실제 기억 내용을 반환합니다.
        results = [
            {"id": mem_id, "content": self.memory_nodes[mem_id]['content']}
            for mem_id in related_memory_ids
        ]
        return results

# --- 2. 예시 실행 ---

if __name__ == "__main__":
    # 그래프 저장소 초기화
    g_store = GraphMemoryStore()

    # LLM 등을 통해 미리 추출했다고 가정한 데이터
    # 실제로는 add_memory 시점에 LLM으로 content에서 entities와 relationships를 추출해야 함
    memory_1_data = {
        "memory_id": "mem_01",
        "content": "사용자는 어제 고양이 '나비'와 함께 공원에 산책을 갔다.",
        "entities": ["사용자", "나비", "공원"],
        "relationships": [("사용자", "OWNER_OF", "나비"), ("나비", "VISITED", "공원")]
    }
    
    memory_2_data = {
        "memory_id": "mem_02",
        "content": "고양이 '나비'는 츄르 간식을 매우 좋아한다.",
        "entities": ["나비", "츄르"],
        "relationships": [("나비", "LIKES", "츄르")]
    }

    memory_3_data = {
        "memory_id": "mem_03",
        "content": "공원 근처에는 맛있는 빵집이 있다.",
        "entities": ["공원", "빵집"],
        "relationships": [("빵집", "LOCATED_NEAR", "공원")]
    }

    # 그래프에 기억 추가
    g_store.add_memory(**memory_1_data)
    g_store.add_memory(**memory_2_data)
    g_store.add_memory(**memory_3_data)
    
    print("\n" + "="*30 + "\n")

    # --- 검색 시나리오 ---
    # 사용자 질문: "나비에 대해 알려줘" -> 질문에서 '나비'라는 개체 추출
    query = "나비"
    print(f"Searching for memories related to '{query}'...")
    
    # '나비'와 관련된 기억 검색 (depth=1: 직접 연결된 기억과 그 이웃까지)
    search_results = g_store.search_related_memories(query_entities=[query], depth=1)
    
    print("\n[Search Results]")
    for result in search_results:
        print(f"- ID: {result['id']}, Content: {result['content']}")
        
    # 결과 분석:
    # '나비'와 직접 연결된 mem_01, mem_02가 검색됩니다.
    # depth를 2로 늘리면 '나비' -> '공원' -> '빵집' 관계를 통해 mem_03까지도 찾을 수 있습니다.
    # 이처럼 GraphRAG는 직접 언급되지 않았지만 관계적으로 연결된 잠재적 컨텍스트를 발견하는 데 강점이 있습니다.
