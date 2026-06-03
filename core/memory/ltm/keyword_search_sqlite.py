import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..MemoryItem import MemoryItem
from .korean_text_analyzer import TextRankKeywordExtractor

logger = logging.getLogger(__name__)

class KeywordSearchSQLite:
    """
    SQLite를 사용하여 장기기억을 저장하고, 키워드 및 메타데이터 필터링을 통해 검색하는 클래스.
    FTS5를 이용한 전문 검색(Full-Text Search)을 지원합니다.
    """

    def __init__(self, db_path: str, keyword_extractor: Optional[callable] = None, **kwargs):
        """
        데이터베이스 연결을 초기화하고 테이블을 생성합니다.

        Args:
            db_path (str): SQLite 데이터베이스 파일 경로.
            keyword_extractor (Optional[callable]): 외부에서 주입된 키워드 추출 함수.
                                                     제공되지 않으면 내부 기본 추출기를 사용합니다.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # SQLite에서 JSON 함수를 사용하기 위해 필요할 수 있습니다.
        try:
            self.conn.enable_load_extension(True)
            self.conn.load_extension("json1")
            self.conn.enable_load_extension(False)
            logger.info("SQLite json1 extension loaded successfully.")
        except sqlite3.OperationalError:
            logger.info("SQLite json1 extension is already loaded or built-in.")
        
        self._create_tables()
        
        # 외부에서 추출기가 주입되지 않은 경우에만 기본 추출기를 생성합니다.
        if keyword_extractor:
            self.keyword_extractor = keyword_extractor
        else:
            logger.info("No keyword extractor provided, initializing default TextRankKeywordExtractor.")
            filter_pos = ["Noun", "Pronoun", "Verb", "Adjective", "Adverb"]
            # 클래스 인스턴스가 아닌, 호출 가능한 'extract_keywords' 메서드를 할당합니다.
            default_extractor_instance = TextRankKeywordExtractor(pos_filter=filter_pos, min_word_length=2, textranking=False)
            self.keyword_extractor = default_extractor_instance.extract_keywords

    def _create_tables(self):
        """데이터베이스에 'memories' 테이블과 FTS용 'memories_fts' 가상 테이블을 생성합니다."""
        cursor = self.conn.cursor()
        # 기억의 모든 원본 데이터를 저장하는 메인 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                item_id TEXT PRIMARY KEY,
                memory_type TEXT DEFAULT 'generic' NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at_timestamp INTEGER NOT NULL,
                source TEXT,
                tags_json TEXT
            )
        """)
        # FTS5 가상 테이블 (내용 기반의 빠른 텍스트 검색용)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(item_id UNINDEXED, content, tokenize = 'porter unicode61')
        """)
        # 두 테이블 간의 데이터 동기화를 위한 트리거
        cursor.executescript("""
            CREATE TRIGGER IF NOT EXISTS memories_after_insert
            AFTER INSERT ON memories
            BEGIN
                INSERT INTO memories_fts(rowid, item_id, content) VALUES (new.rowid, new.item_id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_after_delete
            AFTER DELETE ON memories
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, item_id, content) VALUES ('delete', old.rowid, old.item_id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_after_update
            AFTER UPDATE ON memories
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, item_id, content) VALUES ('delete', old.rowid, old.item_id, old.content);
                INSERT INTO memories_fts(rowid, item_id, content) VALUES (new.rowid, new.item_id, new.content);
            END;
        """)
        self.conn.commit()

    def add_item(self, item: MemoryItem, extract_keywords: bool = True):
        """MemoryItem을 데이터베이스에 추가하거나 업데이트합니다."""
        tags_to_store = item.tags

        # extract_keywords가 True이고, item의 tags가 비어있으며, 추출기가 설정된 경우 키워드를 추출합니다.
        if extract_keywords and not item.tags:
            logger.info(f"Item {item.item_id} has no tags. Running keyword extraction...")
            # self.keyword_extractor는 LTM_Manager로부터 주입된 호출 가능한(callable) 메서드입니다.
            keywords = self.keyword_extractor(item.content, num_keywords=10)
            tags_to_store = keywords
            logger.info(f"Extracted keywords for item {item.item_id}: {keywords}")

        cursor = self.conn.cursor()
        cursor.execute("""
            REPLACE INTO memories (item_id, memory_type, content, metadata_json, created_at_timestamp, source, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            item.item_id,
            item.memory_type,
            item.content,
            json.dumps(item.metadata, ensure_ascii=False) if item.metadata else None, # 한글 깨짐 방지
            item.created_at_timestamp,
            item.source,
            json.dumps(tags_to_store, ensure_ascii=False) if tags_to_store else None # 한글 깨짐 방지
        ))
        self.conn.commit()
        logger.debug(f"Added/Replaced item {item.item_id} in SQLite.")

    def delete_item(self, item_id: str):
        """
        ID를 사용하여 데이터베이스에서 MemoryItem을 삭제합니다.
        트리거에 의해 FTS 테이블의 해당 항목도 자동으로 삭제됩니다.
        """
        if not item_id:
            logger.warning("delete_item called with an empty item_id.")
            return

        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM memories WHERE item_id = ?", (item_id,))
            self.conn.commit()
            logger.info(f"Successfully deleted item {item_id} from SQLite.")
        except Exception as e:
            logger.error(f"Error deleting item {item_id} from SQLite: {e}", exc_info=True)

    def get_item_by_id(self, item_id: str) -> Optional[MemoryItem]:
        """ID로 특정 MemoryItem을 조회합니다."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT item_id, content, memory_type, metadata_json, created_at_timestamp, source, tags_json FROM memories WHERE item_id = ?", (item_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        return MemoryItem(
            item_id=row[0], content=row[1],
            memory_type=row[2],
            metadata=json.loads(row[3]) if row[3] else {},
            created_at_timestamp=row[4], source=row[5],
            tags=json.loads(row[6]) if row[6] else []
        )

    def _build_search_query(self, query: Optional[str], filters: Optional[Dict[str, Any]], top_k: int) -> Tuple[str, List[Any]]:
        """검색 조건에 따라 동적으로 SQL 쿼리를 생성합니다."""
        params = []
        where_clauses = []
        
        # 'memories' 테이블의 직접 필터링 가능한 컬럼
        direct_columns = {'source', 'memory_type'}

        if filters:
            for key, value in filters.items():
                if key in direct_columns:
                    where_clauses.append(f"m.{key} = ?")
                    params.append(value)
                else:
                    # 그 외의 필터는 metadata_json 컬럼 내부에서 검색 (user_id, memory_type 등)
                    where_clauses.append(f"json_extract(m.metadata_json, '$.{key}') = ?")
                    params.append(str(value)) # JSON 값은 텍스트로 비교

        if query:
            # FTS 텍스트 검색이 있는 경우
            select_clause = "SELECT m.*, fts.rank as score"
            from_clause = "FROM memories m JOIN memories_fts fts ON m.rowid = fts.rowid"
            where_clauses.insert(0, "fts.content MATCH ?")
            params.insert(0, query)
            order_by_clause = "ORDER BY score" # FTS5에서는 rank가 낮을수록 관련성 높음
        else:
            # 필터 기반 검색만 있는 경우
            select_clause = "SELECT m.*, 1.0 as score" # 일관성을 위해 더미 점수 추가
            from_clause = "FROM memories m"
            order_by_clause = "ORDER BY m.created_at_timestamp DESC"

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        limit_sql = f"LIMIT ?"
        params.append(top_k)

        final_sql = f"{select_clause} {from_clause} {where_sql} {order_by_clause} {limit_sql};"
        return final_sql, params

    def search(self, query: Optional[str] = None, filters: Optional[Dict[str, Any]] = None, top_k: int = 10) -> List[Dict]:
        """텍스트 쿼리와 메타데이터 필터를 결합하여 기억을 검색합니다."""
        if not query and not filters:
            logger.warning("Search called with no query and no filters.")
            return []

        try:
            sql, params = self._build_search_query(query, filters, top_k)
            logger.debug(f"Executing search query: {sql} with params: {params}")
            
            cursor = self.conn.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            column_names = [desc[0] for desc in cursor.description]
            results = []
            for row in rows:
                res_dict = dict(zip(column_names, row))
                
                # Orchestrator에서 MemoryItem 객체를 쉽게 만들 수 있도록 결과 포맷팅
                item_dict = {
                    "item_id": res_dict["item_id"],
                    "content": res_dict["content"],
                    "metadata": json.loads(res_dict["metadata_json"]) if res_dict.get("metadata_json") else {},
                    "created_at_timestamp": res_dict["created_at_timestamp"],
                    "source": res_dict.get("source"),
                    "tags": json.loads(res_dict["tags_json"]) if res_dict.get("tags_json") else [],
                    "score": res_dict.get("score")
                }
                results.append(item_dict)
            
            return results

        except Exception as e:
            logger.error(f"Error during keyword search: {e}", exc_info=True)
            return []

    def get_all_items(self) -> List[MemoryItem]:
        """디버깅 목적으로 모든 기억을 조회합니다."""
        memories = []
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT item_id, content, memory_type, metadata_json, created_at_timestamp, source, tags_json FROM memories ORDER BY created_at_timestamp DESC")
            rows = cursor.fetchall()
            for row in rows:
                memories.append(MemoryItem(
                    item_id=row[0], content=row[1],
                    memory_type=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    created_at_timestamp=row[4], source=row[5],
                    tags=json.loads(row[6]) if row[6] else []
                ))
        except Exception as e:
            logger.error(f"Error retrieving all memories for debugging: {e}", exc_info=True)
        return memories

    def close(self):
        """데이터베이스 연결을 닫습니다."""
        if self.conn:
            self.conn.close()
            logger.info("SQLite connection closed.")