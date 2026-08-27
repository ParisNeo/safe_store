import sqlite3
from typing import List, Dict, Any, Optional
from ascii_colors import ASCIIColors


class BM25Retriever:
    """
    Lexical search retriever powered by SQLite FTS5 with BM25 ranking.
    Falls back to SQL LIKE matching if FTS5 is not available in the SQLite build.
    """

    def __init__(self, conn: Any):
        if hasattr(conn, "_ensure_connection"):
            conn._ensure_connection()
            self._store = conn
            self._conn = conn.conn
        else:
            self._store = None
            self._conn = conn
        self._fts_available = self._check_fts_availability()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._store:
            self._store._ensure_connection()
            return self._store.conn
        if self._conn is None:
            raise ValueError("Database connection is not available for BM25Retriever.")
        return self._conn

    def _check_fts_availability(self) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'")
            return cursor.fetchone() is not None
        except Exception:
            return False

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        min_relevance_percent: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Searches the database for chunks matching query_text using BM25 with a 0-100 relevance grade.
        Results falling below min_relevance_percent are excluded.
        """
        clean_query = query_text.strip()
        if not clean_query:
            return []

        raw_results = []
        if self._fts_available:
            try:
                raw_results = self._search_fts5(clean_query, top_k * 2)
            except Exception as e:
                ASCIIColors.warning(f"FTS5 query failed, falling back to LIKE: {e}")
                raw_results = self._search_fallback(clean_query, top_k * 2)
        else:
            raw_results = self._search_fallback(clean_query, top_k * 2)

        # Apply threshold filter on normalized 0-100 relevance grade
        filtered = [r for r in raw_results if r["relevance_score"] >= min_relevance_percent]
        return filtered[:top_k] if top_k > 0 else filtered

    def _search_fts5(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        tokens = [t.replace('"', '""') for t in query_text.split() if t.strip()]
        if not tokens:
            return []

        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        sql = """
            SELECT c.chunk_id, c.doc_id, c.chunk_text, c.start_pos, c.end_pos,
                   d.file_path, bm25(chunks_fts) as rank_score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.rowid = c.chunk_id
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?
        """

        results = []
        original_factory = self.conn.text_factory
        self.conn.text_factory = bytes
        cursor = self.conn.cursor()
        rows = cursor.execute(sql, (fts_query, top_k)).fetchall()
        cursor.close()
        self.conn.text_factory = original_factory

        for row in rows:
            chunk_id, doc_id, chunk_text_bytes, start, end, file_path_bytes, rank_score = row

            # Invert FTS5 negative rank score
            norm_score = float(-rank_score) if rank_score is not None else 1.0

            # Map BM25 saturation score to 0-100 grade
            relevance_pct = round(min(100.0, (norm_score / (norm_score + 0.4)) * 100.0), 2) if norm_score > 0 else 0.0


            results.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_text": chunk_text_bytes.decode('utf-8', errors='ignore'),
                "start_pos": start,
                "end_pos": end,
                "file_path": file_path_bytes.decode('utf-8', errors='ignore'),
                "score": norm_score,
                "similarity_score": norm_score,
                "similarity_percent": relevance_pct,
                "relevance_score": relevance_pct
            })

        return results

    def _search_fallback(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        tokens = query_text.lower().split()
        if not tokens:
            return []

        sql = """
            SELECT c.chunk_id, c.doc_id, c.chunk_text, c.start_pos, c.end_pos, d.file_path
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
        """

        original_factory = self.conn.text_factory
        self.conn.text_factory = bytes
        cursor = self.conn.cursor()
        rows = cursor.execute(sql).fetchall()
        cursor.close()
        self.conn.text_factory = original_factory

        scored = []
        total_tokens = max(1, len(tokens))
        for row in rows:
            chunk_id, doc_id, chunk_text_bytes, start, end, file_path_bytes = row
            text = chunk_text_bytes.decode('utf-8', errors='ignore').lower()

            matched_count = sum(1 for token in tokens if token in text)
            if matched_count > 0:
                relevance_pct = round(min(100.0, (float(matched_count) / float(total_tokens)) * 100.0), 2)
                scored.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "chunk_text": chunk_text_bytes.decode('utf-8', errors='ignore'),
                    "start_pos": start,
                    "end_pos": end,
                    "file_path": file_path_bytes.decode('utf-8', errors='ignore'),
                    "score": float(matched_count),
                    "similarity_score": float(matched_count),
                    "similarity_percent": relevance_pct,
                    "relevance_score": relevance_pct
                })

        scored.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored[:top_k]