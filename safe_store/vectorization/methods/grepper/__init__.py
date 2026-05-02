import re
import json
import sqlite3
import numpy as np
from typing import List, Optional, Any, Dict, Tuple
from ...base import BaseVectorizer


class GrepperVectorizer(BaseVectorizer):
    """
    A keyword-based vectorizer that decomposes markdown documents into a tree
    structure and builds an inverted index for ultra-fast text search.
    Stores placeholder vectors in the DB for compatibility with the BaseVectorizer
    interface, but performs real search via an inverted index through hooks.
    """
    class_name = 'GrepperVectorizer'

    def __init__(self, vectorizer_name: str = "grepper", **kwargs):
        super().__init__(vectorizer_name=vectorizer_name)
        self._dim = 1
        self._dtype = np.float32
        # In-memory cache for indices until they are persisted via hooks
        self._pending_indices: Dict[int, List[Dict[str, Any]]] = {}
        self._pending_trees: Dict[int, Dict[str, Any]] = {}

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Returns minimal placeholder vectors for compatibility."""
        if not texts:
            return np.array([], dtype=self._dtype)
        return np.zeros((len(texts), self._dim), dtype=self._dtype)

    def get_tokenizer(self) -> Optional[Any]:
        return None

    @staticmethod
    def list_models(**kwargs) -> List[str]:
        return ["grepper-default"]

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, extract alphanumeric tokens."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _parse_markdown_tree(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown text into a tree of blocks with header hierarchy.
        Returns a list of content blocks with header breadcrumbs.
        """
        lines = text.split('\n')
        blocks = []
        current_block_lines = []
        current_headers = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
        current_header_path = []

        def flush_block():
            if current_block_lines:
                block_text = '\n'.join(current_block_lines).strip()
                if block_text:
                    # Build header path from active headers
                    header_path = []
                    for level in range(1, 7):
                        if current_headers[level] is not None:
                            header_path.append(current_headers[level])
                    blocks.append({
                        'text': block_text,
                        'header_path': header_path.copy(),
                        'header_path_str': ' > '.join(header_path) if header_path else ''
                    })
                current_block_lines.clear()

        i = 0
        while i < len(lines):
            line = lines[i]
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                flush_block()
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                # Update header hierarchy: clear deeper levels, set current
                for l in range(level, 7):
                    current_headers[l] = None
                current_headers[level] = header_text
                # Build current path for tracking
                current_path = []
                for l in range(1, level + 1):
                    if current_headers[l] is not None:
                        current_path.append(current_headers[l])
                # Add header itself as a small block for indexing
                blocks.append({
                    'text': header_text,
                    'header_path': current_path[:-1].copy() if len(current_path) > 1 else [],
                    'header_path_str': ' > '.join(current_path[:-1]) if len(current_path) > 1 else '',
                    'is_header': True,
                    'header_level': level
                })
            else:
                current_block_lines.append(line)
            i += 1

        flush_block()
        return blocks

    def _build_index(self, doc_id: int, chunk_texts: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build inverted index for a document's chunks.
        Returns (index_entries, tree_data).
        """
        all_blocks = []
        tree_structure = {
            'doc_id': doc_id,
            'chunks': []
        }

        for chunk_seq, chunk_text in enumerate(chunk_texts):
            blocks = self._parse_markdown_tree(chunk_text)
            for block in blocks:
                block['chunk_seq'] = chunk_seq
                block['doc_id'] = doc_id
            all_blocks.extend(blocks)
            tree_structure['chunks'].append({
                'seq': chunk_seq,
                'text_preview': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text,
                'num_blocks': len(blocks)
            })

        # Build inverted index: term -> list of occurrences
        term_map: Dict[str, List[Dict[str, Any]]] = {}

        for block_idx, block in enumerate(all_blocks):
            text = block['text']
            tokens = self._tokenize(text)
            token_positions: Dict[str, List[int]] = {}

            for pos, token in enumerate(tokens):
                if token not in token_positions:
                    token_positions[token] = []
                token_positions[token].append(pos)

            for token, positions in token_positions.items():
                if token not in term_map:
                    term_map[token] = []
                term_map[token].append({
                    'block_idx': block_idx,
                    'positions': positions,
                    'header_path': block.get('header_path_str', ''),
                    'text_preview': text[:300],
                    'is_header': block.get('is_header', False),
                    'chunk_seq': block.get('chunk_seq', 0)
                })

        # Flatten to index entries with unique identifiers per block occurrence
        index_entries = []
        for term, occurrences in term_map.items():
            for occ in occurrences:
                index_entries.append({
                    'term': term,
                    'doc_id': doc_id,
                    'chunk_seq': occ['chunk_seq'],
                    'positions_json': json.dumps(occ['positions']),
                    'header_path': occ['header_path'],
                    'content_preview': occ['text_preview'],
                    'is_header': 1 if occ['is_header'] else 0,
                    'tf': len(occ['positions'])
                })

        return index_entries, tree_structure

    def on_document_indexed(self, conn: sqlite3.Connection, doc_id: int, chunk_texts: List[str]) -> None:
        """
        Hook called by SafeStore after a document is chunked and vectorized.
        Persists the inverted index and document tree to SQLite.
        """
        self._ensure_tables(conn)
        index_entries, tree_data = self._build_index(doc_id, chunk_texts)

        try:
            conn.execute("BEGIN")

            # Clear old index entries for this document
            conn.execute("DELETE FROM grepper_index WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM grepper_doc_trees WHERE doc_id = ?", (doc_id,))

            # Insert new index entries
            for entry in index_entries:
                conn.execute("""
                    INSERT INTO grepper_index 
                    (term, doc_id, chunk_seq, positions_json, header_path, content_preview, is_header, tf)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry['term'],
                    entry['doc_id'],
                    entry['chunk_seq'],
                    entry['positions_json'],
                    entry['header_path'],
                    entry['content_preview'],
                    entry['is_header'],
                    entry['tf']
                ))

            # Insert tree
            conn.execute("""
                INSERT INTO grepper_doc_trees (doc_id, tree_json)
                VALUES (?, ?)
            """, (doc_id, json.dumps(tree_data)))

            conn.commit()

        except sqlite3.Error as e:
            if conn.in_transaction:
                conn.rollback()
            raise RuntimeError(f"Failed to persist grepper index for doc_id={doc_id}: {e}") from e

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        """Create grepper-specific tables if they don't exist."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grepper_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                doc_id INTEGER NOT NULL,
                chunk_seq INTEGER NOT NULL,
                positions_json TEXT NOT NULL,
                header_path TEXT,
                content_preview TEXT,
                is_header INTEGER DEFAULT 0,
                tf INTEGER DEFAULT 1
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_grepper_term ON grepper_index (term)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_grepper_doc ON grepper_index (doc_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_grepper_doc_chunk ON grepper_index (doc_id, chunk_seq)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grepper_doc_trees (
                doc_id INTEGER PRIMARY KEY,
                tree_json TEXT NOT NULL
            )
        """)
        conn.commit()

    def _search_index(
        self,
        conn: sqlite3.Connection,
        query_text: str,
        top_k: int,
        min_similarity_percent: float
    ) -> List[Dict[str, Any]]:
        """
        Perform inverted index search.
        Scores by term frequency and exact phrase bonus.
        """
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return []

        # Fetch all index entries for query tokens
        placeholders = ','.join('?' * len(query_tokens))
        sql = f"""
            SELECT term, doc_id, chunk_seq, positions_json, header_path, content_preview, is_header, tf
            FROM grepper_index
            WHERE term IN ({placeholders})
        """

        cursor = conn.execute(sql, tuple(query_tokens))
        rows = cursor.fetchall()

        if not rows:
            return []

        # Group by (doc_id, chunk_seq) and aggregate scores
        chunk_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for row in rows:
            term, doc_id, chunk_seq, positions_json, header_path, content_preview, is_header, tf = row
            key = (doc_id, chunk_seq)

            if key not in chunk_scores:
                chunk_scores[key] = {
                    'doc_id': doc_id,
                    'chunk_seq': chunk_seq,
                    'score': 0.0,
                    'matched_terms': set(),
                    'header_path': header_path or '',
                    'content_preview': content_preview or '',
                    'all_positions': {},  # term -> positions list
                    'is_header': is_header
                }

            chunk_scores[key]['matched_terms'].add(term)
            chunk_scores[key]['score'] += float(tf)  # TF scoring

            positions = json.loads(positions_json)
            chunk_scores[key]['all_positions'][term] = positions

        # Exact phrase bonus: check if query tokens appear consecutively
        query_len = len(query_tokens)
        for key, data in chunk_scores.items():
            if not all(t in data['all_positions'] for t in query_tokens):
                continue

            # Check for consecutive positions across all query tokens
            first_positions = data['all_positions'][query_tokens[0]]
            for start_pos in first_positions:
                consecutive = True
                for i in range(1, query_len):
                    next_token = query_tokens[i]
                    expected_pos = start_pos + i
                    if expected_pos not in data['all_positions'][next_token]:
                        consecutive = False
                        break
                if consecutive:
                    data['score'] += query_len * 3.0  # Strong bonus for exact phrase

        # Header bonus
        for key, data in chunk_scores.items():
            if data['is_header']:
                data['score'] *= 1.5

        # Convert to list and sort by score
        scored_chunks = list(chunk_scores.values())

        # Apply minimum score threshold (scale from percent)
        # min_similarity_percent 0-100 maps to score threshold
        # Base threshold: at least 1 token must match for non-zero percent
        min_score = max(0.5, (min_similarity_percent / 100.0) * 10.0) if min_similarity_percent > 0 else 0.0
        scored_chunks = [c for c in scored_chunks if c['score'] >= min_score]

        if not scored_chunks:
            return []

        scored_chunks.sort(key=lambda x: x['score'], reverse=True)

        # Limit to top_k
        if top_k > 0:
            scored_chunks = scored_chunks[:top_k]

        return scored_chunks

    def custom_search(
        self,
        conn: sqlite3.Connection,
        query_text: str,
        top_k: int = 5,
        min_similarity_percent: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Custom search using the inverted index.
        Returns results in the same format as store.query().
        """
        self._ensure_tables(conn)

        search_results = self._search_index(conn, query_text, top_k, min_similarity_percent)

        if not search_results:
            return []

        # Fetch chunk and document details
        # We need to map (doc_id, chunk_seq) to actual chunk_ids and file paths
        doc_ids = list(set(r['doc_id'] for r in search_results))

        # Get document paths
        doc_paths = {}
        placeholders = ','.join('?' * len(doc_ids))
        cursor = conn.execute(
            f"SELECT doc_id, file_path FROM documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids)
        )
        for row in cursor.fetchall():
            doc_paths[row[0]] = row[1]

        # Get chunk IDs for (doc_id, chunk_seq) pairs
        chunk_keys = [(r['doc_id'], r['chunk_seq']) for r in search_results]
        chunk_id_map = {}
        if chunk_keys:
            # Build query for chunk lookup
            conditions = ' OR '.join(['(doc_id = ? AND chunk_seq = ?)'] * len(chunk_keys))
            flat_params = []
            for dk in chunk_keys:
                flat_params.extend(dk)

            cursor = conn.execute(
                f"SELECT chunk_id, doc_id, chunk_seq, chunk_text, start_pos, end_pos FROM chunks WHERE {conditions}",
                tuple(flat_params)
            )
            for row in cursor.fetchall():
                chunk_id, c_doc_id, c_chunk_seq, chunk_text, start_pos, end_pos = row
                chunk_id_map[(c_doc_id, c_chunk_seq)] = {
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_text,
                    'start_pos': start_pos,
                    'end_pos': end_pos
                }

        # Assemble final results matching store.query() output format
        final_results = []
        for result in search_results:
            key = (result['doc_id'], result['chunk_seq'])
            chunk_info = chunk_id_map.get(key, {})

            chunk_text = chunk_info.get('chunk_text', result['content_preview'])
            if isinstance(chunk_text, bytes):
                try:
                    chunk_text = chunk_text.decode('utf-8')
                except UnicodeDecodeError:
                    chunk_text = str(chunk_text)

            # Build breadcrumb-enhanced text
            breadcrumb = result['header_path']
            if breadcrumb:
                display_text = f"[{breadcrumb}]\n\n{chunk_text}"
            else:
                display_text = chunk_text

            # Calculate normalized similarity score (0 to 1 range roughly)
            raw_score = result['score']
            # Normalize: typical scores are 1-50, cap at 100 for percent calc
            normalized_score = min(raw_score / 20.0, 1.0) if raw_score > 0 else 0
            similarity_percent = round(normalized_score * 100, 2)

            final_results.append({
                'chunk_id': chunk_info.get('chunk_id', 0),
                'chunk_text': display_text,
                'start_pos': chunk_info.get('start_pos', 0),
                'end_pos': chunk_info.get('end_pos', 0),
                'file_path': doc_paths.get(result['doc_id'], ''),
                'document_metadata': None,
                'similarity_score': float(normalized_score),
                'similarity_percent': float(similarity_percent),
                'matched_terms': sorted(list(result['matched_terms'])),
                'header_breadcrumbs': breadcrumb
            })

        # Re-sort by normalized score
        final_results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return final_results

    def fit(self, texts: List[str]) -> None:
        """No-op for compatibility. TF-IDF style fitting is not needed."""
        pass

    @property
    def _fitted(self) -> bool:
        """Always considered fitted."""
        return True