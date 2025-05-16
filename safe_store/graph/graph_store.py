# safe_store/graph/graph_store.py
import sqlite3
import threading
import json
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any, Union, Tuple, Set

from filelock import FileLock, Timeout
from ascii_colors import ASCIIColors, LogLevel

from ..core import db 
from ..core.exceptions import (
    DatabaseError, ConfigurationError, ConcurrencyError, SafeStoreError, 
    EncryptionError, GraphDBError, GraphProcessingError, LLMCallbackError, 
    GraphError, QueryError 
)
from ..security.encryption import Encryptor
from ..store import DEFAULT_LOCK_TIMEOUT

# New callback signatures: they now receive the full prompt from GraphStore
LLMExecutorCallback = Callable[[str], str] # Input: full_prompt, Output: raw_llm_response_string

class GraphStore:
    GRAPH_FEATURES_ENABLED_KEY = "graph_features_enabled"

    # --- Default Prompts ---
    DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE = """
    Extract entities (nodes) and their relationships from the following text.
    Format the output strictly as a JSON object.
    **The entire JSON output MUST be enclosed in a single markdown code block starting with ```json and ending with ```.**
    
    JSON Structure Example:
    ```json
    {{
        "nodes": [
            {{"label": "Person", "properties": {{"name": "John Doe", "title": "Engineer"}}, "unique_id_key": "name"}},
            {{"label": "Company", "properties": {{"name": "Acme Corp", "industry": "Tech"}}, "unique_id_key": "name"}}
        ],
        "relationships": [
            {{"source_node_label": "Person", "source_node_unique_value": "John Doe",
             "target_node_label": "Company", "target_node_unique_value": "Acme Corp",
             "type": "WORKS_AT", "properties": {{"role": "Engineer"}}}}
        ]
    }}
    ```

    For each node:
    - "label": A general type (e.g., "Person", "Company", "Product", "Location", "Organization", "ResearchPaper", "University", "Journal").
    - "properties": Dictionary of relevant attributes.
    - "unique_id_key": A key from "properties" that uniquely identifies the node (e.g., "name"). This field MUST be a sibling to "label" and "properties".

    For each relationship:
    - "source_node_label", "source_node_unique_value": Identify the source node using its label and the value of its "unique_id_key".
    - "target_node_label", "target_node_unique_value": Identify the target node similarly.
    - "type": Relationship type in UPPER_SNAKE_CASE (e.g., "WORKS_AT", "CEO_OF", "PUBLISHED_IN").
    - "properties": Optional dictionary for relationship attributes.

    Text to process:
    ---
    {chunk_text}
    ---

    Extracted JSON (wrapped in ```json ... ```):
    """

    DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE = """
    Parse the following query to identify main entities ("seed_nodes").
    Format the output STRICTLY as a JSON object.
    **The entire JSON output MUST be enclosed in a single markdown code block starting with ```json and ending with ```.**

    JSON structure:
    ```json
    {{
        "seed_nodes": [ 
            {{"label": "EntityType", "properties": {{"identifying_prop_key": "value"}}, "unique_id_key": "identifying_prop_key"}} 
        ],
        "target_relationships": [ {{"type": "REL_TYPE", "direction": "outgoing|incoming|any"}} ],
        "target_node_labels": ["Label1", "Label2"],
        "max_depth": 1 
    }}
    ```
    - "seed_nodes": List of main entities from the query. "unique_id_key" must be a key in its "properties".
    - "target_relationships" (Optional): Desired relationship types and directions.
    - "target_node_labels" (Optional): Desired types of neighbor nodes.
    - "max_depth" (Optional, default 1): Traversal depth.

    Example Query: "Who is Evelyn Reed and what companies is she associated with?"
    Example JSON (wrapped in ```json ... ```):
    ```json
    {{
        "seed_nodes": [ {{"label": "Person", "properties": {{"name": "Evelyn Reed"}}, "unique_id_key": "name"}} ],
        "target_relationships": [ {{"type": "WORKS_AT", "direction": "any"}}, {{"type": "CEO_OF", "direction": "any"}} ],
        "target_node_labels": ["Company", "Organization"],
        "max_depth": 1
    }}
    ```

    If no clear entities, return `{{ "seed_nodes": [] }}`.

    Query: --- {natural_language_query} --- Parsed JSON Query (wrapped in ```json ... ```):
    """


    def __init__(
        self,
        db_path: Union[str, Path],
        llm_executor_callback: LLMExecutorCallback, # Changed callback name and signature
        encryption_key: Optional[str] = None,
        log_level: LogLevel = LogLevel.INFO,
        lock_timeout: int = DEFAULT_LOCK_TIMEOUT,
        graph_extraction_prompt_template: Optional[str] = None,
        query_parsing_prompt_template: Optional[str] = None
    ):
        self.db_path: str = str(Path(db_path).resolve())
        self.llm_executor: LLMExecutorCallback = llm_executor_callback # Store the new callback
        self.lock_timeout: int = lock_timeout
        _db_file_path = Path(self.db_path)
        self.lock_path: str = str(_db_file_path.parent / f"{_db_file_path.name}.lock")

        self.graph_extraction_prompt_template = graph_extraction_prompt_template or self.DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE
        self.query_parsing_prompt_template = query_parsing_prompt_template or self.DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE

        ASCIIColors.info(f"Initializing GraphStore with database: {self.db_path}")
        # ... (rest of __init__ remains the same: encryptor, locks, _connect_and_initialize_graph_db)
        self.conn: Optional[sqlite3.Connection] = None
        self._is_closed: bool = True
        try:
            self.encryptor = Encryptor(encryption_key)
            if self.encryptor.is_enabled: ASCIIColors.info("GraphStore: Encryption enabled for decrypting chunk text.")
        except (ConfigurationError, ValueError) as e: ASCIIColors.critical(f"GraphStore: Encryptor init failed: {e}"); raise
        self._instance_lock = threading.RLock(); self._file_lock = FileLock(self.lock_path, timeout=self.lock_timeout)
        try: self._connect_and_initialize_graph_db()
        except (DatabaseError, Timeout, ConcurrencyError, GraphError) as e: ASCIIColors.critical(f"GraphStore init failed: {e}"); raise


    def _get_graph_extraction_prompt(self, chunk_text: str) -> str:
        return self.graph_extraction_prompt_template.format(chunk_text=chunk_text)

    def _get_query_parsing_prompt(self, natural_language_query: str) -> str:
        return self.query_parsing_prompt_template.format(natural_language_query=natural_language_query)

    # ... ( _connect_and_initialize_graph_db, _ensure_connection, close, __enter__, __exit__ remain the same) ...
    def _connect_and_initialize_graph_db(self) -> None:
        init_lock = FileLock(self.lock_path, timeout=self.lock_timeout)
        try:
            with init_lock:
                ASCIIColors.debug("GraphStore: Acquired init lock for connection/schema setup.")
                if self.conn is None or self._is_closed:
                    self.conn = db.connect_db(self.db_path)
                    db.initialize_schema(self.conn) 
                    self._is_closed = False
                else:
                    ASCIIColors.debug("GraphStore: Connection already established.")
                assert self.conn is not None
                cursor = self.conn.cursor()
                try:
                    cursor.execute("BEGIN")
                    graph_enabled = db.get_store_metadata(self.conn, self.GRAPH_FEATURES_ENABLED_KEY)
                    if graph_enabled != "true":
                        db.set_store_metadata(self.conn, self.GRAPH_FEATURES_ENABLED_KEY, "true")
                        ASCIIColors.info("GraphStore: Marked graph features as enabled in database metadata.")
                    else:
                        ASCIIColors.debug("GraphStore: Graph features already marked as enabled in database.")
                    self.conn.commit()
                except sqlite3.Error as e_trans:
                    ASCIIColors.error(f"GraphStore: Transaction error during metadata check/set: {e_trans}")
                    if self.conn: self.conn.rollback()
                    raise GraphDBError(f"Failed to set graph features metadata: {e_trans}") from e_trans
            ASCIIColors.debug("GraphStore: Released init lock.")
        except Timeout as e:
            msg = f"GraphStore: Timeout acquiring initial lock for DB connection/setup at {self.lock_path}"
            ASCIIColors.error(msg)
            if self.conn:
                try: self.conn.close()
                except Exception: pass
                finally: self.conn = None; self._is_closed = True
            raise ConcurrencyError(msg) from e
        except (DatabaseError, GraphDBError) as e:
            ASCIIColors.error(f"GraphStore: Database error during initial setup: {e}")
            if self.conn:
                try: self.conn.close()
                except Exception: pass
                finally: self.conn = None; self._is_closed = True
            raise
        except Exception as e_unexp:
            msg = f"GraphStore: Unexpected error during initial DB connection/setup: {e_unexp}"
            ASCIIColors.error(msg, exc_info=True)
            if self.conn:
                try: self.conn.close()
                except Exception: pass
                finally: self.conn = None; self._is_closed = True
            raise GraphError(msg) from e_unexp

    def _ensure_connection(self) -> None:
        if self._is_closed or self.conn is None:
            raise ConnectionError("GraphStore: Database connection is closed or not available.")

    def close(self) -> None:
        with self._instance_lock:
            if self._is_closed: 
                ASCIIColors.debug("GraphStore: Connection already closed.")
                return
            if self.conn:
                ASCIIColors.debug("GraphStore: Closing database connection.")
                try:
                    self.conn.close()
                except Exception as e: 
                    ASCIIColors.warning(f"GraphStore: Error closing DB connection: {e}")
                finally:
                    self.conn = None
                    self._is_closed = True
            ASCIIColors.info("GraphStore connection closed.")

    def __enter__(self):
        with self._instance_lock:
            if self._is_closed or self.conn is None:
                ASCIIColors.debug("GraphStore: Re-establishing connection on context manager entry.")
                try:
                    self._connect_and_initialize_graph_db()
                except (DatabaseError, ConcurrencyError, GraphError) as e:
                    ASCIIColors.error(f"GraphStore: Failed to re-establish connection in __enter__: {e}")
                    raise
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type: 
            ASCIIColors.error(f"GraphStore: Context closed with error: {exc_val}", exc_info=(exc_type, exc_val, exc_tb))
        else:
            ASCIIColors.debug("GraphStore: Context closed cleanly.")


    def _process_chunk_for_graph_impl(self, chunk_id: int) -> None:
        assert self.conn is not None
        cursor = self.conn.cursor()
        decrypted_chunk_text: str
        try: # Fetch and decrypt chunk text
            original_text_factory = self.conn.text_factory
            self.conn.text_factory = bytes
            cursor.execute("SELECT chunk_text, is_encrypted FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            self.conn.text_factory = original_text_factory
            if not row: raise GraphProcessingError(f"Chunk {chunk_id} not found.")
            chunk_text_data, is_encrypted_flag = row[0], row[1]
            if is_encrypted_flag:
                if not self.encryptor.is_enabled: raise ConfigurationError(f"Chunk {chunk_id} encrypted, no key.")
                if not isinstance(chunk_text_data, bytes): raise EncryptionError(f"Chunk {chunk_id} encrypted but data not bytes.")
                decrypted_chunk_text = self.encryptor.decrypt(chunk_text_data)
            else:
                decrypted_chunk_text = chunk_text_data.decode('utf-8') if isinstance(chunk_text_data, bytes) else str(chunk_text_data)
        except (sqlite3.Error, EncryptionError, GraphProcessingError, ConfigurationError) as e:
            raise GraphProcessingError(f"Error preparing chunk {chunk_id} for LLM: {e}") from e

        extraction_prompt = self._get_graph_extraction_prompt(decrypted_chunk_text)
        ASCIIColors.debug(f"GraphStore: Processing chunk {chunk_id} with LLM (graph extraction).")
        
        raw_llm_response: str
        try:
            raw_llm_response = self.llm_executor(extraction_prompt)
        except Exception as e:
            raise LLMCallbackError(f"LLM executor for graph extraction failed for chunk {chunk_id}: {e}") from e

        if not raw_llm_response or not raw_llm_response.strip():
            ASCIIColors.warning(f"LLM returned empty response for graph extraction from chunk {chunk_id}.")
            return

        # Standardized JSON cleaning (assuming LLM was asked for markdown block)
        json_candidate = raw_llm_response.strip()
        if json_candidate.startswith("```json"): json_candidate = json_candidate[len("```json"):].strip()
        if json_candidate.startswith("```"): json_candidate = json_candidate[len("```"):].strip()
        if json_candidate.endswith("```"): json_candidate = json_candidate[:-len("```")].strip()
        first_brace, last_brace = json_candidate.find('{'), json_candidate.rfind('}')
        if first_brace != -1 and last_brace > first_brace: json_candidate = json_candidate[first_brace : last_brace+1]
        else: 
            if not json_candidate.strip() or not (json_candidate.startswith("{") and json_candidate.endswith("}")):
                ASCIIColors.warning(f"LLM output for chunk {chunk_id} doesn't look like JSON after cleaning: {json_candidate[:200]}"); return

        try:
            llm_output = json.loads(json_candidate)
        except json.JSONDecodeError as e:
            ASCIIColors.error(f"Failed to decode JSON from LLM graph extraction for chunk {chunk_id}: {e}. Candidate: {json_candidate[:500]}"); return

        if not isinstance(llm_output, dict) or "nodes" not in llm_output or "relationships" not in llm_output:
            raise LLMCallbackError(f"LLM output for chunk {chunk_id} malformed (missing nodes/relationships).")

        processed_nodes_map: Dict[Tuple[str, str], int] = {} 
        for node_data in llm_output.get("nodes", []):
            # ... (node processing logic from previous version, using db.add_or_update_graph_node)
            if not (isinstance(node_data, dict) and node_data.get("label") and isinstance(node_data.get("properties"), dict)):
                ASCIIColors.warning(f"GraphStore: Skipping malformed node data from LLM for chunk {chunk_id}: {node_data}"); continue
            label, properties_dict = str(node_data["label"]), node_data["properties"]
            uid_key = node_data.get("unique_id_key") 
            if not uid_key and isinstance(properties_dict, dict): uid_key = properties_dict.get("unique_id_key")
            if not uid_key: ASCIIColors.warning(f"GraphStore: Skipping node (label: {label}) missing 'unique_id_key' for chunk {chunk_id}: {node_data}"); continue
            uid_key = str(uid_key)
            if uid_key not in properties_dict: ASCIIColors.warning(f"GraphStore: Node (label: {label}) 'unique_id_key' ('{uid_key}') not in 'properties' for chunk {chunk_id}. Skipping. Props: {properties_dict}"); continue
            uid_value, normalized_uid_value = str(properties_dict[uid_key]), str(properties_dict[uid_key]).strip().lower()
            unique_signature = f"{label}:{uid_key}:{normalized_uid_value}"
            try:
                node_id = db.add_or_update_graph_node(self.conn, label, properties_dict, unique_signature)
                db.link_node_to_chunk(self.conn, node_id, chunk_id)
                processed_nodes_map[(label, normalized_uid_value)] = node_id
            except (GraphDBError, json.JSONDecodeError) as e: ASCIIColors.error(f"GraphStore: Error storing node (Sig: {unique_signature}) from chunk {chunk_id}: {e}")

        for rel_data in llm_output.get("relationships", []):
            # ... (relationship processing logic from previous version)
            if not (isinstance(rel_data, dict) and all(k in rel_data for k in ["source_node_label", "source_node_unique_value", "target_node_label", "target_node_unique_value", "type"])):
                ASCIIColors.warning(f"GraphStore: Skipping malformed relationship data from LLM for chunk {chunk_id}: {rel_data}"); continue
            src_label, src_uid_val = str(rel_data["source_node_label"]), str(rel_data["source_node_unique_value"])
            tgt_label, tgt_uid_val = str(rel_data["target_node_label"]), str(rel_data["target_node_unique_value"])
            rel_type, rel_props_dict = str(rel_data["type"]), rel_data.get("properties", {})
            normalized_src_uid_val, normalized_tgt_uid_val = src_uid_val.strip().lower(), tgt_uid_val.strip().lower()
            source_node_id, target_node_id = processed_nodes_map.get((src_label, normalized_src_uid_val)), processed_nodes_map.get((tgt_label, normalized_tgt_uid_val))
            if source_node_id is None or target_node_id is None: ASCIIColors.warning(f"GraphStore: Skipping rel '{rel_type}' missing src/tgt node from chunk {chunk_id}. Src:({src_label},{normalized_src_uid_val}), Tgt:({tgt_label},{normalized_tgt_uid_val}). Map keys: {list(processed_nodes_map.keys())}"); continue
            try:
                rel_props_json = json.dumps(rel_props_dict) if rel_props_dict else None
                db.add_graph_relationship(self.conn, source_node_id, target_node_id, rel_type, rel_props_json)
            except (GraphDBError, json.JSONDecodeError) as e: ASCIIColors.error(f"GraphStore: Error storing rel '{rel_type}' ({source_node_id}->{target_node_id}) from chunk {chunk_id}: {e}")


    # --- Public Graph Building Methods (process_chunk_for_graph, build_graph_for_document, build_graph_for_all_documents) ---
    # These remain structurally the same as in the previous full version, calling the updated _process_chunk_for_graph_impl
    def process_chunk_for_graph(self, chunk_id: int) -> None:
        with self._instance_lock:
            ASCIIColors.debug(f"GraphStore: Attempting to acquire write lock for process_chunk_for_graph: chunk_id {chunk_id}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"GraphStore: Write lock acquired for process_chunk_for_graph: chunk_id {chunk_id}")
                    self._ensure_connection(); assert self.conn is not None
                    try:
                        self.conn.execute("BEGIN")
                        self._process_chunk_for_graph_impl(chunk_id)
                        db.mark_chunks_graph_processed(self.conn, [chunk_id])
                        self.conn.commit()
                        ASCIIColors.success(f"GraphStore: Successfully processed chunk {chunk_id} for graph.")
                    except (GraphDBError, GraphProcessingError, LLMCallbackError, EncryptionError, ConfigurationError, DatabaseError) as e:
                        ASCIIColors.error(f"GraphStore: Error processing chunk {chunk_id} for graph: {e}")
                        if self.conn: self.conn.rollback(); raise 
                    except Exception as e_unexp: 
                        ASCIIColors.error(f"GraphStore: Unexpected error processing chunk {chunk_id} for graph: {e_unexp}", exc_info=True)
                        if self.conn: self.conn.rollback(); raise GraphProcessingError(f"Unexpected error for chunk {chunk_id}: {e_unexp}") from e_unexp
                ASCIIColors.debug(f"GraphStore: Write lock released for process_chunk_for_graph: chunk_id {chunk_id}")
            except Timeout as e_lock:
                msg = f"GraphStore: Timeout ({self.lock_timeout}s) acquiring write lock for process_chunk_for_graph: chunk_id {chunk_id}"
                ASCIIColors.error(msg); raise ConcurrencyError(msg) from e_lock

    def build_graph_for_document(self, doc_id: int) -> None:
        with self._instance_lock:
            ASCIIColors.debug(f"GraphStore: Attempting to acquire write lock for build_graph_for_document: doc_id {doc_id}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"GraphStore: Write lock acquired for build_graph_for_document: doc_id {doc_id}")
                    self._ensure_connection(); assert self.conn is not None
                    cursor = self.conn.cursor()
                    try: cursor.execute("SELECT chunk_id FROM chunks WHERE doc_id = ? AND graph_processed_at IS NULL ORDER BY chunk_seq", (doc_id,))
                    except sqlite3.Error as e: raise GraphDBError(f"Database error fetching chunks for document {doc_id}: {e}") from e
                    chunk_ids_to_process = [row[0] for row in cursor.fetchall()]
                    if not chunk_ids_to_process: ASCIIColors.info(f"GraphStore: No unprocessed chunks found for document {doc_id}. Nothing to do."); return
                    ASCIIColors.info(f"GraphStore: Processing {len(chunk_ids_to_process)} chunks for document {doc_id} to build graph.")
                    try:
                        self.conn.execute("BEGIN")
                        for chunk_id in chunk_ids_to_process: self._process_chunk_for_graph_impl(chunk_id) 
                        db.mark_chunks_graph_processed(self.conn, chunk_ids_to_process) 
                        self.conn.commit()
                        ASCIIColors.success(f"GraphStore: Successfully built graph for document {doc_id}.")
                    except (GraphDBError, GraphProcessingError, LLMCallbackError, EncryptionError, ConfigurationError, DatabaseError) as e:
                        ASCIIColors.error(f"GraphStore: Error building graph for document {doc_id}: {e}")
                        if self.conn: self.conn.rollback(); raise
                    except Exception as e_unexp:
                        ASCIIColors.error(f"GraphStore: Unexpected error building graph for document {doc_id}: {e_unexp}", exc_info=True)
                        if self.conn: self.conn.rollback(); raise GraphProcessingError(f"Unexpected error for document {doc_id}: {e_unexp}") from e_unexp
                ASCIIColors.debug(f"GraphStore: Write lock released for build_graph_for_document: doc_id {doc_id}")
            except Timeout as e_lock:
                msg = f"GraphStore: Timeout ({self.lock_timeout}s) acquiring write lock for build_graph_for_document: doc_id {doc_id}"
                ASCIIColors.error(msg); raise ConcurrencyError(msg) from e_lock

    def build_graph_for_all_documents(self, batch_size_chunks: int = 20) -> None:
        if batch_size_chunks <= 0: raise ValueError("batch_size_chunks must be positive.")
        with self._instance_lock:
            ASCIIColors.debug(f"GraphStore: Attempting to acquire write lock for build_graph_for_all_documents.")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"GraphStore: Write lock acquired for build_graph_for_all_documents.")
                    self._ensure_connection(); assert self.conn is not None
                    processed_total = 0
                    while True: 
                        cursor = self.conn.cursor()
                        try: cursor.execute("SELECT chunk_id FROM chunks WHERE graph_processed_at IS NULL ORDER BY doc_id, chunk_seq LIMIT ?", (batch_size_chunks,))
                        except sqlite3.Error as e: raise GraphDBError(f"Database error fetching batch of unprocessed chunks: {e}") from e
                        chunk_ids_batch = [row[0] for row in cursor.fetchall()]
                        if not chunk_ids_batch: ASCIIColors.info(f"GraphStore: No more unprocessed chunks found. Total processed in this run: {processed_total}."); break
                        ASCIIColors.info(f"GraphStore: Processing batch of {len(chunk_ids_batch)} chunks for graph (total processed so far: {processed_total}).")
                        try:
                            self.conn.execute("BEGIN")
                            for chunk_id in chunk_ids_batch: self._process_chunk_for_graph_impl(chunk_id) 
                            db.mark_chunks_graph_processed(self.conn, chunk_ids_batch) 
                            self.conn.commit()
                            processed_total += len(chunk_ids_batch)
                            ASCIIColors.success(f"GraphStore: Successfully processed batch of {len(chunk_ids_batch)} chunks.")
                        except (GraphDBError, GraphProcessingError, LLMCallbackError, EncryptionError, ConfigurationError, DatabaseError) as e:
                            ASCIIColors.error(f"GraphStore: Error processing batch for graph: {e}. Rolling back batch.")
                            if self.conn: self.conn.rollback(); raise
                        except Exception as e_unexp:
                            ASCIIColors.error(f"GraphStore: Unexpected error processing batch for graph: {e_unexp}", exc_info=True)
                            if self.conn: self.conn.rollback(); raise GraphProcessingError(f"Unexpected error during batch processing: {e_unexp}") from e_unexp
                    ASCIIColors.success(f"GraphStore: Finished building graph for all available documents. Total chunks processed in this run: {processed_total}.")
                ASCIIColors.debug(f"GraphStore: Write lock released for build_graph_for_all_documents.")
            except Timeout as e_lock:
                msg = f"GraphStore: Timeout ({self.lock_timeout}s) acquiring write lock for build_graph_for_all_documents."
                ASCIIColors.error(msg); raise ConcurrencyError(msg) from e_lock

    # --- Graph Read Methods ---
    # ... (get_node_details, get_nodes_by_label, get_relationships, find_neighbors, get_chunks_for_node - all remain the same as previous full version)
    def get_node_details(self, node_id: int) -> Optional[Dict[str, Any]]:
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            try: return db.get_node_details_db(self.conn, node_id)
            except GraphDBError as e: ASCIIColors.error(f"GraphStore: Error getting node details for ID {node_id}: {e}"); raise
            except Exception as e_unexp: ASCIIColors.error(f"GraphStore: Unexpected error getting node details for ID {node_id}: {e_unexp}", exc_info=True); raise GraphError(f"Unexpected error getting node details: {e_unexp}") from e_unexp

    def get_nodes_by_label(self, label: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            cursor = self.conn.cursor(); nodes_found: List[Dict[str, Any]] = []
            try:
                cursor.execute("SELECT node_id, node_label, node_properties, unique_signature FROM graph_nodes WHERE node_label = ? LIMIT ?", (label, limit))
                for row in cursor.fetchall():
                    properties = json.loads(row[2]) if row[2] else {}
                    nodes_found.append({"node_id": row[0], "label": row[1], "properties": properties, "unique_signature": row[3]})
                return nodes_found
            except sqlite3.Error as e: raise GraphDBError(f"DB error finding nodes by label '{label}': {e}") from e
            except json.JSONDecodeError as e: raise GraphDBError(f"JSON decode error for properties while finding nodes by label '{label}': {e}") from e
            except Exception as e_unexp: raise GraphError(f"Unexpected error getting nodes by label: {e_unexp}") from e_unexp

    def get_relationships(self, node_id: int, relationship_type: Optional[str] = None, direction: str = "any", limit: int = 50) -> List[Dict[str, Any]]:
        if direction not in ["outgoing", "incoming", "any"]: raise ValueError("Invalid direction.")
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            try: return db.get_relationships_for_node_db(self.conn, node_id, relationship_type, direction, limit)
            except (GraphDBError, ValueError) as e: ASCIIColors.error(f"GraphStore: Error getting relationships for node ID {node_id}: {e}"); raise
            except Exception as e_unexp: ASCIIColors.error(f"GraphStore: Unexpected error getting relationships for node ID {node_id}: {e_unexp}", exc_info=True); raise GraphError(f"Unexpected error getting relationships: {e_unexp}") from e_unexp

    def find_neighbors(self, node_id: int, relationship_type: Optional[str] = None, direction: str = "outgoing", limit: int = 50) -> List[Dict[str, Any]]:
        if direction not in ["outgoing", "incoming", "any"]: raise ValueError("Invalid direction.")
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            try:
                relationships = db.get_relationships_for_node_db(self.conn, node_id, relationship_type, direction, limit)
                neighbor_nodes: List[Dict[str, Any]] = []; seen_neighbor_ids = set()
                for rel in relationships:
                    neighbor_node_data: Optional[Dict[str, Any]] = None
                    if direction == "outgoing" and rel.get("target_node", {}).get("node_id") != node_id: neighbor_node_data = rel.get("target_node")
                    elif direction == "incoming" and rel.get("source_node", {}).get("node_id") != node_id: neighbor_node_data = rel.get("source_node")
                    elif direction == "any":
                        if rel.get("source_node", {}).get("node_id") == node_id and rel.get("target_node", {}).get("node_id") != node_id: neighbor_node_data = rel.get("target_node")
                        elif rel.get("target_node", {}).get("node_id") == node_id and rel.get("source_node", {}).get("node_id") != node_id: neighbor_node_data = rel.get("source_node")
                    if neighbor_node_data and neighbor_node_data.get("node_id") not in seen_neighbor_ids:
                        neighbor_nodes.append(neighbor_node_data); seen_neighbor_ids.add(neighbor_node_data["node_id"])
                return neighbor_nodes[:limit] 
            except (GraphDBError, ValueError) as e: ASCIIColors.error(f"GraphStore: Error finding neighbors for node ID {node_id}: {e}"); raise
            except Exception as e_unexp: ASCIIColors.error(f"GraphStore: Unexpected error finding neighbors for node ID {node_id}: {e_unexp}", exc_info=True); raise GraphError(f"Unexpected error finding neighbors: {e_unexp}") from e_unexp

    def get_chunks_for_node(self, node_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            try:
                linked_chunk_ids_map = db.get_chunk_ids_for_nodes_db(self.conn, [node_id])
                chunk_ids = linked_chunk_ids_map.get(node_id, [])
                if not chunk_ids: return []
                chunk_details = db.get_chunk_details_db(self.conn, chunk_ids[:limit], self.encryptor)
                return chunk_details
            except GraphDBError as e: ASCIIColors.error(f"GraphStore: Error getting chunks for node ID {node_id}: {e}"); raise
            except Exception as e_unexp: ASCIIColors.error(f"GraphStore: Unexpected error getting chunks for node ID {node_id}: {e_unexp}", exc_info=True); raise GraphError(f"Unexpected error getting chunks for node: {e_unexp}") from e_unexp


    # --- Main Query Method ---
    def query_graph(
        self, 
        natural_language_query: str, 
        output_mode: str = "chunks_summary", 
        llm_parsed_query_override: Optional[Dict[str, Any]] = None
    ) -> Any:
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            ASCIIColors.info(f"GraphStore: Query '{natural_language_query[:50]}...', mode: {output_mode}")

            if output_mode not in ["chunks_summary", "graph_only", "full"]:
                raise ValueError("Invalid output_mode.")

            parsed_query: Dict[str, Any]
            if llm_parsed_query_override:
                parsed_query = llm_parsed_query_override
            else:
                if not self.llm_executor: raise ConfigurationError("GraphStore needs llm_executor_callback.")
                query_parsing_prompt = self._get_query_parsing_prompt(natural_language_query)
                ASCIIColors.debug("GraphStore: Parsing NLQ with LLM...")
                raw_llm_response: str
                try:
                    raw_llm_response = self.llm_executor(query_parsing_prompt)
                except Exception as e: raise LLMCallbackError(f"LLM executor for query parsing failed: {e}") from e
                
                if not raw_llm_response or not raw_llm_response.strip():
                    ASCIIColors.warning("LLM returned empty for query parsing."); 
                    return self._empty_query_result(output_mode)

                json_candidate = raw_llm_response.strip()
                # (Apply same robust JSON cleaning as in _process_chunk_for_graph_impl)
                if json_candidate.startswith("```json"): json_candidate = json_candidate[len("```json"):].strip()
                if json_candidate.startswith("```"): json_candidate = json_candidate[len("```"):].strip()
                if json_candidate.endswith("```"): json_candidate = json_candidate[:-len("```")].strip()
                first_brace, last_brace = json_candidate.find('{'), json_candidate.rfind('}')
                if first_brace != -1 and last_brace > first_brace: json_candidate = json_candidate[first_brace : last_brace+1]
                else: 
                    if not json_candidate.strip() or not (json_candidate.startswith("{") and json_candidate.endswith("}")):
                        ASCIIColors.warning(f"LLM query parse output not JSON: {json_candidate[:200]}"); return self._empty_query_result(output_mode)
                try:
                    parsed_query = json.loads(json_candidate)
                except json.JSONDecodeError as e:
                    raise LLMCallbackError(f"Failed to decode JSON from LLM query parser: {e}. Candidate: {json_candidate[:500]}") from e
            
            ASCIIColors.debug(f"GraphStore: LLM parsed query (raw): {json.dumps(parsed_query, indent=2)}")
            # Validate and default parsed_query structure
            if not isinstance(parsed_query, dict): raise LLMCallbackError("Parsed query is not a dict.")
            if not isinstance(parsed_query.get("seed_nodes"), list): parsed_query["seed_nodes"] = []
            parsed_query.setdefault("target_relationships", [])
            parsed_query.setdefault("target_node_labels", [])
            parsed_query.setdefault("max_depth", 1)
            ASCIIColors.debug(f"GraphStore: LLM parsed query (validated): {json.dumps(parsed_query, indent=2)}")


            current_seed_node_ids: List[int] = []
            for spec in parsed_query["seed_nodes"]:
                if not (isinstance(spec,dict) and spec.get("label") and isinstance(spec.get("properties"),dict) and spec.get("unique_id_key") and spec["unique_id_key"] in spec["properties"]):
                    ASCIIColors.warning(f"Invalid seed_node spec: {spec}. Skipping."); continue
                try:
                    matches = db.get_graph_node_by_label_and_property(self.conn, str(spec["label"]), str(spec["unique_id_key"]), str(spec["properties"][spec["unique_id_key"]]))
                    for node_info in matches: 
                        if node_info["node_id"] not in current_seed_node_ids: current_seed_node_ids.append(node_info["node_id"])
                except GraphDBError as e: ASCIIColors.error(f"DB error finding seed node for {spec}: {e}")
            
            if not current_seed_node_ids:
                ASCIIColors.warning("No seed nodes found in graph for query."); return self._empty_query_result(output_mode)
            ASCIIColors.debug(f"Initial seed node IDs: {current_seed_node_ids}")

            subgraph_nodes: Dict[int, Dict[str, Any]] = {} 
            temp_rel_store_by_id: Dict[int, Dict[str,Any]] = {}
            queue: List[Tuple[int, int]] = []
            for seed_id in current_seed_node_ids:
                if seed_id not in subgraph_nodes:
                    details = db.get_node_details_db(self.conn, seed_id)
                    if details: subgraph_nodes[seed_id] = details
                queue.append((seed_id, 0))
            visited_in_bfs_queue = set(current_seed_node_ids) 

            head = 0
            while head < len(queue):
                curr_node_id, current_depth = queue[head]; head += 1
                if current_depth >= parsed_query["max_depth"]: continue
                
                rels_to_explore = parsed_query["target_relationships"] if parsed_query["target_relationships"] else [{"type": None, "direction": "any"}]
                for rel_spec in rels_to_explore:
                    connected_rels = db.get_relationships_for_node_db(self.conn, curr_node_id, rel_spec.get("type"), rel_spec.get("direction", "any"), limit=100)
                    for rel_detail in connected_rels:
                        if rel_detail["source_node_id"] in subgraph_nodes and rel_detail["target_node_id"] in subgraph_nodes: # Already handled if both ends known
                             temp_rel_store_by_id[rel_detail["relationship_id"]] = rel_detail # Add/update
                        
                        neighbor_node_info: Optional[Dict[str, Any]] = None
                        if rel_detail["source_node_id"] == curr_node_id: neighbor_node_info = rel_detail.get("target_node")
                        elif rel_detail["target_node_id"] == curr_node_id: neighbor_node_info = rel_detail.get("source_node")
                        
                        if neighbor_node_info:
                            neighbor_id, neighbor_label = neighbor_node_info["node_id"], neighbor_node_info["label"]
                            if parsed_query["target_node_labels"] and neighbor_label not in parsed_query["target_node_labels"]: continue 
                            if neighbor_id not in subgraph_nodes: subgraph_nodes[neighbor_id] = neighbor_node_info
                            # Add relationship again here to ensure it's captured if one end was newly added
                            if rel_detail["source_node_id"] in subgraph_nodes and rel_detail["target_node_id"] in subgraph_nodes:
                                temp_rel_store_by_id[rel_detail["relationship_id"]] = rel_detail

                            if neighbor_id not in visited_in_bfs_queue:
                                queue.append((neighbor_id, current_depth + 1)); visited_in_bfs_queue.add(neighbor_id)
            
            final_graph_data = {"nodes": list(subgraph_nodes.values()), "relationships": list(temp_rel_store_by_id.values())}
            ASCIIColors.debug(f"Traversal found {len(final_graph_data['nodes'])} nodes, {len(final_graph_data['relationships'])} rels.")
            return self._format_query_output(final_graph_data, output_mode)

    def _empty_query_result(self, output_mode: str) -> Any:
        # ... (remains the same)
        if output_mode == "chunks_summary": return []
        if output_mode == "graph_only": return {"nodes": [], "relationships": []}
        if output_mode == "full": return {"graph": {"nodes": [], "relationships": []}, "chunks": []}
        return None 

    def _format_query_output(self, graph_data: Dict[str, Any], output_mode: str) -> Any:
        # ... (remains the same)
        assert self.conn is not None 
        chunk_results: List[Dict[str, Any]] = []
        if output_mode in ["chunks_summary", "full"]:
            if graph_data["nodes"]:
                node_ids_in_subgraph = [n["node_id"] for n in graph_data["nodes"]]
                node_to_chunks_map = db.get_chunk_ids_for_nodes_db(self.conn, node_ids_in_subgraph)
                all_linked_chunk_ids = set()
                for ids_list in node_to_chunks_map.values(): all_linked_chunk_ids.update(ids_list)
                
                if all_linked_chunk_ids:
                    chunk_details_list = db.get_chunk_details_db(self.conn, list(all_linked_chunk_ids), self.encryptor)
                    for chunk_detail in chunk_details_list:
                        linked_nodes_info = []
                        for node_id, linked_chunks in node_to_chunks_map.items():
                            if chunk_detail["chunk_id"] in linked_chunks:
                                node_info = next((n for n in graph_data["nodes"] if n["node_id"] == node_id), None)
                                if node_info: linked_nodes_info.append({"node_id": node_id, "label": node_info["label"]})
                        chunk_detail["linked_graph_nodes"] = linked_nodes_info
                        chunk_results.append(chunk_detail)
                    ASCIIColors.debug(f"GraphStore: Retrieved {len(chunk_results)} chunk summaries linked to subgraph.")
            else: ASCIIColors.debug("GraphStore: No nodes in subgraph, so no chunks to retrieve.")

        if output_mode == "chunks_summary": return chunk_results
        if output_mode == "graph_only": return graph_data
        if output_mode == "full": return {"graph": graph_data, "chunks": chunk_results}
        return None 

    def update_node_properties(self, node_id: int, new_properties: Dict[str, Any], merge_strategy: str = "merge_overwrite_new") -> bool:
        # ... (remains the same)
        with self._instance_lock:
            ASCIIColors.debug(f"GraphStore: Attempting to acquire write lock for update_node_properties: node_id {node_id}")
            try:
                with self._file_lock:
                    ASCIIColors.info(f"GraphStore: Write lock acquired for update_node_properties: node_id {node_id}")
                    self._ensure_connection(); assert self.conn is not None
                    try:
                        self.conn.execute("BEGIN")
                        success = db.update_graph_node_properties_db(self.conn, node_id, new_properties, merge_strategy)
                        if success: self.conn.commit(); ASCIIColors.success(f"GraphStore: Updated properties for node {node_id}.")
                        else: self.conn.rollback(); ASCIIColors.info(f"GraphStore: No update for node {node_id}.")
                        return success
                    except (GraphDBError, DatabaseError) as e: ASCIIColors.error(f"GraphStore: Error updating node {node_id} props: {e}"); self.conn.rollback(); raise
                    except Exception as e_unexp: ASCIIColors.error(f"GraphStore: Unexpected error updating node {node_id} props: {e_unexp}", exc_info=True); self.conn.rollback(); raise GraphProcessingError(f"Unexpected error updating node {node_id}: {e_unexp}") from e_unexp
                ASCIIColors.debug(f"GraphStore: Write lock released for update_node_properties: node_id {node_id}")
            except Timeout as e_lock: msg = f"GraphStore: Timeout acquiring write lock for update_node_properties: node_id {node_id}"; ASCIIColors.error(msg); raise ConcurrencyError(msg) from e_lock
            return False 
        
        
    def get_all_nodes_for_visualization(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetches all nodes with minimal data for visualization."""
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            cursor = self.conn.cursor()
            nodes: List[Dict[str, Any]] = []
            try:
                # Fetching id, label, and some key property for display if possible
                # For Vis.js, nodes need an 'id' and 'label' (for display text)
                cursor.execute(
                    "SELECT node_id, node_label, node_properties FROM graph_nodes LIMIT ?", (limit,)
                )
                for row in cursor.fetchall():
                    props = json.loads(row[2]) if row[2] else {}
                    # Try to get a 'name' or 'title' property for the display label, fallback to node_label
                    display_label = props.get('name', props.get('title', row[1])) 
                    nodes.append({
                        "id": row[0], # Vis.js expects 'id'
                        "label": f"{display_label} ({row[1]})", # Display label for Vis.js
                        "title": json.dumps(props, indent=2), # Tooltip for Vis.js
                        "group": row[1], # Group by original label for Vis.js styling
                        "properties": props, # Full properties for potential client-side use
                        "original_label": row[1] # Store original label separately
                    })
                return nodes
            except Exception as e:
                ASCIIColors.error(f"Error fetching all nodes for visualization: {e}"); raise GraphDBError("Failed to fetch all nodes") from e

    def get_all_relationships_for_visualization(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetches all relationships with minimal data for visualization."""
        with self._instance_lock:
            self._ensure_connection(); assert self.conn is not None
            cursor = self.conn.cursor()
            relationships: List[Dict[str, Any]] = []
            try:
                # For Vis.js, edges need 'from', 'to', and optionally 'label'
                cursor.execute(
                    "SELECT relationship_id, source_node_id, target_node_id, relationship_type, relationship_properties FROM graph_relationships LIMIT ?", (limit,)
                )
                for row in cursor.fetchall():
                    props = json.loads(row[4]) if row[4] else {}
                    relationships.append({
                        "id": row[0], # Vis.js can use edge id
                        "from": row[1], # Vis.js expects 'from'
                        "to": row[2],   # Vis.js expects 'to'
                        "label": row[3], # Vis.js can display this on the edge
                        "title": json.dumps(props, indent=2), # Tooltip for Vis.js
                        "properties": props # Full properties
                    })
                return relationships
            except Exception as e:
                ASCIIColors.error(f"Error fetching all relationships for visualization: {e}"); raise GraphDBError("Failed to fetch all relationships") from e