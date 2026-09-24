from __future__ import annotations
import sqlite3
import threading
import json
import uuid
import re
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any, Tuple, TYPE_CHECKING, Set, Union
from ascii_colors import ASCIIColors, trace_exception
from ..core import db
from ..core.exceptions import (
    DatabaseError, ConfigurationError, GraphDBError, GraphProcessingError, LLMCallbackError,
    GraphError, QueryError, NodeNotFoundError, RelationshipNotFoundError
)
from ..utils.json_parsing import robust_json_parser
from ..vectorization.base import BaseVectorizer
from .sparql.engine import SparqlEngine

if TYPE_CHECKING:
    from ..store import SafeStore
    from .cognitive_memory import CognitiveMemoryStore

# Callback signatures
LLMExecutorCallback = Callable[[str], str]
ProgressCallback = Callable[[float, str], None]

def load_prompt(file_name: str) -> str:
    """Loads a prompt template from the 'prompts' subdirectory."""
    path = Path(__file__).parent / "prompts" / f"{file_name}.md"
    return path.read_text()

class GraphStore:
    """
    Manages a knowledge graph within a SafeStore database.
    Provides SPARQL 1.1 querying, graph building, entity fusion, and relational traversal.
    """
    GRAPH_FEATURES_ENABLED_KEY = "graph_features_enabled"
    DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE = load_prompt("graph_extraction_prompt")
    DEFAULT_GRAPH_EXTRACTION_WITH_ONTOLOGY_PROMPT_TEMPLATE = load_prompt("graph_extraction_prompt_with_ontology")
    DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE = load_prompt("query_parsing_prompt")
    DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE = load_prompt("entity_fusion_prompt")
    DEFAULT_SPARQL_GENERATION_PROMPT_TEMPLATE = load_prompt("sparql_generation_prompt")

    # Class-level defaults for backward compatibility and resilient resolution
    graph_extraction_prompt_template: str = DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE
    query_parsing_prompt_template: str = DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE
    entity_fusion_prompt_template: str = DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE
    sparql_generation_prompt_template: str = DEFAULT_SPARQL_GENERATION_PROMPT_TEMPLATE

    def __init__(
        self,
        store: "SafeStore",
        llm_executor_callback: Optional[LLMExecutorCallback] = None,
        lollms_client: Optional[Any] = None,
        ontology: Optional[Union[Dict[str, Any], str]] = None,
        graph_extraction_prompt_template: Optional[str] = None,
        query_parsing_prompt_template: Optional[str] = None,
        entity_fusion_prompt_template: Optional[str] = None,
    ):
        self.store = store
        self.ontology = ontology

        # Soft, lazy resolution of lollms_client to avoid circular dependencies
        self.lollms_client = lollms_client
        if self.lollms_client is None:
            self.lollms_client = self._try_resolve_lollms_client()

        self.llm_executor = llm_executor_callback
        if not self.llm_executor and not self.lollms_client:
            self.llm_executor = lambda p: '{"nodes": [], "relationships": []}'

        self.graph_extraction_prompt_template = graph_extraction_prompt_template or self.DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE
        self.query_parsing_prompt_template = query_parsing_prompt_template or self.DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE
        self.entity_fusion_prompt_template = entity_fusion_prompt_template or self.DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE
        self.sparql_generation_prompt_template = self.DEFAULT_SPARQL_GENERATION_PROMPT_TEMPLATE
        self._sparql_engine: Optional[SparqlEngine] = None
        self._cognitive_memory: Optional[CognitiveMemoryStore] = None

        ASCIIColors.info(f"Initializing GraphStore with shared SafeStore for database: {self.store.db_path}")
        self._initialize_graph_features()

    @staticmethod
    def _try_resolve_lollms_client() -> Optional[Any]:
        """Softly attempts to resolve an active LollmsClient without creating a hard dependency."""
        try:
            from lollms_client.lollms_config_cli_env import get_client_from_env
            client = get_client_from_env()
            if client and getattr(client, "llm", None):
                ASCIIColors.info("GraphStore: Connected to active LOLLMS Client from environment.")
                return client
        except (ImportError, Exception):
            pass
        return None

    def has_structured_lollms_support(self) -> bool:
        """Returns True if an active LOLLMS client with structured generation is connected."""
        return self.lollms_client is not None and hasattr(self.lollms_client, "generate_structured_content")

    @property
    def memory(self) -> CognitiveMemoryStore:
        if self._cognitive_memory is None:
            from .cognitive_memory import CognitiveMemoryStore
            self._cognitive_memory = CognitiveMemoryStore(self)
        return self._cognitive_memory

    @property
    def conn(self) -> sqlite3.Connection:
        self.store._ensure_connection()
        assert self.store.conn is not None, "SafeStore connection is not available."
        return self.store.conn

    @property
    def encryptor(self):
        return self.store.encryptor

    @property
    def embedder(self) -> BaseVectorizer:
        """Directly uses the vectorizer from the parent SafeStore instance."""
        self.store._ensure_connection()
        if not hasattr(self.store, 'vectorizer') or self.store.vectorizer is None:
            raise ConfigurationError("The parent SafeStore has not been initialized with a vectorizer.")
        return self.store.vectorizer

    @property
    def sparql_engine(self) -> SparqlEngine:
        if self._sparql_engine is None or self._sparql_engine.conn != self.conn:
            self._sparql_engine = SparqlEngine(self.conn)
        return self._sparql_engine

    def _initialize_graph_features(self) -> None:
        with self.store._instance_lock, self.store._optional_file_lock_context("Graph feature initialization"):
            try:
                db.initialize_schema(self.conn)
                self.conn.execute("BEGIN")
                if db.get_store_metadata(self.conn, self.GRAPH_FEATURES_ENABLED_KEY) != "true":
                    db.set_store_metadata(self.conn, self.GRAPH_FEATURES_ENABLED_KEY, "true")

                embedder_instance = self.embedder
                if embedder_instance.dim is None:
                    vectorizer_name_for_error = self.store.vectorizer_name if hasattr(self.store, 'vectorizer_name') else 'unknown'
                    raise ConfigurationError(f"GraphStore embedder '{vectorizer_name_for_error}' has an unknown dimension.")
                db.enable_vector_search_on_graph_nodes(self.conn, embedder_instance.dim)
                self.conn.commit()
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError("Failed to initialize GraphStore features.") from e

    def _format_ontology_for_prompt(self) -> str:
        """Formats the ontology for the LLM prompt."""
        if isinstance(self.ontology, str) and self.ontology.strip():
            return self.ontology.strip()

        if isinstance(self.ontology, dict):
            lines = []
            nodes = self.ontology.get("nodes")
            if isinstance(nodes, dict) and nodes:
                lines.append("NODE LABELS and PROPERTIES:")
                for label, details in nodes.items():
                    details = details or {}
                    desc = details.get("description", "")
                    lines.append(f"  - {label}: {desc}")
                    properties = details.get("properties")
                    if isinstance(properties, dict):
                        for prop, prop_desc in properties.items():
                            lines.append(f"    - {prop}: {prop_desc}")

            relationships = self.ontology.get("relationships")
            if isinstance(relationships, dict) and relationships:
                if lines: lines.append("")
                lines.append("RELATIONSHIP TYPES and CONSTRAINTS:")
                for rel_type, details in relationships.items():
                    details = details or {}
                    desc = details.get("description", "")
                    source = details.get("source", "Any")
                    target = details.get("target", "Any")
                    lines.append(f"  - {rel_type} (Source: {source}, Target: {target}): {desc}")

            if lines:
                return "\n".join(lines)

        return "No specific ontology provided. Extract entities and relationships based on the text context."

    def _get_graph_extraction_prompt(self, chunk_text: str, guidance: Optional[str] = None) -> str:
        user_guidance = guidance if guidance and guidance.strip() else "Extract all relevant properties you can identify."
        has_valid_ontology = isinstance(self.ontology, (dict, str)) and self.ontology

        if has_valid_ontology:
            template = self.DEFAULT_GRAPH_EXTRACTION_WITH_ONTOLOGY_PROMPT_TEMPLATE
            ontology_schema = self._format_ontology_for_prompt()
            return template.format(
                chunk_text=chunk_text,
                user_guidance=("" if not ontology_schema else "Ontology:\n"+ontology_schema+"\nGuidance:\n") + user_guidance
            )
        else:
            template = getattr(self, "graph_extraction_prompt_template", None) or self.DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE
            return template.format(chunk_text=chunk_text, user_guidance=user_guidance)

    def _get_query_parsing_prompt(self, natural_language_query: str) -> str:
        template = getattr(self, "query_parsing_prompt_template", None) or self.DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE
        return template.format(natural_language_query=natural_language_query)

    def _get_entity_fusion_prompt(self, node_a_props: Dict, node_b_props: Dict, label: str) -> str:
        template = getattr(self, "entity_fusion_prompt_template", None) or self.DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE
        return template.format(
            node_a_properties=json.dumps(node_a_props, indent=2),
            node_b_properties=json.dumps(node_b_props, indent=2),
            entity_label=label
        )

    def generate_sparql(self, natural_language_query: str, guidance: Optional[str] = None) -> str:
        """
        Translates a natural language question into an executable W3C SPARQL 1.1 query
        using LOLLMS, grounded in the database's live entity classes and relationship types.
        """
        # 1. Gather live graph topology for grounding
        info = self.get_graph_info()
        schema_lines = []
        if info.get("nodes_by_label"):
            schema_lines.append("Existing Entity Classes (Node Labels):")
            for lbl, count in info["nodes_by_label"].items():
                schema_lines.append(f"  - ont:{lbl} ({count} instances)")

        if info.get("relationships_by_type"):
            schema_lines.append("\nExisting Relationship Types (Object Properties):")
            for rtype, count in info["relationships_by_type"].items():
                schema_lines.append(f"  - ex:{rtype} / ont:{rtype} ({count} connections)")

        if info.get("ontology") and isinstance(info["ontology"], dict):
            schema_lines.append(f"\nOntology Summary: {json.dumps(info['ontology'])}")

        schema_context = "\n".join(schema_lines) if schema_lines else "General knowledge graph with nodes and directed relationships."

        template = getattr(self, "sparql_generation_prompt_template", None) or self.DEFAULT_SPARQL_GENERATION_PROMPT_TEMPLATE
        prompt = template.format(
            schema_context=schema_context + (f"\nAdditional Guidance: {guidance}" if guidance else ""),
            natural_language_query=natural_language_query.strip()
        )

        raw_response = ""
        # Priority 1: LOLLMS client
        if self.lollms_client is not None:
            try:
                if hasattr(self.lollms_client, "generate_code"):
                    raw_response = self.lollms_client.generate_code(prompt, language="sparql", temperature=0.1)
                elif hasattr(self.lollms_client, "generate_text"):
                    raw_response = self.lollms_client.generate_text(prompt, max_new_tokens=512, temperature=0.1)
            except Exception as e:
                ASCIIColors.warning(f"LOLLMS client generate_sparql fallback: {e}")

        # Priority 2: LLM Executor callback
        if not raw_response and self.llm_executor:
            raw_response = self.llm_executor(prompt)

        if not raw_response:
            raise SafeStoreError("Failed to generate SPARQL query: LLM returned empty response.")

        # Extract code block if wrapped in markdown
        cleaned = raw_response.strip()
        code_match = re.search(r'```(?:sparql|sql)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
        if code_match:
            cleaned = code_match.group(1).strip()

        # Ensure minimal prefixes exist
        if "PREFIX" not in cleaned.upper():
            prefixes = (
                "PREFIX ex: <http://example.org/>\n"
                "PREFIX ont: <http://example.org/ontology/>\n"
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            )
            cleaned = prefixes + cleaned

        return cleaned

    def get_structured_extraction_schema(self) -> Dict[str, Any]:
        """
        Builds a W3C-compliant JSON Schema for LOLLMS structured generation,
        dynamically constrained by the ontology when provided.
        """
        node_label_schema: Dict[str, Any] = {
            "type": "string",
            "description": "The category or class of the entity (e.g. Person, Concept, Tool, Organization)"
        }
        rel_type_schema: Dict[str, Any] = {
            "type": "string",
            "description": "Uppercase relationship type (e.g. USES, PART_OF, CREATED_BY, RELATES_TO)"
        }

        # Apply enum constraints if an ontology dictionary is present
        if isinstance(self.ontology, dict):
            if "nodes" in self.ontology and self.ontology["nodes"]:
                node_label_schema["enum"] = list(self.ontology["nodes"].keys())
            if "relationships" in self.ontology and self.ontology["relationships"]:
                rel_type_schema["enum"] = list(self.ontology["relationships"].keys())

        return {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of key entities, tools, concepts, organizations, or individuals.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": node_label_schema,
                            "properties": {
                                "type": "object",
                                "description": "Key-value attributes. Must include an identifying_value.",
                                "properties": {
                                    "identifying_value": {
                                        "type": "string",
                                        "description": "Canonical name or unique identifier of the entity"
                                    },
                                    "name": {"type": "string"},
                                    "description": {"type": "string"}
                                },
                                "required": ["identifying_value"]
                            }
                        },
                        "required": ["label", "properties"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "description": "Directed connections between extracted nodes.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_node_label": {"type": "string"},
                            "source_node_identifying_value": {"type": "string"},
                            "target_node_label": {"type": "string"},
                            "target_node_identifying_value": {"type": "string"},
                            "type": rel_type_schema,
                            "properties": {
                                "type": "object",
                                "description": "Optional attributes describing the relationship."
                            }
                        },
                        "required": [
                            "source_node_label",
                            "source_node_identifying_value",
                            "target_node_label",
                            "target_node_identifying_value",
                            "type"
                        ]
                    }
                }
            },
            "required": ["nodes", "relationships"]
        }

    def _sanitize_chunk_for_llm(self, text: Union[str, bytes]) -> str:
        """
        Sanitizes text chunk before sending to LLM:
        - Ensures pure string decoding.
        - Replaces heavy inline base64 image blobs with readable [Image: caption] tags,
          preventing downstream JSON serialization crashes on raw bytes and saving token quota.
        """
        if isinstance(text, bytes):
            clean = text.decode('utf-8', errors='ignore')
        else:
            clean = str(text)

        # Replace markdown inline base64 images: ![alt](data:image/...;base64,...)
        clean = re.sub(r'!\[([^\]]*)\]\(data:image\/[^;]+;base64,[A-Za-z0-9+/=\s]+\)', r'[Image: \1]', clean)
        return clean

    def _extract_and_insert_graph(
        self,
        text: str,
        chunk_ids: List[int],
        guidance: Optional[str] = None,
        source_label: str = "text"
    ) -> Tuple[int, int]:
        """
        Extracts graph elements from text (a single chunk, a batch of chunks, or a whole document)
        and links all extracted nodes to the provided chunk IDs for evidence provenance.
        """
        parsed = None
        sanitized_text = self._sanitize_chunk_for_llm(text)
        if not sanitized_text.strip():
            return 0, 0

        # Priority 1: LOLLMS Structured Content Generation
        if self.lollms_client is not None and hasattr(self.lollms_client, "generate_structured_content"):
            try:
                schema = self.get_structured_extraction_schema()
                ontology_context = self._format_ontology_for_prompt()
                system_prompt = (
                    "You are an expert knowledge graph extraction engine. "
                    "Extract all entities (nodes), their attributes, and directed relationships (triplets) "
                    "from the document content, strictly conforming to the schema.\n\n"
                    f"Ontology Schema / Constraints:\n{ontology_context}\n\n"
                    f"Document Content:\n{sanitized_text}"
                )
                user_prompt = "Create a complete, rich knowledge graph representation of the text as structured triplets."
                if guidance:
                    user_prompt += f" Guidance: {guidance}"

                structured = self.lollms_client.generate_structured_content(
                    system_prompt=system_prompt,
                    prompt=user_prompt,
                    schema=schema
                )

                if isinstance(structured, dict):
                    parsed = structured
                elif isinstance(structured, str):
                    parsed = robust_json_parser(structured)
            except Exception as e:
                ASCIIColors.warning(f"LOLLMS structured generation fallback for {source_label}: {e}")
                parsed = None

        # Priority 2: Fallback to llm_executor callback
        if parsed is None and self.llm_executor:
            prompt = self._get_graph_extraction_prompt(sanitized_text, guidance)
            raw_response = self.llm_executor(prompt)
            if not raw_response:
                ASCIIColors.warning(f"LLM extraction returned empty response for {source_label}.")
                return 0, 0

            try:
                parsed = robust_json_parser(raw_response)
            except Exception as e:
                ASCIIColors.warning(f"Failed to parse LLM extraction response for {source_label}: {e}\nRaw Response preview: {str(raw_response)[:200]}")
                return 0, 0

        nodes_data = parsed.get("nodes", []) if isinstance(parsed, dict) else []
        rels_data = parsed.get("relationships", []) if isinstance(parsed, dict) else []

        node_map: Dict[Tuple[str, str], int] = {}
        nodes_created = 0
        rels_created = 0

        for n in nodes_data:
            if not isinstance(n, dict) or "label" not in n or "properties" not in n:
                continue
            label = str(n["label"])
            props = n["properties"]
            if not isinstance(props, dict):
                continue

            node_id = self._fuse_or_create_node(label, props)
            self._vectorize_and_store_node_update(node_id, label, props)

            # Link node to all corresponding provenance chunk IDs
            for cid in chunk_ids:
                db.link_node_to_chunk(self.conn, node_id, cid)

            id_key, id_val = self._get_node_identifying_parts(props)
            if id_val:
                node_map[(label.lower(), str(id_val).strip().lower())] = node_id
                node_map[(label, str(id_val))] = node_id
            nodes_created += 1

        for r in rels_data:
            if not isinstance(r, dict):
                continue
            src_label = str(r.get("source_node_label", ""))
            src_val = str(r.get("source_node_identifying_value", ""))
            tgt_label = str(r.get("target_node_label", ""))
            tgt_val = str(r.get("target_node_identifying_value", ""))
            rel_type = str(r.get("type", ""))

            src_id = node_map.get((src_label.lower(), src_val.strip().lower())) or node_map.get((src_label, src_val))
            if not src_id and src_label and src_val:
                src_id = db.get_graph_node_by_signature(self.conn, f"{src_label}:{src_val.strip().lower()}")

            tgt_id = node_map.get((tgt_label.lower(), tgt_val.strip().lower())) or node_map.get((tgt_label, tgt_val))
            if not tgt_id and tgt_label and tgt_val:
                tgt_id = db.get_graph_node_by_signature(self.conn, f"{tgt_label}:{tgt_val.strip().lower()}")

            if src_id and tgt_id and rel_type:
                props = r.get("properties", {})
                props_json = json.dumps(props if isinstance(props, dict) else {})
                db.add_graph_relationship(self.conn, src_id, tgt_id, rel_type, props_json)
                rels_created += 1

        if nodes_created > 0 or rels_created > 0:
            ASCIIColors.info(f"{source_label}: Extracted {nodes_created} node(s) and {rels_created} relationship(s).")
        return nodes_created, rels_created

    def _extract_and_insert_graph_for_chunk(self, chunk_id: int, chunk_text: str, guidance: Optional[str] = None) -> Tuple[int, int]:
        """Legacy helper for single-chunk extraction."""
        return self._extract_and_insert_graph(
            text=chunk_text,
            chunk_ids=[chunk_id],
            guidance=guidance,
            source_label=f"Chunk #{chunk_id}"
        )

    def build_graph_for_document(
        self,
        doc_id: int,
        guidance: Optional[str] = None,
        mode: Literal['document', 'batch_chunks', 'chunk'] = 'document',
        chunks_per_batch: int = 5,
        progress_callback: Optional[ProgressCallback] = None
    ) -> Dict[str, int]:
        """
        Builds graph nodes and relationships for a specific document.

        Modes:
        - 'document' (Fastest): Sends the entire reconstructed document text to the LLM at once.
          Takes 1 LLM call instead of N calls, maximizing throughput and capturing cross-chunk relationships.
        - 'batch_chunks' (Balanced): Groups chunks into slices of `chunks_per_batch`.
        - 'chunk' (Granular): Processes each chunk in isolation.
        """
        with self.store._instance_lock, self.store._optional_file_lock_context(f"build_graph_for_document: {doc_id}"):
            cursor = self.conn.execute("SELECT chunk_id, chunk_text, is_encrypted, chunk_seq FROM chunks WHERE doc_id = ? ORDER BY chunk_seq ASC", (doc_id,))
            rows = cursor.fetchall()
            if not rows:
                return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0}

            doc_row = db.get_document_record_by_id(self.conn, doc_id)
            doc_name = Path(doc_row[1].decode('utf-8')).name if doc_row else f"Document #{doc_id}"

            chunk_records = []
            for cid, c_data, is_enc, seq in rows:
                if is_enc:
                    if self.encryptor.is_enabled:
                        try:
                            txt = self.encryptor.decrypt(c_data)
                        except Exception:
                            continue
                    else:
                        continue
                else:
                    txt = c_data.decode('utf-8') if isinstance(c_data, bytes) else str(c_data)
                chunk_records.append((cid, txt, seq))

            if not chunk_records:
                return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0}

            total_nodes = 0
            total_rels = 0
            processed_chunk_ids = [c[0] for c in chunk_records]

            if mode == 'document':
                # Reconstruct full text or assemble sequentially
                full_text = self.store.reconstruct_document_text(doc_id)
                if not full_text:
                    full_text = "\n\n".join(c[1] for c in chunk_records)

                status = f"Extracting graph for full document '{doc_name}' ({len(chunk_records)} chunks in 1 pass)..."
                ASCIIColors.info(status)
                if progress_callback:
                    progress_callback(0.5, status)

                n_cnt, r_cnt = self._extract_and_insert_graph(
                    text=full_text,
                    chunk_ids=processed_chunk_ids,
                    guidance=guidance,
                    source_label=f"Document '{doc_name}'"
                )
                total_nodes += n_cnt
                total_rels += r_cnt

            elif mode == 'batch_chunks':
                # Group chunks into slices
                step = max(1, chunks_per_batch)
                total_batches = (len(chunk_records) + step - 1) // step
                for b_idx in range(0, len(chunk_records), step):
                    slice_records = chunk_records[b_idx:b_idx + step]
                    slice_text = "\n\n".join(f"[Section {c[2]}]:\n{c[1]}" for c in slice_records)
                    slice_cids = [c[0] for c in slice_records]

                    cur_batch_num = (b_idx // step) + 1
                    status = f"Extracting batch {cur_batch_num}/{total_batches} for '{doc_name}'..."
                    if progress_callback:
                        progress_callback(cur_batch_num / total_batches, status)

                    n_cnt, r_cnt = self._extract_and_insert_graph(
                        text=slice_text,
                        chunk_ids=slice_cids,
                        guidance=guidance,
                        source_label=f"Batch {cur_batch_num}/{total_batches} of '{doc_name}'"
                    )
                    total_nodes += n_cnt
                    total_rels += r_cnt

            else:  # 'chunk' mode
                for idx, (cid, txt, seq) in enumerate(chunk_records):
                    status = f"Extracting chunk {idx+1}/{len(chunk_records)} of '{doc_name}'..."
                    if progress_callback:
                        progress_callback((idx + 1) / len(chunk_records), status)

                    n_cnt, r_cnt = self._extract_and_insert_graph(
                        text=txt,
                        chunk_ids=[cid],
                        guidance=guidance,
                        source_label=f"Chunk #{cid} of '{doc_name}'"
                    )
                    total_nodes += n_cnt
                    total_rels += r_cnt

            if processed_chunk_ids:
                db.mark_chunks_graph_processed(self.conn, processed_chunk_ids)

            return {
                "nodes_created": total_nodes,
                "relationships_created": total_rels,
                "chunks_processed": len(processed_chunk_ids)
            }

    def build_graph_for_all_documents(
        self,
        guidance: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        mode: Literal['document', 'batch_chunks', 'chunk'] = 'document',
        chunks_per_batch: int = 5
    ) -> Dict[str, int]:
        """
        Builds knowledge graph across all documents in the store.

        Supports fast large-context extraction modes:
        - 'document' (Default / Recommended): Sends each document as a complete narrative unit in 1 call.
          Transforms a 100-chunk store from 100 calls down to 3–5 calls, executing up to 20x faster.
        - 'batch_chunks': Batches consecutive chunks into slices of `chunks_per_batch`.
        - 'chunk': Classic 1-chunk-per-call mode.
        """
        with self.store._instance_lock, self.store._optional_file_lock_context("build_graph_for_all_documents"):
            # Discover distinct documents with unprocessed chunks
            cursor = self.conn.execute("SELECT DISTINCT doc_id FROM chunks WHERE graph_processed_at IS NULL ORDER BY doc_id ASC")
            doc_ids = [row[0] for row in cursor.fetchall()]
            if not doc_ids:
                cursor = self.conn.execute("SELECT DISTINCT doc_id FROM chunks ORDER BY doc_id ASC")
                doc_ids = [row[0] for row in cursor.fetchall()]

            total_docs = len(doc_ids)
            if total_docs == 0:
                ASCIIColors.warning("No documents available to build graph from.")
                return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0}

            ASCIIColors.info(f"Building knowledge graph across {total_docs} document(s) [Mode: {mode.upper()}]...")

            total_nodes = 0
            total_rels = 0
            total_chunks_processed = 0

            for d_idx, doc_id in enumerate(doc_ids, 1):
                doc_row = db.get_document_record_by_id(self.conn, doc_id)
                doc_name = Path(doc_row[1].decode('utf-8')).name if doc_row else f"Doc #{doc_id}"

                def _sub_progress(fraction, message):
                    overall_progress = ((d_idx - 1) + fraction) / total_docs
                    if progress_callback:
                        progress_callback(overall_progress, f"[{d_idx}/{total_docs}] {message}")

                stats = self.build_graph_for_document(
                    doc_id=doc_id,
                    guidance=guidance,
                    mode=mode,
                    chunks_per_batch=chunks_per_batch,
                    progress_callback=_sub_progress
                )

                total_nodes += stats["nodes_created"]
                total_rels += stats["relationships_created"]
                total_chunks_processed += stats["chunks_processed"]

                status_msg = f"Completed doc {d_idx}/{total_docs} ({doc_name}): +{stats['nodes_created']} nodes, +{stats['relationships_created']} edges (Total: {total_nodes} nodes, {total_rels} edges)"
                ASCIIColors.info(status_msg)
                if progress_callback:
                    progress_callback(d_idx / total_docs, status_msg)

            ASCIIColors.success(f"Graph build finished: {total_nodes} nodes, {total_rels} relationships across {total_chunks_processed} chunks.")
            return {
                "nodes_created": total_nodes,
                "relationships_created": total_rels,
                "chunks_processed": total_chunks_processed
            }

    def _fuse_or_create_node(self, label: str, properties: Dict[str, Any]) -> int:
        id_key, id_value = self._get_node_identifying_parts(properties)
        if id_key and id_value:
            sig = f"{label}:{id_key}:{id_value.strip().lower()}"
            if node_id := db.get_graph_node_by_signature(self.conn, sig):
                db.update_graph_node_properties_db(self.conn, node_id, properties, merge_strategy="overwrite_all")
                return node_id

        temp_text_to_embed = f"An entity of type {label} with properties {json.dumps(properties)}."
        query_vector = self.embedder.vectorize([temp_text_to_embed])[0]
        candidate_ids = db.search_graph_nodes_by_vector(self.conn, query_vector, top_k=3)

        for candidate_id in candidate_ids:
            candidate_details = db.get_node_details_db(self.conn, candidate_id)
            if not candidate_details or candidate_details['label'] != label: continue
            try:
                prompt = self._get_entity_fusion_prompt(candidate_details['properties'], properties, label)
                raw_response = self.llm_executor(prompt)
                decision = robust_json_parser(raw_response)
                if decision.get("is_same") is True:
                    existing_props = candidate_details['properties']
                    other_identifiers = existing_props.get("other_identifiers", [])
                    new_id_key, new_id_value = self._get_node_identifying_parts(properties)

                    if new_id_value and new_id_value not in other_identifiers:
                        other_identifiers.append(new_id_value)

                    merged_props = {**existing_props, **properties}
                    merged_props["other_identifiers"] = sorted(list(set(other_identifiers)))

                    db.update_graph_node_properties_db(self.conn, candidate_id, merged_props, merge_strategy="overwrite_all")
                    return candidate_id
            except Exception:
                pass

        if "other_identifiers" not in properties:
            properties["other_identifiers"] = []

        sig = f"{label}:{id_key}:{id_value.strip().lower()}" if id_key and id_value else f"unidentified:{label}:{uuid.uuid4()}"
        new_node_id = db.add_or_update_graph_node(self.conn, label, properties, sig)
        return new_node_id

    def _get_node_identifying_parts(self, properties: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(properties, dict): return None, None
        if "identifying_value" in properties and properties["identifying_value"]:
            return "identifying_value", str(properties["identifying_value"])
        for key in ["name", "title", "id", "identifier"]:
            if key in properties and properties[key]:
                return key, str(properties[key])
        for key, value in sorted(properties.items()):
            if isinstance(value, (str, int, float)) and value:
                return key, str(value)
        return None, None

    def _vectorize_and_store_node_update(self, node_id: int, label: str, properties: Dict[str, Any]):
        try:
            prop_strings = [f"{key} is {value}" for key, value in properties.items() if isinstance(value, (str, int, float))]
            text_to_embed = f"An entity of type {label} where {' and '.join(prop_strings)}." if prop_strings else f"An entity of type {label}."
            vector = self.embedder.vectorize([text_to_embed])[0]
            db.update_node_vector(self.conn, node_id, vector)
        except Exception as e:
            ASCIIColors.warning(f"Could not generate or store vector for node {node_id}: {e}")

    def query_sparql(self, sparql_query: str) -> Dict[str, Any]:
        """
        Executes a W3C SPARQL 1.1 query (SELECT, ASK, CONSTRUCT, DESCRIBE) against the graph database.
        """
        with self.store._instance_lock, self.store._optional_file_lock_context(f"query_sparql: {sparql_query[:30]}"):
            return self.sparql_engine.execute_query(sparql_query)

    def execute_sparql_update(self, sparql_update: str) -> Dict[str, Any]:
        """
        Executes a W3C SPARQL 1.1 UPDATE command (INSERT DATA, DELETE DATA, DELETE WHERE)
        and synchronizes changes with SQLite graph tables.
        """
        with self.store._instance_lock, self.store._optional_file_lock_context("execute_sparql_update"):
            return self.sparql_engine.execute_update(sparql_update)

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns standard function-calling tool schemas for LLM memory manipulation."""
        return self.memory.get_llm_tool_definitions()

    def dispatch_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Executes a function-calling tool requested by the LLM."""
        return self.memory.dispatch_llm_tool(tool_name, arguments)

    def query_graph(self, natural_language_query: str, output_mode: str = "chunks_summary", top_k_nodes: int = 5) -> Any:
        with self.store._instance_lock, self.store._optional_file_lock_context(f"query_graph: {natural_language_query[:30]}"):
            if output_mode not in ["chunks_summary", "graph_only", "full"]: raise ValueError("Invalid output_mode.")

            query_vector = self.embedder.vectorize([natural_language_query])[0]
            seed_node_ids = db.search_graph_nodes_by_vector(self.conn, query_vector, top_k_nodes)
            if not seed_node_ids:
                return self._empty_query_result(output_mode)

            parsed_guidance = {}
            try:
                raw_llm_response = self.llm_executor(self._get_query_parsing_prompt(natural_language_query))
                parsed_guidance = robust_json_parser(raw_llm_response)
            except Exception:
                pass

            max_depth = parsed_guidance.get("max_depth", 2)
            target_rels = parsed_guidance.get("target_relationships") or [{"type": None, "direction": "any"}]
            target_labels = parsed_guidance.get("target_node_labels") or []

            subgraph_nodes: Dict[int, Dict[str, Any]] = {}
            subgraph_rels: Dict[int, Dict[str, Any]] = {}
            queue: List[Tuple[int, int]] = [(seed_id, 0) for seed_id in seed_node_ids]
            visited: Set[int] = set(seed_node_ids)

            for seed_id in seed_node_ids:
                if details := db.get_node_details_db(self.conn, seed_id): subgraph_nodes[seed_id] = details

            head = 0
            while head < len(queue):
                current_node_id, current_depth = queue[head]; head += 1
                if current_depth >= max_depth: continue

                for rel_spec in target_rels:
                    for rel in db.get_relationships_for_node_db(self.conn, current_node_id, rel_spec.get("type"), rel_spec.get("direction", "any"), limit=100):
                        subgraph_rels[rel["relationship_id"]] = rel
                        neighbor_info = rel.get("target_node") if rel["source_node_id"] == current_node_id else rel.get("source_node")
                        if neighbor_info:
                            neighbor_id, neighbor_label = neighbor_info["node_id"], neighbor_info["label"]
                            if target_labels and neighbor_label not in target_labels: continue
                            if neighbor_id not in subgraph_nodes: subgraph_nodes[neighbor_id] = neighbor_info
                            if neighbor_id not in visited:
                                queue.append((neighbor_id, current_depth + 1))
                                visited.add(neighbor_id)

            final_graph_data = {"nodes": list(subgraph_nodes.values()), "relationships": list(subgraph_rels.values())}
            return self._format_query_output(final_graph_data, output_mode)

    def query_graph_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
        dense_weight: float = 0.4,
        bm25_weight: float = 0.3,
        graph_weight: float = 0.3,
        rrf_k: int = 60,
        min_relevance_percent: float = 0.0
    ) -> Dict[str, Any]:
        """
        Unified Tri-Modal Retrieval combining Graph Traversal (SPARQL/Neighborhood),
        Dense Vector Similarity, and Sparse BM25 Lexical search via Reciprocal Rank Fusion
        with standardized 0-100 relevance grades and threshold filtering.
        """
        with self.store._instance_lock, self.store._optional_file_lock_context(f"query_graph_hybrid: {query_text[:30]}"):
            graph_result = self.query_graph(query_text, output_mode="full", top_k_nodes=top_k)
            graph_chunks = graph_result.get("chunks", []) if isinstance(graph_result, dict) else []

            dense_chunks = self.store.query(query_text, top_k=top_k * 2)

            from ..search.bm25 import BM25Retriever
            from ..search.fusion import reciprocal_rank_fusion
            bm25_retriever = BM25Retriever(self.conn)
            bm25_chunks = bm25_retriever.search(query_text, top_k=top_k * 2)

            fused_chunks = reciprocal_rank_fusion(
                ranked_lists=[dense_chunks, bm25_chunks, graph_chunks],
                weights=[dense_weight, bm25_weight, graph_weight],
                k=rrf_k,
                top_k=top_k,
                min_relevance_percent=min_relevance_percent
            )

            return {
                "query": query_text,
                "ranked_chunks": fused_chunks,
                "subgraph": graph_result.get("graph", {"nodes": [], "relationships": []}) if isinstance(graph_result, dict) else {}
            }

    def _empty_query_result(self, output_mode: str) -> Any:
        if output_mode == "chunks_summary": return []
        if output_mode == "graph_only": return {"nodes": [], "relationships": []}
        if output_mode == "full": return {"graph": {"nodes": [], "relationships": []}, "chunks": []}
        return None

    def _format_query_output(self, graph_data: Dict[str, Any], output_mode: str) -> Any:
        if output_mode in ["chunks_summary", "full"] and graph_data.get("nodes"):
            node_ids = [n["node_id"] for n in graph_data["nodes"]]
            node_to_chunks = db.get_chunk_ids_for_nodes_db(self.conn, node_ids)
            all_chunk_ids = {cid for ids in node_to_chunks.values() for cid in ids}

            chunk_details = db.get_chunk_details_db(self.conn, list(all_chunk_ids), self.encryptor) if all_chunk_ids else []
            for chunk in chunk_details:
                chunk["linked_graph_nodes"] = [
                    {"node_id": n_id, "label": next((n['label'] for n in graph_data['nodes'] if n['node_id'] == n_id), 'Unknown')}
                    for n_id, c_ids in node_to_chunks.items() if chunk["chunk_id"] in c_ids
                ]
            if output_mode == "chunks_summary": return chunk_details
            if output_mode == "full": return {"graph": graph_data, "chunks": chunk_details}

        if output_mode == "graph_only": return graph_data
        if output_mode == "full": return {"graph": graph_data, "chunks": []}
        return []

    def add_node(self, label: str, properties: Dict[str, Any]) -> int:
        with self.store._instance_lock, self.store._optional_file_lock_context("add_node"):
            if "other_identifiers" not in properties:
                properties["other_identifiers"] = []
            id_key, id_value = self._get_node_identifying_parts(properties)
            sig = f"{label}:{id_key}:{id_value.strip().lower()}" if id_key and id_value else f"manual:{label}:{uuid.uuid4()}"
            try:
                self.conn.execute("BEGIN")
                node_id = db.add_or_update_graph_node(self.conn, label, properties, sig)
                self._vectorize_and_store_node_update(node_id, label, properties)
                self.conn.commit()
                ASCIIColors.success(f"Node added successfully with ID: {node_id}")
                return node_id
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Error adding node: {e}") from e

    def get_node_details(self, node_id: int) -> Optional[Dict[str, Any]]:
        with self.store._instance_lock:
            return db.get_node_details_db(self.conn, node_id)

    def get_all_nodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns all graph nodes up to the specified limit."""
        with self.store._instance_lock:
            try:
                cursor = self.conn.execute(
                    "SELECT node_id, node_label, node_properties, unique_signature FROM graph_nodes LIMIT ?",
                    (limit,)
                )
                nodes = []
                for row in cursor.fetchall():
                    nodes.append({
                        "node_id": row[0],
                        "label": row[1],
                        "properties": json.loads(row[2]) if row[2] else {},
                        "unique_signature": row[3]
                    })
                return nodes
            except sqlite3.Error as e:
                raise GraphDBError(f"Error fetching all nodes: {e}") from e

    def get_all_nodes_for_visualization(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Alias for get_all_nodes to support graph visualization pipelines."""
        return self.get_all_nodes(limit=limit)

    def get_graph_info(self) -> Dict[str, Any]:
        """Returns diagnostic metadata about the graph store, node labels, and relationship counts."""
        with self.store._instance_lock:
            self.store._ensure_connection()
            cursor = self.conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM graph_nodes")
            total_nodes = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM graph_relationships")
            total_relationships = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM node_chunk_links")
            total_provenance_links = cursor.fetchone()[0] or 0

            cursor.execute("SELECT node_label, COUNT(*) FROM graph_nodes GROUP BY node_label")
            nodes_by_label = {row[0]: row[1] for row in cursor.fetchall()}

            cursor.execute("SELECT relationship_type, COUNT(*) FROM graph_relationships GROUP BY relationship_type")
            relationships_by_type = {row[0]: row[1] for row in cursor.fetchall()}

            ontology_info = None
            if self.ontology:
                if isinstance(self.ontology, dict):
                    ontology_info = {
                        "defined_node_types": list(self.ontology.get("nodes", {}).keys()),
                        "defined_relationship_types": list(self.ontology.get("relationships", {}).keys())
                    }
                else:
                    ontology_info = {"raw": str(self.ontology)[:300]}

            return {
                "graph_features_enabled": db.get_store_metadata(self.conn, self.GRAPH_FEATURES_ENABLED_KEY) == "true",
                "total_nodes": total_nodes,
                "total_relationships": total_relationships,
                "total_provenance_links": total_provenance_links,
                "nodes_by_label": nodes_by_label,
                "relationships_by_type": relationships_by_type,
                "ontology": ontology_info
            }

    def get_all_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns all graph relationships with source and target node metadata."""
        with self.store._instance_lock:
            try:
                sql = """
                SELECT r.relationship_id, r.source_node_id, r.target_node_id, r.relationship_type, r.relationship_properties,
                       s.node_label as source_label, s.node_properties as source_properties,
                       t.node_label as target_label, t.node_properties as target_properties
                FROM graph_relationships r
                JOIN graph_nodes s ON r.source_node_id = s.node_id
                JOIN graph_nodes t ON r.target_node_id = t.node_id
                LIMIT ?;
                """
                cursor = self.conn.execute(sql, (limit,))
                relationships = []
                for row in cursor.fetchall():
                    relationships.append({
                        "relationship_id": row[0],
                        "source_node_id": row[1],
                        "target_node_id": row[2],
                        "type": row[3],
                        "properties": json.loads(row[4]) if row[4] else {},
                        "source_node": {"node_id": row[1], "label": row[5], "properties": json.loads(row[6]) if row[6] else {}},
                        "target_node": {"node_id": row[2], "label": row[7], "properties": json.loads(row[8]) if row[8] else {}}
                    })
                return relationships
            except sqlite3.Error as e:
                raise GraphDBError(f"Error fetching all relationships: {e}") from e

    def update_node(self, node_id: int, label: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> bool:
        if label is None and properties is None: return True
        with self.store._instance_lock, self.store._optional_file_lock_context(f"update_node: {node_id}"):
            try:
                self.conn.execute("BEGIN")
                current = db.get_node_details_db(self.conn, node_id)
                if not current: raise NodeNotFoundError(f"Node {node_id} not found.")

                if label is not None and label != current["label"]:
                    db.update_graph_node_label_db(self.conn, node_id, label)

                if properties is not None:
                    if "other_identifiers" not in properties and "other_identifiers" in current["properties"]:
                        properties["other_identifiers"] = current["properties"]["other_identifiers"]
                    db.update_graph_node_properties_db(self.conn, node_id, properties, "overwrite_all")

                updated_label = label or current['label']
                updated_props = properties if properties is not None else current['properties']
                self._vectorize_and_store_node_update(node_id, updated_label, updated_props)
                self.conn.commit()
                return True
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Error updating node {node_id}: {e}") from e

    def delete_node(self, node_id: int) -> bool:
        with self.store._instance_lock, self.store._optional_file_lock_context(f"delete_node: {node_id}"):
            try:
                self.conn.execute("BEGIN")
                deleted_count = db.delete_graph_node_and_relationships_db(self.conn, node_id)
                if deleted_count == 0:
                    self.conn.rollback()
                    raise NodeNotFoundError(f"Node with ID {node_id} not found for deletion.")
                self.conn.commit()
                return True
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Error deleting node {node_id}: {e}") from e

    def add_relationship(self, source_node_id: int, target_node_id: int, rel_type: str, properties: Optional[Dict[str, Any]] = None) -> int:
        with self.store._instance_lock, self.store._optional_file_lock_context("add_relationship"):
            try:
                self.conn.execute("BEGIN")
                props_json = json.dumps(properties or {})
                rel_id = db.add_graph_relationship(self.conn, source_node_id, target_node_id, rel_type, props_json)
                self.conn.commit()
                return rel_id
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Error adding relationship: {e}") from e

    def delete_relationship(self, relationship_id: int) -> bool:
        with self.store._instance_lock, self.store._optional_file_lock_context(f"delete_relationship: {relationship_id}"):
            try:
                self.conn.execute("BEGIN")
                deleted_count = db.delete_graph_relationship_db(self.conn, relationship_id)
                if deleted_count == 0:
                    self.conn.rollback()
                    raise RelationshipNotFoundError(f"Relationship {relationship_id} not found for deletion.")
                self.conn.commit()
                return True
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Error deleting relationship {relationship_id}: {e}") from e

    def get_relationship(self, relationship_id: int) -> Optional[Dict[str, Any]]:
        with self.store._instance_lock:
            try:
                cursor = self.conn.execute(
                    "SELECT relationship_id, source_node_id, target_node_id, relationship_type, relationship_properties FROM graph_relationships WHERE relationship_id = ?",
                    (relationship_id,)
                )
                row = cursor.fetchone()
                if not row: return None
                rel_id, src, tgt, rel_type, props_json = row
                return {
                    "relationship_id": rel_id, "source_node_id": src, "target_node_id": tgt,
                    "type": rel_type, "properties": json.loads(props_json) if props_json else {}
                }
            except Exception as e:
                raise GraphDBError(f"Error fetching relationship {relationship_id}: {e}") from e

    def update_relationship(self, relationship_id: int, rel_type: Optional[str] = None, properties: Optional[Dict[str, Any]] = None) -> bool:
        if rel_type is None and properties is None: return True
        with self.store._instance_lock, self.store._optional_file_lock_context(f"update_relationship: {relationship_id}"):
            try:
                self.conn.execute("BEGIN")
                current = self.get_relationship(relationship_id)
                if not current: raise RelationshipNotFoundError(f"Relationship {relationship_id} not found.")

                new_type = rel_type if rel_type is not None else current["type"]
                new_props = properties if properties is not None else current["properties"]

                self.conn.execute(
                    "UPDATE graph_relationships SET relationship_type = ?, relationship_properties = ? WHERE relationship_id = ?",
                    (new_type, json.dumps(new_props), relationship_id)
                )
                self.conn.commit()
                return True
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Error updating relationship {relationship_id}: {e}") from e

    def get_nodes_by_label(self, label: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self.store._instance_lock:
            try:
                return db.get_nodes_by_label_db(self.conn, label, limit)
            except (sqlite3.Error, json.JSONDecodeError) as e: raise GraphDBError(f"DB error finding nodes by label '{label}': {e}") from e

    def find_neighbors(self, node_id: int, relationship_type: Optional[str] = None, direction: str = "any", limit: int = 50) -> List[Dict[str, Any]]:
        if direction not in ["outgoing", "incoming", "any"]: raise ValueError("Invalid direction.")
        with self.store._instance_lock:
            relationships = db.get_relationships_for_node_db(self.conn, node_id, relationship_type, direction, limit)
            neighbor_nodes: List[Dict[str, Any]] = []
            seen_ids: Set[int] = set()
            for rel in relationships:
                node_data: Optional[Dict[str, Any]] = None
                if direction == "any":
                    node_data = rel.get("target_node") if rel.get("source_node_id") == node_id else rel.get("source_node")
                elif direction == "outgoing":
                    node_data = rel.get("target_node")
                elif direction == "incoming":
                    node_data = rel.get("source_node")

                if node_data and node_data.get("node_id") not in seen_ids:
                    neighbor_nodes.append(node_data)
                    seen_ids.add(node_data["node_id"])
            return neighbor_nodes[:limit]