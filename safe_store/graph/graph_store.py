# [FINAL & COMPLETE] safe_store/graph/graph_store.py
from __future__ import annotations
import sqlite3
import threading
import json
import uuid
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any, Tuple, TYPE_CHECKING, Set

from ascii_colors import ASCIIColors
from ..core import db
from ..core.exceptions import (
    DatabaseError, ConfigurationError, GraphDBError, GraphProcessingError, LLMCallbackError,
    GraphError, QueryError, NodeNotFoundError, RelationshipNotFoundError
)
from ..utils.json_parsing import robust_json_parser
from ..vectorization.base import BaseVectorizer

if TYPE_CHECKING:
    from ..store import SafeStore

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

    This class provides functionalities to build a graph from text documents,
    perform vector-based semantic queries on the graph, and manually
    manage graph elements (nodes and relationships). It operates as an
    extension of a SafeStore instance, relying on it for database
    connection, concurrency control, and vectorization.
    """
    GRAPH_FEATURES_ENABLED_KEY = "graph_features_enabled"
    DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE = load_prompt("graph_extraction_prompt")
    DEFAULT_GRAPH_EXTRACTION_WITH_ONTOLOGY_PROMPT_TEMPLATE = load_prompt("graph_extraction_prompt_with_ontology")
    DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE = load_prompt("query_parsing_prompt")
    DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE = load_prompt("entity_fusion_prompt")

    def __init__(
        self,
        store: "SafeStore",
        llm_executor_callback: LLMExecutorCallback,
        ontology: Optional[Dict[str, Any]] = None,
        graph_extraction_prompt_template: Optional[str] = None,
        query_parsing_prompt_template: Optional[str] = None,
        entity_fusion_prompt_template: Optional[str] = None,
    ):
        self.store = store
        self.llm_executor = llm_executor_callback
        self.ontology = ontology
        self.graph_extraction_prompt_template = graph_extraction_prompt_template or self.DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE
        self.query_parsing_prompt_template = query_parsing_prompt_template or self.DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE
        self.entity_fusion_prompt_template = entity_fusion_prompt_template or self.DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE
        ASCIIColors.info(f"Initializing GraphStore with shared SafeStore for database: {self.store.db_path}")
        self._initialize_graph_features()

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
        """
        Formats the ontology for the LLM prompt. Handles structured dicts,
        plain text instructions, or empty/invalid inputs.
        """
        if isinstance(self.ontology, str) and self.ontology.strip():
            ASCIIColors.info("Using plain text ontology instructions for prompt.")
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
                ASCIIColors.info("Formatted structured ontology for prompt.")
                return "\n".join(lines)

        ASCIIColors.info("No valid ontology provided; using default instructions.")
        return "No specific ontology provided. Extract entities and relationships based on the text context."

    def _get_graph_extraction_prompt(self, chunk_text: str, guidance: Optional[str] = None) -> str:
        user_guidance = guidance if guidance and guidance.strip() else "Extract all relevant properties you can identify."
        
        has_valid_ontology = isinstance(self.ontology, (dict, str)) and self.ontology
        
        if has_valid_ontology:
            template = self.DEFAULT_GRAPH_EXTRACTION_WITH_ONTOLOGY_PROMPT_TEMPLATE
            ontology_schema = self._format_ontology_for_prompt()
            ASCIIColors.info("Using ontology-aware prompt for graph extraction.")
            return template.format(
                chunk_text=chunk_text,
                user_guidance=("" if not ontology_schema else "Ontology:\n"+ontology_schema+"\nGuidance:\n") + user_guidance
            )
        else:
            template = self.graph_extraction_prompt_template
            ASCIIColors.info("Using default prompt for graph extraction.")
            return template.format(
                chunk_text=chunk_text,
                user_guidance=user_guidance
            )

    def _get_query_parsing_prompt(self, natural_language_query: str) -> str:
        return self.query_parsing_prompt_template.format(natural_language_query=natural_language_query)
    
    def _get_entity_fusion_prompt(self, node_a_props: Dict, node_b_props: Dict, label: str) -> str:
        """Formats the prompt for the LLM to decide on entity fusion."""
        return self.entity_fusion_prompt_template.format(
            node_a_properties=json.dumps(node_a_props, indent=2),
            node_b_properties=json.dumps(node_b_props, indent=2),
            entity_label=label
        )

    def _fuse_or_create_node(self, label: str, properties: Dict[str, Any]) -> int:
        id_key, id_value = self._get_node_identifying_parts(properties)
        if id_key and id_value:
            sig = f"{label}:{id_key}:{id_value.strip().lower()}"
            if node_id := db.get_graph_node_by_signature(self.conn, sig):
                db.update_graph_node_properties_db(self.conn, node_id, properties, merge_strategy="overwrite_all")
                ASCIIColors.info(f"Fused node via exact signature '{sig}' into existing node {node_id}.")
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
                    ASCIIColors.info(f"LLM confirmed fusion: Merging new data into node {candidate_id}.")
                    
                    # Fusion logic with `other_identifiers`
                    existing_props = candidate_details['properties']
                    other_identifiers = existing_props.get("other_identifiers", [])
                    new_id_key, new_id_value = self._get_node_identifying_parts(properties)
                    
                    if new_id_value and new_id_value not in other_identifiers:
                        other_identifiers.append(new_id_value)
                    
                    # Merge properties, but ensure other_identifiers is handled correctly
                    merged_props = {**existing_props, **properties}
                    merged_props["other_identifiers"] = sorted(list(set(other_identifiers)))

                    db.update_graph_node_properties_db(self.conn, candidate_id, merged_props, merge_strategy="overwrite_all")
                    return candidate_id
            except (LLMCallbackError, json.JSONDecodeError, KeyError) as e:
                ASCIIColors.warning(f"Entity fusion LLM call failed for candidate {candidate_id}: {e}")

        # Ensure new nodes have the other_identifiers property
        if "other_identifiers" not in properties:
            properties["other_identifiers"] = []

        sig = f"{label}:{id_key}:{id_value.strip().lower()}" if id_key and id_value else f"unidentified:{label}:{uuid.uuid4()}"
        new_node_id = db.add_or_update_graph_node(self.conn, label, properties, sig)
        ASCIIColors.info(f"No match found. Created new node {new_node_id} with signature '{sig}'.")
        return new_node_id

    def _get_node_identifying_parts(self, properties: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(properties, dict): return None, None
        # Prioritize "identifying_value" if present, as per the prompt's structure
        if "identifying_value" in properties and properties["identifying_value"]:
            return "identifying_value", str(properties["identifying_value"])
        for key in ["name", "title", "id", "identifier"]:
            if key in properties and properties[key]:
                return key, str(properties[key])
        # Fallback to the first non-empty, sortable property
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

    def _get_llm_extraction_with_retry(
        self, chunk_id: int, chunk_text: str, guidance: Optional[str], max_retries: int
    ) -> Optional[Dict[str, Any]]:
        """
        Calls the LLM to extract graph data, with a retry mechanism for JSON parsing failures.
        Returns the parsed JSON dictionary on success, or None if all retries fail.
        """
        prompt = self._get_graph_extraction_prompt(chunk_text, guidance)
        for attempt in range(max_retries + 1):
            try:
                raw_response = self.llm_executor(prompt)
                if attempt > 0: # Log if we are in a retry attempt
                    ASCIIColors.cyan(f"--- LLM Raw Output for Chunk {chunk_id} (Attempt {attempt + 1}) ---")
                    ASCIIColors.yellow(raw_response)
                    ASCIIColors.cyan("----------------------------------------------------")
                
                parsed_json = robust_json_parser(raw_response)
                return parsed_json
            except Exception as e:
                ASCIIColors.warning(f"LLM output for chunk {chunk_id} failed parsing on attempt {attempt + 1}/{max_retries + 1}. Error: {e}")
                if attempt < max_retries:
                    # Add a corrective instruction to the prompt for the next attempt
                    prompt += (f"\n\n--- Previous Attempt Failed ---"
                               f"\nYour last response could not be parsed as valid JSON. "
                               f"Please review the instructions and provide a single, complete, and well-formed JSON object "
                               f"enclosed in a markdown code block. Raw previous response: {raw_response[:500]}...")
                else:
                    # This was the last attempt
                    ASCIIColors.error(f"Failed to get valid JSON from LLM for chunk {chunk_id} after {max_retries + 1} attempts.")
                    return None
        return None # Should be unreachable, but here for completeness

    def _process_chunk_for_graph_impl(self, chunk_id: int, guidance: Optional[str] = None, llm_retries: int = 1) -> None:
        chunk_details = db.get_chunk_details_db(self.conn, [chunk_id], self.encryptor)[0]
        if not chunk_details:
            raise GraphProcessingError(f"Chunk {chunk_id} not found.")
        decrypted_chunk_text = chunk_details['chunk_text']

        llm_output = self._get_llm_extraction_with_retry(chunk_id, decrypted_chunk_text, guidance, llm_retries)

        if not llm_output:
            ASCIIColors.error(f"Skipping graph processing for chunk {chunk_id} due to persistent LLM JSON parsing failures.")
            return

        if "nodes" not in llm_output or "relationships" not in llm_output:
            ASCIIColors.warning(f"LLM output for chunk {chunk_id} is structured incorrectly (missing keys). Skipping.")
            return

        node_map: Dict[Tuple[str, str], int] = {}
        for node_data in llm_output.get("nodes", []):
            if not (isinstance(node_data, dict) and node_data.get("label") and isinstance(node_data.get("properties"), dict)): continue
            label, props = str(node_data["label"]), node_data["properties"]
            node_id = self._fuse_or_create_node(label, props)
            self._vectorize_and_store_node_update(node_id, label, props)
            db.link_node_to_chunk(self.conn, node_id, chunk_id)
            
            _, id_value = self._get_node_identifying_parts(props)
            if id_value:
                node_map[(label, str(id_value))] = node_id

        for rel_data in llm_output.get("relationships", []):
            req_keys = ["source_node_label", "source_node_identifying_value", "target_node_label", "target_node_identifying_value", "type"]
            if not (isinstance(rel_data, dict) and all(k in rel_data for k in req_keys)): continue
            
            src_label, src_val = str(rel_data["source_node_label"]), str(rel_data["source_node_identifying_value"])
            tgt_label, tgt_val = str(rel_data["target_node_label"]), str(rel_data["target_node_identifying_value"])
            
            src_id = node_map.get((src_label, src_val))
            tgt_id = node_map.get((tgt_label, tgt_val))

            def find_node_id_db(label: str, value: str) -> Optional[int]:
                for key in ["identifying_value", "name", "title", "id", "identifier"]:
                    sig = f"{label}:{key}:{value.strip().lower()}"
                    if node_id := db.get_graph_node_by_signature(self.conn, sig): return node_id
                if similar_nodes := db.find_node_by_label_and_property_value(self.conn, label, value, limit=1):
                    return similar_nodes[0]['node_id']
                return None

            if src_id is None: src_id = find_node_id_db(src_label, src_val)
            if tgt_id is None: tgt_id = find_node_id_db(tgt_label, tgt_val)

            if src_id is None or tgt_id is None:
                ASCIIColors.warning(f"Skipping relationship '{rel_data['type']}'. Src: '{src_val}' (Found: {src_id is not None}), Tgt: '{tgt_val}' (Found: {tgt_id is not None})")
                continue
            
            try:
                props_json = json.dumps(rel_data.get("properties", {}))
                db.add_graph_relationship(self.conn, src_id, tgt_id, str(rel_data["type"]), props_json)
                ASCIIColors.info(f"Successfully created relationship: {src_label} -> {rel_data['type']} -> {tgt_label}")
            except (GraphDBError, json.JSONDecodeError) as e:
                ASCIIColors.error(f"Error storing relationship '{rel_data['type']}': {e}")
    
    def process_chunk_for_graph(self, chunk_id: int, llm_retries: int = 1) -> None:
        with self.store._instance_lock, self.store._optional_file_lock_context(f"process_chunk_for_graph: {chunk_id}"):
            try:
                self.conn.execute("BEGIN")
                self._process_chunk_for_graph_impl(chunk_id, llm_retries=llm_retries)
                db.mark_chunks_graph_processed(self.conn, [chunk_id])
                self.conn.commit()
                ASCIIColors.success(f"Successfully processed chunk {chunk_id} for graph.")
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                ASCIIColors.error(f"Error processing chunk {chunk_id} for graph: {e}")
                raise GraphProcessingError(f"Failed to process chunk {chunk_id}") from e

    def build_graph_for_document(
        self, doc_id: int, guidance: Optional[str] = None, progress_callback: Optional[ProgressCallback] = None, llm_retries: int = 1
    ) -> None:
        with self.store._instance_lock, self.store._optional_file_lock_context(f"build_graph_for_document: {doc_id}"):
            # AND graph_processed_at IS NULL
            chunk_ids = [row[0] for row in self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()]
            if not chunk_ids:
                ASCIIColors.info(f"No unprocessed chunks found for document {doc_id}.")
                if progress_callback: progress_callback(1.0, "No new chunks to process.")
                return

            if progress_callback: progress_callback(0.0, f"Starting to process {len(chunk_ids)} chunks.")
            try:
                self.conn.execute("BEGIN")
                for i, chunk_id in enumerate(chunk_ids):
                    self._process_chunk_for_graph_impl(chunk_id, guidance, llm_retries)
                    if progress_callback: progress_callback((i + 1) / len(chunk_ids), f"Processed chunk {i + 1}/{len(chunk_ids)}.")
                
                db.mark_chunks_graph_processed(self.conn, chunk_ids)
                self.conn.commit()
                if progress_callback: progress_callback(1.0, "Graph building complete.")
                ASCIIColors.success(f"Successfully built graph for document {doc_id}.")
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                if progress_callback: progress_callback(1.0, f"Error: {e}")
                ASCIIColors.error(f"Error building graph for document {doc_id}: {e}")
                raise GraphProcessingError(f"Failed to build graph for document {doc_id}") from e

    def build_graph_for_all_documents(
        self, batch_size_chunks: int = 20, progress_callback: Optional[ProgressCallback] = None, llm_retries: int = 1
    ) -> None:
        with self.store._instance_lock, self.store._optional_file_lock_context("build_graph_for_all_documents"):
            total_unprocessed = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE graph_processed_at IS NULL").fetchone()[0]
            if total_unprocessed == 0:
                ASCIIColors.success("Graph is already up-to-date with all chunks.")
                if progress_callback: progress_callback(1.0, "Graph is up-to-date.")
                return

            ASCIIColors.info(f"Found {total_unprocessed} total unprocessed chunks. Processing in batches of {batch_size_chunks}.")
            processed = 0
            if progress_callback: progress_callback(0, f"Starting build for {total_unprocessed} chunks.")
            while True:
                chunk_ids_batch = [r[0] for r in self.conn.execute("SELECT chunk_id FROM chunks WHERE graph_processed_at IS NULL LIMIT ?", (batch_size_chunks,)).fetchall()]
                if not chunk_ids_batch: break
                try:
                    self.conn.execute("BEGIN")
                    for chunk_id in chunk_ids_batch:
                        self._process_chunk_for_graph_impl(chunk_id, llm_retries=llm_retries)
                    db.mark_chunks_graph_processed(self.conn, chunk_ids_batch)
                    self.conn.commit()
                    processed += len(chunk_ids_batch)
                    ASCIIColors.info(f"Processed batch of {len(chunk_ids_batch)}. Total processed: {processed}/{total_unprocessed}")
                    if progress_callback: progress_callback(processed / total_unprocessed, f"Processed {processed}/{total_unprocessed} chunks.")
                except Exception as e:
                    if self.conn.in_transaction: self.conn.rollback()
                    raise GraphProcessingError("Failed during batch processing.") from e
            ASCIIColors.success("Finished building graph for all available documents.")

    def query_graph(self, natural_language_query: str, output_mode: str = "chunks_summary", top_k_nodes: int = 5) -> Any:
        with self.store._instance_lock, self.store._optional_file_lock_context(f"query_graph: {natural_language_query[:30]}"):
            if output_mode not in ["chunks_summary", "graph_only", "full"]: raise ValueError("Invalid output_mode.")

            query_vector = self.embedder.vectorize([natural_language_query])[0]
            seed_node_ids = db.search_graph_nodes_by_vector(self.conn, query_vector, top_k_nodes)
            if not seed_node_ids:
                ASCIIColors.warning("Vector search yielded no relevant seed nodes for the query.")
                return self._empty_query_result(output_mode)
            ASCIIColors.debug(f"Found seed node IDs via vector search: {seed_node_ids}")

            parsed_guidance = {}
            try:
                raw_llm_response = self.llm_executor(self._get_query_parsing_prompt(natural_language_query))
                parsed_guidance = robust_json_parser(raw_llm_response)
            except Exception as e: ASCIIColors.warning(f"Could not parse LLM guidance for query, using defaults. Error: {e}")

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
                ASCIIColors.success(f"Node {node_id} updated successfully.")
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
                ASCIIColors.success(f"Node {node_id} and all associated data deleted.")
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
                ASCIIColors.success(f"Relationship added successfully with ID: {rel_id}")
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
            
    def get_all_nodes_for_visualization(self, limit: int = 1000) -> List[Dict[str, Any]]:
        with self.store._instance_lock:
            try:
                cursor = self.conn.execute("SELECT node_id, node_label, node_properties FROM graph_nodes LIMIT ?", (limit,))
                nodes = []
                for row in cursor.fetchall():
                    node_id, label, props_json = row
                    properties = {}
                    try:
                        properties = json.loads(props_json)
                        title = json.dumps(properties, indent=2)
                    except json.JSONDecodeError:
                        properties = {"error": "Could not parse properties JSON.", "raw": props_json}
                        title = props_json
                    
                    nodes.append({
                        "id": node_id,
                        "label": label,
                        "title": title,
                        "properties": properties
                    })
                return nodes
            except sqlite3.Error as e:
                raise GraphDBError(f"Database error fetching all nodes: {e}") from e

    def get_all_relationships_for_visualization(self, limit: int = 2000) -> List[Dict[str, Any]]:
        with self.store._instance_lock:
            try:
                cursor = self.conn.execute(
                    "SELECT relationship_id, source_node_id, target_node_id, relationship_type, relationship_properties FROM graph_relationships LIMIT ?", 
                    (limit,)
                )
                edges = []
                for row in cursor.fetchall():
                    rel_id, source_id, target_id, rel_type, props_json = row
                    
                    edge_label = rel_type
                    try:
                        properties = json.loads(props_json)
                        if isinstance(properties, dict):
                            if 'name' in properties and properties['name']:
                                edge_label = str(properties['name'])
                            elif 'label' in properties and properties['label']:
                                edge_label = str(properties['label'])
                    except (json.JSONDecodeError, TypeError):
                        pass

                    edges.append({
                        "id": rel_id,
                        "from": source_id,
                        "to": target_id,
                        "label": edge_label,
                    })
                return edges
            except sqlite3.Error as e:
                raise GraphDBError(f"Database error fetching all relationships: {e}") from e

    def get_chunks_for_node(self, node_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        with self.store._instance_lock:
            chunk_ids = db.get_chunk_ids_for_nodes_db(self.conn, [node_id]).get(node_id, [])
            return db.get_chunk_details_db(self.conn, chunk_ids[:limit], self.encryptor) if chunk_ids else []

    def get_relationships(self, node_id: int, relationship_type: Optional[str] = None, direction: str = "any", limit: int = 50) -> List[Dict[str, Any]]:
        with self.store._instance_lock:
            return db.get_relationships_for_node_db(self.conn, node_id, relationship_type, direction, limit)

    def clear_graph(self) -> None:
        """Deletes all nodes, relationships, and associated data from the graph."""
        with self.store._instance_lock, self.store._optional_file_lock_context("clear_graph"):
            try:
                self.conn.execute("BEGIN")
                # Order matters: clean up linking tables and vector indexes first
                optional_tables = ["graph_node_vectors", "graph_node_to_chunk_link"]
                for table in optional_tables:
                    try: self.conn.execute(f"DROP TABLE IF EXISTS {table}")
                    except sqlite3.Error as e: ASCIIColors.warning(f"Could not drop table {table}: {e}")
                
                self.conn.execute("DELETE FROM graph_relationships")
                self.conn.execute("DELETE FROM graph_nodes")
                self.conn.execute("UPDATE chunks SET graph_processed_at = NULL")
                
                # Re-initialize the graph features to recreate dropped tables
                embedder_instance = self.embedder
                if embedder_instance.dim is not None:
                    db.enable_vector_search_on_graph_nodes(self.conn, embedder_instance.dim)
                
                self.conn.commit()
                ASCIIColors.success("Graph has been completely cleared and re-initialized.")
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Error clearing the graph: {e}") from e

    def remove_graph_elements_for_document(self, doc_id: int) -> None:
        """
        Removes all nodes and relationships that are exclusively derived from a specific document.
        A node is considered exclusive if it is only linked to chunks from the given doc_id.
        """
        with self.store._instance_lock, self.store._optional_file_lock_context(f"remove_graph_for_doc_{doc_id}"):
            try:
                self.conn.execute("BEGIN")
                doc_chunk_ids_cursor = self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
                doc_chunk_ids = {row[0] for row in doc_chunk_ids_cursor}
                
                if not doc_chunk_ids:
                    ASCIIColors.info(f"No chunks found for document {doc_id}, nothing to remove from graph.")
                    self.conn.commit()
                    return
                
                placeholders = ','.join('?' for _ in doc_chunk_ids)
                
                # Find all nodes linked to this document's chunks
                nodes_linked_to_doc_cursor = self.conn.execute(
                    f"SELECT DISTINCT node_id FROM graph_node_to_chunk_link WHERE chunk_id IN ({placeholders})",
                    list(doc_chunk_ids)
                )
                nodes_linked_to_doc = {row[0] for row in nodes_linked_to_doc_cursor}
                
                nodes_to_delete = set()
                if nodes_linked_to_doc:
                    # For each linked node, check if it's linked to ANY OTHER chunk
                    for node_id in nodes_linked_to_doc:
                        other_links_cursor = self.conn.execute(
                            f"SELECT 1 FROM graph_node_to_chunk_link WHERE node_id = ? AND chunk_id NOT IN ({placeholders}) LIMIT 1",
                            [node_id] + list(doc_chunk_ids)
                        )
                        if other_links_cursor.fetchone() is None:
                            nodes_to_delete.add(node_id)
                
                # Unlink all chunks of the document from the graph
                self.conn.execute(f"UPDATE chunks SET graph_processed_at = NULL WHERE doc_id = ?", (doc_id,))
                self.conn.execute(f"DELETE FROM graph_node_to_chunk_link WHERE chunk_id IN ({placeholders})", list(doc_chunk_ids))
                
                # Delete the exclusive nodes
                if nodes_to_delete:
                    ASCIIColors.info(f"Identified {len(nodes_to_delete)} exclusive nodes to delete for document {doc_id}.")
                    for node_id in nodes_to_delete:
                        db.delete_graph_node_and_relationships_db(self.conn, node_id)
                else:
                    ASCIIColors.info(f"No exclusive nodes found for document {doc_id}.")
                
                self.conn.commit()
                ASCIIColors.success(f"Successfully removed graph elements for document {doc_id}.")
            except Exception as e:
                if self.conn.in_transaction: self.conn.rollback()
                raise GraphError(f"Failed to remove graph elements for document {doc_id}: {e}") from e