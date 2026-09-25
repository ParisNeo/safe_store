from __future__ import annotations
import sqlite3
import threading
import json
import uuid
import re
import inspect
from pathlib import Path
from typing import (
    Optional, Callable, Dict, List, Any, Tuple, TYPE_CHECKING,
    Set, Union, Literal, Protocol, runtime_checkable
)
from ascii_colors import ASCIIColors, trace_exception
from ..core import db
from ..core.exceptions import (
    DatabaseError, ConfigurationError, GraphDBError, GraphProcessingError, LLMCallbackError,
    GraphError, QueryError, NodeNotFoundError, RelationshipNotFoundError, SafeStoreError
)
from ..utils.json_parsing import robust_json_parser
from ..vectorization.base import BaseVectorizer
from .sparql.engine import SparqlEngine

if TYPE_CHECKING:
    from ..store import SafeStore
    from .cognitive_memory import CognitiveMemoryStore

# -----------------------------------------------------------------------------
# Standard LLM Generator Callable Specifications
# -----------------------------------------------------------------------------

@runtime_checkable
class LLMGeneratorProtocol(Protocol):
    """
    Standard protocol for user-provided LLM generation callables.
    
    Any custom function, method, or callable object adhering to this signature
    can be plugged into SafeStore to power knowledge graph extraction,
    SPARQL query synthesis, entity fusion, and natural language graph queries.
    """
    def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs: Any
    ) -> str:
        ...

LLMCallable = Union[
    LLMGeneratorProtocol,
    Callable[[str], str],
    Callable[..., str]
]
ProgressCallback = Callable[[float, str], None]


def wrap_llm_callable(target: Optional[Union[LLMCallable, Any]]) -> Optional[Callable[..., str]]:
    """
    Adapts an arbitrary callable or client object into a robust, standardized LLM generator.

    Handles:
    - 4-argument signature: (prompt, system_prompt=..., json_mode=..., **kwargs)
    - 1-argument signature: (prompt) -> str (automatically prepends system_prompt)
    - Client objects with generate_code / generate_text / generate_structured_content
    - MagicMock instances configured for either client methods or direct invocation
    """
    if target is None:
        return None

    def _client_adapter(prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False, **kwargs) -> str:
        full_p = f"System: {system_prompt}\n\n{prompt}" if system_prompt else prompt
        try:
            # 1. Try generate_code if json_mode or prompt asks for code/json
            if hasattr(target, "generate_code") and (json_mode or "json" in prompt.lower() or "sparql" in prompt.lower()):
                lang = "sparql" if "sparql" in prompt.lower() else "json"
                res = target.generate_code(full_p, language=lang, temperature=kwargs.get("temperature", 0.1))
                if isinstance(res, str) and res.strip():
                    return res
                if res is not None and not (hasattr(res, "_mock_return_value") and str(res).startswith("<MagicMock")):
                    return str(res)

            # 2. Try generate_text
            if hasattr(target, "generate_text"):
                res = target.generate_text(full_p, max_new_tokens=kwargs.get("max_tokens", 1024), temperature=kwargs.get("temperature", 0.1))
                if isinstance(res, str) and res.strip():
                    return res
                if res is not None and not (hasattr(res, "_mock_return_value") and str(res).startswith("<MagicMock")):
                    return str(res)

            # 3. Try general generate
            if hasattr(target, "generate"):
                res = target.generate(full_p)
                if isinstance(res, str) and res.strip():
                    return res
                if res is not None and not (hasattr(res, "_mock_return_value") and str(res).startswith("<MagicMock")):
                    return str(res)

            # 4. If callable directly (e.g. MagicMock(return_value=...), function, or lambda)
            if callable(target):
                res = target(full_p)
                if isinstance(res, str):
                    return res
                if res is not None:
                    return str(res)
        except Exception as e:
            ASCIIColors.warning(f"Client adapter generation error: {e}")
        return ""

    # Distinguish client-like objects
    has_client_methods = (
        ("generate_code" in dir(target) or hasattr(target, "generate_code")) or
        ("generate_text" in dir(target) or hasattr(target, "generate_text")) or
        ("generate_structured_content" in dir(target) or hasattr(target, "generate_structured_content"))
    )

    if has_client_methods:
        # Check if generate_code or generate_text was explicitly set with a return value (like a mock client)
        gen_code = getattr(target, "generate_code", None)
        if gen_code is not None and hasattr(gen_code, "_mock_return_value"):
            try:
                from unittest.mock import DEFAULT
                if gen_code._mock_return_value is not DEFAULT:
                    return _client_adapter
            except ImportError:
                pass
        if not callable(target):
            return _client_adapter

    if not callable(target):
        return None

    # Inspect parameter signature of callable
    try:
        sig = inspect.signature(target)
        params = sig.parameters
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        param_names = set(params.keys())
        accepts_system_prompt = "system_prompt" in param_names or has_var_kwargs
        accepts_json_mode = "json_mode" in param_names or has_var_kwargs
    except (ValueError, TypeError):
        has_var_kwargs = True
        accepts_system_prompt = True
        accepts_json_mode = True

    def _universal_generator(
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs: Any
    ) -> str:
        # If target has client methods, attempt them first
        if has_client_methods:
            adapted = _client_adapter(prompt, system_prompt=system_prompt, json_mode=json_mode, **kwargs)
            if adapted and str(adapted).strip():
                return adapted

        if has_var_kwargs or (accepts_system_prompt and accepts_json_mode):
            call_kwargs = dict(kwargs)
            if accepts_system_prompt:
                call_kwargs["system_prompt"] = system_prompt
            if accepts_json_mode:
                call_kwargs["json_mode"] = json_mode
            try:
                res = target(prompt, **call_kwargs)
                return str(res) if res is not None else ""
            except TypeError:
                pass

        full_text = f"{system_prompt}\n\n{prompt}" if system_prompt and system_prompt.strip() else prompt
        try:
            res = target(full_text)
            return str(res) if res is not None else ""
        except Exception as e:
            ASCIIColors.warning(f"Custom LLM callable invocation failed: {e}")
            return ""

    return _universal_generator


def load_prompt(file_name: str) -> str:
    """Loads a prompt template from the 'prompts' subdirectory."""
    path = Path(__file__).parent / "prompts" / f"{file_name}.md"
    return path.read_text(encoding='utf-8')


class GraphStore:
    """
    Manages a knowledge graph within a SafeStore database.
    Provides SPARQL 1.1 querying, graph building, entity fusion, and relational traversal.
    Powered by a tool-agnostic custom LLM generator callable or native LOLLMS client.
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
        llm_generator: Optional[LLMCallable] = None,
        llm_executor_callback: Optional[Callable[[str], str]] = None,
        lollms_client: Optional[Any] = None,
        ontology: Optional[Union[Dict[str, Any], str]] = None,
        graph_extraction_prompt_template: Optional[str] = None,
        query_parsing_prompt_template: Optional[str] = None,
        entity_fusion_prompt_template: Optional[str] = None,
    ):
        self.store = store
        self.ontology = ontology

        # 1. Resolve custom LLM generator callable (prioritizing explicit parameter, then store inheritance)
        candidate_generator = llm_generator or llm_executor_callback
        if candidate_generator is None and hasattr(store, "llm_generator") and store.llm_generator is not None:
            candidate_generator = store.llm_generator

        # 2. Resolve lollms_client instance if provided
        self.lollms_client = lollms_client
        if self.lollms_client is None and hasattr(store, "lollms_client") and store.lollms_client is not None:
            self.lollms_client = store.lollms_client
        if self.lollms_client is None and candidate_generator is None:
            self.lollms_client = self._try_resolve_lollms_client()

        # 3. Build unified, tool-agnostic generator callable
        if candidate_generator is not None:
            self._llm_generator = wrap_llm_callable(candidate_generator)
        elif self.lollms_client is not None:
            self._llm_generator = wrap_llm_callable(self.lollms_client)
        else:
            self._llm_generator = lambda prompt, **kwargs: '{"nodes": [], "relationships": []}'

        self.graph_extraction_prompt_template = graph_extraction_prompt_template or self.DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE
        self.query_parsing_prompt_template = query_parsing_prompt_template or self.DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE
        self.entity_fusion_prompt_template = entity_fusion_prompt_template or self.DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE
        self.sparql_generation_prompt_template = self.DEFAULT_SPARQL_GENERATION_PROMPT_TEMPLATE
        self._sparql_engine: Optional[SparqlEngine] = None
        self._cognitive_memory: Optional[CognitiveMemoryStore] = None

        ASCIIColors.info(f"Initializing GraphStore with shared SafeStore for database: {self.store.db_path}")
        self._initialize_graph_features()

    @property
    def llm_executor(self) -> Callable[..., str]:
        return self._llm_generator

    @llm_executor.setter
    def llm_executor(self, value: Optional[Callable[..., str]]) -> None:
        self.set_llm_generator(value)

    @property
    def llm_generator(self) -> Callable[..., str]:
        return self._llm_generator

    @llm_generator.setter
    def llm_generator(self, value: Optional[Callable[..., str]]) -> None:
        self.set_llm_generator(value)

    def set_llm_generator(self, generator: LLMCallable) -> None:
        """
        Sets or updates the custom LLM generator callable.
        
        Accepts any callable matching:
            generator(prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False, **kwargs) -> str
        or a simple single-argument callable:
            generator(prompt: str) -> str
        """
        self._llm_generator = wrap_llm_callable(generator) if generator is not None else (lambda prompt, **kwargs: '{"nodes": [], "relationships": []}')

    def set_llm_executor(self, executor: Callable[[str], str]) -> None:
        """Backward-compatible alias for set_llm_generator."""
        self.set_llm_generator(executor)

    def set_lollms_client(self, client: Any) -> None:
        """Sets or updates the active LollmsClient instance."""
        self.lollms_client = client
        if client is not None:
            self.set_llm_generator(client)

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
            guidance_text = ("" if not ontology_schema else "Ontology:\n"+ontology_schema+"\nGuidance:\n") + user_guidance
            try:
                return template.format(chunk_text=chunk_text, user_guidance=guidance_text)
            except KeyError:
                return template.replace("{chunk_text}", chunk_text).replace("{user_guidance}", guidance_text)
        else:
            template = getattr(self, "graph_extraction_prompt_template", None) or self.DEFAULT_GRAPH_EXTRACTION_PROMPT_TEMPLATE
            try:
                return template.format(chunk_text=chunk_text, user_guidance=user_guidance)
            except KeyError:
                return template.replace("{chunk_text}", chunk_text).replace("{user_guidance}", user_guidance)

    def _get_query_parsing_prompt(self, natural_language_query: str) -> str:
        template = getattr(self, "query_parsing_prompt_template", None) or self.DEFAULT_QUERY_PARSING_PROMPT_TEMPLATE
        try:
            return template.format(natural_language_query=natural_language_query)
        except KeyError:
            return template.replace("{natural_language_query}", natural_language_query)

    def _get_entity_fusion_prompt(self, node_a_props: Dict, node_b_props: Dict, label: str) -> str:
        template = getattr(self, "entity_fusion_prompt_template", None) or self.DEFAULT_ENTITY_FUSION_PROMPT_TEMPLATE
        try:
            return template.format(
                node_a_properties=json.dumps(node_a_props, indent=2),
                node_b_properties=json.dumps(node_b_props, indent=2),
                entity_label=label
            )
        except KeyError:
            return (
                template
                .replace("{node_a_properties}", json.dumps(node_a_props, indent=2))
                .replace("{node_b_properties}", json.dumps(node_b_props, indent=2))
                .replace("{entity_label}", label)
            )

    def generate_sparql(self, natural_language_query: str, guidance: Optional[str] = None) -> str:
        """
        Translates a natural language question into an executable W3C SPARQL 1.1 query
        using the configured LLM callable, grounded in the database's live entity classes and schema.
        """
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
        try:
            prompt = template.format(
                schema_context=schema_context + (f"\nAdditional Guidance: {guidance}" if guidance else ""),
                natural_language_query=natural_language_query.strip()
            )
        except KeyError:
            prompt = (
                template
                .replace("{schema_context}", schema_context + (f"\nAdditional Guidance: {guidance}" if guidance else ""))
                .replace("{natural_language_query}", natural_language_query.strip())
            )

        system_prompt = (
            "You are an expert semantic web engineer and W3C SPARQL 1.1 query generator. "
            "Translate the user question into an executable SPARQL 1.1 query matching the schema."
        )

        raw_response = self.llm_generator(
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=False
        )

        if not raw_response or not str(raw_response).strip():
            raise SafeStoreError("Failed to generate SPARQL query: LLM generator returned empty response.")

        cleaned = str(raw_response).strip()
        code_match = re.search(r'```(?:sparql|sql)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
        if code_match:
            cleaned = code_match.group(1).strip()

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
        Builds a W3C-compliant JSON Schema for structured generation,
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
        """Sanitizes text chunk before sending to LLM, replacing inline base64 image blobs."""
        if isinstance(text, bytes):
            clean = text.decode('utf-8', errors='ignore')
        else:
            clean = str(text)

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
        Extracts graph elements from text and links all extracted nodes to the
        provided chunk IDs for evidence provenance.
        """
        parsed = None
        sanitized_text = self._sanitize_chunk_for_llm(text)
        if not sanitized_text.strip():
            return 0, 0

        # Priority 1: LOLLMS Structured Content Generation (if client object is attached)
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
                ASCIIColors.warning(f"Structured content generation fallback for {source_label}: {e}")
                parsed = None

        # Priority 2: Generic LLM Callable Generator
        if parsed is None and self.llm_generator:
            system_prompt = (
                "You are an expert knowledge graph extraction engine. "
                "Extract all entities (nodes), their attributes, and directed relationships (triplets) "
                "from the document content, strictly conforming to the requested JSON format."
            )
            prompt = self._get_graph_extraction_prompt(sanitized_text, guidance)
            raw_response = self.llm_generator(
                prompt=prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
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
        progress_callback: Optional[ProgressCallback] = None,
        stop_event: Optional[threading.Event] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Builds graph nodes and relationships for a specific document with pause/resume support.
        """
        if stop_event and stop_event.is_set():
            return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0, "stopped": True}

        with self.store._instance_lock, self.store._optional_file_lock_context(f"build_graph_for_document: {doc_id}"):
            if resume:
                cursor = self.conn.execute(
                    "SELECT chunk_id, chunk_text, is_encrypted, chunk_seq FROM chunks WHERE doc_id = ? AND graph_processed_at IS NULL ORDER BY chunk_seq ASC",
                    (doc_id,)
                )
            else:
                cursor = self.conn.execute(
                    "SELECT chunk_id, chunk_text, is_encrypted, chunk_seq FROM chunks WHERE doc_id = ? ORDER BY chunk_seq ASC",
                    (doc_id,)
                )
            rows = cursor.fetchall()
            if not rows:
                return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0, "already_processed": True}

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
            processed_chunk_ids = []

            if mode == 'document':
                if stop_event and stop_event.is_set():
                    return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0, "stopped": True}

                full_text = self.store.reconstruct_document_text(doc_id)
                if not full_text:
                    full_text = "\n\n".join(c[1] for c in chunk_records)

                status = f"Extracting graph for full document '{doc_name}' ({len(chunk_records)} chunks in 1 pass)..."
                ASCIIColors.info(status)
                if progress_callback:
                    progress_callback(0.5, status)

                cids = [c[0] for c in chunk_records]
                n_cnt, r_cnt = self._extract_and_insert_graph(
                    text=full_text,
                    chunk_ids=cids,
                    guidance=guidance,
                    source_label=f"Document '{doc_name}'"
                )
                total_nodes += n_cnt
                total_rels += r_cnt
                processed_chunk_ids.extend(cids)
                db.mark_chunks_graph_processed(self.conn, cids)
                self.conn.commit()

            elif mode == 'batch_chunks':
                step = max(1, chunks_per_batch)
                total_batches = (len(chunk_records) + step - 1) // step
                for b_idx in range(0, len(chunk_records), step):
                    if stop_event and stop_event.is_set():
                        break

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
                    processed_chunk_ids.extend(slice_cids)
                    db.mark_chunks_graph_processed(self.conn, slice_cids)
                    self.conn.commit()

            else:  # 'chunk' mode
                for idx, (cid, txt, seq) in enumerate(chunk_records):
                    if stop_event and stop_event.is_set():
                        break

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
                    processed_chunk_ids.append(cid)
                    db.mark_chunks_graph_processed(self.conn, [cid])
                    self.conn.commit()

            return {
                "nodes_created": total_nodes,
                "relationships_created": total_rels,
                "chunks_processed": len(processed_chunk_ids),
                "stopped": bool(stop_event and stop_event.is_set())
            }

    def build_graph_for_all_documents(
        self,
        guidance: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        mode: Literal['document', 'batch_chunks', 'chunk'] = 'document',
        chunks_per_batch: int = 5,
        stop_event: Optional[threading.Event] = None,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Builds knowledge graph across documents in the store with real-time cancellation
        and resume support.
        """
        with self.store._instance_lock, self.store._optional_file_lock_context("build_graph_for_all_documents"):
            if not resume:
                self.conn.execute("UPDATE chunks SET graph_processed_at = NULL")
                self.conn.commit()

            cursor = self.conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunks")
            total_docs = cursor.fetchone()[0] or 0

            cursor = self.conn.execute("SELECT DISTINCT doc_id FROM chunks WHERE graph_processed_at IS NULL ORDER BY doc_id ASC")
            pending_doc_ids = [row[0] for row in cursor.fetchall()]

            if not pending_doc_ids:
                if resume and total_docs > 0:
                    status_done = "All documents are already indexed into the knowledge graph."
                    ASCIIColors.info(status_done)
                    if progress_callback:
                        progress_callback(1.0, status_done)
                    return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0, "all_completed": True}
                else:
                    cursor = self.conn.execute("SELECT DISTINCT doc_id FROM chunks ORDER BY doc_id ASC")
                    pending_doc_ids = [row[0] for row in cursor.fetchall()]

            pending_count = len(pending_doc_ids)
            already_processed_count = max(0, total_docs - pending_count)

            if pending_count == 0:
                ASCIIColors.warning("No documents available to build graph from.")
                return {"nodes_created": 0, "relationships_created": 0, "chunks_processed": 0}

            ASCIIColors.info(f"Building knowledge graph: {pending_count} pending doc(s) out of {total_docs} total [Mode: {mode.upper()}]...")

            total_nodes = 0
            total_rels = 0
            total_chunks_processed = 0
            was_stopped = False

            for d_idx, doc_id in enumerate(pending_doc_ids, 1):
                if stop_event and stop_event.is_set():
                    was_stopped = True
                    break

                doc_row = db.get_document_record_by_id(self.conn, doc_id)
                doc_name = Path(doc_row[1].decode('utf-8')).name if doc_row else f"Doc #{doc_id}"
                current_doc_num = already_processed_count + d_idx

                def _sub_progress(fraction, message):
                    overall_progress = max(0.0, min(1.0, ((current_doc_num - 1) + fraction) / max(1, total_docs)))
                    if progress_callback:
                        progress_callback(overall_progress, f"[{current_doc_num}/{total_docs}] {message}")

                stats = self.build_graph_for_document(
                    doc_id=doc_id,
                    guidance=guidance,
                    mode=mode,
                    chunks_per_batch=chunks_per_batch,
                    progress_callback=_sub_progress,
                    stop_event=stop_event,
                    resume=resume
                )

                total_nodes += stats["nodes_created"]
                total_rels += stats["relationships_created"]
                total_chunks_processed += stats["chunks_processed"]
                self.conn.commit()

                if stats.get("stopped") or (stop_event and stop_event.is_set()):
                    was_stopped = True
                    break

                overall_doc_progress = max(0.0, min(1.0, current_doc_num / max(1, total_docs)))
                status_msg = f"[{current_doc_num}/{total_docs}] '{doc_name}': +{stats['nodes_created']} nodes, +{stats['relationships_created']} edges"
                ASCIIColors.info(status_msg)
                if progress_callback:
                    progress_callback(overall_doc_progress, status_msg)

            self.conn.commit()

            if was_stopped:
                ASCIIColors.warning(f"Graph build paused: {total_nodes} nodes, {total_rels} relationships saved to disk. Ready to resume.")
            else:
                ASCIIColors.success(f"Graph build finished: {total_nodes} nodes, {total_rels} relationships across {total_chunks_processed} chunks permanently saved.")

            return {
                "nodes_created": total_nodes,
                "relationships_created": total_rels,
                "chunks_processed": total_chunks_processed,
                "stopped": was_stopped
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
                system_prompt = "You are an entity resolution expert. Compare two entities and determine if they are identical."
                raw_response = self.llm_generator(prompt=prompt, system_prompt=system_prompt, json_mode=True)
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
                system_prompt = "You are a natural language query parser for knowledge graphs. Extract seed entities and relations into JSON."
                raw_llm_response = self.llm_generator(
                    prompt=self._get_query_parsing_prompt(natural_language_query),
                    system_prompt=system_prompt,
                    json_mode=True
                )
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