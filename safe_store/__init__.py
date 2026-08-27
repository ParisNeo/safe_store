"""
safe_store: Simple SQLite Vector & Knowledge Graph Store for RAG.

A Python utility library providing a lightweight, efficient, and file-based
vector and graph database using SQLite. Optimized for local Retrieval-Augmented
Generation (RAG) pipelines for Large Language Models (LLMs).
Includes SPARQL 1.1 engine, TBox/ABox ontology mapping, BM25 lexical search,
and Tri-Modal Reciprocal Rank Fusion.
"""

from .store import SafeStore, LogLevel, TEMP_FILE_DB_INDICATOR, IN_MEMORY_DB_INDICATOR, DEFAULT_LOCK_TIMEOUT
from .graph.graph_store import GraphStore
from .graph.cognitive_memory import CognitiveMemoryStore
from .graph.ontology.tbox import TBoxManager
from .graph.mapping.tabular_mapper import TabularMapper
from .graph.sparql.engine import SparqlEngine
from .search.bm25 import BM25Retriever
from .search.fusion import reciprocal_rank_fusion, weighted_score_fusion
from .datalake.viewer import DatalakeViewer
from .core.exceptions import (
    SafeStoreError,
    DatabaseError,
    FileHandlingError,
    ParsingError,
    IndexingError,  
    VectorizationError,
    QueryError,
    ConfigurationError,
    ConcurrencyError,
    EncryptionError,
    # Graph specific exceptions
    GraphError,
    GraphDBError,
    GraphProcessingError,
    LLMCallbackError,
    NodeNotFoundError,
    RelationshipNotFoundError,
    DocumentNotFoundError,
    GraphEntityFusionError,
    PageIndexError,
    PageNotFoundError
)
from .indexing.parser import SAFE_STORE_SUPPORTED_FILE_EXTENSIONS, parse_document 
from .processing.text_cleaning import basic_text_cleaner
from ascii_colors import ASCIIColors

__version__ = "3.6.0"

__all__ = [
    "SafeStore",
    "GraphStore",
    "CognitiveMemoryStore",
    "TBoxManager",
    "TabularMapper",
    "SparqlEngine",
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    "DatalakeViewer",
    "ASCIIColors",
    "LogLevel",
    # Exceptions
    "SafeStoreError",
    "DatabaseError",
    "FileHandlingError",
    "ParsingError",
    "IndexingError",
    "VectorizationError",
    "QueryError",
    "ConfigurationError",
    "ConcurrencyError",
    "EncryptionError",
    "GraphError",
    "GraphDBError",
    "GraphProcessingError",
    "LLMCallbackError",
    "NodeNotFoundError",
    "RelationshipNotFoundError",
    "DocumentNotFoundError",
    "GraphEntityFusionError",
    "PageIndexError",
    "PageNotFoundError",
    # globals
    "SAFE_STORE_SUPPORTED_FILE_EXTENSIONS",
    "TEMP_FILE_DB_INDICATOR",
    "IN_MEMORY_DB_INDICATOR",
    "DEFAULT_LOCK_TIMEOUT",
    # utilities
    "parse_document",
    "basic_text_cleaner"
]