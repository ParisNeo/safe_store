import sqlite3
import json
from typing import Tuple
from .base import BaseVectorizer
from .methods.sentence_transformer import SentenceTransformerVectorizer
from ..core import db # Import db functions
from ascii_colors import ASCIIColors
import numpy as np

class VectorizationManager:
    """Manages available vectorization methods."""

    def __init__(self):
        self._cache: dict[str, Tuple[BaseVectorizer, int]] = {} # Cache: name -> (instance, method_id)

    def get_vectorizer(self, name: str, conn: sqlite3.Connection) -> Tuple[BaseVectorizer, int]:
        """
        Gets or initializes a vectorizer instance and its corresponding DB method ID.

        Args:
            name: The identifier for the vectorizer (e.g., 'st:all-MiniLM-L6-v2').
            conn: Active database connection.

        Returns:
            A tuple containing the vectorizer instance and its method_id from the DB.
        """
        if name in self._cache:
            ASCIIColors.debug(f"Vectorizer '{name}' found in cache.")
            return self._cache[name]

        ASCIIColors.info(f"Initializing vectorizer: {name}")
        vectorizer = None
        params = {}
        method_type = "unknown"

        # Determine vectorizer type and parameters from name (simple logic for now)
        if name.startswith("st:") or name == "default_sentence_transformer":
             method_type = "sentence_transformer"
             model_name = name.split(":", 1)[1] if ":" in name else SentenceTransformerVectorizer.DEFAULT_MODEL
             params = {"model_name": model_name}
             try:
                 vectorizer = SentenceTransformerVectorizer(model_name=model_name)
             except ImportError as e:
                 ASCIIColors.error(f"Cannot initialize '{name}'. Missing dependency: {e}")
                 raise
             except Exception as e:
                 ASCIIColors.error(f"Failed to initialize SentenceTransformer '{model_name}': {e}")
                 raise

        # Add logic for other types later (tfidf, openai, ollama)
        # elif name.startswith("tfidf"):
        #     method_type = "tfidf"
        #     # ... instantiate TFIDFVectorizer ...

        else:
            ASCIIColors.error(f"Unknown vectorizer name format: {name}")
            raise ValueError(f"Unknown vectorizer name format: {name}")

        if vectorizer:
            # Add/Get the method in the database
            method_id = db.add_or_get_vectorization_method(
                conn=conn,
                name=name, # Use the full identifier as the unique name
                type=method_type,
                dim=vectorizer.dim,
                dtype=np.dtype(vectorizer.dtype).name, # Store dtype as string 'float32' etc.
                params=json.dumps(params)
            )
            self._cache[name] = (vectorizer, method_id)
            ASCIIColors.debug(f"Vectorizer '{name}' initialized and registered with method_id {method_id}.")
            return vectorizer, method_id
        else:
             # Should have raised earlier if vectorizer is None
             raise RuntimeError(f"Vectorizer instance could not be created for '{name}'.")