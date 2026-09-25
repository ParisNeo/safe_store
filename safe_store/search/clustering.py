import json
import sqlite3
from typing import Dict, Any, List, Optional, Union, Literal, Tuple, Callable
from pathlib import Path
import numpy as np
from ascii_colors import ASCIIColors
from ..core import db
from ..core.exceptions import SafeStoreError, ConfigurationError
from ..utils.json_parsing import robust_json_parser


ClusteringAlgorithm = Literal['kmeans', 'agglomerative']


class DocumentClusterer:
    """
    Semantic Document Clustering & Thematic Grouping Engine for SafeStore.

    - Computes normalized document-level vector centroids from chunk embeddings.
    - Groups documents using K-Means or Agglomerative Hierarchical clustering.
    - Automatically determines optimal cluster counts when n_clusters='auto'.
    - Extracts distinctive c-TF-IDF keywords per cluster.
    - Synthesizes conceptual theme titles, descriptions, and topic tags using
      an LLM generator or deterministic lexical heuristics.
    """

    def __init__(self, store: Any):
        self.store = store

    def compute_document_centroids(
        self,
        filter_doc_ids: Optional[List[int]] = None
    ) -> Tuple[List[int], List[str], np.ndarray, Dict[int, Dict[str, Any]]]:
        """
        Calculates normalized vector centroids for all documents with vectors.

        Returns:
            (doc_ids, file_paths, centroid_matrix, doc_metadata_map)
        """
        self.store._ensure_connection()
        conn = self.store.conn
        assert conn is not None

        v_details = self.store.get_vectorization_details() or {}
        dtype_str = v_details.get("dtype", "float32")

        # Fetch documents and their vectors
        sql = """
            SELECT d.doc_id, d.file_path, d.metadata, d.is_encrypted, v.vector_data
            FROM documents d
            JOIN chunks c ON d.doc_id = c.doc_id
            JOIN vectors v ON c.chunk_id = v.chunk_id
        """
        params = []
        if filter_doc_ids:
            placeholders = ','.join('?' * len(filter_doc_ids))
            sql += f" WHERE d.doc_id IN ({placeholders})"
            params.extend(filter_doc_ids)

        sql += " ORDER BY d.doc_id ASC"

        original_factory = conn.text_factory
        conn.text_factory = bytes
        cursor = conn.cursor()
        rows = cursor.execute(sql, tuple(params)).fetchall()
        conn.text_factory = original_factory

        if not rows:
            return [], [], np.empty((0, 1), dtype=np.float32), {}

        # Group vector blobs by document
        doc_vectors_map: Dict[int, List[np.ndarray]] = {}
        doc_meta_map: Dict[int, Dict[str, Any]] = {}
        doc_path_map: Dict[int, str] = {}

        for doc_id, path_bytes, meta_bytes, is_enc, vec_blob in rows:
            if doc_id not in doc_vectors_map:
                doc_vectors_map[doc_id] = []
                file_path = path_bytes.decode('utf-8', errors='ignore')
                doc_path_map[doc_id] = file_path

                meta_dict = None
                if meta_bytes:
                    if is_enc and self.store.encryptor.is_enabled:
                        try:
                            meta_dict = json.loads(self.store.encryptor.decrypt(meta_bytes))
                        except Exception:
                            meta_dict = {}
                    elif not is_enc:
                        try:
                            meta_dict = json.loads(meta_bytes.decode('utf-8'))
                        except Exception:
                            meta_dict = {}

                doc_meta_map[doc_id] = {
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "document_title": Path(file_path).name,
                    "metadata": meta_dict or {}
                }

            vec = db.reconstruct_vector(vec_blob, dtype_str)
            doc_vectors_map[doc_id].append(vec)

        doc_ids = []
        file_paths = []
        centroids = []

        for doc_id, vecs in doc_vectors_map.items():
            if not vecs:
                continue
            doc_ids.append(doc_id)
            file_paths.append(doc_path_map[doc_id])

            # Calculate document centroid (mean of chunk vectors) and normalize
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            normalized_centroid = mean_vec / norm if norm > 0 else mean_vec
            centroids.append(normalized_centroid)

        if not centroids:
            return [], [], np.empty((0, 1), dtype=np.float32), {}

        return doc_ids, file_paths, np.array(centroids, dtype=np.float32), doc_meta_map

    def cluster_documents(
        self,
        n_clusters: Union[int, Literal['auto']] = 'auto',
        method: ClusteringAlgorithm = 'kmeans',
        generate_themes: bool = True,
        save_to_store: bool = True,
        random_state: int = 42,
        filter_doc_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Clusters stored documents and generates thematic metadata for each group.
        """
        doc_ids, file_paths, centroids, doc_info_map = self.compute_document_centroids(filter_doc_ids=filter_doc_ids)
        num_docs = len(doc_ids)

        if num_docs == 0:
            return []

        # Trivial single document case
        if num_docs == 1:
            cluster_docs = [doc_info_map[doc_ids[0]]]
            theme = self._build_theme_for_cluster(cluster_docs, doc_ids) if generate_themes else {
                "theme_title": doc_info_map[doc_ids[0]]["document_title"],
                "theme_description": "Single document group.",
                "key_topics": []
            }
            single_result = [{
                "cluster_id": 0,
                "theme_title": theme["theme_title"],
                "theme_description": theme["theme_description"],
                "key_topics": theme["key_topics"],
                "document_count": 1,
                "documents": cluster_docs,
                "centroid": centroids[0].tolist()
            }]
            if save_to_store:
                self.save_clusters_to_store(single_result)
            return single_result

        # Determine number of clusters
        k = self._determine_k(centroids, n_clusters)
        k = max(1, min(k, num_docs))

        labels = self._execute_clustering(centroids, k, method=method, random_state=random_state)

        # Group documents by cluster label
        clusters_raw: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            cid = int(label)
            if cid not in clusters_raw:
                clusters_raw[cid] = []
            clusters_raw[cid].append(doc_ids[idx])

        # Build cluster representations
        results = []
        for cid, member_doc_ids in sorted(clusters_raw.items()):
            member_docs = [doc_info_map[did] for did in member_doc_ids]
            member_indices = [doc_ids.index(did) for did in member_doc_ids]
            cluster_centroid = np.mean(centroids[member_indices], axis=0)
            norm = np.linalg.norm(cluster_centroid)
            if norm > 0:
                cluster_centroid = cluster_centroid / norm

            if generate_themes:
                theme_info = self._build_theme_for_cluster(member_docs, member_doc_ids)
            else:
                sample_titles = [d["document_title"] for d in member_docs[:3]]
                theme_info = {
                    "theme_title": " / ".join(sample_titles),
                    "theme_description": f"Group of {len(member_docs)} related document(s).",
                    "key_topics": []
                }

            results.append({
                "cluster_id": cid,
                "theme_title": theme_info.get("theme_title", f"Theme {cid + 1}"),
                "theme_description": theme_info.get("theme_description", ""),
                "key_topics": theme_info.get("key_topics", []),
                "document_count": len(member_docs),
                "documents": member_docs,
                "centroid": cluster_centroid.tolist()
            })

        # Sort clusters by document count descending
        results.sort(key=lambda c: c["document_count"], reverse=True)
        # Re-index cluster_id cleanly
        for i, c in enumerate(results):
            c["cluster_id"] = i

        if save_to_store:
            self.save_clusters_to_store(results)

        return results

    def _determine_k(
        self,
        centroids: np.ndarray,
        n_clusters: Union[int, str]
    ) -> int:
        num_docs = centroids.shape[0]
        if isinstance(n_clusters, int) and n_clusters > 0:
            return min(n_clusters, num_docs)

        # Automatic k estimation based on dataset size and heuristic inertia
        if num_docs <= 3:
            return num_docs
        if num_docs <= 7:
            return 2
        if num_docs <= 15:
            return 3

        # Default rule of thumb: sqrt(num_docs / 2) clamped between 3 and 10
        k_est = int(np.round(np.sqrt(num_docs / 2.0)))
        return max(2, min(10, k_est))

    def _execute_clustering(
        self,
        centroids: np.ndarray,
        k: int,
        method: ClusteringAlgorithm = 'kmeans',
        random_state: int = 42
    ) -> np.ndarray:
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
        except ImportError as e:
            raise ConfigurationError("Clustering requires 'scikit-learn'. Install with: pip install scikit-learn") from e

        if k == 1:
            return np.zeros(centroids.shape[0], dtype=int)

        if method == 'kmeans':
            clusterer = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            return clusterer.fit_predict(centroids)
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
            return clusterer.fit_predict(centroids)
        else:
            raise ValueError(f"Unknown clustering method: '{method}'. Supported: 'kmeans', 'agglomerative'.")

    def _extract_cluster_keywords(self, doc_ids: List[int], max_keywords: int = 8) -> List[str]:
        """Extracts top distinctive terms from cluster documents using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return []

        doc_texts = []
        for did in doc_ids:
            text = self.store.reconstruct_document_text(did) or ""
            if text.strip():
                doc_texts.append(text[:3000])

        if not doc_texts:
            return []

        try:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
            )
            tfidf_matrix = vectorizer.fit_transform(doc_texts)
            feature_names = np.array(vectorizer.get_feature_names_out())
            scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            top_indices = np.argsort(scores)[::-1][:max_keywords]
            return [str(feature_names[idx]) for idx in top_indices]
        except Exception:
            return []

    def _build_theme_for_cluster(
        self,
        member_docs: List[Dict[str, Any]],
        doc_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Builds thematic title, description, and tags for a cluster using an LLM generator
        or deterministic lexical analysis.
        """
        keywords = self._extract_cluster_keywords(doc_ids, max_keywords=8)
        titles = [d["document_title"] for d in member_docs]

        # Check if an LLM generator is configured on the store
        llm = getattr(self.store, "llm_generator", None)

        if llm is not None:
            doc_summaries = []
            for d in member_docs[:6]:
                did = d["doc_id"]
                preview = (self.store.reconstruct_document_text(did) or "")[:400].replace("\n", " ")
                doc_summaries.append(f"- '{d['document_title']}': {preview}")

            docs_context = "\n".join(doc_summaries)
            keywords_context = ", ".join(keywords) if keywords else "None extracted"

            system_prompt = (
                "You are an expert thematic analyst and information architect. "
                "Synthesize a clear theme title, a short conceptual summary, and key topic tags "
                "for the group of related documents below, strictly in JSON format."
            )

            user_prompt = f"""Group of Related Documents:
{docs_context}

Extracted Prominent Keywords: {keywords_context}

Respond STRICTLY with a single JSON object in this format:
```json
{{
    "theme_title": "A concise 3 to 6 word descriptive title for this group",
    "theme_description": "A 1-2 sentence summary explaining the core subject unifying these documents.",
    "key_topics": ["topic1", "topic2", "topic3", "topic4"]
}}
```"""
            try:
                raw_response = llm(prompt=user_prompt, system_prompt=system_prompt, json_mode=True)
                parsed = robust_json_parser(raw_response)
                if isinstance(parsed, dict) and "theme_title" in parsed:
                    return {
                        "theme_title": str(parsed.get("theme_title", "")).strip(),
                        "theme_description": str(parsed.get("theme_description", "")).strip(),
                        "key_topics": parsed.get("key_topics", keywords[:5])
                    }
            except Exception as e:
                ASCIIColors.warning(f"LLM theme generation fallback to lexical analysis: {e}")

        # Deterministic Lexical Fallback
        if keywords:
            capitalized_words = [w.capitalize() for w in keywords[:3]]
            title = " & ".join(capitalized_words)
            desc = f"Documents exploring {' and '.join(keywords[:4])} across {len(member_docs)} member document(s)."
            topics = keywords[:6]
        else:
            title = f"{titles[0]} & Related Documents"
            desc = f"Cohesive thematic cluster comprising {len(member_docs)} document(s)."
            topics = [t.split('.')[0] for t in titles[:5]]

        return {
            "theme_title": title,
            "theme_description": desc,
            "key_topics": topics
        }

    def save_clusters_to_store(self, clusters: List[Dict[str, Any]]) -> None:
        """Persists computed clusters and themes into store_metadata."""
        self.store._ensure_connection()
        conn = self.store.conn
        assert conn is not None

        try:
            compact = []
            for c in clusters:
                item = dict(c)
                if "centroid" in item:
                    item.pop("centroid")
                compact.append(item)

            payload_json = json.dumps(compact, indent=2)
            db.set_store_metadata(conn, "document_clusters", payload_json)
        except Exception as e:
            ASCIIColors.warning(f"Failed to persist clusters in database: {e}")

    def get_cached_clusters(self) -> Optional[List[Dict[str, Any]]]:
        """Retrieves cached document clusters from store_metadata if available."""
        self.store._ensure_connection()
        conn = self.store.conn
        assert conn is not None

        raw = db.get_store_metadata(conn, "document_clusters")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None