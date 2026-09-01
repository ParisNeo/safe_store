import json
import sqlite3
import hashlib
from typing import Dict, Any, List, Optional, Union, Literal, Iterator
from pathlib import Path
import numpy as np

from ascii_colors import ASCIIColors
from ..core.exceptions import ConfigurationError, EncryptionError
from ..core import db

ProjectionMethod = Literal['pca', 'tsne', 'umap', 'incremental_pca']
OutputFormat = Literal['dict', 'json_str', 'csv', 'dataframe']


class DatalakeViewer:
    """
    High-performance semantic datalake engine for SafeStore.
    Supports PCA, t-SNE, and UMAP 2D/3D projections with persistent caching,
    lazy stream processing, and interactive visualizer exports.
    """

    def __init__(self, store: Any):
        self.store = store

    def _get_dataset_fingerprint(
        self,
        method: str,
        n_components: int,
        sample_size: Optional[int] = None,
        filter_doc_ids: Optional[List[int]] = None
    ) -> str:
        """Computes a deterministic fingerprint of the current store state and query params."""
        self.store._ensure_connection()
        conn = self.store.conn
        assert conn is not None
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), MAX(doc_id) FROM documents")
        doc_count, max_doc_id = cursor.fetchone() or (0, 0)
        
        cursor.execute("SELECT COUNT(*) FROM vectors")
        vector_count = cursor.fetchone()[0] or 0

        v_info = db.get_store_metadata(conn, "vectorizer_info") or ""

        raw_key = f"{doc_count}_{max_doc_id}_{vector_count}_{v_info}_{method}_{n_components}_{sample_size}_{filter_doc_ids}"
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    def _compute_projections(
        self,
        vectors: np.ndarray,
        method: ProjectionMethod = 'pca',
        n_components: int = 2,
        random_state: int = 42,
        batch_size: int = 500,
        **kwargs
    ) -> np.ndarray:
        """Calculates 2D or 3D coordinate projections for vector matrices."""
        n_samples, dim = vectors.shape
        if n_samples == 0:
            return np.empty((0, n_components), dtype=np.float32)

        if n_samples < n_components:
            padded = np.zeros((n_samples, n_components), dtype=np.float32)
            if n_samples > 0:
                padded[:, :min(dim, n_components)] = vectors[:, :min(dim, n_components)]
            return padded

        method_lower = method.lower()

        if method_lower == 'pca':
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                raise ConfigurationError("PCA projection requires 'scikit-learn'. Install with: pip install scikit-learn")
            
            pca = PCA(n_components=n_components, random_state=random_state)
            return pca.fit_transform(vectors).astype(np.float32)

        elif method_lower == 'incremental_pca':
            try:
                from sklearn.decomposition import IncrementalPCA
            except ImportError:
                raise ConfigurationError("IncrementalPCA projection requires 'scikit-learn'. Install with: pip install scikit-learn")
            
            effective_batch = max(n_components, min(batch_size, n_samples))
            ipca = IncrementalPCA(n_components=n_components, batch_size=effective_batch)
            return ipca.fit_transform(vectors).astype(np.float32)

        elif method_lower == 'tsne':
            try:
                from sklearn.manifold import TSNE
            except ImportError:
                raise ConfigurationError("t-SNE projection requires 'scikit-learn'. Install with: pip install scikit-learn")

            if n_samples < 5:
                ASCIIColors.warning(f"Sample size ({n_samples}) too small for reliable t-SNE. Falling back to PCA.")
                return self._compute_projections(vectors, method='pca', n_components=n_components, random_state=random_state)

            default_perp = kwargs.get('perplexity', 30.0)
            adaptive_perp = min(default_perp, max(2.0, float(n_samples - 1) / 3.0))

            init_mode = 'pca' if dim >= n_components else 'random'
            tsne = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=adaptive_perp,
                init=init_mode,
                learning_rate=kwargs.get('learning_rate', 'auto'),
                max_iter=kwargs.get('max_iter', 1000)
            )
            return tsne.fit_transform(vectors).astype(np.float32)

        elif method_lower == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
                return reducer.fit_transform(vectors).astype(np.float32)
            except ImportError:
                ASCIIColors.warning("UMAP not installed ('umap-learn'). Falling back to t-SNE.")
                return self._compute_projections(vectors, method='tsne', n_components=n_components, random_state=random_state, **kwargs)

        else:
            raise ValueError(f"Unknown projection method: '{method}'. Supported: 'pca', 'tsne', 'umap', 'incremental_pca'.")

    def get_datalake_view(
        self,
        method: ProjectionMethod = 'pca',
        n_components: int = 2,
        use_cache: bool = True,
        sample_size: Optional[int] = None,
        filter_doc_ids: Optional[List[int]] = None,
        output_format: OutputFormat = 'dict',
        include_chunk_text: bool = True
    ) -> Union[List[Dict[str, Any]], str, Any]:
        """
        Retrieves a complete datalake point-cloud view with 2D/3D semantic coordinates,
        document provenance, and metadata.
        """
        if n_components not in (2, 3):
            raise ValueError("n_components must be either 2 or 3.")

        self.store._ensure_connection()
        conn = self.store.conn
        assert conn is not None

        fingerprint = self._get_dataset_fingerprint(method, n_components, sample_size, filter_doc_ids)
        cache_key = f"datalake_cache_{fingerprint}"

        # 1. Check persistent SQLite cache if enabled
        if use_cache:
            cached_json = db.get_store_metadata(conn, cache_key)
            if cached_json:
                ASCIIColors.debug("Returning datalake view from persistent cache.")
                cached_data = json.loads(cached_json)
                return self._format_output(cached_data, output_format)

        # 2. Fetch raw vector records from database
        sql = """
            SELECT v.chunk_id, v.vector_data, d.doc_id, d.file_path, d.metadata, d.is_encrypted AS doc_is_encrypted,
                   c.chunk_text, c.is_encrypted AS chunk_is_encrypted
            FROM vectors v
            JOIN chunks c ON v.chunk_id = c.chunk_id
            JOIN documents d ON c.doc_id = d.doc_id
        """
        params = []
        if filter_doc_ids:
            placeholders = ','.join('?' * len(filter_doc_ids))
            sql += f" WHERE d.doc_id IN ({placeholders})"
            params.extend(filter_doc_ids)

        if sample_size and sample_size > 0:
            sql += f" ORDER BY RANDOM() LIMIT {int(sample_size)}"

        original_factory = conn.text_factory
        conn.text_factory = bytes
        cursor = conn.cursor()
        rows = cursor.execute(sql, tuple(params)).fetchall()
        conn.text_factory = original_factory

        if not rows:
            if output_format == 'dict': return []
            if output_format == 'json_str': return "[]"
            if output_format == 'csv': return ""
            if output_format == 'dataframe':
                try:
                    import pandas as pd
                    return pd.DataFrame()
                except ImportError:
                    return []
            return []

        v_info = self.store.get_vectorization_details() or {}
        dtype_str = v_info.get("dtype", "float32")

        chunk_ids = []
        vectors_list = []
        doc_ids = []
        file_paths = []
        doc_metadatas = []
        chunk_texts = []

        for row in rows:
            cid, v_blob, did, path_b, meta_b, doc_enc, chunk_b, chunk_enc = row
            chunk_ids.append(cid)
            vectors_list.append(db.reconstruct_vector(v_blob, dtype_str))
            doc_ids.append(did)
            file_paths.append(path_b.decode('utf-8'))

            # Handle metadata decryption
            meta_dict = None
            if meta_b:
                if doc_enc and self.store.encryptor.is_enabled:
                    try:
                        meta_dict = json.loads(self.store.encryptor.decrypt(meta_b))
                    except Exception:
                        meta_dict = {"error": "metadata_decryption_failed"}
                elif not doc_enc:
                    try:
                        meta_dict = json.loads(meta_b.decode('utf-8'))
                    except Exception:
                        meta_dict = {}
            doc_metadatas.append(meta_dict or {})

            # Handle chunk text decryption
            if include_chunk_text:
                if chunk_enc:
                    if self.store.encryptor.is_enabled:
                        try:
                            chunk_text = self.store.encryptor.decrypt(chunk_b)
                        except EncryptionError:
                            chunk_text = "[Encrypted Chunk - Decryption Failed]"
                    else:
                        chunk_text = "[Encrypted Chunk - Key Unavailable]"
                else:
                    chunk_text = chunk_b.decode('utf-8', errors='ignore')
                chunk_texts.append(chunk_text)
            else:
                chunk_texts.append("")

        X = np.array(vectors_list)
        projections = self._compute_projections(X, method=method, n_components=n_components)

        datalake_points = []
        for i in range(len(chunk_ids)):
            pt = {
                "chunk_id": int(chunk_ids[i]),
                "doc_id": int(doc_ids[i]),
                "document_path": file_paths[i],
                "document_title": Path(file_paths[i]).name,
                "x": float(projections[i, 0]),
                "y": float(projections[i, 1]),
                "metadata": doc_metadatas[i]
            }
            if n_components == 3:
                pt["z"] = float(projections[i, 2])
            if include_chunk_text:
                pt["chunk_text_preview"] = chunk_texts[i][:200] + ("..." if len(chunk_texts[i]) > 200 else "")

            datalake_points.append(pt)

        # 3. Store in SQLite metadata cache
        if use_cache:
            try:
                db.set_store_metadata(conn, cache_key, json.dumps(datalake_points))
                if not conn.in_transaction:
                    conn.commit()
            except Exception as e:
                ASCIIColors.warning(f"Could not write datalake projection to cache: {e}")


        return self._format_output(datalake_points, output_format)

    def stream_datalake_chunks(
        self,
        batch_size: int = 500,
        method: ProjectionMethod = 'incremental_pca',
        n_components: int = 2
    ) -> Iterator[Dict[str, Any]]:
        """
        Lazy streaming generator yielding datalake points in incremental batches,
        ideal for handling large vector stores without high memory footprint.
        """
        self.store._ensure_connection()
        conn = self.store.conn
        assert conn is not None

        v_info = self.store.get_vectorization_details() or {}
        dtype_str = v_info.get("dtype", "float32")

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vectors")
        total_vectors = cursor.fetchone()[0] or 0

        if total_vectors == 0:
            return

        from sklearn.decomposition import IncrementalPCA
        ipca = IncrementalPCA(n_components=n_components, batch_size=min(batch_size, total_vectors))

        # First pass: fit IncrementalPCA incrementally
        sql = "SELECT vector_data FROM vectors ORDER BY chunk_id ASC"
        cursor.execute(sql)
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            batch_vecs = np.array([db.reconstruct_vector(r[0], dtype_str) for r in rows])
            if len(batch_vecs) >= n_components:
                ipca.partial_fit(batch_vecs)

        # Second pass: stream transformed points
        sql_details = """
            SELECT v.chunk_id, v.vector_data, d.doc_id, d.file_path, d.metadata, d.is_encrypted AS doc_is_encrypted,
                   c.chunk_text, c.is_encrypted AS chunk_is_encrypted
            FROM vectors v
            JOIN chunks c ON v.chunk_id = c.chunk_id
            JOIN documents d ON c.doc_id = d.doc_id
            ORDER BY v.chunk_id ASC
        """
        original_factory = conn.text_factory
        conn.text_factory = bytes
        cursor = conn.cursor()
        cursor.execute(sql_details)

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            batch_vecs = np.array([db.reconstruct_vector(r[1], dtype_str) for r in rows])
            coords = ipca.transform(batch_vecs)

            for i, row in enumerate(rows):
                cid, _, did, path_b, meta_b, doc_enc, chunk_b, chunk_enc = row
                
                meta_dict = None
                if meta_b and not doc_enc:
                    try: meta_dict = json.loads(meta_b.decode('utf-8'))
                    except Exception: meta_dict = {}

                chunk_text = chunk_b.decode('utf-8', errors='ignore') if not chunk_enc else "[Encrypted]"
                
                item = {
                    "chunk_id": int(cid),
                    "doc_id": int(did),
                    "document_path": path_b.decode('utf-8'),
                    "document_title": Path(path_b.decode('utf-8')).name,
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "metadata": meta_dict or {},
                    "chunk_text_preview": chunk_text[:200]
                }
                if n_components == 3:
                    item["z"] = float(coords[i, 2])
                yield item

        conn.text_factory = original_factory

    def export_datalake_html(
        self,
        output_file: Union[str, Path] = "datalake_view.html",
        title: str = "SafeStore Semantic Datalake Explorer",
        method: ProjectionMethod = 'pca',
        n_components: int = 2,
        sample_size: Optional[int] = None
    ) -> Path:
        """
        Generates an interactive standalone HTML datalake visualizer with 2D/3D Plotly canvas,
        hover inspections, and instant search filtering.
        """
        data = self.get_datalake_view(
            method=method,
            n_components=n_components,
            sample_size=sample_size,
            output_format='dict'
        )

        data_json = json.dumps(data)
        out_path = Path(output_file)

        is_3d = (n_components == 3)
        plot_type = 'scatter3d' if is_3d else 'scatter'

        html_template = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>__TITLE__</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body class="bg-slate-900 text-slate-100 font-sans antialiased min-h-screen">
    <main class="container mx-auto p-4 md:p-6">
        <header class="mb-6 flex justify-between items-center border-b border-slate-800 pb-4">
            <div>
                <h1 class="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400">
                    __TITLE__
                </h1>
                <p class="text-sm text-slate-400 mt-1">
                    Projection: <span class="font-semibold text-teal-300">__METHOD__ (__COMPONENTS__D)</span> | Points: <span class="font-semibold text-teal-300">__POINT_COUNT__</span>
                </p>
            </div>
            <div class="flex items-center gap-3">
                <input id="search-input" type="text" placeholder="Filter documents..." class="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500">
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div class="lg:col-span-3 bg-slate-800/80 rounded-xl shadow-xl p-4 h-[75vh]">
                <div id="plot" class="w-full h-full"></div>
            </div>
            <div class="lg:col-span-1 bg-slate-800/80 rounded-xl shadow-xl p-5 flex flex-col h-[75vh]">
                <h2 class="text-lg font-semibold text-teal-400 mb-3 border-b border-slate-700 pb-2">Chunk Inspector</h2>
                <div id="inspector" class="flex-1 overflow-y-auto text-sm text-slate-300 space-y-3 font-mono">
                    <p class="text-slate-500 text-xs italic">Hover over or click a point to inspect semantic content and metadata.</p>
                </div>
            </div>
        </div>
    </main>

    <script>
        const rawData = __DATA_JSON__;
        const is3D = __IS_3D__;
        const plotDiv = document.getElementById('plot');
        const inspector = document.getElementById('inspector');

        function renderPlot(filtered) {
            const docGroups = {};
            filtered.forEach(d => {
                if (!docGroups[d.document_title]) docGroups[d.document_title] = [];
                docGroups[d.document_title].push(d);
            });

            const traces = Object.keys(docGroups).map(title => {
                const group = docGroups[title];
                const trace = {
                    name: title,
                    type: '__PLOT_TYPE__',
                    mode: 'markers',
                    x: group.map(p => p.x),
                    y: group.map(p => p.y),
                    customdata: group,
                    marker: { size: is3D ? 4 : 8, opacity: 0.85 }
                };
                if (is3D) trace.z = group.map(p => p.z);
                return trace;
            });

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#94a3b8' },
                margin: { l: 30, r: 30, t: 30, b: 30 },
                legend: { orientation: 'h', y: -0.1 },
                hovermode: 'closest'
            };

            Plotly.newPlot(plotDiv, traces, layout, { responsive: true });

            plotDiv.on('plotly_hover', function(data) {
                if (data.points.length > 0) {
                    const p = data.points[0].customdata;
                    inspector.innerHTML = `
                        <div class="space-y-2">
                            <div><span class="text-teal-400">Doc:</span> ${p.document_title}</div>
                            <div><span class="text-teal-400">Chunk ID:</span> ${p.chunk_id}</div>
                            <div><span class="text-teal-400">Coordinates:</span> [${p.x.toFixed(3)}, ${p.y.toFixed(3)}${is3D ? ', ' + p.z.toFixed(3) : ''}]</div>
                            ${p.chunk_text_preview ? '<div class="pt-2 border-t border-slate-700 text-slate-200">' + p.chunk_text_preview + '</div>' : ''}
                            <div class="text-xs text-slate-500 pt-2">Metadata: ${JSON.stringify(p.metadata)}</div>
                        </div>
                    `;
                }
            });
        }

        renderPlot(rawData);

        document.getElementById('search-input').addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            const filtered = rawData.filter(d => d.document_title.toLowerCase().includes(term) || JSON.stringify(d.metadata).toLowerCase().includes(term));
            renderPlot(filtered);
        });
    </script>
</body>
</html>"""

        html_content = (
            html_template
            .replace("__TITLE__", title)
            .replace("__METHOD__", method.upper())
            .replace("__COMPONENTS__", str(n_components))
            .replace("__POINT_COUNT__", str(len(data)))
            .replace("__DATA_JSON__", data_json.replace("</", "<\\/"))
            .replace("__IS_3D__", str(is_3d).lower())
            .replace("__PLOT_TYPE__", plot_type)
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content, encoding='utf-8')
        ASCIIColors.success(f"Datalake visualization exported to {out_path.resolve()}")
        return out_path

    def _format_output(self, data: List[Dict[str, Any]], output_format: OutputFormat) -> Any:
        """Formats the datalake payload into the requested output structure."""
        if output_format == 'dict':
            return data
        elif output_format == 'json_str':
            return json.dumps(data)
        elif output_format in ('csv', 'dataframe'):
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                if output_format == 'dataframe':
                    return df
                return df.to_csv(index=False)
            except ImportError:
                if output_format == 'dataframe':
                    raise ConfigurationError("'dataframe' format requires pandas. Install with: pip install pandas")
                if not data:
                    return ""
                headers = list(data[0].keys())
                lines = [",".join(headers)]
                for row in data:
                    lines.append(",".join(f'"{str(row.get(h, ""))}"' for h in headers))
                return "\n".join(lines)
        return data