import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json

import pipmaster as pm

# Ensure UI dependencies
pm.ensure_packages(["nicegui>=1.4.0", "pywebview>=4.0", "plotly>=5.0", "pandas"])

from nicegui import ui, app, Client
import plotly.graph_objects as go
import time
import safe_store
from safe_store import SafeStore, GraphStore, LogLevel

# Global configured database path for initial page loads
CURRENT_DB_PATH: Optional[str] = None
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)


def inspect_store_summary(db_path: Path) -> Dict[str, Any]:
    """Lightweight read-only scanner to extract store metadata without loading neural models."""
    import sqlite3
    db_file = Path(db_path).resolve()
    info: Dict[str, Any] = {
        "path": str(db_file),
        "filename": db_file.name,
        "name": db_file.stem,
        "description": "",
        "doc_count": 0,
        "chunk_count": 0,
        "node_count": 0,
        "rel_count": 0,
        "vectorizer_name": "unknown",
        "vectorizer_model": "",
        "size_mb": round(db_file.stat().st_size / (1024 * 1024), 2) if db_file.exists() else 0,
        "modified_time": time.strftime('%Y-%m-%d %H:%M', time.localtime(db_file.stat().st_mtime)) if db_file.exists() else "",
        "is_encrypted": False,
        "is_valid": False
    }

    if not db_file.exists():
        return info

    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if "documents" in tables:
            cursor.execute("SELECT COUNT(*) FROM documents")
            info["doc_count"] = cursor.fetchone()[0] or 0

            cursor.execute("SELECT 1 FROM documents WHERE is_encrypted = 1 LIMIT 1")
            info["is_encrypted"] = cursor.fetchone() is not None

        if "chunks" in tables:
            cursor.execute("SELECT COUNT(*) FROM chunks")
            info["chunk_count"] = cursor.fetchone()[0] or 0

        if "graph_nodes" in tables:
            cursor.execute("SELECT COUNT(*) FROM graph_nodes")
            info["node_count"] = cursor.fetchone()[0] or 0

        if "graph_relationships" in tables:
            cursor.execute("SELECT COUNT(*) FROM graph_relationships")
            info["rel_count"] = cursor.fetchone()[0] or 0

        if "store_metadata" in tables:
            cursor.execute("SELECT key, value FROM store_metadata WHERE key IN ('store_name', 'store_description', 'vectorizer_info')")
            for k, v in cursor.fetchall():
                if k == 'store_name' and v: info["name"] = v
                elif k == 'store_description' and v: info["description"] = v
                elif k == 'vectorizer_info' and v:
                    try:
                        v_data = json.loads(v)
                        info["vectorizer_name"] = v_data.get("vectorizer_name") or v_data.get("name", "st")
                        cfg = v_data.get("vectorizer_config") or {}
                        info["vectorizer_model"] = cfg.get("model") or cfg.get("model_name", "")
                    except Exception:
                        pass

        conn.close()
        info["is_valid"] = True
    except Exception:
        info["is_valid"] = False

    return info


def discover_local_stores() -> List[Dict[str, Any]]:
    """Scans working directory and projects/ subfolder for SafeStore databases."""
    found_paths = set()
    stores = []

    # 1. Check projects directory
    if PROJECTS_DIR.exists():
        for p in sorted(PROJECTS_DIR.glob("*.db")):
            resolved = str(p.resolve())
            if resolved not in found_paths:
                found_paths.add(resolved)
                stores.append(inspect_store_summary(p))

    # 2. Check current working directory
    for p in sorted(Path(".").glob("*.db")):
        resolved = str(p.resolve())
        if resolved not in found_paths:
            found_paths.add(resolved)
            stores.append(inspect_store_summary(p))

    # Sort: valid stores with most documents/nodes first
    stores.sort(key=lambda s: (s["is_valid"], s["doc_count"] + s["node_count"]), reverse=True)
    return stores

# Inject vis-network into HTML head for interactive knowledge graph rendering
VIS_NETWORK_HEADER = """
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
window.safeStoreNetwork = null;
window.safeStoreNetworkData = { nodes: null, edges: null };
window.safeStoreOriginalNodeColors = {};

window.initSafeStoreGraph = function(containerId, nodesArray, edgesArray) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Guarantee minimum render height if container was previously hidden
    if (container.clientHeight === 0) {
        container.style.height = '560px';
    }

    window.safeStoreOriginalNodeColors = {};
    nodesArray.forEach(n => {
        window.safeStoreOriginalNodeColors[n.id] = n.color;
    });

    const nodes = new vis.DataSet(nodesArray);
    const edges = new vis.DataSet(edgesArray);
    window.safeStoreNetworkData = { nodes: nodes, edges: edges };

    const data = { nodes: nodes, edges: edges };
    const options = {
        nodes: {
            shape: 'dot',
            size: 18,
            font: { color: '#f1f5f9', size: 12, face: 'monospace' },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            color: { color: '#64748b', highlight: '#14b8a6', hover: '#06b6d4' },
            arrows: { to: { enabled: true, scaleFactor: 0.8 } },
            font: { color: '#94a3b8', size: 10, align: 'middle' },
            smooth: { type: 'continuous' }
        },
        physics: {
            stabilization: { iterations: 120 },
            barnesHut: { gravitationalConstant: -10000, centralGravity: 0.3, springLength: 95, springConstant: 0.04 }
        },
        interaction: { hover: true, tooltipDelay: 120 }
    };

    if (window.safeStoreNetwork) {
        window.safeStoreNetwork.destroy();
    }

    window.safeStoreNetwork = new vis.Network(container, data, options);

    // Auto-fit network once DOM reflow settles
    setTimeout(() => {
        if (window.safeStoreNetwork) {
            window.safeStoreNetwork.redraw();
            window.safeStoreNetwork.fit({ animation: { duration: 600 } });
        }
    }, 150);

    window.safeStoreNetwork.on('click', function(params) {
        if (params.nodes.length > 0) {
            container.dispatchEvent(new CustomEvent('safestore_node_clicked', { detail: { node_id: params.nodes[0] } }));
        } else if (params.edges.length > 0) {
            container.dispatchEvent(new CustomEvent('safestore_edge_clicked', { detail: { edge_id: params.edges[0] } }));
        } else {
            container.dispatchEvent(new CustomEvent('safestore_background_clicked', { detail: {} }));
        }
    });
};

window.highlightGraphNodes = function(nodeIds) {
    if (!window.safeStoreNetwork || !window.safeStoreNetworkData.nodes) return;
    const targetSet = new Set(nodeIds.map(Number));
    const updates = [];

    window.safeStoreNetworkData.nodes.forEach(n => {
        if (targetSet.has(Number(n.id))) {
            updates.push({
                id: n.id,
                size: 26,
                borderWidth: 4,
                color: { background: '#14b8a6', border: '#ffffff', highlight: '#06b6d4' },
                font: { color: '#ffffff', size: 14 }
            });
        } else {
            updates.push({
                id: n.id,
                size: 10,
                borderWidth: 1,
                color: { background: '#334155', border: '#1e293b' },
                font: { color: '#64748b', size: 9 }
            });
        }
    });

    window.safeStoreNetworkData.nodes.update(updates);
    if (nodeIds.length > 0) {
        window.safeStoreNetwork.fit({ nodes: nodeIds, animation: { duration: 800, easingFunction: 'easeInOutQuad' } });
    }
};

window.resetGraphHighlight = function() {
    if (!window.safeStoreNetwork || !window.safeStoreNetworkData.nodes) return;
    const updates = [];
    window.safeStoreNetworkData.nodes.forEach(n => {
        const orig = window.safeStoreOriginalNodeColors[n.id] || '#14b8a6';
        updates.push({
            id: n.id,
            size: 16,
            borderWidth: 2,
            color: orig,
            font: { color: '#f1f5f9', size: 12 }
        });
    });
    window.safeStoreNetworkData.nodes.update(updates);
    window.safeStoreNetwork.unselectAll();
    window.safeStoreNetwork.fit({ animation: { duration: 500 } });
};

window.focusGraphNode = function(nodeId) {
    if (!window.safeStoreNetwork) return;
    window.safeStoreNetwork.selectNodes([nodeId]);
    window.safeStoreNetwork.focus(nodeId, { scale: 1.4, animation: { duration: 600 } });
};

window.toggleGraphPhysics = function(enabled) {
    if (!window.safeStoreNetwork) return;
    window.safeStoreNetwork.setOptions({ physics: { enabled: enabled } });
};
</script>
"""


async def pick_file_dialog(
    title: str = "Select Document",
    file_types: Optional[List[Tuple[str, str]]] = None
) -> Optional[str]:
    """
    Opens the native operating system file dialog.
    Executes standard Tkinter dialog in an asyncio worker thread first to prevent
    multiprocessing pickling errors across pywebview WindowProxy queues.
    """
    def _pick_tk():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            types = file_types or [
                ("All Supported Files", "*.pdf;*.docx;*.txt;*.md;*.html;*.htm;*.csv;*.xlsx;*.xls;*.json;*.rst;*.pptx;*.py;*.sql"),
                ("All Files (*.*)", "*.*")
            ]

            selected = filedialog.askopenfilename(
                title=title,
                filetypes=types
            )
            root.destroy()
            return str(Path(selected).resolve()) if selected else None
        except Exception:
            return None

    tk_res = await asyncio.to_thread(_pick_tk)
    if tk_res:
        return tk_res

    # Fallback to pywebview native window using integer constant 10 (avoids pickling function object)
    try:
        if hasattr(app, 'native') and hasattr(app.native, 'main_window') and app.native.main_window:
            dialog_call = app.native.main_window.create_file_dialog(
                10,
                allow_multiple=False
            )
            if asyncio.iscoroutine(dialog_call):
                result = await dialog_call
            else:
                result = dialog_call

            if result and len(result) > 0:
                return str(Path(result[0]).resolve())
    except Exception:
        pass

    return None


class StudioState:
    def __init__(self):
        self.db_path: str = ""
        self.store: Optional[SafeStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.selected_doc_id: Optional[int] = None

    def close_current_store(self):
        """Cleanly releases any open store and unloads model weights."""
        if self.store:
            try:
                self.store.close()
            except Exception:
                pass
        self.store = None
        self.graph_store = None
        self.db_path = ""
        self.selected_doc_id = None

    def open_store(self, path: str):
        self.close_current_store()
        path_str = str(path).strip()
        if path_str.lower() in (":memory:", "memory") or path_str.endswith(":memory:"):
            self.db_path = ":memory:"
        elif path_str.lower() in (":tempfile:", "tempfile") or path_str.endswith(":tempfile:"):
            self.db_path = ":tempfile:"
        else:
            self.db_path = str(Path(path).resolve())
        self.store = SafeStore(db_path=self.db_path, log_level=LogLevel.INFO)
        try:
            self.graph_store = GraphStore(store=self.store)
        except Exception:
            self.graph_store = None
        self.selected_doc_id = None


async def render_studio_page(client: Client, initial_path: Optional[str] = None):
    state = StudioState()

    # Base styling and scripts
    ui.add_head_html(VIS_NETWORK_HEADER)
    ui.colors(primary='#14b8a6', secondary='#06b6d4', accent='#f59e0b', dark='#0f172a')
    ui.query('body').classes('bg-slate-900 text-slate-100 font-sans')

    # =========================================================================
    # HEADER / TOP NAVIGATION BAR
    # =========================================================================
    with ui.header().classes('bg-slate-800/95 backdrop-blur border-b border-slate-700 px-6 py-3 flex justify-between items-center z-50'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('hub', size='md').classes('text-teal-400')
            with ui.column().classes('gap-0'):
                ui.label('SafeStore Studio').classes('text-xl font-bold bg-gradient-to-r from-teal-400 to-cyan-400 bg-clip-text text-transparent leading-tight')
                db_label = ui.label('Projects Hub').classes('text-xs text-slate-400 font-mono')

        with ui.row().classes('items-center gap-2'):
            # Navigation between Projects View and Store Workspace
            btn_all_stores = ui.button('All Stores', icon='grid_view', on_click=lambda: show_projects_view()).props('outline color=teal size=sm')
            btn_all_stores.set_visibility(False)

            # Quick Create Store Button
            ui.button('New Store', icon='add', on_click=lambda: create_store_dialog.open()).props('color=teal size=sm')

            async def browse_and_open_external_db():
                selected = await pick_file_dialog(
                    title="Select SafeStore SQLite Database",
                    file_types=[
                        ("SafeStore SQLite DB (*.db, *.sqlite)", "*.db;*.sqlite;*.sqlite3"),
                        ("All Files (*.*)", "*.*")
                    ]
                )
                if selected:
                    await switch_to_store_workspace(selected)

            ui.button('Open DB File', icon='folder_open', on_click=browse_and_open_external_db).props('outline text-color=grey size=sm')

    # Global Loading Dialog
    loading_overlay = ui.dialog().props('persistent backdrop-filter="blur(4px)"')
    with loading_overlay, ui.card().classes('bg-slate-800/95 text-slate-100 border border-teal-500/50 p-8 items-center text-center space-y-4 shadow-2xl min-w-[22rem] rounded-2xl'):
        ui.spinner('dots', size='xl', color='teal').classes('my-1')
        loading_title = ui.label('Loading SafeStore...').classes('text-lg font-bold text-teal-300')
        loading_text = ui.label('Initializing vectorizer and database schema...').classes('text-xs text-slate-400 font-mono')

    # =========================================================================
    # DIALOG: CREATE NEW STORE
    # =========================================================================
    with ui.dialog() as create_store_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[42rem] border border-slate-700 space-y-4'):
        ui.label('Create New SafeStore Database').classes('text-xl font-bold text-teal-400')
        ui.label('Instantiate a sovereign vector and knowledge graph database with persistent configuration.').classes('text-xs text-slate-400')

        with ui.row().classes('w-full gap-4'):
            new_store_name = ui.input('Store Name', placeholder='e.g. Enterprise Knowledge Base').classes('flex-1').props('dark standout')
            new_store_filename = ui.input('Database Filename', placeholder='e.g. enterprise.db').classes('flex-1').props('dark standout')

        new_store_desc = ui.textarea('Description (Optional)', placeholder='Brief purpose of this knowledge store...').classes('w-full text-xs').props('dark standout rows=2')

        with ui.row().classes('w-full gap-4'):
            new_store_vec = ui.select(
                options={
                    'st': 'SentenceTransformers (Local)',
                    'tfidf': 'TF-IDF (Fast local keyword)',
                    'ollama': 'Ollama (Local daemon)',
                    'openai': 'OpenAI (API)',
                    'grepper': 'Grepper (Tree inverted index)'
                },
                value='st',
                label='Vectorizer Engine'
            ).classes('flex-1').props('dark standout')

            new_store_model = ui.input('Model Name', value='all-MiniLM-L6-v2').classes('flex-1').props('dark standout')

        with ui.row().classes('w-full gap-4 items-center'):
            new_store_shared = ui.checkbox('Use Shared Model Server Daemon (Zero-OOM)', value=True).classes('text-xs text-teal-300')
            new_store_shared.bind_visibility_from(new_store_vec, 'value', value='st')

            new_store_pw = ui.input('Encryption Password (Optional)', placeholder='Leave blank for plaintext').classes('flex-1').props('dark standout type=password')

        async def execute_create_store():
            s_name = new_store_name.value.strip() or "New Store"
            fname = new_store_filename.value.strip()
            if not fname:
                fname = s_name.lower().replace(" ", "_") + ".db"
            if not fname.endswith(".db"):
                fname += ".db"

            target_path = PROJECTS_DIR / fname

            vec_name = new_store_vec.value
            model_name = new_store_model.value.strip() or "all-MiniLM-L6-v2"
            shared_server = bool(new_store_shared.value) if vec_name == 'st' else False
            enc_key = new_store_pw.value.strip() or None

            vec_config = {}
            if vec_name == 'st':
                vec_config = {"model_name": model_name, "use_shared_server": shared_server}
            elif vec_name in ('ollama', 'openai'):
                vec_config = {"model": model_name}

            create_store_dialog.close()
            loading_title.text = f"Creating '{s_name}'"
            loading_text.text = f"Initializing vectorizer '{vec_name}' and database schema..."
            loading_overlay.open()
            await asyncio.sleep(0.05)

            try:
                def _create():
                    st = SafeStore(
                        db_path=target_path,
                        name=s_name,
                        description=new_store_desc.value.strip() or None,
                        vectorizer_name=vec_name,
                        vectorizer_config=vec_config,
                        encryption_key=enc_key,
                        log_level=LogLevel.INFO
                    )
                    st.close()

                await asyncio.to_thread(_create)
                ui.notify(f"Created store '{s_name}' ({fname}) successfully!", color='positive')
                await switch_to_store_workspace(str(target_path.resolve()))
            except Exception as ex:
                ui.notify(f"Failed to create store: {ex}", color='negative', duration=7000)
            finally:
                loading_overlay.close()

        with ui.row().classes('w-full justify-end gap-2 mt-4'):
            ui.button('Cancel', on_click=create_store_dialog.close).props('flat text-color=grey')
            ui.button('Create & Open Store', icon='check', on_click=execute_create_store).props('color=teal')

    # =========================================================================
    # DIALOG: EDIT STORE PROPERTIES
    # =========================================================================
    with ui.dialog() as edit_store_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[36rem] border border-slate-700 space-y-4'):
        ui.label('Edit Store Properties').classes('text-xl font-bold text-teal-400')
        edit_target_path = ui.input('Database Path').classes('w-full hidden')
        edit_store_name = ui.input('Store Name').classes('w-full').props('dark standout')
        edit_store_desc = ui.textarea('Description').classes('w-full text-xs').props('dark standout rows=3')

        async def save_edited_store():
            t_path = edit_target_path.value
            new_n = edit_store_name.value.strip()
            new_d = edit_store_desc.value.strip()
            edit_store_dialog.close()

            try:
                def _update():
                    temp_store = SafeStore(db_path=t_path, log_level=LogLevel.WARNING)
                    temp_store.update_properties({"name": new_n, "description": new_d})
                    temp_store.close()

                await asyncio.to_thread(_update)
                ui.notify(f"Updated properties for '{new_n}'", color='positive')
                refresh_projects_cards()
            except Exception as ex:
                ui.notify(f"Error updating store: {ex}", color='negative')

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=edit_store_dialog.close).props('flat text-color=grey')
            ui.button('Save Changes', icon='save', on_click=save_edited_store).props('color=teal')

    # =========================================================================
    # DIALOG: CONFIRM DELETE STORE
    # =========================================================================
    with ui.dialog() as delete_store_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[32rem] border border-rose-500/50 space-y-4'):
        ui.label('Delete Store Database').classes('text-xl font-bold text-rose-400')
        delete_store_msg = ui.label('').classes('text-xs text-slate-300')
        delete_target_path = ui.input().classes('hidden')

        async def execute_delete_store():
            del_path = Path(delete_target_path.value)
            delete_store_dialog.close()

            # Close if currently opened
            if state.db_path and Path(state.db_path).resolve() == del_path.resolve():
                state.open_store(":memory:")

            try:
                for ext in ["", ".lock", "-wal", "-shm"]:
                    p = Path(f"{del_path}{ext}")
                    p.unlink(missing_ok=True)
                ui.notify(f"Deleted database '{del_path.name}'", color='positive')
                refresh_projects_cards()
            except Exception as ex:
                ui.notify(f"Error deleting file: {ex}", color='negative')

        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=delete_store_dialog.close).props('flat text-color=grey')
            ui.button('Permanently Delete', icon='delete_forever', on_click=execute_delete_store).props('color=negative')

    # =========================================================================
    # VIEW 1: PROJECTS & STORES CARDS HUB (LANDING PAGE)
    # =========================================================================
    projects_container = ui.column().classes('w-full p-8 max-w-7xl mx-auto space-y-6')

    with projects_container:
        with ui.row().classes('w-full justify-between items-end border-b border-slate-800 pb-4'):
            with ui.column().classes('gap-1'):
                ui.label('Stores & Knowledge Bases').classes('text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400')
                ui.label('Select an existing SafeStore to open or create a new sovereign vector and graph database.').classes('text-sm text-slate-400')

            with ui.row().classes('items-center gap-3'):
                project_search = ui.input(placeholder='Search stores...').classes('w-64 text-sm').props('dark standout dense')
                ui.button('Scan Folder', icon='refresh', on_click=lambda: refresh_projects_cards()).props('outline color=teal size=sm')

        projects_grid = ui.element('div').classes('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 w-full')

    # =========================================================================
    # VIEW 2: STORE WORKSPACE (THE 5 TABS)
    # =========================================================================
    workspace_container = ui.column().classes('w-full')
    workspace_container.set_visibility(False)

    with workspace_container:
        # Tab Navigation
        with ui.tabs().classes('w-full bg-slate-800 border-b border-slate-700 text-slate-300') as tabs:
            tab_files = ui.tab('Files & Documents', icon='description')
            tab_datalake = ui.tab('Semantic Datalake', icon='scatter_plot')
            tab_graph = ui.tab('Knowledge Graph & SPARQL', icon='share')
            tab_search = ui.tab('RAG Search Studio', icon='search')
            tab_diagnostics = ui.tab('Database Diagnostics', icon='analytics')

        with ui.tab_panels(tabs, value=tab_files).classes('w-full p-6 bg-slate-900 text-slate-100'):

            # -----------------------------------------------------------------
            # TAB 1: Files & Documents
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_files):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    ui.label('Document Repository').classes('text-2xl font-bold text-teal-400')

                    async def browse_doc_file():
                        selected = await pick_file_dialog(
                            title="Select Document to Index into SafeStore",
                            file_types=[
                                ("Supported Documents", "*.pdf;*.docx;*.txt;*.md;*.html;*.htm;*.csv;*.xlsx;*.xls;*.json;*.rst;*.pptx;*.py;*.sql"),
                                ("PDF Files (*.pdf)", "*.pdf"),
                                ("Word Files (*.docx)", "*.docx"),
                                ("Text / Markdown (*.txt, *.md)", "*.txt;*.md;*.rst"),
                                ("Tabular Data (*.csv, *.xlsx, *.json)", "*.csv;*.xlsx;*.xls;*.json"),
                                ("All Files (*.*)", "*.*")
                            ]
                        )
                        if selected:
                            upload_input.value = selected

                    async def handle_drag_upload(e):
                        temp_dir = Path("temp_uploads")
                        temp_dir.mkdir(exist_ok=True)
                        dest_path = temp_dir / e.name
                        with open(dest_path, "wb") as f:
                            f.write(e.content.read())
                        upload_input.value = str(dest_path.resolve())
                        ui.notify(f"Loaded '{e.name}'. Click 'Add & Index' to process.", color='info')

                    async def add_custom_file():
                        input_file = upload_input.value.strip()
                        if not input_file or not Path(input_file).exists():
                            ui.notify("File does not exist on disk", color='negative')
                            return
                        add_dialog.close()
                        loading_overlay.set_visibility(True)
                        try:
                            res = await asyncio.to_thread(state.store.add_document, input_file)
                            ui.notify(f"Indexed '{Path(input_file).name}': {res['num_chunks_added']} chunks added", color='positive')
                            refresh_files_view()
                        except Exception as e:
                            ui.notify(f"Error adding file: {e}", color='negative')
                        finally:
                            loading_overlay.set_visibility(False)

                    with ui.dialog() as add_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[36rem] border border-slate-700 space-y-4'):
                        ui.label('Add Document to SafeStore').classes('text-xl font-bold text-teal-400')
                        ui.label('Select any PDF, Word DOCX, Markdown, Text, or CSV file to chunk and vectorize.').classes('text-xs text-slate-400')

                        with ui.row().classes('w-full items-center gap-2'):
                            upload_input = ui.input('Selected File Path').classes('flex-1').props('dark standout')
                            ui.button('Browse File...', icon='folder_open', on_click=browse_doc_file).props('color=teal')

                        ui.separator().classes('my-2 border-slate-700')
                        ui.label('Or Drag & Drop File:').classes('text-xs text-slate-400 font-semibold mb-1')
                        ui.upload(on_upload=handle_drag_upload, auto_upload=True, max_files=1).props('dark flat bordered').classes('w-full')

                        with ui.row().classes('w-full justify-end gap-2 mt-4'):
                            ui.button('Cancel', on_click=add_dialog.close).props('flat text-color=grey')
                            ui.button('Add & Index Document', icon='cloud_upload', on_click=add_custom_file).props('color=teal')

                    ui.button('Add Document', icon='add', on_click=add_dialog.open).props('color=teal')

                with ui.row().classes('w-full gap-6 items-start'):
                    doc_table_container = ui.column().classes('w-1/2 bg-slate-800/80 p-4 rounded-xl border border-slate-700')
                    doc_details_container = ui.column().classes('w-1/2 bg-slate-800/80 p-4 rounded-xl border border-slate-700')

            # -----------------------------------------------------------------
            # TAB 2: Semantic Datalake (UMAP / PCA / t-SNE)
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_datalake):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    with ui.row().classes('items-center gap-3'):
                        ui.label('State-of-the-Art Semantic Datalake Explorer').classes('text-2xl font-bold text-teal-400')
                        datalake_method_select = ui.select(['umap', 'pca', 'tsne'], value='umap', label='Method').classes('w-32').props('dark dense standout')
                        datalake_dim_select = ui.select([2, 3], value=2, label='Dimensions').classes('w-32').props('dark dense standout')
                        ui.button('Project', icon='refresh', on_click=lambda: refresh_datalake_view()).props('color=teal size=sm')

                with ui.row().classes('w-full gap-6 items-start'):
                    datalake_plot_container = ui.column().classes('w-3/4 bg-slate-800/80 p-4 rounded-xl border border-slate-700 h-[70vh]')
                    datalake_inspector_container = ui.column().classes('w-1/4 bg-slate-800/80 p-4 rounded-xl border border-slate-700 h-[70vh] overflow-y-auto')

            # -----------------------------------------------------------------
            # TAB 3: Knowledge Graph Visual Studio & W3C SPARQL 1.1 Console
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_graph):
                # Header Bar
                with ui.row().classes('w-full justify-between items-center mb-3'):
                    with ui.row().classes('items-center gap-3'):
                        ui.label('Knowledge Graph Studio').classes('text-2xl font-bold text-teal-400')
                        graph_summary_badge = ui.badge('Loading...', color='slate-700').classes('text-xs font-mono')

                    with ui.row().classes('items-center gap-2'):
                        # Manual Node Dialog
                        with ui.dialog() as add_node_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[32rem] border border-slate-700 space-y-4'):
                            ui.label('Add Graph Node Manually').classes('text-lg font-bold text-teal-400')
                            node_label_input = ui.input('Entity Label / Class', placeholder='e.g. Person, Concept, Tool, API').classes('w-full').props('dark standout')
                            node_name_input = ui.input('Canonical Name / Identifying Value', placeholder='e.g. SafeStore, Alice, LollmsClient').classes('w-full').props('dark standout')
                            node_props_input = ui.textarea('Properties (JSON)', value='{}').classes('w-full font-mono text-xs').props('dark standout rows=4')

                            async def save_manual_node():
                                lbl = node_label_input.value.strip()
                                name = node_name_input.value.strip()
                                if not lbl or not name:
                                    ui.notify("Label and Canonical Name are required.", color='warning')
                                    return
                                try:
                                    props = json.loads(node_props_input.value.strip() or '{}')
                                except Exception:
                                    ui.notify("Properties must be valid JSON.", color='negative')
                                    return
                                props["identifying_value"] = name
                                props["name"] = name
                                try:
                                    node_id = await asyncio.to_thread(state.graph_store.add_node, lbl, props)
                                    ui.notify(f"Node created with ID {node_id}", color='positive')
                                    add_node_dialog.close()
                                    await refresh_graph_view()
                                except Exception as err:
                                    ui.notify(f"Error adding node: {err}", color='negative')

                            with ui.row().classes('w-full justify-end gap-2'):
                                ui.button('Cancel', on_click=add_node_dialog.close).props('flat text-color=grey')
                                ui.button('Create Node', icon='add', on_click=save_manual_node).props('color=teal')

                        # Manual Relationship Dialog
                        with ui.dialog() as add_rel_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[36rem] border border-slate-700 space-y-4'):
                            ui.label('Add Directed Relationship').classes('text-lg font-bold text-teal-400')
                            rel_src_select = ui.select({}, label='Source Node').classes('w-full').props('dark standout')
                            rel_tgt_select = ui.select({}, label='Target Node').classes('w-full').props('dark standout')
                            rel_type_input = ui.input('Relationship Type', placeholder='e.g. USES, DEPENDS_ON, CREATED_BY').classes('w-full').props('dark standout')
                            rel_props_input = ui.textarea('Properties (JSON)', value='{}').classes('w-full font-mono text-xs').props('dark standout rows=3')

                            async def save_manual_rel():
                                src_id = rel_src_select.value
                                tgt_id = rel_tgt_select.value
                                rtype = rel_type_input.value.strip().upper()
                                if not src_id or not tgt_id or not rtype:
                                    ui.notify("Source, Target, and Relationship Type are required.", color='warning')
                                    return
                                try:
                                    props = json.loads(rel_props_input.value.strip() or '{}')
                                except Exception:
                                    ui.notify("Properties must be valid JSON.", color='negative')
                                    return
                                try:
                                    rel_id = await asyncio.to_thread(state.graph_store.add_relationship, int(src_id), int(tgt_id), rtype, props)
                                    ui.notify(f"Relationship created with ID {rel_id}", color='positive')
                                    add_rel_dialog.close()
                                    await refresh_graph_view()
                                except Exception as err:
                                    ui.notify(f"Error adding relationship: {err}", color='negative')

                            with ui.row().classes('w-full justify-end gap-2'):
                                ui.button('Cancel', on_click=add_rel_dialog.close).props('flat text-color=grey')
                                ui.button('Create Relationship', icon='link', on_click=save_manual_rel).props('color=teal')

                        # Fast Graph Builder Configuration Modal
                        with ui.dialog() as build_graph_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[38rem] border border-slate-700 space-y-4'):
                            ui.label('Fast Knowledge Graph Builder').classes('text-xl font-bold text-teal-400')
                            ui.label('Leverage modern high-context LLMs to extract entire documents in 1 single pass.').classes('text-xs text-slate-400')

                            mode_select = ui.select(
                                options={
                                    'document': 'Entire Document (Fastest - 1 call per doc, captures cross-chunk context)',
                                    'batch_chunks': 'Grouped Chunks (Batched - e.g. 5 chunks per LLM call)',
                                    'chunk': 'Individual Chunks (Granular - 1 call per chunk)'
                                },
                                value='document',
                                label='Extraction Mode'
                            ).classes('w-full').props('dark standout')

                            batch_size_slider = ui.slider(min=2, max=30, value=5).classes('w-full')
                            ui.label().bind_text_from(batch_size_slider, 'value', backward=lambda v: f'Chunks per Batch: {v}').classes('text-xs text-slate-400')
                            batch_size_slider.bind_visibility_from(mode_select, 'value', value='batch_chunks')

                            guidance_input = ui.textarea(
                                label='Focus Guidance / Extraction Directives',
                                placeholder='e.g. Focus on software architecture, dependencies, modules, APIs, and team owners.'
                            ).classes('w-full text-xs font-mono').props('dark standout rows=3')

                            build_status_label = ui.label('').classes('text-xs text-teal-300 font-mono')
                            build_progress_bar = ui.linear_progress(value=0.0).classes('w-full')
                            build_progress_bar.set_visibility(False)

                            async def run_fast_graph_build():
                                if not state.graph_store:
                                    ui.notify("GraphStore not initialized on this store.", color='negative')
                                    return

                                chosen_mode = mode_select.value
                                chosen_batch = int(batch_size_slider.value)
                                chosen_guidance = guidance_input.value.strip() or None

                                build_progress_bar.set_visibility(True)
                                build_progress_bar.value = 0.0
                                build_status_label.text = "Initializing extraction pipeline..."
                                start_btn.props('disabled')

                                def _ui_progress(fraction, message):
                                    build_progress_bar.value = fraction
                                    build_status_label.text = message

                                try:
                                    ui.notify(f"Extracting graph using '{chosen_mode.upper()}' mode...", color='info')
                                    stats = await asyncio.to_thread(
                                        state.graph_store.build_graph_for_all_documents,
                                        mode=chosen_mode,
                                        chunks_per_batch=chosen_batch,
                                        guidance=chosen_guidance,
                                        progress_callback=_ui_progress
                                    )
                                    ui.notify(
                                        f"Graph Ready: {stats['nodes_created']} nodes, {stats['relationships_created']} edges created!",
                                        color='positive',
                                        duration=5000
                                    )
                                    build_graph_dialog.close()
                                    await refresh_graph_view()
                                except Exception as e:
                                    ui.notify(f"Graph extraction error: {e}", color='negative', duration=7000)
                                    build_status_label.text = f"Error: {e}"
                                finally:
                                    start_btn.props(remove='disabled')
                                    build_progress_bar.set_visibility(False)

                            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                                ui.button('Cancel', on_click=build_graph_dialog.close).props('flat text-color=grey')
                                start_btn = ui.button('Start Fast Extraction', icon='bolt', on_click=run_fast_graph_build).props('color=teal')

                        ui.button('Add Node', icon='add', on_click=add_node_dialog.open).props('outline color=teal size=sm')

                        async def open_rel_dialog():
                            if not state.graph_store: return
                            nodes = await asyncio.to_thread(state.graph_store.get_all_nodes, 500)
                            opts = {n['node_id']: f"[{n['label']}] {n['properties'].get('name') or n['properties'].get('identifying_value') or n['node_id']}" for n in nodes}
                            rel_src_select.options = opts
                            rel_tgt_select.options = opts
                            if opts:
                                rel_src_select.value = list(opts.keys())[0]
                                rel_tgt_select.value = list(opts.keys())[-1]
                            add_rel_dialog.open()

                        ui.button('Add Relationship', icon='link', on_click=open_rel_dialog).props('outline color=cyan size=sm')
                        ui.button('Build Knowledge Graph', icon='bolt', on_click=build_graph_dialog.open).props('color=teal size=sm')

                # =============================================================
                # 3-PANEL RESPONSIVE WORKSPACE
                # Left Sidebar (Tools) | Center (Hero Graph Canvas) | Right Sidebar (SPARQL & Inspector)
                # =============================================================
                with ui.row().classes('w-full gap-4 items-stretch h-[76vh]'):

                    # ---------------------------------------------------------
                    # PANEL 1: LEFT SIDEBAR (Filters, Metrics, Quick Tools)
                    # ---------------------------------------------------------
                    with ui.card().classes('w-72 bg-slate-800/90 p-4 rounded-xl border border-slate-700 flex flex-col justify-between overflow-y-auto space-y-4'):
                        with ui.column().classes('w-full gap-3'):
                            ui.label('Graph Diagnostics').classes('text-sm font-bold text-teal-300 uppercase tracking-wider')
                            left_metrics_box = ui.column().classes('w-full gap-2 text-xs font-mono text-slate-300 bg-slate-900/60 p-3 rounded-lg border border-slate-700/60')

                            ui.separator().classes('border-slate-700')
                            ui.label('Search Entities').classes('text-sm font-bold text-teal-300 uppercase tracking-wider')
                            node_search_input = ui.input(placeholder='Search node name...').classes('w-full text-xs').props('dark dense standout')

                            async def search_and_focus_node():
                                val = node_search_input.value.strip().lower()
                                if not val or not state.graph_store: return
                                nodes = await asyncio.to_thread(state.graph_store.get_all_nodes, 1000)
                                for n in nodes:
                                    candidate_name = str(n['properties'].get('name') or n['properties'].get('identifying_value') or '').lower()
                                    if val in candidate_name or val in n['label'].lower():
                                        await ui.run_javascript(f"window.focusGraphNode({n['node_id']})")
                                        show_node_inspector(n['node_id'])
                                        ui.notify(f"Focused on [{n['label']}] {candidate_name}", color='info')
                                        return
                                ui.notify(f"No node matching '{val}' found.", color='warning')

                            node_search_input.on('keydown.enter', search_and_focus_node)
                            ui.button('Find & Focus', icon='search', on_click=search_and_focus_node).props('color=teal size=xs class="w-full"')

                            ui.separator().classes('border-slate-700')
                            ui.label('Entity Classes').classes('text-sm font-bold text-teal-300 uppercase tracking-wider')
                            node_types_filter_container = ui.column().classes('w-full gap-1 max-h-48 overflow-y-auto text-xs font-mono')

                        with ui.column().classes('w-full pt-3 border-t border-slate-700 gap-2'):
                            ui.button('Refresh All Graph Data', icon='refresh', on_click=lambda: asyncio.create_task(refresh_graph_view())).props('flat text-color=teal size=sm class="w-full"')

                    # ---------------------------------------------------------
                    # PANEL 2: CENTER HERO VISUAL GRAPH CANVAS
                    # ---------------------------------------------------------
                    with ui.card().classes('flex-1 bg-slate-900 p-0 rounded-xl border border-slate-700 relative overflow-hidden flex flex-col min-h-[560px]'):
                        # Canvas Header Toolbar
                        with ui.row().classes('absolute top-3 left-4 right-4 justify-between items-center z-20 pointer-events-none'):
                            with ui.row().classes('items-center gap-2 pointer-events-auto bg-slate-800/90 backdrop-blur px-3 py-1.5 rounded-lg border border-slate-700 shadow-lg'):
                                ui.icon('schema', size='sm').classes('text-teal-400')
                                ui.label('Interactive Physics Canvas').classes('text-xs font-bold text-slate-200 uppercase tracking-wider')

                            with ui.row().classes('items-center gap-2 pointer-events-auto bg-slate-800/90 backdrop-blur px-2 py-1 rounded-lg border border-slate-700 shadow-lg'):
                                ui.button(icon='filter_center_focus', on_click=lambda: ui.run_javascript('window.safeStoreNetwork && window.safeStoreNetwork.fit({animation: true})')).props('flat dense size=sm title="Fit Entire Network"')
                                physics_toggle = ui.switch('Physics', value=True).props('dark dense').classes('text-xs text-slate-300')
                                physics_toggle.on('update:model-value', lambda e: ui.run_javascript(f'window.toggleGraphPhysics({str(e.args).lower()})'))
                                ui.button('Reset Colors', icon='restart_alt', on_click=lambda: ui.run_javascript('window.resetGraphHighlight()')).props('flat dense text-color=teal size=sm title="Reset Selection"')

                        # The HTML div where vis-network mounts with explicit height
                        graph_canvas_div = ui.element('div').classes('w-full flex-1').style('width: 100%; height: 100%; min-height: 560px;').props('id=graph-network-container')

                        # Attach custom DOM event handlers
                        def on_canvas_node_clicked(e):
                            nid = e.args.get('node_id')
                            if nid is not None:
                                show_node_inspector(int(nid))

                        def on_canvas_edge_clicked(e):
                            rid = e.args.get('edge_id')
                            if rid is not None:
                                show_edge_inspector(int(rid))

                        graph_canvas_div.on('safestore_node_clicked', on_canvas_node_clicked)
                        graph_canvas_div.on('safestore_edge_clicked', on_canvas_edge_clicked)

                    # ---------------------------------------------------------
                    # PANEL 3: RIGHT SIDEBAR (SPARQL Runner & Inspector)
                    # ---------------------------------------------------------
                    with ui.card().classes('w-96 bg-slate-800/90 p-4 rounded-xl border border-slate-700 flex flex-col justify-between overflow-y-auto space-y-4'):
                        with ui.tabs().classes('w-full bg-slate-900/80 rounded-lg text-slate-300 text-xs') as right_tabs:
                            rtab_sparql = ui.tab('SPARQL 1.1', icon='terminal')
                            rtab_inspector = ui.tab('Inspector', icon='info')

                        with ui.tab_panels(right_tabs, value=rtab_sparql).classes('w-full bg-transparent p-0 flex-1'):

                            # SPARQL Panel
                            with ui.tab_panel(rtab_sparql).classes('p-0 space-y-3'):
                                with ui.row().classes('w-full justify-between items-center'):
                                    ui.label('Query & Subselection').classes('text-xs font-bold text-teal-300 uppercase tracking-wider')
                                    sparql_presets = ui.select(
                                        options={
                                            'all': 'Preset: 15 Triples',
                                            'types': 'Preset: Nodes by Class',
                                            'connected': 'Preset: Connected Pairs'
                                        },
                                        value='all',
                                        label='Load Preset'
                                    ).classes('w-36 text-xs').props('dark dense standout')

                                # AI SPARQL Generator Bar
                                with ui.card().classes('w-full bg-slate-900/90 p-2.5 rounded-lg border border-purple-500/40 space-y-1.5'):
                                    with ui.row().classes('w-full items-center justify-between'):
                                        with ui.row().classes('items-center gap-1'):
                                            ui.icon('psychology', size='xs').classes('text-purple-400')
                                            ui.label('AI SPARQL Generator (LOLLMS)').classes('text-[11px] font-bold text-purple-300 uppercase tracking-wider')

                                    ai_prompt_input = ui.input(placeholder="e.g. 'Find all tools that depend on other modules'").classes('w-full text-xs font-sans').props('dark dense standout')

                                    async def generate_sparql_with_ai():
                                        user_q = (ai_prompt_input.value or '').strip()
                                        if not user_q:
                                            ui.notify("Please enter a question to generate SPARQL.", color='warning')
                                            return
                                        if not state.graph_store:
                                            ui.notify("Knowledge Graph not available for this store.", color='negative')
                                            return

                                        gen_btn.props('disabled')
                                        try:
                                            ui.notify("Synthesizing W3C SPARQL 1.1 query with LOLLMS...", color='info')
                                            sparql_code = await asyncio.to_thread(state.graph_store.generate_sparql, user_q)
                                            sparql_editor.value = sparql_code
                                            ui.notify("SPARQL query generated! Click 'Execute & Highlight' to run.", color='positive')
                                        except Exception as err:
                                            ui.notify(f"SPARQL Generation failed: {err}", color='negative', duration=6000)
                                        finally:
                                            gen_btn.props(remove='disabled')

                                    ai_prompt_input.on('keydown.enter', generate_sparql_with_ai)
                                    with ui.row().classes('w-full justify-end'):
                                        gen_btn = ui.button('Generate SPARQL', icon='auto_awesome', on_click=generate_sparql_with_ai).props('color=purple size=xs')

                                sparql_editor = ui.textarea(
                                    label='SPARQL 1.1 Query',
                                    value="""PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
} LIMIT 15"""
                                ).classes('w-full font-mono text-xs').props('dark standout rows=5')

                                def on_preset_change(e):
                                    preset = e.value
                                    if preset == 'all':
                                        sparql_editor.value = """PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
} LIMIT 15"""
                                    elif preset == 'types':
                                        sparql_editor.value = """PREFIX ont: <http://example.org/ontology/>
SELECT ?entity ?label WHERE {
    ?entity ont:label ?label .
} LIMIT 20"""
                                    elif preset == 'connected':
                                        sparql_editor.value = """PREFIX ex: <http://example.org/>
SELECT ?source ?relation ?target WHERE {
    ?source ?relation ?target .
    FILTER(!CONTAINS(STR(?relation), "type") && !CONTAINS(STR(?relation), "label"))
} LIMIT 20"""

                                sparql_presets.on('update:model-value', on_preset_change)

                                async def execute_sparql_and_highlight():
                                    query_str = sparql_editor.value.strip()
                                    sparql_results_drawer.clear()
                                    with sparql_results_drawer:
                                        ui.spinner('dots', size='sm', color='teal')
                                        ui.label('Running SPARQL query...').classes('text-xs text-slate-400')

                                    try:
                                        res = await asyncio.to_thread(state.graph_store.query_sparql, query_str)
                                        sparql_results_drawer.clear()

                                        if "results" in res and "bindings" in res["results"]:
                                            bindings = res["results"]["bindings"]
                                            vars_list = res.get("head", {}).get("vars", [])

                                            # Match node IDs for visual highlighting on canvas
                                            nodes = await asyncio.to_thread(state.graph_store.get_all_nodes, 1000)
                                            name_to_id = {}
                                            for n in nodes:
                                                name_to_id[str(n['node_id'])] = n['node_id']
                                                name_to_id[f"http://example.org/node/{n['node_id']}"] = n['node_id']
                                                c_name = str(n['properties'].get('name') or n['properties'].get('identifying_value') or '')
                                                if c_name:
                                                    name_to_id[c_name.lower()] = n['node_id']
                                                    name_to_id[f"http://example.org/{c_name}".lower()] = n['node_id']

                                            matched_node_ids = set()
                                            for b in bindings:
                                                for var in vars_list:
                                                    val = str(b.get(var, {}).get('value', ''))
                                                    clean_val = val.lower().split('/')[-1]
                                                    if val in name_to_id:
                                                        matched_node_ids.add(name_to_id[val])
                                                    elif clean_val in name_to_id:
                                                        matched_node_ids.add(name_to_id[clean_val])

                                            # Visually highlight matching nodes on canvas!
                                            if matched_node_ids:
                                                await ui.run_javascript(f"window.highlightGraphNodes({json.dumps(list(matched_node_ids))})")

                                            with sparql_results_drawer:
                                                with ui.row().classes('w-full justify-between items-center mb-1'):
                                                    ui.badge(f"{len(bindings)} row(s) returned", color='slate-700')
                                                    if matched_node_ids:
                                                        ui.badge(f"Highlighted {len(matched_node_ids)} nodes", color='teal')

                                                if not bindings:
                                                    ui.label('Empty result set.').classes('text-xs text-slate-400 italic')
                                                else:
                                                    cols = [{'name': v, 'label': v, 'field': v, 'align': 'left'} for v in vars_list]
                                                    rows = [{v: b.get(v, {}).get('value', '').split('/')[-1] for v in vars_list} for b in bindings]
                                                    ui.table(columns=cols, rows=rows).classes('w-full').props('dark flat dense wrap-cells')

                                        elif "boolean" in res:
                                            sparql_results_drawer.clear()
                                            with sparql_results_drawer:
                                                ui.badge(f"ASK Result: {res['boolean']}", color='positive' if res['boolean'] else 'negative').classes('text-sm p-2')
                                        elif "triples" in res:
                                            triples = res["triples"]
                                            sparql_results_drawer.clear()
                                            with sparql_results_drawer:
                                                ui.label(f"Constructed {len(triples)} triple(s):").classes('text-xs text-slate-300 font-semibold mb-1')
                                                with ui.scroll_area().classes('h-40 font-mono text-xs'):
                                                    for t in triples:
                                                        ui.label(f"({t['subject']['value'].split('/')[-1]}) --[{t['predicate']['value'].split('/')[-1]}]--> ({t['object']['value'].split('/')[-1]})").classes('text-teal-300')
                                    except Exception as ex:
                                        sparql_results_drawer.clear()
                                        with sparql_results_drawer:
                                            ui.label(f"SPARQL Error: {ex}").classes('text-xs text-rose-400 font-mono')

                                with ui.row().classes('w-full justify-between items-center'):
                                    ui.button('Execute & Highlight', icon='play_arrow', on_click=execute_sparql_and_highlight).props('color=teal size=sm class="flex-1"')
                                    ui.button(icon='restart_alt', on_click=lambda: ui.run_javascript('window.resetGraphHighlight()')).props('outline text-color=grey size=sm title="Reset Highlight"')

                                sparql_results_drawer = ui.column().classes('w-full max-h-48 overflow-y-auto bg-slate-900/80 p-2 rounded-lg border border-slate-700/60')

                            # Inspector Panel
                            with ui.tab_panel(rtab_inspector).classes('p-0 space-y-3'):
                                ui.label('Element Details & Grounding').classes('text-xs font-bold text-teal-300 uppercase tracking-wider')
                                inspector_drawer = ui.column().classes('w-full space-y-2 text-xs font-mono')
                                with inspector_drawer:
                                    ui.label('Click any node or relationship on the canvas to inspect properties and source text chunks.').classes('text-slate-500 italic')

                # Element Inspection Helper Functions
                def show_node_inspector(node_id: int):
                    right_tabs.value = rtab_inspector
                    inspector_drawer.clear()
                    node = state.graph_store.get_node_details(node_id)
                    if not node: return

                    # Fetch linked chunk provenance
                    chunk_map = db.get_chunk_ids_for_nodes_db(state.store.conn, [node_id])
                    chunk_ids = chunk_map.get(node_id, [])
                    chunk_details = db.get_chunk_details_db(state.store.conn, chunk_ids, state.store.encryptor) if chunk_ids else []

                    # Fetch adjacent relationships
                    outgoing = state.graph_store.find_neighbors(node_id, direction='outgoing', limit=15)
                    incoming = state.graph_store.find_neighbors(node_id, direction='incoming', limit=15)

                    with inspector_drawer:
                        with ui.row().classes('w-full justify-between items-center'):
                            ui.badge(f"ID #{node['node_id']}", color='slate-700')
                            ui.badge(node['label'], color='teal').classes('font-bold')

                        name = node['properties'].get('name') or node['properties'].get('identifying_value') or node['label']
                        ui.label(name).classes('text-sm font-bold text-teal-300 break-words')

                        ui.label('Attributes:').classes('text-xs text-slate-400 font-bold mt-2')
                        with ui.column().classes('w-full bg-slate-900/80 p-2 rounded border border-slate-700/60 gap-1'):
                            for k, v in node['properties'].items():
                                if k not in ("name", "identifying_value", "other_identifiers"):
                                    ui.label(f"• {k}: {v}").classes('text-xs text-slate-300 break-words')

                        if outgoing:
                            ui.label(f'Outgoing Connections ({len(outgoing)}):').classes('text-xs text-slate-400 font-bold mt-2')
                            with ui.column().classes('w-full gap-1'):
                                for out_n in outgoing[:5]:
                                    o_name = out_n['properties'].get('name') or out_n['properties'].get('identifying_value') or out_n['label']
                                    ui.label(f"→ [{out_n['label']}] {o_name}").classes('text-xs text-cyan-300')

                        if chunk_details:
                            ui.label(f'Grounded Evidence ({len(chunk_details)} chunks):').classes('text-xs text-slate-400 font-bold mt-2')
                            with ui.scroll_area().classes('w-full max-h-36 bg-slate-900/80 p-2 rounded border border-slate-700/60'):
                                for c in chunk_details[:3]:
                                    ui.label(f"[Chunk #{c['chunk_id']} from {Path(c['file_path']).name}]:").classes('text-teal-400 font-bold text-[10px]')
                                    ui.label(c['chunk_text'][:180] + '...').classes('text-slate-300 text-[11px] mb-2')

                        async def delete_inspected_node():
                            try:
                                await asyncio.to_thread(state.graph_store.delete_node, node_id)
                                ui.notify(f"Deleted node ID {node_id}", color='positive')
                                inspector_drawer.clear()
                                await refresh_graph_view()
                            except Exception as ex:
                                ui.notify(f"Error deleting node: {ex}", color='negative')

                        ui.button('Delete Node', icon='delete', on_click=delete_inspected_node).props('flat text-color=red size=xs class="w-full mt-2"')

                def show_edge_inspector(edge_id: int):
                    right_tabs.value = rtab_inspector
                    inspector_drawer.clear()
                    rel = state.graph_store.get_relationship(edge_id)
                    if not rel: return

                    src_n = state.graph_store.get_node_details(rel['source_node_id'])
                    tgt_n = state.graph_store.get_node_details(rel['target_node_id'])

                    src_name = src_n['properties'].get('name') or src_n['label'] if src_n else f"#{rel['source_node_id']}"
                    tgt_name = tgt_n['properties'].get('name') or tgt_n['label'] if tgt_n else f"#{rel['target_node_id']}"

                    with inspector_drawer:
                        ui.badge(f"Edge ID #{rel['relationship_id']}", color='slate-700')
                        ui.label(f"[{rel['type']}]").classes('text-sm font-bold text-cyan-400')
                        ui.label(f"Source: {src_name}").classes('text-xs text-slate-300')
                        ui.label(f"Target: {tgt_name}").classes('text-xs text-slate-300')

                        if rel.get('properties'):
                            ui.label('Properties:').classes('text-xs text-slate-400 font-bold mt-2')
                            ui.label(json.dumps(rel['properties'], indent=2)).classes('text-xs text-slate-300 bg-slate-900/80 p-2 rounded')

                        async def delete_inspected_rel():
                            try:
                                await asyncio.to_thread(state.graph_store.delete_relationship, edge_id)
                                ui.notify(f"Deleted relationship ID {edge_id}", color='positive')
                                inspector_drawer.clear()
                                await refresh_graph_view()
                            except Exception as ex:
                                ui.notify(f"Error deleting relationship: {ex}", color='negative')

                        ui.button('Delete Relationship', icon='delete', on_click=delete_inspected_rel).props('flat text-color=red size=xs class="w-full mt-2"')

            # -----------------------------------------------------------------
            # TAB 4: RAG Search Studio
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_search):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    ui.label('RAG Search & Chunk Reconstruction Studio').classes('text-2xl font-bold text-teal-400')

                with ui.card().classes('w-full bg-slate-800/80 p-4 rounded-xl border border-slate-700 mb-6'):
                    with ui.row().classes('w-full gap-4 items-center'):
                        query_input = ui.input(placeholder='Enter semantic query or exact token (e.g. error code)...').classes('flex-1').props('dark standout')
                        query_input.on('keydown.enter', lambda: execute_search())
                        search_mode = ui.select(['hybrid', 'dense', 'bm25'], value='hybrid', label='Mode').classes('w-36').props('dark dense standout')
                        top_k_input = ui.number('Top K', value=3, min=1, max=50).classes('w-24').props('dark dense standout')
                        threshold_slider = ui.slider(min=0, max=100, value=0).classes('w-48')
                        ui.label().bind_text_from(threshold_slider, 'value', backward=lambda v: f'Min Rel: {v}%').classes('text-xs text-slate-400')
                        reconstruct_checkbox = ui.checkbox('Reconstruct Chunks', value=True).classes('text-teal-300 font-semibold')
                        ui.button('Search', icon='search', on_click=lambda: execute_search()).props('color=teal')

                search_results_container = ui.column().classes('w-full space-y-4')

            # -----------------------------------------------------------------
            # TAB 5: Database Diagnostics
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_diagnostics):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    ui.label('Database Diagnostics & Schema Verification').classes('text-2xl font-bold text-teal-400')
                    ui.button('Refresh Diagnostics', icon='refresh', on_click=lambda: refresh_diagnostics_view()).props('color=teal size=sm')

                diagnostics_content = ui.column().classes('w-full')

    # View Renderers
    def refresh_files_view():
        doc_table_container.clear()
        doc_details_container.clear()
        if not state.store:
            return

        docs = state.store.list_documents()

        with doc_table_container:
            ui.label(f'Indexed Documents ({len(docs)})').classes('text-lg font-bold text-teal-300 mb-2')
            if not docs:
                ui.label('No documents indexed yet. Click "Add Document" to begin.').classes('text-sm text-slate-400 italic')
                return

            # Detect duplicate filenames to disambiguate titles
            all_names = [Path(d['file_path']).name for d in docs]
            name_counts = {}
            for n in all_names:
                name_counts[n] = name_counts.get(n, 0) + 1

            columns = [
                {'name': 'doc_id', 'label': 'ID', 'field': 'doc_id', 'align': 'left', 'sortable': True},
                {'name': 'document_title', 'label': 'Document Title', 'field': 'document_title', 'align': 'left', 'sortable': True},
                {'name': 'file_path', 'label': 'Origin Path', 'field': 'file_path', 'align': 'left'},
                {'name': 'added_timestamp', 'label': 'Date Added', 'field': 'added_timestamp', 'align': 'left', 'sortable': True}
            ]
            rows = []
            for d in docs:
                p_obj = Path(d['file_path'])
                base_name = p_obj.name
                if name_counts.get(base_name, 0) > 1:
                    parent_name = p_obj.parent.name
                    display_title = f"{base_name} ({parent_name}/ #{d['doc_id']})" if parent_name else f"{base_name} (#{d['doc_id']})"
                else:
                    display_title = base_name

                clean_date = str(d.get('added_timestamp', '')).replace("b'", "").replace("'", "")[:19]
                rows.append({
                    'doc_id': d['doc_id'],
                    'document_title': display_title,
                    'file_path': d['file_path'],
                    'added_timestamp': clean_date,
                    'metadata': d.get('metadata')
                })

            def on_row_click(e):
                row = e.args[1]
                state.selected_doc_id = row['doc_id']
                show_doc_details(row['doc_id'], row['file_path'], row['metadata'], row['document_title'])

            table = ui.table(columns=columns, rows=rows, row_key='doc_id').classes('w-full').props('dark flat dense')
            table.on('rowClick', on_row_click)

    def show_doc_details(doc_id: int, file_path: str, metadata: Any, display_title: Optional[str] = None):
        doc_details_container.clear()
        with doc_details_container:
            title_text = display_title or Path(file_path).name
            ui.label(f'Document #{doc_id}: {title_text}').classes('text-lg font-bold text-teal-300 mb-1')
            ui.label(f'Full Origin: {file_path}').classes('text-xs text-slate-400 font-mono break-all mb-2')
            if metadata:
                ui.label(f'Metadata: {json.dumps(metadata)}').classes('text-xs text-slate-400 bg-slate-900/60 p-2 rounded mb-3')

            full_text = state.store.reconstruct_document_text(doc_id)
            ui.label('Reconstructed Full Content:').classes('text-sm font-semibold text-slate-300 mt-2 mb-1')
            with ui.scroll_area().classes('w-full h-96 p-3 bg-slate-900 rounded font-mono text-xs whitespace-pre-wrap border border-slate-700'):
                ui.label(full_text or "(Empty content)")

    async def refresh_datalake_view():
        datalake_plot_container.clear()
        datalake_inspector_container.clear()

        method = datalake_method_select.value
        n_comp = datalake_dim_select.value

        with datalake_inspector_container:
            ui.label('Chunk Inspector').classes('text-lg font-bold text-teal-300 mb-2 border-b border-slate-700 pb-2 w-full')
            inspector_detail = ui.column().classes('space-y-2 text-xs font-mono text-slate-300')
            with inspector_detail:
                ui.label('Hover or click on any point in the plot to inspect chunk content and metadata.').classes('text-slate-500 italic')

        with datalake_plot_container:
            plot_loading = ui.row().classes('w-full h-full items-center justify-center')
            with plot_loading:
                ui.spinner(size='lg', color='teal')
                ui.label(f'Computing {method.upper()} projection in worker thread...').classes('text-sm text-slate-400 ml-3')

        points = await asyncio.to_thread(
            state.store.get_datalake_view,
            method=method,
            n_components=n_comp,
            use_cache=True,
            output_format='dict'
        )

        datalake_plot_container.clear()
        with datalake_plot_container:
            if not points:
                ui.label('No vectorized points found. Index documents to view point cloud.').classes('text-sm text-slate-400 italic m-auto')
                return

            fig = go.Figure()
            doc_groups = {}
            for p in points:
                doc = p['document_title']
                if doc not in doc_groups:
                    doc_groups[doc] = []
                doc_groups[doc].append(p)

            # Store centroid metadata for interactive inspector lookup
            cog_lookup = {}

            for doc_title, pts in doc_groups.items():
                x_vals = [p['x'] for p in pts]
                y_vals = [p['y'] for p in pts]

                # Compute Center of Gravity (Centroid)
                cog_x = float(sum(x_vals) / len(x_vals))
                cog_y = float(sum(y_vals) / len(y_vals))

                hover_texts = [f"<b>{doc_title}</b><br>Chunk #{p['chunk_id']}<br>Path: {p.get('document_path', '')}" for p in pts]

                if n_comp == 3:
                    z_vals = [p['z'] for p in pts]
                    cog_z = float(sum(z_vals) / len(z_vals))
                    cog_lookup[doc_title] = {'x': cog_x, 'y': cog_y, 'z': cog_z, 'count': len(pts), 'path': pts[0].get('document_path', '')}

                    # Document Chunk Points Trace
                    fig.add_trace(go.Scatter3d(
                        x=x_vals, y=y_vals, z=z_vals,
                        mode='markers', name=doc_title,
                        text=hover_texts, hoverinfo='text',
                        marker=dict(size=5, opacity=0.7)
                    ))

                    # Center of Gravity (Centroid) Trace
                    fig.add_trace(go.Scatter3d(
                        x=[cog_x], y=[cog_y], z=[cog_z],
                        mode='markers+text',
                        name=f"⌖ CoG: {doc_title}",
                        text=[f"⌖ {doc_title}"],
                        textposition="top center",
                        textfont=dict(size=10, color='#f1f5f9'),
                        hovertext=[f"<b>[Center of Gravity] {doc_title}</b><br>Total Chunks: {len(pts)}<br>Centroid: ({cog_x:.3f}, {cog_y:.3f}, {cog_z:.3f})"],
                        hoverinfo='text',
                        marker=dict(size=9, symbol='diamond', line=dict(color='#ffffff', width=2))
                    ))
                else:
                    cog_lookup[doc_title] = {'x': cog_x, 'y': cog_y, 'count': len(pts), 'path': pts[0].get('document_path', '')}

                    # Document Chunk Points Trace
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode='markers', name=doc_title,
                        text=hover_texts, hoverinfo='text',
                        marker=dict(size=8, opacity=0.7)
                    ))

                    # Center of Gravity (Centroid) Trace
                    fig.add_trace(go.Scatter(
                        x=[cog_x], y=[cog_y],
                        mode='markers+text',
                        name=f"⌖ CoG: {doc_title}",
                        text=[f"⌖ {doc_title}"],
                        textposition="top center",
                        textfont=dict(size=11, color='#f1f5f9'),
                        hovertext=[f"<b>[Center of Gravity] {doc_title}</b><br>Total Chunks: {len(pts)}<br>Centroid: ({cog_x:.3f}, {cog_y:.3f})"],
                        hoverinfo='text',
                        marker=dict(size=14, symbol='diamond', line=dict(color='#ffffff', width=2))
                    ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation='h', y=-0.1),
                hovermode='closest'
            )

            plot = ui.plotly(fig).classes('w-full h-full')

            def on_point_click(e):
                try:
                    if e.args and 'points' in e.args and len(e.args['points']) > 0:
                        pt_data = e.args['points'][0]
                        pt_idx = pt_data.get('pointIndex', 0)
                        curve_idx = pt_data.get('curveNumber', 0)

                        # Each document has 2 traces: (chunks_trace, cog_trace)
                        doc_keys = list(doc_groups.keys())
                        doc_idx = curve_idx // 2
                        is_cog = (curve_idx % 2 == 1)

                        if doc_idx < len(doc_keys):
                            target_doc = doc_keys[doc_idx]
                            inspector_detail.clear()

                            if is_cog:
                                cog_info = cog_lookup.get(target_doc, {})
                                with inspector_detail:
                                    ui.badge('Center of Gravity', color='amber-500').classes('text-xs mb-1')
                                    ui.label(f"Document: {target_doc}").classes('font-bold text-teal-400')
                                    ui.label(f"Full Origin: {cog_info.get('path', '')}").classes('text-xs text-slate-400 font-mono break-all')
                                    coords_str = f"({cog_info.get('x', 0):.3f}, {cog_info.get('y', 0):.3f}{', ' + str(round(cog_info.get('z', 0), 3)) if n_comp == 3 else ''})"
                                    ui.label(f"Centroid Coordinates: {coords_str}")
                                    ui.label(f"Cluster Size: {cog_info.get('count', 0)} chunks").classes('font-semibold text-teal-300 mt-1')
                                    ui.label('The Center of Gravity represents the mean semantic embedding coordinate for all chunks in this document.').classes('text-slate-400 text-xs italic mt-2')
                            else:
                                pts = doc_groups[target_doc]
                                if pt_idx < len(pts):
                                    selected_p = pts[pt_idx]
                                    with inspector_detail:
                                        ui.badge('Document Chunk', color='teal').classes('text-xs mb-1')
                                        ui.label(f"Document: {selected_p['document_title']}").classes('font-bold text-teal-400')
                                        ui.label(f"Origin Path: {selected_p.get('document_path', '')}").classes('text-xs text-slate-400 font-mono break-all')
                                        ui.label(f"Chunk ID: {selected_p['chunk_id']}")
                                        coords_str = f"({selected_p['x']:.3f}, {selected_p['y']:.3f}{', ' + str(round(selected_p['z'], 3)) if n_comp == 3 else ''})"
                                        ui.label(f"Coordinates: {coords_str}")
                                        ui.label('Content Preview:').classes('font-semibold text-slate-200 mt-2')
                                        ui.label(selected_p.get('chunk_text_preview', '(No text preview)')).classes('p-2 bg-slate-900 rounded border border-slate-700 whitespace-pre-wrap')
                                        ui.label(f"Metadata: {json.dumps(selected_p.get('metadata', {}))}").classes('text-slate-400 text-xs mt-1')
                except Exception:
                    pass

            plot.on('plotly_click', on_point_click)

    async def refresh_graph_view():
        if not state.graph_store: return

        info = await asyncio.to_thread(state.graph_store.get_graph_info)
        graph_summary_badge.text = f"Nodes: {info['total_nodes']} | Edges: {info['total_relationships']} | Provenance: {info['total_provenance_links']}"
        graph_summary_badge.props('color=teal')

        # Update left sidebar metrics
        left_metrics_box.clear()
        with left_metrics_box:
            ui.label(f"• Total Nodes : {info['total_nodes']}")
            ui.label(f"• Total Edges : {info['total_relationships']}")
            ui.label(f"• Chunk Links : {info['total_provenance_links']}")
            ui.label(f"• Class Count : {len(info['nodes_by_label'])}")

        # Fetch nodes and edges for visual rendering
        nodes_raw = await asyncio.to_thread(state.graph_store.get_all_nodes, 1000)
        edges_raw = await asyncio.to_thread(state.graph_store.get_all_relationships, 2000)

        # Palette for label coloring
        palette = ['#14b8a6', '#06b6d4', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6', '#10b981', '#f97316', '#6366f1', '#e11d48']
        label_colors = {}
        for idx, lbl in enumerate(sorted(info['nodes_by_label'].keys())):
            label_colors[lbl] = palette[idx % len(palette)]

        # Render category chips in left sidebar
        node_types_filter_container.clear()
        with node_types_filter_container:
            for lbl, count in sorted(info['nodes_by_label'].items(), key=lambda x: x[1], reverse=True):
                col = label_colors.get(lbl, '#14b8a6')
                with ui.row().classes('w-full justify-between items-center py-0.5 cursor-pointer hover:bg-slate-700/50 px-1 rounded'):
                    with ui.row().classes('items-center gap-1.5'):
                        ui.element('span').classes('w-2.5 h-2.5 rounded-full').style(f'background-color: {col}')
                        ui.label(lbl).classes('text-slate-200 font-bold')
                    ui.label(str(count)).classes('text-slate-400')

        # Prepare vis-network payload
        vis_nodes = []
        for n in nodes_raw:
            lbl = n['label']
            name = str(n['properties'].get('name') or n['properties'].get('identifying_value') or n['properties'].get('title') or lbl)
            color = label_colors.get(lbl, '#14b8a6')
            vis_nodes.append({
                'id': n['node_id'],
                'label': name[:25] + ('...' if len(name) > 25 else ''),
                'title': f"<b>[{lbl}] {name}</b><br>ID #{n['node_id']}<br>" + "<br>".join(f"{k}: {v}" for k, v in list(n['properties'].items())[:5]),
                'color': { 'background': color, 'border': '#0f172a', 'highlight': { 'background': '#ffffff', 'border': color } },
                'category': lbl
            })

        vis_edges = []
        for r in edges_raw:
            vis_edges.append({
                'id': r['relationship_id'],
                'from': r['source_node_id'],
                'to': r['target_node_id'],
                'label': r['type'],
                'title': f"<b>[{r['type']}]</b> (ID #{r['relationship_id']})"
            })

        # Mount vis-network on canvas
        await ui.run_javascript(f"window.initSafeStoreGraph('graph-network-container', {json.dumps(vis_nodes)}, {json.dumps(vis_edges)})")

    async def execute_search():
        raw_val = query_input.value
        query_text = (raw_val or "").strip()
        if not query_text:
            search_results_container.clear()
            with search_results_container:
                ui.label("Please enter a query in the search box.").classes('text-sm text-amber-400 italic')
            return

        mode = search_mode.value
        k = int(top_k_input.value or 3)
        thresh = float(threshold_slider.value or 0.0)
        reconstruct = bool(reconstruct_checkbox.value)

        # 1. Mount loading spinner cleanly inside container
        search_results_container.clear()
        with search_results_container:
            with ui.row().classes('items-center gap-2 text-teal-400 p-2'):
                ui.spinner('dots', size='md', color='teal')
                ui.label(f"Executing {mode.upper()} search across embeddings...").classes('text-sm text-slate-300')

        # 2. Perform search in background thread OUTSIDE container context
        try:
            if mode == 'hybrid':
                hits = await asyncio.to_thread(
                    state.store.hybrid_query,
                    query_text,
                    top_k=k,
                    min_relevance_percent=thresh,
                    reconstruct_overlapping_chunks=reconstruct
                )
            elif mode == 'dense':
                hits = await asyncio.to_thread(
                    state.store.query,
                    query_text,
                    top_k=k,
                    min_relevance_percent=thresh,
                    reconstruct_overlapping_chunks=reconstruct
                )
            elif mode == 'bm25':
                from safe_store import BM25Retriever
                bm25 = BM25Retriever(state.store.conn)
                raw_hits = await asyncio.to_thread(bm25.search, query_text, top_k=k, min_relevance_percent=thresh)
                hits = await asyncio.to_thread(state.store.reconstruct_overlapping_chunks, raw_hits) if reconstruct else raw_hits
            else:
                hits = []
        except Exception as e:
            search_results_container.clear()
            with search_results_container:
                ui.label(f"Search Execution Error: {e}").classes('text-sm text-rose-400 font-mono')
            return

        # 3. Clear container and re-enter context to render results
        search_results_container.clear()
        with search_results_container:
            ui.label(f"Query Results for: '{query_text}' ({mode.upper()} Search, {len(hits)} hit(s))").classes('text-lg font-bold text-teal-300')

            if not hits:
                ui.label(f"No results found matching '{query_text}' above the {thresh:.1f}% threshold.").classes('text-sm text-slate-400 italic')
                return

            for i, hit in enumerate(hits, 1):
                with ui.card().classes('w-full bg-slate-800/90 border border-slate-700 p-4 rounded-xl shadow-lg'):
                    with ui.row().classes('w-full justify-between items-center mb-2'):
                        with ui.row().classes('items-center gap-2'):
                            ui.badge(f"Rank #{i}", color='slate-700')
                            ui.label(hit.get('document_title', Path(hit.get('file_path', '')).name)).classes('font-bold text-teal-300')
                        with ui.row().classes('items-center gap-2'):
                            rel_grade = float(hit.get('relevance_score', hit.get('similarity_percent', 0.0)))
                            ui.badge(f"Relevance: {rel_grade:.1f}%", color='teal' if rel_grade >= 50 else 'cyan')
                            if hit.get('is_reconstructed'):
                                ui.badge("Reconstructed Contiguous", color='indigo')

                    if hit.get('fused_chunk_ids'):
                        ui.label(f"Fused Chunks: IDs={hit['fused_chunk_ids']} | Sequences={hit.get('chunk_seqs')}").classes('text-xs text-slate-400 font-mono mb-2')

                    with ui.scroll_area().classes('w-full max-h-56 p-3 bg-slate-900 rounded font-mono text-xs whitespace-pre-wrap border border-slate-700 text-slate-200'):
                        ui.label(hit.get('chunk_text', '(Empty chunk text)'))

    def refresh_diagnostics_view():
        diagnostics_content.clear()
        if not state.store:
            return

        with diagnostics_content:
            try:
                diag = state.store.get_database_info()
                with ui.row().classes('w-full gap-6'):
                    with ui.card().classes('w-1/2 bg-slate-800/80 p-4 rounded-xl border border-slate-700'):
                        ui.label('Store & Vectorizer Diagnostics').classes('text-lg font-bold text-teal-300 mb-2')
                        ui.label(f"Store Name: {diag['store_name']}")
                        ui.label(f"Database Path: {diag['database_path']}")
                        ui.label(f"Encryption: {'Enabled (Fernet)' if diag['encryption_enabled'] else 'Plaintext'}")
                        ui.label(f"Vectorizer: {diag['vectorizer'].get('name')} (Dim: {diag['vectorizer'].get('dim')}, Dtype: {diag['vectorizer'].get('dtype')})")
                        ui.label(f"Chunking: {diag['chunking_configuration']['strategy']} (Size: {diag['chunking_configuration']['chunk_size']}, Overlap: {diag['chunking_configuration']['chunk_overlap']})")

                    with ui.card().classes('w-1/2 bg-slate-800/80 p-4 rounded-xl border border-slate-700'):
                        ui.label('Corpus & Graph Inventory').classes('text-lg font-bold text-teal-300 mb-2')
                        ui.label(f"Total Documents: {diag['documents']['total_documents']}")
                        ui.label(f"Total Chunks: {diag['documents']['total_chunks']}")
                        ui.label(f"Total Graph Nodes: {diag['knowledge_graph']['total_nodes']}")
                        ui.label(f"Total Graph Relationships: {diag['knowledge_graph']['total_relationships']}")
                        ui.label(f"Provenance Links: {diag['knowledge_graph']['total_provenance_links']}")
            except Exception as e:
                ui.label(f"Diagnostics error: {e}").classes('text-rose-400')

    async def refresh_all_views():
        refresh_files_view()
        await refresh_datalake_view()
        await refresh_graph_view()
        refresh_diagnostics_view()

    # Reactive Tab Switching Listener
    async def on_tab_change(e):
        # Defensively resolve tab target from ValueChangeEventArguments, GenericEventArguments, or tabs.value
        target = getattr(e, 'value', None)
        if target is None and hasattr(e, 'args'):
            target = e.args
        if target is None:
            target = tabs.value

        target_name = getattr(target, 'name', target) if not isinstance(target, str) else target

        if target == tab_graph or target_name == getattr(tab_graph, 'name', 'tab_graph'):
            await asyncio.sleep(0.05)
            await refresh_graph_view()
            await ui.run_javascript('setTimeout(() => { if (window.safeStoreNetwork) { window.safeStoreNetwork.redraw(); window.safeStoreNetwork.fit({animation: true}); } }, 200);')
        elif target == tab_datalake or target_name == getattr(tab_datalake, 'name', 'tab_datalake'):
            await refresh_datalake_view()
        elif target == tab_diagnostics or target_name == getattr(tab_diagnostics, 'name', 'tab_diagnostics'):
            refresh_diagnostics_view()
        elif target == tab_files or target_name == getattr(tab_files, 'name', 'tab_files'):
            refresh_files_view()

    tabs.on_value_change(on_tab_change)

    # =========================================================================
    # VIEW NAVIGATION CONTROLLER
    # =========================================================================
    def show_projects_view():
        # Do not load any store or vectorizer while viewing the project hub
        state.close_current_store()
        workspace_container.set_visibility(False)
        projects_container.set_visibility(True)
        btn_all_stores.set_visibility(False)
        db_label.text = 'Projects Hub'
        refresh_projects_cards()

    async def switch_to_store_workspace(db_path_str: str):
        store_fname = Path(db_path_str).name
        loading_title.text = f"Opening '{store_fname}'"
        loading_text.text = "Loading vectorizer model & database schema..."
        loading_overlay.open()
        await asyncio.sleep(0.05)

        try:
            await asyncio.to_thread(state.open_store, db_path_str)
            db_label.text = f"DB: {Path(state.db_path).name}"
            projects_container.set_visibility(False)
            workspace_container.set_visibility(True)
            btn_all_stores.set_visibility(True)
            tabs.value = tab_files
            refresh_files_view()
            refresh_diagnostics_view()
        except Exception as ex:
            ui.notify(f"Failed to open store: {ex}", color='negative', duration=7000)
            show_projects_view()
        finally:
            loading_overlay.close()

    def refresh_projects_cards():
        projects_grid.clear()
        all_stores = discover_local_stores()
        term = (project_search.value or "").strip().lower()

        with projects_grid:
            # Card 1: + Add New Project Card
            with ui.card().classes('border-2 border-dashed border-slate-700 hover:border-teal-500 bg-slate-800/40 hover:bg-slate-800/80 rounded-2xl flex flex-col items-center justify-center p-8 cursor-pointer transition-all duration-300 min-h-[260px] text-center group') as add_card:
                add_card.on('click', lambda: create_store_dialog.open())
                with ui.column().classes('items-center gap-2'):
                    with ui.element('div').classes('w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700 group-hover:border-teal-500 group-hover:bg-teal-500/10 transition-colors'):
                        ui.icon('add', size='md').classes('text-slate-400 group-hover:text-teal-400')
                    ui.label('Create New Store').classes('font-bold text-slate-200 group-hover:text-teal-400 text-base')
                    ui.label('Initialize a new vector & graph database').classes('text-xs text-slate-500 max-w-[14rem]')

            # Individual Store Project Cards
            for store_info in all_stores:
                if term and term not in store_info["name"].lower() and term not in store_info["filename"].lower() and term not in store_info["description"].lower():
                    continue

                def _bind_card(info=store_info):
                    with ui.card().classes('bg-slate-800/80 hover:bg-slate-800 border border-slate-700 hover:border-teal-500/60 rounded-2xl p-5 flex flex-col justify-between shadow-lg hover:shadow-2xl transition-all duration-300 cursor-pointer min-h-[260px] group relative overflow-hidden'):
                        # Card Click: Open Store Workspace
                        ui.element('div').classes('absolute inset-0 z-0').on('click', lambda: asyncio.create_task(switch_to_store_workspace(info["path"])))

                        # Top Bar: Name & Actions
                        with ui.row().classes('w-full justify-between items-start z-10 pointer-events-none'):
                            with ui.row().classes('items-center gap-2.5 flex-1 pr-2'):
                                with ui.element('div').classes('w-10 h-10 rounded-xl bg-teal-500/10 border border-teal-500/30 flex items-center justify-center text-teal-400'):
                                    ui.icon('lock' if info['is_encrypted'] else 'database', size='sm')
                                with ui.column().classes('gap-0 flex-1'):
                                    ui.label(info['name']).classes('font-bold text-slate-100 text-base leading-snug truncate w-full')
                                    ui.label(info['filename']).classes('text-xs text-slate-400 font-mono truncate w-full')

                            with ui.row().classes('items-center gap-1 pointer-events-auto opacity-0 group-hover:opacity-100 transition-opacity'):
                                def open_edit():
                                    edit_target_path.value = info["path"]
                                    edit_store_name.value = info["name"]
                                    edit_store_desc.value = info["description"]
                                    edit_store_dialog.open()

                                def open_delete():
                                    delete_target_path.value = info["path"]
                                    delete_store_msg.text = f"Are you sure you want to permanently delete '{info['name']}' ({info['filename']})? This action cannot be undone."
                                    delete_store_dialog.open()

                                ui.button(icon='edit', on_click=open_edit).props('flat round dense size=sm text-color=grey')
                                ui.button(icon='delete', on_click=open_delete).props('flat round dense size=sm text-color=red')

                        # Middle: Description
                        with ui.column().classes('w-full my-2 z-10 pointer-events-none'):
                            desc_text = info['description'] or 'No description provided.'
                            ui.label(desc_text).classes('text-xs text-slate-400 line-clamp-2 h-8 leading-relaxed')

                        # Badges: Model, Docs, Chunks, Graph
                        with ui.column().classes('w-full gap-2 z-10 pointer-events-none pt-2 border-t border-slate-700/60'):
                            with ui.row().classes('gap-1.5 flex-wrap'):
                                vec_label = f"{info['vectorizer_name'].upper()}" + (f": {info['vectorizer_model']}" if info['vectorizer_model'] else "")
                                ui.badge(vec_label[:28], color='slate-700').classes('text-[10px] font-mono')
                                if info['is_encrypted']:
                                    ui.badge('Encrypted', color='amber-500').classes('text-[10px]')

                            with ui.row().classes('w-full justify-between items-center text-xs text-slate-400 font-mono pt-1'):
                                with ui.row().classes('items-center gap-1'):
                                    ui.icon('description', size='xs').classes('text-teal-400')
                                    ui.label(f"{info['doc_count']} docs")
                                    ui.label('•')
                                    ui.label(f"{info['chunk_count']} chunks")

                                if info['node_count'] > 0:
                                    with ui.row().classes('items-center gap-1'):
                                        ui.icon('share', size='xs').classes('text-cyan-400')
                                        ui.label(f"{info['node_count']} nodes")

                            with ui.row().classes('w-full justify-between items-center text-[10px] text-slate-500 font-mono'):
                                ui.label(f"{info['size_mb']} MB")
                                ui.label(info['modified_time'])

                _bind_card()

        project_search.on('input', refresh_projects_cards)

    # --- Initial Page Load Strategy ---
    await client.connected()
    refresh_projects_cards()

    # If launched with a specific database argument, open it immediately
    if initial_path and Path(initial_path).exists():
        await switch_to_store_workspace(str(Path(initial_path).resolve()))
    else:
        show_projects_view()


# Register asynchronous root page handler with expanded response timeout
@ui.page('/', response_timeout=60.0)
async def index(client: Client):
    await render_studio_page(client, CURRENT_DB_PATH)


def launch_studio(
    db_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    native: bool = True
):
    """Launches SafeStore Studio via NiceGUI and pywebview."""
    global CURRENT_DB_PATH
    CURRENT_DB_PATH = str(Path(db_path).resolve()) if db_path else None

    # Pre-check pywebview availability for native window
    if native:
        try:
            import webview
        except ImportError:
            print("[!] pywebview not installed. Running in browser mode...")
            native = False

    try:
        ui.run(
            title="SafeStore Studio",
            host=host,
            port=port,
            native=native,
            window_size=(1366, 850),
            reload=False
        )
    except Exception as e:
        if native:
            print(f"[!] Native desktop window failed ({e}). Falling back to browser mode at http://{host}:{port}...")
            ui.run(
                title="SafeStore Studio",
                host=host,
                port=port,
                native=False,
                reload=False
            )
        else:
            raise e


def main():
    parser = argparse.ArgumentParser(description="SafeStore Studio: Visual VectorDB & Knowledge Graph RAG App")
    parser.add_argument("db_path", nargs="?", default=None, help="Optional path to SafeStore SQLite database (.db). If omitted, opens the Projects Hub.")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--browser", action="store_true", help="Launch in default browser instead of native pywebview window")

    args = parser.parse_args()
    launch_studio(
        db_path=args.db_path,
        host=args.host,
        port=args.port,
        native=not args.browser
    )


if __name__ == "__main__":
    main()