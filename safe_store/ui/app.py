import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import json
import threading
import pipmaster as pm
from ascii_colors import ASCIIColors
# Ensure UI dependencies
pm.ensure_packages(["nicegui>=1.4.0", "pywebview>=4.0", "plotly>=5.0", "pandas"])

from nicegui import ui, app, Client
import plotly.graph_objects as go
import time
import safe_store
from safe_store import SafeStore, GraphStore, LogLevel
import numpy as np
# Global configured database path for initial page loads
CURRENT_DB_PATH: Optional[str] = None
CURRENT_LOLLMS_CLIENT: Optional[Any] = None
CURRENT_LLM_GENERATOR: Optional[Callable[..., str]] = None
PROJECTS_DIR = Path("projects")
STUDIO_CONFIG_FILE = Path("studio_llm_config.json")


def load_studio_llm_config() -> Dict[str, Any]:
    """Loads saved LLM settings from disk if available."""
    if STUDIO_CONFIG_FILE.exists():
        try:
            return json.loads(STUDIO_CONFIG_FILE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {
        "provider": "lollms",
        "binding_name": "ollama",
        "model_name": "mistral:latest",
        "host_address": "http://localhost:11434",
        "service_key": "",
        "temperature": 0.1
    }


def save_studio_llm_config(cfg: Dict[str, Any]) -> None:
    """Persists LLM settings to disk."""
    try:
        STUDIO_CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    except Exception:
        pass


def build_lollms_generator_from_config(cfg: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Callable[..., str]]]:
    """Builds a LollmsClient instance and a standard generator callable from user configuration."""
    provider = cfg.get("provider", "lollms")
    if provider == "lollms":
        try:
            from lollms_client import LollmsClient
            client = LollmsClient(
                llm_binding_name=cfg.get("binding_name", "ollama"),
                llm_binding_config={
                    "model_name": cfg.get("model_name", "mistral:latest"),
                    "host_address": cfg.get("host_address", "http://localhost:11434"),
                    "service_key": cfg.get("service_key", "")
                }
            )
            from safe_store.graph.graph_store import wrap_llm_callable
            generator = wrap_llm_callable(client)
            return client, generator
        except Exception as e:
            ASCIIColors.warning(f"Could not initialize LollmsClient from config: {e}")
            return None, None
    elif provider == "openai":
        try:
            from openai import OpenAI
            api_key = cfg.get("service_key") or os.environ.get("OPENAI_API_KEY", "")
            base_url = cfg.get("host_address") or None
            client = OpenAI(api_key=api_key or "not_needed", base_url=base_url)
            model_name = cfg.get("model_name", "gpt-4o-mini")

            def _openai_gen(prompt: str, system_prompt: str|None = None, json_mode: bool = False, **kwargs) -> str:
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": prompt})
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    response_format={"type": "json_object"} if json_mode else None,
                    temperature=kwargs.get("temperature", 0.1)
                )
                return resp.choices[0].message.content or ""

            return client, _openai_gen
        except Exception as e:
            ASCIIColors.warning(f"Could not initialize OpenAI client from config: {e}")
            return None, None
    return None, None
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
<style>
.vis-tooltip { display: none !important; }
</style>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
window.safeStoreNetwork = null;
window.safeStoreNetworkData = { nodes: null, edges: null };
window.safeStoreOriginalNodeColors = {};

window.initSafeStoreGraph = function(containerId, nodesArray, edgesArray) {
    let container = document.getElementById(containerId);
    if (!container) {
        container = document.getElementById('graph-network-container');
    }
    if (!container) {
        container = document.querySelector('[id*="graph-network"]');
    }
    if (!container) {
        console.error("SafeStore Graph container not found for ID:", containerId);
        return;
    }

    if (container.clientHeight === 0) {
        container.style.height = '560px';
    }

    window.safeStoreOriginalNodeColors = {};
    nodesArray.forEach(n => {
        window.safeStoreOriginalNodeColors[n.id] = (n.color && n.color.background) ? n.color.background : n.color;
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
    }, 100);
    setTimeout(() => {
        if (window.safeStoreNetwork) {
            window.safeStoreNetwork.fit({ animation: { duration: 400 } });
        }
    }, 350);

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
    def __init__(self, llm_generator: Optional[Callable[..., str]] = None, lollms_client: Optional[Any] = None):
        self.db_path: str = ""
        self.store: Optional[SafeStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.selected_doc_id: Optional[int] = None
        self.chat_history: List[Dict[str, str]] = []
        self.llm_generator: Optional[Callable[..., str]] = llm_generator
        self.lollms_client: Optional[Any] = lollms_client

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
        self.chat_history.clear()

    def open_store(self, path: str, encryption_key: Optional[str] = None):
        self.close_current_store()
        path_str = str(path).strip()
        if path_str.lower() in (":memory:", "memory") or path_str.endswith(":memory:"):
            self.db_path = ":memory:"
        elif path_str.lower() in (":tempfile:", "tempfile") or path_str.endswith(":tempfile:"):
            self.db_path = ":tempfile:"
        else:
            self.db_path = str(Path(path).resolve())
        self.store = SafeStore(
            db_path=self.db_path,
            log_level=LogLevel.INFO,
            encryption_key=encryption_key,
            llm_generator=self.llm_generator,
            lollms_client=self.lollms_client
        )
        try:
            self.graph_store = GraphStore(
                store=self.store,
                llm_generator=self.llm_generator,
                lollms_client=self.lollms_client
            )
        except Exception:
            self.graph_store = None
        self.selected_doc_id = None


async def render_studio_page(
    client: Client,
    initial_path: Optional[str] = None,
    llm_generator: Optional[Callable[..., str]] = None,
    lollms_client: Optional[Any] = None
):
    state = StudioState(llm_generator=llm_generator, lollms_client=lollms_client)

    # Automatically load and initialize saved LLM configuration on startup
    if state.llm_generator is None:
        startup_cfg = load_studio_llm_config()
        c_inst, g_func = await asyncio.to_thread(build_lollms_generator_from_config, startup_cfg)
        if g_func is not None:
            state.llm_generator = g_func
            state.lollms_client = c_inst

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

            # Database Import Action (Clones external .db via SQLite Online Backup)
            async def import_external_db():
                selected = await pick_file_dialog(
                    title="Select SQLite Database to Import",
                    file_types=[("SQLite DB (*.db, *.sqlite)", "*.db;*.sqlite;*.sqlite3"), ("All Files (*.*)", "*.*")]
                )
                if selected:
                    target_name = Path(selected).name
                    dest_path = PROJECTS_DIR / target_name
                    try:
                        loading_title.text = f"Importing '{target_name}'"
                        loading_text.text = "Copying SQLite database with online backup..."
                        loading_overlay.open()
                        await asyncio.to_thread(SafeStore.import_db_file, selected, dest_path)
                        ui.notify(f"Imported database '{target_name}' successfully!", color='positive', icon='check_circle')
                        refresh_projects_cards()
                        await switch_to_store_workspace(str(dest_path.resolve()))
                    except Exception as err:
                        ui.notify(f"Import failed: {err}", color='negative')
                    finally:
                        loading_overlay.close()

            ui.button('Import DB', icon='file_upload', on_click=import_external_db).props('outline color=teal size=sm title="Import external .db file"')

            # Database Export Action (Saves current .db cleanly via SQLite Online Backup)
            async def export_current_db():
                if not state.store or not state.db_path or state.db_path in (":memory:", ":tempfile:"):
                    ui.notify("Open a persistent database to export.", color='warning')
                    return
                src_name = Path(state.db_path).name
                dest_dir = Path.home() / "Downloads"
                dest_path = dest_dir / f"backup_{src_name}"
                try:
                    await asyncio.to_thread(state.store.export_db_file, dest_path)
                    ui.notify(f"Database exported to '{dest_path}'!", color='positive', icon='file_download', duration=7000)
                except Exception as err:
                    ui.notify(f"Export failed: {err}", color='negative')

            btn_export_db = ui.button('Export DB', icon='file_download', on_click=export_current_db).props('outline color=teal size=sm title="Export database file"')
            btn_export_db.bind_visibility_from(btn_all_stores, 'visible')

            # LLM Configuration Center Dialog
            ui.button('LLM Setup', icon='tune', on_click=lambda: llm_config_dialog.open()).props('outline color=teal size=sm title="Configure LoLLMS / LLM Backend"')

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

    # =========================================================================
    # DIALOG: LLM CONFIGURATION CENTER
    # =========================================================================
    current_cfg = load_studio_llm_config()
    with ui.dialog() as llm_config_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[40rem] border border-slate-700 space-y-4 shadow-2xl'):
        with ui.row().classes('w-full justify-between items-center'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('smart_toy', size='md').classes('text-teal-400')
                ui.label('LLM & LoLLMS Configuration Center').classes('text-xl font-bold text-teal-300')
            llm_status_badge = ui.badge('Configured' if state.llm_generator else 'Not Connected', color='teal' if state.llm_generator else 'slate-700')

        ui.label('Configure the AI model backend used for Sourced Agentic Chat, Knowledge Graph extraction, and SPARQL generation.').classes('text-xs text-slate-400')

        llm_provider_select = ui.select(
            options={
                'lollms': 'LoLLMS Client (Ollama, vLLM, WebUI, etc.)',
                'openai': 'OpenAI SDK / Compatible REST Endpoint'
            },
            value=current_cfg.get('provider', 'lollms'),
            label='Backend Provider'
        ).classes('w-full').props('dark standout')

        llm_binding_input = ui.input('Binding Name (e.g. ollama, vllm, open_ai)', value=current_cfg.get('binding_name', 'ollama')).classes('w-full').props('dark standout')
        llm_binding_input.bind_visibility_from(llm_provider_select, 'value', value='lollms')

        with ui.row().classes('w-full gap-4'):
            llm_model_input = ui.input('Model Name', value=current_cfg.get('model_name', 'mistral:latest')).classes('flex-1').props('dark standout')
            llm_host_input = ui.input('Host Address / URL', value=current_cfg.get('host_address', 'http://localhost:11434')).classes('flex-1').props('dark standout')

        llm_key_input = ui.input('API / Service Key (Optional)', value=current_cfg.get('service_key', '')).classes('w-full').props('dark standout type=password')

        async def test_and_save_llm():
            cfg = {
                "provider": llm_provider_select.value,
                "binding_name": llm_binding_input.value.strip(),
                "model_name": llm_model_input.value.strip(),
                "host_address": llm_host_input.value.strip(),
                "service_key": llm_key_input.value.strip()
            }
            save_studio_llm_config(cfg)
            ui.notify("Testing connection to AI backend...", color='info')
            client_inst, gen_func = await asyncio.to_thread(build_lollms_generator_from_config, cfg)

            if gen_func is not None:
                state.llm_generator = gen_func
                state.lollms_client = client_inst
                if state.store:
                    state.store.set_llm_generator(gen_func)
                llm_status_badge.text = "Connected"
                llm_status_badge.props('color=positive')
                ui.notify(f"Successfully connected to {cfg['model_name']}!", color='positive', icon='check_circle')
                llm_config_dialog.close()
            else:
                llm_status_badge.text = "Failed"
                llm_status_badge.props('color=negative')
                ui.notify("Connection test failed. Check host address, model name, and service status.", color='negative', duration=6000)

        with ui.row().classes('w-full justify-end gap-2 mt-4'):
            ui.button('Cancel', on_click=llm_config_dialog.close).props('flat text-color=grey')
            ui.button('Test & Save Configuration', icon='save', on_click=test_and_save_llm).props('color=teal')

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

        with ui.column().classes('w-full bg-slate-900/70 p-3.5 rounded-xl border border-slate-700/80 space-y-2'):
            with ui.row().classes('items-center gap-2'):
                ui.icon('lock', size='sm').classes('text-amber-400')
                ui.label('Zero-Leakage Encryption at Rest').classes('text-sm font-bold text-amber-300')

            new_store_enc_toggle = ui.checkbox('Encrypt Entire Database (Documents, Paths, Chunks, and Graph)', value=False).classes('text-xs text-slate-200')
            with ui.row().classes('w-full gap-3').bind_visibility_from(new_store_enc_toggle, 'value'):
                new_store_pw = ui.input('Passphrase', placeholder='Enter strong encryption key').classes('flex-1').props('dark standout type=password')
                new_store_pw_confirm = ui.input('Confirm Passphrase', placeholder='Repeat passphrase').classes('flex-1').props('dark standout type=password')

        with ui.row().classes('w-full gap-4 items-center'):
            new_store_shared = ui.checkbox('Use Shared Model Server Daemon (Zero-OOM)', value=True).classes('text-xs text-teal-300')
            new_store_shared.bind_visibility_from(new_store_vec, 'value', value='st')

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
            enc_enabled = bool(new_store_enc_toggle.value)
            enc_key = new_store_pw.value.strip() if enc_enabled else None
            enc_confirm = new_store_pw_confirm.value.strip() if enc_enabled else None

            if enc_enabled:
                if not enc_key:
                    ui.notify("Encryption passphrase cannot be empty when encryption is enabled.", color='warning')
                    return
                if enc_key != enc_confirm:
                    ui.notify("Encryption passphrases do not match.", color='negative')
                    return

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
            tab_clusters = ui.tab('Clusters & Themes', icon='category')
            tab_chat = ui.tab('Agentic Chat & Q&A', icon='chat')
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
                with ui.card().classes('w-full bg-slate-800/90 p-4 rounded-2xl border border-slate-700 shadow-xl mb-4'):
                    with ui.row().classes('w-full justify-between items-center flex-wrap gap-4'):
                        with ui.column().classes('gap-0.5'):
                            ui.label('Semantic Datalake Explorer').classes('text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-cyan-400')
                            with ui.row().classes('items-center gap-2'):
                                datalake_stats_label = ui.label('Projecting document chunk embeddings into semantic manifold space.').classes('text-xs text-slate-400 font-mono')
                                datalake_outdated_badge = ui.badge('⚠️ Outdated: New documents added', color='amber-500').classes('text-[10px] font-bold')
                                datalake_outdated_badge.set_visibility(False)

                        with ui.row().classes('items-center gap-2.5 flex-wrap'):
                            datalake_method_select = ui.select(
                                {'umap': 'UMAP (Manifold)', 'pca': 'PCA (Variance)', 'tsne': 't-SNE (Cluster)'},
                                value='umap',
                                label='Method'
                            ).classes('w-40 text-xs').props('dark dense standout')

                            datalake_dim_select = ui.select(
                                {2: '2D Plane', 3: '3D Space'},
                                value=2,
                                label='Dimension'
                            ).classes('w-32 text-xs').props('dark dense standout')

                            datalake_show_cog_switch = ui.checkbox('Show Centroids (⌖)', value=True).classes('text-xs text-teal-300 font-semibold')
                            datalake_show_labels_switch = ui.checkbox('Doc Labels', value=True).classes('text-xs text-slate-300')

                            async def export_datalake_standalone_html():
                                if not state.store: return
                                try:
                                    out_file = Path("datalake_export.html")
                                    await asyncio.to_thread(
                                        state.store.export_datalake_html,
                                        output_file=out_file,
                                        method=datalake_method_select.value,
                                        n_components=int(datalake_dim_select.value)
                                    )
                                    ui.notify(f"Exported interactive HTML to '{out_file.name}'!", color='positive', icon='download')
                                except Exception as err:
                                    ui.notify(f"Export failed: {err}", color='negative')

                            ui.button(icon='file_download', on_click=export_datalake_standalone_html).props('outline color=teal size=sm title="Export Standalone HTML Visualizer"')
                            datalake_project_btn = ui.button('Project Embeddings', icon='auto_awesome', on_click=lambda: refresh_datalake_view()).props('color=teal size=sm')

                # Main 2-Panel Layout: Plot Canvas (Left) | Semantic Inspector (Right)
                with ui.row().classes('w-full gap-4 items-stretch h-[75vh]'):
                    datalake_plot_container = ui.card().classes('flex-1 bg-slate-900/90 rounded-2xl border border-slate-700/80 p-3 flex flex-col relative overflow-hidden shadow-2xl')
                    datalake_inspector_container = ui.card().classes('w-96 bg-slate-800/90 rounded-2xl border border-slate-700/80 p-4 flex flex-col justify-between overflow-y-auto shadow-xl space-y-3')

            # -----------------------------------------------------------------
            # TAB 2.5: Document Clusters & Thematic Analysis
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_clusters):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    with ui.column().classes('gap-1'):
                        ui.label('Document Clusters & Thematic Analysis').classes('text-2xl font-bold text-teal-400')
                        ui.label('Group corpus documents using semantic vector centroids and extract conceptual themes.').classes('text-xs text-slate-400')

                    with ui.row().classes('items-center gap-3'):
                        cluster_k_input = ui.number('Clusters (0=Auto)', value=0, min=0, max=30).classes('w-36 text-xs').props('dark dense standout')
                        cluster_algo_select = ui.select(['kmeans', 'agglomerative'], value='kmeans', label='Algorithm').classes('w-36 text-xs').props('dark dense standout')
                        cluster_themes_switch = ui.checkbox('Generate Thematic Titles & Summaries', value=True).classes('text-xs text-teal-300')
                        cluster_run_btn = ui.button('Cluster Documents', icon='auto_awesome', on_click=lambda: run_document_clustering()).props('color=teal size=sm')

                with ui.row().classes('w-full items-center justify-between mb-2'):
                    clusters_outdated_banner = ui.card().classes('w-full bg-amber-500/10 border border-amber-500/40 p-2.5 rounded-xl flex justify-between items-center')
                    with clusters_outdated_banner:
                        with ui.row().classes('items-center gap-2 text-amber-300 text-xs font-mono'):
                            ui.icon('warning', size='sm').classes('text-amber-400')
                            ui.label('New documents added since clusters were generated. Click "Re-Cluster" to update.').classes('font-bold')
                        ui.button('Re-Cluster Now', icon='refresh', on_click=lambda: run_document_clustering()).props('color=amber-500 text-color=black size=xs font-bold')
                    clusters_outdated_banner.set_visibility(False)

                clusters_summary_bar = ui.row().classes('w-full items-center justify-between bg-slate-800/80 p-3 rounded-xl border border-slate-700 text-xs font-mono text-slate-300 mb-4')
                clusters_grid_container = ui.element('div').classes('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full')

            # -----------------------------------------------------------------
            # TAB 2.7: Sourced Agentic Chat & Q&A Workspace
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_chat):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    with ui.column().classes('gap-0.5'):
                        ui.label('Talk to Your Documents (Sourced Agentic Q&A)').classes('text-2xl font-bold text-teal-400')
                        ui.label('Agentic knowledge recovery combining vector semantic search, BM25 keywords, and SPARQL logic with clickable source citations.').classes('text-xs text-slate-400')

                    with ui.row().classes('items-center gap-2'):
                        chat_top_k_select = ui.select({3: 'Top 3 Chunks', 5: 'Top 5 Chunks', 8: 'Top 8 Chunks'}, value=5, label='Context Span').classes('w-36 text-xs').props('dark dense standout')
                        chat_include_graph = ui.checkbox('Query Graph Logic (SPARQL)', value=True).classes('text-xs text-teal-300 font-semibold')
                        ui.button('Clear Chat', icon='delete_sweep', on_click=lambda: clear_agentic_chat()).props('flat text-color=grey size=sm')

                # Main Chat Container: Stream (Left) | Source Evidence Cards (Right)
                with ui.row().classes('w-full gap-4 items-stretch h-[72vh]'):
                    with ui.card().classes('flex-1 bg-slate-900/90 rounded-2xl border border-slate-700 p-4 flex flex-col justify-between overflow-hidden shadow-2xl'):
                        chat_scroll_area = ui.scroll_area().classes('w-full flex-1 pr-3 space-y-4')
                        with chat_scroll_area:
                            chat_messages_container = ui.column().classes('w-full space-y-4')

                        with ui.row().classes('w-full items-center gap-2 pt-3 border-t border-slate-800'):
                            chat_input = ui.input(placeholder='Ask any question about your indexed documents...').classes('flex-1 font-sans text-sm').props('dark standout')
                            chat_send_btn = ui.button('Ask', icon='send', on_click=lambda: send_agentic_chat()).props('color=teal')
                            chat_input.on('keydown.enter', lambda: send_agentic_chat())

                    # Right Side: Live Source Evidence Inspector
                    with ui.card().classes('w-96 bg-slate-800/90 rounded-2xl border border-slate-700 p-4 flex flex-col justify-between overflow-y-auto shadow-xl space-y-3'):
                        with ui.row().classes('w-full justify-between items-center border-b border-slate-700 pb-2'):
                            with ui.row().classes('items-center gap-1.5'):
                                ui.icon('menu_book', size='sm').classes('text-teal-400')
                                ui.label('Grounding Citations').classes('text-sm font-bold text-slate-200')
                            chat_source_badge = ui.badge('0 Sources', color='slate-700').classes('text-[10px] font-mono')

                        chat_sources_drawer = ui.column().classes('w-full space-y-3 text-xs font-mono text-slate-300 flex-1')
                        with chat_sources_drawer:
                            with ui.column().classes('items-center justify-center h-48 text-center text-slate-500 m-auto space-y-2'):
                                ui.icon('source', size='lg').classes('text-slate-600')
                                ui.label('Retrieved chunk citations and SPARQL facts will appear here as the agent recovers knowledge.').classes('text-xs italic max-w-xs')

            # -----------------------------------------------------------------
            # TAB 3: Knowledge Graph Visual Studio & W3C SPARQL 1.1 Console
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_graph):
                # Header Bar
                with ui.row().classes('w-full justify-between items-center mb-3'):
                    with ui.row().classes('items-center gap-3'):
                        ui.label('Knowledge Graph Studio').classes('text-2xl font-bold text-teal-400')
                        graph_scope_select = ui.select(
                            options={
                                'abox': 'ABox (Instances & Content)',
                                'tbox': 'TBox (Ontology Schema)',
                                'both': 'Unified (ABox Content + TBox Schema)'
                            },
                            value='abox',
                            label='Graph Scope'
                        ).classes('w-64 text-xs').props('dark dense standout')
                        graph_summary_badge = ui.badge('Loading...', color='slate-700').classes('text-xs font-mono')

                    # Dynamic Action Toolbar: Changes based on Scope (ABox vs TBox)
                    with ui.row().classes('items-center gap-2'):
                        # TBOX AUTHORING DIALOG 1: Add OWL Class
                        with ui.dialog() as add_class_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[36rem] border border-purple-500/50 space-y-4'):
                            with ui.row().classes('w-full justify-between items-center'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('category', size='sm').classes('text-purple-400')
                                    ui.label('Add Ontology Class (owl:Class)').classes('text-lg font-bold text-purple-300')
                                ui.button(icon='close', on_click=add_class_dialog.close).props('flat dense round text-color=grey')

                            class_name_input = ui.input('Class Name / Identifier', placeholder='e.g. Microservice, Drone, Scientist').classes('w-full').props('dark standout')
                            class_label_input = ui.input('Display Label (Optional)', placeholder='e.g. Microservice Component').classes('w-full').props('dark standout')
                            class_parent_select = ui.select({}, label='Superclass (rdfs:subClassOf)').classes('w-full').props('dark standout')
                            class_disjoint_select = ui.select({}, label='Disjoint Class Exclusivity (owl:disjointWith)').classes('w-full').props('dark standout')

                            async def execute_add_class():
                                c_name = class_name_input.value.strip()
                                if not c_name:
                                    ui.notify("Class name is required.", color='warning')
                                    return
                                try:
                                    state.graph_store.tbox.add_class(
                                        class_name_or_uri=c_name,
                                        label=class_label_input.value.strip() or None,
                                        sub_class_of=class_parent_select.value or None,
                                        disjoint_with=class_disjoint_select.value or None
                                    )
                                    state.graph_store.save_ontology_to_db()
                                    ui.notify(f"Added ontology class '{c_name}'!", color='positive', icon='check_circle')
                                    add_class_dialog.close()
                                    await refresh_graph_view()
                                except Exception as err:
                                    ui.notify(f"Error adding class: {err}", color='negative')

                            with ui.row().classes('w-full justify-end gap-2'):
                                ui.button('Cancel', on_click=add_class_dialog.close).props('flat text-color=grey')
                                ui.button('Create Class', icon='add', on_click=execute_add_class).props('color=purple')

                        # TBOX AUTHORING DIALOG 2: Add Object Property (Link between classes)
                        with ui.dialog() as add_prop_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[40rem] border border-cyan-500/50 space-y-4'):
                            with ui.row().classes('w-full justify-between items-center'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('link', size='sm').classes('text-cyan-400')
                                    ui.label('Add Object Property (Class Link)').classes('text-lg font-bold text-cyan-300')
                                ui.button(icon='close', on_click=add_prop_dialog.close).props('flat dense round text-color=grey')

                            prop_name_input = ui.input('Property Name / Predicate', placeholder='e.g. dependsOn, communicatesWith, manages').classes('w-full').props('dark standout')
                            with ui.row().classes('w-full gap-4'):
                                prop_dom_select = ui.select({}, label='Domain Class (Source)').classes('flex-1').props('dark standout')
                                prop_rng_select = ui.select({}, label='Range Class (Target)').classes('flex-1').props('dark standout')

                            prop_inv_input = ui.input('Inverse Property Name (Optional)', placeholder='e.g. dependedOnBy').classes('w-full').props('dark standout')

                            with ui.row().classes('w-full gap-4 items-center'):
                                prop_trans_chk = ui.checkbox('Transitive (A->B->C => A->C)', value=False).classes('text-xs text-cyan-300 font-mono')
                                prop_sym_chk = ui.checkbox('Symmetric (A->B <=> B->A)', value=False).classes('text-xs text-cyan-300 font-mono')
                                prop_func_chk = ui.checkbox('Functional (At most 1 value)', value=False).classes('text-xs text-cyan-300 font-mono')

                            async def execute_add_object_property():
                                p_name = prop_name_input.value.strip()
                                if not p_name or not prop_dom_select.value or not prop_rng_select.value:
                                    ui.notify("Property name, Domain, and Range are required.", color='warning')
                                    return
                                try:
                                    state.graph_store.tbox.add_object_property(
                                        prop_name_or_uri=p_name,
                                        domain=prop_dom_select.value,
                                        range_=prop_rng_select.value,
                                        is_transitive=bool(prop_trans_chk.value),
                                        is_symmetric=bool(prop_sym_chk.value),
                                        is_functional=bool(prop_func_chk.value),
                                        inverse_of=prop_inv_input.value.strip() or None
                                    )
                                    state.graph_store.save_ontology_to_db()
                                    ui.notify(f"Added object property '{p_name}'!", color='positive', icon='check_circle')
                                    add_prop_dialog.close()
                                    await refresh_graph_view()
                                except Exception as err:
                                    ui.notify(f"Error adding property: {err}", color='negative')

                            with ui.row().classes('w-full justify-end gap-2'):
                                ui.button('Cancel', on_click=add_prop_dialog.close).props('flat text-color=grey')
                                ui.button('Create Property', icon='add', on_click=execute_add_object_property).props('color=cyan')

                        # TBOX AUTHORING DIALOG 3: Add Datatype Property (Class Attribute)
                        with ui.dialog() as add_attr_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[36rem] border border-amber-500/50 space-y-4'):
                            with ui.row().classes('w-full justify-between items-center'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('label', size='sm').classes('text-amber-400')
                                    ui.label('Add Class Attribute (Datatype Property)').classes('text-lg font-bold text-amber-300')
                                ui.button(icon='close', on_click=add_attr_dialog.close).props('flat dense round text-color=grey')

                            attr_name_input = ui.input('Attribute Name', placeholder='e.g. ipAddress, serialNumber, rating').classes('w-full').props('dark standout')
                            attr_dom_select = ui.select({}, label='Target Class (Domain)').classes('w-full').props('dark standout')
                            attr_type_select = ui.select(
                                ['xsd:string', 'xsd:integer', 'xsd:float', 'xsd:boolean', 'xsd:dateTime'],
                                value='xsd:string',
                                label='Data Type'
                            ).classes('w-full').props('dark standout')
                            attr_func_chk = ui.checkbox('Functional Attribute (Single value per entity)', value=True).classes('text-xs text-amber-300 font-mono')

                            async def execute_add_attribute():
                                a_name = attr_name_input.value.strip()
                                if not a_name or not attr_dom_select.value:
                                    ui.notify("Attribute name and Target Class are required.", color='warning')
                                    return
                                try:
                                    state.graph_store.tbox.add_datatype_property(
                                        prop_name_or_uri=a_name,
                                        domain=attr_dom_select.value,
                                        range_data_type=attr_type_select.value,
                                        is_functional=bool(attr_func_chk.value)
                                    )
                                    state.graph_store.save_ontology_to_db()
                                    ui.notify(f"Added attribute '{a_name}' on {attr_dom_select.value}!", color='positive', icon='check_circle')
                                    add_attr_dialog.close()
                                    await refresh_graph_view()
                                except Exception as err:
                                    ui.notify(f"Error adding attribute: {err}", color='negative')

                            with ui.row().classes('w-full justify-end gap-2'):
                                ui.button('Cancel', on_click=add_attr_dialog.close).props('flat text-color=grey')
                                ui.button('Create Attribute', icon='add', on_click=execute_add_attribute).props('color=amber-500')

                        # TBOX BUTTONS (Visible when TBox mode active)
                        async def open_class_dialog():
                            if not state.graph_store: return
                            classes = state.graph_store.tbox.get_classes()
                            opts = {c: c.split('/')[-1].split('#')[-1] for c in classes}
                            opts_with_empty = {"": "(None)"}
                            opts_with_empty.update(opts)
                            class_parent_select.options = opts_with_empty
                            class_parent_select.value = ""
                            class_disjoint_select.options = opts_with_empty
                            class_disjoint_select.value = ""
                            add_class_dialog.open()

                        async def open_prop_dialog():
                            if not state.graph_store: return
                            classes = state.graph_store.tbox.get_classes()
                            opts = {c: c.split('/')[-1].split('#')[-1] for c in classes}
                            prop_dom_select.options = opts
                            prop_rng_select.options = opts
                            if opts:
                                prop_dom_select.value = list(opts.keys())[0]
                                prop_rng_select.value = list(opts.keys())[-1]
                            add_prop_dialog.open()

                        async def open_attr_dialog():
                            if not state.graph_store: return
                            classes = state.graph_store.tbox.get_classes()
                            opts = {c: c.split('/')[-1].split('#')[-1] for c in classes}
                            attr_dom_select.options = opts
                            if opts: attr_dom_select.value = list(opts.keys())[0]
                            add_attr_dialog.open()

                        btn_add_class = ui.button('Add Class (OWL)', icon='category', on_click=open_class_dialog).props('color=purple size=sm')
                        btn_add_obj_prop = ui.button('Add Link / Relation', icon='link', on_click=open_prop_dialog).props('color=cyan size=sm')
                        btn_add_attr = ui.button('Add Attribute', icon='label', on_click=open_attr_dialog).props('outline color=amber-500 size=sm')

                        # ABOX BUTTONS (Visible when ABox mode active)
                        # Tabular CSV Mapping Modal
                        with ui.dialog() as csv_import_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[48rem] border border-slate-700 space-y-4 shadow-2xl'):
                            with ui.row().classes('w-full justify-between items-center'):
                                with ui.row().classes('items-center gap-2'):
                                    ui.icon('table_chart', size='md').classes('text-teal-400')
                                    ui.label('Import Tabular CSV Data to Knowledge Graph').classes('text-xl font-bold text-teal-300')
                                ui.button(icon='close', on_click=csv_import_dialog.close).props('flat dense round text-color=grey')

                            ui.label('Ingest CSV rows as grounded ABox entities and relationships. Choose manual mapping or let AI synthesize the mapping for review.').classes('text-xs text-slate-400')

                            with ui.row().classes('w-full gap-3 items-center'):
                                csv_data_input = ui.input('Selected Data CSV File').classes('flex-1').props('dark standout')
                                async def browse_data_csv():
                                    sel = await pick_file_dialog("Select Data CSV File", [("CSV Files (*.csv)", "*.csv")])
                                    if sel: csv_data_input.value = sel
                                ui.button('Browse CSV...', icon='folder_open', on_click=browse_data_csv).props('color=teal size=sm')

                            with ui.row().classes('w-full gap-4'):
                                csv_mode_select = ui.select(
                                    options={
                                        'ai': 'Auto-Generate Mapping with AI (Review & Apply)',
                                        'file': 'Provide Existing Mapping File (.csv / .json)'
                                    },
                                    value='ai',
                                    label='Mapping Strategy'
                                ).classes('flex-1').props('dark standout')

                            with ui.row().classes('w-full gap-3 items-center') as mapping_file_row:
                                csv_mapping_input = ui.input('Selected Mapping File (.csv/.json)').classes('flex-1').props('dark standout')
                                async def browse_mapping_file():
                                    sel = await pick_file_dialog("Select Mapping File", [("Mapping Files (*.csv;*.json)", "*.csv;*.json")])
                                    if sel: csv_mapping_input.value = sel
                                ui.button('Browse...', icon='folder_open', on_click=browse_mapping_file).props('outline color=teal size=sm')
                            mapping_file_row.bind_visibility_from(csv_mode_select, 'value', value='file')

                            csv_guidance_input = ui.textarea('Mapping Guidance (Optional)', placeholder='e.g. Map employee_id to Person, dept_id to Organization, and creates worksFor edges.').classes('w-full text-xs font-mono').props('dark standout rows=2')
                            csv_guidance_input.bind_visibility_from(csv_mode_select, 'value', value='ai')

                            ui.label('Proposed Declarative Mapping Rules (JSON):').classes('text-xs font-bold text-teal-400')
                            mapping_json_editor = ui.textarea(
                                placeholder='Click "Generate AI Mapping" or upload a mapping file to review and edit rules...'
                            ).classes('w-full font-mono text-xs').props('dark standout rows=8')

                            async def run_ai_mapping_proposal():
                                d_file = csv_data_input.value.strip()
                                if not d_file or not Path(d_file).exists():
                                    ui.notify("Please select a valid Data CSV file first.", color='warning')
                                    return
                                from safe_store import TabularMapper
                                mapper = TabularMapper(store=state.store, tbox=state.graph_store.tbox)
                                ui.notify("Analyzing CSV headers and generating mapping with AI...", color='info')
                                try:
                                    rules = await asyncio.to_thread(
                                        mapper.propose_mapping_with_llm,
                                        csv_data_or_path=d_file,
                                        user_guidance=csv_guidance_input.value.strip() or None
                                    )
                                    mapping_json_editor.value = json.dumps(rules, indent=2)
                                    ui.notify("AI Mapping generated! Review the rules below before applying.", color='positive')
                                except Exception as err:
                                    ui.notify(f"Mapping generation failed: {err}", color='negative')

                            async def apply_csv_import():
                                d_file = csv_data_input.value.strip()
                                if not d_file or not Path(d_file).exists():
                                    ui.notify("Please select a valid Data CSV file.", color='warning')
                                    return

                                mode = csv_mode_select.value
                                from safe_store import TabularMapper
                                mapper = TabularMapper(store=state.store, tbox=state.graph_store.tbox)

                                try:
                                    if mode == 'file' and csv_mapping_input.value.strip():
                                        m_file = csv_mapping_input.value.strip()
                                        res = await asyncio.to_thread(mapper.map_csv_with_mapping_file, d_file, m_file)
                                    else:
                                        rules_text = mapping_json_editor.value.strip()
                                        if not rules_text:
                                            ui.notify("Mapping rules are empty. Generate or enter rules first.", color='warning')
                                            return
                                        rules = json.loads(rules_text)
                                        res = await asyncio.to_thread(mapper.map_csv, d_file, rules)

                                    ui.notify(f"Import complete! Created {res['entities_created']} entities and {res['triples_generated']} triples.", color='positive', icon='check_circle')
                                    csv_import_dialog.close()
                                    await refresh_graph_view()
                                except Exception as err:
                                    ui.notify(f"Import failed: {err}", color='negative', duration=7000)

                            with ui.row().classes('w-full justify-between items-center mt-3'):
                                ui.button('Generate AI Mapping', icon='auto_awesome', on_click=run_ai_mapping_proposal).props('outline color=purple size=sm').bind_visibility_from(csv_mode_select, 'value', value='ai')
                                with ui.row().classes('gap-2'):
                                    ui.button('Cancel', on_click=csv_import_dialog.close).props('flat text-color=grey')
                                    ui.button('Verify & Apply Import', icon='cloud_upload', on_click=apply_csv_import).props('color=teal')

                        ui.button('Import CSV Data', icon='table_chart', on_click=csv_import_dialog.open).props('outline color=teal size=sm')
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

                            resume_checkbox = ui.checkbox('Resume unprocessed documents only', value=True).classes('text-xs text-teal-300 font-mono')

                            with ui.column().classes('w-full gap-1'):
                                with ui.row().classes('w-full justify-between items-center text-xs text-slate-300 font-mono'):
                                    build_status_label = ui.label('Ready').classes('text-xs text-teal-300 font-mono truncate flex-1')
                                    build_percent_label = ui.label('0.0%').classes('text-xs font-bold text-teal-400 font-mono ml-2')
                                build_progress_bar = ui.linear_progress(value=0.0, show_value=False).props('rounded size=14px color=teal track-color=slate-700').classes('w-full')

                            active_extraction_stop = threading.Event()

                            def request_stop_extraction():
                                active_extraction_stop.set()
                                build_status_label.text = "Stopping... finishing current document and saving progress."
                                ui.notify("Stopping after current step completes...", color='warning')
                                stop_btn.props('disabled')

                            async def run_fast_graph_build():
                                if not state.graph_store:
                                    ui.notify("GraphStore not initialized on this store.", color='negative')
                                    return

                                active_extraction_stop.clear()
                                chosen_mode = mode_select.value
                                chosen_batch = int(batch_size_slider.value)
                                chosen_guidance = guidance_input.value.strip() or None
                                should_resume = bool(resume_checkbox.value)

                                build_progress_bar.value = 0.0
                                build_percent_label.text = "0.0%"
                                build_status_label.text = "Initializing extraction pipeline..."

                                start_btn.set_visibility(False)
                                stop_btn.set_visibility(True)
                                stop_btn.props(remove='disabled')
                                close_btn.props('disabled')

                                def _ui_progress(fraction, message):
                                    with client:
                                        clamped_f = max(0.0, min(1.0, float(fraction)))
                                        build_progress_bar.value = clamped_f
                                        build_percent_label.text = f"{clamped_f * 100.0:.1f}%"
                                        build_status_label.text = message

                                try:
                                    action_name = "Resuming" if should_resume else "Starting fresh"
                                    ui.notify(f"{action_name} graph extraction using '{chosen_mode.upper()}' mode...", color='info')

                                    stats = await asyncio.to_thread(
                                        state.graph_store.build_graph_for_all_documents,
                                        mode=chosen_mode,
                                        chunks_per_batch=chosen_batch,
                                        guidance=chosen_guidance,
                                        progress_callback=_ui_progress,
                                        stop_event=active_extraction_stop,
                                        resume=should_resume
                                    )

                                    if stats.get("stopped"):
                                        ui.notify(
                                            f"Extraction paused. Saved: {stats['nodes_created']} nodes, {stats['relationships_created']} edges. You can resume anytime!",
                                            color='warning',
                                            duration=6000
                                        )
                                        build_status_label.text = f"Paused ({stats['nodes_created']} nodes, {stats['relationships_created']} edges saved). Ready to resume."
                                        start_btn.text = "Resume Fast Extraction"
                                    elif stats.get("all_completed"):
                                        ui.notify("All documents in this store are already extracted into the knowledge graph!", color='positive', duration=5000)
                                        build_status_label.text = "All documents already extracted."
                                        build_progress_bar.value = 1.0
                                        build_percent_label.text = "100.0%"
                                    else:
                                        ui.notify(
                                            f"Graph Ready: {stats['nodes_created']} nodes, {stats['relationships_created']} edges created!",
                                            color='positive',
                                            duration=5000
                                        )
                                        build_status_label.text = f"Completed ({stats['nodes_created']} nodes, {stats['relationships_created']} edges created)."
                                        build_progress_bar.value = 1.0
                                        build_percent_label.text = "100.0%"
                                        build_graph_dialog.close()

                                    await refresh_graph_view()
                                except Exception as e:
                                    ui.notify(f"Graph extraction error: {e}", color='negative', duration=7000)
                                    build_status_label.text = f"Error: {e}"
                                finally:
                                    start_btn.set_visibility(True)
                                    start_btn.props(remove='disabled')
                                    stop_btn.set_visibility(False)
                                    close_btn.props(remove='disabled')

                            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                                close_btn = ui.button('Cancel', on_click=build_graph_dialog.close).props('flat text-color=grey')
                                stop_btn = ui.button('Stop Extraction', icon='stop', on_click=request_stop_extraction).props('color=negative')
                                stop_btn.set_visibility(False)
                                start_btn = ui.button('Start Fast Extraction', icon='bolt', on_click=run_fast_graph_build).props('color=teal')

                        btn_add_node = ui.button('Add Node', icon='add', on_click=add_node_dialog.open).props('outline color=teal size=sm')

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

                        btn_add_rel = ui.button('Add Relationship', icon='link', on_click=open_rel_dialog).props('outline color=cyan size=sm')
                        btn_build_kg = ui.button('Build Knowledge Graph', icon='bolt', on_click=build_graph_dialog.open).props('color=teal size=sm')
                        btn_import_csv = ui.button('Import CSV Data', icon='table_chart', on_click=csv_import_dialog.open).props('outline color=teal size=sm')

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
                                with client:
                                    val = node_search_input.value.strip().lower()
                                    if not val or not state.graph_store: return
                                    nodes = await asyncio.to_thread(state.graph_store.get_all_nodes, 1000)
                                    for n in nodes:
                                        candidate_name = str(n['properties'].get('name') or n['properties'].get('identifying_value') or '').lower()
                                        if val in candidate_name or val in n['label'].lower():
                                            await client.run_javascript(f"window.focusGraphNode({n['node_id']})")
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
                                ui.button(icon='filter_center_focus', on_click=lambda: client.run_javascript('window.safeStoreNetwork && window.safeStoreNetwork.fit({animation: true})')).props('flat dense size=sm title="Fit Entire Network"')
                                physics_toggle = ui.switch('Physics', value=True).props('dark dense').classes('text-xs text-slate-300')
                                physics_toggle.on('update:model-value', lambda e: client.run_javascript(f'window.toggleGraphPhysics({str(e.args).lower()})'))
                                ui.button('Reset Colors', icon='restart_alt', on_click=lambda: client.run_javascript('window.resetGraphHighlight()')).props('flat dense text-color=teal size=sm title="Reset Selection"')

                        # The HTML div where vis-network mounts with explicit height
                        graph_canvas_div = ui.element('div').classes('w-full flex-1').style('width: 100%; height: 100%; min-height: 560px;')
                        graph_canvas_div._props['id'] = 'graph-network-container'
                        graph_canvas_div.props('id=graph-network-container')

                        # Clipboard helper
                        async def copy_to_clipboard(text: str, success_msg: str = "Copied to clipboard!"):
                            with client:
                                js_cmd = f"navigator.clipboard.writeText({json.dumps(text)});"
                                await client.run_javascript(js_cmd)
                                ui.notify(success_msg, color='positive', icon='content_paste')

                        # Real Interactive Information Panel (Floating HUD on canvas)
                        hud_info_card = ui.card().classes(
                            'absolute bottom-4 left-4 z-30 bg-slate-850/95 backdrop-blur-md border border-teal-500/40 '
                            'rounded-xl p-4 shadow-2xl max-w-sm w-96 space-y-3 transition-all duration-300'
                        ).style('background-color: rgba(15, 23, 42, 0.95);')
                        hud_info_card.set_visibility(False)

                        def show_hud_node_info(node_id: int):
                            if not state.graph_store: return
                            node = state.graph_store.get_node_details(node_id)
                            if not node: return

                            hud_info_card.clear()
                            hud_info_card.set_visibility(True)

                            name = str(node['properties'].get('name') or node['properties'].get('identifying_value') or node['properties'].get('title') or node['label'])
                            clean_props = {k: v for k, v in node['properties'].items() if k != "other_identifiers"}
                            json_str = json.dumps(clean_props, indent=2, ensure_ascii=False)

                            with hud_info_card:
                                # Header Bar
                                with ui.row().classes('w-full justify-between items-center'):
                                    with ui.row().classes('items-center gap-1.5'):
                                        ui.badge(node['label'], color='teal').classes('font-bold text-xs uppercase tracking-wider')
                                        ui.badge(f"ID #{node['node_id']}", color='slate-700').classes('text-xs font-mono text-slate-300')
                                    ui.button(icon='close', on_click=lambda: hud_info_card.set_visibility(False)).props('flat dense round size=xs text-color=grey')

                                # Entity Title & Quick Copy
                                with ui.row().classes('w-full justify-between items-start bg-slate-900/90 p-2.5 rounded-lg border border-slate-700/70'):
                                    with ui.column().classes('flex-1 pr-2 gap-0'):
                                        ui.label('Entity Name').classes('text-[10px] text-teal-400 font-mono uppercase tracking-wider')
                                        ui.label(name).classes('text-sm font-bold text-slate-100 break-words')
                                    ui.button(icon='content_copy', on_click=lambda n=name: asyncio.create_task(copy_to_clipboard(n, f"Copied '{n}'"))).props('flat dense size=sm text-color=teal title="Copy Name"')

                                # Properties View
                                with ui.column().classes('w-full gap-1'):
                                    ui.label('Attributes & Properties').classes('text-[10px] text-slate-400 font-mono uppercase tracking-wider')
                                    with ui.scroll_area().classes('w-full max-h-40 bg-slate-900/90 p-2 rounded-lg border border-slate-700/70 font-mono text-xs'):
                                        for k, v in clean_props.items():
                                            if k not in ("name", "identifying_value"):
                                                with ui.row().classes('w-full justify-between items-start py-0.5 border-b border-slate-800/80'):
                                                    ui.label(f"{k}:").classes('text-teal-300 font-semibold text-[11px]')
                                                    ui.label(str(v)).classes('text-slate-300 text-[11px] break-all flex-1 text-right ml-2')

                                # Bottom Action Toolbar
                                with ui.row().classes('w-full justify-between items-center pt-1 border-t border-slate-700/60'):
                                    ui.button('Copy JSON', icon='code', on_click=lambda j=json_str: asyncio.create_task(copy_to_clipboard(j, "Copied node JSON!"))).props('outline color=teal size=xs')
                                    ui.button('Focus', icon='filter_center_focus', on_click=lambda nid=node_id: client.run_javascript(f"window.focusGraphNode({nid})")).props('outline color=cyan size=xs')
                                    
                                    def _inspect_more(nid=node_id):
                                        show_node_inspector(nid)
                                        right_tabs.value = rtab_inspector

                                    ui.button('Inspect Evidence', icon='search', on_click=_inspect_more).props('color=teal size=xs')

                        def show_hud_edge_info(edge_id: int):
                            if not state.graph_store: return
                            rel = state.graph_store.get_relationship(edge_id)
                            if not rel: return

                            src_n = state.graph_store.get_node_details(rel['source_node_id'])
                            tgt_n = state.graph_store.get_node_details(rel['target_node_id'])

                            src_name = src_n['properties'].get('name') or src_n['label'] if src_n else f"#{rel['source_node_id']}"
                            tgt_name = tgt_n['properties'].get('name') or tgt_n['label'] if tgt_n else f"#{rel['target_node_id']}"
                            json_str = json.dumps(rel.get('properties', {}), indent=2)

                            hud_info_card.clear()
                            hud_info_card.set_visibility(True)

                            with hud_info_card:
                                with ui.row().classes('w-full justify-between items-center'):
                                    ui.badge(rel['type'], color='cyan').classes('font-bold text-xs')
                                    ui.badge(f"Edge #{rel['relationship_id']}", color='slate-700').classes('text-xs font-mono')
                                    ui.button(icon='close', on_click=lambda: hud_info_card.set_visibility(False)).props('flat dense round size=xs text-color=grey')

                                with ui.column().classes('w-full bg-slate-900/90 p-2.5 rounded-lg border border-slate-700 font-mono text-xs gap-1'):
                                    ui.label(f"From: {src_name}").classes('text-slate-200 font-bold')
                                    ui.label(f"  ──[{rel['type']}]──►").classes('text-cyan-400')
                                    ui.label(f"To:   {tgt_name}").classes('text-slate-200 font-bold')

                                with ui.row().classes('w-full justify-end pt-1'):
                                    ui.button('Copy JSON', icon='code', on_click=lambda j=json_str: asyncio.create_task(copy_to_clipboard(j, "Copied edge JSON!"))).props('outline color=cyan size=xs')

                        # Attach custom DOM event handlers
                        def on_canvas_node_clicked(e):
                            nid = e.args.get('node_id')
                            if nid is not None:
                                show_hud_node_info(int(nid))
                                show_node_inspector(int(nid))

                        def on_canvas_edge_clicked(e):
                            rid = e.args.get('edge_id')
                            if rid is not None:
                                show_hud_edge_info(int(rid))
                                show_edge_inspector(int(rid))

                        def on_canvas_bg_clicked(e):
                            hud_info_card.set_visibility(False)

                        graph_canvas_div.on('safestore_node_clicked', on_canvas_node_clicked)
                        graph_canvas_div.on('safestore_edge_clicked', on_canvas_edge_clicked)
                        graph_canvas_div.on('safestore_background_clicked', on_canvas_bg_clicked)


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
                                    sparql_reasoning_switch = ui.checkbox('OWL/RDFS Reasoning', value=False).classes('text-xs text-teal-300 font-mono')
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
                                    preset = getattr(e, 'value', None)
                                    if preset is None and hasattr(e, 'args'):
                                        preset = e.args
                                    if preset is None:
                                        preset = sparql_presets.value

                                    preset_str = str(preset or '')
                                    if 'all' in preset_str:
                                        sparql_editor.value = """PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
} LIMIT 15"""
                                    elif 'types' in preset_str:
                                        sparql_editor.value = """PREFIX ont: <http://example.org/ontology/>
SELECT ?entity ?label WHERE {
    ?entity ont:label ?label .
} LIMIT 20"""
                                    elif 'connected' in preset_str:
                                        sparql_editor.value = """PREFIX ex: <http://example.org/>
SELECT ?source ?relation ?target WHERE {
    ?source ?relation ?target .
    FILTER(!CONTAINS(STR(?relation), "type") && !CONTAINS(STR(?relation), "label"))
} LIMIT 20"""

                                sparql_presets.on_value_change(on_preset_change)

                                async def execute_sparql_and_highlight():
                                    query_str = sparql_editor.value.strip()
                                    sparql_results_drawer.clear()
                                    with sparql_results_drawer:
                                        ui.spinner('dots', size='sm', color='teal')
                                        ui.label('Running SPARQL query...').classes('text-xs text-slate-400')

                                    try:
                                        use_reasoning = bool(sparql_reasoning_switch.value)
                                        res = await asyncio.to_thread(state.graph_store.query_sparql, query_str, enable_reasoning=use_reasoning)
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
                                                await client.run_javascript(f"window.highlightGraphNodes({json.dumps(list(matched_node_ids))})")

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

                                # Consistency Verification Action
                                async def run_consistency_check():
                                    if not state.graph_store: return
                                    try:
                                        report = await asyncio.to_thread(state.graph_store.check_consistency)
                                        sparql_results_drawer.clear()
                                        with sparql_results_drawer:
                                            if report["is_consistent"]:
                                                ui.badge('Logical Graph Consistency: 100% Valid (0 Conflicts)', color='positive').classes('text-xs p-2 w-full text-center')
                                                ui.label('All exclusivity, disjoint class, and functional constraints are satisfied.').classes('text-[11px] text-slate-300 italic mt-1')
                                            else:
                                                ui.badge(f'Consistency Violations Detected: {report["violation_count"]}', color='negative').classes('text-xs p-2 w-full text-center font-bold')
                                                for v in report["violations"]:
                                                    with ui.card().classes('w-full bg-slate-950 p-2 rounded border border-rose-500/60 text-xs font-mono space-y-1 mt-1'):
                                                        ui.label(f"[{v['severity']}] {v['type']}").classes('text-rose-400 font-bold')
                                                        ui.label(v['conflict']).classes('text-slate-300')
                                    except Exception as ex:
                                        ui.notify(f"Consistency check error: {ex}", color='negative')

                                with ui.row().classes('w-full justify-between items-center gap-1.5'):
                                    ui.button('Execute & Highlight', icon='play_arrow', on_click=execute_sparql_and_highlight).props('color=teal size=sm class="flex-1"')
                                    ui.button('Verify Exclusivity', icon='verified_user', on_click=run_consistency_check).props('outline color=cyan size=sm title="Check Disjointness & Contradictions"')
                                    ui.button(icon='restart_alt', on_click=lambda: client.run_javascript('window.resetGraphHighlight()')).props('outline text-color=grey size=sm title="Reset Highlight"')

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
                                    with ui.row().classes('w-full justify-between items-center'):
                                        ui.label(f"[Chunk #{c['chunk_id']} from {Path(c['file_path']).name}]:").classes('text-teal-400 font-bold text-[10px]')
                                        ui.button(icon='content_copy', on_click=lambda t=c['chunk_text']: asyncio.create_task(copy_to_clipboard(t, "Copied chunk text!"))).props('flat dense size=xs text-color=teal')
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
        n_comp = int(datalake_dim_select.value)
        show_cog = bool(datalake_show_cog_switch.value)
        show_labels = bool(datalake_show_labels_switch.value)

        # Check and display outdated state
        if state.store and hasattr(state.store, "is_datalake_outdated"):
            datalake_outdated_badge.set_visibility(state.store.is_datalake_outdated())

        # Consistent, distinct color palette for documents
        DOC_PALETTE = [
            '#2dd4bf',  # Teal
            '#38bdf8',  # Sky blue
            '#f43f5e',  # Rose
            '#a855f7',  # Purple
            '#fbbf24',  # Amber
            '#34d399',  # Emerald
            '#fb923c',  # Orange
            '#818cf8',  # Indigo
            '#ec4899',  # Pink
            '#4ade80',  # Green
            '#60a5fa',  # Blue
            '#e879f9',  # Fuchsia
            '#facc15',  # Yellow
            '#94a3b8'   # Slate
        ]

        with datalake_inspector_container:
            with ui.row().classes('w-full justify-between items-center border-b border-slate-700/80 pb-2 mb-2'):
                with ui.row().classes('items-center gap-1.5'):
                    ui.icon('radar', size='sm').classes('text-teal-400')
                    ui.label('Semantic Inspector').classes('text-base font-bold text-slate-200')
                inspector_badge = ui.badge('Idle', color='slate-700').classes('text-[10px] font-mono')

            inspector_body = ui.column().classes('w-full space-y-3 text-xs font-mono text-slate-300 flex-1')
            with inspector_body:
                with ui.column().classes('items-center justify-center h-48 text-center text-slate-500 m-auto space-y-2'):
                    ui.icon('touch_app', size='lg').classes('text-slate-600')
                    ui.label('Click any chunk dot or ⌖ diamond to inspect semantic details and distances.').classes('text-xs italic max-w-xs')

        with datalake_plot_container:
            plot_loading = ui.row().classes('w-full h-full items-center justify-center space-x-3')
            with plot_loading:
                ui.spinner('dots', size='lg', color='teal')
                ui.label(f'Calculating {method.upper()} manifold projection in worker thread...').classes('text-sm text-slate-300 font-mono')

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
                ui.label('No vectorized points found. Index documents to view semantic point cloud.').classes('text-sm text-slate-400 italic m-auto')
                return

            # Group points by document
            doc_groups: Dict[str, List[Dict[str, Any]]] = {}
            for p in points:
                doc = p['document_title']
                if doc not in doc_groups:
                    doc_groups[doc] = []
                doc_groups[doc].append(p)

            datalake_stats_label.text = f"{len(points)} chunks across {len(doc_groups)} document(s) projected via {method.upper()} ({n_comp}D)."

            fig = go.Figure()

            # Iterate through each document and assign an identical, shared color for both chunks and CoG
            for doc_idx, (doc_title, pts) in enumerate(doc_groups.items()):
                doc_color = DOC_PALETTE[doc_idx % len(DOC_PALETTE)]

                x_vals = [p['x'] for p in pts]
                y_vals = [p['y'] for p in pts]

                cog_x = float(sum(x_vals) / len(x_vals))
                cog_y = float(sum(y_vals) / len(y_vals))

                # Compute distances of each chunk to the document's center of gravity
                if n_comp == 3:
                    z_vals = [p['z'] for p in pts]
                    cog_z = float(sum(z_vals) / len(z_vals))
                    chunk_dists = [float(np.sqrt((p['x'] - cog_x)**2 + (p['y'] - cog_y)**2 + (p['z'] - cog_z)**2)) for p in pts]
                else:
                    cog_z = 0.0
                    z_vals = None
                    chunk_dists = [float(np.sqrt((p['x'] - cog_x)**2 + (p['y'] - cog_y)**2)) for p in pts]

                avg_dist = float(sum(chunk_dists) / len(chunk_dists)) if chunk_dists else 0.0
                max_dist = max(chunk_dists) if chunk_dists else 0.0

                # Build custom data payloads for direct, unambiguous inspector access
                chunk_customdata = []
                for p, dist in zip(pts, chunk_dists):
                    chunk_customdata.append({
                        'is_cog': False,
                        'doc_title': doc_title,
                        'doc_path': p.get('document_path', ''),
                        'doc_id': p.get('doc_id'),
                        'chunk_id': p['chunk_id'],
                        'x': p['x'],
                        'y': p['y'],
                        'z': p.get('z'),
                        'text': p.get('chunk_text_preview', ''),
                        'metadata': p.get('metadata', {}),
                        'dist_to_cog': round(dist, 3),
                        'color': doc_color
                    })

                cog_customdata = [{
                    'is_cog': True,
                    'doc_title': doc_title,
                    'doc_path': pts[0].get('document_path', ''),
                    'doc_id': pts[0].get('doc_id'),
                    'x': round(cog_x, 3),
                    'y': round(cog_y, 3),
                    'z': round(cog_z, 3) if n_comp == 3 else None,
                    'chunk_count': len(pts),
                    'avg_dist': round(avg_dist, 3),
                    'max_dist': round(max_dist, 3),
                    'color': doc_color
                }]

                hover_template = (
                    f"<b>{doc_title}</b><br>" +
                    "Chunk #%{customdata.chunk_id}<br>" +
                    "Dist to CoG: %{customdata.dist_to_cog}<extra></extra>"
                )

                # 1. Document Chunk Points Trace (Colored with doc_color)
                if n_comp == 3:
                    fig.add_trace(go.Scatter3d(
                        x=x_vals, y=y_vals, z=z_vals,
                        mode='markers',
                        name=doc_title,
                        legendgroup=doc_title,
                        customdata=chunk_customdata,
                        hovertemplate=hover_template,
                        marker=dict(
                            size=6,
                            color=doc_color,
                            opacity=0.85,
                            line=dict(color='#0f172a', width=0.5)
                        )
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode='markers',
                        name=doc_title,
                        legendgroup=doc_title,
                        customdata=chunk_customdata,
                        hovertemplate=hover_template,
                        marker=dict(
                            size=8,
                            color=doc_color,
                            opacity=0.85,
                            line=dict(color='#0f172a', width=0.5)
                        )
                    ))

                # 2. Document Center of Gravity (Centroid) Trace:
                # MATCHES THE EXACT SAME doc_color AS THE CHUNKS OF THIS DOCUMENT!
                if show_cog:
                    cog_hover = (
                        f"<b>⌖ Center of Gravity: {doc_title}</b><br>" +
                        f"Chunks in Doc: {len(pts)}<br>" +
                        f"Mean Radius: {avg_dist:.3f}<extra></extra>"
                    )

                    text_list = [f"⌖ {doc_title}"] if show_labels else None
                    cog_mode = 'markers+text' if show_labels else 'markers'

                    if n_comp == 3:
                        fig.add_trace(go.Scatter3d(
                            x=[cog_x], y=[cog_y], z=[cog_z],
                            mode=cog_mode,
                            name=f"⌖ {doc_title}",
                            legendgroup=doc_title,
                            showlegend=False,
                            text=text_list,
                            textposition="top center",
                            textfont=dict(size=10, color='#f1f5f9', family='monospace'),
                            customdata=cog_customdata,
                            hovertemplate=cog_hover,
                            marker=dict(
                                size=12,
                                symbol='diamond',
                                color=doc_color,  # <-- SAME COLOR AS CHUNKS!
                                line=dict(color='#ffffff', width=2)  # White border makes it pop
                            )
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=[cog_x], y=[cog_y],
                            mode=cog_mode,
                            name=f"⌖ {doc_title}",
                            legendgroup=doc_title,
                            showlegend=False,
                            text=text_list,
                            textposition="top center",
                            textfont=dict(size=10, color='#f1f5f9', family='monospace'),
                            customdata=cog_customdata,
                            hovertemplate=cog_hover,
                            marker=dict(
                                size=15,
                                symbol='diamond',
                                color=doc_color,  # <-- SAME COLOR AS CHUNKS!
                                line=dict(color='#ffffff', width=2)  # White border makes it pop
                            )
                        ))

            # Enterprise Dark Theme Styling for Gridlines and Background
            grid_style = dict(
                gridcolor='#1e293b',      # Subtle dark slate grid line
                zerolinecolor='#334155',  # Slightly accented center line
                showgrid=True,
                zeroline=True,
                tickfont=dict(color='#94a3b8', size=10)
            )

            layout_kwargs = dict(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15, 23, 42, 0.4)',
                font=dict(color='#94a3b8', family='monospace'),
                margin=dict(l=20, r=20, t=20, b=40),
                legend=dict(
                    orientation='h',
                    y=-0.12,
                    x=0.0,
                    xanchor='left',
                    font=dict(size=10, color='#cbd5e1')
                ),
                hovermode='closest'
            )

            if n_comp == 3:
                layout_kwargs['scene'] = dict(
                    xaxis=grid_style,
                    yaxis=grid_style,
                    zaxis=grid_style,
                    bgcolor='rgba(15, 23, 42, 0.4)'
                )
            else:
                layout_kwargs['xaxis'] = grid_style
                layout_kwargs['yaxis'] = grid_style

            fig.update_layout(**layout_kwargs)

            plot = ui.plotly(fig).classes('w-full h-full')

            def on_point_click(e):
                try:
                    if not e.args or 'points' not in e.args or not e.args['points']:
                        return
                    pt_entry = e.args['points'][0]
                    cdata = pt_entry.get('customdata')
                    if not cdata:
                        return

                    inspector_body.clear()
                    is_cog = cdata.get('is_cog', False)
                    doc_color = cdata.get('color', '#14b8a6')

                    if is_cog:
                        inspector_badge.text = "Center of Gravity"
                        inspector_badge.props('color=amber-500')
                        with inspector_body:
                            # Header Bar
                            with ui.row().classes('w-full justify-between items-center bg-slate-900/90 p-2.5 rounded-xl border border-slate-700/80'):
                                with ui.row().classes('items-center gap-2 flex-1 truncate'):
                                    ui.element('span').classes('w-3.5 h-3.5 rounded-sm rotate-45').style(f'background-color: {doc_color}; border: 1.5px solid white;')
                                    ui.label(cdata['doc_title']).classes('font-bold text-sm text-slate-100 truncate flex-1')
                                ui.badge(f"{cdata['chunk_count']} chunks", color='slate-700').classes('text-[10px]')

                            # Coordinates & Distribution Box
                            with ui.column().classes('w-full bg-slate-900/60 p-3 rounded-xl border border-slate-700/60 space-y-1.5'):
                                ui.label('Centroid Coordinates:').classes('text-[10px] text-teal-400 font-bold uppercase tracking-wider')
                                coord_text = f"X: {cdata['x']:.3f} | Y: {cdata['y']:.3f}" + (f" | Z: {cdata['z']:.3f}" if cdata.get('z') is not None else "")
                                ui.label(coord_text).classes('text-xs text-slate-200 font-bold')

                                ui.separator().classes('border-slate-800')
                                with ui.row().classes('w-full justify-between text-xs text-slate-300'):
                                    ui.label('Mean Chunk Spread:')
                                    ui.label(f"{cdata['avg_dist']:.3f} units").classes('text-teal-300 font-bold')
                                with ui.row().classes('w-full justify-between text-xs text-slate-300'):
                                    ui.label('Max Chunk Radius:')
                                    ui.label(f"{cdata['max_dist']:.3f} units").classes('text-amber-400 font-bold')

                            with ui.column().classes('w-full gap-0.5'):
                                ui.label('Document Origin:').classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                                ui.label(cdata['doc_path']).classes('text-[11px] text-slate-400 break-all bg-slate-900/40 p-2 rounded')

                            # Quick Actions Bridge
                            with ui.column().classes('w-full gap-2 pt-2 border-t border-slate-700/60'):
                                def _search_doc():
                                    query_input.value = cdata['doc_title'].split('(')[0].strip()
                                    tabs.value = tab_search

                                ui.button('Search This Document in RAG', icon='search', on_click=_search_doc).props('color=teal size=xs class="w-full"')

                    else:
                        inspector_badge.text = "Document Chunk"
                        inspector_badge.props('color=teal')
                        with inspector_body:
                            with ui.row().classes('w-full justify-between items-center bg-slate-900/90 p-2.5 rounded-xl border border-slate-700/80'):
                                with ui.row().classes('items-center gap-2 flex-1 truncate'):
                                    ui.element('span').classes('w-2.5 h-2.5 rounded-full').style(f'background-color: {doc_color};')
                                    ui.label(cdata['doc_title']).classes('font-bold text-sm text-slate-100 truncate flex-1')
                                ui.badge(f"ID #{cdata['chunk_id']}", color='slate-700').classes('text-[10px]')

                            # Coordinates & Outlier Distance
                            with ui.column().classes('w-full bg-slate-900/60 p-2.5 rounded-xl border border-slate-700/60 space-y-1'):
                                coord_text = f"X: {cdata['x']:.3f} | Y: {cdata['y']:.3f}" + (f" | Z: {cdata['z']:.3f}" if cdata.get('z') is not None else "")
                                ui.label(coord_text).classes('text-xs text-slate-200 font-bold')
                                with ui.row().classes('w-full justify-between text-xs text-slate-300'):
                                    ui.label('Distance to Document CoG:')
                                    ui.label(f"{cdata['dist_to_cog']:.3f} units").classes('text-teal-300 font-bold')

                            # Chunk Content Preview
                            with ui.column().classes('w-full space-y-1'):
                                with ui.row().classes('w-full justify-between items-center'):
                                    ui.label('Chunk Text Preview:').classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                                    ui.button(icon='content_copy', on_click=lambda t=cdata.get('text', ''): asyncio.create_task(copy_to_clipboard(t, "Copied chunk text!"))).props('flat dense size=xs text-color=teal')

                                with ui.scroll_area().classes('w-full max-h-48 p-2.5 bg-slate-900 rounded-xl border border-slate-700/80 text-[11px] whitespace-pre-wrap text-slate-200'):
                                    ui.label(cdata.get('text') or '(No text preview available)')

                            if cdata.get('metadata'):
                                with ui.column().classes('w-full gap-0.5'):
                                    ui.label('Metadata:').classes('text-[10px] text-slate-400 font-bold uppercase tracking-wider')
                                    ui.label(json.dumps(cdata['metadata'])).classes('text-[10px] text-slate-400 break-all bg-slate-900/40 p-1.5 rounded')

                except Exception as err:
                    ASCIIColors.warning(f"Datalake inspector click error: {err}")

            plot.on('plotly_click', on_point_click)

    async def refresh_graph_view():
        if not state.graph_store:
            graph_summary_badge.text = "No GraphStore"
            graph_summary_badge.props('color=grey')
            return

        try:
            scope_raw = str(graph_scope_select.value or '').lower()
            info = await asyncio.to_thread(state.graph_store.get_graph_info)
            tbox_classes_count = len(state.graph_store.tbox.get_classes())

            # Update left sidebar metrics
            left_metrics_box.clear()
            with left_metrics_box:
                ui.label(f"• Total ABox Nodes : {info['total_nodes']}").classes('text-slate-200 font-bold')
                ui.label(f"• Total ABox Edges : {info['total_relationships']}").classes('text-slate-200')
                ui.label(f"• TBox Classes     : {tbox_classes_count}").classes('text-purple-300 font-bold')
                ui.label(f"• Provenance Links : {info['total_provenance_links']}").classes('text-slate-200')

            # Robust scope matching (handles both key and display string)
            is_tbox_mode = 'tbox' in scope_raw
            is_both_mode = 'both' in scope_raw or 'unified' in scope_raw

            # Toggle toolbar button visibility according to scope
            btn_add_class.set_visibility(is_tbox_mode)
            btn_add_obj_prop.set_visibility(is_tbox_mode)
            btn_add_attr.set_visibility(is_tbox_mode)

            btn_add_node.set_visibility(not is_tbox_mode)
            btn_add_rel.set_visibility(not is_tbox_mode)
            btn_build_kg.set_visibility(not is_tbox_mode)
            btn_import_csv.set_visibility(not is_tbox_mode)

            if is_tbox_mode:
                graph_data = await asyncio.to_thread(state.graph_store.get_tbox_graph)
                graph_summary_badge.text = f"TBox Schema: {len(graph_data['nodes'])} Classes | {len(graph_data['edges'])} Axioms"
                graph_summary_badge.props('color=purple')
            elif is_both_mode:
                graph_data = await asyncio.to_thread(state.graph_store.get_unified_graph, 800)
                graph_summary_badge.text = f"Unified: {len(graph_data['nodes'])} Nodes | {len(graph_data['edges'])} Links"
                graph_summary_badge.props('color=cyan')
            else:
                graph_data = await asyncio.to_thread(state.graph_store.get_abox_graph, 1000)
                graph_summary_badge.text = f"ABox Content: {len(graph_data['nodes'])} Instances | {len(graph_data['edges'])} Facts"
                graph_summary_badge.props('color=teal')

            # Palette for label coloring with string sanitization
            palette = ['#14b8a6', '#06b6d4', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6', '#10b981', '#f97316', '#6366f1', '#e11d48']
            clean_labels = [str(k) for k in info['nodes_by_label'].keys() if k is not None]
            label_colors = {}
            for idx, lbl in enumerate(sorted(clean_labels)):
                label_colors[lbl] = palette[idx % len(palette)]

            # Render category chips in left sidebar
            node_types_filter_container.clear()
            with node_types_filter_container:
                for lbl_raw, count in sorted(info['nodes_by_label'].items(), key=lambda x: x[1], reverse=True):
                    lbl = str(lbl_raw or "Unlabeled")
                    col = label_colors.get(lbl, '#14b8a6')
                    with ui.row().classes('w-full justify-between items-center py-0.5 cursor-pointer hover:bg-slate-700/50 px-1 rounded'):
                        with ui.row().classes('items-center gap-1.5'):
                            ui.element('span').classes('w-2.5 h-2.5 rounded-full').style(f'background-color: {col}')
                            ui.label(lbl).classes('text-slate-200 font-bold')
                        ui.label(str(count)).classes('text-slate-400')

            # Mount vis-network on canvas passing element ID
            div_id = f"c{graph_canvas_div.id}"
            await client.run_javascript(f"window.initSafeStoreGraph('{div_id}', {json.dumps(graph_data['nodes'])}, {json.dumps(graph_data['edges'])})")
        except Exception as e:
            trace_exception(e)
            ui.notify(f"Graph view error: {e}", color='negative')

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

    async def run_document_clustering():
        if not state.store: return
        cluster_run_btn.props('disabled')
        clusters_grid_container.clear()
        with clusters_grid_container:
            ui.spinner('dots', size='lg', color='teal')
            ui.label('Clustering documents and generating thematic summaries...').classes('text-sm text-slate-300 ml-2')

        k_val = int(cluster_k_input.value or 0)
        n_clusters = 'auto' if k_val <= 0 else k_val
        method = cluster_algo_select.value
        with_themes = bool(cluster_themes_switch.value)

        try:
            clusters = await asyncio.to_thread(
                state.store.cluster_documents,
                n_clusters=n_clusters,
                method=method,
                generate_themes=with_themes,
                save_to_store=True
            )
            render_clusters_view(clusters)
            ui.notify(f"Successfully clustered documents into {len(clusters)} theme(s)!", color='positive')
        except Exception as ex:
            clusters_grid_container.clear()
            with clusters_grid_container:
                ui.label(f"Clustering error: {ex}").classes('text-rose-400 text-sm font-mono')
            ui.notify(f"Error: {ex}", color='negative')
        finally:
            cluster_run_btn.props(remove='disabled')

    def render_clusters_view(clusters: List[Dict[str, Any]]):
        clusters_grid_container.clear()
        clusters_summary_bar.clear()

        total_clustered_docs = sum(c.get("document_count", 0) for c in clusters)
        with clusters_summary_bar:
            ui.label(f"Thematic Groups: {len(clusters)} | Total Clustered Documents: {total_clustered_docs}").classes('font-bold text-teal-300')
            ui.button('Re-Cluster', icon='refresh', on_click=run_document_clustering).props('flat text-color=teal size=xs')

        if not clusters:
            with clusters_grid_container:
                ui.label('No documents available to cluster. Add documents to the store first.').classes('text-sm text-slate-400 italic col-span-full')
            return

        palette = ['#14b8a6', '#06b6d4', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6', '#10b981', '#f97316']

        with clusters_grid_container:
            for idx, c in enumerate(clusters):
                border_color = palette[idx % len(palette)]
                with ui.card().classes('bg-slate-800/90 rounded-2xl p-5 border shadow-xl flex flex-col justify-between space-y-4 hover:border-teal-400 transition-all duration-300').style(f'border-color: {border_color}88;'):
                    # Top Header
                    with ui.column().classes('w-full gap-2'):
                        with ui.row().classes('w-full justify-between items-center'):
                            with ui.row().classes('items-center gap-2'):
                                ui.element('span').classes('w-3 h-3 rounded-full').style(f'background-color: {border_color};')
                                ui.label(f"Theme #{c['cluster_id'] + 1}").classes('text-xs font-bold font-mono text-slate-400 uppercase tracking-wider')
                            ui.badge(f"{c['document_count']} doc(s)", color='slate-700').classes('text-xs font-mono font-bold')

                        ui.label(c.get('theme_title', 'Unnamed Theme')).classes('text-lg font-bold text-slate-100 leading-snug break-words')
                        if c.get('theme_description'):
                            ui.label(c['theme_description']).classes('text-xs text-slate-300 leading-relaxed bg-slate-900/60 p-2.5 rounded-lg border border-slate-700/50')

                        # Key topics chips
                        if c.get('key_topics'):
                            with ui.row().classes('gap-1 flex-wrap pt-1'):
                                for topic in c['key_topics']:
                                    ui.badge(f"#{topic}", color='teal').classes('text-[10px] font-mono')

                    # Member Documents List
                    with ui.column().classes('w-full pt-3 border-t border-slate-700/60 gap-1'):
                        ui.label('Member Documents:').classes('text-[11px] font-bold text-slate-400 font-mono uppercase tracking-wider')
                        with ui.scroll_area().classes('w-full max-h-32 text-xs font-mono'):
                            for doc in c.get('documents', []):
                                with ui.row().classes('w-full justify-between items-center py-1 border-b border-slate-800/60'):
                                    ui.label(doc.get('document_title', 'Untitled')).classes('text-teal-300 truncate flex-1')
                                    ui.label(f"ID #{doc.get('doc_id')}").classes('text-[10px] text-slate-500 ml-2')

                    # Quick Action: Query in RAG Studio
                    def _query_theme(title=c.get('theme_title', '')):
                        query_input.value = title
                        tabs.value = tab_search

                    ui.button('Query in Search Studio', icon='search', on_click=_query_theme).props('outline color=teal size=xs class="w-full mt-2"')

    def clear_agentic_chat():
        state.chat_history.clear()
        chat_messages_container.clear()
        chat_sources_drawer.clear()
        chat_source_badge.text = "0 Sources"
        with chat_sources_drawer:
            with ui.column().classes('items-center justify-center h-48 text-center text-slate-500 m-auto space-y-2'):
                ui.icon('source', size='lg').classes('text-slate-600')
                ui.label('Retrieved chunk citations and SPARQL facts will appear here as the agent recovers knowledge.').classes('text-xs italic max-w-xs')

    async def send_agentic_chat():
        raw_prompt = chat_input.value
        question = (raw_prompt or "").strip()
        if not question or not state.store:
            return

        chat_input.value = ""
        chat_send_btn.props('disabled')

        # 1. Contextual Query Augmentation for Short/Referential Prompts (e.g. "try again")
        search_query = question
        if len(question.split()) <= 4 and state.chat_history:
            last_user_query = ""
            for msg in reversed(state.chat_history):
                if msg["role"] == "user":
                    last_user_query = msg["content"]
                    break
            if last_user_query:
                search_query = f"{last_user_query} {question}"

        # 2. Record User Turn in History
        state.chat_history.append({"role": "user", "content": question})

        # 3. Render User Message in Stream
        with chat_messages_container:
            with ui.row().classes('w-full justify-end'):
                with ui.card().classes('bg-teal-600/90 text-white p-3.5 rounded-2xl max-w-xl shadow-md'):
                    ui.label(question).classes('text-sm whitespace-pre-wrap')

        # 4. Add Thinking Bubble
        with chat_messages_container:
            thinking_card = ui.card().classes('bg-slate-800 border border-slate-700 p-3.5 rounded-2xl max-w-2xl shadow-md space-y-2')
            with thinking_card:
                with ui.row().classes('items-center gap-2 text-teal-400'):
                    ui.spinner('dots', size='sm', color='teal')
                    ui.label('Recovering semantic facts and conversation context...').classes('text-xs font-mono')

        chat_scroll_area.scroll_to(percent=1.0)
        top_k = int(chat_top_k_select.value or 5)
        include_sparql = bool(chat_include_graph.value)

        # 5. Agentic Retrieval Pipeline: Semantic Chunks + Graph Logic
        def _gather_evidence():
            hits = state.store.hybrid_query(
                query_text=search_query,
                top_k=top_k,
                min_relevance_percent=25.0,
                reconstruct_overlapping_chunks=True
            )

            graph_facts = []
            if include_sparql and state.graph_store:
                try:
                    subgraph = state.graph_store.query_graph(search_query, output_mode="graph_only", top_k_nodes=3)
                    if isinstance(subgraph, dict) and subgraph.get("relationships"):
                        for r in subgraph["relationships"][:5]:
                            s_name = r.get("source_node", {}).get("properties", {}).get("name") or r.get("source_node", {}).get("label")
                            t_name = r.get("target_node", {}).get("properties", {}).get("name") or r.get("target_node", {}).get("label")
                            graph_facts.append(f"({s_name}) --[{r.get('type')}]--> ({t_name})")
                except Exception:
                    pass

            return hits, graph_facts

        hits, graph_facts = await asyncio.to_thread(_gather_evidence)

        # 4. Update Sources Drawer
        chat_sources_drawer.clear()
        chat_source_badge.text = f"{len(hits)} Chunk(s) | {len(graph_facts)} Fact(s)"
        with chat_sources_drawer:
            if not hits and not graph_facts:
                ui.label('No matching sources found in knowledge base.').classes('text-xs text-slate-500 italic')
            else:
                if graph_facts:
                    ui.label('Graph Relational Facts:').classes('text-xs font-bold text-cyan-400 font-mono')
                    with ui.column().classes('w-full bg-slate-900/80 p-2.5 rounded-xl border border-slate-700/60 gap-1'):
                        for f in graph_facts:
                            ui.label(f"• {f}").classes('text-[11px] text-cyan-300 font-mono')

                if hits:
                    ui.label('Document Evidence Chunks:').classes('text-xs font-bold text-teal-400 font-mono mt-1')
                    for h_idx, hit in enumerate(hits, 1):
                        with ui.card().classes('w-full bg-slate-900/80 p-3 rounded-xl border border-slate-700/80 space-y-1.5'):
                            with ui.row().classes('w-full justify-between items-center'):
                                ui.badge(f"[{h_idx}] {hit.get('document_title', 'Doc')}", color='teal').classes('text-[10px] font-bold')
                                ui.label(f"{hit.get('relevance_score', 0):.1f}%").classes('text-[10px] font-bold text-teal-300')
                            ui.label(hit.get('chunk_text', '')[:220] + '...').classes('text-[11px] text-slate-300 whitespace-pre-wrap')

        # 6. Synthesize Multi-Turn Answer using Generator
        def _generate_answer():
            if not hits and not graph_facts:
                return "I couldn't find any relevant facts or passages in the database to answer your question."

            context_blocks = []
            for idx, h in enumerate(hits, 1):
                clean_text = h.get('chunk_text', '').replace('\n', ' ')
                context_blocks.append(f"[Source {idx}: '{h.get('document_title', '')}'] {clean_text}")

            if graph_facts:
                context_blocks.append("[Graph Relationships]: " + " | ".join(graph_facts))

            full_context = "\n\n".join(context_blocks)

            # Assemble conversation history (last 6 turns prior to current question)
            history_lines = []
            prior_messages = state.chat_history[:-1]
            if prior_messages:
                for m in prior_messages[-6:]:
                    r = "User" if m["role"] == "user" else "Assistant"
                    snippet = m["content"].replace("\n", " ").strip()
                    history_lines.append(f"{r}: {snippet}")

            history_context = ("\n\nRecent Conversation History:\n" + "\n".join(history_lines)) if history_lines else ""

            system_prompt = (
                "You are an expert autonomous knowledge assistant grounded in the user's SafeStore database. "
                "You have full awareness of the conversation history. "
                "Answer the user's question accurately, taking previous messages into account when asked to clarify, retry, or expand. "
                "Base your response on the provided grounding sources and graph facts, and explicitly cite your sources "
                "using tags like [Source 1] or [Source 2] next to your claims."
            )

            user_prompt = f"""Grounding Knowledge Sources:
{full_context}{history_context}

Current User Message: {question}

Sourced Answer:"""

            llm = state.llm_generator or getattr(state.store, "llm_generator", None)
            if llm:
                try:
                    return llm(prompt=user_prompt, system_prompt=system_prompt, json_mode=False)
                except Exception as ex:
                    return f"Error from LLM generator: {ex}"
            else:
                return (
                    "**[Offline Extraction Preview - Configure LLM in Header for Full AI Synthesis]**\n\n"
                    f"Found **{len(hits)} relevant source chunk(s)** in your documents:\n\n" +
                    "\n".join(f"• **{h.get('document_title')}** (Score: {h.get('relevance_score', 0):.1f}%): \"{h.get('chunk_text', '')[:140]}...\"" for h in hits)
                )

        answer_text = await asyncio.to_thread(_generate_answer)

        # Record assistant answer in conversation history
        state.chat_history.append({"role": "assistant", "content": answer_text})

        # 6. Render Agent Answer in Chat Stream
        thinking_card.clear()
        with thinking_card:
            with ui.column().classes('w-full space-y-2'):
                with ui.row().classes('items-center gap-1.5 text-teal-400'):
                    ui.icon('psychology', size='sm')
                    ui.label('SafeStore Knowledge Agent').classes('text-xs font-bold font-mono uppercase tracking-wider')

                ui.markdown(answer_text).classes('text-sm text-slate-200 leading-relaxed font-sans')

                # Clickable source pills
                if hits:
                    with ui.row().classes('gap-1.5 flex-wrap pt-2 border-t border-slate-700/60'):
                        ui.label('Citations:').classes('text-[10px] text-slate-400 font-mono')
                        for idx, h in enumerate(hits, 1):
                            ui.badge(f"[{idx}] {h.get('document_title', '')}", color='slate-700').classes('text-[10px] font-mono cursor-pointer hover:bg-teal-600')

        chat_send_btn.props(remove='disabled')
        chat_scroll_area.scroll_to(percent=1.0)

    async def refresh_clusters_view():
        if not state.store: return

        # Update outdated banner visibility
        if hasattr(state.store, "is_clusters_outdated"):
            clusters_outdated_banner.set_visibility(state.store.is_clusters_outdated())

        cached = await asyncio.to_thread(state.store.get_document_clusters, True)
        if cached:
            render_clusters_view(cached)
        else:
            clusters_summary_bar.clear()
            clusters_grid_container.clear()
            with clusters_summary_bar:
                ui.label('No clusters computed yet. Click "Cluster Documents" to begin thematic analysis.').classes('text-xs text-slate-400 italic')

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
    # Reactive Tab Switching Listener
    async def on_tab_change(e):
        target = getattr(e, 'value', None)
        if target is None and hasattr(e, 'args'):
            target = e.args
        if target is None:
            target = tabs.value

        target_str = str(target or '')
        target_name = getattr(target, 'name', target) if not isinstance(target, str) else target

        if 'Graph' in target_str or target == tab_graph or 'graph' in str(target_name).lower():
            await asyncio.sleep(0.05)
            await refresh_graph_view()
            await client.run_javascript('setTimeout(() => { if (window.safeStoreNetwork) { window.safeStoreNetwork.redraw(); window.safeStoreNetwork.fit({animation: true}); } }, 250);')
        elif 'Datalake' in target_str or target == tab_datalake or 'datalake' in str(target_name).lower():
            await refresh_datalake_view()
        elif 'Clusters' in target_str or target == tab_clusters or 'clusters' in str(target_name).lower():
            await refresh_clusters_view()
        elif 'Chat' in target_str or target == tab_chat or 'chat' in str(target_name).lower():
            pass
        elif 'Diagnostics' in target_str or target == tab_diagnostics or 'diagnostics' in str(target_name).lower():
            refresh_diagnostics_view()
        elif 'Files' in target_str or target == tab_files or 'files' in str(target_name).lower():
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

        # Check if database is encrypted and challenges password
        info = inspect_store_summary(Path(db_path_str))
        if info.get("is_encrypted") and not getattr(state, "active_encryption_key", None):
            loading_overlay.close()
            prompt_unlock_dialog(db_path_str)
            return

        try:
            active_key = getattr(state, "active_encryption_key", None)
            await asyncio.to_thread(state.open_store, db_path_str, active_key)
            db_label.text = f"DB: {Path(state.db_path).name}"
            projects_container.set_visibility(False)
            workspace_container.set_visibility(True)
            btn_all_stores.set_visibility(True)
            tabs.value = tab_files
            refresh_files_view()
            refresh_diagnostics_view()
            asyncio.create_task(refresh_graph_view())
        except Exception as ex:
            ui.notify(f"Failed to open store: {ex}", color='negative', duration=7000)
            show_projects_view()
        finally:
            loading_overlay.close()

    # Password Challenge Dialog for Encrypted Stores
    with ui.dialog() as unlock_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[32rem] border border-amber-500/60 space-y-4 shadow-2xl'):
        with ui.row().classes('items-center gap-2'):
            ui.icon('lock', size='md').classes('text-amber-400')
            ui.label('Encrypted Database').classes('text-xl font-bold text-amber-300')
        ui.label('This SafeStore is protected with authenticated encryption. Enter the passphrase to unlock.').classes('text-xs text-slate-400')
        unlock_target_path = ui.input().classes('hidden')
        unlock_pw_input = ui.input('Decryption Passphrase', placeholder='Enter passphrase').classes('w-full').props('dark standout type=password')

        async def execute_unlock():
            t_path = unlock_target_path.value
            pw = unlock_pw_input.value.strip()
            if not pw:
                ui.notify("Passphrase is required.", color='warning')
                return
            unlock_dialog.close()

            # Test password validity
            try:
                def _test():
                    test_s = SafeStore(db_path=t_path, encryption_key=pw)
                    docs = test_s.list_documents()
                    test_s.close()

                await asyncio.to_thread(_test)
                state.active_encryption_key = pw
                await switch_to_store_workspace(t_path)
            except Exception:
                ui.notify("Decryption failed: Invalid passphrase or corrupted store.", color='negative', duration=6000)

        unlock_pw_input.on('keydown.enter', execute_unlock)
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=unlock_dialog.close).props('flat text-color=grey')
            ui.button('Unlock Database', icon='lock_open', on_click=execute_unlock).props('color=amber-500 text-color=black font-bold')

    def prompt_unlock_dialog(db_path_str: str):
        unlock_target_path.value = db_path_str
        unlock_pw_input.value = ""
        unlock_dialog.open()

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

                                async def export_card_db(info=info):
                                    dest_dir = Path.home() / "Downloads"
                                    dest = dest_dir / f"exported_{info['filename']}"
                                    try:
                                        tmp = SafeStore(db_path=info['path'])
                                        await asyncio.to_thread(tmp.export_db_file, dest)
                                        tmp.close()
                                        ui.notify(f"Exported to '{dest}'!", color='positive', icon='file_download')
                                    except Exception as ex:
                                        ui.notify(f"Export error: {ex}", color='negative')

                                ui.button(icon='file_download', on_click=lambda i=info: export_card_db(i)).props('flat round dense size=sm text-color=teal title="Export DB file"')
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
    await render_studio_page(
        client,
        CURRENT_DB_PATH,
        llm_generator=CURRENT_LLM_GENERATOR,
        lollms_client=CURRENT_LOLLMS_CLIENT
    )


def launch_studio(
    db_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    native: bool = True,
    llm_generator: Optional[Callable[..., str]] = None,
    lollms_client: Optional[Any] = None
):
    """Launches SafeStore Studio via NiceGUI and pywebview with an optional custom LLM generator."""
    global CURRENT_DB_PATH, CURRENT_LLM_GENERATOR, CURRENT_LOLLMS_CLIENT
    CURRENT_DB_PATH = str(Path(db_path).resolve()) if db_path else None
    CURRENT_LLM_GENERATOR = llm_generator
    CURRENT_LOLLMS_CLIENT = lollms_client

    # Pre-check pywebview availability for native window and configure maximized startup
    if native:
        try:
            import webview
            if hasattr(app, 'native') and hasattr(app.native, 'window_args'):
                app.native.window_args['maximized'] = True
        except (ImportError, AttributeError):
            print("[!] pywebview not installed. Running in browser mode...")
            native = False

    try:
        ui.run(
            title="SafeStore Studio",
            host=host,
            port=port,
            native=native,
            window_size=(1920, 1080),
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