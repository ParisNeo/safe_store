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

import safe_store
from safe_store import SafeStore, GraphStore, LogLevel

# Global configured database path for initial page loads
CURRENT_DB_PATH: str = str(Path("safe_store.db").resolve())


async def pick_file_dialog(
    title: str = "Select Document",
    file_types: Optional[List[Tuple[str, str]]] = None
) -> Optional[str]:
    """
    Opens the native operating system file dialog.
    Properly awaits pywebview's async WindowProxy.create_file_dialog when active,
    falling back to Tkinter in a worker thread.
    """
    # 1. Try pywebview native window file dialog if in native mode
    try:
        import webview
        if hasattr(app, 'native') and hasattr(app.native, 'main_window') and app.native.main_window:
            dialog_call = app.native.main_window.create_file_dialog(
                webview.OPEN_DIALOG,
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

    # 2. Universal Tkinter native file dialog (runs in worker thread to prevent event loop blocking)
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

    return await asyncio.to_thread(_pick_tk)


class StudioState:
    def __init__(self):
        self.db_path: str = ""
        self.store: Optional[SafeStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.selected_doc_id: Optional[int] = None

    def open_store(self, path: str):
        if self.store:
            try:
                self.store.close()
            except Exception:
                pass
        self.db_path = str(Path(path).resolve())
        self.store = SafeStore(db_path=self.db_path, log_level=LogLevel.INFO)
        try:
            self.graph_store = GraphStore(store=self.store)
        except Exception:
            self.graph_store = None
        self.selected_doc_id = None


async def render_studio_page(client: Client, initial_path: str):
    state = StudioState()

    # Base styling
    ui.colors(primary='#14b8a6', secondary='#06b6d4', accent='#f59e0b', dark='#0f172a')
    ui.query('body').classes('bg-slate-900 text-slate-100 font-sans')

    # Header / Toolbar (Paints immediately)
    with ui.header().classes('bg-slate-800/95 backdrop-blur border-b border-slate-700 px-6 py-3 flex justify-between items-center z-50'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('hub', size='md').classes('text-teal-400')
            with ui.column().classes('gap-0'):
                ui.label('SafeStore Studio').classes('text-xl font-bold bg-gradient-to-r from-teal-400 to-cyan-400 bg-clip-text text-transparent leading-tight')
                db_label = ui.label(f'DB: {Path(initial_path).name}').classes('text-xs text-slate-400 font-mono')

        with ui.row().classes('items-center gap-2'):
            async def change_db():
                new_path = path_input.value.strip()
                if new_path:
                    db_dialog.close()
                    loading_overlay.set_visibility(True)
                    try:
                        await asyncio.to_thread(state.open_store, new_path)
                        db_label.text = f'DB: {Path(state.db_path).name}'
                        ui.notify(f"Opened database: {Path(new_path).name}", color='positive')
                        await refresh_all_views()
                    except Exception as ex:
                        ui.notify(f"Failed to open store: {ex}", color='negative')
                    finally:
                        loading_overlay.set_visibility(False)

            async def browse_db_file():
                selected = await pick_file_dialog(
                    title="Select SafeStore SQLite Database",
                    file_types=[
                        ("SafeStore SQLite DB (*.db, *.sqlite)", "*.db;*.sqlite;*.sqlite3"),
                        ("All Files (*.*)", "*.*")
                    ]
                )
                if selected:
                    path_input.value = selected

            with ui.dialog() as db_dialog, ui.card().classes('bg-slate-800 text-slate-100 p-6 w-[30rem] border border-slate-700 space-y-4'):
                ui.label('Open / Switch Database').classes('text-lg font-bold text-teal-400')
                with ui.row().classes('w-full items-center gap-2'):
                    path_input = ui.input('Database Path', value=initial_path).classes('flex-1').props('dark standout')
                    ui.button(icon='folder_open', on_click=browse_db_file).props('color=teal size=md title="Browse..."')
                with ui.row().classes('w-full justify-end gap-2 mt-4'):
                    ui.button('Cancel', on_click=db_dialog.close).props('flat text-color=grey')
                    ui.button('Open Database', on_click=change_db).props('color=teal')

            ui.button('Switch Store', icon='folder_open', on_click=db_dialog.open).props('outline color=teal size=sm')

    # Instant Startup Spinner Container (Visible immediately upon page mount)
    startup_loading_container = ui.column().classes('w-full h-[80vh] items-center justify-center')
    with startup_loading_container:
        with ui.card().classes('bg-slate-800/90 border border-slate-700 p-8 items-center text-center shadow-2xl rounded-2xl space-y-4 max-w-md'):
            ui.spinner(size='xl', color='teal', thickness=3)
            with ui.column().classes('gap-1 items-center'):
                ui.label('Initializing SafeStore Engine').classes('text-xl font-bold text-teal-400')
                ui.label(f'Loading model & database: {Path(initial_path).name}').classes('text-xs text-slate-400 font-mono')
                ui.label('Allocating tensor buffers and verifying vectorizer compatibility...').classes('text-xs text-slate-500 italic mt-2')

    # Global Async Loading Overlay for heavy operations
    loading_overlay = ui.dialog().props('persistent')
    with loading_overlay, ui.card().classes('bg-slate-800 border border-slate-700 p-6 items-center text-center space-y-3'):
        ui.spinner(size='lg', color='teal')
        ui.label('Processing in background thread...').classes('text-sm text-slate-300')

    # Main Application Container (Initially hidden until model is ready)
    main_container = ui.column().classes('w-full')
    main_container.set_visibility(False)

    with main_container:
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
            # TAB 3: Knowledge Graph & SPARQL
            # -----------------------------------------------------------------
            with ui.tab_panel(tab_graph):
                with ui.row().classes('w-full justify-between items-center mb-4'):
                    ui.label('Knowledge Graph & W3C SPARQL 1.1 Console').classes('text-2xl font-bold text-teal-400')

                    async def build_graph_with_lollms():
                        if not state.graph_store:
                            ui.notify("GraphStore not initialized on this store.", color='negative')
                            return

                        # Check if LOLLMS is available or can be resolved
                        if not state.graph_store.has_structured_lollms_support():
                            # Try on-demand resolution
                            client = state.graph_store._try_resolve_lollms_client()
                            if client:
                                state.graph_store.lollms_client = client
                            else:
                                ui.notify(
                                    "lollms-client is not detected or LOLLMS server is unreachable. Run: pip install lollms-client",
                                    color='warning',
                                    duration=6000
                                )
                                return

                        loading_overlay.set_visibility(True)
                        try:
                            ui.notify("Extracting structured triplets across documents via LOLLMS...", color='info')
                            stats = await asyncio.to_thread(state.graph_store.build_graph_for_all_documents)
                            ui.notify(
                                f"Graph Build Complete: {stats['nodes_created']} nodes, {stats['relationships_created']} relationships added!",
                                color='positive'
                            )
                            refresh_graph_view()
                        except Exception as e:
                            ui.notify(f"Graph extraction failed: {e}", color='negative')
                        finally:
                            loading_overlay.set_visibility(False)

                    ui.button('Build Graph with LOLLMS', icon='auto_awesome', on_click=build_graph_with_lollms).props('color=teal')

                with ui.row().classes('w-full gap-6 items-start'):
                    with ui.column().classes('w-1/2 space-y-4'):
                        graph_stats_card = ui.card().classes('w-full bg-slate-800/80 p-4 rounded-xl border border-slate-700')
                        sparql_editor_card = ui.card().classes('w-full bg-slate-800/80 p-4 rounded-xl border border-slate-700')

                    with ui.column().classes('w-1/2 space-y-4'):
                        sparql_results_card = ui.card().classes('w-full bg-slate-800/80 p-4 rounded-xl border border-slate-700 min-h-[50vh]')

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
            use_cache=False,
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

    def refresh_graph_view():
        graph_stats_card.clear()
        sparql_editor_card.clear()

        with graph_stats_card:
            ui.label('Graph Store Information').classes('text-lg font-bold text-teal-300 mb-2')
            if not state.graph_store:
                ui.label('GraphStore not initialized on this database.').classes('text-sm text-slate-400 italic')
                return

            info = state.graph_store.get_graph_info()
            with ui.row().classes('gap-4 mb-2'):
                ui.badge(f"Nodes: {info['total_nodes']}", color='teal')
                ui.badge(f"Edges: {info['total_relationships']}", color='cyan')
                ui.badge(f"Provenance Links: {info['total_provenance_links']}", color='indigo')

            ui.label(f"Node Types: {json.dumps(info['nodes_by_label'])}").classes('text-xs text-slate-400 font-mono')
            ui.label(f"Relationship Types: {json.dumps(info['relationships_by_type'])}").classes('text-xs text-slate-400 font-mono')

        with sparql_editor_card:
            ui.label('W3C SPARQL 1.1 Query Runner').classes('text-lg font-bold text-teal-300 mb-2')
            sparql_query_input = ui.textarea(
                label='SPARQL Query',
                value="""PREFIX ex: <http://example.org/>
PREFIX ont: <http://example.org/ontology/>
SELECT ?subject ?predicate ?object WHERE {
    ?subject ?predicate ?object .
} LIMIT 10"""
            ).classes('w-full font-mono text-xs').props('dark standout rows=6')

            async def run_sparql():
                query_str = sparql_query_input.value.strip()
                sparql_results_card.clear()
                with sparql_results_card:
                    ui.label('Query Results').classes('text-lg font-bold text-teal-300 mb-2 border-b border-slate-700 pb-1')
                    try:
                        res = await asyncio.to_thread(state.graph_store.query_sparql, query_str)
                        if "results" in res and "bindings" in res["results"]:
                            bindings = res["results"]["bindings"]
                            vars_list = res.get("head", {}).get("vars", [])
                            if not bindings:
                                ui.label('Empty result set.').classes('text-sm text-slate-400 italic')
                            else:
                                cols = [{'name': v, 'label': v, 'field': v, 'align': 'left'} for v in vars_list]
                                rows = []
                                for b in bindings:
                                    rows.append({v: b.get(v, {}).get('value', '') for v in vars_list})
                                ui.table(columns=cols, rows=rows).classes('w-full').props('dark flat dense')
                        elif "boolean" in res:
                            ui.badge(f"ASK Result: {res['boolean']}", color='positive' if res['boolean'] else 'negative').classes('text-lg p-3')
                        elif "triples" in res:
                            triples = res["triples"]
                            ui.label(f"Constructed {len(triples)} triples:").classes('text-sm text-slate-300 mb-2')
                            with ui.scroll_area().classes('h-64'):
                                for t in triples:
                                    ui.label(f"({t['subject']['value']}) --[{t['predicate']['value']}]--> ({t['object']['value']})").classes('text-xs font-mono text-teal-300')
                    except Exception as e:
                        ui.label(f"SPARQL Error: {e}").classes('text-sm text-rose-400 font-mono')

            ui.button('Execute SPARQL', icon='play_arrow', on_click=run_sparql).props('color=teal')

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
        refresh_graph_view()
        refresh_diagnostics_view()

    # --- Step 1: Immediately Await WebSocket Handshake ---
    # This fulfills the HTTP 200 GET request in <50ms, showing the spinner to the user!
    await client.connected()

    # --- Step 2: Offload Long-Running Model Loading to Worker Thread ---
    try:
        await asyncio.to_thread(state.open_store, initial_path)
        db_label.text = f'DB: {Path(state.db_path).name}'
        startup_loading_container.set_visibility(False)
        main_container.set_visibility(True)
        await refresh_all_views()
    except Exception as err:
        startup_loading_container.clear()
        with startup_loading_container:
            with ui.card().classes('bg-slate-800 border border-rose-500/50 p-6 items-center text-center space-y-3'):
                ui.icon('error', size='lg').classes('text-rose-400')
                ui.label('Failed to Load SafeStore').classes('text-xl font-bold text-rose-400')
                ui.label(str(err)).classes('text-xs text-slate-300 font-mono max-w-lg break-all')


# Register asynchronous root page handler with expanded response timeout
@ui.page('/', response_timeout=60.0)
async def index(client: Client):
    await render_studio_page(client, CURRENT_DB_PATH)


def launch_studio(
    db_path: str = "safe_store.db",
    host: str = "127.0.0.1",
    port: int = 8080,
    native: bool = True
):
    """Launches SafeStore Studio via NiceGUI and pywebview."""
    global CURRENT_DB_PATH
    CURRENT_DB_PATH = str(Path(db_path).resolve())

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
    parser.add_argument("db_path", nargs="?", default="safe_store.db", help="Path to SafeStore SQLite database (.db)")
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