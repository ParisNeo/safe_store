# [FINAL & COMPLETE] server.py
import shutil
from pathlib import Path
import asyncio
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import json
import time
import uuid
import argparse

from safe_store import SafeStore, GraphStore
from safe_store.core.exceptions import NodeNotFoundError, RelationshipNotFoundError, GraphError
from lollms_client import LollmsClient
from lollms_client.lollms_llm_binding import get_available_bindings
from ascii_colors import ASCIIColors

app = FastAPI()
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)
CONFIG_FILE = Path("config.json")

PROJECTS: Dict[str, SafeStore] = {}
GRAPH_STORES: Dict[str, GraphStore] = {}
TASK_MANAGER: Dict[str, Dict] = {}
LLM_CLIENT: Optional[LollmsClient] = None

class ProjectCreate(BaseModel): 
    name: str
    description: Optional[str] = ""

class LLMSettings(BaseModel): binding_name: str; config: Dict[str, Any]
class OntologyUpdate(BaseModel): ontology: Dict[str, Any]
class QueryRequest(BaseModel): question: str
class NodeCreate(BaseModel): label: str; properties: Dict[str, Any]
class NodeUpdate(BaseModel): label: Optional[str] = None; properties: Optional[Dict[str, Any]] = None
class RelationshipCreate(BaseModel): source_node_id: int; target_node_id: int; rel_type: str; properties: Optional[Dict[str, Any]] = None

def load_config() -> Dict:
    if CONFIG_FILE.exists():
        try: return json.loads(CONFIG_FILE.read_text())
        except json.JSONDecodeError: pass
    return {"binding_name": "ollama", "config": {"model_name": "mistral:latest"}}

def save_config(settings: LLMSettings):
    global LLM_CLIENT
    CONFIG_FILE.write_text(settings.json(indent=2))
    LLM_CLIENT = None

def get_lollms_client(force_reload: bool = False) -> LollmsClient:
    global LLM_CLIENT
    if LLM_CLIENT is not None and not force_reload: return LLM_CLIENT
    config = load_config()
    ASCIIColors.info(f"Initializing LollmsClient with binding: {config['binding_name']}")
    try:
        client = LollmsClient(llm_binding_name=config["binding_name"], llm_binding_config=config["config"])
        if not client.llm: raise Exception("LLM binding failed to load or model not found.")
        LLM_CLIENT = client
        ASCIIColors.success("LollmsClient initialized successfully.")
        return LLM_CLIENT
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient: {e}")
        LLM_CLIENT = None; raise

def llm_executor_callback(prompt: str) -> str:
    return get_lollms_client().generate_code(prompt, language="json", temperature=0.1)

def create_task(name: str, project_name: str) -> str:
    task_id = str(uuid.uuid4())
    TASK_MANAGER[task_id] = {"id": task_id, "name": name, "status": "pending", "project": project_name, "progress": 0, "logs": [f"Task '{name}' created."], "start_time": time.time()}
    return task_id

def update_task_progress(task_id: str, progress: float, log_message: str):
    if task_id in TASK_MANAGER:
        TASK_MANAGER[task_id]["progress"] = round(progress, 2)
        TASK_MANAGER[task_id]["logs"].append(log_message)
        TASK_MANAGER[task_id]["status"] = "running"


def finish_task(task_id: str, status: str, final_message: str):
    if task_id in TASK_MANAGER:
        TASK_MANAGER[task_id].update({"status": status, "progress": 100, "end_time": time.time()})
        TASK_MANAGER[task_id]["logs"].append(final_message)

def _build_graph_worker(project_name: str, task_id: str):
    try:
        graph_store = get_graph_store(project_name)
        with graph_store.store:
            def progress_callback(progress, message): update_task_progress(task_id, progress * 100, message)
            update_task_progress(task_id, 5, "Starting graph build process...")
            graph_store.build_graph_for_all_documents(progress_callback=progress_callback)
            finish_task(task_id, "complete", "Graph building finished successfully.")
    except Exception as e:
        ASCIIColors.error(f"Task {task_id} failed: {e}")
        finish_task(task_id, "failed", f"Error: {e}")

def _rebuild_graph_worker(project_name: str, task_id: str):
    try:
        graph_store = get_graph_store(project_name)
        with graph_store.store:
            def progress_callback(progress, message):
                scaled_progress = 10 + (progress * 90)
                update_task_progress(task_id, scaled_progress, message)

            update_task_progress(task_id, 0, "Clearing existing graph...")
            graph_store.clear_graph()
            
            update_task_progress(task_id, 5, "Re-initializing graph schema...")
            graph_store._initialize_graph_features()

            update_task_progress(task_id, 10, "Graph cleared. Starting rebuild process...")
            graph_store.build_graph_for_all_documents(progress_callback=progress_callback)
            finish_task(task_id, "complete", "Graph rebuild finished successfully.")
    except Exception as e:
        ASCIIColors.error(f"Task {task_id} (rebuild) failed: {e}")
        finish_task(task_id, "failed", f"Error during rebuild: {e}")

def _add_document_worker(project_name: str, file_path: Path, task_id: str):
    try:
        store = get_project_store(project_name)
        with store:
            update_task_progress(task_id, 10, f"Indexing document: {file_path.name}")
            store.add_document(file_path)
            update_task_progress(task_id, 90, "Document indexed. Invalidate graph store for next build.")
            if project_name in GRAPH_STORES: del GRAPH_STORES[project_name]
            finish_task(task_id, "complete", "Document processing finished.")
    except Exception as e:
        ASCIIColors.error(f"Task {task_id} failed: {e}")
        finish_task(task_id, "failed", f"Error: {e}")

def get_project_store(project_name: str) -> SafeStore:
    if project_name not in PROJECTS:
        db_path = PROJECTS_DIR / f"{project_name}.db"
        if not db_path.exists(): raise HTTPException(status_code=404, detail="Project not found.")
        PROJECTS[project_name] = SafeStore(db_path=db_path)
    return PROJECTS[project_name]

def get_graph_store(project_name: str) -> GraphStore:
    if project_name not in GRAPH_STORES:
        store = get_project_store(project_name)
        properties = store.get_properties() or {}
        metadata = properties.get("metadata") or {}
        ontology = metadata.get("ontology") or {}
        GRAPH_STORES[project_name] = GraphStore(store=store, llm_executor_callback=llm_executor_callback, ontology=ontology)
    return GRAPH_STORES[project_name]

@app.get("/", response_class=FileResponse)
async def get_index(): return "index.html"
@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return Response(status_code=204)
@app.get("/api/llm/status")
async def get_llm_status():
    try: get_lollms_client(); return {"status": "ok"}
    except Exception as e: return {"status": "error", "message": str(e)}
@app.get("/api/settings/llm-bindings")
async def list_llm_bindings(): return [{"name": b["binding_name"], "parameters": b.get("input_parameters", [])} for b in get_available_bindings()]
@app.get("/api/settings")
async def get_settings(): return load_config()
@app.post("/api/settings")
async def set_settings(settings: LLMSettings):
    save_config(settings)
    try:
        get_lollms_client(force_reload=True)
        return {"status": "ok", "message": "Settings saved and LLM connected successfully."}
    except Exception as e: raise HTTPException(status_code=400, detail=f"Failed to connect with new settings: {e}")

@app.get("/api/tasks")
async def get_tasks(): return list(TASK_MANAGER.values())

@app.get("/api/projects")
async def list_projects():
    project_details = []
    db_files = sorted(list(PROJECTS_DIR.glob("*.db")))
    for p in db_files:
        project_name = p.stem
        try:
            store = SafeStore(db_path=p)
            with store:
                props = store.get_properties() or {}
                description = props.get("description", "")
                project_details.append({"name": project_name, "description": description})
        except Exception as e:
            ASCIIColors.warning(f"Could not read properties for project {project_name}: {e}")
            project_details.append({"name": project_name, "description": "Error reading project details."})
    return project_details

@app.post("/api/projects")
async def create_project(project: ProjectCreate):
    project_name = project.name.strip().lower().replace(" ", "_")
    if not project_name: raise HTTPException(status_code=400, detail="Project name cannot be empty.")
    db_path = PROJECTS_DIR / f"{project_name}.db"
    if db_path.exists(): raise HTTPException(status_code=409, detail="Project already exists.")
    
    with SafeStore(db_path=db_path) as store:
        if project.description:
            store.update_properties({"description": project.description})
    
    return {"status": "created", "name": project_name, "description": project.description}

@app.delete("/api/projects/{project_name}", status_code=204)
async def delete_project(project_name: str):
    db_path = PROJECTS_DIR / f"{project_name}.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Project not found.")

    ASCIIColors.warning(f"Attempting to delete project: {project_name}")
    try:
        # Close any open connections before deleting
        if project_name in PROJECTS:
            PROJECTS[project_name].close()

        # Delete from in-memory dicts
        PROJECTS.pop(project_name, None)
        GRAPH_STORES.pop(project_name, None)

        # Delete database files
        db_path.unlink(missing_ok=True)
        (PROJECTS_DIR / f"{project_name}.db-shm").unlink(missing_ok=True)
        (PROJECTS_DIR / f"{project_name}.db-wal").unlink(missing_ok=True)
        
        # Delete associated documents folder
        docs_dir = PROJECTS_DIR / project_name
        if docs_dir.exists() and docs_dir.is_dir():
            shutil.rmtree(docs_dir)

        ASCIIColors.success(f"Successfully deleted project: {project_name}")
        return Response(status_code=204)
    except Exception as e:
        ASCIIColors.error(f"Error deleting project {project_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not delete project files: {e}")


@app.post("/api/projects/{project_name}/documents")
async def upload_document(project_name: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    docs_dir = PROJECTS_DIR / project_name / "docs"; docs_dir.mkdir(parents=True, exist_ok=True)
    file_path = docs_dir / file.filename
    with file_path.open("wb") as buffer: shutil.copyfileobj(file.file, buffer)
    task_id = create_task(f"Index document: {file.filename}", project_name)
    background_tasks.add_task(_add_document_worker, project_name, file_path, task_id)
    return {"status": "task_started", "task_id": task_id}

@app.get("/api/projects/{project_name}/documents")
async def list_project_documents(project_name: str):
    try:
        store = get_project_store(project_name)
        with store:
            documents = store.list_documents()
            for doc in documents:
                doc['filename'] = Path(doc['file_path']).name
            return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

@app.delete("/api/projects/{project_name}/documents/{doc_id}", status_code=204)
async def delete_project_document(project_name: str, doc_id: int):
    try:
        store = get_project_store(project_name)
        graph_store = get_graph_store(project_name)
        with store:
            graph_store.remove_graph_elements_for_document(doc_id)
            store.delete_document_by_id(doc_id)
        return Response(status_code=204)
    except Exception as e:
        ASCIIColors.error(f"Failed to delete document {doc_id} from {project_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")

@app.post("/api/projects/{project_name}/build-graph")
async def build_graph(project_name: str, background_tasks: BackgroundTasks):
    task_id = create_task(f"Build graph for: {project_name}", project_name)
    background_tasks.add_task(_build_graph_worker, project_name, task_id)
    return {"status": "task_started", "task_id": task_id}
    
@app.post("/api/projects/{project_name}/rebuild-graph")
async def rebuild_graph(project_name: str, background_tasks: BackgroundTasks):
    task_id = create_task(f"Rebuild graph for: {project_name}", project_name)
    background_tasks.add_task(_rebuild_graph_worker, project_name, task_id)
    return {"status": "task_started", "task_id": task_id}

@app.get("/api/projects/{project_name}/graph")
async def get_graph_data(project_name: str):
    try:
        graph_store = get_graph_store(project_name)
        return {"nodes": graph_store.get_all_nodes_for_visualization(limit=1000), "edges": graph_store.get_all_relationships_for_visualization(limit=2000)}
    except Exception as e:
        ASCIIColors.error(f"Failed to get graph data for {project_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Could not initialize project backend or connect to LLM. Error: {e}")

@app.get("/api/projects/{project_name}/ontology")
async def get_ontology(project_name: str):
    try:
        store = get_project_store(project_name)
        properties = store.get_properties() or {}
        return (properties.get("metadata", {})).get("ontology", {})
    except Exception as e:
        ASCIIColors.error(f"Failed to get ontology for {project_name}: {e}")
        raise HTTPException(status_code=503, detail=f"Could not initialize project backend. Error: {e}")

@app.post("/api/projects/{project_name}/ontology")
async def set_ontology(project_name: str, body: OntologyUpdate):
    store = get_project_store(project_name)
    with store:
        store.update_properties(metadata={"ontology": body.ontology}, overwrite_metadata=True)
        if project_name in GRAPH_STORES: del GRAPH_STORES[project_name]
    return {"status": "Ontology updated."}
    
@app.post("/api/projects/{project_name}/query")
async def query_graph(project_name: str, request: QueryRequest):
    try:
        client = get_lollms_client(); graph_store = get_graph_store(project_name)
        with graph_store.store:
            result = graph_store.query_graph(request.question, output_mode="full")
            context = f"Graph Info: {json.dumps(result.get('graph'))}\n\nText Chunks: {[c['chunk_text'] for c in result.get('chunks', [])]}"
            prompt = f"Answer the question based ONLY on the context.\nContext: {context}\n\nQuestion: {request.question}"
            answer = client.generate_text(prompt, n_predict=512)
            return {"question": request.question, "answer": answer, "context": result}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Query failed: {e}")

@app.post("/api/projects/{project_name}/graph/nodes", status_code=201)
async def add_node(project_name: str, node_data: NodeCreate):
    try:
        gs = get_graph_store(project_name)
        node_id = gs.add_node(label=node_data.label, properties=node_data.properties)
        details = gs.get_node_details(node_id)
        if not details: raise HTTPException(status_code=500, detail="Node created but not found.")
        props = details.get("properties", {})
        return {"id": details["node_id"], "label": details["label"], "title": json.dumps(props, indent=2), "properties": props}
    except GraphError as e: raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/projects/{project_name}/graph/nodes/{node_id}")
async def update_node(project_name: str, node_id: int, node_data: NodeUpdate):
    try:
        gs = get_graph_store(project_name)
        gs.update_node(node_id=node_id, label=node_data.label, properties=node_data.properties)
        details = gs.get_node_details(node_id)
        if not details: raise NodeNotFoundError(f"Node {node_id} not found after update.")
        props = details.get("properties", {})
        return {"id": details["node_id"], "label": details["label"], "title": json.dumps(props, indent=2), "properties": props}
    except NodeNotFoundError as e: raise HTTPException(status_code=404, detail=str(e))
    except GraphError as e: raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_name}/graph/nodes/{node_id}", status_code=204)
async def delete_node(project_name: str, node_id: int):
    try: get_graph_store(project_name).delete_node(node_id=node_id); return Response(status_code=204)
    except NodeNotFoundError as e: raise HTTPException(status_code=404, detail=str(e))
    except GraphError as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_name}/graph/relationships", status_code=201)
async def add_relationship(project_name: str, rel_data: RelationshipCreate):
    try:
        gs = get_graph_store(project_name)
        rel_id = gs.add_relationship(source_node_id=rel_data.source_node_id, target_node_id=rel_data.target_node_id, rel_type=rel_data.rel_type, properties=rel_data.properties)
        return {"id": rel_id, "from": rel_data.source_node_id, "to": rel_data.target_node_id, "label": rel_data.rel_type}
    except GraphError as e:
        if "FOREIGN KEY constraint failed" in str(e).lower(): raise HTTPException(status_code=404, detail="Source or target node not found.")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_name}/graph/relationships/{relationship_id}", status_code=204)
async def delete_relationship(project_name: str, relationship_id: int):
    try: get_graph_store(project_name).delete_relationship(relationship_id=relationship_id); return Response(status_code=204)
    except RelationshipNotFoundError as e: raise HTTPException(status_code=404, detail=str(e))
    except GraphError as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple script to demonstrate parsing host and port."
    )

    parser.add_argument('--host', 
                        type=str, 
                        default='localhost', # Changed default host to 'localhost'
                        help='The host to bind the server to (default: localhost)')

    parser.add_argument('--port', 
                        type=int, 
                        default=9601, 
                        help='The port to listen on (default: 9601)')

    args = parser.parse_args()

    HOST = args.host
    PORT = args.port

    uvicorn.run(app, host=HOST, port=PORT)