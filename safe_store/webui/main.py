# webui/main.py
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import asyncio
import pipmaster as pm
import sys
pm.ensure_packages([
    "lollms_client",
    "uvicorn",
    "fastapi",
    "python-multipart",
    "toml",
    "pydantic",
    "python-socketio" # Added for explicitness
])
# Ajoute le répertoire parent (safe_store/safe_store/) au chemin de recherche de Python
# Cela permet à Python de trouver config_manager.py
sys.path.insert(0, str(Path(__file__).parent))

# Maintenant, l'importation originale fonctionnera
from config_manager import get_config, get_log_level_from_str, save_config, DATABASES_ROOT
from lollms_client import LollmsClient
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body, Path as FastApiPath, Query, status, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse 
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field # For request/response models
import socketio

import safe_store # Assuming safe_store is in PYTHONPATH or installed
from safe_store import GraphStore, SafeStore, LogLevel as SafeStoreLogLevel # Renamed to avoid conflict
from safe_store.core.exceptions import DocumentNotFoundError # For specific exception handling
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors, trace_exception, LogLevel

# Import the configuration manager
from config_manager import get_config, get_log_level_from_str, save_config # Relative import

# --- Load Configuration ---
config = get_config()

# --- Global variables for the active database ---
# These will be set on startup based on the active_database_name in the config
ACTIVE_DB_FILE: Optional[Path] = None
ACTIVE_DOC_DIR: Optional[Path] = None

# --- Use Configuration Values ---
TEMP_UPLOAD_DIR = Path(config["webui"]["temp_upload_dir"])

# LOLLMS Client Configuration from config
BINDING_NAME = config["lollms"]["binding_name"]
HOST_ADDRESS = config["lollms"]["host_address"]
MODEL_NAME = config["lollms"]["model_name"]
SERVICE_KEY = config["lollms"].get("service_key") 

# WebUI settings
WEBUI_HOST = config["webui"]["host"]
WEBUI_PORT = int(config["webui"]["port"])
APP_LOG_LEVEL_STR = config["webui"]["log_level"]
APP_LOG_LEVEL = get_log_level_from_str(APP_LOG_LEVEL_STR) # This is ascii_colors.LogLevel

# SafeStore settings from config
SS_DEFAULT_VECTORIZER = config["safestore"]["default_vectorizer"]
SS_CHUNK_SIZE = int(config["safestore"]["chunk_size"])
SS_CHUNK_OVERLAP = int(config["safestore"]["chunk_overlap"])

# GraphStore settings
GS_FUSION_ENABLED = config["graphstore"].get("fusion", {}).get("enabled", False)

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """
Use the following context from a knowledge graph to answer the user's question.
The context consists of text chunks from documents.
If the context is not sufficient, say that you don't have enough information.

**Context:**
---
{context}
---

**User Question:**
{question}

**Answer:**
"""

# Global instances
app = FastAPI(title="SafeStore Graph WebUI")

# --- Socket.IO Server Setup ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, socketio_path='sio')
app.mount('/sio', socket_app)

current_dir = Path(__file__).parent
app.mount("/static_assets", StaticFiles(directory=current_dir / "static_assets"), name="static_assets")

ss_instance: Optional[SafeStore] = None
gs_instance: Optional[GraphStore] = None
lc_client: Optional[LollmsClient] = None

# --- Pydantic Models for API ---
class NodeModel(BaseModel):
    id: Union[int, str] 
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    x: Optional[float] = None
    y: Optional[float] = None
    original_label: Optional[str] = None

class NodeCreateModel(BaseModel):
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

class NodeUpdateModel(BaseModel):
    label: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class EdgeModel(BaseModel):
    id: Union[int, str]
    from_node_id: Union[int, str] = Field(..., alias="from") 
    to_node_id: Union[int, str] = Field(..., alias="to")     
    label: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

class EdgeCreateModel(BaseModel):
    from_node_id: Union[int, str]
    to_node_id: Union[int, str]
    label: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

class EdgeUpdateModel(BaseModel):
    label: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class FuseRequest(BaseModel):
    sid: str

class DatabaseConfigModel(BaseModel):
    name: str
    db_file: str
    doc_dir: str

class DatabaseCreationRequest(BaseModel):
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$", description="The unique name for the database. No spaces or special characters.")
class PathRequest(BaseModel):
    start_node_id: int
    end_node_id: int
    directed: bool = True

class ChatRequest(BaseModel):
    query: str

class UploadResponseTask(BaseModel):
    filename: str
    task_id: str

# --- LLM Executor Callback ---
def llm_executor(prompt_to_llm: str) -> str:
    global lc_client
    if not lc_client:
        ASCIIColors.error("LLM Client not initialized for executor callback!")
        raise ConnectionError("LLM Client not ready for executor callback.")
    ASCIIColors.debug(f"WebUI LLM Executor: Sending prompt (len {len(prompt_to_llm)}) to LLM...")
    try:
        response = lc_client.generate_text(prompt_to_llm, max_new_tokens=1024)
        ASCIIColors.debug(f"WebUI LLM Executor: Raw response from LLM: {response[:200]}...")
        return response if response else ""
    except Exception as e:
        ASCIIColors.error(f"Error during LLM execution in webui callback: {e}")
        trace_exception(e)
        raise RuntimeError(f"LLM execution failed: {e}") from e

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    global ss_instance, gs_instance, lc_client, ACTIVE_DB_FILE, ACTIVE_DOC_DIR, config

    # Ensure the root databases directory exists
    DATABASES_ROOT.mkdir(parents=True, exist_ok=True)

    # Determine active database
    active_db_name = config["webui"]["active_database_name"]
    db_configs = config["databases"]
    active_db_config = next((db for db in db_configs if db["name"] == active_db_name), None)

    if not active_db_config:
        ASCIIColors.warning(f"Active database '{active_db_name}' not found. Falling back to the first available database.")
        if db_configs:
            active_db_config = db_configs[0]
            config["webui"]["active_database_name"] = active_db_config["name"]
            save_config(config)
        else:
            ASCIIColors.error("No databases are configured in config.toml. Cannot start services.")
            return

    ACTIVE_DB_FILE = Path(active_db_config["db_file"])
    ACTIVE_DOC_DIR = Path(active_db_config["doc_dir"])
    ASCIIColors.info(f"Activating database '{active_db_config['name']}' (db: {ACTIVE_DB_FILE}, docs: {ACTIVE_DOC_DIR})")

    safe_store_log_level_map = {
        LogLevel.DEBUG: SafeStoreLogLevel.DEBUG,
        LogLevel.INFO: SafeStoreLogLevel.INFO,
        LogLevel.WARNING: SafeStoreLogLevel.WARNING,
        LogLevel.ERROR: SafeStoreLogLevel.ERROR,
        LogLevel.CRITICAL: SafeStoreLogLevel.CRITICAL,
    }
    ss_gs_log_level = safe_store_log_level_map.get(APP_LOG_LEVEL, SafeStoreLogLevel.INFO)
    
    ASCIIColors.set_log_level(APP_LOG_LEVEL) 

    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_DOC_DIR.mkdir(parents=True, exist_ok=True)

    ASCIIColors.info("Initializing LollmsClient for WebUI...")
    try:
        lc_params: Dict[str, Any] = {
            "binding_name": BINDING_NAME, "host_address": HOST_ADDRESS, 
            "model_name": MODEL_NAME,
        }
        if SERVICE_KEY: lc_params["service_key"] = SERVICE_KEY
        if lc_params.get("host_address") is None and BINDING_NAME in ["openai", "cohere", "anthropic", "palm", "mistral_ai", "perplexity_ai", "groq"]: 
            del lc_params["host_address"]
        
        lc_client = LollmsClient(**lc_params)
        if not hasattr(lc_client, 'binding') or lc_client.binding is None: 
            ASCIIColors.error(f"LollmsClient binding '{BINDING_NAME}' could not be loaded. Check configuration.")
            lc_client = None 
        else:
            ASCIIColors.success("LollmsClient initialized for WebUI.")
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient for WebUI: {e}")
        trace_exception(e)
        lc_client = None

    ASCIIColors.info("Initializing SafeStore for WebUI...")
    try:
        ss_instance = SafeStore(db_path=ACTIVE_DB_FILE, log_level=ss_gs_log_level)
        ASCIIColors.success("SafeStore initialized.")
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize SafeStore: {e}"); trace_exception(e); ss_instance = None

    ASCIIColors.info("Initializing GraphStore for WebUI...")
    if lc_client and ss_instance:
        try:
            gs_instance = GraphStore(
                db_path=ACTIVE_DB_FILE, 
                llm_executor_callback=llm_executor,
                log_level=ss_gs_log_level
            )
            ASCIIColors.success(f"GraphStore initialized.")
        except Exception as e:
            ASCIIColors.error(f"Failed to initialize GraphStore: {e}"); trace_exception(e); gs_instance = None
    else:
        ASCIIColors.warning("GraphStore not initialized (missing LLM client or SafeStore).")

@app.on_event("shutdown")
async def shutdown_event():
    global ss_instance, gs_instance
    if gs_instance:
        try: gs_instance.close(); ASCIIColors.info("GraphStore closed.")
        except Exception as e: ASCIIColors.error(f"Error closing GraphStore: {e}")
    if ss_instance:
        try: ss_instance.close(); ASCIIColors.info("SafeStore closed.")
        except Exception as e: ASCIIColors.error(f"Error closing SafeStore: {e}")

# --- Socket.IO Handlers ---
@sio.event
async def connect(sid, environ):
    ASCIIColors.info(f"Socket.IO client connected: {sid}")

@sio.event
async def disconnect(sid):
    ASCIIColors.info(f"Socket.IO client disconnected: {sid}")

# --- Progress Update Emitter ---
async def send_progress_update(sid: str, progress: float, message: str, task_id: str, filename: str):
    try:
        await sio.emit('progress_update', {
            "task_id": task_id,
            "filename": filename,
            "progress": progress,
            "message": message
        }, to=sid)
    except Exception as e:
        ASCIIColors.error(f"Error sending progress to client {sid}: {e}")

# --- Background Task Wrappers ---
def run_build_graph_task_wrapper(doc_id: int, filename: str, guidance: Optional[str], sid: str, task_id: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def sync_callback(progress, message):
        loop.run_until_complete(send_progress_update(sid, progress, message, task_id, filename))

    try:
        with gs_instance:
            gs_instance.build_graph_for_document(doc_id, guidance=guidance, progress_callback=sync_callback)
    except Exception as e:
        ASCIIColors.error(f"Error in background task 'build_graph_for_document' for doc {doc_id} ('{filename}'): {e}")
        trace_exception(e)
        sync_callback(1.0, f"An error occurred: {str(e)}")
    finally:
        loop.close()

def run_fuse_entities_task_wrapper(sid: str, task_id: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def sync_callback(progress, message):
        # Fuse task doesn't have a specific filename, so we can pass a generic one
        loop.run_until_complete(send_progress_update(sid, progress, message, task_id, "Entity Fusion"))

    try:
        with gs_instance:
            gs_instance.fuse_all_similar_entities(progress_callback=sync_callback)
    except Exception as e:
        ASCIIColors.error(f"Error in background task 'fuse_all_similar_entities': {e}")
        trace_exception(e)
        sync_callback(1.0, f"An error occurred: {str(e)}")
    finally:
        loop.close()


# --- FastAPI Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_root():
    index_html_path = current_dir / "static" / "index.html"
    if not index_html_path.is_file():
        ASCIIColors.error(f"index.html not found at expected path: {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found. Ensure it is in the 'webui/static/' directory.")
    return FileResponse(index_html_path)


@app.post("/upload-file/", response_model=List[UploadResponseTask])
async def upload_files_and_process(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    guidance: str = Form(""),
    sid: str = Form(...)
):
    global ss_instance, gs_instance, ACTIVE_DOC_DIR
    if not ss_instance: raise HTTPException(status_code=503, detail="SafeStore not initialized.")
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    if not ACTIVE_DOC_DIR: raise HTTPException(status_code=503, detail="Active document directory not set.")

    response_tasks = []
    
    for file in files:
        target_path_for_safestore_processing = None
        try:
            safe_filename = Path(file.filename).name
            unique_doc_filename = f"{uuid.uuid4()}_{safe_filename}"
            target_path_for_safestore_processing = ACTIVE_DOC_DIR / unique_doc_filename
            
            # Save the file
            with open(target_path_for_safestore_processing, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            ASCIIColors.info(f"File '{safe_filename}' saved to '{target_path_for_safestore_processing}'")

            # Add to SafeStore
            with ss_instance:
                ss_instance.add_document(
                    file_path=target_path_for_safestore_processing,
                    vectorizer_name=SS_DEFAULT_VECTORIZER,
                    chunk_size=SS_CHUNK_SIZE,
                    chunk_overlap=SS_CHUNK_OVERLAP
                )
                
                # Retrieve the doc_id for the just-added document
                docs = ss_instance.list_documents()
                found_doc = next((doc for doc in reversed(docs) if Path(doc['file_path']).name == unique_doc_filename), None)
                if not found_doc:
                    raise DocumentNotFoundError(f"Failed to find document '{unique_doc_filename}' in SafeStore after adding.")
                doc_id_found = found_doc['doc_id']
                ASCIIColors.info(f"Document '{safe_filename}' (ID: {doc_id_found}) confirmed in SafeStore.")

            # Schedule the background task for graph processing
            task_id = f"graph_build_{uuid.uuid4()}"
            background_tasks.add_task(run_build_graph_task_wrapper, doc_id_found, safe_filename, guidance, sid, task_id)
            response_tasks.append({"filename": safe_filename, "task_id": task_id})

        except Exception as e:
            ASCIIColors.error(f"Error processing file {file.filename}: {e}"); trace_exception(e)
            if target_path_for_safestore_processing and target_path_for_safestore_processing.exists():
                target_path_for_safestore_processing.unlink(missing_ok=True)
            # We skip this file and continue with others, but might want to report error back.
            # For now, it just won't be in the response list.
        finally:
            await file.close()
            
    if not response_tasks:
        raise HTTPException(status_code=500, detail="No files could be processed.")

    return response_tasks


@app.get("/graph-data/")
async def get_graph_data_endpoint(): 
    global gs_instance
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            nodes = gs_instance.get_all_nodes_for_visualization(limit=1000) 
            relationships = gs_instance.get_all_relationships_for_visualization(limit=2000)
        return {"nodes": nodes, "edges": relationships}
    except Exception as e:
        ASCIIColors.error(f"Error fetching graph data: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error fetching graph data: {str(e)}")

@app.post("/graph/fuse/")
async def fuse_entities_endpoint(background_tasks: BackgroundTasks, request: FuseRequest):
    global gs_instance
    sid = request.sid
    if not gs_instance:
        raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    if not lc_client:
        raise HTTPException(status_code=503, detail="LLM Client not initialized, which is required for fusion.")
    if not sid:
        raise HTTPException(status_code=400, detail="Socket.IO session ID (sid) is required.")
        
    task_id = f"fuse_{uuid.uuid4()}"
    background_tasks.add_task(run_fuse_entities_task_wrapper, sid, task_id)
    return JSONResponse(status_code=202, content={"message": "Entity fusion process started in background.", "task_id": task_id})


@app.get("/graph/search/")
async def search_graph(q: str = Query(..., min_length=1, description="Text to search for in node labels and properties.")):
    global gs_instance
    if not gs_instance:
        raise HTTPException(status_code=503, detail="GraphStore not initialized.")

    try:
        with gs_instance:
            all_nodes = gs_instance.get_all_nodes_for_visualization(limit=-1)
            all_edges = gs_instance.get_all_relationships_for_visualization(limit=-1)

        search_term = q.lower()
        matching_nodes = []
        matching_node_ids = set()
        
        ASCIIColors.info(f"Performing simple text search for: '{q}'")

        for node in all_nodes:
            if node['id'] in matching_node_ids: continue
            if search_term in node.get('label', '').lower():
                matching_nodes.append(node)
                matching_node_ids.add(node['id'])
                continue
            for key, value in node.get('properties', {}).items():
                if isinstance(value, str) and search_term in value.lower():
                    matching_nodes.append(node)
                    matching_node_ids.add(node['id'])
                    break

        matching_edges = []
        if matching_node_ids:
            for edge in all_edges:
                if edge.get('from') in matching_node_ids or edge.get('to') in matching_node_ids:
                    matching_edges.append(edge)
        
        ASCIIColors.success(f"Search found {len(matching_nodes)} nodes and {len(matching_edges)} related edges.")
        return {"nodes": matching_nodes, "edges": matching_edges}

    except Exception as e:
        ASCIIColors.error(f"Error during graph search for query '{q}': {e}")
        trace_exception(e)
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {str(e)}")

@app.get("/graph/node/{node_id}/neighbors")
async def get_node_neighbors(node_id: int, limit: int = Query(50, ge=1, le=200)):
    """
    Get the neighbors of a specific node.
    """
    global gs_instance
    if not gs_instance:
        raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            # The get_neighbors method returns both nodes and edges
            neighbor_data = gs_instance.get_neighbors(node_id, limit=limit)
            return neighbor_data
    except Exception as e:
        ASCIIColors.error(f"Error fetching neighbors for node {node_id}: {e}")
        trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error fetching neighbors: {str(e)}")


@app.post("/graph/path")
async def find_path_endpoint(path_request: PathRequest):
    """
    Finds the shortest path between two nodes.
    """
    global gs_instance
    if not gs_instance:
        raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            # Pass the directed flag to the GraphStore method
            path = gs_instance.find_shortest_path(
                start_node_id=path_request.start_node_id, 
                end_node_id=path_request.end_node_id,
                directed=path_request.directed
            )
            if path is None:
                raise HTTPException(status_code=404, detail="No path found between the specified nodes.")
            return path
    except HTTPException:
        raise
    except Exception as e:
        ASCIIColors.error(f"Error finding path between {path_request.start_node_id} and {path_request.end_node_id}: {e}")
        trace_exception(e)
        raise HTTPException(status_code=500, detail=f"An error occurred while finding the path: {str(e)}")

@app.post("/api/chat/rag")
async def chat_rag_endpoint(chat_request: ChatRequest):
    """
    Handles a chat request using the RAG pipeline.
    """
    global gs_instance, lc_client
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    if not lc_client: raise HTTPException(status_code=503, detail="LLM Client not initialized.")

    try:
        with gs_instance:
            # Use query_graph to retrieve relevant context chunks
            context_chunks = gs_instance.query_graph(
                natural_language_query=chat_request.query,
                output_mode="chunks_summary"
            )
        
        if not context_chunks:
            return {"answer": "I couldn't find any information in the knowledge graph related to your question."}

        # Format the context for the LLM prompt
        context_str = "\n\n".join([chunk['chunk_text'] for chunk in context_chunks])
        
        # Build the final prompt for the LLM
        final_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context_str,
            question=chat_request.query
        )

        # Generate the answer
        answer = llm_executor(final_prompt)
        
        return {"answer": answer}
        
    except Exception as e:
        ASCIIColors.error(f"Error during RAG chat for query '{chat_request.query}': {e}")
        trace_exception(e)
        raise HTTPException(status_code=500, detail=f"An error occurred during RAG chat: {str(e)}")


# --- Database Management Endpoints ---
@app.get("/api/databases", response_model=List[DatabaseConfigModel])
async def get_all_databases():
    """Returns a list of all configured databases."""
    db_list = get_config().get("databases", [])
    active_name = get_config().get("webui", {}).get("active_database_name")
    # Add an is_active flag for the frontend
    for db in db_list:
        db["is_active"] = (db["name"] == active_name)
    return db_list

@app.post("/api/databases", response_model=DatabaseConfigModel, status_code=status.HTTP_201_CREATED)
async def create_database(request: DatabaseCreationRequest):
    """Creates a new database configuration and its directory structure."""
    global config
    db_name = request.name
    if any(db["name"] == db_name for db in config["databases"]):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Database with name '{db_name}' already exists.")
    
    db_folder = DATABASES_ROOT / db_name
    db_file = db_folder / f"{db_name}.db"
    doc_dir = db_folder / "docs"

    new_db_dict = {"name": db_name, "db_file": str(db_file), "doc_dir": str(doc_dir)}
    
    try:
        doc_dir.mkdir(parents=True, exist_ok=True)
        ASCIIColors.info(f"Created database directory structure at: {db_folder.resolve()}")
        
        config["databases"].append(new_db_dict)
        save_config(config)
        return new_db_dict
    except Exception as e:
        ASCIIColors.error(f"Failed to create database configuration: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.delete("/api/databases/{db_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_database(db_name: str):
    """Deletes a database configuration. Does not delete files on disk."""
    global config
    if len(config["databases"]) <= 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete the last remaining database.")
    
    original_dbs = list(config["databases"])
    config["databases"] = [db for db in original_dbs if db["name"] != db_name]
    
    if len(config["databases"]) == len(original_dbs):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Database '{db_name}' not found.")
    
    if config["webui"]["active_database_name"] == db_name:
        config["webui"]["active_database_name"] = config["databases"][0]["name"]
        ASCIIColors.warning(f"Deleted active database. Switched active DB to '{config['databases'][0]['name']}'.")

    save_config(config)

@app.put("/api/databases/{db_name}/activate", status_code=status.HTTP_200_OK)
async def activate_database(db_name: str):
    """Sets the specified database as the active one for the next startup."""
    global config
    if not any(db["name"] == db_name for db in config["databases"]):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Database '{db_name}' not found.")
    
    config["webui"]["active_database_name"] = db_name
    save_config(config)
    return {"message": f"Database '{db_name}' set as active. Please reload the application."}


# --- Graph Element CRUD Endpoints (unchanged) ---
@app.post("/graph/node/", response_model=NodeModel, status_code=201)
async def add_graph_node(node_data: NodeCreateModel):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            node_id = gs_instance.add_node(label=node_data.label, properties=node_data.properties)
            created_node_dict = gs_instance.get_node_by_id(node_id) 
            if not created_node_dict:
                raise HTTPException(status_code=500, detail="Failed to retrieve created node.")
            return NodeModel(**created_node_dict)
    except Exception as e:
        ASCIIColors.error(f"Error adding node: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error adding node: {str(e)}")

@app.put("/graph/node/{node_id}", response_model=NodeModel)
async def update_graph_node(node_id: int, node_data: NodeUpdateModel):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            gs_instance.update_node(node_id, label=node_data.label, properties=node_data.properties)
            updated_node_dict = gs_instance.get_node_by_id(node_id)
            if not updated_node_dict:
                 raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found post-update.")
            return NodeModel(**updated_node_dict)
    except safe_store.core.exceptions.NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        ASCIIColors.error(f"Error updating node {node_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error updating node: {str(e)}")

@app.delete("/graph/node/{node_id}", status_code=204)
async def delete_graph_node(node_id: int):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            if not gs_instance.delete_node(node_id):
                 raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found for deletion.")
    except safe_store.core.exceptions.NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        ASCIIColors.error(f"Error deleting node {node_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error deleting node: {str(e)}")

@app.post("/graph/edge/", response_model=EdgeModel, status_code=201)
async def add_graph_edge(edge_data: EdgeCreateModel):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            edge_id = gs_instance.add_relationship(
                source_node_id=int(edge_data.from_node_id),
                target_node_id=int(edge_data.to_node_id),
                label=edge_data.label,
                properties=edge_data.properties
            )
            created_edge_dict = gs_instance.get_relationship_by_id(edge_id)
            if not created_edge_dict:
                raise HTTPException(status_code=500, detail="Failed to retrieve created edge.")
            
            api_edge_data = {"id": created_edge_dict["id"], "from": created_edge_dict["source_node_id"], "to": created_edge_dict["target_node_id"], "label": created_edge_dict["label"], "properties": created_edge_dict["properties"]}
            return EdgeModel(**api_edge_data)
    except safe_store.core.exceptions.NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError: 
        raise HTTPException(status_code=400, detail="Invalid node ID format.")
    except Exception as e:
        ASCIIColors.error(f"Error adding edge: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error adding edge: {str(e)}")

@app.put("/graph/edge/{edge_id}", response_model=EdgeModel)
async def update_graph_edge(edge_id: int, edge_data: EdgeUpdateModel):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            gs_instance.update_relationship(edge_id, label=edge_data.label, properties=edge_data.properties)
            updated_edge_dict = gs_instance.get_relationship_by_id(edge_id)
            if not updated_edge_dict:
                 raise HTTPException(status_code=404, detail=f"Edge with ID {edge_id} not found.")
            api_edge_data = {"id": updated_edge_dict["id"], "from": updated_edge_dict["source_node_id"], "to": updated_edge_dict["target_node_id"], "label": updated_edge_dict["label"], "properties": updated_edge_dict["properties"]}
            return EdgeModel(**api_edge_data)
    except safe_store.core.exceptions.RelationshipNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        ASCIIColors.error(f"Error updating edge {edge_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error updating edge: {str(e)}")

@app.delete("/graph/edge/{edge_id}", status_code=204)
async def delete_graph_edge(edge_id: int):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            if not gs_instance.delete_relationship(edge_id):
                raise HTTPException(status_code=404, detail=f"Edge with ID {edge_id} not found for deletion.")
    except safe_store.core.exceptions.RelationshipNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        ASCIIColors.error(f"Error deleting edge {edge_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error deleting edge: {str(e)}")


# --- Launcher function for command-line entry point ---
def launch_webui():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir) # Change CWD to the script's dir for relative path consistency

    global config
    config = get_config() 

    _webui_host = config["webui"]["host"]
    _webui_port = int(config["webui"]["port"])
    _app_log_level_str = config["webui"]["log_level"] 

    ASCIIColors.info(f"Launching SafeStore WebUI on http://{_webui_host}:{_webui_port}")
    ASCIIColors.info(f"Uvicorn log level will be based on application setting: {_app_log_level_str.lower()}")
    
    uvicorn.run(
        app,
        host=_webui_host,
        port=_webui_port,
        log_level=_app_log_level_str.lower(), 
        reload=False 
    )

if __name__ == "__main__":
    launch_webui()