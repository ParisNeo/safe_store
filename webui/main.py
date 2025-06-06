# webui/main.py
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import pipmaster as pm
pm.ensure_packages([
    "lollms_client",
    "uvicorn",
    "fastapi",
    "python-multipart",
    "toml",
    "pydantic" # Added for request/response models
])
from config_manager import get_config, get_log_level_from_str, save_config, DEFAULT_CONFIG, CONFIG_FILE_PATH # Explicitly import 
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body, Path as FastApiPath
from fastapi.responses import FileResponse, JSONResponse 
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field # For request/response models

import safe_store # Assuming safe_store is in PYTHONPATH or installed
from safe_store import GraphStore, SafeStore, LogLevel as SafeStoreLogLevel # Renamed to avoid conflict
from safe_store.core.exceptions import DocumentNotFoundError # For specific exception handling
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors, trace_exception, LogLevel

# Import the configuration manager
from config_manager import get_config, get_log_level_from_str, save_config # Relative import

# --- Load Configuration ---
config = get_config()

# --- Use Configuration Values ---
TEMP_UPLOAD_DIR = Path(config["webui"]["temp_upload_dir"])
DB_FILE = Path(config["safestore"]["db_file"])
DOC_DIR_FOR_SAFESTORE = Path(config["safestore"]["doc_dir"])

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


# Global instances
app = FastAPI(title="SafeStore Graph WebUI")

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


# --- LLM Executor Callback ---
def llm_executor(prompt_to_llm: str) -> str:
    global lc_client
    if not lc_client:
        ASCIIColors.error("LLM Client not initialized for executor callback!")
        raise ConnectionError("LLM Client not ready for executor callback.")
    ASCIIColors.debug(f"WebUI LLM Executor: Sending prompt (len {len(prompt_to_llm)}) to LLM...")
    try:
        response = lc_client.generate_code(
            prompt_to_llm, language="json" 
        ) 
        ASCIIColors.debug(f"WebUI LLM Executor: Raw response from LLM: {response[:200]}...")
        return response if response else ""
    except Exception as e:
        ASCIIColors.error(f"Error during LLM execution in webui callback: {e}")
        trace_exception(e)
        raise RuntimeError(f"LLM execution failed: {e}") from e

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    global ss_instance, gs_instance, lc_client
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
    DOC_DIR_FOR_SAFESTORE.mkdir(parents=True, exist_ok=True)

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
        # Assuming SafeStore is initialized without encryption key for now, or key is handled by SafeStore internally
        ss_instance = SafeStore(db_path=DB_FILE, log_level=ss_gs_log_level) # encryption_key=None
        ASCIIColors.success("SafeStore initialized.")
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize SafeStore: {e}"); trace_exception(e); ss_instance = None

    ASCIIColors.info("Initializing GraphStore for WebUI...")
    if lc_client and ss_instance:
        try:
            gs_instance = GraphStore(
                db_path=DB_FILE, 
                llm_executor_callback=llm_executor,
                log_level=ss_gs_log_level # encryption_key=None (assuming GraphStore uses SafeStore's encryptor or handles its own)
            )
            ASCIIColors.success("GraphStore initialized.")
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

# --- FastAPI Endpoints ---
@app.get("/", response_class=FileResponse)
async def read_root():
    index_html_path = current_dir / "static" / "index.html"
    if not index_html_path.is_file():
        ASCIIColors.error(f"index.html not found at expected path: {index_html_path}")
        raise HTTPException(status_code=404, detail="index.html not found. Ensure it is in the 'webui/static/' directory.")
    return FileResponse(index_html_path)


@app.post("/upload-file/")
async def upload_file_and_process(file: UploadFile = File(...)):
    global ss_instance, gs_instance
    if not ss_instance: raise HTTPException(status_code=503, detail="SafeStore not initialized.")
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized. LLM may be unavailable.")
    if not lc_client: raise HTTPException(status_code=503, detail="LLM Client not initialized. Graph building requires LLM.")

    processed_doc_path_in_safestore: Optional[Path] = None
    doc_id_found = -1

    try:
        safe_filename = Path(file.filename).name 
        # Use a unique name for the file in SafeStore's doc_dir to avoid collisions
        # SafeStore might also rename or manage this internally.
        unique_doc_filename = f"{uuid.uuid4()}_{safe_filename}"
        
        # Path where the file will be initially saved for SafeStore to pick up
        target_path_for_safestore_processing = DOC_DIR_FOR_SAFESTORE / unique_doc_filename
        
        # Save uploaded file to the DOC_DIR_FOR_SAFESTORE
        # No intermediate temp_upload_dir step unless SafeStore strictly requires it.
        # If DOC_DIR_FOR_SAFESTORE is watched, this is usually enough.
        with open(target_path_for_safestore_processing, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        ASCIIColors.info(f"File '{file.filename}' saved to SafeStore processing directory '{target_path_for_safestore_processing}'")

        with ss_instance: # Context manager for SafeStore
            ss_instance.add_document(
                file_path=target_path_for_safestore_processing, 
                vectorizer_name=SS_DEFAULT_VECTORIZER,
                chunk_size=SS_CHUNK_SIZE,
                chunk_overlap=SS_CHUNK_OVERLAP
            )
            ASCIIColors.info(f"Document '{target_path_for_safestore_processing.name}' submitted to SafeStore for processing.")

            # Retrieve the doc_id. This part depends on how SafeStore manages documents.
            # Option 1: If SafeStore renames/moves files and tracks them, get by original path (if possible) or new path.
            # Option 2: List documents and find the most recent or match by filename.
            # We'll use a robust way to find the document, assuming SafeStore might alter the path or name.
            # Using get_document_by_path if available and path remains consistent, or list_documents.
            
            # Attempt to get the document by its path (which might be what SafeStore uses as an identifier)
            # This assumes `get_document_by_path` exists and `target_path_for_safestore_processing` is a valid lookup key.
            # If SafeStore moves the file, we need its new path.
            # A common pattern is that `add_document` processes and stores metadata, including a stable `doc_id`.
            
            # Let's try to find it using list_documents, as it's more generic.
            # We need the actual path SafeStore uses if it modifies it.
            # For now, assume target_path_for_safestore_processing is what we can query for.
            # A better way would be if SafeStore's add_document could signal the final path or ID.

            docs = ss_instance.list_documents() # Get most recent first
            found_doc = None
            for doc_meta in docs:
                # We need a reliable way to match. The path given to add_document might be best.
                # Or if SafeStore provides the final path in its metadata.
                # For now, let's assume `doc_meta['file_path']` is the key path.
                # The file `target_path_for_safestore_processing` was the input.
                # SafeStore might have moved or renamed it.
                # The most reliable way is if list_documents returns the *original* path or enough metadata.
                # Let's assume `doc_meta['file_path']` IS the path we can match against the one we saved.
                # However, SafeStore often processes into its own structure.
                # The original `webui/main.py` used `Path(doc['file_path']).resolve() == saved_file_path.resolve()`
                # `saved_file_path` in that context was `DOC_DIR_FOR_SAFESTORE / temp_file_name`
                
                # If SafeStore keeps the file name part:
                if Path(doc_meta['file_path']).name == unique_doc_filename:
                    found_doc = doc_meta
                    break
            
            if found_doc:
                doc_id_found = found_doc['doc_id']
                processed_doc_path_in_safestore = Path(found_doc['file_path'])
                ASCIIColors.info(f"Document '{unique_doc_filename}' (ID: {doc_id_found}) confirmed in SafeStore. Path: {processed_doc_path_in_safestore}")
            else:
                # If not found by name, maybe it's the absolute latest one and we risk a race condition.
                # This part is tricky without knowing exactly how SafeStore's list_documents and path management work.
                ASCIIColors.warning(f"Could not definitively find '{unique_doc_filename}' by name in SafeStore listing. Trying most recent.")
                if docs: # Fallback to most recent if list is not empty
                    doc_id_found = docs[0]['doc_id']
                    processed_doc_path_in_safestore = Path(docs[0]['file_path'])
                    ASCIIColors.warning(f"Assuming most recent document (ID: {doc_id_found}, Path: {processed_doc_path_in_safestore}) is the one just added.")
                else:
                    raise DocumentNotFoundError(f"Failed to find document '{unique_doc_filename}' in SafeStore after adding, and no documents listed.")

        if doc_id_found == -1: # Should be caught by DocumentNotFoundError now
             raise HTTPException(status_code=500, detail=f"Failed to obtain doc_id for '{file.filename}' from SafeStore.")

        ASCIIColors.info(f"Doc '{file.filename}' (ID: {doc_id_found}) ready. Building graph...")
        with gs_instance: # Context manager for GraphStore
            gs_instance.build_graph_for_document(doc_id_found)
        ASCIIColors.success(f"Graph built for doc '{file.filename}' (ID: {doc_id_found}).")
        
        return JSONResponse(
            status_code=200, 
            content={"message": f"File '{file.filename}' processed.", "doc_id": doc_id_found}
        )
    except safe_store.LLMCallbackError as e: 
        ASCIIColors.error(f"LLM Callback Error for {file.filename}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"LLM processing error: {e}")
    except DocumentNotFoundError as e:
        ASCIIColors.error(f"Document not found error for {file.filename}: {e}"); trace_exception(e)
        # Cleanup the initially saved file if it's still there and SafeStore failed to process/recognize it
        if target_path_for_safestore_processing and target_path_for_safestore_processing.exists():
            target_path_for_safestore_processing.unlink(missing_ok=True)
            ASCIIColors.debug(f"Cleaned up file from SafeStore doc dir due to DocumentNotFoundError: {target_path_for_safestore_processing}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm document in SafeStore: {e}")
    except HTTPException: raise 
    except Exception as e:
        ASCIIColors.error(f"Error processing file {file.filename}: {e}"); trace_exception(e)
        if 'target_path_for_safestore_processing' in locals() and target_path_for_safestore_processing.exists():
            target_path_for_safestore_processing.unlink(missing_ok=True)
            ASCIIColors.debug(f"Cleaned up file from SafeStore doc dir due to error: {target_path_for_safestore_processing}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if file: await file.close()


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

# --- Graph Element CRUD Endpoints ---
@app.post("/graph/node/", response_model=NodeModel, status_code=201)
async def add_graph_node(node_data: NodeCreateModel):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            node_id = gs_instance.add_node(label=node_data.label, properties=node_data.properties)
            created_node_dict = gs_instance.get_node_by_id(node_id) 
            if not created_node_dict:
                raise HTTPException(status_code=500, detail="Failed to retrieve created node.")
            return NodeModel(**created_node_dict) # Use Pydantic model directly with dict
    except Exception as e:
        ASCIIColors.error(f"Error adding node: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error adding node: {str(e)}")

@app.put("/graph/node/{node_id}", response_model=NodeModel)
async def update_graph_node(node_id: int = FastApiPath(..., title="The ID of the node to update"), node_data: NodeUpdateModel = Body(...)):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            success = gs_instance.update_node(
                node_id, 
                label=node_data.label if node_data.label is not None else None, # Explicitly pass None if not provided
                properties=node_data.properties if node_data.properties is not None else None
            )
            if not success: # update_node should raise NodeNotFound if not found, or return False for other non-update.
                            # Let's assume it raises NodeNotFound, so this case is for "no change needed" or other logical non-update.
                # Check if node still exists to differentiate "not found" from "no change"
                current_node = gs_instance.get_node_by_id(node_id)
                if not current_node:
                     raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found.")
                # If it exists but update returned false, maybe no change was made.
                ASCIIColors.info(f"Update call for node {node_id} resulted in no change or logical non-update.")
                return NodeModel(**current_node) # Return current state

            updated_node_dict = gs_instance.get_node_by_id(node_id)
            if not updated_node_dict: 
                 raise HTTPException(status_code=500, detail="Failed to retrieve updated node post-update.")
            return NodeModel(**updated_node_dict)
    except safe_store.core.exceptions.NodeNotFoundError as e: # Catch specific exception from GraphStore
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException: raise
    except Exception as e:
        ASCIIColors.error(f"Error updating node {node_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error updating node: {str(e)}")

@app.delete("/graph/node/{node_id}", status_code=204)
async def delete_graph_node(node_id: int = FastApiPath(..., title="The ID of the node to delete")):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            success = gs_instance.delete_node(node_id)
            if not success: # If delete_node returns False for "not found"
                 raise HTTPException(status_code=404, detail=f"Node with ID {node_id} not found for deletion.")
            # HTTP 204 No Content implies success, no body needed.
    except safe_store.core.exceptions.NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException: raise
    except Exception as e:
        ASCIIColors.error(f"Error deleting node {node_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error deleting node: {str(e)}")

@app.post("/graph/edge/", response_model=EdgeModel, status_code=201)
async def add_graph_edge(edge_data: EdgeCreateModel):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            from_node_id_int = int(edge_data.from_node_id)
            to_node_id_int = int(edge_data.to_node_id)

            edge_id = gs_instance.add_relationship(
                source_node_id=from_node_id_int,
                target_node_id=to_node_id_int,
                label=edge_data.label,
                properties=edge_data.properties
            )
            created_edge_dict = gs_instance.get_relationship_by_id(edge_id)
            if not created_edge_dict:
                raise HTTPException(status_code=500, detail="Failed to retrieve created edge.")
            
            # Map GraphStore dict to EdgeModel, ensuring 'from' and 'to' aliases
            api_edge_data = {
                "id": created_edge_dict["id"],
                "from": created_edge_dict["source_node_id"], # Alias for frontend
                "to": created_edge_dict["target_node_id"],   # Alias for frontend
                "label": created_edge_dict["label"],
                "properties": created_edge_dict["properties"]
            }
            return EdgeModel(**api_edge_data)
    except safe_store.core.exceptions.NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) # If source/target node not found
    except ValueError: 
        raise HTTPException(status_code=400, detail="Invalid node ID format. Node IDs must be integers.")
    except HTTPException: raise
    except Exception as e:
        ASCIIColors.error(f"Error adding edge: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error adding edge: {str(e)}")

@app.put("/graph/edge/{edge_id}", response_model=EdgeModel)
async def update_graph_edge(edge_id: int = FastApiPath(..., title="The ID of the edge to update"), edge_data: EdgeUpdateModel = Body(...)):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            success = gs_instance.update_relationship(
                edge_id, 
                label=edge_data.label if edge_data.label is not None else None,
                properties=edge_data.properties if edge_data.properties is not None else None
            )
            if not success:
                current_edge = gs_instance.get_relationship_by_id(edge_id)
                if not current_edge:
                    raise HTTPException(status_code=404, detail=f"Edge with ID {edge_id} not found.")
                ASCIIColors.info(f"Update call for edge {edge_id} resulted in no change.")
                api_edge_data = {"id": current_edge["id"], "from": current_edge["source_node_id"], "to": current_edge["target_node_id"], "label": current_edge["label"], "properties": current_edge["properties"]}
                return EdgeModel(**api_edge_data)

            updated_edge_dict = gs_instance.get_relationship_by_id(edge_id)
            if not updated_edge_dict:
                 raise HTTPException(status_code=500, detail="Failed to retrieve updated edge.")
            api_edge_data = {"id": updated_edge_dict["id"], "from": updated_edge_dict["source_node_id"], "to": updated_edge_dict["target_node_id"], "label": updated_edge_dict["label"], "properties": updated_edge_dict["properties"]}
            return EdgeModel(**api_edge_data)
    except safe_store.core.exceptions.RelationshipNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException: raise
    except Exception as e:
        ASCIIColors.error(f"Error updating edge {edge_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error updating edge: {str(e)}")

@app.delete("/graph/edge/{edge_id}", status_code=204)
async def delete_graph_edge(edge_id: int = FastApiPath(..., title="The ID of the edge to delete")):
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            success = gs_instance.delete_relationship(edge_id)
            if not success: # If delete_relationship returns False for "not found"
                raise HTTPException(status_code=404, detail=f"Edge with ID {edge_id} not found for deletion.")
    except safe_store.core.exceptions.RelationshipNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException: raise
    except Exception as e:
        ASCIIColors.error(f"Error deleting edge {edge_id}: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error deleting edge: {str(e)}")


# --- Launcher function for command-line entry point ---
def launch_webui():
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    ASCIIColors.info(f"WebUI Launcher: Changed CWD to: {os.getcwd()} for reliable relative path resolution.")

    global config 
    config = get_config() 

    _webui_host = config["webui"]["host"]
    _webui_port = int(config["webui"]["port"])
    _app_log_level_str = config["webui"]["log_level"] 

    if not CONFIG_FILE_PATH.exists():
         ASCIIColors.warning(f"config.toml not found at {CONFIG_FILE_PATH}. Creating default.")
         save_config(DEFAULT_CONFIG)

    ASCIIColors.info(f"Launching SafeStore WebUI on http://{_webui_host}:{_webui_port}")
    ASCIIColors.info(f"Uvicorn log level will be based on application setting: {_app_log_level_str.lower()}")

    uvicorn.run(
        "main:app", 
        host=_webui_host,
        port=_webui_port,
        log_level=_app_log_level_str.lower(), 
        reload=False 
    )

if __name__ == "__main__":
    launch_webui()