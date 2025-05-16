# webui/main.py
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
import pipmaster as pm
pm.ensure_packages([
    "lollms_client",
    "uvicorn",
    "fastapi",
    "python-multipart",
    "toml"
])
from config_manager import get_config, get_log_level_from_str, save_config, DEFAULT_CONFIG, CONFIG_FILE_PATH # Explicitly import 
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse # Changed HTMLResponse to FileResponse
from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates # REMOVED Jinja2Templates

import safe_store # Assuming safe_store is in PYTHONPATH or installed
from safe_store import GraphStore, SafeStore, LogLevel
from lollms_client import LollmsClient
from ascii_colors import ASCIIColors, trace_exception

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
SERVICE_KEY = config["lollms"].get("service_key") # Use .get for optional keys

# WebUI settings
WEBUI_HOST = config["webui"]["host"]
WEBUI_PORT = int(config["webui"]["port"])
APP_LOG_LEVEL_STR = config["webui"]["log_level"]
APP_LOG_LEVEL = get_log_level_from_str(APP_LOG_LEVEL_STR)

# SafeStore settings from config
SS_DEFAULT_VECTORIZER = config["safestore"]["default_vectorizer"]
SS_CHUNK_SIZE = int(config["safestore"]["chunk_size"])
SS_CHUNK_OVERLAP = int(config["safestore"]["chunk_overlap"])


# Global instances
app = FastAPI(title="SafeStore Graph WebUI")
# templates = Jinja2Templates(directory="templates") # REMOVED

# Mount static files relative to this main.py file's location
current_dir = Path(__file__).parent
# This serves all files in the 'static' directory under the /static URL path
# e.g., /static/style.css will serve webui/static/style.css
app.mount("/static_assets", StaticFiles(directory=current_dir / "static_assets"), name="static_assets")
# Note: Renamed mount path to "/static_assets" to avoid potential conflict if you named your index.html serving path "/static/index.html"
# If you prefer `/static/index.html` and `/static/style.css`, you can name the mount path "/static"

ss_instance: Optional[SafeStore] = None
gs_instance: Optional[GraphStore] = None
lc_client: Optional[LollmsClient] = None

# --- LLM Executor Callback (remains the same) ---
def llm_executor(prompt_to_llm: str) -> str:
    global lc_client
    if not lc_client:
        ASCIIColors.error("LLM Client not initialized for executor callback!")
        raise ConnectionError("LLM Client not ready for executor callback.")
    ASCIIColors.debug(f"WebUI LLM Executor: Sending prompt (len {len(prompt_to_llm)}) to LLM...")
    try:
        response = lc_client.generate_code(
            prompt_to_llm, language="json", temperature=0.1, max_size=16096
        )
        ASCIIColors.debug(f"WebUI LLM Executor: Raw response from generate_code: {response[:200]}...")
        return response if response else ""
    except Exception as e:
        ASCIIColors.error(f"Error during LLM execution in webui callback: {e}")
        trace_exception(e)
        raise RuntimeError(f"LLM execution failed: {e}") from e

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    global ss_instance, gs_instance, lc_client
    ASCIIColors.set_log_level(APP_LOG_LEVEL) # Set global log level from config

    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR_FOR_SAFESTORE.mkdir(parents=True, exist_ok=True)

    ASCIIColors.info("Initializing LollmsClient for WebUI...")
    try:
        lc_params: Dict[str, Any] = {
            "binding_name": BINDING_NAME, "host_address": HOST_ADDRESS, 
            "model_name": MODEL_NAME,
        }
        if SERVICE_KEY: lc_params["service_key"] = SERVICE_KEY
        if lc_params.get("host_address") is None and BINDING_NAME in ["openai"]: del lc_params["host_address"]
        
        lc_client = LollmsClient(**lc_params)
        if not hasattr(lc_client, 'binding') or lc_client.binding is None:
            raise ConnectionError(f"Binding {BINDING_NAME} could not be loaded.")
        ASCIIColors.success("LollmsClient initialized for WebUI.")
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize LollmsClient for WebUI: {e}")
        lc_client = None

    ASCIIColors.info("Initializing SafeStore for WebUI...")
    try:
        ss_instance = SafeStore(db_path=DB_FILE, log_level=APP_LOG_LEVEL)
    except Exception as e:
        ASCIIColors.error(f"Failed to initialize SafeStore: {e}"); ss_instance = None

    ASCIIColors.info("Initializing GraphStore for WebUI...")
    if lc_client and ss_instance:
        try:
            gs_instance = GraphStore(
                db_path=DB_FILE,
                llm_executor_callback=llm_executor,
                log_level=APP_LOG_LEVEL
            )
        except Exception as e:
            ASCIIColors.error(f"Failed to initialize GraphStore: {e}"); gs_instance = None
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
async def read_root(): # Request object is optional if not used
    index_html_path = current_dir / "static" / "index.html"
    if not index_html_path.is_file():
        ASCIIColors.error(f"index.html not found at expected path: {index_html_path}")
        # Fallback or more robust error handling can be added here
        # For now, rely on FastAPI's default 404 if FileResponse can't find the file,
        # or raise HTTPException explicitly.
        raise HTTPException(status_code=404, detail="index.html not found. Ensure it is in the 'webui/static/' directory.")
    return FileResponse(index_html_path)


@app.post("/upload-file/")
async def upload_file_and_process(file: UploadFile = File(...)):
    global ss_instance, gs_instance
    if not ss_instance: raise HTTPException(status_code=503, detail="SafeStore not initialized.")
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")

    saved_file_path: Optional[Path] = None 
    try:
        # Sanitize filename a bit and ensure unique name in upload dir
        safe_filename = Path(file.filename).name # Basic sanitization
        temp_file_name = f"{uuid.uuid4()}_{safe_filename}"
        
        # Changed save path to TEMP_UPLOAD_DIR for uploaded files, then copy to DOC_DIR_FOR_SAFESTORE
        temp_saved_path = TEMP_UPLOAD_DIR / temp_file_name
        
        with open(temp_saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        ASCIIColors.info(f"File '{file.filename}' uploaded to temporary location '{temp_saved_path}'")

        # Now copy to the directory SafeStore will process from
        # This two-step helps if SafeStore wants to manage its own doc directory content more strictly
        saved_file_path = DOC_DIR_FOR_SAFESTORE / temp_file_name # Use the same unique name
        shutil.copy(temp_saved_path, saved_file_path)
        ASCIIColors.info(f"File '{file.filename}' copied to SafeStore processing directory '{saved_file_path}'")
        temp_saved_path.unlink() # Clean up temp uploaded file

        doc_id = -1
        with ss_instance:
            ss_instance.add_document(
                saved_file_path, 
                vectorizer_name=SS_DEFAULT_VECTORIZER,
                chunk_size=SS_CHUNK_SIZE,
                chunk_overlap=SS_CHUNK_OVERLAP
            )
            docs = ss_instance.list_documents()
            for doc in reversed(docs): # Check most recent first
                if Path(doc['file_path']).resolve() == saved_file_path.resolve():
                    doc_id = doc['doc_id']
                    break
        
        if doc_id == -1:
            # No need to delete saved_file_path here as add_document might have already processed/moved it or failed before.
            # If SafeStore internally moves/deletes source after processing, this is fine.
            raise HTTPException(status_code=500, detail=f"Failed to add doc '{file.filename}' to SafeStore or get ID.")

        ASCIIColors.info(f"Doc '{file.filename}' (ID: {doc_id}) added. Building graph...")
        with gs_instance: gs_instance.build_graph_for_document(doc_id)
        ASCIIColors.success(f"Graph built for doc '{file.filename}' (ID: {doc_id}).")
        
        return JSONResponse(
            status_code=200, 
            content={"message": f"File '{file.filename}' processed.", "doc_id": doc_id}
        )
    except HTTPException: raise
    except safe_store.LLMCallbackError as e:
        ASCIIColors.error(f"LLM Callback Error for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing error: {e}")
    except Exception as e:
        ASCIIColors.error(f"Error processing file {file.filename}: {e}"); trace_exception(e)
        # Cleanup potentially partially processed file from SafeStore's dir if it wasn't fully processed
        if saved_file_path and saved_file_path.exists():
            saved_file_path.unlink(missing_ok=True)
            ASCIIColors.debug(f"Cleaned up file from SafeStore doc dir: {saved_file_path}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        if file: await file.close()


@app.get("/graph-data/")
async def get_graph_data():
    global gs_instance
    if not gs_instance: raise HTTPException(status_code=503, detail="GraphStore not initialized.")
    try:
        with gs_instance:
            nodes = gs_instance.get_all_nodes_for_visualization(limit=500) 
            relationships = gs_instance.get_all_relationships_for_visualization(limit=1000)
        return {"nodes": nodes, "edges": relationships}
    except Exception as e:
        ASCIIColors.error(f"Error fetching graph data: {e}"); trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Error fetching graph data: {str(e)}")

# --- Launcher function for command-line entry point ---
def launch_webui():
    """
    Launches the SafeStore WebUI using Uvicorn.
    This function is intended to be called by the command-line script.
    """
    # Ensure CWD is webui directory for relative paths to templates/static
    # This logic is important if the CLI command is run from anywhere.
    # We need to ensure that when uvicorn.run is called, relative paths
    # for StaticFiles, config_manager, etc., resolve correctly.

    # Get the directory where this main.py script is located.
    script_dir = Path(__file__).parent.resolve()
    
    # Change current working directory to the script's directory.
    # This makes relative paths within the webui folder (like 'static', 'config.toml') work reliably.
    os.chdir(script_dir)
    ASCIIColors.info(f"WebUI Launcher: Changed CWD to: {os.getcwd()} for reliable relative path resolution.")

    # Reload configuration from the new CWD context if necessary,
    # or ensure config_manager uses absolute paths or paths relative to its own location.
    # Current config_manager.py uses Path("config.toml") which will now be webui/config.toml
    global config # Reload config based on new CWD
    config = get_config() # This will re-evaluate CONFIG_FILE_PATH relative to the new CWD

    # Re-apply webui host and port from potentially reloaded config
    # These are already defined globally, but good to be explicit if config could change
    _webui_host = config["webui"]["host"]
    _webui_port = int(config["webui"]["port"])
    _app_log_level_str = config["webui"]["log_level"] # For Uvicorn's log_level

    # Ensure the default config.toml is created if it doesn't exist in webui/
    if not CONFIG_FILE_PATH.exists(): # CONFIG_FILE_PATH from config_manager
         ASCIIColors.warning(f"config.toml not found at {CONFIG_FILE_PATH}. Creating default.")
         save_config(DEFAULT_CONFIG) # DEFAULT_CONFIG from config_manager

    ASCIIColors.info(f"Launching SafeStore WebUI on http://{_webui_host}:{_webui_port}")
    ASCIIColors.info(f"Uvicorn log level will be based on application setting: {_app_log_level_str.lower()}")

    uvicorn.run(
        "main:app", # Points to the 'app' instance in the 'main.py' (now 'webui.main')
        host=_webui_host,
        port=_webui_port,
        log_level=_app_log_level_str.lower(), # Uvicorn expects lowercase string
        reload=False # Typically, reload is False for installed CLI tools. Set to True for dev.
    )

# The existing if __name__ == "__main__": block can call this new function
# or be kept for direct script execution testing if preferred.
if __name__ == "__main__":
    # This block is for when you run `python webui/main.py` directly.
    # The `launch_webui` function will handle CWD changes.
    launch_webui()