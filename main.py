import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import sys
import logging
from llama_cpp import Llama

# -----------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout) # Log to console
    ]
)
log = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Warcamp API Initializing
# -----------------------------------------------------------------

# Read the Model Directory from the environment variable set by the PowerShell script
MODELS_ROOT = os.environ.get("WARCAMP_MODELS_ROOT")
if not MODELS_ROOT:
    log.error("FATAL ERROR: WARCAMP_MODELS_ROOT environment variable not set.")
    sys.exit(1)

log.info(f"Warcamp 🏕️ initializing... Model Root: {MODELS_ROOT}")

app = FastAPI(
    title="Warcamp 🏕️ - Dev Orch Backend",
    description="API for orchestrating local LLM agents (Gemma, CodeGemma) via llama-cpp-python.",
    version="0.1.0"
)

# -----------------------------------------------------------------
# Pydantic Models (Request/Response Bodies)
# -----------------------------------------------------------------
class LoadModelRequest(BaseModel):
    model_name: str # e.g., "TheCouncil_Gemma1b"
    model_filename: str # e.g., "gemma-2b-it.gguf". Must be inside MODELS_ROOT
    n_gpu_layers: int = -1 # -1 = all layers, 0 = no layers
    n_ctx: int = 4096

class GenerateRequest(BaseModel):
    model_name: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

class MemoryQueryRequest(BaseModel):
    query: str
    top_k: int = 3

class AdminExecRequest(BaseModel):
    command: str # The shell command to execute

class AdminExecResponse(BaseModel):
    stdout: str
    stderr: str
    return_code: int

# -----------------------------------------------------------------
# Global State (In-memory stores)
# -----------------------------------------------------------------
# This dict will hold our loaded model instances ("agents")
# e.g., {"TheCouncil_Gemma1b": <Llama object>}
loaded_models = {}

# -----------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------

@app.get("/", tags=["Status"])
async def get_root():
    """Root endpoint to check if the Warcamp is online."""
    return {"message": "Warcamp is online. The Council is listening."}

# --- Model Management ---

@app.post("/api/v1/models/load", tags=["Model Management"])
async def load_model(request: LoadModelRequest):
    """
    Load a GGUF model into VRAM and make it available for inference.
    """
    if request.model_name in loaded_models:
        log.warning(f"Load request for '{request.model_name}', which is already loaded.")
        return {"message": f"Model '{request.model_name}' is already loaded."}

    model_path = os.path.join(MODELS_ROOT, request.model_filename)
    
    if not os.path.exists(model_path):
        log.error(f"Model file not found at path: {model_path}")
        raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_filename}")

    log.info(f"Loading model '{request.model_name}' from '{model_path}'...")
    log.info(f"  n_gpu_layers: {request.n_gpu_layers}, n_ctx: {request.n_ctx}")

    try:
        model = Llama(
            model_path=model_path,
            n_gpu_layers=request.n_gpu_layers,
            n_ctx=request.n_ctx,
            verbose=True
        )
        loaded_models[request.model_name] = model
        log.info(f"SUCCESS: Model '{request.model_name}' is loaded and online.")
        return {"message": f"Model '{request.model_name}' loaded successfully."}
    except Exception as e:
        log.error(f"Failed to load model '{request.model_name}'. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/unload", tags=["Model Management"])
async def unload_model(model_name: str):
    """
    Unload a model from VRAM to free up resources.
    """
    if model_name not in loaded_models:
        log.warning(f"Unload request for '{model_name}', but it's not loaded.")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    try:
        log.info(f"Unloading model '{model_name}'...")
        # Get the model, delete it, and force garbage collection
        model_instance = loaded_models.pop(model_name)
        del model_instance
        # TODO: Add torch.cuda.empty_cache() if using GPU tensors explicitly
        
        log.info(f"SUCCESS: Model '{model_name}' unloaded and VRAM freed.")
        return {"message": f"Model '{model_name}' unloaded successfully."}
    except Exception as e:
        log.error(f"An error occurred while unloading '{model_name}'. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/list", tags=["Model Management"])
async def list_models():
    """
    List all currently loaded models.
    """
    return {"loaded_models": list(loaded_models.keys())}

# --- Inference ---

@app.post("/api/v1/generate", tags=["Inference"])
async def generate_completion(request: GenerateRequest):
    """
    Run inference on a pre-loaded model.
    """
    if request.model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found.")

    # TODO: Get model from 'loaded_models'
    model = loaded_models[request.model_name]
    
    # TODO: Run model.create_completion(prompt=...)
    
    # Placeholder response
    return {"message": f"TODO: Run generation on '{request.model_name}' with prompt: '{request.prompt[:50]}...'"}

# --- RAG/Memory ---

@app.post("/api/v1/memory/query", tags=["RAG / Memory"])
async def query_memory(request: MemoryQueryRequest):
    """
    Query the RAG vector store with a string.
    """
    # TODO: Add EmbeddingGemma/Vector DB logic
    return {"message": "TODO: Query vector store", "query": request.query}

# --- Admin Shell ---

@app.post("/api/v1/admin/exec", tags=["Admin"], response_model=AdminExecResponse)
async def admin_exec(request: AdminExecRequest):
    """
    (SECURE) Execute a shell command on the Warcamp host.
    This is for the Council to self-manage the environment.
    """
    # TODO: Add security (e.g., API key)
    log.info(f"AdminExec executing command: {request.command}")
    try:
        process = subprocess.run(
            request.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30 # 30-second timeout
        )
        return AdminExecResponse(
            stdout=process.stdout,
            stderr=process.stderr,
            return_code=process.returncode
        )
    except Exception as e:
        log.error(f"AdminExec failed. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Entry Point ---

if __name__ == "__main__":
    """
    This block allows running the server directly for debugging.
    'python main.py'
    
    For production, you would use:
    'uvicorn main:app --host 0.0.0.0 --port 8000 --reload'
    """
    log.info(f"Starting Uvicorn server in debug mode (reload=True)...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)