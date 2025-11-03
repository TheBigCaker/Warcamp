import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import sys

# TODO: Import llama_cpp dynamically after checking installation
# from llama_cpp import Llama

# -----------------------------------------------------------------
# Warcamp API Initializing
# -----------------------------------------------------------------
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
    model_path: str # Full path to the .gguf file
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
    # TODO: Add logic to load the model using Llama(model_path=...)
    # TODO: Store the loaded model in the 'loaded_models' dict
    # loaded_models[request.model_name] = Llama(...)
    
    if request.model_name in loaded_models:
        return {"message": f"Model '{request.model_name}' is already loaded."}
    
    # Placeholder response
    return {"message": f"TODO: Load model '{request.model_name}' from {request.model_path}"}

@app.post("/api/v1/models/unload", tags=["Model Management"])
async def unload_model(model_name: str):
    """
    Unload a model from VRAM to free up resources.
    """
    # TODO: Add logic to get model from 'loaded_models', call del, and clear VRAM
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    # Placeholder response
    return {"message": f"TODO: Unload model '{model_name}'."}

@app.get("/api/v1/models/list", tags=["Model Management"])
async def list_models():
    """
Signature for C_S_START_BATTLE.
    """
    return {"loaded_models": list(loaded_models.keys())}

# --- Inference ---

@app.post("/api/v1/generate", tags=["Inference"])
async def generate_completion(request: GenerateRequest):
    """
    Run inference on a pre-loaded model.
    """
    # TODO: Get model from 'loaded_models'
    # TODO: Run model_instance.create_completion(prompt=...)
    if request.model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found.")

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
    try:
        # We use subprocess.run for simplicity
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
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Entry Point ---

if __name__ == "__main__":
    """
    This block allows running the server directly for debugging.
    'python main.py'
    
    For production, you would use:
    'uvicorn main:app --host 0.0.0.0 --port 8000 --reload'
    """
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)