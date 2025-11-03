import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import subprocess
import os
import sys
import logging
import json
from typing import Iterator
from enum import Enum
from llama_cpp import Llama
from dotenv import load_dotenv

# --- RAG Imports ---
import numpy as np
import faiss

# -----------------------------------------------------------------
# Environment & Logging Setup
# -----------------------------------------------------------------

# Load environment variables from .env file (for VS Code/Visual Studio debugging)
load_dotenv() 

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

# Read the Model Directory from the environment variable
MODELS_ROOT = os.environ.get("WARCAMP_MODELS_ROOT")
if not MODELS_ROOT:
    log.error("FATAL ERROR: WARCAMP_MODELS_ROOT environment variable not set.")
    log.error("Ensure it is set in your .env file or system environment.")
    sys.exit(1)

# Read the Port from the environment variable, default to 8001
APP_PORT = int(os.environ.get("WARCAMP_PORT", 8001))

log.info(f"Warcamp 🏕️ initializing... Model Root: {MODELS_ROOT}")

# --- Dynamic Enum Creation for Swagger UI ---
def _get_model_file_enum() -> Enum:
    """
    Scans the MODELS_ROOT directory and creates a dynamic Enum of .gguf files
    for use in the Swagger UI drop-down.
    """
    if not os.path.exists(MODELS_ROOT):
        log.warning(f"Model directory not found: {MODELS_ROOT}. Creating placeholder enum.")
        return Enum("ModelFileEnum", {"NO_MODELS_FOUND": "NO_MODELS_FOUND"})
    
    try:
        model_files = [f for f in os.listdir(MODELS_ROOT) if f.endswith(".gguf")]
        if not model_files:
            log.warning(f"No .gguf models found in {MODELS_ROOT}. Creating placeholder enum.")
            return Enum("ModelFileEnum", {"NO_MODELS_FOUND": "NO_MODELS_FOUND"})
            
        # Create an Enum where the name and value are both the filename
        # e.g., "gemma-1b.gguf" = "gemma-1b.gguf"
        return Enum("ModelFileEnum", {f: f for f in model_files})
    except Exception as e:
        log.error(f"Error scanning model directory '{MODELS_ROOT}': {e}")
        return Enum("ModelFileEnum", {"ERROR_SCANNING_MODELS": "ERROR_SCANNING_MODELS"})

ModelFileEnum = _get_model_file_enum()
# --- End Dynamic Enum Creation ---


# --- RAG System Initialization ---
# This system is now based on a GGUF embedding model
embedding_model_instance = None
faiss_index = None
document_store = []
# Dimension corrected in V2.7 from 256 to 768 based on model metadata
EMBEDDING_DIM = 768 # Specific to embeddinggemma-300M

try:
    EMBEDDING_MODEL_FILENAME = os.environ.get("WARCAMP_EMBEDDING_MODEL_FILENAME")
    if not EMBEDDING_MODEL_FILENAME:
        log.error("FATAL ERROR: WARCAMP_EMBEDDING_MODEL_FILENAME not set in .env")
        raise ValueError("Embedding model filename not specified.")
        
    embedding_model_path = os.path.join(MODELS_ROOT, EMBEDDING_MODEL_FILENAME)
    if not os.path.exists(embedding_model_path):
        log.error(f"FATAL ERROR: Embedding model file not found at: {embedding_model_path}")
        raise FileNotFoundError(f"Missing embedding model: {EMBEDDING_MODEL_FILENAME}")

    log.info(f"Loading RAG embedding model from: {embedding_model_path}...")
    
    # Load the GGUF model in embedding mode
    embedding_model_instance = Llama(
        model_path=embedding_model_path,
        embedding=True,
        n_ctx=512, # Context for embeddings can be smaller
        n_gpu_layers=-1, # Offload embedding model fully to GPU
        verbose=True
    )
    
    log.info(f"Embedding model '{EMBEDDING_MODEL_FILENAME}' loaded successfully.")

    # Create a FAISS index (Flat, L2 distance)
    # This index lives in RAM
    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # In-memory store for the actual text content (the "documents")
    log.info(f"In-memory FAISS index (IndexFlatL2, Dim={EMBEDDING_DIM}) initialized.")

except Exception as e:
    log.error(f"FATAL ERROR: Could not initialize RAG system: {e}")
    sys.exit(1)
# --- End RAG Initialization ---


app = FastAPI(
    title="Warcamp 🏕️ - Dev Orch Backend",
    description="API for orchestrating local LLM agents (Gemma, CodeGemma) via llama-cpp-python.",
    version="0.1.0"
)

# -----------------------------------------------------------------
# Pydantic Models (Request/Response Bodies)
# -----------------------------------------------------------------
class LoadModelRequest(BaseModel):
    model_name: str = Field(..., example="council") # The nickname you will use for this model
    model_filename: ModelFileEnum # This will be a drop-down in /docs
    n_gpu_layers: int = -1 # -1 = all layers, 0 = no layers
    n_ctx: int = 4096

class UnloadModelRequest(BaseModel):
    model_name: str = Field(..., example="council") # The nickname of the model to unload

class GenerateRequest(BaseModel):
    model_name: str = Field(..., example="council") # This must match a name you've already loaded
    prompt: str = Field(..., example="USER: Hello, what is your status?\nASSISTANT:")
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True # Controls whether to stream the response

class MemoryAddRequest(BaseModel):
    text: str = Field(..., example="Dev Orch is an AI-driven system to write software.")
    doc_id: str = Field(..., example="doc_id_001") # A unique ID for this text chunk

class MemoryQueryRequest(BaseModel):
    query: str = Field(..., example="What is Dev Orch?")
    top_k: int = 3

class AdminExecRequest(BaseModel):
    command: str = Field(..., example="ls -l") # The shell command to execute

class AdminExecResponse(BaseModel):
    stdout: str
    stderr: str
    return_code: int

# -----------------------------------------------------------------
# Global State (In-memory stores)
# -----------------------------------------------------------------
# This dict will hold our loaded model instances ("agents")
# e.g., {"council": <Llama object>}
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

    # Convert Enum member to its string value (the filename)
    model_filename_str = request.model_filename.value
    model_path = os.path.join(MODELS_ROOT, model_filename_str)
    
    if not os.path.exists(model_path):
        log.error(f"Model file not found at path: {model_path}")
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_filename_str}")
    
    # Check that we are not trying to load the embedding model as a chat model
    if model_filename_str == EMBEDDING_MODEL_FILENAME:
        log.warning(f"Preventing load: '{model_filename_str}' is the active embedding model.")
        raise HTTPException(status_code=400, detail="Cannot load the active embedding model as a chat model.")

    log.info(f"Loading model '{request.model_name}' from '{model_path}'...")
    log.info(f"  n_gpu_layers: {request.n_gpu_layers}, n_ctx: {request.n_ctx}")

    try:
        model = Llama(
            model_path=model_path,
            n_gpu_layers=request.n_gpu_layers,
            n_ctx=request.n_ctx,
            verbose=True
            # Note: embedding=False is the default
        )
        loaded_models[request.model_name] = model
        log.info(f"SUCCESS: Model '{request.model_name}' is loaded and online.")
        return {"message": f"Model '{request.model_name}' loaded successfully."}
    except Exception as e:
        log.error(f"Failed to load model '{request.model_name}'. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/models/unload", tags=["Model Management"])
async def unload_model(request: UnloadModelRequest):
    """
    Unload a model from VRAM to free up resources.
    """
    model_name = request.model_name
    if model_name not in loaded_models:
        log.warning(f"Unload request for '{model_name}', but it's not loaded.")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    
    try:
        log.info(f"Unloading model '{model_name}'...")
        # Get the model, delete it, and force garbage collection
        model_instance = loaded_models.pop(model_name)
        del model_instance
        
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

async def stream_generator(model_name: str, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
    """
    Generator function to yield streaming completion chunks.
    This allows us to send the response token by token.
    """
    log.info(f"Starting generation stream for '{model_name}'...")
    try:
        model = loaded_models.get(model_name)
        if not model:
            log.error(f"Model '{model_name}' not found in stream_generator.")
            yield f"data: {json.dumps({'error': f'Model {model_name} not found.'})}\n\n"
            return

        # Create the completion stream
        stream = model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        # Yield each chunk as a Server-Sent Event (SSE)
        for output in stream:
            chunk = output.get('choices', [{}])[0].get('text', '')
            if chunk:
                # Format as Server-Sent Event (SSE)
                # This 'data: ... \n\n' format is crucial
                yield f"data: {json.dumps({'token': chunk})}\n\n"
        
        log.info(f"Generation stream for '{model_name}' complete.")
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    except Exception as e:
        log.error(f"Error during generation stream for '{model_name}': {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/api/v1/generate", tags=["Inference"])
async def generate_completion(request: GenerateRequest):
    """
    Run inference on a pre-loaded model.
    """
    if request.model_name not in loaded_models:
        log.error(f"Generate request for '{request.model_name}', but it's not loaded.")
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found.")

    model = loaded_models[request.model_name]
    
    if request.stream:
        # Use the streaming generator
        return StreamingResponse(
            stream_generator(
                model_name=request.model_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ),
            media_type="text/event-stream"
        )
    else:
        # Perform a blocking, non-streaming completion
        log.info(f"Starting non-streaming generation for '{request.model_name}'...")
        try:
            output = model.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False
            )
            log.info(f"Non-streaming generation for '{request.model_name}' complete.")
            return output
        except Exception as e:
            log.error(f"Error during non-streaming generation for '{request.model_name}': {e}")
            raise HTTPException(status_code=500, detail=str(e))

# --- RAG/Memory ---

@app.post("/api/v1/memory/add", tags=["RAG / Memory"])
async def add_to_memory(request: MemoryAddRequest):
    """
    Add a chunk of text to the in-memory vector store.
    """
    try:
        log.info(f"Adding to memory, doc_id: {request.doc_id}")
        
        # 1. Create embedding for the text using our GGUF model
        vector_list = embedding_model_instance.embed(request.text)
        
        # 2. Convert to NumPy array, ensure float32, and reshape for FAISS
        vector_np = np.array(vector_list).astype('float32').reshape(1, -1)
        
        if vector_np.shape[1] != EMBEDDING_DIM:
            log.error(f"Embedding dimension mismatch! Model returned {vector_np.shape[1]} but index requires {EMBEDDING_DIM}.")
            raise ValueError("Embedding dimension mismatch.")
            
        # 3. Add vector to FAISS index
        faiss_index.add(vector_np)
        
        # 4. Store the text content, keyed by the new index ID
        new_index_id = faiss_index.ntotal - 1
        document_store.append({
            "id": request.doc_id,
            "text": request.text
        })
        
        # Sanity check
        if new_index_id != (len(document_store) - 1):
            log.warning("FAISS index and document store are out of sync!")
            
        log.info(f"Successfully added doc_id '{request.doc_id}' to vector store. Total documents: {faiss_index.ntotal}")
        return {
            "message": "Text added to memory successfully.",
            "new_index_id": new_index_id,
            "doc_id": request.doc_id,
            "total_documents": faiss_index.ntotal
        }
    except Exception as e:
        log.error(f"Failed to add text to memory. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/memory/query", tags=["RAG / Memory"])
async def query_memory(request: MemoryQueryRequest):
    """
    Query the RAG vector store with a string.
    """
    # TODO: Implement the query logic:
    # 1. Create embedding for request.query using embedding_model_instance.embed()
    # 2. Convert to NumPy array and reshape
    # 3. Use faiss_index.search(embedding, k=request.top_k)
    # 4. Get the indices (I) and distances (D) from the result
    # 5. Loop through the indices (I[0])
    # 6. Use each index i to look up the original text in document_store[i]
    # 7. Return the found texts
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
    log.info(f"Starting Uvicorn server in debug mode (reload=True) on port {APP_PORT}...")
    uvicorn.run("main:app", host="127.0.0.1", port=APP_PORT, reload=True)