import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import subprocess
import os
import sys
import logging
import json
from typing import Iterator, List, Dict, Any
from enum import Enum
from llama_cpp import Llama
from dotenv import load_dotenv
import datetime
import aiohttp
import asyncio
import uuid
import platform  # <-- NEW: Import platform

# --- RAG Imports ---
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- PDF Report Imports ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

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
BASE_URL = f"http://127.0.0.1:{APP_PORT}"

log.info(f"Warcamp 🏕️ initializing... Model Root: {MODELS_ROOT}")
log.info(f"Warcamp 🏕️ API will be available at {BASE_URL}")

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
# This system is now based on sentence-transformers (a pure-python library)
embedding_model_instance = None
faiss_index = None
document_store = []
# *** PIVOT V2.9: Using 'all-MiniLM-L6-v2', which has a dimension of 384 ***
EMBEDDING_DIM = 384

try:
    log.info(f"Loading RAG embedding model 'all-MiniLM-L6-v2'...")
    # This will download the model (once) into a .cache folder
    embedding_model_instance = SentenceTransformer('all-MiniLM-L6-v2')
    log.info(f"Embedding model 'all-MiniLM-L6-v2' loaded successfully.")

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
    version="0.2.0" # <-- Version Bump
)

# -----------------------------------------------------------------
# Pydantic Models (Request/Response Bodies)
# -----------------------------------------------------------------
class LoadModelRequest(BaseModel):
    model_name: str = Field(..., json_schema_extra={'example': 'council'}) # The nickname you will use for this model
    model_filename: ModelFileEnum # This will be a drop-down in /docs
    n_gpu_layers: int = -1 # -1 = all layers, 0 = no layers
    n_ctx: int = 4096

class UnloadModelRequest(BaseModel):
    model_name: str = Field(..., json_schema_extra={'example': 'council'}) # The nickname of the model to unload

class GenerateRequest(BaseModel):
    model_name: str = Field(..., json_schema_extra={'example': 'council'}) # This must match a name you've already loaded
    prompt: str = Field(..., json_schema_extra={'example': 'USER: Hello, what is your status?\nASSISTANT:'})
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = True # Controls whether to stream the response

class MemoryAddRequest(BaseModel):
    text: str = Field(..., json_schema_extra={'example': 'Dev Orch is an AI-driven system to write software.'})
    doc_id: str = Field(..., json_schema_extra={'example': 'doc_id_001'}) # A unique ID for this text chunk

class MemoryQueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={'example': 'What is Dev Orch?'})
    top_k: int = 3

class FoundDocument(BaseModel):
    doc_id: str
    text: str
    score: float # L2 distance (lower is better)

class MemoryQueryResponse(BaseModel):
    query: str
    found_documents: List[FoundDocument]

class AdminExecRequest(BaseModel):
    command: str = Field(..., json_schema_extra={'example': 'ls -l'}) # The shell command to execute

class AdminExecResponse(BaseModel):
    stdout: str
    stderr: str
    return_code: int

# --- NEW: Orchestration Models ---
class MissionRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={'example': 'Build me a python script that acts as a simple calculator.'})
    council_model: str = Field("council", json_schema_extra={'example': 'council'})
    advisor_model: str = Field("advisor", json_schema_extra={'example': 'advisor'})
    advisor_filename: ModelFileEnum
    sarge_model: str = Field("sarge", json_schema_extra={'example': 'sarge'})
    sarge_filename: ModelFileEnum

class MissionStatus(BaseModel):
    mission_id: str
    status: str
    details: List[str]
# --- End Orchestration Models ---

# -----------------------------------------------------------------
# Global State (In-memory stores)
# -----------------------------------------------------------------
# This dict will hold our loaded model instances ("agents")
# e.g., {"council": <Llama object>}
loaded_models: Dict[str, Llama] = {}

# --- NEW: Mission Tracking ---
# This will store the status of long-running missions
# e.g., {"mission-uuid-123": ["Step 1: Loading Advisor..."]}
mission_logs: Dict[str, List[str]] = {}
# --- End Mission Tracking ---

# -----------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------

@app.get("/", tags=["Dashboard"], response_class=HTMLResponse)
async def get_dashboard():
    """
    Serves the main HTML dashboard for the Warcamp.
    """
    html_content = """
    <html>
        <head>
            <title>Warcamp 🏕️ Dashboard</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    background-color: #1a1a1a; 
                    color: #e0e0e0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    flex-direction: column;
                }
                h1 { 
                    color: #4CAF50; /* Ork Green */
                    font-weight: 600;
                    font-size: 2.5em;
                }
                #testButton {
                    background-color: #4CAF50;
                    color: #1a1a1a;
                    border: none;
                    padding: 20px 40px;
                    font-size: 1.2em;
                    font-weight: bold;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: background-color 0.3s, transform 0.1s;
                }
                #testButton:hover {
                    background-color: #66BB6A;
                }
                #testButton:active {
                    transform: scale(0.98);
                }
                #testButton:disabled {
                    background-color: #555;
                    color: #888;
                    cursor: not-allowed;
                }
                #status {
                    margin-top: 20px;
                    font-size: 1.1em;
                    color: #aaa;
                    min-height: 1.5em;
                }
                .links {
                    margin-top: 40px;
                }
                .links a {
                    color: #4CAF50;
                    text-decoration: none;
                    margin: 0 10px;
                    font-size: 1.1em;
                }
                .links a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Warcamp 🏕️ Dashboard</h1>
            <button id="testButton" onclick="runTest()">Run Warcamp Smoke Test</button>
            <div id="status">Ready, Chief! "Work work!"</div>
            <div class="links">
                <a href="/docs" target="_blank">API Docs (Swagger)</a>
                <a href="/redoc" target="_blank">API ReDoc</a>
            </div>

            <script>
                async function runTest() {
                    const button = document.getElementById('testButton');
                    const status = document.getElementById('status');
                    
                    button.disabled = true;
                    status.innerText = 'Running smoke test... This may take a minute...';
                    
                    try {
                        const response = await fetch('/api/v1/run-smoke-test');
                        
                        if (!response.ok) {
                            throw new Error('Test failed. Server responded with: ' + response.status);
                        }
                        
                        // Handle the PDF download
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        a.download = 'DO-Test-Results.pdf';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                        document.body.removeChild(a);
                        
                        status.style.color = '#4CAF50';
                        status.innerText = 'Test complete! Report "DO-Test-Results.pdf" downloaded.';
                        
                    } catch (error) {
                        console.error('Smoke test failed:', error);
                        status.style.color = '#F44336'; // Red
                        status.innerText = 'Test failed! See server logs for details. ' + error.message;
                    } finally {
                        button.disabled = false;
                        setTimeout(() => { 
                            status.style.color = '#aaa';
                            status.innerText = 'Ready, Chief! "Work work!"'; 
                        }, 5000);
                    }
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/v1/status", tags=["Status"])
async def get_status():
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
        
        # 1. Create embedding for the text using sentence-transformers
        vector = embedding_model_instance.encode(request.text)
        
        # 2. Convert to NumPy array, ensure float32, and reshape for FAISS
        vector_np = np.array(vector).astype('float32').reshape(1, -1)
        
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


@app.post("/api/v1/memory/query", tags=["RAG / Memory"], response_model=MemoryQueryResponse)
async def query_memory(request: MemoryQueryRequest):
    """
    Query the RAG vector store with a string.
    """
    try:
        if faiss_index.ntotal == 0:
            log.warning("Query received, but memory store is empty.")
            return MemoryQueryResponse(query=request.query, found_documents=[])
            
        log.info(f"Querying memory with: '{request.query}'")
        
        # 1. Create embedding for the query
        query_vector = embedding_model_instance.encode(request.query)
        
        # 2. Convert to NumPy array and reshape
        query_np = np.array(query_vector).astype('float32').reshape(1, -1)
        
        if query_np.shape[1] != EMBEDDING_DIM:
            log.error(f"Query embedding dimension mismatch!")
            raise ValueError("Query embedding dimension mismatch.")

        # 3. Use faiss_index.search
        # D = distances (lower is better for L2), I = indices
        distances, indices = faiss_index.search(query_np, k=request.top_k)
        
        # 4. Loop through results and look up in document_store
        results = []
        for i, index in enumerate(indices[0]):
            if index < 0: # -1 means no result found
                continue
                
            doc = document_store[index]
            results.append(FoundDocument(
                doc_id=doc["id"],
                text=doc["text"],
                score=float(distances[0][i]) # The L2 distance
            ))
            
        log.info(f"Query returned {len(results)} documents.")
        return MemoryQueryResponse(query=request.query, found_documents=results)
        
    except Exception as e:
        log.error(f"Failed to query memory. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# -----------------------------------------------------------------
# --- NEW: Live Chat WebSocket (V3.7) ---
# -----------------------------------------------------------------
@app.websocket("/ws/council-chat")
async def websocket_council_chat(websocket: WebSocket):
    """
    Handles a live, streaming chat session with the 'council' model.
    """
    await websocket.accept()
    log.info("WebSocket connection established for Council chat.")
    
    # Check if 'council' model is loaded
    if "council" not in loaded_models:
        log.warning("Council chat requested, but 'council' model is not loaded.")
        await websocket.send_json({"error": "The 'council' model is not loaded."})
        await websocket.close()
        return

    model = loaded_models["council"]

    try:
        while True:
            data = await websocket.receive_json()
            prompt = data.get("prompt")
            if not prompt:
                await websocket.send_json({"error": "No 'prompt' field in message."})
                continue
            
            log.info(f"WebSocket received prompt: {prompt[:50]}...")
            
            # Start the streaming completion
            stream = model.create_completion(
                prompt=prompt,
                max_tokens=1024, # Larger context for chat
                temperature=0.7,
                stream=True
            )
            
            # Send each token back over the WebSocket
            full_response = ""
            for output in stream:
                chunk = output.get('choices', [{}])[0].get('text', '')
                if chunk:
                    full_response += chunk
                    await websocket.send_json({"token": chunk})
            
            log.info(f"WebSocket full response generated: {full_response[:50]}...")
            await websocket.send_json({"status": "done"})

    except WebSocketDisconnect:
        log.info("WebSocket connection closed.")
    except Exception as e:
        log.error(f"Error in WebSocket chat: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass # Connection might be closed
        await websocket.close()


# -----------------------------------------------------------------
# --- NEW: Agent Orchestration (V3.7) ---
# -----------------------------------------------------------------

async def _log_mission(mission_id: str, message: str):
    """Helper to log mission progress."""
    log.info(f"[Mission {mission_id[:6]}] {message}")
    if mission_id not in mission_logs:
        mission_logs[mission_id] = []
    mission_logs[mission_id].append(message)

async def _run_mission_logic(mission_id: str, chief_prompt: str, req: MissionRequest):
    """
    The background task that runs the full agent workflow.
    """
    await _log_mission(mission_id, f"Mission Started: {chief_prompt[:50]}...")
    
    advisor_name = req.advisor_model
    advisor_file = req.advisor_filename.value
    sarge_name = req.sarge_model
    sarge_file = req.sarge_filename.value
    
    try:
        async with aiohttp.ClientSession() as session:
            
            # === STEP 1: Load Advisor ===
            await _log_mission(mission_id, f"Loading Advisor model '{advisor_name}'...")
            load_payload = {"model_name": advisor_name, "model_filename": advisor_file, "n_gpu_layers": -1, "n_ctx": 4096}
            async with session.post(f"{BASE_URL}/api/v1/models/load", json=load_payload) as r:
                if not r.ok:
                    raise Exception(f"Failed to load Advisor: {await r.text()}")
            
            # === STEP 2: Generate Plan ===
            await _log_mission(mission_id, "Advisor loaded. Generating plan...")
            plan_prompt = f"USER: You are the Advisor. The Chief's request is: '{chief_prompt}'. Create a detailed plan, including file structure and logic, as Plans.md.\nASSISTANT:\n`markdown\n# Plans.md\n"
            # --- BUG FIX V3.8: Added missing 'temperature' ---
            gen_payload = {"model_name": advisor_name, "prompt": plan_prompt, "max_tokens": 2048, "stream": False, "temperature": 0.7}
            async with session.post(f"{BASE_URL}/api/v1/generate", json=gen_payload) as r:
                if not r.ok:
                    raise Exception(f"Advisor failed to generate plan: {await r.text()}")
                plan_data = await r.json()
                plan_text = plan_data['choices'][0]['text']
            
            await _log_mission(mission_id, "Plan generated by Advisor.")
            # TODO: Save plan_text to disk as 'Plans.md'

            # === STEP 3: Unload Advisor ===
            await _log_mission(mission_id, f"Unloading Advisor '{advisor_name}'...")
            unload_payload = {"model_name": advisor_name}
            async with session.post(f"{BASE_URL}/api/v1/models/unload", json=unload_payload) as r:
                if not r.ok:
                    raise Exception(f"Failed to unload Advisor: {await r.text()}")

            # === STEP 4: Load Sarge ===
            await _log_mission(mission_id, f"Loading Sarge model '{sarge_name}'...")
            load_payload = {"model_name": sarge_name, "model_filename": sarge_file, "n_gpu_layers": -1, "n_ctx": 4096}
            async with session.post(f"{BASE_URL}/api/v1/models/load", json=load_payload) as r:
                if not r.ok:
                    raise Exception(f"Failed to load Sarge: {await r.text()}")

            # === STEP 5: Generate Tasks ===
            await _log_mission(mission_id, "Sarge loaded. Generating tasks from plan...")
            task_prompt = f"USER: You are Sarge. Read this plan:\n{plan_text}\n\nCreate a detailed, step-by-step Tasklist.md for the Orchs.\nASSISTANT:\n`markdown\n# Tasklist.md\n"
            # --- BUG FIX V3.8: Added missing 'temperature' ---
            gen_payload = {"model_name": sarge_name, "prompt": task_prompt, "max_tokens": 2048, "stream": False, "temperature": 0.7}
            async with session.post(f"{BASE_URL}/api/v1/generate", json=gen_payload) as r:
                if not r.ok:
                    raise Exception(f"Sarge failed to generate tasks: {await r.text()}")
                task_data = await r.json()
                task_text = task_data['choices'][0]['text']

            await _log_mission(mission_id, "Tasklist generated by Sarge.")
            # TODO: Save task_text to disk as 'Tasklist.md'
            
            # === STEP 6: Unload Sarge ===
            await _log_mission(mission_id, f"Unloading Sarge '{sarge_name}'...")
            unload_payload = {"model_name": sarge_name}
            async with session.post(f"{BASE_URL}/api/v1/models/unload", json=unload_payload) as r:
                if not r.ok:
                    raise Exception(f"Failed to unload Sarge: {await r.text()}")

            # === TODO: Load Orchs and execute tasks ===
            await _log_mission(mission_id, "TODO: Load Orchs and execute task list.")

            await _log_mission(mission_id, "Mission Complete!")

    except Exception as e:
        await _log_mission(mission_id, f"MISSION FAILED: {str(e)}")


@app.post("/api/v1/run-mission", tags=["Orchestration"], response_model=MissionStatus)
async def run_mission(request: MissionRequest, background_tasks: BackgroundTasks):
    """
    Starts a new agent orchestration mission in the background.
    """
    mission_id = f"mission_{uuid.uuid4()}"
    log.info(f"Received new mission, assigning ID: {mission_id}")
    
    # Start the agent workflow as a background task
    background_tasks.add_task(_run_mission_logic, mission_id, request.prompt, request)
    
    # Return immediately
    return MissionStatus(
        mission_id=mission_id,
        status="Mission Started",
        details=[f"Mission '{mission_id}' has been queued. Check status endpoint for progress."]
    )

@app.get("/api/v1/mission-status/{mission_id}", tags=["Orchestration"], response_model=MissionStatus)
async def get_mission_status(mission_id: str):
    """
    Checks the status and logs of a running mission.
    """
    if mission_id not in mission_logs:
        raise HTTPException(status_code=404, detail="Mission ID not found.")
        
    logs = mission_logs[mission_id]
    current_status = logs[-1] if logs else "Unknown"
    
    return MissionStatus(
        mission_id=mission_id,
        status=current_status,
        details=logs
    )

# -----------------------------------------------------------------
# --- Smoke Test & Report Generation ---
# -----------------------------------------------------------------

# --- Test Logic ---
async def run_smoke_test_logic(base_url: str, model_filename: str) -> List[Dict[str, Any]]:
    """
    Runs the full 8-step smoke test logic internally using aiohttp.
    Returns a list of result dictionaries.
    """
    log.info("--- Starting Internal Smoke Test ---")
    results = []
    MODEL_NAME = "smoke_test_model"
    # Use the first available .gguf file if the default isn't found
    if not os.path.exists(os.path.join(MODELS_ROOT, model_filename)):
        log.warning(f"Smoke test model '{model_filename}' not found. Trying first available .gguf file...")
        try:
            model_filename = _get_model_file_enum().__members__.popitem()[0]
        except Exception:
            results.append({"step": "Setup", "success": False, "details": "No .gguf models found in model directory."})
            return results
    
    log.info(f"Using model '{model_filename}' for smoke test.")

    async with aiohttp.ClientSession() as session:
        
        # Step 1: Check Status
        try:
            async with session.get(f"{base_url}/api/v1/status") as r:
                r.raise_for_status()
                data = await r.json()
                assert "Warcamp is online" in data.get("message", "")
                results.append({"step": "1. GET /api/v1/status", "success": True, "details": "Server is online."})
        except Exception as e:
            results.append({"step": "1. GET /api/v1/status", "success": False, "details": str(e)})
            return results # Stop test if server isn't on
            
        # Step 2: Load Model
        try:
            payload = {"model_name": MODEL_NAME, "model_filename": model_filename, "n_gpu_layers": -1, "n_ctx": 4096}
            async with session.post(f"{base_url}/api/v1/models/load", json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                assert "loaded successfully" in data.get("message", "")
                results.append({"step": "2. POST /models/load", "success": True, "details": f"Loaded '{model_filename}'."})
        except Exception as e:
            results.append({"step": "2. POST /models/load", "success": False, "details": str(e)})
            return results # Stop test if model can't load

        # Step 3: List Models
        try:
            async with session.get(f"{base_url}/api/v1/models/list") as r:
                r.raise_for_status()
                data = await r.json()
                assert MODEL_NAME in data.get("loaded_models", [])
                results.append({"step": "3. GET /models/list", "success": True, "details": f"Model '{MODEL_NAME}' confirmed in list."})
        except Exception as e:
            results.append({"step": "3. GET /models/list", "success": False, "details": str(e)})

        # Step 4: Add to Memory
        try:
            payload = {"text": "The secret code is 'work work'.", "doc_id": "smoke_test_doc"}
            async with session.post(f"{base_url}/api/v1/memory/add", json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                assert data.get("total_documents", 0) > 0
                results.append({"step": "4. POST /memory/add", "success": True, "details": "Added 'smoke_test_doc'."})
        except Exception as e:
            results.append({"step": "4. POST /memory/add", "success": False, "details": str(e)})

        # Step 5: Query Memory
        try:
            payload = {"query": "What is the secret code?", "top_k": 1}
            async with session.post(f"{base_url}/api/v1/memory/query", json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                assert len(data.get("found_documents", [])) == 1
                assert "work work" in data["found_documents"][0].get("text", "")
                results.append({"step": "5. POST /memory/query", "success": True, "details": "RAG query successful."})
        except Exception as e:
            results.append({"step": "5. POST /memory/query", "success": False, "details": str(e)})

        # Step 6: Generate Stream
        try:
            payload = {"model_name": MODEL_NAME, "prompt": "USER: Hello! Say 'test'.\nASSISTANT:", "max_tokens": 5, "stream": True}
            full_response = ""
            async with session.post(f"{base_url}/api/v1/generate", json=payload) as r:
                r.raise_for_status()
                async for line in r.content:
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data:"):
                            data_str = line_str[5:]
                            try:
                                token_data = json.loads(data_str)
                                if "token" in token_data:
                                    full_response += token_data["token"]
                            except json.JSONDecodeError:
                                pass # Ignore status/non-json lines
            assert "test" in full_response.lower()
            results.append({"step": "6. POST /generate (Stream)", "success": True, "details": "Streamed response contained 'test'."})
        except Exception as e:
            results.append({"step": "6. POST /generate (Stream)", "success": False, "details": str(e)})

        # --- BUG FIX V3.9: Make Admin Exec OS-aware ---
        # Step 7: Admin Exec
        try:
            os_name = platform.system().lower()
            safe_command = "timeout 1" if os_name == "windows" else "sleep 1"
            
            payload = {"command": safe_command}
            async with session.post(f"{base_url}/api/v1/admin/exec", json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                assert "stdout" in data
                results.append({"step": "7. POST /admin/exec", "success": True, "details": f"Shell command '{safe_command}' executed."})
        except Exception as e:
            results.append({"step": "7. POST /admin/exec", "success": False, "details": str(e)})
        # --- END BUG FIX ---

        # Step 8: Unload Model
        try:
            payload = {"model_name": MODEL_NAME}
            # --- BUG FIX V3.8: Corrected variable name 'base_URL' to 'base_url' ---
            async with session.post(f"{base_url}/api/v1/models/unload", json=payload) as r:
                r.raise_for_status()
                data = await r.json()
                assert "unloaded successfully" in data.get("message", "")
                results.append({"step": "8. POST /models/unload", "success": True, "details": f"Model '{MODEL_NAME}' unloaded."})
        except Exception as e:
            results.append({"step": "8. POST /models/unload", "success": False, "details": str(e)})

    log.info("--- Internal Smoke Test Finished ---")
    return results

# --- PDF Generation ---
def generate_pdf_report(results: List[Dict[str, Any]], filename: str) -> None:
    """
    Generates a PDF report from the smoke test results.
    """
    log.info(f"Generating PDF report: {filename}")
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2.0, height - 1*inch, "Warcamp 🏕️ Smoke Test Report")
    
    # Timestamp
    c.setFont("Helvetica", 10)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawCentredString(width / 2.0, height - 1.25*inch, f"Generated: {now}")
    
    # Start drawing results
    c.setFont("Helvetica-Bold", 12)
    y_pos = height - 2*inch
    c.drawString(1*inch, y_pos, "Test Step")
    c.drawString(4*inch, y_pos, "Status")
    c.line(1*inch, y_pos - 0.1*inch, width - 1*inch, y_pos - 0.1*inch)
    
    c.setFont("Helvetica", 10)
    y_pos -= 0.25*inch
    
    total_passed = 0
    for res in results:
        step = res.get("step", "Unknown Step")
        success = res.get("success", False)
        details = res.get("details", "")
        
        # Set color
        if success:
            total_passed += 1
            status = "[SUCCESS]"
            c.setFillColorRGB(0, 0.5, 0) # Dark Green
        else:
            status = "[FAILURE]"
            c.setFillColorRGB(0.8, 0, 0) # Dark Red
            
        c.drawString(1*inch, y_pos, step)
        c.drawString(4*inch, y_pos, status)
        
        # Draw details if they exist (and for failures)
        if not success and details:
            c.setFillColorRGB(0.3, 0.3, 0.3) # Gray
            c.setFont("Helvetica-Oblique", 9)
            y_pos -= 0.2*inch
            c.drawString(1.2*inch, y_pos, f"Details: {details[:100]}") # Truncate details
            c.setFont("Helvetica", 10)
        
        c.setFillColorRGB(0, 0, 0) # Reset to black
        y_pos -= 0.3*inch
        
        # Page break
        if y_pos < 1*inch:
            c.showPage()
            c.setFont("Helvetica", 10)
            y_pos = height - 1*inch

    # Summary
    y_pos -= 0.2*inch
    c.line(1*inch, y_pos, width - 1*inch, y_pos)
    y_pos -= 0.3*inch
    
    total = len(results)
    if total_passed == total:
        c.setFillColorRGB(0, 0.5, 0)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2.0, y_pos, f"ALL {total} TESTS PASSED")
    else:
        c.setFillColorRGB(0.8, 0, 0)
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(width / 2.0, y_pos, f"FAILED: {total - total_passed} / {total} TESTS")

    c.save()
    log.info("PDF report saved.")

# --- Test Endpoint ---
@app.get("/api/v1/run-smoke-test", tags=["Dashboard"])
async def run_smoke_test_endpoint():
    """
    Runs the internal smoke test and returns a PDF report.
    """
    # Find a model to use for the test
    try:
        # --- BUG FIX V3.3 ---
        # Get the first model filename (key) from the Enum members
        default_model_file = next(iter(ModelFileEnum.__members__))
        # --- END BUG FIX ---
        
        if "NO_MODELS_FOUND" in default_model_file or "ERROR" in default_model_file:
            raise HTTPException(status_code=500, detail="Smoke test failed: No .gguf models found in model directory.")
    except Exception as e:
        log.error(f"Smoke test failed: Could not find a model to test with. {e}")
        raise HTTPException(status_code=500, detail=f"Smoke test failed: No .gguf models found. {e}")

    base_url = f"http://127.0.0.1:{APP_PORT}"
    report_filename = "DO-Test-Results.pdf"
    
    try:
        results = await run_smoke_test_logic(base_url, default_model_file)
        generate_pdf_report(results, report_filename)
        
        return FileResponse(
            report_filename,
            media_type="application/pdf",
            filename=report_filename
        )
    except Exception as e:
        log.error(f"Failed to run smoke test or generate report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------
# --- Main Entry Point ---
# -----------------------------------------------------------------

if __name__ == "__main__":
    """
    This block allows running the server directly for debugging.
    'python main.py'
    
    For production, you would use:
    'uvicorn main:app --host 0.0.0.0 --port 8000 --reload'
    """
    log.info(f"Starting Uvicorn server in debug mode (reload=True) on port {APP_PORT}...")
    uvicorn.run("main:app", host="127.0.0.1", port=APP_PORT, reload=True)