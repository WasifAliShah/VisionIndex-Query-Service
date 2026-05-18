"""
Unified Query Service
FastAPI for post-processing image/text queries against Qdrant.
"""
import os
import sys
import torch
import traceback
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
from dotenv import load_dotenv
from urllib.parse import urlparse

from query_image import handle_image_query
from query_text import handle_text_query

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI(title="Unified Query Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant client
client = None
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_host = os.environ.get("QDRANT_HOST")
qdrant_port = os.environ.get("QDRANT_PORT") or os.environ.get("QDRANT_GRPC_PORT")

try:
    from qdrant_client import QdrantClient
    
    if qdrant_url:
        parsed = urlparse(qdrant_url)
        if parsed.port == 6334:
            host = parsed.hostname or "localhost"
            port = parsed.port
            client = QdrantClient(host=host, port=port, prefer_grpc=True)
        else:
            client = QdrantClient(url=qdrant_url)
    elif qdrant_host and qdrant_port:
        client = QdrantClient(host=qdrant_host, port=int(qdrant_port), prefer_grpc=True)
    else:
        client = QdrantClient()
    
    client.get_collections()
    print(f"✓ Connected to Qdrant")
except Exception as e:
    print(f"✗ Failed to connect to Qdrant: {e}")

# Initialize InsightFace
face_analyzer = None
try:
    from insightface.app import FaceAnalysis
    face_analyzer = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    print("✓ InsightFace initialized")
except Exception as e:
    print(f"✗ InsightFace not available: {e}")

# Initialize CLIP
clip_model = None
clip = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    import clip as clip_lib
    clip = clip_lib
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    print(f"✓ CLIP initialized on {DEVICE}")
except Exception as e:
    print(f"✗ CLIP not available: {e}")


# ==========================================================
# Flask Mocking Layer (To keep query_*.py unchanged)
# ==========================================================
class MockFile:
    def __init__(self, file_bytes):
        self._bytes = file_bytes
    def read(self):
        return self._bytes

class MockFlaskRequest:
    def __init__(self, json_data=None, form_data=None, files_data=None):
        self.json = json_data or {}
        self.form = form_data or {}
        self.files = {k: MockFile(v) for k, v in (files_data or {}).items()}
        self.is_json = bool(json_data is not None)
    
    def get_json(self, silent=True):
        return self.json


# ==========================================================
# FastAPI Endpoints
# ==========================================================
class TextQueryRequest(BaseModel):
    text: str
    video_id: Any
    top_k: Optional[int] = 10

@app.get('/health')
def health():
    """Health check endpoint."""
    return {
        'status': 'ok',
        'qdrant': client is not None,
        'insightface': face_analyzer is not None,
        'clip': clip_model is not None
    }

@app.post('/query/text')
def query_by_text(req: TextQueryRequest):
    """Query Qdrant semantically by text using CLIP embeddings."""
    try:
        mock_req = MockFlaskRequest(json_data=req.model_dump())
        payload, status_code = handle_text_query(mock_req, client, clip_model, clip, DEVICE)
        return JSONResponse(content=payload, status_code=status_code)
    except Exception as e:
        print(f"✗ Error in text query: {e}")
        traceback.print_exc()
        return JSONResponse(content={'success': False, 'error': str(e)}, status_code=500)

@app.post('/query/image')
async def query_by_image(
    image: UploadFile = File(...),
    video_id: str = Form(...),
    top_k: int = Form(10)
):
    """Query Qdrant by face image."""
    try:
        image_bytes = await image.read()
        mock_req = MockFlaskRequest(
            form_data={'video_id': video_id, 'top_k': top_k},
            files_data={'image': image_bytes}
        )
        payload, status_code = handle_image_query(mock_req, client, face_analyzer)
        return JSONResponse(content=payload, status_code=status_code)
    except Exception as e:
        print(f"✗ Error in image query: {e}")
        traceback.print_exc()
        return JSONResponse(content={'success': False, 'error': str(e)}, status_code=500)


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('QUERY_SERVICE_PORT', os.environ.get('IMAGE_QUERY_PORT', 5001)))
    print(f"\n{'='*60}")
    print(f"🚀 Unified FastAPI Query Service starting on port {port}")
    print(f"{'='*60}\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
