from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from detectors.pipeline import analyze_document, analyze_pdf

app = FastAPI(title="Plagiarism Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    methods: Optional[List[str]] = None
    sensitivity: Optional[int] = 80

@app.get("/")
def root():
    return {"status": "active", "message": "Plagiarism Detection API is running"}

@app.post('/analyze/text')
def analyze_text(request: TextRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
            
        result = analyze_document(request.text, request.methods, request.sensitivity)
        analysis_id = str(uuid4())
        return {"id": analysis_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

import traceback

@app.post('/analyze/pdf')
async def analyze_pdf_endpoint(file: UploadFile = File(...), methods: Optional[List[str]] = None, sensitivity: Optional[int] = 80):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        result = analyze_pdf(content, methods, sensitivity)
        analysis_id = str(uuid4())
        return {"id": analysis_id, "result": result}
    except Exception as e:
        print("PDF analysis error:", e)
        traceback.print_exc()  # Print full error stack to console
        raise HTTPException(status_code=500, detail=f"PDF analysis error: {str(e)}")

@app.get('/results/{id}')
def get_results(id: str):
    return {"id": id, "result": "Not implemented"}

@app.post('/batch')
def batch_process(requests: List[TextRequest]):
    results = [analyze_document(r.text, r.methods, r.sensitivity) for r in requests]
    return {"results": results}

