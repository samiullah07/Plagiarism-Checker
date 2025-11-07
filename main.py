from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from detectors.pipeline import analyze_document, analyze_pdf

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    methods: Optional[List[str]] = None
    sensitivity: Optional[int] = 80

@app.post('/analyze/text')
def analyze_text(request: TextRequest):
    result = analyze_document(request.text, request.methods, request.sensitivity)
    analysis_id = str(uuid4())
    # Persist result as needed...
    return {"id": analysis_id, "result": result}

@app.post('/analyze/pdf')
async def analyze_pdf_endpoint(file: UploadFile = File(...), methods: Optional[List[str]] = None, sensitivity: Optional[int] = 80):
    content = await file.read()
    result = analyze_pdf(content, methods, sensitivity)
    analysis_id = str(uuid4())
    # Persist result as needed...
    return {"id": analysis_id, "result": result}

@app.get('/results/{id}')
def get_results(id: str):
    # Load results from store...
    return {"id": id, "result": "Not implemented"}

@app.post('/batch')
def batch_process(requests: List[TextRequest]):
    results = [analyze_document(r.text, r.methods, r.sensitivity) for r in requests]
    return {"results": results}
