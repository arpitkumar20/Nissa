from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List
from fastapi.responses import JSONResponse
from ..controllers.extract_controller import ExtractController

router = APIRouter(
    prefix="/extract",
    tags=["extract"]
)

@router.post("/content/")
async def extract_content(
    files: List[UploadFile] = File(...),
    company_name: str = Form(...)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if not company_name:
        raise HTTPException(status_code=400, detail="'company_name' is required")

    results = ExtractController.extract_files(files, company_name)
    return JSONResponse(content=[r.dict() for r in results])
