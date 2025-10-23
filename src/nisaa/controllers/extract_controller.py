import uuid
from fastapi import UploadFile
from typing import List
from ..helpers.tika_client import TikaClient
from ..models.extract_model import ExtractContentResponse

class ExtractController:
    @staticmethod
    def extract_files(files: List[UploadFile], company_name: str) -> List[ExtractContentResponse]:
        """
        Process uploaded files and extract text using Tika.
        """
        results = []
        for f in files:
            extracted = TikaClient.extract_text(f)
            results.append(ExtractContentResponse(
                filename=f.filename,
                raw_text=extracted.get("extracted_text", ""),
                company_name=company_name
            ))
        return results
