from pydantic import BaseModel
from typing import List

class ExtractContentResponse(BaseModel):
    filename: str
    raw_text: str
    company_name: str