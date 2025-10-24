import os
import requests
import logging as log
from typing import Dict
from fastapi import UploadFile

TIKA_URL = os.getenv('TIKA_URL')
if not TIKA_URL:
    log.error("ENV TIKA_URL missing")
    raise EnvironmentError("Missing required environment variable 'TIKA_URL'")

session = requests.Session()
TIMEOUT = 300

class TikaClient:
    @staticmethod
    def extract_text(upload_file: UploadFile) -> Dict[str, str]:
        try:
            headers = {'Accept': 'text/plain'}
            file_bytes = upload_file.file.read()
            response = session.put(
                url=TIKA_URL,
                data=file_bytes,
                headers=headers,
                timeout=TIMEOUT
            )
            response.raise_for_status()
            return {"extracted_text": response.text}
        except requests.RequestException as e:
            raise Exception(f"Error extracting text from {upload_file.filename}: {e}")