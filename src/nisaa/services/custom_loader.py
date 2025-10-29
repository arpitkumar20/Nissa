import json
from typing import Any, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader


class JSONStringLoader(BaseLoader):
    """Custom loader that converts JSON to clean formatted string"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def json_to_text(self, data: Any, indent: int = 0) -> str:
        """Convert JSON to clean readable text format"""
        lines = []
        indent_str = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{indent_str}{key}:")
                    lines.append(self.json_to_text(value, indent + 1))
                elif isinstance(value, list):
                    lines.append(f"{indent_str}{key}:")
                    for item in value:
                        lines.append(self.json_to_text(item, indent + 1))
                else:
                    lines.append(f"{indent_str}{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                lines.append(self.json_to_text(item, indent))
                lines.append("")
        else:
            lines.append(f"{indent_str}{data}")

        return "\n".join(lines)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load JSON file as formatted text"""
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        json_string = self.json_to_text(data)

        yield Document(page_content=json_string, metadata={"source": self.file_path})