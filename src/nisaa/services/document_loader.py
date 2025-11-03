import json
import os
from tqdm import tqdm
from typing import Any, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyMuPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_community.document_loaders.xml import UnstructuredXMLLoader
from yaml import BaseLoader

from src.nisaa.helpers.logger import logger

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


class DocumentLoader:
    """Handles loading documents from various file formats"""

    def __init__(self, directory_path: str, company_namespace: Optional[str] = None):
        self.directory_path = directory_path
        self.company_namespace = company_namespace or "default"
        self.loaders_map = {
            ".pdf": PyMuPDFLoader,
            ".xml": UnstructuredXMLLoader,
            ".csv": CSVLoader,
            ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader,
        }

    def create_directory_loader(self, file_type: str) -> DirectoryLoader:
        return DirectoryLoader(
            path=self.directory_path,
            glob=f"**/*{file_type}",
            loader_cls=self.loaders_map[file_type],
            show_progress=False,
            use_multithreading=True,
        )

    def create_json_loader(self) -> DirectoryLoader:
        return DirectoryLoader(
            path=self.directory_path,
            glob="**/*.json",
            loader_cls=JSONStringLoader,
            show_progress=False,
        )

    def create_text_loader(self) -> DirectoryLoader:
        return DirectoryLoader(
            path=self.directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )

    def load_all_documents(self, exclude_json: bool = True) -> List[Document]:
        """
        Load all documents from directory with progress tracking
        
        Args:
            exclude_json: If True, skip JSON files (they'll be processed separately)
        """
        logger.info(f"📁 Loading documents for company: {self.company_namespace}")
        logger.info(f"Directory: {self.directory_path}")

        try:
            file_types = [".pdf", ".txt", ".xml", ".csv", ".docx", ".xlsx", ".xls"]
            
            loaders = [
                self.create_directory_loader(".pdf"),
                self.create_text_loader(),
                self.create_directory_loader(".xml"),
                self.create_directory_loader(".csv"),
                self.create_directory_loader(".docx"),
                self.create_directory_loader(".xlsx"),
                self.create_directory_loader(".xls"),
            ]
            
            if not exclude_json:
                file_types.append(".json")
                loaders.append(self.create_json_loader())

            documents = []

            with tqdm(
                total=len(loaders),
                desc="📁 File Types",
                unit="type",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            ) as pbar:
                for i, loader in enumerate(loaders):
                    file_type = file_types[i]
                    pbar.set_postfix_str(f"Loading {file_type}")
                    try:
                        docs = loader.load()
                        documents.extend(docs)
                        logger.info(f"Loaded {len(docs)} {file_type} documents")
                    except Exception as e:
                        logger.debug(f"No {file_type} files found or error: {e}")
                    pbar.update(1)

            for doc in documents:
                doc.metadata["company_namespace"] = self.company_namespace
                doc.metadata["source_type"] = "file"

            logger.info(f"✅ Successfully loaded {len(documents)} documents from files")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def get_json_files(self) -> List[str]:
        """Get list of JSON files in the directory"""
        json_files = []
        for root, dirs, files in os.walk(self.directory_path):
            for file in files:
                if file.lower().endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files
    
    def load_specific_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load only specific files by their paths
        
        Args:
            file_paths: List of absolute file paths to load
        
        Returns:
            List of Document objects
        """
        logger.info(f"📁 Loading {len(file_paths)} specific files")
        
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            file_ext = os.path.splitext(file_path)[1].lower()
            
            try:
                # Map extensions to loaders
                if file_ext == '.pdf':
                    loader = PyMuPDFLoader(file_path)
                elif file_ext == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                elif file_ext == '.xml':
                    loader = UnstructuredXMLLoader(file_path)
                elif file_ext == '.csv':
                    loader = CSVLoader(file_path)
                elif file_ext == '.docx':
                    loader = Docx2txtLoader(file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    loader = UnstructuredExcelLoader(file_path)
                elif file_ext == '.json':
                    loader = JSONStringLoader(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file_ext}")
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                logger.debug(f"Loaded {len(docs)} documents from {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        # Add metadata
        for doc in documents:
            doc.metadata["company_namespace"] = self.company_namespace
            doc.metadata["source_type"] = "file"
        
        logger.info(f"✅ Successfully loaded {len(documents)} documents from {len(file_paths)} files")
        return documents