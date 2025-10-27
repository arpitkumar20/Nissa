from tqdm import tqdm
from typing import List, Optional
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

from nisaa.helpers.logger import logger
from nisaa.services.custom_loader import JSONStringLoader

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

    def load_all_documents(self) -> List[Document]:
        """Load all documents from directory with progress tracking"""
        logger.info(f"Loading documents for company: {self.company_namespace}")
        logger.info(f"Directory: {self.directory_path}")

        try:
            file_types = [
                ".pdf",
                ".txt",
                ".xml",
                ".csv",
                ".docx",
                ".xlsx",
                ".xls",
                ".json",
            ]

            loaders = [
                self.create_directory_loader(".pdf"),
                self.create_text_loader(),
                self.create_directory_loader(".xml"),
                self.create_directory_loader(".csv"),
                self.create_directory_loader(".docx"),
                self.create_directory_loader(".xlsx"),
                self.create_directory_loader(".xls"),
                self.create_json_loader(),
            ]

            documents = []

            with tqdm(
                total=len(loaders),
                desc="File Types",
                unit="type",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            ) as pbar:
                for i, loader in enumerate(loaders):
                    file_type = file_types[i]
                    pbar.set_postfix_str(f"Loading {file_type}")
                    try:
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        logger.debug(f"No {file_type} files found or error: {e}")
                    pbar.update(1)

            for doc in documents:
                doc.metadata["company_namespace"] = self.company_namespace
                doc.metadata["source_type"] = "file"

            logger.info(f"Successfully loaded {len(documents)} documents from files")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise