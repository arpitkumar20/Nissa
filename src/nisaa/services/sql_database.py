"""
SQL Database ingestion - EXISTING LOGIC INTACT
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.url import make_url
from tqdm import tqdm
from src.nisaa.helpers.logger import logger

class SQLDatabaseIngester:
    """Handles SQL database ingestion with progress tracking"""

    def __init__(self, company_namespace: str):
        self.company_namespace = company_namespace
        self.stats = {
            "total_databases": 0,
            "total_tables": 0,
            "total_documents": 0,
            "failed_connections": 0,
        }

    def parse_uri(self, uri: str) -> Dict[str, Any]:
        """Parse database URI into components"""
        url = make_url(uri)
        return {
            "dialect": url.get_backend_name(),
            "driver": url.get_driver_name(),
            "host": url.host,
            "port": url.port,
            "username": url.username,
            "database": url.database,
        }

    def get_valid_schemas(self, engine, dialect: str, db_name: str) -> List[str]:
        """Get valid schemas from database"""
        inspector = inspect(engine)
        if dialect == "mysql":
            return [db_name]
        return [
            s
            for s in inspector.get_schema_names()
            if s not in ("pg_catalog", "information_schema")
        ]

    def get_all_tables(self, inspector, schema: str) -> List[str]:
        """Get all tables from schema"""
        return inspector.get_table_names(schema=schema)

    def load_table_data(
        self,
        db: SQLDatabase,
        query: str,
        metadata: Dict[str, Any],
        schema: str,
        table: str,
    ) -> List[Document]:
        """Load data from a single table"""
        try:
            loader = SQLDatabaseLoader(db=db, query=query)
            docs = loader.load()

            for doc in docs:
                doc.metadata.update(metadata)
                doc.metadata["schema"] = schema
                doc.metadata["table_name"] = table
                doc.metadata["company_namespace"] = self.company_namespace
                doc.metadata["source_type"] = "database"

            self.stats["total_documents"] += len(docs)
            return docs

        except Exception as e:
            logger.warning(f"Error loading table '{schema}.{table}': {e}")
            return []

    def ingest_database(self, uri: str, pbar: tqdm = None) -> List[Document]:
        """Ingest all data from a database"""
        try:
            metadata = self.parse_uri(uri)
            engine = create_engine(uri)
            inspector = inspect(engine)
            dialect = metadata["dialect"]

            schemas = self.get_valid_schemas(engine, dialect, metadata["database"])
            logger.info(
                f"Connected to '{metadata['database']}' ({dialect}) - Schemas: {schemas}"
            )

            db = SQLDatabase(engine)
            all_docs = []

            # Count total tables for progress
            total_tables = sum(
                len(self.get_all_tables(inspector, schema)) for schema in schemas
            )

            with tqdm(
                total=total_tables,
                desc=f"  📊 {metadata['database']}",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tables",
            ) as table_pbar:
                for schema in schemas:
                    tables = self.get_all_tables(inspector, schema)
                    self.stats["total_tables"] += len(tables)

                    for table in tables:
                        full_table_name = (
                            f"{schema}.{table}"
                            if dialect != "mysql" and schema != "public"
                            else table
                        )
                        query = f"SELECT * FROM {full_table_name};"
                        docs = self.load_table_data(db, query, metadata, schema, table)
                        all_docs.extend(docs)
                        table_pbar.update(1)

            self.stats["total_databases"] += 1
            if pbar:
                pbar.update(1)

            return all_docs

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.stats["failed_connections"] += 1
            if pbar:
                pbar.update(1)
            return []

    def ingest_multiple_databases(self, db_uris: List[str]) -> List[Document]:
        """Ingest data from multiple databases with progress tracking"""
        if not db_uris:
            return []
            
        logger.info(f"🗄️ Starting SQL database ingestion for {len(db_uris)} databases")

        all_documents = []

        with tqdm(
            total=len(db_uris),
            desc="🗄️ Databases",
            unit="db",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as pbar:
            for uri in db_uris:
                docs = self.ingest_database(uri, pbar)
                all_documents.extend(docs)

        logger.info(
            f"✅ SQL ingestion complete - {self.stats['total_documents']} documents from "
            f"{self.stats['total_databases']} databases, {self.stats['total_tables']} tables"
        )

        return all_documents