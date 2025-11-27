"""
PRODUCTION-READY: SQL Database Ingestion with Stable Row Ordering
Ensures consistent row order across multiple runs
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import make_url
from tqdm import tqdm
from src.nisaa.config.logger import logger


def get_database_tables(db_uri: str) -> List[Dict[str, Any]]:
    """
    Get all tables with metadata from a database
    
    Returns tables sorted alphabetically for consistency
    """
    try:
        engine = create_engine(db_uri)
        inspector = inspect(engine)

        url = make_url(db_uri)
        dialect = url.get_backend_name()
        db_name = url.database

        if dialect == "mysql":
            schemas = [db_name]
        else:
            schemas = [
                s
                for s in inspector.get_schema_names()
                if s not in ("pg_catalog", "information_schema")
            ]

        tables_info = []

        with engine.connect() as conn:
            for schema in sorted(schemas):  # Sort schemas for consistency
                table_names = sorted(inspector.get_table_names(schema=schema))  # Sort tables

                for table in table_names:
                    try:
                        full_table_name = (
                            f"{schema}.{table}"
                            if dialect != "mysql" and schema != "public"
                            else table
                        )

                        result = conn.execute(
                            text(f"SELECT COUNT(*) FROM {full_table_name}")
                        )
                        row_count = result.scalar()

                        tables_info.append(
                            {
                                "table_name": table,
                                "schema": schema,
                                "row_count": row_count or 0,
                                "full_name": full_table_name,
                            }
                        )

                    except Exception as e:
                        logger.warning(
                            f"Could not get row count for {schema}.{table}: {e}"
                        )
                        tables_info.append(
                            {
                                "table_name": table,
                                "schema": schema,
                                "row_count": 0,
                                "full_name": f"{schema}.{table}",
                            }
                        )

        logger.info(f"Found {len(tables_info)} tables in database")
        return tables_info

    except Exception as e:
        logger.error(f"Failed to get database tables: {e}")
        raise


class SQLDatabaseIngester:
    """
    FIXED: SQL database ingestion with STABLE row ordering
    
    Key Features:
    - Always orders query results by primary key or first column
    - Ensures consistent row order across multiple runs
    - Prevents document ordering changes that break checkpoints
    """

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
        """Get valid schemas from database, sorted for consistency"""
        inspector = inspect(engine)
        if dialect == "mysql":
            return [db_name]
        
        schemas = [
            s
            for s in inspector.get_schema_names()
            if s not in ("pg_catalog", "information_schema")
        ]
        return sorted(schemas)  # Sort for consistency

    def get_all_tables(self, inspector, schema: str) -> List[str]:
        """Get all tables from schema, sorted for consistency"""
        return sorted(inspector.get_table_names(schema=schema))

    def get_order_by_clause(
        self,
        inspector,
        table_name: str,
        schema: str,
        dialect: str
    ) -> str:
        """
        CRITICAL: Determine ORDER BY clause for stable row ordering
        
        Priority:
        1. Primary key columns (if exist)
        2. First column (fallback)
        3. No ORDER BY (last resort - logs warning)
        """
        try:
            # Try to get primary key
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
            pk_columns = pk_constraint.get('constrained_columns', [])
            
            if pk_columns:
                order_by = ', '.join(pk_columns)
                logger.debug(f"Using PRIMARY KEY for ORDER BY: {order_by}")
                return order_by
            
            # Fallback: use first column
            columns = inspector.get_columns(table_name, schema=schema)
            if columns:
                first_col = columns[0]['name']
                logger.debug(f"Using first column for ORDER BY: {first_col}")
                return first_col
            
            # Last resort: no ORDER BY
            logger.warning(
                f"⚠️ No PRIMARY KEY or columns found for {table_name}. "
                f"Row order may be non-deterministic!"
            )
            return None
            
        except Exception as e:
            logger.warning(f"Failed to determine ORDER BY for {table_name}: {e}")
            return None

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

    def ingest_specific_tables(
        self, db_uri: str, table_names: List[str]
    ) -> List[Document]:
        """
        FIXED: Ingest specific tables with STABLE row ordering
        
        Key Changes:
        - Processes tables in sorted order
        - Adds ORDER BY clause to all queries
        - Ensures consistent document order across runs
        """
        try:
            metadata = self.parse_uri(db_uri)
            engine = create_engine(db_uri)
            inspector = inspect(engine)
            dialect = metadata["dialect"]

            schemas = self.get_valid_schemas(engine, dialect, metadata["database"])
            logger.info(
                f"Connected to '{metadata['database']}' - Processing {len(table_names)} specific tables"
            )

            db = SQLDatabase(engine)
            all_docs = []

            # Build table map
            table_map = {}
            for schema in schemas:
                tables = self.get_all_tables(inspector, schema)
                for table in tables:
                    table_map[table] = {
                        "schema": schema,
                        "full_name": (
                            f"{schema}.{table}"
                            if dialect != "mysql" and schema != "public"
                            else table
                        ),
                    }

            # CRITICAL: Process tables in SORTED order
            for table_name in sorted(table_names):
                if table_name not in table_map:
                    logger.warning(f"Table '{table_name}' not found in database")
                    continue

                table_info = table_map[table_name]
                schema = table_info["schema"]
                full_name = table_info["full_name"]

                # CRITICAL: Get ORDER BY clause for stable ordering
                order_by = self.get_order_by_clause(
                    inspector, table_name, schema, dialect
                )

                if order_by:
                    query = f"SELECT * FROM {full_name} ORDER BY {order_by};"
                    logger.info(f"Loading {full_name} (ordered by: {order_by})")
                else:
                    query = f"SELECT * FROM {full_name};"
                    logger.warning(
                        f"⚠️ Loading {full_name} WITHOUT ORDER BY - "
                        f"row order may be non-deterministic!"
                    )

                docs = self.load_table_data(db, query, metadata, schema, table_name)
                all_docs.extend(docs)

                logger.info(f"✓ Loaded {len(docs)} documents from {full_name}")

            self.stats["total_tables"] += len(table_names)
            self.stats["total_databases"] += 1

            return all_docs

        except Exception as e:
            logger.error(f"Failed to ingest specific tables: {e}")
            self.stats["failed_connections"] += 1
            return []

    def ingest_database(self, uri: str, pbar: tqdm = None) -> List[Document]:
        """
        FIXED: Ingest all data from a database with STABLE ordering
        """
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

            total_tables = sum(
                len(self.get_all_tables(inspector, schema)) for schema in schemas
            )

            with tqdm(
                total=total_tables,
                desc=f"{metadata['database']}",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tables",
            ) as table_pbar:
                # Process schemas in sorted order
                for schema in schemas:
                    tables = self.get_all_tables(inspector, schema)
                    self.stats["total_tables"] += len(tables)

                    # Process tables in sorted order
                    for table in tables:
                        full_table_name = (
                            f"{schema}.{table}"
                            if dialect != "mysql" and schema != "public"
                            else table
                        )
                        
                        # Get ORDER BY clause
                        order_by = self.get_order_by_clause(
                            inspector, table, schema, dialect
                        )
                        
                        if order_by:
                            query = f"SELECT * FROM {full_table_name} ORDER BY {order_by};"
                        else:
                            query = f"SELECT * FROM {full_table_name};"
                            logger.warning(
                                f"⚠️ {full_table_name} loaded without ORDER BY"
                            )
                        
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

        logger.info(f"Starting SQL database ingestion for {len(db_uris)} databases")

        all_documents = []

        with tqdm(
            total=len(db_uris),
            desc="Databases",
            unit="db",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as pbar:
            for uri in db_uris:
                docs = self.ingest_database(uri, pbar)
                all_documents.extend(docs)

        logger.info(
            f"SQL ingestion complete - {self.stats['total_documents']} documents from "
            f"{self.stats['total_databases']} databases, {self.stats['total_tables']} tables"
        )

        return all_documents