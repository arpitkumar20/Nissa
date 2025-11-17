"""
Checkpoint Manager for Data Ingestion Recovery
Tracks progress during embedding and upserting to enable crash recovery
"""
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from psycopg2 import pool

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint/recovery state for ingestion pipeline
    Stores progress in both database and local JSON for redundancy
    """

    def __init__(self, db_pool: pool.SimpleConnectionPool, checkpoint_dir: str = "checkpoints"):
        self.pool = db_pool
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_checkpoint_table()

    def _ensure_checkpoint_table(self):
        """Create checkpoint tracking table if not exists"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ingestion_checkpoints (
                        id SERIAL PRIMARY KEY,
                        job_id VARCHAR(50) NOT NULL,
                        company_name VARCHAR(255) NOT NULL,
                        phase VARCHAR(50) NOT NULL,
                        checkpoint_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(job_id, phase)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_checkpoint_job_phase 
                    ON ingestion_checkpoints(job_id, phase);
                    
                    CREATE INDEX IF NOT EXISTS idx_checkpoint_company 
                    ON ingestion_checkpoints(company_name);
                """)
            conn.commit()
            logger.info("✓ Checkpoint table initialized")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create checkpoint table: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def save_checkpoint(
        self,
        job_id: str,
        company_name: str,
        phase: str,
        checkpoint_data: Dict[str, Any]
    ):
        """Save checkpoint with file fallback if DB fails"""
        checkpoint_data.update({
            "job_id": job_id,
            "company_name": company_name,
            "phase": phase,
            "timestamp": datetime.now().isoformat()
        })

        checkpoint_file = self.checkpoint_dir / f"{job_id}_{phase}.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"✓ Checkpoint saved to file: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to file: {e}")

        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ingestion_checkpoints 
                    (job_id, company_name, phase, checkpoint_data, updated_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (job_id, phase) 
                    DO UPDATE SET
                        checkpoint_data = EXCLUDED.checkpoint_data,
                        updated_at = NOW()
                """, (job_id, company_name, phase, json.dumps(checkpoint_data)))
            conn.commit()
            logger.info(f"✓ Checkpoint saved: {job_id} - {phase}")
        except Exception as e:
            logger.warning(f"DB checkpoint save failed (file saved): {e}")
            # Don't raise - file checkpoint is sufficient
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass

    def load_checkpoint(
        self,
        job_id: str,
        phase: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint from database, fallback to local file
        
        Returns:
            Checkpoint data dict or None if not found
        """
        conn = None
        try:
            # Try database first
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT checkpoint_data
                    FROM ingestion_checkpoints
                    WHERE job_id = %s AND phase = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (job_id, phase))
                
                row = cur.fetchone()
                if row:
                    logger.info(f"✓ Checkpoint loaded from DB: {job_id} - {phase}")
                    return row[0]
            
            # Fallback to local file
            checkpoint_file = self.checkpoint_dir / f"{job_id}_{phase}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"✓ Checkpoint loaded from file: {job_id} - {phase}")
                return data
            
            logger.info(f"No checkpoint found for {job_id} - {phase}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
        finally:
            if conn:
                self.pool.putconn(conn)

    def clear_checkpoint(self, job_id: str, phase: str = None):
        """Clear checkpoint after successful completion"""
        
        try:
            if phase:
                checkpoint_file = self.checkpoint_dir / f"{job_id}_{phase}.json"
                data_file = self.checkpoint_dir / f"{job_id}_{phase}_data.json"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                if data_file.exists():
                    data_file.unlink()
            else:
                for file in self.checkpoint_dir.glob(f"{job_id}_*.json"):
                    file.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint files: {e}")
        
        conn = None
        try:
            if self.pool.closed:
                logger.warning("DB pool closed, skipping database checkpoint cleanup")
                return
            
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                if phase:
                    cur.execute("DELETE FROM ingestion_checkpoints WHERE job_id = %s AND phase = %s", (job_id, phase))
                else:
                    cur.execute("DELETE FROM ingestion_checkpoints WHERE job_id = %s", (job_id,))
            
            conn.commit()
            logger.info(f"✓ Checkpoint cleared: {job_id} - {phase or 'all phases'}")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to clear checkpoint from DB: {e}")
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass

    def get_all_checkpoints(self, job_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all checkpoints for a job"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT phase, checkpoint_data, updated_at
                    FROM ingestion_checkpoints
                    WHERE job_id = %s
                    ORDER BY updated_at DESC
                """, (job_id,))
                
                checkpoints = {}
                for row in cur.fetchall():
                    phase, data, updated_at = row
                    data['last_updated'] = updated_at.isoformat() if updated_at else None
                    checkpoints[phase] = data
                
                return checkpoints
                
        except Exception as e:
            logger.error(f"Failed to get checkpoints: {e}")
            return {}
        finally:
            if conn:
                self.pool.putconn(conn)


class ProcessedItemTracker:
    """
    Tracks individual items (documents, chunks, vectors) that have been processed
    Enables fine-grained recovery
    """
    
    def __init__(self, db_pool: pool.SimpleConnectionPool):
        self.pool = db_pool
        self._ensure_tracking_table()
    
    def _ensure_tracking_table(self):
        """Create item tracking table"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS processed_items (
                        id SERIAL PRIMARY KEY,
                        job_id VARCHAR(50) NOT NULL,
                        item_type VARCHAR(50) NOT NULL,
                        item_hash VARCHAR(64) NOT NULL,
                        batch_index INTEGER,
                        phase VARCHAR(50) NOT NULL,
                        processed_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(job_id, item_type, item_hash, phase)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_processed_items_job 
                    ON processed_items(job_id, phase);
                    
                    CREATE INDEX IF NOT EXISTS idx_processed_items_hash 
                    ON processed_items(item_hash);
                """)
            conn.commit()
            logger.info("✓ Item tracking table initialized")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to create tracking table: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def mark_items_processed(
        self,
        job_id: str,
        item_type: str,
        item_hashes: List[str],
        phase: str,
        batch_index: int = None
    ):
        """Mark multiple items as processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                values = [
                    (job_id, item_type, hash_val, batch_index, phase)
                    for hash_val in item_hashes
                ]
                
                cur.executemany("""
                    INSERT INTO processed_items 
                    (job_id, item_type, item_hash, batch_index, phase)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (job_id, item_type, item_hash, phase) 
                    DO NOTHING
                """, values)
            
            conn.commit()
            logger.debug(f"Marked {len(item_hashes)} items as processed")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to mark items processed: {e}")
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def is_item_processed(
        self,
        job_id: str,
        item_type: str,
        item_hash: str,
        phase: str
    ) -> bool:
        """Check if item has been processed"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM processed_items
                        WHERE job_id = %s 
                        AND item_type = %s 
                        AND item_hash = %s 
                        AND phase = %s
                    )
                """, (job_id, item_type, item_hash, phase))
                
                return cur.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to check item: {e}")
            return False
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def get_processed_count(
        self,
        job_id: str,
        item_type: str,
        phase: str
    ) -> int:
        """Get count of processed items"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM processed_items
                    WHERE job_id = %s 
                    AND item_type = %s 
                    AND phase = %s
                """, (job_id, item_type, phase))
                
                return cur.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def clear_items(self, job_id: str, phase: str = None):
        """Clear processed items after completion"""
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                if phase:
                    cur.execute("""
                        DELETE FROM processed_items
                        WHERE job_id = %s AND phase = %s
                    """, (job_id, phase))
                else:
                    cur.execute("""
                        DELETE FROM processed_items
                        WHERE job_id = %s
                    """, (job_id,))
            
            conn.commit()
            logger.info(f"✓ Cleared processed items: {job_id} - {phase or 'all phases'}")
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to clear items: {e}")
        finally:
            if conn:
                self.pool.putconn(conn)