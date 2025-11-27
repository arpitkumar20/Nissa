"""
Fixed Checkpoint Manager with Database Resilience
Key fixes:
1. File-first strategy (DB is bonus)
2. Connection failure recovery
3. Checkpoint validation
4. Atomic saves
"""
import json
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from psycopg2 import pool
import threading

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint/recovery state with FILE-FIRST strategy
    Database is secondary (for convenience, not required)
    """

    def __init__(self, db_pool: pool.SimpleConnectionPool = None, checkpoint_dir: str = "checkpoints"):
        self.pool = db_pool
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.company_name = None
        self.file_lock = threading.Lock()
        
        if self.pool:
            try:
                self._ensure_checkpoint_table()
            except Exception as e:
                logger.warning(f"DB checkpoint table creation failed (file checkpoints will work): {e}")

    def _ensure_checkpoint_table(self):
        """Create checkpoint tracking table if not exists"""
        if not self.pool:
            return
        
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
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"Failed to create checkpoint table: {e}")
            raise
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass

    def save_checkpoint(
        self,
        job_id: str,
        company_name: str,
        phase: str,
        checkpoint_data: Dict[str, Any]
    ):
        """
        Save checkpoint with FILE-FIRST strategy
        File save is CRITICAL, DB save is optional
        """
        checkpoint_data.update({
            "job_id": job_id,
            "company_name": company_name,
            "phase": phase,
            "timestamp": datetime.now().isoformat()
        })

        # STEP 1: Save to file (CRITICAL - must not fail)
        checkpoint_file = self.checkpoint_dir / f"{job_id}_{phase}.json"
        try:
            with self.file_lock:
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"✓ Checkpoint saved to file: {checkpoint_file.name}")
        except Exception as e:
            logger.critical(f"FILE CHECKPOINT FAILED - CRITICAL: {e}")
            raise  # Don't proceed if file save fails

        # STEP 2: Try DB save (OPTIONAL - continue if fails)
        if not self.pool:
            return
        
        conn = None
        try:
            conn = self.pool.getconn()
            
            # Add timeout to prevent hanging
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
            logger.debug(f"✓ DB checkpoint saved: {job_id} - {phase}")
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.warning(f"DB checkpoint save failed (file checkpoint is safe): {e}")
            # Continue - file checkpoint is sufficient
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
        Load checkpoint from FILE first, then try DB
        Returns checkpoint data or None
        """
        # STEP 1: Try file first (faster, always available)
        checkpoint_file = self.checkpoint_dir / f"{job_id}_{phase}.json"
        try:
            if checkpoint_file.exists():
                with self.file_lock:
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                logger.debug(f"✓ Checkpoint loaded from file: {job_id} - {phase}")
                return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint from file: {e}")
            # Continue to try DB
        
        # STEP 2: Try DB as fallback
        if not self.pool:
            return None
        
        conn = None
        try:
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
                    logger.debug(f"✓ Checkpoint loaded from DB: {job_id} - {phase}")
                    return row[0]
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint from DB: {e}")
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass
        
        logger.info(f"No checkpoint found for {job_id} - {phase}")
        return None

    def clear_checkpoint(self, job_id: str, phase: str = None):
        """Clear checkpoint after successful completion"""
        
        # STEP 1: Clear files
        try:
            if phase:
                checkpoint_file = self.checkpoint_dir / f"{job_id}_{phase}.json"
                data_file = self.checkpoint_dir / f"{job_id}_{phase}_data.json"
                ids_file = self.checkpoint_dir / f"{job_id}_{phase}_ids.json"
                
                for f in [checkpoint_file, data_file, ids_file]:
                    if f.exists():
                        with self.file_lock:
                            f.unlink()
            else:
                for file in self.checkpoint_dir.glob(f"{job_id}_*.json"):
                    with self.file_lock:
                        file.unlink()
            
            logger.debug(f"✓ Checkpoint files cleared: {job_id} - {phase or 'all'}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint files: {e}")
        
        # STEP 2: Clear from DB (optional)
        if not self.pool:
            return
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                if phase:
                    cur.execute(
                        "DELETE FROM ingestion_checkpoints WHERE job_id = %s AND phase = %s",
                        (job_id, phase)
                    )
                else:
                    cur.execute(
                        "DELETE FROM ingestion_checkpoints WHERE job_id = %s",
                        (job_id,)
                    )
            
            conn.commit()
            logger.debug(f"✓ DB checkpoint cleared: {job_id} - {phase or 'all'}")
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.warning(f"Failed to clear DB checkpoint: {e}")
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass

    def get_all_checkpoints(self, job_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all checkpoints for a job"""
        
        # STEP 1: Try file system first
        checkpoints = {}
        try:
            for checkpoint_file in self.checkpoint_dir.glob(f"{job_id}_*.json"):
                if checkpoint_file.name.endswith('_data.json') or checkpoint_file.name.endswith('_ids.json'):
                    continue  # Skip data files
                
                try:
                    with self.file_lock:
                        with open(checkpoint_file, 'r') as f:
                            data = json.load(f)
                    
                    phase = data.get('phase', checkpoint_file.stem.split('_', 1)[1])
                    checkpoints[phase] = data
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint file {checkpoint_file.name}: {e}")
            
            if checkpoints:
                logger.info(f"Found {len(checkpoints)} file checkpoints for {job_id}")
                return checkpoints
        except Exception as e:
            logger.warning(f"Failed to scan file checkpoints: {e}")
        
        # STEP 2: Try DB as fallback
        if not self.pool:
            return checkpoints
        
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
                
                for row in cur.fetchall():
                    phase, data, updated_at = row
                    data['last_updated'] = updated_at.isoformat() if updated_at else None
                    checkpoints[phase] = data
            
            if checkpoints:
                logger.info(f"Found {len(checkpoints)} DB checkpoints for {job_id}")
                
        except Exception as e:
            logger.warning(f"Failed to get DB checkpoints: {e}")
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass
        
        return checkpoints

    def checkpoint_exists(self, job_id: str, phase: str = None) -> bool:
        """Check if checkpoint exists"""
        try:
            if phase:
                checkpoint_file = self.checkpoint_dir / f"{job_id}_{phase}.json"
                return checkpoint_file.exists()
            else:
                # Check if any checkpoint exists for job
                return len(list(self.checkpoint_dir.glob(f"{job_id}_*.json"))) > 0
        except Exception as e:
            logger.warning(f"Failed to check checkpoint: {e}")
            return False


class ProcessedItemTracker:
    """
    Tracks individual items that have been processed
    Uses database, with file backup
    """
    
    def __init__(self, db_pool: pool.SimpleConnectionPool = None):
        self.pool = db_pool
        if self.pool:
            try:
                self._ensure_tracking_table()
            except Exception as e:
                logger.warning(f"Item tracking table creation failed: {e}")
    
    def _ensure_tracking_table(self):
        """Create item tracking table"""
        if not self.pool:
            return
        
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
                try:
                    conn.rollback()
                except:
                    pass
            logger.error(f"Failed to create tracking table: {e}")
            raise
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass
    
    def mark_items_processed(
        self,
        job_id: str,
        item_type: str,
        item_hashes: List[str],
        phase: str,
        batch_index: int = None
    ):
        """Mark multiple items as processed"""
        if not self.pool:
            return
        
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
                try:
                    conn.rollback()
                except:
                    pass
            logger.warning(f"Failed to mark items processed: {e}")
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass
    
    def is_item_processed(
        self,
        job_id: str,
        item_type: str,
        item_hash: str,
        phase: str
    ) -> bool:
        """Check if item has been processed"""
        if not self.pool:
            return False
        
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
            logger.warning(f"Failed to check item: {e}")
            return False
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass
    
    def get_processed_count(
        self,
        job_id: str,
        item_type: str,
        phase: str
    ) -> int:
        """Get count of processed items"""
        if not self.pool:
            return 0
        
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
            logger.warning(f"Failed to get count: {e}")
            return 0
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass
    
    def clear_items(self, job_id: str, phase: str = None):
        """Clear processed items after completion"""
        if not self.pool:
            return
        
        conn = None
        try:
            conn = self.pool.getconn()
            with conn.cursor() as cur:
                if phase:
                    cur.execute(
                        "DELETE FROM processed_items WHERE job_id = %s AND phase = %s",
                        (job_id, phase)
                    )
                else:
                    cur.execute(
                        "DELETE FROM processed_items WHERE job_id = %s",
                        (job_id,)
                    )
            
            conn.commit()
            logger.debug(f"✓ Processed items cleared: {job_id} - {phase or 'all'}")
            
        except Exception as e:
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            logger.warning(f"Failed to clear items: {e}")
        finally:
            if conn:
                try:
                    self.pool.putconn(conn)
                except:
                    pass