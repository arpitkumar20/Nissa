import os
import sys
import asyncio
import warnings
from dotenv import load_dotenv
from src.nisaa.server import run_server

load_dotenv()
warnings.filterwarnings("ignore")

# Global set to track running background tasks
background_tasks_running = set()


async def wait_for_background_tasks(timeout: float = 30.0):
    """
    Wait for all background tasks to complete before shutdown
    
    Args:
        timeout: Maximum time to wait in seconds
    """
    if not background_tasks_running:
        return
    
    print(f"\n⏳ Waiting for {len(background_tasks_running)} background task(s) to complete...")
    
    try:
        # Wait for all tasks with timeout
        await asyncio.wait_for(
            asyncio.gather(*background_tasks_running, return_exceptions=True),
            timeout=timeout
        )
        print("✓ All background tasks completed")
    except asyncio.TimeoutError:
        print(f"⚠️  Timeout waiting for tasks after {timeout}s. Forcing shutdown...")
        # Cancel remaining tasks
        for task in background_tasks_running:
            if not task.done():
                task.cancel()
    except Exception as e:
        print(f"⚠️  Error waiting for background tasks: {e}")


def main():
    """
    Main entry point for starting the Nisaa API server
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "4011"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    try:
        required_vars = [
            # OpenAI Configuration
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "EMBEDDING_MODEL",
            "EMBEDDING_BATCH_SIZE",
            "MIN_BATCH_DELAY",
            "MAX_WORKERS",
            "SIMILARITY_THRESHOLD",
            # Pinecone Configuration
            "PINECONE_API_KEY",
            "PINECONE_ENV",
            "PINECONE_INDEX",
            "TOP_K",
            # WATI Configuration
            "WATI_API_KEY",
            "WATI_BASE_URL",
            "WATI_TENANT_ID",
            "WATI_CHANNEL_NUMBER",
            # Database Configuration
            "DATABASE_HOST",
            "DATABASE_PORT",
            "DATABASE_USER",
            "DATABASE_PASS",
            "DB_NAME",
            # S3 Configuration
            "BUCKET_NAME",
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            # Local WATI Template Configs
            "BOOKING_TEMPLATE",
            "BOOKING_BROADCAST_NAME",
            "CANCEL_TEMPLATE",
            "CANCEL_BROADCAST_NAME",
        ]
        
        # Check for missing environment variables
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        # Print warnings for missing variables
        if missing_vars:
            print("WARNING: Missing environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            print()
        
        # Start the server
        print("✓ Starting server...\n")
        run_server(host=host, port=port, debug=debug)
        
    except ImportError as e:
        print("ERROR: Failed to import required modules")
        print(f"Details: {e}")
        sys.exit(1)
    except ValueError as e:
        print("ERROR: Invalid configuration")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print("ERROR: Failed to start server")
        print(f"Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()