import os
import sys
import warnings
from dotenv import load_dotenv
from src.nisaa.api.rest_server import run_server

warnings.filterwarnings("ignore")

def main():
    """
    Main entry point for starting the Nisaa API server
    """
    try:
        load_dotenv()
      
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "4011"))
        debug = os.getenv("DEBUG", "false").lower() == "true"

        print(f" Host: {host}")
        print(f" Port: {port}")

        required_vars = [
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "DATABASE_HOST",
            "DATABASE_USER",
            "DATABASE_PASS",
            "DB_NAME",
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            print("WARNING: Missing environment variables:")
            for var in missing_vars:
                print(f"{var}")

        # Start the server
        print("Starting server...\n")
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
 