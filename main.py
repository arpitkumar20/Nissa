"""
Main entry point for the Nisaa application
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to start the Nisaa REST API server"""
    try:
        from nisaa.api.rest_server import run_server
        
        # Get configuration from environment variables
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "4011"))
        debug = os.getenv("DEBUG", "false").lower() == "true"
        
        print(f"Starting Nisaa API server on {host}:{port}")
        print(f"Debug mode: {debug}")
        print(f"Health check available at: http://{host}:{port}/health/")
        print(f"API documentation at: http://{host}:{port}/docs")
        
        # Start the server
        run_server(host=host, port=port, debug=debug)
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please install dependencies first: pip install -e .")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()