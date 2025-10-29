import os
import sys
from dotenv import load_dotenv
# Import server module
from src.nisaa.api.rest_server import run_server


def main():
    """
    Main entry point for starting the Nisaa API server
    """
    try:
        # Load environment variables
        load_dotenv()
        print("✅ Environment variables loaded")

        # Get configuration from environment
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "4011"))
        debug = os.getenv("DEBUG", "false").lower() == "true"

        # Display startup information
        print("\n" + "=" * 80)
        print("🚀 NISAA API SERVER WITH AGENTIC RAG")
        print("=" * 80)
        print(f"📍 Host: {host}")
        print(f"🔌 Port: {port}")
        print(f"🐛 Debug Mode: {debug}")
        print(f"🌐 Base URL: http://{host}:{port}")
        print("-" * 80)
        print(f"💚 Health Check: http://{host}:{port}/health/")
        print(f"📊 Detailed Health: http://{host}:{port}/health/detailed")
        print(f"📚 API Docs: http://{host}:{port}/docs")
        print(f"📖 ReDoc: http://{host}:{port}/redoc")
        print("-" * 80)
        print("🔗 Available Endpoints:")
        print(f"   • WhatsApp Webhook: http://{host}:{port}/wati_webhook")
        print(f"   • Chatbot: http://{host}:{port}/chatbot/")
        print(f"   • Zoho: http://{host}:{port}/zoho/")
        print(f"   • Extraction: http://{host}:{port}/extract/")
        print(f"   • Ingestion: http://{host}:{port}/ingest/")
        print("=" * 80 + "\n")

        # Verify critical environment variables
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
            print("⚠️  WARNING: Missing environment variables:")
            for var in missing_vars:
                print(f"   • {var}")
            print("\n💡 Some features may not work properly.")
            print("   Please check your .env file.\n")

        # Start the server
        print("🎯 Starting server...\n")
        run_server(host=host, port=port, debug=debug)

    except ImportError as e:
        print("\n" + "=" * 80)
        print("❌ ERROR: Failed to import required modules")
        print("=" * 80)
        print(f"Details: {e}")
        print("\n💡 Solution:")
        print("   1. Make sure you're in the correct directory")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Or install in development mode: pip install -e .")
        print("=" * 80 + "\n")
        sys.exit(1)

    except ValueError as e:
        print("\n" + "=" * 80)
        print("❌ ERROR: Invalid configuration")
        print("=" * 80)
        print(f"Details: {e}")
        print("\n💡 Solution:")
        print(
            "   Check your .env file for invalid values (e.g., PORT must be a number)"
        )
        print("=" * 80 + "\n")
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR: Failed to start server")
        print("=" * 80)
        print(f"Details: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if the port is already in use")
        print("   2. Verify database connection settings")
        print("   3. Ensure all API keys are valid")
        print("   4. Check logs for more details")
        print("=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
 