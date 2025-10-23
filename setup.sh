#!/bin/bash

# Setup script for Nisaa API
echo "Setting up Nisaa API..."

# Install dependencies
echo "Installing dependencies..."
pip install -e .

echo "Setup complete!"
echo ""
echo "To start the server:"
echo "  python main.py"
echo ""
echo "Or run directly:"
echo "  python -m nisaa.api.rest_server"
echo ""
echo "Health check endpoints will be available at:"
echo "  http://localhost:4011/health/"
echo "  http://localhost:4011/health/detailed"
echo ""
echo "API documentation will be available at:"
echo "  http://localhost:4011/docs"
