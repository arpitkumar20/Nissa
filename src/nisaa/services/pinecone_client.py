import os
from pinecone import Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
pinecone_index = pinecone_client.Index(PINECONE_INDEX)