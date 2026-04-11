import os
from openai import OpenAI
from pinecone import Pinecone

_openai_client = None
_pinecone_index = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ["PINECONE_INDEX_NAME"]
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


def embed_text(text: str) -> list[float]:
    client = get_openai_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding
