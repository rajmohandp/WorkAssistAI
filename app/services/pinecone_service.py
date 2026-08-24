"""Application service for Pinecone index status."""

from app.api.schemas import PineconeStatusResponse
from src.vector_store import get_vector_store_status


class PineconeService:
    def status(self) -> PineconeStatusResponse:
        status = get_vector_store_status()
        return PineconeStatusResponse(
            connected=True,
            index_name=status.index_name,
            indexed_documents=status.indexed_documents,
            total_vectors=status.vector_count,
        )
