"""Administrative document synchronization service."""

from dataclasses import asdict

from app.api.schemas import SyncResponse
from src.vector_store import synchronize_s3_documents


class SyncService:
    """Synchronize S3 documents with Pinecone outside the UI process."""

    def synchronize(self) -> SyncResponse:
        statistics = synchronize_s3_documents()
        payload = asdict(statistics)
        payload.pop("failures", None)
        return SyncResponse(**payload)
