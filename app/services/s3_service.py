"""Application service for Amazon S3 repository status."""

from app.api.schemas import S3StatusResponse
from src.s3_loader import get_bucket_name, list_documents


class S3Service:
    def status(self) -> S3StatusResponse:
        bucket = get_bucket_name()
        documents = list_documents(bucket)
        return S3StatusResponse(
            connected=True,
            bucket=bucket,
            supported_documents=len(documents),
            documents=sorted(
                {document["file_name"] for document in documents},
                key=str.casefold,
            ),
        )
