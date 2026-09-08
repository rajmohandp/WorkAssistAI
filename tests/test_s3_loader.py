"""Tests for the Amazon S3 document repository boundary."""

from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from langchain_core.documents import Document

from src.s3_loader import (
    S3AccessDeniedError,
    S3BucketNotFoundError,
    S3CredentialsError,
    S3NetworkError,
    _create_s3_client,
    _loader_for,
    list_documents,
    load_documents,
)


class FakePaginator:
    def __init__(self, pages=None, error=None):
        self.pages = pages or []
        self.error = error

    def paginate(self, **kwargs):
        assert kwargs == {"Bucket": "documents"}
        if self.error:
            raise self.error
        return self.pages


class FakeClient:
    def __init__(self, pages=None, error=None):
        self.paginator = FakePaginator(pages, error)

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self.paginator

    def download_file(self, bucket, key, filename):
        assert bucket == "documents"
        Path(filename).write_text(f"Downloaded {key}", encoding="utf-8")


def client_error(code, status):
    return ClientError(
        {
            "Error": {"Code": code, "Message": "sensitive AWS response"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "ListObjectsV2",
    )


def test_lists_only_supported_documents_with_metadata():
    modified = datetime(2026, 1, 2, tzinfo=UTC)
    pages = [
        {
            "Contents": [
                {"Key": "reports/Annual.PDF", "Size": 120, "LastModified": modified, "ETag": '"abc"'},
                {"Key": "notes.txt", "Size": 8, "LastModified": modified, "ETag": '"def"'},
                {"Key": "draft.docx", "Size": 99, "LastModified": modified, "ETag": '"ghi"'},
                {"Key": "image.png", "Size": 15, "LastModified": modified},
                {"Key": "folder/", "Size": 0, "LastModified": modified},
            ]
        }
    ]

    result = list_documents("documents", s3_client=FakeClient(pages))

    assert [item["file_name"] for item in result] == [
        "Annual.PDF",
        "notes.txt",
        "draft.docx",
    ]
    assert result[0] == {
        "file_name": "Annual.PDF",
        "s3_key": "reports/Annual.PDF",
        "file_type": "pdf",
        "file_size": 120,
        "last_modified": modified,
        "etag": "abc",
    }


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (NoCredentialsError(), S3CredentialsError),
        (client_error("InvalidAccessKeyId", 403), S3CredentialsError),
        (client_error("NoSuchBucket", 404), S3BucketNotFoundError),
        (client_error("AccessDenied", 403), S3AccessDeniedError),
        (
            EndpointConnectionError(endpoint_url="https://s3.example.invalid"),
            S3NetworkError,
        ),
    ],
)
def test_maps_aws_errors_to_safe_repository_errors(error, expected):
    with pytest.raises(expected) as caught:
        list_documents("documents", s3_client=FakeClient(error=error))

    assert "sensitive AWS response" not in str(caught.value)


def test_s3_client_uses_region_and_optional_profile_without_explicit_keys(monkeypatch):
    captured = {}
    expected_client = object()

    class FakeSession:
        def __init__(self, **kwargs):
            captured["session"] = kwargs

        def client(self, service_name):
            captured["service"] = service_name
            return expected_client

    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("AWS_PROFILE", "workassist-local")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-be-forwarded")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-be-forwarded")
    monkeypatch.setattr("src.s3_loader.boto3.Session", FakeSession)

    assert _create_s3_client() is expected_client
    assert captured == {
        "session": {
            "region_name": "us-west-2",
            "profile_name": "workassist-local",
        },
        "service": "s3",
    }


@pytest.mark.parametrize("with_keys", [True, False])
def test_s3_client_loads_dotenv_credentials_or_preserves_default_chain(monkeypatch, tmp_path, with_keys):
    from src.config import EnvironmentSettings

    for name in ("AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AWS_ACCESS_KEY_ID=test-key\nAWS_SECRET_ACCESS_KEY=test-secret\nAWS_SESSION_TOKEN=test-token\n"
        if with_keys else "",
        encoding="utf-8",
    )
    settings = EnvironmentSettings(_env_file=env_file)
    captured = {}

    class FakeSession:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def client(self, service_name):
            return service_name

    monkeypatch.setattr("src.s3_loader.get_environment_settings", lambda: settings)
    monkeypatch.setattr("src.s3_loader.boto3.Session", FakeSession)
    assert _create_s3_client() == "s3"
    if with_keys:
        assert captured["aws_access_key_id"] == "test-key"
        assert captured["aws_secret_access_key"] == "test-secret"
        assert captured["aws_session_token"] == "test-token"
        assert "test-secret" not in repr(settings)
    else:
        assert not any(name.startswith("aws_") for name in captured)


def test_missing_object_is_reported_safely_and_ingestion_continues():
    class MissingObjectClient(FakeClient):
        def download_file(self, bucket, key, filename):
            raise client_error("NoSuchKey", 404)

    metadata = [
        {
            "file_name": "missing.pdf",
            "s3_key": "private/missing.pdf",
            "file_type": "pdf",
            "file_size": 42,
            "last_modified": datetime(2026, 1, 2, tzinfo=UTC),
            "etag": "gone",
        }
    ]

    result = load_documents(
        "documents", document_metadata=metadata, s3_client=MissingObjectClient()
    )

    assert result.loaded_files == 0
    assert result.documents == []
    assert result.failed_documents == [
        {
            "filename": "missing.pdf",
            "s3_key": "private/missing.pdf",
            "error": "The S3 object no longer exists.",
        }
    ]


def test_loads_documents_and_normalizes_metadata(monkeypatch):
    downloaded_paths = []

    class FakeLoader:
        def __init__(self, path):
            downloaded_paths.append(path)

        def load(self):
            return [
                Document(
                    page_content="Extracted content",
                    metadata={"source": str(downloaded_paths[-1]), "page": 0},
                )
            ]

    monkeypatch.setattr(
        "src.s3_loader._loader_for", lambda _file_type, path: FakeLoader(path)
    )
    metadata = [
        {
            "file_name": "report.pdf",
            "s3_key": "folder/report.pdf",
            "file_type": "pdf",
            "file_size": 42,
            "last_modified": datetime(2026, 1, 2, tzinfo=UTC),
            "etag": "abc123",
        }
    ]

    result = load_documents(
        "documents", document_metadata=metadata, s3_client=FakeClient()
    )

    assert result.loaded_files == 1
    assert result.failed_documents == []
    assert result.documents[0].page_content == "Extracted content"
    normalized = result.documents[0].metadata
    assert normalized["source"] == "s3://documents/folder/report.pdf"
    assert normalized["s3_key"] == "folder/report.pdf"
    assert normalized["filename"] == "report.pdf"
    assert normalized["file_type"] == "pdf"
    assert normalized["file_size"] == 42
    assert normalized["etag"] == "abc123"
    assert normalized["last_modified"] == "2026-01-02T00:00:00+00:00"
    assert normalized["document_id"]
    assert normalized["document_version"]
    assert normalized["page"] == 0
    assert normalized["page_number"] == 1
    assert {
        "source": "s3://documents/folder/report.pdf",
        "s3_key": "folder/report.pdf",
    }.items() <= normalized.items()
    assert not Path(downloaded_paths[0]).exists()


def test_continues_after_an_unreadable_document(monkeypatch):
    class BrokenLoader:
        def load(self):
            raise ValueError("corrupt content details")

    monkeypatch.setattr(
        "src.s3_loader._loader_for", lambda _file_type, _path: BrokenLoader()
    )
    metadata = [
        {
            "file_name": "broken.docx",
            "s3_key": "broken.docx",
            "file_type": "docx",
            "file_size": 10,
            "last_modified": datetime(2026, 1, 2, tzinfo=UTC),
            "etag": "broken",
        }
    ]

    result = load_documents(
        "documents", document_metadata=metadata, s3_client=FakeClient()
    )

    assert result.loaded_files == 0
    assert result.documents == []
    assert result.failed_documents == [
        {
            "filename": "broken.docx",
            "s3_key": "broken.docx",
            "error": "The document could not be downloaded or read.",
        }
    ]


def test_txt_and_docx_loaders_extract_content(tmp_path):
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("DocuVerse text content", encoding="utf-8")

    docx_path = tmp_path / "sample.docx"
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body><w:p><w:r><w:t>DocuVerse DOCX content</w:t></w:r></w:p></w:body>
    </w:document>"""
    with ZipFile(docx_path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    txt_documents = _loader_for("txt", txt_path).load()
    docx_documents = _loader_for("docx", docx_path).load()

    assert "DocuVerse text content" in txt_documents[0].page_content
    assert "DocuVerse DOCX content" in docx_documents[0].page_content
