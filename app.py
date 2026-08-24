"""HTTP-only Streamlit frontend for DocuVerse."""

from __future__ import annotations

import base64
import html
import logging
from pathlib import Path
from typing import Any

import streamlit as st

from app.frontend_client import (
    API_BASE_URL,
    DocuVerseAPIError,
    ask_question_http,
    synchronize_repository,
)
from app.frontend_client import (
    get_repository_status as fetch_repository_status,
)
from auth import (
    is_admin,
    is_authenticated,
    login,
    logout,
)
from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("docuverse.app")
selected_document: str | None = None

st.set_page_config(page_title="DocuVerse", page_icon="📚", layout="wide")


def get_base64_image(image_path: str) -> str:
    """Encode a local image for use in Streamlit's injected CSS."""

    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


background = get_base64_image("assets/docuverse_background.png")

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(
                rgba(255, 255, 255, 0.82),
                rgba(255, 255, 255, 0.82)
            ),
            url("data:image/png;base64,{background}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E5EAF3;
    }}
    [data-testid="stSidebar"] [data-testid="stImage"] {{
        margin-top: 0;
        margin-bottom: 0.25rem;
    }}
    [data-testid="stSidebar"] [data-testid="stImage"] img {{
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}
    .docuverse-tagline {{
        color: #65708A;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 0;
        margin-bottom: 1.25rem;
    }}
    .account-card {{
        background: rgba(248, 250, 255, 0.8);
        border: 1px solid #E5EAF3;
        border-radius: 10px;
        color: #26324A;
        line-height: 1.8;
        margin: 0.75rem 0;
        padding: 0.8rem 1rem;
    }}
    h1 {{
        color: #101A45;
        font-weight: 700;
    }}
    .block-container {{max-width: 980px; padding-top: 2rem;}}
    [data-testid="stChatMessage"] {{
        border-radius: 0.8rem;
        padding: 0.35rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_login() -> None:
    """Render the authentication gate before any repository access."""

    st.title("DocuVerse")
    st.subheader("Sign in")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True,
        )
    if submitted:
        if login(st.session_state, username, password):
            st.session_state.app_mode = (
                "Admin" if is_admin(st.session_state) else "Chat"
            )
            st.rerun()
        else:
            st.error("Invalid username or password.")


if not is_authenticated(st.session_state):
    render_login()
    st.stop()


@st.cache_data(ttl=30, show_spinner=False)
def get_repository_status() -> dict[str, Any]:
    """Read S3 and Pinecone status through FastAPI only."""

    return fetch_repository_status()


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Display source documents separately beneath a grounded answer."""

    if not sources:
        return
    with st.expander("Sources Used"):
        for index, source in enumerate(sources, start=1):
            document = source.get("document", "Unknown document")
            page = source.get("page")
            label = f"{index}. {document}"
            if page is not None:
                label += f" — Page {page}"
            st.write(label)


def render_evaluation(evaluation: dict[str, Any] | None) -> None:
    """Display optional quality metrics without affecting the answer."""

    if not evaluation or evaluation.get("status") != "evaluated":
        return
    with st.expander("Answer Quality"):
        faithfulness, relevance, confidence = st.columns(3)
        faithfulness.metric(
            "Faithfulness",
            f"{evaluation['faithfulness_percentage']:.1f}%",
        )
        relevance.metric(
            "Answer relevance",
            f"{evaluation['answer_relevance_percentage']:.1f}%",
        )
        confidence.metric(
            "Overall confidence",
            f"{evaluation['overall_confidence_percentage']:.1f}%",
        )
        retrieval = evaluation.get("retrieval_confidence_percentage")
        if retrieval is not None:
            st.caption(f"Retrieval confidence: {retrieval:.1f}%")


def render_sync_controls() -> None:
    """Render HTTP-based synchronization controls for administrators."""

    if not is_admin(st.session_state):
        return
    st.subheader("Administration")
    if st.button(
        "Sync Documents",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner("Synchronizing S3 documents with Pinecone..."):
                sync_result = synchronize_repository()
            st.session_state.sync_result = sync_result
            get_repository_status.clear()
            st.success("Document synchronization completed.")
        except DocuVerseAPIError as exc:
            logger.error(
                "Document synchronization failed through API",
                extra={
                    "operation": "application",
                    "event": "frontend_sync_failed",
                    "error_type": type(exc).__name__,
                },
            )
            st.error(str(exc))

    if sync_result := st.session_state.get("sync_result"):
        with st.expander("Last Synchronization Summary"):
            st.write(f"Documents scanned: {sync_result['documents_scanned']}")
            st.write(f"Documents added: {sync_result['documents_added']}")
            st.write(f"Documents updated: {sync_result['documents_updated']}")
            st.write(f"Documents removed: {sync_result['documents_removed']}")
            st.write(f"Unchanged documents: {sync_result['unchanged_documents']}")
            st.write(f"Chunks added: {sync_result['chunks_added']}")
            removed_vectors = (
                sync_result["chunks_removed"]
                + sync_result["vectors_removed_from_pinecone"]
            )
            st.write(f"Vectors/chunks removed: {removed_vectors}")
            st.write(f"Failures: {sync_result['failed_documents']}")


with st.sidebar:
    st.image("assets/docuverse_logo.png", width=250)
    st.markdown(
        '<p class="docuverse-tagline">Your Documents. Instantly Explore.</p>',
        unsafe_allow_html=True,
    )
    username = html.escape(str(st.session_state.username))
    role = html.escape(str(st.session_state.role).title())
    st.markdown(
        f"""
        <div class="account-card">
            <div>Logged in as: <strong>{username}</strong></div>
            <div>Role: <strong>{role}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Logout", use_container_width=True):
        logout(st.session_state)
        for key in ("app_mode", "messages", "sync_result"):
            st.session_state.pop(key, None)
        st.rerun()

    if is_admin(st.session_state):
        selected_mode = st.radio(
            "Mode",
            ("Chat", "Admin"),
            horizontal=True,
            key="app_mode",
        )
    else:
        selected_mode = "Chat"
        st.caption("Mode: User")

    if selected_mode == "Admin" and is_admin(st.session_state):
        render_sync_controls()

    st.divider()
    st.header("Document Repository")
    try:
        status_payload = get_repository_status()
    except DocuVerseAPIError as exc:
        logger.warning(
            "Repository status unavailable through API",
            extra={
                "operation": "application",
                "event": "frontend_repository_status_failed",
                "error_type": type(exc).__name__,
            },
        )
        st.error("Backend: unavailable")
        st.caption(str(exc))
    else:
        s3_status = status_payload.get("s3", {})
        pinecone_status = status_payload.get("pinecone", {})
        if s3_status.get("connected"):
            st.success("S3: connected")
            st.caption(f"Bucket: {s3_status.get('bucket', 'Unavailable')}")
            st.metric(
                "Available documents",
                s3_status.get("supported_documents", 0),
            )
            documents = s3_status.get("documents", [])
            if documents:
                with st.expander("Available Documents"):
                    for document in documents:
                        st.write(f"• {document}")
                if selected_mode == "Chat":
                    search_scope = st.selectbox(
                        "Search Document",
                        ["All Documents", *documents],
                    )
                    if search_scope != "All Documents":
                        selected_document = search_scope
            else:
                st.caption("No supported documents found.")
        else:
            st.error("S3: unavailable")

        if pinecone_status.get("connected"):
            st.success("Pinecone: connected")
            if selected_mode == "Admin":
                indexed_documents = pinecone_status.get("indexed_documents")
                if indexed_documents is not None:
                    st.metric("Indexed documents", indexed_documents)
                st.metric(
                    "Indexed chunks",
                    pinecone_status.get("total_vectors", 0),
                )
        else:
            st.error("Pinecone: unavailable")

    st.divider()
    st.caption(f"API: {API_BASE_URL}")
    if selected_mode == "Chat" and st.button(
        "Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


if selected_mode == "Admin" and is_admin(st.session_state):
    st.title("DocuVerse Administration")
    st.subheader("Document Synchronization")
    st.caption(
        "Use Sync Documents in the sidebar to incrementally synchronize "
        "Amazon S3 with Pinecone."
    )
    st.stop()


st.title("DocuVerse")
st.subheader("Intelligent Search Across Your Documents")
st.caption(
    "Ask questions and receive answers grounded in your organization's "
    "document repository."
)
if selected_document:
    st.info(f"Searching document: {selected_document}")
else:
    st.info("Searching: All Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            render_evaluation(message.get("evaluation"))

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document repository..."):
                result = ask_question_http(question, document=selected_document)
            answer = result["answer"]
            sources = result.get("sources", [])
            evaluation = result.get("evaluation")
            st.markdown(answer)
            render_sources(sources)
            render_evaluation(evaluation)
        except DocuVerseAPIError as exc:
            logger.error(
                "Chat API request failed",
                extra={
                    "operation": "application",
                    "event": "frontend_chat_request_failed",
                    "error_type": type(exc).__name__,
                },
            )
            answer = str(exc)
            sources = []
            evaluation = None
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "evaluation": evaluation,
        }
    )
