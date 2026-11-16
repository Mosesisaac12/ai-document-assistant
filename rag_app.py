import os
import subprocess
import sys
import time
import urllib.request

import streamlit as st

from rag_engine import (
    get_chroma_client,
    index_document,
    answer_question,
)


# ---------------------------------------------------------
# MAIN APPLICATION FLOW
# ---------------------------------------------------------

def main():
    """Run the complete Streamlit application."""

    configure_page()

    initialize_session_state()

    chroma_client = load_chroma_client()

    display_header()

    uploaded_file = display_sidebar()

    collection = process_uploaded_document(
        uploaded_file=uploaded_file,
        chroma_client=chroma_client,
    )

    display_chat_history()

    handle_user_question(
        collection=collection,
    )



# ---------------------------------------------------------
# APPLICATION CONFIGURATION
# ---------------------------------------------------------

def configure_page():
    """Configure the Streamlit browser page."""

    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📄",
        layout="wide",
    )


# ---------------------------------------------------------
# DATABASE RESOURCE
# ---------------------------------------------------------

@st.cache_resource
def load_chroma_client():
    """
    Create the ChromaDB client once and reuse it
    during Streamlit reruns.
    """

    return get_chroma_client()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

def initialize_session_state():
    """Create the required Streamlit session variables."""

    if "messages" not in st.session_state:
        st.session_state.messages = []



# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

def display_header():
    """Display the application heading and description."""

    st.title("📄 AI Document Assistant")

    st.caption(
        "Upload a PDF and ask questions about its contents."
    )

# ---------------------------------------------------------
# EVALUATION DASHBOARD CONTROLLER
# ---------------------------------------------------------

@st.cache_resource
def get_evaluation_process_state():
    """Keep the evaluation process shared across Streamlit reruns."""
    return {"process": None}


def is_evaluation_running():
    """Check whether the evaluation dashboard process is running."""

    process = get_evaluation_process_state()["process"]

    if process is None:
        return False

    return process.poll() is None


def start_evaluation_dashboard():
    """Start the evaluation dashboard inside this container."""

    state = get_evaluation_process_state()

    if is_evaluation_running():
        return True

    app_directory = os.path.dirname(os.path.abspath(__file__))

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "evaluation_app.py",
        "--server.address=0.0.0.0",
        "--server.port=8502",
        "--server.headless=true",
    ]

    try:
        process = subprocess.Popen(
            command,
            cwd=app_directory,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        state["process"] = process

        for _ in range(40):
            if process.poll() is not None:
                return False

            try:
                urllib.request.urlopen(
                    "http://127.0.0.1:8502/_stcore/health",
                    timeout=1,
                )
                return True
            except Exception:
                time.sleep(0.5)

        return False

    except Exception:
        return False


def stop_evaluation_dashboard():
    """Stop the evaluation dashboard without stopping the main app."""

    state = get_evaluation_process_state()
    process = state["process"]

    if process is None or process.poll() is not None:
        state["process"] = None
        return True

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

    state["process"] = None

    return True

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

def display_sidebar():
    """
    Display the sidebar controls.

    Returns:
        The PDF uploaded by the user, or None.
    """

    with st.sidebar:

        st.header("Document")

        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
        )

        if st.button(
            "Clear chat",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        with st.expander("🛠️ Developer Options"):

            st.caption(
                "Tools for testing and evaluating RAG performance."
            )

            if st.button(
                "Start Evaluation Dashboard",
                use_container_width=True,
            ):

                with st.spinner("Starting Evaluation Dashboard..."):

                    started = start_evaluation_dashboard()

                if started:
                    st.success("Evaluation Dashboard is running.")
                else:
                    st.error("Could not start Evaluation Dashboard.")

            if is_evaluation_running():

                st.link_button(
                    "Open Evaluation Dashboard",
                    "http://localhost:8502",
                    use_container_width=True,
                )

                if st.button(
                    "Stop Evaluation Dashboard",
                    use_container_width=True,
                ):
                    stop_evaluation_dashboard()
                    st.success("Evaluation Dashboard stopped.")
                    st.rerun()

    return uploaded_file


# ---------------------------------------------------------
# DOCUMENT INFORMATION
# ---------------------------------------------------------

def display_document_information(
    uploaded_file,
    document_info,
):
    """Display information about the indexed PDF."""

    if document_info["already_indexed"]:

        st.info(
            f"Using the previously indexed document: "
            f"{uploaded_file.name}"
        )

    else:

        st.success(
            f"Document indexed successfully: "
            f"{uploaded_file.name}"
        )

    with st.sidebar:

        st.subheader("Document details")

        st.write(
            f"**File:** {uploaded_file.name}"
        )

        st.write(
            f"**Pages:** {document_info['page_count']}"
        )

        # st.write(
        #     f"**Stored chunks:** "
        #     f"{document_info['chunk_count']}"
        # )


# ---------------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------------

def process_uploaded_document(
    uploaded_file,
    chroma_client,
):
    """
    Convert the uploaded PDF to bytes and index it.

    Returns:
        The ChromaDB collection for the document.
        Returns None when no PDF is uploaded or processing fails.
    """

    if uploaded_file is None:
        return None

    pdf_bytes = uploaded_file.getvalue()

    try:

        with st.spinner("Preparing the document..."):

            document_info = index_document(
                client=chroma_client,
                pdf_bytes=pdf_bytes,
                source_name=uploaded_file.name
            )

        display_document_information(
            uploaded_file=uploaded_file,
            document_info=document_info,
        )

        return document_info["collection"]

    except ValueError as error:

        st.error(str(error))
        return None

    except Exception as error:

        st.error(
            f"An error occurred while processing the PDF: "
            f"{error}"
        )

        return None


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

def display_chat_history():
    """Display all previously stored chat messages."""

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("source_text")
            ):
                st.caption(message["source_text"])


# ---------------------------------------------------------
# STORE A MESSAGE
# ---------------------------------------------------------

def save_message(
    role,
    content,
    source_text="",
):
    """Save a chat message in Streamlit session state."""

    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "source_text": source_text,
        }
    )


# ---------------------------------------------------------
# RETRIEVAL RESULTS
# ---------------------------------------------------------

def display_retrieval_results(result):
    """Display the chunks retrieved from ChromaDB."""

    documents = result["documents"]
    metadatas = result["metadatas"]
    distances = result["distances"]

    with st.expander(
        "View retrieved document sections"
    ):

        for position, (
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            page_number = metadata.get(
                "page",
                "Unknown",
            )

            st.markdown(
                f"### Result {position} — "
                f"Page {page_number}"
            )

            st.write(document)

            st.caption(
                f"Vector distance: {distance:.4f}"
            )

            st.divider()


# ---------------------------------------------------------
# GENERATE AND DISPLAY ANSWER
# ---------------------------------------------------------

def generate_and_display_answer(
    question,
    collection,
):
    """Generate an answer and display its sources."""

    previous_messages = (
        st.session_state.messages[:-1]
    )

    try:

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the document and "
                "generating an answer..."
            ):

                result = answer_question(
                    user_question=question,
                    collection=collection,
                    previous_messages=previous_messages,
                )

            answer = result["answer"]

            source_text = result.get(
                "source_text",
                "",
            )

            st.markdown(answer)

            if source_text:
                st.caption(source_text)

            # display_retrieval_results(result)

        save_message(
            role="assistant",
            content=answer,
            source_text=source_text,
        )

    except Exception as error:

        error_message = (
            f"An error occurred while answering: {error}"
        )

        with st.chat_message("assistant"):
            st.error(error_message)


# ---------------------------------------------------------
# HANDLE USER QUESTION
# ---------------------------------------------------------

def handle_user_question(collection):
    """Accept and process the user's question."""

    question = st.chat_input(
        "Ask a question about the uploaded document"
    )

    if not question:
        return

    save_message(
        role="user",
        content=question,
    )

    with st.chat_message("user"):
        st.markdown(question)

    if collection is None:

        warning_message = (
            "Please upload a PDF before asking a question."
        )

        with st.chat_message("assistant"):
            st.warning(warning_message)

        save_message(
            role="assistant",
            content=warning_message,
        )

        return

    generate_and_display_answer(
        question=question,
        collection=collection,
    )




# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
