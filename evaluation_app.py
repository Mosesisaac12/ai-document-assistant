import streamlit as st

from rag_engine import(
    get_chroma_client,
    index_document,
)

from evaluation import run_evaluation


# ---------------------------------------------
# MAIN APPLICATION FORM
# ---------------------------------------------

def main():
    configure_page()
    initialize_session_state()
    chroma_client = load_chroma_client()
    display_header()

    uploaded_file = display_document_uploader()

    collection = process_uploaded_document(
        uploaded_file,
        chroma_client,
    )

    display_test_case_form()
    display_saved_test_cases()
    run_saved_test_cases(collection)
    display_evaluation_results()


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

def configure_page():
    "Configure the evaluation applicatiom"
    st.set_page_config(
        page_title="RAG Evaluation Dashboard",
        page_icon="🧪",
        layout="wide",
    )



# ------------------------------------
# APPLICATION RESOURCES
# ------------------------------------

@st.cache_resource
def load_chroma_client():
    """Create and cache the ChromaDB client."""

    return get_chroma_client()


# ------------------------------------
# SESSION STATE
# ------------------------------------

def initialize_session_state():
    """Initialize evaluation session variables"""

    if "test_cases" not in st.session_state:
        st.session_state.test_cases = []

    if "evaluation_output" not in st.session_state:
        st.session_state.evaluation_output = None

    

# ------------------------------------
# PAGE HEADER
# ------------------------------------

def display_header():
    """Display the evaluation application heading."""

    st.title("🧪 RAG Evaluation Dashboard")

    st.caption(
        "Upload a PDF, create test cases and evaluate "
        "retrieval quality, answer coverage and response time."
    )

# -------------------------------------
# DOCUMENT UPLOAD
# -------------------------------------

def display_document_uploader():
    """Display the PDF uploader and return the uploaded file"""
    st.subheader("1.Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"],
        key="evaluation_pdf",
    )

    return uploaded_file


def process_uploaded_document(
        uploaded_file,
        chroma_client,
):
    """ Index the uploaded PDF and return its collection"""
    if uploaded_file is None:
        st.info("Please upload a PDF document to begin evaluation.")
        return None

    try:
        with st.spinner("Processing and indexing the PDF..."):

            document_info = index_document(
                client=chroma_client,
                pdf_bytes=uploaded_file.getvalue(),
                source_name=uploaded_file.name,
            )

        # Store document information for later use.
        st.session_state.document_info = document_info

        collection = document_info["collection"]

        st.success("PDF processed successfully")

        # Display indexing information.
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Pages",
            document_info["page_count"],
        )

        col2.metric(
            "Total Chunks",
            document_info["chunk_count"],
        )

        col3.metric(
            "Stored Chunks",
            collection.count(),
        )

        if document_info["already_indexed"]:
            st.caption(
                "This document was already indexed."
                "Existing embeddings were reused."
            )

        return collection

    except Exception as error:
        st.error(f"Document processing failed: {error}")
        return None

    


# -----------------------------------------
# TEST-CASE MANAGEMENT
# -----------------------------------------

def display_test_case_form():
    """Display inputs for creating an evaluation test case."""
    st.subheader("2. Create Evaluation Test Case")

    with st.form("test_case_form"):
        question = st.text_area(
            "Test Question",
            placeholder="Example: What is the transformer architecture",
        )

        expected_answer = st.text_area(
            "Expected Answer",
            placeholder= "Enter the refernce for comparison.",
        )

        expected_pages_input = st.text_input(
            "Expected PDF Pages",
            placeholder="Example: 1,2",
            help="Enter the relevant PDf page numbers separated by commas.",
        )

        expected_keywords_input = st.text_input(
            "Expected Keywords",
            placeholder="Example: attention, self-attention,parallel",
            help="Enter keywords or phrases separated by commas."
        )

        submitted = st.form_submit_button(
            "Add Test Case"
        )

        if submitted:

            if not question.strip() or not expected_answer.strip():
                st.warning(
                    "Please enter both a question and an expected answer."
                )

            else:
                try: 
                    expected_pages = [
                        int(page.strip())
                        for page in expected_pages_input.split(",")
                        if page.strip()
                    ]

                    if any(page < 1 for page in expected_pages):
                        raise ValueError(
                            "Page numbers must be posistive integers."
                        )

                except ValueError:
                    st.error(
                        "Enter valid page numbers separated by commas,"
                        "for example:1, 2, 3."
                    )

                    return

                expected_keywords = [
                    keyword.strip()
                    for keyword in expected_keywords_input.split(",")
                    if keyword.strip()
                ]

                test_case = {
                    "question": question.strip(),
                    "expected_answer": expected_answer.strip(),
                    "expected_pages": expected_pages,
                    "expected_keywords": expected_keywords,
                }

                st.session_state.test_cases.append(
                    test_case
                )

                # Clear any old results whem tst cases change.
                st.session_state.evaluation_output = None

                st.success("Test case added successfully")


def display_saved_test_cases():
    """ Display the casees currently stored in the session"""
    st.subheader("3. Saved Evaluation Test Cases")

    test_cases = st.session_state.test_cases

    if not test_cases:
        st.info(
            "No test cases added yet. "
            "Create a test case using the form above."
        )
        return

    st.write(f"Total Test Cases: {len(test_cases)}")

    for index, test_case in enumerate(test_cases):

        with st.expander(
            f"Test Case {index + 1}",
            expanded=False
        ):
            st.markdown("**Questions**")
            st.write(test_case["question"])

            st.markdown("**Expected Answer**")
            st.write(test_case["expected_answer"])

            st.markdown("Expected PDF Pages")
            pages = test_case.get("expected_pages",[])
            st.write(pages if pages else "Not specified")

            st.markdown("**Expected keywords**")
            keywords = test_case.get("expected_keywords",[])
            st.write(keywords if keywords else "Not specified")

            if st.button(
                "Delete Test Case",
                key=f"delete_test_case_{index}",
            ):
                st.session_state.test_cases.pop(index)
                st.session_state.evaluation_output = None
                st.rerun()

# -------------------------------------------
# RUN EVALUATION
# -------------------------------------------

def run_saved_test_cases(collection):
    """Run all saved test cases against the current document"""

    st.subheader("4. Run Evaluation")

    if collection is None:
        st.info("Upload a PDF document before running evaluation.")
        return

    if not st.session_state.test_cases:
        st.info("Add at least one test case before running evaluation.")
        return

    if st.button("Run Evaluation", type="primary"):
        try:
            with st.spinner("Evaluating test cases..."):

                evaluation_output = run_evaluation(
                    collection=collection,
                    test_cases=st.session_state.test_cases,
                )

            st.session_state.evaluation_output = evaluation_output

            st.success("Evaluation completed successfully!")

        except Exception as error:
            st.error(f"Evaluation failed: {error}")

# --------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------

def display_evaluation_results():
    """Display detailed results and the evaluation summary."""

    st.subheader("5. Evaluation Results")

    evaluation_output = st.session_state.evaluation_output

    if evaluation_output is None:
        st.info("Run an evaluation to see the results here.")
        return

    results = evaluation_output["results"]
    summary = evaluation_output["summary"]

    # Display summary metrics.
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Test Cases",
        summary["total_test"],
    )

    col2.metric(
        "Retrieval Passes",
        summary["retrieval_passes"],
    )

    col3.metric("Automatic Passes",
                summary["automatic_passes"],
                )
    col4.metric(
        "Average Response Time",
        f'{summary["average_response_time_seconds"]} s'
    )

    st.divider()

    # Display individual test-case results
    st.subheader("Individual Test Results")

    for index, result in enumerate(results):

        with st.expander(
            f"Test Case {index + 1}",
        ):
            st.write("**Question:**", result["question"])
            st.write("**Expected Answer:**", result.get("expected_answer",""))
            st.write("**Generated Answer:**", result.get("generated_answer",result.get("answer","")))

            st.write(
                "**Retrieval Pass:**",
                "Pass" if result.get("retrieval_pass") is True
                else "Fail" if result.get("retrieval_pass") is False
                else "Not evaluated"
            )

            st.write(
                "** Expected Pages:**",
                result.get("expected_pages",[]),
            )

            st.write(
                "**Retrieved Pages:**",
                result.get("retrieved_pages",[]),
            )

            st.write(
                "**Matched Pages:**",
                result.get("matched_pages", []),

            )

            st.write(
                "** Missing Pages:**",
                result.get("missing_pages",[])
            )
            page_recall = result.get("page_recall")
            st.write(
                "**Page Recall:**"
                f"{page_recall:.0%}" if page_recall is not None
                else "Not evaluated",
            )

            st.write(
                "**Expected Keywords:**",
                result.get("expected_keywords",[]),
            )

            st.write(
                "**Matched Keywords:**",
                result.get("matched_keywords",[]),
            )

            st.write(
                "**Missing Keywords:**",
                result.get("missing_keywords", []),
            )

            keyword_coverage = result.get("keyword_coverage")
            st.write(
                "**Keyword Coverage:**",
                f"{keyword_coverage:0%}"
                if keyword_coverage is not None
                else "Not evaluated",
            )
            st.write(
                "**Automatic Pass:**",
                "Pass" if result.get("automatic_pass") is True
                else "Fail" if result.get("automatic_pass") is False
                else "Not evaluated",
            )

            st.write(
                "**Response Time:**",
                f'{result["response_time_seconds"]} seconds',
            )


if __name__ == "__main__":
    main()