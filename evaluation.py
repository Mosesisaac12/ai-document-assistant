import re
from time import perf_counter
import unicodedata

from rag_engine import answer_question

# -------------------------------------
# TEXT NORMALIZATION
# -------------------------------------

def normalize_text(text):
    """
    Convert text into a consistent format for comparison.
    """

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    # Convert differnt dash characters to a standard hyphen.
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()

# -----------------------------------
# KEYWORD EVALUATION
# -----------------------------------

def evaluate_keywords(
    answer,
    expected_keywords,
):

    """
    Check how many expected keywords or phrases appear
    in the generated answer.
    """

    if not expected_keywords:
        return {
            "matched_keywords":[],
            "missing_keywords":[],
            "keyword_coverage": None,
        }

    normalized_answer = normalize_text(answer)

    matched_keywords = []
    missing_keywords = []

    for keyword in expected_keywords:

        normalized_keyword = normalize_text(keyword)

        if normalized_keyword in normalized_answer:
            matched_keywords.append(keyword)

        else:
            missing_keywords.append(keyword)
    
    keyword_coverage = (
        len(matched_keywords)
        / len(expected_keywords)

    )

    return {
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keyword_coverage": keyword_coverage,
    }

# -----------------------------------------
# PAGE-RETRIEVAL EVALUATION
# -----------------------------------------


def evaluate_retrieved_pages(
    retrieved_metadatas,
    expected_pages,
):

    """
    Compare retrieved pages with the expected pages
    """

    retrieved_pages = sorted(
        {
            metadata["page"]
            for metadata in retrieved_metadatas
            if "page" in metadata
        }
    )

    if not expected_pages:
        return {
            "retrieved_pages": retrieved_pages,
            "matched_pages": [],
            "missing_pages": [],
            "page_recall": None,
            "retrieval_pass": None,
        }

    expected_page_set = set(expected_pages)
    retrieved_page_set = set(retrieved_pages)
    
    matched_pages = sorted(
        expected_page_set
        & retrieved_page_set
    )

    missing_pages = sorted(
        expected_page_set - retrieved_page_set
    )

    page_recall = (
        len(matched_pages)/ len(expected_page_set)

    )

    retrieval_pass = (
        page_recall == 1.0
    )

    return {
        "retrieved_pages": retrieved_pages,
        "matched_pages": matched_pages,
        "missing_pages": missing_pages,
        "page_recall": page_recall,
        "retrieval_pass":retrieval_pass,
    }

# -------------------------------------------
# SINGLE TEST CASE
# -------------------------------------------
def evaluate_test_case(
    collection,
    question,
    expected_answer="",
    expected_pages=None,
    expected_keywords=None
):

    """
    Run one question through the complete RAG pipeline.

    Returns retrieval results, answer results and timing.
    """

    expected_pages = expected_pages or []
    expected_keywords = expected_keywords or []

    start_time = perf_counter()

    result = answer_question(
        user_question=question,
        collection=collection,
        previous_messages=[],
    )

    response_time = (
        perf_counter() - start_time
    )

    answer = result["answer"]

    page_evaluation = evaluate_retrieved_pages(
        retrieved_metadatas=result["metadatas"],
        expected_pages=expected_pages,
    )

    keyword_evaluation = evaluate_keywords(
        answer=answer,
        expected_keywords=expected_keywords,
    )

    retrieval_pass = page_evaluation[
        "retrieval_pass"
    ]

    keyword_coverage = keyword_evaluation[
        "keyword_coverage"
    ]

    if(
        retrieval_pass is None
        or keyword_coverage is None
    ):

        automatic_pass = None

    else:
        automatic_pass = (
            retrieval_pass and keyword_coverage == 1.0
        )

    return {
        "question" : question,
        "expected_answer": expected_answer,
        "generated_answer" : answer,
        "expected_pages" : expected_pages,
        "retrieved_pages" : page_evaluation[
            "retrieved_pages"
        ],
        "matched_pages" : page_evaluation[
            "matched_pages"
        ],
        "missing_pages" : page_evaluation[
            "missing_pages"
        ],
        "page_recall" : page_evaluation[
            "page_recall"
        ],
        "retrieval_pass": retrieval_pass,
        "expected_keywords": expected_keywords,
        "matched_keywords": keyword_evaluation[
            "matched_keywords"
        ],
        "missing_keywords" : keyword_evaluation[
            "missing_keywords"
        ],
        "keyword_coverage": keyword_coverage,
        "automatic_pass": automatic_pass,
        "response_time_seconds": round(
            response_time,
            2,
        ),
        "documents" : result["documents"],
        "metadatas" : result["metadatas"],
        "distances" : result["distances"],
    }

# --------------------------------------------
# COMPLETE TEST SUITE
# --------------------------------------------

def evaluate_test_suite(
    collection,
    test_cases,
):

    """
    Evaluate a list of test cases.

    Each test case should be a dictionary conatining:
        question
        expected_answer
        expected_pages
        expected_keywords
    """

    evaluation_results = []

    for test_case in test_cases:

        result = evaluate_test_case(
            collection=collection,
            question=test_case["question"],
            expected_answer=test_case.get("expected_answer",""),
            expected_pages=test_case.get(
                "expected_pages",
                [],
            ),
            expected_keywords=test_case.get(
                "expected_keywords",
                [],
            )
        )

        evaluation_results.append(result)

    return evaluation_results

# -----------------------------------------------
# EVALUATION SUMMARY
# -----------------------------------------------

def create_evaluation_summary(
    evaluation_results,
):

    """
    Create a summary of all completed test cases.
    """

    total_test = len(evaluation_results)

    retrieval_passes = sum(
        result["retrieval_pass"] is True
        for result in evaluation_results
    )

    automatic_passes = sum(
        result["automatic_pass"] is True
        for result in evaluation_results
    )

    evaluated_response_times = [
        result["response_time_seconds"]
        for result in evaluation_results
    ]

    if evaluated_response_times:
        average_response_time = (
            sum(evaluated_response_times)
            / len(evaluated_response_times)
        )

    else:
        average_response_time = 0

    return {
        "total_test" : total_test,
        "retrieval_passes" : retrieval_passes,
        "automatic_passes" : automatic_passes,
        "average_response_time_seconds" :round(
            average_response_time,
            2,
        )
    }

# -----------------------------------------------
# MAIN EVALUATION ENTRY POINT
# -----------------------------------------------

def run_evaluation(
    collection,
    test_cases,
):

    """
    Run the complete evaluation process.

    This is the main entry point of evaluation.py
    """

    evaluation_results = evaluate_test_suite(
        collection=collection,
        test_cases=test_cases,
    )

    evaluation_summary = create_evaluation_summary(
        evaluation_results
    )

    return {
        "results": evaluation_results,
        "summary": evaluation_summary,
    }
