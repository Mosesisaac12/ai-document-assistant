from evaluation import(
    normalize_text,
    evaluate_keywords,
    evaluate_retrieved_pages,
    create_evaluation_summary,
)

def test_normalize_text():
    result = normalize_text("  HELLO WORLD ")
    assert result == "hello world"


def test_evaluate_keywords_all_match():
    result = evaluate_keywords(
        "the Transformer uses attention mechanishms.",
        ["Transformer","attention"],
    )

    assert result["keyword_coverage"] == 1.0
    assert result["missing_keywords"] == []

def test_evaluate_keywords_partial_match():
    result = evaluate_keywords(
        "The Transformer uses attention. ",
        ["Transformer", "convolution"],
    )

    assert result["keyword_coverage"] == 0.5
    assert result["missing_keywords"] == ["convolution"]

def test_evaluate_retrieved_pages_all_match():
    result = evaluate_retrieved_pages(
        retrieved_metadatas=[
            {"page":1},
            {"page":2},
        ],
        expected_pages=[1,2],
    )

    assert result["page_recall"] == 1.0
    assert result["retrieval_pass"] is True
    assert result["missing_pages"] == []

def test_evaluate_retrieved_pages_missing_page():
    result = evaluate_retrieved_pages(
        retrieved_metadatas=[
            {"page": 1},
        ],
        expected_pages=[1,3],
    )
    
    assert result["page_recall"] == 0.5
    assert result["retrieval_pass"] is False
    assert result["missing_pages"] == [3]

def test_create_evaluation_summary():
    evaluation_results = [
        {
            "retrieval_pass": True,
            "automatic_pass": True,
            "response_time_seconds":2.0,
        },
        {
            "retrieval_pass": False,
            "automatic_pass": False,
            "response_time_seconds":4.0,
        },
    ]

    result = create_evaluation_summary(evaluation_results)

    assert result["total_test"] == 2
    assert result["retrieval_passes"] == 1
    assert result["automatic_passes"] == 1
    assert result["average_response_time_seconds"] == 3.0