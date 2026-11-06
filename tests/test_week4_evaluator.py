from scripts.evaluate_ragas import RagasEvaluator


def test_ragas_evaluator_metrics():
    question = "Kiến trúc RAG doanh nghiệp tách biệt làm mấy đường ống?"
    answer = "Kiến trúc RAG doanh nghiệp tách biệt 2 đường ống độc lập gồm Ingestion và Query."
    contexts = ["Kiến trúc RAG doanh nghiệp tách biệt làm 2 đường ống độc lập: Ingestion Pipeline và Query Pipeline."]
    ground_truth = "Hệ thống RAG cấp doanh nghiệp tách biệt 2 đường ống độc lập: Ingestion Pipeline và Query Pipeline."

    scores = RagasEvaluator.evaluate_sample(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    assert scores["faithfulness"] >= 0.8
    assert scores["answer_relevance"] >= 0.7
    assert scores["context_recall"] >= 0.8


def test_ragas_evaluator_unsupported_refusal():
    question = "Hệ thống có hỗ trợ dịch chuyển tức thời không?"
    answer = "Không tìm thấy thông tin trong tài liệu."
    contexts = ["Hệ thống chỉ hỗ trợ tìm kiếm tài liệu văn bản."]
    ground_truth = "Không có thông tin."

    scores = RagasEvaluator.evaluate_sample(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
    )

    # When the model properly refuses hallucination, faithfulness should be high
    assert scores["faithfulness"] == 1.0
