import asyncio
from scripts.evaluate_ragas import RagasEvaluator
from src.api.routes import evaluate_pipeline, delete_document


def test_ragas_evaluator_metrics():
    question = "Kiến trúc RAG doanh nghiệp tách biệt làm mấy đường ống?"
    answer = "Kiến trúc RAG doanh nghiệp tách biệt 2 đường ống độc lập gồm Ingestion và Query."
    contexts = [
        "Kiến trúc RAG doanh nghiệp tách biệt làm 2 đường ống độc lập: Ingestion Pipeline và Query Pipeline."
    ]
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

    assert scores["faithfulness"] == 1.0


def test_api_evaluate_and_delete_endpoints():
    eval_res = asyncio.run(evaluate_pipeline(_="127.0.0.1"))
    assert "chunking_benchmark" in eval_res
    assert "ragas_evaluation" in eval_res
    assert len(eval_res["chunking_benchmark"]) == 3
    assert len(eval_res["ragas_evaluation"]) == 3

    del_res = asyncio.run(delete_document(document_id="temp_doc", _="127.0.0.1"))
    assert del_res["success"] is True
