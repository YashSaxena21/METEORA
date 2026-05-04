import unittest

from meteora import HashingEncoder, MeteoraReranker


class LangChainLikeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class RerankerTest(unittest.TestCase):
    def test_rerank_returns_selected_result_objects(self):
        reranker = MeteoraReranker(HashingEncoder(), fallback_to_query_rationale=True)

        results = reranker.rerank(
            query="assignment consent",
            documents=[
                "The agreement may not be assigned without consent.",
                "Invoices are due in thirty days.",
            ],
            order="document",
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].rank, 1)
        self.assertIn("agreement", results[0].text)

    def test_filter_returns_original_documents(self):
        documents = [
            LangChainLikeDocument("Assignment requires prior consent.", {"id": "a"}),
            LangChainLikeDocument("Invoice terms.", {"id": "b"}),
        ]
        reranker = MeteoraReranker(HashingEncoder())

        selected = reranker.filter("assignment consent", documents, order="document")

        self.assertIs(selected[0], documents[0])

    def test_duplicate_document_indices_preserve_original_mapping(self):
        documents = [
            {"index": 0, "text": "Assignment requires prior consent.", "id": "assignment"},
            {"index": 0, "text": "Successors and assigns language.", "id": "successors"},
        ]
        reranker = MeteoraReranker(HashingEncoder(), fallback_to_query_rationale=False)

        results = reranker.rerank(
            "assignment",
            documents,
            rationales=["assignment consent", "successors assigns"],
            order="document",
        )

        self.assertGreaterEqual(len(results), 1)
        for result in results:
            self.assertEqual(result.text, result.document["text"])

    def test_rationale_generator_is_used(self):
        calls = []

        def generator(query, documents):
            calls.append((query, documents))
            return ["assignment consent"]

        reranker = MeteoraReranker(HashingEncoder(), rationale_generator=generator)
        reranker.rerank(
            "does assignment require consent?",
            ["Assignment requires prior consent.", "Invoice terms."],
        )

        self.assertEqual(calls[0][0], "does assignment require consent?")
        self.assertEqual(len(calls[0][1]), 2)

    def test_verifier_filters_rerank_results(self):
        responses = iter(
            [
                {"relevant": True, "flagged": False},
                {"relevant": True, "flagged": True, "flag_types": ["CONTRADICTION"]},
            ]
        )
        reranker = MeteoraReranker(HashingEncoder(), verifier=lambda prompt: next(responses))

        results = reranker.rerank(
            "assignment consent",
            [
                "Assignment requires prior consent.",
                "Assignment is unrestricted.",
            ],
            rationales=["assignment consent"],
            order="document",
        )

        self.assertEqual([result.index for result in results], [0])


if __name__ == "__main__":
    unittest.main()
