import unittest
from types import SimpleNamespace
from unittest.mock import patch

from meteora_rag import SentenceTransformerEncoder
from meteora_rag.embeddings import encode_texts


class EmbeddingsTest(unittest.TestCase):
    def test_incomplete_embedding_tokenizer_is_rejected(self):
        for vocab_size in (5, 30522):
            model = SimpleNamespace(
                tokenizer=SimpleNamespace(vocab_size=vocab_size, all_special_ids=[0, 1, 2, 3, 4])
            )
            dependency = SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: model)
            with (
                self.subTest(vocab_size=vocab_size),
                patch.dict("sys.modules", {"sentence_transformers": dependency}),
            ):
                if vocab_size == 5:
                    with self.assertRaisesRegex(ValueError, "no trained vocabulary"):
                        SentenceTransformerEncoder("example-model")
                else:
                    self.assertIs(SentenceTransformerEncoder("example-model").model, model)

    def test_backend_type_error_does_not_retry_encoding(self):
        class BrokenEncoder:
            calls = 0

            def encode(self, texts, **kwargs):
                self.calls += 1
                raise TypeError("Embedding backend failed")

        encoder = BrokenEncoder()
        with self.assertRaisesRegex(TypeError, "Embedding backend failed"):
            encode_texts(encoder, ["evidence"])
        self.assertEqual(encoder.calls, 1)

    def test_scalar_embedding_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            encode_texts(lambda texts: 0.5, ["evidence"])


if __name__ == "__main__":
    unittest.main()
