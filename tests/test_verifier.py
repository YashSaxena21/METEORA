import json
import unittest

from meteora import (
    Chunk,
    HashingEncoder,
    MeteoraSelector,
    MeteoraVerifier,
    parse_verifier_response,
)


class VerifierTest(unittest.TestCase):
    def test_parse_json_verifier_response(self):
        chunk = Chunk(text="Evidence text", index=3)

        result = parse_verifier_response(
            '{"relevant": true, "flagged": false, "reason": "Supports the query.", "confidence": 0.9}',
            chunk=chunk,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "Supports the query.")
        self.assertEqual(result.confidence, 0.9)

    def test_parse_error_fails_closed_by_default(self):
        chunk = Chunk(text="Evidence text", index=3)

        result = parse_verifier_response("not json", chunk=chunk)

        self.assertFalse(result.accepted)
        self.assertTrue(result.flagged)
        self.assertEqual(result.flag_types, ("PARSE_ERROR",))

    def test_raw_response_is_json_serializable(self):
        class ClientResponse:
            pass

        chunk = Chunk(text="Evidence text", index=3)

        result = parse_verifier_response(ClientResponse(), chunk=chunk)

        json.dumps(result.to_dict())
        self.assertIsInstance(result.to_dict()["raw_response"], str)

    def test_verify_chunks_filters_flagged_and_irrelevant_chunks(self):
        responses = iter(
            [
                {"relevant": True, "flagged": False, "reason": "Good evidence."},
                {
                    "relevant": True,
                    "flagged": True,
                    "reason": "Contradicts accepted evidence.",
                    "flag_types": ["CONTRADICTION"],
                },
                {"relevant": False, "flagged": False, "reason": "Off topic."},
            ]
        )

        def fake_model(prompt):
            self.assertIn("Previously accepted evidence", prompt)
            return next(responses)

        verifier = MeteoraVerifier(fake_model)
        verified = verifier.verify_chunks(
            query="Is assignment restricted?",
            chunks=["Assignment requires consent.", "Assignment is unrestricted.", "Invoice terms."],
            rationales=["Look for assignment restrictions."],
        )

        self.assertEqual(verified.accepted_indices, (0,))
        self.assertEqual(verified.flagged_indices, (1,))
        self.assertEqual(verified.rejected_indices, (1, 2))

    def test_verify_selection_shortcut(self):
        selector = MeteoraSelector(HashingEncoder(), expansion_window=0)
        selection = selector.select(
            chunks=["Assignment requires consent.", "Invoice terms."],
            rationales=["assignment consent"],
        )
        verifier = MeteoraVerifier(lambda prompt: {"relevant": True, "flagged": False})

        verified = verifier.verify_selection("Is assignment restricted?", selection)

        self.assertEqual(verified.accepted_indices, tuple(selection.selected_indices))


if __name__ == "__main__":
    unittest.main()
