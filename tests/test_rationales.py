import unittest

from meteora import extract_flag_instructions, parse_rationale_texts, parse_rationales
from meteora.rationales import coerce_rationales


class RationalesTest(unittest.TestCase):
    def test_parse_xml_rationales_with_labels_and_flags(self):
        response = """
Query: What licenses are granted?
<rationale_2>[Licensing boundary] Search for limits on redistribution. Flag Instructions: flag reversed permissions.</rationale_2>
<rationale_1>[Grant language] Look for explicit license grants.</rationale_1>
"""

        rationales = parse_rationales(response)

        self.assertEqual([r.index for r in rationales], [1, 2])
        self.assertEqual(rationales[0].label, "Grant language")
        self.assertEqual(rationales[1].flag_instructions, "flag reversed permissions.")

    def test_parse_numbered_fallback(self):
        response = "1. [Assignment] Look for assignment limits.\n2. Search for consent language."

        self.assertEqual(
            parse_rationale_texts(response),
            ["Look for assignment limits.", "Search for consent language."],
        )

    def test_extract_flag_instructions(self):
        response = "<rationale_1>Find X. Flag Instructions: reject contradictions.</rationale_1>"

        self.assertEqual(extract_flag_instructions(response), [(1, "reject contradictions.")])

    def test_plain_multiline_rationales_can_be_coerced_as_separate_items(self):
        rationales = coerce_rationales(
            "Look for assignment restrictions.\nLook for consent requirements."
        )

        self.assertEqual([rationale.text for rationale in rationales], [
            "Look for assignment restrictions.",
            "Look for consent requirements.",
        ])


if __name__ == "__main__":
    unittest.main()
