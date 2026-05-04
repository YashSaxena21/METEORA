import unittest

from meteora import chunk_text, find_spanning_chunks


class ChunkingTest(unittest.TestCase):
    def test_chunk_text_preserves_offsets_with_extra_spaces(self):
        text = "alpha   beta gamma delta"

        chunks = chunk_text(text, chunk_size=2, overlap=0)

        self.assertEqual([chunk.text for chunk in chunks], ["alpha   beta", "gamma delta"])
        self.assertEqual(text[chunks[0].start_pos : chunks[0].end_pos], "alpha   beta")

    def test_find_spanning_chunks_returns_consecutive_range(self):
        text = "alpha beta gamma delta epsilon"
        chunks = chunk_text(text, chunk_size=2, overlap=0)

        found = find_spanning_chunks(6, 18, chunks)

        self.assertEqual(found, [0, 1])


if __name__ == "__main__":
    unittest.main()
