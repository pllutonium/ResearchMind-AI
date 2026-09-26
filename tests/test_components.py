"""
Unit and integration tests for ResearchMind AI core components.
"""
import sys
import os
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.ingestion.chunker import chunk_text, _get_overlap_prefix
from src.db.metadata_store import init_db, save_paper_metadata, get_paper_metadata, list_all_papers
from src.llm.client import OllamaClient


class TestChunker(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_single_chunk(self):
        text = "This is a brief academic sentence."
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_boundary_aware_splitting(self):
        text = (
            "First section discussing methods.\n\n"
            "Second section discussing experimental results.\n\n"
            "Third section concluding the research."
        )
        chunks = chunk_text(text, chunk_size=60, overlap=10)
        self.assertGreaterEqual(len(chunks), 2)
        # Ensure chunks don't start or end with broken words
        for c in chunks:
            self.assertTrue(len(c) > 0)
            self.assertFalse(c.startswith(" "))

    def test_overlap_word_boundary(self):
        prefix = _get_overlap_prefix("Transformers achieve state of the art results", overlap=15)
        # Should not start with a fragmented word
        self.assertFalse(prefix.startswith(" "))
        self.assertIn("results", prefix)


class TestMetadataStore(unittest.TestCase):
    def test_save_and_retrieve_paper(self):
        init_db()
        test_id = "test_paper_001"
        save_paper_metadata(
            paper_id=test_id,
            filename="test_paper.pdf",
            page_count=12,
            chunk_count=45,
            title="A Test Paper on Graph Neural Networks",
        )
        meta = get_paper_metadata(test_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["paper_id"], test_id)
        self.assertEqual(meta["page_count"], 12)
        self.assertEqual(meta["chunk_count"], 45)

        papers = list_all_papers()
        ids = [p["paper_id"] for p in papers]
        self.assertIn(test_id, ids)


class TestOllamaClient(unittest.TestCase):
    def test_client_defaults(self):
        client = OllamaClient()
        self.assertIsNotNone(client.model_name)
        opts = client._get_default_options({"temperature": 0.5})
        self.assertEqual(opts["temperature"], 0.5)
        self.assertEqual(opts["num_ctx"], config.LLM_NUM_CTX)

    def test_backend_switching(self):
        client = OllamaClient()
        self.assertEqual(client.backend, "ollama")
        client.set_backend("groq", groq_api_key="gsk_test123", model="llama-3.3-70b-versatile")
        self.assertEqual(client.backend, "groq")
        self.assertEqual(client.groq_api_key, "gsk_test123")
        self.assertIn("llama-3.3-70b-versatile", client.list_available_models())


if __name__ == "__main__":
    unittest.main()

