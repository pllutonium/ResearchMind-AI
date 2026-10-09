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
        client = OllamaClient(backend="ollama")
        self.assertEqual(client.backend, "ollama")
        client.set_backend("groq", groq_api_key="gsk_test123", model="llama-3.3-70b-versatile")
        self.assertEqual(client.backend, "groq")
        self.assertEqual(client.groq_api_key, "gsk_test123")
        self.assertIn("llama-3.3-70b-versatile", client.list_available_models())


class TestCRAGComponents(unittest.TestCase):
    def test_sqlite_docstore_crud(self):
        from src.ingestion.docstore import DocumentStore, ParentDocument
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_docstore.sqlite3"
            store = DocumentStore(storage_path=db_path)
            
            p1 = ParentDocument(
                parent_id="p1",
                paper_id="paper_a",
                page_number=1,
                title="Paper A",
                text="Content of parent document 1.",
                metadata={"test": True},
            )
            store.put(p1)
            self.assertEqual(store.count(), 1)
            
            retrieved = store.get("p1")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.parent_id, "p1")
            self.assertEqual(retrieved.text, "Content of parent document 1.")
            self.assertTrue(retrieved.metadata.get("test"))

    def test_hierarchical_chunker_continuous_stream(self):
        from src.ingestion.chunking import HierarchicalChunker
        from src.ingestion.parser import ParsedPaper, ExtractedPage

        paper = ParsedPaper(
            paper_id="paper_test",
            file_path="",
            title="Continuous Processing",
            total_pages=2,
            pages=[
                ExtractedPage(page_number=1, text="Word1 Word2 Word3 Word4 Word5", char_count=29),
                ExtractedPage(page_number=2, text="Word6 Word7 Word8 Word9 Word10", char_count=30),
            ],
            full_text="",
        )
        chunker = HierarchicalChunker(parent_size=6, parent_overlap=2, child_size=3, child_overlap=1)
        parents, children = chunker.chunk_paper(paper)
        self.assertGreater(len(parents), 0)
        self.assertGreater(len(children), 0)
        # Ensure parent contains text spanning across page 1 and page 2
        self.assertTrue(any("Word5" in p.text and "Word6" in p.text for p in parents))

    def test_bm25_idempotence(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.ingestion.chunking import ChildChunk
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            bm25_path = Path(tmpdir) / "bm25.json"
            retriever = HybridRetriever(bm25_path=bm25_path, collection_name="test_bm25_coll")
            
            chunk = ChildChunk(
                child_id="c1",
                parent_id="p1",
                paper_id="paper_test",
                page_number=1,
                title="Test Paper",
                text="Attention mechanism and transformer architecture",
            )
            # Index chunk once
            retriever.index_chunks([chunk])
            initial_count = len(retriever.bm25_doc_ids)
            
            # Index exact same chunk again (should be idempotent)
            retriever.index_chunks([chunk])
            self.assertEqual(len(retriever.bm25_doc_ids), initial_count)
            self.assertEqual(len(retriever.bm25_corpus), initial_count)


if __name__ == "__main__":
    unittest.main()

