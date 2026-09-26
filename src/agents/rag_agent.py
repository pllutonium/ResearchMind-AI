"""
Module: rag_agent.py
Description: End-to-end Retrieval-Augmented Generation orchestrator.
"""
from typing import Dict, Any, List, Optional
from src.embeddings.embedder import embed_single_text
from src.vectorstore.chroma_store import search_similar_chunks
from src.llm.client import OllamaClient


SYSTEM_INSTRUCTION = """
You are an expert academic research assistant. 
Answer the user query based strictly on the provided research paper excerpts.
Adhere strictly to the following parameters:
1. Synthesize accurate, objective responses derived entirely from the supplied context.
2. Explicitly cite the originating page numbers using the syntax: [Page X].
3. If the provided context does not contain sufficient factual evidence to resolve the query, state: "The provided literature does not contain adequate evidence to address this query."
"""


class RAGPipeline:
    """
    Coordinates semantic vector retrieval and context-grounded response generation.
    """

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.llm = OllamaClient()

    def query(self, user_query: str, paper_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a research inquiry through vector retrieval and generative synthesis.

        Args:
            user_query (str): Input query text.
            paper_id (Optional[str]): Optional paper identifier to filter context to a single document.

        Returns:
            Dict[str, Any]: Serialized result containing synthesized answer and supporting citations.
        """
        # Step 1: Encode query to dense vector
        query_vector = embed_single_text(user_query)

        # Step 2: Retrieve relevant contextual passages
        retrieved_passages = search_similar_chunks(query_vector, top_k=self.top_k, paper_id=paper_id)


        if not retrieved_passages:
            return {
                "query": user_query,
                "response": "No relevant documents found in the vector index.",
                "sources": [],
            }

        # Step 3: Compile context block
        context_blocks = []
        for idx, item in enumerate(retrieved_passages, start=1):
            source_tag = f"Excerpt {idx} (Document: {item['paper_id']}, Page: {item['page_number']})"
            context_blocks.append(f"{source_tag}:\n{item['text']}")

        formatted_context = "\n\n---\n\n".join(context_blocks)

        augmented_prompt = (
            f"Context Documents:\n{formatted_context}\n\n"
            f"Research Inquiry: {user_query}\n\n"
            f"Synthesized Answer:"
        )

        # Step 4: Execute LLM inference
        answer = self.llm.generate_response(
            prompt=augmented_prompt,
            system_prompt=SYSTEM_INSTRUCTION
        )

        return {
            "query": user_query,
            "response": answer,
            "sources": [
                {"paper_id": p["paper_id"], "page_number": p["page_number"]}
                for p in retrieved_passages
            ],
        }