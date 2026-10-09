"""
Module: src/agents/state.py
Description: Typed State definition for the Agentic Corrective RAG (CRAG) state graph.
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """Execution state tracking the complete lifecycle of a user research query through CRAG."""
    query: str
    rewritten_query: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    retrieval_grade: str                    # "RELEVANT" or "INSUFFICIENT"
    retrieval_reasoning: str
    arxiv_docs: List[Dict[str, Any]]        # Fallback arXiv abstracts
    candidate_answer: str
    hallucination_grade: str                # "GROUNDED" or "HALLUCINATED"
    hallucination_reasoning: str
    hallucination_retries: int              # Loop guard (e.g., max 2 retries)
    final_response: str
    citations: List[str]
    trace_url: Optional[str]
    error: Optional[str]
