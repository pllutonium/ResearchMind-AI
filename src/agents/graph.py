"""
Module: src/agents/graph.py
Description: LangGraph state machine constructing the Agentic Corrective RAG (CRAG) workflow.
Connects retrieval, quality grading, external arXiv fallback, synthesis, and hallucination checks.
"""
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import (
    retriever_node,
    retrieval_evaluator_node,
    arxiv_fallback_node,
    synthesis_node,
    hallucination_grader_node,
)
from src.observability.tracer import tracer


def _route_after_evaluation(state: AgentState) -> Literal["synthesis", "arxiv_fallback"]:
    """Conditional edge: routes to synthesis if local context is sufficient, else triggers arXiv fallback."""
    grade = state.get("retrieval_grade", "RELEVANT")
    if grade == "INSUFFICIENT":
        return "arxiv_fallback"
    return "synthesis"


def _route_after_hallucination_check(state: AgentState) -> Literal["synthesis", "__end__"]:
    """Conditional edge: loops back to synthesis if hallucinated (up to 2 retries), else completes."""
    grade = state.get("hallucination_grade", "GROUNDED")
    retries = state.get("hallucination_retries", 0)

    if grade == "HALLUCINATED" and retries < 2:
        return "synthesis"
    return "__end__"


def build_crag_graph():
    """Builds and compiles the production LangGraph state graph for Corrective RAG."""
    workflow = StateGraph(AgentState)

    # 1. Register discrete processing nodes
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("evaluator", retrieval_evaluator_node)
    workflow.add_node("arxiv_fallback", arxiv_fallback_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("hallucination_grader", hallucination_grader_node)

    # 2. Define workflow edge transitions
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "evaluator")

    workflow.add_conditional_edges(
        "evaluator",
        _route_after_evaluation,
        {
            "synthesis": "synthesis",
            "arxiv_fallback": "arxiv_fallback",
        },
    )

    workflow.add_edge("arxiv_fallback", "synthesis")
    workflow.add_edge("synthesis", "hallucination_grader")

    workflow.add_conditional_edges(
        "hallucination_grader",
        _route_after_hallucination_check,
        {
            "synthesis": "synthesis",
            "__end__": END,
        },
    )

    return workflow.compile()


# Compiled runnable graph
build_crag_app = build_crag_graph
create_crag_app = build_crag_graph
crag_pipeline = build_crag_graph()
app = crag_pipeline


def run_agentic_rag(query: str, user_id: str = "researcher") -> AgentState:
    """
    Executes the complete Agentic Corrective RAG state graph for a research query.
    Attaches Langfuse callback handler for full observability and latency telemetry.
    """
    initial_state: AgentState = {
        "query": query,
        "rewritten_query": None,
        "retrieved_docs": [],
        "retrieval_grade": "PENDING",
        "retrieval_reasoning": "",
        "arxiv_docs": [],
        "candidate_answer": "",
        "hallucination_grade": "PENDING",
        "hallucination_reasoning": "",
        "hallucination_retries": 0,
        "final_response": "",
        "citations": [],
        "trace_url": None,
        "error": None,
    }

    callbacks = []
    handler = tracer.get_callback_handler(trace_name="CRAG-Inference", user_id=user_id)
    if handler:
        callbacks.append(handler)

    config = {"callbacks": callbacks} if callbacks else {}
    result_state = crag_pipeline.invoke(initial_state, config=config)

    if handler:
        tracer.flush(handler)
        result_state["trace_url"] = tracer.get_trace_url(handler)

    return result_state
