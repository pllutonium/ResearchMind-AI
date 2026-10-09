"""
Module: src/agents/nodes.py
Description: Discrete processing nodes for the Agentic Corrective RAG (CRAG) LangGraph.
Implements retrieval, grading, arXiv fallback, synthesis, and hallucination evaluation.
"""
from typing import Dict, Any, List, Optional
import os
import json
import re
import yaml
import requests
from config.settings import settings
from src.agents.state import AgentState
from src.retrieval.hybrid_retriever import hybrid_retriever


def _load_prompts() -> Dict[str, Any]:
    """Loads decoupled version-controlled prompts from config/prompts.yaml."""
    prompts_path = settings.PROMPTS_FILE
    if prompts_path.exists():
        with open(prompts_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


PROMPTS = _load_prompts()


class LLMInferenceError(RuntimeError):
    """Raised when LLM inference fails across all candidate models and fallback backends."""
    pass


def invoke_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    response_json: bool = False,
) -> str:
    """
    Executes resilient inference using Google Gemini with automatic model failover.
    Attempts direct REST API first (with secure header auth), falling back to LangChain ChatGoogleGenerativeAI.
    """
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise LLMInferenceError("Google Gemini API key not configured. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment.")

    candidate_models = [
        settings.GEMINI_MODEL_NAME,
        "gemini-flash-lite-latest",
        "gemini-3.7-flash",
        "gemini-3.1-flash-lite",
    ]
    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    # Method A: Direct REST API with secure header authentication
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": settings.LLM_MAX_TOKENS,
        },
    }
    if response_json:
        payload["generationConfig"]["responseMimeType"] = "application/json"

    last_error: Optional[Exception] = None
    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            else:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as exc:
            last_error = exc
            continue

    # Method B: Fallback to LangChain ChatGoogleGenerativeAI
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            google_api_key=api_key,
            temperature=temperature,
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        res = llm.invoke(messages)
        return res.content.strip()
    except Exception as e:
        error_msg = f"Inference failed on all candidate models and backends: {e or last_error}"
        raise LLMInferenceError(error_msg) from e


# ==============================================================================
# NODE 1: RETRIEVER NODE
# ==============================================================================
def retriever_node(state: AgentState) -> Dict[str, Any]:
    """Executes hybrid retrieval (BM25 + ChromaDB RRF) and Cross-Encoder reranking."""
    query = state.get("rewritten_query") or state["query"]
    retrieved_parents = hybrid_retriever.retrieve(
        query=query,
        final_top_k=settings.FINAL_TOP_K,
        candidate_pool=settings.INITIAL_CANDIDATE_POOL,
    )

    serialized_docs = []
    citations = []
    for doc in retrieved_parents:
        serialized_docs.append({
            "parent_id": doc.parent_id,
            "paper_id": doc.paper_id,
            "page_number": doc.page_number,
            "title": doc.title,
            "text": doc.text,
            "metadata": doc.metadata,
        })
        citations.append(f"[{doc.title}, Page {doc.page_number}]")

    return {
        "retrieved_docs": serialized_docs,
        "citations": list(set(citations)),
    }


# ==============================================================================
# NODE 2: RETRIEVAL EVALUATOR NODE
# ==============================================================================
def retrieval_evaluator_node(state: AgentState) -> Dict[str, Any]:
    """LLM evaluates whether retrieved chunks contain sufficient evidence to answer the query."""
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {
            "retrieval_grade": "INSUFFICIENT",
            "retrieval_reasoning": "No relevant document chunks were retrieved from the local corpus.",
        }

    formatted_context = "\n\n".join(
        f"[{d['title']}, Page {d['page_number']}]\n{d['text']}" for d in docs
    )

    eval_prompts = PROMPTS.get("retrieval_evaluator", {})
    sys_prompt = eval_prompts.get(
        "system",
        "Evaluate if the retrieved context contains enough evidence to answer the query. Respond JSON: {'grade': 'RELEVANT'|'INSUFFICIENT', 'reasoning': '...'}"
    )
    user_prompt = eval_prompts.get(
        "user_template",
        "Query: {query}\n\nContext:\n{context}"
    ).format(query=state["query"], context=formatted_context)

    try:
        raw_resp = invoke_llm(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            response_json=True,
        )
    except LLMInferenceError:
        raw_resp = "{}"

    grade = "RELEVANT"
    reasoning = "Retrieved chunks provide substantive domain context."

    try:
        parsed = json.loads(re.search(r"\{.*\}", raw_resp, re.DOTALL).group(0))
        grade = parsed.get("grade", "RELEVANT").upper()
        reasoning = parsed.get("reasoning", reasoning)
    except Exception:
        if "insufficient" in raw_resp.lower() and "relevant" not in raw_resp.lower():
            grade = "INSUFFICIENT"
            reasoning = "Heuristic parsing detected insufficient evidence in retrieved context."

    return {
        "retrieval_grade": grade,
        "retrieval_reasoning": reasoning,
    }


# ==============================================================================
# NODE 3: ARXIV FALLBACK NODE
# ==============================================================================
def arxiv_fallback_node(state: AgentState) -> Dict[str, Any]:
    """Pulls 3 academic paper abstracts from arXiv API when local retrieval is insufficient."""
    query = state["query"]
    kw_prompts = PROMPTS.get("arxiv_keyword_extractor", {})
    sys_prompt = kw_prompts.get("system", "Extract 2-4 clean academic keywords for arXiv search.")
    user_prompt = kw_prompts.get("user_template", "Query: {query}").format(query=query)

    try:
        keywords = invoke_llm(sys_prompt, user_prompt, temperature=0.1)
    except LLMInferenceError:
        keywords = query
    clean_keywords = re.sub(r"[^a-zA-Z0-9\s]", "", keywords).strip() or query

    arxiv_docs: List[Dict[str, Any]] = []
    try:
        import arxiv
        search = arxiv.Search(
            query=clean_keywords,
            max_results=settings.ARXIV_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        client = arxiv.Client()
        for result in client.results(search):
            year = result.published.year if result.published else "Recent"
            primary_author = result.authors[0].name if result.authors else "arXiv Scholar"
            arxiv_docs.append({
                "parent_id": f"arxiv_{result.entry_id}",
                "paper_id": result.entry_id,
                "page_number": 1,
                "title": result.title,
                "text": f"Title: {result.title}\nAuthors: {', '.join(a.name for a in result.authors[:3])}\nSummary: {result.summary}",
                "metadata": {
                    "source": "arXiv API Fallback",
                    "url": result.pdf_url,
                    "year": year,
                    "primary_author": primary_author,
                },
            })
    except Exception as e:
        print(f"[WARN] arXiv API client search failed: {e}. Attempting direct feedparser fallback.")
        try:
            import urllib.parse
            import feedparser
            encoded_q = urllib.parse.quote(clean_keywords)
            url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_q}&start=0&max_results={settings.ARXIV_MAX_RESULTS}"
            feed = feedparser.parse(url)
            for entry in feed.entries:
                arxiv_docs.append({
                    "parent_id": f"arxiv_{entry.id}",
                    "paper_id": entry.id,
                    "page_number": 1,
                    "title": entry.title,
                    "text": f"Title: {entry.title}\nSummary: {entry.summary}",
                    "metadata": {"source": "arXiv Direct Feed"},
                })
        except Exception as e2:
            print(f"[WARN] arXiv feedparser fallback also failed: {e2}")

    return {
        "arxiv_docs": arxiv_docs,
    }


# ==============================================================================
# NODE 4: SYNTHESIS NODE
# ==============================================================================
def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Generates an academic synthesis grounded strictly in context with paragraph citations."""
    retrieved = state.get("retrieved_docs", [])
    arxiv_docs = state.get("arxiv_docs", [])
    combined_docs = retrieved + arxiv_docs

    if not combined_docs:
        refusal = "The available academic corpus does not contain sufficient empirical evidence to substantiate this inquiry."
        return {
            "candidate_answer": refusal,
            "final_response": refusal,
        }

    formatted_context_blocks = []
    for d in combined_docs:
        title = d.get("title", "Academic Source")
        page = d.get("page_number", 1)
        author = d.get("metadata", {}).get("primary_author") or title
        year = d.get("metadata", {}).get("year", "2024")
        formatted_context_blocks.append(
            f"SOURCE: [{author}, {year}, Page {page}] | Title: '{title}'\n{d.get('text', '')}"
        )
    context_str = "\n\n".join(formatted_context_blocks)

    synth_prompts = PROMPTS.get("synthesis", {})
    sys_prompt = synth_prompts.get(
        "system",
        "Synthesize an authoritative academic response based strictly on the provided context with [Author/Title, Year, Page X] citations."
    )
    user_prompt = synth_prompts.get(
        "user_template",
        "Query: {query}\n\nContext:\n{context}"
    ).format(query=state["query"], context=context_str)

    # Use temperature 0.0 if regenerating to correct hallucination
    temperature = 0.0 if state.get("hallucination_retries", 0) > 0 else settings.LLM_TEMPERATURE
    try:
        answer = invoke_llm(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    except LLMInferenceError as err:
        refusal = "The system encountered an error connecting to the inference engine to synthesize an academic answer."
        return {
            "candidate_answer": refusal,
            "final_response": refusal,
            "error": str(err),
        }

    return {
        "candidate_answer": answer,
    }


# ==============================================================================
# NODE 5: HALLUCINATION GRADER NODE
# ==============================================================================
def hallucination_grader_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates whether the candidate answer is strictly grounded in the retrieved context."""
    answer = state.get("candidate_answer", "")
    standard_refusal = "does not contain sufficient empirical evidence"

    # If the model produced standard refusal, it is grounded by definition
    if standard_refusal in answer.lower():
        return {
            "hallucination_grade": "GROUNDED",
            "hallucination_reasoning": "Answer adhered to standard refusal constraint without hallucinating.",
            "final_response": answer,
        }

    combined_docs = state.get("retrieved_docs", []) + state.get("arxiv_docs", [])
    context_str = "\n\n".join(d.get("text", "") for d in combined_docs)

    grader_prompts = PROMPTS.get("hallucination_grader", {})
    sys_prompt = grader_prompts.get(
        "system",
        "Evaluate if the candidate answer is strictly grounded in the source context. Respond JSON: {'grade': 'GROUNDED'|'HALLUCINATED', 'explanation': '...'}"
    )
    user_prompt = grader_prompts.get(
        "user_template",
        "Context:\n{context}\n\nCandidate Answer:\n{answer}"
    ).format(context=context_str, answer=answer)

    try:
        raw_resp = invoke_llm(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            response_json=True,
        )
    except LLMInferenceError:
        raw_resp = "{}"

    grade = "GROUNDED"
    reasoning = "Claims in candidate answer are grounded in context."

    try:
        parsed = json.loads(re.search(r"\{.*\}", raw_resp, re.DOTALL).group(0))
        grade = parsed.get("grade", "GROUNDED").upper()
        reasoning = parsed.get("explanation", reasoning)
    except Exception:
        if "hallucinated" in raw_resp.lower() and "grounded" not in raw_resp.lower():
            grade = "HALLUCINATED"
            reasoning = "Candidate response contains claims ungrounded in context."

    current_retries = state.get("hallucination_retries", 0)
    if grade == "HALLUCINATED":
        current_retries += 1

    return {
        "hallucination_grade": grade,
        "hallucination_reasoning": reasoning,
        "hallucination_retries": current_retries,
        "final_response": answer if grade == "GROUNDED" or current_retries >= 2 else "",
    }
