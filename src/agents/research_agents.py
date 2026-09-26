"""
Module: research_agents.py
Description: The 5 Specialized Sub-Agents and the AICoordinator for collaborative pipeline execution.
"""
from typing import Dict, Any, List, Optional
import re
import json
from src.llm.client import OllamaClient
from src.vectorstore.chroma_store import VectorStore



def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Shared helper: turn retrieved chunks into a labeled context block for the LLM prompt."""
    return "\n\n".join(
        f"[{c.get('paper_id', 'unknown')} - Page {c.get('page_number', 'N/A')}] {c.get('text', '')}"
        for c in chunks
    )


class LiteratureSearchAgent:
    def __init__(self, vector_store: VectorStore, llm: OllamaClient):
        self.vector_store = vector_store
        self.llm = llm

    def run(self, query: str, n_results: int = 4) -> Dict[str, Any]:
        results = self.vector_store.query_similar(query_text=query, n_results=n_results)

        if not results:
            return {"query": query, "results": [], "summary": "No indexed documents matched this query yet. Ingest a paper first."}

        system_prompt = "You are the Literature Search Agent, acting as an elite Senior Academic Synthesis Director."
        prompt = f"""Analyze these retrieved literature segments and provide an executive synthesis detailing key themes, methodologies, and findings.

Retrieved Literature Segments:
{_format_context(results)}

Format your response cleanly:
- **Core Findings & Methodologies**: 2-3 substantive technical takeaways citing the source [Paper ID - Page X].
- **Executive Synthesis**: A dense, authoritative paragraph (under 120 words) summarizing the state of the literature.

Executive Overview:"""
        summary = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            options={"num_predict": 1024, "temperature": 0.2},
        )
        return {"query": query, "results": results, "summary": summary}



class PaperReaderAgent:
    """
    Extracts structured fields (Problem, Methodology, Dataset, Results, Limitations,
    Future Work) from ONE specific paper, grounded strictly in that paper's own chunks.

    Every field starts as 'Not found in retrieved context' to eliminate hallucination.
    """

    FIELD_LABELS = {
        "Problem Statement": "PROBLEM",
        "Methodology": "METHODOLOGY",
        "Dataset Used": "DATASET",
        "Results & Findings": "RESULTS",
        "Limitations": "LIMITATIONS",
        "Future Work": "FUTURE_WORK",
    }

    def __init__(self, vector_store: VectorStore, llm: OllamaClient):
        self.vector_store = vector_store
        self.llm = llm

    def run(self, paper_id: str) -> Dict[str, Any]:
        if not paper_id:
            return {"paper_id": None, "table": {}, "citations": "", "error": "No paper_id was provided."}

        query = (
            "problem statement methodology architecture dataset training "
            "evaluation benchmark results limitations future work"
        )
        retrieved = self.vector_store.query_similar(query_text=query, n_results=5, paper_id=paper_id)

        if not retrieved:
            return {
                "paper_id": paper_id,
                "table": {k: "Not found in retrieved context" for k in self.FIELD_LABELS},
                "citations": "",
                "error": f"No indexed chunks found for paper_id='{paper_id}'.",
            }

        context = _format_context(retrieved)
        citations = ", ".join(sorted({f"Page {doc.get('page_number', 'N/A')}" for doc in retrieved}))

        system_prompt = "You are the Paper Reader Agent, acting as a high-precision Academic Information Architect."
        prompt = f"""Extract findings strictly using the exact labeled format below, grounded ONLY in the provided context.
For each extracted dimension, include the specific page citation in parentheses, e.g. '(Page 4)'.
If the context does not contain enough information for a field, write 'Not found in retrieved context' for that field.
Do not use markdown bolding inside values. Be precise, technical, and concrete.

Context:
{context}

Format:
PROBLEM:
METHODOLOGY:
DATASET:
RESULTS:
LIMITATIONS:
FUTURE_WORK:"""

        raw_output = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            options={"num_predict": 1024, "temperature": 0.2},
        )

        table_data = {key: "Not found in retrieved context" for key in self.FIELD_LABELS}

        # 1. Attempt JSON parsing if output contains a structured JSON block
        try:
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                for key, label in self.FIELD_LABELS.items():
                    val = (
                        parsed_json.get(label)
                        or parsed_json.get(key)
                        or parsed_json.get(label.lower())
                        or parsed_json.get(key.lower())
                    )
                    if val and isinstance(val, str):
                        clean_val = val.strip().replace("\n", " ")
                        if len(clean_val) > 4 and "not found" not in clean_val.lower():
                            table_data[key] = clean_val
        except Exception:
            pass

        # 2. Resilient pattern matching for structured plain text or markdown labels
        all_labels_pattern = "|".join(re.escape(lbl) for lbl in self.FIELD_LABELS.values())
        for key, label in self.FIELD_LABELS.items():
            if table_data[key] != "Not found in retrieved context":
                continue
            pattern = (
                rf"(?:^|\n)\s*(?:\d+[\.\)]\s*)?(?:\*\*|#+)?\s*{re.escape(label)}[:\-]?\s*(?:\*\*)?\s*"
                rf"([^\n].*?)(?=(?:\n\s*(?:\d+[\.\)]\s*)?(?:\*\*|#+)?\s*(?:{all_labels_pattern})[:\-]?)|\Z)"
            )
            match = re.search(pattern, raw_output, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                value = re.sub(r"^\s*[-*•]\s*", "", value)
                value = value.replace("**", "").replace("`", "").replace("\n", " ").strip()
                if len(value) > 4 and "not found" not in value.lower():
                    table_data[key] = value

        return {"paper_id": paper_id, "table": table_data, "citations": citations}



class ComparisonAgent:
    def __init__(self, vector_store: VectorStore, llm: OllamaClient):
        self.vector_store = vector_store
        self.llm = llm

    def run(self, query: str) -> Dict[str, Any]:
        results = self.vector_store.query_similar(query_text=query, n_results=4)
        if not results:
            return {"query": query, "comparison": "No indexed documents matched this query yet."}

        system_prompt = "You are the Comparison Agent, acting as a Comparative Benchmark & Architecture Specialist."
        prompt = f"""Using ONLY the context below, produce a rigorous comparative analysis contrasting the core methodology, benchmark metrics, and computational trade-offs against baseline approaches mentioned in the literature.

Context:
{_format_context(results)}

Format your response as:
1. **Architectural & Methodological Nuances**: Direct structural differences between the proposed method and traditional approaches.
2. **Empirical Benchmarks & Trade-offs**: Quantitative performance gains vs. computational/memory footprint.
3. **Summary Verdict**: Concise synthesis of trade-offs and ideal operational regimes.

Comparative Synthesis:"""
        report = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            options={"num_predict": 1024, "temperature": 0.2},
        )
        return {"query": query, "comparison": report}


class ResearchGapAgent:
    def __init__(self, vector_store: VectorStore, llm: OllamaClient):
        self.vector_store = vector_store
        self.llm = llm

    def run(self, topic: str) -> Dict[str, Any]:
        q = f"limitations bottlenecks open challenges future work {topic}"
        results = self.vector_store.query_similar(query_text=q, n_results=4)
        if not results:
            return {"topic": topic, "analysis": "No indexed documents matched this topic yet."}

        system_prompt = "You are the Research Gap Agent, acting as an NSF Principal Investigator and Scientific Hypothesis Formulator."
        prompt = f"""Read deeply between the lines of the retrieved literature segments, focusing on stated limitations, unaddressed assumptions, and algorithmic bottlenecks.

Context:
{_format_context(results)}

Formulate a rigorous academic gap analysis in this exact format:
### 1. Identified Research Gaps
- **Gap 1 (Theoretical / Structural)**: The primary unaddressed architectural bottleneck or structural assumption.
- **Gap 2 (Empirical / Operational)**: The limitation in evaluation scope, dataset diversity, or deployment scalability.

### 2. Formulated Testable Hypotheses
- **H1 (Empirical Hypothesis)**: "If [proposed modification/technique] is integrated into [architecture/framework], then [quantifiable metric] will improve by [expected effect] under [specific constraints] because [theoretical justification]."
- **H2 (Ablation Hypothesis)**: "Removing or replacing [component] will induce [specific failure mode], proving [theoretical mechanism]."

### 3. Proposed Experimental Validation Protocol
Briefly outline the benchmark dataset, baseline model, and evaluation metric required to empirically test H1.

Research Gaps & Hypotheses:"""
        analysis = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            options={"num_predict": 1024, "temperature": 0.2},
        )
        return {"topic": topic, "analysis": analysis}


class ReviewerAgent:
    def __init__(self, vector_store: VectorStore, llm: OllamaClient):
        self.vector_store = vector_store
        self.llm = llm

    def run(self, proposal_text: str) -> Dict[str, Any]:
        if not proposal_text or not proposal_text.strip():
            return {"proposal": proposal_text, "critique": "No proposal text was provided to review."}

        system_prompt = "You are an elite Senior Area Chair and Peer Reviewer for top-tier AI conferences (NeurIPS / ICLR style)."
        prompt = f"""Evaluate the proposal or methodology below with unsparing intellectual rigor, uncompromising technical precision, and zero superficial pleasantries.

Proposal / Paper Context:
{proposal_text}

Provide an official Reviewer Evaluation report structured as follows:
### 1. Summary of Contributions & Novelty Assessment
What is genuinely novel vs. standard practice or incremental engineering?

### 2. Methodological Risks & Technical Flaws
Probe for missing baseline comparisons, dataset contamination/leakage, lack of ablation, or unverified statistical claims.

### 3. Empirical Soundness & Reproducibility
Evaluate metric validity, variance across random seeds, and compute hardware realism.

### 4. Mandatory Rebuttal Questions for Authors
Formulate 2 pointed, critical technical questions that the authors must answer to justify acceptance.

### 5. Overall Recommendation & Score
State an explicit decision (**Strong Accept** | **Weak Accept** | **Borderline** | **Weak Reject** | **Strong Reject**) accompanied by a concise, authoritative academic rationale.

Reviewer Evaluation:"""
        critique = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            options={"num_predict": 1024, "temperature": 0.2},
        )
        return {"proposal": proposal_text, "critique": critique}



class AICoordinator:
    """Routes a free-text query to the right specialized agent, or runs all 5 in sequence."""

    def __init__(self, vector_store: VectorStore, llm: OllamaClient):
        self.vector_store = vector_store
        self.llm = llm
        self.search_agent = LiteratureSearchAgent(vector_store, llm)
        self.reader_agent = PaperReaderAgent(vector_store, llm)
        self.comparison_agent = ComparisonAgent(vector_store, llm)
        self.gap_agent = ResearchGapAgent(vector_store, llm)
        self.reviewer_agent = ReviewerAgent(vector_store, llm)

    def _default_paper_id(self) -> Optional[str]:
        """Picks the first indexed paper as a default when none is specified."""
        paper_ids = self.vector_store.list_paper_ids()
        return paper_ids[0] if paper_ids else None

    def route_query(self, user_query: str) -> Dict[str, Any]:
        q = user_query.lower()
        if any(w in q for w in ["read", "extract", "problem statement", "methodology", "table", "اقرأ", "استخرج", "لخص", "ملخص"]):
            return {"agent": "Paper Reader Agent", "result": self.reader_agent.run(self._default_paper_id())}
        elif any(w in q for w in ["compare", "comparison", "difference", "vs", "versus", "قارن", "مقارنة", "الفرق"]):
            return {"agent": "Comparison Agent", "result": self.comparison_agent.run(user_query)}
        elif any(w in q for w in ["gap", "hypothesis", "limitations", "future work", "bottleneck", "open challenges", "فجوة", "فجوات", "قصور"]):
            return {"agent": "Research Gap Agent", "result": self.gap_agent.run(user_query)}
        elif any(w in q for w in ["review", "critique", "evaluate", "proposal", "recommendation", "ضعف", "نقد", "تقييم"]):
            return {"agent": "Proposal Reviewer Agent", "result": self.reviewer_agent.run(user_query)}
        else:
            return {"agent": "Literature Search Agent", "result": self.search_agent.run(user_query)}


    def run_collaborative_synthesis_stream(self, paper_id: Optional[str] = None):
        """Yields live progress updates and intermediate agent results as each completes."""
        paper_id = paper_id or self._default_paper_id()
        if not paper_id:
            yield {"stage": 0, "status": "error", "error": "No papers are indexed yet."}
            return

        # 1. Literature Search
        yield {"stage": 1, "status": "running", "agent": "Literature Search Agent"}
        search_res = self.search_agent.run(query=f"key contributions and findings of {paper_id}")
        yield {"stage": 1, "status": "done", "key": "search", "result": search_res}

        # 2. Paper Reader
        yield {"stage": 2, "status": "running", "agent": "Paper Reader Agent"}
        reader_res = self.reader_agent.run(paper_id=paper_id)
        yield {"stage": 2, "status": "done", "key": "reader", "result": reader_res}

        table = reader_res.get("table", {})
        methodology = table.get("Methodology", "the method described in the paper")
        limitations = table.get("Limitations", "the limitations described in the paper")

        # 3. Comparison Agent
        yield {"stage": 3, "status": "running", "agent": "Comparison Agent"}
        comp_res = self.comparison_agent.run(
            query=f"Compare {methodology} against alternative approaches mentioned in the paper"
        )
        yield {"stage": 3, "status": "done", "key": "comparison", "result": comp_res}

        # 4. Research Gap Agent
        yield {"stage": 4, "status": "running", "agent": "Research Gap Agent"}
        gap_res = self.gap_agent.run(topic=f"{limitations} in {paper_id}")
        yield {"stage": 4, "status": "done", "key": "gap", "result": gap_res}

        # 5. Reviewer Agent
        yield {"stage": 5, "status": "running", "agent": "Proposal Reviewer Agent"}
        critique_payload = f"Evaluated Method: {methodology}\nIdentified Bottlenecks & Gaps:\n{gap_res.get('analysis', '')}"
        rev_res = self.reviewer_agent.run(proposal_text=critique_payload)
        yield {"stage": 5, "status": "done", "key": "reviewer", "result": rev_res}

        yield {
            "stage": 6,
            "status": "complete",
            "full_results": {
                "paper_id": paper_id,
                "search": search_res,
                "reader": reader_res,
                "comparison": comp_res,
                "gap": gap_res,
                "reviewer": rev_res,
            },
        }

    def run_collaborative_synthesis(self, paper_id: Optional[str] = None) -> Dict[str, Any]:
        """Orchestrates an end-to-end collaborative pipeline across all 5 specialized agents."""
        final_data = {}
        for event in self.run_collaborative_synthesis_stream(paper_id):
            if event.get("status") == "error":
                return {"error": event.get("error")}
            if event.get("status") == "complete":
                final_data = event.get("full_results", {})
        return final_data

