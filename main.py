"""
Module: main.py
Description: Production CLI entrypoint for ResearchMind AI Agentic RAG system.
Supports PDF ingestion into hierarchical docstore/ChromaDB and live interactive query execution.
"""
from typing import Optional
from pathlib import Path
import argparse
import sys
import os

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from config.settings import settings
from src.ingestion.parser import pdf_parser
from src.ingestion.chunking import hierarchical_chunker
from src.retrieval.hybrid_retriever import hybrid_retriever
from src.ingestion.docstore import docstore
from src.agents.graph import run_agentic_rag

console = Console(highlight=False)


def ingest_pdf(file_path: str) -> None:
    """
    Ingests an academic PDF into the hierarchical docstore and hybrid retrieval indexes.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}")
        sys.exit(1)

    console.print(f"\n[bold cyan]1. Parsing Academic PDF:[/bold cyan] [white]{path.name}[/white]")
    with console.status("[bold green]Extracting document pages and structures...[/bold green]"):
        paper = pdf_parser.parse(path)

    console.print(f"[bold green][OK][/bold green] Parsed [bold yellow]{paper.total_pages}[/bold yellow] pages. Title: [italic]{paper.title}[/italic]")

    console.print(f"\n[bold cyan]2. Hierarchical Parent-Child Chunking:[/bold cyan]")
    with console.status("[bold green]Building parent (800 tokens) and child (200 tokens) windows...[/bold green]"):
        parent_docs, child_chunks = hierarchical_chunker.chunk_paper(paper)

    console.print(
        f"[bold green][OK][/bold green] Generated [bold green]{len(parent_docs)}[/bold green] Parent Documents (persisted in DocStore) and "
        f"[bold green]{len(child_chunks)}[/bold green] Child Chunks."
    )

    console.print(f"\n[bold cyan]3. Indexing into Vector & Sparse Stores:[/bold cyan]")
    with console.status("[bold green]Embedding child vectors in ChromaDB and building BM25 index...[/bold green]"):
        indexed_count = hybrid_retriever.index_chunks(child_chunks)

    console.print(f"[bold green][OK][/bold green] Indexed [bold green]{indexed_count}[/bold green] chunks successfully into ChromaDB & BM25.")
    console.print(
        Panel(
            f"[bold green]Ingestion Complete for:[/bold green] {paper.title}\n"
            f"[bold]Total Chunks in Vector Store:[/bold] {hybrid_retriever.collection.count()}\n"
            f"[bold]Total Parent Documents in Store:[/bold] {docstore.count()}",
            title="Ingestion Summary",
            border_style="green",
        )
    )


def execute_query(query: str, user_id: str = "researcher") -> None:
    """
    Runs a research query through the full Corrective RAG (CRAG) state graph.
    """
    console.print(f"\n[bold cyan]Research Inquiry:[/bold cyan] [bold white]{query}[/bold white]\n")

    with console.status("[bold cyan]Executing Agentic Corrective RAG Workflow...[/bold cyan]"):
        state = run_agentic_rag(query=query, user_id=user_id)

    # 1. Retrieval & Grading Information
    retrieval_grade = state.get("retrieval_grade", "UNKNOWN")
    grade_color = "green" if retrieval_grade == "RELEVANT" else "yellow"
    
    table = Table(title="CRAG Execution Trace", border_style="cyan")
    table.add_column("Pipeline Stage", style="bold white")
    table.add_column("Agent Decision / Output", style="white")

    table.add_row(
        "Local Retrieval",
        f"Retrieved {len(state.get('retrieved_docs', []))} Parent Documents via Hybrid RRF + Cross-Encoder",
    )
    table.add_row(
        "Retrieval Evaluator",
        f"[{grade_color}]{retrieval_grade}[/{grade_color}] - {state.get('retrieval_reasoning', '')}",
    )

    arxiv_docs = state.get("arxiv_docs", [])
    if arxiv_docs:
        table.add_row(
            "arXiv Fallback",
            f"[bold yellow]Triggered:[/bold yellow] Fetched {len(arxiv_docs)} external research abstracts",
        )

    hallucination_grade = state.get("hallucination_grade", "GROUNDED")
    h_color = "green" if hallucination_grade == "GROUNDED" else "red"
    table.add_row(
        "Hallucination Grader",
        f"[{h_color}]{hallucination_grade}[/{h_color}] (Retries: {state.get('hallucination_retries', 0)})",
    )

    if state.get("trace_url"):
        table.add_row("Langfuse Telemetry", f"[link={state['trace_url']}]{state['trace_url']}[/link]")

    console.print(table)
    console.print()

    # 2. Render Final Academic Synthesis
    final_text = state.get("final_response") or state.get("candidate_answer") or "No response generated."
    console.print(
        Panel(
            Markdown(final_text),
            title="[bold green]Authoritative Academic Synthesis[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # 3. Render Sources & Citations
    docs = state.get("retrieved_docs", []) + state.get("arxiv_docs", [])
    if docs:
        console.print("\n[bold cyan]Referenced Academic Contexts & Sources:[/bold cyan]")
        for idx, doc in enumerate(docs[:4], 1):
            title = doc.get("title", "Document")
            page = doc.get("page_number", 1)
            source = doc.get("metadata", {}).get("source") or f"Page {page}"
            console.print(f"  [bold cyan][{idx}][/bold cyan] [bold white]{title}[/bold white] - [yellow]{source}[/yellow]")
            excerpt = doc.get("text", "")[:180].replace("\n", " ") + "..."
            console.print(f"      [dim]{excerpt}[/dim]")
    console.print()


def interactive_mode():
    """Runs continuous interactive research session in terminal."""
    console.print(
        Panel(
            "[bold white]Welcome to ResearchMind AI Interactive Academic Terminal[/bold white]\n"
            "Ask research inquiries across your ingested literature. Type [bold red]'exit'[/bold red] to quit.",
            border_style="cyan",
        )
    )
    while True:
        try:
            query = console.input("\n[bold cyan]ResearchMind > [/bold cyan]").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Exiting interactive session. Goodbye![/yellow]")
                break
            execute_query(query)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session interrupted.[/yellow]")
            break


def main():
    parser = argparse.ArgumentParser(
        description="ResearchMind AI — Enterprise Agentic Corrective RAG System"
    )
    parser.add_argument("--ingest", type=str, help="Path to academic PDF file to ingest")
    parser.add_argument("--query", type=str, help="Research question to synthesize")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive academic shell")

    args = parser.parse_args()

    if args.ingest:
        ingest_pdf(args.ingest)
    elif args.query:
        execute_query(args.query)
    elif args.interactive:
        interactive_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
