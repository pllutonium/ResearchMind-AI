"""
Script: ask.py
Description: Interactive terminal interface for querying the research corpus via RAG.
"""
import os
import sys
import logging

# Suppress telemetry prior to imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

from src.agents.rag_agent import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline(top_k=4)

    print("=" * 60)
    print("ResearchMind-AI: Interactive Document Query Interface")
    print("Type 'exit' or 'quit' to terminate the session.")
    print("=" * 60)

    while True:
        try:
            query_input = input("\nEnter research question: ").strip()
            if not query_input:
                continue

            if query_input.lower() in ("exit", "quit"):
                print("\nTerminating query session.")
                break

            print("\n[INFO] Retrieving relevant passages and synthesizing response...")
            output = pipeline.query(query_input)

            print("\n" + "-" * 50)
            print("Answer:")
            print(output["response"])
            print("-" * 50)

            print("Referenced Sources:")
            for src in output["sources"]:
                print(f" - Document: {src['paper_id']} | Page: {src['page_number']}")

        except KeyboardInterrupt:
            print("\nTerminating query session.")
            sys.exit(0)


if __name__ == "__main__":
    main()