"""
Module: src/observability/tracer.py
Description: Production observability and tracing integration using Langfuse.
Captures LangChain / LangGraph latency, token counts, prompt variables, and execution graphs.
"""
from typing import Optional, List, Any
import os
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse

# Load environment configuration
load_dotenv()

# Explicit initialization
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

from config.settings import settings


class ObservabilityTracer:
    """Manages Langfuse tracing callbacks and telemetry reporting."""

    def __init__(self, client: Optional[Langfuse] = None):
        self.client = client or langfuse_client
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or settings.LANGFUSE_PUBLIC_KEY
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY") or settings.LANGFUSE_SECRET_KEY
        self.host = os.getenv("LANGFUSE_HOST", os.getenv("LANGFUSE_BASE_URL", settings.LANGFUSE_BASE_URL))
        self._handler = None
        self._is_enabled = bool(self.public_key and self.secret_key)

    def is_enabled(self) -> bool:
        """Returns True if valid Langfuse credentials are configured."""
        return self._is_enabled

    def get_callback_handler(self, trace_name: str = "ResearchMind-Agentic-RAG", user_id: str = "academic-researcher") -> Optional[Any]:
        """
        Instantiates and returns a Langfuse CallbackHandler for LangGraph / LangChain invocations.
        Returns None gracefully if Langfuse credentials are not configured or connection fails.
        """
        if not self._is_enabled:
            return None

        try:
            try:
                from langfuse.langchain import CallbackHandler
            except ImportError:
                from langfuse.callback import CallbackHandler

            try:
                handler = CallbackHandler(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
            except TypeError:
                try:
                    handler = CallbackHandler(public_key=self.public_key)
                except TypeError:
                    handler = CallbackHandler()
            return handler
        except Exception as e:
            print(f"[WARN] Failed to initialize Langfuse CallbackHandler: {e}. Proceeding without active tracing.")
            return None

    def get_trace_url(self, handler: Any) -> Optional[str]:
        """Extracts the web URL for the trace from the callback handler."""
        if handler and hasattr(handler, "get_trace_url"):
            try:
                return handler.get_trace_url()
            except Exception:
                pass
        return None

    def flush(self, handler: Any = None) -> None:
        """Flushes queued trace events to the Langfuse cloud server."""
        if handler and hasattr(handler, "flush"):
            try:
                handler.flush()
            except Exception:
                pass
        elif self.client and hasattr(self.client, "flush"):
            try:
                self.client.flush()
            except Exception:
                pass


# Global observability tracer singleton
tracer = ObservabilityTracer(client=langfuse_client)
get_langfuse_handler = tracer.get_callback_handler

# Expose observe decorator for modular tracing
try:
    from langfuse import observe
except ImportError:
    try:
        from langfuse.decorators import observe
    except ImportError:
        def observe(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
