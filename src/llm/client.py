"""
Module: client.py
Description: Multi-Backend LLM Client supporting:
1. Google Gemini Cloud (Recommended: 100% Free, 1M context, 1s response, zero laptop load)
2. Groq Cloud (Sub-second speed, 70B open weights)
3. Local Ollama (100% offline fallback)
"""
import requests
import json
import os
from typing import Optional, Dict, Any, List

import config

OLLAMA_BASE_URL = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class OllamaClient:
    """
    Unified LLM Client capable of executing against:
    - Google Gemini Cloud API
    - Groq Cloud API
    - Local Ollama runtime
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        self.base_url = base_url
        self.backend = backend or getattr(config, "DEFAULT_LLM_BACKEND", "gemini")

        # Gemini settings
        self.gemini_api_key = (
            gemini_api_key
            or getattr(config, "GEMINI_API_KEY", "")
            or os.getenv("GEMINI_API_KEY", "")
        ).strip().strip('"').strip("'")
        self.gemini_model_name = getattr(config, "GEMINI_MODEL_NAME", "gemini-flash-lite-latest")

        # Groq settings
        self.groq_api_key = (
            groq_api_key
            or getattr(config, "GROQ_API_KEY", "")
            or os.getenv("GROQ_API_KEY", "")
        ).strip().strip('"').strip("'")
        self.groq_model_name = getattr(config, "GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

        # Local Ollama settings
        self.model_name = (
            model_name
            or getattr(config, "LLM_MODEL_NAME", None)
            or "llama3.1"
        )
        self._verify_and_fallback_model()

    def set_backend(
        self,
        backend: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Dynamically switches active backend ('gemini', 'groq', or 'ollama')."""
        self.backend = backend.lower()
        if "groq_api_key" in kwargs and kwargs["groq_api_key"]:
            self.groq_api_key = str(kwargs["groq_api_key"]).strip().strip('"').strip("'")
        if "gemini_api_key" in kwargs and kwargs["gemini_api_key"]:
            self.gemini_api_key = str(kwargs["gemini_api_key"]).strip().strip('"').strip("'")

        if api_key is not None:
            clean_key = api_key.strip().strip('"').strip("'")
            if self.backend == "gemini":
                self.gemini_api_key = clean_key
            elif self.backend == "groq":
                self.groq_api_key = clean_key

        if model:
            if self.backend == "gemini":
                self.gemini_model_name = model
            elif self.backend == "groq":
                self.groq_model_name = model
            else:
                self.model_name = model

    def _verify_and_fallback_model(self) -> None:
        """Verifies local Ollama model if running in local mode."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                if models and self.model_name not in models:
                    self.model_name = models[0]
        except Exception:
            pass

    def list_available_models(self) -> List[str]:
        """Returns the supported models for the active backend."""
        if self.backend == "gemini":
            return ["gemini-flash-lite-latest", "gemini-3.7-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
        elif self.backend == "groq":
            return [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ]
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", []) if m.get("name")]
                if models:
                    return models
        except Exception:
            pass
        return [self.model_name]

    def _get_default_options(self, custom_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merges custom generation options with speed-optimized defaults for CPU."""
        default_ctx = getattr(config, "LLM_NUM_CTX", 2048)
        opts: Dict[str, Any] = {
            "num_predict": 180,
            "temperature": 0.2,
            "num_ctx": default_ctx,
        }
        if custom_options:
            opts.update(custom_options)
        return opts

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        response_format: Optional[str] = None,
    ) -> str:
        # Route 1: Google Gemini Cloud (Ultra-Fast 1-2s Cloud Inference)
        if self.backend == "gemini":
            if not self.gemini_api_key:
                return "[ERROR] Gemini API key is missing. Set GEMINI_API_KEY in .env."
            return self._generate_gemini_response(prompt, system_prompt, options, response_format)

        # Route 2: Groq Cloud
        elif self.backend == "groq":
            if not self.groq_api_key:
                return "[ERROR] Groq API key is missing. Set GROQ_API_KEY in .env."
            return self._generate_groq_response(prompt, system_prompt, options, response_format)

        # Route 3: Local Ollama (Only executed when explicitly selected)
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": self._get_default_options(options),
        }
        if system_prompt:
            payload["system"] = system_prompt
        if response_format:
            payload["format"] = response_format

        try:
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            return f"[ERROR] Failed to communicate with Ollama backend ({self.model_name}): {e}."

    def _generate_gemini_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """Executes sub-second cloud inference via Google Gemini REST API with resilient auto-failover."""
        candidate_models = [
            self.gemini_model_name,
            "gemini-flash-lite-latest",
            "gemini-3.7-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
        ]
        seen = set()
        models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": options.get("temperature", 0.2) if options else 0.2,
                "maxOutputTokens": options.get("num_predict", 1024) if options else 1024,
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
        if response_format == "json":
            payload["generationConfig"]["responseMimeType"] = "application/json"

        last_error = ""
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            self.gemini_model_name = model
                            return parts[0].get("text", "").strip()
                else:
                    last_error = f"{response.status_code}: {response.text[:120]}"
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                continue

        return f"[ERROR] Gemini Cloud error across models: {last_error}"

    def _generate_groq_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        response_format: Optional[str] = None,
    ) -> str:
        """Executes cloud inference via Groq API."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.groq_model_name,
            "messages": messages,
            "temperature": options.get("temperature", 0.2) if options else 0.2,
            "max_tokens": 1024,
        }
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""
        except requests.exceptions.RequestException as e:
            return f"[ERROR] Groq API error: {e}. Check your GROQ_API_KEY in the sidebar."

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Yields streaming token responses in real-time."""
        if self.backend == "gemini" and self.gemini_api_key:
            yield from self._generate_gemini_stream(prompt, system_prompt, options)
            return
        elif self.backend == "groq" and self.groq_api_key:
            yield from self._generate_groq_stream(prompt, system_prompt, options)
            return

        # Local Ollama stream
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": self._get_default_options(options),
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with requests.post(url, json=payload, stream=True, timeout=600) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        yield chunk.get("response", "")
        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Ollama stream failed: {e}"

    def _generate_gemini_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Streams responses from Google Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_name}:streamGenerateContent?key={self.gemini_api_key}&alt=sse"
        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": options.get("temperature", 0.2) if options else 0.2,
                "maxOutputTokens": options.get("num_predict", 1024) if options else 1024,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data: "):
                            chunk_data = json.loads(decoded[6:])
                            candidates = chunk_data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    text = p.get("text", "")
                                    if text:
                                        yield text
        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Gemini streaming failed: {e}"

    def _generate_groq_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """Streams responses from Groq API."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.groq_model_name,
            "messages": messages,
            "temperature": options.get("temperature", 0.2) if options else 0.2,
            "max_tokens": 1024,
            "stream": True,
        }

        try:
            with requests.post(GROQ_API_URL, headers=headers, json=payload, stream=True, timeout=30) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded = line.decode("utf-8").strip()
                        if decoded.startswith("data: ") and not decoded.endswith("[DONE]"):
                            chunk_json = json.loads(decoded[6:])
                            delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
        except requests.exceptions.RequestException as e:
            yield f"[ERROR] Groq streaming failed: {e}"


# Backward compatibility alias
LLMClient = OllamaClient