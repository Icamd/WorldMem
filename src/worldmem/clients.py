"""Shared LLM and embedding interfaces, with optional model dependencies loaded lazily."""

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np


@dataclass
class LLMResponse:
    content: str
    usage: dict
    model: str
    finish_reason: str = "stop"
    response_time: float = 0.0


def parse_json(text):
    """Read objects or arrays, ignoring an optional reasoning prefix and code fences."""
    text = re.sub("<think>.*?</think>", "", text or "", flags=re.S).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    candidates = re.findall("```(?:json)?\\s*(.*?)```", text, flags=re.S) + [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, (dict, list)):
                return value
        except ValueError:
            pass
        for match in re.finditer("[\\[{]", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
                if isinstance(value, (dict, list)):
                    return value
            except ValueError:
                pass
    raise ValueError("Model response contains no valid JSON object or array")


class LLMClient:
    """Chat Completions for vLLM/GPT, or in-process Hugging Face inference."""

    def __init__(
        self,
        *,
        model="Qwen3-8B",
        backend="vllm",
        base_url=None,
        api_key=None,
        enable_thinking=False,
        timeout=600.0,
        max_retries=3,
        reasoning_effort="minimal",
        min_completion_tokens=4096,
        device="auto",
        dtype="bfloat16",
    ):
        self.model_name = self.model = model
        self.backend = backend
        self.base_url = base_url
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort
        self.min_completion_tokens = min_completion_tokens
        self.max_retries = max_retries
        self._policy = threading.local()
        self._lock = threading.RLock()
        self.tokenizer = None
        if backend == "hf":
            from .hf import HuggingFaceBackend

            self._hf = HuggingFaceBackend(model, device=device, dtype=dtype)
            self.tokenizer = self._hf.tokenizer
        elif backend in {"vllm", "openai_api"}:
            from openai import OpenAI

            key = api_key or os.environ.get(
                "OPENAI_API_KEY" if backend == "openai_api" else "VLLM_API_KEY"
            )
            if backend == "openai_api" and (not key):
                raise ValueError("Set OPENAI_API_KEY before using the external API backend")
            self.client = OpenAI(
                api_key=key or "EMPTY",
                base_url=base_url,
                timeout=timeout,
                max_retries=max(0, max_retries - 1),
            )
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

    @contextmanager
    def api_call_scope(self, *, reasoning_effort=None, min_completion_tokens=None):
        old = getattr(self._policy, "value", {})
        self._policy.value = {
            **old,
            **{
                k: v
                for k, v in dict(
                    reasoning_effort=reasoning_effort, min_completion_tokens=min_completion_tokens
                ).items()
                if v is not None
            },
        }
        try:
            yield
        finally:
            self._policy.value = old

    def chat_completion(
        self,
        messages,
        temperature=0.0,
        max_tokens=4096,
        enable_thinking=None,
        top_p=None,
        top_k=None,
        min_p=None,
        api_reasoning_effort=None,
        api_min_completion_tokens=None,
        **kwargs,
    ):
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        if self.backend == "hf":
            with self._lock:
                return self._hf.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=thinking,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
        start = time.monotonic()
        request = dict(model=self.model_name, messages=messages)
        if self.backend == "openai_api":
            policy = getattr(self._policy, "value", {})
            effort = api_reasoning_effort or policy.get("reasoning_effort", self.reasoning_effort)
            minimum = api_min_completion_tokens or policy.get(
                "min_completion_tokens", self.min_completion_tokens
            )
            request.update(
                max_completion_tokens=max(max_tokens or 4096, minimum, self.min_completion_tokens),
                reasoning_effort=effort,
            )
        else:
            extra = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
            if top_k is not None:
                extra["top_k"] = top_k
            if min_p is not None and min_p > 0:
                extra["min_p"] = min_p
            request.update(temperature=temperature, max_tokens=max_tokens, extra_body=extra)
            if top_p is not None:
                request["top_p"] = top_p
        result = self.client.chat.completions.create(**request)
        choice = result.choices[0]
        raw = result.usage.model_dump() if result.usage else {}
        completion = raw.get("completion_tokens", 0) or 0
        prompt = raw.get("prompt_tokens", 0) or 0
        reasoning = (raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        usage = dict(
            raw,
            prompt_tokens_est=prompt,
            completion_tokens_est=completion,
            total_tokens_est=raw.get("total_tokens", prompt + completion),
            thinking_tokens_est=reasoning,
            visible_completion_tokens_est=max(0, completion - reasoning),
        )
        content = choice.message.content or ""
        if not content.strip():
            raise RuntimeError(
                f"Empty model answer (finish_reason={choice.finish_reason}); increase the completion budget"
            )
        content = re.sub("<think>.*?</think>", "", content, flags=re.S).strip()
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[-1].strip()
        return LLMResponse(
            content,
            usage,
            self.model_name,
            choice.finish_reason or "stop",
            time.monotonic() - start,
        )

    def generate_text_response_with_usage(self, prompt, system_prompt=None, **kwargs):
        messages = [dict(role="system", content=system_prompt)] if system_prompt else []
        messages.append(dict(role="user", content=prompt))
        return self.chat_completion(messages, **kwargs)

    def generate_text_response(self, prompt, **kwargs):
        return self.generate_text_response_with_usage(prompt, **kwargs).content

    def generate_response(self, prompt, **kwargs):
        return self.generate_text_response(prompt, **kwargs)

    def generate_json_response_with_usage(
        self, prompt, default_response=None, max_retries=None, **kwargs
    ):
        last = None
        for _ in range(max_retries or self.max_retries):
            last = self.generate_text_response_with_usage(prompt, **kwargs)
            try:
                parsed = parse_json(last.content)
                if default_response is not None and (
                    not isinstance(parsed, type(default_response))
                ):
                    raise ValueError("JSON response has an unexpected top-level type")
                return (parsed, last)
            except ValueError:
                continue
        raise ValueError("Failed to parse a structured model response after retries")

    def generate_json_response(self, prompt, **kwargs):
        return self.generate_json_response_with_usage(prompt, **kwargs)[0]

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text)) if self.tokenizer else max(1, len(text) // 4)

    def get_model_info(self):
        return dict(
            backend=self.backend,
            model=self.model_name,
            default_enable_thinking=self.enable_thinking,
            token_count="exact" if self.tokenizer else "estimated",
        )

    def close(self):
        if hasattr(self, "client"):
            self.client.close()


class EmbeddingClient:
    def __init__(
        self,
        *,
        model=None,
        local_model_path=None,
        local_device="cpu",
        backend="local",
        base_url=None,
    ):
        self.model = model or local_model_path or "sentence-transformers/all-MiniLM-L6-v2"
        self.backend = backend
        self._lock = threading.RLock()
        if backend == "local":
            from sentence_transformers import SentenceTransformer

            self.local_model = SentenceTransformer(self.model, device=local_device)
        elif backend == "api":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=os.environ.get("EMBEDDING_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or "EMPTY",
                base_url=base_url,
            )
        elif backend != "hash":
            raise ValueError(f"Unsupported embedding backend: {backend}")

    def embed_text(self, text):
        if self.backend == "hash":
            vector = np.zeros(128, dtype=float)
            for token in re.findall("\\w+", str(text).lower()):
                digest = hashlib.sha256(token.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % len(vector)] += 1
            norm = np.linalg.norm(vector)
            return (vector / norm if norm else vector).tolist()
        if self.backend == "api":
            return self.client.embeddings.create(model=self.model, input=[text]).data[0].embedding
        with self._lock:
            return self.local_model.encode(str(text), show_progress_bar=False).tolist()

    @staticmethod
    def cosine_similarity(left, right):
        a, b = (np.asarray(left), np.asarray(right))
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denominator) if denominator else 0.0

    def close(self):
        if hasattr(self, "client"):
            self.client.close()
