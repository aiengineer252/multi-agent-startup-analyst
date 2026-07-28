"""LLM provider abstraction.

The analyst pipeline needs exactly one operation from a model provider:

    provider.complete(system_prompt, user_prompt) -> markdown report (str)

Two implementations cover every major API:

- AnthropicProvider  — Claude via the official Anthropic SDK. First-class:
  streaming, adaptive thinking, and optional server-side web search.
- OpenAICompatProvider — anything that speaks the OpenAI Chat Completions
  protocol: OpenAI itself, Groq, Together, OpenRouter, Mistral, DeepSeek,
  local Ollama, etc. Point it at a base_url + API key and it works.

Presets for common providers live in PROVIDER_PRESETS; `create_provider()`
is the factory the CLI uses.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class ProviderError(RuntimeError):
    """A configuration problem (missing key, package, or model name).
    The CLI catches this and prints it as a friendly error."""


# --------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------- #

class LLMProvider(ABC):
    name: str = "base"
    model: str = ""
    supports_web_search: bool = False

    @abstractmethod
    def complete(self, system: str, user: str, allow_web_search: bool = False) -> str:
        """Run one agent call and return the response text (Markdown)."""

    @property
    def label(self) -> str:
        return f"{self.name}/{self.model}"


# --------------------------------------------------------------------- #
# Anthropic (Claude) — the default
# --------------------------------------------------------------------- #

class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_web_search = True

    # Server-side tool: runs on Anthropic's infra, nothing to implement here.
    # Capped so one run can't rack up unbounded search charges.
    WEB_SEARCH_TOOL = {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 3,
    }

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 16000,
        use_web_search: bool = False,
    ):
        from anthropic import Anthropic

        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model
        self.max_tokens = max_tokens
        self.use_web_search = use_web_search

    def _thinking_config(self) -> Optional[dict]:
        """Adaptive thinking where the model supports it.

        - Fable/Mythos: thinking is always on; the param must be omitted.
        - Haiku 4.5 / Sonnet 4.5 and older: no adaptive mode; omit.
        - Opus 4.6+ / Sonnet 4.6+ / Sonnet 5: adaptive, explicitly set.
        """
        m = self.model
        if any(s in m for s in ("fable", "mythos", "haiku", "sonnet-4-5")):
            return None
        return {"type": "adaptive"}

    def complete(self, system: str, user: str, allow_web_search: bool = False) -> str:
        messages = [{"role": "user", "content": user}]
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        thinking = self._thinking_config()
        if thinking:
            kwargs["thinking"] = thinking
        if self.use_web_search and allow_web_search:
            kwargs["tools"] = [self.WEB_SEARCH_TOOL]

        # Stream to completion; resume if server-side web search pauses the turn.
        while True:
            with self.client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
            if response.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            return "_This analysis was declined by the model's safety system._"

        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()


# --------------------------------------------------------------------- #
# OpenAI-compatible (OpenAI, Groq, Ollama, OpenRouter, Together, ...)
# --------------------------------------------------------------------- #

class OpenAICompatProvider(LLMProvider):
    supports_web_search = False

    def __init__(
        self,
        name: str,
        model: str,
        api_key_env: str,
        base_url: Optional[str] = None,
        max_tokens: int = 16000,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "The 'openai' package is required for this provider.\n"
                "Install it with:  pip install openai"
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderError(
                f"{api_key_env} is not set. Add it to your .env file "
                f"(see .env.example) or export it in your shell."
            )

        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str, allow_web_search: bool = False) -> str:
        import openai

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Newer OpenAI models (gpt-5.x, o-series) require max_completion_tokens;
        # some compatible servers only know the older max_tokens. Try new, fall
        # back to old.
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_tokens,
            )
        except openai.BadRequestError:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )

        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        if not text and getattr(msg, "refusal", None):
            return f"_Declined by the provider: {msg.refusal}_"
        return text


# --------------------------------------------------------------------- #
# Presets & factory
# --------------------------------------------------------------------- #

PROVIDER_PRESETS = {
    "anthropic": {
        "default_model": "claude-opus-4-8",
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    "openai": {
        "default_model": "gpt-5.1",
        "key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "groq": {
        "default_model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "custom": {
        # Any OpenAI-compatible endpoint: Ollama, OpenRouter, Together,
        # Mistral, DeepSeek, vLLM, LM Studio, ...
        "default_model": None,          # must be supplied with --model
        "key_env": "CUSTOM_API_KEY",    # override with --api-key-env
        "base_url": None,               # supply with --base-url or CUSTOM_BASE_URL
    },
}


def create_provider(
    provider_name: str,
    model: Optional[str] = None,
    use_web_search: bool = False,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    max_tokens: int = 16000,
) -> LLMProvider:
    """Build a provider from a preset name plus optional overrides."""
    preset = PROVIDER_PRESETS.get(provider_name)
    if preset is None:
        raise ProviderError(
            f"Unknown provider '{provider_name}'. "
            f"Choose from: {', '.join(PROVIDER_PRESETS)}"
        )

    model = model or preset["default_model"]
    if not model:
        raise ProviderError(
            f"Provider '{provider_name}' has no default model — "
            "pass one with --model (e.g. --model llama3.1:8b)."
        )

    if provider_name == "anthropic":
        return AnthropicProvider(
            model=model, max_tokens=max_tokens, use_web_search=use_web_search
        )

    resolved_base_url = base_url or preset["base_url"]
    if provider_name == "custom" and not resolved_base_url:
        resolved_base_url = os.environ.get("CUSTOM_BASE_URL")
        if not resolved_base_url:
            raise ProviderError(
                "The 'custom' provider needs an endpoint. Pass --base-url "
                "(e.g. --base-url http://localhost:11434/v1 for Ollama) "
                "or set CUSTOM_BASE_URL in your .env."
            )

    return OpenAICompatProvider(
        name=provider_name,
        model=model,
        api_key_env=api_key_env or preset["key_env"],
        base_url=resolved_base_url,
        max_tokens=max_tokens,
    )
