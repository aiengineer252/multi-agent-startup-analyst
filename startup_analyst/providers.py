"""LLM provider abstraction.

The analyst pipeline needs exactly one operation from a model provider:

    provider.complete(system_prompt, user_prompt) -> Completion

`Completion` carries the text PLUS the telemetry monitoring needs (tokens,
latency, errors). Returning that from the lowest layer means every call site
gets measured automatically — you can never forget to instrument one.

Implementations:
- AnthropicProvider    — Claude via the official SDK. Streaming, adaptive
                         thinking, optional server-side web search.
- OpenAICompatProvider — anything speaking the OpenAI Chat Completions
                         protocol: OpenAI, Groq, Together, OpenRouter,
                         Mistral, DeepSeek, local Ollama, ...
- MockProvider         — deterministic fake reports, no network, no cost.
                         Used to test the pipeline, the evals, and the
                         dashboards without burning a single token.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


class ProviderError(RuntimeError):
    """A configuration problem (missing key, package, or model name).
    The CLI catches this and prints it as a friendly error."""


@dataclass
class Completion:
    """One model call's result plus everything monitoring wants to record."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""
    provider: str = ""
    stop_reason: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# --------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------- #

class LLMProvider(ABC):
    name: str = "base"
    model: str = ""
    supports_web_search: bool = False

    @abstractmethod
    def complete(
        self, system: str, user: str, allow_web_search: bool = False
    ) -> Completion:
        """Run one agent call."""

    @property
    def label(self) -> str:
        return f"{self.name}/{self.model}"


# --------------------------------------------------------------------- #
# Anthropic (Claude) — the default
# --------------------------------------------------------------------- #

class AnthropicProvider(LLMProvider):
    name = "anthropic"
    supports_web_search = True

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

    def complete(
        self, system: str, user: str, allow_web_search: bool = False
    ) -> Completion:
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

        started = time.time()
        in_tok = out_tok = 0

        # Stream to completion; resume if server-side web search pauses the turn.
        while True:
            with self.client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
            in_tok += response.usage.input_tokens
            out_tok += response.usage.output_tokens
            if response.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": response.content})

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        if response.stop_reason == "refusal":
            text = "_This analysis was declined by the model's safety system._"

        return Completion(
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_s=time.time() - started,
            model=self.model,
            provider=self.name,
            stop_reason=response.stop_reason or "",
        )


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

    def complete(
        self, system: str, user: str, allow_web_search: bool = False
    ) -> Completion:
        import openai

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.time()
        # Newer OpenAI models require max_completion_tokens; some compatible
        # servers only know the older max_tokens. Try new, fall back to old.
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
            text = f"_Declined by the provider: {msg.refusal}_"

        usage = resp.usage
        return Completion(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.time() - started,
            model=self.model,
            provider=self.name,
            stop_reason=resp.choices[0].finish_reason or "",
        )


# --------------------------------------------------------------------- #
# Mock — offline testing of the pipeline, evals, and dashboards
# --------------------------------------------------------------------- #

class MockProvider(LLMProvider):
    """Generates deterministic, spec-shaped fake reports. No network, no cost.

    Why this exists: you should be able to run the full pipeline, the eval
    suite, and the drift report in CI without an API key or a bill. It is also
    how you test that your GRADER works — set quality below 1.0 and the mock
    starts omitting sections, so you can prove the evals actually catch it.
    """

    name = "mock"
    supports_web_search = False

    def __init__(self, model: str = "mock-1", quality: float = 1.0, seed: int = 7):
        self.model = model
        self.quality = quality  # 1.0 = fully compliant, 0.0 = garbage
        self.seed = seed

    def complete(
        self, system: str, user: str, allow_web_search: bool = False
    ) -> Completion:
        from .specs import SPECS

        started = time.time()
        # Identify the agent by EXACT-matching its rendered contract, which the
        # orchestrator appends verbatim. Substring-matching a section title
        # would misfire on the synthesizer, whose prompt embeds all five
        # specialist reports.
        spec = next(
            (s for s in SPECS.values() if s.render_contract() in user),
            SPECS["market"],
        )
        rng = random.Random(
            self.seed + int(hashlib.md5(user.encode()).hexdigest()[:8], 16)
        )

        keep = max(1, round(len(spec.sections) * self.quality))
        parts = [f"# {spec.agent_key.title()} Analysis\n"]
        for section in spec.sections[:keep]:
            parts.append(f"## {section.title}\n")
            parts.append(
                f"Estimated at {rng.randint(2, 90)}% with roughly "
                f"{rng.randint(10, 900)}k users and ${rng.randint(5, 500)}M "
                f"in {rng.randint(2026, 2030)}. Assumption: {rng.randint(1, 40)}% "
                f"conversion at ${rng.randint(5, 99)} per month. "
                + "Filler sentence for realistic length. " * 30
            )
        if spec.requires_table:
            parts.append(
                "\n| Item | Value | Note |\n|---|---|---|\n"
                "| A | 12 | ok |\n| B | 34 | ok |\n"
            )
        score = rng.randint(4, 9)
        if spec.verdict_choices:
            parts.append(
                f"\n## Final Verdict\n\n**{rng.choice(spec.verdict_choices)}** — "
                "justification text here."
            )
        else:
            parts.append(
                f"\n**{spec.verdict_label}**: Plausible but unproven. {score}/10"
            )

        text = "\n".join(parts)
        time.sleep(rng.uniform(0.01, 0.05))  # tiny, so latency stats aren't all zero
        return Completion(
            text=text,
            input_tokens=len(user) // 4,
            output_tokens=len(text) // 4,
            latency_s=time.time() - started,
            model=self.model,
            provider=self.name,
            stop_reason="end_turn",
        )


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
    "mock": {
        "default_model": "mock-1",
        "key_env": None,                # no key needed — runs fully offline
        "base_url": None,
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

    if provider_name == "mock":
        # MOCK_QUALITY lets you simulate a degraded model (0.0-1.0) so you can
        # watch the evals fail and the drift detector fire, without waiting for
        # a real regression to happen in production.
        return MockProvider(
            model=model, quality=float(os.environ.get("MOCK_QUALITY", "1.0"))
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
