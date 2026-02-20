"""AWS Bedrock LLM client for MDP Interface Discovery.

Wraps the Bedrock Converse API to generate MDP interface code.
Takes prompt dicts from PromptBuilder, calls the model, extracts code.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from mdp_discovery.config import LLMConfig
from mdp_discovery.prompts import extract_code

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Bedrock model configuration."""
    model_id: str
    max_tokens: int = 4096
    supports_system: bool = True


# Short aliases → Bedrock model IDs / inference profile ARNs.
# Users can also pass a raw model ID or ARN directly.
MODELS: Dict[str, ModelInfo] = {
    # Claude
    "claude-3-haiku": ModelInfo("anthropic.claude-3-haiku-20240307-v1:0", 4096),
    "claude-3.5-sonnet": ModelInfo("anthropic.claude-3-5-sonnet-20241022-v2:0", 8192),
    "claude-4-sonnet": ModelInfo("us.anthropic.claude-sonnet-4-20250514-v1:0", 8192),
    "claude-sonnet-4-6": ModelInfo("us.anthropic.claude-sonnet-4-6", 8192),
    "claude-4-opus": ModelInfo("us.anthropic.claude-opus-4-20250514-v1:0", 8192),
    # Llama
    "llama3.1-8b": ModelInfo("us.meta.llama3-1-8b-instruct-v1:0", 2048),
    "llama3.1-70b": ModelInfo("us.meta.llama3-1-70b-instruct-v1:0", 2048),
    "llama3.3-70b": ModelInfo("us.meta.llama3-3-70b-instruct-v1:0", 2048),
    "llama4-scout": ModelInfo("us.meta.llama4-scout-17b-instruct-v1:0", 2048),
    "llama4-maverick": ModelInfo("us.meta.llama4-maverick-17b-instruct-v1:0", 2048),
    # Mistral
    "mistral-large": ModelInfo("mistral.mistral-large-2402-v1:0", 4096),
}


def _resolve_model_id(name: str) -> ModelInfo:
    """Resolve a short model name or raw ARN/ID to a ModelInfo."""
    if name in MODELS:
        return MODELS[name]
    # Treat as raw Bedrock model ID or ARN
    return ModelInfo(model_id=name)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Result from a single LLM call."""
    text: str
    code: Optional[str]
    input_tokens: int
    output_tokens: int
    model: str


class LLMClient:
    """Bedrock-backed LLM client for generating MDP interface code.

    Usage:
        client = LLMClient(config)
        prompt = prompt_builder.build_prompt(...)
        response = client.generate(prompt)
        if response.code:
            # response.code contains the extracted Python
            ...
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._local = threading.local()
        self._model_info = _resolve_model_id(config.model_name)
        logger.info(
            "LLMClient initialized: model=%s region=%s",
            config.model_name,
            config.region_name,
        )

    @property
    def client(self):
        """Return a thread-local boto3 Bedrock client (created lazily)."""
        c = getattr(self._local, "client", None)
        if c is None:
            c = boto3.client(
                "bedrock-runtime",
                region_name=self.config.region_name,
            )
            self._local.client = c
        return c

    def generate(self, prompt: Dict[str, str]) -> LLMResponse:
        """Call the LLM with a prompt dict from PromptBuilder.

        Args:
            prompt: {"system": str, "user": str} as returned by
                    PromptBuilder.build_prompt().

        Returns:
            LLMResponse with the raw text, extracted code (if any),
            and token usage.
        """
        system_msg = prompt["system"]
        user_msg = prompt["user"]

        max_tokens = min(self.config.max_tokens, self._model_info.max_tokens)

        converse_params: Dict[str, Any] = {
            "modelId": self._model_info.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": user_msg}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": self.config.temperature,
            },
        }

        if self._model_info.supports_system and system_msg:
            converse_params["system"] = [{"text": system_msg}]

        # Retry loop with exponential backoff for throttling
        last_error: Optional[Exception] = None
        delay = self.config.retry_delay

        for attempt in range(1, self.config.retries + 1):
            try:
                response = self.client.converse(**converse_params)

                text = response["output"]["message"]["content"][0]["text"]
                usage = response.get("usage", {})
                code = extract_code(text)

                return LLMResponse(
                    text=text,
                    code=code,
                    input_tokens=usage.get("inputTokens", 0),
                    output_tokens=usage.get("outputTokens", 0),
                    model=self.config.model_name,
                )

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                last_error = e

                if error_code == "ThrottlingException" and attempt < self.config.retries:
                    logger.warning(
                        "Throttled (attempt %d/%d), retrying in %.1fs...",
                        attempt,
                        self.config.retries,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue

                logger.error("Bedrock API error: %s — %s", error_code, e)
                raise

            except Exception as e:
                last_error = e
                if attempt < self.config.retries:
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt,
                        self.config.retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise

        raise RuntimeError(
            f"LLM call failed after {self.config.retries} attempts: {last_error}"
        )

    def generate_code(self, prompt: Dict[str, str]) -> Optional[str]:
        """Convenience method: generate and return just the extracted code.

        Returns None if the LLM response doesn't contain a Python fence.
        """
        response = self.generate(prompt)
        return response.code


class LLMEnsemble:
    """Weighted ensemble of LLM clients with random model selection.

    When config.models is empty, behaves as a single-model wrapper around
    LLMClient (full backward compatibility). When populated, selects a model
    per generate() call using weighted random sampling.

    Usage:
        ensemble = LLMEnsemble(config)
        response = ensemble.generate(prompt)  # same interface as LLMClient
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        if not config.models:
            # Single model mode — backward compatible
            self._clients = [LLMClient(config)]
            self._weights = [1.0]
            self._names = [config.model_name]
        else:
            self._clients = []
            self._weights = []
            self._names = []

            for model_cfg in config.models:
                # Create per-model LLMConfig inheriting defaults
                model_config = LLMConfig(
                    region_name=model_cfg.region_name or config.region_name,
                    model_name=model_cfg.name,
                    temperature=(
                        model_cfg.temperature
                        if model_cfg.temperature is not None
                        else config.temperature
                    ),
                    max_tokens=(
                        model_cfg.max_tokens
                        if model_cfg.max_tokens is not None
                        else config.max_tokens
                    ),
                    retries=config.retries,
                    retry_delay=config.retry_delay,
                )
                self._clients.append(LLMClient(model_config))
                self._weights.append(model_cfg.weight)
                self._names.append(model_cfg.name)

            # Normalize weights
            total = sum(self._weights)
            if total > 0:
                self._weights = [w / total for w in self._weights]

        logger.info(
            "LLMEnsemble initialized: %s",
            ", ".join(
                f"{n} (w={w:.2f})" for n, w in zip(self._names, self._weights)
            ),
        )

    def generate(self, prompt: Dict[str, str]) -> LLMResponse:
        """Generate using a weighted-random selected model."""
        selected = random.choices(self._clients, weights=self._weights, k=1)[0]
        return selected.generate(prompt)
