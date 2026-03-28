"""Azure OpenAI-based signal decision agent with rule-based fallback."""

import asyncio
import base64
import logging
import os
import re
import time
from functools import lru_cache
from typing import Any, Tuple, cast

import cv2
import instructor
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

from agent.models import IntersectionState, SignalDecision
from agent.prompts import INDIA_SYSTEM_PROMPT
from agent.rule_based import rule_based_decision
from agent.state_builder import build_state_prompt

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

_LLM_COOLDOWN_UNTIL = 0.0
_LAST_COOLDOWN_REASON = ""


def _is_azure_configured() -> bool:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    return bool(endpoint and api_key)


def _normalize_azure_endpoint(raw_endpoint: str) -> str:
    endpoint = (raw_endpoint or "").strip()
    if not endpoint:
        return endpoint
    endpoint = endpoint.rstrip("/")

    # Accept either resource endpoint or a copied full deployments URL.
    marker = "/openai/deployments/"
    idx = endpoint.lower().find(marker)
    if idx >= 0:
        endpoint = endpoint[:idx]

    return endpoint


def _extract_retry_seconds(text: str) -> int | None:
    match = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return max(1, int(float(match.group(1))))
    except ValueError:
        return None


def _compute_cooldown_seconds(exc: Exception) -> int:
    msg = str(exc).lower()
    parsed = _extract_retry_seconds(str(exc))
    if parsed is not None:
        return parsed
    if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate limit" in msg:
        return 30
    if "10013" in msg or "11001" in msg or "getaddrinfo" in msg or "timed out" in msg:
        return 15
    return 10


def _set_llm_cooldown(exc: Exception) -> None:
    global _LLM_COOLDOWN_UNTIL
    global _LAST_COOLDOWN_REASON

    cooldown_s = _compute_cooldown_seconds(exc)
    _LLM_COOLDOWN_UNTIL = time.monotonic() + cooldown_s
    _LAST_COOLDOWN_REASON = str(exc)


def _cooldown_remaining_s() -> int:
    return max(0, int(_LLM_COOLDOWN_UNTIL - time.monotonic()))


def _encode_frame_to_data_url(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@lru_cache(maxsize=4)
def _client(model_name: str = DEFAULT_MODEL):
    if not _is_azure_configured():
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
        )

    endpoint = _normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION).strip() or DEFAULT_API_VERSION

    azure_client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
    return instructor.from_openai(azure_client)


async def _decide_with_client(state: IntersectionState, frame: np.ndarray, model_name: str) -> SignalDecision:
    prompt = build_state_prompt(state)
    image_data_url = _encode_frame_to_data_url(frame)
    client = _client(model_name)
    messages_mm = [
        {"role": "system", "content": INDIA_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]
    try:
        result: Any = client.create(
            response_model=SignalDecision,
            model=model_name,
            messages=messages_mm,
        )
    except Exception as exc:
        # Some deployments reject multimodal content payloads.
        if "Unsupported content item type" not in str(exc):
            raise
        logger.info("LLM deployment rejected multimodal payload; retrying with text-only prompt")
        text_only_prompt = (
            f"{prompt}\n\n"
            "Image context (data URL, for multimodal-compatible providers):\n"
            f"{image_data_url}"
        )
        result = client.create(
            response_model=SignalDecision,
            model=model_name,
            messages=[
                {"role": "system", "content": INDIA_SYSTEM_PROMPT},
                {"role": "user", "content": text_only_prompt},
            ],
        )
    if asyncio.iscoroutine(result):
        result = await result
    return cast(SignalDecision, result)


def decide_signal(
    state: IntersectionState,
    frame: np.ndarray,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[SignalDecision, str, float]:
    start = time.perf_counter()

    if _cooldown_remaining_s() > 0:
        remaining = _cooldown_remaining_s()
        logger.info("LLM temporarily in cooldown (%ss remaining); using rule-based fallback", remaining)
        fallback = rule_based_decision(state)
        return fallback, "rule_based", (time.perf_counter() - start) * 1000.0

    try:
        decision = asyncio.run(_decide_with_client(state, frame, model_name))
        return decision, "llm", (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        _set_llm_cooldown(exc)
        remaining = _cooldown_remaining_s()
        logger.warning("LLM decision failed, using rule-based fallback: %s", exc)
        if remaining > 0:
            logger.warning("LLM cooldown activated for %ss after failure", remaining)
        fallback = rule_based_decision(state)
        return fallback, "rule_based", (time.perf_counter() - start) * 1000.0
