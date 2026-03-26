"""Gemini-based signal decision agent with rule-based fallback."""

import asyncio
import base64
import os
import time
from functools import lru_cache
from typing import Any, Tuple, cast

import cv2
import instructor
import numpy as np
from dotenv import load_dotenv

from agent.models import IntersectionState, SignalDecision
from agent.prompts import INDIA_SYSTEM_PROMPT
from agent.rule_based import rule_based_decision
from agent.state_builder import build_state_prompt

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "google/gemini-2.5-flash")


def _encode_frame_to_data_url(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@lru_cache(maxsize=4)
def _client(model_name: str = DEFAULT_MODEL):
    return instructor.from_provider(model_name)


async def _decide_with_client(state: IntersectionState, frame: np.ndarray, model_name: str) -> SignalDecision:
    prompt = build_state_prompt(state)
    image_data_url = _encode_frame_to_data_url(frame)
    client = _client(model_name)
    result: Any = client.create(
        response_model=SignalDecision,
        messages=[
            {"role": "system", "content": INDIA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
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
    try:
        decision = asyncio.run(_decide_with_client(state, frame, model_name))
        return decision, "llm", (time.perf_counter() - start) * 1000.0
    except Exception:
        fallback = rule_based_decision(state)
        return fallback, "rule_based", (time.perf_counter() - start) * 1000.0
