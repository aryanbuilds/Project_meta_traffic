#!/usr/bin/env python3
"""
Test instructor + Gemini API integration BEFORE Day 3 (T03.4)

This script verifies the correct API syntax for instructor with Google Gemini 2.0 Flash.
Research shows Pattern 1 (from_provider) is correct for 2025.

Run: python test_instructor_api.py
"""

import os
import sys
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("=" * 70)
print("Testing instructor + Gemini API Integration")
print("=" * 70)

# ── Check API Key ────────────────────────────────────────────────────
print("\n1. Checking GEMINI_API_KEY environment variable...")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("[X] GEMINI_API_KEY not found in environment")
    print("  Please set it: export GEMINI_API_KEY='your-api-key-here'")
    print("  Get free key: https://aistudio.google.com")
    sys.exit(1)
else:
    print(f"[OK] GEMINI_API_KEY found ({len(api_key)} characters)")

# ── Test Pattern 1: from_provider (2025 current API) ─────────────────
print("\n2. Testing Pattern 1: instructor.from_provider() [CURRENT 2025 API]")
try:
    import instructor
    print("  [OK] instructor imported")
except ImportError as e:
    print(f"  [X] instructor import failed: {e}")
    print("  Install: pip install instructor")
    sys.exit(1)

try:
    from google import genai
    print("  [OK] google.genai imported")
except ImportError as e:
    print(f"  [X] google.genai import failed: {e}")
    print("  Install: pip install google-genai")
    sys.exit(1)

# Define test Pydantic model (matches SignalDecision structure)
class TestSignalDecision(BaseModel):
    """Test model matching NeuroSignal SignalDecision structure"""
    phase: Literal['north', 'south', 'east', 'west']
    duration_s: int = Field(ge=15, le=60, description="Green duration in seconds")
    reasoning: str = Field(min_length=20, description="Chain-of-thought reasoning")

print("\n3. Initializing instructor client...")
try:
    client = instructor.from_provider("google/gemini-2.0-flash-lite")
    print("  [OK] Client created with instructor.from_provider()")
except Exception as e:
    print(f"  [X] Client creation failed: {e}")
    sys.exit(1)

# ── Make test API call ───────────────────────────────────────────────
print("\n4. Making test API call to Gemini 2.0 Flash Lite...")
test_prompt = """
You are a Delhi intersection traffic signal controller.

Current traffic state:
- NORTH: 12 vehicles, PCE=15.5, wait_time=67s
- SOUTH: 5 vehicles, PCE=6.0, wait_time=15s
- EAST: 8 vehicles, PCE=9.5, wait_time=42s
- WEST: 0 vehicles, PCE=0.0, wait_time=0s

Decide which direction to give green signal and for how long.
Provide detailed reasoning explaining your decision.
"""

try:
    response = client.create(
        response_model=TestSignalDecision,
        messages=[
            {
                "role": "user",
                "content": test_prompt
            }
        ],
        max_retries=0  # No retries to avoid spamming API
    )

    print("  [OK] API call successful!")
    print("\n5. Response validation:")
    print(f"  Phase: {response.phase}")
    print(f"  Duration: {response.duration_s}s")
    print(f"  Reasoning length: {len(response.reasoning)} chars")
    print(f"  Reasoning preview: {response.reasoning[:100]}...")

    # Validate Pydantic constraints
    assert 15 <= response.duration_s <= 60, f"Duration {response.duration_s} outside [15,60]"
    assert len(response.reasoning) >= 20, f"Reasoning too short: {len(response.reasoning)} chars"
    assert response.phase in ['north', 'south', 'east', 'west'], f"Invalid phase: {response.phase}"

    print("\n  [OK] All Pydantic validations passed!")

except Exception as e:
    print(f"  [X] API call failed: {e}")
    print("\nDEBUG INFO:")
    print(f"  Exception type: {type(e).__name__}")
    print(f"  Exception args: {e.args}")
    sys.exit(1)

# ── Test Pattern 2: from_gemini (deprecated?) ────────────────────────
print("\n6. Testing Pattern 2: instructor.from_gemini() [STAGES.JSON PATTERN]")
try:
    # This may fail if the API has changed
    client2 = instructor.from_gemini(
        genai.GenerativeModel('gemini-2.0-flash')
    )
    print("  [OK] Pattern 2 works (from_gemini)")
except AttributeError as e:
    print(f"  [X] Pattern 2 failed: {e}")
    print("     This confirms Pattern 1 (from_provider) is the correct 2025 API")
except Exception as e:
    print(f"  [X] Pattern 2 failed with unexpected error: {e}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: instructor API Verification")
print("=" * 70)
print("[OK] instructor + Gemini 2.0 Flash working correctly")
print("[OK] Correct API pattern: instructor.from_provider('google/gemini-2.0-flash')")
print("[OK] Pydantic structured output validation working")
print("[OK] Ready to implement T03.4 on Day 3")
print("\nACTION REQUIRED:")
print("  Update Stages.json T03.4 to use Pattern 1 (from_provider)")
print("=" * 70)
