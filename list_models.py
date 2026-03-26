#!/usr/bin/env python3
"""List available Gemini models"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No API key found")
    exit(1)

print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
print("\nListing available models...\n")

client = genai.Client(api_key=api_key)

try:
    models = client.models.list()
    print("Available models:")
    for model in models:
        print(f"  - {model.name}")
        if hasattr(model, 'supported_generation_methods'):
            print(f"    Methods: {model.supported_generation_methods}")
except Exception as e:
    print(f"Error: {e}")
