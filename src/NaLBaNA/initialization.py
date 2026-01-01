"""
initialization.py

Initializes the OpenAI client.
"""
import openai
from openai import OpenAI

openai_client = None  # initially empty

def openai_setup(key: str):
    """Set up the OpenAI client with the given key."""
    global openai_client
    openai_client = OpenAI(api_key=key)

def get_client():
    """Return the OpenAI client; raises error if not initialized."""
    if openai_client is None:
        raise RuntimeError(
            "You must call openai_setup(key) before using the OpenAI client."
        )
    return openai_client
