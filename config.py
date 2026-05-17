"""
config.py — Environment configuration loader.

Usage:
  Place your API key in a .env file (never commit this):
    ANTHROPIC_API_KEY=sk-ant-...
  
  Or set it directly:
    export ANTHROPIC_API_KEY=sk-ant-...

.env file is gitignored automatically if you add it to .gitignore.
"""

import os
from pathlib import Path


def load_env():
    """Load .env file if present (falls back gracefully if python-dotenv not installed)."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            # Manual parse will be used if dotenv is not installed
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        os.environ.setdefault(key.strip(), val.strip())


def validate_env():
    load_env()
    if not (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    ):
        raise EnvironmentError(
            "No API key found. Set GROQ_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY."
        )


# Auto-load will be done on import
load_env()
