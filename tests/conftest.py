"""Pytest configuration for grafeo-memory tests."""

import sys
from pathlib import Path

# Add tests directory to sys.path so mock_llm can be imported
sys.path.insert(0, str(Path(__file__).parent))
