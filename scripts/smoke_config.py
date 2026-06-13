"""Manual smoke test script: load and print the configured application settings.

Simple validation that the YAML config file can be parsed and loaded
into the Pydantic models without errors.
"""

from src.core.config import load_config

config = load_config()

print(config)
