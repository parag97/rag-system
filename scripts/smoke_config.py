"""Manual smoke script: load and print the configured application settings."""

from src.core.config import load_config

config = load_config()

print(config)
