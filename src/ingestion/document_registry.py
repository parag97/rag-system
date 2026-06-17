"""Persistent registry for tracking ingested documents."""

import json
from pathlib import Path

from pydantic import BaseModel


class RegistryEntry(BaseModel):
    """
    Tracks a single ingested document.
    """

    document_id: str
    file_hash: str


class DocumentRegistry:
    """
    Simple JSON-backed document registry.

    Example:

    {
        "data/raw/amazon.pdf": {
            "document_id": "amazon-abc123",
            "file_hash": "xyz789"
        }
    }
    """

    def __init__(
        self,
        registry_path: str | Path,
    ) -> None:
        self._registry_path = Path(registry_path)

        if not self._registry_path.exists():
            self._registry_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._save_raw({})

    def get(
        self,
        source_path: str,
    ) -> RegistryEntry | None:
        data = self._load_raw()

        entry = data.get(source_path)

        if entry is None:
            return None

        return RegistryEntry.model_validate(entry)

    def set(
        self,
        source_path: str,
        entry: RegistryEntry,
    ) -> None:
        data = self._load_raw()

        data[source_path] = entry.model_dump()

        self._save_raw(data)

    def remove(
        self,
        source_path: str,
    ) -> None:
        data = self._load_raw()

        if source_path in data:
            del data[source_path]

            self._save_raw(data)

    def exists(
        self,
        source_path: str,
    ) -> bool:
        data = self._load_raw()

        return source_path in data

    def list_entries(
        self,
    ) -> dict[str, RegistryEntry]:
        raw = self._load_raw()

        return {
            key: RegistryEntry.model_validate(value)
            for key, value in raw.items()
        }

    def _load_raw(
        self,
    ) -> dict:
        with open(
            self._registry_path,
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _save_raw(
        self,
        data: dict,
    ) -> None:
        with open(
            self._registry_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=4,
            )