"""Chunking utilities and chunker implementations.

This package contains implementations and interfaces used to split
documents into indexable chunks for embedding and retrieval.
"""

from .base import Chunker
from .recursive_chunker import RecursiveChunker

__all__ = ["Chunker", "RecursiveChunker"]
