"""Adapter layer: the only place the app touches models, storage, or vendor SDKs.

Business code depends on the interfaces in `base`, never on a vendor directly. Swapping
local for hosted, or one vendor for another, is a config change, not a rewrite.
"""
from .base import Chunk, Embedder, LLMClient, VectorStore

__all__ = ["Chunk", "Embedder", "LLMClient", "VectorStore"]
