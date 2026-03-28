"""Local vector index for offline issue search (Phase 2 placeholder)."""

from __future__ import annotations


class LocalIndex:
    """Placeholder for local FAISS/Chroma vector index.

    Will be implemented in Phase 2.4 to complement realtime GitHub search
    with pre-indexed issue embeddings for improved recall.
    """

    def __init__(self, index_path: str | None = None):
        self._index_path = index_path

    async def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search the local index. Returns empty list until Phase 2."""
        return []

    async def build(self, issues: list[dict]) -> None:
        """Build/rebuild the index from issue data. Not yet implemented."""
        raise NotImplementedError("Local index will be implemented in Phase 2.4")
