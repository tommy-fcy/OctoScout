"""Local vector index for offline issue search using FAISS + sentence-transformers.

Requires optional dependencies: pip install octoscout[vector]
"""

from __future__ import annotations

import json
from pathlib import Path

from octoscout.matrix.models import ExtractedIssueInfo
from octoscout.models import GitHubIssueRef

# Lazy-loaded dependencies (optional)
_faiss = None
_SentenceTransformer = None
_np = None


def _ensure_deps():
    """Lazy-load optional dependencies."""
    global _faiss, _SentenceTransformer, _np
    if _faiss is None:
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer
            _faiss = faiss
            _np = np
            _SentenceTransformer = SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local vector index requires optional dependencies. "
                "Install them with: pip install octoscout[vector]"
            ) from e


_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class LocalIndex:
    """FAISS-based vector index for semantic issue search."""

    def __init__(self, index_dir: Path):
        self._index_dir = index_dir
        self._index = None
        self._metadata: list[dict] | None = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            _ensure_deps()
            self._model = _SentenceTransformer(_DEFAULT_MODEL)
        return self._model

    def _load_index(self):
        """Lazy-load the FAISS index and metadata."""
        if self._index is not None:
            return

        _ensure_deps()
        index_path = self._index_dir / "faiss.index"
        metadata_path = self._index_dir / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found at {self._index_dir}. "
                "Run 'octoscout matrix index' to build it."
            )

        self._index = _faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    async def build(self, extracted_dir: Path) -> int:
        """Build the FAISS index from extracted JSONL files.

        Returns the number of indexed issues.
        """
        _ensure_deps()
        model = self._load_model()

        # Read all extracted issues
        issues: list[ExtractedIssueInfo] = []
        for jsonl_file in extracted_dir.glob("*.jsonl"):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            issues.append(ExtractedIssueInfo.from_dict(json.loads(line)))
                        except (json.JSONDecodeError, KeyError):
                            pass

        if not issues:
            return 0

        # Create embedding texts
        texts = []
        metadata = []
        for issue in issues:
            text = f"{issue.title} {issue.error_message_summary} {issue.solution_detail or ''}"
            texts.append(text.strip())

            # Parse repo and number from issue_id (format: "owner/repo#number")
            repo = ""
            number = 0
            if "#" in issue.issue_id:
                repo, num_str = issue.issue_id.rsplit("#", 1)
                try:
                    number = int(num_str)
                except ValueError:
                    pass

            metadata.append({
                "issue_id": issue.issue_id,
                "repo": repo,
                "number": number,
                "title": issue.title,
                "url": f"https://github.com/{repo}/issues/{number}" if repo else "",
                "snippet": issue.error_message_summary[:300],
            })

        # Encode and normalize
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        embeddings = _np.array(embeddings, dtype="float32")

        # Build FAISS index (Inner Product on normalized vectors = cosine similarity)
        dim = embeddings.shape[1]
        index = _faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save
        self._index_dir.mkdir(parents=True, exist_ok=True)
        _faiss.write_index(index, str(self._index_dir / "faiss.index"))
        with open(self._index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self._index = index
        self._metadata = metadata

        return len(issues)

    async def search(self, query: str, top_k: int = 10) -> list[GitHubIssueRef]:
        """Search the local index for semantically similar issues."""
        self._load_index()
        model = self._load_model()

        # Encode query
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = _np.array(query_vec, dtype="float32")

        # Search
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []

        scores, indices = self._index.search(query_vec, k)

        results: list[GitHubIssueRef] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            results.append(
                GitHubIssueRef(
                    repo=meta.get("repo", ""),
                    number=meta.get("number", 0),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    snippet=meta.get("snippet", ""),
                    relevance_score=float(score),
                )
            )

        return results

    @property
    def index_size(self) -> int:
        """Number of indexed issues."""
        if self._index is None:
            try:
                self._load_index()
            except FileNotFoundError:
                return 0
        return self._index.ntotal if self._index else 0
