"""Download pre-built compatibility matrix from GitHub Releases."""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

_console = Console()

REPO = "tommy-fcy/OctoScout"
ASSET_NAME = "matrix.json.gz"
GITHUB_API = "https://api.github.com"


class DownloadError(Exception):
    """Raised when matrix download fails."""


async def download_matrix(
    data_dir: Path,
    token: str | None = None,
    force: bool = False,
) -> Path:
    """Download pre-built matrix.json from the latest GitHub Release.

    Args:
        data_dir: Directory to save matrix.json into.
        token: Optional GitHub token for higher rate limits.
        force: Overwrite existing matrix.json if present.

    Returns:
        Path to the downloaded matrix.json.

    Raises:
        DownloadError: If the download fails.
    """
    output_path = data_dir / "matrix.json"

    if output_path.exists() and not force:
        raise DownloadError(
            f"Matrix already exists at {output_path}. Use --force to overwrite."
        )

    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    headers = _build_headers(token)

    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        # Step 1: Get latest release info
        release_url = f"{GITHUB_API}/repos/{REPO}/releases/latest"
        try:
            resp = await client.get(release_url, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise DownloadError(
                    "No releases found. The pre-built matrix has not been published yet.\n"
                    "You can build it locally with: octoscout matrix crawl --all && octoscout matrix build"
                ) from e
            raise DownloadError(f"Failed to fetch release info: {e}") from e
        except httpx.HTTPError as e:
            raise DownloadError(f"Network error fetching release info: {e}") from e

        release = resp.json()
        release_tag = release.get("tag_name", "unknown")

        # Step 2: Find the matrix asset
        asset = _find_asset(release)
        if asset is None:
            raise DownloadError(
                f"Release {release_tag} does not contain '{ASSET_NAME}' asset.\n"
                "This release may not include pre-built matrix data."
            )

        download_url = asset["browser_download_url"]
        asset_size = asset.get("size", 0)

        _console.print(
            f"[bold cyan]Downloading matrix from release {release_tag}...[/bold cyan]"
        )

        # Step 3: Download with progress bar
        gz_path = data_dir / ASSET_NAME
        try:
            await _download_file(client, download_url, gz_path, asset_size, headers)
        except httpx.HTTPError as e:
            # Clean up partial download
            gz_path.unlink(missing_ok=True)
            raise DownloadError(f"Download failed: {e}") from e

        # Step 4: Decompress
        try:
            _decompress_gzip(gz_path, output_path)
        except Exception as e:
            output_path.unlink(missing_ok=True)
            raise DownloadError(f"Failed to decompress matrix data: {e}") from e
        finally:
            gz_path.unlink(missing_ok=True)

        # Step 5: Validate
        _validate_matrix(output_path)

        return output_path


async def check_update(
    data_dir: Path,
    token: str | None = None,
) -> dict | None:
    """Check if a newer matrix is available.

    Returns:
        A dict with ``tag``, ``published_at``, ``local_built_at`` if an update
        is available, or None if up-to-date.
    """
    matrix_path = data_dir / "matrix.json"

    headers = _build_headers(token)

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        try:
            resp = await client.get(
                f"{GITHUB_API}/repos/{REPO}/releases/latest", headers=headers
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            return None  # Can't check — not an error

        release = resp.json()
        asset = _find_asset(release)
        if asset is None:
            return None

        published_at = release.get("published_at", "")

        # Compare with local matrix
        local_built_at = _get_local_built_at(matrix_path)

        if not local_built_at:
            return {
                "tag": release.get("tag_name", "unknown"),
                "published_at": published_at,
                "local_built_at": None,
            }

        try:
            remote_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            local_dt = datetime.fromisoformat(local_built_at)
            # Ensure both are timezone-aware
            if local_dt.tzinfo is None:
                local_dt = local_dt.replace(tzinfo=timezone.utc)
            if remote_dt > local_dt:
                return {
                    "tag": release.get("tag_name", "unknown"),
                    "published_at": published_at,
                    "local_built_at": local_built_at,
                }
        except (ValueError, TypeError):
            pass  # Can't compare — assume up-to-date

        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_headers(token: str | None) -> dict[str, str]:
    """Build HTTP headers for GitHub API requests."""
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "OctoScout",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _find_asset(release: dict) -> dict | None:
    """Find the matrix asset in a release."""
    for asset in release.get("assets", []):
        if asset.get("name") == ASSET_NAME:
            return asset
    return None


async def _download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    total_size: int,
    headers: dict[str, str],
) -> None:
    """Download a file with a progress bar."""
    # For asset downloads, use Accept: application/octet-stream
    dl_headers = {**headers, "Accept": "application/octet-stream"}

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=_console,
    ) as progress:
        task = progress.add_task("matrix.json.gz", total=total_size or None)

        async with client.stream("GET", url, headers=dl_headers) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))


def _decompress_gzip(src: Path, dest: Path) -> None:
    """Decompress a .gz file to destination."""
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _validate_matrix(path: Path) -> None:
    """Quick validation that the downloaded file is a valid matrix."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "entries" not in data:
            raise DownloadError("Invalid matrix file: missing 'entries' key.")
        entry_count = len(data.get("entries", {}))
        built_at = data.get("built_at", "unknown")
        _console.print(
            f"[bold green]Matrix downloaded successfully.[/bold green] "
            f"{entry_count} version pairs (built: {built_at})"
        )
    except json.JSONDecodeError as e:
        path.unlink(missing_ok=True)
        raise DownloadError(f"Downloaded file is not valid JSON: {e}") from e


def _get_local_built_at(path: Path) -> str | None:
    """Read the built_at timestamp from a local matrix.json."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("built_at")
    except (json.JSONDecodeError, OSError):
        return None
