from dataclasses import dataclass

import asyncio
import fnmatch
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, AsyncIterator, Coroutine
import redis.asyncio as redis
import wcmatch.glob as wcglob
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
)


class _AsyncThread(threading.Thread):
    """helper thread class for running async coroutines in a separate thread"""

    def __init__(self, coroutine: Coroutine[Any, Any, Any]):
        self.coroutine = coroutine
        self.result = None
        self.exception = None

        super().__init__(daemon=True)

    def run(self):
        try:
            self.result = asyncio.run(self.coroutine)
        except Exception as e:
            self.exception = e


def run_async_safely[T](coroutine: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """safely runs a coroutine with handling of an existing event loop.

    This function detects if there's already a running event loop and uses
    a separate thread if needed to avoid the "asyncio.run() cannot be called
    from a running event loop" error. This is particularly useful in environments
    like Jupyter notebooks, FastAPI applications, or other async frameworks.

    Args:
        coroutine: The coroutine to run
        timeout: max seconds to wait for. None means hanging forever

    Returns:
        The result of the coroutine

    Raises:
        Any exception raised by the coroutine
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # There's a running loop, use a separate thread
        thread = _AsyncThread(coroutine)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError("The operation timed out after %f seconds" % timeout)

        if thread.exception:
            raise thread.exception

        return thread.result
    else:
        if timeout:
            coroutine = asyncio.wait_for(coroutine, timeout)

        return asyncio.run(coroutine)


def _utcnow_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _make_text_file_data(content: str) -> dict[str, Any]:
    """Build the canonical JSON payload used by text-oriented backends."""
    now = _utcnow_iso()
    return {
        "content": content.splitlines(),
        "created_at": now,
        "modified_at": now,
    }


def _read_text_payload(
    file_path: str,
    data: dict[str, Any] | None,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Render stored line-array content using Deep Agents' read format."""
    if data is None:
        return f"Error: File '{file_path}' not found"

    lines = data.get("content", [])
    if not lines:
        empty_msg = check_empty_content("")
        if empty_msg:
            return empty_msg

    if offset >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    selected = lines[offset : offset + limit]
    return format_content_with_line_numbers(selected, start_line=offset + 1)


def _edit_text_payload(
    data: dict[str, Any] | None,
    *,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Apply Deep Agents string replacement semantics to stored file data."""
    if data is None:
        return None, None, "file_not_found"

    content = "\n".join(data.get("content", []))
    result = perform_string_replacement(content, old_string, new_string, replace_all)
    if isinstance(result, str):
        return None, None, result

    new_content, occurrences = result
    data["content"] = new_content.splitlines()
    return data, int(occurrences), None


def _normalize_virtual_path(path: str) -> str:
    """Normalize any path into a slash-prefixed virtual path."""
    return "/" + path.lstrip("/")


def _build_direct_listing(
    directory_path: str,
    files: list[tuple[str, int | None, str | None]],
) -> list[FileInfo]:
    """Collapse recursive file metadata into direct child file/dir entries."""
    normalized_dir = "/" if directory_path == "/" else "/" + directory_path.strip("/")
    base = normalized_dir.strip("/")
    prefix = f"{base}/" if base else ""

    direct_files: list[FileInfo] = []
    direct_dirs: set[str] = set()

    for virtual_path, size, modified_at in files:
        clean = virtual_path.lstrip("/")
        if prefix:
            if not clean.startswith(prefix):
                continue
            rel = clean[len(prefix) :]
        else:
            rel = clean

        if not rel:
            continue

        child, sep, _rest = rel.partition("/")
        if sep:
            direct_dirs.add(child)
            continue

        direct_files.append(
            {
                "path": _normalize_virtual_path(clean),
                "is_dir": False,
                "size": size or 0,
                "modified_at": modified_at,
            }
        )

    results = direct_files + [
        {
            "path": (
                f"{normalized_dir.rstrip('/')}/{dir_name}/"
                if normalized_dir != "/"
                else f"/{dir_name}/"
            ),
            "is_dir": True,
        }
        for dir_name in sorted(direct_dirs)
    ]
    results.sort(key=lambda item: item.get("path", ""))
    return results


def _matches_glob(pattern: str, path: str, virtual_path: str) -> bool:
    """Return whether a path matches a glob relative to a search root or absolutely."""
    rel_path = (
        virtual_path[len(path) :].lstrip("/") if path != "/" else virtual_path[1:]
    )
    return fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(virtual_path, pattern)


def _status_code_from_error(error: Exception) -> int | None:
    """Extract an HTTP-like status code from client exceptions when available."""
    return getattr(error, "status", getattr(error, "code", None))


@dataclass
class RedisConfig:
    """Configuration for Redis/Valkey storage."""

    url: str = "redis://localhost:6379/0"
    prefix: str = ""
    namespace: str = "deepagents"


class RedisBackend(BackendProtocol):
    """Redis/Valkey backend for Deep Agents file operations."""

    def __init__(self, config: RedisConfig) -> None:
        self._config = config
        self._prefix = config.prefix.strip("/")
        if self._prefix:
            self._prefix += "/"
        self._client = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._namespace = config.namespace
        self._index_key = f"{self._namespace}:__index__"

    def _storage_path(self, path: str) -> str:
        return f"{self._prefix}{path.lstrip('/')}"

    def _virtual_path(self, storage_path: str) -> str:
        if self._prefix and storage_path.startswith(self._prefix):
            storage_path = storage_path[len(self._prefix) :]
        return _normalize_virtual_path(storage_path)

    def _data_key(self, path: str) -> str:
        return f"{self._namespace}:file:{self._storage_path(path)}"

    async def close(self) -> None:
        """Close the Redis client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._client_loop = None

    async def _ensure_client(self):
        """Lazily initialize the Redis client inside an event loop."""
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is not loop:
            try:
                await self._client.aclose()
            except RuntimeError:
                pass
            self._client = None
            self._client_loop = None

        if self._client is None:
            self._client = redis.from_url(self._config.url, decode_responses=False)
            self._client_loop = loop
        return self._client

    async def _get_file_data(self, path: str) -> dict[str, Any] | None:
        client = await self._ensure_client()
        payload = await client.get(self._data_key(path))
        if payload is None:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return json.loads(payload)

    async def _put_file_data(
        self,
        path: str,
        data: dict[str, Any],
        *,
        update_modified: bool = True,
    ) -> None:
        if update_modified:
            data["modified_at"] = _utcnow_iso()
        storage_path = self._storage_path(path)
        client = await self._ensure_client()
        await client.set(
            f"{self._namespace}:file:{storage_path}",
            json.dumps(data).encode("utf-8"),
        )
        await client.sadd(self._index_key, storage_path)

    async def _exists(self, path: str) -> bool:
        client = await self._ensure_client()
        return bool(await client.exists(self._data_key(path)))

    async def _list_paths(self, path: str = "/") -> list[tuple[str, int | None, str | None]]:
        storage_prefix = self._storage_path(path)
        client = await self._ensure_client()
        members = await client.smembers(self._index_key)
        storage_paths = sorted(
            (
                member.decode("utf-8") if isinstance(member, bytes) else str(member)
                for member in members
            ),
            key=str,
        )
        matching = [
            storage_path
            for storage_path in storage_paths
            if storage_path.startswith(storage_prefix)
        ]
        payloads = await client.mget(
            [f"{self._namespace}:file:{storage_path}" for storage_path in matching]
        )

        results: list[tuple[str, int | None, str | None]] = []
        for storage_path, payload in zip(matching, payloads, strict=False):
            if payload is None:
                continue
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            data = json.loads(payload)
            results.append(
                (
                    self._virtual_path(storage_path),
                    len(data.get("content", [])),
                    data.get("modified_at"),
                )
            )
        return results

    def ls_info(self, path: str) -> list[FileInfo]:
        return run_async_safely(self.als_info(path))

    async def als_info(self, path: str) -> list[FileInfo]:
        return _build_direct_listing(path, await self._list_paths(path))

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return run_async_safely(self.aread(file_path, offset, limit))

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return _read_text_payload(file_path, await self._get_file_data(file_path), offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        return run_async_safely(self.awrite(file_path, content))

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        if await self._exists(file_path):
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                "Read and then make an edit, or write to a new path."
            )

        try:
            await self._put_file_data(file_path, _make_text_file_data(content), update_modified=False)
            return WriteResult(path=file_path, files_update=None)
        except Exception as exc:
            return WriteResult(error=f"Error writing file '{file_path}': {exc}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return run_async_safely(
            self.aedit(file_path, old_string, new_string, replace_all)
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        updated, occurrences, error = _edit_text_payload(
            await self._get_file_data(file_path),
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        if error == "file_not_found":
            return EditResult(error=f"Error: File '{file_path}' not found")
        if error:
            return EditResult(error=error)

        try:
            await self._put_file_data(file_path, updated or {})
            return EditResult(path=file_path, files_update=None, occurrences=occurrences or 0)
        except Exception as exc:
            return EditResult(error=f"Error editing file '{file_path}': {exc}")

    def grep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list[GrepMatch] | str:
        return run_async_safely(self.agrep_raw(pattern, path, glob))

    async def agrep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> list[GrepMatch] | str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Invalid regex pattern: {exc}"

        matches: list[GrepMatch] = []
        for virtual_path, _size, _modified in await self._list_paths(path or "/"):
            filename = PurePosixPath(virtual_path).name
            if glob and not wcglob.globmatch(filename, glob, flags=wcglob.BRACE):
                continue
            data = await self._get_file_data(virtual_path)
            if data is None:
                continue
            for line_num, line in enumerate(data.get("content", []), 1):
                if regex.search(line):
                    matches.append({"path": virtual_path, "line": line_num, "text": line})
        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return run_async_safely(self.aglob_info(pattern, path))

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        results: list[FileInfo] = []
        for virtual_path, size, modified_at in await self._list_paths(path):
            if _matches_glob(pattern, path, virtual_path):
                results.append(
                    {
                        "path": virtual_path,
                        "is_dir": False,
                        "size": size or 0,
                        "modified_at": modified_at,
                    }
                )
        results.sort(key=lambda item: item.get("path", ""))
        return results

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return run_async_safely(self.aupload_files(files))

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                await self._put_file_data(
                    path,
                    _make_text_file_data(content.decode("utf-8", errors="replace")),
                    update_modified=False,
                )
                responses.append(FileUploadResponse(path=path, error=None))
            except Exception:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return run_async_safely(self.adownload_files(paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                data = await self._get_file_data(path)
                if data is None:
                    responses.append(
                        FileDownloadResponse(path=path, content=None, error="file_not_found")
                    )
                    continue
                content = "\n".join(data.get("content", [])).encode("utf-8")
                responses.append(FileDownloadResponse(path=path, content=content, error=None))
            except Exception:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
        return responses