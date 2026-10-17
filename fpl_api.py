"""Read-only FPL HTTP client with verified, age-limited disk caching."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_BASE = "https://fantasy.premierleague.com/api/"


class APIError(RuntimeError):
    """A request failed; old data has not been substituted."""


class DataValidationError(ValueError):
    """Response or snapshot cannot safely be used for predictions."""


def utc_now():
    return datetime.now(timezone.utc)


def parse_timestamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise DataValidationError(f"Invalid timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise DataValidationError("Timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def atomic_write(path, data):
    """Replace a file only after all bytes have reached disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False)
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


class FPLClient:
    """Public GETs only. Expired/corrupt cache never masks an outage.

    get_json accepts an API path or full HTTPS URL. Use a separate ignored
    cache directory for personal team responses.
    """

    def __init__(self, cache_dir="data/cache", timeout=30, retries=3):
        if timeout <= 0 or retries < 1:
            raise ValueError("timeout and retries must be positive")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = retries
        self.records = []
        self._lock = threading.Lock()

    def _get(self, path, *, max_age_seconds=21600, force=False):
        if max_age_seconds < 0:
            raise ValueError("max_age_seconds cannot be negative")
        url = path if path.startswith("https://") else API_BASE + path.lstrip("/")
        if "://" in path and not path.startswith("https://"):
            raise ValueError("Only HTTPS sources are supported")
        key = hashlib.sha256(url.encode()).hexdigest()
        payload_path = self.cache_dir / (key + ".body")
        metadata_path = self.cache_dir / (key + ".json")
        now = utc_now()
        if not force and payload_path.exists() and metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text())
                age = (now - parse_timestamp(meta["fetched_at"])).total_seconds()
                payload = payload_path.read_bytes()
                valid = (meta["url"] == url and 0 <= age <= max_age_seconds
                         and hashlib.sha256(payload).hexdigest() == meta["sha256"])
            except (OSError, KeyError, ValueError, TypeError):
                valid = False
            if valid:
                with self._lock:
                    self.records.append({**meta, "cache_hit": True})
                return payload
        error = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(url, headers={
                    "User-Agent": "fpl26-data-client/0.4",
                    "Accept": "application/json,text/csv,text/plain",
                })
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if response.status != 200:
                        raise APIError(f"HTTP {response.status}: {url}")
                    payload = response.read()
                if not payload:
                    raise APIError(f"Empty response: {url}")
                meta = {"url": url, "fetched_at": utc_now().isoformat(),
                        "sha256": hashlib.sha256(payload).hexdigest()}
                atomic_write(payload_path, payload)
                atomic_write(metadata_path, json_bytes(meta))
                with self._lock:
                    self.records.append({**meta, "cache_hit": False})
                return payload
            except urllib.error.HTTPError as exc:
                error = exc
                if exc.code != 429 and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError, APIError) as exc:
                error = exc
            if attempt + 1 < self.retries:
                time.sleep(min(2 ** attempt, 4))
        raise APIError(f"Could not fetch {url}: {error}; no stale-cache fallback") from error

    def get_json(self, path, *, max_age_seconds=21600, force=False):
        payload = self._get(path, max_age_seconds=max_age_seconds, force=force)
        try:
            return json.loads(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            raise DataValidationError(f"Invalid JSON from {path}") from exc

    def get_csv(self, url, *, max_age_seconds=2592000, force=False):
        payload = self._get(url, max_age_seconds=max_age_seconds, force=force)
        try:
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
            rows = list(reader)
        except (csv.Error, UnicodeDecodeError) as exc:
            raise DataValidationError(f"Invalid CSV from {url}") from exc
        if not rows or not reader.fieldnames or len(reader.fieldnames) < 2:
            raise DataValidationError(f"Empty or invalid CSV from {url}")
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise DataValidationError(f"Ragged CSV from {url}")
        return rows
