from __future__ import annotations

import json
import os
import time
import threading
import ssl
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path
from .core import append_jsonl, canonical, digest, jsonl

API = "https://openrouter.ai/api/v1"


class RateLimited(RuntimeError):
    """Explicit rejected HTTP429 response, distinct from an ambiguous network failure."""


def fetch_json(url, key=None, payload=None, timeout=180):
    headers = {"User-Agent": "finrag-reconstruction/0.1"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None
    if payload is not None:
        data = canonical(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, headers=headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimited("Explicit HTTP429 rate-limit rejection") from None
        # Never echo a request, headers, key, or a provider error that might reflect them.
        raise RuntimeError(f"HTTP {exc.code} from {urllib.parse.urlsplit(url).hostname}; no automatic paid retry") from None
    except (urllib.error.URLError, TimeoutError, ssl.SSLError):
        raise RuntimeError("Network failure; paid requests are not automatically retried (billing may be ambiguous)") from None


def read_key(path=None):
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and path and Path(path).exists():
        key = Path(path).read_text().strip()
    if not key:
        raise ValueError("OpenRouter key is missing. Save it in the local .secrets/openrouter.key file or set OPENROUTER_API_KEY.")
    if "\n" in key or " " in key or not key.startswith("sk-or-"):
        raise ValueError("The local key must contain only a valid OpenRouter key, without quotes or a variable name")
    return key


def preflight(config, key=None):
    models = fetch_json(API + "/models")["data"]
    model = next((m for m in models if m["id"] == config["model"]), None)
    if model is None:
        raise ValueError(f"Configured model {config['model']} is not in the live OpenRouter catalogue. No substitution made.")
    endpoints = fetch_json(API + "/models/" + config["model"] + "/endpoints")["data"]["endpoints"]
    provider = config.get("provider")
    if not provider:
        raise ValueError("Set one explicit provider tag from the model endpoints before a paid run; automatic provider selection is disabled")
    endpoint = next((e for e in endpoints if e.get("tag") == provider), None)
    if endpoint is None:
        available = [e.get("tag") for e in endpoints]
        raise ValueError(f"Pinned provider {provider} is unavailable. Available tags: {available}")
    required = {"max_tokens", "temperature", "seed"}
    if config.get("reasoning") is not None:
        required.add("reasoning")
    if not required.issubset(set(endpoint.get("supported_parameters", []))):
        raise ValueError("Pinned endpoint does not advertise all requested generation parameters")
    pricing = endpoint.get("pricing", {})
    if any(k not in pricing or Decimal(str(pricing[k])) < 0 for k in ("prompt", "completion")):
        raise ValueError("Missing/invalid endpoint pricing")
    result = {"model": model["id"], "provider": provider, "endpoint": endpoint,
              "pricing": pricing, "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if key:
        info = fetch_json(API + "/key", key=key)["data"]
        # Save only non-secret spending fields, never the key metadata/label.
        result["key_limits"] = {k: info.get(k) for k in ("limit", "limit_remaining", "include_byok_in_limit")}
        limit = info.get("limit")
        if limit is None or Decimal(str(limit)) > Decimal(str(config["budget_usd"])):
            raise ValueError("Set a dedicated OpenRouter key credit limit at or below the configured budget before running")
        if info.get("limit_remaining") is not None and info["limit_remaining"] <= 0:
            raise ValueError("Dedicated key has no remaining credit")
    return result


class OpenRouterClient:
    def __init__(self, config, run_dir, key, metadata):
        self.config, self.key = config, key
        self.folder = Path(run_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "calls.jsonl"
        self.ledger_path = self.folder / "budget.jsonl"
        self.pricing = metadata["pricing"]
        self.context = metadata["endpoint"].get("context_length", 0)
        self.cache = {r["request_hash"]: r for r in jsonl(self.path)}
        self.reserved = sum(Decimal(str(r["reserved_usd"])) for r in jsonl(self.ledger_path))
        self.guard = threading.RLock()
        self.request_locks = {}
        self.start_guard = threading.Lock()
        self.last_started = 0.0

    def complete(self, messages, tag):
        payload = {"model": self.config["model"], "messages": messages,
                   "temperature": self.config["temperature"], "seed": self.config["seed"],
                   "max_tokens": self.config["max_tokens"], "stream": False, "transforms": [],
                   "usage": {"include": True},
                   "provider": {"only": [self.config["provider"]], "allow_fallbacks": False,
                                "require_parameters": True}}
        if self.config.get("reasoning") is not None:
            payload["reasoning"] = self.config["reasoning"]
        request_hash = digest(payload)
        with self.guard:
            request_lock = self.request_locks.setdefault(request_hash, threading.Lock())
        with request_lock:
            return self._execute(payload, request_hash, tag)

    def _execute(self, payload, request_hash, tag):
        with self.guard:
            if request_hash in self.cache:
                if self.cache[request_hash]["usage"].get("cost") is None:
                    raise RuntimeError("Cached response has no billing cost; reconcile it before further calls")
                return self.cache[request_hash]
            if request_hash in {r["request_hash"] for r in jsonl(self.ledger_path)}:
                raise RuntimeError("An earlier request has reserved cost but no saved response. Reconcile ambiguous billing; automatic retry is disabled.")
            # Reserve under the shared lock BEFORE allowing concurrent network calls.
            input_allowance = sum(len(m["content"].encode()) + 128 for m in payload["messages"]) + 1024
            reserve = (Decimal(input_allowance) * Decimal(str(self.pricing["prompt"])) +
                       Decimal(payload["max_tokens"]) * Decimal(str(self.pricing["completion"]))) * Decimal("1.10")
            if self.reserved + reserve > Decimal(str(self.config["budget_usd"])):
                raise ValueError("Local conservative spending allowance reached; resume only after reviewing the budget")
            append_jsonl(self.ledger_path, {"request_hash": request_hash, "reserved_usd": str(reserve), "tag": tag})
            self.reserved += reserve
        with self.start_guard:
            interval = float(self.config.get("request_interval_seconds", 0))
            delay = interval - (time.monotonic() - self.last_started)
            if delay > 0:
                time.sleep(delay)
            self.last_started = time.monotonic()
        started = time.monotonic()
        for retry in range(4):
            try:
                result = fetch_json(API + "/chat/completions", key=self.key, payload=payload)
                break
            except RateLimited:
                # Only explicit rejections retry. Never retry ambiguous timeouts/TLS failures.
                delay = min(20 * (2 ** retry), 45)
                with self.guard:
                    append_jsonl(self.folder / "transport-events.jsonl", {
                        "request_hash": request_hash, "http_status": 429, "retry": retry,
                        "backoff_seconds": delay if retry < 3 else None,
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                if retry == 3:
                    raise RuntimeError("Repeated HTTP429; run halted after three bounded retries") from None
                time.sleep(delay)
        elapsed = time.monotonic() - started
        if result.get("error"):
            raise RuntimeError("Provider returned an error; request allowance retained; no automatic paid retry")
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError("No completion choices; no automatic paid retry")
        choice = choices[0]
        content = choice.get("message", {}).get("content") or ""
        if not isinstance(content, str):
            raise RuntimeError("Unexpected completion content type")
        actual_model = result.get("model", "")
        # Provider may return an alias with a dated suffix, but store it for audit.
        record = {"request_hash": request_hash, "tag": tag, "request": payload,
                  "response_id": result.get("id"), "returned_model": actual_model,
                  "returned_provider": result.get("provider"), "system_fingerprint": result.get("system_fingerprint"),
                  "content": content, "finish_reason": choice.get("finish_reason"),
                  "usage": result.get("usage", {}), "elapsed_seconds": elapsed,
                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with self.guard:
            append_jsonl(self.path, record)
            self.cache[request_hash] = record
        if record["usage"].get("cost") is None:
            raise RuntimeError("Response saved but billing cost missing; inspect before making further calls")
        return record
