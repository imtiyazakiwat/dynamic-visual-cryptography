"""Phase 4 data contracts: per-share integrity tags, share envelopes, and the session manifest."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass

import numpy as np

from .encoder import share_canonical_bytes
from .entropy import ALGORITHM_VERSION

MANIFEST_KEY_ENV = "DVC_MANIFEST_KEY"


def share_tag(session_id: str, index: int, scheme: str, k: int, n: int, array: np.ndarray) -> str:
    """SHA-256 over the share bytes, bound to its session and position.

    Binding the session id means a share replayed from another session fails the
    tag even if an attacker rewrites its envelope.
    """
    prefix = f"{ALGORITHM_VERSION}|{session_id}|share={index}|{scheme}|k={k}|n={n}|".encode()
    return hashlib.sha256(prefix + share_canonical_bytes(array)).hexdigest()


@dataclass
class Envelope:
    """One share in transit plus the metadata it carries."""

    session_id: str
    index: int
    scheme: str
    k: int
    n: int
    nonce: str
    tag: str
    array: np.ndarray

    def header(self) -> dict:
        return {
            "version": ALGORITHM_VERSION,
            "session_id": self.session_id,
            "index": self.index,
            "scheme": self.scheme,
            "k": self.k,
            "n": self.n,
            "nonce": self.nonce,
            "tag": self.tag,
            "shape": list(self.array.shape),
        }

    def copy(self) -> "Envelope":
        return Envelope(self.session_id, self.index, self.scheme, self.k, self.n,
                        self.nonce, self.tag, self.array.copy())


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def manifest_key() -> bytes | None:
    value = os.environ.get(MANIFEST_KEY_ENV)
    return value.encode() if value else None


def seal_manifest(body: dict, key: bytes | None = None) -> dict:
    """Return the manifest with a SHA-256 checksum and, if a key is set, an HMAC-SHA256 signature."""
    sealed = dict(body)
    sealed.pop("checksum", None)
    sealed.pop("signature", None)
    canonical = _canonical_json(sealed)
    sealed["checksum"] = hashlib.sha256(canonical).hexdigest()
    if key:
        sealed["signature"] = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return sealed


def check_manifest(manifest: dict, key: bytes | None = None) -> tuple[bool, str]:
    body = {k: v for k, v in manifest.items() if k not in ("checksum", "signature")}
    canonical = _canonical_json(body)
    if hashlib.sha256(canonical).hexdigest() != manifest.get("checksum"):
        return False, "manifest checksum mismatch"
    if key:
        signature = manifest.get("signature")
        if not signature:
            return False, "manifest is unsigned but a manifest key is configured"
        expected = hmac.new(key, canonical, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return False, "manifest HMAC signature mismatch"
    return True, "ok"


def build_manifest(*, session_id: str, k: int, n: int, recovery: str, scheme: str,
                   image_mode: str, shape: tuple, nonce: str, timestamp: float,
                   freshness_window: float, tags: list[dict], seed_commitment: str,
                   input_digest: str, key: bytes | None = None) -> dict:
    body = {
        "version": ALGORITHM_VERSION,
        "session_id": session_id,
        "k": k,
        "n": n,
        "recovery": recovery,
        "scheme": scheme,
        "image_mode": image_mode,
        "height": int(shape[0]),
        "width": int(shape[1]),
        "channels": int(shape[2]),
        "nonce": nonce,
        "timestamp": timestamp,
        "freshness_window_s": freshness_window,
        "tags": tags,
        "seed_commitment": seed_commitment,
        "input_digest": input_digest,
    }
    return seal_manifest(body, key)
