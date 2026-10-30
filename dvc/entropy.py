"""Phase 2: session entropy, key derivation, and keystream generation.

Authoritative randomness is 256 bits from the OS CSPRNG (``secrets``). It is bound
to the session context (nonce, timestamp, image digest, parameters) with
HKDF-SHA256 (RFC 5869). Per-channel keystreams are produced by SHAKE-256, a
cryptographic extendable-output function, so every byte the encoders consume is
CSPRNG-grade. A logistic map (r = 3.99) can optionally be XOR-mixed in as
auxiliary diffusion; it is never load-bearing for secrecy.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field

import numpy as np

ALGORITHM_VERSION = "DVC-v1"
SEED_BYTES = 32
NONCE_BYTES = 16
LOGISTIC_R = 3.99
LOGISTIC_BURN_IN = 128
LOGISTIC_LANES = 4096


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if not salt:
        salt = b"\x00" * hashlib.sha256().digest_size
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    if length > 255 * 32:
        raise ValueError("HKDF-SHA256 cannot expand beyond 8160 bytes")
    output = b""
    block = b""
    counter = 1
    while len(output) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        output += block
        counter += 1
    return output[:length]


def new_session_id() -> str:
    return secrets.token_hex(8)


@dataclass
class SessionKey:
    """All entropy and context for one session. Never reused across sessions."""

    seed: bytes
    nonce: bytes
    timestamp: float
    session_id: str
    image_digest: str
    k: int
    n: int
    mode: str
    chaos: bool
    _prk: bytes = field(init=False, repr=False)

    def __post_init__(self):
        self._prk = hkdf_extract(self.nonce, self.seed)

    @classmethod
    def fresh(cls, image_digest: str, k: int, n: int, mode: str, chaos: bool,
              operator_nonce: bytes | None = None) -> "SessionKey":
        nonce = secrets.token_bytes(NONCE_BYTES)
        if operator_nonce:
            nonce = hashlib.sha256(nonce + operator_nonce).digest()[:NONCE_BYTES]
        return cls(
            seed=secrets.token_bytes(SEED_BYTES),
            nonce=nonce,
            timestamp=time.time(),
            session_id=new_session_id(),
            image_digest=image_digest,
            k=k,
            n=n,
            mode=mode,
            chaos=chaos,
        )

    def context(self) -> bytes:
        return "|".join(
            [
                ALGORITHM_VERSION,
                self.session_id,
                f"{self.timestamp:.6f}",
                self.image_digest,
                f"k={self.k}",
                f"n={self.n}",
                f"mode={self.mode}",
                f"chaos={int(self.chaos)}",
            ]
        ).encode()

    def seed_commitment(self) -> str:
        return hashlib.sha256(b"DVC-seed-commit|" + self.seed + self.nonce).hexdigest()

    def subkey(self, label: str) -> bytes:
        return hkdf_expand(self._prk, self.context() + b"|" + label.encode(), 32)

    def keystream(self, label: str, length: int) -> np.ndarray:
        """Return ``length`` keystream bytes as a uint8 array for the given label."""
        if length <= 0:
            return np.zeros(0, dtype=np.uint8)
        key = self.subkey(label)
        stream = np.frombuffer(hashlib.shake_256(key).digest(length), dtype=np.uint8)
        if self.chaos:
            chaos_key = self.subkey(label + "|logistic")
            stream = stream ^ logistic_stream(chaos_key, length)
        return stream


def logistic_stream(key: bytes, length: int, r: float = LOGISTIC_R,
                    lanes: int = LOGISTIC_LANES) -> np.ndarray:
    """Byte stream from many parallel logistic maps x <- r x (1 - x).

    Each lane is seeded from SHAKE-256(key) so the map is freshly seeded per session.
    Lanes run in lock-step so the loop is vectorized across lanes.
    """
    lanes = max(1, min(lanes, length))
    raw = np.frombuffer(hashlib.shake_256(b"logistic-seed|" + key).digest(lanes * 8), dtype=np.uint64)
    # Map to the open interval (0.05, 0.95) to avoid the fixed points 0 and 1 - 1/r.
    x = 0.05 + 0.9 * (raw.astype(np.float64) / float(2**64))
    for _ in range(LOGISTIC_BURN_IN):
        x = r * x * (1.0 - x)
    iterations = -(-length // lanes)
    out = np.empty((iterations, lanes), dtype=np.uint8)
    scale = float(2**32)
    for i in range(iterations):
        x = r * x * (1.0 - x)
        # Low-order bits of the scaled state are close to uniform; high bits follow the arcsine law.
        out[i] = (np.floor(x * scale).astype(np.uint64) & 0xFF).astype(np.uint8)
    return out.reshape(-1)[:length]
