"""Phase 5: admission gating and verification.

Every received share is checked in a fixed order, and the first failure
decides its status:

1. Session binding: the session id and nonce must match the manifest (else STALE).
2. Parameters: k, n, and scheme must match the manifest (else PARAM_MISMATCH).
3. Integrity: the recomputed SHA-256 tag must equal the manifest's tag for that
   index (else TAMPER). A cropped share has different bytes, so it fails here too.

The whole session aborts if the manifest checksum/signature fails (TAMPER), the
manifest is outside the freshness window (STALE), the manifest disagrees with
the receiver's agreed parameters (PARAM_MISMATCH), or fewer than k shares are
valid (MISSING). The receiver never attempts a best-effort decode.
"""

from __future__ import annotations

import hmac
import time
from dataclasses import dataclass, field

from .errors import MISSING, PARAM_MISMATCH, STALE, TAMPER, DVCError
from .manifest import Envelope, check_manifest, share_tag

STATUS_VALID = "VALID"


@dataclass
class ShareCheck:
    index: int
    status: str
    detail: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class VerificationReport:
    checks: list[ShareCheck] = field(default_factory=list)
    admitted: list[int] = field(default_factory=list)
    quorum_met: bool = False
    k: int = 0
    n: int = 0

    def failed(self) -> list[ShareCheck]:
        return [check for check in self.checks if check.status != STATUS_VALID]

    def to_dict(self) -> dict:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "admitted": list(self.admitted),
            "quorum_met": self.quorum_met,
            "k": self.k,
            "n": self.n,
        }


def verify_manifest(manifest: dict | None, *, expected: dict | None = None,
                    key: bytes | None = None, now: float | None = None) -> None:
    if manifest is None:
        raise DVCError(MISSING, "No manifest was received for this session.", "verify")
    ok, reason = check_manifest(manifest, key)
    if not ok:
        raise DVCError(TAMPER, f"Manifest rejected: {reason}.", "verify")
    now = time.time() if now is None else now
    age = now - float(manifest["timestamp"])
    window = float(manifest.get("freshness_window_s", 0))
    if window > 0 and (age > window or age < -60):
        raise DVCError(STALE, f"Manifest is {age:.0f}s old; the freshness window is {window:.0f}s.", "verify")
    if expected:
        for name, value in expected.items():
            if value is not None and manifest.get(name) != value:
                raise DVCError(
                    PARAM_MISMATCH,
                    f"Manifest has {name}={manifest.get(name)!r} but the receiver expects {value!r}.",
                    "verify",
                )


def check_share(envelope: Envelope, manifest: dict, expected_index: int) -> ShareCheck:
    index = expected_index
    if envelope.session_id != manifest["session_id"] or envelope.nonce != manifest["nonce"]:
        return ShareCheck(index, STALE, f"share belongs to session {envelope.session_id}, not {manifest['session_id']}")
    if envelope.index != expected_index:
        return ShareCheck(index, PARAM_MISMATCH, f"share claims index {envelope.index} but arrived on channel {expected_index}")
    for name in ("k", "n", "scheme"):
        if getattr(envelope, name) != manifest[name]:
            return ShareCheck(index, PARAM_MISMATCH,
                              f"share {name}={getattr(envelope, name)!r}, manifest {name}={manifest[name]!r}")
    expected_shape = (manifest["height"], manifest["width"], manifest["channels"])
    tags = {entry["index"]: entry["tag"] for entry in manifest["tags"]}
    manifest_tag = tags.get(index)
    if manifest_tag is None:
        return ShareCheck(index, PARAM_MISMATCH, "manifest has no tag for this share index")
    actual_tag = share_tag(manifest["session_id"], index, manifest["scheme"], manifest["k"], manifest["n"], envelope.array)
    if tuple(envelope.array.shape) != expected_shape:
        return ShareCheck(index, TAMPER,
                          f"shape {tuple(envelope.array.shape)} differs from {expected_shape}; SHA-256 mismatch")
    if not hmac.compare_digest(actual_tag, manifest_tag) or not hmac.compare_digest(envelope.tag, manifest_tag):
        return ShareCheck(index, TAMPER, "SHA-256 tag mismatch")
    return ShareCheck(index, STATUS_VALID, "tag, nonce, and parameters verified")


def admit(received: dict[int, Envelope], manifest: dict, *, enforce_quorum: bool = True) -> VerificationReport:
    k = int(manifest["k"])
    n = int(manifest["n"])
    report = VerificationReport(k=k, n=n)
    for index in range(1, n + 1):
        envelope = received.get(index)
        if envelope is None:
            report.checks.append(ShareCheck(index, MISSING, "no share arrived on this channel"))
            continue
        check = check_share(envelope, manifest, index)
        report.checks.append(check)
        if check.status == STATUS_VALID:
            report.admitted.append(index)
    report.quorum_met = len(report.admitted) >= k
    if enforce_quorum and not report.quorum_met:
        failures = ", ".join(f"#{c.index}:{c.status}" for c in report.failed())
        error = DVCError(MISSING, f"Only {len(report.admitted)} of the required {k} shares are valid ({failures}).",
                         "verify")
        error.report = report
        raise error
    return report
