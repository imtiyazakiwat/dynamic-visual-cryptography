"""Session orchestration: runs the seven gated phases and persists a replayable log.

A session either passes every gate or aborts with a ``DVCError`` whose code and
phase are recorded. Nothing crosses session boundaries except the algorithm
version: each run draws a fresh 256-bit seed and nonce.

Security note on the log: replay needs the session seed, and anyone holding the
seed plus the input can regenerate every share. Logs written with
``store_seed=True`` are lab artifacts and must be protected like the secret
itself. With ``store_seed=False`` only a SHA-256 seed commitment is stored and
replay is disabled.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics
from .encoder import (
    RECOVERY_XOR, SCHEME_GF256, SCHEME_RG_OR, SCHEME_XOR_NN, Share, encode, ink_bits,
    scheme_for, share_digest, validate_params,
)
from .entropy import ALGORITHM_VERSION, SessionKey
from .errors import MISSING, REPLAY_MISMATCH, TAMPER, UNSUPPORTED_MODE, DVCError
from .manifest import Envelope, build_manifest, manifest_key, share_tag
from .preprocess import CanonicalImage, array_to_pil, load_image, null_path_psnr
from .reconstruct import Reconstruction, combine_gf256, combine_xor, reconstruct, stack_or
from .transport import FAULT_NONE, FAULT_REPLAY, DirectoryTransport, FaultPlan
from .verify import VerificationReport, admit, verify_manifest

DEFAULT_WORKDIR = Path(__file__).resolve().parent.parent / "runs"
NULL_PATH_MIN_PSNR = 40.0
LOG_FILE = "sessions.jsonl"

PHASES = [
    ("preprocess", "1. Acquisition & normalization"),
    ("entropy", "2. Session entropy & grid derivation"),
    ("encode", "3. Threshold encoding"),
    ("dispatch", "4. Tagging & dispatch"),
    ("verify", "5. Admission gating & verification"),
    ("reconstruct", "6. Reconstruction"),
    ("score", "7. Scoring & logging"),
]


@dataclass
class SessionConfig:
    k: int = 2
    n: int = 3
    recovery: str = RECOVERY_XOR
    chaos: bool = False
    force_mode: str | None = None
    faults: dict[int, str] = field(default_factory=dict)
    persistent_faults: bool = False
    max_retries: int = 1
    freshness_window: float = 3600.0
    store_seed: bool = True
    sharpen: bool = False
    manifest_fault: str = FAULT_NONE      # "none" or "tamper"
    expect: dict | None = None            # receiver-side agreed parameters
    fault_seed: int = 0
    compute_secrecy: bool = True
    log: bool = True

    def to_dict(self) -> dict:
        data = dict(self.__dict__)
        data["faults"] = {str(k): v for k, v in self.faults.items()}
        return data


@dataclass
class SessionResult:
    session_id: str | None = None
    status: str = "pending"
    error: dict | None = None
    config: SessionConfig | None = None
    secret: CanonicalImage | None = None
    key: SessionKey | None = None
    shares: list[Share] = field(default_factory=list)
    manifest: dict | None = None
    receipts: list[dict] = field(default_factory=list)
    attempts: list[VerificationReport] = field(default_factory=list)
    reconstruction: Reconstruction | None = None
    below_threshold: np.ndarray | None = None
    scores: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    record: dict | None = None
    channel_dir: Path | None = None

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @property
    def verification(self) -> VerificationReport | None:
        return self.attempts[-1] if self.attempts else None


class _PhaseClock:
    def __init__(self, result: SessionResult):
        self.result = result

    def run(self, phase: str, func, *args, **kwargs):
        start = time.perf_counter()
        self.result.gates[phase] = "running"
        try:
            value = func(*args, **kwargs)
        except DVCError as error:
            error.phase = error.phase or phase
            self.result.gates[phase] = "failed"
            raise
        finally:
            self.result.timings[f"{phase}_ms"] = (time.perf_counter() - start) * 1000.0
        self.result.gates[phase] = "passed"
        return value


# ---------------------------------------------------------------- sender side

def _phase_preprocess(source, config: SessionConfig) -> tuple[CanonicalImage, float]:
    validate_params(config.k, config.n, config.recovery)
    canonical = source if isinstance(source, CanonicalImage) else load_image(source, config.force_mode)
    fidelity = null_path_psnr(canonical)
    if fidelity < NULL_PATH_MIN_PSNR:
        raise DVCError(UNSUPPORTED_MODE, f"Null-path PSNR {fidelity:.1f} dB is below {NULL_PATH_MIN_PSNR} dB.")
    return canonical, fidelity


def _make_envelopes(shares: list[Share], key: SessionKey, k: int, n: int) -> tuple[list[Envelope], list[dict]]:
    envelopes, tags = [], []
    for share in shares:
        tag = share_tag(key.session_id, share.index, share.scheme, k, n, share.array)
        envelopes.append(Envelope(key.session_id, share.index, share.scheme, k, n, key.nonce.hex(), tag, share.array))
        tags.append({"index": share.index, "tag": tag, "sha256": share_digest(share.array)})
    return envelopes, tags


def _stale_envelopes(canonical: CanonicalImage, config: SessionConfig) -> dict[int, Envelope]:
    """Genuine shares from an earlier session of the same image, used by the replay injector."""
    old_key = SessionKey.fresh(canonical.digest(), config.k, config.n, config.recovery, config.chaos)
    old_key.timestamp -= 3 * 86400
    old_shares = encode(canonical.array, canonical.mode, old_key, config.k, config.n, config.recovery)
    envelopes, _ = _make_envelopes(old_shares, old_key, config.k, config.n)
    return {envelope.index: envelope for envelope in envelopes}


def _phase_dispatch(canonical, key, shares, config, workdir):
    envelopes, tags = _make_envelopes(shares, key, config.k, config.n)
    manifest = build_manifest(
        session_id=key.session_id, k=config.k, n=config.n, recovery=config.recovery,
        scheme=shares[0].scheme, image_mode=canonical.mode, shape=canonical.array.shape,
        nonce=key.nonce.hex(), timestamp=key.timestamp, freshness_window=config.freshness_window,
        tags=tags, seed_commitment=key.seed_commitment(), input_digest=canonical.digest(),
        key=manifest_key(),
    )
    plan = FaultPlan(faults=dict(config.faults), seed=config.fault_seed, persistent=config.persistent_faults)
    if FAULT_REPLAY in config.faults.values():
        plan.stale_envelopes = _stale_envelopes(canonical, config)
    transport = DirectoryTransport(Path(workdir) / "channels", key.session_id, plan)
    receipts = [transport.send(envelope) for envelope in envelopes]
    sent_manifest = manifest
    if config.manifest_fault == "tamper":
        sent_manifest = json.loads(json.dumps(manifest))
        first = sent_manifest["tags"][0]
        first["tag"] = ("0" if first["tag"][0] != "0" else "1") + first["tag"][1:]
    transport.send_manifest(sent_manifest)
    return envelopes, manifest, transport, receipts


# -------------------------------------------------------------- receiver side

def _phase_verify(transport: DirectoryTransport, envelopes: list[Envelope], config: SessionConfig,
                  result: SessionResult) -> tuple[dict, dict[int, Envelope], VerificationReport]:
    received_manifest = transport.receive_manifest()
    expected = config.expect or {"version": ALGORITHM_VERSION, "k": config.k, "n": config.n,
                                 "recovery": config.recovery}
    verify_manifest(received_manifest, expected=expected, key=manifest_key())

    received = transport.receive()
    report = admit(received, received_manifest, enforce_quorum=False)
    result.attempts.append(report)

    retries = 0
    while not report.quorum_met and retries < config.max_retries:
        # Selective repair: only the shares that failed are sent again.
        failed = [check.index for check in report.failed()]
        for index in failed:
            result.receipts.append(transport.send(envelopes[index - 1]).to_dict())
        received = transport.receive()
        report = admit(received, received_manifest, enforce_quorum=False)
        result.attempts.append(report)
        retries += 1

    if not report.quorum_met:
        failures = ", ".join(f"#{c.index}:{c.status}" for c in report.failed())
        raise DVCError(MISSING, f"Only {len(report.admitted)} of the required {report.k} shares are valid "
                                f"after {retries} retr{'y' if retries == 1 else 'ies'} ({failures}).")
    return received_manifest, received, report


# ---------------------------------------------------------------- scoring

def _below_threshold_combine(shares: list[Share], k: int) -> np.ndarray:
    """What an attacker holding k-1 shares gets by applying the decoder anyway."""
    subset = shares[: k - 1]
    scheme = shares[0].scheme
    arrays = [share.array for share in subset]
    if scheme == SCHEME_RG_OR:
        return stack_or(arrays)
    if scheme == SCHEME_GF256:
        return combine_gf256(arrays, [share.index for share in subset])
    return combine_xor(arrays)


def score_session(canonical: CanonicalImage, shares: list[Share], recon: Reconstruction,
                  config: SessionConfig, null_psnr: float) -> tuple[dict, np.ndarray]:
    secret = canonical.array
    scheme = shares[0].scheme
    byte_shares = scheme in (SCHEME_XOR_NN, SCHEME_GF256)

    fidelity = {
        "psnr_db": metrics.psnr(secret, recon.display),
        "ssim": metrics.ssim(secret, recon.display),
        "bit_exact": bool(np.array_equal(secret, recon.raw)),
        "null_path_psnr_db": null_psnr,
    }
    fidelity.update({f"contrast_{key}": value for key, value in metrics.contrast_retention(secret, recon.display).items()})
    if scheme == SCHEME_RG_OR:
        ink = ink_bits(secret, canonical.mode)
        fidelity["vc_relative_contrast"] = metrics.vc_relative_contrast(ink, recon.raw)
        fidelity["raw_stack_psnr_db"] = metrics.psnr(secret, recon.raw)

    per_share = []
    for share in shares:
        per_share.append({
            "index": share.index,
            "entropy_bits": metrics.shannon_entropy(share.array),
            "corr_to_secret": metrics.pearson(share.array, secret),
            "adjacent_corr": metrics.adjacent_correlation(share.array),
            "shape": list(share.array.shape),
        })

    below = _below_threshold_combine(shares, config.k)
    secrecy = {
        "per_share": per_share,
        "min_share_entropy": min(item["entropy_bits"] for item in per_share),
        "max_abs_share_corr": max(abs(item["corr_to_secret"]) for item in per_share),
        "below_threshold_corr": metrics.pearson(below, secret),
        "below_threshold_ssim": metrics.ssim(secret, below),
        "zero_expansion": all(tuple(share.array.shape) == secret.shape for share in shares),
    }
    if scheme == SCHEME_RG_OR:
        secrecy["below_threshold_vc_contrast"] = metrics.vc_relative_contrast(ink_bits(secret, canonical.mode), below)

    if config.compute_secrecy:
        # Re-encrypt the identical image under a fresh session and compare share by share.
        other_key = SessionKey.fresh(canonical.digest(), config.k, config.n, config.recovery, config.chaos)
        other = encode(secret, canonical.mode, other_key, config.k, config.n, config.recovery)
        npcrs = [metrics.npcr(a.array, b.array) for a, b in zip(shares, other)]
        uacis = [metrics.uaci(a.array, b.array) for a, b in zip(shares, other)]
        corrs = [metrics.pearson(a.array, b.array) for a, b in zip(shares, other)]
        secrecy["cross_session"] = {
            "npcr_min": min(npcrs), "npcr_mean": float(np.mean(npcrs)),
            "uaci_mean": float(np.mean(uacis)), "max_abs_corr": max(abs(c) for c in corrs),
        }

    targets = []

    def target(name, value, op, limit, applies, note=""):
        if value is None:
            passed = None
        elif op == ">":
            passed = value > limit
        elif op == "<":
            passed = value < limit
        else:
            passed = bool(value)
        targets.append({"metric": name, "value": value, "target": f"{op} {limit}" if op in "<>" else "true",
                        "passed": passed if applies else None, "applies": applies, "note": note})

    xor_path = scheme in (SCHEME_XOR_NN, SCHEME_GF256)
    target("PSNR (XOR path, dB)", fidelity["psnr_db"], ">", 30, xor_path,
           "" if xor_path else "OR stacking is lossy by design; see VC contrast")
    target("SSIM (XOR path)", fidelity["ssim"], ">", 0.90, xor_path)
    target("Null-path PSNR (dB)", null_psnr, ">", NULL_PATH_MIN_PSNR, True)
    target("Share entropy (bits/pixel)", secrecy["min_share_entropy"], ">", 7.9, byte_shares,
           "" if byte_shares else "OR shares are binary (max 1 bit)")
    target("Share-to-secret |correlation|", secrecy["max_abs_share_corr"], "<", 0.05, True)
    target("Below-quorum |correlation|", abs(secrecy["below_threshold_corr"]), "<", 0.05, True)
    if "cross_session" in secrecy:
        target("Cross-session NPCR (%)", secrecy["cross_session"]["npcr_min"], ">", 99.0, byte_shares,
               "" if byte_shares else "binary shares: ideal NPCR is 50%")
        target("Cross-session |correlation|", secrecy["cross_session"]["max_abs_corr"], "<", 0.05, True)
    target("Zero expansion (share = H x W x C)", secrecy["zero_expansion"], "=", True, True)
    if scheme == SCHEME_RG_OR:
        target("VC relative contrast (k shares)", fidelity["vc_relative_contrast"], ">", 0.0, True)

    return {"fidelity": fidelity, "secrecy": secrecy, "targets": targets}, below


# ---------------------------------------------------------------- logging

def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value


def _write_log(result: SessionResult, workdir: Path) -> dict:
    config = result.config
    key = result.key
    record = {
        "logged_at": time.time(),
        "version": ALGORITHM_VERSION,
        "session_id": result.session_id,
        "status": result.status,
        "error": result.error,
        "config": config.to_dict(),
        "scheme": scheme_for(config.k, config.n, config.recovery),
        "gates": result.gates,
        "timings": result.timings,
        "receipts": result.receipts,
        "verification": [attempt.to_dict() for attempt in result.attempts],
        "scores": result.scores,
    }
    if result.secret is not None:
        record.update({
            "input_digest": result.secret.digest(),
            "image_mode": result.secret.mode,
            "shape": list(result.secret.array.shape),
        })
    if key is not None:
        record.update({
            "nonce": key.nonce.hex(),
            "timestamp": key.timestamp,
            "seed_commitment": key.seed_commitment(),
            "seed": key.seed.hex() if config.store_seed else None,
        })
    if result.manifest is not None:
        record["tags"] = result.manifest["tags"]
        record["manifest_checksum"] = result.manifest["checksum"]
    if result.reconstruction is not None:
        record["output_digest"] = result.reconstruction.raw_digest
        record["used_indices"] = result.reconstruction.used_indices

    workdir.mkdir(parents=True, exist_ok=True)
    if config.store_seed and result.secret is not None:
        inputs = workdir / "inputs"
        inputs.mkdir(exist_ok=True)
        path = inputs / f"{result.secret.digest()}.png"
        if not path.exists():
            result.secret.to_pil().save(path)
    record = _json_safe(record)
    with (workdir / LOG_FILE).open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


# ---------------------------------------------------------------- entry points

def run_session(source, config: SessionConfig | None = None, workdir: Path | str = DEFAULT_WORKDIR) -> SessionResult:
    """Run all seven phases. Never raises DVCError: failures are reported in the result."""
    config = config or SessionConfig()
    workdir = Path(workdir)
    result = SessionResult(config=config)
    clock = _PhaseClock(result)
    try:
        canonical, null_psnr = clock.run("preprocess", _phase_preprocess, source, config)
        result.secret = canonical

        key = clock.run("entropy", SessionKey.fresh, canonical.digest(), config.k, config.n,
                        config.recovery, config.chaos)
        result.key = key
        result.session_id = key.session_id

        shares = clock.run("encode", encode, canonical.array, canonical.mode, key, config.k, config.n,
                           config.recovery)
        result.shares = shares

        envelopes, manifest, transport, receipts = clock.run("dispatch", _phase_dispatch, canonical, key, shares,
                                                             config, workdir)
        result.manifest = manifest
        result.receipts = [receipt.to_dict() for receipt in receipts]
        result.channel_dir = transport.session_dir

        received_manifest, received, report = clock.run("verify", _phase_verify, transport, envelopes, config, result)

        admitted = {index: received[index].array for index in report.admitted}
        recon = clock.run("reconstruct", reconstruct, admitted, received_manifest["scheme"], config.k, config.n,
                          config.sharpen)
        result.reconstruction = recon
        result.timings["decode_core_ms"] = recon.decode_ms

        scores, below = clock.run("score", score_session, canonical, shares, recon, config, null_psnr)
        result.scores = scores
        result.below_threshold = below
        result.status = "success"
    except DVCError as error:
        result.status = "aborted"
        result.error = error.to_dict()
    for phase, _ in PHASES:
        result.gates.setdefault(phase, "skipped")
    if config.log:
        result.record = _write_log(result, workdir)
    return result


def read_log(workdir: Path | str = DEFAULT_WORKDIR) -> list[dict]:
    path = Path(workdir) / LOG_FILE
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def find_record(session_id: str, workdir: Path | str = DEFAULT_WORKDIR) -> dict:
    for record in reversed(read_log(workdir)):
        if record.get("session_id") == session_id:
            return record
    raise DVCError(MISSING, f"No logged session with id '{session_id}'.", "replay")


def replay(session_id: str, workdir: Path | str = DEFAULT_WORKDIR) -> dict:
    """Regenerate a logged session from its seed and check every share tag and the output digest."""
    workdir = Path(workdir)
    record = find_record(session_id, workdir)
    if not record.get("seed"):
        raise DVCError(REPLAY_MISMATCH, "This session was logged with a seed commitment only; replay is disabled.",
                       "replay")
    if record.get("tags") is None:
        raise DVCError(REPLAY_MISMATCH, "The session aborted before shares were tagged; nothing to replay.", "replay")
    path = workdir / "inputs" / f"{record['input_digest']}.png"
    if not path.exists():
        raise DVCError(MISSING, f"Logged input image {path.name} is missing.", "replay")

    start = time.perf_counter()
    canonical = load_image(path, force_mode=record["image_mode"])
    if canonical.digest() != record["input_digest"]:
        raise DVCError(TAMPER, "Stored input image does not match the logged digest.", "replay")

    config = record["config"]
    k, n, recovery = int(config["k"]), int(config["n"]), config["recovery"]
    key = SessionKey(seed=bytes.fromhex(record["seed"]), nonce=bytes.fromhex(record["nonce"]),
                     timestamp=float(record["timestamp"]), session_id=record["session_id"],
                     image_digest=record["input_digest"], k=k, n=n, mode=recovery, chaos=bool(config["chaos"]))
    if key.seed_commitment() != record["seed_commitment"]:
        raise DVCError(TAMPER, "Logged seed does not match its commitment.", "replay")

    shares = encode(canonical.array, canonical.mode, key, k, n, recovery)
    logged = {entry["index"]: entry["tag"] for entry in record["tags"]}
    tag_matches = {}
    for share in shares:
        tag_matches[share.index] = share_tag(key.session_id, share.index, share.scheme, k, n, share.array) == logged.get(share.index)

    output_match = None
    reconstruction = None
    if record.get("used_indices"):
        subset = {share.index: share.array for share in shares if share.index in record["used_indices"]}
        reconstruction = reconstruct(subset, shares[0].scheme, k, n, bool(config.get("sharpen", False)))
        output_match = reconstruction.raw_digest == record.get("output_digest")

    bit_exact = all(tag_matches.values()) and output_match is not False
    return {
        "session_id": session_id,
        "bit_exact": bit_exact,
        "tag_matches": tag_matches,
        "output_match": output_match,
        "replay_ms": (time.perf_counter() - start) * 1000.0,
        "shares": shares,
        "reconstruction": reconstruction,
        "secret": canonical,
        "record": record,
    }


def save_artifacts(result: SessionResult, out_dir: Path | str) -> list[Path]:
    """Write secret, shares, reconstruction, and the manifest to ``out_dir`` for inspection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if result.secret is not None:
        path = out_dir / "secret.png"
        result.secret.to_pil().save(path)
        written.append(path)
    for share in result.shares:
        path = out_dir / f"share_{share.index}.png"
        array_to_pil(share.array).save(path)
        written.append(path)
    if result.reconstruction is not None:
        path = out_dir / "reconstructed.png"
        array_to_pil(result.reconstruction.display).save(path)
        written.append(path)
        if result.reconstruction.display is not result.reconstruction.raw:
            path = out_dir / "reconstructed_raw_stack.png"
            array_to_pil(result.reconstruction.raw).save(path)
            written.append(path)
    if result.manifest is not None:
        path = out_dir / "manifest.json"
        path.write_text(json.dumps(result.manifest, indent=2))
        written.append(path)
    return written
