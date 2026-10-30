"""Phase 4: dispatch of share envelopes over logically independent channels.

Channels are directory drops: channel i writes to ``<root>/<session>/channel_<i>/``.
The share pixels are stored as a lossless PNG and the envelope header as JSON, so
the files can be inspected by hand or moved over any real transport.

A ``FaultPlan`` describes adverse conditions to inject per channel for evaluation:
drop, delay (beyond the receive timeout), single-pixel flip, crop, and stale-share
replay from an earlier session.
"""

from __future__ import annotations

import io
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .manifest import Envelope

FAULT_NONE = "none"
FAULT_DROP = "drop"
FAULT_DELAY = "delay"
FAULT_FLIP = "flip"
FAULT_CROP = "crop"
FAULT_REPLAY = "replay"
FAULT_SUBSTITUTE = "substitute"
FAULT_KINDS = [FAULT_NONE, FAULT_DROP, FAULT_DELAY, FAULT_FLIP, FAULT_CROP, FAULT_REPLAY, FAULT_SUBSTITUTE]

FAULT_DESCRIPTIONS = {
    FAULT_NONE: "Delivered intact",
    FAULT_DROP: "Share lost in transit",
    FAULT_DELAY: "Share delayed past the receive timeout",
    FAULT_FLIP: "One pixel altered in transit",
    FAULT_CROP: "Border cropped in transit",
    FAULT_REPLAY: "Stale share from an earlier session injected",
    FAULT_SUBSTITUTE: "Share pixels replaced by an attacker (tag left unchanged)",
}


@dataclass
class FaultPlan:
    """Per-channel faults, keyed by 1-based share index. Applied on first send only."""

    faults: dict[int, str] = field(default_factory=dict)
    seed: int = 0
    stale_envelopes: dict[int, Envelope] = field(default_factory=dict)
    persistent: bool = False

    def fault_for(self, index: int, attempt: int) -> str:
        if attempt > 1 and not self.persistent:
            return FAULT_NONE
        return self.faults.get(index, FAULT_NONE)


@dataclass
class Receipt:
    index: int
    channel: str
    attempt: int
    status: str            # "ack" or "timeout"
    fault: str
    latency_ms: float
    bytes_written: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _apply_fault(envelope: Envelope, fault: str, rng: np.random.Generator,
                 plan: FaultPlan) -> Envelope | None:
    if fault in (FAULT_NONE, FAULT_DELAY):
        return envelope
    if fault == FAULT_DROP:
        return None
    if fault == FAULT_FLIP:
        mutated = envelope.copy()
        row = int(rng.integers(mutated.array.shape[0]))
        col = int(rng.integers(mutated.array.shape[1]))
        chan = int(rng.integers(mutated.array.shape[2]))
        original = int(mutated.array[row, col, chan])
        mutated.array[row, col, chan] = 255 - original if original in (0, 255) else original ^ 0x01
        return mutated
    if fault == FAULT_CROP:
        mutated = envelope.copy()
        margin = max(1, min(mutated.array.shape[0], mutated.array.shape[1]) // 20)
        mutated.array = mutated.array[margin:-margin, margin:-margin, :].copy()
        return mutated
    if fault == FAULT_SUBSTITUTE:
        mutated = envelope.copy()
        mutated.array = rng.integers(0, 256, size=mutated.array.shape, dtype=np.uint8)
        if np.all(np.isin(envelope.array, (0, 255))):
            mutated.array = np.where(mutated.array >= 128, 255, 0).astype(np.uint8)
        return mutated
    if fault == FAULT_REPLAY:
        stale = plan.stale_envelopes.get(envelope.index)
        if stale is None:
            # Without a captured share from an earlier session, forge one by relabeling
            # the header as if it came from a different session.
            stale = envelope.copy()
            stale.session_id = "stale-" + envelope.session_id[:8]
            stale.nonce = "00" * 16
        else:
            stale = stale.copy()
        return stale
    raise ValueError(f"Unknown fault '{fault}'")


def _write_envelope(directory: Path, envelope: Envelope) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    array = envelope.array
    image = Image.fromarray(array[:, :, 0], mode="L") if array.shape[2] == 1 else Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    (directory / f"share_{envelope.index}.png").write_bytes(png_bytes)
    header_bytes = json.dumps(envelope.header(), indent=2).encode()
    (directory / f"share_{envelope.index}.json").write_bytes(header_bytes)
    return len(png_bytes) + len(header_bytes)


def read_envelope(path_png: Path) -> Envelope:
    header = json.loads(path_png.with_suffix(".json").read_text())
    with Image.open(path_png) as image:
        array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    return Envelope(
        session_id=header["session_id"],
        index=int(header["index"]),
        scheme=header["scheme"],
        k=int(header["k"]),
        n=int(header["n"]),
        nonce=header["nonce"],
        tag=header["tag"],
        array=np.ascontiguousarray(array, dtype=np.uint8),
    )


class DirectoryTransport:
    """Writes each share to its own channel directory and reads back whatever arrived."""

    def __init__(self, root: Path, session_id: str, plan: FaultPlan | None = None):
        self.root = Path(root)
        self.session_dir = self.root / session_id
        self.plan = plan or FaultPlan()
        self.rng = np.random.default_rng(self.plan.seed)
        self.receipts: list[Receipt] = []
        self.attempts: dict[int, int] = {}

    def channel_dir(self, index: int) -> Path:
        return self.session_dir / f"channel_{index}"

    def send(self, envelope: Envelope) -> Receipt:
        attempt = self.attempts.get(envelope.index, 0) + 1
        self.attempts[envelope.index] = attempt
        fault = self.plan.fault_for(envelope.index, attempt)
        channel = self.channel_dir(envelope.index)
        if channel.exists():
            shutil.rmtree(channel)

        start = time.perf_counter()
        delivered = _apply_fault(envelope, fault, self.rng, self.plan)
        written = 0
        if delivered is not None and fault != FAULT_DELAY:
            written = _write_envelope(channel, delivered)
        else:
            channel.mkdir(parents=True, exist_ok=True)
        latency_ms = (time.perf_counter() - start) * 1000.0

        status = "timeout" if fault in (FAULT_DROP, FAULT_DELAY) else "ack"
        receipt = Receipt(envelope.index, channel.name, attempt, status, fault, latency_ms, written)
        self.receipts.append(receipt)
        return receipt

    def send_manifest(self, manifest: dict) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def receive(self, indices: list[int] | None = None) -> dict[int, Envelope]:
        """Collect envelopes that arrived on each channel, keyed by channel index."""
        arrived: dict[int, Envelope] = {}
        if not self.session_dir.exists():
            return arrived
        for channel in sorted(self.session_dir.glob("channel_*")):
            index = int(channel.name.split("_")[1])
            if indices is not None and index not in indices:
                continue
            pngs = sorted(channel.glob("share_*.png"))
            if not pngs:
                continue
            arrived[index] = read_envelope(pngs[0])
        return arrived

    def receive_manifest(self) -> dict | None:
        path = self.session_dir / "manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())
