"""Benchmark: one frozen scorecard across every image, (k, n) setting, and recovery mode.

Outputs (in reports/):
  scorecard.csv        one row per (image, k, n, mode)
  fault_suite.csv      detection outcome for every injected corruption
  threshold_curves.png fidelity vs. (k, n) per mode
  summary.md           acceptance-target table with pass counts

Usage:  .venv/bin/python bench.py [--samples samples] [--trials 5]
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dvc.session import SessionConfig, run_session  # noqa: E402

SETTINGS = [(2, 2), (2, 3), (3, 3), (3, 5), (4, 5)]
MODES = ["xor", "or"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FAULT_SUITE = ["flip", "crop", "substitute", "replay", "drop", "delay"]


def finite(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None


def run_scorecard(images: list[Path], workdir: Path) -> list[dict]:
    rows = []
    for image in images:
        for mode in MODES:
            for k, n in SETTINGS:
                start = time.perf_counter()
                result = run_session(image, SessionConfig(k=k, n=n, recovery=mode, log=False), workdir)
                total_ms = (time.perf_counter() - start) * 1000
                if not result.ok:
                    rows.append({"image": image.stem, "mode": mode, "k": k, "n": n, "status": result.error["code"]})
                    continue
                fid, sec = result.scores["fidelity"], result.scores["secrecy"]
                rows.append({
                    "image": image.stem, "image_mode": result.secret.mode, "mode": mode, "scheme": result.shares[0].scheme,
                    "k": k, "n": n, "status": "ok",
                    "height": result.secret.height, "width": result.secret.width, "channels": result.secret.channels,
                    "psnr_db": fid["psnr_db"], "ssim": fid["ssim"], "bit_exact": fid["bit_exact"],
                    "contrast_retention": fid["contrast_retention"],
                    "vc_contrast": fid.get("vc_relative_contrast"),
                    "min_share_entropy": sec["min_share_entropy"],
                    "max_abs_share_corr": sec["max_abs_share_corr"],
                    "below_threshold_corr": sec["below_threshold_corr"],
                    "below_threshold_ssim": sec["below_threshold_ssim"],
                    "cross_npcr_min": sec["cross_session"]["npcr_min"],
                    "cross_uaci_mean": sec["cross_session"]["uaci_mean"],
                    "cross_max_abs_corr": sec["cross_session"]["max_abs_corr"],
                    "zero_expansion": sec["zero_expansion"],
                    "encode_ms": result.timings["encode_ms"],
                    "decode_ms": result.timings["decode_core_ms"],
                    "verify_ms": result.timings["verify_ms"],
                    "session_ms": total_ms,
                    "share_bytes": result.secret.array.nbytes,
                    "integrity_bytes": 32 * n,
                })
                print(f"  {image.stem:<22} {mode} ({k},{n})  psnr={fid['psnr_db']:.2f}  ssim={fid['ssim']:.3f}  "
                      f"H={sec['min_share_entropy']:.3f}")
    return rows


def run_fault_suite(images: list[Path], workdir: Path, trials: int) -> list[dict]:
    """Inject each fault on one channel; check it is caught and (for k < n) recovery still succeeds."""
    rows = []
    for image in images:
        for mode in MODES:
            for fault in FAULT_SUITE:
                for trial in range(trials):
                    # (2,3): one bad channel must be quarantined while the other two recover the image.
                    result = run_session(image, SessionConfig(
                        k=2, n=3, recovery=mode, faults={2: fault}, persistent_faults=True, max_retries=0,
                        compute_secrecy=False, log=False, fault_seed=trial), workdir)
                    status = result.attempts[-1].checks[1].status if result.attempts else "n/a"
                    caught = status != "VALID"
                    # (2,2): the same fault leaves no quorum, so the session must abort with no output.
                    strict = run_session(image, SessionConfig(
                        k=2, n=2, recovery=mode, faults={1: fault}, persistent_faults=True, max_retries=0,
                        compute_secrecy=False, log=False, fault_seed=trial), workdir)
                    rows.append({
                        "image": image.stem, "mode": mode, "fault": fault, "trial": trial,
                        "detected_status": status, "detected": caught,
                        "recovered_with_quorum": result.ok and bool(
                            result.scores["fidelity"]["bit_exact"] if mode == "xor" else True),
                        "aborted_below_quorum": (not strict.ok) and strict.reconstruction is None,
                        "abort_code": (strict.error or {}).get("code"),
                    })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict], path: Path) -> None:
    ok = [row for row in rows if row["status"] == "ok"]
    labels = [f"({k},{n})" for k, n in SETTINGS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def mean_of(mode, metric):
        values = []
        for k, n in SETTINGS:
            subset = [finite(r[metric]) for r in ok if r["mode"] == mode and r["k"] == k and r["n"] == n]
            subset = [v for v in subset if v is not None]
            values.append(float(np.mean(subset)) if subset else np.nan)
        return values

    axes[0].plot(labels, mean_of("or", "ssim"), "o-", label="OR stacking (enhanced)")
    axes[0].plot(labels, mean_of("xor", "ssim"), "s-", label="XOR / GF(256)")
    axes[0].axhline(0.90, color="gray", ls="--", lw=1, label="target 0.90")
    axes[0].set_title("SSIM vs. threshold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].plot(labels, mean_of("or", "vc_contrast"), "o-", color="tab:blue")
    axes[1].set_title("OR stacking: relative contrast (P_white - P_black)")
    axes[1].set_ylim(0, None)

    axes[2].plot(labels, mean_of("or", "psnr_db"), "o-", label="OR stacking (enhanced)")
    axes[2].set_title("OR stacking: PSNR (dB); XOR path is lossless (inf)")
    axes[2].legend()

    for axis in axes:
        axis.set_xlabel("(k, n)")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarize(score_rows: list[dict], fault_rows: list[dict], images: list[Path], path: Path) -> str:
    ok = [r for r in score_rows if r["status"] == "ok"]
    xor = [r for r in ok if r["mode"] == "xor"]
    orr = [r for r in ok if r["mode"] == "or"]

    def share(rows, predicate):
        hits = sum(1 for r in rows if predicate(r))
        return f"{hits}/{len(rows)}"

    tamper = [r for r in fault_rows if r["fault"] in ("flip", "crop", "substitute", "replay")]
    loss = [r for r in fault_rows if r["fault"] in ("drop", "delay")]
    lines = [
        "# Benchmark summary",
        "",
        f"Images: {len(images)} ({', '.join(p.stem for p in images)})",
        f"Settings: {', '.join(f'({k},{n})' for k, n in SETTINGS)} x modes {', '.join(MODES)}",
        "",
        "## Acceptance targets (PDF section 13)",
        "",
        "| Metric | Target | Result |",
        "|---|---|---|",
        f"| PSNR (XOR path) | > 30 dB | {share(xor, lambda r: r['psnr_db'] > 30)} sessions (all bit-exact: "
        f"{share(xor, lambda r: r['bit_exact'])}) |",
        f"| SSIM (XOR path) | > 0.90 | {share(xor, lambda r: r['ssim'] > 0.90)} |",
        f"| Share entropy, 8-bit shares | > 7.9 bits/pixel | {share(xor, lambda r: r['min_share_entropy'] > 7.9)} "
        f"(min {min(r['min_share_entropy'] for r in xor):.4f}) |",
        f"| Share-to-secret correlation | within +-0.05 | {share(ok, lambda r: abs(r['max_abs_share_corr']) < 0.05)} |",
        f"| Cross-session NPCR, 8-bit shares | > 99% | {share(xor, lambda r: r['cross_npcr_min'] > 99)} "
        f"(min {min(r['cross_npcr_min'] for r in xor):.3f}%) |",
        f"| Cross-session correlation | within +-0.05 | {share(ok, lambda r: r['cross_max_abs_corr'] < 0.05)} |",
        f"| Tamper detection (flip, crop, substitute, replay) | 100% | {share(tamper, lambda r: r['detected'])} |",
        f"| Recovery with one bad channel, (2,3) | succeeds | {share(fault_rows, lambda r: r['recovered_with_quorum'])} |",
        f"| Below-quorum abort, (2,2) with one bad channel | no output | "
        f"{share(fault_rows, lambda r: r['aborted_below_quorum'])} |",
        f"| Loss detection (drop, delay) | 100% | {share(loss, lambda r: r['detected'])} |",
        f"| Below-quorum leakage (k-1 shares combined) | \\|corr\\| < 0.05 | "
        f"{share(ok, lambda r: abs(r['below_threshold_corr']) < 0.05)} |",
        f"| Zero expansion | share = H x W | {share(ok, lambda r: r['zero_expansion'])} |",
        "",
        "## OR stacking (visual path)",
        "",
        "OR shares are binary, so 8-bit entropy and NPCR targets do not apply (ideal values are 1 bit and 50%).",
        "Fidelity is limited by design; the table shows mean values per threshold.",
        "",
        "| (k, n) | VC contrast | SSIM (enhanced) | PSNR dB (enhanced) |",
        "|---|---|---|---|",
    ]
    for k, n in SETTINGS:
        subset = [r for r in orr if r["k"] == k and r["n"] == n]
        if subset:
            lines.append(f"| ({k},{n}) | {np.mean([r['vc_contrast'] for r in subset]):.3f} | "
                         f"{np.mean([r['ssim'] for r in subset]):.3f} | {np.mean([r['psnr_db'] for r in subset]):.2f} |")
    lines += [
        "",
        "## Timing (mean over images)",
        "",
        "| Mode | (k, n) | encode ms | verify ms | decode ms |",
        "|---|---|---|---|---|",
    ]
    for mode in MODES:
        for k, n in SETTINGS:
            subset = [r for r in ok if r["mode"] == mode and r["k"] == k and r["n"] == n]
            if subset:
                lines.append(f"| {mode} | ({k},{n}) | {np.mean([r['encode_ms'] for r in subset]):.1f} | "
                             f"{np.mean([r['verify_ms'] for r in subset]):.1f} | "
                             f"{np.mean([r['decode_ms'] for r in subset]):.1f} |")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    images = sorted(p for p in args.samples.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {args.samples}")
    args.reports.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        print("Scorecard ...")
        score_rows = run_scorecard(images, workdir)
        print("Fault suite ...")
        fault_rows = run_fault_suite(images, workdir, args.trials)

    write_csv(args.reports / "scorecard.csv", score_rows)
    write_csv(args.reports / "fault_suite.csv", fault_rows)
    plot_curves(score_rows, args.reports / "threshold_curves.png")
    print(summarize(score_rows, fault_rows, images, args.reports / "summary.md"))


if __name__ == "__main__":
    main()
