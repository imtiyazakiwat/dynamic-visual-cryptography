"""Operator dashboard for Dynamic Visual Cryptography for Secure Image Transmission.

Run with:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from dvc.encoder import MAX_SHARES
from dvc.errors import EXPLANATIONS
from dvc.preprocess import array_to_pil
from dvc.reconstruct import stack_or
from dvc.session import DEFAULT_WORKDIR, PHASES, SessionConfig, read_log, replay, run_session
from dvc.transport import FAULT_DESCRIPTIONS, FAULT_KINDS

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
WORKDIR = DEFAULT_WORKDIR

st.set_page_config(page_title="Dynamic Visual Cryptography", layout="wide")

GATE_ICON = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "running": "⏳"}
STATUS_ICON = {"VALID": "✅", "MISSING": "📭", "TAMPER": "🛑", "STALE": "⌛", "PARAM_MISMATCH": "⚠️"}


def to_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    array_to_pil(array).save(buffer, format="PNG")
    return buffer.getvalue()


def fmt(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isinf(value):
            return "∞ (lossless)"
        return f"{value:.4f}"
    if isinstance(value, str) and value in ("inf", "Infinity"):
        return "∞ (lossless)"
    return str(value)


def sample_files() -> list[Path]:
    if not SAMPLES.exists():
        return []
    return sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


# ------------------------------------------------------------------ sidebar

st.sidebar.title("Session setup")

source_kind = st.sidebar.radio("Secret image", ["Sample", "Upload"], horizontal=True)
image_source = None
image_label = ""
if source_kind == "Sample":
    files = sample_files()
    if files:
        choice = st.sidebar.selectbox("Sample image", files, format_func=lambda p: p.stem)
        image_source = choice
        image_label = choice.stem
    else:
        st.sidebar.warning("No images in samples/.")
else:
    upload = st.sidebar.file_uploader("PNG, JPEG, TIFF or BMP (max 2048 x 2048)",
                                      type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if upload is not None:
        image_source = upload.getvalue()
        image_label = upload.name

preset = st.sidebar.selectbox(
    "Threshold preset",
    ["(2, 3) single-fault tolerance", "(2, 2) two-party escrow", "(3, 5) committee custody", "Custom"],
)
if preset == "Custom":
    n = st.sidebar.number_input("n (shares created)", min_value=2, max_value=MAX_SHARES, value=4)
    k = st.sidebar.number_input("k (shares needed)", min_value=2, max_value=int(n), value=min(3, int(n)))
else:
    k, n = {"(2, 3) single-fault tolerance": (2, 3), "(2, 2) two-party escrow": (2, 2),
            "(3, 5) committee custody": (3, 5)}[preset]
k, n = int(k), int(n)

recovery_label = st.sidebar.radio(
    "Recovery mode", ["XOR (digital, lossless)", "OR (visual stacking)"],
    help="XOR rebuilds the exact image. OR simulates stacking printed transparencies: "
         "the secret appears as a contrast difference, dimmer and noisier by design.",
)
recovery = "xor" if recovery_label.startswith("XOR") else "or"

with st.sidebar.expander("Transmission faults (simulation)"):
    st.caption("Inject a problem on any channel to see how the receiver reacts.")
    faults = {}
    for index in range(1, n + 1):
        kind = st.selectbox(f"Channel {index}", FAULT_KINDS, key=f"fault_{index}_{n}",
                            format_func=lambda f: f"{f} – {FAULT_DESCRIPTIONS[f]}")
        if kind != "none":
            faults[index] = kind
    persistent = st.checkbox("Faults also hit retransmissions", value=False)
    retries = st.number_input("Retry budget (selective retransmission)", min_value=0, max_value=3, value=1)
    tamper_manifest = st.checkbox("Tamper with the manifest", value=False)

with st.sidebar.expander("Advanced"):
    chaos = st.checkbox("Mix logistic-map stream (auxiliary whitening)", value=False)
    store_seed = st.checkbox("Store seed in log (enables bit-exact replay)", value=True,
                             help="Anyone with the seed and the input can regenerate the shares. "
                                  "Off = only a seed commitment is logged and replay is disabled.")
    sharpen = st.checkbox("Unsharp-mask OR output", value=False)
    secrecy = st.checkbox("Cross-session secrecy test (re-encrypts once more)", value=True)

run_clicked = st.sidebar.button("Run session", type="primary", disabled=image_source is None,
                                use_container_width=True)

# ------------------------------------------------------------------ header

st.title("Dynamic Visual Cryptography for Secure Image Transmission")
st.caption(f"Any **{k}** of **{n}** shares rebuild the image; {k - 1} or fewer reveal nothing. "
           "Every session uses fresh 256-bit randomness, SHA-256-tagged shares, and strict quorum gating.")

tab_run, tab_history, tab_compare, tab_help = st.tabs(["Session", "History & replay", "Compare sessions", "How it works"])

if run_clicked and image_source is not None:
    config = SessionConfig(k=k, n=n, recovery=recovery, chaos=chaos, faults=faults, persistent_faults=persistent,
                           max_retries=int(retries), store_seed=store_seed, sharpen=sharpen,
                           manifest_fault="tamper" if tamper_manifest else "none", compute_secrecy=secrecy)
    with st.spinner("Running all seven phases ..."):
        st.session_state["result"] = run_session(image_source, config, WORKDIR)
        st.session_state["label"] = image_label

# ------------------------------------------------------------------ session tab

with tab_run:
    result = st.session_state.get("result")
    if result is None:
        st.info("Choose an image and settings in the sidebar, then press **Run session**.")
    else:
        config = result.config
        head = st.columns([2, 1, 1, 1])
        if result.ok:
            head[0].success(f"Session **{result.session_id}** delivered and verified.")
        else:
            error = result.error
            head[0].error(f"Session aborted with **{error['code']}** in phase *{error['phase']}*.\n\n"
                          f"{error['message']}\n\n{EXPLANATIONS.get(error['code'], '')}")
        head[1].metric("Threshold", f"({config.k}, {config.n})")
        head[2].metric("Mode", config.recovery.upper())
        total = sum(v for key, v in result.timings.items() if key.endswith("_ms") and key != "decode_core_ms")
        head[3].metric("Session time", f"{total:.0f} ms")

        st.subheader("Pipeline gates")
        gate_cols = st.columns(len(PHASES))
        for column, (phase, title) in zip(gate_cols, PHASES):
            state = result.gates.get(phase, "skipped")
            timing = result.timings.get(f"{phase}_ms")
            column.markdown(f"{GATE_ICON.get(state, '')} **{title}**  \n{state}"
                            + (f" · {timing:.1f} ms" if timing is not None else ""))

        if result.secret is not None:
            st.subheader("Secret image")
            info = st.columns([1, 2])
            info[0].image(to_png(result.secret.array), caption=f"{st.session_state.get('label', '')}", width="stretch")
            info[1].markdown(
                f"- Detected mode: **{result.secret.mode}** (source `{result.secret.source_mode}`)\n"
                f"- Size: **{result.secret.width} × {result.secret.height}**, {result.secret.channels} channel(s)\n"
                f"- EXIF orientation applied: {fmt(result.secret.orientation_applied)}; "
                f"alpha flattened: {fmt(result.secret.alpha_flattened)}\n"
                f"- Input SHA-256: `{result.secret.digest()[:32]}…`\n"
                + (f"- Share scheme: **{result.shares[0].scheme}**\n" if result.shares else "")
                + (f"- Session nonce: `{result.key.nonce.hex()}`\n- Seed commitment: `{result.key.seed_commitment()[:32]}…`"
                   if result.key else "")
            )

        if result.shares:
            st.subheader("Shares (each one alone is noise)")
            per_row = min(5, len(result.shares))
            rows = [result.shares[i:i + per_row] for i in range(0, len(result.shares), per_row)]
            tags = {entry["index"]: entry["tag"] for entry in result.manifest["tags"]} if result.manifest else {}
            for row in rows:
                columns = st.columns(per_row)
                for column, share in zip(columns, row):
                    column.image(to_png(share.array), caption=f"Share {share.index}", width="stretch")
                    column.caption(f"tag `{tags.get(share.index, '')[:16]}…`")

        if result.receipts:
            st.subheader("Channels and admission")
            final = result.verification
            checks = {check.index: check for check in final.checks} if final else {}
            table = []
            for receipt in result.receipts:
                table.append({
                    "share": receipt["index"], "channel": receipt["channel"], "attempt": receipt["attempt"],
                    "transport": receipt["status"], "injected fault": receipt["fault"],
                    "bytes": receipt["bytes_written"],
                })
            left, right = st.columns(2)
            left.markdown("**Dispatch receipts**")
            left.dataframe(table, hide_index=True, width="stretch")
            if final:
                right.markdown(f"**Verification (attempt {len(result.attempts)})** — "
                               f"admitted {len(final.admitted)} of {final.n}, need {final.k}")
                right.dataframe([{"share": c.index, "status": f"{STATUS_ICON.get(c.status, '')} {c.status}",
                                  "detail": c.detail} for c in final.checks], hide_index=True, width="stretch")
                if len(result.attempts) > 1:
                    first = result.attempts[0]
                    right.caption("First attempt: " + ", ".join(f"#{c.index} {c.status}" for c in first.checks)
                                  + ". Only the failed shares were retransmitted.")

        if result.ok:
            recon = result.reconstruction
            st.subheader("Reconstruction")
            columns = st.columns(3)
            columns[0].image(to_png(result.secret.array), caption="Original", width="stretch")
            if config.recovery == "or":
                columns[1].image(to_png(recon.raw), caption=f"Raw OR stack of shares {recon.used_indices}",
                                 width="stretch")
                columns[2].image(to_png(recon.display), caption="Enhanced for viewing (low-pass + stretch)",
                                 width="stretch")
            else:
                columns[1].image(to_png(recon.display), caption=f"Recovered from shares {recon.used_indices}",
                                 width="stretch")
                columns[2].image(to_png(result.below_threshold),
                                 caption=f"Attacker's view: decoder on only {config.k - 1} share(s)", width="stretch")
            if config.recovery == "or":
                st.image(to_png(result.below_threshold), width=360,
                         caption=f"Attacker's view: stacking only {config.k - 1} share(s) shows nothing")

            st.subheader("Scorecard")
            targets = result.scores["targets"]
            st.dataframe([{
                "metric": t["metric"], "value": fmt(t["value"]), "target": t["target"],
                "result": {True: "✅ pass", False: "❌ fail", None: "— n/a"}[t["passed"]], "note": t["note"],
            } for t in targets], hide_index=True, width="stretch")

            with st.expander("Per-share secrecy details"):
                st.dataframe([{
                    "share": item["index"], "entropy (bits)": round(item["entropy_bits"], 4),
                    "corr. to secret": round(item["corr_to_secret"], 4),
                    "adjacent corr. (h / v / d)": " / ".join(f"{v:.3f}" for v in item["adjacent_corr"].values()),
                } for item in result.scores["secrecy"]["per_share"]], hide_index=True, width="stretch")
                cross = result.scores["secrecy"].get("cross_session")
                if cross:
                    st.write(f"Cross-session test (same image, fresh session): NPCR min {cross['npcr_min']:.3f}%, "
                             f"UACI mean {cross['uaci_mean']:.3f}%, max |corr| {cross['max_abs_corr']:.4f}")

            with st.expander("Interactive stacking: choose any subset of shares"):
                chosen = st.multiselect("Shares to combine", [s.index for s in result.shares],
                                        default=[s.index for s in result.shares][: config.k], key="subset")
                if chosen:
                    from dvc.reconstruct import combine_gf256, combine_xor, enhance_or
                    arrays = [result.shares[i - 1].array for i in chosen]
                    scheme = result.shares[0].scheme
                    if scheme == "rg-or":
                        combined = enhance_or(stack_or(arrays))
                    elif scheme == "gf256":
                        combined = combine_gf256(arrays, chosen)
                    else:
                        combined = combine_xor(arrays)
                    enough = len(chosen) >= config.k if scheme != "xor-nn" else len(chosen) == config.n
                    st.image(to_png(combined), width=420,
                             caption=("Quorum met: secret visible" if enough else "Below quorum: noise only"))

            downloads = io.BytesIO()
            with zipfile.ZipFile(downloads, "w") as archive:
                for share in result.shares:
                    archive.writestr(f"share_{share.index}.png", to_png(share.array))
                archive.writestr("manifest.json", json.dumps(result.manifest, indent=2))
                archive.writestr("reconstructed.png", to_png(recon.display))
            st.download_button("Download shares + manifest (.zip)", downloads.getvalue(),
                               file_name=f"dvc_{result.session_id}.zip", mime="application/zip")

# ------------------------------------------------------------------ history tab

with tab_history:
    records = read_log(WORKDIR)
    if not records:
        st.info("No sessions logged yet.")
    else:
        st.caption(f"Log file: `{WORKDIR / 'sessions.jsonl'}` (one JSON line per session)")
        st.dataframe([{
            "session": r.get("session_id"), "status": r["status"],
            "error": (r.get("error") or {}).get("code", ""),
            "k": r["config"]["k"], "n": r["config"]["n"], "mode": r["config"]["recovery"],
            "image": r.get("image_mode", ""), "faults": ", ".join(f"#{i}:{f}" for i, f in r["config"]["faults"].items()),
            "replayable": bool(r.get("seed")) and bool(r.get("tags")),
        } for r in reversed(records)], hide_index=True, width="stretch")

        replayable = [r["session_id"] for r in reversed(records) if r.get("seed") and r.get("tags")]
        if replayable:
            target = st.selectbox("Session to replay", replayable)
            if st.button("Replay bit-exact from log"):
                try:
                    outcome = replay(target, WORKDIR)
                    if outcome["bit_exact"]:
                        st.success(f"Replay of {target} is bit-exact: all {len(outcome['tag_matches'])} share tags "
                                   f"and the output digest match ({outcome['replay_ms']:.0f} ms).")
                    else:
                        st.error(f"Replay mismatch: tags {outcome['tag_matches']}, output {outcome['output_match']}")
                    if outcome["reconstruction"] is not None:
                        st.image(to_png(outcome["reconstruction"].display), width=360, caption="Replayed output")
                except Exception as exc:  # surface any replay failure to the operator
                    st.error(str(exc))

# ------------------------------------------------------------------ compare tab

with tab_compare:
    st.markdown("Run the same image through several thresholds and both modes, then compare their scores.")
    if image_source is None:
        st.info("Select an image in the sidebar first.")
    elif st.button("Run comparison: (2,2), (2,3), (3,5) × XOR, OR"):
        rows = []
        previews = []
        progress = st.progress(0.0)
        settings = [(2, 2), (2, 3), (3, 5)]
        total = len(settings) * 2
        done = 0
        for mode in ("xor", "or"):
            for kk, nn in settings:
                res = run_session(image_source, SessionConfig(k=kk, n=nn, recovery=mode), WORKDIR)
                done += 1
                progress.progress(done / total)
                if not res.ok:
                    rows.append({"setting": f"({kk},{nn}) {mode}", "session": f"aborted: {res.error['code']}"})
                    continue
                fid, sec = res.scores["fidelity"], res.scores["secrecy"]
                rows.append({
                    "setting": f"({kk},{nn}) {mode}", "session": res.session_id,
                    "PSNR dB": fmt(fid["psnr_db"]), "SSIM": round(fid["ssim"], 4),
                    "VC contrast": f'{fid["vc_relative_contrast"]:.3f}' if "vc_relative_contrast" in fid else "—",
                    "min entropy": round(sec["min_share_entropy"], 4),
                    "max |corr|": round(sec["max_abs_share_corr"], 4),
                    "NPCR min %": round(sec["cross_session"]["npcr_min"], 3),
                    "encode ms": round(res.timings["encode_ms"], 1),
                    "decode ms": round(res.timings["decode_core_ms"], 1),
                })
                previews.append((f"({kk},{nn}) {mode}", res.reconstruction.display))
        st.dataframe(rows, hide_index=True, width="stretch")
        columns = st.columns(len(previews) or 1)
        for column, (label, array) in zip(columns, previews):
            column.image(to_png(array), caption=label, width="stretch")
    curves = ROOT / "reports" / "threshold_curves.png"
    if curves.exists():
        st.subheader("Threshold operating curves (from bench.py over all samples)")
        st.image(str(curves), width="stretch")

# ------------------------------------------------------------------ help tab

with tab_help:
    st.markdown("""
### What happens in a session

1. **Acquisition & normalization**: the image is read, rotated per EXIF, alpha is flattened onto white, and the
   mode (binary, gray, color) is detected. A lossless round-trip check (PSNR > 40 dB) certifies the input.
2. **Session entropy**: 256 fresh bits from the operating system's CSPRNG plus a nonce are bound to the session
   (timestamp, image digest, parameters) with HKDF-SHA256. SHAKE-256 expands them into keystreams. An optional
   logistic map (r = 3.99) can be mixed in, but secrecy never depends on it.
3. **Threshold encoding** (shares are exactly the image size):
   * **XOR, k = n**: n−1 shares are pure random bytes; the last is the image XOR all of them.
   * **XOR, k < n**: every pixel byte becomes a random polynomial over GF(256); share *i* stores its value at *i*.
     Any k shares rebuild the image exactly with XOR and table lookups.
   * **OR**: each channel is halftoned to dots, then the Chen–Tsao random-grid rule splits every dot across n shares.
     Stacking k transparencies shows the image as darker/lighter regions.
4. **Tagging & dispatch**: each share gets a SHA-256 tag bound to its session and position. A manifest lists
   (k, n), size, nonce, timestamp and all tags, sealed with a checksum (and an HMAC if `DVC_MANIFEST_KEY` is set).
   Shares go to separate channel folders; faults can be injected here.
5. **Admission gating**: the receiver checks the manifest, freshness, and every share. Bad shares are labelled
   **MISSING**, **TAMPER**, **STALE** or **PARAM_MISMATCH**; only failed shares are re-sent. With fewer than k valid
   shares the session aborts: no image is better than a wrong image.
6. **Reconstruction**: XOR / GF(256) output is bit-exact. OR output is the raw stack plus an enhanced view.
7. **Scoring & logging**: PSNR, SSIM, contrast, entropy, correlation, NPCR/UACI are computed and the whole
   session is appended to `runs/sessions.jsonl` for bit-exact replay.

### Choosing settings
* **(2, 2)** two-party escrow · **(2, 3)** survives one lost channel · **(3, 5)** committee custody, survives two.
* Use **XOR** for archival fidelity; use **OR** to demonstrate classical visual stacking.

### Security note
With "Store seed in log" on, `runs/` contains everything needed to regenerate the shares. Treat it like the
secret image itself, or turn the option off for real data.
""")
