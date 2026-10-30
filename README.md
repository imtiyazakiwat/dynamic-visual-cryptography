# Dynamic Visual Cryptography for Secure Image Transmission

This project splits a secret image into **n** noise-like shares. Any **k** of the shares rebuild the image, and
**k − 1** or fewer reveal nothing. Every transmission uses fresh randomness. Every share is SHA-256 tagged and
checked before decoding, and every session is scored and logged so it can be replayed bit for bit.

## Requirements

- **Python 3.12 or newer** (the pinned NumPy 2.5 needs 3.12+). Tested on Python 3.12 and 3.14.
- macOS, Linux, or Windows. No GPU or database needed.
- Check your version with `python3 --version` (Windows: `python --version`).

## How to run

### 1. Get the code

```bash
git clone https://github.com/imtiyazakiwat/dynamic-visual-cryptography.git
cd dynamic-visual-cryptography
```

### 2. Create a virtual environment and install the dependencies

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use
Command Prompt with `.venv\Scripts\activate.bat`.

If `python3` on your machine is older than 3.12, name a newer one explicitly, for example `python3.12 -m venv .venv`.

All commands below assume the virtual environment is active (your prompt shows `(.venv)`). Run them from the
project folder.

### 3. Start the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in a browser (Streamlit usually opens it for you). Then:

1. In the sidebar, choose a sample image or upload your own (PNG, JPEG, TIFF, BMP, up to 2048 × 2048).
2. Choose a threshold preset, for example **(2, 3)**: 3 shares are created and any 2 rebuild the image.
3. Choose **XOR** (exact digital recovery) or **OR** (simulated stacking of printed transparencies).
4. Optional: open **Transmission faults** and break a channel (drop, flip, crop, replay, ...).
5. Click **Run session**.

The **Session** tab shows the 7 pipeline gates, the shares, channel status, the rebuilt image, what an attacker
with too few shares sees, and the scorecard. **History & replay** re-runs any logged session bit for bit.
**Compare sessions** runs several thresholds side by side. Stop the server with `Ctrl+C`.

### 4. Run the tests

```bash
python -m pytest -q
```

Expected: `65 passed`.

### 5. Run the benchmark (optional)

```bash
python bench.py
```

This scores every image in `samples/` at (2,2), (2,3), (3,3), (3,5) and (4,5) in both modes, runs the fault
suite, and writes `reports/scorecard.csv`, `reports/fault_suite.csv`, `reports/threshold_curves.png` and
`reports/summary.md`. Use `python bench.py --trials 1` for a quicker run.

### 6. Command line (optional)

```bash
# Run one full session and save secret, shares, output and manifest into out/
python -m dvc.cli run samples/peppers_color.png --k 3 --n 5 --mode xor --out out/

# Visual (OR) mode with a pixel flipped on channel 1
python -m dvc.cli run samples/secret_text_binary.png --k 2 --n 3 --mode or --fault 1=flip --out out/

# List logged sessions, then replay one by its id
python -m dvc.cli history
python -m dvc.cli replay <session_id>

# All options
python -m dvc.cli run --help
```

Fault kinds: `drop`, `delay`, `flip`, `crop`, `replay`, `substitute`. A fault hits only the first send by default,
so a retry repairs it. Add `--persistent` to make faults hit retransmissions too, and `--retries N` to change the
retry budget (default 1). `run` exits with code 0 on success and 2 if the session aborted.

Sessions are logged to `runs/sessions.jsonl` (created on first run). This folder is ignored by git; see the
security notes below.

## Pipeline (PDF section 10)

| Phase | Module | What it does |
|---|---|---|
| 1. Acquisition | `dvc/preprocess.py` | Detects binary/gray/color, applies EXIF orientation, flattens alpha, enforces 8–2048 px, checks a lossless PNG round trip (PSNR > 40 dB) |
| 2. Entropy | `dvc/entropy.py` | 256-bit seed from `secrets` plus a nonce, bound with HKDF-SHA256 to session id, timestamp, image digest and (k, n, mode); SHAKE-256 keystreams; optional logistic-map mixing (r = 3.99) |
| 3. Encoding | `dvc/encoder.py`, `dvc/gf256.py` | Zero-expansion shares, see below |
| 4. Dispatch | `dvc/manifest.py`, `dvc/transport.py` | Per-share tags, sealed manifest, one directory per channel, fault injector |
| 5. Verification | `dvc/verify.py` | Manifest checksum/HMAC, freshness, parameters, per-share tags, quorum rule; codes `MISSING`, `TAMPER`, `STALE`, `PARAM_MISMATCH`; selective retry |
| 6. Reconstruction | `dvc/reconstruct.py` | XOR / GF(256) combine (bit-exact) or OR stacking with an enhanced viewing copy |
| 7. Scoring & log | `dvc/metrics.py`, `dvc/session.py` | PSNR, SSIM, Michelson contrast, entropy, correlation, NPCR, UACI; JSONL log; bit-exact replay |

### Share schemes

| Recovery mode | (k, n) | Scheme | Result |
|---|---|---|---|
| XOR | k = n | n − 1 random shares + secret XOR all of them | Bit-exact |
| XOR | k < n | Per-byte polynomial sharing over GF(2⁸) (Shamir), decoded with XOR and table lookups | Bit-exact |
| OR | any | Floyd–Steinberg halftone, then the Chen–Tsao (k, n) random-grid rule | Visual: the secret appears as a contrast difference |

Plain random-grid XOR is only lossless when k = n. To meet the PDF's XOR-path targets (PSNR > 30 dB,
SSIM > 0.90) for (2, 3) and (3, 5), XOR mode with k < n uses GF(256) sharing. Its shares are the same size and
uniformly random, and decoding is still lightweight.

## Results (`reports/summary.md`, 8 images × 5 thresholds × 2 modes)

| Target (PDF section 13) | Result |
|---|---|
| PSNR > 30 dB, SSIM > 0.90 (XOR path) | 40/40 sessions, all bit-exact |
| Share entropy > 7.9 bits (8-bit shares) | 40/40, min 7.997 |
| Share-to-secret \|corr\| < 0.05 | 80/80 |
| Cross-session NPCR > 99% (8-bit shares) | 40/40, min 99.56% |
| Tamper detection (flip, crop, substitute, replay) | 192/192 |
| Recovery with one bad channel at (2, 3) | 288/288 |
| Clean abort below quorum | 288/288 |
| Zero expansion | 80/80 |

OR shares are binary, so their entropy is 1 bit and their ideal cross-session NPCR is 50%. The 8-bit targets
don't apply to them. Their relative contrast falls as k grows (0.50 at (2,2), 0.33 at (2,3), 0.09 at (3,5)),
which is a known property of random-grid schemes and is plotted in `reports/threshold_curves.png`.

## Layout

```
dvc/            library (one module per pipeline layer)
app.py          Streamlit dashboard
bench.py        benchmark -> reports/
tests/          pytest suite
samples/        USC-SIPI Mandrill, Peppers, House, Clock, Bridge + synthetic test images
reports/        latest benchmark output
runs/           created at runtime (git-ignored): channel folders, sessions.jsonl, logged inputs
```

`samples/` contains the USC-SIPI test images Mandrill, Peppers, House, Clock and Bridge
(https://sipi.usc.edu/database/), plus three synthetic images made for this project.

## Security notes

- `runs/sessions.jsonl` stores the session seed by default so sessions can be replayed. Seed + logged input =
  every share. Treat `runs/` like the secret images, or turn off "Store seed" (CLI: `--no-seed`), which logs only
  a SHA-256 seed commitment.
- The manifest has a SHA-256 checksum. Set `DVC_MANIFEST_KEY` on sender and receiver to add an HMAC-SHA256
  signature, so an attacker can't rewrite the manifest and tags together.
- Channels are simulated with local folders. A live network, video, and multiple secrets per share set are out of
  scope (PDF section 14).
- Lena (USC-SIPI 4.2.04) is no longer distributed by USC-SIPI and isn't included. Medical and biometric samples
  aren't included either. Put any images in `samples/` and `bench.py` will include them.
