# Benchmark summary

Images: 8 (bridge_gray, clock_gray, house_color, mandrill_color, pattern_gray, peppers_color, secret_text_binary, shapes_color)
Settings: (2,2), (2,3), (3,3), (3,5), (4,5) x modes xor, or

## Acceptance targets (PDF section 13)

| Metric | Target | Result |
|---|---|---|
| PSNR (XOR path) | > 30 dB | 40/40 sessions (all bit-exact: 40/40) |
| SSIM (XOR path) | > 0.90 | 40/40 |
| Share entropy, 8-bit shares | > 7.9 bits/pixel | 40/40 (min 7.9967) |
| Share-to-secret correlation | within +-0.05 | 80/80 |
| Cross-session NPCR, 8-bit shares | > 99% | 40/40 (min 99.551%) |
| Cross-session correlation | within +-0.05 | 80/80 |
| Tamper detection (flip, crop, substitute, replay) | 100% | 192/192 |
| Recovery with one bad channel, (2,3) | succeeds | 288/288 |
| Below-quorum abort, (2,2) with one bad channel | no output | 288/288 |
| Loss detection (drop, delay) | 100% | 96/96 |
| Below-quorum leakage (k-1 shares combined) | \|corr\| < 0.05 | 80/80 |
| Zero expansion | share = H x W | 80/80 |

## OR stacking (visual path)

OR shares are binary, so 8-bit entropy and NPCR targets do not apply (ideal values are 1 bit and 50%).
Fidelity is limited by design; the table shows mean values per threshold.

| (k, n) | VC contrast | SSIM (enhanced) | PSNR dB (enhanced) |
|---|---|---|---|
| (2,2) | 0.501 | 0.374 | 16.38 |
| (2,3) | 0.333 | 0.252 | 14.28 |
| (3,3) | 0.250 | 0.262 | 13.68 |
| (3,5) | 0.092 | 0.122 | 10.50 |
| (4,5) | 0.050 | 0.104 | 9.81 |

## Timing (mean over images)

| Mode | (k, n) | encode ms | verify ms | decode ms |
|---|---|---|---|---|
| xor | (2,2) | 0.6 | 1.9 | 0.1 |
| xor | (2,3) | 1.5 | 2.7 | 0.7 |
| xor | (3,3) | 1.1 | 2.8 | 0.2 |
| xor | (3,5) | 4.1 | 4.5 | 1.0 |
| xor | (4,5) | 6.0 | 4.4 | 1.3 |
| or | (2,2) | 10.0 | 2.0 | 3.8 |
| or | (2,3) | 16.2 | 2.8 | 3.9 |
| or | (3,3) | 14.0 | 2.8 | 3.8 |
| or | (3,5) | 27.7 | 4.6 | 4.2 |
| or | (4,5) | 25.6 | 4.5 | 4.1 |
