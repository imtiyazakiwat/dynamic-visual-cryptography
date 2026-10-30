import io
import json

import numpy as np
import pytest
from PIL import Image

from dvc.errors import (DIMENSION_LIMIT, INVALID_PARAMS, MISSING, PARAM_MISMATCH, REPLAY_MISMATCH, STALE, TAMPER,
                        UNSUPPORTED_MODE, DVCError)
from dvc.manifest import MANIFEST_KEY_ENV
from dvc.preprocess import load_image, null_path_psnr
from dvc.session import SessionConfig, read_log, replay, run_session, save_artifacts


def cfg(**kwargs):
    kwargs.setdefault("compute_secrecy", False)
    return SessionConfig(**kwargs)


# ---------------------------------------------------------------- preprocessing

def test_mode_detection(color_image, gray_image, binary_image):
    assert load_image(color_image).mode == "color"
    assert load_image(gray_image).mode == "gray"
    assert load_image(binary_image).mode == "binary"
    assert load_image(binary_image.convert("1")).mode == "binary"
    gray_as_rgb = load_image(gray_image.convert("RGB"))
    assert gray_as_rgb.mode == "gray" and gray_as_rgb.channels == 1


def test_alpha_is_flattened_onto_white():
    rgba = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
    canonical = load_image(rgba)
    assert canonical.alpha_flattened
    assert np.all(canonical.array == 255)


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (40, 20), (10, 200, 30))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    canonical = load_image(buffer.getvalue())
    assert canonical.orientation_applied
    assert (canonical.width, canonical.height) == (20, 40)


def test_bad_inputs_get_typed_errors():
    with pytest.raises(DVCError) as info:
        load_image(b"definitely not an image")
    assert info.value.code == UNSUPPORTED_MODE
    with pytest.raises(DVCError) as info:
        load_image(Image.new("L", (2049, 16)))
    assert info.value.code == DIMENSION_LIMIT
    with pytest.raises(DVCError) as info:
        load_image(Image.new("L", (4, 4)))
    assert info.value.code == DIMENSION_LIMIT
    with pytest.raises(DVCError) as info:
        load_image(np.zeros((10, 10), dtype=np.float32))
    assert info.value.code == UNSUPPORTED_MODE


def test_null_path_is_lossless(color_image):
    assert null_path_psnr(load_image(color_image)) == float("inf")


# ---------------------------------------------------------------- happy paths

@pytest.mark.parametrize("k,n,mode", [(2, 2, "xor"), (2, 3, "xor"), (3, 5, "xor"), (2, 3, "or"), (3, 5, "or")])
def test_full_session_succeeds(color_image, workdir, k, n, mode):
    result = run_session(color_image, SessionConfig(k=k, n=n, recovery=mode), workdir)
    assert result.ok, result.error
    assert all(state == "passed" for state in result.gates.values())
    assert len(result.shares) == n
    assert all(share.array.shape == result.secret.array.shape for share in result.shares)
    targets = {t["metric"]: t for t in result.scores["targets"]}
    assert all(t["passed"] in (True, None) for t in targets.values()), targets
    if mode == "xor":
        assert result.scores["fidelity"]["bit_exact"]
        assert targets["Share entropy (bits/pixel)"]["passed"]
        assert targets["Cross-session NPCR (%)"]["passed"]


def test_gray_and_binary_sessions(gray_image, binary_image, workdir):
    for image in (gray_image, binary_image):
        for mode in ("xor", "or"):
            result = run_session(image, cfg(k=2, n=3, recovery=mode), workdir)
            assert result.ok, result.error


def test_sessions_are_unlinkable(color_image, workdir):
    a = run_session(color_image, cfg(), workdir)
    b = run_session(color_image, cfg(), workdir)
    assert a.session_id != b.session_id
    assert a.key.seed != b.key.seed
    for x, y in zip(a.shares, b.shares):
        assert np.mean(x.array != y.array) > 0.99


# ---------------------------------------------------------------- fault handling

def test_one_lost_share_is_tolerated(color_image, workdir):
    result = run_session(color_image, cfg(k=2, n=3, faults={1: "drop"}, max_retries=0), workdir)
    assert result.ok
    assert result.verification.admitted == [2, 3]
    assert result.verification.checks[0].status == MISSING


@pytest.mark.parametrize("fault,code", [("flip", TAMPER), ("crop", TAMPER), ("substitute", TAMPER), ("replay", STALE)])
def test_each_corruption_class_is_quarantined(color_image, workdir, fault, code):
    result = run_session(color_image, cfg(k=2, n=3, faults={2: fault}, persistent_faults=True, max_retries=0), workdir)
    assert result.ok
    assert result.verification.checks[1].status == code
    assert 2 not in result.verification.admitted
    assert result.scores["fidelity"]["bit_exact"]


@pytest.mark.parametrize("fault", ["flip", "crop", "substitute", "replay"])
def test_corruption_below_quorum_aborts_with_no_output(color_image, workdir, fault):
    result = run_session(color_image, cfg(k=2, n=2, faults={1: fault}, persistent_faults=True, max_retries=1), workdir)
    assert not result.ok
    assert result.error["code"] == MISSING
    assert result.error["phase"] == "verify"
    assert result.reconstruction is None


def test_or_mode_tamper_is_detected(color_image, workdir):
    result = run_session(color_image, cfg(k=2, n=3, recovery="or", faults={1: "flip", 3: "crop"},
                                          persistent_faults=True), workdir)
    assert not result.ok and result.error["code"] == MISSING
    assert [c.status for c in result.verification.checks] == [TAMPER, "VALID", TAMPER]


def test_selective_retry_resends_only_failed_shares(color_image, workdir):
    result = run_session(color_image, cfg(k=3, n=3, faults={1: "drop", 3: "flip"}, max_retries=1), workdir)
    assert result.ok
    assert len(result.attempts) == 2
    retried = [r["index"] for r in result.receipts if r["attempt"] == 2]
    assert sorted(retried) == [1, 3]


def test_delay_is_a_timeout(color_image, workdir):
    result = run_session(color_image, cfg(k=2, n=2, faults={2: "delay"}, max_retries=0), workdir)
    assert not result.ok and result.error["code"] == MISSING
    assert result.receipts[1]["status"] == "timeout"


def test_tampered_manifest_aborts(color_image, workdir):
    result = run_session(color_image, cfg(manifest_fault="tamper"), workdir)
    assert not result.ok and result.error["code"] == TAMPER


def test_stale_manifest_aborts(color_image, workdir):
    result = run_session(color_image, cfg(freshness_window=1e-9), workdir)
    assert not result.ok and result.error["code"] == STALE


def test_parameter_mismatch_aborts(color_image, workdir):
    result = run_session(color_image, cfg(k=2, n=3, expect={"k": 3}), workdir)
    assert not result.ok and result.error["code"] == PARAM_MISMATCH


def test_invalid_parameters_abort_in_preprocess(color_image, workdir):
    result = run_session(color_image, cfg(k=4, n=3), workdir)
    assert not result.ok and result.error["code"] == INVALID_PARAMS
    assert result.error["phase"] == "preprocess"


def test_hmac_signed_manifest(color_image, workdir, monkeypatch):
    monkeypatch.setenv(MANIFEST_KEY_ENV, "lab-secret")
    result = run_session(color_image, cfg(), workdir)
    assert result.ok and "signature" in result.manifest
    manifest_path = result.channel_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["k"] = 3
    from dvc.manifest import seal_manifest
    manifest_path.write_text(json.dumps(seal_manifest(manifest)))  # attacker reseals without the key
    from dvc.verify import verify_manifest
    with pytest.raises(DVCError) as info:
        verify_manifest(json.loads(manifest_path.read_text()), key=b"lab-secret")
    assert info.value.code == TAMPER


# ---------------------------------------------------------------- logging and replay

def test_replay_is_bit_exact(color_image, workdir):
    for mode in ("xor", "or"):
        result = run_session(color_image, cfg(k=2, n=3, recovery=mode), workdir)
        outcome = replay(result.session_id, workdir)
        assert outcome["bit_exact"]
        assert outcome["output_match"]


def test_replay_disabled_without_seed(color_image, workdir):
    result = run_session(color_image, cfg(store_seed=False), workdir)
    record = read_log(workdir)[-1]
    assert record["seed"] is None and record["seed_commitment"]
    with pytest.raises(DVCError) as info:
        replay(result.session_id, workdir)
    assert info.value.code == REPLAY_MISMATCH


def test_aborted_sessions_are_logged(color_image, workdir):
    run_session(color_image, cfg(k=2, n=2, faults={1: "drop"}, max_retries=0), workdir)
    record = read_log(workdir)[-1]
    assert record["status"] == "aborted"
    assert record["error"]["code"] == MISSING
    assert record["gates"]["reconstruct"] == "skipped"


def test_save_artifacts(color_image, workdir, tmp_path):
    result = run_session(color_image, cfg(k=2, n=3, recovery="or"), workdir)
    files = {path.name for path in save_artifacts(result, tmp_path / "out")}
    assert {"secret.png", "share_1.png", "share_3.png", "reconstructed.png", "manifest.json"} <= files
