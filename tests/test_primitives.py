import hashlib
import hmac
import itertools
import math

import numpy as np
import pytest

from dvc import gf256, metrics
from dvc.encoder import encode, scheme_for, validate_params
from dvc.entropy import SessionKey, hkdf_expand, hkdf_extract, logistic_stream
from dvc.errors import INVALID_PARAMS, DVCError
from dvc.reconstruct import combine_gf256, combine_xor, stack_or


def fixed_key(k=2, n=3, mode="xor", chaos=False):
    return SessionKey(seed=b"\x01" * 32, nonce=b"\x02" * 16, timestamp=1_700_000_000.0,
                      session_id="fixture", image_digest="d" * 64, k=k, n=n, mode=mode, chaos=chaos)


# ---------------------------------------------------------------- HKDF / entropy

def test_hkdf_rfc5869_case1():
    ikm = bytes.fromhex("0b" * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    prk = hkdf_extract(salt, ikm)
    assert prk.hex() == "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
    okm = hkdf_expand(prk, info, 42)
    assert okm.hex() == ("3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
                         "34007208d5b887185865")


def test_keystream_deterministic_and_label_separated():
    key = fixed_key()
    a = key.keystream("x", 1000)
    assert np.array_equal(a, fixed_key().keystream("x", 1000))
    assert not np.array_equal(a, key.keystream("y", 1000))
    assert key.keystream("x", 0).size == 0


def test_fresh_sessions_never_repeat():
    keys = [SessionKey.fresh("d", 2, 3, "xor", False) for _ in range(20)]
    assert len({key.seed for key in keys}) == 20
    assert len({key.nonce for key in keys}) == 20
    assert len({key.session_id for key in keys}) == 20


def test_logistic_stream_is_seeded_and_balanced():
    a = logistic_stream(b"k1", 50_000)
    assert np.array_equal(a, logistic_stream(b"k1", 50_000))
    assert not np.array_equal(a, logistic_stream(b"k2", 50_000))
    assert metrics.shannon_entropy(a) > 7.9


def test_chaos_mixing_changes_stream_but_keeps_entropy():
    plain = fixed_key(chaos=False).keystream("x", 65536)
    mixed = fixed_key(chaos=True).keystream("x", 65536)
    assert not np.array_equal(plain, mixed)
    assert metrics.shannon_entropy(mixed) > 7.99


# ---------------------------------------------------------------- GF(256)

def test_gf256_known_products_and_inverses():
    assert gf256.gf_mul(0x57, 0x83) == 0xC1  # FIPS-197 example
    assert gf256.gf_mul(0x57, 0x13) == 0xFE
    for a in range(1, 256):
        assert gf256.gf_mul(a, gf256.gf_inv(a)) == 1
    assert gf256.gf_mul(0, 77) == 0
    with pytest.raises(ZeroDivisionError):
        gf256.gf_inv(0)


def test_lagrange_rejects_bad_coordinates():
    with pytest.raises(ValueError):
        gf256.lagrange_at_zero([1, 1])
    with pytest.raises(ValueError):
        gf256.lagrange_at_zero([0, 2])


# ---------------------------------------------------------------- parameter validation

@pytest.mark.parametrize("k,n,mode", [(1, 3, "xor"), (4, 3, "xor"), (2, 17, "or"), (2, 3, "and"), (2.0, 3, "xor")])
def test_invalid_params_rejected(k, n, mode):
    with pytest.raises(DVCError) as info:
        validate_params(k, n, mode)
    assert info.value.code == INVALID_PARAMS


def test_scheme_selection():
    assert scheme_for(2, 2, "xor") == "xor-nn"
    assert scheme_for(3, 3, "xor") == "xor-nn"
    assert scheme_for(2, 3, "xor") == "gf256"
    assert scheme_for(3, 5, "or") == "rg-or"


# ---------------------------------------------------------------- threshold correctness

SECRET = np.random.default_rng(7).integers(0, 256, size=(32, 40, 3), dtype=np.uint8)


@pytest.mark.parametrize("k,n", [(2, 2), (3, 3), (2, 3), (3, 5), (4, 6), (5, 5)])
def test_xor_modes_recover_every_quorum_bit_exact(k, n):
    key = fixed_key(k, n)
    shares = encode(SECRET, "color", key, k, n, "xor")
    assert all(share.array.shape == SECRET.shape for share in shares)
    for subset in itertools.combinations(shares, k):
        arrays = [share.array for share in subset]
        indices = [share.index for share in subset]
        if k == n:
            recovered = combine_xor(arrays)
        else:
            recovered = combine_gf256(arrays, indices)
        assert np.array_equal(recovered, SECRET)
    if k < n:
        # Any k+1 shares also work: the decoder uses the first k.
        recovered = combine_gf256([s.array for s in shares[:k]], [s.index for s in shares[:k]])
        assert np.array_equal(recovered, SECRET)


@pytest.mark.parametrize("k,n", [(2, 2), (2, 3), (3, 5)])
def test_xor_modes_below_threshold_reveal_nothing(k, n):
    secret = np.zeros((128, 128, 1), dtype=np.uint8)
    secret[32:96, 32:96] = 255
    shares = encode(secret, "gray", fixed_key(k, n), k, n, "xor")
    for subset in itertools.combinations(shares, k - 1):
        arrays = [share.array for share in subset]
        attempt = combine_xor(arrays) if len(arrays) > 1 else arrays[0]
        assert abs(metrics.pearson(attempt, secret)) < 0.05
        assert metrics.shannon_entropy(attempt) > 7.9


def test_gf256_shares_are_uniform_for_constant_secret():
    # The hardest case for leakage: a constant secret. Every single share must still look uniform.
    secret = np.full((128, 128, 1), 200, dtype=np.uint8)
    shares = encode(secret, "gray", fixed_key(3, 5), 3, 5, "xor")
    for share in shares:
        assert metrics.shannon_entropy(share.array) > 7.98


@pytest.mark.parametrize("k,n", [(2, 2), (2, 3), (3, 3), (3, 5), (4, 5)])
def test_random_grid_or_contrast_and_secrecy(k, n):
    secret = np.full((128, 128, 1), 255, dtype=np.uint8)
    secret[:, 64:] = 0  # right half is black (ink)
    shares = encode(secret, "binary", fixed_key(k, n, "or"), k, n, "or")
    ink = (secret < 128).astype(np.uint8)
    for share in shares:
        assert set(np.unique(share.array).tolist()) <= {0, 255}
        assert share.array.shape == secret.shape
    for subset in itertools.combinations(shares, k):
        stacked = stack_or([share.array for share in subset])
        if k == n:
            assert np.all(stacked[ink.astype(bool)] == 0), "with k == n, ink pixels always stack to full ink"
        else:
            assert np.mean(stacked[ink.astype(bool)] == 0) > 0.75
        assert metrics.vc_relative_contrast(ink, stacked) > 0.02
    for subset in itertools.combinations(shares, k - 1):
        stacked = stack_or([share.array for share in subset])
        assert abs(metrics.vc_relative_contrast(ink, stacked)) < 0.03


def test_random_grid_contrast_matches_theory_for_2_2():
    secret = np.full((200, 200, 1), 255, dtype=np.uint8)
    secret[:, 100:] = 0
    shares = encode(secret, "binary", fixed_key(2, 2, "or"), 2, 2, "or")
    stacked = stack_or([share.array for share in shares])
    ink = (secret < 128).astype(np.uint8)
    # Kafri-Keren (2,2): white areas are clear with probability 1/2, black never.
    assert metrics.vc_relative_contrast(ink, stacked) == pytest.approx(0.5, abs=0.02)


def test_encoding_is_deterministic_under_fixed_seed():
    a = encode(SECRET, "color", fixed_key(2, 3), 2, 3, "xor")
    b = encode(SECRET, "color", fixed_key(2, 3), 2, 3, "xor")
    assert all(np.array_equal(x.array, y.array) for x, y in zip(a, b))


# ---------------------------------------------------------------- metrics

def test_psnr_and_ssim_reference_values():
    a = np.full((32, 32), 100, dtype=np.uint8)
    b = a.copy()
    b[0, 0] = 110
    expected = 10 * math.log10(255**2 / (100 / 1024))
    assert metrics.psnr(a, b) == pytest.approx(expected)
    assert metrics.psnr(a, a) == math.inf
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    assert metrics.ssim(image, image) == pytest.approx(1.0)
    yy, xx = np.mgrid[0:64, 0:64]
    smooth = (xx * 2 + yy).astype(np.uint8)
    noisy = np.clip(smooth.astype(int) + rng.integers(-60, 60, smooth.shape), 0, 255).astype(np.uint8)
    assert 0 < metrics.ssim(smooth, noisy) < 0.5


def test_entropy_npcr_uaci_and_correlation_bounds():
    assert metrics.shannon_entropy(np.zeros(100, dtype=np.uint8)) == 0.0
    assert metrics.shannon_entropy(np.arange(256, dtype=np.uint8)) == pytest.approx(8.0)
    assert metrics.shannon_entropy(np.zeros(0, dtype=np.uint8)) == 0.0
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.full((10, 10), 255, dtype=np.uint8)
    assert metrics.npcr(a, b) == 100.0
    assert metrics.uaci(a, b) == 100.0
    assert metrics.npcr(a, a) == 0.0
    assert metrics.pearson(a, b) == 0.0  # constant inputs -> defined as 0
    x = np.arange(100, dtype=np.uint8)
    assert metrics.pearson(x, x) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.npcr(a, b[:5])
