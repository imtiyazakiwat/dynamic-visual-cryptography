"""Typed error codes shared by every layer of the pipeline."""

UNSUPPORTED_MODE = "UNSUPPORTED_MODE"
DIMENSION_LIMIT = "DIMENSION_LIMIT"
INVALID_PARAMS = "INVALID_PARAMS"
MISSING = "MISSING"
TAMPER = "TAMPER"
STALE = "STALE"
PARAM_MISMATCH = "PARAM_MISMATCH"
REPLAY_MISMATCH = "REPLAY_MISMATCH"

EXPLANATIONS = {
    UNSUPPORTED_MODE: "The input could not be decoded as a supported raster image.",
    DIMENSION_LIMIT: "The image is larger than the allowed maximum or smaller than the minimum size.",
    INVALID_PARAMS: "The (k, n) threshold or recovery mode is not valid.",
    MISSING: "Fewer than k valid shares arrived, so the secret cannot be reconstructed.",
    TAMPER: "A SHA-256 tag or manifest checksum did not match; the data was altered in transit.",
    STALE: "The share or manifest belongs to another session or is outside the freshness window.",
    PARAM_MISMATCH: "Share or manifest parameters do not match what the receiver expects.",
    REPLAY_MISMATCH: "Re-running the logged session did not reproduce the same shares.",
}


class DVCError(Exception):
    """Pipeline failure carrying a machine-readable code and the phase that raised it."""

    def __init__(self, code: str, message: str, phase: str | None = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.phase = phase

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "phase": self.phase,
            "explanation": EXPLANATIONS.get(self.code, ""),
        }
