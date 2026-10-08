"""Explicit sentinels raised only after recognizing a recorded defect signature.

Strict xfail markers accept these sentinels, never arbitrary assertion/runtime
failures. Unrelated failures propagate normally; a repaired contract is XPASS.
"""


class MissingCSVTextDefect(AssertionError):
    """Exactly the known missing-text row becomes literal 'nan'."""


class AuthMiddlewareDefect(AssertionError):
    """The intended 401 exception escapes middleware instead of becoming a response."""


class UndefinedProviderDefect(AssertionError):
    """KG metadata construction references the absent provider name."""


class CreatorSignatureDefect(AssertionError):
    """Only the endpoint's extra final provider argument breaks signature binding."""


class ReadbackIdentityDefect(AssertionError):
    """Internal numeric identities are compared against string application IDs."""


class ReadbackDriverLeak(AssertionError):
    """A completed readback request exits its session without closing its driver."""
