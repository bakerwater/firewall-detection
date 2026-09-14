"""Public TPT module.

The implementation remains in gate.py so existing imports continue to work.
"""

from .gate import (
    DEFAULT_FIRE_CONFIG,
    DEFAULT_OTHER_CONFIG,
    DEFAULT_SMOKE_CONFIG,
    TPTClassConfig,
    TemporalPersistenceGate as _TemporalPersistenceGate,
)

class TemporalPersistenceGate(_TemporalPersistenceGate):
    """Public TPT gate; inherits the backward-compatible implementation."""


__all__ = [
    "DEFAULT_FIRE_CONFIG",
    "DEFAULT_OTHER_CONFIG",
    "DEFAULT_SMOKE_CONFIG",
    "TPTClassConfig",
    "TemporalPersistenceGate",
]
