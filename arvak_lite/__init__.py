"""arvak-lite — Lightweight quantum circuit compiler.

Drop-in replacement for framework transpilers.
One function. All frameworks. 3000x faster.

    from arvak_lite import transpile
    compiled = transpile(circuit)
"""

from __future__ import annotations

from arvak_lite.transpile import transpile

__version__ = "0.1.0"
__all__ = ["transpile"]
