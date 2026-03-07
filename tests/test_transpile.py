"""Test arvak-lite transpile function."""

from __future__ import annotations

import pytest

from arvak_lite import transpile


BELL_QASM = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

GHZ_QASM = """OPENQASM 3.0;
qubit[3] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
"""


def test_transpile_qasm_string():
    result = transpile(BELL_QASM)
    assert isinstance(result, str)
    assert "OPENQASM" in result


def test_transpile_qasm_returns_qasm():
    """QASM in → QASM out."""
    result = transpile(BELL_QASM)
    assert "qubit" in result or "qreg" in result


def test_transpile_arvak_circuit():
    import arvak
    qc = arvak.Circuit("test", num_qubits=2)
    qc.h(0).cx(0, 1)
    result = transpile(qc)
    assert isinstance(result, arvak.Circuit)


def test_transpile_with_metrics():
    result, metrics = transpile(BELL_QASM, return_metrics=True)
    assert isinstance(result, str)
    assert "time_ms" in metrics
    assert metrics["time_ms"] > 0
    assert "gate_count" in metrics
    assert metrics["gate_count"] >= 1


def test_transpile_optimization_levels():
    for level in range(4):
        result = transpile(BELL_QASM, optimization_level=level)
        assert "OPENQASM" in result


def test_transpile_with_basis_gates():
    result = transpile(BELL_QASM, basis_gates=["cx", "rz", "sx", "x"])
    assert isinstance(result, str)


def test_transpile_ghz():
    result, metrics = transpile(GHZ_QASM, return_metrics=True)
    assert metrics["gate_count"] >= 2


def test_transpile_invalid_type():
    with pytest.raises(TypeError, match="Unsupported circuit type"):
        transpile(42)


def test_transpile_invalid_qasm():
    with pytest.raises(RuntimeError):
        transpile("not valid qasm at all")
