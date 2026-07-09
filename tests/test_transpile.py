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


class TestCouplingMapEdgeList:
    """Regression: arvak >= 2.0 changed CouplingMap(n) to take a qubit
    count; the edge-list path must go through from_edge_list. Broken in
    0.1.1 against arvak 2.x (TypeError on the num_qubits argument)."""

    def test_transpile_with_edge_list_and_basis(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.ccx(0, 1, 2)
        qc.measure(range(3), range(3))

        out = transpile(
            qc,
            coupling_map=[[0, 1], [1, 2]],
            basis_gates=["cx", "rz", "sx", "x"],
        )
        ops = out.count_ops()
        assert "ccx" not in ops, "ccx must be decomposed for the target"
        assert all(g in {"cx", "rz", "sx", "x", "measure", "barrier"} for g in ops)

    def test_edge_list_wider_than_circuit(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Device has 5 qubits, circuit only 2 — num_qubits must follow
        # the highest edge index, not the circuit width.
        out = transpile(qc, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]])
        assert out.num_qubits >= 2
