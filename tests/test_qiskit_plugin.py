"""Tests for Qiskit transpiler plugin registration and functionality.

Verifies that arvak-lite registers as a valid Qiskit transpiler plugin
and produces correct circuits through the standard transpile() API.
"""

from __future__ import annotations

import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class TestPluginRegistration:
    """Verify arvak appears in Qiskit's plugin registry."""

    def test_init_stage_registered(self):
        assert "arvak" in list_stage_plugins("init")

    def test_routing_stage_registered(self):
        assert "arvak" in list_stage_plugins("routing")

    def test_optimization_stage_registered(self):
        assert "arvak" in list_stage_plugins("optimization")


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

class TestPluginFunctionality:
    """Verify arvak plugin produces correct circuits."""

    def _bell(self) -> QuantumCircuit:
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc

    def _ghz(self, n: int = 4) -> QuantumCircuit:
        qc = QuantumCircuit(n, n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        return qc

    def test_optimization_method(self):
        """transpile(circuit, optimization_method='arvak') works."""
        compiled = transpile(self._bell(), optimization_method="arvak")
        assert compiled.num_qubits == 2
        assert compiled.num_clbits == 2
        ops = compiled.count_ops()
        assert "h" in ops or "u" in ops or "rz" in ops  # some gate exists
        assert "measure" in ops

    def test_routing_method_no_coupling(self):
        """routing_method='arvak' with no coupling map passes through."""
        compiled = transpile(self._bell(), routing_method="arvak")
        assert compiled.num_qubits == 2
        assert "measure" in compiled.count_ops()

    def test_routing_method_with_coupling(self):
        """routing_method='arvak' with linear coupling map."""
        coupling = [[0, 1], [1, 0], [1, 2], [2, 1]]
        compiled = transpile(
            self._ghz(3),
            coupling_map=coupling,
            routing_method="arvak",
        )
        assert compiled.num_qubits >= 3
        assert "measure" in compiled.count_ops()

    def test_init_method(self):
        """init_method='arvak' works."""
        compiled = transpile(self._bell(), init_method="arvak")
        assert compiled.num_qubits == 2

    def test_all_stages_arvak(self):
        """All three stages set to arvak."""
        compiled = transpile(
            self._ghz(4),
            init_method="arvak",
            routing_method="arvak",
            optimization_method="arvak",
        )
        assert compiled.num_qubits >= 4
        assert "measure" in compiled.count_ops()

    def test_generate_preset_pass_manager(self):
        """Works with the modern generate_preset_pass_manager API."""
        pm = generate_preset_pass_manager(
            optimization_level=3,
            optimization_method="arvak",
        )
        compiled = pm.run(self._bell())
        assert compiled.num_qubits == 2
        assert "measure" in compiled.count_ops()

    def test_preserves_measurement_count(self):
        """Measurements are preserved through compilation."""
        qc = self._ghz(3)
        compiled = transpile(qc, optimization_method="arvak")
        assert compiled.count_ops().get("measure", 0) == 3

    def test_gate_count_not_worse(self):
        """Arvak should not increase gate count on simple circuits."""
        qc = self._bell()
        original_gates = sum(
            v for k, v in qc.count_ops().items() if k != "measure"
        )
        compiled = transpile(qc, optimization_method="arvak")
        compiled_gates = sum(
            v for k, v in compiled.count_ops().items() if k != "measure"
        )
        assert compiled_gates <= original_gates + 2  # allow small overhead from decomposition
