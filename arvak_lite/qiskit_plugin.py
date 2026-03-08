"""Qiskit transpiler plugin for Arvak.

Registers Arvak as a drop-in stage plugin for Qiskit's transpile() function.
Once installed, users can invoke Arvak compilation via:

    from qiskit import transpile
    compiled = transpile(circuit, optimization_method="arvak")

Or use Arvak for the full pipeline (init + routing + optimization):

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(
        optimization_level=3,
        init_method="arvak",
        routing_method="arvak",
        optimization_method="arvak",
    )
    compiled = pm.run(circuit)

Plugin registration is handled via setuptools entry_points in pyproject.toml.
"""

from __future__ import annotations

from typing import Any

import arvak
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


# ---------------------------------------------------------------------------
# Core pass: delegates to Arvak's Rust compiler
# ---------------------------------------------------------------------------

class ArvakCompilePass(TransformationPass):
    """A Qiskit TransformationPass that compiles circuits using Arvak.

    Converts DAGCircuit → QuantumCircuit → arvak.Circuit, compiles with
    Arvak's Rust-native optimizer, then converts back.
    """

    def __init__(
        self,
        optimization_level: int = 3,
        coupling_map: Any | None = None,
        basis_gates: list[str] | None = None,
    ):
        super().__init__()
        self.optimization_level = optimization_level
        self._coupling_map = coupling_map
        self._basis_gates = basis_gates

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # DAG → QuantumCircuit
        circuit = dag_to_circuit(dag)

        # QuantumCircuit → arvak.Circuit
        try:
            integration = arvak.get_integration("qiskit")
            arvak_circuit = integration.to_arvak(circuit)
        except (ImportError, ValueError):
            from qiskit.qasm3 import dumps
            arvak_circuit = arvak.from_qasm(dumps(circuit))

        # Build compilation kwargs
        compile_kwargs: dict[str, Any] = {
            "optimization_level": self.optimization_level,
        }
        if self._coupling_map is not None:
            compile_kwargs["coupling_map"] = self._coupling_map
        if self._basis_gates is not None:
            compile_kwargs["basis_gates"] = arvak.BasisGates(self._basis_gates)

        # Compile with Arvak
        compiled = arvak.compile(arvak_circuit, **compile_kwargs)

        # arvak.Circuit → QuantumCircuit
        try:
            integration = arvak.get_integration("qiskit")
            result_circuit = integration.from_arvak(compiled)
        except (ImportError, ValueError):
            from qiskit.qasm3 import loads
            result_circuit = loads(arvak.to_qasm(compiled))

        # Preserve measurements from original circuit.
        # Arvak strips measurements during compilation; re-add them.
        original_measures = [
            (circuit.find_bit(instr.qubits[0]).index,
             circuit.find_bit(instr.clbits[0]).index)
            for instr in circuit.data
            if instr.operation.name == "measure"
        ]
        result_has_measures = "measure" in result_circuit.count_ops()

        if original_measures and not result_has_measures:
            from qiskit import QuantumCircuit as QC
            n_q = max(result_circuit.num_qubits, circuit.num_qubits)
            n_c = circuit.num_clbits
            qc = QC(n_q, n_c) if result_circuit.num_clbits == 0 else QC(n_q)
            if result_circuit.num_clbits == 0:
                qc.compose(result_circuit, qubits=range(result_circuit.num_qubits), inplace=True)
            else:
                qc = result_circuit.copy()
                if qc.num_clbits == 0:
                    from qiskit.circuit import ClassicalRegister
                    qc.add_register(ClassicalRegister(n_c))
            for q_idx, c_idx in original_measures:
                if q_idx < qc.num_qubits and c_idx < qc.num_clbits:
                    qc.measure(q_idx, c_idx)
            result_circuit = qc

        return circuit_to_dag(result_circuit)


# ---------------------------------------------------------------------------
# Stage plugins: registered via entry_points
# ---------------------------------------------------------------------------

def _extract_coupling_map(config: PassManagerConfig) -> Any | None:
    """Extract coupling map from PassManagerConfig for Arvak."""
    if config.coupling_map is not None:
        edges = [tuple(e) for e in config.coupling_map.get_edges()]
        if edges:
            n_qubits = max(max(a, b) for a, b in edges) + 1
            return arvak.CouplingMap.from_edge_list(n_qubits, edges)
    return None


class ArvakInitPlugin(PassManagerStagePlugin):
    """Arvak as the init stage — full circuit optimization early."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level: int | None = None,
    ) -> PassManager:
        level = optimization_level if optimization_level is not None else 3
        return PassManager([
            ArvakCompilePass(
                optimization_level=level,
                coupling_map=_extract_coupling_map(pass_manager_config),
                basis_gates=pass_manager_config.basis_gates,
            ),
        ])


class ArvakRoutingPlugin(PassManagerStagePlugin):
    """Arvak as the routing stage — topology-aware qubit mapping."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level: int | None = None,
    ) -> PassManager:
        level = optimization_level if optimization_level is not None else 3
        coupling_map = _extract_coupling_map(pass_manager_config)
        if coupling_map is None:
            # No coupling map → nothing to route, pass through
            return PassManager()
        return PassManager([
            ArvakCompilePass(
                optimization_level=level,
                coupling_map=coupling_map,
                basis_gates=pass_manager_config.basis_gates,
            ),
        ])


class ArvakOptimizationPlugin(PassManagerStagePlugin):
    """Arvak as the optimization stage — gate reduction and simplification."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level: int | None = None,
    ) -> PassManager:
        level = optimization_level if optimization_level is not None else 3
        return PassManager([
            ArvakCompilePass(
                optimization_level=level,
                coupling_map=_extract_coupling_map(pass_manager_config),
                basis_gates=pass_manager_config.basis_gates,
            ),
        ])
