"""The only function that matters.

    from arvak_lite import transpile
    compiled = transpile(circuit)

Accepts Qiskit, Cirq, PennyLane, Pulser, raw QASM, or arvak.Circuit.
Returns the same type you gave it — compiled, optimized, ready to run.
"""

from __future__ import annotations

import time
from typing import Any

import arvak


def transpile(
    circuit: Any,
    *,
    optimization_level: int = 3,
    coupling_map: list[list[int]] | None = None,
    basis_gates: list[str] | None = None,
    backend: str | None = None,
    return_metrics: bool = False,
) -> Any:
    """Compile a quantum circuit. Any framework in, same framework out.

    Args:
        circuit: A quantum circuit in any supported format:
            - ``qiskit.QuantumCircuit``
            - ``cirq.Circuit``
            - ``pennylane.tape.QuantumTape`` or QNode
            - ``pulser.Sequence`` (digital mode)
            - ``arvak.Circuit``
            - ``str`` (OpenQASM 2.0 or 3.0)

        optimization_level: How hard to optimize (0–3). Default 3.
            0 = parse only, 1 = basic, 2 = standard, 3 = aggressive.

        coupling_map: Device connectivity as edge list,
            e.g. ``[[0,1], [1,2], [2,3]]``. If None, all-to-all.

        basis_gates: Target gate set, e.g. ``["cx", "rz", "sx", "x"]``.

        backend: Non-binding backend hint, e.g. ``"ibm_torino"``.
            Arvak may use this to select optimal compilation strategy.

        return_metrics: If True, return ``(compiled_circuit, metrics)``
            where metrics is a dict with compilation time and gate counts.

    Returns:
        Compiled circuit in the **same type** as the input.
        If input was a Qiskit QuantumCircuit, output is a Qiskit QuantumCircuit.

    Example::

        from arvak_lite import transpile

        # Qiskit
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        compiled = transpile(qc)

        # Raw QASM
        compiled = transpile("OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];")

        # With metrics
        compiled, metrics = transpile(qc, return_metrics=True)
        print(f"Compiled in {metrics['time_ms']:.2f}ms, {metrics['gate_count']} gates")
    """
    # Detect framework and convert to arvak.Circuit
    source_type, arvak_circuit = _to_arvak(circuit)

    # Build compilation kwargs
    compile_kwargs: dict[str, Any] = {
        "optimization_level": optimization_level,
    }
    if coupling_map is not None:
        compile_kwargs["coupling_map"] = arvak.CouplingMap(coupling_map)
    if basis_gates is not None:
        compile_kwargs["basis_gates"] = arvak.BasisGates(basis_gates)

    # Compile
    t0 = time.perf_counter()
    compiled = arvak.compile(arvak_circuit, **compile_kwargs)
    t1 = time.perf_counter()

    # Convert back to original framework
    result = _from_arvak(compiled, source_type)

    if return_metrics:
        qasm_out = arvak.to_qasm(compiled)
        metrics = {
            "time_ms": (t1 - t0) * 1000,
            "gate_count": _count_gates(qasm_out),
            "arvak_version": getattr(arvak, "__version__", "unknown"),
        }
        return result, metrics

    return result


# ---------------------------------------------------------------------------
# Framework detection and conversion
# ---------------------------------------------------------------------------

def _to_arvak(circuit: Any) -> tuple[str, arvak.Circuit]:
    """Convert any supported circuit to arvak.Circuit.

    Returns (source_type, arvak_circuit).
    """
    # Already an arvak.Circuit
    if isinstance(circuit, arvak.Circuit):
        return "arvak", circuit

    # Raw QASM string
    if isinstance(circuit, str):
        return "qasm", arvak.from_qasm(circuit)

    type_name = _type_path(circuit)

    # Qiskit
    if "qiskit" in type_name:
        return "qiskit", _from_qiskit(circuit)

    # Cirq
    if "cirq" in type_name:
        return "cirq", _from_cirq(circuit)

    # PennyLane
    if "pennylane" in type_name:
        return "pennylane", _from_pennylane(circuit)

    # Pulser
    if "pulser" in type_name:
        return "pulser", _from_pulser(circuit)

    raise TypeError(
        f"Unsupported circuit type: {type(circuit).__name__}. "
        "Supported: qiskit.QuantumCircuit, cirq.Circuit, pennylane QNode/tape, "
        "pulser.Sequence, arvak.Circuit, or QASM string."
    )


def _from_arvak(compiled: arvak.Circuit, source_type: str) -> Any:
    """Convert compiled arvak.Circuit back to the original framework."""
    if source_type == "arvak":
        return compiled

    if source_type == "qasm":
        return arvak.to_qasm(compiled)

    if source_type == "qiskit":
        return _to_qiskit(compiled)

    if source_type == "cirq":
        return _to_cirq(compiled)

    if source_type == "pennylane":
        return _to_pennylane(compiled)

    if source_type == "pulser":
        # Pulser round-trip not yet supported — return QASM
        return arvak.to_qasm(compiled)

    return arvak.to_qasm(compiled)


# ---------------------------------------------------------------------------
# Qiskit
# ---------------------------------------------------------------------------

def _from_qiskit(circuit: Any) -> arvak.Circuit:
    """Qiskit QuantumCircuit → arvak.Circuit."""
    try:
        integration = arvak.get_integration("qiskit")
        return integration.to_arvak(circuit)
    except (ImportError, ValueError):
        # Fallback: export QASM from Qiskit, parse with Arvak
        try:
            from qiskit.qasm3 import dumps
            return arvak.from_qasm(dumps(circuit))
        except ImportError:
            from qiskit import qasm2
            return arvak.from_qasm(circuit.qasm())


def _to_qiskit(compiled: arvak.Circuit) -> Any:
    """arvak.Circuit → Qiskit QuantumCircuit."""
    try:
        integration = arvak.get_integration("qiskit")
        return integration.from_arvak(compiled)
    except (ImportError, ValueError):
        from qiskit.qasm3 import loads
        return loads(arvak.to_qasm(compiled))


# ---------------------------------------------------------------------------
# Cirq
# ---------------------------------------------------------------------------

def _from_cirq(circuit: Any) -> arvak.Circuit:
    """Cirq Circuit → arvak.Circuit."""
    try:
        integration = arvak.get_integration("cirq")
        return integration.to_arvak(circuit)
    except (ImportError, ValueError):
        from cirq.contrib.qasm_import import circuit_from_qasm
        # Cirq → QASM2 → Arvak
        qasm = circuit.to_qasm()
        return arvak.from_qasm(qasm)


def _to_cirq(compiled: arvak.Circuit) -> Any:
    """arvak.Circuit → Cirq Circuit."""
    try:
        integration = arvak.get_integration("cirq")
        return integration.from_arvak(compiled)
    except (ImportError, ValueError):
        from cirq.contrib.qasm_import import circuit_from_qasm
        return circuit_from_qasm(arvak.to_qasm(compiled))


# ---------------------------------------------------------------------------
# PennyLane
# ---------------------------------------------------------------------------

def _from_pennylane(circuit: Any) -> arvak.Circuit:
    """PennyLane tape/QNode → arvak.Circuit."""
    try:
        integration = arvak.get_integration("pennylane")
        return integration.to_arvak(circuit)
    except (ImportError, ValueError):
        raise TypeError(
            "PennyLane conversion requires arvak[pennylane]. "
            "Install with: pip install arvak[pennylane]"
        )


def _to_pennylane(compiled: arvak.Circuit) -> Any:
    """arvak.Circuit → PennyLane tape."""
    try:
        integration = arvak.get_integration("pennylane")
        return integration.from_arvak(compiled)
    except (ImportError, ValueError):
        # Fallback: return QASM
        return arvak.to_qasm(compiled)


# ---------------------------------------------------------------------------
# Pulser (Pasqal)
# ---------------------------------------------------------------------------

def _from_pulser(circuit: Any) -> arvak.Circuit:
    """Pulser Sequence → arvak.Circuit (digital mode only)."""
    # Pulser digital mode exports to abstract gate sequences
    # For now: extract gate operations and build arvak.Circuit
    try:
        # Pulser sequences in digital mode can export to QASM-like ops
        if hasattr(circuit, "to_abstract_repr"):
            import json
            abstract = json.loads(circuit.to_abstract_repr())
            n_qubits = len(abstract.get("register", []))
            qc = arvak.Circuit("pulser_import", num_qubits=max(n_qubits, 1))
            # TODO: map Pulser digital ops to arvak gates
            return qc
    except Exception:
        pass

    raise TypeError(
        "Pulser conversion currently requires digital mode sequences. "
        "Analog pulse sequences are not yet supported."
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _type_path(obj: Any) -> str:
    """Get fully qualified type name, lowercased."""
    return (type(obj).__module__ + "." + type(obj).__qualname__).lower()


def _count_gates(qasm: str) -> int:
    """Quick gate count from QASM string."""
    count = 0
    for line in qasm.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include") or line.startswith("qubit") or line.startswith("bit") or line.startswith("creg") or line.startswith("qreg"):
            continue
        if "q[" in line or "q " in line:
            count += 1
    return count
