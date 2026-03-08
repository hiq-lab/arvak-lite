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

    # Qrisp
    if "qrisp" in type_name:
        return "qrisp", _from_qrisp(circuit)

    raise TypeError(
        f"Unsupported circuit type: {type(circuit).__name__}. "
        "Supported: qiskit.QuantumCircuit, cirq.Circuit, pennylane QNode/tape, "
        "pulser.Sequence, qrisp.QuantumCircuit, arvak.Circuit, or QASM string."
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

    if source_type == "qrisp":
        return _to_qrisp(compiled)

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
# Qrisp
# ---------------------------------------------------------------------------

def _from_qrisp(circuit: Any) -> arvak.Circuit:
    """Qrisp QuantumCircuit/QuantumSession → arvak.Circuit via QASM."""
    try:
        from qrisp import QuantumSession
        if isinstance(circuit, QuantumSession):
            circuit = circuit.compile()
    except ImportError:
        pass

    # Qrisp exports QASM 2.0 — convert to 3.0 for Arvak's parser
    try:
        integration = arvak.get_integration("qrisp")
        return integration.to_arvak(circuit)
    except (ImportError, ValueError):
        qasm2 = circuit.qasm()
        qasm3 = _qasm2_to_qasm3(qasm2)
        return arvak.from_qasm(qasm3)


def _to_qrisp(compiled: arvak.Circuit) -> Any:
    """arvak.Circuit → Qrisp QuantumCircuit via QASM."""
    try:
        integration = arvak.get_integration("qrisp")
        return integration.from_arvak(compiled)
    except (ImportError, ValueError):
        from qrisp import QuantumCircuit as QrispCircuit
        qasm3 = arvak.to_qasm(compiled)
        qasm2 = _qasm3_to_qasm2(qasm3)
        return QrispCircuit.from_qasm_str(qasm2)


def _qasm2_to_qasm3(qasm2: str) -> str:
    """Minimal QASM 2.0 → 3.0 conversion for Arvak's parser."""
    lines = []
    for line in qasm2.splitlines():
        stripped = line.strip()
        if stripped.startswith("OPENQASM 2"):
            lines.append("OPENQASM 3.0;")
            continue
        if stripped.startswith("include") and "qelib1" in stripped:
            lines.append('include "stdgates.inc";')
            continue
        if stripped.startswith("qreg "):
            # qreg q[2]; → qubit[2] q;
            name = stripped.split()[1].split("[")[0]
            size = stripped.split("[")[1].split("]")[0]
            lines.append(f"qubit[{size}] {name};")
            continue
        if stripped.startswith("creg "):
            name = stripped.split()[1].split("[")[0]
            size = stripped.split("[")[1].split("]")[0]
            lines.append(f"bit[{size}] {name};")
            continue
        lines.append(line)
    return "\n".join(lines)


def _qasm3_to_qasm2(qasm3: str) -> str:
    """Minimal QASM 3.0 → 2.0 conversion for Qrisp's parser."""
    lines = []
    for line in qasm3.splitlines():
        stripped = line.strip()
        if stripped.startswith("OPENQASM 3"):
            lines.append("OPENQASM 2.0;")
            continue
        if stripped.startswith("include") and "stdgates" in stripped:
            lines.append('include "qelib1.inc";')
            continue
        if stripped.startswith("qubit["):
            size = stripped.split("[")[1].split("]")[0]
            name = stripped.rstrip(";").split()[-1].rstrip(";")
            lines.append(f"qreg {name}[{size}];")
            continue
        if stripped.startswith("bit["):
            size = stripped.split("[")[1].split("]")[0]
            name = stripped.rstrip(";").split()[-1].rstrip(";")
            lines.append(f"creg {name}[{size}];")
            continue
        # Skip QASM3-style measure assignments: c[0] = measure q[0];
        if "= measure" in stripped:
            parts = stripped.split("=")
            creg = parts[0].strip()
            qreg = parts[1].replace("measure", "").strip().rstrip(";")
            lines.append(f"measure {qreg} -> {creg};")
            continue
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _type_path(obj: Any) -> str:
    """Get fully qualified type name, lowercased."""
    return (type(obj).__module__ + "." + type(obj).__qualname__).lower()


def _to_pyquil(compiled: Any) -> Any:
    """arvak.Circuit → pyquil.Program with standard 'ro' register.

    qBraid's transpile(circuit, 'pyquil') creates separate DECLARE per
    classical bit (m0, m1, ...) which Rigetti QVM rejects. We build the
    Program manually with a single 'ro' register.
    """
    from pyquil import Program
    from pyquil.gates import H, CNOT, CZ, X, Y, Z, S, T, RX, RY, RZ, MEASURE
    from pyquil.quilbase import Declare

    qasm = arvak.to_qasm(compiled)
    ops = _parse_qasm_ops(qasm)
    n_qubits = _count_qubits_from_ops(qasm, ops)

    p = Program()
    ro = p.declare("ro", "BIT", n_qubits)

    gate_map = {
        "h": H, "x": X, "y": Y, "z": Z, "s": S, "t": T,
        "cx": CNOT, "cnot": CNOT, "cz": CZ,
    }

    for gate_name, qubits in ops:
        if gate_name in gate_map:
            p += gate_map[gate_name](*qubits)

    for i in range(n_qubits):
        p += MEASURE(i, ro[i])

    return p


def _parse_qasm_ops(qasm: str) -> list[tuple[str, list[int]]]:
    """Extract gate operations from QASM as (name, [qubit_indices])."""
    import re
    ops: list[tuple[str, list[int]]] = []
    for line in qasm.splitlines():
        line = line.strip().rstrip(";")
        if not line or line.startswith(("//", "OPENQASM", "include")):
            continue
        if line.startswith(("qubit", "bit", "creg", "qreg", "measure")):
            continue
        if "= measure" in line:
            continue
        match = re.match(r"(\w+)(?:\([^)]*\))?\s+(.+)", line)
        if match:
            gate = match.group(1).lower()
            qubit_str = match.group(2)
            qubit_indices = [int(x) for x in re.findall(r"q\[(\d+)\]", qubit_str)]
            if qubit_indices:
                ops.append((gate, qubit_indices))
    return ops


def _count_qubits_from_ops(qasm: str, ops: list) -> int:
    """Determine qubit count from QASM or operations."""
    for line in qasm.splitlines():
        line = line.strip()
        if line.startswith("qubit["):
            return int(line.split("[")[1].split("]")[0])
    if ops:
        return max(q for _, qubits in ops for q in qubits) + 1
    return 1


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
