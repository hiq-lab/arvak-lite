# arvak-lite

**One function. All frameworks. 3000x faster.**

Drop-in replacement for `qiskit.transpile`, `cirq.optimize`, and other framework transpilers.
Powered by [Arvak](https://arvak.io), the high-performance quantum circuit compiler.

```python
from arvak_lite import transpile

compiled = transpile(circuit)
```

## Why arvak-lite?

| Feature | Qiskit transpile | Cirq optimize | arvak-lite |
|---|---|---|---|
| Frameworks | Qiskit only | Cirq only | All of them |
| Speed (100q) | ~45s | ~12s | ~15ms |
| API surface | 30+ params | varies | 1 function |
| Output type | Qiskit only | Cirq only | Same as input |

arvak-lite detects your circuit's framework, compiles it through the Arvak compiler,
and returns the result in the **same type** you gave it.

## Installation

```bash
pip install arvak-lite
```

With framework extras:

```bash
pip install arvak-lite[qiskit]      # Qiskit integration
pip install arvak-lite[cirq]        # Cirq integration
pip install arvak-lite[pennylane]   # PennyLane integration
pip install arvak-lite[pulser]      # Pulser (Pasqal) integration
pip install arvak-lite[all]         # Everything
```

## Quick start

### Qiskit

```python
from qiskit import QuantumCircuit
from arvak_lite import transpile

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

compiled = transpile(qc)  # Returns a QuantumCircuit
```

### Cirq

```python
import cirq
from arvak_lite import transpile

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

compiled = transpile(circuit)  # Returns a cirq.Circuit
```

### Raw QASM

```python
from arvak_lite import transpile

qasm = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

compiled = transpile(qasm)  # Returns a QASM string
```

### With metrics

```python
compiled, metrics = transpile(circuit, return_metrics=True)
print(f"Compiled in {metrics['time_ms']:.2f}ms, {metrics['gate_count']} gates")
```

## Parameters

```python
transpile(
    circuit,                          # Any supported circuit type
    optimization_level=3,             # 0=parse, 1=basic, 2=standard, 3=aggressive
    coupling_map=[[0,1], [1,2]],      # Device connectivity (None = all-to-all)
    basis_gates=["cx", "rz", "sx"],   # Target gate set
    backend="ibm_torino",             # Backend hint for compilation strategy
    return_metrics=True,              # Return (circuit, metrics) tuple
)
```

## Supported frameworks

| Framework | Import | Round-trip |
|---|---|---|
| Qiskit | `qiskit.QuantumCircuit` | Yes |
| Cirq | `cirq.Circuit` | Yes |
| PennyLane | `pennylane` QNode/tape | Yes |
| Pulser | `pulser.Sequence` (digital) | QASM output |
| OpenQASM | `str` (2.0 or 3.0) | Yes |
| Arvak | `arvak.Circuit` | Yes |

## How it works

1. **Detect** the input framework from the circuit's type
2. **Convert** to `arvak.Circuit` (Arvak's internal representation)
3. **Compile** using Arvak's Rust-based optimizer
4. **Convert back** to the original framework type

No QASM serialization overhead for frameworks with native Arvak integrations.

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).

Built by [The HAL Contract](https://hal-contract.org).
