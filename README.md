# arvak-lite

[![Qiskit Ecosystem](https://img.shields.io/badge/Qiskit-Ecosystem-blueviolet)](https://qiskit.github.io/ecosystem/)

Lightweight quantum circuit compiler wrapping [Arvak](https://arvak.io).
Takes circuits from any major framework, compiles them, and returns the same type.

```python
from arvak_lite import transpile

compiled = transpile(circuit)
```

Also registers as a [Qiskit transpiler plugin](https://docs.quantum.ibm.com/guides/transpiler-plugins):

```python
from qiskit.compiler import transpile

compiled = transpile(circuit, optimization_method="arvak")
```

## Installation

```bash
pip install arvak-lite
```

With framework extras:

```bash
pip install arvak-lite[qiskit]      # Qiskit + transpiler plugin
pip install arvak-lite[cirq]        # Cirq
pip install arvak-lite[pennylane]   # PennyLane
pip install arvak-lite[pulser]      # Pulser (Pasqal)
pip install arvak-lite[qrisp]       # Qrisp
pip install arvak-lite[qbraid]      # qBraid multi-backend
pip install arvak-lite[all]         # Everything
```

## Usage

### Any framework

```python
from arvak_lite import transpile

compiled = transpile(circuit)  # Qiskit in → Qiskit out, Cirq in → Cirq out, etc.
```

### Qiskit transpiler plugin

Once installed, `arvak` is available as a named plugin in Qiskit's `transpile()`:

```python
from qiskit.compiler import transpile

# Use Arvak for optimization
compiled = transpile(circuit, optimization_method="arvak")

# Use Arvak for routing
compiled = transpile(circuit, coupling_map=coupling, routing_method="arvak")

# Or via pass manager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(optimization_level=3, optimization_method="arvak")
compiled = pm.run(circuit)
```

### With metrics

```python
compiled, metrics = transpile(circuit, return_metrics=True)
print(f"{metrics['time_ms']:.2f}ms, {metrics['gate_count']} gates")
```

## Parameters

```python
transpile(
    circuit,                          # Any supported circuit type
    optimization_level=3,             # 0=parse, 1=basic, 2=standard, 3=aggressive
    coupling_map=[[0,1], [1,2]],      # Device connectivity (None = all-to-all)
    basis_gates=["cx", "rz", "sx"],   # Target gate set
    return_metrics=True,              # Return (circuit, metrics) tuple
)
```

## Supported frameworks

| Framework | Type | Round-trip |
|---|---|---|
| Qiskit | `QuantumCircuit` | Yes |
| Cirq | `cirq.Circuit` | Yes |
| PennyLane | QNode / tape | Yes |
| Pulser | `Sequence` (digital) | QASM output |
| Qrisp | `QuantumCircuit` / `QuantumSession` | Yes |
| OpenQASM | `str` (2.0 / 3.0) | Yes |
| Arvak | `arvak.Circuit` | Yes |

## How it works

arvak-lite is a thin Python wrapper (~12 KB). The actual compilation happens in
[Arvak](https://github.com/hiq-lab/arvak), a Rust-native compiler exposed via
PyO3. The full compiler fits in a ~600 KB wheel — no LLVM, no JVM, no heavy
runtime. The speed comes from Rust, not from skipping optimization passes.

1. Detect the input framework from the circuit's type
2. Convert to `arvak.Circuit`
3. Compile in Rust (gate decomposition, routing, optimization)
4. Convert back to the original framework type

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
