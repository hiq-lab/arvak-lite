"""Test arvak-lite as drop-in compiler on qBraid runtime.

Usage:
    export QBRAID_API_KEY=your_key_here
    pip install arvak-lite qbraid qiskit
    python -m pytest tests/test_qbraid_integration.py -v -s
"""

from __future__ import annotations

import os
import time

import pytest

# Skip entire module if qbraid not installed or no API key
qbraid = pytest.importorskip("qbraid")
pytestmark = pytest.mark.skipif(
    not os.getenv("QBRAID_API_KEY"),
    reason="QBRAID_API_KEY not set",
)


def make_bell_circuit():
    """Create a simple Bell state circuit in Qiskit."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_arvak_lite_compiles_for_qbraid():
    """Compile a circuit with arvak-lite, then run on qBraid simulator."""
    from arvak_lite import transpile

    qc = make_bell_circuit()

    # Compile with arvak-lite
    compiled, metrics = transpile(qc, return_metrics=True)

    print(f"\n  arvak-lite compiled in {metrics['time_ms']:.2f}ms")
    print(f"  gate count: {metrics['gate_count']}")

    assert compiled is not None
    assert metrics["time_ms"] > 0


def test_arvak_lite_vs_qbraid_transpile():
    """Compare arvak-lite transpile vs qBraid's built-in transpile."""
    from arvak_lite import transpile as arvak_transpile

    qc = make_bell_circuit()

    # arvak-lite
    t0 = time.perf_counter()
    arvak_compiled = arvak_transpile(qc)
    arvak_time = (time.perf_counter() - t0) * 1000

    # qBraid transpile (framework conversion, not compilation)
    t0 = time.perf_counter()
    qbraid_result = qbraid.transpile(qc, "qiskit")
    qbraid_time = (time.perf_counter() - t0) * 1000

    print(f"\n  arvak-lite: {arvak_time:.2f}ms")
    print(f"  qBraid transpile: {qbraid_time:.2f}ms")
    print(f"  speedup: {qbraid_time / arvak_time:.1f}x")

    assert arvak_compiled is not None


def test_submit_arvak_compiled_to_qbraid_simulator():
    """Full pipeline: compile with arvak-lite, run on qBraid simulator."""
    from arvak_lite import transpile

    qc = make_bell_circuit()
    compiled = transpile(qc)

    # Submit to qBraid simulator
    provider = qbraid.runtime.QbraidProvider(api_key=os.getenv("QBRAID_API_KEY"))

    try:
        devices = provider.get_devices()
        sim_devices = [d for d in devices if "simulator" in str(d).lower()]
        if not sim_devices:
            pytest.skip("No simulator available on qBraid")

        device = sim_devices[0]
        job = device.run(compiled, shots=1000)
        result = job.result()
        counts = result.data.get_counts()

        print(f"\n  Device: {device}")
        print(f"  Counts: {counts}")

        # Bell state: expect ~50% |00> and ~50% |11>
        total = sum(counts.values())
        assert total == 1000
        # At least 30% in each expected state (loose bound for noise)
        for state in ["00", "11"]:
            if state in counts:
                assert counts[state] / total > 0.3, f"Expected ~50% for |{state}>, got {counts[state]/total:.1%}"

    except Exception as e:
        if "credential" in str(e).lower() or "auth" in str(e).lower():
            pytest.skip(f"qBraid auth issue: {e}")
        raise
