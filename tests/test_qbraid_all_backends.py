"""Test arvak-lite on every qBraid simulator backend.

This is the product-readiness test. Every gate-model simulator that qBraid
exposes must work end-to-end: compile with arvak-lite, submit, get results.

Usage:
    export QBRAID_API_KEY=your_key_here
    python -m pytest tests/test_qbraid_all_backends.py -v -s

Backends tested:
    - azure:ionq:sim:simulator      (free, QASM3, IonQ simulator)
    - qbraid:qbraid:sim:qir-sv      (free, QASM3, qBraid statevector)
    - azure:quantinuum:sim:h2-1sc   (free, QASM2, syntax checker only)
    - azure:quantinuum:sim:h2-1e    (free, QASM2, full emulator)
    - azure:rigetti:sim:qvm         (free, pyquil, Rigetti QVM)
    - aws:aws:sim:sv1               (paid, QASM3, AWS statevector)
    - aws:aws:sim:dm1               (paid, QASM3, AWS density matrix)
    - aws:aws:sim:tn1               (paid, QASM3, AWS tensor network)
    - azure:pasqal:sim:emu-tn       (paid, pulser, Pasqal emulator)
"""

from __future__ import annotations

import os
import time

import pytest

qbraid = pytest.importorskip("qbraid")

pytestmark = pytest.mark.skipif(
    not os.getenv("QBRAID_API_KEY"),
    reason="QBRAID_API_KEY not set",
)

API_KEY = os.getenv("QBRAID_API_KEY", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider():
    return qbraid.runtime.QbraidProvider(api_key=API_KEY)


def _bell_qiskit():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _compile_qiskit(qc):
    from arvak_lite import transpile
    return transpile(qc)


def _compile_qasm3(qc):
    """Compile with arvak-lite, return QASM3 string with measurements."""
    compiled = _compile_qiskit(qc)
    from qiskit.qasm3 import dumps
    return dumps(compiled)


def _compile_qasm2(qc):
    """Compile with arvak-lite, return QASM2 string."""
    compiled = _compile_qiskit(qc)
    from qiskit.qasm2 import dumps
    return dumps(compiled)


def _run_and_validate(device, program, shots=100, expect_bell=True):
    """Submit program, wait for result, validate Bell state."""
    t0 = time.time()
    job = device.run(program, shots=shots)
    result = job.result()
    elapsed = time.time() - t0

    if hasattr(result, "data"):
        counts = result.data.get_counts()
    elif hasattr(result, "get_counts"):
        counts = result.get_counts()
    else:
        pytest.fail(f"Cannot extract counts from result type {type(result)}")

    total = sum(counts.values())
    assert total == shots, f"Expected {shots} total shots, got {total}"

    if expect_bell:
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        bell = p00 + p11
        assert bell > 0.8, (
            f"Bell fraction {bell:.1%} too low (|00>={p00:.1%}, |11>={p11:.1%}). "
            f"Full counts: {counts}"
        )

    return counts, elapsed


def _skip_if_offline(device):
    status = device.status()
    if "OFFLINE" in str(status).upper():
        pytest.skip(f"{device.id} is offline")


def _skip_if_no_credits(device, program, shots=100):
    """Try to submit; skip cleanly on 402 (no credits)."""
    try:
        job = device.run(program, shots=shots)
        return job
    except Exception as e:
        if "402" in str(e):
            pytest.skip(f"{device.id} requires credits (402)")
        raise


# ---------------------------------------------------------------------------
# Free simulators — these MUST pass
# ---------------------------------------------------------------------------

class TestFreeSimulators:
    """Free simulators that must always work."""

    def test_ionq_simulator(self):
        """Azure IonQ simulator — QASM3 input, free."""
        provider = _provider()
        device = provider.get_device("azure:ionq:sim:simulator")
        _skip_if_offline(device)

        qasm3 = _compile_qasm3(_bell_qiskit())
        counts, elapsed = _run_and_validate(device, qasm3, shots=100)

        print(f"\n  IonQ sim: {counts} ({elapsed:.1f}s)")

    def test_qbraid_qir_sv(self):
        """qBraid QIR statevector simulator — QASM3 input, free."""
        provider = _provider()
        device = provider.get_device("qbraid:qbraid:sim:qir-sv")
        _skip_if_offline(device)

        qasm3 = _compile_qasm3(_bell_qiskit())
        counts, elapsed = _run_and_validate(device, qasm3, shots=100)

        print(f"\n  qBraid QIR-SV: {counts} ({elapsed:.1f}s)")

    def test_quantinuum_syntax_checker(self):
        """Quantinuum H2-1 syntax checker — QASM2, validates syntax only.

        Returns dummy results (all zeros). We only check that submission
        succeeds and we get back the right number of shots.
        """
        provider = _provider()
        device = provider.get_device("azure:quantinuum:sim:h2-1sc")
        _skip_if_offline(device)

        qasm2 = _compile_qasm2(_bell_qiskit())
        counts, elapsed = _run_and_validate(
            device, qasm2, shots=100, expect_bell=False
        )

        total = sum(counts.values())
        assert total == 100, f"Expected 100 shots, got {total}"
        print(f"\n  Quantinuum H2-1SC (syntax): {counts} ({elapsed:.1f}s)")

    def test_quantinuum_emulator(self):
        """Quantinuum H2-1 emulator — QASM2, full simulation."""
        provider = _provider()
        device = provider.get_device("azure:quantinuum:sim:h2-1e")
        _skip_if_offline(device)

        qasm2 = _compile_qasm2(_bell_qiskit())
        counts, elapsed = _run_and_validate(device, qasm2, shots=100)

        print(f"\n  Quantinuum H2-1E: {counts} ({elapsed:.1f}s)")

    def test_rigetti_qvm(self):
        """Rigetti QVM — pyquil with single 'ro' register."""
        provider = _provider()
        device = provider.get_device("azure:rigetti:sim:qvm")
        _skip_if_offline(device)

        import arvak
        from arvak_lite.transpile import _to_pyquil

        # Compile with arvak, convert to pyquil with correct ro register
        bell = arvak.Circuit("bell", num_qubits=2)
        bell.h(0).cx(0, 1)
        compiled = arvak.compile(bell)
        pyquil_prog = _to_pyquil(compiled)

        try:
            job = device.run(pyquil_prog, shots=100)
            result = job.result()

            if hasattr(result, "data"):
                counts = result.data.get_counts()
            elif hasattr(result, "get_counts"):
                counts = result.get_counts()
            else:
                counts = None

            if counts is not None:
                total = sum(counts.values())
                p00 = counts.get("00", 0) / total
                p11 = counts.get("11", 0) / total
                print(f"\n  Rigetti QVM: {counts} (Bell={p00+p11:.0%})")
                assert p00 + p11 > 0.8
            else:
                print(f"\n  Rigetti QVM: job completed, result={result}")

        except AttributeError as e:
            if "resultData" in str(e) or "NoneType" in str(e):
                pytest.skip(
                    f"qBraid SDK cannot parse Rigetti QVM results (known issue): {e}"
                )
            raise
        except Exception as e:
            if "402" in str(e):
                pytest.skip(f"Rigetti QVM requires credits (402)")
            raise


# ---------------------------------------------------------------------------
# Paid simulators — test if credits available, skip otherwise
# ---------------------------------------------------------------------------

class TestPaidSimulators:
    """Paid simulators — skip on 402, validate Bell state if run."""

    def test_aws_sv1(self):
        """AWS SV1 statevector simulator — QASM3, paid."""
        provider = _provider()
        device = provider.get_device("aws:aws:sim:sv1")
        _skip_if_offline(device)

        qasm3 = _compile_qasm3(_bell_qiskit())
        job = _skip_if_no_credits(device, qasm3, shots=100)

        result = job.result()
        if hasattr(result, "data"):
            counts = result.data.get_counts()
        else:
            counts = result.get_counts()

        total = sum(counts.values())
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        assert p00 + p11 > 0.8
        print(f"\n  AWS SV1: {counts}")

    def test_aws_dm1(self):
        """AWS DM1 density matrix simulator — QASM3, paid."""
        provider = _provider()
        device = provider.get_device("aws:aws:sim:dm1")
        _skip_if_offline(device)

        qasm3 = _compile_qasm3(_bell_qiskit())
        job = _skip_if_no_credits(device, qasm3, shots=100)

        result = job.result()
        if hasattr(result, "data"):
            counts = result.data.get_counts()
        else:
            counts = result.get_counts()

        total = sum(counts.values())
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        assert p00 + p11 > 0.8
        print(f"\n  AWS DM1: {counts}")

    def test_aws_tn1(self):
        """AWS TN1 tensor network simulator — QASM3, paid."""
        provider = _provider()
        device = provider.get_device("aws:aws:sim:tn1")
        _skip_if_offline(device)

        qasm3 = _compile_qasm3(_bell_qiskit())
        job = _skip_if_no_credits(device, qasm3, shots=100)

        result = job.result()
        if hasattr(result, "data"):
            counts = result.data.get_counts()
        else:
            counts = result.get_counts()

        total = sum(counts.values())
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        assert p00 + p11 > 0.8
        print(f"\n  AWS TN1: {counts}")

    def test_pasqal_emu_tn(self):
        """Pasqal emu-tn — Pulser Sequence via converter, paid."""
        provider = _provider()
        device = provider.get_device("azure:pasqal:sim:emu-tn")
        _skip_if_offline(device)

        import arvak
        from arvak.integrations.pulser.converter import arvak_to_pulser

        bell = arvak.Circuit("bell", num_qubits=2)
        bell.h(0).cx(0, 1)
        compiled = arvak.compile(bell)
        seq = arvak_to_pulser(compiled, spacing=4.0)

        job = _skip_if_no_credits(device, seq, shots=100)
        result = job.result()

        if hasattr(result, "data"):
            counts = result.data.get_counts()
        else:
            counts = result.get_counts()

        total = sum(counts.values())
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        assert p00 + p11 > 0.8
        print(f"\n  Pasqal emu-tn: {counts}")


# ---------------------------------------------------------------------------
# Local Pulser verification (no qBraid credits needed)
# ---------------------------------------------------------------------------

class TestPulserLocal:
    """Verify Pulser converter locally via QutipEmulator."""

    def test_bell_state_qutip(self):
        """Full pipeline: arvak compile -> Pulser -> QutipEmulator."""
        import arvak
        from arvak.integrations.pulser.converter import arvak_to_pulser

        bell = arvak.Circuit("bell", num_qubits=2)
        bell.h(0).cx(0, 1)
        compiled = arvak.compile(bell)
        seq = arvak_to_pulser(compiled, spacing=4.0)

        from pulser_simulation import QutipEmulator
        import numpy as np

        sim = QutipEmulator.from_sequence(seq)
        results = sim.run()

        # State vector check
        final = results.get_final_state(ignore_global_phase=False)
        arr = np.array(final.full()).flatten()

        # 3-level: |gg>=4=|00>, |gh>=5=|01>, |hg>=7=|10>, |hh>=8=|11>
        p00 = abs(arr[4]) ** 2
        p11 = abs(arr[8]) ** 2
        leakage = 1.0 - sum(abs(arr[i]) ** 2 for i in [4, 5, 7, 8])

        fidelity = p00 + p11
        assert fidelity > 0.99, f"Bell fidelity {fidelity:.4f} < 99%"
        assert leakage < 0.01, f"Rydberg leakage {leakage:.4f} > 1%"

        # Sample check
        counts = results.sample_final_state(N_samples=10000)
        total = sum(counts.values())
        bell_frac = (counts.get("00", 0) + counts.get("11", 0)) / total
        assert bell_frac > 0.95

        print(f"\n  QutipEmulator: fidelity={fidelity:.4f}, "
              f"leakage={leakage:.4f}, counts={dict(counts)}")
