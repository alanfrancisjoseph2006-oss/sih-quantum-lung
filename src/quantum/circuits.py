"""
Quantum machine-learning branch (PennyLane).

Design rationale
----------------
The CNN embedding is 256-d; a variational quantum circuit can only accept a
handful of features. So the pipeline is:

    256-d CNN embedding
      -> StandardScaler
      -> PCA (32 components, fitted on TRAIN ONLY)
      -> linear projection to n_qubits, scaled into [-pi, pi]
      -> angle encoding (RY)
      -> `Q_LAYERS` x [ RY/RZ rotations + ring of CNOTs ]
      -> Pauli-Z expectation on every wire
      -> n_qubits quantum features -> linear read-out -> 3 logits

Deliberate constraints, per the brief:
  * Simulator backend (`default.qubit` / `lightning.qubit`), not real hardware.
  * Shallow circuit (3 layers) and few qubits (4/6/8) -- deep circuits on this
    little data would be untrainable (barren plateaus) and slow to simulate.
  * `n_qubits` is configurable and swept experimentally.

No claim of quantum advantage is made anywhere. The circuit is evaluated on the
same held-out test split as every classical model, and reported side by side.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from src.config import Q_LAYERS, QUANTUM_BACKEND, NUM_CLASSES


def make_qnode(n_qubits: int, n_layers: int = Q_LAYERS, backend: str = QUANTUM_BACKEND):
    """
    Build the parameterised circuit as a Torch-differentiable QNode.

    Falls back from `lightning.qubit` to `default.qubit` if the fast C++
    simulator is unavailable, so the pipeline never hard-fails on backend setup.
    """
    try:
        dev = qml.device(backend, wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # --- Angle encoding: one rotation per feature, one feature per qubit ---
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # --- Variational layers: rotations + ring entanglement ---
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(weights[l, w, 0], wires=w)
                qml.RZ(weights[l, w, 1], wires=w)
            if n_qubits > 1:
                for w in range(n_qubits):
                    qml.CNOT(wires=[w, (w + 1) % n_qubits])

        # --- Measurement: <Z> on each wire -> n_qubits real features ---
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit, dev


class QuantumLayer(nn.Module):
    """Torch module wrapping the QNode, with trainable circuit weights."""

    def __init__(self, n_qubits: int, n_layers: int = Q_LAYERS,
                 backend: str = QUANTUM_BACKEND):
        super().__init__()
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.circuit, self.dev = make_qnode(n_qubits, n_layers, backend)
        self.backend_name = getattr(self.dev, "name", str(backend))
        # Small init keeps the circuit near identity at the start, which helps
        # avoid barren plateaus on a shallow ansatz.
        self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.circuit(x, self.weights)
        if isinstance(out, (list, tuple)):
            out = torch.stack([o if torch.is_tensor(o) else torch.as_tensor(o)
                               for o in out], dim=-1)
        return out.to(x.dtype).reshape(x.shape[0], self.n_qubits) \
            if out.ndim > 1 else out.to(x.dtype).reshape(1, self.n_qubits)

    def circuit_info(self) -> dict:
        specs = {}
        try:
            dummy_x = torch.zeros(1, self.n_qubits)
            s = qml.specs(self.circuit)(dummy_x, self.weights)
            res = s.resources
            specs = {"gate_count": int(res.num_gates),
                     "circuit_depth": int(res.depth),
                     "gate_types": {str(k): int(v) for k, v in dict(res.gate_types).items()}}
        except Exception:
            pass
        return {"n_qubits": self.n_qubits, "n_variational_layers": self.n_layers,
                "encoding": "Angle encoding (RY rotation, 1 feature per qubit)",
                "entanglement": "Ring of CNOTs (nearest-neighbour, periodic)",
                "measurement": f"Pauli-Z expectation on all {self.n_qubits} wires",
                "backend": self.backend_name,
                "trainable_circuit_params": int(self.weights.numel()),
                "differentiation": "backprop through the statevector simulator",
                **specs}


class QuantumClassifier(nn.Module):
    """
    PCA features -> learned projection -> quantum circuit -> linear read-out.

    The projection is trainable, so the network learns *which* directions in
    embedding space are worth encoding into the limited qubit budget, rather
    than us hard-coding the top-n PCA components.
    """

    def __init__(self, in_dim: int, n_qubits: int, n_classes: int = NUM_CLASSES,
                 n_layers: int = Q_LAYERS, backend: str = QUANTUM_BACKEND):
        super().__init__()
        self.pre = nn.Linear(in_dim, n_qubits)
        self.pre_norm = nn.BatchNorm1d(n_qubits)
        self.q = QuantumLayer(n_qubits, n_layers, backend)
        self.head = nn.Linear(n_qubits, n_classes)
        self.n_qubits = n_qubits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map features into valid rotation angles in [-pi, pi]."""
        return torch.tanh(self.pre_norm(self.pre(x))) * np.pi

    def forward(self, x: torch.Tensor, return_qfeat: bool = False):
        qin = self.encode(x)
        qf = self.q(qin)
        logits = self.head(qf)
        return (logits, qf) if return_qfeat else logits


def draw_circuit(n_qubits: int = 4, n_layers: int = 2) -> str:
    """ASCII circuit diagram for the Quantum Explainer page."""
    circuit, _ = make_qnode(n_qubits, n_layers)
    x = torch.zeros(1, n_qubits)
    w = torch.zeros(n_layers, n_qubits, 2)
    try:
        return qml.draw(circuit, max_length=200)(x, w)
    except Exception as e:
        return f"(circuit diagram unavailable: {e})"
