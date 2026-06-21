"""
src/core/neural_network.py
──────────────────────────
Mạng thần kinh feedforward nhỏ gọn cho bài toán Neuroevolution.

Kiến trúc mặc định: 5 → 6 → 4 → 2 (tanh activation)
  - Input  (5): khoảng cách 5 tia cảm biến, chuẩn hóa về [0, 1]
  - Hidden1 (6): phát hiện pattern nguy hiểm
  - Hidden2 (4): tổng hợp quyết định
  - Output  (2): [turn ∈ (-1,1), engine ∈ (-1,1)]

Toàn bộ weights + biases được flatten thành 1 vector 1D (genome)
để Genetic Algorithm có thể crossover và mutate trực tiếp.
"""

from __future__ import annotations
import numpy as np
from typing import List


ACTIVATIONS = {
    "tanh":    (np.tanh,    lambda a: 1 - a ** 2),
    "relu":    (lambda x: np.maximum(0, x), lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), lambda a: a * (1 - a)),
}


class NeuralNetwork:
    """
    Feedforward Neural Network — tối ưu cho Neuroevolution.

    Genome = vector 1D gồm tất cả weights và biases flatten.
    Không lưu gradient, không cần optimizer — chỉ cần forward pass.
    """

    def __init__(
        self,
        architecture: List[int] = None,
        activation: str = "tanh",
        genome: np.ndarray = None,
        weight_init_std: float = 0.8,
    ):
        self.arch = architecture or [5, 6, 4, 2]
        self.activation_fn, _ = ACTIVATIONS[activation]
        self.activation_name = activation
        self._weight_init_std = weight_init_std

        # Tính kích thước genome
        self.genome_size = sum(
            self.arch[i] * self.arch[i + 1] + self.arch[i + 1]
            for i in range(len(self.arch) - 1)
        )

        if genome is not None:
            assert len(genome) == self.genome_size, (
                f"Genome size mismatch: expected {self.genome_size}, got {len(genome)}"
            )
            self.genome = genome.copy()
        else:
            self.genome = np.random.randn(self.genome_size) * weight_init_std

        self._weights, self._biases = self._unpack_genome(self.genome)

    # ── Genome packing / unpacking ────────────────────────────────────

    def _unpack_genome(self, genome: np.ndarray):
        """Chuyển vector 1D → list các weight matrix + bias vector."""
        weights, biases = [], []
        idx = 0
        for i in range(len(self.arch) - 1):
            fan_in, fan_out = self.arch[i], self.arch[i + 1]
            w_size = fan_in * fan_out
            W = genome[idx: idx + w_size].reshape(fan_out, fan_in)
            idx += w_size
            b = genome[idx: idx + fan_out].reshape(fan_out, 1)
            idx += fan_out
            weights.append(W)
            biases.append(b)
        return weights, biases

    def set_genome(self, genome: np.ndarray) -> None:
        """Cập nhật genome (dùng sau crossover/mutation)."""
        self.genome = genome.copy()
        self._weights, self._biases = self._unpack_genome(self.genome)

    # ── Forward pass ──────────────────────────────────────────────────

    def forward(self, inputs: List[float]) -> np.ndarray:
        """
        Nhận 5 giá trị cảm biến → trả về [turn, engine].

        Args:
            inputs: List[float] độ dài = arch[0], mỗi giá trị ∈ [0, 1]

        Returns:
            np.ndarray shape (arch[-1],) — giá trị ∈ (-1, 1) do tanh
        """
        a = np.array(inputs, dtype=np.float32).reshape(-1, 1)
        for W, b in zip(self._weights, self._biases):
            a = self.activation_fn(W @ a + b)
        return a.flatten()

    # ── Utilities ─────────────────────────────────────────────────────

    def clone(self) -> "NeuralNetwork":
        return NeuralNetwork(
            architecture=self.arch,
            activation=self.activation_name,
            genome=self.genome.copy(),
        )

    def __repr__(self) -> str:
        arch_str = "→".join(map(str, self.arch))
        return (
            f"NeuralNetwork({arch_str}, "
            f"activation={self.activation_name}, "
            f"genome_size={self.genome_size})"
        )
