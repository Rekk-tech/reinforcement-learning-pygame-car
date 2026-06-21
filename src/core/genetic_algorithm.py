"""
src/core/genetic_algorithm.py
──────────────────────────────
Thuật toán Tiến hóa (Genetic Algorithm) dành cho Neuroevolution.

Pipeline mỗi generation:
  1. Evaluate  → thu thập fitness score từng xe
  2. Select    → chọn top-N elites (không bị đột biến)
  3. Crossover → lai tạo genome từ 2 elite ngẫu nhiên
  4. Mutate    → nhiễu ngẫu nhiên Gaussian lên genome con
  5. Reproduce → tạo thế hệ mới kích thước cố định

Lưu ý thiết kế:
  - Elitism: elite giữ nguyên genome, bảo toàn behavior tốt nhất
  - Tournament selection tùy chọn thay thế top-N thuần túy
  - Genome là np.ndarray → toàn bộ thao tác vector hóa (nhanh)
"""

from __future__ import annotations
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from src.core.neural_network import NeuralNetwork


class GeneticAlgorithm:
    """
    Quản lý toàn bộ vòng lặp tiến hóa.

    Attributes:
        population_size (int): Số cá thể mỗi thế hệ
        n_elites (int)       : Số elite giữ nguyên mỗi gen
        mutation_rate (float): Xác suất đột biến mỗi weight
        mutation_strength (float): Biên độ nhiễu Gaussian
        crossover_prob (float): Ngưỡng uniform crossover
        generation (int)     : Đếm số thế hệ đã qua
        best_fitness_history : Lịch sử best fitness mỗi gen
    """

    def __init__(
        self,
        population_size: int = 20,
        n_elites: int = 4,
        mutation_rate: float = 0.08,
        mutation_strength: float = 0.30,
        crossover_prob: float = 0.50,
        nn_architecture: List[int] = None,
        nn_activation: str = "tanh",
    ):
        self.population_size = population_size
        self.n_elites = n_elites
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_prob = crossover_prob
        self.nn_architecture = nn_architecture or [5, 6, 4, 2]
        self.nn_activation = nn_activation

        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []

        # Khởi tạo thế hệ đầu tiên (genome ngẫu nhiên)
        self.population: List[NeuralNetwork] = self._init_population()

    # ── Initialization ────────────────────────────────────────────────

    def _init_population(self) -> List[NeuralNetwork]:
        return [
            NeuralNetwork(
                architecture=self.nn_architecture,
                activation=self.nn_activation,
            )
            for _ in range(self.population_size)
        ]

    def _make_nn(self, genome: np.ndarray) -> NeuralNetwork:
        return NeuralNetwork(
            architecture=self.nn_architecture,
            activation=self.nn_activation,
            genome=genome,
        )

    # ── Core GA operations ────────────────────────────────────────────

    def select_elites(
        self, fitness_scores: List[float]
    ) -> List[NeuralNetwork]:
        """
        Chọn top-N cá thể theo fitness (Elitist Selection).
        Trả về list NeuralNetwork đã clone (an toàn, không reference chung).
        """
        ranked = sorted(
            zip(fitness_scores, self.population),
            key=lambda x: x[0],
            reverse=True,
        )
        return [nn.clone() for _, nn in ranked[: self.n_elites]]

    def crossover(
        self, parent_a: NeuralNetwork, parent_b: NeuralNetwork
    ) -> np.ndarray:
        """
        Uniform crossover trên genome vector.
        Mỗi weight độc lập lấy từ parent_a hoặc parent_b với p=0.5.
        """
        mask = np.random.rand(parent_a.genome_size) < self.crossover_prob
        child_genome = np.where(mask, parent_a.genome, parent_b.genome)
        return child_genome

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """
        Gaussian mutation: thêm nhiễu vào mutation_rate% các weights.

        noise ~ N(0, mutation_strength²)
        mask  ~ Bernoulli(mutation_rate)
        genome_new = genome + noise * mask
        """
        noise = np.random.randn(len(genome)) * self.mutation_strength
        mask = np.random.rand(len(genome)) < self.mutation_rate
        return genome + noise * mask

    # ── Generation step ───────────────────────────────────────────────

    def evolve(self, fitness_scores: List[float]) -> List[NeuralNetwork]:
        """
        Nhận fitness scores của generation hiện tại,
        trả về population mới cho generation tiếp theo.

        Args:
            fitness_scores: List[float] cùng thứ tự với self.population

        Returns:
            List[NeuralNetwork] — thế hệ mới kích thước population_size
        """
        assert len(fitness_scores) == len(self.population), (
            f"fitness_scores length {len(fitness_scores)} "
            f"!= population size {len(self.population)}"
        )

        # Log thống kê
        best = max(fitness_scores)
        mean = float(np.mean(fitness_scores))
        self.best_fitness_history.append(best)
        self.mean_fitness_history.append(mean)
        self.generation += 1

        # 1. Selection
        elites = self.select_elites(fitness_scores)

        # 2. Tạo thế hệ mới
        new_population: List[NeuralNetwork] = list(elites)  # giữ elites

        while len(new_population) < self.population_size:
            # Chọn 2 parent từ elite pool
            idx_a, idx_b = np.random.choice(len(elites), size=2, replace=False)
            parent_a, parent_b = elites[idx_a], elites[idx_b]

            # Crossover + Mutate
            child_genome = self.crossover(parent_a, parent_b)
            child_genome = self.mutate(child_genome)
            new_population.append(self._make_nn(child_genome))

        self.population = new_population
        return self.population

    # ── Checkpointing ─────────────────────────────────────────────────

    def save_checkpoint(self, output_dir: str = "./checkpoints") -> str:
        """
        Lưu genome của best elite ra file .npy + metadata JSON.

        Args:
            output_dir: Thư mục lưu checkpoint

        Returns:
            Đường dẫn file .npy đã lưu
        """
        ckpt_dir = Path(output_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Lấy genome của xe tốt nhất (population[0] = top elite sau evolve)
        best_genome = self.population[0].genome

        # Lưu trọng số
        weights_path = ckpt_dir / "best.npy"
        np.save(str(weights_path), best_genome)

        # Lưu metadata
        metadata = {
            "generation": self.generation,
            "best_fitness": self.best_fitness_history[-1] if self.best_fitness_history else 0.0,
            "all_time_best": max(self.best_fitness_history) if self.best_fitness_history else 0.0,
            "architecture": self.nn_architecture,
            "activation": self.nn_activation,
            "population_size": self.population_size,
            "timestamp": datetime.now().isoformat(),
        }
        meta_path = ckpt_dir / "metadata.json"
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return str(weights_path)

    def load_checkpoint(self, output_dir: str = "./checkpoints") -> dict | None:
        """
        Tải genome đã lưu và inject vào population[0].

        Args:
            output_dir: Thư mục chứa checkpoint

        Returns:
            Dict metadata nếu load thành công, None nếu không tìm thấy file
        """
        ckpt_dir = Path(output_dir)
        weights_path = ckpt_dir / "best.npy"
        meta_path = ckpt_dir / "metadata.json"

        if not weights_path.exists():
            return None

        # Tải genome
        genome = np.load(str(weights_path))
        loaded_nn = self._make_nn(genome)

        # Inject vào slot đầu tiên (best position)
        self.population[0] = loaded_nn

        # Tải metadata
        metadata = {}
        if meta_path.exists():
            with open(str(meta_path), "r", encoding="utf-8") as f:
                metadata = json.load(f)
            # Khôi phục generation counter
            self.generation = metadata.get("generation", 0)

        return metadata

    # ── Stats & Utilities ─────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Trả về thống kê generation hiện tại."""
        if not self.best_fitness_history:
            return {}
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness_history[-1],
            "mean_fitness": self.mean_fitness_history[-1],
            "all_time_best": max(self.best_fitness_history),
        }

    def log_generation(self, fitness_scores: List[float]) -> str:
        """Format log string cho 1 generation."""
        best = max(fitness_scores)
        mean = float(np.mean(fitness_scores))
        worst = min(fitness_scores)
        return (
            f"Gen {self.generation:3d} | "
            f"best={best:7.1f} | mean={mean:7.1f} | worst={worst:7.1f} | "
            f"alive={sum(1 for f in fitness_scores if f > 0):2d}/{len(fitness_scores)}"
        )

    def __repr__(self) -> str:
        return (
            f"GeneticAlgorithm("
            f"pop={self.population_size}, "
            f"elites={self.n_elites}, "
            f"mut_rate={self.mutation_rate}, "
            f"gen={self.generation})"
        )

