# Deep Learning Cars — Neuroevolution Simulation

Hệ thống mô phỏng xe tự lái học cách lái bằng **Neuroevolution**: kết hợp Mạng thần kinh nhân tạo (Neural Network) và Thuật toán Tiến hóa (Genetic Algorithm) trong bài toán Học tăng cường (Reinforcement Learning) — hoàn toàn không dùng gradient descent.

---

## Giới thiệu

Mỗi chiếc xe là một **Agent** độc lập, sở hữu một "bộ não" là một mạng thần kinh nhỏ. Thay vì dạy xe cụ thể cách lái, hệ thống chỉ đặt ra một mục tiêu duy nhất: **sống sót và đi xa nhất có thể**. Qua nhiều thế hệ tiến hóa, các chiếc xe dần học được cách bẻ lái tránh tường, tăng tốc trên đường thẳng và xử lý các khúc cua phức tạp.

```
Thế hệ 1:  Xe đâm tường ngay lập tức (hành vi ngẫu nhiên)
Thế hệ 5:  Xe biết rẽ cơ bản, tránh tường gần
Thế hệ 20: Xe hoàn thành vòng đua, tối ưu đường đi
```

---

## Kiến trúc hệ thống

```
deep-learning-cars/
│
├── README.md                   # Tài liệu tổng quan (file này)
├── main.py                     # Entry point — chạy simulation
├── requirements.txt            # Dependencies Python
│
├── configs/
│   └── config.yaml             # Tham số GA, NN, vật lý, render
│
├── src/
│   ├── core/                   # Lõi AI: NN + Genetic Algorithm
│   │   ├── __init__.py
│   │   ├── neural_network.py   # Feedforward NN (5→6→4→2, tanh)
│   │   └── genetic_algorithm.py# Selection, crossover, mutation
│   │
│   ├── simulation/             # Vật lý + môi trường
│   │   ├── __init__.py
│   │   ├── car.py              # Agent: cảm biến, động cơ, va chạm
│   │   ├── track.py            # Sa hình: waypoints, tường, checkpoints
│   │   └── physics.py          # Quán tính, góc lái, tốc độ
│   │
│   ├── rendering/              # Hiển thị
│   │   ├── __init__.py
│   │   ├── renderer.py         # Vẽ track, xe, sensor rays
│   │   └── hud.py              # Overlay: generation, fitness, graphs
│   │
│   └── ui/                     # Giao diện điều khiển
│       ├── __init__.py
│       └── controls.py         # Speed slider, pause/reset, stats panel
│
├── tests/
│   ├── test_neural_network.py  # Unit test: forward pass, shapes
│   ├── test_genetic_algorithm.py# Unit test: crossover, mutation, elites
│   └── test_simulation.py      # Integration test: car + track
│
└── docs/
    ├── architecture.md         # Chi tiết kiến trúc AI
    └── training_guide.md       # Hướng dẫn huấn luyện và tuning
```

---

## Thành phần AI

### Neural Network (Bộ não xe)

Kiến trúc `5 → 6 → 4 → 2` với activation `tanh`:

| Layer | Neurons | Vai trò |
|-------|---------|---------|
| Input | 5 | Khoảng cách 5 tia cảm biến [0,1] |
| Hidden 1 | 6 | Phát hiện pattern tránh chướng ngại |
| Hidden 2 | 4 | Tổng hợp quyết định lái |
| Output | 2 | [turn ∈ (-1,1), engine ∈ (-1,1)] |

Tổng: **74 parameters** — nhỏ gọn, tiến hóa nhanh.

### Genetic Algorithm (Bộ não tiến hóa)

| Bước | Kỹ thuật | Tham số mặc định |
|------|----------|-----------------|
| Selection | Elitist | Top 4/20 xe |
| Crossover | Uniform per weight | p=0.5 |
| Mutation | Gaussian noise + mask | rate=8%, strength=0.3 |
| Population | Fixed size | 20 xe/generation |

---

## Cài đặt & Chạy

```bash
# Cài dependencies
pip install -r requirements.txt

# Chạy simulation với config mặc định
python main.py

# Chạy với config tùy chỉnh
python main.py --config configs/config.yaml --track oval

# Chỉ train không render (nhanh hơn)
python main.py --headless --generations 100
```

---

## Tham số quan trọng (`configs/config.yaml`)

```yaml
genetic_algorithm:
  population_size: 20    # Tăng → đa dạng hơn, chậm hơn
  n_elites: 4            # Giữ lại top N mỗi thế hệ
  mutation_rate: 0.08    # Xác suất đột biến mỗi weight
  mutation_strength: 0.3 # Biên độ đột biến (Gaussian std)

neural_network:
  architecture: [5,6,4,2]
  activation: tanh

simulation:
  n_sensors: 5           # Số tia cảm biến
  sensor_range: 120      # Pixel, tầm nhìn tối đa
  max_speed: 3.0         # Pixel/frame
  friction: 0.02         # Hãm tốc tự nhiên
```

---

## Kết quả kỳ vọng

| Generation | Hành vi điển hình |
|------------|------------------|
| 1–3 | Ngẫu nhiên, đa số đâm tường ngay |
| 4–8 | Một số xe biết rẽ, đi được vài giây |
| 10–15 | Elite xe hoàn thành 1 vòng cơ bản |
| 20+ | Xe tối ưu đường đi, tốc độ cao ổn định |

---

## Công nghệ

- **Python 3.10+** với NumPy (tính toán NN)
- **Pygame** (simulation + rendering)
- **PyYAML** (cấu hình)
- **Matplotlib** (biểu đồ fitness theo thế hệ)
