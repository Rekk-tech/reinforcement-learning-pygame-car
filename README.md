# 🏎️ Deep Learning Cars

Dự án mô phỏng xe tự lái 2D sử dụng hai thuật toán học máy:
- **Genetic Algorithm (GA)** — Neuroevolution tiến hóa thế hệ xe
- **Proximal Policy Optimization (PPO)** — Deep Reinforcement Learning

---

## 🗂 Cấu trúc Dự án

```
deep-learning-cars/
├── main.py                     # Entry point chính
├── mlflow_tracking.py          # MLflow experiment tracker
├── configs/config.yaml         # Toàn bộ tham số cấu hình
├── src/
│   ├── core/
│   │   ├── ppo_agent.py        # PPO Agent (ActorCritic + RolloutBuffer)
│   │   ├── genetic_algorithm.py# GA + Elitism + Crossover + Mutation
│   │   ├── neural_network.py   # Feed-forward NN cho GA
│   │   └── cnn_encoder.py      # CNN Encoder (pixel mode, tuỳ chọn)
│   ├── simulation/
│   │   ├── car.py              # Agent xe (sensor, physics, gym API)
│   │   ├── track.py            # Track definitions + PRESET_TRACKS
│   │   └── reward.py           # Shaped Reward Function cho PPO
│   ├── rendering/
│   │   ├── renderer.py         # Pygame rendering engine
│   │   └── hud.py              # HUD overlay (stats, fitness)
│   └── ui/controls.py          # Keyboard/mouse controls
├── checkpoints/
│   ├── best_ppo_model.pt       # Model tốt nhất từ trước đến nay
│   ├── best_ppo_model_1437_backup.pt  # Backup kỷ lục 1437.1 (oval ep581)
│   └── ppo_model.pt            # Checkpoint mới nhất (dùng để resume)
├── docs/
│   ├── PROJECT_REVIEW.md       # Phân tích kỹ thuật chi tiết
│   ├── training_guide.md       # Hướng dẫn huấn luyện từng bước
│   └── document.md             # Tài liệu thuyết trình tổng thể
└── mlruns/                     # MLflow experiment data
```

---

## ⚡ Cài đặt & Chạy nhanh

```bash
# Tạo môi trường ảo
python -m venv venv
.\venv\Scripts\activate        # Windows

# Cài thư viện
pip install -r requirements.txt

# Chạy PPO (thuật toán chính) trên city_simple
.\venv\Scripts\python.exe main.py --algorithm ppo --track city_simple

# Resume từ model đã train
.\venv\Scripts\python.exe main.py --algorithm ppo --track city_simple --resume

# Chạy Genetic Algorithm
.\venv\Scripts\python.exe main.py --algorithm ga --track oval

# Xem MLflow dashboard
.\venv\Scripts\python.exe -m mlflow ui   # → http://localhost:5000
```

---

## 🗺 Danh sách Map

| Map           | Loại              | Độ khó | Mô tả                               |
|---------------|-------------------|--------|--------------------------------------|
| `map_straight`| Đường thẳng       | ⭐     | Học cơ bản: ga, cảm biến            |
| `map_u_turn`  | U-Turn (Hairpin)  | ⭐⭐⭐ | Cua 180° — buộc phải học phanh      |
| `map_zigzag`  | Zíczắc            | ⭐⭐⭐ | Đổi hướng liên tục trái-phải        |
| `oval`        | Oval              | ⭐⭐   | Cua bo tròn rộng, tốc độ cao        |
| `city_simple` | Chữ nhật bo góc   | ⭐⭐   | Đường thành phố cơ bản              |
| `city_oval`   | Kết hợp           | ⭐⭐⭐ | Đoạn thẳng dài + cua oval mượt      |
| `city`        | Số 8 vuông        | ⭐⭐⭐⭐| Thách thức cao nhất (inner/outer)   |

---

## 🧠 Thuật toán

### Proximal Policy Optimization (PPO)
- **Kiến trúc**: Actor-Critic, shared backbone `obs(6) → FC[64] → FC[64] → Actor(2) + Critic(1)`
- **Observation (6D)**: 5 tia cảm biến (raycast) + angle_diff đến checkpoint
- **Action (2D)**: `[turn ∈ (-1,1), engine ∈ (-1,1)]` liên tục
- **Hàm Reward**: Context-Aware Speed Bonus + Checkpoint + nhiều hình phạt

### Genetic Algorithm (GA)
- **Neuroevolution**: tiến hóa trọng số neural network qua nhiều thế hệ
- **Elitism**: giữ nguyên top-N cá thể tốt nhất
- **Operators**: Uniform Crossover + Gaussian Mutation với Annealing

---

## 📊 Tracking & Experiment

MLflow tự động ghi lại mọi run (hyperparameters, metrics, model artifacts):
```bash
.\venv\Scripts\python.exe -m mlflow ui   # → http://localhost:5000
```

---

## 🏆 Kết quả Huấn luyện

| Mốc đạt được    | Map           | Best Reward | Episodes |
|-----------------|---------------|-------------|---------|
| Tốt nghiệp      | map_straight  | ~500        | ~100    |
| Tốt nghiệp      | oval          | **1437.1**  | 581     |
| Tốt nghiệp      | city_simple   | ~1045       | 582+    |
| Đang thử nghiệm | city_oval     | 1246+       | 583+    |
