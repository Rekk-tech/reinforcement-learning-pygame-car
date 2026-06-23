# Deep Learning Cars — Autonomous Driving Simulation

Hệ thống mô phỏng xe tự lái học cách điều khiển thông qua **Trí tuệ nhân tạo (AI)**. Dự án hỗ trợ cả **Neuroevolution (Genetic Algorithm)** và **Deep Reinforcement Learning (PPO)**, cho phép so sánh sự khác biệt giữa các phương pháp tiếp cận từ đơn giản (mạng FC nhỏ xíu với cảm biến) đến phức tạp (mạng CNN lớn với hình ảnh pixel).

---

## 🌟 Các tính năng chính

1. **3 Chế độ hoạt động (Modes)**:
   - **GA + Sensor**: Mạng nơ-ron bé gọn (74 tham số) tối ưu bằng thuật toán tiến hóa. Nhanh và nhẹ.
   - **PPO + Sensor**: Agent RL (Proximal Policy Optimization) học từ phần thưởng (Shaped Reward) qua 5 tia cảm biến.
   - **PPO + Pixel (CNN)**: Agent RL quan sát hình ảnh camera (84x84) với 4 khung hình gộp lại (Frame Stacking) qua mạng Convolutional Neural Network (CNN) ~1.7 triệu tham số.

2. **Môi trường & Vật lý (Physics)**:
   - Các track (sa hình) linh hoạt: `oval` (cơ bản), `figure8` (khó hơn), `city` (góc cua vuông).
   - Công thức vật lý bao gồm tốc độ, gia tốc, ma sát và góc lái phụ thuộc vận tốc.

3. **Giao diện & HUD trực quan (UI)**:
   - Rendering Pygame mượt mà.
   - Bảng điều khiển tốc độ mô phỏng trực tiếp (Speed Slider 1x-10x), nút Pause, Reset.
   - Heads-Up Display (HUD): Cập nhật biểu đồ fitness/reward thời gian thực, progress bars, thông tin thế hệ/episode.

4. **MLOps Tracking**:
   - Tích hợp **MLflow** theo dõi toàn bộ hyperparameters, metrics (reward, loss, fitness), và tự động lưu model artifacts (checkpoints).
   - Hỗ trợ MLflow server qua Docker Compose.

---

## 📁 Kiến trúc hệ thống

```text
d:\files\
├── main.py                       # Entry point — chạy GA / PPO
├── mlflow_tracking.py            # Quản lý MLflow logs & artifacts
├── requirements.txt              # Dependencies
├── docker-compose.yml            # Khởi chạy MLflow Server
│
├── configs/
│   └── config.yaml               # Mọi hyperparams được quản lý tập trung ở đây
│
├── src/
│   ├── core/                     # Lõi AI
│   │   ├── neural_network.py     # Feedforward NN cho GA
│   │   ├── genetic_algorithm.py  # Thuật toán tiến hóa
│   │   ├── ppo_agent.py          # Actor-Critic, GAE, PPO loss
│   │   └── cnn_encoder.py        # CNN (3 lớp Conv2d -> 512d)
│   │
│   ├── simulation/               # Vật lý môi trường
│   │   ├── car.py                # Agent xe (cảm biến, điều khiển, gym_step)
│   │   ├── track.py              # Sa hình (waypoints, collision)
│   │   ├── camera.py             # Crop 84x84, grayscale, frame stacking
│   │   └── reward.py             # Hàm phần thưởng (alive, speed, penalty)
│   │
│   ├── rendering/                # Hiển thị
│   │   ├── renderer.py           # Vẽ đồ họa Pygame
│   │   └── hud.py                # HUD (biểu đồ, stats)
│   │
│   └── ui/
│       └── controls.py           # Giao diện tương tác (Pause, Reset, Speed)
│
└── docs/
    └── PROJECT_REVIEW.md         # Bản đánh giá chi tiết toán học và kiến trúc
```

---

## 🚀 Cài đặt & Chạy

### Cài đặt môi trường

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
.\venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

### Cách chạy mô phỏng

```bash
# 1. GA (Neuroevolution) + Sensor + Oval track (Mặc định)
python main.py

# 2. GA nhưng đổi sang sa hình số 8, chạy không giao diện (headless) nhanh x10
python main.py --track figure8 --headless --generations 100

# 3. PPO + Sensor (Cần chuyển mode: sensor trong config.yaml)
python main.py --algorithm ppo --track city

# 4. Resume training từ checkpoint
python main.py --algorithm ppo --resume
```

**Phím tắt giao diện**:
- `Space`: Tạm dừng / Tiếp tục
- `R`: Reset lại từ đầu
- `+/-` (hoặc `1-9`, `0`): Chỉnh tốc độ mô phỏng (1x-10x)
- `Esc`: Thoát

---

## 🧠 Chuyển đổi giữa Sensor và Pixel Mode

Bạn có thể thay đổi thiết lập trong file `configs/config.yaml`:

```yaml
perception:
  # Chọn "sensor" (5 tia cảm biến, nhẹ) hoặc "pixel" (CNN + 84x84 cam, nặng)
  mode: sensor
```
*Lưu ý: Pixel mode rất nặng trên CPU vì mô hình CNN có 1.7M tham số. Bạn nên huấn luyện dài (2000+ episodes) ở chế độ này.*

---

## 📈 Theo dõi với MLflow

Mọi phiên huấn luyện đều được lưu log tự động.

1. **Khởi động Server MLflow**:
```bash
python -m mlflow ui
```
Hoặc dùng Docker:
```bash
docker-compose up -d
```

2. **Truy cập**: Vào URL `http://127.0.0.1:5000` trên trình duyệt để xem biểu đồ hội tụ, so sánh các phiên bản chạy thử và lấy file trọng số (`.npy` hoặc `.pt`).
