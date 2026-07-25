# 🎓 Training Guide — Deep Learning Cars (PPO)

Hướng dẫn huấn luyện từng bước từ đầu đến khi AI tự lái thành thạo.

---

## Lộ trình Curriculum Learning

```
Giai đoạn 0       Giai đoạn 1       Giai đoạn 2       Giai đoạn 3
──────────────    ──────────────    ──────────────    ──────────────
map_straight  →  oval / city_simple →  city_oval   →  city (cuối)
 (khởi động)      (tốt nghiệp)      (kết hợp)       (thành thạo)
```

---

## Giai đoạn 0: Khởi động từ đầu

```bash
.\venv\Scripts\python.exe main.py --algorithm ppo --track map_straight
```

Mục tiêu: AI học phản xạ cơ bản — nhấn ga, đọc cảm biến.
Dừng khi: `avg100 > 300` hoặc `best > 400`.

---

## Giai đoạn 1: map tốt nghiệp Oval / City Simple

```bash
# Sau khi đã có model từ map_straight
.\venv\Scripts\python.exe main.py --algorithm ppo --track oval --resume
```

Mục tiêu: AI học ôm cua rộng ở tốc độ cao.
Dừng khi: `best > 1000` (tương đương đi được 20+ checkpoint liên tiếp).
**Kỷ lục đạt được**: 1437.1 điểm tại ep 581.

---

## Giai đoạn 2: Map kết hợp City Oval

```bash
.\venv\Scripts\python.exe main.py --algorithm ppo --track city_oval --resume
```

Mục tiêu: Học kết hợp đường thẳng dài + cua oval mượt mà.

---

## Giai đoạn 3: Thách thức Final — City

```bash
.\venv\Scripts\python.exe main.py --algorithm ppo --track city --resume
```

---

## Resume & Checkpoint

Khi chạy `--resume`, script tìm file theo thứ tự ưu tiên:
1. `checkpoints/ppo_model.pt` (checkpoint mới nhất)
2. `checkpoints/best_ppo_model.pt` (tốt nhất nếu không có cái trên)

### Khôi phục model từ MLflow (nếu file bị ghi đè)

```python
import torch, shutil
src = "mlruns/2/<run_id>/artifacts/best_ppo_model.pt"
shutil.copy(src, "checkpoints/best_ppo_model.pt")
```

---

## Tham số Reward Shaping — Khi nào điều chỉnh?

| Tình huống | Hành động |
|-----------|-----------|
| AI không chịu phanh (lút ga đâm tường) | Tăng `speed_angle_penalty` |
| Điểm âm quá sâu (khủng hoảng) | Giảm `speed_angle_penalty` |
| AI đứng im (scared) | Giảm `stuck_penalty` hoặc tăng `alive_bonus` |
| AI chạy giật cục (zigzag) | Tăng `jerk_penalty`, `turn_penalty` |
| AI bám sát tường | Tăng `proximity_penalty` |
| AI không chịu khám phá | Tăng `entropy_coeff` trong config |

---

## Environment Relaxation — Kỹ thuật dạy phanh

Khi AI bị "nghiện chân ga" (Policy Inertia):

```yaml
# Giảm tốc độ để AI dễ ôm cua (trong car.py)
MAX_SPEED: float = 1.5   # Giảm từ 3.0

# Hoặc tăng entropy để ép khám phá (trong config.yaml)
entropy_coeff: 0.05      # Tăng từ 0.01
```

Sau khi AI học được hành vi phanh, tăng dần lại:
```python
MAX_SPEED = 2.0  # → rồi 3.0
```

---

## Quan sát UI trong lúc Train

| Màu sắc | Ý nghĩa |
|---------|---------|
| 🟠 Xe cam | Xe tốt nhất hiện tại |
| 🔵 Xe xanh | Xe thường |
| 🔴 Xe đỏ | Xe vừa chết |
| Tia trắng | 5 tia cảm biến (raycast) |
| Mũi tên cam | La bàn hướng đến checkpoint |
| Vòng tròn vàng | Checkpoint |

| HUD Item | Ý nghĩa |
|----------|---------|
| `reward` | Reward tích lũy của episode này |
| `avg100` | Trung bình reward 100 episode gần nhất |
| `best` | Kỷ lục episode reward trong phiên chạy |
| `steps` | Tổng số steps từ đầu train |

---

## MLflow Dashboard

```bash
.\venv\Scripts\python.exe -m mlflow ui
# Mở http://localhost:5000
```

Xem được:
- So sánh nhiều run song song
- Biểu đồ reward theo thời gian
- Download model artifact từ bất kỳ run nào

---

## Lưu ý An toàn (Tránh mất model)

> Khi script khởi động, `best_episode_reward = -inf` nên episode đầu tiên BẤT KỲ cũng sẽ ghi đè `best_ppo_model.pt`!

**Quy tắc bảo vệ model quý giá:**
1. Luôn giữ file `best_ppo_model_1437_backup.pt` (hoặc đặt tên theo kỷ lục)
2. Trước khi thử nghiệm rủi ro, copy backup:
   ```powershell
   Copy-Item checkpoints\best_ppo_model.pt checkpoints\backup_before_exp.pt
   ```
3. Dùng MLflow để tra cứu và phục hồi bất kỳ model nào trong lịch sử
