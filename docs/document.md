# 🚗 Deep Learning Cars — Tài liệu Thuyết trình

> Hành trình xây dựng một chiếc xe tự lái từ số 0 bằng Deep Reinforcement Learning

---

## 1. Giới thiệu Dự án

**Deep Learning Cars** là một dự án mô phỏng xe tự lái 2D, nơi một mạng nơ-ron học cách điều khiển chiếc xe qua nhiều đường đua khác nhau — hoàn toàn tự động, không có quy tắc được lập trình cứng.

### Mục tiêu
- Xây dựng một agent AI có khả năng tự học cách lái xe
- Áp dụng hai paradigm học máy: **Tiến hóa (GA)** và **Học tăng cường (PPO)**
- Khám phá các thách thức thực tế: reward hacking, policy inertia, curriculum learning

### Công nghệ sử dụng
- **Python 3.13** + **PyTorch 2.x** — nền tảng Deep Learning
- **Pygame 2.6** — môi trường mô phỏng 2D
- **MLflow** — tracking và quản lý experiments

---

## 2. Hai Thuật toán — Hai Triết lý

### 2A. Genetic Algorithm (GA) — "Tiến hóa"

```
Thế hệ 1 (20 xe ngẫu nhiên)
       ↓ Evaluate (đua thực tế)
       ↓ Select (top 4 elites)
       ↓ Crossover (lai gen)
       ↓ Mutate (đột biến ngẫu nhiên)
Thế hệ 2 (20 xe mới)
       ↓ ...
Thế hệ N (xe champion)
```

**Ưu điểm**: Đơn giản, không cần gradient, tốt cho bài toán exploration
**Nhược điểm**: Chậm hội tụ, không học từng bước nhỏ

### 2B. Proximal Policy Optimization (PPO) — "Học từ kinh nghiệm"

```
Xe quan sát môi trường (6 số liệu)
       ↓
Mạng nơ-ron ra quyết định (2 actions)
       ↓
Môi trường phản hồi reward
       ↓ (lặp 4096 lần)
Tính Advantage bằng GAE
       ↓
Cập nhật trọng số bằng gradient (10 epochs)
       ↓ (lặp liên tục)
Policy ngày càng tốt hơn
```

**Ưu điểm**: Sample-efficient, hội tụ nhanh, có thể fine-tune
**Nhược điểm**: Dễ bị stuck ở local optima (policy inertia)

---

## 3. Cảm giác của một chiếc xe AI

### Xe "nhìn" thế giới như thế nào?

Tưởng tượng bạn bị bịt mắt và chỉ có 5 que dò phía trước. Đó chính xác là cách AI "nhìn":

```
          🏁 (checkpoint)
          ↑
    s0  s1  s2  s3  s4
    ╲   ╲   |   /   /
     ╲   ╲  |  /   /
      ╲   ╲ | /   /
       ════════════  ← chiếc xe
```

- **5 tia sensor**: Đo khoảng cách đến tường trong 5 góc khác nhau
- **1 la bàn**: Đo góc lệch đến checkpoint tiếp theo

Kết quả: 6 con số `[0.8, 0.9, 0.3, 1.0, 1.0, -0.4]` — đây là tất cả những gì AI biết về thế giới.

### Xe "hành động" như thế nào?

Mạng nơ-ron xuất ra 2 con số:
- **turn** ∈ (-1, 1): Âm = trái, Dương = phải
- **engine** ∈ (-1, 1): Dương = ga, Âm = phanh

---

## 4. Kiến trúc Mạng Nơ-ron (Actor-Critic)

```
Observation (6D)
       │
       ├─ Linear(6→64) → Tanh
       │
       └─ Linear(64→64) → Tanh
              │
       ┌──────┴──────┐
       │             │
  Actor Head     Critic Head
  Linear(64→2)   Linear(64→1)
  + log_std      
       │             │
  Normal(μ,σ)    V(state)
  → tanh          (giá trị kỳ vọng)
       │
  [turn, engine]
```

**Actor**: Quyết định hành động (Policy π)
**Critic**: Đánh giá trạng thái hiện tại có tốt không (Value V)

---

## 5. Hàm Reward — Ngôn ngữ của Giáo viên

Reward là cách chúng ta "nói chuyện" với AI, cho nó biết hành động nào tốt/xấu.

### Cấu trúc Reward hiện tại

| Loại | Công thức | Ý nghĩa |
|------|----------|---------|
| ✅ Alive | +0.05/frame | Sống thêm 1 frame |
| ✅ Checkpoint | +50.0 | Qua được điểm đích |
| ✅ Speed* | Phụ thuộc ngữ cảnh | Thưởng nhanh ở thẳng, phạt nhanh ở cua |
| ❌ Wall crash | -10.0 | Va chạm tường |
| ❌ Stuck | -0.5/frame | Đứng im |
| ❌ Zigzag | -0.05×|turn| | Lạng lách |
| ❌ Speed at curve | -2.0 | Chạy nhanh khi góc gắt |

### ✨ Thiết kế đột phá: Context-Aware Speed Bonus

```
reward_speed = weight × (v/v_max) × (1 - 2|Δθ|/π)
```

| Góc lệch | Hệ số | Hiệu quả |
|---------|-------|---------|
| 0° (thẳng) | ×1.0 | Thưởng ga đầy đủ |
| 45° (cua nhẹ) | ×0.0 | Trung tính |
| 90° (cua gắt) | ×-1.0 | Phạt càng nhanh càng nặng |

Ý tưởng: **Một công thức duy nhất thay thế 2 điều khoản phức tạp** — không cần threshold binary, không có reward hacking.

---

## 6. Quá trình Huấn luyện — Câu chuyện AI "trưởng thành"

### Timeline thực tế của dự án

```
Episodes 1-100:    AI điên loạn, va tường liên tục
Episodes 100-300:  AI học đi thẳng, tránh tường cơ bản
Episodes 300-450:  AI phá kỷ lục city_simple (448→∞)
Episodes 450-581:  AI chinh phục map oval, đạt 1437.1 điểm 🏆
Episodes 582+:     AI tiếp tục chinh phục city_simple, city_oval
```

### Vấn đề "Nghiện Chân Ga" (Policy Inertia)

Đây là bài học thực tế thú vị nhất của dự án. Sau khi AI học thành công trên các map cua rộng, nó hình thành "niềm tin" cứng nhắc: **"Nhấn ga = tốt"**. Khi chuyển sang map U-turn (cua 180°), nó tiếp tục lút ga và đâm thẳng vào tường.

**Các kỹ thuật chúng tôi đã thử:**
1. ❌ Tăng penalty: AI vẫn không thay đổi (gradient quá yếu)
2. ❌ Look-ahead compass: Gây State Space Shift, model cũ không đọc được
3. ✅ Environment Relaxation: Giảm MAX_SPEED xuống 1.5, buộc AI học lại
4. ✅ Context-Aware Reward: Thiết kế lại reward function từ gốc

---

## 7. Curriculum Learning — Dạy học theo trình độ

Triết lý giáo dục: Không thể dạy tích phân cho người chưa biết cộng trừ.

```
Level 1: map_straight
"Học cách nhấn ga và đọc cảm biến"
→ Không cần phanh, không cần cua

Level 2: oval / city_simple
"Học cua mượt mà ở tốc độ cao"
→ Cua rộng, không đòi hỏi phanh

Level 3: city_oval (map tổng hợp)
"Kết hợp đường thẳng dài + cua oval"
→ Thử thách tổng hợp

Level 4: map_u_turn / city
"Học phanh và cua hairpin"
→ Cần kỹ năng phanh thực sự
```

---

## 8. Công cụ Tracking — MLflow

Mỗi lần chạy thử nghiệm, hệ thống tự động ghi lại:
- Toàn bộ hyperparameters
- Reward theo từng episode
- Policy loss, Value loss, Entropy
- File model weights (artifact)

Điều này cho phép chúng tôi:
- So sánh hiệu quả của nhiều cách điều chỉnh reward
- Phục hồi bất kỳ model nào trong lịch sử (như file 1437.1 được khôi phục nhiều lần)
- Không bao giờ mất công sức training

---

## 9. Bài học Rút ra

### Kỹ thuật
- **State Space Shift**: Thay đổi ý nghĩa input sẽ làm model cũ "điên loạn" — không thể dùng `--resume` sau khi thay đổi cấu trúc observation
- **Reward Hacking**: Nếu reward per-frame quá lớn, AI sẽ tìm cách lạm dụng thay vì học đúng hành vi
- **Policy Inertia**: Model đã hội tụ rất khó thay đổi hành vi bằng penalty nhỏ — cần thay đổi cấu trúc reward hoặc làm mềm môi trường

### Thực tế
- Checkpoint và backup model là cực kỳ quan trọng — một dòng code sai có thể xóa sổ hàng giờ training
- MLflow không chỉ là tool tracking mà còn là "bảo hiểm" cho model
- Việc hiểu bản chất vật lý của bài toán (bán kính cong vs tốc độ tối đa) giúp thiết kế reward hiệu quả hơn nhiều so với thử sai

---

## 10. Kết quả & Demo

| Thành tựu | Số liệu |
|-----------|---------|
| Kỷ lục cao nhất | **1437.1 điểm** (map oval, ep 581) |
| Số map đã tốt nghiệp | 3 (map_straight, oval, city_simple) |
| Tổng số steps training | ~260,000+ |
| Số experiments MLflow | 20+ runs |
| Tổng thời gian phát triển | ~2 tuần |

### Demo trực tiếp
```bash
# Xem AI phá đảo city_simple
.\venv\Scripts\python.exe main.py --algorithm ppo --track city_simple --resume

# Xem AI thách thức city_oval
.\venv\Scripts\python.exe main.py --algorithm ppo --track city_oval --resume

# So sánh AI vs GA
.\venv\Scripts\python.exe main.py --algorithm ga --track oval
```

---

## 11. Hướng phát triển tiếp theo

- [ ] **CNN Mode**: Sử dụng camera pixel thay vì sensor để AI "nhìn" như người thật
- [ ] **Multi-agent**: Nhiều xe cùng train song song để tăng sample efficiency
- [ ] **Curriculum tự động**: Hệ thống tự động nâng cấp map khi AI đạt ngưỡng điểm
- [ ] **Transfer Learning**: Thử nghiệm chuyển model sang các bài toán lái xe thực tế
- [ ] **Adaptive Reward**: Tự động điều chỉnh hệ số reward dựa trên hiệu suất training

---

*Dự án được phát triển bằng phương pháp Iterative Design — thất bại nhanh, học nhanh, cải tiến liên tục.*
