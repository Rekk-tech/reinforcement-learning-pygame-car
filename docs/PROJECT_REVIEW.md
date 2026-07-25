# 📋 Project Review — Deep Learning Cars

> Phân tích kỹ thuật toàn diện về kiến trúc, thuật toán, và quá trình huấn luyện.

---

## 1. Tổng quan Kiến trúc

### Sơ đồ luồng dữ liệu

```
Environment (Track + Physics)
        │
        ▼
   Car.cast_sensors()         ← 5 tia raycast, chuẩn hoá [0,1]
   Car.get_observation()      ← 6D vector: [s0..s4, angle_diff]
        │
        ▼
   PPOAgent.select_action()   ← ActorCritic forward pass
        │ action = [turn, engine] ∈ (-1, 1)
        ▼
   Car.gym_step(action, track)
        │ next_obs, base_reward, done, info
        ▼
   RewardCalculator.compute() ← Shaped reward
        │
        ▼
   RolloutBuffer.store()
        │ (sau mỗi 4096 steps)
        ▼
   PPOAgent.update()          ← GAE → PPO clipped objective
        │
        ▼
   Checkpoint save (best / latest)
   MLflow log (params, metrics, artifacts)
```

---

## 2. Observation Space (6 chiều)

| Index | Ký hiệu  | Mô tả                                          | Phạm vi  |
|-------|----------|------------------------------------------------|----------|
| 0     | s\_left2 | Cảm biến góc -0.6 rad (tia trái xa)           | [0, 1]   |
| 1     | s\_left1 | Cảm biến góc -0.3 rad (tia trái gần)          | [0, 1]   |
| 2     | s\_front | Cảm biến góc 0.0 rad (tia thẳng)              | [0, 1]   |
| 3     | s\_right1| Cảm biến góc +0.3 rad (tia phải gần)          | [0, 1]   |
| 4     | s\_right2| Cảm biến góc +0.6 rad (tia phải xa)           | [0, 1]   |
| 5     | angle    | Góc lệch đến checkpoint hiện tại / π          | [-1, 1]  |

> **Ghi chú**: 0 = tường ngay sát, 1 = không thấy tường. Sensor range = 120 pixels.

---

## 3. Action Space (2 chiều liên tục)

| Index | Ký hiệu | Mô tả                                  | Phạm vi  |
|-------|---------|----------------------------------------|----------|
| 0     | turn    | Góc lái: âm = trái, dương = phải      | [-1, 1]  |
| 1     | engine  | Ga: dương = tăng tốc, âm = phanh/lùi  | [-1, 1]  |

Công thức vật lý áp dụng (`car.py`):
```python
self.angle += turn * MAX_STEER * (speed / MAX_SPEED + 0.3)
self.speed += engine * ACCELERATION
self.speed -= FRICTION
self.speed = max(0.0, min(self.speed, MAX_SPEED))
```

---

## 4. Hằng số Vật lý

| Tham số       | Giá trị | Mô tả                                      |
|---------------|---------|--------------------------------------------|
| MAX_SPEED     | 1.5     | Tốc độ tối đa (px/frame). Giảm từ 3.0 để dễ học phanh. |
| ACCELERATION  | 0.15    | Tăng tốc mỗi frame khi nhấn ga            |
| FRICTION      | 0.02    | Ma sát tự nhiên làm chậm xe               |
| MAX_STEER     | 0.12    | Góc lái tối đa (rad/frame)                |
| SENSOR_RANGE  | 120     | Tầm nhìn sensor (pixels)                  |
| N_SENSORS     | 5       | Số tia raycast                             |

---

## 5. Kiến trúc Mạng Neural (PPO)

### ActorCritic Network

```
Input(6) → Linear(64) → Tanh → Linear(64) → Tanh ┬→ Linear(2)  [Actor: mean]
                                                   │   + log_std (learnable)
                                                   │   → Normal distribution → tanh
                                                   └→ Linear(1)  [Critic: V(s)]
```

- **Weight init**: Orthogonal (gain=√2 cho backbone, gain=0.01 cho actor output)
- **Std**: Learnable log_std parameter, không phụ thuộc state
- **Action sampling**: `action = tanh(Normal(mean, std).sample())` → bound [-1, 1]

---

## 6. Hàm Reward (Shaped Reward)

### Công thức tổng hợp mỗi frame

| Thành phần | Điều kiện | Giá trị | Mục đích |
|-----------|-----------|---------|----------|
| `alive_bonus` | Mỗi frame sống | +0.05 | Khuyến khích sống lâu |
| `speed_bonus` | Context-aware | `+0.1 × (v/vmax) × (1 - 2|Δθ|/π)` | Thưởng nhanh ở thẳng, phạt nhanh ở cua |
| `checkpoint_bonus` | Qua checkpoint mới | +50.0 | Mục tiêu chính |
| `wall_penalty` | Va chạm tường | -10.0 | Phạt chết |
| `stuck_penalty` | `speed < 0.1` | -0.5/frame | Chống đứng im |
| `turn_penalty` | Đánh lái | `-0.05 × |turn|` | Chống lạng lách |
| `jerk_penalty` | Thay đổi lái | `-0.15 × jerk` | Chống giật cục |
| `proximity_penalty` | Sensor < 20% | `-0.02 × (1-d)` | Chống bám tường |
| `speed_angle_penalty` | `|Δθ|>45° & v>1.0` | -2.0 | Phạt cua nhanh |

### Context-Aware Speed Bonus (thiết kế chủ chốt)

```
speed_reward = weight × (v / v_max) × (1.0 - 2.0 × |angle_diff| / π)
```

| angle_diff | Hệ số | Tác dụng |
|-----------|-------|----------|
| 0° (thẳng) | +1.0 | Thưởng tốc độ đầy đủ |
| 45° (cua nhẹ) | 0.0 | Trung tính |
| 90°+ (cua gắt) | -1.0 | Phạt tốc độ càng nhanh càng nặng |

---

## 7. PPO Hyperparameters

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `lr` | 0.0003 | Learning rate (Adam) |
| `gamma` | 0.99 | Discount factor (tầm nhìn xa) |
| `gae_lambda` | 0.95 | GAE λ (cân bằng bias/variance) |
| `clip_epsilon` | 0.2 | PPO clipping range |
| `epochs` | 10 | Số lần update mỗi batch |
| `batch_size` | 128 | Mini-batch size |
| `entropy_coeff` | 0.05 | Tăng 5× để ép khám phá |
| `value_coeff` | 0.5 | Hệ số value loss |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `update_every` | 4096 | Steps thu thập trước mỗi update |

### PPO Loss Function

```
L = L_policy + 0.5 × L_value + 0.05 × L_entropy

L_policy = -min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
         với r(θ) = π_θ(a|s) / π_θ_old(a|s)

L_value  = MSE(V_θ(s), returns)

L_entropy = -H[π_θ(·|s)]   (entropy bonus)
```

### GAE (Generalized Advantage Estimation)

```
δt = r_t + γ × V(s_{t+1}) × (1-done) - V(s_t)
A_t = δt + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

---

## 8. Checkpoint & Lưu trữ Model

| File | Mô tả | Ghi lên khi |
|------|-------|------------|
| `ppo_model.pt` | Checkpoint mới nhất | Mỗi 10 episodes |
| `best_ppo_model.pt` | Model tốt nhất trong phiên | Khi `episode_reward > session_best` |
| `best_ppo_model_1437_backup.pt` | Bản sao kỷ lục 1437.1 | Được tạo thủ công để bảo tồn |

Nội dung file `.pt`:
```python
{
    "network_state_dict": ...,   # Trọng số mạng neural
    "optimizer_state_dict": ..., # Trạng thái optimizer Adam
    "episode_count": 581,        # Số episode đã train
    "total_steps": 245779,       # Tổng số steps
    "episode_rewards": [...],    # Lịch sử rewards
}
```

> **Lưu ý**: `best_episode_reward` được reset về `-inf` mỗi lần khởi động script.
> Điều này cho phép model tốt nhất của một map mới (dù điểm thấp hơn map cũ) vẫn được lưu vào `best_ppo_model.pt`.

---

## 9. Genetic Algorithm

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `population_size` | 20 | Số xe mỗi thế hệ |
| `n_elites` | 4 | Số elite giữ nguyên |
| `mutation_rate` | 0.08 | Xác suất đột biến mỗi weight |
| `mutation_strength` | 0.30 | Biên độ nhiễu Gaussian |
| `crossover_prob` | 0.50 | Uniform crossover |

Pipeline mỗi generation:
1. **Evaluate**: Chạy toàn bộ xe đến khi dead/timeout
2. **Sort**: Sắp xếp theo fitness
3. **Elitism**: Giữ nguyên top-N genome
4. **Crossover**: Lai gen ngẫu nhiên từ 2 elite
5. **Mutation**: Thêm nhiễu Gaussian vào genome con

---

## 10. MLflow Tracking

Mỗi lần chạy tự động ghi:
- **Params**: lr, gamma, track, epochs, batch_size, ...
- **Metrics**: episode_reward, policy_loss, value_loss, entropy (theo step)
- **Artifacts**: best_ppo_model.pt (mỗi khi đạt kỷ lục mới)

```bash
.\venv\Scripts\python.exe -m mlflow ui   # Dashboard tại http://localhost:5000
```

---

## 11. Curriculum Learning (Lộ trình Huấn luyện)

```
map_straight → oval/city_simple → city_oval → map_u_turn → city
    (GA)          (PPO, ⭐⭐)       (PPO, ⭐⭐⭐)   (PPO, ⭐⭐⭐)   (PPO, ⭐⭐⭐⭐)
```

Triết lý: Dạy từ dễ đến khó, mỗi map là một "kỹ năng" mới cần học.
