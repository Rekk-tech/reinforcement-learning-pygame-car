# Deep Learning Cars — Project Review

> Tai lieu review toan dien du an xe tu lai hoc bang AI.
> Cap nhat: 2026-06-21

---

## Muc Luc

1. [Tong Quan Du An](#1-tong-quan-du-an)
2. [Kien Truc He Thong](#2-kien-truc-he-thong)
3. [3 Loai Mang No-ron](#3-ba-loai-mang-no-ron)
4. [Thuat Toan Huan Luyen](#4-thuat-toan-huan-luyen)
5. [Luong Input — Sensor vs Pixel](#5-luong-input--sensor-vs-pixel)
6. [Vat Ly Xe](#6-vat-ly-xe)
7. [Checkpointing va Resume](#7-checkpointing-va-resume)
8. [MLflow Tracking](#8-mlflow-tracking)
9. [Cach Chay Du An](#9-cach-chay-du-an)
10. [Cac Van De Can Luu Y](#10-cac-van-de-can-luu-y)
11. [Khuyen Nghi Cai Thien](#11-khuyen-nghi-cai-thien)

---

## 1. Tong Quan Du An

Du an mo phong xe o to tu lai tren sa hinh 2D, su dung AI de hoc cach
dieu khien xe khong bi dam tuong. Du an ho tro **3 che do** hoat dong:

| Che do          | Thuat toan         | Input cua AI         | Mang no-ron          | So trong so    |
|-----------------|--------------------|----------------------|----------------------|----------------|
| GA + Sensor     | Genetic Algorithm  | 5 tia cam bien       | FC 5->6->4->2        | **74**         |
| PPO + Sensor    | PPO (Deep RL)      | 5 tia cam bien       | FC 5->64->64->2      | **~4,741**     |
| PPO + Pixel     | PPO (Deep RL)      | Anh 84x84 x4 frames  | CNN 3 Conv + FC 512  | **~1,685,669** |

---

## 2. Kien Truc He Thong

### 2.1 So Do Luong Du Lieu Tong The

```
main.py (Entry Point)
  |
  |--- [algorithm = "ga"] ---> run_ga()
  |                              |
  |                              |--- GeneticAlgorithm  (src/core/genetic_algorithm.py)
  |                              |     |--- Khoi tao 20 NeuralNetwork ngau nhien
  |                              |     |--- Moi generation: evaluate -> select -> crossover -> mutate
  |                              |     |--- Luu/tai checkpoint (best.npy + metadata.json)
  |                              |
  |                              |--- NeuralNetwork     (src/core/neural_network.py)
  |                              |     |--- Feedforward: 5 -> 6 -> 4 -> 2 (tanh)
  |                              |     |--- Genome = flatten tat ca weights + biases = 74 tham so
  |                              |
  |                              |--- Car               (src/simulation/car.py)
  |                              |     |--- step(): observe -> NN.forward() -> act -> physics -> collision
  |                              |     |--- cast_sensors(): phat 5 tia, tra ve [0,1] x 5
  |                              |
  |                              |--- Track             (src/simulation/track.py)
  |                              |     |--- Waypoints khep kin (oval / figure8 / city)
  |                              |     |--- is_off_track(): kiem tra va cham
  |                              |
  |                              |--- Renderer          (src/rendering/renderer.py)
  |                              |     |--- Ve track, xe, sensor, HUD
  |                              |
  |                              |--- MLflowTracker     (mlflow_tracking.py)
  |                                    |--- Log params, metrics, artifacts
  |
  |
  |--- [algorithm = "ppo"] ---> run_ppo()
                                   |
                                   |--- PPOAgent         (src/core/ppo_agent.py)
                                   |     |--- ActorCritic (sensor mode) hoac CNNActorCritic (pixel mode)
                                   |     |--- RolloutBuffer: luu transitions
                                   |     |--- GAE: tinh advantage
                                   |     |--- PPO Update: clipped surrogate objective
                                   |
                                   |--- RewardCalculator  (src/simulation/reward.py)
                                   |     |--- 5 thanh phan: alive, speed, wall, stuck, checkpoint
                                   |
                                   |--- CameraSensor      (src/simulation/camera.py)   [chi khi pixel mode]
                                   |     |--- Crop 84x84 quanh xe
                                   |     |--- Grayscale + Frame stacking (4 frames)
                                   |
                                   |--- CNNEncoder        (src/core/cnn_encoder.py)    [chi khi pixel mode]
                                   |     |--- 3 lop Conv2d -> FC 512 -> feature vector
                                   |
                                   |--- Car, Track, Renderer, MLflowTracker (giong GA)
```

### 2.2 Cau Truc Thu Muc

```
d:\files\
|
|-- main.py                          # Entry point — switch GA/PPO, tich hop MLflow
|-- mlflow_tracking.py               # MLflow wrapper (tu dong fallback khi chua cai)
|-- requirements.txt                 # pygame, pyyaml, matplotlib, numpy, torch, mlflow
|-- Dockerfile                       # Multi-stage build cho Docker
|-- docker-compose.yml               # App + MLflow server
|-- README.md                        # Gioi thieu du an
|
|-- configs/
|   +-- config.yaml                  # TAT CA hyperparams o day
|
|-- src/
|   |-- core/
|   |   |-- neural_network.py        # Feedforward NN — genome cho GA
|   |   |-- genetic_algorithm.py     # GA: selection, crossover, mutation, checkpoint
|   |   |-- ppo_agent.py             # PPO: ActorCritic, GAE, clipped update, checkpoint
|   |   +-- cnn_encoder.py           # CNN: 3 Conv layers -> 512d features
|   |
|   |-- simulation/
|   |   |-- car.py                   # Agent xe: physics, sensors, gym_step API
|   |   |-- track.py                 # Sa hinh: waypoints, collision detection
|   |   |-- reward.py                # Shaped reward function (5 thanh phan)
|   |   +-- camera.py                # Pixel capture 84x84 + frame stacking
|   |
|   +-- rendering/
|       +-- renderer.py              # Pygame renderer + HUD overlay
|
|-- docs/
|   |-- training_guide.md            # Huong dan training chi tiet
|   +-- PROJECT_REVIEW.md            # << BAN DANG DOC FILE NAY
|
|-- checkpoints/                     # Thu muc luu trong so model
|   |-- best.npy                     # GA genome (74 floats)
|   |-- metadata.json                # GA metadata (gen, fitness, timestamp)
|   +-- ppo_model.pt                 # PPO weights (PyTorch state_dict)
|
|-- .github/workflows/
|   +-- ci.yml                       # GitHub Actions: pytest + smoke test + Docker
|
+-- tests/                           # Unit tests
```

---

## 3. Ba Loai Mang No-ron

### 3.1 NeuralNetwork — Feedforward cho GA

**File:** `src/core/neural_network.py`

```
                    KIEN TRUC MANG GA
    ============================================

    Input Layer (5 neurons)               Sensor Readings
    +---------+                           +-----------------+
    | S0: 0.85|  <--- tia -0.6 rad        | Trai xa         |
    | S1: 0.62|  <--- tia -0.3 rad        | Trai gan        |
    | S2: 0.31|  <--- tia  0.0 rad        | Thang truoc     |
    | S3: 0.78|  <--- tia +0.3 rad        | Phai gan        |
    | S4: 0.92|  <--- tia +0.6 rad        | Phai xa         |
    +---------+
         |
         | 5x6 weights + 6 biases = 36 tham so
         v
    Hidden Layer 1 (6 neurons, tanh)
    +--+--+--+--+--+--+
    |  |  |  |  |  |  |   <--- Phat hien pattern nguy hiem
    +--+--+--+--+--+--+
         |
         | 6x4 weights + 4 biases = 28 tham so
         v
    Hidden Layer 2 (4 neurons, tanh)
    +--+--+--+--+
    |  |  |  |  |             <--- Tong hop quyet dinh
    +--+--+--+--+
         |
         | 4x2 weights + 2 biases = 10 tham so
         v
    Output Layer (2 neurons, tanh)
    +--------+--------+
    | turn   | engine |
    | (-1,1) | (-1,1) |
    +--------+--------+
      |          |
      v          v
    Re trai/   Phanh/
    Re phai    Tang toc

    TONG CONG: 36 + 28 + 10 = 74 tham so
```

**Dac diem:**
- Khong co gradient, khong co optimizer
- Tat ca 74 trong so duoc gom thanh 1 vector "genome"
- Khoi tao ngau nhien: `genome ~ N(0, 0.8^2)`
- Toi uu hoa bang qua trinh tien hoa (GA) thay vi backpropagation
- Forward pass cuc nhanh (chi la phep nhan ma tran nho)

### 3.2 ActorCritic — Fully Connected cho PPO + Sensor

**File:** `src/core/ppo_agent.py` (class ActorCritic)

```
                    KIEN TRUC MANG PPO (SENSOR MODE)
    =====================================================

    Input (5 sensor readings)
         |
         | FC(5, 64) + Tanh
         v
    Backbone Layer 1 (64 neurons)     384 tham so
         |
         | FC(64, 64) + Tanh
         v
    Backbone Layer 2 (64 neurons)     4,160 tham so
         |
         +------------------+------------------+
         |                                     |
         v                                     v
    Actor Head                            Critic Head
    FC(64, 2) -> mean                     FC(64, 1) -> V(s)
    + log_std (learnable, 2 params)
         |                                     |
         v                                     v
    Normal(mean, exp(log_std))            Gia tri V(s)
         |                                (uoc luong reward tuong lai)
         v
    action = tanh(sample)
    [turn, engine] in [-1, 1]

    Actor: 130 + 2 = 132 tham so
    Critic: 65 tham so
    TONG: 384 + 4160 + 132 + 65 = ~4,741 tham so
```

**Dac diem:**
- Shared backbone — actor va critic dung chung 2 lop an
- Continuous action space — output la phan phoi Normal, khong phai softmax
- `log_std` la tham so hoc duoc (learnable), khong phai co dinh
- Khoi tao weights bang Orthogonal initialization (on dinh hon random)

### 3.3 CNNActorCritic — Convolutional cho PPO + Pixel

**File:** `src/core/cnn_encoder.py`

```
                    KIEN TRUC MANG PPO (PIXEL MODE)
    =====================================================

    Input: (batch, 4, 84, 84)
    4 khung hinh xam lien tiep, moi pixel trong [0.0, 1.0]

    +-------------------------------------------+
    |  Conv2d(4, 32, kernel=8, stride=4) + ReLU |   8,224 tham so
    |  Output: (batch, 32, 20, 20)              |
    +-------------------------------------------+
                      |
                      v
    +-------------------------------------------+
    |  Conv2d(32, 64, kernel=4, stride=2) + ReLU|   32,832 tham so
    |  Output: (batch, 64, 9, 9)                |
    +-------------------------------------------+
                      |
                      v
    +-------------------------------------------+
    |  Conv2d(64, 64, kernel=3, stride=1) + ReLU|   36,928 tham so
    |  Output: (batch, 64, 7, 7) = 3136 values  |
    +-------------------------------------------+
                      |
                      v
    +-------------------------------------------+
    |  Flatten -> FC(3136, 512) + ReLU          |   1,606,144 tham so
    +-------------------------------------------+
                      |
         +------------+------------+
         |                         |
         v                         v
    Actor Head                Critic Head
    FC(512, 2) + log_std(2)   FC(512, 1)
    1,028 tham so             513 tham so
         |                         |
         v                         v
    [turn, engine]            V(s)

    TONG: 8,224 + 32,832 + 36,928 + 1,606,144 + 1,028 + 513
        = ~1,685,669 tham so
```

**Dac diem:**
- Kien truc lay cam hung tu DQN cua DeepMind (Mnih et al., 2015)
- Kaiming Normal initialization cho Conv layers
- Frame stacking (4 frames) de AI nhan biet chuyen dong va toc do
- FC layer 3136 -> 512 chiem ~95% tong so tham so (bottleneck)

### 3.4 So Sanh 3 Mang

```
    So luong tham so (logarithmic scale):

    GA (74)          |##
    PPO Sensor (4.7k)|#########
    PPO CNN (1.69M)  |###############################################

    Toc do forward pass (tuong doi):

    GA (74)          |## (< 0.01ms)
    PPO Sensor (4.7k)|#### (~ 0.1ms)
    PPO CNN (1.69M)  |################################ (~ 5-10ms, CPU)
```

---

## 4. Thuat Toan Huan Luyen

### 4.1 Genetic Algorithm (Neuroevolution)

**File:** `src/core/genetic_algorithm.py`

```
    VONG LAP TIEN HOA (GA)
    ======================

    The he 0: Khoi tao 20 genome ngau nhien
         |
         v
    +---> BUOC 1: EVALUATE
    |     - Tao 20 xe, moi xe gan 1 NeuralNetwork
    |     - Chay mo phong (toi da 3000 frames/generation)
    |     - Moi frame: sensor -> NN -> action -> physics -> fitness += speed
    |     - Xe chet khi: dam tuong HOAC dung im > 120 frames
    |     - Ket qua: 20 fitness scores
    |         |
    |         v
    |    BUOC 2: SELECT ELITES
    |     - Xep hang 20 xe theo fitness (cao -> thap)
    |     - Chon 4 xe tot nhat (elites)
    |     - Clone genome cua chung (tranh reference sharing)
    |         |
    |         v
    |    BUOC 3: CROSSOVER
    |     - Voi moi vi tri con lai (16 slots):
    |       - Chon 2 parents ngau nhien tu 4 elites
    |       - Moi gene (weight): lay tu parent A voi p=0.5, parent B voi p=0.5
    |       -> Tao 1 child genome
    |         |
    |         v
    |    BUOC 4: MUTATION
    |     - Voi moi child genome:
    |       - Moi weight co 8% co hoi bi dot bien
    |       - Neu bi dot bien: weight += N(0, 0.3^2)
    |         |
    |         v
    |    BUOC 5: REPRODUCE
    |     - The he moi = 4 elites (nguyen ven) + 16 children (da mutate)
    |     - generation += 1
    |         |
    +-------- (lap lai tu BUOC 1)
```

**Tham so:**

| Tham so           | Gia tri | Mo ta                                      |
|-------------------|---------|--------------------------------------------|
| population_size   | 20      | So xe moi the he                           |
| n_elites          | 4       | So elite giu nguyen (20%)                  |
| mutation_rate     | 0.08    | Xac suat dot bien moi weight (8%)          |
| mutation_strength | 0.30    | Do lon nhieu Gaussian (std = 0.3)          |
| crossover_prob    | 0.50    | Ty le chon gene tu parent A vs B           |
| max_frames        | 3000    | So frame toi da moi generation             |

**Fitness Function (GA):**
```
Moi frame:
    fitness += speed      (speed in [0, 3.0])

Xe chet khi:
    - track.is_off_track(x, y) == True     -> dam tuong
    - stuck_counter >= 120                  -> dung im qua lau

=> Fitness = tong quang duong xe di duoc truoc khi chet
=> Fitness tot = xe di nhanh + di xa + khong dam tuong
```

### 4.2 PPO (Proximal Policy Optimization)

**File:** `src/core/ppo_agent.py`

```
    VONG LAP HUAN LUYEN PPO
    =======================

    Khoi tao: ActorCritic network + Adam optimizer
         |
         v
    +---> BUOC 1: RESET EPISODE
    |     - Dat xe ve vi tri xuat phat
    |     - Reset reward calculator
    |     - Lay observation dau tien (sensor hoac pixel)
    |         |
    |         v
    |    BUOC 2: COLLECT ROLLOUT (toi da 3000 steps/episode)
    |     +---> a) Agent quan sat (obs)
    |     |     b) Policy network -> sample action (turn, engine)
    |     |     c) Xe thuc hien action -> new_obs, reward, done
    |     |     d) Tinh shaped reward (5 thanh phan)
    |     |     e) Luu (obs, action, reward, value, log_prob, done) vao buffer
    |     |     f) obs = new_obs
    |     |     g) Neu done -> ket thuc episode
    |     |         |
    |     |         v
    |     |    Buffer du 2048 transitions?
    |     |     |          |
    |     |    Chua       Roi --> BUOC 3: PPO UPDATE
    |     +----+                    |
    |                               v
    |    BUOC 3: PPO UPDATE (10 epochs)
    |     a) Tinh GAE (Generalized Advantage Estimation):
    |        - delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    |        - A_t = delta_t + (gamma * lambda) * A_{t+1}
    |        - Returns = Advantages + Values
    |        - Chuan hoa Advantages: (A - mean) / (std + 1e-8)
    |
    |     b) Voi moi epoch (10 lan):
    |        - Shuffle va chia thanh mini-batches (size 64)
    |        - Voi moi batch:
    |          * Tinh log_prob_new, value_new, entropy tu policy hien tai
    |          * ratio = exp(log_prob_new - log_prob_old)
    |          * L_clip = min(ratio*A, clip(ratio, 0.8, 1.2)*A)
    |          * L_value = MSE(value_new, returns)
    |          * L_entropy = mean(entropy)
    |          * Loss = -L_clip + 0.5*L_value - 0.01*L_entropy
    |          * Backprop + clip grad norm (0.5) + Adam step
    |
    |     c) Xoa buffer
    |         |
    |         v
    |    BUOC 4: LOG & SAVE
    |     - In: episode, reward, avg100, best, steps
    |     - Luu model moi 10 episodes
    |         |
    +-------- (lap lai tu BUOC 1)
```

**Tham so PPO:**

| Tham so       | Gia tri | Mo ta                                          |
|---------------|---------|------------------------------------------------|
| lr            | 3e-4    | Learning rate (Adam optimizer)                 |
| gamma         | 0.99    | Discount factor (tam nhin xa)                  |
| gae_lambda    | 0.95    | GAE lambda (can bang bias-variance)            |
| clip_epsilon  | 0.2     | PPO clipping range (gioi han thay doi policy)  |
| epochs        | 10      | So lan lap lai PPO update moi batch            |
| batch_size    | 64      | Kich thuoc mini-batch                          |
| entropy_coeff | 0.01    | He so entropy bonus (khuyen khich exploration) |
| value_coeff   | 0.5     | He so value loss                               |
| max_grad_norm | 0.5     | Gradient clipping                              |
| update_every  | 2048    | So steps giua moi lan PPO update               |

**Shaped Reward Function:**

```
Moi frame khi xe con song:
    reward  = +0.05                          (alive bonus)
    reward += +0.1 * (speed / 3.0)           (speed bonus, toi da +0.1)
    reward += -0.5 neu speed < 0.1           (stuck penalty)
    reward += +50.0 neu qua checkpoint moi   (checkpoint bonus)

Khi xe chet (dam tuong):
    reward = -10.0                           (wall penalty)
```

### 4.3 So Sanh GA vs PPO

| Tieu chi              | GA (Neuroevolution)           | PPO (Deep RL)                        |
|-----------------------|-------------------------------|--------------------------------------|
| Cach toi uu           | Tien hoa (khong gradient)     | Gradient descent (backprop)          |
| So luong agent        | 20 xe dong thoi               | 1 xe duy nhat                        |
| Fitness / Reward      | Tong quang duong              | Shaped reward 5 thanh phan           |
| Toc do hoi tu         | Nhanh (10-20 gen)             | Cham hon (200-500 episodes)          |
| Kha nang mo rong      | Han che (mang nho)            | Tot (mang lon, CNN)                  |
| Do phuc tap code      | Don gian                      | Phuc tap (GAE, clipping, buffer)     |
| Uu diem               | De hieu, hoi tu nhanh         | Policy manh me, scale duoc           |
| Nhuoc diem            | Khong scale voi mang lon      | Can nhieu tuning, cham tren CPU      |

---

## 5. Luong Input — Sensor vs Pixel

### 5.1 Sensor Mode (Mac dinh)

```
    LUONG DU LIEU SENSOR
    ====================

    Track (sa hinh 2D)
         |
         |  Car.cast_sensors(track)
         |  Phat 5 tia raycast tu vi tri xe
         |
         |  Moi tia:
         |    - Xuat phat tu (car.x, car.y)
         |    - Huong: car.angle + delta_angle
         |    - Di moi 3 pixel mot
         |    - Dung khi gap tuong (track.is_off_track == True)
         |    - Gia tri = khoang_cach / sensor_range
         v
    [0.85, 0.62, 0.31, 0.78, 0.92]     5 so float, moi so trong [0, 1]
         |                               0.0 = tuong sat mat
         |                               1.0 = khong thay tuong (xa > 120px)
         v
    NeuralNetwork.forward(inputs)        GA mode
    hoac
    PPOAgent.select_action(obs)          PPO mode
         |
         v
    [turn, engine]                       2 so float, moi so trong [-1, 1]
         |
         v
    Car.act(turn, engine)                Ap dung vao vat ly xe
```

**5 tia cam bien:**
```
              S2 (0.0 rad, thang truoc)
               |
        S1     |     S3
       /       |       \
      / -0.3   |  +0.3  \
     /         |         \
    S0         |          S4
   (-0.6 rad)  |    (+0.6 rad)
               |
           [XE O TO]
```

### 5.2 Pixel Mode (CNN Vision)

```
    LUONG DU LIEU PIXEL
    ===================

    BUOC 1: Ve hinh
    Renderer.draw_frame([car], track, {})
    -> Pygame Surface 1280x720 RGB
    (Ve track, xe, sensor rays len surface)

    BUOC 2: Chup anh
    CameraSensor.capture(surface, car)
         |
         |  a) Crop vung 84x84 quanh vi tri xe
         |     left = car.x - 42, top = car.y - 42
         |     (clamp vao trong surface neu sat ria)
         |
         |  b) pygame.surfarray.array3d() -> numpy (84, 84, 3)
         |
         |  c) Chuyen sang grayscale (1 channel):
         |     gray = 0.2989*R + 0.5870*G + 0.1140*B
         |
         |  d) Chuan hoa: pixel / 255.0 -> [0.0, 1.0]
         v
    1 frame: shape (1, 84, 84)

    BUOC 3: Frame stacking
    CameraSensor.observe()
         |
         |  Gom 4 frame gan nhat tu deque buffer
         |  np.concatenate([frame_t-3, frame_t-2, frame_t-1, frame_t])
         v
    Stacked observation: shape (4, 84, 84)
    (4 anh xam lien tiep -> AI biet xe dang di nhanh hay cham)

    BUOC 4: CNN xu ly
    CNNActorCritic.get_action(obs_tensor)
         |
         |  Conv2d x3 -> Flatten -> FC -> 512d feature vector
         |  -> Actor head: action [turn, engine]
         |  -> Critic head: V(s)
         v
    [turn, engine] va V(s)
```

**Tai sao can Frame Stacking?**
- 1 anh tinh khong cho biet xe dang di nhanh hay cham
- 4 anh lien tiep cho phep CNN suy ra: van toc, gia toc, huong di chuyen
- Tuong tu nhu mat nguoi can nhieu khung hinh de cam nhan chuyen dong

---

## 6. Vat Ly Xe

**File:** `src/simulation/car.py`

### 6.1 Tham So Vat Ly

| Tham so        | Gia tri          | Don vi       | Mo ta                           |
|----------------|------------------|--------------|---------------------------------|
| MAX_SPEED      | 3.0              | pixel/frame  | Toc do toi da                   |
| ACCELERATION   | 0.15             | px/frame^2   | Tang toc moi frame              |
| FRICTION       | 0.02             | px/frame^2   | Giam toc tu nhien (ma sat)      |
| MAX_STEER      | 0.08             | rad/frame    | Goc lai toi da moi frame        |
| SENSOR_RANGE   | 120              | pixels       | Tam nhin cam bien               |
| MAX_STUCK_FRAMES| 120             | frames       | Gioi han dung im truoc khi chet |

### 6.2 Cong Thuc Vat Ly

```python
# === GOC LAI ===
# He so (speed/MAX_SPEED + 0.3) lam cho:
#   - Xe dung im (speed=0): van xoay duoc nhe (he so 0.3)
#   - Xe chay nhanh (speed=3): re manh hon (he so 1.3)
#   -> Mo phong Ackermann steering don gian

angle += turn * MAX_STEER * (speed / MAX_SPEED + 0.3)
#        |         |              |
#    [-1, 1]    0.08 rad      [0.3, 1.3]
#    tu NN      toi da        phu thuoc toc do


# === TOC DO ===
speed += engine * ACCELERATION     # engine in [-1, 1], ACCEL = 0.15
speed -= FRICTION                  # ma sat giam 0.02 moi frame
speed = clamp(0.0, MAX_SPEED)      # khong am, khong vuot 3.0


# === VI TRI ===
new_x = x + cos(angle) * speed
new_y = y + sin(angle) * speed

# Kiem tra va cham
if track.is_off_track(new_x, new_y):
    alive = False    # XE CHET
else:
    x, y = new_x, new_y
    fitness += speed   # Cong quang duong vao fitness
```

### 6.3 Co Che Chong Stuck

```python
# Moi 30 frame, kiem tra fitness co tang khong
if frame_count % 30 == 0:
    if fitness - last_fitness < 1.0:   # Khong tien duoc
        stuck_counter += 30
    else:
        stuck_counter = 0              # Reset neu co tien
    last_fitness = fitness

# Neu stuck qua 120 frames (khoang 2 giay) -> xe tu chet
if stuck_counter >= 120:
    alive = False
```

### 6.4 Va Cham (Collision Detection)

**File:** `src/simulation/track.py`

```
Track duoc dinh nghia boi chuoi waypoints khep kin.
"Tren duong" = khoang cach tu diem den segment gan nhat <= track_width/2

Vi du track "oval" co 13 waypoints:
(150,90) -> (320,70) -> (490,80) -> ... -> (100,130) -> (150,90)

Kiem tra is_off_track(x, y):
  1. Voi moi cap waypoints lien tiep (A, B):
     - Tinh khoang cach vuong goc tu (x,y) den doan AB
     - Dung cong thuc projection: t = dot(P-A, B-A) / |B-A|^2
     - Clamp t vao [0, 1]
     - closest = A + t * (B-A)
     - dist = |P - closest|
  2. Lay min(dist) trong tat ca cac segments
  3. Neu min_dist > track_width/2 (28px) -> off track -> xe chet
```

---

## 7. Checkpointing va Resume

### 7.1 GA Checkpoint

```
checkpoints/
|-- best.npy           # Numpy array 74 floats (genome cua xe tot nhat)
+-- metadata.json      # Thong tin kem theo
```

**metadata.json:**
```json
{
  "generation": 50,
  "best_fitness": 8966.85,
  "all_time_best": 8966.85,
  "architecture": [5, 6, 4, 2],
  "activation": "tanh",
  "population_size": 20,
  "timestamp": "2026-06-21T10:15:30"
}
```

**Cach luu:** Tu dong luu moi 10 generation (config: `save_best_every: 10`)
**Cach tai:** `python main.py --resume` -> inject genome vao population[0]

### 7.2 PPO Checkpoint

```
checkpoints/
+-- ppo_model.pt       # PyTorch state_dict
```

**Noi dung file .pt:**
```python
{
    "network_state_dict": ...,     # Tat ca weights cua ActorCritic
    "optimizer_state_dict": ...,   # Trang thai Adam optimizer
    "episode_count": 150,          # So episode da train
    "total_steps": 450000,         # Tong so steps
    "episode_rewards": [...],      # Lich su reward moi episode
}
```

**Cach luu:** Tu dong moi 10 episodes
**Cach tai:** `python main.py --algorithm ppo --resume`

---

## 8. MLflow Tracking

### 8.1 Tong Quan

MLflow la cong cu tracking experiments, giup ban:
- So sanh cac lan chay (runs) voi nhau
- Xem bieu do reward/fitness theo thoi gian
- Luu lai hyperparams va model artifacts
- Quay lai bat ky phien train nao truoc do

### 8.2 Nhung Gi Duoc Track

**GA mode (experiment: "dl-cars-ga"):**

| Loai    | Ten                | Mo ta                           |
|---------|--------------------|---------------------------------|
| Param   | population_size    | So xe moi the he                |
| Param   | mutation_rate      | Xac suat dot bien               |
| Param   | architecture       | Kien truc mang [5,6,4,2]        |
| Param   | track              | Ten sa hinh (oval, figure8...)   |
| Metric  | best_fitness       | Fitness tot nhat moi generation  |
| Metric  | mean_fitness       | Fitness trung binh moi generation|
| Artifact| best.npy           | Genome tot nhat                  |

**PPO mode (experiment: "dl-cars-ppo"):**

| Loai    | Ten               | Mo ta                            |
|---------|-------------------|----------------------------------|
| Param   | is_cnn            | True neu dung pixel mode         |
| Param   | lr                | Learning rate                    |
| Param   | gamma             | Discount factor                  |
| Param   | clip_epsilon      | PPO clipping range               |
| Param   | epochs            | So epoch update                  |
| Param   | batch_size        | Mini-batch size                  |
| Metric  | episode_reward    | Reward moi episode               |
| Metric  | episode_steps     | So steps moi episode             |
| Metric  | policy_loss       | Policy loss sau moi update       |
| Metric  | value_loss        | Value loss sau moi update        |
| Metric  | entropy           | Policy entropy                   |
| Artifact| ppo_model.pt      | Model weights                    |

### 8.3 Cach Su Dung MLflow

**Buoc 1: Cai dat (da co trong requirements.txt)**
```bash
.\venv\Scripts\python.exe -m pip install mlflow
```

**Buoc 2: Chay training (MLflow tu dong ghi lai)**
```bash
# GA mode
.\venv\Scripts\python.exe main.py --headless --generations 50

# PPO mode
.\venv\Scripts\python.exe main.py --algorithm ppo --headless --generations 100
```

**Buoc 3: Mo MLflow UI de xem ket qua**
```bash
.\venv\Scripts\python.exe -m mlflow ui
```
Sau do mo trinh duyet tai: http://127.0.0.1:5000

**Buoc 4: Xem tren giao dien MLflow**
```
+--------------------------------------------------+
|  MLflow UI (http://127.0.0.1:5000)               |
|                                                   |
|  Experiments:                                     |
|    [x] dl-cars-ga                                 |
|    [x] dl-cars-ppo                                |
|                                                   |
|  Runs:                                            |
|  +------+--------+----------+------------------+  |
|  | Run  | Status | Duration | best_fitness     |  |
|  +------+--------+----------+------------------+  |
|  | run1 | DONE   | 2m 30s   | 8966.85         |  |
|  | run2 | DONE   | 5m 12s   | 9234.21         |  |
|  +------+--------+----------+------------------+  |
|                                                   |
|  Charts:                                          |
|  best_fitness  ^                                  |
|           9000 |        ____----                   |
|           6000 |   ___/                            |
|           3000 |  /                                |
|              0 +--------+---------> generation     |
|                0       25       50                 |
+--------------------------------------------------+
```

### 8.4 Docker Compose (MLflow Server)

```bash
# Chay ca training + MLflow server
docker-compose up

# Chi chay MLflow server
docker-compose up mlflow
# -> Truy cap tai http://localhost:5000
```

---

## 9. Cach Chay Du An

### 9.1 Cai Dat

```bash
# 1. Tao moi truong ao
python -m venv venv

# 2. Kich hoat
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # Linux/Mac

# 3. Cai dependencies
pip install -r requirements.txt
```

### 9.2 Chay GA (Mac Dinh)

```bash
# Co giao dien (nhin xe chay)
python main.py

# Khong giao dien (nhanh 5-10x)
python main.py --headless --generations 100

# Doi track
python main.py --track figure8

# Tang toc mo phong (2x speed)
python main.py --speed 2

# Tiep tuc tu checkpoint
python main.py --resume
```

### 9.3 Chay PPO

```bash
# PPO voi sensor (nhanh, nhe)
python main.py --algorithm ppo --headless --generations 500

# PPO voi pixel/CNN (can doi config.yaml: perception.mode = pixel)
python main.py --algorithm ppo --headless --generations 2000

# PPO co giao dien
python main.py --algorithm ppo

# Resume PPO
python main.py --algorithm ppo --resume
```

### 9.4 Chuyen Doi Sensor <-> Pixel

Sua file `configs/config.yaml`, dong 42:

```yaml
perception:
  mode: sensor    # Dung 5 tia cam bien (nhanh, nhe)
  # mode: pixel   # Dung camera anh 84x84 + CNN (nang, cham)
```

### 9.5 Xem MLflow

```bash
# Mo MLflow UI
python -m mlflow ui

# Mo trinh duyet: http://127.0.0.1:5000
```

---

## 10. Cac Van De Can Luu Y

### 10.1 Config Hien Tai Dang De Pixel Mode

File `configs/config.yaml` dong 42 hien tai la:
```yaml
perception:
  mode: pixel    # <--- DAY!
```

Dieu nay co nghia la moi khi chay `python main.py --algorithm ppo`,
he thong se:
- Khoi tao mang CNN 1.7 trieu tham so (thay vi 4,700)
- Tao Pygame surface de chup anh (ke ca khi headless)
- Toc do train cham hon 10-50x so voi sensor mode

**De chay nhanh, doi lai thanh `mode: sensor`.**

### 10.2 PPO Reward Am Lien Tuc

Khi moi bat dau train PPO, reward se am lien tuc (khoang -100 den -200).
Day la binh thuong vi:
- Xe chua biet lai, dam tuong lien tuc -> nhan wall_penalty = -10.0
- Xe dung im -> nhan stuck_penalty = -0.5/frame
- Can it nhat 100-200 episodes de bat dau thay cai thien

**Giai phap:**
- Tang `entropy_coeff` len 0.05 (nhieu exploration hon)
- Giam `lr` xuong 1e-4 (on dinh hon)
- Train it nhat 500 episodes (sensor) hoac 2000 episodes (pixel)

### 10.3 GA Mode Khong Dung CNN

GA luon dung NeuralNetwork 74 tham so + 5 tia cam bien,
BAT KE config `perception.mode` la gi. Day la thiet ke co chu dich vi:
- GA toi uu bang tien hoa, khong can gradient
- Mang 74 tham so du nho de tien hoa nhanh
- CNN 1.7 trieu tham so qua lon cho GA (khong gian tim kiem qua rong)

### 10.4 Windows Console va Unicode

Windows console (cp1252) khong ho tro emoji Unicode.
Tat ca print trong code da dung ky tu ASCII thay the:
- `[SAVE]` thay vi emoji save
- `[OK]` thay vi emoji check
- `[--]` thay vi emoji warning

---

## 11. Khuyen Nghi Cai Thien

| #  | Van de                                    | Giai phap de xuat                                       | Do kho |
|----|-------------------------------------------|---------------------------------------------------------|--------|
| 1  | PPO sensor reward am (-100 -> -200)       | Tang entropy_coeff=0.05, giam lr=1e-4, train 500+ ep    | De     |
| 2  | CNN mode cham tren CPU                    | Dung GPU (CUDA), hoac giam capture_size xuong 42        | TB     |
| 3  | is_off_track() la performance bottleneck  | Dung bitmap mask (pre-render) thay vi ray-segment check  | TB     |
| 4  | Camera crop lech o ria man hinh           | Them zero-padding hoac cuon camera theo xe               | De     |
| 5  | Chua co evaluation mode                   | Them flag --eval de chay model ma khong update weights   | De     |
| 6  | Chua co visualize training progress       | Them matplotlib plot fitness/reward curve sau khi train  | De     |
| 7  | GA chi select tu elite pool               | Them tournament selection de tang da dang                | TB     |
| 8  | PPO chua dung checkpoint_bonus            | Can tich hop checkpoint detection vao track.py           | Kho    |
| 9  | Chua co multi-agent PPO                   | Chay nhieu xe PPO song song de tang sample efficiency    | Kho    |
| 10 | Chua co curriculum learning               | Bat dau voi track de -> kho dan de tang toc hoi tu       | Kho    |

---

*Tai lieu nay duoc tao tu dong boi AI assistant.*
*Moi noi dung da duoc review va kiem tra dua tren source code thuc te.*
