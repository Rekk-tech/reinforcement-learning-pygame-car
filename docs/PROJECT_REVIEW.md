# Deep Learning Cars — Project Review

> Tai lieu review toan dien du an xe tu lai hoc bang AI.
> Cap nhat: 2026-06-29

---

## Muc Luc

1. [Tong Quan Du An](#1-tong-quan-du-an)
2. [Kien Truc He Thong](#2-kien-truc-he-thong)
3. [Pipeline Huan Luyen](#3-pipeline-huan-luyen)
4. [Thuat Toan Tien Hoa (GA)](#4-thuat-toan-tien-hoa-ga)
5. [Thuat Toan PPO](#5-thuat-toan-ppo)
6. [Input / Output cua AI](#6-input--output-cua-ai)
7. [He Thong Diem — Thuong / Phat / Fitness](#7-he-thong-diem)
8. [Vat Ly Xe va Sa Hinh](#8-vat-ly-xe-va-sa-hinh)
9. [Checkpoint va Resume](#9-checkpoint-va-resume)
10. [MLflow Tracking](#10-mlflow-tracking)
11. [Cach Chay Du An](#11-cach-chay-du-an)
12. [Cai Dat va Setup](#12-cai-dat-va-setup)

---

## 1. Tong Quan Du An

Du an mo phong xe o to tu lai tren sa hinh 2D. Xe khong biet gi ve duong dua —
no chi co **cam bien** va **bo nao AI**. Thong qua qua trinh huan luyen,
AI tu hoc cach:
- Tranh tuong
- Di dung huong (theo checkpoint)
- Toi uu hoa toc do

### Muc tieu cuoi cung

Xe tu lai hoan toan 1 vong duong dua ma khong can tuong tac cua nguoi.
AI phai tu quyet dinh: bao nhieu ga? re bao nhieu? dua tren cam bien va la ban.

### 3 Che Do Hoat Dong

| Che do | Thuat toan | Input AI | Mang no-ron | So trong so |
|---|---|---|---|---|
| **GA + Sensor** | Genetic Algorithm | 7 tia laser + 1 goc | FC 8→8→4→2 | **118** |
| **PPO + Sensor** | PPO (Deep RL) | 7 tia laser + 1 goc | FC 8→64→64→2 | **~4,933** |
| **PPO + Pixel** | PPO (Deep RL) | Anh 84×84 × 4 frames | CNN 3Conv + FC512 | **~1,685,669** |

---

## 2. Kien Truc He Thong

### 2.1 Cau Truc Thu Muc

```
deep-learning-cars/
├── main.py                        # Entry point: parse args, goi run_ga() hoac run_ppo()
├── mlflow_tracking.py             # MLflow wrapper (tu dong fallback khi chua cai)
├── requirements.txt               # pygame, pyyaml, matplotlib, numpy, torch, mlflow
├── Dockerfile                     # Multi-stage build
├── docker-compose.yml             # App + MLflow server
│
├── configs/
│   └── config.yaml                # TAT CA hyperparams cua du an
│
├── src/
│   ├── core/                      # Loi AI
│   │   ├── neural_network.py      # Feedforward NN [8,8,4,2] — genome cho GA
│   │   ├── genetic_algorithm.py   # GA: selection, crossover, mutation, checkpoint
│   │   ├── ppo_agent.py           # PPO: ActorCritic, GAE, clipped update
│   │   └── cnn_encoder.py         # CNN: 3 Conv → FC512 (cho pixel mode)
│   │
│   ├── simulation/                # Moi truong vat ly
│   │   ├── car.py                 # Agent xe: sensor, physics, checkpoint tracking
│   │   ├── track.py               # Sa hinh: waypoints (x,y,w), collision, checkpoints
│   │   ├── camera.py              # Camera 84x84 + frame stacking (pixel mode)
│   │   └── reward.py              # Shaped reward: 5 thanh phan (PPO)
│   │
│   ├── rendering/
│   │   ├── renderer.py            # Pygame renderer + checkpoint visualization
│   │   └── hud.py                 # HUD overlay (bieu do, stats)
│   │
│   └── ui/
│       └── controls.py            # Giao dien tuong tac (Pause, Speed, Reset)
│
├── checkpoints/                   # Thu muc luu trong so model
│   ├── best.npy                   # GA genome (118 floats)
│   ├── metadata.json              # GA metadata (gen, fitness, architecture)
│   └── ppo_model.pt               # PPO weights (PyTorch state_dict)
│
└── docs/
    └── PROJECT_REVIEW.md          # << BAN DANG DOC FILE NAY
```

### 2.2 So Do Luong Du Lieu

```
main.py
  │
  ├─[algorithm = "ga"]──► run_ga()
  │    │
  │    ├─ GeneticAlgorithm         Quan ly quan the 20 NeuralNetwork
  │    │    └─ Moi the he: evaluate → select → crossover → mutate
  │    │
  │    ├─ Car                      Moi xe co NN rieng, tu chay doc lap
  │    │    └─ step(): sensors → NN.forward() → act → physics → checkpoint
  │    │
  │    ├─ Track                    Sinh checkpoints tu dong tu waypoints
  │    │    └─ is_off_track(): kiem tra va cham voi do rong bien thien
  │    │
  │    └─ Renderer                 Ve track, xe, tia sensor, checkpoint, HUD
  │
  └─[algorithm = "ppo"]──► run_ppo()
       │
       ├─ PPOAgent                 ActorCritic network + Adam optimizer
       │    └─ collect rollout → compute GAE → clipped PPO update
       │
       ├─ RewardCalculator         5 thanh phan: alive + speed + checkpoint + stuck + wall
       │
       └─ CameraSensor             (chi khi pixel mode) Crop 84x84 → grayscale → stack 4 frames
```

---

## 3. Pipeline Huan Luyen

### 3.1 GA Pipeline (Tien Hoa)

```
Khoi tao 20 xe voi brain ngau nhien (118 trong so)
         │
         ▼
┌───► EVALUATE: Chay tat ca 20 xe dong thoi tren sa hinh
│     - Moi frame: sensors → NN → action → physics → checkpoint
│     - Xe chet khi: dam tuong / di lui / quet lau khong qua checkpoint
│     - Ket qua: 20 fitness scores
│         │
│         ▼
│    SELECT: Xep hang theo fitness, giu 4 xe tot nhat (elites)
│         │
│         ▼
│    CROSSOVER: Tao 16 xe con bang lai ghep gene tu 2 elite cha me
│    - Uniform crossover: moi gene co 50% lay tu cha hoac me
│         │
│         ▼
│    MUTATE: Them nhieu Gaussian vao trong so cua xe con
│    - 8% co hoi moi weight bi dot bien
│    - Nhieu ~ N(0, 0.30²), giam 1% moi the he (san 0.05)
│         │
│         ▼
│    The he moi = 4 elites (nguyen ven) + 16 children
│    generation += 1
│         │
└─────────┘ (lap lai)
```

### 3.2 PPO Pipeline (Deep RL)

```
Khoi tao 1 xe duy nhat voi ActorCritic network
         │
         ▼
┌───► EPISODE BAT DAU: Reset xe ve diem xuat phat
│     │
│     ▼
│    ┌──► STEP: Agent quan sat (obs) → Policy sinh action → Xe di chuyen
│    │    - Luu (obs, action, reward, value, log_prob, done) vao Buffer
│    │    - Neu done → ket thuc episode
│    │         │
│    │         ▼
│    │    Buffer du 2048 transitions?
│    │    - Chua → tiep tuc step
│    │    - Roi  → PPO UPDATE:
│    │              a) Tinh GAE (Generalized Advantage Estimation)
│    │              b) 10 epochs: shuffle → mini-batch (64) → clip PPO loss
│    │              c) Xoa buffer
│    │         │
│    └─────────┘
│    Log episode reward, luu model moi 10 episodes
│         │
└─────────┘ (episode moi)
```

---

## 4. Thuat Toan Tien Hoa (GA)

**File:** `src/core/genetic_algorithm.py`

### 4.1 Kien Truc Mang GA

```
Input (8 neurons)
  │  S0..S6 = 7 tia laser (khoang cach den tuong, chuan hoa [0,1])
  │  S7     = Goc lech toi checkpoint tiep theo (chuan hoa [-1,1])
  │
  │  8×8 weights + 8 biases = 72 tham so
  ▼
Hidden 1 (8 neurons, tanh)
  │
  │  8×4 weights + 4 biases = 36 tham so
  ▼
Hidden 2 (4 neurons, tanh)
  │
  │  4×2 weights + 2 biases = 10 tham so
  ▼
Output (2 neurons, tanh)
  ├─ turn   ∈ (-1, 1)   re trai / re phai
  └─ engine ∈ (-1, 1)   phanh / tang ga

TONG: 72 + 36 + 10 = 118 tham so (= 1 genome)
```

### 4.2 Cac Phep Toan Tien Hoa

| Phep toan | Cong thuc | Mo ta |
|---|---|---|
| **Selection** | Top-4 theo fitness | Giu nguyen elite, khong dot bien |
| **Crossover** | `mask = rand() < 0.5` → lay gene tu cha/me | Uniform crossover |
| **Mutation** | `genome += N(0, σ²) × Bernoulli(0.08)` | 8% weight bi dot bien |
| **Decay** | `σ *= 0.99` moi gen, san `σ_min = 0.05` | Exploration → Exploitation |

### 4.3 Hyperparameters GA

| Tham so | Gia tri | Mo ta |
|---|---|---|
| population_size | 20 | So xe moi the he |
| n_elites | 4 | So elite giu nguyen (20%) |
| mutation_rate | 0.08 | Xac suat dot bien moi weight |
| mutation_strength | 0.30 | Do lon nhieu khoi tao (std) |
| mutation_decay | 0.99 | Giam 1% moi the he |
| min_mutation_strength | 0.05 | San toi thieu |
| crossover_prob | 0.50 | Xac suat lay gene tu cha |
| max_frames | 3000 | So frame toi da moi generation |

---

## 5. Thuat Toan PPO

**File:** `src/core/ppo_agent.py`

### 5.1 Kien Truc Mang PPO (Sensor Mode)

```
Input (8)  →  Backbone [FC(8,64)+Tanh, FC(64,64)+Tanh]
                │
                ├─ Actor head:  FC(64,2) + log_std(2)  →  Normal(mean, exp(log_std))
                │                                          │
                │                                          ▼
                │                                     tanh(sample) → [turn, engine]
                │
                └─ Critic head: FC(64,1) → V(s)  (gia tri uoc luong reward tuong lai)

TONG: ~4,933 tham so
```

### 5.2 Kien Truc Mang PPO (Pixel Mode — CNN)

```
Input: (batch, 4, 84, 84)   4 khung hinh xam xep chong

→ Conv2d(4,  32, kernel=8, stride=4) + ReLU   →  [20×20×32]     8,224 tham so
→ Conv2d(32, 64, kernel=4, stride=2) + ReLU   →  [9×9×64]      32,832 tham so
→ Conv2d(64, 64, kernel=3, stride=1) + ReLU   →  [7×7×64]      36,928 tham so
→ Flatten → FC(3136, 512) + ReLU                             1,606,144 tham so
        │
        ├─ Actor head:  FC(512,2) + log_std(2)                1,028 tham so
        └─ Critic head: FC(512,1)                               513 tham so

TONG: ~1,685,669 tham so
```

### 5.3 Hyperparameters PPO

| Tham so | Gia tri | Mo ta |
|---|---|---|
| lr | 3e-4 | Learning rate (Adam) |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda (can bang bias-variance) |
| clip_epsilon | 0.2 | PPO clipping range |
| epochs | 10 | So lan lap PPO update moi batch |
| batch_size | 64 | Mini-batch size |
| entropy_coeff | 0.01 | He so entropy (khuyen khich exploration) |
| value_coeff | 0.5 | He so value loss |
| max_grad_norm | 0.5 | Gradient clipping |
| update_every | 2048 | Steps giua moi lan PPO update |

---

## 6. Input / Output cua AI

### 6.1 Input: 8 Neurons

**7 tia laser (sensor raycasting):**

```
             S3 (0.0 rad — thang truoc)
              │
       S2     │     S4
      /       │       \
     / -0.3   │  +0.3  \
    /         │         \
   S1         │          S5
  (-0.6)      │     (+0.6)
  /           │           \
S0            │            S6
(-1.5 rad)    │       (+1.5 rad)
              │
          [XE O TO]
```

Moi tia bat ra tu tam xe, do khoang cach den tuong gan nhat:
- Gia tri `0.0` = tuong sat mat
- Gia tri `1.0` = khong thay tuong (xa > 120px)

**Input thu 8 — La ban (Compass):**

```python
target_angle = atan2(checkpoint.y - car.y, checkpoint.x - car.x)
angle_diff   = normalize(target_angle - car.angle)    # → [-π, π]
sensor[7]    = angle_diff / π                          # → [-1, 1]
```

| Gia tri sensor[7] | Y nghia |
|---|---|
| ≈ 0 | Xe dang di dung huong checkpoint |
| < 0 | Checkpoint o ben trai |
| > 0 | Checkpoint o ben phai |

### 6.2 Output: 2 Neurons

| Output | Pham vi | Y nghia |
|---|---|---|
| `turn` | (-1, 1) | -1 = re het trai, +1 = re het phai |
| `engine` | (-1, 1) | -1 = phanh, +1 = tang toc toi da |

---

## 7. He Thong Diem — Thuong / Phat / Fitness

### 7.1 Fitness (GA Mode)

```
Fitness = (So_checkpoint_da_qua × 1000) − Khoang_cach_toi_checkpoint_tiep_theo
```

**Vi du:**
- Xe qua duoc 5 checkpoint, cach checkpoint thu 6 la 200px:
  `Fitness = 5 × 1000 − 200 = 4800`
- Xe qua duoc 10 checkpoint, cach checkpoint thu 11 la 50px:
  `Fitness = 10 × 1000 − 50 = 9950`

**Loi ich:** Xe PHAI di qua checkpoint theo thu tu, khong the gian lan bang cach chay vong tron.

### 7.2 Shaped Reward (PPO Mode)

**File:** `src/simulation/reward.py`

| Thanh phan | Gia tri | Khi nao |
|---|---|---|
| Alive bonus | +0.05 / frame | Moi frame xe con song |
| Speed bonus | +0.1 × (speed/3.0) | Thuong di nhanh |
| Checkpoint bonus | +50.0 | Moi lan qua checkpoint moi |
| Stuck penalty | −0.5 / frame | Khi toc do < 0.1 |
| Wall penalty | −10.0 | Khi dam tuong (chet) |

### 7.3 Luat Tu Hinh (Anti-Cheat)

```python
# CHET 1: Di lui — goc lech voi checkpoint > 90 do
if abs(angle_diff) > π/2:
    car.alive = False

# CHET 2: Le me — 150 frames (2.5 giay) khong qua checkpoint moi
if frames_since_last_checkpoint >= 150:
    car.alive = False

# CHET 3: Dam tuong — vi tri xe nam ngoai bien duong
if track.is_off_track(car.x, car.y):
    car.alive = False
```

---

## 8. Vat Ly Xe va Sa Hinh

### 8.1 Cong Thuc Vat Ly

**File:** `src/simulation/car.py`

```python
# GOC LAI (phu thuoc toc do)
angle += turn × MAX_STEER × (speed / MAX_SPEED + 0.3)
#        [-1,1]   0.08 rad      [0.3, 1.3]

# TOC DO
speed += engine × ACCELERATION    # engine ∈ [-1, 1]
speed -= FRICTION                 # ma sat 0.02/frame
speed = clamp(0, MAX_SPEED)       # [0, 3.0]

# VI TRI
x += cos(angle) × speed
y += sin(angle) × speed
```

| Tham so | Gia tri | Don vi | Mo ta |
|---|---|---|---|
| MAX_SPEED | 3.0 | pixel/frame | Toc do toi da |
| ACCELERATION | 0.15 | px/frame² | Tang toc |
| FRICTION | 0.02 | px/frame² | Giam toc tu nhien |
| MAX_STEER | 0.08 | rad/frame | Goc lai toi da |
| SENSOR_RANGE | 120 | pixels | Tam nhin cam bien |

### 8.2 Sa Hinh (Track)

**File:** `src/simulation/track.py`

Moi track la 1 chuoi waypoints khep kin. Waypoint co the co do rong rieng `(x, y, w)`:

```python
"city_simple": [
    (200, 100, 60),    # doan thang, rong 60px
    (600, 100, 60),
    (700, 200, 80),    # goc cua, rong 80px (mo rong de xe de re)
    ...
]
```

**Ham is_off_track():** Dung noi suy do rong `lerp(wA, wB, t)` de tinh bien duong
tai moi vi tri. Bo tron goc cua bang `pygame.draw.circle` tai cac khop noi.

### 8.3 Cac Sa Hinh Co San

| Track | Mo ta | Do kho |
|---|---|---|
| `oval` | Hinh bau duc, rong deu | ⭐ Co ban |
| `figure8` | So 8, co diem cheo | ⭐⭐ Trung binh |
| `city_simple` | 1 block, 4 goc 90° | ⭐⭐ Curriculum 1 |
| `city` | 2 block, nga ba nga tu | ⭐⭐⭐ Curriculum 2 |

### 8.4 He Thong Checkpoint

Track tu dong sinh cac **vung tron tang hinh** doc theo duong dua (cach nhau ~60px).
Xe phai di qua checkpoint theo dung thu tu:
- Checkpoint 1 → 2 → 3 → ... → quay lai 1 (1 vong)
- Bo qua hoac di nguoc → khong duoc tinh diem

```
Renderer hien thi:
  - Vong tron vang nhat: Vi tri checkpoint
  - Tia laser vang: Duong tu xe den checkpoint tiep theo (chi xe dau)
```

---

## 9. Checkpoint va Resume

### 9.1 GA Checkpoint

```
checkpoints/
├── best.npy          # Numpy array 118 floats (genome xe tot nhat)
└── metadata.json     # Thong tin kem theo
```

**metadata.json:**
```json
{
  "generation": 50,
  "best_fitness": 148902.9,
  "all_time_best": 148902.9,
  "architecture": [8, 8, 4, 2],
  "activation": "tanh",
  "population_size": 20,
  "timestamp": "2026-06-29T20:27:06"
}
```

**Bao mat:** Khi load checkpoint, he thong kiem tra `len(genome) == expected_size`.
Neu kien truc khong khop (vi du checkpoint cu 74 tham so voi kien truc moi 118),
he thong se in canh bao va bo qua thay vi crash.

### 9.2 PPO Checkpoint

```
checkpoints/
└── ppo_model.pt      # PyTorch state_dict
```

Noi dung: `network_state_dict`, `optimizer_state_dict`, `episode_count`,
`total_steps`, `episode_rewards`.

### 9.3 Cach Su Dung

```bash
# Luu: tu dong moi 10 gen/episode (config: save_best_every: 10)

# Tai: them flag --resume
.\venv\Scripts\python.exe main.py --resume

# PPO resume
.\venv\Scripts\python.exe main.py --algorithm ppo --resume
```

---

## 10. MLflow Tracking

**File:** `mlflow_tracking.py`

Moi phien huan luyen duoc tu dong log:

**GA mode (experiment: "dl-cars-ga"):**

| Loai | Ten | Mo ta |
|---|---|---|
| Param | population_size | So xe moi the he |
| Param | mutation_rate | Xac suat dot bien |
| Param | architecture | Kien truc mang [8,8,4,2] |
| Param | track | Ten sa hinh |
| Metric | best_fitness | Fitness tot nhat moi generation |
| Metric | mean_fitness | Fitness trung binh |
| Artifact | best.npy | Genome tot nhat |

**PPO mode (experiment: "dl-cars-ppo"):**

| Loai | Ten | Mo ta |
|---|---|---|
| Param | is_cnn, lr, gamma, clip_epsilon | Hyperparams |
| Metric | episode_reward | Reward moi episode |
| Metric | policy_loss, value_loss, entropy | Sau moi PPO update |
| Artifact | ppo_model.pt | Model weights |

**Xem ket qua:**
```bash
.\venv\Scripts\python.exe -m mlflow ui
# Mo trinh duyet: http://127.0.0.1:5000
```

---

## 11. Cach Chay Du An

### 11.1 GA Mode

```bash
# Co giao dien (nhin xe chay, co tia laser + checkpoint)
.\venv\Scripts\python.exe main.py

# Khong giao dien (nhanh 5-10x)
.\venv\Scripts\python.exe main.py --headless --generations 100

# Doi track
.\venv\Scripts\python.exe main.py --track city

# Tang toc mo phong (2x)
.\venv\Scripts\python.exe main.py --speed 2

# Tiep tuc tu checkpoint
.\venv\Scripts\python.exe main.py --resume
```

### 11.2 PPO Mode

```bash
# PPO + Sensor
.\venv\Scripts\python.exe main.py --algorithm ppo --track city_simple

# PPO + Pixel (phai doi config.yaml: perception.mode = pixel)
.\venv\Scripts\python.exe main.py --algorithm ppo --headless --generations 2000

# Resume PPO
.\venv\Scripts\python.exe main.py --algorithm ppo --resume
```

### 11.3 Phim Tat Giao Dien

| Phim | Chuc nang |
|---|---|
| `Space` | Tam dung / Tiep tuc |
| `R` | Reset lai tu dau |
| `+` / `-` | Tang / Giam toc do mo phong |
| `1`-`9`, `0` | Dat toc do 1x-10x |
| `Esc` | Thoat |

---

## 12. Cai Dat va Setup

### 12.1 Yeu Cau He Thong

- Python 3.10+
- Windows / Linux / macOS
- GPU (tuy chon, chi can cho PPO Pixel mode)

### 12.2 Cai Dat

```bash
# 1. Tao moi truong ao
python -m venv venv

# 2. Kich hoat (Windows)
.\venv\Scripts\activate

# 3. Cai thu vien
pip install -r requirements.txt
```

### 12.3 Chuyen Doi Sensor ↔ Pixel

Sua file `configs/config.yaml`:

```yaml
perception:
  mode: sensor    # 7 tia cam bien (nhanh, nhe)
  # mode: pixel   # Camera 84x84 + CNN (nang, can train lau)
```

### 12.4 Docker

```bash
# Chay ca training + MLflow server
docker-compose up

# Chi chay MLflow server
docker-compose up mlflow
# → http://localhost:5000
```

---

*Tai lieu nay duoc tao dua tren source code thuc te cua du an.*
*Cap nhat lan cuoi: 2026-06-29*
