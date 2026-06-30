# Deep Learning Cars — Autonomous Driving Simulation

Mo phong xe tu lai hoc cach dieu khien bang **Tri tue nhan tao (AI)** tren sa hinh 2D.
Du an ho tro ca **Neuroevolution (Genetic Algorithm)** va **Deep Reinforcement Learning (PPO)**,
cho phep so sanh su khac biet giua cac phuong phap — tu mang nho gon 118 tham so den mang CNN 1.7 trieu tham so.

---

## Tinh Nang Chinh

- **3 Che do AI**: GA + Sensor | PPO + Sensor | PPO + Pixel (CNN)
- **He thong Checkpoint**: Dan duong AI bang cac diem kiem soat tu dong, kem la ban dinh huong
- **Tien hoa thong minh**: Mutation decay giam dan — tu kham pha sang toi uu hoa
- **4 sa hinh**: `oval` → `figure8` → `city_simple` → `city` (tang dan do kho)
- **Duong dua do rong bien thien**: Nga tu rong hon, doan thang hep hon
- **Giao dien truc quan**: HUD thoi gian thuc, bieu do fitness, tia laser, tia dan duong
- **MLflow Tracking**: Tu dong log moi phien huan luyen
- **Docker**: Chay nhanh voi Docker Compose

---

## Kien Truc

```
deep-learning-cars/
├── main.py                        # Entry point
├── mlflow_tracking.py             # MLflow wrapper
├── configs/config.yaml            # Tat ca hyperparams
│
├── src/
│   ├── core/                      # AI: NN, GA, PPO, CNN
│   ├── simulation/                # Vat ly: xe, track, reward, camera
│   ├── rendering/                 # Pygame: renderer, HUD
│   └── ui/                        # Controls: pause, speed, reset
│
├── checkpoints/                   # Trong so model (best.npy, ppo_model.pt)
└── docs/PROJECT_REVIEW.md         # Tai lieu ky thuat chi tiet
```

---

## Cai Dat

```bash
# 1. Tao moi truong ao
python -m venv venv

# 2. Kich hoat (Windows)
.\venv\Scripts\activate

# 3. Cai thu vien
pip install -r requirements.txt
```

**Yeu cau:** Python 3.10+, Windows/Linux/macOS

---

## Cach Chay

### GA Mode (Mac dinh — nhanh, don gian)

```bash
# Co giao dien (nhin xe chay voi tia laser + checkpoint)
.\venv\Scripts\python.exe main.py

# Khong giao dien (nhanh 5-10x)
.\venv\Scripts\python.exe main.py --headless --generations 100

# Doi sa hinh
.\venv\Scripts\python.exe main.py --track city

# Tiep tuc tu checkpoint
.\venv\Scripts\python.exe main.py --resume
```

### PPO Mode (Deep RL — manh me hon)

```bash
# PPO + Sensor
.\venv\Scripts\python.exe main.py --algorithm ppo --track city_simple

# PPO + Pixel (doi config.yaml: perception.mode = pixel)
.\venv\Scripts\python.exe main.py --algorithm ppo --headless --generations 2000

# Resume PPO
.\venv\Scripts\python.exe main.py --algorithm ppo --resume
```

### Phim Tat Giao Dien

| Phim | Chuc nang |
|---|---|
| `Space` | Tam dung / Tiep tuc |
| `R` | Reset |
| `+` / `-` | Tang / Giam toc do |
| `1`-`9`, `0` | Toc do 1x-10x |
| `Esc` | Thoat |

---

## 3 Che Do AI

| Che do | Thuat toan | Input | Mang | So tham so |
|---|---|---|---|---|
| **GA + Sensor** | Genetic Algorithm | 7 laser + 1 goc | FC 8→8→4→2 | 118 |
| **PPO + Sensor** | PPO | 7 laser + 1 goc | FC 8→64→64→2 | ~4,933 |
| **PPO + Pixel** | PPO | 84×84×4 frames | CNN 3Conv+FC512 | ~1,685,669 |

### Chuyen doi Sensor ↔ Pixel

Sua `configs/config.yaml`:

```yaml
perception:
  mode: sensor    # 7 tia cam bien (nhanh)
  # mode: pixel   # Camera 84x84 + CNN (nang, can GPU)
```

---

## He Thong Diem

### GA Fitness

```
Fitness = (So checkpoint da qua × 1000) − Khoang cach toi checkpoint tiep theo
```

### PPO Shaped Reward

| Thanh phan | Gia tri | Khi nao |
|---|---|---|
| Alive bonus | +0.05 / frame | Song sot |
| Speed bonus | +0.1 × (speed/3.0) | Di nhanh |
| Checkpoint | +50.0 | Qua checkpoint moi |
| Stuck penalty | −0.5 / frame | Toc do < 0.1 |
| Wall penalty | −10.0 | Dam tuong |

### Luat Tu Hinh

- **Di lui**: Goc lech > 90° → chet ngay
- **Le me**: 150 frames khong qua checkpoint → chet
- **Dam tuong**: Ra ngoai bien duong → chet

---

## MLflow Tracking

Moi phien train tu dong log metrics va hyperparams.

```bash
# Mo MLflow UI
.\venv\Scripts\python.exe -m mlflow ui
# → http://127.0.0.1:5000
```

Hoac dung Docker:
```bash
docker-compose up mlflow
```

---

## Tai Lieu Chi Tiet

Xem [docs/PROJECT_REVIEW.md](docs/PROJECT_REVIEW.md) de biet:
- Pipeline huan luyen chi tiet (GA va PPO)
- Kien truc mang no-ron (so do day du)
- Cong thuc vat ly xe
- Cach thuc checkpoint hoat dong
- Hyperparameters mac dinh
- Cach dung MLflow

---

*Du an nay su dung Pygame, PyTorch, NumPy, MLflow.*
