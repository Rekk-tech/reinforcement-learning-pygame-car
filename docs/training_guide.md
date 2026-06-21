# Training Guide — Deep Learning Cars

## Tong Quan

Du an ho tro 2 thuat toan training:
- **GA (Genetic Algorithm)**: Neuroevolution — don gian, hoi tu nhanh voi mang nho
- **PPO (Proximal Policy Optimization)**: Deep RL — manh me hon, scale duoc voi mang lon

---

## 1. Genetic Algorithm (GA)

### Cach chay

```bash
# Mode co giao dien (render)
python main.py

# Mode khong giao dien (nhanh hon)
python main.py --headless --generations 100

# Tiep tuc tu checkpoint
python main.py --resume
```

### Hyperparams quan trong

| Tham so | Default | Goi y tuning |
|---------|---------|-------------|
| `population_size` | 20 | Tang len 40-60 de tang da dang |
| `n_elites` | 4 | Giu 10-20% population |
| `mutation_rate` | 0.08 | Tang khi bi ket local optima |
| `mutation_strength` | 0.30 | Giam khi da hoi tu co ban |

### Ket qua ky vong

| Generation | Hanh vi dien hinh |
|------------|------------------|
| 1-3 | Ngau nhien, da so dam tuong |
| 4-8 | Mot so xe biet re, di duoc vai giay |
| 10-15 | Elite xe hoan thanh 1 vong co ban |
| 20+ | Xe toi uu duong di, toc do cao on dinh |

---

## 2. PPO (Proximal Policy Optimization)

### Cach chay

```bash
# PPO headless
python main.py --algorithm ppo --headless --generations 500

# PPO voi render
python main.py --algorithm ppo

# Resume PPO
python main.py --algorithm ppo --resume
```

### Hyperparams quan trong

| Tham so | Default | Goi y |
|---------|---------|-------|
| `lr` | 3e-4 | Giam xuong 1e-4 neu training khong on dinh |
| `gamma` | 0.99 | Giu nguyen, day la gia tri chuan |
| `clip_epsilon` | 0.2 | Giu nguyen |
| `epochs` | 10 | Tang len 15-20 neu batch lon |
| `entropy_coeff` | 0.01 | Tang len 0.05 neu can nhieu exploration |

### Ket qua ky vong

| Episode | Hanh vi dien hinh |
|---------|------------------|
| 1-50 | Reward am, xe chua biet di |
| 50-150 | Reward tang dan, xe bat dau di thang |
| 150-300 | Xe biet re, hoan thanh cac khuc cua |
| 300+ | Xe toi uu toc do va duong di |

---

## 3. Config Reference

File config: `configs/config.yaml`

### Chuyen doi thuat toan

```yaml
# Trong config.yaml
algorithm: ga    # hoac: ppo
```

Hoac dung command line:
```bash
python main.py --algorithm ppo
```

### Shaped Reward (PPO)

```yaml
reward:
  speed_bonus_weight: 0.1   # Thuong toc do
  wall_penalty: -10.0       # Phat dam tuong
  checkpoint_bonus: 50.0    # Thuong qua checkpoint
  stuck_penalty: -0.5       # Phat dung im
  alive_bonus: 0.05         # Thuong song sot
```

---

## 4. Checkpointing

### GA checkpoint
- Luu tai: `checkpoints/best.npy` + `checkpoints/metadata.json`
- Tu dong luu moi 10 gen (config: `training.save_best_every`)

### PPO checkpoint
- Luu tai: `checkpoints/ppo_model.pt`
- Chua: network weights, optimizer state, episode count

### Resume training
```bash
python main.py --resume                    # GA
python main.py --algorithm ppo --resume    # PPO
```

---

## 5. Tips & Tricks

1. **GA khong hoi tu?** Tang `mutation_rate` len 0.15 va `population_size` len 40
2. **PPO reward giam?** Giam `lr` xuong 1e-4 hoac tang `entropy_coeff`
3. **Xe bi stuck?** Giam `max_stuck_frames` xuong 80 de ket thuc som hon
4. **Muon train nhanh?** Dung `--headless` mode (nhanh hon 5-10x)
5. **Track kho?** Thu `figure8` hoac `city` thay vi `oval`
