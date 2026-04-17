# NT549_SDNControllerLoadBalancing
# SDN Controller Load Balancing với Reinforcement Learning

Hệ thống cân bằng tải thông minh cho cụm SDN Controller (Cluster Controllers) sử dụng Deep Reinforcement Learning (DQN), triển khai trên nền tảng Mininet + Ryu.

---

## Tổng Quan Đồ Án

Bài toán cốt lõi: Khi mạng SDN mở rộng, một controller đơn lẻ dễ bị **quá tải (overload)** do xử lý quá nhiều packet-in từ switch, dẫn đến tăng độ trễ và giảm thông lượng. Đồ án này áp dụng **Reinforcement Learning** để agent tự học cách **di chuyển switch (switch migration)** giữa các controller sao cho tải được phân phối đều — thay vì dùng thuật toán tĩnh như Round-Robin hay Least-Load.

### Mô hình hóa MDP (Markov Decision Process)

| Thành phần | Mô tả |
|---|---|
| **State** | Vector `[CPU_c1, RAM_c1, packet_in_c1, ..., CPU_cN, RAM_cN, packet_in_cN]` — tải thực tế của từng controller |
| **Action** | Discrete: chọn cặp `(switch_id, target_controller_id)` để thực hiện migration |
| **Reward** | `R = -α·Var(CPU) - β·avg_latency + γ·bonus_nếu_cân_bằng` |

### Luồng hoạt động

```
Mininet (topology + traffic)
        ↓
Cluster Ryu Controllers (port 6633, 6634, 6635)
        ↓
Module Giám Sát (psutil + Ryu REST API)
        │
        ├──→  State vector  →  RL Agent (DQN / Multi-Agent)
        │                              ↓
        │                      Action: migrate switch X → controller Y
        │                              ↓
        │                      Module Hành Động (OpenFlow Role Request)
        │                              ↓
        │                      Đo reward → cập nhật policy → lặp lại
        │
        └──→  Prometheus Exporter  →  Prometheus  →  Grafana Dashboard
                   (metrics CPU/RAM/latency/reward theo thời gian thực)
```

---

## Cấu Trúc Dự Án

```
sdn-rl-loadbalancer/
├── controllers/
│   ├── ryu_app_c1.py            # Ryu app cho controller 1 (port 6633)
│   ├── ryu_app_c2.py            # Ryu app cho controller 2 (port 6634)
│   ├── ryu_app_c3.py            # Ryu app cho controller 3 (port 6635)
│   └── monitor_api.py           # REST API thu thập metrics (packet-in, flow stats)
├── mininet/
│   ├── custom_topo.py           # Topology tree/fat-tree với nhiều switch
│   └── traffic_generator.py     # Sinh traffic: iperf, scapy (burst, Poisson, DDoS-like)
├── monitoring/
│   ├── system_monitor.py        # Thu thập CPU/RAM từng Ryu process qua psutil → state vector cho RL
│   ├── prometheus_exporter.py   # Expose metrics lên Prometheus endpoint (/metrics)
│   ├── prometheus.yml           # Cấu hình Prometheus scrape jobs
│   └── grafana/
│       └── dashboard.json       # Dashboard Grafana import sẵn
├── rl_agent/
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── sdn_env.py           # Single-agent Gymnasium environment
│   │   └── sdn_multiagent_env.py # Multi-agent PettingZoo environment
│   ├── train_dqn.py             # Huấn luyện DQN (Stable-Baselines3)
│   ├── train_multiagent.py      # Huấn luyện Multi-Agent (Independent DQN / MADDPG)
│   └── evaluate.py              # Đánh giá & so sánh với baselines
├── baselines/
│   ├── round_robin.py           # Baseline: Round-Robin
│   └── least_load.py            # Baseline: Least-Load
├── utils/
│   ├── api_client.py            # Client gọi Ryu REST API
│   ├── migration_executor.py    # Thực thi switch migration qua OpenFlow Role Request
│   └── visualizer.py            # Vẽ biểu đồ kết quả (matplotlib)
├── scenarios/
│   ├── scenario1_burst.py       # Kịch bản 1: Burst traffic
│   ├── scenario2_dynamic_topo.py # Kịch bản 2: Topology động
│   ├── scenario3_controller_fault.py # Kịch bản 3: Lỗi controller
│   └── scenario4_random_traffic.py   # Kịch bản 4: Traffic ngẫu nhiên (Poisson)
├── models/                      # Model DQN đã huấn luyện (.zip)
├── logs/                        # TensorBoard logs
├── data/                        # Kết quả đo đạc, biểu đồ (.png, .csv)
├── requirements.txt
├── README.md
└── main.py                      # Script khởi động toàn bộ hệ thống
```

---

## Yêu Cầu Hệ Thống

- **OS**: Ubuntu 20.04+ (khuyến nghị dùng VM)
- **Python**: 3.8+
- **Quyền**: root/sudo (bắt buộc cho Mininet)
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB nếu chạy Docker)

---

## Cài Đặt

### 1. Cài đặt Python

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y

echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Cài đặt Mininet và Open vSwitch

```bash
sudo apt-get update
git clone https://github.com/mininet/mininet
cd mininet
sudo ./util/install.sh -a
sudo mn -c   # Dọn dẹp môi trường cũ nếu có
```

### 3. Cài đặt python cho `/mininet`

```bash
cd mininet
sudo python3 setup.py install
sudo python3 -c "import mininet; print('OK')"
```

### 4. Cài đặt Prometheus và Grafana

```bash
# Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v3.11.2/prometheus-3.11.2.linux-amd64.tar.gz
tar xvf prometheus-*.tar.gz
sudo mv prometheus-3.11.2.linux-amd64 /usr/local/bin/

# Grafana
sudo apt-get install -y apt-transport-https software-properties-common
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update && sudo apt-get install -y grafana
```

### 5. Cài đặt Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install "setuptools<58"
pip install "wheel<0.38"

pip install ryu==4.34 --no-build-isolation
pip install eventlet==0.30.2

pip install python-dateutil==2.8.2
pip install "gymnasium<1.1.0"

pip install -r requirements.txt
```

**Nội dung `requirements.txt`:**

```
stable-baselines3[extra]
psutil
scapy
matplotlib
seaborn
pettingzoo
tensorboard
requests
prometheus_client
```

### 6. Kiểm tra Mininet + Ryu cơ bản

```bash
# Terminal 1: Khởi động Ryu controller đơn
ryu-manager controllers/ryu_app_c1.py controllers/monitor_api.py

# Terminal 2: Tạo topology nhỏ và kiểm tra
sudo python3 mininets/custom_topo.py --topo linear --switches 4
# Trong Mininet CLI:
# mininet> pingall
# mininet> h1 iperf -s &
# mininet> h2 iperf -c h1
```

---

## Khởi Động Cluster Controllers

Đây là bước quan trọng — chạy **3 Ryu instance** trên các port khác nhau để tạo cluster thực sự:

```bash
# Terminal 1 — Controller 1
ryu-manager controllers/ryu_app_c1.py controllers/monitor_api.py \
    --ofp-tcp-listen-port 6633 --wsapi-port 8080

# Terminal 2 — Controller 2
ryu-manager controllers/ryu_app_c2.py \
    --ofp-tcp-listen-port 6634 --wsapi-port 8081

# Terminal 3 — Controller 3
ryu-manager controllers/ryu_app_c3.py \
    --ofp-tcp-listen-port 6635 --wsapi-port 8082

# Terminal 4 — Tạo topology kết nối với cả 3 controller
sudo python mininet/custom_topo.py --topo tree --depth 2 --fanout 3 \
    --controllers 127.0.0.1:6633,127.0.0.1:6634,127.0.0.1:6635
```

---

## Khởi Động Monitoring Stack (Prometheus + Grafana)

Prometheus và Grafana chạy **song song** với hệ thống RL, dùng để **trực quan hóa** — không tham gia trực tiếp vào vòng lặp RL.

```bash
# Terminal 5 — Khởi động Prometheus exporter (expose metrics từ psutil + Ryu)
python monitoring/prometheus_exporter.py
# Metrics có tại: http://localhost:9090/metrics

# Terminal 6 — Khởi động Prometheus server
prometheus --config.file=monitoring/prometheus.yml
# Web UI tại: http://localhost:9090

# Terminal 7 — Khởi động Grafana
sudo systemctl start grafana-server
# Dashboard tại: http://localhost:3000 (admin/admin)
# Import dashboard: Grafana → Import → upload monitoring/grafana/dashboard.json
```

### Các metrics được expose lên Prometheus

| Metric | Mô tả | Nguồn |
|---|---|---|
| `sdn_controller_cpu_percent` | CPU usage từng Ryu process | psutil |
| `sdn_controller_ram_mb` | RAM usage từng Ryu process | psutil |
| `sdn_controller_packet_in_rate` | Packet-in rate từng controller | Ryu REST API |
| `sdn_switch_count` | Số switch kết nối mỗi controller | Ryu REST API |
| `sdn_rl_reward` | Reward nhận được mỗi step | RL Agent |
| `sdn_migration_count` | Số lần migrate trong episode | RL Agent |
| `sdn_avg_latency_ms` | Latency trung bình toàn mạng | Mininet ping |

> **Lưu ý**: `sdn_controller_cpu_percent` và `sdn_controller_ram_mb` được psutil cung cấp **đồng thời** cho cả state vector RL lẫn Prometheus exporter. Hai luồng này chạy độc lập, không ảnh hưởng nhau.



### Kiểm tra giám sát và migration thủ công

```bash
# Xem metrics CPU/RAM từng controller process
python monitoring/system_monitor.py

# Thực hiện migrate switch thủ công (test trước khi dùng RL)
python utils/migration_executor.py --switch 1 --target-controller 2
```

### Huấn luyện RL Agent (Single-Agent DQN)

```bash
python rl_agent/train_dqn.py \
    --total-timesteps 200000 \
    --learning-rate 1e-3 \
    --batch-size 64 \
    --buffer-size 50000 \
    --exploration-fraction 0.15

# Theo dõi quá trình huấn luyện
tensorboard --logdir logs/
```

### Huấn luyện Multi-Agent RL

```bash
# Mỗi Ryu controller là một agent độc lập
python rl_agent/train_multiagent.py \
    --algorithm independent_dqn \
    --n-agents 3 \
    --total-timesteps 300000
```

### Đánh Giá và So Sánh Baselines

```bash
python rl_agent/evaluate.py \
    --model models/dqn_best.zip \
    --compare round_robin least_load \
    --episodes 50
```

Kết quả xuất ra `data/comparison.png` và `data/metrics.csv`.

---

## Môi Trường RL (Gymnasium)

### State Space

```python
# Vector state — ví dụ với 3 controllers
state = [
    cpu_c1, ram_c1, packet_in_rate_c1,   # Controller 1
    cpu_c2, ram_c2, packet_in_rate_c2,   # Controller 2
    cpu_c3, ram_c3, packet_in_rate_c3,   # Controller 3
]
# shape: (n_controllers * 3,)
```

### Action Space

```python
# Discrete: index tương ứng với cặp (switch_id, target_controller_id)
# Ví dụ: 10 switches × 3 controllers = 30 actions
action_space = Discrete(n_switches * n_controllers)
```

### Reward Function

```python
# Reward dương khi cân bằng, âm khi mất cân bằng hoặc migration thất bại
reward = (
    - alpha * np.var(cpu_loads)          # Phạt chênh lệch CPU
    - beta  * avg_latency_normalized     # Phạt độ trễ cao
    + gamma * balance_bonus              # Thưởng nếu tất cả controllers trong ngưỡng an toàn
    - delta * migration_penalty          # Phạt nếu migration gây overload mới
)
```

---

## 4 Kịch Bản Thử Nghiệm

### Kịch Bản 1 — Burst Traffic

```bash
python scenarios/scenario1_burst.py
# Traffic thấp ban đầu → đột ngột tăng packet-in trên nhóm switch
# Mục tiêu: Agent phát hiện overload và migrate trong < 5 giây
# Kỳ vọng: Giảm latency từ ~50ms xuống < 20ms, variance CPU giảm 30%
```

### Kịch Bản 2 — Topology Động

```bash
python scenarios/scenario2_dynamic_topo.py
# Thêm/xóa switch động trong lúc chạy
# Mục tiêu: Agent duy trì cân bằng lâu dài khi topo thay đổi
# Kỳ vọng: Throughput loss < 5% (so với Round-Robin là 20%)
```

### Kịch Bản 3 — Lỗi Controller Tạm Thời

```bash
python scenarios/scenario3_controller_fault.py
# Giả lập overload CPU một controller bằng stress tool
# Mục tiêu: Agent migrate switch sang controllers khỏe mạnh, phục hồi khi controller ổn định
# Kỳ vọng: Uptime 99%, reward hội tụ sau ~100 episodes
```

### Kịch Bản 4 — Traffic Ngẫu Nhiên (Poisson)

```bash
python scenarios/scenario4_random_traffic.py
# Packet-in rate theo phân phối Poisson, peak bất ngờ nhiều controller
# Mục tiêu: Agent xử lý multi-overload, ưu tiên switch có impact lớn nhất
# Kỳ vọng: Max latency giảm từ ~100ms xuống ~30ms, RAM < 70% tất cả controllers
```

---

## Kết Quả Đầu Ra

| File | Nội dung |
|---|---|
| `models/dqn_best.zip` | Model DQN tốt nhất sau training |
| `logs/` | TensorBoard event logs (reward, loss, episode length) |
| `data/reward_curve.png` | Reward vs Episodes — chứng minh hội tụ |
| `data/comparison.png` | So sánh RL vs Round-Robin vs Least-Load (latency, throughput) |
| `data/cpu_variance.png` | CPU variance theo thời gian giữa các controllers |
| `data/metrics.csv` | Số liệu thô toàn bộ thí nghiệm |

### Kết Quả Mong Đợi

```
=== Single-Agent DQN ===
  Mean reward:     -0.21  (baseline random: -0.78)
  Avg latency:     18ms   (Round-Robin: 45ms)
  CPU variance:    0.03   (Least-Load: 0.09)
  Migration count: 12/episode

=== Multi-Agent DQN ===
  Mean reward:     -0.14
  Avg latency:     14ms
  CPU variance:    0.02
  Migration count: 8/episode  (ít conflict hơn single-agent)
```

---

## Tham Số Huấn Luyện

| Tham số | Giá trị mặc định | Ghi chú |
|---|---|---|
| `total_timesteps` | 200,000 | Tăng lên 500,000 nếu chưa hội tụ |
| `learning_rate` | 1e-3 | Thử 1e-4 nếu training không ổn định |
| `exploration_fraction` | 0.15 | Tăng nếu agent hội tụ quá sớm vào local optima |
| `batch_size` | 64 | Giảm xuống 32 nếu hết RAM |
| `buffer_size` | 50,000 | Giảm nếu OutOfMemory |
| `alpha` (reward) | 1.0 | Trọng số penalty variance CPU |
| `beta` (reward) | 0.5 | Trọng số penalty latency |
| `gamma` (reward) | 0.3 | Trọng số bonus cân bằng |

---

## Xử Lý Sự Cố

### Ryu không khởi động được

```bash
# Kiểm tra port có đang bị chiếm không
lsof -i :6633
# Kill process cũ nếu cần
sudo fuser -k 6633/tcp
```

### Mininet bị treo hoặc lỗi topology

```bash
# Dọn dẹp toàn bộ Mininet
sudo mn -c
# Khởi động lại Open vSwitch
sudo service openvswitch-switch restart
```

### Agent không hội tụ

- Tăng `exploration_fraction` (0.2 → 0.3) để agent khám phá nhiều hơn
- Normalize state vector về `[0, 1]` trước khi đưa vào network
- Clip reward về `[-1, 1]` để tránh gradient exploding
- Kiểm tra reward function: đảm bảo không bị NaN khi controllers có load = 0

### Lỗi kết nối API Ryu

```bash
# Kiểm tra REST API controller 1 hoạt động
curl http://127.0.0.1:8080/stats/switches

# Kiểm tra controller 2
curl http://127.0.0.1:8081/stats/switches
```

---

## Công Cụ Sử Dụng

| Công cụ | Vai trò | Lý do chọn |
|---|---|---|
| **Mininet** | Mô phỏng topology SDN + generate traffic | Miễn phí, hỗ trợ OpenFlow đầy đủ |
| **Ryu** | Cluster controller + REST API giám sát | Python, dễ mở rộng, hỗ trợ multi-instance |
| **Gymnasium** | Định nghĩa MDP custom (State/Action/Reward) | Chuẩn RL, tích hợp tốt với SB3 |
| **Stable-Baselines3** | Implement & train DQN/PPO | Code ít, kết quả tốt, có TensorBoard |
| **PettingZoo** | Multi-agent environment | Chuẩn cho multi-agent RL |
| **psutil** | Đo CPU/RAM Ryu process → **cấp số liệu cho state vector RL** | Nhanh, trực tiếp, không overhead |
| **Prometheus** | Thu thập & lưu time-series metrics để phân tích | Chuẩn industry, scrape tự động |
| **Grafana** | Dashboard trực quan hóa real-time khi demo | Đẹp, dễ import/export dashboard |
| **Scapy** | Sinh traffic phức tạp (burst, DDoS-like) | Tùy biến cao hơn iperf |

---

## Tài Liệu Tham Khảo

- [Ryu Documentation](https://ryu.readthedocs.io/)
- [Mininet Walkthrough](http://mininet.org/walkthrough/)
- [Gymnasium Custom Environments](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [PettingZoo Multi-Agent](https://pettingzoo.farama.org/)
- [OpenFlow 1.3 Spec](https://www.opennetworking.org/wp-content/uploads/2014/10/openflow-spec-v1.3.0.pdf)
- [PettingZoo Multi-Agent](https://pettingzoo.farama.org/)
- [OpenFlow 1.3 Spec](https://www.opennetworking.org/wp-content/uploads/2014/10/openflow-spec-v1.3.0.pdf)
