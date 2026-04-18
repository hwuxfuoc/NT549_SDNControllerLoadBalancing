import sys
import time
import logging
import argparse
import subprocess
import threading
from pathlib import Path
from typing import Optional

# Đảm bảo root dir trong sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

# ------------------------------------------------------------------
# Cấu hình mặc định
# ------------------------------------------------------------------

DEFAULT_CONFIG = {
    "num_controllers": 3,
    "num_switches": 12,
    "total_timesteps": 200000,
    "n_episodes": 20,
    "model_path": "models/best_model.zip",
    "model_dir": "models/",
    "log_dir": "logs/",
    "data_dir": "data/",
    "use_mock": True,
    "prometheus_port": 9100,
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def print_banner(text: str) -> None:
    line = "=" * 60
    logger.info(f"\n{line}\n  {text}\n{line}")

def print_startup_guide() -> None:
    """In hướng dẫn khởi động Ryu cluster và Mininet."""
    guide = """
╔═════════════════════════════════════════════════════════╗
║         HƯỚNG DẪN KHỞI ĐỘNG THỦ CÔNG                    ║
╠═════════════════════════════════════════════════════════╣
║  Mở 3 terminal riêng để chạy Ryu cluster:               ║
║                                                         ║
║  Terminal 1 — Controller 1:                             ║
║    ryu-manager controllers/ryu_app_c1.py \\             ║
║               controllers/monitor_api.py \\             ║
║               --ofp-tcp-listen-port 6633 \\             ║
║               --wsapi-port 8080                         ║
║                                                         ║
║  Terminal 2 — Controller 2:                             ║
║    ryu-manager controllers/ryu_app_c2.py \\             ║
║               --ofp-tcp-listen-port 6634 \\             ║
║               --wsapi-port 8081                         ║
║                                                         ║
║  Terminal 3 — Controller 3:                             ║
║    ryu-manager controllers/ryu_app_c3.py \\             ║
║               --ofp-tcp-listen-port 6635 \\             ║
║               --wsapi-port 8082                         ║
║                                                         ║
║  Terminal 4 — Mininet topology:                         ║
║    sudo python mininet/custom_topo.py \\                ║
║               --topo tree --depth 2 --fanout 3 \\       ║
║               --controllers 127.0.0.1:6633,\\           ║
║                             127.0.0.1:6634,\\           ║
║                             127.0.0.1:6635              ║
║                                                         ║
║  Terminal 5 — Prometheus exporter:                      ║
║    python monitoring/prometheus_exporter.py             ║
║                                                         ║
║  Terminal 6 — Prometheus server:                        ║
║    prometheus --config.file=monitoring/prometheus.yml   ║
║                                                         ║
║  Grafana: http://localhost:3000 (admin/admin)           ║
║    Import: monitoring/grafana/dashboard.json            ║
╚═════════════════════════════════════════════════════════╝
"""
    print(guide)

def start_prometheus_exporter(port: int = 9100) -> Optional[threading.Thread]:
    """Khởi động Prometheus exporter trong background thread."""
    try:
        from monitoring.prometheus_exporter import SDNPrometheusExporter, start_http_server
        from prometheus_client import start_http_server as prom_start

        prom_start(port)
        exporter = SDNPrometheusExporter()
        exporter.start()
        logger.info(f"[Prometheus] Exporter chạy tại http://localhost:{port}/metrics")
        return exporter
    
    except ImportError as e:
        logger.warning(f"[Prometheus] Không load được exporter: {e} — bỏ qua")
        return None
    
    except Exception as e:
        logger.warning(f"[Prometheus] Lỗi khởi động exporter: {e} — bỏ qua")
        return None

def wait_for_user(msg: str = "Nhấn Enter để tiếp tục...") -> None:
    input(f"\n  ⏸  {msg}")

# ------------------------------------------------------------------
# Các chế độ chạy
# ------------------------------------------------------------------

def mode_train(cfg: dict) -> None:
    """Train single-agent với thuật toán được chỉ định (DQN/PPO)."""
    algo = cfg.get("algo", "dqn").upper()
    print_banner(f"TRAIN SINGLE-AGENT — {algo}")
    from rl_agent.train import train
 
    model = train(
        algo=cfg.get("algo", "dqn"),
        total_timesteps=cfg["total_timesteps"],
        num_controllers=cfg["num_controllers"],
        num_switches=cfg["num_switches"],
        model_dir=cfg["model_dir"],
        log_dir=cfg["log_dir"],
        use_mock=cfg["use_mock"],
    )
    logger.info(f"✓ Training hoàn tất. Best model: {cfg['model_dir']}best_model.zip")
    logger.info(f"  Xem TensorBoard: tensorboard --logdir {cfg['log_dir']}")

def mode_train_multi(cfg: dict) -> None:
    """Train Multi-Agent DQN."""
    print_banner("TRAIN MULTI-AGENT DQN")
    from rl_agent.train_multiagent import MultiAgentSDNTrainer

    trainer = MultiAgentSDNTrainer(
        num_controllers=cfg["num_controllers"],
        num_switches=cfg["num_switches"],
        model_dir=str(Path(cfg["model_dir"]) / "multiagent/"),
    )

    trainer.train(total_timesteps_per_agent=cfg["total_timesteps"] // 2)
    trainer.evaluate(num_episodes=10)
    logger.info("✓ Multi-Agent training hoàn tất.")

def mode_evaluate(cfg: dict) -> None:
    """So sánh DQN agent vs Round-Robin vs Least-Load."""
    print_banner("ĐÁNH GIÁ & SO SÁNH BASELINES")
    from rl_agent.evaluate import compare_all

    model_path = cfg["model_path"]

    if not Path(model_path).exists():
        logger.error(f"Model không tồn tại: {model_path}")
        logger.info("Chạy training trước: python main.py --mode train")
        return

    agent_metrics, baseline_metrics = compare_all(
        model_path = model_path,
        n_episodes = cfg["n_episodes"],
        num_controllers = cfg["num_controllers"],
        num_switches = cfg["num_switches"],
        output_dir = cfg["data_dir"],
    )

    logger.info("✓ Đánh giá hoàn tất.")

def _run_scenario(scenario_num: int, model_path: str, data_dir: str, n_episodes: int) -> None:
    """Helper chạy một kịch bản."""
    scenario_map = {
        1: ("scenarios.scenario1_burst", "main", "data/scenario1"),
        2: ("scenarios.scenario2_dynamic_topo", "main", "data/scenario2"),
        3: ("scenarios.scenario3_controller_fault", "main", "data/scenario3"),
        4: ("scenarios.scenario4_random_traffic", "main", "data/scenario4"),
    }

    if scenario_num not in scenario_map:
        logger.error(f"Kịch bản {scenario_num} không hợp lệ (1-4)")
        return

    module_name, func_name, output_dir = scenario_map[scenario_num]
    import importlib
    mod = importlib.import_module(module_name)
    # Override N_EPISODES trong module
    mod.N_EPISODES = n_episodes
    getattr(mod, func_name)(model_path = model_path, output_dir = output_dir)

def mode_scenario(scenario_num: int, cfg: dict) -> None:
    """Chạy một kịch bản cụ thể."""
    scenario_names = {
        1: "BURST TRAFFIC",
        2: "DYNAMIC TOPOLOGY",
        3: "CONTROLLER FAULT",
        4: "POISSON TRAFFIC",
    }

    print_banner(f"KỊCH BẢN {scenario_num}: {scenario_names.get(scenario_num, '')}")

    model_path = cfg["model_path"]
    if not Path(model_path).exists():
        logger.error(f"Model không tồn tại: {model_path}")
        logger.info("Chạy training trước: python main.py --mode train")
        return

    _run_scenario(scenario_num, model_path, cfg["data_dir"], cfg["n_episodes"])
    logger.info(f"✓ Kịch bản {scenario_num} hoàn tất.")

def mode_all_scenarios(cfg: dict) -> None:
    """Chạy tất cả 4 kịch bản tuần tự."""
    print_banner("CHẠY TẤT CẢ 4 KỊCH BẢN")

    model_path = cfg["model_path"]
    if not Path(model_path).exists():
        logger.error(f"Model không tồn tại: {model_path}")
        return

    results = {}
    for i in range(1, 5):
        logger.info(f"\n>>> Bắt đầu Kịch Bản {i}...")
        try:
            _run_scenario(i, model_path, cfg["data_dir"], cfg["n_episodes"])
            results[i] = "✓ PASS"
        except Exception as e:
            logger.error(f"Kịch bản {i} lỗi: {e}")
            results[i] = f"✗ FAIL ({e})"

    print_banner("TỔNG KẾT 4 KỊCH BẢN")

    for num, status in results.items():
        logger.info(f"  Kịch bản {num}: {status}")

def mode_monitor(cfg: dict) -> None:
    """Chỉ chạy monitoring stack (system_monitor + prometheus exporter)."""
    print_banner("MONITORING STACK")
    from monitoring.system_monitor import ControllerProcessMonitor

    monitor = ControllerProcessMonitor(num_controllers = cfg["num_controllers"], monitor_interval = 2.0)
    monitor.start()

    exporter = start_prometheus_exporter(cfg["prometheus_port"])

    logger.info("Monitoring stack đang chạy. Ctrl+C để dừng.")
    logger.info(f"Metrics: http://localhost:{cfg['prometheus_port']}/metrics")

    try:
        while True:
            time.sleep(10)
            monitor.print_summary()

    except KeyboardInterrupt:
        monitor.stop()
        if exporter:
            exporter.stop()
        logger.info("Đã dừng monitoring stack.")

def mode_full(cfg: dict) -> None:
    """Workflow đầy đủ: train → evaluate → all scenarios."""
    print_startup_guide()
    wait_for_user("Đã khởi động Ryu cluster và Mininet? Nhấn Enter để tiếp tục...")

    # Khởi động Prometheus exporter
    exporter = start_prometheus_exporter(cfg["prometheus_port"])

    # 1. Train
    mode_train(cfg)

    # 2. Evaluate
    mode_evaluate(cfg)

    # 3. All scenarios
    mode_all_scenarios(cfg)

    print_banner("HOÀN THÀNH TOÀN BỘ PIPELINE")
    logger.info(f"  Model:   {cfg['model_dir']}")
    logger.info(f"  Logs:    {cfg['log_dir']}")
    logger.info(f"  Results: {cfg['data_dir']}")
    logger.info(f"\n  TensorBoard: tensorboard --logdir {cfg['log_dir']}")
    logger.info(f"  Grafana:     http://localhost:3000")

    if exporter:
        exporter.stop()

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SDN RL Load Balancer — Main Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py --mode train
  python main.py --mode train --timesteps 500000
  python main.py --mode evaluate --model models/dqn_best.zip
  python main.py --mode scenario1 --episodes 30
  python main.py --mode all-scenarios
  python main.py --mode full
        """
    )

    parser.add_argument(
        "--mode",
        choices = ["train", "train-multi", "evaluate", "scenario1", "scenario2", "scenario3", "scenario4", "all-scenarios", "monitor", "full", "guide"],
        default = "guide",
        help = "Chế độ chạy (default: guide — in hướng dẫn khởi động)"
    )
    parser.add_argument("--model", default = DEFAULT_CONFIG["model_path"], help = f"Đường dẫn model (default: {DEFAULT_CONFIG['model_path']})")
    parser.add_argument("--algo", choices = ["dqn", "ppo"], default = "dqn", help = "Thuật toán RL để train (default: dqn)")
    parser.add_argument("--timesteps", type = int, default = DEFAULT_CONFIG["total_timesteps"], help = f"Timesteps training (default: {DEFAULT_CONFIG['total_timesteps']})")
    parser.add_argument("--episodes", type = int, default = DEFAULT_CONFIG["n_episodes"], help = f"Số episodes đánh giá (default: {DEFAULT_CONFIG['n_episodes']})")
    parser.add_argument("--controllers", type=int, default=DEFAULT_CONFIG["num_controllers"], help = "Số controllers (default: 3)")
    parser.add_argument("--switches", type=int, default= DEFAULT_CONFIG["num_switches"], help = "Số switches (default: 12)")
    parser.add_argument("--real", action="store_true", help = "Dùng Ryu thật thay vì mock (yêu cầu Ryu + Mininet đang chạy)")
    parser.add_argument("--prom-port", type=int, default = DEFAULT_CONFIG["prometheus_port"], help = f"Prometheus exporter port (default: {DEFAULT_CONFIG['prometheus_port']})")

    args = parser.parse_args()

    cfg = {
        **DEFAULT_CONFIG,
        "model_path": args.model,
        "algo": args.algo,
        "total_timesteps": args.timesteps,
        "n_episodes": args.episodes,
        "num_controllers": args.controllers,
        "num_switches": args.switches,
        "use_mock": not args.real,
        "prometheus_port": args.prom_port,
    }

    # Tạo thư mục output
    for d in [cfg["model_dir"], cfg["log_dir"], cfg["data_dir"]]:
        Path(d).mkdir(parents = True, exist_ok = True)

    # Dispatch
    mode = args.mode
    if mode == "guide":
        print_startup_guide()
    elif mode == "train":
        mode_train(cfg)
    elif mode == "train-multi":
        mode_train_multi(cfg)
    elif mode == "evaluate":
        mode_evaluate(cfg)
    elif mode == "scenario1":
        mode_scenario(1, cfg)
    elif mode == "scenario2":
        mode_scenario(2, cfg)
    elif mode == "scenario3":
        mode_scenario(3, cfg)
    elif mode == "scenario4":
        mode_scenario(4, cfg)
    elif mode == "all-scenarios":
        mode_all_scenarios(cfg)
    elif mode == "monitor":
        mode_monitor(cfg)
    elif mode == "full":
        mode_full(cfg)

if __name__ == "__main__":
    main()