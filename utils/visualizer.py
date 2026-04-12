import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Style config — nhất quán toàn bộ đồ án
# ------------------------------------------------------------------

COLORS = {
    "dqn_agent":   "#2196F3",   # xanh dương — RL agent
    "round_robin": "#FF9800",   # cam — Round-Robin
    "least_load":  "#F44336",   # đỏ — Least-Load
    "random":      "#9E9E9E",   # xám — Random baseline
    "multiagent":  "#4CAF50",   # xanh lá — Multi-agent
    "variance":    "#9C27B0",   # tím — CPU variance
    "latency":     "#00BCD4",   # cyan — latency
}

FIGSIZE_SINGLE = (10, 5)
FIGSIZE_COMPARE = (12, 6)
FIGSIZE_GRID = (14, 10)
DPI = 150

def _save(fig: plt.Figure, output_dir: str, filename: str) -> str:
    """Lưu figure và đóng lại để giải phóng memory."""
    os.makedirs(output_dir, exist_ok = True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi = DPI, bbox_inches = "tight")
    plt.close(fig)
    logger.info(f"[Visualizer] Đã lưu: {path}")
    return path

def _smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Moving average để làm mượt đường biểu đồ."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window

    return np.convolve(values, kernel, mode = "valid")

# ------------------------------------------------------------------
# 1. Reward curve — chứng minh agent hội tụ
# ------------------------------------------------------------------

def plot_reward_curve(rewards: List[float], output_dir: str = "data/", title: str = "DQN Training — Reward vs Episodes", smooth_window: int = 20) -> str:
    """
    Vẽ reward theo episodes trong quá trình training.

    Args:
        rewards:       List reward mỗi episode.
        smooth_window: Cửa sổ moving average để làm mượt đường.
    """
    fig, ax = plt.subplots(figsize = FIGSIZE_SINGLE)

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha = 0.3, color = COLORS["dqn_agent"], linewidth = 0.8, label = "Raw reward")

    if len(rewards) >= smooth_window:
        smoothed = _smooth(rewards, window = smooth_window)
        ep_smooth = np.arange(smooth_window - 1, len(rewards))
        ax.plot(ep_smooth, smoothed, color = COLORS["dqn_agent"], linewidth = 2.0, label = f"Smoothed (window = {smooth_window})")

    ax.axhline(y=np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards), color = "black", linestyle = "--", linewidth = 1.0, alpha = 0.6, label = "Mean (last 50 ep)")

    ax.set_xlabel("Episode", fontsize = 12)
    ax.set_ylabel("Total Reward", fontsize = 12)
    ax.set_title(title, fontsize = 14, fontweight = "bold")
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)

    return _save(fig, output_dir, "reward_curve.png")

# ------------------------------------------------------------------
# 2. CPU Variance — mục tiêu chính cần minimize
# ------------------------------------------------------------------

def plot_cpu_variance(variance_history: List[float], output_dir: str = "data/", algorithm_label: str = "DQN Agent", smooth_window: int = 10) -> str:
    """
    Vẽ CPU variance giữa các controllers theo thời gian.
    Variance thấp = cân bằng tốt.
    """
    fig, ax = plt.subplots(figsize = FIGSIZE_SINGLE)

    steps = np.arange(len(variance_history))
    ax.fill_between(steps, variance_history, alpha = 0.2, color = COLORS["variance"])
    ax.plot(steps, variance_history, color = COLORS["variance"], linewidth = 1.0, alpha = 0.5, label = "Raw variance")

    if len(variance_history) >= smooth_window:
        smoothed = _smooth(variance_history, window = smooth_window)
        ax.plot(np.arange(smooth_window - 1, len(variance_history)), smoothed, color = COLORS["variance"], linewidth = 2.0, label = f"Smoothed ({algorithm_label})")

    ax.axhline(y=0.02, color="green", linestyle = "--", linewidth = 1.2, label="Target (variance < 0.02)")

    ax.set_xlabel("Step", fontsize = 12)
    ax.set_ylabel("CPU Variance", fontsize = 12)
    ax.set_title("CPU Load Variance giữa các Controllers", fontsize = 14, fontweight = "bold")
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)

    return _save(fig, output_dir, "cpu_variance.png")

# ------------------------------------------------------------------
# 3. Comparison — RL vs baselines (biểu đồ chính để báo cáo)
# ------------------------------------------------------------------

def plot_comparison(agent_metrics: Dict[str, float], baseline_metrics: Dict[str, Dict[str, float]], output_dir: str = "data/") -> str:
    """
    Vẽ biểu đồ so sánh mean reward + std giữa DQN agent và các baselines.

    Args:
        agent_metrics:    Dict từ evaluate.py, có key "mean_reward", "std_reward".
        baseline_metrics: Dict {baseline_name: {"mean_reward": ..., "std_reward": ...}}.
    """
    # Gộp tất cả policies để vẽ
    policy_order = ["dqn_agent"] + list(baseline_metrics.keys())
    labels_map = {
        "dqn_agent":   "DQN Agent",
        "round_robin": "Round-Robin",
        "least_load":  "Least-Load",
        "random":      "Random",
        "multiagent":  "Multi-Agent DQN",
    }

    all_data = {"dqn_agent": agent_metrics}
    all_data.update(baseline_metrics)

    names  = [labels_map.get(p, p) for p in policy_order]
    means  = [all_data[p].get("mean_reward", 0.0) for p in policy_order]
    stds   = [all_data[p].get("std_reward", 0.0)  for p in policy_order]
    colors = [COLORS.get(p, "#607D8B") for p in policy_order]

    fig, ax = plt.subplots(figsize = FIGSIZE_COMPARE)

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr = stds, capsize = 6, color = colors, alpha = 0.85, edgecolor = "white", linewidth = 1.2)

    # Thêm giá trị lên mỗi cột
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.3f}",
            ha = "center", va = "bottom", fontsize = 10, fontweight = "bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize = 11)
    ax.set_ylabel("Mean Reward (±Std)", fontsize = 12)
    ax.set_title("So Sánh Hiệu Năng: RL Agent vs Baselines", fontsize = 14, fontweight = "bold")
    ax.grid(True, axis="y", alpha = 0.3)

    # Legend màu
    patches = [mpatches.Patch(color=COLORS.get(p, "#607D8B"), label=labels_map.get(p, p)) for p in policy_order]
    ax.legend(handles=patches, fontsize = 10, loc = "lower right")

    return _save(fig, output_dir, "comparison.png")

# ------------------------------------------------------------------
# 4. Migration count per episode
# ------------------------------------------------------------------

def plot_migration_count(migration_counts: List[int], output_dir: str = "data/", label: str = "DQN Agent") -> str:
    """Vẽ số lần migrate mỗi episode."""
    fig, ax = plt.subplots(figsize = FIGSIZE_SINGLE)

    episodes = np.arange(len(migration_counts))
    ax.bar(episodes, migration_counts, color = COLORS["dqn_agent"], alpha = 0.7, width = 0.8)
    ax.plot(episodes, _smooth(migration_counts, window = 10), color = "black", linewidth = 1.5, label = "Smoothed")

    ax.set_xlabel("Episode", fontsize = 12)
    ax.set_ylabel("Migration Count", fontsize = 12)
    ax.set_title(f"Số Lần Migration Mỗi Episode — {label}", fontsize = 14, fontweight = "bold")
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)

    return _save(fig, output_dir, "migration_count.png")

# ------------------------------------------------------------------
# 5. Latency over time
# ------------------------------------------------------------------

def plot_latency(latency_histories: Dict[str, List[float]], output_dir: str = "data/") -> str:
    """
    Vẽ latency theo thời gian cho nhiều policies.

    Args:
        latency_histories: {"dqn_agent": [...], "round_robin": [...], ...}
    """
    fig, ax = plt.subplots(figsize = FIGSIZE_COMPARE)

    labels_map = {
        "dqn_agent":   "DQN Agent",
        "round_robin": "Round-Robin",
        "least_load":  "Least-Load",
        "random":      "Random",
    }

    for policy, values in latency_histories.items():
        steps = np.arange(len(values))
        color = COLORS.get(policy, "#607D8B")
        label = labels_map.get(policy, policy)
        ax.plot(steps, values, alpha = 0.3, color = color, linewidth = 0.8)
        if len(values) >= 10:
            ax.plot(np.arange(9, len(values)), _smooth(values, 10), color = color, linewidth = 2.0, label = label)
        else:
            ax.plot(steps, values, color = color, linewidth = 2.0, label = label)

    ax.set_xlabel("Step", fontsize = 12)
    ax.set_ylabel("Latency (normalized)", fontsize = 12)
    ax.set_title("Latency Trung Bình theo Thời Gian", fontsize = 14, fontweight = "bold")
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)

    return _save(fig, output_dir, "latency_comparison.png")

# ------------------------------------------------------------------
# 6. Tổng hợp kịch bản — grid 4 plots
# ------------------------------------------------------------------

def plot_scenario_summary(rewards: List[float], variances: List[float], latencies: List[float], migration_counts: List[int], output_dir: str = "data/", scenario_name: str = "Kịch Bản") -> str:
    """
    Vẽ tổng hợp 4 biểu đồ cho một kịch bản test vào 1 figure.
    Dùng trong scenarios/*.py.
    """
    fig, axes = plt.subplots(2, 2, figsize = FIGSIZE_GRID)
    fig.suptitle(f"Kết Quả: {scenario_name}", fontsize = 16, fontweight = "bold")

    steps = np.arange(len(rewards))

    # Reward
    ax = axes[0, 0]
    ax.plot(steps, rewards, color = COLORS["dqn_agent"], alpha = 0.4, linewidth = 0.8)
    if len(rewards) >= 10:
        ax.plot(np.arange(9, len(rewards)), _smooth(rewards, 10), color = COLORS["dqn_agent"], linewidth = 2.0)
    ax.set_title("Reward", fontweight = "bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha = 0.3)

    # CPU Variance
    ax = axes[0, 1]
    ax.fill_between(np.arange(len(variances)), variances, alpha = 0.2, color = COLORS["variance"])
    ax.plot(np.arange(len(variances)), variances, color = COLORS["variance"], linewidth = 1.5)
    ax.axhline(y = 0.02, color = "green", linestyle = "--", linewidth = 1.0, label = "Target")
    ax.set_title("CPU Variance", fontweight = "bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Variance")
    ax.legend(fontsize = 9)
    ax.grid(True, alpha = 0.3)

    # Latency
    ax = axes[1, 0]
    ax.plot(np.arange(len(latencies)), latencies, color=COLORS["latency"], linewidth = 1.5)
    ax.set_title("Latency", fontweight = "bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Latency (normalized)")
    ax.grid(True, alpha = 0.3)

    # Migration count
    ax = axes[1, 1]
    ax.bar(np.arange(len(migration_counts)), migration_counts, color=COLORS["dqn_agent"], alpha = 0.7)
    ax.set_title("Migration Count / Episode", fontweight = "bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Count")
    ax.grid(True, alpha = 0.3)

    plt.tight_layout()
    filename = f"scenario_{scenario_name.lower().replace(' ', '_')}.png"
    return _save(fig, output_dir, filename)

# ------------------------------------------------------------------
# Entry point — test visualizer với dữ liệu giả
# ------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level = logging.INFO)

    output_dir = "data/test_plots"
    n_ep = 100
    n_steps = 200

    # Giả lập dữ liệu
    rewards_dqn = [-1.5 + 0.01 * i + np.random.normal(0, 0.2) for i in range(n_ep)]
    variances   = [0.3 * np.exp(-0.02 * i) + np.random.normal(0, 0.01) for i in range(n_steps)]
    variances   = [max(0, v) for v in variances]
    latencies   = [0.3 - 0.001 * i + np.random.normal(0, 0.02) for i in range(n_steps)]
    migrations  = [np.random.randint(0, 5) for _ in range(n_ep)]

    print("Vẽ reward curve...")
    plot_reward_curve(rewards_dqn, output_dir = output_dir)

    print("Vẽ CPU variance...")
    plot_cpu_variance(variances, output_dir = output_dir)

    print("Vẽ comparison...")
    agent_metrics = {"mean_reward": -0.45, "std_reward": 0.12}
    baseline_metrics = {
        "round_robin": {"mean_reward": -0.78, "std_reward": 0.20},
        "least_load":  {"mean_reward": -0.62, "std_reward": 0.18},
    }
    plot_comparison(agent_metrics, baseline_metrics, output_dir = output_dir)

    print("Vẽ migration count...")
    plot_migration_count(migrations, output_dir = output_dir)

    print("Vẽ scenario summary...")
    plot_scenario_summary(rewards_dqn, variances[:n_ep], latencies[:n_ep], migrations, output_dir = output_dir, scenario_name = "Burst Traffic")

    print(f"\nTất cả biểu đồ đã lưu tại: {output_dir}/")