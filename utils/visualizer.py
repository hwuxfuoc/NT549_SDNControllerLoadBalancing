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

# Key = label chính xác xuất hiện trong evaluate.py → compare_all()
COLORS = {
    "DQN Best":        "#1565C0",   # xanh dương đậm
    "DQN Final":       "#2196F3",   # xanh dương nhạt
    "PPO Best":        "#BF360C",   # đỏ đậm
    "PPO Final":       "#FF5722",   # đỏ nhạt
    "Multi-Agent DQN": "#4CAF50",   # xanh lá
    "random":          "#9E9E9E",   # xám
    "round_robin":     "#FF9800",   # cam
    "least_load":      "#F44336",   # đỏ tươi
    # fallback nội bộ (dùng trong plot_reward_curve, plot_cpu_variance)
    "_dqn":            "#2196F3",
    "_variance":       "#9C27B0",
    "_latency":        "#00BCD4",
}

# Tên hiển thị cho từng key baseline (RL labels đã đủ nghĩa)
DISPLAY_NAMES = {
    "random":      "Random",
    "round_robin": "Round-Robin",
    "least_load":  "Least-Load",
}

FIGSIZE_SINGLE  = (10, 5)
FIGSIZE_COMPARE = (13, 6)
FIGSIZE_GRID    = (14, 10)
DPI = 150


def _save(fig: plt.Figure, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Visualizer] Đã lưu: {path}")
    return path


def _smooth(values: List[float], window: int = 10) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _display(key: str) -> str:
    """Trả về tên hiển thị cho 1 policy key."""
    return DISPLAY_NAMES.get(key, key)


def _color(key: str) -> str:
    return COLORS.get(key, "#607D8B")

# ------------------------------------------------------------------
# 1. Reward curve
# ------------------------------------------------------------------

def plot_reward_curve(
    rewards: List[float],
    output_dir: str = "data/",
    title: str = "Training — Reward vs Episodes",
    smooth_window: int = 20,
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    episodes = np.arange(len(rewards))
    c = COLORS["_dqn"]

    ax.plot(episodes, rewards, alpha=0.3, color=c, linewidth=0.8, label="Raw reward")
    if len(rewards) >= smooth_window:
        smoothed = _smooth(rewards, smooth_window)
        ax.plot(
            np.arange(smooth_window - 1, len(rewards)), smoothed,
            color=c, linewidth=2.0, label=f"Smoothed (window={smooth_window})"
        )
    baseline = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
    ax.axhline(y=baseline, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Mean (last 50 ep)")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "reward_curve.png")

# ------------------------------------------------------------------
# 2. CPU Variance
# ------------------------------------------------------------------

def plot_cpu_variance(
    variance_history: List[float],
    output_dir: str = "data/",
    algorithm_label: str = "Agent",
    smooth_window: int = 10,
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    steps = np.arange(len(variance_history))
    c = COLORS["_variance"]

    ax.fill_between(steps, variance_history, alpha=0.2, color=c)
    ax.plot(steps, variance_history, color=c, linewidth=1.0, alpha=0.5, label="Raw variance")
    if len(variance_history) >= smooth_window:
        smoothed = _smooth(variance_history, smooth_window)
        ax.plot(
            np.arange(smooth_window - 1, len(variance_history)), smoothed,
            color=c, linewidth=2.0, label=f"Smoothed ({algorithm_label})"
        )
    ax.axhline(y=0.02, color="green", linestyle="--", linewidth=1.2, label="Target (< 0.02)")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("CPU Variance", fontsize=12)
    ax.set_title("CPU Load Variance giữa các Controllers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "cpu_variance.png")

# ------------------------------------------------------------------
# 3. Comparison — nhận dict gộp từ evaluate.py
# ------------------------------------------------------------------

def plot_comparison(
    all_results: Dict[str, Dict[str, float]],
    output_dir: str = "data/",
) -> str:
    """
    Vẽ biểu đồ so sánh mean reward ± std cho tất cả policies.

    Args:
        all_results: Dict {label: {"mean_reward": ..., "std_reward": ...}}
                     Truyền thẳng {**rl_results, **baseline_results} từ compare_all().
        output_dir:  Thư mục lưu ảnh.
    """
    if not all_results:
        logger.warning("[Visualizer] all_results rỗng — bỏ qua plot_comparison")
        return ""

    labels = list(all_results.keys())
    means  = [all_results[k].get("mean_reward", 0.0) for k in labels]
    stds   = [all_results[k].get("std_reward",  0.0) for k in labels]
    colors = [_color(k) for k in labels]
    names  = [_display(k) for k in labels]

    fig, ax = plt.subplots(figsize=FIGSIZE_COMPARE)
    x    = np.arange(len(names))
    bars = ax.bar(
        x, means, yerr=stds, capsize=6,
        color=colors, alpha=0.85, edgecolor="white", linewidth=1.2,
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + abs(min(means)) * 0.01 + 0.01,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Mean Reward (±Std)", fontsize=12)
    ax.set_title("So Sánh Hiệu Năng: RL vs Baselines", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    patches = [mpatches.Patch(color=_color(k), label=_display(k)) for k in labels]
    ax.legend(handles=patches, fontsize=9, loc="lower right")

    plt.tight_layout()
    return _save(fig, output_dir, "comparison.png")

# ------------------------------------------------------------------
# 4. Migration count
# ------------------------------------------------------------------

def plot_migration_count(
    migration_counts: List[int],
    output_dir: str = "data/",
    label: str = "Agent",
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    episodes = np.arange(len(migration_counts))
    ax.bar(episodes, migration_counts, color=COLORS["_dqn"], alpha=0.7, width=0.8)
    if len(migration_counts) >= 10:
        ax.plot(
            np.arange(9, len(migration_counts)),
            _smooth(migration_counts, 10),
            color="black", linewidth=1.5, label="Smoothed",
        )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Migration Count", fontsize=12)
    ax.set_title(f"Số Lần Migration Mỗi Episode — {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "migration_count.png")

# ------------------------------------------------------------------
# 5. Latency
# ------------------------------------------------------------------

def plot_latency(
    latency_histories: Dict[str, List[float]],
    output_dir: str = "data/",
) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE_COMPARE)
    for policy, values in latency_histories.items():
        steps = np.arange(len(values))
        c     = _color(policy)
        name  = _display(policy)
        ax.plot(steps, values, alpha=0.3, color=c, linewidth=0.8)
        if len(values) >= 10:
            ax.plot(
                np.arange(9, len(values)), _smooth(values, 10),
                color=c, linewidth=2.0, label=name,
            )
        else:
            ax.plot(steps, values, color=c, linewidth=2.0, label=name)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Latency (normalized)", fontsize=12)
    ax.set_title("Latency Trung Bình theo Thời Gian", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "latency_comparison.png")

# ------------------------------------------------------------------
# 6. Scenario summary — grid 4 plots
# ------------------------------------------------------------------

def plot_scenario_summary(
    rewards: List[float],
    variances: List[float],
    latencies: List[float],
    migration_counts: List[int],
    output_dir: str = "data/",
    scenario_name: str = "Kịch Bản",
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID)
    fig.suptitle(f"Kết Quả: {scenario_name}", fontsize=16, fontweight="bold")

    # Reward
    ax = axes[0, 0]
    ax.plot(rewards, color=COLORS["_dqn"], alpha=0.4, linewidth=0.8)
    if len(rewards) >= 10:
        ax.plot(np.arange(9, len(rewards)), _smooth(rewards, 10), color=COLORS["_dqn"], linewidth=2.0)
    ax.set_title("Reward", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # CPU Variance
    ax = axes[0, 1]
    ax.fill_between(np.arange(len(variances)), variances, alpha=0.2, color=COLORS["_variance"])
    ax.plot(variances, color=COLORS["_variance"], linewidth=1.5)
    ax.axhline(y=0.02, color="green", linestyle="--", linewidth=1.0, label="Target")
    ax.set_title("CPU Variance", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("Variance")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Latency
    ax = axes[1, 0]
    ax.plot(latencies, color=COLORS["_latency"], linewidth=1.5)
    ax.set_title("Latency", fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("Latency (normalized)")
    ax.grid(True, alpha=0.3)

    # Migration count
    ax = axes[1, 1]
    ax.bar(np.arange(len(migration_counts)), migration_counts, color=COLORS["_dqn"], alpha=0.7)
    ax.set_title("Migration Count / Episode", fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"scenario_{scenario_name.lower().replace(' ', '_')}.png"
    return _save(fig, output_dir, filename)

# ------------------------------------------------------------------
# Entry point — test với dữ liệu giả
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_dir = "data/test_plots"
    n_ep, n_steps = 100, 200

    rewards   = [-1.5 + 0.01 * i + np.random.normal(0, 0.2) for i in range(n_ep)]
    variances = [max(0, 0.3 * np.exp(-0.02 * i) + np.random.normal(0, 0.01)) for i in range(n_steps)]
    latencies = [0.3 - 0.001 * i + np.random.normal(0, 0.02) for i in range(n_steps)]
    migrations = [np.random.randint(0, 5) for _ in range(n_ep)]

    plot_reward_curve(rewards, output_dir=output_dir)
    plot_cpu_variance(variances, output_dir=output_dir)

    # Test plot_comparison với format mới (dict gộp)
    all_results = {
        "DQN Best":    {"mean_reward": -0.40, "std_reward": 0.10},
        "DQN Final":   {"mean_reward": -0.45, "std_reward": 0.12},
        "PPO Best":    {"mean_reward": -0.42, "std_reward": 0.11},
        "PPO Final":   {"mean_reward": -0.48, "std_reward": 0.13},
        "random":      {"mean_reward": -0.90, "std_reward": 0.25},
        "round_robin": {"mean_reward": -0.78, "std_reward": 0.20},
        "least_load":  {"mean_reward": -0.62, "std_reward": 0.18},
    }
    plot_comparison(all_results, output_dir=output_dir)
    plot_migration_count(migrations, output_dir=output_dir)
    plot_scenario_summary(rewards, variances[:n_ep], latencies[:n_ep], migrations,
                          output_dir=output_dir, scenario_name="Burst Traffic")

    print(f"\nTất cả biểu đồ đã lưu tại: {output_dir}/")