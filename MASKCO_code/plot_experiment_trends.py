from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TSP_RESULTS = {
    "TSP100": {
        "cycles": [1, 40, 80, 160, 320],
        "gap": [5.178588, 0.022789, 0.013032, 0.008136, 0.005459],
    },
    "TSP500": {
        "cycles": [1, 40, 80, 160, 320],
        "gap": [7.282582, 0.228421, 0.134909, 0.084025, 0.035471],
    },
    "TSP1000": {
        "cycles": [1, 40, 80, 160, 320],
        "gap": [12.914725, 5.709113, 5.265668, 4.839186, 4.346584],
    },
}

TSP1000_TUNING_RESULTS = {
    "configs": [
        "base\nkr=0.2\n2opt=1",
        "kr=0.1\n2opt=5",
        "kr=0.05\n2opt=5",
        "kr=0.1\n2opt=10",
        "kr=0.05\n2opt=10",
        "kr=0.1\n2opt=20",
        "kr=0.05\n2opt=20",
    ],
    "gap": [4.346584, 2.264765, 2.074212, 1.384764, 1.298093, 0.636431, 0.616160],
}

CVRP_RESULTS = {
    "CVRP100": {
        "cycles": [40, 80, 160, 320],
        "gap": [0.229693, 0.178143, 0.141593, 0.111656],
    },
    "CVRP500": {
        "cycles": [40, 80, 160, 320],
        "gap": [0.543088, 0.458105, 0.393387, 0.341145],
    },
}

MIS_RESULTS = {
    "MIS ER": {
        "cycles": [1, 40, 80, 160, 320],
        "avg_size": [32.734375, 39.179688, 39.679688, 40.101562, 40.492188],
    },
    "MIS RB": {
        "cycles": [40, 80, 160, 320],
        "avg_size": [19.664000, 19.744000, 19.796000, 19.842000],
    },
}


def save_line_plot(
    results: dict[str, dict[str, list[float]]],
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
    use_log_x: bool = True,
) -> None:
    plt.figure(figsize=(8, 5), dpi=150)
    for name, values in results.items():
        plt.plot(
            values["cycles"],
            values[metric_key],
            marker="o",
            linewidth=2,
            label=name,
        )

    if use_log_x:
        plt.xscale("log", base=2)
    plt.xlabel("Cycles")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def save_tsp1000_tuning_plot() -> None:
    plt.figure(figsize=(9, 5), dpi=150)
    configs = TSP1000_TUNING_RESULTS["configs"]
    gaps = TSP1000_TUNING_RESULTS["gap"]
    bars = plt.bar(configs, gaps)

    for bar, gap in zip(bars, gaps):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{gap:.3f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("Gap (%)")
    plt.title("TSP1000 Parameter Tuning Results")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsp1000_parameter_tuning.png")
    plt.close()


def main() -> None:
    save_line_plot(
        TSP_RESULTS,
        metric_key="gap",
        ylabel="Gap (%)",
        title="TSP: Cycles vs Gap",
        filename="tsp_cycles_vs_gap.png",
    )
    save_tsp1000_tuning_plot()
    save_line_plot(
        CVRP_RESULTS,
        metric_key="gap",
        ylabel="Gap (%)",
        title="CVRP: Cycles vs Gap",
        filename="cvrp_cycles_vs_gap.png",
    )
    save_line_plot(
        MIS_RESULTS,
        metric_key="avg_size",
        ylabel="Average Independent Set Size",
        title="MIS: Cycles vs Average Size",
        filename="mis_cycles_vs_avg_size.png",
    )

    print(f"Saved figures to: {OUTPUT_DIR}")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
