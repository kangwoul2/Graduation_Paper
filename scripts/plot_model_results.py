from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "model_performance_results.csv"
OUTPUT = ROOT / "docs" / "assets" / "model_metrics_generated.png"


def main() -> None:
    df = pd.read_csv(RESULTS)
    metrics = ["Test Accuracy", "Custom Accuracy", "Test F1"]
    ax = df.set_index("Algorithm")[metrics].plot(kind="bar")
    ax.set_title("Model evaluation metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.65)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=180)


if __name__ == "__main__":
    main()
