"""Create a cross-market summary from saved real-data ML experiments."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"])
    parser.add_argument("--output", default="reports/portfolio")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for symbol in args.symbols:
        path = Path("outputs") / symbol.lower() / "results.json"
        report = json.loads(path.read_text(encoding="utf-8"))

        if "synthetic" in report["data_source"].lower():
            raise ValueError("Synthetic data cannot enter the real-data report")

        selected = report["selected_by_validation_only"]

        rows.append({
            "symbol": symbol.upper(),
            "selected": selected,
            "cv": report["models"][selected]["cv_balanced_accuracy_mean"],
            "validation": report["models"][selected]["validation"]["balanced_accuracy"],
            "test": report["held_out_test"]["balanced_accuracy"],
            "baseline": report["held_out_baseline"]["balanced_accuracy"],
            "test_rows": report["rows"]["test"],
            "test_period": report["windows"]["test"]["start"] + " to " + report["windows"]["test"]["end"],
            "sha256": report["sha256"],
        })

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.bar([v - .18 for v in x],
           [r["test"] for r in rows], .35, label="ML model")
    ax.bar([v + .18 for v in x],
           [r["baseline"] for r in rows], .35, label="Majority baseline")

    ax.set_xticks(x, [r["symbol"] for r in rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out balanced accuracy")
    ax.set_title("Out-of-sample model vs baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "baseline_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for field, label in [
        ("cv", "Walk-forward CV"),
        ("validation", "Validation"),
        ("test", "Test")
    ]:
        ax.plot(
            [r["symbol"] for r in rows],
            [r[field] for r in rows],
            marker="o",
            label=label
        )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Generalization across chronological evaluation stages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "generalization.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Cross-Market ML Experiment Results",
        "",
        "Independent experiments using real historical daily market data.",
        "Models were selected using earlier validation data, not test results.",
        "",
        "| Market | Selected model | Walk-forward CV | Validation | Test | Baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['symbol']} | {r['selected']} | "
            f"{r['cv']:.3f} | {r['validation']:.3f} | "
            f"{r['test']:.3f} | {r['baseline']:.3f} |"
        )

    lines += ["", "## Findings", ""]

    for r in rows:
        verdict = (
            "outperformed" if r["test"] > r["baseline"]
            else "underperformed" if r["test"] < r["baseline"]
            else "matched"
        )
        lines.append(
            f"- **{r['symbol']}** {verdict} its baseline "
            f"({r['test']:.3f} vs {r['baseline']:.3f}); "
            f"{r['test_rows']} test records; {r['test_period']}."
        )

    lines += [
        "",
        "These results measure held-out next-day classification, "
        "not HFT performance or trading profitability.",
        "",
        "![Baseline comparison](baseline_comparison.png)",
        "",
        "![Generalization analysis](generalization.png)",
        "",
        "## Reproducibility",
        "",
        "All experiments use chronological splits, a one-bar embargo, "
        "and expanding-window cross-validation.",
        "Each market is trained and evaluated independently.",
        "Raw data are downloaded from Yahoo Finance through yfinance.",
        ""
    ]

    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nCROSS-MARKET RESULTS")
    for r in rows:
        print(
            f"{r['symbol']}: selected={r['selected']}, "
            f"test={r['test']:.3f}, baseline={r['baseline']:.3f}"
        )

    print("\nReport and visualizations saved to:", out)


if __name__ == "__main__":
    main()
