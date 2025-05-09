# Improved statistical_validation.py

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Configuration
NUM_RUNS = 5  # Keep at 5 due to compute constraints
OUTPUT_DIR = "statistical_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Results storage with additional metrics
results = {
    3: {"realism": [], "fid": [], "is_mean": [], "is_std": []},
    5: {"realism": [], "fid": [], "is_mean": [], "is_std": []},
    7: {"realism": [], "fid": [], "is_mean": [], "is_std": []},
    10: {"realism": [], "fid": [], "is_mean": [], "is_std": []},
}

# Run the existing code multiple times with different seeds
for run in range(NUM_RUNS):
    print(f"Starting run {run + 1}/{NUM_RUNS}")

    # Set environment variable for the seed
    os.environ["RANDOM_SEED"] = str(42 + run)

    # Run your training script
    subprocess.run(["python", "dcgan_training.py"], check=True)

    # Parse the results from your output files
    for client_count in [3, 5, 7, 10]:
        # Parse realism scores
        result_file = f"federated_learning/evaluation_results/clusters_{client_count}/round_2_scores.txt"
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                for line in f:
                    if "Average Realism Score:" in line:
                        score = float(line.split(":")[-1].strip())
                        results[client_count]["realism"].append(score)
                        break

        # Parse standard metrics (FID, IS)
        metrics_file = f"federated_learning/evaluation_results/standard_metrics/metrics_{client_count}clients.txt"
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "FID Score:" in line:
                        fid = float(line.split(":")[-1].strip())
                        results[client_count]["fid"].append(fid)
                    elif "Inception Score:" in line:
                        parts = line.split(":")[-1].strip().split("±")
                        is_mean = float(parts[0].strip())
                        is_std = float(parts[1].strip())
                        results[client_count]["is_mean"].append(is_mean)
                        results[client_count]["is_std"].append(is_std)

# Calculate comprehensive statistics
with open(f"{OUTPUT_DIR}/statistics_summary.txt", "w") as f:
    f.write("Statistical Summary of FedGAN Performance\n")
    f.write("=======================================\n\n")

    # Create a DataFrame for clean tabular formatting
    stats_df = pd.DataFrame(
        columns=["Client Count", "Metric", "Mean", "Std Dev", "95% CI"]
    )

    # Add all metrics to the DataFrame
    row_idx = 0
    for client_count, metrics in results.items():
        for metric_name, values in metrics.items():
            if values:  # Only process metrics with data
                mean = np.mean(values)
                std = np.std(values)

                # Calculate 95% confidence interval
                if len(values) > 1:
                    ci = stats.t.interval(
                        0.95,
                        len(values) - 1,
                        loc=mean,
                        scale=std / np.sqrt(len(values)),
                    )
                    ci_str = f"({ci[0]:.4f}, {ci[1]:.4f})"
                else:
                    ci_str = "N/A"

                # Add to DataFrame
                stats_df.loc[row_idx] = [
                    client_count,
                    metric_name,
                    f"{mean:.4f}",
                    f"{std:.4f}",
                    ci_str,
                ]
                row_idx += 1

                # Write directly to text file
                if metric_name == "realism":
                    f.write(
                        f"{client_count}-client setting (Realism Score): {mean:.4f} ± {std:.4f}\n"
                    )
                elif metric_name == "fid":
                    f.write(
                        f"{client_count}-client setting (FID): {mean:.4f} ± {std:.4f}\n"
                    )
                elif metric_name == "is_mean":
                    f.write(
                        f"{client_count}-client setting (IS): {mean:.4f} ± {std:.4f}\n"
                    )

    # Add DataFrame to file
    f.write("\n\nComprehensive Statistics Table:\n")
    f.write(stats_df.to_string(index=False))

    # Write state-of-the-art comparison
    f.write("\n\nComparison with State-of-the-Art Methods:\n")
    f.write("Method                  | FID       | IS        | Domain\n")
    f.write("------------------------|-----------|-----------|------------------\n")
    f.write(
        "FedGAN (Ours, 3 clients)| %-9.4f | %-9.4f | Diabetic Retinopathy\n"
        % (
            np.mean(results[3]["fid"]) if results[3]["fid"] else 0,
            np.mean(results[3]["is_mean"]) if results[3]["is_mean"] else 0,
        )
    )
    f.write("FedDAG (Che et al.)     | 35.62     | 3.21      | Medical Imaging\n")
    f.write("FedSynthCT (Raggio)     | 28.43     | N/A       | Brain MRI→CT\n")
    f.write("DFGM (Shi & Wang)       | 42.18     | 2.89      | Brain Tumors\n")


# Create visualizations for each metric
def create_metric_boxplot(metric_name, ylabel, filename, lower_better=False):
    plt.figure(figsize=(10, 6))

    # Collect data for boxplot
    data = []
    labels = []

    for client_count in [3, 5, 7, 10]:
        if metric_name in results[client_count] and results[client_count][metric_name]:
            data.append(results[client_count][metric_name])
            labels.append(f"{client_count} Clients")

    if not data:
        print(f"No data available for {metric_name} metric")
        return

    # Create boxplot
    box = plt.boxplot(data, patch_artist=True)

    # Add colors
    colors = ["lightblue", "lightgreen", "lightpink", "lightyellow"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Add individual data points
    for i, d in enumerate(data):
        # Add jitter
        x = np.random.normal(i + 1, 0.04, size=len(d))
        plt.scatter(x, d, alpha=0.6, s=30)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel(ylabel)
    if lower_better:
        plt.title(f"{ylabel} by Client Count (lower is better)")
    else:
        plt.title(f"{ylabel} by Client Count (higher is better)")

    # Add mean values as text
    for i, d in enumerate(data):
        if d:
            plt.text(
                i + 1,
                np.mean(d),
                f"{np.mean(d):.4f}",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()


# Create plots for each metric
create_metric_boxplot(
    "realism", "Realism Score", "realism_scores.png", lower_better=False
)
create_metric_boxplot("fid", "FID Score", "fid_scores.png", lower_better=True)
create_metric_boxplot(
    "is_mean", "Inception Score", "inception_scores.png", lower_better=False
)

# Statistical significance tests with better reporting
print("\nStatistical Significance Tests:")
with open(f"{OUTPUT_DIR}/significance_tests.txt", "w") as f:
    f.write("Statistical Significance Tests\n")
    f.write("=============================\n\n")

    metrics = ["realism", "fid", "is_mean"]
    metric_names = {
        "realism": "Realism Score",
        "fid": "FID Score",
        "is_mean": "Inception Score",
    }

    for metric in metrics:
        f.write(f"\n{metric_names[metric]}:\n")
        f.write("-" * (len(metric_names[metric]) + 1) + "\n")

        for i, client1 in enumerate([3, 5, 7, 10]):
            for client2 in [3, 5, 7, 10][i + 1 :]:
                # Check if we have enough data for a valid test
                if (
                    metric in results[client1]
                    and metric in results[client2]
                    and results[client1][metric]
                    and results[client2][metric]
                    and len(results[client1][metric]) == len(results[client2][metric])
                    and len(results[client1][metric]) > 1
                ):
                    t_stat, p_value = stats.ttest_rel(
                        results[client1][metric], results[client2][metric]
                    )

                    significance = (
                        "Significant" if p_value < 0.05 else "Not Significant"
                    )

                    result_str = f"{client1} vs {client2} clients: t={t_stat:.4f}, p={p_value:.4f} ({significance})"
                    print(result_str)
                    f.write(result_str + "\n")

                    # Add effect size (Cohen's d)
                    d = (
                        np.mean(results[client1][metric])
                        - np.mean(results[client2][metric])
                    ) / np.sqrt(
                        (
                            np.std(results[client1][metric]) ** 2
                            + np.std(results[client2][metric]) ** 2
                        )
                        / 2
                    )

                    effect_size = (
                        "Large"
                        if abs(d) > 0.8
                        else "Medium"
                        if abs(d) > 0.5
                        else "Small"
                        if abs(d) > 0.2
                        else "Negligible"
                    )
                    f.write(f"   Effect size (Cohen's d): {d:.4f} ({effect_size})\n")

print(f"\nStatistical analysis results saved to {OUTPUT_DIR}/")
