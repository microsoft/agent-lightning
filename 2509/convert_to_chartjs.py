"""
Convert three CSV files with val/reward and critic/rewards/mean to ChartJS JSON format.
"""

import csv
import json
from pathlib import Path


def read_csv_data(csv_path):
    """Read CSV file and extract step and val/reward columns."""
    steps = []
    val_rewards = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = float(row["_step"])
            steps.append(step)

            # val/reward may be empty string, convert to None
            val_reward = row["val/reward"]
            if val_reward == "":
                val_rewards.append(None)
            else:
                val_rewards.append(float(val_reward))

    return steps, val_rewards


def create_chartjs_json(csv_files, output_file, chart_title="Training Results"):
    """
    Create ChartJS JSON from multiple CSV files.

    Args:
        csv_files: List of tuples (csv_path, label_name)
        output_file: Path to output JSON file
        chart_title: Title for the chart
    """
    # Read all CSV files
    all_data = []
    steps = None

    for csv_path, label in csv_files:
        file_steps, val_rewards = read_csv_data(csv_path)
        all_data.append((label, val_rewards))

        # Use steps from first file (assuming all have same steps)
        if steps is None:
            steps = file_steps

    # Filter out steps where all datasets have null values
    filtered_steps = []
    filtered_data = [[] for _ in all_data]

    for i in range(len(steps)):
        # Check if at least one dataset has a non-null value at this step
        has_value = any(val_rewards[i] is not None for _, val_rewards in all_data)

        if has_value:
            filtered_steps.append(steps[i])
            for j, (_, val_rewards) in enumerate(all_data):
                filtered_data[j].append(val_rewards[i])

    # Update steps and data with filtered values
    steps = filtered_steps
    all_data = [(label, filtered_data[j]) for j, (label, _) in enumerate(all_data)]

    # Create datasets for ChartJS - only validation data
    datasets = []

    for label, val_rewards in all_data:
        datasets.append({"label": label, "data": val_rewards, "spanGaps": True})

    # Create ChartJS configuration
    chart_config = {
        "type": "line",
        "data": {"labels": steps, "datasets": datasets},
        "options": {
            "interaction": {"mode": "nearest", "intersect": False},
            "plugins": {
                "legend": {"display": True, "position": "top"},
                "title": {"display": True, "text": chart_title},
            },
            "scales": {
                "x": {"title": {"display": True, "text": "Step"}},
                "y": {"title": {"display": True, "text": "Reward"}},
            },
        },
    }

    # Write to file
    with open(output_file, "w") as f:
        json.dump(chart_config, f, indent=2)

    print(f"ChartJS JSON written to: {output_file}")
    print(f"Total steps: {len(steps)}")
    print(f"Datasets: {len(datasets)}")


def main():
    # Define the three CSV files
    base_dir = Path(__file__).parent

    csv_files = [
        (base_dir / "AgentZero_720dfb66-aware-panther-train_calc_metrics.csv", "calc"),
        (base_dir / "AgentZero_9d3076ad-bold-alien-train_calc_g1_metrics.csv", "calc_g1"),
        (base_dir / "AgentZero_c38c8374-aware-panther-train_calc_instrument_metrics.csv", "calc_instrument"),
    ]

    # Output file
    output_file = base_dir / "chartjs_output.json"

    # Create ChartJS JSON
    create_chartjs_json(csv_files, output_file, chart_title="Agent Training Results Comparison")

    # Also print the JSON to stdout for easy copy-paste
    print("\n" + "=" * 80)
    print("ChartJS JSON (for copy-paste into markdown):")
    print("=" * 80)
    with open(output_file, "r") as f:
        print(f.read())


if __name__ == "__main__":
    main()
