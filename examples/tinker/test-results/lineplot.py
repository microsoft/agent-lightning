import csv
import json

import pandas as pd

# Price configuration
TINKER_PRICE_TABLE = {
    "Qwen/Qwen3-4B-Instruct-2507": {
        "prompt": 0.07,  # per M tokens
        "completion": 0.22,
        "training": 0.22,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "prompt": 0.12,
        "completion": 0.30,
        "training": 0.36,
    },
}

A100_PRICE_PER_HOUR = 3.673

# File mappings
training_files = {
    "q20_no_search_4b": {
        "file": "train_q20_no_search_4b.jsonl",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "name": "Tinker Qwen3-4B",
    },
    "q20_no_search_30b": {
        "file": "train_q20_no_search_30b.jsonl",
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "name": "Tinker Qwen3-30B",
    },
    "q20_search_4b": {
        "file": "train_q20_search_4b.jsonl",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "name": "Tinker Qwen3-4B + Tool",
    },
}

verl_file = "AgentLightningQ20VERL_05kd8o8n_metrics.csv"


def load_training_data(filename):
    """Load training metrics from JSONL file."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry)
    return data


def load_verl_data(filename):
    """Load VERL metrics from CSV file."""
    df = pd.read_csv(filename)
    return df


def compute_tinker_cost(entry, model_name):
    """Compute cost for a Tinker training step."""
    prices = TINKER_PRICE_TABLE[model_name]

    # Get tokens
    prompt_tokens = entry.get("env/all/total_ob_tokens", 0)
    completion_tokens = entry.get("env/all/total_ac_tokens", 0)

    # Training tokens = 4 * (prompt + completion)
    training_tokens = 4 * (prompt_tokens + completion_tokens)

    # Calculate cost (prices are per million tokens)
    cost = (
        prompt_tokens * prices["prompt"] / 1_000_000
        + completion_tokens * prices["completion"] / 1_000_000
        + training_tokens * prices["training"] / 1_000_000
    )

    return cost


# Load all training data
all_training_data = {}
for key, config in training_files.items():
    all_training_data[key] = load_training_data(config["file"])

# Load VERL data
verl_data = load_verl_data(verl_file)

# ===== 1. Four separate figures for training and val accuracy =====
for key, config in training_files.items():
    data = all_training_data[key]

    chart_data = []
    for entry in data:
        step = entry.get("step", 0)

        # Training accuracy
        if "env/all/reward/total" in entry:
            chart_data.append({"step": step, "accuracy": entry["env/all/reward/total"], "type": "Train"})

        # Val accuracy
        if "test/env/all/reward/total" in entry:
            chart_data.append({"step": step, "accuracy": entry["test/env/all/reward/total"], "type": "Val"})

    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": f"Training and Val Accuracy - {config['name']}",
        "width": 800,
        "height": 400,
        "config": {
            "axis": {"labelFontSize": 14, "titleFontSize": 16, "titleFontWeight": "normal"},
            "legend": {
                "labelFontSize": 14,
                "titleFontSize": 14,
                "fillColor": "white",
                "strokeColor": "#ccc",
                "padding": 10,
                "cornerRadius": 5,
            },
        },
        "data": {"values": chart_data},
        "mark": {"type": "line", "point": True, "strokeWidth": 2.5},
        "encoding": {
            "x": {
                "field": "step",
                "type": "quantitative",
                "title": "Step",
                "scale": {"domain": [0, max(d["step"] for d in chart_data)]},
            },
            "y": {"field": "accuracy", "type": "quantitative", "title": "Accuracy"},
            "color": {
                "field": "type",
                "type": "nominal",
                "title": "Type",
                "scale": {"domain": ["Train", "Val"], "range": ["#2E86AB", "#E63946"]},
                "legend": {
                    "orient": "bottom-right" if "no_search" in key else "bottom-left",
                },
            },
        },
    }

    with open(f"lineplot_accuracy_{key}.json", "w") as f:
        json.dump(vega_spec, f, indent=2)

# VERL training and val accuracy
verl_chart_data = []
for _, row in verl_data.iterrows():
    step = row["_step"]

    if pd.notna(row.get("training/reward")):
        verl_chart_data.append({"step": step, "accuracy": row["training/reward"], "type": "Train"})

    if pd.notna(row.get("val/reward")):
        verl_chart_data.append({"step": step, "accuracy": row["val/reward"], "type": "Val"})

vega_spec_verl = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Training and Val Accuracy - VERL",
    "width": 800,
    "height": 400,
    "config": {
        "axis": {"labelFontSize": 14, "titleFontSize": 16, "titleFontWeight": "normal"},
        "legend": {
            "labelFontSize": 14,
            "titleFontSize": 14,
            "fillColor": "white",
            "strokeColor": "#ccc",
            "padding": 10,
            "cornerRadius": 5,
        },
    },
    "data": {"values": verl_chart_data},
    "mark": {"type": "line", "point": True, "strokeWidth": 2.5},
    "encoding": {
        "x": {
            "field": "step",
            "type": "quantitative",
            "title": "Step",
            "scale": {"domain": [0, max(d["step"] for d in verl_chart_data)]},
        },
        "y": {"field": "accuracy", "type": "quantitative", "title": "Accuracy"},
        "color": {
            "field": "type",
            "type": "nominal",
            "title": "Type",
            "scale": {"domain": ["Train", "Val"], "range": ["#2E86AB", "#E63946"]},
            "legend": {
                "orient": "bottom-right",
            },
        },
    },
}

with open("lineplot_accuracy_verl.json", "w") as f:
    json.dump(vega_spec_verl, f, indent=2)

# ===== 2. Compare env/all/ac_tokens_per_turn =====
tokens_chart_data = []
for key in ["q20_no_search_4b", "q20_no_search_30b", "q20_search_4b"]:
    config = training_files[key]
    data = all_training_data[key]

    for entry in data:
        step = entry.get("step", 0)
        if "env/all/ac_tokens_per_turn" in entry:
            tokens_chart_data.append(
                {
                    "step": step,
                    "tokens_per_turn": entry["env/all/ac_tokens_per_turn"],
                    "configuration": config["name"],
                }
            )

vega_spec_tokens = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Action Tokens Per Turn Comparison",
    "width": 800,
    "height": 400,
    "config": {
        "axis": {"labelFontSize": 14, "titleFontSize": 16, "titleFontWeight": "normal"},
        "legend": {
            "labelFontSize": 14,
            "titleFontSize": 14,
            "fillColor": "white",
            "strokeColor": "#ccc",
            "padding": 10,
            "cornerRadius": 5,
        },
    },
    "data": {"values": tokens_chart_data},
    "mark": {"type": "line", "point": True, "strokeWidth": 2.5},
    "encoding": {
        "x": {
            "field": "step",
            "type": "quantitative",
            "title": "Step",
            "scale": {"domain": [0, max(d["step"] for d in tokens_chart_data)]},
        },
        "y": {
            "field": "tokens_per_turn",
            "type": "quantitative",
            "title": "Action Tokens Per Turn",
        },
        "color": {
            "field": "configuration",
            "type": "nominal",
            "title": "Configuration",
            "scale": {
                "domain": ["Tinker Qwen3-4B", "Tinker Qwen3-30B", "Tinker Qwen3-4B + Tool"],
                "range": ["#2E86AB", "#E63946", "#06A77D"],
            },
            "legend": {
                "orient": "top-left",
            },
        },
    },
}

with open("lineplot_tokens_per_turn.json", "w") as f:
    json.dump(vega_spec_tokens, f, indent=2)

# ===== 3. Compare val accuracy vs cost =====
cost_chart_data = []

# Process Tinker data
for key in ["q20_no_search_4b", "q20_no_search_30b"]:
    config = training_files[key]
    data = all_training_data[key]

    cumulative_cost = 0
    cost_chart_data.append(
        {
            "cost": 0,
            "val_accuracy": data[0]["test/env/all/reward/total"],
            "configuration": config["name"],
        }
    )
    for entry in data[1:]:
        step = entry.get("step", 0)
        cumulative_cost += compute_tinker_cost(entry, config["model"])

        if "test/env/all/reward/total" in entry:
            cost_chart_data.append(
                {
                    "cost": cumulative_cost,
                    "val_accuracy": entry["test/env/all/reward/total"],
                    "configuration": config["name"],
                }
            )

# Process VERL data
cumulative_verl_cost = 0
cost_chart_data.append(
    {
        "cost": 0,
        "val_accuracy": verl_data["val/reward"].iloc[0],
        "configuration": "VERL Qwen2.5-3B",
    }
)
for _, row in verl_data.iterrows():
    if pd.notna(row.get("timing_s/step")):
        # Cost = (timing_s/step / 3600) * A100_PRICE_PER_HOUR
        step_cost = (row["timing_s/step"] / 3600) * A100_PRICE_PER_HOUR
        cumulative_verl_cost += step_cost

        if pd.notna(row.get("val/reward")):
            cost_chart_data.append(
                {
                    "cost": cumulative_verl_cost,
                    "val_accuracy": row["val/reward"],
                    "configuration": "VERL Qwen2.5-3B",
                }
            )

vega_spec_cost = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Val Accuracy vs Cost",
    "width": 800,
    "height": 400,
    "config": {
        "axis": {"labelFontSize": 14, "titleFontSize": 16, "titleFontWeight": "normal"},
        "legend": {
            "labelFontSize": 14,
            "titleFontSize": 14,
            "fillColor": "white",
            "strokeColor": "#ccc",
            "padding": 10,
            "cornerRadius": 5,
        },
    },
    "data": {"values": cost_chart_data},
    "mark": {"type": "line", "point": True, "strokeWidth": 2.5},
    "encoding": {
        "x": {
            "field": "cost",
            "type": "quantitative",
            "title": "Cost ($)",
        },
        "y": {"field": "val_accuracy", "type": "quantitative", "title": "Val Accuracy"},
        "color": {
            "field": "configuration",
            "type": "nominal",
            "title": "Configuration",
            "scale": {
                "domain": ["Tinker Qwen3-4B", "Tinker Qwen3-30B", "VERL Qwen2.5-3B"],
                "range": ["#2E86AB", "#E63946", "#06A77D"],
            },
            "legend": {
                "orient": "bottom-right",
            },
        },
    },
}

with open("lineplot_cost_vs_accuracy.json", "w") as f:
    json.dump(vega_spec_cost, f, indent=2)

# ===== 4. Compare val accuracy vs time =====
time_chart_data = []

# Process Tinker data
for key in ["q20_no_search_4b", "q20_no_search_30b"]:
    config = training_files[key]
    data = all_training_data[key]

    cumulative_time = 0
    time_chart_data.append(
        {
            "time_hours": 0,
            "val_accuracy": data[0]["test/env/all/reward/total"],
            "configuration": config["name"],
        }
    )
    for entry in data[1:]:
        step = entry.get("step", 0)
        # Use time/total field if available (in seconds)
        if "time/total" in entry:
            cumulative_time += entry["time/total"]

        if "test/env/all/reward/total" in entry:
            time_chart_data.append(
                {
                    "time_hours": cumulative_time / 3600,  # Convert to hours
                    "val_accuracy": entry["test/env/all/reward/total"],
                    "configuration": config["name"],
                }
            )

# Process VERL data
cumulative_verl_time = 0
time_chart_data.append(
    {
        "time_hours": 0,
        "val_accuracy": verl_data["val/reward"].iloc[0],
        "configuration": "VERL Qwen2.5-3B",
    }
)
for _, row in verl_data.iterrows():
    if pd.notna(row.get("timing_s/step")):
        cumulative_verl_time += row["timing_s/step"]

        if pd.notna(row.get("val/reward")):
            time_chart_data.append(
                {
                    "time_hours": cumulative_verl_time / 3600,  # Convert to hours
                    "val_accuracy": row["val/reward"],
                    "configuration": "VERL Qwen2.5-3B",
                }
            )

vega_spec_time = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Val Accuracy vs Time",
    "width": 800,
    "height": 400,
    "config": {
        "axis": {"labelFontSize": 14, "titleFontSize": 16, "titleFontWeight": "normal"},
        "legend": {
            "labelFontSize": 14,
            "titleFontSize": 14,
            "fillColor": "white",
            "strokeColor": "#ccc",
            "padding": 10,
            "cornerRadius": 5,
        },
    },
    "data": {"values": time_chart_data},
    "mark": {"type": "line", "point": True, "strokeWidth": 2.5},
    "encoding": {
        "x": {
            "field": "time_hours",
            "type": "quantitative",
            "title": "Time (hours)",
        },
        "y": {"field": "val_accuracy", "type": "quantitative", "title": "Val Accuracy"},
        "color": {
            "field": "configuration",
            "type": "nominal",
            "title": "Configuration",
            "scale": {
                "domain": ["Tinker Qwen3-4B", "Tinker Qwen3-30B", "VERL Qwen2.5-3B"],
                "range": ["#2E86AB", "#E63946", "#06A77D"],
            },
            "legend": {
                "orient": "bottom-right",
            },
        },
    },
}

with open("lineplot_time_vs_accuracy.json", "w") as f:
    json.dump(vega_spec_time, f, indent=2)

print("Created line plot specifications:")
print("  - lineplot_accuracy_q20_no_search_4b.json")
print("  - lineplot_accuracy_q20_no_search_30b.json")
print("  - lineplot_accuracy_q20_search_4b.json")
print("  - lineplot_accuracy_verl.json")
print("  - lineplot_tokens_per_turn.json")
print("  - lineplot_cost_vs_accuracy.json")
print("  - lineplot_time_vs_accuracy.json")

# Print summary statistics
print("\n=== Final Validation Accuracy ===")
for key, config in training_files.items():
    data = all_training_data[key]
    final_val_acc = None
    for entry in reversed(data):
        if "test/env/all/reward/total" in entry:
            final_val_acc = entry["test/env/all/reward/total"]
            break
    if final_val_acc is not None:
        print(f"{config['name']:25s}: {final_val_acc:.4f}")

verl_final_val = verl_data[verl_data["val/reward"].notna()]["val/reward"].iloc[-1]
print(f"{'VERL':25s}: {verl_final_val:.4f}")

print("\n=== Total Cost ===")
for key, config in training_files.items():
    data = all_training_data[key]
    total_cost = sum(compute_tinker_cost(entry, config["model"]) for entry in data)
    print(f"{config['name']:25s}: ${total_cost:.2f}")

verl_total_cost = sum(
    (row["timing_s/step"] / 3600) * A100_PRICE_PER_HOUR
    for _, row in verl_data.iterrows()
    if pd.notna(row.get("timing_s/step"))
)
print(f"{'VERL':25s}: ${verl_total_cost:.2f}")

print("\n=== Total Time (hours) ===")
for key, config in training_files.items():
    data = all_training_data[key]
    total_time = sum(entry.get("time/total", 0) for entry in data) / 3600
    print(f"{config['name']:25s}: {total_time:.2f} hours")

verl_total_time = (
    sum(row["timing_s/step"] for _, row in verl_data.iterrows() if pd.notna(row.get("timing_s/step"))) / 3600
)
print(f"{'VERL':25s}: {verl_total_time:.2f} hours")
