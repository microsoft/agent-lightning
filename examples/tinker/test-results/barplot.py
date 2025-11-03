import csv
import json
from collections import defaultdict

# File mappings
files = {
    "qwen4b_withtool": "twenty_questions_qwen4b_withtool41_gpt5mini_20251027.jsonl",
    "qwen4b_notool": "twenty_questions_qwen4b_notool_gpt5mini_20251026.jsonl",
    "qwen30b_withtool": "twenty_questions_qwen30b_withtool41_gpt5mini_20251027.jsonl",
    "qwen30b_notool": "twenty_questions_qwen30b_notool_gpt5mini_20251026.jsonl",
    "qwen235b_notool": "twenty_questions_qwen235b_notool_gpt5mini_20251026.jsonl",
    "gpt41_notool": "twenty_questions_gpt41_notool_gpt5mini_20251026.jsonl",
}

# Display names for configurations
display_names = {
    "qwen4b_withtool": "Qwen-4B + Tool",
    "qwen4b_notool": "Qwen-4B No Tool",
    "qwen30b_withtool": "Qwen-30B + Tool",
    "qwen30b_notool": "Qwen-30B No Tool",
    "qwen235b_notool": "Qwen-235B No Tool",
    "gpt41_notool": "GPT-4.1 No Tool",
}


def load_split_mapping():
    """Load the CSV mapping from answer to train/test split."""
    split_map = {}
    with open("../q20_nouns.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_map[row["answer"].lower()] = row["split"]
    return split_map


def load_data(filename):
    """Load JSONL data and extract relevant information."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            entry = json.loads(line)
            if "correct" in entry and "category" in entry and "answer" in entry:
                data.append({"correct": entry["correct"], "category": entry["category"], "answer": entry["answer"]})
            else:
                print(f"Skipping entry: {entry}")
                continue
    return data


# Load split mapping
split_map = load_split_mapping()

# Load all data
all_data = {config: load_data(filename) for config, filename in files.items()}

# Calculate overall success rates
overall_data = []
for config, data in all_data.items():
    total = len(data)
    correct = sum(1 for item in data if item["correct"])
    success_rate = (correct / total * 100) if total > 0 else 0
    overall_data.append({"configuration": display_names[config], "success_rate": success_rate})

# Calculate success rate by category for specific configurations
configs_to_compare = ["qwen4b_withtool", "qwen30b_notool", "gpt41_notool"]
category_data = []

for config in configs_to_compare:
    data = all_data[config]
    categories = set(item["category"] for item in data)

    for category in categories:
        category_items = [item for item in data if item["category"] == category]
        total = len(category_items)
        correct = sum(1 for item in category_items if item["correct"])
        success_rate = (correct / total * 100) if total > 0 else 0
        category_data.append(
            {
                "category": category.capitalize(),
                "configuration": display_names[config],
                "success_rate": success_rate,
            }
        )

# Calculate success rate by train/test split for all configurations
split_data = []
for config, data in all_data.items():
    for split_type in ["train", "test"]:
        split_items = [item for item in data if split_map.get(item["answer"].lower()) == split_type]
        total = len(split_items)
        correct = sum(1 for item in split_items if item["correct"])
        success_rate = (correct / total * 100) if total > 0 else 0
        split_data.append(
            {"split": split_type.capitalize(), "configuration": display_names[config], "success_rate": success_rate}
        )

# Vega-Lite specification for overall success rate
vega_spec_overall = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Overall Success Rate Across Six Configurations",
    "width": 600,
    "height": 400,
    "config": {
        "axis": {"labelFontSize": 14, "titleFontSize": 16, "titleFontWeight": "normal"},
        "legend": {"labelFontSize": 14, "titleFontSize": 14},
    },
    "data": {"values": overall_data},
    "mark": {"type": "bar", "color": "#2E86AB", "opacity": 0.8},
    "encoding": {
        "x": {
            "field": "configuration",
            "type": "nominal",
            "title": "Configuration",
            "axis": {"labelAngle": -45, "labelAlign": "right"},
        },
        "y": {
            "field": "success_rate",
            "type": "quantitative",
            "title": "Success Rate (%)",
            "scale": {"domain": [0, 100]},
        },
    },
}

# Vega-Lite specification for success rate by category
vega_spec_category = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Success Rate by Category: Model Comparison",
    "width": 600,
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
    "data": {"values": category_data},
    "mark": {"type": "bar", "opacity": 0.8},
    "encoding": {
        "x": {
            "field": "category",
            "type": "nominal",
            "title": "Category",
            "axis": {"labelAngle": -45, "labelAlign": "right"},
        },
        "y": {
            "field": "success_rate",
            "type": "quantitative",
            "title": "Success Rate (%)",
            "scale": {"domain": [0, 100]},
        },
        "color": {
            "field": "configuration",
            "type": "nominal",
            "title": "Configuration",
            "scale": {
                "domain": [
                    display_names["qwen4b_withtool"],
                    display_names["qwen30b_notool"],
                    display_names["gpt41_notool"],
                ],
                "range": ["#2E86AB", "#E63946", "#06A77D"],
            },
        },
        "xOffset": {"field": "configuration"},
    },
}

# Vega-Lite specification for success rate by train/test split
vega_spec_split = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Success Rate by Train/Test Split: All Configurations",
    "width": 600,
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
    "data": {"values": split_data},
    "mark": {"type": "bar", "opacity": 0.8},
    "encoding": {
        "x": {
            "field": "configuration",
            "type": "nominal",
            "title": "Configuration",
            "axis": {"labelAngle": -45, "labelAlign": "right"},
        },
        "y": {
            "field": "success_rate",
            "type": "quantitative",
            "title": "Success Rate (%)",
            "scale": {"domain": [0, 100]},
        },
        "color": {
            "field": "split",
            "type": "nominal",
            "title": "Split",
            "scale": {
                "domain": ["Train", "Test"],
                "range": ["#2E86AB", "#E63946"],
            },
        },
        "xOffset": {"field": "split"},
    },
}

# Save specifications
with open("barplot_overall.json", "w") as f:
    json.dump(vega_spec_overall, f, indent=2)

with open("barplot_category.json", "w") as f:
    json.dump(vega_spec_category, f, indent=2)

with open("barplot_split.json", "w") as f:
    json.dump(vega_spec_split, f, indent=2)

print("Created barplot_overall.json, barplot_category.json, and barplot_split.json")

# Print summary statistics
print("\n=== Overall Success Rates ===")
for item in overall_data:
    print(f"{item['configuration']:25s}: {item['success_rate']:.1f}%")

print("\n=== Success Rates by Category ===")
categories = sorted(set(item["category"] for item in category_data))
for category in categories:
    print(f"\n{category}:")
    for item in category_data:
        if item["category"] == category:
            print(f"  {item['configuration']:25s}: {item['success_rate']:.1f}%")

print("\n=== Success Rates by Train/Test Split ===")
for split_type in ["Train", "Test"]:
    print(f"\n{split_type}:")
    for item in split_data:
        if item["split"] == split_type:
            print(f"  {item['configuration']:25s}: {item['success_rate']:.1f}%")
