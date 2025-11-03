import json

import pandas as pd

# Read the JSONL file
df = pd.read_json("hello-metrics.jsonl", lines=True)

# Extract the relevant columns
data = []
for _, row in df.iterrows():
    step = row["step"]

    # Add train reward
    if "env/all/reward/total" in row and pd.notna(row["env/all/reward/total"]):
        data.append({"step": step, "reward": row["env/all/reward/total"], "type": "Train"})

    # Add test reward if it exists
    if "test/env/all/reward/total" in row and pd.notna(row["test/env/all/reward/total"]):
        data.append({"step": step, "reward": row["test/env/all/reward/total"], "type": "Test"})

# Get max step for x-axis domain
max_step = max(d["step"] for d in data)

# Create Vega-Lite specification
vega_spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "description": "Train and Test Rewards over Steps",
    "width": 600,
    "height": 300,
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
    "data": {"values": data},
    "mark": {"type": "line", "point": True, "strokeWidth": 2.5},
    "encoding": {
        "x": {"field": "step", "type": "quantitative", "title": "Step", "scale": {"domain": [0, max_step]}},
        "y": {"field": "reward", "type": "quantitative", "title": "Reward"},
        "color": {
            "field": "type",
            "type": "nominal",
            "title": "Type",
            "scale": {"domain": ["Train", "Test"], "range": ["#2E86AB", "#E63946"]},
            "legend": {"orient": "none", "legendX": 520, "legendY": 220},
        },
        "strokeDash": {"field": "type", "type": "nominal", "legend": None},
    },
}

# Save as JSON
with open("hello-metrics-chart.json", "w") as f:
    json.dump(vega_spec, f, indent=2)

# # Create HTML file for visualization
# html_content = f"""<!DOCTYPE html>
# <html>
# <head>
#     <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
#     <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
#     <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
#     <title>Train vs Test Rewards</title>
# </head>
# <body>
#     <div id="vis"></div>
#     <script type="text/javascript">
#         var spec = {json.dumps(vega_spec, indent=2)};
#         vegaEmbed('#vis', spec);
#     </script>
# </body>
# </html>"""

# with open("hello-metrics-chart.html", "w") as f:
#     f.write(html_content)

# print("Created hello-metrics-chart.json and hello-metrics-chart.html")
# print(f"Total data points: {len(data)}")
# print(f"Train points: {sum(1 for d in data if d['type'] == 'train')}")
# print(f"Test points: {sum(1 for d in data if d['type'] == 'test')}")
