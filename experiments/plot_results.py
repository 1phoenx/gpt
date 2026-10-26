from __future__ import annotations

import json
from pathlib import Path


COLORS = {"baseline": "#1565c0", "no_residual": "#c62828", "post_norm": "#2e7d32", "context_64": "#6a1b9a"}


def metrics(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("type") == "metrics":
            rows.append(row)
    return rows


def path_points(rows, key, x0, y0, width, height, ymin, ymax, xmax):
    points = []
    for row in rows:
        x = x0 + row["step"] / xmax * width
        y = y0 + height - (row[key] - ymin) / max(ymax - ymin, 1e-9) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.joinpath("results").glob("*.jsonl"))
    series = {path.stem: metrics(path) for path in files}
    width, height = 1200, 760
    panels = [(70, 55, 500, 270, "train_loss", "Train loss"), (630, 55, 500, 270, "valid_loss", "Validation loss"),
              (70, 425, 500, 270, "lr", "Learning rate"), (630, 425, 500, 270, "tokens_seen", "Tokens seen")]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#f8fafc"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.5}.line{fill:none;stroke-width:3}</style>']
    for x0, y0, w, h, key, title in panels:
        values = [row[key] for rows in series.values() for row in rows]
        ymin, ymax = min(values), max(values)
        if key in {"train_loss", "valid_loss"}:
            ymin, ymax = min(0, ymin - 0.3), ymax + 0.3
        else:
            ymin, ymax = ymin * 0.9, ymax * 1.1 if ymax else 1
        xmax = max(row["step"] for rows in series.values() for row in rows)
        svg.append(f'<text x="{x0}" y="{y0 - 18}" font-size="20" font-weight="bold">{title}</text>')
        for fraction in (0, .25, .5, .75, 1):
            y = y0 + h * fraction
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+h}" class="axis"/><line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" class="axis"/>')
        for name, rows in series.items():
            svg.append(f'<polyline points="{path_points(rows, key, x0, y0, w, h, ymin, ymax, xmax)}" class="line" stroke="{COLORS.get(name, "#111827")}"/>')
    legend_x = 850
    for index, name in enumerate(series):
        x = legend_x + (index % 2) * 140
        y = 350 + (index // 2) * 24
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x+24}" y2="{y}" class="line" stroke="{COLORS.get(name, "#111827")}"/><text x="{x+30}" y="{y+6}" font-size="14">{name}</text>')
    svg.append('</svg>')
    root.joinpath("artifacts/training_curves.svg").write_text("\n".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
