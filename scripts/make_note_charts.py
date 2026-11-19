#!/usr/bin/env python3
"""Draw the five charts of notes/lab-notes.md from the saved reports, as SVG files in notes/charts/.

    .venv/bin/python scripts/make_note_charts.py

Reads runs/target, runs/grader, runs/backup-10env and runs/final-backup. Nothing is trained or executed.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notes" / "charts"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8d8c86"
FONT = "font-family='-apple-system, Helvetica, Arial, sans-serif'"


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def wilson(k: float, n: int) -> tuple[float, float]:
    """95% Wilson interval for k successes in n."""
    if n == 0:
        return 0.0, 1.0
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


class Svg:
    def __init__(self, width: int, height: int, title: str, subtitle: str = ""):
        self.w, self.h, self.parts = width, height, []
        self.text(24, 30, title, size=16, weight=600)
        if subtitle:
            self.text(24, 50, subtitle, size=12, fill=MUTED)

    def text(self, x, y, s, size=12, fill=INK, anchor="start", weight=400):
        s = str(s).replace("&", "&amp;").replace("<", "&lt;")
        self.parts.append(f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}' {FONT}>{s}</text>")

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1.0, dash=""):
        extra = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='{stroke}' stroke-width='{width}'{extra}/>")

    def rect(self, x, y, w, h, fill, radius=0.0):
        self.parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(0.0, w):.1f}' height='{max(0.0, h):.1f}' fill='{fill}' rx='{radius}'/>")

    def circle(self, x, y, r, fill):
        self.parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{SURFACE}' stroke-width='2'/>")

    def path(self, points, stroke, width=2.0):
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        self.parts.append(f"<path d='{d}' fill='none' stroke='{stroke}' stroke-width='{width}' stroke-linejoin='round'/>")

    def save(self, name: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.parts)
        (OUT / name).write_text(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' viewBox='0 0 {self.w} {self.h}'>\n"
            f"<rect width='{self.w}' height='{self.h}' fill='{SURFACE}'/>\n{body}\n</svg>\n",
            encoding="utf-8",
        )
        print("wrote", (OUT / name).relative_to(ROOT))


def interval_rows(svg: Svg, rows, x0, x1, y0, step, label_x):
    """One row per item: a dot at the share with its 95% interval; rows = (label, k, n, colour, note)."""
    scale = lambda v: x0 + v * (x1 - x0)
    bottom = y0 + step * len(rows) - step / 2
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        svg.line(scale(tick), y0 - step / 2, scale(tick), bottom)
        svg.text(scale(tick), bottom + 16, f"{tick:.0%}", size=11, fill=MUTED, anchor="middle")
    for i, (label, k, n, colour, note) in enumerate(rows):
        y = y0 + i * step
        lo, hi = wilson(k, n)
        svg.text(label_x, y + 4, label, size=12, anchor="end")
        svg.line(scale(lo), y, scale(hi), y, stroke=colour, width=2)
        svg.circle(scale(k / n), y, 5, colour)
        svg.text(x1 + 12, y + 4, f"{k / n:.0%}  ({round(k)} of {n}){note}", size=11, fill=MUTED)
    return bottom


def chart_models():
    final, first = load("runs/target/report.json"), load("runs/backup-10env/target/report.json")
    export = load("webgpu-demo/export.json")["check"]
    rest, n = final["unused_held_out"], len(final["samples"])
    rows = [
        ("Base LFM3, no adapter", final["execution"]["base"]["correct"] * n, n, GREY, ""),
        ("First run: 162 chains, 10 environments", first["execution"]["P_good"]["correct"] * 40, 40, GREY, "  its own held-out set"),
        ("Final: 401 chains, 20 environments", final["execution"]["P_good"]["correct"] * n, n, BLUE, "  rows also used to pick settings"),
        ("Final, requests used for nothing else", rest["correct"] * rest["rows"], rest["rows"], BLUE, ""),
        ("Final, all held-out, 6 faulty requests left out", final["execution"]["P_good"]["correct"] * n + rest["correct"] * rest["rows"], n + rest["rows"] - FAULTY_REQUESTS, BLUE, ""),
        ("Final, exported to 4-bit ONNX", export["correct"] * export["requests"], export["requests"], BLUE, ""),
    ]
    svg = Svg(900, 336, "Calls that return the reference value when executed", "Held-out environments. Dot: share correct. Line: 95% interval for that many requests.")
    interval_rows(svg, rows, 300, 640, 96, 36, 288)
    svg.save("1-executed-correct.svg")


def chart_data_size():
    grader_now, grader_800, grader_exp = load("runs/grader/report.json"), load("runs/final-backup/grader/report.json"), load("runs/grader/experiments.json")
    small = next(r for r in grader_exp["placement_ablation"] if r["config"]["placement"] == "mlp_only")
    points = [
        (160, small["metrics"]["type_accuracy"]["1.0"], 100, "160 rows, 1 epoch"),
        (800, grader_800["runs"][0]["metrics"]["type_accuracy"]["1.0"], 100, "800 rows, 2 epochs"),
        (2400, grader_now["runs"][0]["metrics"]["type_accuracy"]["1.0"], grader_now["runs"][0]["dev_rows"], "2,400 rows, 2 epochs"),
    ]
    final, first = load("runs/target/report.json"), load("runs/backup-10env/target/report.json")
    target = [
        (first["rows"]["fit"], first["execution"]["P_good"]["correct"], 40, f"{first['rows']['fit']} rows (162 chains)"),
        (final["rows"]["fit"], final["unused_held_out"]["correct"], final["unused_held_out"]["rows"], f"{final['rows']['fit']} rows (401 chains)"),
    ]
    svg = Svg(900, 360, "Score against the amount of training data", "Bars: 95% interval. Grader: same settings throughout. Target: the two runs also differ in settings and held-out set.")
    for (x0, title, series, colour, lo_x, hi_x) in ((70, "Grader (D1): failure type named correctly", points, ORANGE, 100, 4000), (500, "Target (LFM3): call returns the reference value", target, BLUE, 200, 1200)):
        x1, top, bottom = x0 + 340, 100, 300
        xs = lambda v: x0 + (math.log10(v) - math.log10(lo_x)) / (math.log10(hi_x) - math.log10(lo_x)) * (x1 - x0)
        ys = lambda v: bottom - v * (bottom - top)
        svg.text(x0, 82, title, size=12, weight=600)
        for tick in (0, 0.25, 0.5, 0.75, 1.0):
            svg.line(x0, ys(tick), x1, ys(tick))
            svg.text(x0 - 8, ys(tick) + 4, f"{tick:.0%}", size=11, fill=MUTED, anchor="end")
        svg.path([(xs(x), ys(y)) for x, y, _, _ in series], colour)
        for x, y, n, label in series:
            lo, hi = wilson(y * n, n)
            svg.line(xs(x), ys(lo), xs(x), ys(hi), stroke=colour, width=2)
            svg.circle(xs(x), ys(y), 5, colour)
            svg.text(xs(x), ys(hi) - 8, f"{y:.0%}", size=12, anchor="middle", weight=600)
            svg.text(xs(x), bottom + 18, label, size=10.5, fill=MUTED, anchor="middle")
        svg.text((x0 + x1) / 2, bottom + 38, "training rows (log scale)", size=11, fill=MUTED, anchor="middle")
    svg.save("2-data-size.svg")


def chart_sweep():
    target, grader = load("runs/target/experiments.json"), load("runs/grader/experiments.json")
    name = lambda c: f"{c.get('placement', 'all_linear').replace('_', ' ')}, rank {c['rank']}, lr {c['lr']:g}"
    key = lambda c: f"{c.get('placement', 'all_linear')}-r{c['rank']}-lr{c['lr']:g}"
    t_rows = [(name(r["config"]), r["correct"] * 24, 24, BLUE if key(r["config"]) == target["chosen"] else GREY, "") for r in target["sweep"] + target["placement_ablation"]]
    g_rows = [(name(r["config"]), r["metrics"]["type_accuracy"]["1.0"] * 100, 100, ORANGE if r["config"]["name"] == grader["chosen"] else GREY, "") for r in grader["sweep"] + grader["placement_ablation"]]
    svg = Svg(900, 760, "Sweep and layer ablation: every setting with its 95% interval", "Coloured: the setting carried forward. One short training run per setting, one seed.")
    svg.text(24, 84, "Target: 200 rows, 1 epoch, scored on 24 held-out requests", size=12, weight=600)
    bottom = interval_rows(svg, t_rows, 300, 640, 110, 30, 288)
    svg.text(24, bottom + 56, "Grader: 160 rows, 1 epoch, failure type on 100 held-out rows", size=12, weight=600)
    interval_rows(svg, g_rows, 300, 640, bottom + 82, 30, 288)
    svg.save("3-sweep-and-ablation.svg")


def chart_margins():
    totals = load("runs/target/guidance_tokens.json")
    margins, floor = totals["margin"], math.log(10)
    width, top_cap = 0.5, 20.0
    bins = [0] * int(top_cap / width)
    for m in margins:
        bins[min(len(bins) - 1, int(m / width))] += 1
    svg = Svg(900, 380, "How far ahead P_good's first-choice token is of its runner-up", f"{len(margins):,} tokens of {totals['calls']} reference calls. Guidance can only pick a token left of the line (floor 0.1 = {floor:.1f} nats).")
    x0, x1, top, bottom = 60, 860, 90, 320
    peak = max(bins)
    xs = lambda v: x0 + v / top_cap * (x1 - x0)
    ys = lambda v: bottom - v / peak * (bottom - top)
    for tick in range(0, peak + 1, 50):
        svg.line(x0, ys(tick), x1, ys(tick))
        svg.text(x0 - 8, ys(tick) + 4, tick, size=11, fill=MUTED, anchor="end")
    for i, count in enumerate(bins):
        lead = i * width
        svg.rect(xs(lead) + 1, ys(count), xs(width) - xs(0) - 2, bottom - ys(count), ORANGE if lead < floor else BLUE, radius=2)
    for tick in range(0, int(top_cap) + 1, 2):
        svg.text(xs(tick), bottom + 16, tick, size=11, fill=MUTED, anchor="middle")
    svg.text((x0 + x1) / 2, bottom + 36, "lead in nats: log p(first choice) − log p(runner-up); last bar holds everything above 19.5", size=11, fill=MUTED, anchor="middle")
    svg.line(xs(floor), top - 6, xs(floor), bottom, stroke=INK, width=1.5, dash="4 3")
    under = sum(m <= floor for m in margins)
    svg.text(xs(floor) + 8, top + 4, f"floor: {under} of {len(margins):,} tokens are within reach ({under / len(margins):.1%})", size=12, weight=600)
    median = sorted(margins)[len(margins) // 2]
    svg.text(xs(median), ys(peak) - 6, f"median {median:.1f} nats", size=11, fill=MUTED, anchor="middle")
    svg.save("4-guidance-margins.svg")


FAULTY_REQUESTS = 6  # held-out requests that do not contain what the reference needs; found by reading the wrong calls

# The 17 wrong calls of the final adapter on all 231 held-out requests, read one by one (see notes/lab-notes.md).
MISSES = [
    ("Leaves out a step the request names", 6, BLUE, "second `enumerate` (3), `last 1` (3)"),
    ("Writes a different step than asked", 5, BLUE, "extra sorts, `split chars` for words, sum for count"),
    ("Request does not contain what the reference needs", 6, GREY, "key or extension lost in an analogy, cut-off or echoed instruction"),
]


def chart_misses():
    svg = Svg(900, 250, "The 17 wrong calls of the final adapter, by cause", "All 231 held-out requests. Grey: the request itself is at fault, so no model could match the reference.")
    x0, peak = 330, 8
    for i, (label, count, colour, note) in enumerate(MISSES):
        y = 92 + i * 46
        svg.text(x0 - 12, y + 5, label, size=12, anchor="end")
        svg.rect(x0, y - 10, count / peak * 360, 22, colour, radius=4)
        svg.text(x0 + count / peak * 360 + 10, y + 5, count, size=12, weight=600)
        svg.text(x0 + count / peak * 360 + 32, y + 5, note, size=10.5, fill=MUTED)
    svg.save("5-wrong-calls.svg")


if __name__ == "__main__":
    for draw in (chart_models, chart_data_size, chart_sweep, chart_misses, chart_margins):
        draw()
