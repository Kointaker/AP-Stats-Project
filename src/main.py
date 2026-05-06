"""
TV Size vs Price — AP Statistics Analysis
Modules: pandas, numpy, scipy, plotly
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ─────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────

df = pd.read_csv("stats_spreadsheet.csv")

# Strip $ and commas, cast to float
df["Price"] = df["Price ($)"].str.replace(r"[$,]", "", regex=True).astype(float)
df["Size"]  = df["Screen Size (in)"]

# Normalize minor panel tech labels
df["Panel Tech"] = df["Panel Tech"].replace({"QNED": "LED", "LCD": "LED"})

# ─────────────────────────────────────────
# 2. OUTLIER DETECTION (IQR method)
# ─────────────────────────────────────────

Q1, Q3 = df["Price"].quantile(0.25), df["Price"].quantile(0.75)
IQR     = Q3 - Q1
lower   = Q1 - 1.5 * IQR
upper   = Q3 + 1.5 * IQR

outliers = df[(df["Price"] < lower) | (df["Price"] > upper)]
df_clean = df[(df["Price"] >= lower) & (df["Price"] <= upper)]

print("=" * 55)
print("  OUTLIERS DETECTED (IQR method, excluded from regression)")
print("=" * 55)
for _, row in outliers.iterrows():
    print(f"  {row['Brand']:12} {row['Size']:>4}\"  ${row['Price']:>10,.2f}  [{row['Panel Tech']}]")

# ─────────────────────────────────────────
# 3. SUMMARY STATISTICS
# ─────────────────────────────────────────

print("\n" + "=" * 55)
print("  OVERALL SUMMARY  (outliers removed, n={})".format(len(df_clean)))
print("=" * 55)
print(f"  Mean Price:    ${df_clean['Price'].mean():>10,.2f}")
print(f"  Median Price:  ${df_clean['Price'].median():>10,.2f}")
print(f"  Std Dev:       ${df_clean['Price'].std():>10,.2f}")
print(f"  Min:           ${df_clean['Price'].min():>10,.2f}")
print(f"  Max:           ${df_clean['Price'].max():>10,.2f}")

print("\n── Stats by Panel Technology ──")
panel_stats = (
    df_clean.groupby("Panel Tech")["Price"]
    .agg(Count="count", Mean="mean", Median="median", StdDev="std")
    .round(2)
)
print(panel_stats.to_string())

print("\n── Stats by Resolution ──")
res_stats = (
    df_clean.groupby("Resolution")["Price"]
    .agg(Count="count", Mean="mean", Median="median", StdDev="std")
    .round(2)
)
print(res_stats.to_string())

# ─────────────────────────────────────────
# 4. LINEAR REGRESSION
# ─────────────────────────────────────────

def run_regression(data, label=""):
    slope, intercept, r, p, se = stats.linregress(data["Size"], data["Price"])
    print(f"\n  [{label}]  n={len(data)}")
    print(f"    Slope:     ${slope:.2f} per inch")
    print(f"    Intercept: ${intercept:.2f}")
    print(f"    R²:        {r**2:.4f}")
    print(f"    r:         {r:.4f}")
    print(f"    p-value:   {p:.6f}  {'✓ significant' if p < 0.05 else '✗ not significant'}")
    return slope, intercept, r**2

print("\n" + "=" * 55)
print("  LINEAR REGRESSION RESULTS")
print("=" * 55)

slope_all, intercept_all, r2_all = run_regression(df_clean, "All Panel Types Combined")

print("\n── Regression by Panel Technology ──")
tech_reg = {}
for tech in sorted(df_clean["Panel Tech"].unique()):
    sub = df_clean[df_clean["Panel Tech"] == tech]
    if len(sub) >= 4:
        s, i, r2 = run_regression(sub, tech)
        tech_reg[tech] = (s, i, r2, len(sub))

# ─────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────

COLORS = {"LED": "#4C9BE8", "QLED": "#F4A100", "OLED": "#E84C4C"}

# ── Chart 1: Scatter — Size vs Price, colored by Panel Tech ──
x_line = np.linspace(df_clean["Size"].min(), df_clean["Size"].max(), 200)
y_line = slope_all * x_line + intercept_all

fig1 = px.scatter(
    df_clean, x="Size", y="Price",
    color="Panel Tech",
    color_discrete_map=COLORS,
    hover_data=["Brand", "Resolution", "Refresh Rate (Hz)", "Retailer"],
    title="TV Screen Size vs Price by Panel Technology",
    labels={"Size": "Screen Size (inches)", "Price": "Price ($)"},
    template="plotly_white",
)
fig1.add_trace(
    go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name=f"Overall Regression (R²={r2_all:.3f})",
        line=dict(dash="dash", color="gray", width=2),
    )
)
# Add per-tech regression lines
for tech, (s, i, _, n) in tech_reg.items():
    sub = df_clean[df_clean["Panel Tech"] == tech]
    x_t = np.linspace(sub["Size"].min(), sub["Size"].max(), 100)
    fig1.add_trace(
        go.Scatter(
            x=x_t, y=s * x_t + i,
            mode="lines",
            name=f"{tech} fit",
            line=dict(color=COLORS.get(tech, "purple"), width=1.5, dash="dot"),
            visible="legendonly",  # toggle on if needed
        )
    )
fig1.show()

# ── Chart 2: Box Plot — Price by Panel Tech ──
fig2 = px.box(
    df_clean, x="Panel Tech", y="Price",
    color="Panel Tech",
    color_discrete_map=COLORS,
    points="all",
    title="Price Distribution by Panel Technology",
    labels={"Price": "Price ($)"},
    template="plotly_white",
)
fig2.show()

# ── Chart 3: Box Plot — Price by Size Range ──
bins   = [0, 44, 54, 64, 74, 84, 120]
labels = ['32–43"', '44–54"', '55–64"', '65–74"', '75–84"', '85+"']
df_clean = df_clean.copy()
df_clean["Size Range"] = pd.cut(df_clean["Size"], bins=bins, labels=labels)

fig3 = px.box(
    df_clean, x="Size Range", y="Price",
    color="Panel Tech",
    color_discrete_map=COLORS,
    title="Price Distribution by Size Range (colored by Panel Tech)",
    labels={"Price": "Price ($)", "Size Range": "Size Range"},
    template="plotly_white",
)
fig3.show()

# ── Chart 4: Bar — Mean Price by Panel Tech + Size Range ──
pivot = (
    df_clean.groupby(["Size Range", "Panel Tech"])["Price"]
    .mean()
    .reset_index()
    .rename(columns={"Price": "Mean Price"})
)
fig4 = px.bar(
    pivot, x="Size Range", y="Mean Price", color="Panel Tech",
    barmode="group",
    color_discrete_map=COLORS,
    title="Mean Price by Size Range and Panel Technology",
    labels={"Mean Price": "Mean Price ($)"},
    template="plotly_white",
)
fig4.show()

print("\n✓ All charts rendered. Analysis complete.")