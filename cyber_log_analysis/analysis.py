import sys
import io

# ---------------------------------------------------------------------------
# Force UTF-8 on Windows cp1252 terminals  (no-op on Linux/macOS)
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

# =============================================
# GLOBAL THEME CONFIG
# =============================================
DARK_BG       = "#0f1117"
CARD_BG       = "#1a1c24"
TEXT_PRIMARY  = "#e8eaf0"
TEXT_DIM      = "#6b7280"
ACCENT_CYAN   = "#22d3ee"
ACCENT_AMBER  = "#fbbf24"
ACCENT_ROSE   = "#fb7185"
ACCENT_VIOLET = "#a78bfa"

PALETTE_MAIN = [
    "#22d3ee", "#a78bfa", "#fb7185",
    "#fbbf24", "#34d399", "#60a5fa",
    "#f472b6", "#fb923c", "#a3e635", "#7dd3fc"
]

DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]


# =============================================
# REUSABLE THEME HELPERS
# =============================================
def _apply_dark_theme(ax, title="", xlabel="", ylabel=""):
    """Apply the dark theme to a single Axes object."""
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_PRIMARY, labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=10, labelpad=10)
    ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=10, labelpad=10)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=14,
                 fontweight="bold", pad=16, loc="left")
    ax.xaxis.set_tick_params(which="both", length=0)
    ax.yaxis.set_tick_params(which="both", length=0)
    ax.yaxis.grid(True, color="#2a2d36", linewidth=0.6, linestyle="--")
    ax.set_axisbelow(True)


def _style_figure(fig):
    """Dark background for the whole figure."""
    fig.patch.set_facecolor(DARK_BG)


def _style_cbar(cbar):
    """Consistently style a colorbar without fragile plt.getp calls."""
    cbar.ax.tick_params(colors=TEXT_PRIMARY, labelsize=8)
    cbar.outline.set_edgecolor(TEXT_DIM)


# =============================================
# LOAD & CLEAN DATA
# =============================================
df = pd.read_csv("data/login_logs.csv")
df.columns = df.columns.str.strip()

print("\n[Preview] Data:")
print(df.head())
print("\n[Info] Column Info:")
print(df.info())

# -----------------------------
# CREATE TIMESTAMP FEATURES
# -----------------------------
df["timestamp"] = pd.to_datetime(
    df["Month"] + " " +
    df["DayOftheMonth"].astype(str) + " 2025 " +
    df["Time"]
)
df["hour"]     = df["timestamp"].dt.hour
df["day_name"] = df["timestamp"].dt.day_name()

# -----------------------------
# BASIC STATS
# -----------------------------
print(f"\n[Stats] Total Records   : {len(df)}")
print(f"[Stats] Unique IPs      : {df['IPAddress'].nunique()}")
print(f"[Stats] Unique Usernames: {df['Username'].nunique()}")


# =============================================
# 1  TOP 10 ATTACKING IPs  (horizontal bar)
# =============================================
top_ips = df["IPAddress"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(11, 5.5))
_style_figure(fig)
_apply_dark_theme(ax, title="Top 10 IPs by Login Attempts",
                  xlabel="Attempts", ylabel="")

# reverse so rank #1 sits at the top
ip_labels  = top_ips.index.to_numpy()[::-1]
ip_values  = top_ips.to_numpy()[::-1]

bars = ax.barh(ip_labels, ip_values,
               color=PALETTE_MAIN[:len(ip_values)],
               edgecolor="none", height=0.55)

# value labels at the end of each bar
for bar in bars:
    w = bar.get_width()
    ax.text(w + ip_values.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w)}", va="center", ha="left",
            color=TEXT_PRIMARY, fontsize=9, fontweight="bold")

ax.set_xlim(0, ip_values.max() * 1.12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.tick_params(axis="y", labelsize=8.5)
plt.tight_layout(pad=2.4)
plt.show()


# =============================================
# 2  MOST TARGETED USERNAMES  (horizontal bar)
# =============================================
top_users = df["Username"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(11, 5.5))
_style_figure(fig)
_apply_dark_theme(ax, title="Most Targeted Usernames",
                  xlabel="Attempts", ylabel="")

user_labels = top_users.index.to_numpy()[::-1]
user_values = top_users.to_numpy()[::-1]

bars = ax.barh(user_labels, user_values,
               color=PALETTE_MAIN[:len(user_values)],
               edgecolor="none", height=0.55)

for bar in bars:
    w = bar.get_width()
    ax.text(w + user_values.max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(w)}", va="center", ha="left",
            color=TEXT_PRIMARY, fontsize=9, fontweight="bold")

ax.set_xlim(0, user_values.max() * 1.12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.tick_params(axis="y", labelsize=8.5)
plt.tight_layout(pad=2.4)
plt.show()


# =============================================
# 3  ATTACKS BY HOUR  (area + peak callout)
# =============================================
hourly = df["hour"].value_counts().sort_index().reindex(range(24), fill_value=0)

fig, ax = plt.subplots(figsize=(11, 4.5))
_style_figure(fig)
_apply_dark_theme(ax, title="Login Attempts by Hour of Day",
                  xlabel="Hour (24 h)", ylabel="Attempts")

ax.fill_between(hourly.index, hourly.values,
                color=ACCENT_CYAN, alpha=0.15)
ax.plot(hourly.index, hourly.values,
        color=ACCENT_CYAN, linewidth=2.2, marker="o",
        markersize=5, markerfacecolor=CARD_BG,
        markeredgecolor=ACCENT_CYAN, markeredgewidth=2)

# highlight peak hour
peak_h   = hourly.idxmax()
peak_val = int(hourly[peak_h])

ax.scatter([peak_h], [peak_val],
           s=120, color=ACCENT_ROSE, zorder=5, edgecolors="none")
ax.annotate(f"Peak  {peak_val}",
            xy=(peak_h, peak_val),
            xytext=(peak_h + 1.4, peak_val + hourly.max() * 0.07),
            color=ACCENT_ROSE, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ACCENT_ROSE, lw=1.5))

ax.set_xticks(range(0, 24))
ax.set_xlim(-0.5, 23.5)
plt.tight_layout(pad=2.4)
plt.show()


# =============================================
# 4  ATTACK HEATMAP  (day x hour)
# =============================================
heatmap_data = (
    df.pivot_table(index="day_name", columns="hour",
                   values="IPAddress", aggfunc="count")
    .reindex(DAY_ORDER)
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(14, 5))
_style_figure(fig)
ax.set_facecolor(CARD_BG)

cmap_ylrd = sns.color_palette("YlOrRd", as_cmap=True)

# sns.heatmap returns the Axes -- grab colorbar cleanly from it
g = sns.heatmap(
    heatmap_data, ax=ax, cmap=cmap_ylrd,
    linewidths=0.5, linecolor=DARK_BG,
    annot=True, fmt=".0f",
    annot_kws={"size": 7.5, "color": TEXT_PRIMARY},
    cbar_kws={"shrink": 0.85, "pad": 0.02}
)
_style_cbar(g.collections[0].colorbar)

ax.set_title("Attack Heatmap - Day vs Hour",
             color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=16, loc="left")
ax.set_xlabel("Hour (24 h)", color=TEXT_DIM, fontsize=10, labelpad=10)
ax.set_ylabel("",           color=TEXT_DIM, fontsize=10)
ax.tick_params(colors=TEXT_PRIMARY, labelsize=9)
ax.spines[:].set_visible(False)
plt.tight_layout(pad=2.4)
plt.show()


# =============================================
# 5  MOST TARGETED PORTS  (vertical bar)
# =============================================
top_ports = df["Port"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
_style_figure(fig)
_apply_dark_theme(ax, title="Most Targeted Ports",
                  xlabel="Port", ylabel="Attempts")

port_labels = top_ports.index.astype(str).to_numpy()
port_values = top_ports.to_numpy()

bars = ax.bar(port_labels, port_values,
              color=PALETTE_MAIN[:len(port_values)],
              edgecolor="none", width=0.6)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            h + port_values.max() * 0.015,
            f"{int(h)}", ha="center", va="bottom",
            color=TEXT_PRIMARY, fontsize=9, fontweight="bold")

ax.set_ylim(0, port_values.max() * 1.1)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
plt.tight_layout(pad=2.4)
plt.show()


# =============================================
# 6  RISK SCORING & SUSPICIOUS IPs
# =============================================
# FIX: transform("size") replaces deprecated transform("count")
df["ip_attempts"] = df.groupby("IPAddress")["IPAddress"].transform("size")

THRESHOLD      = 20
suspicious_ips = df.loc[df["ip_attempts"] > THRESHOLD, "IPAddress"].unique()

print(f"\n[!] Suspicious IPs (>{THRESHOLD} attempts):")
print(suspicious_ips)

high_risk_records = df[df["IPAddress"].isin(suspicious_ips)]
print("\n[Records] Sample High-Risk Records:")
print(high_risk_records.head(10))


# =============================================
# 7  CORRELATION HEATMAP  (lower-triangle only)
# =============================================
numeric_cols = df.select_dtypes(include=np.number)
corr_matrix  = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
_style_figure(fig)
ax.set_facecolor(CARD_BG)

# custom diverging palette: violet  <-- 0 -->  cyan
cmap_div = mcolors.LinearSegmentedColormap.from_list(
    "viol_cyan", [ACCENT_VIOLET, DARK_BG, ACCENT_CYAN]
)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)   # hide upper triangle

g = sns.heatmap(
    corr_matrix, ax=ax, mask=mask, cmap=cmap_div,
    vmin=-1, vmax=1,
    annot=True, fmt=".2f",
    annot_kws={"size": 9, "color": TEXT_PRIMARY},
    linewidths=0.8, linecolor=DARK_BG,
    cbar_kws={"shrink": 0.8, "pad": 0.02}
)
_style_cbar(g.collections[0].colorbar)

ax.set_title("Feature Correlation Matrix",
             color=TEXT_PRIMARY, fontsize=14, fontweight="bold", pad=16, loc="left")
ax.tick_params(colors=TEXT_PRIMARY, labelsize=9)
ax.spines[:].set_visible(False)
plt.tight_layout(pad=2.4)
plt.show()


# =============================================
# 8  COMBINED DASHBOARD  (2 x 2 grid)
# =============================================
fig = plt.figure(figsize=(18, 10))
_style_figure(fig)
gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

# ---- A) Top IPs ----------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
_apply_dark_theme(ax1, title="Top 10 Attacking IPs", ylabel="Attempts")
ax1.barh(ip_labels, ip_values,
         color=PALETTE_MAIN[:10], edgecolor="none", height=0.55)
ax1.tick_params(axis="y", labelsize=7.5)
ax1.set_xlim(0, ip_values.max() * 1.15)

# ---- B) Hourly trend -----------------------------------
ax2 = fig.add_subplot(gs[0, 1])
_apply_dark_theme(ax2, title="Attempts by Hour",
                  xlabel="Hour", ylabel="Attempts")
ax2.fill_between(hourly.index, hourly.values,
                 color=ACCENT_CYAN, alpha=0.15)
ax2.plot(hourly.index, hourly.values, color=ACCENT_CYAN,
         linewidth=2, marker="o", markersize=4,
         markerfacecolor=CARD_BG,
         markeredgecolor=ACCENT_CYAN, markeredgewidth=1.8)
ax2.set_xticks(range(0, 24, 2))
ax2.set_xlim(-0.5, 23.5)

# ---- C) Targeted ports ---------------------------------
ax3 = fig.add_subplot(gs[1, 0])
_apply_dark_theme(ax3, title="Most Targeted Ports",
                  xlabel="Port", ylabel="Attempts")
ax3.bar(port_labels, port_values,
        color=PALETTE_MAIN[:len(port_values)], edgecolor="none", width=0.6)
ax3.set_ylim(0, port_values.max() * 1.12)
ax3.tick_params(axis="x", rotation=30, labelsize=8)

# ---- D) Correlation -------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD_BG)
g4 = sns.heatmap(corr_matrix, ax=ax4, mask=mask, cmap=cmap_div,
                 vmin=-1, vmax=1, annot=True, fmt=".2f",
                 annot_kws={"size": 7.5, "color": TEXT_PRIMARY},
                 linewidths=0.6, linecolor=DARK_BG,
                 cbar_kws={"shrink": 0.75, "pad": 0.02})
_style_cbar(g4.collections[0].colorbar)
ax4.set_title("Correlation Matrix", color=TEXT_PRIMARY,
              fontsize=13, fontweight="bold", pad=12, loc="left")
ax4.tick_params(colors=TEXT_PRIMARY, labelsize=8)
ax4.spines[:].set_visible(False)

# suptitle + layout tweak so it does NOT clip
fig.suptitle("Login-Attempt Threat Dashboard",
             color=TEXT_PRIMARY, fontsize=20, fontweight="bold")
fig.subplots_adjust(top=0.93)
plt.show()

print("\n[Done] Analysis Completed")
