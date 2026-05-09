"""
PRISM Dataset Analysis — N subunits × function × diversity
"""

import json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

# ── Load and enrich ──────────────────────────────────────────────────────────

df = pd.read_csv("data/interim/labeled_dataset.tsv", sep="\t")
uids = df.uniprot_ids.apply(lambda x: json.loads(x) if pd.notna(x) else [])
df["n_unique"]  = uids.apply(lambda x: len(set(x)))
df["diversity"] = df["n_unique"] / df["n_subunits"].clip(lower=1)

# Classify repeat structure
def repeat_class(row):
    if row.diversity >= 1.0:          return "All-unique\n(pure heteromer)"
    if row.diversity <= 1 / row.n_subunits + 1e-6 and row.n_unique == 1:
                                      return "All-same\n(pure homomer)"
    return "Mixed\n(repeated subunits)"

df["repeat_class"] = df.apply(repeat_class, axis=1)

# Functional keyword categories
FUNC_CATS = {
    "Kinase /\nPhosphatase":    r"kinas|phosphatas|phosphoryl",
    "Transcription /\nRNA Pol": r"transcri|rna pol|mediator",
    "Protease /\nPeptidase":    r"proteas|peptidas|caspas|trypsin",
    "Transport /\nChannel":     r"transport|channel|pump|import|export",
    "Signaling /\nGTPase":      r"signal|gtpas|ras |rho |rab |ran |g.protein",
    "Chaperone /\nFolding":     r"chaperoni|groel|groes|hsp|heat shock",
    "Ubiquitin /\nLigase":      r"ubiquitin|ligase|cullin|skp|ring.*finger",
    "Ribosome /\nTranslation":  r"ribosom|translat|eif|elongation factor",
    "DNA Repair /\nReplication":r"dna rep|repair|pcna|helicase|primase",
}

def classify(name):
    if not isinstance(name, str): return "Other"
    n = name.lower()
    for cat, pat in FUNC_CATS.items():
        if re.search(pat, n): return cat
    return "Other"

df["func_cat"] = df.name.apply(classify)

# Source labels
SRC_LABEL = {
    "corum": "CORUM 5.2",
    "rcsb_search": "RCSB PDB",
    "complex_portal": "Complex Portal",
    "experimental_gt": "Experimental GT",
}
df["source_label"] = df.source.map(SRC_LABEL)

TIER_LABEL = {1: "Tier 1 ★\n(Experimental)", 2: "Tier 2 ◆\n(Structure + BSA)", 3: "Tier 3 ◇\n(Seq-length BSA)"}
df["tier_label"] = df.gt_tier.map(TIER_LABEL)

# ── Color palettes ────────────────────────────────────────────────────────────

SRC_COLORS  = {"CORUM 5.2": "#4e9af1", "RCSB PDB": "#f1a54e",
               "Complex Portal": "#6bd16b", "Experimental GT": "#e05c5c"}
TIER_COLORS = {"Tier 1 ★\n(Experimental)": "#d29922",
               "Tier 2 ◆\n(Structure + BSA)": "#4e9af1",
               "Tier 3 ◇\n(Seq-length BSA)": "#8b949e"}
FUNC_COLORS = plt.cm.tab10(np.linspace(0, 1, len(FUNC_CATS) + 1))

# ── Figure ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 18), facecolor="#0d1117")
fig.suptitle("PRISM Training Corpus — Complex Composition & Diversity Analysis",
             fontsize=16, fontweight="bold", color="#e6edf3", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

ax_style = dict(facecolor="#161b22", labelcolor="#8b949e",
                tick_params=dict(colors="#8b949e"))

def style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="#e6edf3", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color="#8b949e", fontsize=9)
    ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.grid(axis="y", color="#30363d", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

# ── A: N_subunits distribution stacked by source ─────────────────────────────

ax_a = fig.add_subplot(gs[0, :2])
style(ax_a, "A — Subunit Count Distribution by Source",
      "Number of subunits per complex", "Complex count")

n_vals = range(2, 13)
src_order = ["CORUM 5.2", "RCSB PDB", "Complex Portal", "Experimental GT"]
bottom = np.zeros(len(n_vals))
for src in src_order:
    counts = [len(df[(df.source_label == src) & (df.n_subunits == n)]) for n in n_vals]
    bars = ax_a.bar(list(n_vals), counts, bottom=bottom,
                    color=SRC_COLORS[src], label=src, width=0.7, alpha=0.9)
    bottom += np.array(counts)

ax_a.set_xticks(list(n_vals))
ax_a.legend(loc="upper right", fontsize=8, framealpha=0.2,
            labelcolor="#e6edf3", facecolor="#161b22")

# Annotate total per bar
for i, n in enumerate(n_vals):
    total = int(bottom[i])
    if total > 0:
        ax_a.text(n, total + 30, f"{total:,}", ha="center", va="bottom",
                  color="#8b949e", fontsize=7)

# ── B: Tier breakdown per source ─────────────────────────────────────────────

ax_b = fig.add_subplot(gs[0, 2])
style(ax_b, "B — Tier Distribution", "Source", "% of complexes")

tier_order = list(TIER_LABEL.values())
x = np.arange(len(src_order))
w = 0.22
for ti, tlabel in enumerate(tier_order):
    vals = []
    for src in src_order:
        sub = df[df.source_label == src]
        vals.append(100 * len(sub[sub.tier_label == tlabel]) / max(len(sub), 1))
    ax_b.bar(x + (ti - 1) * w, vals, width=w,
             color=TIER_COLORS[tlabel], label=tlabel.replace("\n", " "), alpha=0.9)

ax_b.set_xticks(x)
ax_b.set_xticklabels([s.replace(" ", "\n") for s in src_order], fontsize=7)
ax_b.legend(fontsize=6.5, framealpha=0.2, labelcolor="#e6edf3", facecolor="#161b22")
ax_b.set_ylim(0, 105)
ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# ── C: N_subunits by functional category (violin / box) ──────────────────────

ax_c = fig.add_subplot(gs[1, :])
style(ax_c, "C — Subunit Count by Functional Category",
      "Functional category (name-keyword classified)", "N subunits")

cat_order = sorted(FUNC_CATS.keys(), key=lambda c: -len(df[df.func_cat == c]))
cat_order.append("Other")
cat_data  = [df.loc[df.func_cat == c, "n_subunits"].values for c in cat_order]
cat_counts= [len(d) for d in cat_data]
cat_labels= [f"{c}\n(n={cnt:,})" for c, cnt in zip(cat_order, cat_counts)]

parts = ax_c.violinplot(cat_data, positions=range(len(cat_order)),
                         showmedians=True, showextrema=False, widths=0.7)
for i, (pc, cat) in enumerate(zip(parts["bodies"], cat_order)):
    c = FUNC_COLORS[i % len(FUNC_COLORS)]
    pc.set_facecolor(c)
    pc.set_alpha(0.75)
    pc.set_edgecolor("#30363d")

parts["cmedians"].set_color("#e6edf3")
parts["cmedians"].set_linewidth(1.5)

# Overlay mean dots
means = [np.mean(d) for d in cat_data]
ax_c.scatter(range(len(cat_order)), means, color="white", s=25, zorder=5)

ax_c.set_xticks(range(len(cat_order)))
ax_c.set_xticklabels(cat_labels, fontsize=7.5)
ax_c.set_yticks(range(2, 13))
ax_c.set_ylim(1.3, 13)
ax_c.axhline(df.n_subunits.mean(), color="#58a6ff", linewidth=0.8,
             linestyle="--", alpha=0.6, label=f"Overall mean ({df.n_subunits.mean():.2f})")
ax_c.legend(fontsize=8, framealpha=0.2, labelcolor="#e6edf3", facecolor="#161b22")

# ── D: Diversity ratio heatmap: n_subunits × n_unique ────────────────────────

ax_d = fig.add_subplot(gs[2, 0])
style(ax_d, "D — Subunit Identity Matrix\n(N total vs N unique)", "N unique proteins", "N total subunits")

n_range = range(1, 13)
heat = np.zeros((12, 12))
for _, row in df.iterrows():
    nt, nu = int(row.n_subunits), int(row.n_unique)
    if 1 <= nt <= 12 and 1 <= nu <= 12:
        heat[nt - 1, nu - 1] += 1

# Log scale for visibility
heat_log = np.log1p(heat)
cmap = LinearSegmentedColormap.from_list("prism", ["#161b22", "#1f3f6e", "#4e9af1", "#e6edf3"])
im = ax_d.imshow(heat_log, cmap=cmap, aspect="auto", origin="lower",
                 extent=[0.5, 12.5, 0.5, 12.5])
plt.colorbar(im, ax=ax_d, label="log(count+1)", shrink=0.8).ax.yaxis.set_tick_params(color="#8b949e")

ax_d.set_xlabel("N unique proteins", color="#8b949e", fontsize=9)
ax_d.set_ylabel("N total subunits", color="#8b949e", fontsize=9)
ax_d.plot([0.5, 12.5], [0.5, 12.5], "w--", linewidth=0.8, alpha=0.5, label="N unique = N total\n(pure heteromer)")
ax_d.legend(fontsize=7, framealpha=0.2, labelcolor="#e6edf3", facecolor="#161b22")
ax_d.set_xticks(range(1, 13))
ax_d.set_yticks(range(1, 13))

# ── E: Repeat class donut ────────────────────────────────────────────────────

ax_e = fig.add_subplot(gs[2, 1])
ax_e.set_facecolor("#161b22")
ax_e.set_title("E — Subunit Repetition Class", color="#e6edf3",
               fontsize=11, fontweight="bold", pad=8)

rc_counts = df.repeat_class.value_counts()
rc_labels = rc_counts.index.tolist()
rc_vals   = rc_counts.values
rc_colors = ["#4e9af1", "#f0883e", "#3fb950"]
wedges, texts, autotexts = ax_e.pie(
    rc_vals, labels=None, autopct="%1.1f%%", startangle=90,
    colors=rc_colors[:len(rc_labels)],
    wedgeprops=dict(width=0.55, edgecolor="#0d1117"),
    pctdistance=0.78,
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_color("#e6edf3")

ax_e.legend(
    [Patch(facecolor=rc_colors[i]) for i in range(len(rc_labels))],
    [f"{l}: {v:,}" for l, v in zip(rc_labels, rc_vals)],
    loc="lower center", fontsize=7.5, framealpha=0.2,
    labelcolor="#e6edf3", facecolor="#161b22", ncol=1,
    bbox_to_anchor=(0.5, -0.18),
)

# ── F: Mean N subunits per functional category bar ───────────────────────────

ax_f = fig.add_subplot(gs[2, 2])
style(ax_f, "F — Avg Subunits per\nFunctional Category", "Category", "Mean N subunits")

func_means = df.groupby("func_cat").n_subunits.mean().sort_values(ascending=True)
# Exclude "Other" for cleaner view
func_means = func_means[func_means.index != "Other"]
colors_f   = [FUNC_COLORS[list(FUNC_CATS.keys()).index(c) % len(FUNC_COLORS)]
              if c in FUNC_CATS else "#8b949e" for c in func_means.index]

bars = ax_f.barh(range(len(func_means)), func_means.values, color=colors_f, alpha=0.85)
ax_f.set_yticks(range(len(func_means)))
ax_f.set_yticklabels([c.replace("\n", " ") for c in func_means.index], fontsize=7.5)
ax_f.axvline(df.n_subunits.mean(), color="#58a6ff", linewidth=0.8,
             linestyle="--", alpha=0.7, label=f"Overall μ={df.n_subunits.mean():.2f}")
ax_f.legend(fontsize=7.5, framealpha=0.2, labelcolor="#e6edf3", facecolor="#161b22")
for bar, val in zip(bars, func_means.values):
    ax_f.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
              f"{val:.2f}", va="center", color="#e6edf3", fontsize=7.5)
ax_f.set_xlim(0, func_means.max() + 0.5)
ax_f.grid(axis="x", color="#30363d", linewidth=0.5)
ax_f.grid(axis="y", visible=False)
for spine in ax_f.spines.values():
    spine.set_edgecolor("#30363d")
ax_f.tick_params(colors="#8b949e", labelsize=8)

# ── Save & show ───────────────────────────────────────────────────────────────

out = "/tmp/prism_corpus_analysis.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
