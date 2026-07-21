"""
Presentation-quality EDA + Domain Analysis plots
Saves all figures to plots/ directory
"""
import pandas as pd
import numpy as np
import re
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import Counter

os.makedirs("plots", exist_ok=True)

# ── colour palette (consistent across all charts) ───────────────────────────
BLUE       = "#2563EB"
ORANGE     = "#F97316"
GREEN      = "#16A34A"
RED        = "#DC2626"
PURPLE     = "#7C3AED"
TEAL       = "#0D9488"
YELLOW     = "#CA8A04"
GREY       = "#6B7280"
LIGHT_GREY = "#E5E7EB"
BG         = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        LIGHT_GREY,
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

# ── load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/raw/final_dataset.csv")
df["am_words"] = df["amharic"].apply(lambda x: len(str(x).split()))
df["en_words"] = df["english"].apply(lambda x: len(str(x).split()))
df["ratio"]    = df["am_words"] / (df["en_words"] + 1e-6)

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

with open("data/v2/import_report.json") as f:
    report = json.load(f)
v2_df = pd.read_csv("data/v2/verified_additions.csv")

# ── domain keyword analysis ──────────────────────────────────────────────────
DOMAINS = {
    "Religious / Bible":         ["god","lord","jesus","christ","holy","spirit","prayer","faith",
                                   "church","gospel","worship","scripture","jehovah","kingdom",
                                   "heaven","sin","salvation","prophet","apostle","angel",
                                   "baptism","psalm","christian","sabbath","temple","priest",
                                   "offering","covenant"],
    "Family / Relationships":    ["father","mother","son","daughter","husband","wife","brother",
                                   "sister","family","child","children","marriage","love","friend",
                                   "parent","born","home","wed","widow","grandfather"],
    "News / Politics":           ["president","war","peace","country","nation","africa","ethiopia",
                                   "world","report","news","crisis","military","attack","people",
                                   "human","leader","force","conflict","vote","party"],
    "Daily Life / Conversational":["eat","food","drink","house","work","want","need","money","buy",
                                   "sleep","walk","talk","phone","car","road","market","cook",
                                   "wear","clothes"],
    "Legal / Government":        ["law","court","government","rights","constitution","parliament",
                                   "minister","election","policy","justice","judge","authority",
                                   "regulation","treaty","legislation","act","decree","shall"],
    "Nature / Agriculture":      ["water","land","earth","rain","river","mountain","tree","animal",
                                   "farm","crop","soil","harvest","field","seed","plant","forest",
                                   "fish","cattle","ground"],
    "Health / Medical":          ["health","disease","doctor","hospital","medicine","patient",
                                   "treatment","pain","blood","death","sick","heal","virus",
                                   "infection","body","care","died","wound"],
    "Education / Science":       ["school","student","teacher","learn","study","science","research",
                                   "knowledge","university","education","book","read","write",
                                   "technology","training","college"],
    "Software / UI":             ["click","button","file","menu","window","open","close","save",
                                   "error","cancel","settings","install","software","application",
                                   "dialog","option","select","delete","edit","toolbar","folder"],
}

DOMAIN_COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, YELLOW, GREY, "#EC4899"]

print("Running domain analysis...")
en_texts = df["english"].dropna().str.lower().tolist()
domain_counts = Counter()
for text in en_texts:
    words = set(re.findall(r"\b\w+\b", text))
    for domain, kws in DOMAINS.items():
        if words & set(kws):
            domain_counts[domain] += 1

domain_labels  = [d for d, _ in domain_counts.most_common()]
domain_values  = [c for _, c in domain_counts.most_common()]
domain_pcts    = [v / len(df) * 100 for v in domain_values]

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Corpus overview  (2×2 grid)
# ════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 1: Corpus Overview...")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Amharic-English NMT — Corpus Overview", fontsize=18, fontweight="bold", y=1.01)
fig.patch.set_facecolor(BG)

# ── 1a: Data source composition (pie) ───────────────────────────────────────
ax = axes[0, 0]
src_labels = ["OPUS-100\n(web-crawled)", "Bible\nparallel", "AmharicDataset\n(general)",
              "FLORES\nbenchmark", "Rush Hour\nsubtitles"]
src_sizes  = [200000, 30000, 5000, 1000, 1009]
src_colors = [BLUE, ORANGE, GREEN, PURPLE, TEAL]
wedges, texts, autotexts = ax.pie(
    src_sizes, labels=src_labels, colors=src_colors,
    autopct="%1.1f%%", startangle=140,
    pctdistance=0.78, textprops={"fontsize": 9},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")
    at.set_color("white")
ax.set_title("Data Source Composition\n(~229K total pairs)", fontweight="bold", pad=12)
ax.set_facecolor(BG)

# ── 1b: Split size bar chart ─────────────────────────────────────────────────
ax = axes[0, 1]
split_names  = ["Train", "Validation", "Test"]
split_sizes  = [217996, 5737, 5737]
split_colors = [BLUE, ORANGE, GREEN]
bars = ax.bar(split_names, split_sizes, color=split_colors, width=0.5, edgecolor="white", linewidth=1.2)
for bar, size in zip(bars, split_sizes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
            f"{size:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    pct = size / sum(split_sizes) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f"{pct:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
ax.set_title("Train / Val / Test Split Sizes", fontweight="bold")
ax.set_ylabel("Number of Sentence Pairs")
ax.set_ylim(0, 240000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", visible=False)

# ── 1c: Sentence length distribution (overlapping histograms) ────────────────
ax = axes[1, 0]
bins = range(0, 55, 2)
ax.hist(df["en_words"].clip(upper=54), bins=bins, color=ORANGE, alpha=0.75, label="English", edgecolor="white", linewidth=0.4)
ax.hist(df["am_words"].clip(upper=54), bins=bins, color=BLUE,   alpha=0.75, label="Amharic", edgecolor="white", linewidth=0.4)
ax.axvline(df["en_words"].median(), color=ORANGE, linestyle="--", linewidth=1.8, label=f"EN median={int(df['en_words'].median())}")
ax.axvline(df["am_words"].median(), color=BLUE,   linestyle="--", linewidth=1.8, label=f"AM median={int(df['am_words'].median())}")
ax.set_title("Sentence Length Distribution (words)", fontweight="bold")
ax.set_xlabel("Words per Sentence  (clipped at 54)")
ax.set_ylabel("Number of Sentences")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)

# ── 1d: Length ratio histogram ───────────────────────────────────────────────
ax = axes[1, 1]
ratio_clipped = df["ratio"].clip(0.1, 2.5)
ax.hist(ratio_clipped, bins=60, color=TEAL, edgecolor="white", linewidth=0.4)
ax.axvline(df["ratio"].mean(),   color=RED,    linestyle="--", linewidth=2, label=f"Mean={df['ratio'].mean():.2f}")
ax.axvline(df["ratio"].median(), color=YELLOW, linestyle="--", linewidth=2, label=f"Median={df['ratio'].median():.2f}")
ax.set_title("Amharic/English Word-Count Ratio", fontweight="bold")
ax.set_xlabel("Ratio  (clipped 0.1–2.5)")
ax.set_ylabel("Number of Sentences")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("plots/fig1_corpus_overview.png", dpi=180, bbox_inches="tight")
plt.close()
print("  ✓ plots/fig1_corpus_overview.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Domain distribution
# ════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 2: Domain Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Corpus Domain Distribution", fontsize=18, fontweight="bold")
fig.patch.set_facecolor(BG)

# ── 2a: Horizontal bar chart ─────────────────────────────────────────────────
ax = axes[0]
y_pos = range(len(domain_labels))
bars  = ax.barh(list(y_pos), domain_pcts, color=DOMAIN_COLORS, edgecolor="white",
                linewidth=1.0, height=0.65)
for bar, pct, val in zip(bars, domain_pcts, domain_values):
    ax.text(pct + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:,}  ({pct:.1f}%)", va="center", fontsize=9.5, fontweight="bold")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(domain_labels, fontsize=10)
ax.set_xlabel("% of corpus sentences  (keyword-based, multi-label)")
ax.set_title("Domains by Frequency", fontweight="bold")
ax.set_xlim(0, 42)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)

# ── 2b: Donut chart ──────────────────────────────────────────────────────────
ax = axes[1]
all_matched = set()
for i, text in enumerate(en_texts):
    words = set(re.findall(r"\b\w+\b", text))
    for kws in DOMAINS.values():
        if words & set(kws):
            all_matched.add(i)
            break
unmatched = len(df) - len(all_matched)

pie_labels = domain_labels + ["Other / Mixed"]
pie_values = domain_values + [unmatched]
pie_colors = DOMAIN_COLORS + [LIGHT_GREY]

wedges, texts, autotexts = ax.pie(
    pie_values, labels=None, colors=pie_colors,
    autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
    startangle=90, pctdistance=0.82,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5, "width": 0.55},
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight("bold")
    at.set_color("white")

ax.legend(wedges, pie_labels, loc="lower left", bbox_to_anchor=(-0.15, -0.15),
          fontsize=8.5, framealpha=0.6, ncol=2)
ax.set_title("Domain Share (donut)", fontweight="bold")

plt.tight_layout()
plt.savefig("plots/fig2_domain_distribution.png", dpi=180, bbox_inches="tight")
plt.close()
print("  ✓ plots/fig2_domain_distribution.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Data quality checks
# ════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 3: Data Quality...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Data Quality Analysis", fontsize=18, fontweight="bold")
fig.patch.set_facecolor(BG)

# ── 3a: Ge'ez purity histogram ───────────────────────────────────────────────
ax = axes[0]
def geez_purity(text):
    if not isinstance(text, str): return 0.0
    geez = len(re.findall(r"[\u1200-\u137F]", text))
    total = len(text.replace(" ", ""))
    return geez / total if total > 0 else 0.0

print("  Computing Ge'ez purity (may take ~10s)...")
purity = df["amharic"].apply(geez_purity)
ax.hist(purity, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
ax.axvline(0.5, color=RED, linestyle="--", linewidth=2, label="50% threshold")
low_purity = (purity < 0.5).sum()
ax.text(0.05, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 1000,
        f"Below 50%:\n{low_purity:,} rows\n({low_purity/len(df)*100:.2f}%)",
        fontsize=9, color=RED, fontweight="bold")
ax.set_title("Ge'ez Script Purity\n(Amharic column)", fontweight="bold")
ax.set_xlabel("Fraction of chars that are Ge'ez")
ax.set_ylabel("Number of Sentences")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)

# ── 3b: Quality issue breakdown (stacked bar) ────────────────────────────────
ax = axes[1]
issues = {
    "Very short\nAmharic (<3w)": (df["am_words"] < 3).sum(),
    "Very short\nEnglish (<3w)": (df["en_words"] < 3).sum(),
    "Very long\nAmharic (>50w)": (df["am_words"] > 50).sum(),
    "Very long\nEnglish (>50w)": (df["en_words"] > 50).sum(),
    "Extreme\nlength ratio":     ((df["ratio"] < 0.3) | (df["ratio"] > 3.0)).sum(),
}
issue_colors = [BLUE, ORANGE, GREEN, RED, PURPLE]
bars = ax.bar(list(issues.keys()), list(issues.values()),
              color=issue_colors, edgecolor="white", linewidth=1.0, width=0.6)
for bar, val in zip(bars, issues.values()):
    pct = val / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f"{val:,}\n({pct:.2f}%)", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold")
ax.set_title("Potential Quality Issues\n(out of 229,472 pairs)", fontweight="bold")
ax.set_ylabel("Number of Sentences")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="x", visible=False)

# ── 3c: Word count box plot comparison ───────────────────────────────────────
ax = axes[2]
bp_data  = [df["am_words"].clip(upper=60), df["en_words"].clip(upper=60)]
bp_colors = [BLUE, ORANGE]
bp = ax.boxplot(bp_data, patch_artist=True, widths=0.45,
                medianprops={"color":"white","linewidth":2.5},
                whiskerprops={"linewidth":1.5},
                capprops={"linewidth":1.5},
                flierprops={"marker":"o","markersize":2,"alpha":0.3})
for patch, color in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.set_xticks([1, 2])
ax.set_xticklabels(["Amharic", "English"], fontsize=12)
ax.set_title("Word Count Distribution\n(clipped at 60, shows IQR)", fontweight="bold")
ax.set_ylabel("Words per Sentence")

plt.tight_layout()
plt.savefig("plots/fig3_data_quality.png", dpi=180, bbox_inches="tight")
plt.close()
print("  ✓ plots/fig3_data_quality.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — V2 candidate data analysis
# ════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 4: V2 Candidate Data...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("V2 Data Expansion — Import Analysis", fontsize=18, fontweight="bold")
fig.patch.set_facecolor(BG)

# ── 4a: Accepted vs rejected per source ──────────────────────────────────────
ax = axes[0]
sources      = list(report["sources"].keys())
total_import = [report["sources"][s] for s in sources]
accepted     = [report["accepted_by_source"][s] for s in sources]
rejected     = [report["sources"][s] - report["accepted_by_source"][s] for s in sources]
x = np.arange(len(sources))
w = 0.38
ax.bar(x - w/2, rejected, w, label="Rejected", color=RED,   edgecolor="white", linewidth=1)
ax.bar(x + w/2, accepted, w, label="Accepted", color=GREEN, edgecolor="white", linewidth=1)
for i, (rej, acc) in enumerate(zip(rejected, accepted)):
    acc_rate = acc / (rej + acc) * 100
    ax.text(i, max(rej, acc) + 200, f"{acc_rate:.0f}%\nacc.", ha="center", fontsize=8.5, fontweight="bold", color=GREEN)
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in sources], fontsize=10)
ax.set_title("Accepted vs Rejected\nby Source", fontweight="bold")
ax.set_ylabel("Sentence Pairs")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=10)
ax.grid(axis="x", visible=False)

# ── 4b: Rejection reason breakdown ───────────────────────────────────────────
ax = axes[1]
rej_reasons = report["rejections_by_reason"]
rej_labels  = [r.replace("_", "\n") for r in rej_reasons.keys()]
rej_values  = list(rej_reasons.values())
rej_colors  = [RED, ORANGE, PURPLE, BLUE, GREEN, TEAL, YELLOW, GREY][:len(rej_labels)]
total_rej   = sum(rej_values)
bars = ax.barh(rej_labels, rej_values, color=rej_colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, rej_values):
    ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
            f"{val:,}  ({val/total_rej*100:.1f}%)", va="center", fontsize=8.5)
ax.set_title("Rejection Reasons\n(55,341 total rejected)", fontweight="bold")
ax.set_xlabel("Count")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(0, max(rej_values) * 1.35)

# ── 4c: Accepted source composition (pie) ────────────────────────────────────
ax = axes[2]
src_colors_v2 = [BLUE, ORANGE, GREEN, PURPLE]
wedges, texts, autotexts = ax.pie(
    accepted,
    labels=[f"{s.capitalize()}\n{a:,}" for s, a in zip(sources, accepted)],
    colors=src_colors_v2,
    autopct="%1.1f%%", startangle=120,
    pctdistance=0.75, textprops={"fontsize": 9},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_fontweight("bold")
    at.set_color("white")
ax.set_title(f"Accepted Pairs by Source\n(7,600 total)", fontweight="bold")
ax.set_facecolor(BG)

plt.tight_layout()
plt.savefig("plots/fig4_v2_analysis.png", dpi=180, bbox_inches="tight")
plt.close()
print("  ✓ plots/fig4_v2_analysis.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Tokenizer statistics
# ════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 5: Tokenizer Stats...")
try:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load("data/processed/am_en_bpe.model")

    # Read vocab for composition analysis
    vocab_tokens = []
    with open("data/processed/am_en_bpe.vocab", "r", encoding="utf-8") as f:
        for line in f:
            vocab_tokens.append(line.split("\t")[0])

    geez_t  = sum(1 for t in vocab_tokens if re.search(r"[\u1200-\u137F]", t))
    latin_t = sum(1 for t in vocab_tokens if re.search(r"[a-zA-Z]", t) and not re.search(r"[\u1200-\u137F]", t))
    digit_t = sum(1 for t in vocab_tokens if re.search(r"\d", t))
    other_t = len(vocab_tokens) - geez_t - latin_t - digit_t

    # Token lengths on 5000 sample pairs
    sample = train_df.sample(5000, random_state=42)
    am_tl, en_tl = [], []
    for _, row in sample.iterrows():
        am_tl.append(len(sp.encode_as_ids(str(row["amharic"]))))
        en_tl.append(len(sp.encode_as_ids(str(row["english"]))))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Tokenizer Analysis — SentencePiece BPE (32K vocab)", fontsize=18, fontweight="bold")
    fig.patch.set_facecolor(BG)

    # ── 5a: Vocab composition ────────────────────────────────────────────────
    ax = axes[0]
    voc_labels = ["Ge'ez script\ntokens", "Latin script\ntokens", "Digit\ntokens", "Other /\nspecial"]
    voc_values = [geez_t, latin_t, digit_t, other_t]
    voc_colors = [BLUE, ORANGE, GREEN, GREY]
    wedges, texts, autotexts = ax.pie(
        voc_values, labels=voc_labels, colors=voc_colors,
        autopct="%1.1f%%", startangle=100,
        pctdistance=0.78, textprops={"fontsize": 9.5},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Vocabulary Composition\n(32,000 tokens)", fontweight="bold")
    ax.set_facecolor(BG)

    # ── 5b: Token length distribution ────────────────────────────────────────
    ax = axes[1]
    bins = range(0, 80, 3)
    ax.hist(en_tl, bins=bins, color=ORANGE, alpha=0.75, label="English", edgecolor="white", linewidth=0.4)
    ax.hist(am_tl, bins=bins, color=BLUE,   alpha=0.75, label="Amharic", edgecolor="white", linewidth=0.4)
    ax.axvline(128, color=RED, linestyle="--", linewidth=2, label="MAX_LENGTH=128")
    ax.axvline(np.median(en_tl), color=ORANGE, linestyle=":", linewidth=1.8, label=f"EN median={int(np.median(en_tl))}")
    ax.axvline(np.median(am_tl), color=BLUE,   linestyle=":", linewidth=1.8, label=f"AM median={int(np.median(am_tl))}")
    ax.set_title("Token Length Distribution\n(5,000 training samples)", fontweight="bold")
    ax.set_xlabel("Tokens per Sentence")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── 5c: Words → tokens expansion ratio ────────────────────────────────────
    ax = axes[2]
    am_expansion = [t / w if w > 0 else 0 for t, w in
                    zip(am_tl, sample["amharic"].apply(lambda x: len(str(x).split())))]
    en_expansion = [t / w if w > 0 else 0 for t, w in
                    zip(en_tl, sample["english"].apply(lambda x: len(str(x).split())))]
    ax.hist(en_expansion, bins=40, range=(0, 4), color=ORANGE, alpha=0.75, label="English", edgecolor="white", linewidth=0.4)
    ax.hist(am_expansion, bins=40, range=(0, 4), color=BLUE,   alpha=0.75, label="Amharic", edgecolor="white", linewidth=0.4)
    ax.axvline(np.mean(en_expansion), color=ORANGE, linestyle="--", linewidth=2, label=f"EN mean={np.mean(en_expansion):.2f}")
    ax.axvline(np.mean(am_expansion), color=BLUE,   linestyle="--", linewidth=2, label=f"AM mean={np.mean(am_expansion):.2f}")
    ax.set_title("Tokens per Word (Fertility)\n(5,000 training samples)", fontweight="bold")
    ax.set_xlabel("Tokens / Word")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig("plots/fig5_tokenizer.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  ✓ plots/fig5_tokenizer.png")

except ImportError:
    print("  ⚠ sentencepiece not installed — skipping Figure 5")
except Exception as e:
    print(f"  ⚠ Tokenizer figure error: {e}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Single-slide summary (for title slide / poster)
# ════════════════════════════════════════════════════════════════════════════
print("Plotting Figure 6: Summary Dashboard...")
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("#1E293B")   # dark background for impact slide

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

# ── KPI tiles (top row) ───────────────────────────────────────────────────────
kpis = [
    ("229,472", "Total Sentence Pairs"),
    ("5 Sources", "Bible · OPUS · FLORES\nGlobalVoices · Subtitles"),
    ("32,000", "BPE Vocabulary Size\n(63% Ge'ez tokens)"),
    ("99.99%", "Ge'ez Script Coverage"),
]
for i, (value, label) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor("#334155")
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=22,
            fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=10,
            color="#94A3B8", transform=ax.transAxes, multialignment="center")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
        spine.set_linewidth(1.5)
    ax.set_xticks([]); ax.set_yticks([])

# ── Bottom-left: Domain bar chart ────────────────────────────────────────────
ax_dom = fig.add_subplot(gs[1, :2])
ax_dom.set_facecolor("#1E293B")
ax_dom.spines["bottom"].set_color("#475569")
ax_dom.spines["left"].set_color("#475569")
ax_dom.spines["top"].set_visible(False)
ax_dom.spines["right"].set_visible(False)
ax_dom.grid(color="#334155", linewidth=0.8)
y_pos = range(len(domain_labels))
bars = ax_dom.barh(list(y_pos), domain_pcts, color=DOMAIN_COLORS,
                   edgecolor="#1E293B", linewidth=0.8, height=0.6)
for bar, pct in zip(bars, domain_pcts):
    ax_dom.text(pct + 0.3, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=9, color="white", fontweight="bold")
ax_dom.set_yticks(list(y_pos))
ax_dom.set_yticklabels(domain_labels, fontsize=9, color="white")
ax_dom.set_xlabel("% of corpus", color="#94A3B8", fontsize=10)
ax_dom.set_title("Domain Distribution", color="white", fontweight="bold", fontsize=12, pad=8)
ax_dom.invert_yaxis()
ax_dom.tick_params(colors="#94A3B8")
ax_dom.set_xlim(0, 40)

# ── Bottom-right: Split + sentence length side by side ───────────────────────
ax_sp = fig.add_subplot(gs[1, 2])
ax_sp.set_facecolor("#1E293B")
for spine in ax_sp.spines.values():
    spine.set_color("#475569")
ax_sp.spines["top"].set_visible(False)
ax_sp.spines["right"].set_visible(False)
ax_sp.grid(color="#334155", linewidth=0.8)
split_pcts = [95.0, 2.5, 2.5]
bars2 = ax_sp.bar(["Train", "Val", "Test"], split_pcts,
                  color=[BLUE, ORANGE, GREEN], edgecolor="#1E293B", linewidth=0.8)
for bar, pct, size in zip(bars2, split_pcts, [217996, 5737, 5737]):
    ax_sp.text(bar.get_x() + bar.get_width()/2, pct + 1,
               f"{size:,}", ha="center", color="white", fontsize=9, fontweight="bold")
ax_sp.set_title("Split Sizes", color="white", fontweight="bold", fontsize=12, pad=8)
ax_sp.set_ylabel("% of corpus", color="#94A3B8")
ax_sp.tick_params(colors="#94A3B8")
ax_sp.set_ylim(0, 105)

ax_len = fig.add_subplot(gs[1, 3])
ax_len.set_facecolor("#1E293B")
for spine in ax_len.spines.values():
    spine.set_color("#475569")
ax_len.spines["top"].set_visible(False)
ax_len.spines["right"].set_visible(False)
ax_len.grid(color="#334155", linewidth=0.8)
stats = {"Mean": [14.17, 20.12], "Median": [12, 18], "Max": [94, 100]}
x2 = np.arange(len(stats))
w2 = 0.32
ax_len.bar(x2 - w2/2, [v[0] for v in stats.values()], w2, color=BLUE,   label="Amharic", edgecolor="#1E293B")
ax_len.bar(x2 + w2/2, [v[1] for v in stats.values()], w2, color=ORANGE, label="English", edgecolor="#1E293B")
ax_len.set_xticks(x2)
ax_len.set_xticklabels(list(stats.keys()), color="#94A3B8")
ax_len.set_title("Sentence Length (words)", color="white", fontweight="bold", fontsize=12, pad=8)
ax_len.legend(fontsize=9, facecolor="#334155", labelcolor="white", edgecolor="#475569")
ax_len.tick_params(colors="#94A3B8")

plt.suptitle("Amharic-English NMT — Data at a Glance",
             fontsize=20, fontweight="bold", color="white", y=1.01)
plt.savefig("plots/fig6_summary_dashboard.png", dpi=180, bbox_inches="tight",
            facecolor="#1E293B")
plt.close()
print("  ✓ plots/fig6_summary_dashboard.png")

# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("All figures saved to  plots/")
print("="*60)
files = sorted(os.listdir("plots"))
for f in files:
    size_kb = os.path.getsize(f"plots/{f}") // 1024
    print(f"  {f}  ({size_kb} KB)")
