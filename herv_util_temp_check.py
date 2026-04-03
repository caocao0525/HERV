# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple
import tempfile
import os
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import gseapy as gp

# Import ChromBERT utilities
import css_utility as crb

# %%
# essential data
LTR="../database/LTR/hg38_RepeatMasker_LTR.bed"

# Liver Cell Line (E066 from ROADMAP)
liver_cell = "../database/ROADMAP/hg38_mnemonics/unzipped/E066_15_coreMarks_hg38lift_mnemonics.bed"
# HepG2 Hepatocellular Carcinoma Cell Line (E118 from ROADMAP)
hepg2_cell = "../database/ROADMAP/hg38_mnemonics/unzipped/E118_15_coreMarks_hg38lift_mnemonics.bed"

# Lung Cell Line (E096 from ROADMAP)
lung_cell = "../database/ROADMAP/hg38_mnemonics/unzipped/E096_15_coreMarks_hg38lift_mnemonics.bed"
# A549 Lung Carcinoma Cell Line (E114 from ROADMAP)
a549_cell = "../database/ROADMAP/hg38_mnemonics/unzipped/E114_15_coreMarks_hg38lift_mnemonics.bed"

# Monocyte Cell Line (E029 from ROADMAP)
monocyte_cell = "../database/ROADMAP/hg38_mnemonics/unzipped/E029_15_coreMarks_hg38lift_mnemonics.bed"
# K562 Leukemia Cell Line (E123 from ROADMAP)
k562_cell = "../database/ROADMAP/hg38_mnemonics/unzipped/E123_15_coreMarks_hg38lift_mnemonics.bed"

# Gene annotation file (GENCODE v40)
gene_annotation = "../database/refGene.txt"

# %%
def load_refgene_full(gene_annotation):
    gene_df = pd.read_csv(
        gene_annotation,
        sep="\t",
        header=None,
        names=[
            "bin","name","chrom","strand",
            "txStart","txEnd",
            "cdsStart","cdsEnd",
            "exonCount","exonStarts","exonEnds",
            "score","name2","cdsStartStat","cdsEndStat","exonFrames"
        ]
    )

    # cleanup
    gene_df["txStart"] = pd.to_numeric(gene_df["txStart"], errors="coerce")
    gene_df["txEnd"] = pd.to_numeric(gene_df["txEnd"], errors="coerce")
    gene_df = gene_df.dropna(subset=["txStart","txEnd"]).copy()

    gene_df["txStart"] = gene_df["txStart"].astype(int)
    gene_df["txEnd"] = gene_df["txEnd"].astype(int)

    # add TSS
    gene_df["tss"] = np.where(
        gene_df["strand"] == "+",
        gene_df["txStart"],
        gene_df["txEnd"]
    )

    gene_df["gene_name"] = gene_df["name2"]

    return gene_df

# %%
gene_df=load_refgene_full(
    gene_annotation,
)
gene_df.head(10)

# %%
def make_ltr(inp="../database/LTR/hg38_RepeatMasker_LTR.bed",
                   out="./hg38_RepeatMasker_LTR_pm1kb.bed", dist=1000):
    
    # Read as 6-column BED
    df_ltr = pd.read_csv(
        inp,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
        comment="#",
        engine="python"
    )

    # Keep only real BED rows
    df_ltr = df_ltr[df_ltr["chrom"].astype(str).str.startswith("chr")].copy()

    # Ensure numeric types
    df_ltr["start"] = pd.to_numeric(df_ltr["start"], errors="coerce")
    df_ltr["end"]   = pd.to_numeric(df_ltr["end"], errors="coerce")
    df_ltr = df_ltr.dropna(subset=["start", "end"])

    df_ltr["start"] = df_ltr["start"].astype(int)
    df_ltr["end"]   = df_ltr["end"].astype(int)

    # Expand ±1 kb
    df_ltr["start"] = (df_ltr["start"] - dist).clip(lower=0)
    df_ltr["end"]   = df_ltr["end"] + dist

    # Save BED
    df_ltr.to_csv(out, sep="\t", header=False, index=False)

    return df_ltr

# %%
df_ltr = make_ltr(
    inp=LTR,
    out="hg38_RepeatMasker_LTR_pm1kb.bed",
    dist=1000
)

# %%
df_ltr.head()

# %%
def make_ltr_df_on_celltype_from_mnemonics(
    mnemonics_bed,
    ltr_df,
    canonical_chroms=None,
    chrom_order=None,
    bin_size=200
):
    if canonical_chroms is None:
        canonical_chroms = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

    if chrom_order is None:
        chrom_order = {f"chr{i}": i for i in range(1, 23)}
        chrom_order.update({"chrX": 23, "chrY": 24})

    # 1. read original mnemonics BED
    df_raw = pd.read_csv(
        mnemonics_bed,
        sep="\t",
        header=None,
        names=["chromosome", "start", "end", "state"],
        usecols=[0, 1, 2, 3]
    )

    # 2. keep canonical chromosomes
    df_raw = df_raw[df_raw["chromosome"].isin(canonical_chroms)].copy()

    # 3. numeric cleanup
    df_raw["start"] = pd.to_numeric(df_raw["start"], errors="coerce")
    df_raw["end"] = pd.to_numeric(df_raw["end"], errors="coerce")
    df_raw = df_raw.dropna(subset=["start", "end"]).copy()
    df_raw["start"] = df_raw["start"].astype(int)
    df_raw["end"] = df_raw["end"].astype(int)

    # 4. sort in Python
    df_raw["chrom_order"] = df_raw["chromosome"].map(chrom_order)
    df_raw = (
        df_raw
        .sort_values(["chrom_order", "start", "end"])
        .drop(columns="chrom_order")
        .reset_index(drop=True)
    )

    # 5. convert mnemonic -> state number only
    #    e.g. 15_Quies -> 15
    df_raw["state"] = df_raw["state"].astype(str).str.replace(r"_.*", "", regex=True)

    # 6. write temporary stateno BED for crb.bed2df_expanded()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".stateno.bed", delete=False) as tmp:
        tmp_path = tmp.name
        df_raw.to_csv(tmp_path, sep="\t", header=False, index=False)

    try:
        # 7. use YOUR function here
        df_state = crb.bed2df_expanded(tmp_path)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 8. keep canonical chromosomes and sort
    df_state = (
        df_state[df_state["chromosome"].isin(canonical_chroms)]
        .assign(chrom_order=lambda x: x["chromosome"].map(chrom_order))
        .sort_values(["chrom_order", "start", "end"])
        .drop(columns="chrom_order")
        .reset_index(drop=True)
    )

    # 9. build chromosome-wide state strings from state_seq_full
    chr_seq_map = (
        df_state.groupby("chromosome", sort=False)["state_seq_full"]
        .apply("".join)
        .to_dict()
    )

    # 10. slice onto LTR windows
    def slice_chr_seq(chrom, start_bp, end_bp):
        seq = chr_seq_map.get(chrom)
        if seq is None:
            return None

        bin_start = start_bp // bin_size
        bin_end = (end_bp - 1) // bin_size + 1
        return seq[bin_start:bin_end]

    df_ltr_on_cell = ltr_df.copy()
    df_ltr_on_cell = df_ltr_on_cell[df_ltr_on_cell["chrom"].isin(canonical_chroms)].copy()

    df_ltr_on_cell["length"] = df_ltr_on_cell["end"] - df_ltr_on_cell["start"]

    df_ltr_on_cell["state_seq_full_slice"] = df_ltr_on_cell.apply(
        lambda r: slice_chr_seq(r["chrom"], int(r["start"]), int(r["end"])),
        axis=1
    )

    df_ltr_on_cell["n_bins"] = df_ltr_on_cell["state_seq_full_slice"].str.len()

    df_ltr_on_cell["chrom_order"] = df_ltr_on_cell["chrom"].map(chrom_order)

    df_ltr_on_cell = (
        df_ltr_on_cell
        .sort_values(["chrom_order", "start", "end"])
        .drop(columns="chrom_order")
        .reset_index(drop=True)
    )

    df_ltr_on_cell = df_ltr_on_cell[df_ltr_on_cell["n_bins"] > 0]

    return df_ltr_on_cell

# %%
df_ltr_E066 = make_ltr_df_on_celltype_from_mnemonics(
    liver_cell,
    df_ltr
)

# %%
df_ltr_E066

# %%
df_ltr_E118 = make_ltr_df_on_celltype_from_mnemonics(
    hepg2_cell,
    df_ltr
)
df_ltr_E118

# %%

# rewitten, exitsing below (tidy later)
def count_transitions(seq):
    if not isinstance(seq, str) or len(seq) == 0:
        return 0
    return sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))

    df_ltr_trans["n_transitions"] = df_ltr_trans["state_seq_full_slice"].apply(count_transitions)

    # transition density per kb
    df_ltr_trans = df_ltr_trans[df_ltr_trans["length"] >= min_length_bp].copy()
    df_ltr_trans["transition_density_per_kb"] = (
        df_ltr_trans["n_transitions"] / (df_ltr_trans["length"] / 1000)
    )
        
    return df_ltr_trans

# %%
##### test overall LTR loci in normal cell vs. cancer cell
df_ltr_E066 = df_ltr_E066[
    df_ltr_E066["state_seq_full_slice"].apply(
        lambda s: isinstance(s, str) and set(s) != {"O"}
    )
].copy()
df_ltr_E118 = df_ltr_E118[
    df_ltr_E118["state_seq_full_slice"].apply(
        lambda s: isinstance(s, str) and set(s) != {"O"}
    )
].copy()

# %%
df_merged = df_ltr_E066.merge(
    df_ltr_E118,
    on=["chrom", "start", "end", "name"],
    suffixes=("_normal", "_cancer")
)

# %%
df_merged

# %%
def frac_of(states, counter):
    return sum(counter.get(s, 0) for s in states)

# %%
def get_frac(seq):
    c = Counter(seq)
    total = sum(c.values())
    return {k: v/total for k, v in c.items()}

# %%
bivalent = {"J","K","L"}
active = {"A","B","C","D","E","F","G"}
repressed = {"H","I","M","N","O"}

# %%
df_merged["frac_normal"] = df_merged["state_seq_full_slice_normal"].apply(get_frac)
df_merged["frac_cancer"] = df_merged["state_seq_full_slice_cancer"].apply(get_frac)

df_merged["biv_normal"] = df_merged["frac_normal"].apply(lambda x: frac_of(bivalent, x))
df_merged["biv_cancer"] = df_merged["frac_cancer"].apply(lambda x: frac_of(bivalent, x))

# %%
def classify(row):
    n = row["biv_normal"]
    c = row["biv_cancer"]

    if n < 0.2 and c > 0.4:
        return "gain_bivalent"
    elif n > 0.4 and c < 0.2:
        return "loss_bivalent"
    elif n > 0.4 and c > 0.4:
        return "stable_bivalent"
    else:
        return "other"



# %%


# %%
df_merged["switch_type"] = df_merged.apply(classify, axis=1)
df_merged["switch_type"].value_counts(normalize=True)

# %%
df_merged[["biv_normal","biv_cancer"]].describe()

# %%
df_merged

# %%
#### Function 1: assign category
def assign_state_category(frac_dict,
                          active_states={"A","B","C","D","E","F","G"},
                          bivalent_states={"J","K","L"},
                          repressed_states={"H","I","M","N","O"},
                          min_dominance=0.5):
    """
    Assign one coarse chromatin category to a locus based on state fractions.

    Returns one of:
    - 'active'
    - 'bivalent'
    - 'repressed'
    - 'mixed'
    """

    active_frac = sum(frac_dict.get(s, 0) for s in active_states)
    bivalent_frac = sum(frac_dict.get(s, 0) for s in bivalent_states)
    repressed_frac = sum(frac_dict.get(s, 0) for s in repressed_states)

    category_fracs = {
        "active": active_frac,
        "bivalent": bivalent_frac,
        "repressed": repressed_frac
    }

    top_cat = max(category_fracs, key=category_fracs.get)
    top_val = category_fracs[top_cat]

    if top_val < min_dominance:
        return "mixed"
    return top_cat

# %%
# Function 2: add normal/cancer categories + transition label
def add_coarse_state_transitions(df_merged, min_dominance=0.5):
    """
    Add coarse chromatin-state category labels and transition labels
    for normal vs cancer.
    """

    df_out = df_merged.copy()

    df_out["cat_normal"] = df_out["frac_normal"].apply(
        lambda x: assign_state_category(x, min_dominance=min_dominance)
    )
    df_out["cat_cancer"] = df_out["frac_cancer"].apply(
        lambda x: assign_state_category(x, min_dominance=min_dominance)
    )

    df_out["cat_transition"] = (
        df_out["cat_normal"] + " -> " + df_out["cat_cancer"]
    )

    return df_out

# %%
df_merged = add_coarse_state_transitions(df_merged, min_dominance=0.5)

# %%
df_merged

# %%
df_merged["cat_transition"].value_counts()

# %%
df_merged["cat_transition"].value_counts(normalize=True)

# %%
def plot_transition_heatmap(
    df,
    transition_col="cat_transition",
    order=("active", "repressed", "bivalent", "mixed"),
    figsize=(6,5),
    cmap="Blues",
    title="Chromatin state transitions (LTR)"
):
    """
    Plot chromatin state transition heatmap (%).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column like 'cat_transition' with format "A -> B"

    transition_col : str
        Column name containing transition labels

    order : tuple/list
        Order of categories for rows and columns

    figsize : tuple
        Figure size

    cmap : str
        Colormap for heatmap

    title : str
        Plot title
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # copy
    df_tmp = df.copy()

    # split transitions
    df_tmp[["normal", "cancer"]] = df_tmp[transition_col].str.split(" -> ", expand=True)

    # create matrix
    mat = pd.crosstab(df_tmp["normal"], df_tmp["cancer"], normalize="all")

    # reorder
    mat = mat.reindex(index=order, columns=order)

    # convert to %
    mat_pct = mat * 100

    # plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        mat_pct,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% of LTRs"}
    )

    plt.title(title)
    plt.xlabel("Cancer")
    plt.ylabel("Normal")

    plt.tight_layout()
    plt.show()

    return mat_pct

# %%
mat_pct = plot_transition_heatmap(df_merged, cmap="Blues")

# %%
def plot_transition_heatmap_by_row_with_counts(
    df,
    transition_col="cat_transition",
    order=("active", "repressed", "bivalent", "mixed"),
    figsize=(6, 5),
    cmap="Blues",
    title="Chromatin state transitions within each normal-state group"
):
    """
    Row-normalized heatmap with BOTH percentage and absolute counts.
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_tmp = df.copy()
    df_tmp[["normal", "cancer"]] = df_tmp[transition_col].str.split(" -> ", expand=True)

    # absolute counts
    count_mat = pd.crosstab(df_tmp["normal"], df_tmp["cancer"])
    count_mat = count_mat.reindex(index=order, columns=order).fillna(0)

    # row-normalized (%)
    frac_mat = pd.crosstab(df_tmp["normal"], df_tmp["cancer"], normalize="index")
    frac_mat = frac_mat.reindex(index=order, columns=order).fillna(0)

    pct_mat = frac_mat * 100

    # build annotation labels
    annot = pct_mat.copy().astype(str)

    for i in pct_mat.index:
        for j in pct_mat.columns:
            pct = pct_mat.loc[i, j]
            cnt = int(count_mat.loc[i, j])
            annot.loc[i, j] = f"{pct:.1f}%\n({cnt})"

    # plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        pct_mat,
        annot=annot,
        fmt="",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% within normal-state group"}
    )

    plt.title(title)
    plt.xlabel("Cancer")
    plt.ylabel("Normal")

    plt.tight_layout()
    plt.show()

    return pct_mat, count_mat


# %%
mat_row_pct_liver = plot_transition_heatmap_by_row_with_counts(df_merged)

# %%
df_merged

# %%
def annotate_bivalent_ltrs_with_nearest_tss(
    df_ltr,
    gene_df,
    bivalent_col="biv_normal",
    threshold=0.2,
    gene_name_col="gene_name"
):
    """
    Select LTR loci with substantial bivalent signal and annotate them with nearest gene TSS.

    Parameters
    ----------
    df_ltr : pd.DataFrame
        DataFrame containing LTR loci. Expected columns:
        - chrom
        - start
        - end
        - one bivalent fraction column such as 'biv_normal' or 'biv_cancer'

    gene_df : pd.DataFrame
        Gene annotation DataFrame. Expected columns:
        - chrom
        - strand
        - tss
        - gene_name_col

    bivalent_col : str
        Which bivalent fraction column to use for filtering.
        Example: 'biv_normal' or 'biv_cancer'

    threshold : float
        Minimum bivalent fraction required to keep a locus.
        Example: 0.2

    gene_name_col : str
        Column in gene_df containing gene symbols/names.

    Returns
    -------
    bivalent_annot : pd.DataFrame
        Filtered LTR dataframe annotated with nearest gene and TSS distance.
    """

    import numpy as np
    import pandas as pd

    # 1. filter loci by bivalent threshold
    bivalent_annot = df_ltr[df_ltr[bivalent_col] >= threshold].copy()

    # 2. keep only needed gene columns
    gene_use = gene_df[["chrom", "strand", "tss", gene_name_col]].copy()
    gene_use = gene_use.dropna(subset=["chrom", "tss"]).copy()
    gene_use["tss"] = pd.to_numeric(gene_use["tss"], errors="coerce")
    gene_use = gene_use.dropna(subset=["tss"]).copy()
    gene_use["tss"] = gene_use["tss"].astype(int)

    # 3. split genes by chromosome for speed
    gene_by_chr = {}
    for chrom, sub in gene_use.groupby("chrom"):
        sub = sub.sort_values("tss").reset_index(drop=True)
        gene_by_chr[chrom] = sub

    # 4. helper: nearest TSS to an interval
    def find_nearest_tss(chrom, start, end):
        if chrom not in gene_by_chr:
            return pd.Series({
                "nearest_gene": None,
                "nearest_gene_strand": None,
                "nearest_tss": np.nan,
                "tss_distance": np.nan
            })

        sub = gene_by_chr[chrom]
        tss_array = sub["tss"].values

        # 0 if TSS falls inside the interval
        # otherwise distance to nearest interval edge
        dist = np.where(
            tss_array < start,
            start - tss_array,
            np.where(tss_array > end, tss_array - end, 0)
        )

        idx = dist.argmin()

        return pd.Series({
            "nearest_gene": sub.iloc[idx][gene_name_col],
            "nearest_gene_strand": sub.iloc[idx]["strand"],
            "nearest_tss": sub.iloc[idx]["tss"],
            "tss_distance": int(dist[idx])
        })

    # 5. annotate filtered loci
    annot_df = bivalent_annot.apply(
        lambda r: find_nearest_tss(r["chrom"], int(r["start"]), int(r["end"])),
        axis=1
    )

    bivalent_annot = pd.concat(
        [bivalent_annot.reset_index(drop=True), annot_df.reset_index(drop=True)],
        axis=1
    )

    return bivalent_annot

# %%
# Normal-cell bivalent-associated LTRs (Liver)
biv_normal_annot = annotate_bivalent_ltrs_with_nearest_tss(
    df_merged,
    gene_df,
    bivalent_col="biv_normal",
    threshold=0.2,
    gene_name_col="gene_name"
)
# Cancer-cell bivalent-associated LTRs (Liver)
biv_cancer_annot = annotate_bivalent_ltrs_with_nearest_tss(
    df_merged,
    gene_df,
    bivalent_col="biv_cancer",
    threshold=0.2,
    gene_name_col="gene_name"
)

# %%
# How many loci? (Liver)
len(biv_normal_annot), len(biv_cancer_annot)

# %%
biv_normal_annot[[
    "chrom", "start", "end", "name",
    "biv_normal", "nearest_gene", "tss_distance",
    "state_seq_full_slice_normal", "state_seq_full_slice_cancer"
]].head(3)

# %%
# TSS proximity summary
print((biv_normal_annot["tss_distance"] == 0).mean())
print((biv_normal_annot["tss_distance"] <= 1000).mean())
print((biv_normal_annot["tss_distance"] <= 5000).mean())
print((biv_normal_annot["tss_distance"] <= 10000).mean())
print((biv_normal_annot["tss_distance"] > 10000).mean())

# %% [markdown]
# **Observation**: The majority of bivalent-associated LTR loci are distal from annotated TSS, suggesting that any regulatory role is likely mediated through distal regulatory mechanisms rather than promoter-proximal effects.

# %%
# compare distributions
biv_normal_annot["tss_distance"].describe()

# %%
biv_cancer_annot["tss_distance"].describe()

# %%
#  Is bivalent increase specific (localized) or global (uniform shift)?
#  but this result is consistent with this expansion: So your result is likely not just global drift
df_merged["biv_normal"].mean(), df_merged["biv_cancer"].mean()

# %%
def plot_tss_distance_categories(
    df,
    title="TSS distance distribution",
    color="steelblue",
    figsize=(6, 4),
    annotate=True
):
    """
    Plot TSS distance category distribution as a bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column 'tss_distance'

    title : str
        Plot title

    color : str or list
        Bar color (single color or list of colors)

    figsize : tuple
        Figure size (width, height)

    annotate : bool
        Whether to show percentage labels on bars
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    dist = df["tss_distance"]

    categories = pd.Series({
        "TSS (0)": (dist == 0).mean(),
        "<=1kb": (dist <= 1000).mean(),
        "<=5kb": (dist <= 5000).mean(),
        "<=10kb": (dist <= 10000).mean(),
        ">10kb": (dist > 10000).mean()
    }) * 100

    plt.figure(figsize=figsize)

    ax = categories.plot(kind="bar", color=color)

    plt.ylabel("Percentage (%)")
    plt.title(title)
    plt.xticks(rotation=45)

    # optional annotations
    if annotate:
        for i, v in enumerate(categories.values):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

    return categories

# %%
categories_normal_liver=plot_tss_distance_categories(biv_normal_annot, title="Liver Normal-cell bivalent LTRs TSS distance", color="steelblue", figsize=(4, 4),)
categories_normal_liver

# %%
def plot_tss_distance_compare(df1, df2, label1="Normal", label2="Cancer"):
    import pandas as pd
    import matplotlib.pyplot as plt

    def get_bins(df):
        d = df["tss_distance"]
        return pd.Series({
            "<=1kb": (d <= 1000).mean(),
            "1-5kb": ((d > 1000) & (d <= 5000)).mean(),
            "5-10kb": ((d > 5000) & (d <= 10000)).mean(),
            ">10kb": (d > 10000).mean()
        })

    s1 = get_bins(df1)
    s2 = get_bins(df2)

    df_plot = pd.DataFrame({
        label1: s1 * 100,
        label2: s2 * 100
    })

    df_plot.plot(kind="bar")

    plt.ylabel("Percentage (%)")
    plt.title("TSS distance comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df_plot

# %%
plot_tss_distance_compare(biv_normal_annot, biv_cancer_annot, label1="Liver_Normal", label2="Liver_Cancer")

# %%
def plot_tss_distance_box(df1, df2):
    import matplotlib.pyplot as plt
    import numpy as np

    data = [
        np.log10(df1["tss_distance"] + 1),
        np.log10(df2["tss_distance"] + 1)
    ]

    plt.figure()
    plt.boxplot(data, labels=["Normal", "Cancer"])

    plt.ylabel("log10(TSS distance + 1)")
    plt.title("TSS distance distribution")
    plt.tight_layout()
    plt.show()

# %%
plot_tss_distance_box(biv_normal_annot, biv_cancer_annot)

# %%
def plot_tss_distance_violin(df1, df2):
    import matplotlib.pyplot as plt
    import numpy as np

    data = [
        np.log10(df1["tss_distance"] + 1),
        np.log10(df2["tss_distance"] + 1)
    ]

    plt.figure()
    plt.violinplot(data, showmedians=True)

    plt.xticks([1, 2], ["Normal", "Cancer"])
    plt.ylabel("log10(TSS distance + 1)")
    plt.title("TSS distance distribution")
    plt.tight_layout()
    plt.show()

# %%
plot_tss_distance_violin(biv_normal_annot, biv_cancer_annot)

# %% [markdown]
# ### What is the fate of LTR loci with substantial bivalent chromatin in the normal state?

# %%
def plot_ltr_family_fate_summary(
    biv_df,
    background_df,
    top_n=15,
    figsize_bar=(8, 5),
    figsize_bar_major=(7, 5),
    figsize_heatmap=(8, 4),
    cmap="Blues",
    title_bar="LTR family composition by fate",
    title_bar_major="Major LTR family composition by fate",
    title_heatmap="Major LTR family enrichment by fate"
):
    """
    Plot LTR family composition and enrichment by fate for a bivalent-associated subset.

    Produces 3 plots:
    1) Detailed family composition by fate (top_n families)
    2) Major family-class composition by fate
    3) Major family-class enrichment heatmap by fate

    Parameters
    ----------
    biv_df : pd.DataFrame
        Subset dataframe, e.g. biv_normal_annot.
        Must contain:
        - 'cat_transition'
        - 'name'

    background_df : pd.DataFrame
        Full comparison dataframe, e.g. df_merged.
        Used to compute background family frequencies.
        Must contain:
        - 'name'

    top_n : int
        Number of most frequent detailed families in biv_df to display.

    figsize_bar : tuple
        Figure size for detailed stacked bar plot.

    figsize_bar_major : tuple
        Figure size for major-family stacked bar plot.

    figsize_heatmap : tuple
        Figure size for major-family enrichment heatmap.

    cmap : str
        Colormap for heatmap.

    title_bar : str
        Title for detailed stacked bar plot.

    title_bar_major : str
        Title for major family-class stacked bar plot.

    title_heatmap : str
        Title for major family-class enrichment heatmap.

    Returns
    -------
    results : dict
        Dictionary containing detailed and major-family summary tables.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ---------------------------
    # helper: map detailed family to major class
    # ---------------------------
    def map_family_to_major(name):
        if pd.isna(name):
            return "Other"

        x = str(name).upper()

        # ERVK-like
        if x.startswith("HERVK") or x.startswith("LTR5"):
            return "ERVK"

        # ERVL / MaLR-like
        if x.startswith("THE1"):
            return "ERVL-MaLR"

        if (
            x.startswith("MLT")
            or x.startswith("MST")
            or x.startswith("LTR16")
            or x.startswith("LTR33")
            or x.startswith("LTR46")
            or x.startswith("LTR67")
            or x.startswith("LTR78")
            or "ERVL" in x
        ):
            return "ERVL"

        # ERV1-like
        if (
            x.startswith("MER")
            or x.startswith("LTR12")
            or x.startswith("ERV3")
            or x.startswith("HUERS")
            or x.startswith("HERV9")
            or x.startswith("LTR8")
            or x.startswith("LTR10")
            or x.startswith("LTR13")
            or x.startswith("LTR14")
            or x.startswith("LTR15")
            or x.startswith("LTR18")
            or x.startswith("LTR25")
            or x.startswith("LTR26")
            or x.startswith("LTR40")
            or x.startswith("LTR50")
            or x.startswith("LTR70")
            or x.startswith("LTR75")
            or x.startswith("LTR79")
            or x.startswith("LTR81")
            or x.startswith("LTR83")
            or x.startswith("LTR84")
            or x.startswith("LTR87")
            or x.startswith("LTR89")
            or x.startswith("LTR90")
            or "ERV1" in x
        ):
            return "ERV1"

        return "Other"

    # ---------------------------
    # copy and define fate
    # ---------------------------
    biv = biv_df.copy()
    biv["fate"] = biv["cat_transition"].str.split(" -> ").str[1]

    # ============================================================
    # PART 1. Detailed family composition (top_n)
    # ============================================================
    family_counts = (
        biv.groupby(["fate", "name"])
        .size()
        .reset_index(name="count")
    )

    family_counts["fraction"] = (
        family_counts["count"] /
        family_counts.groupby("fate")["count"].transform("sum")
    )

    top_families = (
        biv["name"]
        .value_counts()
        .head(top_n)
        .index
    )

    family_counts_top = family_counts[
        family_counts["name"].isin(top_families)
    ].copy()

    pivot_frac = family_counts_top.pivot_table(
        index="fate",
        columns="name",
        values="fraction",
        aggfunc="sum"
    ).fillna(0)

    plt.figure(figsize=figsize_bar)
    pivot_frac.plot(
        kind="bar",
        stacked=True,
        figsize=figsize_bar
    )
    plt.ylabel("Fraction")
    plt.xlabel("fate")
    plt.title(title_bar)
    plt.legend(title="LTR family", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # PART 2. Major family-class composition
    # ============================================================
    biv["major_family"] = biv["name"].apply(map_family_to_major)
    background_major = background_df.copy()
    background_major["major_family"] = background_major["name"].apply(map_family_to_major)

    major_counts = (
        biv.groupby(["fate", "major_family"])
        .size()
        .reset_index(name="count")
    )

    major_counts["fraction"] = (
        major_counts["count"] /
        major_counts.groupby("fate")["count"].transform("sum")
    )

    pivot_major_frac = major_counts.pivot_table(
        index="fate",
        columns="major_family",
        values="fraction",
        aggfunc="sum"
    ).fillna(0)

    plt.figure(figsize=figsize_bar_major)
    pivot_major_frac.plot(
        kind="bar",
        stacked=True,
        figsize=figsize_bar_major,
        color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    )
    plt.ylabel("Fraction")
    plt.xlabel("fate")
    plt.title(title_bar_major)
    plt.legend(title="Major family", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # PART 3. Major family-class enrichment heatmap
    # ============================================================
    background_major_freq = background_major["major_family"].value_counts(normalize=True)

    major_counts["enrichment"] = major_counts.apply(
        lambda r: r["fraction"] / background_major_freq.get(r["major_family"], 1e-9),
        axis=1
    )

    enrich_major_mat = major_counts.pivot_table(
        index="fate",
        columns="major_family",
        values="enrichment",
        aggfunc="sum"
    ).fillna(0)

    count_major_mat = major_counts.pivot_table(
        index="fate",
        columns="major_family",
        values="count",
        aggfunc="sum"
    ).fillna(0)

    # make sure same order/layout
    count_major_mat = count_major_mat.reindex(
        index=enrich_major_mat.index,
        columns=enrich_major_mat.columns
    ).fillna(0)

    # annotation labels: enrichment + count
    annot_mat = enrich_major_mat.copy().astype(str)

    for i in enrich_major_mat.index:
        for j in enrich_major_mat.columns:
            enrich_val = enrich_major_mat.loc[i, j]
            count_val = int(count_major_mat.loc[i, j])
            annot_mat.loc[i, j] = f"{enrich_val:.2f}\n({count_val})"

    plt.figure(figsize=figsize_heatmap)
    sns.heatmap(
        enrich_major_mat,
        annot=annot_mat,
        fmt="",
        cmap=cmap
    )
    #####
    
    plt.title(title_heatmap)
    plt.xlabel("Major family")
    plt.ylabel("Fate")
    plt.tight_layout()
    plt.show()

    return {
        "family_counts": family_counts,
        "family_counts_top": family_counts_top,
        "pivot_frac": pivot_frac,
        "major_counts": major_counts,
        "pivot_major_frac": pivot_major_frac,
        "enrich_major_mat": enrich_major_mat
    }

# %%
results_liver = plot_ltr_family_fate_summary(
    biv_df=biv_normal_annot,
    background_df=df_merged,
    top_n=15,
    cmap="Blues",
    title_bar="LTR family composition by fate (normal bivalent loci, liver)",
    title_bar_major="Major LTR family composition by fate (liver)",
    title_heatmap="Major LTR family enrichment by fate (liver)"
)

# %%
df_merged["is_hotspot_normal"] = df_merged["n_trans_normal"] >= 4
df_merged["is_hotspot_cancer"] = df_merged["n_trans_cancer"] >= 4
# df_merged = df_merged.drop(columns=["is_hotspot"])

# %%
# Are mixed loci enriched for hotspot?
df_merged.groupby("cat_normal")["is_hotspot_normal"].mean()

# %%
df_merged.groupby("cat_normal")["n_trans_normal"].mean()

# %%
df_merged

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%
# A subset of LTR loci exhibits de novo acquisition of bivalent chromatin in cancer.

df_de_novo = df_merged[
    (df_merged["biv_normal"] == 0) &
    (df_merged["biv_cancer"] > 0.2)
]

count = len(df_de_novo)
fraction = count / len(df_merged)

print("Count:", count)
print("Fraction:", fraction)

# %%
df_de_novo

# %%
df_de_novo["name"].value_counts().head(20)

# %%
df_de_novo[["state_seq_full_slice_normal","state_seq_full_slice_cancer"]]

# %%
df_ltr_E096 = make_ltr_df_on_celltype_from_mnemonics(
    lung_cell,
    df_ltr
)
df_ltr_E114 = make_ltr_df_on_celltype_from_mnemonics(
    a549_cell,
    df_ltr
)

# %%
##### test overall LTR loci in normal cell vs. cancer cell
df_ltr_E096 = df_ltr_E096[
    df_ltr_E096["state_seq_full_slice"].apply(
        lambda s: isinstance(s, str) and set(s) != {"O"}
    )
].copy()
df_ltr_E114 = df_ltr_E114[
    df_ltr_E114["state_seq_full_slice"].apply(
        lambda s: isinstance(s, str) and set(s) != {"O"}
    )
].copy()

# %%
df_merged_lung = df_ltr_E096.merge(
    df_ltr_E114,
    on=["chrom", "start", "end", "name"],
    suffixes=("_normal", "_cancer")
)

# %%
df_merged_lung["frac_normal"] = df_merged_lung["state_seq_full_slice_normal"].apply(get_frac)
df_merged_lung["frac_cancer"] = df_merged_lung["state_seq_full_slice_cancer"].apply(get_frac)

df_merged_lung["biv_normal"] = df_merged_lung["frac_normal"].apply(lambda x: frac_of(bivalent, x))
df_merged_lung["biv_cancer"] = df_merged_lung["frac_cancer"].apply(lambda x: frac_of(bivalent, x))

# %%
df_merged_lung["switch_type"] = df_merged_lung.apply(classify, axis=1)
df_merged_lung["switch_type"].value_counts(normalize=True)

# %%
df_merged_lung[["biv_normal","biv_cancer"]].describe()

# %%
df_merged_lung=add_coarse_state_transitions(df_merged_lung, min_dominance=0.5)
mat_pct_lung = plot_transition_heatmap(df_merged_lung, cmap="Greens")

# %%
df_merged_lung["cat_transition"].value_counts()

# %%
mat_row_pct_lung = plot_transition_heatmap_by_row_with_counts(df_merged_lung, cmap="Greens")

# %%
# Normal-cell bivalent-associated LTRs (Lung)
biv_normal_annot_lung = annotate_bivalent_ltrs_with_nearest_tss(
    df_merged_lung,
    gene_df,
    bivalent_col="biv_normal",
    threshold=0.2,
    gene_name_col="gene_name"
)
# Cancer-cell bivalent-associated LTRs (Lung)
biv_cancer_annot_lung = annotate_bivalent_ltrs_with_nearest_tss(
    df_merged_lung,
    gene_df,
    bivalent_col="biv_cancer",
    threshold=0.2,
    gene_name_col="gene_name"
)       

# %%
# How many loci? (Lung)
len(biv_normal_annot_lung), len(biv_cancer_annot_lung)

# %%
categories_normal_lung=plot_tss_distance_categories(biv_normal_annot_lung, title="Lung Normal-cell bivalent LTRs TSS distance", color="olivedrab",figsize=(4, 4))
categories_normal_lung

# %%
plot_tss_distance_compare(biv_normal_annot_lung, biv_cancer_annot_lung, label1="Lung_Normal", label2="Lung_Cancer")

# %%
results_lung = plot_ltr_family_fate_summary(
    biv_df=biv_normal_annot_lung,
    background_df=df_merged_lung,
    top_n=15,
    cmap="Greens",
    title_bar="LTR family composition by fate (normal bivalent loci, lung)",
    title_bar_major="Major LTR family composition by fate (lung)",
    title_heatmap="Major LTR family enrichment by fate (lung)"
)

# %%
df_merged_lung["is_hotspot_normal"] = df_merged_lung["n_trans_normal"] >= 4
df_merged_lung["is_hotspot_cancer"] = df_merged_lung["n_trans_cancer"] >= 4
# df_merged_lung = df_merged_lung.drop(columns=["is_hotspot"])

# %%
df_merged_lung.groupby("cat_normal")["is_hotspot_normal"].mean()

# %%
df_merged_lung.groupby("cat_normal")["n_trans_normal"].mean()

# %%
df_merged_lung

# %%


# %%


# %%


# %%
df_ltr_E029 = make_ltr_df_on_celltype_from_mnemonics(
    monocyte_cell,
    df_ltr
)
df_ltr_E123 = make_ltr_df_on_celltype_from_mnemonics(
    k562_cell,
    df_ltr
)
df_merged_blood = df_ltr_E029.merge(
    df_ltr_E123,
    on=["chrom", "start", "end", "name"],
    suffixes=("_normal", "_cancer")
)

# %%
df_merged_blood["frac_normal"] = df_merged_blood["state_seq_full_slice_normal"].apply(get_frac)
df_merged_blood["frac_cancer"] = df_merged_blood["state_seq_full_slice_cancer"].apply(get_frac)

df_merged_blood["biv_normal"] = df_merged_blood["frac_normal"].apply(lambda x: frac_of(bivalent, x))
df_merged_blood["biv_cancer"] = df_merged_blood["frac_cancer"].apply(lambda x: frac_of(bivalent, x))

# %%
df_merged_blood["switch_type"] = df_merged_blood.apply(classify, axis=1)
df_merged_blood["switch_type"].value_counts(normalize=True)

# %%
df_merged_blood=add_coarse_state_transitions(df_merged_blood, min_dominance=0.5)
mat_pct_blood = plot_transition_heatmap(df_merged_blood, cmap="Reds")

# %%
df_merged_blood["cat_transition"].value_counts()

# %%


# %%
mat_row_pct_blood = plot_transition_heatmap_by_row_with_counts(df_merged_blood, cmap="Reds")

# %%
# Normal-cell bivalent-associated LTRs (Blood)
biv_normal_annot_blood = annotate_bivalent_ltrs_with_nearest_tss(
    df_merged_blood,
    gene_df,
    bivalent_col="biv_normal",
    threshold=0.2,
    gene_name_col="gene_name"
)
# Cancer-cell bivalent-associated LTRs (Blood)
biv_cancer_annot_blood = annotate_bivalent_ltrs_with_nearest_tss(
    df_merged_blood,
    gene_df,
    bivalent_col="biv_cancer",
    threshold=0.2,
    gene_name_col="gene_name"
)       

# %%
# How many loci? (Blood)
len(biv_normal_annot_blood), len(biv_cancer_annot_blood)

# %%
categories_normal_blood = plot_tss_distance_categories(biv_normal_annot_blood, title="Blood Normal-cell bivalent LTRs TSS distance", color="firebrick",figsize=(4, 4))
categories_normal_blood

# %%
plot_tss_distance_compare(biv_normal_annot_blood, biv_cancer_annot_blood, label1="Blood_Normal", label2="Blood_Cancer")

# %%
results_blood = plot_ltr_family_fate_summary(
    biv_df=biv_normal_annot_blood,
    background_df=df_merged_blood,
    top_n=15,
    cmap="Reds",
    title_bar="LTR family composition by fate (normal bivalent loci, blood)",
    title_bar_major="Major LTR family composition by fate (blood)",
    title_heatmap="Major LTR family enrichment by fate (blood)"
)

# %%
set(df_merged.columns) - set(df_merged_blood.columns)

# %%
df_merged_blood=add_transition_counts(df_merged_blood)

# %%
df_merged

# %%


# %%


# %%


# %%
df_de_novo_lung = df_merged_lung[
    (df_merged_lung["biv_normal"] == 0) &
    (df_merged_lung["biv_cancer"] > 0.2)
]

count = len(df_de_novo_lung)
fraction = count / len(df_merged_lung)

print("Count:", count)
print("Fraction:", fraction)

# %%
df_de_novo_lung = add_transition_counts(df_de_novo_lung)

# %%
df_de_novo_lung

# %%
df_de_novo_lung[[
    "state_seq_full_slice_normal",
    "state_seq_full_slice_cancer",
    "n_trans_normal",
    "n_trans_cancer"
]].head(10)

# %%
df_merged_lung=add_transition_counts(df_merged_lung)

# %%
df_merged_lung["is_hotspot"] = df_merged_lung["n_trans_cancer"] >= 4

# %%
# compare fractions:
p_all_lung = df_merged_lung["is_hotspot"].mean()
p_denovo_lung = df_de_novo_lung["n_trans_cancer"].ge(4).mean()

print("All LTR hotspot fraction:", p_all_lung)
print("De novo hotspot fraction:", p_denovo_lung)

# %%


# %%


# %%


# %%
(df_merged["biv_cancer"] > df_merged["biv_normal"]).mean()

# %%
import matplotlib.pyplot as plt

plt.hist(df_merged["biv_cancer"], bins=50, alpha=0.5, color="white", edgecolor="red", label="Cancer")
plt.hist(df_merged["biv_normal"], bins=50, alpha=0.5, color="white", edgecolor="blue", label="Normal")

plt.yscale("log")
plt.legend()
plt.xlabel("Bivalent fraction")
plt.ylabel("Count (log)")
plt.title("Bivalent signal in LTR loci")

plt.show()

# %%
from scipy.stats import mannwhitneyu

stat, p = mannwhitneyu(
    df_merged["biv_normal"],
    df_merged["biv_cancer"],
    alternative="two-sided"
)

print(p)

# %%
df_merged["biv_cancer"].mean() - df_merged["biv_normal"].mean()

# %%
(df_merged["biv_cancer"] > df_merged["biv_normal"]).mean()

# %%
# this existed below just copied here 
# transition count
def count_transitions(seq):
    if not isinstance(seq, str) or len(seq) == 0:
        return 0
    return sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))

# %%
def add_transition_counts(df):
    """
    Add transition counts for normal and cancer state sequences.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - state_seq_full_slice_normal
        - state_seq_full_slice_cancer

    Returns
    -------
    df_out : pd.DataFrame
        Copy of input with:
        - n_trans_normal
        - n_trans_cancer
    """

    def count_transitions(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return 0
        return sum(seq[i] != seq[i-1] for i in range(1, len(seq)))

    df_out = df.copy()

    df_out["n_trans_normal"] = df_out["state_seq_full_slice_normal"].apply(count_transitions)
    df_out["n_trans_cancer"] = df_out["state_seq_full_slice_cancer"].apply(count_transitions)

    return df_out

# %%
df_de_novo = add_transition_counts(df_de_novo)

# %%
df_de_novo

# %%
df_de_novo[[
    "state_seq_full_slice_normal",
    "state_seq_full_slice_cancer",
    "n_trans_normal",
    "n_trans_cancer"
]].head(10)

# %%
# define hotspot on ALL data:
df_merged["is_hotspot"] = df_merged["n_trans_cancer"] >= 4

# %%
# compare fractions:
p_all = df_merged["is_hotspot"].mean()
p_denovo = df_de_novo["n_trans_cancer"].ge(4).mean()

print("All LTR hotspot fraction:", p_all)
print("De novo hotspot fraction:", p_denovo)

# %%


# %%
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple
import tempfile
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import gseapy as gp

# Import ChromBERT utilities
import css_utility as crb

# %%


# %%


# %%
def make_ltr_transition_df(
    df_ltr_on_cell,
    remove_all_O=True,
    min_length_bp=1
):
    """
    Build an analysis-ready LTR transition dataframe from a sliced LTR dataframe.

    Parameters
    ----------
    df_ltr_on_cell : pd.DataFrame
        Output of make_ltr_df_on_celltype_from_mnemonics(), expected to contain:
        chrom, start, end, name, score, strand, length, state_seq_full_slice, n_bins

    remove_all_O : bool, default True
        If True, remove rows whose state_seq_full_slice contains only 'O'.

    min_length_bp : int, default 1
        Minimum length required to compute transition density.

    Returns
    -------
    df_ltr_trans : pd.DataFrame
        Dataframe with additional columns:
        state_comp, n_states, n_transitions, transition_density_per_kb
    """

    df_ltr_trans = df_ltr_on_cell.copy()

    # state composition
    df_ltr_trans["state_comp"] = df_ltr_trans["state_seq_full_slice"].apply(
        lambda s: Counter(s) if isinstance(s, str) else {}
    )

    # optionally remove all-O rows
    if remove_all_O:
        df_ltr_trans = df_ltr_trans[
            df_ltr_trans["state_comp"].apply(lambda c: any(k != "O" for k in c))
        ].copy()

    # distinct number of states
    df_ltr_trans["n_states"] = df_ltr_trans["state_comp"].apply(len)

    # transition count
    def count_transitions(seq):
        if not isinstance(seq, str) or len(seq) == 0:
            return 0
        return sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))

    df_ltr_trans["n_transitions"] = df_ltr_trans["state_seq_full_slice"].apply(count_transitions)

    # transition density per kb
    df_ltr_trans = df_ltr_trans[df_ltr_trans["length"] >= min_length_bp].copy()
    df_ltr_trans["transition_density_per_kb"] = (
        df_ltr_trans["n_transitions"] / (df_ltr_trans["length"] / 1000)
    )

    return df_ltr_trans

# %%
df_ltr_trans_E066=make_ltr_transition_df(df_ltr_E066)
df_sorted_E066 = df_ltr_trans_E066.sort_values("transition_density_per_kb", ascending=False)
df_sorted_E066.head(10)

# %%
df_ltr_trans_E118=make_ltr_transition_df(df_ltr_E118)
df_sorted_E118 = df_ltr_trans_E118.sort_values("transition_density_per_kb", ascending=False)
df_sorted_E118.head(10)

# %%
# def load_refgene_full(gene_annotation):
#     gene_df = pd.read_csv(
#         gene_annotation,
#         sep="\t",
#         header=None,
#         names=[
#             "bin","name","chrom","strand",
#             "txStart","txEnd",
#             "cdsStart","cdsEnd",
#             "exonCount","exonStarts","exonEnds",
#             "score","name2","cdsStartStat","cdsEndStat","exonFrames"
#         ]
#     )

#     # cleanup
#     gene_df["txStart"] = pd.to_numeric(gene_df["txStart"], errors="coerce")
#     gene_df["txEnd"] = pd.to_numeric(gene_df["txEnd"], errors="coerce")
#     gene_df = gene_df.dropna(subset=["txStart","txEnd"]).copy()

#     gene_df["txStart"] = gene_df["txStart"].astype(int)
#     gene_df["txEnd"] = gene_df["txEnd"].astype(int)

#     # add TSS
#     gene_df["tss"] = np.where(
#         gene_df["strand"] == "+",
#         gene_df["txStart"],
#         gene_df["txEnd"]
#     )

#     gene_df["gene_name"] = gene_df["name2"]

#     return gene_df

# %%
# gene_df=load_refgene_full(
#     gene_annotation,
# )
# gene_df

# %%
def annotate_hotspots_with_nearest_tss(
    df_ltr_trans,
    gene_df,
    mode="n_transitions",
    threshold=4,
    gene_name_col="gene_name"
):
    """
    Annotate hotspot LTRs with nearest gene TSS.

    Parameters
    ----------
    df_ltr_trans : pd.DataFrame
        LTR transition dataframe, expected to contain:
        chrom, start, end, name, length, n_states, n_transitions,
        transition_density_per_kb, state_seq_full_slice

    gene_df : pd.DataFrame
        Gene annotation dataframe, expected to contain:
        chrom, strand, tss, and gene_name_col

    mode : str
        "n_transitions" or "transition_density_per_kb"

    threshold : float
        Threshold for defining hotspots.
        Example:
            mode="n_transitions", threshold=4
            mode="transition_density_per_kb", threshold=2.0

    gene_name_col : str
        Column in gene_df containing gene symbols/names.

    Returns
    -------
    hotspot_annot : pd.DataFrame
        Hotspot dataframe annotated with nearest gene and TSS distance.
    """

    if mode not in ["n_transitions", "transition_density_per_kb"]:
        raise ValueError("mode must be 'n_transitions' or 'transition_density_per_kb'")

    # 1. define hotspots
    if mode == "n_transitions":
        hotspot_annot = df_ltr_trans[df_ltr_trans["n_transitions"] >= threshold].copy()
    else:
        hotspot_annot = df_ltr_trans[df_ltr_trans["transition_density_per_kb"] >= threshold].copy()

    # 2. keep only needed gene columns
    gene_use = gene_df[["chrom", "strand", "tss", gene_name_col]].copy()
    gene_use = gene_use.dropna(subset=["chrom", "tss"]).copy()
    gene_use["tss"] = pd.to_numeric(gene_use["tss"], errors="coerce")
    gene_use = gene_use.dropna(subset=["tss"]).copy()
    gene_use["tss"] = gene_use["tss"].astype(int)

    # 3. pre-split genes by chromosome for speed
    gene_by_chr = {}
    for chrom, sub in gene_use.groupby("chrom"):
        sub = sub.sort_values("tss").reset_index(drop=True)
        gene_by_chr[chrom] = sub

    # 4. nearest TSS helper
    def find_nearest_tss(chrom, start, end):
        if chrom not in gene_by_chr:
            return pd.Series({
                "nearest_gene": None,
                "nearest_gene_strand": None,
                "nearest_tss": np.nan,
                "tss_distance": np.nan
            })

        sub = gene_by_chr[chrom]
        tss_array = sub["tss"].values

        # distance from interval to TSS:
        # 0 if TSS inside [start, end]
        # otherwise min distance to interval edges
        dist = np.where(
            tss_array < start,
            start - tss_array,
            np.where(tss_array > end, tss_array - end, 0)
        )

        idx = dist.argmin()

        return pd.Series({
            "nearest_gene": sub.iloc[idx][gene_name_col],
            "nearest_gene_strand": sub.iloc[idx]["strand"],
            "nearest_tss": sub.iloc[idx]["tss"],
            "tss_distance": int(dist[idx])
        })

    # 5. annotate hotspots
    annot_df = hotspot_annot.apply(
        lambda r: find_nearest_tss(r["chrom"], int(r["start"]), int(r["end"])),
        axis=1
    )

    hotspot_annot = pd.concat([hotspot_annot.reset_index(drop=True),
                               annot_df.reset_index(drop=True)], axis=1)

    return hotspot_annot

# %%
hotspot_annot_E066=annotate_hotspots_with_nearest_tss(
    df_ltr_trans_E066,
    gene_df,
    mode="n_transitions",
    threshold=4,
    gene_name_col="gene_name"
)
hotspot_annot_E066

# %%
hotspot_annot_E118=annotate_hotspots_with_nearest_tss(
    df_ltr_trans_E118,
    gene_df,
    mode="n_transitions",
    threshold=4,
    gene_name_col="gene_name"
)
hotspot_annot_E118

# %%
print((hotspot_annot_E066["tss_distance"] == 0).mean())
print((hotspot_annot_E066["tss_distance"] <= 1000).mean())
print((hotspot_annot_E066["tss_distance"] <= 5000).mean())
print((hotspot_annot_E066["tss_distance"] <= 10000).mean())

# %%
bins = [
    (0, 0),
    (1, 1000),
    (1001, 5000),
    (5001, 10000),
    (10001, 20000)
]
dist = hotspot_annot_E066["tss_distance"].values
labels = ["0", "1-1kb", "1kb-5kb", "5kb-10kb", "10kb-20kb"]

fractions = [
    ((dist >= low) & (dist <= high)).mean()
    for (low, high) in bins
]

plt.figure(figsize=(6,4))

plt.bar(labels, fractions, color="cornflowerblue")

plt.ylabel("Fraction of hotspots")
plt.title("TSS distance bins (E066 Liver Cell Line)")
plt.ylim(0, max(fractions)*1.2)

plt.show()

# %%
def compute_bin_fraction(dist, bins):
    return [((dist >= low) & (dist <= high)).mean() for (low, high) in bins]

dist_A = hotspot_annot_E066["tss_distance"].values
dist_B = hotspot_annot_E118["tss_distance"].values

bins = [
    (0, 0),
    (1, 1000),
    (1001, 5000),
    (5001, 10000),
    (10001, 20000)
]

labels = ["0", "1-1kb", "1kb-5kb", "5kb-10kb", "10kb-20kb"]

frac_A = compute_bin_fraction(dist_A, bins)
frac_B = compute_bin_fraction(dist_B, bins)

x = np.arange(len(labels))

plt.figure(figsize=(7,4))

plt.bar(x - 0.2, frac_A, width=0.4, label="E066 (normal liver cell line)", color="cornflowerblue")
plt.bar(x + 0.2, frac_B, width=0.4, label="E118 (cancer liver cell line)", color="lightcoral")

plt.xticks(x, labels)
plt.ylabel("Fraction of hotspots")
plt.title("TSS proximity comparison")
plt.legend()

plt.show()

# %%
def extract_hotspot_nearby_genes(
    hotspot_annot,
    max_tss_distance=5000,
    gene_col="nearest_gene",
    return_df=False
):
    """
    Extract genes near hotspot LTRs within a given TSS distance.

    Parameters
    ----------
    hotspot_annot : pd.DataFrame
        Output of annotate_hotspots_with_nearest_tss()

    max_tss_distance : int, default 5000
        Maximum allowed distance to nearest TSS

    gene_col : str, default "nearest_gene"
        Column containing gene names

    return_df : bool, default False
        If True, also return the filtered dataframe

    Returns
    -------
    genes : list
        Unique nearby genes

    filtered_df : pd.DataFrame, optional
        Filtered hotspot dataframe
    """

    filtered_df = hotspot_annot[
        hotspot_annot["tss_distance"].notna() &
        (hotspot_annot["tss_distance"] <= max_tss_distance)
    ].copy()

    genes = (
        filtered_df[gene_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if return_df:
        return genes, filtered_df
    return genes

# %%
genes_E066_5kb, hotspot_E066_5kb = extract_hotspot_nearby_genes(
    hotspot_annot_E066,
    max_tss_distance=5000,
    return_df=True
)

# %%
len(genes_E066_5kb)

# %%
genes_E118_5kb, hotspot_E118_5kb = extract_hotspot_nearby_genes(
    hotspot_annot_E118,
    max_tss_distance=5000,
    return_df=True
)

# %%
len(genes_E118_5kb)

# %%
hotspot_E118_5kb[[
    "chrom", "start", "end", "name",
    "nearest_gene", "tss_distance"
]].head(20)

# %%
print(len(set(genes_E066_5kb) & set(genes_E118_5kb)))
print(len(set(genes_E118_5kb) - set(genes_E066_5kb)))
print(len(set(genes_E066_5kb) - set(genes_E118_5kb)))

# %%
# Values
only_E066 = len(set(genes_E066_5kb) - set(genes_E118_5kb))
only_E118 = len(set(genes_E118_5kb) - set(genes_E066_5kb))
both = len(set(genes_E066_5kb) & set(genes_E118_5kb))

# Plot
plt.figure()
venn2(subsets=(only_E066, only_E118, both),
      set_labels=('E066 Liver cell (≤5kb)', 'E118 Liver Cancer Cell (≤5kb)'))

plt.title('Gene Overlap within 5kb of Hotspots')
plt.show()

# %%
genes_shared = list(set(genes_E066_5kb) & set(genes_E118_5kb))
genes_cancer_only = list(set(genes_E118_5kb) - set(genes_E066_5kb))
genes_normal_only = list(set(genes_E066_5kb) - set(genes_E118_5kb))

# %%
def run_go_enrichment(gene_list, label, gene_sets="GO_Biological_Process_2021", top_n=10):
    """
    Run GO enrichment using Enrichr via gseapy.
    """

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism="human",
        outdir=None
    )

    res = enr.results.sort_values("Adjusted P-value").head(top_n)

    print(f"\n=== {label} ===")
    display(res[["Term", "Adjusted P-value", "Overlap"]])

    return res

# %%
res_shared = run_go_enrichment(genes_shared, "Shared genes")
res_cancer = run_go_enrichment(genes_cancer_only, "Cancer-specific genes")
res_normal = run_go_enrichment(genes_normal_only, "Normal-specific genes")

# %%



