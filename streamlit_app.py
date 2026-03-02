"""
narrative_tool_app.py

Streamlit dashboard wrapper for:
  prototype_5_narrative_tool.py
  prototype_5_narrative_tool.py was originally produced in google colab

- Uses the SAME workflow / methodology as the core script by:
    * Loading that script from disk
    * Patching the user-tunable parameters (MIN_DF, MAX_DF, etc.) to read from
      environment variables
    * Executing the patched script in-process and then exposing its outputs

REQUIREMENTS (pip install):
  streamlit
  pandas
  numpy
  matplotlib
  sentence-transformers
  scikit-learn
  scipy
  ruptures
  networkx
  openpyxl

USAGE:
  1. Place this file AND
     `prototype_5_narrative_tool.py`
     in the same folder.
  2. `streamlit run narrative_tool_app.py`
"""

import io
import os
import re
from pathlib import Path
from datetime import date
from typing import Dict, Any, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.figure import Figure

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

CORE_SCRIPT_PATH = Path(
    "prototype_5_narrative_tool.py"
)

# Streamlit-wide plotting defaults
plt.rcParams["figure.dpi"] = 160


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def ensure_single_date(val: Union[date, Sequence[date]]) -> date:
    """Streamlit's date_input can return a date or a tuple of dates.
    This helper always returns a single date (first element if a sequence)."""
    if isinstance(val, (list, tuple)):
        return val[0]

    # Help the type checker: at this point it must be a `date`
    assert isinstance(val, date)
    return val


def save_uploaded_file(uploaded, target_name: str) -> Path:
    """
    Save an uploaded CSV or Excel file to a fixed name on disk.

    The core script expects Excel files, so if a CSV is uploaded we convert it to xlsx.
    """
    target_path = Path(target_name)

    if uploaded is None:
        raise ValueError("No file uploaded")

    suffix = Path(uploaded.name).suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        # Write bytes directly
        with open(target_path, "wb") as f:
            f.write(uploaded.getbuffer())
    elif suffix == ".csv":
        df = pd.read_csv(uploaded)
        df.to_excel(target_path, index=False)
    else:
        raise ValueError("Only .csv, .xlsx, or .xls files are supported")

    return target_path


def patch_core_script_text(text: str) -> str:
    """
    Patch the core script so that the tunable parameters read from environment
    variables instead of hard-coded constants. This preserves all of the original
    workflow while letting Streamlit control the knobs.
    """
    # MIN_DF, MAX_DF, NGRAM_RANGE, WINDOW_DAYS, SIM_THRESHOLD, TOPK_PER_TWEET,
    # MIN_OUTLETS_STD, MIN_OUTLETS_STRICT, MIN_TWEETS (two places)
    text = re.sub(
        r"MIN_DF\s*=\s*5",
        "MIN_DF = int(os.getenv('MIN_DF', '5'))",
        text,
    )
    text = re.sub(
        r"MAX_DF\s*=\s*0\.80",
        "MAX_DF = float(os.getenv('MAX_DF', '0.80'))",
        text,
    )
    text = re.sub(
        r"NGRAM_RANGE\s*=\s*\(1,\s*2\)",
        (
            "NGRAM_RANGE = tuple("
            "int(x) for x in os.getenv('NGRAM_RANGE', '1,2').split(',')"
            ")"
        ),
        text,
    )
    text = re.sub(
        r"WINDOW_DAYS\s*=\s*7",
        "WINDOW_DAYS = int(os.getenv('WINDOW_DAYS', '7'))",
        text,
    )
    text = re.sub(
        r"SIM_THRESHOLD\s*=\s*0\.30",
        "SIM_THRESHOLD = float(os.getenv('SIM_THRESHOLD', '0.30'))",
        text,
    )
    text = re.sub(
        r"TOPK_PER_TWEET\s*=\s*3",
        "TOPK_PER_TWEET = int(os.getenv('TOPK_PER_TWEET', '3'))",
        text,
    )
    text = re.sub(
        r"MIN_OUTLETS_STD\s*=\s*1",
        "MIN_OUTLETS_STD = int(os.getenv('MIN_OUTLETS_STD', '1'))",
        text,
    )
    text = re.sub(
        r"MIN_OUTLETS_STRICT\s*=\s*2",
        "MIN_OUTLETS_STRICT = int(os.getenv('MIN_OUTLETS_STRICT', '2'))",
        text,
    )
    text = re.sub(
        r"MIN_TWEETS\s*=\s*3",
        "MIN_TWEETS = int(os.getenv('MIN_TWEETS', '3'))",
        text,
    )

    return text


def run_core_pipeline(env_overrides: Dict[str, str]) -> Dict[str, Any]:
    """
    Execute the patched core script in a fresh global namespace, using the
    provided environment variable overrides.

    Returns the namespace dict, which contains:
      - corpus, ts, matches
      - narr_top_terms, narr_labels, cluster_stats_table
      - leaderboard_std, leaderboard_strict
      - top_by_narr, top_table_NER
      - influencer_clusters, ao_edges, top_authors, top_outlets, etc.
    """
    if not CORE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Core script not found at {CORE_SCRIPT_PATH}. "
            "Place the prototype_5_narrative_tool.py "
            "file in the same directory as this app."
        )

    # Apply env overrides (as strings)
    for k, v in env_overrides.items():
        if v is None:
            continue
        os.environ[k] = str(v)

    # Read and patch the script
    text = CORE_SCRIPT_PATH.read_text(encoding="utf-8")
    patched = patch_core_script_text(text)

    # Execute in an isolated namespace
    ns: Dict[str, Any] = {}
    exec(compile(patched, str(CORE_SCRIPT_PATH), "exec"), ns)

    return ns


def fig_to_png_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------
# PLOT BUILDERS (using outputs from the core script)
# ---------------------------------------------------------------------


def plot_narrative_centroids(ns: Dict[str, Any]) -> Figure:
    """
    Rebuild the 'Narrative centroids (size = cluster volume)' plot
    using core-script globals: XY, corpus, comp, narr_labels.
    """
    XY = ns["XY"]
    corpus = ns["corpus"]
    comp = ns["cluster_stats_table"]

    labels = corpus["cluster"].to_numpy()

    dfp = (
        pd.DataFrame(XY, columns=["x", "y"])
        .assign(cluster=labels, kind=corpus["_kind"].to_numpy())
    )

    # Rebuild centroid table with volume
    cent = (
        dfp[["x", "y", "cluster"]]
        .assign(cluster=labels)
        .groupby("cluster")[["x", "y"]]
        .mean()
        .reset_index()
        .merge(comp[["cluster", "n_docs"]], on="cluster", how="left")
    )

    narr_labels = ns.get("narr_labels", {})

    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = 80 * (cent["n_docs"] / cent["n_docs"].max()).clip(0.2, 1.0)
    ax.scatter(cent["x"], cent["y"], s=sizes)

    for _, r in cent.iterrows():
        label = narr_labels.get(int(r["cluster"]), str(int(r["cluster"])))
        ax.text(
            r["x"],
            r["y"],
            label,
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title("Narrative centroids (size = cluster volume)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return fig


def plot_propagation_speed(matches: pd.DataFrame) -> Figure:
    """
    Median Tweet -> Media lead per narrative cluster.
    """
    df = (
        matches.dropna(subset=["tweet_cluster", "lead_days"])
        .assign(
            tweet_cluster=lambda d: d["tweet_cluster"].astype(int),
            lead_days=lambda d: pd.to_numeric(d["lead_days"], errors="coerce"),
        )
        .query("lead_days >= 0")
    )

    agg = (
        df.groupby("tweet_cluster")
        .agg(median_lead=("lead_days", "median"), n=("lead_days", "size"))
        .reset_index()
        .rename(columns={"tweet_cluster": "cluster"})
    )

    agg = agg.sort_values("median_lead", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        agg["cluster"].astype(str) + ": "
        + agg["median_lead"].round(2).astype(str)
        + "d",
        agg["median_lead"],
    )
    ax.set_xlabel("Median Lead Days")
    ax.set_title("Median Tweet → Media Lag by Narrative Cluster")

    for i, (med, n) in enumerate(zip(agg["median_lead"], agg["n"])):
        ax.text(
            med + 0.05,
            i,
            f"{med:.2f}d  (n={n})",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_top_outlets(matches: pd.DataFrame) -> Figure:
    counts = matches["media_outlet"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts[::-1].plot(kind="barh", ax=ax, edgecolor="black")
    ax.set_title("Top 15 Media Outlets by Tweet Matches")
    ax.set_xlabel("Number of Matched Tweets")
    ax.set_ylabel("Media Outlet")
    fig.tight_layout()
    return fig


def plot_top_influencers(leaderboard: pd.DataFrame, title: str) -> Figure:
    df = leaderboard.copy()
    df["InfluenceScore"] = pd.to_numeric(df["InfluenceScore"], errors="coerce")
    topN = df.nlargest(15, "InfluenceScore").copy()

    def short(a: Any) -> str:
        a = str(a)
        return a if len(a) <= 28 else a[:25] + "…"

    labels = [short(a) for a in topN["author"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], topN["InfluenceScore"][::-1])
    ax.set_xlabel("InfluenceScore")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_role_composition(
    leaderboard_std: pd.DataFrame,
    leaderboard_strict: pd.DataFrame,
) -> Figure:
    def find_role_col(df: pd.DataFrame) -> str:
        for c in df.columns:
            if "role" in c.lower():
                return c
        raise ValueError("No role column found in leaderboard")

    role_col = find_role_col(leaderboard_std)

    def role_percentages(df: pd.DataFrame) -> pd.DataFrame:
        s = df[role_col].value_counts(normalize=True).mul(100)
        return s.rename_axis("role").reset_index(name="pct")

    std_pct = role_percentages(leaderboard_std)
    std_pct["leaderboard"] = "Standard"

    strict_pct = role_percentages(leaderboard_strict)
    strict_pct["leaderboard"] = "Strict"

    comp = pd.concat([std_pct, strict_pct], ignore_index=True)

    # pivot to stacked bars
    pivot = comp.pivot_table(
        index="leaderboard", columns="role", values="pct", fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(pivot), dtype=float)

    for role in pivot.columns:
        # ensure we have a plain float64 NumPy array
        vals = pivot[role].to_numpy(dtype=float)

        ax.bar(pivot.index, vals, bottom=bottom, label=role)
        bottom = bottom + vals   # or bottom += vals, both fine now

    ax.set_ylabel("Share of entries (%)")
    ax.set_title("Role Composition by Leaderboard Type")
    ax.legend()

    # Add labels only for larger segments
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{height:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    fig.tight_layout()
    return fig


def plot_ner_breadth(top_table_NER: pd.DataFrame) -> Figure:
    """
    'Cross-Cluster Reach (Narrative Breadth) – Top 20 Names'
    Uses n_clusters_appeared_in column in the long NER table.
    """
    # Expect a column that counts distinct clusters per name.
    # If not present, derive it.
    if "n_clusters" in top_table_NER.columns:
        breadth = (
            top_table_NER.groupby("name")["n_clusters"]
            .max()
            .sort_values(ascending=False)
        )
    else:
        breadth = (
            top_table_NER.groupby("name")["cluster"]
            .nunique()
            .sort_values(ascending=False)
        )

    breadth = breadth.head(20)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(breadth.index.to_list()[::-1], breadth.to_numpy(dtype=float)[::-1])
    ax.set_xlabel("Number of Narrative Clusters Appeared In")
    ax.set_title("Cross-Cluster Reach (Narrative Breadth) – Top 20 Names")
    fig.tight_layout()
    return fig


def plot_timeline_for_cluster(ns: Dict[str, Any], cluster_id: int) -> Figure:
    ts = ns["ts"]
    plot_fn = ns["plot_narrative_timeline"]
    fig, _ = plot_fn(ts, cluster_id, save=False, show=False)
    return fig


def plot_spike_authors(
    ns: Dict[str, Any],
    cluster_id: int,
    start_date: str,
    end_date: str,
    leaderboard_option: Optional[str],
) -> Figure:
    """
    Wraps the core script's plot_top_authors_for_window:

    - Uses the core function to compute the top_authors table (so aggregation
      logic stays identical to the original script).
    - Builds the Matplotlib figure here for use in Streamlit.
    """
    corpus = ns["corpus"]

    # Choose which leaderboard to use (if any)
    leaderboard_df = None
    if leaderboard_option == "Standard leaderboard":
        leaderboard_df = ns["leaderboard_std"]
    elif leaderboard_option == "Strict leaderboard":
        leaderboard_df = ns["leaderboard_strict"]

    plot_fn = ns["plot_top_authors_for_window"]

    # Core function returns a DataFrame (or None), not a figure
    top_authors = plot_fn(
        corpus,
        cluster_id,
        start_date,
        end_date,
        leaderboard_df=leaderboard_df,
    )

    # If nothing came back, make a simple empty figure with a message
    if top_authors is None or len(top_authors) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            f"No tweets found for cluster {cluster_id}\n"
            f"between {start_date} and {end_date}",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")
        fig.tight_layout()
        return fig

    # Replicate the original plotting style using the returned table
    author_col = "author"
    role_col = "role" if "role" in top_authors.columns else ""
    df_plot = top_authors.head(10).sort_values("tweet_count", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df_plot[author_col], df_plot["tweet_count"])

    # Add labels "count (role)" similar to the core script
    for i, (cnt, role) in enumerate(
        zip(
            df_plot["tweet_count"],
            df_plot[role_col] if role_col else [""] * len(df_plot),
        )
    ):
        label = str(cnt)
        if isinstance(role, str) and role:
            label += f" ({role})"
        ax.text(cnt + 0.3, i, label, va="center", fontsize=8)

    # Simple title using the provided date strings
    ax.set_title(
        f"Cluster {cluster_id} – Top authors\n{start_date} to {end_date}"
    )
    ax.set_xlabel("Tweet count in window")
    fig.tight_layout()
    return fig


def make_top_centrality_fig(df: pd.DataFrame, title: str) -> Figure:
    df_plot = df.head(15).sort_values("hub_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_plot["name"], df_plot["hub_score"])
    ax.set_xlabel("Hub score (in_degree + out_degree + 10×betweenness)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Story Propagation and Influence Network – SPIN",
        layout="wide",
    )

    LOGO_PATH = "app_assets/spin_logo.png"

    # ---- Centered, smaller logo ----
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        st.image(
            LOGO_PATH,
            width=205,          # <<< Adjust the logo size here (200–260)
            use_container_width=False
        )

    st.title("Story Propagation and Influence Network – SPIN")
    st.markdown(
        """
This tool maps how narratives spread across social and traditional media, 
identifies the key actors driving or amplifying those narratives, and 
measures their influence over time.

This app wraps the 'prototype_5_narrative_tool.py`
pipeline and exposes it as an interactive dashboard.

Upload tweet & media files, set analysis parameters, click **Run**, and then
explore the narrative clusters, propagation patterns, leaderboards, and NER results.
"""
    )

    # -------------------- SIDEBAR: INPUTS & CONTROLS -----------------------------
    with st.sidebar:
        st.header("1. Upload data")
        tweets_file = st.file_uploader(
            "Tweets file (.csv, .xlsx)", type=["csv", "xlsx", "xls"], key="tweets"
        )
        media_file = st.file_uploader(
            "Media file (.csv, .xlsx)", type=["csv", "xlsx", "xls"], key="media"
        )

        st.header("2. Date range")
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Start date", value=None)
        with col_b:
            end_date = st.date_input("End date", value=None)

        st.header("3. Text & matching knobs")

        min_df = st.slider("MIN_DF (min docs per term)", 1, 5, 5)
        max_df = st.slider(
            "MAX_DF (max doc fraction per term)", 0.5, 1.0, 0.80, 0.05
        )

        ngram_option = st.selectbox(
            "NGRAM_RANGE",
            options=["(1,1)", "(1,2)", "(1,3)", "(2,2)", "(3,3)"],
            index=1,
        )

        if ngram_option is None:
            ngram_option = "(1,2)"   # safety fallback

        # Parse "(1,2)" -> (1, 2)
        ngram_tuple = tuple(int(x)
                            for x in ngram_option.strip("()").split(","))

        window_days = st.slider("WINDOW_DAYS", 1, 7, 7)
        sim_threshold = st.slider("SIM_THRESHOLD", 0.20, 0.40, 0.30, 0.01)
        topk_per_tweet = st.slider("TOPK_PER_TWEET", 1, 10, 3)

        st.header("4. Leaderboard knobs")
        min_outlets_std = st.slider("MIN_OUTLETS_STD", 1, 5, 1)
        min_outlets_strict = st.slider("MIN_OUTLETS_STRICT", 1, 5, 2)
        min_tweets = st.slider("MIN_TWEETS (per author)", 3, 5, 3)

        st.header("5. Embedding model")
        embed_model = st.selectbox(
            "Embedding model",
            options=["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"],
            index=0,
        )

        run_clicked = st.button("Run analysis", type="primary")

    # -------------------- RUN PIPELINE ------------------------------------------
    if run_clicked:
        if tweets_file is None or media_file is None:
            st.error("Please upload BOTH tweets and media files.")
        elif start_date is None or end_date is None:
            st.error("Please select both start and end dates.")
        else:
            # Save uploaded files to the filenames expected by the core script
            save_uploaded_file(
                tweets_file, "tweets_input.xlsx")
            save_uploaded_file(media_file, "media_input.xlsx")

            # Normalize streamlit date_input outputs to single dates
            start_dt = ensure_single_date(start_date)
            end_dt = ensure_single_date(end_date)

            # Prepare env overrides
            env_overrides = {
                # Date filter (core script already reads these)
                "NARR_DATE_START": start_dt.isoformat(),
                "NARR_DATE_END": end_dt.isoformat(),
                # Embedding model override
                "EMBED_MODEL_NAME": embed_model,
                # Tunable knobs (patched into script)
                "MIN_DF": str(min_df),
                "MAX_DF": str(max_df),
                "NGRAM_RANGE": f"{ngram_tuple[0]},{ngram_tuple[1]}",
                "WINDOW_DAYS": str(window_days),
                "SIM_THRESHOLD": str(sim_threshold),
                "TOPK_PER_TWEET": str(topk_per_tweet),
                "MIN_OUTLETS_STD": str(min_outlets_std),
                "MIN_OUTLETS_STRICT": str(min_outlets_strict),
                "MIN_TWEETS": str(min_tweets),
            }

            progress = st.progress(0)
            with st.spinner(
                "Running narrative pipeline (this may take several minutes)..."
            ):
                progress.progress(5)
                ns = run_core_pipeline(env_overrides)
                progress.progress(100)

            # --- NER backend status message ---
            has_spacy = bool(ns.get("HAS_SPACY"))
            used_spacy = bool(ns.get("USED_SPACY_FOR_NER"))
            spacy_on_gpu = bool(ns.get("SPACY_ON_GPU"))

            if used_spacy:
                backend = "spaCy (GPU)" if spacy_on_gpu else "spaCy (CPU)"
                st.success(f"NER backend: {backend}")
            elif has_spacy:
                # spaCy is installed but failed and we fell back to regex
                st.warning(
                    "NER backend: regex-only fallback (spaCy available but not used for this run).")
            else:
                # no spaCy at all
                st.warning(
                    "NER backend: regex-only (spaCy not installed or could not be loaded).")

            # DEBUG: check whether spaCy was loaded in the core script############
            # st.write("HAS_SPACY from core script:", ns.get("HAS_SPACY"))
            # st.write("USED_SPACY_FOR_NER from core script:",
            #          ns.get("USED_SPACY_FOR_NER"))
            st.write("HAS_SPACY from core script:", ns.get("HAS_SPACY"))
            st.write("USED_SPACY_FOR_NER from core script:",
                     ns.get("USED_SPACY_FOR_NER"))

            st.success("Analysis complete.")

            # Pull out key tables for convenience
            results = {
                "narr_top_terms": ns.get("narr_top_terms"),
                "narr_labels": ns.get("narr_labels"),
                "cluster_stats_table": ns.get("cluster_stats_table"),
                "ts": ns.get("ts"),
                "leaderboard_std": ns.get("leaderboard_std"),
                "leaderboard_strict": ns.get("leaderboard_strict"),
                "top_by_narr": ns.get("top_by_narr"),
                "top_table_NER": ns.get("top_table_NER"),
                "matches": ns.get("matches"),
                "influencer_clusters": ns.get("influencer_clusters"),
                "ao_edges": ns.get("ao_edges"),
                "top_authors": ns.get("top_authors"),
                "top_outlets": ns.get("top_outlets"),
                "corpus_with_hits_non_media": ns.get("corpus_with_hits_non_media"),
            }

            st.session_state["ns"] = ns
            st.session_state["results"] = results

    # -------------------- MAIN TABS ---------------------------------------------
    if "ns" not in st.session_state:
        st.info("Upload data, configure parameters, and click **Run analysis**.")
        return

    ns = st.session_state["ns"]
    results = st.session_state["results"]

    tabs = st.tabs(
        [
            "NARRATIVE CLUSTERS",
            "NARRATIVE PROPAGATION",
            "LEADERBOARDS & TOP INFLUENCERS",
            "TOP MENTIONS",
            "DOWNLOADS",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: NARRATIVE CLUSTERS
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Narrative clusters – top terms")

        narr_top_terms = results["narr_top_terms"]
        if isinstance(narr_top_terms, pd.Series):
            df_terms = narr_top_terms.to_frame(name="top_terms").reset_index()
            df_terms = df_terms.rename(columns={"index": "cluster"})
        else:
            df_terms = narr_top_terms

        st.markdown("**Narrative Clusters (top terms)**")
        st.dataframe(df_terms)

        st.markdown("**Narrative auto-labels**")
        narr_labels = results["narr_labels"] or {}
        df_labels = (
            pd.DataFrame(
                {
                    "cluster": list(narr_labels.keys()),
                    "label": list(narr_labels.values()),
                }
            )
            .sort_values("cluster")
            .reset_index(drop=True)
        )
        st.dataframe(df_labels)

        st.markdown("**Cluster statistics**")
        cluster_stats_table = results["cluster_stats_table"]
        if cluster_stats_table is not None:
            st.dataframe(cluster_stats_table)

        st.markdown("**Narrative centroids (size = cluster volume)**")
        fig_centroids = plot_narrative_centroids(ns)
        st.pyplot(fig_centroids)
        st.download_button(
            "Download centroids plot (PNG)",
            data=fig_to_png_bytes(fig_centroids),
            file_name="narrative_centroids.png",
            mime="image/png",
        )

    # ------------------------------------------------------------------
    # TAB 2: NARRATIVE PROPAGATION
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Narrative propagation")

        matches = results["matches"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Median Tweet → Media Lag by Narrative Cluster**")
            fig_prop = plot_propagation_speed(matches)
            st.pyplot(fig_prop)
            st.download_button(
                "Download propagation speed plot (PNG)",
                data=fig_to_png_bytes(fig_prop),
                file_name="narrative_propagation_speed.png",
                mime="image/png",
            )

        with col2:
            st.markdown("**Top 15 Media Outlets by Tweet Matches**")
            fig_outlets = plot_top_outlets(matches)
            st.pyplot(fig_outlets)
            st.download_button(
                "Download media outlet matches plot (PNG)",
                data=fig_to_png_bytes(fig_outlets),
                file_name="top15_media_outlets_by_matches.png",
                mime="image/png",
            )

        st.markdown("---")
        st.subheader("Narrative timelines")

        ts = results["ts"]
        if ts is not None:
            cluster_ids = sorted(int(c) for c in ts["cluster"].unique())
            cid = st.selectbox(
                "Choose cluster for timeline",
                cluster_ids,
                index=0,
            )

            # For Pylance: cid is not None and is an int
            assert cid is not None
            cid_int = int(cid)

            fig_timeline = plot_timeline_for_cluster(ns, cid_int)
            st.pyplot(fig_timeline)
            st.download_button(
                f"Download cluster {cid_int} timeline (PNG)",
                data=fig_to_png_bytes(fig_timeline),
                file_name=f"cluster_{cid_int}_timeline.png",
                mime="image/png",
            )

        st.markdown("---")
        st.subheader("Spike author attribution")

        cluster_ids = sorted(int(c) for c in ts["cluster"].unique())
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            cid_spike = st.selectbox(
                "Cluster ID", cluster_ids, key="spike_cluster"
            )
        with col_b:
            start_spike = st.date_input(
                "Start date (YYYY-MM-DD)", value=None, key="spike_start")
        with col_c:
            end_spike = st.date_input(
                "End date (YYYY-MM-DD)", value=None, key="spike_end")

        leaderboard_choice = st.selectbox(
            "Leaderboard for attribution",
            ["None", "Standard leaderboard", "Strict leaderboard"],
            index=1,
        )

        if st.button("Plot top authors for window"):
            if start_spike is None or end_spike is None:
                st.error("Please choose both a start date and an end date.")
            else:
                # Normalize (handles the rare case of date ranges) and convert to strings
                start_spike_date = ensure_single_date(start_spike)
                end_spike_date = ensure_single_date(end_spike)
                start_spike = start_spike_date.isoformat()
                end_spike = end_spike_date.isoformat()

                # For Pylance: cid_spike is not None and is an int
                assert cid_spike is not None
                cid_spike_int = int(cid_spike)

                fig_spike = plot_spike_authors(
                    ns,
                    cluster_id=cid_spike_int,
                    start_date=start_spike,
                    end_date=end_spike,
                    leaderboard_option=(
                        leaderboard_choice if leaderboard_choice != "None" else None
                    ),
                )
                st.pyplot(fig_spike)
                st.download_button(
                    "Download spike attribution plot (PNG)",
                    data=fig_to_png_bytes(fig_spike),
                    file_name=f"spike_attribution_cluster_{cid_spike_int}.png",
                    mime="image/png",
                )

    # ------------------------------------------------------------------
    # TAB 3: LEADERBOARDS & TOP INFLUENCERS
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Leaderboards and top influencers")

        leaderboard_std = results["leaderboard_std"]
        leaderboard_strict = results["leaderboard_strict"]

        st.markdown("### Standard Leaderboard (Top 15)")
        st.dataframe(leaderboard_std.head(15))
        fig_top_std = plot_top_influencers(
            leaderboard_std, "Top 15 Influencers (Standard Leaderboard)"
        )
        st.pyplot(fig_top_std)
        st.download_button(
            "Download standard top-15 plot (PNG)",
            data=fig_to_png_bytes(fig_top_std),
            file_name="top15_influencers_standard.png",
            mime="image/png",
        )

        st.markdown("---")
        st.markdown("### Strict Leaderboard (Top 15)")
        st.dataframe(leaderboard_strict.head(15))
        fig_top_strict = plot_top_influencers(
            leaderboard_strict, "Top 15 Influencers (Strict Leaderboard)"
        )
        st.pyplot(fig_top_strict)
        st.download_button(
            "Download strict top-15 plot (PNG)",
            data=fig_to_png_bytes(fig_top_strict),
            file_name="top15_influencers_strict.png",
            mime="image/png",
        )

        st.markdown("---")
        st.markdown("### Role Composition by Leaderboard Type")
        fig_roles = plot_role_composition(leaderboard_std, leaderboard_strict)
        st.pyplot(fig_roles)
        st.download_button(
            "Download role composition plot (PNG)",
            data=fig_to_png_bytes(fig_roles),
            file_name="role_composition_by_leaderboard_type.png",
            mime="image/png",
        )

        st.markdown("---")
        st.markdown("### Top influencers per narrative")
        top_by_narr = results["top_by_narr"]
        if top_by_narr is not None:
            st.dataframe(top_by_narr)

        st.markdown("---")
        st.markdown("### Network centrality (authors & outlets)")

        top_authors = results["top_authors"]
        top_outlets = results["top_outlets"]

        if top_authors is not None and top_outlets is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Central Authors in Narrative Network**")
                fig_auth_central = make_top_centrality_fig(
                    top_authors, "Top Central Authors in Narrative Network"
                )
                st.pyplot(fig_auth_central)
                st.download_button(
                    "Download central authors plot (PNG)",
                    data=fig_to_png_bytes(fig_auth_central),
                    file_name="top_central_authors.png",
                    mime="image/png",
                )

            with col2:
                st.markdown("**Top Central Outlets in Narrative Network**")
                fig_out_central = make_top_centrality_fig(
                    top_outlets, "Top Central Outlets in Narrative Network"
                )
                st.pyplot(fig_out_central)
                st.download_button(
                    "Download central outlets plot (PNG)",
                    data=fig_to_png_bytes(fig_out_central),
                    file_name="top_central_outlets.png",
                    mime="image/png",
                )

    # ------------------------------------------------------------------
    # TAB 4: TOP MENTIONS
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Top mentions (NER)")

        top_table_NER = results["top_table_NER"]
        if top_table_NER is not None:
            st.markdown(
                "**NER TOP 25 NAMES PER NARRATIVE CLUSTER (long table)**"
            )
            st.dataframe(top_table_NER)

            st.markdown(
                "**Cross-Cluster Reach (Narrative Breadth) – Top 20 Names**"
            )
            fig_ner = plot_ner_breadth(top_table_NER)
            st.pyplot(fig_ner)
            st.download_button(
                "Download NER cross-cluster breadth plot (PNG)",
                data=fig_to_png_bytes(fig_ner),
                file_name="ner_cross_cluster_breadth_top20.png",
                mime="image/png",
            )
        else:
            st.info("NER table not found in core-script outputs.")

    # ------------------------------------------------------------------
    # TAB 5: DOWNLOADS
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Downloads")

        st.markdown(
            "Download key CSV outputs generated from the core narrative tool."
        )

        def csv_download_button(
            label: str, df: Optional[pd.DataFrame], filename: str
        ):
            if df is None:
                return
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label,
                data=csv_bytes,
                file_name=filename,
                mime="text/csv",
            )

        csv_download_button(
            "Download influencer_clusters.csv",
            results["influencer_clusters"],
            "influencer_clusters.csv",
        )
        csv_download_button(
            "Download leaderboard_standard_with_roles.csv",
            results["leaderboard_std"],
            "leaderboard_standard_with_roles.csv",
        )
        csv_download_button(
            "Download leaderboard_strict_with_roles.csv",
            results["leaderboard_strict"],
            "leaderboard_strict_with_roles.csv",
        )
        csv_download_button(
            "Download narrative_top_terms.csv",
            results["narr_top_terms"]
            if isinstance(results["narr_top_terms"], pd.DataFrame)
            else results["narr_top_terms"].to_frame("top_terms")
            if results["narr_top_terms"] is not None
            else None,
            "narrative_top_terms.csv",
        )
        csv_download_button(
            "Download propagation_edges_author_to_outlet.csv",
            results["ao_edges"],
            "propagation_edges_author_to_outlet.csv",
        )
        csv_download_button(
            "Download top_central_authors.csv",
            results["top_authors"],
            "top_central_authors.csv",
        )
        csv_download_button(
            "Download top_central_outlets.csv",
            results["top_outlets"],
            "top_central_outlets.csv",
        )
        csv_download_button(
            "Download top_names_per_cluster.csv",
            results["top_table_NER"],
            "top_names_per_cluster.csv",
        )
        csv_download_button(
            "Download tweet_media_matches.csv",
            results["matches"],
            "tweet_media_matches.csv",
        )
        csv_download_button(
            "Download corpus_with_hits_non_media.csv",
            results["corpus_with_hits_non_media"],
            "corpus_with_hits_non_media.csv",
        )

        st.markdown(
            """
For PNGs of specific plots, you can also use the **Download** buttons
directly underneath each plot in the other tabs.
"""
        )


if __name__ == "__main__":
    main()
