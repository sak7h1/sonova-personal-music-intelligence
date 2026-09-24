import pandas as pd
import numpy as np


def generate_recommendations(df: pd.DataFrame, context: str = "General", n: int = 10) -> pd.DataFrame:
    """
    Generates personalized recommendations based on actual historical listening behavior.
    Contexts:
    - General: Balances lifetime frequency, low skip rate, and recency.
    - Night: Prioritizes tracks disproportionately played between 22:00 and 05:00.
    - Weekend: Prioritizes tracks played during weekends (Saturday and Sunday).
    - Exploration: Prioritizes low-frequency tracks with zero or low skip rates.
    - Repeat Favorites: Prioritizes top repeated tracks with highest completion loyalty.

    No emotion detection is claimed; all results stem from empirical listening actions.
    """
    # Track base aggregation
    grouped = (
        df.groupby(
            ["master_metadata_track_name", "master_metadata_album_artist_name"]
        )
        .agg(
            total_plays=("master_metadata_track_name", "size"),
            skip_rate=("skipped", "mean"),
            last_played=("timestamp", "max"),
        )
        .reset_index()
    )

    latest_date = df["timestamp"].max()
    grouped["days_since_play"] = (
        (latest_date - grouped["last_played"]).dt.total_seconds() / 86400.0
    ).fillna(9999)

    # Core score components
    grouped["play_score"] = np.log1p(grouped["total_plays"])
    grouped["skip_score"] = 1.0 - grouped["skip_rate"].clip(0, 1)
    grouped["recency_score"] = 1.0 / (1.0 + grouped["days_since_play"] / 30.0)

    # Contextual filtering
    ctx_clean = context.strip().lower()

    if "night" in ctx_clean:
        context_df = df[(df["hour"] >= 22) | (df["hour"] <= 5)]
        desc_rationale = "Elevated night listening activity (22:00 – 05:00)"
    elif "weekend" in ctx_clean:
        context_df = df[df["is_weekend"] == True]
        desc_rationale = "Frequent weekend session listening"
    else:
        context_df = df
        desc_rationale = "Consistent all-hours listening loyalty"

    context_counts = (
        context_df.groupby(
            ["master_metadata_track_name", "master_metadata_album_artist_name"]
        )
        .size()
        .reset_index(name="context_plays")
    )

    merged = grouped.merge(
        context_counts,
        on=["master_metadata_track_name", "master_metadata_album_artist_name"],
        how="left",
    )
    merged["context_plays"] = merged["context_plays"].fillna(0)

    if "exploration" in ctx_clean:
        # Favor tracks with fewer plays (e.g., 2 to 15) but very low skip rate
        merged["exploration_score"] = np.where(
            (merged["total_plays"] >= 2) & (merged["total_plays"] <= 20),
            1.5 / (np.log1p(merged["total_plays"])),
            0.1,
        )
        merged["final_score"] = (
            0.20 * merged["play_score"]
            + 0.50 * merged["skip_score"]
            + 0.10 * merged["recency_score"]
            + 0.20 * merged["exploration_score"]
        )
        desc_rationale = "Low-frequency discovery track with clean completion"

    elif "repeat" in ctx_clean or "favorite" in ctx_clean:
        # Heavily favor play volume and completion loyalty
        merged["final_score"] = (
            0.60 * merged["play_score"]
            + 0.30 * merged["skip_score"]
            + 0.10 * merged["recency_score"]
        )
        desc_rationale = "High-frequency heavy rotation loyalty"

    else:
        # Context ratio (night / weekend / general)
        merged["context_ratio"] = merged["context_plays"] / merged["total_plays"].clip(lower=1)
        base_recommendation_score = (
            0.50 * merged["play_score"]
            + 0.30 * merged["skip_score"]
            + 0.20 * merged["recency_score"]
        )
        merged["final_score"] = (
            0.75 * base_recommendation_score
            + 0.25 * merged["context_ratio"]
        )

    top_recs = (
        merged.sort_values("final_score", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    top_recs["Behavioral Context"] = desc_rationale
    top_recs["Skip Rate"] = (top_recs["skip_rate"] * 100).round(1).astype(str) + "%"
    top_recs["Total Plays"] = top_recs["total_plays"]
    top_recs["Score"] = top_recs["final_score"].round(3)

    return top_recs[
        [
            "master_metadata_track_name",
            "master_metadata_album_artist_name",
            "Total Plays",
            "Skip Rate",
            "Score",
            "Behavioral Context",
        ]
    ].rename(
        columns={
            "master_metadata_track_name": "Track Name",
            "master_metadata_album_artist_name": "Artist",
        }
    )
