import pandas as pd
import numpy as np


def create_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes monthly behavioral features where one month = one observation.
    Features:
    - total_plays
    - unique_tracks
    - unique_artists
    - total_minutes
    - average_minutes
    - skip_rate
    - night_listening_ratio (22:00 to 05:00)
    - weekend_listening_ratio
    - repeat_rate (tracks played > 1 time / unique tracks in month)
    - discovery_rate (tracks played == 1 time / unique tracks in month)
    """
    monthly = (
        df.groupby("month")
        .agg(
            total_plays=("master_metadata_track_name", "size"),
            unique_tracks=("master_metadata_track_name", "nunique"),
            unique_artists=("master_metadata_album_artist_name", "nunique"),
            total_minutes=("minutes_played", "sum"),
            average_minutes=("minutes_played", "mean"),
            skip_rate=("skipped", "mean"),
            night_listening_ratio=("hour", lambda x: ((x >= 22) | (x <= 5)).mean()),
            weekend_listening_ratio=("is_weekend", "mean"),
        )
        .reset_index()
    )

    # Monthly track frequency for repeat & discovery rates
    track_counts = (
        df.groupby(["month", "master_metadata_track_name"])
        .size()
        .reset_index(name="plays")
    )

    repeat_rate = (
        track_counts.assign(repeated=lambda x: x["plays"] > 1)
        .groupby("month")["repeated"]
        .mean()
        .rename("repeat_rate")
        .reset_index()
    )

    discovery_rate = (
        track_counts.assign(one_play=lambda x: x["plays"] == 1)
        .groupby("month")["one_play"]
        .mean()
        .rename("discovery_rate")
        .reset_index()
    )

    monthly = monthly.merge(repeat_rate, on="month", how="left")
    monthly = monthly.merge(discovery_rate, on="month", how="left")

    monthly = monthly.sort_values("month").reset_index(drop=True)
    return monthly


def compute_overview_metrics(df: pd.DataFrame) -> dict:
    """
    Computes core Music DNA KPI metrics for the entire listening dataset.
    """
    total_plays = len(df)
    total_minutes = df["minutes_played"].sum()
    total_hours = total_minutes / 60.0
    unique_tracks = df["master_metadata_track_name"].nunique()
    unique_artists = df["master_metadata_album_artist_name"].nunique()
    skip_rate = df["skipped"].mean()

    # Track repeat behavior across entire dataset
    track_counts = df.groupby("master_metadata_track_name").size()
    repeat_rate = (track_counts > 1).mean()
    discovery_rate = (track_counts == 1).mean()

    return {
        "total_plays": total_plays,
        "total_hours": total_hours,
        "total_minutes": total_minutes,
        "unique_tracks": unique_tracks,
        "unique_artists": unique_artists,
        "skip_rate": skip_rate,
        "repeat_rate": repeat_rate,
        "discovery_rate": discovery_rate,
    }


def get_top_artists(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Returns the top N artists by total plays and listening time.
    """
    top = (
        df.groupby("master_metadata_album_artist_name")
        .agg(
            total_plays=("master_metadata_track_name", "size"),
            total_hours=("minutes_played", lambda x: round(x.sum() / 60.0, 1)),
            skip_rate=("skipped", lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
        .rename(columns={"master_metadata_album_artist_name": "artist"})
        .sort_values("total_plays", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return top


def get_top_tracks(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Returns the top N tracks by total plays.
    """
    top = (
        df.groupby(["master_metadata_track_name", "master_metadata_album_artist_name"])
        .agg(
            total_plays=("master_metadata_track_name", "size"),
            total_hours=("minutes_played", lambda x: round(x.sum() / 60.0, 1)),
            skip_rate=("skipped", lambda x: round(x.mean() * 100, 1)),
        )
        .reset_index()
        .rename(
            columns={
                "master_metadata_track_name": "track",
                "master_metadata_album_artist_name": "artist",
            }
        )
        .sort_values("total_plays", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return top


def get_hourly_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns listening distribution grouped by hour of day (0-23).
    """
    hourly = (
        df.groupby("hour")
        .agg(
            plays=("master_metadata_track_name", "size"),
            skip_rate=("skipped", "mean"),
        )
        .reset_index()
        .sort_values("hour")
    )
    # Ensure all 24 hours are present
    all_hours = pd.DataFrame({"hour": list(range(24))})
    hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)
    return hourly


def create_listening_clock_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a Day of Week (rows) x Hour of Day (columns) matrix of play counts.
    """
    days_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    pivot = df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="master_metadata_track_name",
        aggfunc="count",
        fill_value=0,
    )

    # Reindex to ensure standard day ordering and all 24 hours
    pivot = pivot.reindex(index=days_order, fill_value=0)
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot[sorted(pivot.columns)]

    return pivot
