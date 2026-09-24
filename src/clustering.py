import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


CLUSTER_FEATURE_COLS = [
    "total_plays",
    "unique_tracks",
    "unique_artists",
    "total_minutes",
    "average_minutes",
    "skip_rate",
    "night_listening_ratio",
    "weekend_listening_ratio",
    "repeat_rate",
    "discovery_rate",
]


def run_monthly_clustering(monthly_df: pd.DataFrame, max_clusters: int = 8):
    """
    Performs K-Means clustering on monthly listening behavior observations.
    Standardizes features, tests K from 2 to min(max_clusters, len(df)-1),
    computes silhouette scores, selects the optimal K, and generates
    evidence-based behavioral descriptions based on empirical feature means.
    """
    if len(monthly_df) < 3:
        return {
            "error": "At least 3 months of listening history are required for K-Means clustering.",
            "monthly_clustered": monthly_df,
            "silhouette_scores": {},
            "best_k": None,
            "best_score": None,
            "cluster_profiles": None,
            "cluster_descriptions": {},
        }

    # Ensure all feature columns exist and are imputed
    X = monthly_df[CLUSTER_FEATURE_COLS].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    upper_k = min(max_clusters, len(monthly_df) - 1)
    silhouette_scores = {}

    for k in range(2, upper_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        # Check if more than 1 cluster assigned
        if len(set(labels)) > 1:
            silhouette_scores[k] = float(silhouette_score(X_scaled, labels))
        else:
            silhouette_scores[k] = 0.0

    if not silhouette_scores:
        best_k = 2
        best_score = 0.0
    else:
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_k]

    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    monthly_clustered = monthly_df.copy()
    monthly_clustered["cluster"] = final_kmeans.fit_predict(X_scaled)

    # Compute cluster feature averages
    cluster_profiles = (
        monthly_clustered.groupby("cluster")[CLUSTER_FEATURE_COLS]
        .mean()
        .round(3)
        .reset_index()
    )

    # Generate evidence-based descriptions strictly from observed numbers
    descriptions = {}
    overall_mean_repeat = monthly_df["repeat_rate"].mean()
    overall_mean_skip = monthly_df["skip_rate"].mean()
    overall_mean_volume = monthly_df["total_plays"].mean()
    overall_mean_discovery = monthly_df["discovery_rate"].mean()

    for _, row in cluster_profiles.iterrows():
        c_id = int(row["cluster"])
        c_vol = row["total_plays"]
        c_repeat = row["repeat_rate"]
        c_skip = row["skip_rate"]
        c_disc = row["discovery_rate"]

        traits = []
        if c_vol >= overall_mean_volume:
            traits.append("High volume immersion")
        else:
            traits.append("Moderate/light listening")

        if c_repeat >= overall_mean_repeat and c_disc < overall_mean_discovery:
            traits.append("focused repeat favorites")
        elif c_disc >= overall_mean_discovery and c_repeat < overall_mean_repeat:
            traits.append("broad discovery & exploration")
        else:
            traits.append("balanced library browsing")

        if c_skip > overall_mean_skip * 1.15:
            traits.append("elevated skip rate")
        elif c_skip < overall_mean_skip * 0.85:
            traits.append("attentive low-skip sessions")

        descriptions[c_id] = " — ".join(traits).capitalize()

    return {
        "monthly_clustered": monthly_clustered,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "best_score": best_score,
        "cluster_profiles": cluster_profiles,
        "cluster_descriptions": descriptions,
        "scaler": scaler,
        "model": final_kmeans,
    }
