import json
import zipfile
import pandas as pd
import numpy as np


SENSITIVE_COLUMNS = [
    "ip_addr_decrypted",
    "username",
    "conn_country",
    "user_agent_decrypted",
    "client_ip_address",
    "postal_code",
    "latitude",
    "longitude",
]


def load_spotify_zip(uploaded_file):
    """
    Parses a Spotify Extended Streaming History ZIP file.
    Finds and loads all Streaming_History_Audio_*.json files,
    strips sensitive metadata, and creates clean derived time features.
    """
    records = []

    try:
        with zipfile.ZipFile(uploaded_file) as z:
            namelist = z.namelist()
            # Look for audio streaming history json files
            json_files = [
                name
                for name in namelist
                if ("Streaming_History_Audio" in name or "StreamingHistory" in name)
                and name.endswith(".json")
            ]

            # Fallback: any JSON with expected structure if naming differs
            if not json_files:
                json_files = [
                    name
                    for name in namelist
                    if name.endswith(".json") and not name.startswith("__MACOSX")
                ]

            if not json_files:
                raise ValueError(
                    "No Spotify streaming history JSON files were found in the uploaded ZIP."
                )

            for name in json_files:
                try:
                    with z.open(name) as f:
                        raw = f.read()

                    if not raw.strip():
                        continue

                    data = json.loads(raw.decode("utf-8-sig", errors="replace"))

                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict) and "records" in data:
                        records.extend(data["records"])

                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    except zipfile.BadZipFile:
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    if not records:
        raise ValueError(
            "This doesn't look like a Spotify Extended Streaming History export. "
            "Please download your Spotify personal data and upload the ZIP containing your Extended Streaming History."
        )

    df = pd.DataFrame(records)

    # Standardize column names if legacy format
    rename_map = {
        "endTime": "ts",
        "trackName": "master_metadata_track_name",
        "artistName": "master_metadata_album_artist_name",
        "msPlayed": "ms_played",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_columns = [
        "ts",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "ms_played",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required Spotify fields in data: {missing}. "
            "Please ensure you are uploading the Extended Streaming History package."
        )

    # Discard sensitive personal fields
    for col in SENSITIVE_COLUMNS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Filter to valid music tracks
    df = df[
        df["master_metadata_track_name"].notna()
        & df["master_metadata_album_artist_name"].notna()
    ].copy()

    # Time features
    df["timestamp"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])

    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce").fillna(0)
    df["minutes_played"] = df["ms_played"] / 60000.0
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["hour"] = df["timestamp"].dt.hour
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    # Target & auxiliary attributes
    if "skipped" in df.columns:
        df["skipped"] = df["skipped"].fillna(False).astype(bool)
    else:
        # If skipped not present, proxy by ms_played < 30000 ms
        df["skipped"] = df["ms_played"] < 30000

    if "shuffle" in df.columns:
        df["shuffle"] = df["shuffle"].fillna(False).astype(bool)
    else:
        df["shuffle"] = False

    if "reason_start" not in df.columns:
        df["reason_start"] = "trackdone"
    else:
        df["reason_start"] = df["reason_start"].fillna("unknown").astype(str)

    df = df.drop_duplicates().reset_index(drop=True)

    return df
