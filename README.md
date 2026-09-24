# SONOVA

**Your listening, decoded.**

SONOVA is a personal music intelligence system that transforms Spotify Extended Streaming History into meaningful insights about listening behavior, music preferences, patterns, and personalized recommendations.

---

## What is SONOVA?

SONOVA analyzes your personal Spotify listening history and turns raw listening events into an interactive music intelligence dashboard.

Instead of simply showing your most-played songs, SONOVA looks at **how, when, and what you listen to** and applies data analysis and machine learning to uncover patterns in your listening behavior.

---

## What SONOVA Can Explore

### 🎧 Music DNA

Understand your overall listening profile through:

- Total listening time
- Most-played artists
- Most-played tracks
- Listening frequency
- Repeat listening patterns
- Discovery vs. familiar listening behavior

### 🕐 Listening Habits

Explore when you listen:

- Listening by hour
- Listening by day of the week
- Daily and monthly listening patterns
- Peak listening periods
- Changes in listening behavior over time

### 🧠 Listening Phases

SONOVA uses **K-Means clustering** to identify recurring patterns in listening behavior.

Different listening periods can be grouped based on features such as:

- Listening frequency
- Listening duration
- Time-based behavior
- Skip behavior
- Artist and track activity

### ⏭️ Skip Prediction

SONOVA explores machine learning models that estimate whether a listening event may result in a track being skipped.

The model uses behavioral and track-level features extracted from listening history.

### 🎵 Personalized Recommendations

SONOVA generates recommendations using information from your listening history, including:

- Listening frequency
- Track popularity within your history
- Skip behavior
- Previously played tracks
- Listening patterns

---

## How It Works

## Machine Learning Pipeline

```text
Spotify Extended Streaming History
                ↓
        Data Preprocessing
                ↓
      Exploratory Data Analysis
                ↓
        Feature Engineering
                ↓
       ┌────────┴─────────┐
       ↓                  ↓
 K-Means Clustering   Skip Prediction
       ↓                  ↓
Listening Phases      Skip Behavior
       └────────┬─────────┘
                ↓
       Personalized Recommendations
                ↓
        SONOVA Insights
```
