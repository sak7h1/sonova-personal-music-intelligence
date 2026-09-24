import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

PREDICTION_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "shuffle",
    "reason_start",
    "master_metadata_album_artist_name",
]

CATEGORICAL_FEATURES = [
    "day_of_week",
    "reason_start",
    "master_metadata_album_artist_name",
]

NUMERIC_FEATURES = [
    "hour",
    "is_weekend",
    "shuffle",
]


def build_preprocessor():
    """
    Builds a robust scikit-learn ColumnTransformer.
    Uses OneHotEncoder with min_frequency=5 to handle high-cardinality artists.
    """
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", cat_pipe, CATEGORICAL_FEATURES),
            ("numeric", num_pipe, NUMERIC_FEATURES),
        ]
    )
    return preprocessor


def evaluate_skip_models(df: pd.DataFrame, sample_size: int = 3000):
    """
    Trains and evaluates 5 supervised ML models on Spotify event-level listening data:
    1. Logistic Regression (balanced)
    2. Decision Tree
    3. K-NN
    4. SVC (Calibrated LinearSVC for fast convergence and probability calibration)
    5. Random Forest

    Evaluates: Accuracy, Precision, Recall, F1, ROC-AUC.
    Computes 5-Fold Stratified Cross-Validation on F1 score.
    Computes Train vs Test F1 with overfitting diagnostics.
    Selects the Final Model using Test F1 as the primary project criterion.
    """
    # Sample if necessary for snappy performance while preserving class balance
    if len(df) > sample_size:
        eval_df = df.sample(n=sample_size, random_state=42)
    else:
        eval_df = df.copy()

    X = eval_df[PREDICTION_FEATURES].copy()
    X["is_weekend"] = X["is_weekend"].astype(int)
    X["shuffle"] = X["shuffle"].astype(int)
    y = eval_df["skipped"].astype(bool)

    # Check for single-class edge case
    if len(y.unique()) < 2:
        return {
            "error": "The dataset contains only one target class (no skips or only skips). Classification requires both classes."
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    model_definitions = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "K-NN": KNeighborsClassifier(n_neighbors=15),
        "SVC": CalibratedClassifierCV(
            LinearSVC(dual=False, random_state=42, class_weight="balanced", max_iter=1000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
    }

    trained_pipelines = {}
    performance_records = []
    cv_records = []
    train_test_records = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in model_definitions.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Test set evaluation
        y_test_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_test_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = None

        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, zero_division=0)
        rec = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        auc = (
            roc_auc_score(y_test, y_test_proba)
            if y_test_proba is not None
            else 0.50
        )

        # Train set evaluation for overfitting diagnostic
        y_train_pred = pipe.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        f1_gap = train_f1 - test_f1

        if f1_gap > 0.15:
            overfit_diag = "Possible overfitting (elevated train/test gap)"
        elif f1_gap < -0.05:
            overfit_diag = "Test performance slightly exceeds train"
        else:
            overfit_diag = "Similar train/test performance (balanced generalization)"

        performance_records.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(test_f1, 4),
                "ROC-AUC": round(auc, 4),
            }
        )

        train_test_records.append(
            {
                "Model": name,
                "Train F1": round(train_f1, 4),
                "Test F1": round(test_f1, 4),
                "Gap": round(f1_gap, 4),
                "Diagnostic": overfit_diag,
            }
        )

        # 5-fold Stratified CV on F1
        scores = cross_val_score(
            pipe, X_train, y_train, scoring="f1", cv=skf, n_jobs=1
        )
        cv_records.append(
            {
                "Model": name,
                "Fold 1": round(scores[0], 4),
                "Fold 2": round(scores[1], 4),
                "Fold 3": round(scores[2], 4),
                "Fold 4": round(scores[3], 4),
                "Fold 5": round(scores[4], 4),
                "Mean F1": round(scores.mean(), 4),
                "Std Dev": round(scores.std(), 4),
            }
        )

    performance_df = pd.DataFrame(performance_records)
    train_test_df = pd.DataFrame(train_test_records)
    cv_df = pd.DataFrame(cv_records)

    # Select final model using Test F1 as primary project criterion
    best_row = performance_df.sort_values("F1", ascending=False).iloc[0]
    final_model_name = best_row["Model"]
    final_pipeline = trained_pipelines[final_model_name]

    # Confusion matrix for final selected model on actual test set
    y_final_pred = final_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_final_pred)
    # cm format: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    return {
        "performance_df": performance_df,
        "train_test_df": train_test_df,
        "cv_df": cv_df,
        "final_model_name": final_model_name,
        "final_pipeline": final_pipeline,
        "confusion_matrix": {
            "matrix": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "test_sample_count": len(y_test),
        "trained_pipelines": trained_pipelines,
    }


def predict_single_event(pipeline, hour, day_of_week, is_weekend, shuffle, reason_start, artist):
    """
    Uses the trained final pipeline to classify an arbitrary listening event.
    Returns:
    - outcome: "LIKELY TO SKIP" or "LIKELY TO LISTEN"
    - probability: float percentage (e.g. 72.4)
    """
    input_df = pd.DataFrame(
        [
            {
                "hour": int(hour),
                "day_of_week": str(day_of_week),
                "is_weekend": int(is_weekend),
                "shuffle": int(shuffle),
                "reason_start": str(reason_start),
                "master_metadata_album_artist_name": str(artist),
            }
        ]
    )

    pred = bool(pipeline.predict(input_df)[0])
    prob_skip = 0.50

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_df)[0]
        # index 1 corresponds to True (skipped)
        prob_skip = float(probs[1])
    else:
        prob_skip = 0.85 if pred else 0.15

    outcome = "LIKELY TO SKIP" if pred else "LIKELY TO LISTEN"
    return outcome, round(prob_skip * 100, 1)
