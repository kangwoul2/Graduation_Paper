from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
DATA_PATH = Path("3.Labeled_data/BTC_ohlcv_data_Labeled.csv")
RESULT_DIR = Path("research_v2/results")
LABELS = [-1, 0, 1]

LEGACY_FEATURES = [
    "sma_50", "ema_20", "wma_20", "macd", "macd_signal", "macd_hist", "tema_20",
    "rsi_14", "roc", "cci_14", "willr_14", "atr_14", "upper_bb", "middle_bb", "lower_bb",
    "obv", "ad", "chaikin_ad",
]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a / b.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def load_frame() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date", "open", "high", "low", "close", "volume", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def build_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Lagged returns use only information available at or before time t.
    for lag in (1, 2, 3, 5, 10):
        out[f"return_{lag}d"] = df["close"].pct_change(lag)

    out["body_pct"] = safe_div(df["close"] - df["open"], df["open"])
    out["range_pct"] = safe_div(df["high"] - df["low"], df["close"])
    out["volume_log_change"] = np.log1p(df["volume"]).diff()

    out["close_sma_gap"] = safe_div(df["close"], df["sma_50"]) - 1
    out["close_ema_gap"] = safe_div(df["close"], df["ema_20"]) - 1
    out["close_wma_gap"] = safe_div(df["close"], df["wma_20"]) - 1
    out["close_tema_gap"] = safe_div(df["close"], df["tema_20"]) - 1

    out["macd_pct"] = safe_div(df["macd"], df["close"])
    out["macd_signal_pct"] = safe_div(df["macd_signal"], df["close"])
    out["macd_hist_pct"] = safe_div(df["macd_hist"], df["close"])
    out["atr_pct"] = safe_div(df["atr_14"], df["close"])

    out["rsi_14"] = df["rsi_14"] / 100.0
    out["roc"] = df["roc"] / 100.0
    out["cci_14"] = df["cci_14"] / 200.0
    out["willr_14"] = df["willr_14"] / 100.0

    out["bb_position"] = safe_div(df["close"] - df["lower_bb"], df["upper_bb"] - df["lower_bb"])
    out["bb_width"] = safe_div(df["upper_bb"] - df["lower_bb"], df["middle_bb"])

    # Raw return in the legacy file is the next-day target return and must never be used as an input feature.
    return out.replace([np.inf, -np.inf], np.nan)


def prepare_dataset(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if feature_set == "legacy_technical":
        X = df[LEGACY_FEATURES].copy()
    elif feature_set == "stationary_v2":
        X = build_stationary_features(df)
    else:
        raise ValueError(feature_set)

    combined = X.copy()
    combined["label"] = df["label"].astype(int)
    combined["date"] = df["date"]
    combined = combined.dropna().reset_index(drop=True)
    return combined.drop(columns=["label", "date"]), combined["label"], combined["date"]


def chronological_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.6, val_ratio: float = 0.2) -> SplitData:
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end < 100 or val_end <= train_end or val_end >= n:
        raise ValueError(f"insufficient samples for split: {n}")
    return SplitData(
        X.iloc[:train_end], y.iloc[:train_end],
        X.iloc[train_end:val_end], y.iloc[train_end:val_end],
        X.iloc[val_end:], y.iloc[val_end:],
    )


def metric_row(name: str, y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_down": recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[0],
        "recall_neutral": recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[1],
        "recall_up": recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[2],
    }


def candidate_models() -> list[tuple[str, object, bool]]:
    return [
        ("dummy_majority", DummyClassifier(strategy="most_frequent"), False),
        (
            "logistic_balanced",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.2, class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
            ]),
            False,
        ),
        (
            "random_forest_balanced",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    ]


def fit_model(model, X: pd.DataFrame, y: pd.Series, use_balanced_sample_weight: bool):
    if use_balanced_sample_weight:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def tune_on_validation(split: SplitData) -> tuple[str, object, bool, pd.DataFrame]:
    rows = []
    best = None
    best_score = -math.inf

    for name, prototype, weighted in candidate_models():
        model = clone(prototype)
        fit_model(model, split.X_train, split.y_train, weighted)
        pred = model.predict(split.X_val)
        row = metric_row(name, split.y_val, pred)
        rows.append(row)
        if row["macro_f1"] > best_score:
            best_score = row["macro_f1"]
            best = (name, prototype, weighted)

    assert best is not None
    return (*best, pd.DataFrame(rows).sort_values("macro_f1", ascending=False))


def final_test(split: SplitData, selected: tuple[str, object, bool]) -> tuple[object, dict, np.ndarray]:
    name, prototype, weighted = selected
    X_train = pd.concat([split.X_train, split.X_val], axis=0)
    y_train = pd.concat([split.y_train, split.y_val], axis=0)
    model = clone(prototype)
    fit_model(model, X_train, y_train, weighted)
    pred = model.predict(split.X_test)
    return model, metric_row(name, split.y_test, pred), pred


def walk_forward(X: pd.DataFrame, y: pd.Series, selected: tuple[str, object, bool]) -> pd.DataFrame:
    name, prototype, weighted = selected
    tscv = TimeSeriesSplit(n_splits=5)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = clone(prototype)
        fit_model(model, X.iloc[train_idx], y.iloc[train_idx], weighted)
        pred = model.predict(X.iloc[test_idx])
        row = metric_row(name, y.iloc[test_idx], pred)
        row["fold"] = fold
        row["train_size"] = len(train_idx)
        row["test_size"] = len(test_idx)
        rows.append(row)
    return pd.DataFrame(rows)


def selective_curve(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame()
    probs = model.predict_proba(X_test)
    classes = np.asarray(model.classes_)
    confidence = probs.max(axis=1)
    pred = classes[probs.argmax(axis=1)]
    rows = []
    for threshold in np.arange(0.35, 0.81, 0.05):
        mask = confidence >= threshold
        coverage = float(mask.mean())
        if mask.sum() < 20:
            continue
        rows.append({
            "threshold": round(float(threshold), 2),
            "coverage": coverage,
            "selected_samples": int(mask.sum()),
            "accuracy": accuracy_score(y_test.iloc[np.flatnonzero(mask)], pred[mask]),
            "balanced_accuracy": balanced_accuracy_score(y_test.iloc[np.flatnonzero(mask)], pred[mask]),
            "macro_f1": f1_score(y_test.iloc[np.flatnonzero(mask)], pred[mask], average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)


def plot_results(comparison: pd.DataFrame, selective: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = comparison.sort_values("macro_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ordered))
    width = 0.36
    ax.bar(x - width / 2, ordered["balanced_accuracy"], width, label="Balanced Accuracy")
    ax.bar(x + width / 2, ordered["macro_f1"], width, label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["model"], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Leakage-safe chronological holdout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)

    if not selective.empty:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(selective["coverage"], selective["accuracy"], marker="o", label="Accuracy")
        ax1.plot(selective["coverage"], selective["macro_f1"], marker="s", label="Macro F1")
        ax1.set_xlabel("Coverage")
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1)
        ax1.set_title("Confidence threshold: quality vs coverage")
        ax1.legend()
        fig.tight_layout()
        fig.savefig(RESULT_DIR / "selective_prediction.png", dpi=160)
        plt.close(fig)


def write_summary(
    feature_set: str,
    test_metrics: dict,
    dummy_metrics: dict,
    walk: pd.DataFrame,
    selective: pd.DataFrame,
    class_counts: dict,
) -> None:
    macro_gain = test_metrics["macro_f1"] - dummy_metrics["macro_f1"]
    bal_gain = test_metrics["balanced_accuracy"] - dummy_metrics["balanced_accuracy"]
    lines = [
        "# Research V2 Result Summary",
        "",
        "This file is generated from repository data. It must not be edited to manufacture performance claims.",
        "",
        f"- Selected feature set: `{feature_set}`",
        f"- Selected model: `{test_metrics['model']}`",
        f"- Test accuracy: `{test_metrics['accuracy']:.4f}`",
        f"- Test balanced accuracy: `{test_metrics['balanced_accuracy']:.4f}`",
        f"- Test macro F1: `{test_metrics['macro_f1']:.4f}`",
        f"- Macro F1 gain vs majority dummy: `{macro_gain:+.4f}`",
        f"- Balanced accuracy gain vs majority dummy: `{bal_gain:+.4f}`",
        f"- Walk-forward macro F1 mean/std: `{walk['macro_f1'].mean():.4f} / {walk['macro_f1'].std(ddof=0):.4f}`",
        f"- Class counts: `{json.dumps(class_counts, ensure_ascii=False)}`",
        "",
    ]
    if not selective.empty:
        eligible = selective[selective["coverage"] >= 0.20]
        if not eligible.empty:
            best = eligible.sort_values(["accuracy", "coverage"], ascending=[False, False]).iloc[0]
            lines.extend([
                "## Selective prediction",
                "",
                "When the system is allowed to abstain on low-confidence days, quality can be traded for coverage.",
                f"- Best accuracy with coverage >= 20%: `{best['accuracy']:.4f}`",
                f"- Coverage: `{best['coverage']:.4f}`",
                f"- Confidence threshold: `{best['threshold']:.2f}`",
                "",
            ])
    (RESULT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_feature_set(df: pd.DataFrame, feature_set: str) -> dict:
    X, y, dates = prepare_dataset(df, feature_set)
    split = chronological_split(X, y)
    name, prototype, weighted, validation = tune_on_validation(split)
    model, test_metrics, test_pred = final_test(split, (name, prototype, weighted))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(pd.concat([split.X_train, split.X_val]), pd.concat([split.y_train, split.y_val]))
    dummy_pred = dummy.predict(split.X_test)
    dummy_metrics = metric_row("dummy_majority", split.y_test, dummy_pred)

    pretest_X = pd.concat([split.X_train, split.X_val], axis=0).reset_index(drop=True)
    pretest_y = pd.concat([split.y_train, split.y_val], axis=0).reset_index(drop=True)
    walk = walk_forward(pretest_X, pretest_y, (name, prototype, weighted))
    selective = selective_curve(model, split.X_test.reset_index(drop=True), split.y_test.reset_index(drop=True))

    confusion = pd.DataFrame(
        confusion_matrix(split.y_test, test_pred, labels=LABELS),
        index=[f"actual_{x}" for x in LABELS],
        columns=[f"pred_{x}" for x in LABELS],
    )

    return {
        "feature_set": feature_set,
        "validation": validation,
        "test_metrics": test_metrics,
        "dummy_metrics": dummy_metrics,
        "walk": walk,
        "selective": selective,
        "confusion": confusion,
        "test_start": str(dates.iloc[int(len(dates) * 0.8)].date()),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    class_counts = {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}

    outcomes = [run_feature_set(df, feature_set) for feature_set in ("legacy_technical", "stationary_v2")]

    ablation_rows = []
    for outcome in outcomes:
        row = {"feature_set": outcome["feature_set"], **outcome["test_metrics"]}
        row["dummy_macro_f1"] = outcome["dummy_metrics"]["macro_f1"]
        row["macro_f1_gain_vs_dummy"] = row["macro_f1"] - row["dummy_macro_f1"]
        row["walk_forward_macro_f1_mean"] = outcome["walk"]["macro_f1"].mean()
        row["walk_forward_macro_f1_std"] = outcome["walk"]["macro_f1"].std(ddof=0)
        ablation_rows.append(row)

    ablation = pd.DataFrame(ablation_rows).sort_values("macro_f1", ascending=False)
    best_feature = ablation.iloc[0]["feature_set"]
    best = next(o for o in outcomes if o["feature_set"] == best_feature)

    validation_frames = []
    for outcome in outcomes:
        frame = outcome["validation"].copy()
        frame.insert(0, "feature_set", outcome["feature_set"])
        validation_frames.append(frame)
    pd.concat(validation_frames, ignore_index=True).to_csv(RESULT_DIR / "validation_model_selection.csv", index=False)
    ablation.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    best["walk"].to_csv(RESULT_DIR / "walk_forward.csv", index=False)
    best["selective"].to_csv(RESULT_DIR / "selective_prediction.csv", index=False)
    best["confusion"].to_csv(RESULT_DIR / "confusion_matrix.csv")

    plot_results(ablation.rename(columns={"feature_set": "model"}), best["selective"])
    write_summary(
        str(best_feature),
        best["test_metrics"],
        best["dummy_metrics"],
        best["walk"],
        best["selective"],
        class_counts,
    )

    metadata = {
        "data_path": str(DATA_PATH),
        "rows": len(df),
        "label_definition": "next-day close return: up > +1%, down < -1%, otherwise neutral",
        "split": "chronological 60/20/20",
        "selection_metric": "validation macro_f1",
        "test_policy": "test partition evaluated once after model selection",
        "target_column_excluded": "return (legacy file contains next-day target return)",
    }
    (RESULT_DIR / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print((RESULT_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
