from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path('.')
BTC_PATH = ROOT / '3.Labeled_data/BTC_ohlcv_data_Labeled.csv'
RESULT_DIR = ROOT / 'research_v2/results_extended'
ASSETS = ['SPY', 'QQQ', 'DIA', 'IWM', 'EFA', 'EEM', 'XLK', 'XLF', 'XLE', 'IAU']
RANDOM_STATE = 42


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def btc_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3, 5, 7, 10, 14, 21):
        x[f'btc_return_{lag}d'] = df['close'].pct_change(lag)

    x['body_pct'] = safe_div(df['close'] - df['open'], df['open'])
    x['range_pct'] = safe_div(df['high'] - df['low'], df['close'])
    x['volume_log_change'] = np.log1p(df['volume']).diff()
    x['close_sma_gap'] = safe_div(df['close'], df['sma_50']) - 1
    x['close_ema_gap'] = safe_div(df['close'], df['ema_20']) - 1
    x['close_tema_gap'] = safe_div(df['close'], df['tema_20']) - 1
    x['macd_pct'] = safe_div(df['macd'], df['close'])
    x['macd_hist_pct'] = safe_div(df['macd_hist'], df['close'])
    x['atr_pct'] = safe_div(df['atr_14'], df['close'])
    x['rsi_14'] = df['rsi_14'] / 100.0
    x['roc'] = df['roc'] / 100.0
    x['cci_14'] = df['cci_14'] / 200.0
    x['willr_14'] = df['willr_14'] / 100.0
    x['bb_position'] = safe_div(df['close'] - df['lower_bb'], df['upper_bb'] - df['lower_bb'])
    x['bb_width'] = safe_div(df['upper_bb'] - df['lower_bb'], df['middle_bb'])

    daily_ret = df['close'].pct_change()
    x['volatility_7d'] = daily_ret.rolling(7).std()
    x['volatility_21d'] = daily_ret.rolling(21).std()
    x['momentum_7d'] = df['close'].pct_change(7)
    x['momentum_21d'] = df['close'].pct_change(21)
    x['volume_z_21d'] = (
        (np.log1p(df['volume']) - np.log1p(df['volume']).rolling(21).mean())
        / np.log1p(df['volume']).rolling(21).std().replace(0, np.nan)
    )
    return x


def exogenous_features(target_dates: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame({'date': pd.to_datetime(target_dates)}).set_index('date')

    for asset in ASSETS:
        path = ROOT / '1.used_data' / f'{asset}_ohlcv_data.csv'
        raw = pd.read_csv(path, parse_dates=['Date']).sort_values('Date')
        close = raw.set_index('Date')['Closing Price'].astype(float)
        frame = pd.DataFrame(index=close.index)

        # Shift one observed market row so the feature is known before the target date.
        frame[f'{asset.lower()}_ret_1d_lag1'] = close.pct_change().shift(1)
        frame[f'{asset.lower()}_ret_5d_lag1'] = close.pct_change(5).shift(1)
        frame[f'{asset.lower()}_vol_10d_lag1'] = close.pct_change().rolling(10).std().shift(1)

        # Reindex to the BTC calendar and carry only the latest already-known market observation.
        frame = frame.reindex(result.index.union(frame.index)).sort_index().ffill().reindex(result.index)
        result = result.join(frame)

    return result.reset_index(drop=True)


def prepare() -> pd.DataFrame:
    df = pd.read_csv(BTC_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    x_btc = btc_features(df)
    x_macro = exogenous_features(df['date'])

    combined = pd.concat([x_btc.reset_index(drop=True), x_macro.reset_index(drop=True)], axis=1)
    combined['date'] = df['date']
    combined['target_return'] = df['return']
    combined['label_3class'] = df['label'].astype(int)
    combined = combined.dropna().reset_index(drop=True)
    return combined


def models(binary: bool):
    class_weight = 'balanced'
    return [
        ('dummy', DummyClassifier(strategy='most_frequent'), False),
        ('logistic', Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(C=0.15, class_weight=class_weight, max_iter=4000, random_state=RANDOM_STATE)),
        ]), False),
        ('random_forest', RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=7,
            class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1,
        ), False),
        ('extra_trees', ExtraTreesClassifier(
            n_estimators=500, max_depth=9, min_samples_leaf=6,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
        ), False),
        ('hist_gradient_boosting', HistGradientBoostingClassifier(
            learning_rate=0.04, max_iter=350, max_leaf_nodes=15,
            l2_regularization=3.0, random_state=RANDOM_STATE,
        ), True),
    ]


def metric(y_true, pred, prob=None) -> dict:
    out = {
        'accuracy': accuracy_score(y_true, pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, pred),
        'macro_f1': f1_score(y_true, pred, average='macro', zero_division=0),
    }
    if prob is not None and len(np.unique(y_true)) == 2:
        out['roc_auc'] = roc_auc_score(y_true, prob)
    return out


def split_indexes(n: int):
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return np.arange(0, train_end), np.arange(train_end, val_end), np.arange(val_end, n)


def fit(model, X, y, weighted: bool):
    if weighted:
        model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    else:
        model.fit(X, y)
    return model


def evaluate_task(frame: pd.DataFrame, *, task: str, feature_columns: list[str]) -> dict:
    if task == 'three_class':
        task_df = frame.copy()
        y = task_df['label_3class'].astype(int).reset_index(drop=True)
        binary = False
    elif task == 'actionable_direction':
        task_df = frame[frame['label_3class'] != 0].reset_index(drop=True)
        y = (task_df['label_3class'] == 1).astype(int)
        binary = True
    elif task == 'actionable_move':
        task_df = frame.copy()
        y = (task_df['target_return'].abs() > 0.01).astype(int).reset_index(drop=True)
        binary = True
    else:
        raise ValueError(task)

    X = task_df[feature_columns].reset_index(drop=True)
    train_idx, val_idx, test_idx = split_indexes(len(X))

    validation_rows = []
    candidates = []
    for name, prototype, weighted in models(binary):
        fitted = fit(clone(prototype), X.iloc[train_idx], y.iloc[train_idx], weighted)
        pred = fitted.predict(X.iloc[val_idx])
        row = {'model': name, **metric(y.iloc[val_idx], pred)}
        validation_rows.append(row)
        candidates.append((row['macro_f1'], name, prototype, weighted))

    _, selected_name, selected_proto, selected_weighted = max(candidates, key=lambda x: x[0])
    train_val = np.concatenate([train_idx, val_idx])
    selected = fit(clone(selected_proto), X.iloc[train_val], y.iloc[train_val], selected_weighted)
    pred = selected.predict(X.iloc[test_idx])
    prob = None
    if binary and hasattr(selected, 'predict_proba'):
        prob = selected.predict_proba(X.iloc[test_idx])[:, 1]
    test_metric = metric(y.iloc[test_idx], pred, prob)

    dummy = DummyClassifier(strategy='most_frequent').fit(X.iloc[train_val], y.iloc[train_val])
    dummy_pred = dummy.predict(X.iloc[test_idx])
    dummy_metric = metric(y.iloc[test_idx], dummy_pred)

    selective = []
    if binary and hasattr(selected, 'predict_proba'):
        probs = selected.predict_proba(X.iloc[test_idx])
        confidence = probs.max(axis=1)
        for threshold in np.arange(0.55, 0.91, 0.05):
            mask = confidence >= threshold
            if mask.sum() < 20:
                continue
            selective.append({
                'threshold': round(float(threshold), 2),
                'coverage': float(mask.mean()),
                'n': int(mask.sum()),
                **metric(y.iloc[test_idx][mask], pred[mask]),
            })

    return {
        'task': task,
        'selected_model': selected_name,
        'n': len(task_df),
        'test_start': str(task_df['date'].iloc[test_idx[0]].date()),
        'test': test_metric,
        'dummy': dummy_metric,
        'gain_accuracy': test_metric['accuracy'] - dummy_metric['accuracy'],
        'gain_balanced_accuracy': test_metric['balanced_accuracy'] - dummy_metric['balanced_accuracy'],
        'gain_macro_f1': test_metric['macro_f1'] - dummy_metric['macro_f1'],
        'validation': validation_rows,
        'selective': selective,
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare()
    excluded = {'date', 'target_return', 'label_3class'}
    all_features = [c for c in frame.columns if c not in excluded]
    btc_only = [c for c in all_features if not any(c.startswith(asset.lower() + '_') for asset in ASSETS)]

    experiments = []
    for feature_set_name, columns in [('btc_only', btc_only), ('btc_plus_cross_market', all_features)]:
        for task in ('three_class', 'actionable_direction', 'actionable_move'):
            result = evaluate_task(frame, task=task, feature_columns=columns)
            result['feature_set'] = feature_set_name
            result['feature_count'] = len(columns)
            experiments.append(result)

    summary_rows = []
    for exp in experiments:
        summary_rows.append({
            'feature_set': exp['feature_set'],
            'task': exp['task'],
            'selected_model': exp['selected_model'],
            'n': exp['n'],
            'feature_count': exp['feature_count'],
            **{f'test_{k}': v for k, v in exp['test'].items()},
            **{f'dummy_{k}': v for k, v in exp['dummy'].items()},
            'gain_accuracy': exp['gain_accuracy'],
            'gain_balanced_accuracy': exp['gain_balanced_accuracy'],
            'gain_macro_f1': exp['gain_macro_f1'],
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_DIR / 'extended_summary.csv', index=False)
    (RESULT_DIR / 'extended_results.json').write_text(json.dumps(experiments, indent=2), encoding='utf-8')

    print('# Extended Research Summary')
    print(summary.to_string(index=False))
    print('\n# Best selective binary slices (coverage >= 20%)')
    for exp in experiments:
        eligible = [row for row in exp['selective'] if row['coverage'] >= 0.20]
        if not eligible:
            continue
        best = max(eligible, key=lambda row: (row['balanced_accuracy'], row['coverage']))
        print(exp['feature_set'], exp['task'], best)


if __name__ == '__main__':
    main()
