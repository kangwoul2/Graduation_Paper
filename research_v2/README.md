# Research V2: Leakage-safe Market Direction Prediction

## 1. Why this second experiment exists

The original undergraduate experiment compared LSTM, GRU and Transformer models on a three-class next-day Bitcoin direction problem.

The target rule in the legacy notebook is:

```python
return_t_plus_1 = close.pct_change().shift(-1)
label = 1   if return_t_plus_1 > 0.01
label = -1  if return_t_plus_1 < -0.01
label = 0   otherwise
```

The first experiment produced useful model-comparison evidence, but a later audit found two evaluation-design issues in `4.standard.ipynb`:

1. `StandardScaler.fit_transform()` was applied before the train/test split.
2. `train_test_split(..., stratify=y)` randomly mixed observations from different time periods.

For ordinary IID data this pattern can be acceptable after careful preprocessing, but for financial time-series it weakens the validity of out-of-sample claims. Future-regime distribution information can enter preprocessing, and random splitting does not reproduce the real deployment question: **can a model trained on the past generalize to a later market regime?**

Research V2 therefore treats the original deep-learning results as a historical baseline and introduces a stricter evaluation protocol.

---

## 2. Research question

The revised question is not simply:

> Which neural network has the highest accuracy?

It is:

> After removing temporal leakage and using a chronological evaluation protocol, do technical indicators contain enough stable information to outperform a naive baseline, and can feature engineering or selective prediction improve the quality/coverage trade-off?

This changes the emphasis from architecture competition to **evidence quality**.

---

## 3. Causal chain of the redesign

```text
Weak absolute model scores
        ↓
Question whether model complexity is justified
        ↓
Audit the evaluation protocol
        ↓
Detect random split + scaler fitted before split
        ↓
Rebuild a leakage-safe chronological baseline
        ↓
Compare simple and nonlinear models
        ↓
Create stationary / relative features
        ↓
Evaluate by Macro-F1 + Balanced Accuracy
        ↓
Check walk-forward stability
        ↓
Measure confidence vs coverage
        ↓
Decide what can actually be claimed
```

The important point is that each new technique is introduced because the previous stage exposes a concrete limitation.

---

## 4. Evaluation protocol

### 4.1 Chronological 60 / 20 / 20

```text
Past                                             Future
|---------------- Train ----------------|---- Val ----|---- Test ----|
             60%                         20%           20%
```

- Train: fit model and preprocessing
- Validation: select model/feature set by Macro-F1
- Test: evaluate once after selection

The test partition is not reused to choose hyperparameters.

### 4.2 Walk-forward validation

A single holdout can depend heavily on one market regime. The selected candidate is therefore also evaluated with `TimeSeriesSplit`.

The mean score describes expected quality across folds, while the standard deviation is treated as a **stability metric**.

### 4.3 Metrics

| Metric | Why it is used |
|---|---|
| Accuracy | overall hit ratio |
| Balanced Accuracy | class-wise recall averaged equally |
| Macro-F1 | gives each class equal importance |
| Weighted F1 | reflects support-weighted quality |
| Per-class recall | reveals which market direction is ignored |
| Confusion Matrix | shows the structure of errors |
| Walk-forward mean/std | measures temporal stability |
| Coverage | fraction of days on which the model chooses to act |

A three-class market problem can be misleading when only Accuracy is reported, especially when the neutral class is dominant. Macro-F1 and Balanced Accuracy are therefore the main model-selection metrics.

---

## 5. Baselines before complexity

Research V2 intentionally adds simple models before another neural network.

```text
Dummy Majority
    ↓
Balanced Logistic Regression
    ↓
Balanced Random Forest
    ↓
Histogram Gradient Boosting
```

This answers an important engineering question:

> Does a more complex model actually add predictive value over a cheap baseline?

If a neural model cannot beat a simple leakage-safe baseline under the same protocol, model complexity is not justified by the evidence.

---

## 6. Feature engineering hypothesis

The legacy experiment uses many level-dependent indicators directly. Bitcoin price levels change substantially across market regimes, so Research V2 additionally creates relative features.

Examples:

```text
close / SMA - 1
close / EMA - 1
MACD / close
ATR / close
Bollinger-band position
Bollinger-band width
1 / 2 / 3 / 5 / 10-day historical returns
intraday body ratio
high-low range ratio
log-volume change
```

The hypothesis is that relative features are more portable across regimes than absolute indicator levels.

The existing `return` column is **not** used as a feature because the original labeling notebook defines it with `shift(-1)`, meaning it represents the next day's target return.

---

## 7. Selective prediction

A practical prediction system does not always need to emit a trading signal every day.

Research V2 therefore adds a confidence threshold experiment.

```text
all predictions
      ↓
model probability
      ↓
confidence >= threshold ?
     /                  \
   yes                  no
    ↓                    ↓
use prediction          abstain
```

The result is reported as a quality/coverage curve rather than hiding rejected observations.

For example, an increase in Accuracy is meaningful only together with the corresponding decrease in Coverage.

This prevents a misleading claim such as "accuracy improved" when the model simply predicts on a much smaller subset.

---

## 8. Reproduction

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements-research-v2.txt
python research_v2/run_research.py
```

Generated outputs:

```text
research_v2/results/
├── validation_model_selection.csv
├── model_comparison.csv
├── walk_forward.csv
├── selective_prediction.csv
├── confusion_matrix.csv
├── protocol.json
├── summary.md
├── model_comparison.png
└── selective_prediction.png
```

GitHub Actions runs the same script and uploads the result directory as an artifact.

---

## 9. Claim policy

The following rules are intentionally strict.

- Existing thesis numbers are never edited to look stronger.
- Research V2 numbers are generated only by `run_research.py`.
- The legacy and V2 metrics are not treated as directly comparable when their evaluation protocols differ.
- "Improvement" is reported against a baseline measured under the **same chronological protocol**.
- Confidence filtering must always report both quality and coverage.
- A single strong holdout result is not enough; walk-forward variance is also reported.

This policy makes the repository easier to defend in a technical interview because every number has a reproducible origin.
