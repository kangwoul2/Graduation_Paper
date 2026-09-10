# Verified Research V2 Results

이 문서는 저장소의 실제 데이터와 `research_v2/` 코드를 GitHub Actions에서 실행한 결과를 고정 기록합니다.

- Workflow: `research-v2`
- Successful run: `34449931162`
- Commit evaluated: `d81f044c91f41e65ff1e4758b530033067e2d1b9`
- Environment: GitHub Actions Ubuntu runner / Python 3.11
- Result policy: 측정되지 않은 수치나 수동으로 개선한 수치를 기록하지 않음

## 1. Leakage-safe three-class benchmark

시간 순서를 보존하고 train-only preprocessing을 사용한 첫 재검증에서 선택된 구성은 다음과 같습니다.

- Feature set: `stationary_v2`
- Model: `HistGradientBoosting`
- Accuracy: `0.3249`
- Balanced Accuracy: `0.3173`
- Macro F1: `0.3127`
- Walk-forward Macro F1 mean/std: `0.3485 / 0.0198`
- Majority dummy 대비 Macro F1 gain: `+0.1612`

이 결과는 절대적인 예측력이 강하다는 의미가 아닙니다. 오히려 모든 날짜를 강제로 3-class로 예측하는 문제가 어렵다는 것을 보여줍니다.

## 2. Selective prediction

confidence가 낮은 예측을 거절하는 방식에서 다음 trade-off가 관찰됐습니다.

- confidence threshold: `0.65`
- coverage: `0.3193`
- accuracy: `0.4211`

즉 전체 날짜 중 약 31.9%만 예측하는 대신 accuracy가 약 42.1%로 높아졌습니다. 이는 모델을 모든 날짜에 강제로 사용하는 것보다 **high-conviction prediction의 품질/coverage trade-off를 별도로 평가해야 한다**는 근거가 되었습니다.

## 3. Extended benchmark

두 번째 실험에서는 세 가지 질문을 분리했습니다.

1. `three_class`: 하락 / 중립 / 상승 3-class
2. `actionable_direction`: ±1% 이상 움직인 날에서 상승/하락 방향
3. `actionable_move`: 큰 움직임이 발생하는 날인지 여부

또한 BTC-only feature와 ETF/Gold를 포함한 cross-market feature를 비교했습니다.

### Best verified results

| Task | Feature set | Model | Accuracy | Balanced Acc. | Macro F1 | Dummy Macro F1 |
|---|---|---|---:|---:|---:|---:|
| Three-class | BTC only | ExtraTrees | **0.4000** | **0.3875** | **0.3844** | 0.1511 |
| Actionable direction | BTC + cross-market | RandomForest | **0.5459** | **0.5482** | **0.5451** | 0.3245 |
| Actionable move | BTC + cross-market | Logistic | 0.5352 | **0.5693** | **0.5347** | 0.3783 |

### Three-class interpretation

`BTC-only + ExtraTrees`는 dummy 대비:

- Accuracy: `+0.1070`
- Balanced Accuracy: `+0.0542`
- Macro F1: `+0.2333`

을 기록했습니다.

원래 논문에서 보고된 GRU Test Accuracy 0.3739와 새 0.4000을 직접적인 2.61%p 개선이라고 주장하지 않습니다. **원래 실험은 random split과 전체 데이터 preprocessing을 사용했고, Research V2는 chronological split과 train-only preprocessing을 사용하기 때문에 평가 조건이 다릅니다.**

따라서 새 결과의 의미는 "기존 숫자를 더 높였다"가 아니라 **더 엄격한 평가 프로토콜에서도 dummy baseline보다 명확히 높은 성능을 재현했다**는 데 있습니다.

### Cross-market interpretation

3-class에서는 cross-market feature가 BTC-only보다 좋아지지 않았습니다.

- BTC-only three-class Macro F1: `0.3844`
- BTC + cross-market three-class Macro F1: `0.3772`

반면 actionable direction에서는 cross-market feature가 도움이 됐습니다.

- BTC-only Macro F1: `0.5189`
- BTC + cross-market Macro F1: `0.5451`

따라서 "외부 시장 데이터를 추가하면 항상 좋아진다"고 결론 내리지 않습니다. **feature value는 task에 따라 달라졌습니다.**

## 4. Selective binary slices

Coverage 20% 이상 조건에서 확인된 주요 selective slice:

| Feature / Task | Threshold | Coverage | Accuracy | Balanced Acc. | Macro F1 |
|---|---:|---:|---:|---:|---:|
| BTC only / actionable direction | 0.55 | 0.6026 | 0.5362 | 0.5348 | 0.5347 |
| BTC only / actionable move | 0.65 | 0.2310 | **0.6463** | **0.6494** | **0.6420** |
| Cross-market / actionable direction | 0.55 | 0.5109 | 0.4957 | 0.5053 | 0.4920 |
| Cross-market / actionable move | 0.60 | 0.4986 | 0.5254 | 0.5651 | 0.5199 |

특히 `BTC-only actionable_move`는 coverage 약 23.1%에서 accuracy 약 64.6%를 기록했습니다. 이는 전체 날짜 예측 성능과 직접 비교할 수 있는 동일한 문제는 아니며, **모델 confidence를 활용해 예측을 포기할 수 있을 때의 trade-off**를 보여주는 보조 실험입니다.

## 5. Conclusion

Research V2의 결론은 다음과 같습니다.

1. 금융 시계열 3-class 방향 예측은 단순 accuracy 하나로 강한 성능을 주장하기 어렵다.
2. random split보다 chronological split과 train-only preprocessing이 더 타당한 평가 프로토콜이다.
3. dummy baseline을 반드시 함께 제시해야 모델의 실제 추가 가치를 판단할 수 있다.
4. cross-market feature는 모든 task에 일관되게 이득을 주지 않았지만 actionable direction에서는 개선을 보였다.
5. confidence를 활용한 selective prediction은 품질과 coverage 사이의 명확한 trade-off를 만들었다.
6. 결과를 좋게 보이게 만들기 위해 숫자를 수정하는 대신, **평가 설계를 개선하고 실패한 가설도 결과로 남기는 것이 연구의 핵심 개선**이었다.
