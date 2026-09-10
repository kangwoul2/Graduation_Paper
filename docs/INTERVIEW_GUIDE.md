# Market Prediction Research Interview Guide

## 1. 기존 연구 결과가 왜 충분하지 않다고 판단했나요?

모델 숫자보다 평가 protocol을 다시 봤습니다. 원본 경로는 전체 데이터에 scaler를 fit한 뒤 random split을 사용하는 구조라 금융 시계열의 실제 inference 상황과 다릅니다. 따라서 모델을 더 튜닝하기 전에 chronological split과 train-only preprocessing으로 평가 자체를 수정했습니다.

## 2. 왜 random split이 시계열에서 문제인가요?

실제 예측에서는 미래를 학습에 사용할 수 없습니다. random split은 미래 날짜가 train에 들어가고 과거 날짜가 test에 들어갈 수 있어 temporal leakage와 과도하게 낙관적인 평가를 만들 수 있습니다.

## 3. scaler도 leakage를 만들 수 있나요?

네. 전체 데이터 평균/분산으로 scaler를 fit하면 test period의 분포가 train preprocessing parameter에 반영됩니다. 그래서 split 후 train에만 fit하고 validation/test에는 transform만 적용해야 합니다.

## 4. 왜 Accuracy 말고 Balanced Accuracy와 Macro F1을 같이 봤나요?

3-class label 비율이 동일하지 않으면 majority class만 잘 맞혀도 Accuracy가 높아질 수 있습니다. Balanced Accuracy는 class별 recall을 균등하게 보고, Macro F1은 각 class F1을 동일 가중으로 봅니다.

## 5. Dummy baseline이 왜 필요한가요?

모델 Accuracy가 40%라고 해도 majority dummy가 39%면 가치가 거의 없습니다. Research V2 three-class best는 Accuracy 0.4000, dummy 0.2930이고 Macro F1은 0.3844 vs 0.1511이어서 단순 class prior 이상의 signal이 있는지 비교할 수 있습니다.

## 6. 기존 GRU 37.39%와 새 ExtraTrees 40.00%는 성능 개선인가요?

같은 평가 protocol이 아니므로 직접적인 before/after 개선율로 주장하지 않습니다. 원본은 random split/global preprocessing이고 V2는 chronological/train-only preprocessing입니다. 새 결과의 의미는 더 엄격한 조건에서 dummy baseline보다 높은 성능을 재현했다는 것입니다.

## 7. 왜 Deep Learning이 아니라 ExtraTrees가 더 좋았나요?

engineered tabular feature에서는 tree ensemble이 nonlinear interaction을 효율적으로 포착할 수 있습니다. 데이터 크기와 feature representation이 sequence deep model에 반드시 유리한 조건은 아니었습니다. 모델 복잡성보다 validation performance로 선택했습니다.

## 8. Cross-market feature를 왜 넣었나요?

BTC 가격이 전통시장 위험선호와 함께 움직일 수 있다는 가설을 검증하기 위해 broad-market, sector, gold proxy를 lagged feature로 사용했습니다. 중요한 것은 설명만 한 것이 아니라 BTC-only와 ablation으로 비교했다는 점입니다.

## 9. Cross-market 결과는 좋았나요?

모든 task에서 그렇지는 않았습니다. three-class에서는 BTC-only Macro F1 0.3844가 cross-market 0.3772보다 높았습니다. 반면 ±1% 이상 움직인 날의 방향 task에서는 cross-market Macro F1 0.5451이 BTC-only 0.5189보다 높았습니다. 그래서 feature 효과는 task-dependent라고 결론냈습니다.

## 10. Selective prediction이 무엇인가요?

모델 confidence가 낮으면 예측을 포기하고 높은 경우에만 action을 내는 방식입니다. accuracy만 높여 보이게 하지 않기 위해 coverage를 반드시 같이 제시합니다.

## 11. 64.63% Accuracy는 어떤 의미인가요?

BTC-only actionable-move binary task에서 confidence 0.65 이상인 약 23.1%의 test sample만 선택했을 때의 Accuracy입니다. 전체 sample prediction 정확도가 아닙니다. 그래서 coverage 23.1%, Balanced Accuracy 64.94%, Macro F1 64.20%를 함께 보고합니다.

## 12. Walk-forward validation은 왜 하나요?

단일 split이 특정 시장 regime에 우연히 맞았을 수 있기 때문입니다. 시간축을 앞으로 이동하며 여러 fold를 평가하면 성능의 평균과 변동성을 확인할 수 있습니다. 첫 V2 benchmark의 Macro F1 mean/std는 0.3485/0.0198이었습니다.

## 13. 이 모델로 실제 투자할 수 있나요?

그렇게 주장하지 않습니다. prediction metric과 trading profitability는 다릅니다. transaction cost, slippage, execution delay, position sizing, drawdown을 포함한 strategy backtest가 추가로 필요합니다.

## 14. 연구에서 가장 크게 배운 것은 무엇인가요?

좋은 모델을 찾는 것보다 **좋은 평가를 설계하는 것이 먼저**라는 점입니다. leakage를 막고, baseline을 두고, 실패한 feature hypothesis도 남기고, confidence와 coverage를 함께 보면서 결론을 좁혔습니다.

## 30-second answer

> 학사 연구에서는 Bitcoin 방향 예측에 LSTM, GRU, Transformer를 비교했는데, 포트폴리오 정리 과정에서 전체 데이터 scaler fit과 random split이 금융 시계열 평가에 적절하지 않다는 문제를 발견했습니다. 그래서 원본 결과는 보존하되 chronological split, train-only preprocessing, dummy baseline, walk-forward validation을 적용한 Research V2를 만들었습니다. GitHub Actions에서 실제 데이터를 재실행한 결과 BTC-only ExtraTrees가 three-class Accuracy 40.0%, Macro F1 0.384를 기록했고 dummy보다 명확히 높았습니다. Cross-market feature는 모든 task에서 좋아지지는 않았지만 ±1% 이상 움직인 날의 방향 분류에서는 Macro F1이 0.545까지 올라갔습니다. 성능 숫자를 바꾸기보다 평가 설계를 개선하고 어떤 가설이 언제 유효한지 검증한 연구입니다.
