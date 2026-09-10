# Research Evaluation Decisions

## RD-001. Temporal split before model tuning

금융 시계열의 production inference는 과거로 미래를 예측합니다. random split은 이 시간축을 깨므로 Research V2는 chronological train/validation/test split을 사용합니다.

## RD-002. Preprocessing fit은 train only

Scaler와 feature transform parameter는 train subset에서만 학습하고 validation/test에는 transform만 적용합니다. 미래 분포가 preprocessing parameter에 들어가는 leakage를 막기 위함입니다.

## RD-003. Dummy baseline을 모든 task에 포함

Accuracy 숫자만으로 signal을 판단하지 않습니다. majority dummy와 Accuracy, Balanced Accuracy, Macro F1을 함께 비교합니다.

## RD-004. Original study 결과는 삭제하지 않음

원래 LSTM/GRU/Transformer 결과는 역사적 연구 결과로 보존합니다. Research V2와 평가 protocol이 다르므로 동일 조건의 before/after처럼 비교하지 않습니다.

## RD-005. Cross-market feature는 ablation으로 검증

ETF/Gold feature를 사용한 이유를 설명만 하지 않고 BTC-only와 동일 task에서 직접 비교합니다. 3-class에서는 개선되지 않았고 actionable direction에서는 개선되었습니다. 실패한 가설도 결과로 유지합니다.

## RD-006. Selective prediction은 coverage를 함께 보고

confidence threshold를 높이면 accuracy가 오를 수 있지만 예측 표본 수가 줄어듭니다. 따라서 selective result는 Accuracy 단독이 아니라 Coverage, Balanced Accuracy, Macro F1과 함께 기록합니다.

## RD-007. Trading profitability를 주장하지 않음

분류 accuracy는 실제 전략 수익률과 동일하지 않습니다. transaction cost, slippage, position sizing, risk control을 포함한 backtest가 없으므로 투자 성과를 주장하지 않습니다.

## RD-008. Reproducibility를 CI에서 검증

GitHub Actions가 strict temporal benchmark와 extended cross-market benchmark를 직접 실행하고 artifact를 생성합니다. README의 핵심 V2 수치는 successful workflow run의 출력에서 고정 기록합니다.
