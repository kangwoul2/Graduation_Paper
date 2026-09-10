# Model Evaluation

`model_performance_results.csv` 기준 결과입니다.

| Model | Test Loss | Test Accuracy | Custom Accuracy | Test F1 |
|---|---:|---:|---:|---:|
| LSTM | 1.492900 | 0.297872 | 0.375000 | 0.257359 |
| GRU | 1.505537 | 0.373860 | 0.532051 | 0.373038 |
| Transformer | 1.134376 | 0.331307 | 0.544000 | 0.325961 |

## Interpretation

### Test Accuracy
GRU > Transformer > LSTM

### Test F1
GRU > Transformer > LSTM

### Custom Accuracy
Transformer > GRU > LSTM

### Test Loss
Transformer < LSTM < GRU

한 모델이 모든 지표에서 우세하지 않으므로 모델 선택 기준을 먼저 정의해야 합니다.

## Interview angle

이 결과는 백엔드 성능 측정과도 같은 교훈을 줍니다. 평균 latency만 낮아졌다고 성공으로 판단하지 않고 throughput, p95/p99, error rate와 함께 평가해야 합니다. 즉 **측정 지표는 시스템이 최적화하려는 사용자 경험/비즈니스 오류 비용과 연결되어야 합니다.**
