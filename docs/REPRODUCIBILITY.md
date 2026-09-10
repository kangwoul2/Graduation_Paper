# Reproducibility Notes

## Source of truth

- 원문 연구 설명: 학사학위논문 PDF
- 모델 비교 숫자: `model_performance_results.csv`
- 연구 실행 흔적: numbered notebooks
- 저장 모델: `LSTM_model.h5`, `GRU_model.h5`, `Transformer_model.keras`

## Why notebooks are not rewritten

기존 notebook은 연구 당시의 탐색 과정과 실패 기록을 포함합니다. 결과를 더 깔끔하게 보이게 만들기 위해 실행 출력을 지우거나 결과를 바꾸지 않습니다.

## Reporting rule

README나 이력서에 성능 수치를 사용할 때는 `model_performance_results.csv`에 존재하는 값만 사용합니다. 새 실험을 수행한 경우 새 결과 파일을 별도로 추가하고 기존 결과와 구분합니다.

## Recommended future reproduction

1. 데이터 수집 시점과 ticker 목록을 config로 고정
2. train/validation/test를 시간 순서로 분리
3. scaler fit은 train partition에서만 수행
4. 모든 random seed 기록
5. model config를 YAML/JSON으로 분리
6. 한 command로 train/evaluate/report 생성
7. experiment tracking 도구로 parameter/metric lineage 관리
