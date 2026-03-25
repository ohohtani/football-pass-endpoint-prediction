# Football Pass Endpoint Prediction Pipeline

단일 스크립트로 작성했던 축구 패스 종점 예측 코드를, 역할 기준으로 분리한 버전입니다.  
기존 로직은 최대한 유지하면서도, `설정 / 피처 엔지니어링 / 모델 학습 / 추론` 흐름이 보이도록 구조를 정리했습니다.

## Overview

이 프로젝트는 경기 이벤트 시퀀스를 기반으로 각 `game_episode`의 패스 종점 좌표(`end_x`, `end_y`)를 예측하는 파이프라인입니다.

모델 구조는 2단계로 구성되어 있습니다.

1. **Step 1**: AutoGluon으로 `delta_x`, `delta_y` 예측  
2. **Step 2**: CatBoost로 Step 1의 residual을 추가 보정  
3. 각 fold별 모델 성능을 비교한 뒤, 상위 fold 예측을 평균 앙상블하여 최종 제출 파일 생성

즉, 절대 좌표를 바로 예측하는 대신,
- 먼저 시작 좌표 기준 이동량(`delta_x`, `delta_y`)을 예측하고
- 이후 residual correction으로 오차를 한 번 더 줄이는 구조입니다.

---

## Project Structure

```bash
project/
├─ data/
│  ├─ train.csv
│  ├─ test.csv
│  ├─ match_info.csv
│  └─ sample_submission.csv
├─ outputs/                    # 학습 결과, 모델, fold 결과 저장
├─ src/
│  ├─ __init__.py
│  ├─ config.py                # 전역 설정값 및 경로
│  ├─ utils.py                 # 공통 유틸 함수
│  ├─ stats.py                 # player / team 통계 생성
│  ├─ features.py              # feature engineering
│  ├─ model_utils.py           # AutoGluon / CatBoost 관련 함수
│  ├─ train.py                 # 학습 파이프라인
│  └─ predict.py               # 추론 및 submission 생성
├─ run_all.py                  # 학습 + 추론 전체 실행
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## File Description

### `src/config.py`
- 데이터 경로
- random seed
- fold 설정
- feature 제거 목록
- Optuna trial 수
- output 경로

실험 시 가장 먼저 수정하게 되는 값들을 모아둔 파일입니다.

### `src/utils.py`
- seed 및 환경 설정
- inf / nan 처리
- categorical casting
- 좌표 정규화 및 로그 변환
- 후처리 함수

여러 단계에서 공통으로 쓰이는 보조 함수들을 분리했습니다.

### `src/stats.py`
- 선수별 통계
- 팀별 통계

원본 이벤트 데이터에서 별도 통계 feature를 만들기 위한 로직입니다.

### `src/features.py`
- train/test 공통 feature engineering
- CV 분할에 필요한 helper
- x / y 예측용 feature column 분리

가장 많은 feature 생성 로직이 들어가는 핵심 모듈입니다.

### `src/model_utils.py`
- AutoGluon 학습
- CatBoost residual model 학습
- Optuna 기반 Step2 파라미터 탐색
- 모델 저장 / 로드
- 컬럼 정렬 및 Pool 생성

모델 관련 기능만 따로 모아서 관리합니다.

### `src/train.py`
전체 학습 파이프라인입니다.

주요 흐름:
- train 데이터 로드
- base feature 생성
- fold별 train/valid 분리
- Step1 AutoGluon 학습
- residual 계산
- Step2 CatBoost + Optuna 튜닝
- fold 성능 저장
- 상위 fold 정보 저장

### `src/predict.py`
학습 완료 후 저장된 모델을 이용해 test 데이터를 예측합니다.

주요 흐름:
- pipeline meta 로드
- test raw episode 로드
- feature engineering 적용
- Step1 예측
- Step2 residual correction
- top fold 평균 앙상블
- submission 저장

### `run_all.py`
학습부터 제출 파일 생성까지 한 번에 실행하는 진입점입니다.

---

## Modeling Pipeline

### Step 1: Base prediction
AutoGluon을 사용해 `label_delta_x`, `label_delta_y`를 각각 회귀로 예측합니다.

### Step 2: Residual correction
Step 1 예측값과 원래 정답의 차이(residual)를 CatBoost로 학습합니다.

이때,
- `dx`, `dy`를 각각 별도로 튜닝하고
- Optuna로 CatBoost 파라미터와 residual 반영 비율(`alpha_x`, `alpha_y`)을 함께 탐색합니다.

### Final prediction
최종 예측은 아래 형태입니다.

```text
final_x = start_x + (pred_dx + alpha_x * residual_dx)
final_y = start_y + (pred_dy + alpha_y * residual_dy)
```

이후 좌표 clipping 및 일부 conservative postprocessing을 적용합니다.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

아래 파일들을 `data/` 폴더에 넣습니다.

- `train.csv`
- `test.csv`
- `match_info.csv`
- `sample_submission.csv`

### 3. Run full pipeline

```bash
python run_all.py
```

### 4. Or run separately

학습만 실행:

```bash
python -m src.train
```

추론만 실행:

```bash
python -m src.predict
```

---

## Output Files

실행 후 다음 파일들이 생성됩니다.

- `outputs/model_split_2/` : Step1 AutoGluon 모델
- `outputs/model_2/` : Step2 CatBoost 모델
- `outputs/fold_results.csv` : fold별 성능 요약
- `outputs/pipeline_meta.json` : 추론용 메타 정보
- `submission.csv` : 최종 제출 파일

---
