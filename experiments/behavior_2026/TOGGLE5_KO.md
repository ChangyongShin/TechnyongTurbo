# 전원 조작 계열 5개 task 공동 학습

대상: 0(라디오), 66(조명 끄기), 77(모뎀 설치), 89(팩스 설치),
92(스캐너 설치). 설치 작업에는 전원 조작 외 배치 목표도 포함된다.
공동 학습이 task 0에 도움이 되는지 확인하는 실험이며 개선을 보장하지 않는다.

## 실험 조건

- 기존 task 0의 180/20 분할 파일을 그대로 읽고 변경하지 않는다.
- 다른 task도 seed 42로 에피소드 단위 90/10 분할한다.
- 데이터 revision은 기존 `turbovla_source.json`의 SHA로 고정한다.
- task를 같은 확률로 선택하고, 그 안의 프레임을 균등하게 뽑는다.
  정확히 매 배치 20%씩이 아니라 장기 기대값 20%다. 복원 추출이므로
  모든 에피소드를 학습 풀에 포함해도 짧은 실행에서 모든 프레임을 방문하지는 않는다.
- DINOv3/BERT 사전학습 가중치와 seed 42에서 fusion/action을 새로 초기화한다.
  기존 task 0 체크포인트를 resume/initial weights로 사용하지 않는다.
- batch 32, 누적 1, lr 1e-4, 증강 없음으로 시작한다.
- 첫 비교는 총 12,048step으로 계산량을 맞춘다. task 0 노출은 기존의 약 1/5다.
  task 0 노출량을 맞추는 장기 비교는 약 60,237step(385,512×5/32 올림)이며
  별도 실험으로 기록한다. manifest의 계산값을 사용한다.
- 여러 task를 섞어 계산한 정규화로 학습하므로, 정규화가 다른 기존 L1과
  새 `validation/task_000/l1_mean`을 직접 비교하지 않는다.

## 1. 준비 계획 확인

아스트라고의 기존 저장소와 모델·데이터·캐시가 유지되어 있어야 한다.

```bash
cd /workspace/TechnyongTurbo
source .venv-behavior/bin/activate
python -m turbovla.data.toggle_tasks --phase plan
```

메타데이터만 받아 task별 에피소드 수·학습 프레임·RGB 캐시 예상량을 출력하고
`experiments/behavior_2026/toggle5_split.json`을 저장한다. 저장공간과 워크로드
quota를 확인한다. 기존 task 0 캐시는 재사용하고 다른 4개 task는
`/workspace/data/behavior-2026-toggle-cache`에 저장한다. 캐시 예상량은 RGB만의 값이다.

## 2. 다운로드·전처리 (백그라운드)

```bash
mkdir -p logs
nohup bash -c 'set -e; source .venv-behavior/bin/activate; python -u -m turbovla.data.toggle_tasks --phase download; python -u -m turbovla.data.toggle_tasks --phase prepare' \
  > logs/toggle5-prepare.log 2>&1 < /dev/null &
echo "준비 PID: $!"
tail -f logs/toggle5-prepare.log
```

기존 캐시는 metadata·원본 revision·task·길이·해상도를 검사하고 건너뛴다.
중단으로 생긴 metadata 없는 불완전 디렉터리는 삭제하지 않고 오류로 보고한다.
새로 생성한 캐시도 오류 없이 전체 task 준비가 완료됐는지 로그를 확인한다.
`Ctrl+C`는 tail 보기만 중단한다.

## 3. 짧은 실행 확인

준비가 끝난 뒤 실행한다. 5개 task 전체 검증을 하므로 50step 학습보다
마지막 검증이 오래 걸릴 수 있다. 학습 시간과 검증 시간을 구분한다.

```bash
unset RESUME_CKPT INITIAL_WEIGHTS WANDB_NAME
export WANDB_ENTITY=scy3942-seoul-national-university-ofscience-and-technology
export RUN_DIR=outputs/toggle5-check
nohup env STEPS=50 BATCH_SIZE=32 GRAD_ACCUMULATION=1 \
  bash scripts/behavior_2026/train_toggle5.sh \
  > logs/toggle5-check.log 2>&1 < /dev/null &
tail -f logs/toggle5-check.log
```

baseline `outputs/task0-batch32-epoch1/config.json`이 필요하다. 다른 위치라면
`REFERENCE_CONFIG` 환경변수로 경로를 지정한다.

## 4. 첫 공동 학습

짧은 실행 성공 후 새 output에서 같은 초기화로 시작한다. 체크포인트 50step을
재개하지 않으므로 초기화부터의 비교가 명확하다.

```bash
unset RESUME_CKPT INITIAL_WEIGHTS WANDB_NAME
export RUN_DIR=outputs/toggle5-compute12048
nohup env STEPS=12048 SAVE_EVERY=3012 BATCH_SIZE=32 GRAD_ACCUMULATION=1 \
  bash scripts/behavior_2026/train_toggle5.sh \
  > logs/toggle5-compute12048.log 2>&1 < /dev/null &
echo "학습 PID: $!"
tail -f logs/toggle5-compute12048.log
```

## W&B와 비교 척도

- `train/task_XXX_sample_fraction`: 현재 프로세스가 뽑은 task별 누적 비율
- `validation/task_000/l1_mean` 등: 공동 정규화 기준 task별 검증 평균
- `validation/task_macro_l1_mean`: task별 평균의 산술평균, 긴 task 독점 방지
- `validation/l1_mean`: 전체 유효 행동 시점 기준 평균, 긴 task 비중이 큼
- `validation/task_000/reference_l1_mean`: 기존 config의 action 범위로 평가한 값

reference 평가는 prediction을 현재 범위에서 원시 행동으로 선형 복원하고,
고정 기준 범위로 다시 변환한다. 예측은 원래 학습 손실처럼 clip하지 않고 정답은
기존 정규화 규칙대로 clip한다. 기준 범위가 상수인 축은 reference 지표에서 0으로
제외된다. 따라서 기존 지표에 상수 축의 잔여 예측 오차가 포함됐다면 기존
`0.029268`과 정확히 같은 정의가 아니다. 단독 모델도 같은 evaluator로 재평가해
엄밀히 비교해야 한다. 기준 min/max 초과 정답 clipping 비율과 물리 단위 축별
오차도 후속 평가에서 확인한다. reference metric은 학습 loss에 사용하지 않는다.

task별 검증 출력은 정답 시연에 대한 행동 일치도다. 전원 조작 성공률은 별도의
RTX OmniGibson 평가가 필요하다. 아직 실제 5개 task GPU 실행 결과는 없다.
