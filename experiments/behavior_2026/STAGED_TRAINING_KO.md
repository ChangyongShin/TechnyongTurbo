# 시각·언어 → 동작 판단 → 제어: 별도 실험 경로

## 구현 범위

기존 TurboVLA는 시각·언어 융합 특징과 proprioception에서 행동을 바로 예측한다.
새 `--staged` 경로는 다음 계산을 명시적으로 구성한다.

```text
카메라 3개 + 지시
  → DINOv3/BERT·기존 융합 모듈의 시각/언어 특징
  → 이동 / 회전 / 조작 / 유지 판단 확률
  → 판단 확률을 control memory token으로 전달
  → 현재 상태와 함께 23차원 absolute/controller 행동 chunk 예측
```

**라디오 탐지기나 완성된 탐색 planner를 구현한 것은 아니다.** 현재 캐시에 target
visibility·2D 위치·3D 목표 위치의 정답이 없다. 시각 특징 학습과 목표 물체 인지의
검증을 혼동하지 않는다. 향후 실제 TRAIN/DEV target 가시성·위치 라벨 또는 공식
skill annotation을 확보하면 별도 supervision을 추가해야 한다. test 영상을 정답
생성이나 학습에 사용하지 않는다.

판단 head는 state를 직접 입력받지 않고 시각/언어 특징으로 예측한다. 제어기는
predicted probability token과 state를 함께 받는다. 학습 때도 GT 단계 token을 넣지
않으므로 추론에서만 판단을 사용하게 되는 teacher forcing 불일치를 피한다.
단계 확률은 제어기의 조건일 뿐 강제적 stop/navigation gate가 아니다.

## 전처리 보완

원본 RGB·state·action 캐시를 덮어쓰지 않는다. `PhaseDataset`이 저차원 상태·행동에서
단계 라벨을 계산한다. 이미지 디코딩이나 대형 캐시를 다시 만들 필요는 없다.
단계는 의미적 search/approach/success가 아닌 **동작 기반 proxy**다.

우선순위: XY 명령 norm≥0.01이면 navigate, 아니면 yaw 절댓값≥0.02이면 turn,
아니면 팔·몸통 목표와 현재 자세 차이≥0.01 또는 gripper 명령 변화≥0.01이면
manipulate, 나머지는 hold. 단위는 raw controller input이며 임계값은 실험 시작값이다.
base 이동과 회전이 동시에 있으면 navigate로 표시하지만 행동 target은 양쪽 모두
보존한다. 정지 접촉, 작업 성공, 물체 가시성을 이 규칙으로 판단하지 않는다.
gripper 변화량 계산은 episode마다 초기화한다.

구간 보고서:

```bash
python -m turbovla.data.stages --cache /workspace/data/task0-audit-cache --episodes 0 1 --out outputs/stage-profile-01.json
```

학습 샘플러는 이용 가능한 단계를 균등 확률로 선택한 뒤 해당 프레임을 선택한다.
정지 구간을 삭제하지 않는다. `--task-balanced --phase-balanced`를 함께 쓰면
task를 균등하게 고른 뒤 해당 task의 단계들을 균등하게 고른다. 이는 복원 추출이며
1회 sampler 순회가 전체 데이터 방문을 보장하지 않는다. 검증은 원래 분포 전체를
순회한다. episode split과 TRAIN 전용 정규화는 유지한다.

## 학습 보완

- 초기 일부 step에서는 단계 cross entropy만 학습할 수 있다.
- 이후 제어 L1 + 0.1×단계CE + 0.1×chunk내 행동 변화량 오차를 공동 학습한다.
- 제어 L1은 base·몸통·양팔·양그리퍼 각 그룹 평균을 같은 비중으로 계산한다.
- 변화량 오차는 예측 chunk 차분과 전문가 chunk 차분을 비교하며, 정상 움직임을
  무조건 평평하게 만드는 smoothing이 아니다. 에피소드 끝의 무효 pair는 제외한다.
- 이 항목들의 이득은 아직 검증되지 않았다. 통합 후보 경로를 우선 점검한 뒤
  `--staged`, `--phase-balanced`, `--group-balanced-loss`, `--temporal-weight`를
  분리한 실험으로 개선 원인을 평가해야 한다.
- 새 모델은 decision head가 추가돼 기존 체크포인트와 구조가 다르다. 기본 launcher는
  baseline 가중치를 resume하지 않고 공통 pretrained backbone과 seed42에서 시작한다.
- 기존 모델에는 새 옵션을 기본 적용하지 않는다. 예전 checkpoint는 그대로 로딩된다.

## 기록·판정

`train/control_l1`, `train/decision_ce`, `train/temporal_delta_l1`, `train/loss_total`을
분리 기록한다. `train/l1`은 control loss이며 staged 경로에서는 그룹 균등 L1이다.
과거 좌표 균등 L1과 정의가 다르므로 곧바로 비교하지 않는다. 판단 사전학습 때는
control L1이 최적화되지 않으므로 그 구간 제어 성능을 좋다고 해석하지 않는다.

검증에는 원래 23축 오차, task별 오차, 단계별·그룹별 L1, decision accuracy와
confusion matrix를 기록한다. 단계는 원시 명령에서 만든 proxy여서 높은 accuracy가
물체 인식 정확도를 의미하지 않는다. `reference` 척도의 상수 축 처리 차이도
이전 TOGGLE5 안내대로 적용된다.

정책 서버는 새 checkpoint config를 읽어 같은 판단·제어 경로로 추론한다.
trace에는 decision_probabilities가 추가된다. 판단 head만 학습한 중간 checkpoint를
작업 평가에 사용하지 않는다. 최종 모델도 DEV에서 이동·회전·조작·실패 회복과
실제 성공률을 검사하기 전에는 개선됐다고 주장하지 않는다.

## 처음 실행: 작은 기능 점검

아스트라고의 audit worktree에서 최신 커밋을 받은 후, 기존 Python을 사용한다.
검증용 episode2 캐시가 없다면 원본에서 episode2만 별도 전처리한다.

```bash
PYTHONPATH="$PWD" /workspace/TechnyongTurbo/.venv-behavior/bin/python -m turbovla.data.prepare --root /workspace/data/behavior-2026-rgb --episodes 2 --out /workspace/data/task0-audit-cache --image-size 224 --format mmap
```

이미 episode2가 있으면 위 명령은 생략한다. 불완전 캐시는 자동 삭제하지 않는다.

```bash
/workspace/TechnyongTurbo/.venv-behavior/bin/python scripts/behavior_2026/train_staged.py --cache /workspace/data/task0-audit-cache --baseline-config /workspace/TechnyongTurbo/outputs/task0-batch32-epoch1/config.json --out /workspace/TechnyongTurbo/outputs/task0-staged-smoke01 --smoke --steps 2 --save-every 2 --batch-size 2
```

이 실행은 TRAIN0·1 / VAL2만의 기능 점검이며 full-task 학습이 아니다. baseline config가
다른 split이면 그 파일의 처음 두 TRAIN과 처음 한 VAL을 사용하고 실제 ID를 출력한다.
launcher는 cached task_id·episode_id·원본 revision을 확인한다.

## task 0 전체 에피소드로 전환

`scripts/behavior_2026/run_task0_full_staged.sh`는 기존 baseline config에 저장된
180개 TRAIN / 20개 validation 분할을 그대로 사용한다. 기본 설정은 배치 64,
5,000 optimizer step, 1,000 step마다 검증·저장, 최초 100 step 판단 CE 사전학습이다.
기존 소규모 실험의 가중치를 이어받지 않고 새 모델을 학습한다.

먼저 `prepare_task0_full.py`가 원본 revision과 에피소드 메타데이터를 검사한다.
`/workspace/data/task0-audit-cache`의 완성된 224px RGB 캐시는 재사용하고 나머지만
전처리한다. 전체 RGB 캐시는 약 180.8 GiB이며 상태·행동과 여유 공간이 추가로 필요하다.
백업을 가리키는 원본 데이터 경로에는 쓰지 않는다. 원본 영상이 없거나 기존 캐시가
불완전하면 파일을 삭제하지 않고 중단한다. 전처리 및 전체 200개 검사에 성공해야
학습으로 넘어가므로, 캐시 준비 중 조기 학습하는 문제를 방지한다.

```bash
mkdir -p /workspace/TechnyongTurbo/logs
unset WANDB_NAME
nohup bash scripts/behavior_2026/run_task0_full_staged.sh > /workspace/TechnyongTurbo/logs/task0-staged-full-b64-5000.log 2>&1 < /dev/null &
tail -f /workspace/TechnyongTurbo/logs/task0-staged-full-b64-5000.log
```

기존 소규모 학습은 종료한 뒤 새 학습을 시작한다. `tail -f`는 Ctrl+C로 종료해도
백그라운드 작업은 계속된다. 서버 워크로드 종료·재시작까지 견디는 기능은 아니다.
출력 폴더는 `/workspace/TechnyongTurbo/outputs/task0-staged-full-b64-5000`이다.
같은 출력 폴더에 결과가 있으면 실행을 거부하므로 재실험에는 새 `RUN_DIR`을 지정한다.
`STEPS`, `BATCH_SIZE`, `SAVE_EVERY`, `CACHE_ROOT`, `RUN_DIR`, `PYTHON_BIN`을 환경변수로
바꿀 수 있다.

배치 64 × 5,000 step은 320,000개의 샘플 추출이다. TRAIN 385,512 프레임에 대한
단순 비율은 약 0.83이지만, 단계별 균등 복원 추출이므로 모든 프레임을 한 번씩
학습하는 epoch를 뜻하지 않는다. 180개 전체 학습 에피소드가 샘플링 대상이 된다.
동일한 검증 20개에서 오차와 이동·회전 단계 성능을 관찰한 뒤 학습량을 연장한다.

full-task는 `--smoke`를 빼고 180/20의 모든 캐시가 존재하는 경로를 지정한다.
현재 백업에서 캐시가 일부만 복원돼 있다면 전부 준비되기 전 실행하지 않는다.
이번 전체 데이터 실험은 5,000step으로 시작하고 검증 결과에 따라 연장을 결정한다.
이 분량은 공식 권장 학습량이나 최적값이 아니다.
