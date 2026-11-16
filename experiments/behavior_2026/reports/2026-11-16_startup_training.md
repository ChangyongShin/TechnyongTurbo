# 초반 이동·회전 학습 보완

## 목적과 한계

instance 301은 라디오가 영상 후반에 들어오지만, 이것만으로 모델이 그때부터 목표를 인식·탐색했다고 단정하지 않는다. 초반 명령의 정체, 잘못된 방향 선택, 목표 인식 실패를 구분해야 한다. 이번 변경은 **기존 전문가 시연의 초반 행동을 더 충실하게 학습하는 옵션**이다. 라디오 위치·가시성 정답이나 탐색 planner를 추가한 것은 아니다.

실제 학습 및 평가 서버 실행은 하지 않았다. 합성 배열과 mock으로 동작만 검증했다.

## 추가한 코드

- `turbovla/data/startup.py`: 초반 샘플 혼합 추출, 초반 시연 분포 보고, 첫 base 행동의 보조 L1, 초반 검증 지표.
- `turbovla/data/dataset.py`: 시연 내부 frame_index와 episode_id 반환. 에피소드 경계에서 frame_index가 0으로 초기화된다. 두 값은 샘플링/손실/지표에만 사용하며 모델 입력에 넣지 않는다.
- `turbovla/training/trainer.py`: 옵션·손실·학습 표본 비율·검증 지표·체크포인트 설정 기록. resume 시 설정 변경을 거부하고, warm-start는 새로운 실험으로 허용한다.
- `scripts/behavior_2026/train_staged.py`, `run_task0_full_staged.sh`: 새 옵션 전달.
- `turbovla/data/stages.py`: 학습 없이 동작 분포를 확인하는 기존 audit 명령에 초반 단계 수와 초반 첫 이동 시점 추가.

## 샘플링

`--startup-seconds 10 --startup-fraction 0.25`이면 학습 추출의 기대 25%를 시연 초반 10초 구간에서 고른다. 나머지 75%는 기존 sampler를 따른다. 초반 분기에서는 (task 균형 사용 시 task →) episode → 그 episode 초반에 존재하는 동작 단계 → frame 순서로 균등 추출한다. 희귀한 초기 회전 구간도 반복해서 학습할 수 있다. 조작/정지 단계만 있는 시작 상태에도 없는 이동 정답을 넣지 않는다.

전체 초반 표본 비율은 기존 sampler에서도 초반 프레임이 뽑히므로 25%보다 클 수 있다. `train/startup_sample_fraction`으로 실제 비율을 확인한다. 복원 추출이므로 epoch 수와 전체 프레임 방문을 뜻하지 않으며, 소수 시작 장면을 과도하게 반복할 위험이 있다. 검증은 원래 분포 전체를 그대로 사용한다.

## 첫 행동의 이동·회전 보조 손실

`--startup-base-weight 0.1`을 주면 초반 관측에서 **첫 번째 예측 행동의 base XY·yaw 3축**과 시연 정답 사이 정규화 L1을 기존 손실에 0.1배 더한다. 실제 execute-steps=1에서 실행되는 명령에 초점을 둔다. 초반 프레임 또는 유효 행동이 배치에 없으면 보조 손실은 0이다. 전문가가 멈추면 정답도 정지다. 결정 모듈 전용 사전학습 구간에는 다른 제어 손실처럼 최적화에 포함되지 않는다.

이 옵션은 목표 물체 위치나 가시성을 직접 가르치지 않는다. 전문가와 다른 관측에서 어느 방향이 맞는지 새 정답을 만들지도 않는다. 원래 제어 손실에도 base가 포함돼 있으므로 이 항목은 가중치 증가이며 독립적인 새 supervision이 아니다.

## 초반 검증

매 validation에서 `startup_validation` JSON과 W&B의 `validation/startup/*`를 기록한다.

- 첫 행동 base raw L1, 축별로 기록. 물리 단위가 다른 축을 합쳐 해석하지 않는다.
- 모든 base 명령을 0으로 예측하는 기준의 축별 L1.
- 이동/회전이 정답일 때 모델도 움직임을 예측한 비율(motion_recall).
- 모델의 움직임 예측 중 정답도 움직임인 비율(motion_precision).
- 멈춰야 하는 정답에서 움직임을 내는 비율(false_motion_rate).

움직임 임계값은 기존 proxy와 동일하게 raw XY norm 0.01 또는 raw yaw 절댓값 0.02다. 평가기 스케일 적용 후 m/s 임계값으로 해석하지 않는다. 검증 예측은 실제 정책과 같은 inverse_action clipping을 거친다. 정답은 raw_actions 첫 행동이다.

recall만 높아지고 오작동 비율 또는 방향 오차가 늘면 개선으로 보지 않는다. 이 지표는 시연 관측에서의 결과이며 실제 라디오 최초 가시 시점·접근 시간·켜기 성공은 폐루프 평가에서 별도로 측정해야 한다.

## 실행할 때의 비교 순서

기본값은 startup-fraction=0, startup-base-weight=0이므로 기존 학습 목적은 유지된다. 10초·25%·0.1은 검증된 최적값이 아닌 실험 시작값이다.

1. 같은 정책 가중치를 warm-start하고 기존 설정 그대로 실행하는 대조군.
2. 같은 가중치·학습률·분할·학습 표본 예산에서 샘플링만 변경:

```bash
--startup-seconds 10 --startup-fraction 0.25 --startup-base-weight 0
```

3. 샘플링 효과를 본 뒤 별도 출력 폴더에서 첫 base 손실을 추가:

```bash
--startup-seconds 10 --startup-fraction 0.25 --startup-base-weight 0.1
```

위 옵션은 `train_staged.py` 또는 core trainer에 추가한다. shell launcher에서는 각각 `STARTUP_SECONDS`, `STARTUP_FRACTION`, `STARTUP_BASE_WEIGHT` 환경변수로 지정한다. 기존 정책 이식은 `WARM_START`, 학습률은 `LEARNING_RATE`로 지정할 수 있다. warm-start 때 판단 전용 학습의 기본값은 0이고 새 출력 폴더가 필요하다. 추론 앙상블은 고정해 학습 변경 효과와 섞지 않는다.

학습 전에는 기존 `python -m turbovla.data.stages --cache ... --episodes ... --out 새파일.json --startup-seconds 10`으로 **TRAIN episode만** 검사한다. 여기서 초반에 전문가도 정지한다면 이번 샘플링은 빠른 탐색의 새 정답을 제공하지 못한다. 그런 경우 학습용 환경에서 즉시 목표를 찾고 접근하는 시연과 실패 상태의 회복 시연을 보강하는 쪽이 필요하다.

## 검증

새 단위 테스트 7개 통과: 샘플 혼합률·정지/조작 시작 보존·seed 재현·짧은 episode·첫 base 축 gradient·유효 mask·초반 지표·dataset 경계·launcher 옵션 전달. 전체 모델 학습 테스트는 이번 요청에서 실행하지 않았다. 실제 성공률 개선은 미확인이다.
