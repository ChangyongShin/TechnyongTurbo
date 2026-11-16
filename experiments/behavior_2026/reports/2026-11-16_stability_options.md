# 반복 팔 동작 진단 및 정책 가중치 이식

## 확인된 코드 경로와 남은 불확실성

기본 BehaviorPolicy는 horizon 12 중 첫 행동 1개만 실행하고 매 관측에서 재추론한다. 입력 상태는 이 실행 결과에 따라 바뀐다. 이전 목표와 다른 관절 목표를 계속 예측하면 팔 왕복을 만들 수 있다. 현재 temporal loss는 한 관측에서 예측한 chunk 내부 변화량만 모방하므로 연속 관측 사이 첫 행동의 일관성을 직접 학습하지 않는다. 또한 모델의 시간 축 입력은 마지막 프레임만 사용하므로 관측 이력을 이용하는 정책이 아니다.

영상은 팔의 반복 이동을 보여주지만 특정 손목 축의 연속 회전인지, 여러 관절의 왕복인지 구분하지 못한다. 모델의 명령과 실제 관절값, 제어기 계약을 함께 기록하기 전에는 원인을 확정할 수 없다. 목표가 안 보이는 상태의 회복 실패와, 목표가 보이는 상태의 정밀 조작 실패도 별도 문제다.

## 추가한 옵션

1. `--warm-start`: 기존 BEHAVIOR 정책 가중치를 이식한다. normalization, preprocessing, 행동 순서, 학습/검증 분할, 데이터 출처가 같아야 한다. 모델 구성은 backbone 경로·영상 계산 정밀도·판단 conditioning의 차이만 허용하며 공통 정책 tensor 누락·shape 차이는 거부한다. optimizer와 step은 새로 시작한다. 저장 파일에 원본 해시·step·새로 초기화한 key를 기록한다. 기존 `--initial-weights`의 느슨한 부분 이식과 구분한다.
2. `--ensemble-window`: 매 시점 재추론하면서 과거 chunk가 **현재 시점에 대해 예측한 행동**을 시간 정렬해 평균한다. 몸통과 팔 관절만 평균하고 base·gripper는 최신 명령을 유지한다. 기본 1은 기존 동작이다. 1보다 크면 execute-steps=1이어야 한다. 큰 window나 작은 decay는 과거 예측의 영향을 늘려 반응을 둔하게 만들 수 있다. 정확한 접촉이나 빠른 회복이 악화될 수 있어 기본으로 활성화하지 않았다.
3. trace에 보정 전 `unfiltered_action`, 실행 `action`, 각각의 인접 명령 변화량, 이전 명령 유무를 추가했다. reset 시 기록과 앙상블 이력을 함께 지운다. 명령 변화량은 제어기 입력 단위이며 실제 관절속도와 다르다.

판단 conditioning이 없는 baseline에서 staged 모델로 이식하면 새로운 판단 모듈은 무작위 초기화되고 attention에 토큰이 추가된다. 가중치를 가져와도 첫 출력이 baseline과 정확히 같지는 않다. 해당 변경의 영향을 완전히 분리하려면 기존 구조 그대로의 trainer에서 warm-start 대조군도 실행한다. 이식 자체가 성공을 보장하지 않는다.

## 먼저 재학습 없이 비교

기존 서버 실행 명령에 아래 옵션만 추가한다. 새 체크포인트로 교체하지 말고 같은 파일·초기 상태·seed·시간 상한을 사용한다. 진단 trace는 이미지 저장으로 지연을 늘릴 수 있으므로 최종 성능 비교와 별도로 수집한다.

```bash
# 대조군: 기존 동작
--execute-steps 1 --ensemble-window 1

# 실험군: 최근 최대 4개 chunk의 시간 정렬 예측 (실험 시작값)
--execute-steps 1 --ensemble-window 4 --ensemble-decay 1.0

# 별도 짧은 진단 실행: 새 폴더를 지정
--trace-dir /workspace/traces/task0-stability-001 --trace-limit 120
```

trace의 `unfiltered_action_delta`와 `action_delta`에서 왼팔 7:14, 오른팔 15:22를 비교한다. 명령 변화만 감소하고 성공률이 늘지 않으면 안정화만 된 것이다. `request_state_raw`의 대응 관절과 출력 목표의 오차를 함께 보되 실제 평가기 state 의미를 먼저 확인한다. 동일 상태의 반복 추론에서 변화가 생기는지도 점검한다.

## 가중치 이식 학습 예시

경로는 기존 서버 구조 예시이며 실제 파일을 확인해 바꾼다. 새 출력 폴더만 사용한다. 아래 3e-5와 3,000 step은 검증된 최적값이 아닌 비교 실험 시작값이다.

```bash
python scripts/behavior_2026/train_staged.py \
  --cache /workspace/data/task0-audit-cache \
  --baseline-config /workspace/TechnyongTurbo/outputs/task0-batch32-epoch1/config.json \
  --warm-start /workspace/TechnyongTurbo/outputs/task0-batch32-epoch1/step_12048.pt \
  --out /workspace/TechnyongTurbo/outputs/task0-staged-warmstart-001 \
  --steps 3000 --batch-size 64 --save-every 1000 \
  --lr 3e-5 --decision-pretrain-steps 0
```

`train_staged.py`는 warm-start 때 판단 전용 사전학습을 기본 0으로 한다. 순수 resume가 아니며 기존 step에 이어 optimizer 상태를 복원하지 않는다. 엄격한 계약 검사가 실패하면 억지로 우회하지 말고 데이터/정규화 차이를 먼저 확인한다. staged launcher의 기존 단계 샘플링·그룹 손실·chunk 변화량 목적함수는 그대로 적용된다.

학습 자체의 문제를 분리하려면 같은 `turbovla.training.trainer`의 기존 구조(판단 conditioning 없이)로 기존 split을 지정해 `--warm-start`를 사용한 대조군도 비교한다. 한 번에 학습 변경과 앙상블을 모두 바꾸지 않는다.

## 검증 범위

로컬 단위 테스트: 시간 정렬, 가중 평균, base/gripper 보존, reset, 잘못된 설정 거부, 가중치 계약·공통 tensor 누락 거부. 추론 통합 테스트: 기존 queue 행동 보존, 신규 trace 값, reset. 작은 모델 학습 통합 테스트: 학습·저장·warm-start·재학습·추론. 실제 GPU 학습과 OmniGibson 성공률 평가는 수행하지 않았다.
