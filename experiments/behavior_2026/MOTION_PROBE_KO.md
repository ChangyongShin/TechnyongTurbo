# TRAIN 시연의 이동·회전 및 시각 활용 진단

`python -m turbovla.evaluation.motion_probe`는 시뮬레이터나 정책 서버에 연결하지 않고
기존 RGB 체크포인트와 TRAIN 캐시로 검사한다. 기존 파일을 수정하거나 재학습하지 않는다.

## 자동 검사

1. XY 명령 norm≥0.01, yaw 명령 절댓값≥0.02, 팔 목표-현재 관절 차이≥0.01인
   구간과 나머지 quiet_commands를 구한다. 단위는 원시 controller input이며
   물리 속도·성공 판정 기준이 아니다. 팔 차이는 실제 운동 여부와 같지 않다.
2. 에피소드마다 각 구간에서 최대16개 시점을 균등하게 선택한다. 구간은 겹칠 수 있다.
3. 같은 state·instruction에 정상 RGB, 검은 RGB, 같은 에피소드 절반 뒤 시점 RGB를
   각각 넣는다. 세 카메라를 한 묶음으로 바꾸고 역할 순서를 유지한다.
4. 매 조건 전에 큐를 reset하여 h=1의 새 예측을 비교한다.
5. 모델과 전문가 행동의 축별/그룹별 오차, XY 크기 비율·방향 일치도,
   움직임 구간에서 모델 명령이0.01 미만인 비율을 저장한다.
6. 현재 자세 유지(그리퍼는 평균 명령)와 선택 TRAIN 평균 명령도 비교한다.

## 실행

아스트라고에서 기존 worktree의 진단 코드를 최신 진단 커밋으로 갱신한 후:

```bash
cd /workspace/TechnyongTurbo-audit-295e1d1
PYTHONPATH="$PWD" /workspace/TechnyongTurbo/.venv-behavior/bin/python -u -m turbovla.evaluation.motion_probe --checkpoint /workspace/TechnyongTurbo/outputs/task0-batch32-epoch1/step_12048.pt --cache /workspace/data/task0-audit-cache --episodes 0 1 --samples-per-phase 16 --dinov3 /workspace/models/dinov3-vitb --bert /workspace/models/bert-base-uncased --out /workspace/TechnyongTurbo/outputs/task0-motion-probe-01
```

`motion-probe.json`과 터미널 출력에 구간별 결과가 남는다. 기존 audit의 같은0·1
TRAIN 에피소드를 사용하며 public-test rollout은 학습·통계에 섞지 않는다.
기존 out 폴더는 덮어쓰지 않는다. 폴더가 있으면 새 out 경로를 사용한다.

## 판단 범위

- 정상 RGB에서도 이동 구간 명령이 작거나 방향이 틀리면 TRAIN 시연에서의
  행동 예측 문제다. 정지 시점에 가려졌던 이동 오차를 분리할 수 있다.
- 정상 RGB가 검은/다른 시점 RGB보다 정확한지는 시각 정보 활용의 근거다.
  검은 영상은 분포 밖이고 다른 시점 영상은 state와 불일치하므로
  차이가 난다는 것만으로 라디오 위치를 의미적으로 인식했다고 결론내리지 않는다.
- 다른 시점에도 배경·라디오가 유사하게 보일 수 있어 차이가 작다고
  시각 입력을 전혀 안 쓴다고 확정하지 않는다.
- 추출 시점의 통계는 전체 에피소드의 frame-weighted 평균이 아니다.
- TRAIN에서 잘 맞춰도 새로운 초기 자세나 폐루프 실패를 배제하지 않는다.
- 같은 관절 상태 유지 기준은 현재 state/action의 선언된 순서를 가정한다.
  제어 의미 자체의 검증은 expert replay와 런타임 계약 대조가 필요하다.

진단 도구는 실패 원인을 분리하기 위한 것으로 재학습 개선 결과를 주장하지 않는다.
