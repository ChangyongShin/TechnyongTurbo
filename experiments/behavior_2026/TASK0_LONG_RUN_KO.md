# staged 5,000-step 정책을 총 100,000 step까지 재개

사용자가 지정한 출발점은 무증강 12,048-step 모델이 아니라 active_policy의 staged 5,000-step 모델이다.
그 모델의 구조·손실·샘플링을 유지하고 step만 늘린다.
`scripts/behavior_2026/train_task0_long.py`는 실행 준비만 담당한다. 모델과 trainer 소스는 수정하지 않는다.

- 출발점: `/input/Technyong_TurboVLA/active_policy/step_5000.pt`.
- 선택한 SHA-256: `08fb14be979a66c637ec9911002c97851ade7caab571587f61c9cff9793475d5`. 다른 batch8 파일이나 이후 startup 1,000-step 모델을 선택하지 않는다.
- 총 목표: 100,000 optimizer step. 기존 5,000을 포함하므로 신규 업데이트는 95,000회다.
- batch·LR·gradient accumulation·worker·seed·180/20 split은 체크포인트에서 읽는다. 지정한 모델은 batch64, lr1e-4, 누적1이다.
- `--resume`으로 모델과 optimizer를 복구한다. 기존 trainer가 RNG·sampler 위치까지 복구하는 것은 아니므로 프레임 순서가 중단 없이 진행한 실행과 같다는 보장은 없다.
- 5,000 step마다 원래 validation을 수행하고 저장한다. 기존 판단 conditioning·단계 균형·그룹 손실·decision/temporal 가중치를 복원한다. 초반 샘플 강화·joint ensemble은 추가하지 않는다.
- decision_pretrain_steps=100도 기존 기록과 동일하게 전달하지만 5,001 step부터 재개하므로 판단 전용 구간을 다시 실행하지 않는다.
- 출발 가중치를 새 실험 폴더에 복사하고 해시를 확인해 그 복사본으로 resume한다. active_policy가 나중에 교체돼도 출발점이 바뀌지 않는다.
- source revision과 task 0 지시문, 준비된 TRAIN 정규화를 원래 기록과 대조한다. 모델 설정이 현재 trainer와 다르면 전처리 전에 중단한다.
- 누락된 RGB 캐시만 영속 저장소 `/input/Technyong_TurboVLA/caches/task0-rgb-224`에 준비한다. 약 180GiB가 필요하다. 이미 있는 불완전/다른 캐시는 삭제하지 않고 오류로 중단한다.
- W&B online 모드에서는 인증을 캐시 준비 전에 확인한다. offline/disabled는 명시적 CLI 선택으로만 변경한다.
- 결과 폴더는 새 경로를 쓴다. 이전 가중치·원본 시연·이전 실험 결과를 덮어쓰지 않는다.

원복한 서버 작업 폴더는 그대로 둔다. wrapper가 `bbba897` 코드를 새 실험 폴더의 code 디렉터리에 추출해 실행한다. 이 별도 코드에는 원래 staged 모델이 있으며, 최근 원복한 초반 샘플링·앙상블 변경은 없다. Git 이력을 병합할 필요도 없다.

```bash
cd /workspace/TechnyongTurbo
git fetch origin
git show origin/main:scripts/behavior_2026/run_task0_staged_100k.sh > scripts/behavior_2026/run_task0_staged_100k.sh
source .venv-behavior/bin/activate
python -m wandb login

export RUN_DIR="/input/Technyong_TurboVLA/task0-staged-100k-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
export WANDB_NAME="$(basename "$RUN_DIR")"
nohup bash scripts/behavior_2026/run_task0_staged_100k.sh > "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo "PID: $!"
echo "LOG: $RUN_DIR/train.log"
```

`--check-only`는 checkpoint/source/settings 검사만 하고 캐시 생성·학습을 시작하지 않는다.
실행 단계에서는 GPU/W&B 확인 → 캐시 재사용/전처리 → 정규화 대조 → 학습 재개 순서다.
기본 경로가 다르면 `--checkpoint`, `--data-root`, `--cache`, `--dinov3`, `--bert`로 명시한다.
인증키는 채팅이나 코드에 넣지 않는다. 작업 도중 컨테이너 자체가 종료되면 nohup으로 유지되지 않는다.

학습량 증가만으로 성공이 보장되지는 않는다. 중간 checkpoint의 이동·회전 예측과 개발 환경의 목표 접근·작업 성공도 함께 비교한다.
