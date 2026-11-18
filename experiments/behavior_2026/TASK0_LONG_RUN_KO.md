# 원복 코드에서 기존 task0 baseline을 총 100,000 step까지 재개

사용자 요청은 모델 구조·손실·샘플링을 추가 변경하지 않고 step만 늘리는 실험이다.
`scripts/behavior_2026/train_task0_long.py`는 실행 준비만 담당한다. 모델과 trainer 소스는 수정하지 않는다.

- 출발점: `outputs/task0-batch32-epoch1/step_12048.pt`. staged `active_policy/step_5000.pt`를 사용하지 않는다.
- 총 목표: 100,000 optimizer step. 기존 12,048을 포함하므로 신규 업데이트는 87,952회다.
- batch·LR·gradient accumulation·worker·seed·180/20 split은 체크포인트에서 읽는다. 알려진 baseline은 batch32, lr1e-4, 누적1이다.
- `--resume`으로 모델과 optimizer를 복구한다. 기존 trainer가 RNG·sampler 위치까지 복구하는 것은 아니므로 프레임 순서가 중단 없이 진행한 실행과 같다는 보장은 없다.
- 5,000 step마다 원래 validation을 수행하고 저장한다. 판단 conditioning·단계 균형·초반 샘플 강화·색 증강·보조 손실은 추가하지 않는다.
- source revision과 task 0 지시문, 준비된 TRAIN 정규화를 원래 기록과 대조한다. 모델 설정이 현재 trainer와 다르면 전처리 전에 중단한다.
- 누락된 RGB 캐시만 영속 저장소 `/input/Technyong_TurboVLA/caches/task0-rgb-224`에 준비한다. 약 180GiB가 필요하다. 이미 있는 불완전/다른 캐시는 삭제하지 않고 오류로 중단한다.
- W&B online 모드에서는 인증을 캐시 준비 전에 확인한다. offline/disabled는 명시적 CLI 선택으로만 변경한다.
- 결과 폴더는 새 경로를 쓴다. 이전 가중치·원본 시연·이전 실험 결과를 덮어쓰지 않는다.

Astrago는 서버 전용 커밋 `3f4a19e`를 유지할 수 있다. 원복한 핵심 코드를 다시 병합하지 않고, 새 launcher 한 파일만 가져온다.

```bash
cd /workspace/TechnyongTurbo
git fetch origin
git show origin/main:scripts/behavior_2026/train_task0_long.py > scripts/behavior_2026/train_task0_long.py
source .venv-behavior/bin/activate
python -m wandb login

RUN_DIR="/input/Technyong_TurboVLA/task0-baseline-100k-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
if python scripts/behavior_2026/train_task0_long.py --out "$RUN_DIR/checkpoints" --check-only; then
    nohup python -u scripts/behavior_2026/train_task0_long.py --out "$RUN_DIR/checkpoints" > "$RUN_DIR/train.log" 2>&1 < /dev/null &
    echo "PID: $!"
    echo "LOG: $RUN_DIR/train.log"
fi
```

`--check-only`는 checkpoint/source/settings 검사만 하고 캐시 생성·학습을 시작하지 않는다.
실행 단계에서는 GPU/W&B 확인 → 캐시 재사용/전처리 → 정규화 대조 → 학습 재개 순서다.
기본 경로가 다르면 `--checkpoint`, `--data-root`, `--cache`, `--dinov3`, `--bert`로 명시한다.
인증키는 채팅이나 코드에 넣지 않는다. 작업 도중 컨테이너 자체가 종료되면 nohup으로 유지되지 않는다.

학습량 증가만으로 성공이 보장되지는 않는다. 중간 checkpoint의 이동·회전 예측과 개발 환경의 목표 접근·작업 성공도 함께 비교한다.
