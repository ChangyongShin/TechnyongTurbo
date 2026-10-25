# A100에서 TurboVLA로 BEHAVIOR 2026 학습 시작하기

목표는 공식 시연 데이터 → 실제 GPU 학습 1 step → 체크포인트 재로딩 →
공식 평가 클라이언트 연결이 가능한 RGB/RGB-D 모방학습 구성이다.
이 문서의 작업 디렉터리는 `pyproject.toml`이 있는 TurboVLA 루트다.

## 학습과 시뮬레이터의 분리

- **A100 서버:** Linux, Python 3.11, CUDA PyTorch. 데이터 준비·학습·정책 서버.
- **RTX 서버:** BEHAVIOR-1K `v3.9.3`, 해당 버전의 공식 설치 절차로 준비한
  OmniGibson/Isaac Sim과 에셋·2026 task instances. 평가 환경은 별도 유지한다.

A100은 학습에 쓸 수 있지만 Isaac Sim이 공식 지원하는 렌더링 GPU가 아니다.
`headless`도 RGB 렌더링을 수행하므로 이 조건을 없애지 않는다.
정책 서버와 평가기를 websocket으로 연결하면 학습 패키지와 시뮬레이터의
PyTorch/NumPy/Isaac Sim 의존성 충돌을 피할 수 있다.

출처: [Isaac Sim GPU 요구사항](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html),
[2026 baseline 설치 안내](https://behavior.stanford.edu/challenge/baselines.html).
첨부된 BEHAVIOR checkout의 문서에는 `v3.9.2`가 남아 있을 수 있다.
현재 공식 안내의 `v3.9.3`로 평가 환경을 준비하고 사용한 commit을 기록한다.

## 1. A100 학습 환경

서버에 Python 3.11이 준비되어 있어야 한다. 아래 스크립트는 프로젝트 안에
`.venv-behavior`를 만들며 시스템 Python이나 시뮬레이터 환경을 수정하지 않는다.

```bash
bash scripts/behavior_2026/setup_a100.sh
source .venv-behavior/bin/activate
```

CUDA 12.6용 torch 2.7.1, torchvision 0.22.1과 고정한 학습 의존성을 설치한다.
서버 NVIDIA 드라이버가 해당 wheel과 호환되어야 한다. `pip check`, CUDA/BF16
검사 및 CPU 계약 테스트가 모두 성공한 뒤 진행한다. CUDA toolkit을 별도로
빌드하거나 flash-attn, DeepSpeed, TensorFlow를 설치할 필요가 없는 경로다.
단일 A100부터 시작한다. 현재 학습기는 DDP를 구현하지 않아 여러 프로세스
`torchrun` 실행은 명시적으로 거절한다.

## 2. 모델과 소량의 공식 시연 데이터

HF DINOv3 **ViT-B**와 BERT를 준비한다. DINOv3 접근이 제한된 계정에서는 먼저
모델 페이지에서 라이선스 접근 승인을 받은 뒤 `hf auth login`을 수행한다.

```bash
export DINOV3_MODEL_PATH=/data/models/dinov3-vitb
export BERT_MODEL_PATH=/data/models/bert-base-uncased
hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir "$DINOV3_MODEL_PATH"
hf download google-bert/bert-base-uncased --local-dir "$BERT_MODEL_PATH"

export DATA_ROOT=/data/behavior-2026-rgb
export BEHAVIOR_CACHE=/data/behavior-2026-cache
python -m turbovla.data.download \
  --root "$DATA_ROOT" --task-id 0 --episodes 0 1 2
```

`task-id 0`, episode `0 1 2`는 첫 task로 시작하는 예시다. 다른 task는 해당 chunk의
episode metadata에서 실제 ID를 선택한다. downloader는 명시한 task chunk의
data parquet와 선택한 episode에 필요한 **3개 RGB 카메라의 video shards**를
받고 HF commit SHA를 `turbovla_source.json`에 기록한다. 동일 shard에 여러
episode가 포함될 수 있어 3개 episode만 선택해도 파일 크기가 작지는 않을 수 있다.
Depth 영상과 전체 100 task는 이 RGB 시작 구성에 필요하지 않다.

```bash
python -m turbovla.training.preflight \
  --root "$DATA_ROOT" --episodes 0 1 2 \
  --dinov3 "$DINOV3_MODEL_PATH" --bert "$BERT_MODEL_PATH"
python -m turbovla.data.prepare \
  --root "$DATA_ROOT" --episodes 0 1 2 --out "$BEHAVIOR_CACHE" \
  --image-size 224 --format mmap
```

Preflight는 GPU와 모델 설정·데이터 schema·영상 파일 존재를 검사한다.
모델 weights의 실제 로딩은 학습 1 step으로, 영상의 실제 디코딩과 타임스탬프
정렬은 prepare로 확인한다. 각 episode 디렉터리에 `rgb.npy`, `states.npy`,
`actions.npy`, 완료 표시인 `metadata.json`을 만든다. RGB는 영상 frame 단위로
메모리 매핑 캐시에 기록하고 학습에서도 메모리 매핑으로 읽는다.
224 RGB 캐시는 frame당 약 451,584 bytes로, 원본 압축 영상보다 커진다.
저장공간을 계산해 task 단위로 늘리고 처음부터 전체 데이터를 변환하지 않는다.
prepare 실패로 생긴 미완료 episode 디렉터리는 완료 metadata가 없어 로딩되지 않는다.
재시도는 새 cache 경로를 사용하거나 해당 미완료 디렉터리만 정리한 뒤 수행한다.

## 3. 한 step 학습 → 작은 학습

학습기는 기본적으로 W&B에 자동 기록한다. 최초 한 번 `wandb login`으로
로그인한다. 기본 프로젝트는 `TechnyongTurbo`, 실행 이름은 `RUN_DIR`의 마지막
폴더 이름이다. `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_NAME`으로 변경할 수 있다.
매 optimizer step의 `train/l1`, `train/lr`와 검증 시점의 `validation/l1_mean`,
23개 `validation/action_XX_l1`를 학습 step 기준으로 기록하고 학습 설정과
episode split을 저장한다. 체크포인트 파일은 로컬 output 경로에 저장한다.
재개 시에는 시작 step을 기록한 새 W&B run을 만든다.
오프라인 기록은 `WANDB_MODE=offline`, 기록을 끄려면 `WANDB_MODE=disabled`로
실행한다. W&B 초기화 실패 시 오류를 표시하므로 인증과 네트워크를 확인한다.
학습 실패 시에도 W&B run은 실패 상태로 종료된다.

```bash
export TRAIN_EPISODES="0 1"
export VAL_EPISODES="2"
export CUDA_VISIBLE_DEVICES=0
export RUN_DIR=outputs/behavior-smoke
STEPS=1 BATCH_SIZE=1 GRAD_ACCUMULATION=1 bash scripts/behavior_2026/train_a100.sh
```

실제 forward/backward, finite loss, 검증 loss와 `step_1.pt` 저장까지 확인한다.
성공하면 새 output 경로에서 작은 학습을 실행한다.

```bash
export RUN_DIR=outputs/behavior-small
STEPS=1000 BATCH_SIZE=2 GRAD_ACCUMULATION=4 bash scripts/behavior_2026/train_a100.sh
```

위 값은 시작 설정이며 A100 메모리/속도를 실측한 권장 최대 batch가 아니다.
DINOv3/BERT는 frozen, vision은 BF16 autocast, fusion/action은 FP32다.
유효 batch는 `BATCH_SIZE × GRAD_ACCUMULATION`. 실제 메모리를 보며 batch를 늘린다.
정규화는 training episode만으로 계산하고 동일 episode ID가 validation에
중복되는 경우 거절한다. episode 끝 action padding은 loss에서 제외된다.

재시작은 `RESUME_CKPT=/.../step_500.pt`, 새 `RUN_DIR`, 목표 총 `STEPS=1000`으로
실행한다. 모델/optimizer를 복구하며 데이터 split·정규화·모델 설정 변경을 거절한다.
샘플 순서와 RNG까지 복원하는 bitwise 재현 재개는 아니다.
캐시 RGB는 RAM에 전부 복사하지 않지만 학습 shuffle 및 파일 수의 비용은 남는다.
전체 20,000 episode 규모의 처리량 검증은 별도 필요하다.

이 학습 경로는 DINOv3/BERT 사전학습 가중치를 사용하고 fusion/action은 기본적으로
새로 학습한다. 기존 LIBERO/RoboTwin TurboVLA와는 action/state 크기가 다르다.
`INITIAL_WEIGHTS`는 core key와 shape가 맞는 tensor만 이식하며, GroundingDINO
체크포인트 key 변환이나 LIBERO 행동 head를 BEHAVIOR head로 변환하지 않는다.

## 4. 체크포인트를 평가 환경에 연결

A100 서버의 학습 환경에서:

```bash
python -m turbovla.evaluation.serve \
  --checkpoint outputs/behavior-smoke/step_1.pt --data-root "$DATA_ROOT" \
  --dinov3 "$DINOV3_MODEL_PATH" --bert "$BERT_MODEL_PATH" \
  --host 127.0.0.1 --port 8000 --execute-steps 1
```

다른 서버로 checkpoint를 옮겼다면 `--dinov3`, `--bert`로 현재 머신의 모델
경로를 지정한다. 서버는 checkpoint의 정규화·이미지 전처리·모델 구성을 사용한다.
공식 dataset task metadata로 `task_id`를 instruction으로 바꾸며 알 수 없는
task, 잘못된 관측 shape, NaN 행동은 즉시 오류로 처리한다.

RTX 평가 서버에서 A100으로 SSH 연결할 수 있다면 터널을 열어 두고:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@A100_HOST
```

다른 터미널의 기존 `behavior` 평가 환경에서:

```bash
python -m omnigibson.eval.eval \
  --task-name TASK_NAME_FOR_SELECTED_TASK_ID --host 127.0.0.1 --port 8000 \
  --instance-indices 0 --num-rollouts 1 --output-dir outputs/turbovla-smoke \
  --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper --write-video
```

`TASK_NAME_FOR_SELECTED_TASK_ID`는 선택한 task-id의 공식 task 이름으로 바꾼다.
예시의 학습/평가는 동일 task여야 한다. RGB-D wrapper에서 RGB만 사용한다.
depth, camera-relative pose와 다른 추가 관측은 모델 입력으로 전달하지 않는다.
`/healthz` → msgpack metadata → observation/action → `reset` 순서를 지원한다.
공식 reset 메시지는 응답을 기다리지 않으므로 reset ACK를 보내지 않는다.
한 정책 서버는 한 rollout만 처리하고 reset/연결 종료 때 action chunk를 비운다.

## 환경 결합 시 고정할 계약

| 항목 | 현재 최소 구성 |
| --- | --- |
| 카메라 순서 | head → left wrist → right wrist |
| 상태 | `robot_r1::proprio`, 61차원, 공식 R1Pro proprio_obs 순서 |
| 행동 | base[0:3], trunk[3:7], left arm[7:14], left gripper[14:15], right arm[15:22], right gripper[22:23] |
| 제어 | 공식 `r1pro.yaml`, `action_normalize: false`, base local-frame velocity, absolute joint commands, 연속 gripper |
| 주기 | 시연 30 Hz, horizon 12, 최초 execute_steps=1 |
| 버전 | dataset HF SHA 기록, 평가 BEHAVIOR v3.9.3 |

base 명령에 wheel radius나 속도 상수를 다시 곱하지 않는다. proprio의 base velocity도
현재 공식 robot 코드가 local frame으로 제공하므로 정책 서버에서 재회전하지 않는다.
그리퍼를 이진화하거나 14차원 RoboTwin head를 그대로 연결하지 않는다.
로봇 이름·카메라 이름을 바꿨다면 서버 `--rgb-keys`와 `--state-key`를 명시한다.
custom controller/proprio 순서를 사용하면 별도 검증된 매핑을 작성해야 한다.

## RGB-D로 학습하기

RGB-D 경로도 구현되어 있다. 위 RGB 예시의 네 단계에 `--rgbd`를 추가하고
별도 cache와 output을 사용한다. 기존 RGB 다운로드 root에는 기록된 동일 HF
revision의 depth shards를 추가로 다운로드할 수 있다.

```bash
# 기존 다운로드가 있으면 turbovla_source.json의 revision을 아래 값으로 사용한다.
export DATA_REVISION=main
export BEHAVIOR_CACHE=/data/behavior-2026-rgbd-cache
python -m turbovla.data.download \
  --root "$DATA_ROOT" --task-id 0 --episodes 0 1 2 --revision "$DATA_REVISION" --rgbd
python -m turbovla.training.preflight \
  --root "$DATA_ROOT" --episodes 0 1 2 \
  --dinov3 "$DINOV3_MODEL_PATH" --bert "$BERT_MODEL_PATH" --rgbd
python -m turbovla.data.prepare \
  --root "$DATA_ROOT" --episodes 0 1 2 --out "$BEHAVIOR_CACHE" --format mmap --rgbd
export RUN_DIR=outputs/behavior-rgbd-smoke
RGBD=1 STEPS=1 BATCH_SIZE=1 GRAD_ACCUMULATION=1 bash scripts/behavior_2026/train_a100.sh
```

원본 depth 영상의 `gray12le` 12비트 로그 코드를 직접 읽고 m 단위로 복원한다.
`video.output_unit=mm`는 공식 loader가 반환하는 단위 표기이며 raw video
픽셀의 단위가 아니다. RGB-D cache는 float32 `depth.npy [T,3,224,224]`를 포함한다.
같은 episode-relative 시각에 각각의 RGB/depth video offset을 적용해 정렬한다.
RGB와 depth의 원래 해상도가 다르거나 depth encoding metadata가 다르면 거절한다.

모델 입력은 `[B,3,4,224,224]`로, 네 번째 채널은 depth metres/10이다.
RGB mean/std는 RGB 세 채널에만 적용한다. depth는 nearest resize를 사용한다.
RGB-D 입력 patch의 새 depth kernel은 0으로 시작하고 **학습 가능**하게 유지한다.
기존 RGB kernel·DINOv3 나머지·BERT는 frozen으로 유지한다. depth kernel에
gradient를 전달하기 위해 DINOv3 연산 graph가 필요하므로 RGB보다 학습 메모리가
더 필요할 수 있다. A100에서 batch/VRAM을 실제로 측정해야 한다.

`serve`는 RGB-D checkpoint에서 자동으로 depth 관측을 요구하므로 추가 modality
플래그가 필요 없다. 평가 명령은 `RGBDFullResWrapper`를 그대로 사용한다.
실시간 sensor `depth_linear`는 m 단위다. +inf는 시연 quantizer와 동일하게 10 m로
포화시키고 NaN/-inf/음수는 거절한다. 누락된 depth를 alpha나 0으로 대신하지 않는다.
RGB-D cache를 RGB 학습에 넘기거나 RGB cache를 RGB-D 학습에 넘기면 거절한다.

## 검증 범위

```bash
python -m unittest discover -s tests -p 'test_behavior*.py' -v
```

계약 테스트는 정규화, mmap cache, task metadata, 관측 순서, websocket 직렬화,
health check/reset/실제 loopback 왕복, lossless 12비트 HEVC 영상→RGB-D cache→
학습 전처리, 소형 모델의 depth gradient·RGB 가중치 보존·checkpoint 재로딩,
실제 train CLI의 소형 synthetic 1-step까지 검사한다. 추가 의존성이 없는 로컬 환경에서는
일부 테스트가 실행되지 않을 수 있다. A100 설치 스크립트가 모든 의존성을 설치한다.
실제 시연 영상 디코딩, A100 forward/backward와 checkpoint reload, OmniGibson
closed-loop 평가는 해당 서버에서 위 순서대로 통과해야 한다.
1-step 체크포인트는 인터페이스 검증용으로 task 성공률을 보장하지 않는다.
