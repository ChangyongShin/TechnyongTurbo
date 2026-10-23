# 패치 적용 완료된 전체 코드

기존 TechnyongTurboVLA 2bdfe3f46c93bb27b1741bbfcc106e1b19084182에 BEHAVIOR 수정 패치를 적용했습니다.
BEHAVIOR-1K 서브모듈은 b1979916ec1549b10a4e65e630bc6504a9af1b00의 소스 코드를 포함합니다.
이 ZIP에는 패치가 이미 적용되어 있으므로 git apply를 다시 실행하지 마세요.

압축을 푼 폴더에서 [A100 실행 안내서](experiments/behavior_2026/A100_SETUP_KO.md)의 설치, 데이터 준비, 학습 절차를 따르세요.
학습 데이터, 장면/물체 자산, 모델 가중치와 Python 의존성은 별도로 준비해야 합니다.
.git 디렉터리는 포함하지 않았으므로 ZIP으로 받은 코드에서 git submodule update를 실행할 필요는 없습니다.

## 2026-10-04 추가 수정

- A100 단일 GPU용 설치/학습 스크립트, 공식 task parquet, HF revision 기록, RGB mmap cache와 정책 서버를 추가했습니다.
- RGB-D 다운로드 → 12비트 depth 디코딩 → metric cache → 4채널 학습 → checkpoint → 평가 서버 입력 경로를 구현했습니다.
- 상태 61차원과 행동 23차원을 유지하며, 세 카메라 순서는 head/left wrist/right wrist입니다.
- raw depth는 log12 코드로 읽어 m로 복원합니다. 공식 loader의 mm 출력 표기를 raw 영상 픽셀 단위로 오해하지 않습니다.
- 추가 depth kernel만 학습하고 기존 RGB kernel은 고정합니다. depth를 0으로 추가한 뒤 전부 freeze해 무시되는 문제를 방지했습니다.
- checkpoint 경로 이동, RGB/RGB-D modality 불일치, 누락된 depth, episode reset을 처리합니다.

검증: CPU 계약/통합 테스트 **24개 통과, skip 없음**, Bash 문법 및 prepare/train CLI 검사 통과.
합성 lossless 12비트 HEVC 영상의 디코딩과 mmap/NPZ RGB-D cache, 소형 로컬 HF 모델의
forward/backward·depth gradient·RGB 가중치 보존·checkpoint 정책 재로딩 및 실제 train CLI 1-step을 검증했습니다.

공식 시연의 실제 episode 영상, A100 CUDA/BF16 학습과 OmniGibson closed-loop 실행은 아직 미검증입니다.
로컬 CPU 테스트 버전은 A100 고정 환경과 다릅니다. 상세 범위는 [검증 기록](experiments/behavior_2026/VALIDATION_KO.md)을 확인하세요.
`PACKAGE_SHA256.json`은 변경/추가한 소스까지 갱신하며 `.test-deps`와 가상환경은 패키지 목록에서 제외합니다.
