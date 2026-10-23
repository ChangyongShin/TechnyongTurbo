# BEHAVIOR RGB/RGB-D 검증 기록

검증일: 2026-10-04. CPU 테스트 24개 모두 통과했으며 skip은 없습니다.

```bash
python -m unittest discover -s tests -p 'test_behavior*.py' -v
```

검증된 항목:

- 23차원 행동 정규화, 61차원 상태, episode 경계의 chunk mask.
- 공식 R1Pro 관측 key·head/left/right 카메라 순서, task parquet/JSONL.
- 실제 loopback websocket의 health check, metadata, array 직렬화와 reset 무응답 프로토콜.
- 실제 인코딩한 합성 RGB MP4와 lossless 12비트 gray HEVC를 episode offset에 맞춰 디코딩.
- native 12비트 plane의 row padding 보존, log12→m 변환과 원본 RGB/depth resolution 검사.
- RGB-D mmap/NPZ cache→학습 batch, train/inference 전처리 일치, 누락/NaN depth 거절.
- live depth +inf를 공식 시연 quantizer와 동일하게 10 m로 포화.
- 소형 로컬 HF DINOv3/BERT를 사용한 RGB-D forward/backward, 실제 depth gradient와 RGB kernel 보존.
- 학습된 depth kernel을 포함한 checkpoint를 다시 로딩한 정책의 예측 일치.
- 실제 `train.py --rgbd`의 한 synthetic optimizer step·검증 loss·checkpoint 저장.
- `prepare/train --help`, Python compile 및 A100 Bash 스크립트 문법.

CPU 테스트 환경: Windows/Python 3.14.7, torch 2.14.1+cpu,
torchvision 0.29.1+cpu, transformers 4.56.2, PyAV 19.0.1,
PyArrow 25.0.1, Pillow 12.3.0, NumPy 2.5.3. 테스트 라이브러리는
프로젝트 `.test-deps`에 격리했으며 대회 데이터 학습은 수행하지 않았습니다.

서버 설치 스크립트의 Python 3.11/torch 2.7.1 CUDA 12.6/NumPy 1.26.4 등
고정 환경은 서버에서 아직 실행하지 않았습니다. CPU 통과가 그 환경의 설치와
BF16 실행까지 검증했다는 의미는 아닙니다.

공식 depth schema는 [공식 info.json](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/blob/main/meta/info.json)과
포함된 OmniGibson quantize/dequantize 코드에 대조했습니다. 이 검증은 합성 영상을
사용하므로 실제 배포 영상의 codec/shard/episode metadata가 완전히 일치하는지는
선택한 실제 episode로 `preflight → prepare → A100 1-step → serve → RTX evaluation`을
수행해 확인해야 합니다. GPU 메모리/처리량, full-dataset 확장성과 task 성공률은 미측정입니다.
