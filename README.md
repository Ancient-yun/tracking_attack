# St4RTrack 전체 프레임 PGD 실험

**현재 상태(2026-10-04): 사용자의 요청으로 tracking 3D 로스까지 완료하고 중단했습니다.** Point Odyssey 4클립 + Dynamic Replica 4클립의 원본 128프레임 전체를 사용해 clean 8 + `tracking_mse` 8 + `tracking_3d` 8 = **24/48조건**을 완료했습니다. 공격은 PGD-20입니다. `reconstruction_3d`, `confidence`, `joint_training`은 시작하지 않았고 자동 재개도 없습니다. [중단 확인 기록](reports/lossstudy_allframes_20261004_183750/pause_verification.json)과 [부분 결과 요약](reports/lossstudy_allframes_20261004_183750/summary.json)을 공개합니다. 전체 48조건 감사와 최종 분석은 실행하지 않았습니다.

기본 계획은 **Point Odyssey 4클립 + Dynamic Replica 4클립, 각 원본 128프레임 전체, 다섯 목적함수별 PGD-20**입니다. 현재 설정과 전체 재현 절차는 [실험 재현 안내](docs/EXPERIMENT_REPRODUCTION.md), 손실 수식은 [loss 구조 설명](docs/loss_structure.md), Docker 명령은 [Docker 실행 안내](docker/README.md)를 참고하세요. 아래 16/64프레임 결과와 설정은 이전 검증 기록입니다.

```powershell
git clone https://github.com/Ancient-yun/tracking_attack.git
cd tracking_attack
```

이 저장소에는 구현 코드, 설정, manifest, 테스트, 문서와 `reports/`의 작은 상태·지표 기록을 포함합니다. 데이터, 체크포인트, 가상환경, 캐시, 대용량 실행 배열은 로컬에 보관합니다. 설치와 데이터 다운로드 후 실행하세요. 문서의 `C:\code\st4rtrack_pgd`는 원래 실험 경로이며 새 환경에서는 clone한 폴더 경로로 바꿉니다. Docker에서는 `docker/manifests/*.json`의 상대 경로 manifest를 사용합니다. 공식 St4RTrack 소스는 Docker 빌드 또는 아래 clone 명령으로 고정 commit에서 별도로 가져옵니다.

공식 St4RTrack Seq 모델과 WorldTrack 데이터를 사용하여 clean, uniform noise, FGSM, PGD의 3D 추적 성능을 비교합니다. 실험 폴더는 `C:\code\st4rtrack_pgd`이며, 기존 프로젝트와 분리된 Python 환경을 사용합니다.

공식 소스: <https://github.com/HavenFeng/St4RTrack>  
고정 commit: `0f9a3f44a7ebac76600cd31ec9eea5228ad7db91`  
체크포인트: 저자 배포 `St4RTrack_Seqmode_reweightMax5.pth`  
데이터: <https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV>

Docker 구성과 명령은 [docker/README.md](docker/README.md)에 있습니다. 기본 Docker 실행은 PO·DR 각각 2프레임의 스모크 테스트이며 전체 실험은 별도 명령으로 시작합니다.

## 실제 검증 결과

2026-10-02, RTX 5080에서 PO 2개 클립 × 첫 16프레임으로 실행했습니다. epsilon=4/255, PGD-20, step=1/255, restart=1입니다. 아래 값은 두 시퀀스 지표의 단순 평균이며, 전체 100개 본 평가의 결과가 아닙니다.

| 조건 | APD3D 전체 (%) | APD3D 동적 (%) | EPE 전체 (m) | EPE 동적 (m) |
|---|---:|---:|---:|---:|
| Clean | 71.8809 | 78.5532 | 0.2714 | 0.2065 |
| Uniform noise | 72.8437 | 78.5340 | 0.2665 | 0.2133 |
| FGSM | 38.8585 | 45.5503 | 0.8499 | 0.9579 |
| PGD-20 | 10.6854 | 15.6599 | 12.4265 | 7.0848 |

8개 조건 모두 완료했고 실패는 0개입니다. 공식 clean 출력과 공격용 forward의 오차는 0이며, 저장한 float 입력을 재실행한 궤적 오차도 모든 조건에서 0입니다. CPU 단위·통합 테스트는 32개 및 추가 검사 11개를 통과했습니다. DR 공식 adapter의 64프레임 입력 검사도 통과했습니다.

공식 모델의 2프레임 gradient 검증도 통과했습니다. full graph와 재계산의 relative L2 차이는 0.03375%였고, 각각 반복한 backward의 차이는 0.03138%와 0.03489%였습니다. 방향 수치 미분은 인접한 두 step에서 기준을 만족했으며 선택된 probe의 오차는 4.13%였습니다. 원소별 엄격 비교는 일부 값에서 실패했으므로 GPU gradient의 bit 일치로 표현하지 않습니다. 전체 수치와 단일 pixel 수치 미분 진단은 `runs/implementation_validation.json`에 있습니다.

결과는 `runs/po_validation_verified/sequence_metrics.csv`, `aggregate_metrics.csv`, `summary.json`에 있습니다. PGD-20은 16프레임 한 클립에서 평균 약 315초가 걸렸습니다. 단순히 프레임 수에 비례한다고 가정하면 이 PC에서 100개 × 64프레임 × epsilon 3종의 PGD만 약 105시간이므로, 본 평가 실행 시간을 별도로 확보해야 합니다. 이는 작은 검증 run으로부터의 거친 예상이며 실제 본 평가를 측정한 시간은 아닙니다.

## 데이터와 모델 가중치 다운로드

아래 명령은 clone한 `tracking_attack` 폴더에서 실행합니다. 본 실험은 저자가 전처리해 배포한 WorldTrack NPZ와 정확한 Seq 체크포인트를 사용합니다. [공식 St4RTrack 다운로드 안내](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/README.md#download-checkpoints)에서도 이 체크포인트를 권장합니다.

### 공식 다운로드 링크

| 파일 | 다운로드 위치 | 이 프로젝트에서 저장할 경로 |
|---|---|---|
| WorldTrack 전체 배포 폴더 | [WorldTrack mini-release](https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV) | `data/worldtrack_release/` |
| Point Odyssey NPZ 50개 | [po_mini 폴더](https://drive.google.com/drive/folders/1ToXkVJKlHs6xhCBaY4LB29whViH4RAHn) | `data/worldtrack_release/po_mini/*.npz` |
| Dynamic Replica NPZ 50개 | [ds_mini 폴더](https://drive.google.com/drive/folders/1jl9Og6MWLj1q3runKkpkE5iH14ba6_3T) | `data/worldtrack_release/ds_mini/*.npz` |
| `St4RTrack_Seqmode_reweightMax5.pth` | [정확한 Seq 체크포인트 파일](https://drive.google.com/file/d/1ElgLYxWNHmps7-xvmHz2D6w4B0kZbBd1/view), [저자 체크포인트 폴더](https://drive.google.com/drive/folders/1uSfnZbzqa8pfIb6k383-BerLQ0m9-R1l) | `assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth` |

모델 파일의 크기는 **4,411,404,231 bytes(약 4.41GB)**입니다. 데이터와 모델은 Git 저장소에 포함하지 않으며 로컬에 별도로 받습니다.

### 자동 다운로드: Windows PowerShell

Python 3.12가 설치되어 있어야 합니다. 다운로드만 할 때는 아래 두 Python 패키지만 필요하며, 호스트의 PyTorch 또는 GPU를 사용하지 않습니다. 이미 `.venv`가 준비되어 있으면 첫 줄은 생략합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "gdown>=6.4" requests
.\.venv\Scripts\python.exe scripts\download_assets.py
```

기본값은 **PO 50개 + DR 50개 + Seq 체크포인트**를 모두 다운로드합니다. 현재 고정 8클립 실험의 명단은 [재현 안내](docs/EXPERIMENT_REPRODUCTION.md#4-데이터셋과-고정-8클립-명단)에 있습니다. `--limit 4`는 공개 폴더 목록의 앞 4개를 받을 뿐이므로 본 실험의 선정 클립과 일치하지 않습니다.

### 자동 다운로드: Linux

Python 3.12와 venv 기능이 설치된 환경에서 실행합니다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'gdown>=6.4' requests
.venv/bin/python scripts/download_assets.py
```

### 가중치만 또는 데이터만 받기

Windows에서는 다음 옵션을 사용합니다. Linux에서는 Python 실행 경로를 `.venv/bin/python`으로 바꾸면 됩니다.

```powershell
# 가중치만 다운로드
.\.venv\Scripts\python.exe scripts\download_assets.py --checkpoint-only

# PO와 DR 데이터만 다운로드
.\.venv\Scripts\python.exe scripts\download_assets.py --skip-checkpoint

# 특정 데이터셋만 다운로드
.\.venv\Scripts\python.exe scripts\download_assets.py --datasets po_mini --skip-checkpoint
.\.venv\Scripts\python.exe scripts\download_assets.py --datasets ds_mini --skip-checkpoint
```

다운로드가 중단되면 같은 명령을 다시 실행합니다. 완료된 정상 파일은 재사용하고 부분 다운로드는 재개합니다. 다운로드 목록, 공개 폴더 snapshot, 파일별 SHA-256, 실패 내역은 `assets/download_manifest.json`과 `assets/*_files.json` 등에 기록합니다. 다운로더는 파일 크기와 ZIP/NPZ/PyTorch archive 형식을 검사하여 HTML 응답과 잘린 파일을 거부합니다. Google Drive 다운로드 제한이나 네트워크 오류가 발생하면 기록의 `failed` 항목을 확인하고 나중에 재시도합니다. 동시 다운로드 수를 줄이려면 `--workers 1`을 추가합니다.

### 수동 다운로드와 저장 경로

브라우저에서 위 공식 링크를 열어 받아도 됩니다. 폴더 다운로드가 ZIP으로 묶여 내려오면 압축을 풀어 **원본 NPZ 파일**을 아래 경로에 둡니다. NPZ 내부 파일을 풀거나 다시 압축하지 않습니다. 다른 variant의 가중치를 이름만 바꿔 사용하지 않습니다.

```text
tracking_attack/
├── assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth
└── data/worldtrack_release/
    ├── po_mini/*.npz
    └── ds_mini/*.npz
```

가중치의 예상 SHA-256은 다음과 같습니다. 다운로드 후 실제 값과 비교합니다. Docker 스모크는 이 값과 일치하는지 검사합니다. 실험 runner는 모델 hash를 실행 기록에 남기고 manifest에 고정된 데이터 hash를 확인합니다.

```text
cae4712e0265f7d5ecadade19a0cb2ccf7b7e786fa0f230a80603a3437bcdfd3
```

```powershell
# Windows
Get-FileHash -Algorithm SHA256 assets\checkpoints\St4RTrack_Seqmode_reweightMax5.pth
```

```bash
# Linux
sha256sum assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth
```

Docker는 이 경로를 읽기 전용으로 연결합니다. 파일 준비가 끝나면 [Docker 빌드와 스모크 안내](docker/README.md#빌드와-스모크)를 따릅니다. ADT·PStudio는 사용자가 받은 공식 NPZ를 같은 형식으로 추가하면 runner에서 사용할 수 있습니다.

## Python 환경 설치

Docker 대신 호스트에서 모델을 실행하려면 전체 Python 의존성을 설치합니다. 아래 명령은 환경 구성과 데이터·모델 다운로드를 함께 수행하며, CUDA를 사용할 수 있으면 환경 검증 중 작은 GPU 연산도 수행합니다.

```powershell
# Windows, 프로젝트 루트에서
.\scripts\bootstrap.ps1 -DownloadAssets
```

```bash
# Linux, 프로젝트 루트에서
bash scripts/bootstrap.sh --download-assets
```

현재 GPU RTX 5080에서는 Python 3.12 / PyTorch 2.7.1 / CUDA 12.8을 사용합니다. 저자의 PyTorch 2.5.1 / CUDA 12.1 환경과 다르므로 동일 환경 재현으로 표현하지 않습니다. 지원되는 이전 세대 GPU의 Linux에서는 `TORCH_PLATFORM=cu121 bash scripts/bootstrap.sh --download-assets`로 저자 버전을 설치할 수 있습니다. 새 머신에 복사할 때는 `.venv`를 복사하지 않고 해당 OS에서 bootstrap을 실행합니다.

공식 checkout을 다시 구성할 때:

```powershell
git clone --recursive https://github.com/HavenFeng/St4RTrack.git vendor\St4RTrack
git -C vendor\St4RTrack checkout 0f9a3f44a7ebac76600cd31ec9eea5228ad7db91
```

## 실행

2개 PO 클립 × 첫 16프레임, PGD-20, epsilon=4/255, step=1/255:

```powershell
.\scripts\run_validation.ps1
```

위 스크립트는 제공된 manifest를 사용하고 날짜·시간이 붙은 새 run 폴더에 저장합니다. 완료된 기존 결과는 `runs/po_validation_verified`에 있습니다. 같은 이름의 manifest를 덮어쓰지 않으며, 데이터 파일 hash를 실행 전에 확인합니다. 조건을 바꾸면 새 run 폴더를 사용합니다. 동일 config·manifest·코드·모델·실행 환경의 중단된 작업은 `--resume`으로 재개합니다.

PO 10개 × 64프레임 예비 실험:

```powershell
.\.venv\Scripts\python.exe -m st4rtrack_pgd.run_attack --config configs\pilot.json --manifest manifests\po_pilot.json --run-dir runs\po_pilot
```

PO·DR 각 50개 × 64프레임, epsilon=2/255,4/255,8/255:

```powershell
.\.venv\Scripts\python.exe -m st4rtrack_pgd.run_attack --config configs\full.json --manifest manifests\synthetic_full.json --run-dir runs\synthetic_full
```

50개 release 파일을 모두 사용할 때도 논문과 동일한 시퀀스인지 확인되지 않았다는 사실을 manifest에 기록합니다. 저자 평가 목록이 별도로 있다면 `--sequence-list`에 `dataset/filename.npz` 형식의 텍스트 파일을 전달합니다. 공격 설정을 고르는 개발 데이터와 본 평가 데이터는 별도로 유지해야 합니다. `configs/strength.json`은 고정 검증 manifest에서 PGD-100/restart=3을 실행합니다. `--steps 50`으로 PGD-50도 실행할 수 있습니다.

## 구현과 평가 규칙

- 공격 공간: 공식 전처리 후, 정규화 전 RGB `[0,1]`. 각 프레임마다 독립적인 L-infinity 교란을 가지며, frame 0은 모든 `(0,t)` 쌍 및 `(0,0)` 양쪽에서 공유합니다.
- 실제 공식 평가의 전처리는 `load_images(size=512, square_ok=True, crop=False)`입니다. 종횡비를 유지하는 resize 후 크기를 16의 배수로 맞추는 두 번째 resize를 수행합니다. 첨부 초안의 일반적인 crop 표현과 달리 실제 평가 규칙을 따릅니다.
- GT는 공식 `load_npz_data(normalize_cam=True)`의 첫 카메라 world 좌표계입니다. query는 첫 프레임 visibility로 선택하고 최종 해상도로 xy를 변환한 뒤 정수 truncation으로 `pred1['pts3d']`에서 읽습니다. 이후 가려진 시점도 공식 평가처럼 평가에 포함합니다.
- 동적 점은 선택한 GT 궤적의 프레임 간 이동거리 합이 `0.01m`보다 큰 점입니다. confidence로 점을 제외하거나 loss를 가중하지 않습니다. 가중치는 동결하고 `eval()`로 실행하며 TTA를 사용하지 않습니다.
- loss는 시퀀스 공통 `median(||GT||)/median(||pred||)` 정렬 후 평균 제곱 3D 거리입니다. 짝수 원소 median은 NumPy처럼 가운데 두 값의 평균이며 scale의 gradient도 계산합니다.
- 메모리 절약: query 궤적을 먼저 모으고 전체 loss의 출력 gradient를 계산한 뒤 각 쌍을 재계산해 입력 gradient를 합칩니다. 출력 재계산의 일치 검사를 기본으로 켭니다. 모든 gradient를 모은 뒤 모든 프레임을 한 번에 갱신합니다.
- PGD는 clean, 모든 초기화와 반복 중 loss가 가장 큰 입력을 저장합니다. noise는 한 번의 uniform 표본, FGSM은 clean에서 한 번의 epsilon sign step을 그대로 저장합니다.
- 지표는 고정 official `track_eval_util.py` 함수를 직접 호출합니다. APD3D는 `0.1,0.3,0.5,1.0m` 성공률의 평균이며 CSV에서는 `%`로 표시합니다. EPE 단위는 m입니다. 전체/동적 결과를 모두 저장하며, 동적 subset은 공식 코드와 같이 subset에서 scale을 별도 계산합니다. 집계는 시퀀스 지표의 단순 평균입니다.
- clean 비교는 공식 inference의 `batch_size=1`에 맞춥니다. 일부 공식 모델 구조의 temporal positional encoding은 batch 구성에 영향을 받으므로 임의의 batch size 결과와 동일하다고 가정하지 않습니다.

## 검증과 결과 재평가

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m st4rtrack_pgd.validate --sequence data\worldtrack_release\po_mini\cab_e_3rd_1.npz --num-frames 2 --finite-difference
.\.venv\Scripts\python.exe -m st4rtrack_pgd.evaluate_attack --run-dir runs\po_validation_verified
```

CPU toy 검증은 전체 graph와 두 번 계산 gradient의 일치, scale gradient의 수치 미분, `(0,0)` 양쪽 입력 gradient, 모든 프레임의 유한 gradient, projection, seed, FGSM/PGD 동작, float 입력 저장·복원을 검사합니다. 실제 클립 검증은 공식 clean 출력 일치와 두 방식의 gradient를 검사합니다. 저자 DPT의 CUDA bilinear interpolation backward에는 [PyTorch 2.7이 문서화한 비결정성](https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html)이 있어, 각 방식을 두 번 계산해 관측된 변동량을 기록합니다. 두 방식의 차이는 관측된 반복 오차의 2배 이내여야 하며 relative L2 상한은 0.1%입니다. 엄격 원소별 비교 결과도 따로 남깁니다. 수치 미분은 unit-L2 gradient 방향에서 검사하고, 상대 오차 15% 이하인 인접 probe 두 개가 서로 10% 이내로 일치해야 합니다. 단일 pixel probe도 실패 여부를 판단할 수 있도록 전체 값을 보존합니다. runner는 각 조건의 float 입력을 다시 읽어 모델을 재실행하고 궤적과 지표가 유지되는지 확인합니다.

결과 파일:

| 파일 | 내용 |
|---|---|
| `run.json` | config, manifest, 소스 commit/hash, 체크포인트 SHA-256, Python/PyTorch/CUDA/GPU |
| `sequence_metrics.csv` | 시퀀스·조건별 APD/EPE, clean 대비 하락/증가, 실행 시간 |
| `aggregate_metrics.csv`, `summary.json` | dataset·조건별 집계, 완료 수/예정 수, 실패 내역 |
| `sequences/.../tracks.npz` | 예측, GT, 고정 mask, query xy, intrinsics |
| `rgb_float32.npy`, `delta_float32.npy` | 양자화 없는 실제 공격 입력과 교란 |
| `history.json`, `perturbation_norms.json` | 반복 loss/scale, 프레임별 입력 gradient와 교란 크기 |
| `result.json` | 조건 설정, 지표, peak VRAM, 입력 재평가, artifact hash |
| `failures.jsonl` | NaN/Inf, 모델/데이터/검증 실패의 traceback |

오류는 조용히 제외하지 않습니다. 불완전 run은 nonzero exit code로 끝나고 실패 및 완료 수가 남습니다. 전체 본 평가에는 여러 시간의 GPU 실행과 저장 공간이 필요할 수 있으므로, 검증 run의 처리 시간과 VRAM부터 확인합니다. 공식 코드·모델의 비상업 연구 라이선스는 upstream README를 따릅니다.
