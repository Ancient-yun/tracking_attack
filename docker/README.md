# Docker 실행

이 구성의 기본 실행은 스모크 테스트입니다. PO와 DR에서 각각 한 클립의 첫 2프레임만 읽고, 공식 clean forward와 gradient 및 한 번의 FGSM 갱신을 검사합니다. 기존 16/64프레임 실험은 별도 Action을 지정해야 실행됩니다.

## 빌드와 스모크

빌드 전에 [README의 데이터와 모델 가중치 다운로드](../README.md#데이터와-모델-가중치-다운로드)를 따라 PO/DR NPZ와 `St4RTrack_Seqmode_reweightMax5.pth`를 준비합니다. Windows·Linux 다운로드 명령, 공식 링크, 저장 경로와 SHA-256 확인 방법이 있습니다. Docker용 다운로드만 할 때는 호스트의 PyTorch나 GPU가 필요하지 않습니다.

현재 Windows의 Docker Desktop Linux engine과 NVIDIA GPU 연결을 사용합니다. 이미지에는 Python 3.12.15와 PyTorch 2.7.1 / torchvision 0.22.1의 CUDA 12.8 wheel을 설치합니다. CUDA runtime 라이브러리는 wheel에 포함된 NVIDIA 패키지로 설치되며 호스트 driver는 Docker GPU 연결을 통해 사용합니다. [공식 PyTorch 설치 조합](https://pytorch.org/get-started/previous-versions/)과 [Docker GPU 설정](https://docs.docker.com/compose/how-tos/gpu-support/)을 따릅니다.

~~~powershell
cd C:\code\st4rtrack_pgd
.\scripts\docker.ps1 -Action Build
.\scripts\docker.ps1 -Action Smoke
.\scripts\docker.ps1 -Action Test
~~~

이미지는 st4rtrack-pgd:torch2.7.1-cu128입니다. 인자 없이 .\scripts\docker.ps1을 실행해도 Smoke를 실행합니다. docker compose run --rm experiment 또는 docker compose up의 기본 명령도 Smoke입니다. PowerShell 스크립트는 일회용 컨테이너를 종료 후 제거합니다. docker compose up을 직접 사용한 뒤에는 docker compose down으로 해당 컨테이너를 정리합니다.

Linux에서는 프로젝트 루트에서 docker compose build, docker compose run --rm experiment, docker compose run --rm experiment python -m pytest -q -p no:cacheprovider를 사용합니다. 준비된 docker/manifests/*.json이 있으므로 이미지 빌드에 호스트 Python은 필수가 아닙니다.

## 연결하는 파일

| 호스트 경로 | 컨테이너 경로 | 권한 |
|---|---|---|
| assets/checkpoints | /workspace/assets/checkpoints | 읽기 전용 |
| data/worldtrack_release | /workspace/data/worldtrack_release | 읽기 전용 |
| runs/docker | /workspace/runs/docker | 읽기·쓰기 |
| .cache/docker | /workspace/.cache | 읽기·쓰기 |
| configs | /workspace/configs | 읽기 전용 |
| docker/manifests | /workspace/docker/manifests | 읽기 전용 |

모델·데이터·Windows 가상환경·기존 결과는 이미지에 넣지 않습니다. 공식 코드는 이미지 안에서 고정 commit 0f9a3f44a7ebac76600cd31ec9eea5228ad7db91로 clone하며 Git metadata를 유지해 provenance 검사가 작동합니다. 실험 패키지는 /workspace에서 editable 설치해 원래 ROOT 경로 규칙을 유지합니다. 이미지 파일 시스템은 실행 중 읽기 전용이며 임시 파일은 /tmp의 tmpfs를 사용합니다.

Docker용 manifest는 기존 선택 순서, 시퀀스, 프레임 수, 데이터 SHA-256을 그대로 유지하고 경로만 프로젝트 상대 경로로 바꾼 사본입니다. 원본 manifests/*.json과 기존 Windows 결과는 유지됩니다. 설정과 Docker manifest는 호스트에서 읽기 전용으로 연결하므로 새 설정을 실행할 때 이미지를 다시 빌드할 필요가 없습니다. 실행 시작 시 설정과 명단의 사본 및 hash를 run.json에 기록합니다.

## 스모크 범위와 출력

스모크의 제한 시간은 Compose에서 240초입니다. 전체 benchmark runner를 호출하지 않습니다. 검사 범위는 다음과 같습니다.

- 공식 source commit·clean 상태와 저자 체크포인트 SHA-256
- PO/DR 각각 2프레임의 공식 전처리, query/GT 좌표, RGB 범위
- CUDA에서 모델의 strict 로딩, 공식 batch_size=1 clean 출력 일치
- 동결된 모델과 모든 입력 프레임의 finite·nonzero gradient
- 모든 pair의 공유 frame 0 및 (0,0) 두 branch의 gradient 합산
- epsilon=4/255의 한 번의 FGSM 갱신과 프레임별 projection
- float 입력·교란 저장, 재로딩한 입력의 궤적·공식 지표 재현

runs/docker/smoke/report.json과 같은 폴더의 *_smoke_tracks.npz에 결과를 저장합니다. 실패하면 JSON에 현재 단계와 traceback을 기록하고 exit 1로 종료합니다. 이 결과는 2프레임 스모크의 결과이며 64프레임의 메모리 사용량이나 PGD 성능을 검증한 결과로 해석하지 않습니다. PGD의 projection, incumbent, restarts와 gradient 수학은 별도 CPU 테스트에서 검사합니다.

이미지 내부의 전체 Python 패키지 목록은 /workspace/docker/image-environment-freeze.txt에 있습니다. 빌드와 컨테이너 정리 기록은 runs/docker_setup_20261004에 저장했습니다.

## 실험을 실행할 때 사용하는 명령

아래 명령은 사용자가 실제 실험을 시작할 때 선택합니다. 기본 Smoke와 별개입니다.

2026-10-04에 요청한 PGD 한 조건은 PO 2클립·DR 2클립, 각 64프레임, epsilon=4/255, PGD-20, restart=1입니다. 원본 clean을 비교용으로 평가합니다. Noise·FGSM과 다른 epsilon은 실행하지 않습니다.

~~~powershell
.\scripts\docker.ps1 -Action PGD
~~~

~~~powershell
# PO 2클립 × 16프레임, PGD-20
.\scripts\docker.ps1 -Action Validation

# PO 10클립 × 64프레임
.\scripts\docker.ps1 -Action Pilot

# PO/DR 각각 50클립 × 64프레임
.\scripts\docker.ps1 -Action Full
~~~

새 실험은 runs/docker/<action>_<timestamp>에 저장됩니다. 중단된 Docker 실험은 같은 이미지·설정·manifest를 유지하고 다음과 같이 재개합니다.

~~~powershell
.\scripts\docker.ps1 -Action Full -RunName full_YYYYMMDD_HHMMSS -Resume
~~~

기존 Windows 실험은 platform·환경·경로가 달라 동일 run으로 resume할 수 없습니다. Docker 결과 폴더를 사용합니다. 세부 공격 설정과 지표 정의는 프로젝트 루트의 README.md를 참조합니다.


## 과거 64프레임 사전 설정과 시간 추정

`configs/loss_components.json`과 `docker/manifests/loss_components_14clips.json`은 Point Odyssey 7클립·Dynamic Replica 7클립의 첫 64프레임을 사용합니다. 기존에 측정한 4클립을 포함하고, 모든 목적에 같은 14클립과 같은 GT/query/mask를 사용합니다. 목적은 `tracking_mse`, `tracking_3d`, `reconstruction_3d`, `confidence`, `joint_training`이며, 각각 epsilon=4/255, PGD-20, step size=epsilon×0.25, restart=1, seed=20261002입니다. `alpha=0.2`, `reweight_scale=5`를 설정하며 모델은 동결합니다.

실제 다운로드된 PO/DR 각 50개 중 첫64프레임 GT가 finite이고 dynamic·static query가 모두 존재하는 후보는 각각 48개입니다. 원본 manifest 순서를 유지하고 기존 두 클립씩을 보존한 뒤, `Random(20261002)` 한 RNG에서 PO 추가6개, DR 추가6개를 순서대로 뽑아 각 앞5개를 사용했습니다. 제외 ID·GT 개수·추가6개 목록·원본 SHA-256·선정 알고리즘은 새 manifest에 기록되어 있습니다. 선택된14개 파일 SHA-256과 전체64프레임 depth finite 여부 및 프레임별 positive depth 개수를 CPU에서 확인했습니다. Depth=0 픽셀은 GT invalid로 처리해야 합니다. 이 명단은 개발용 부분집합이며 논문 평가 시퀀스 목록과의 일치는 검증하지 않았습니다.

완료된4클립 실측은 PGD 구간 1,819.40초, clean 포함 attack 구간 1,848.10초, 컨테이너 전체 2,046.02초(34분6초)였습니다. 고정 startup/finalization/shutdown 30.40초와 클립당503.91초를 외삽하면 14클립은 목적당118.08분, 16클립은134.88분입니다. 따라서14클립×5목적은 약10시간 예상입니다. 이 수치는 기존 tracking-MSE 실행을 바탕으로 한 예상이며 새 loss의 추가 계산 비용·GPU 부하·I/O에 따라 달라집니다. 실행 후 저장된 timing 결과를 실측값으로 보고해야 합니다. 기존 저장 방식 기준 출력은 목적당 약6.38GB이고 새 dense target/output 저장분은 추가됩니다.

새 코드가 포함된 loss 전용 이미지는 `st4rtrack-pgd:loss-components-cu128`입니다. `compose.loss.yaml`은 이미지 태그만 바꾸므로 기본 bounded smoke 명령을 그대로 상속합니다. 기존 `st4rtrack-pgd:torch2.7.1-cu128` 태그는 유지합니다. loss 이미지에는 새 모듈을 bake해야 하므로 아래 별도 빌드를 먼저 실행합니다. Compose override는 호스트에서 읽는 파일이므로 이미지에 복사하지 않습니다.

~~~powershell
cd C:\code\st4rtrack_pgd
docker compose -f compose.yaml -f compose.loss.yaml build experiment
# 기본 명령은 첫2프레임 bounded smoke
docker compose -f compose.yaml -f compose.loss.yaml run --rm --no-deps experiment
# 실제14클립×5목적 PGD
.\scripts\docker.ps1 -Action LossStudy -LossConfig loss_components.json -LossManifest loss_components_14clips.json -FailFast
~~~

실험 출력은 `runs/docker/lossstudy_<timestamp>`입니다. 재개할 때는 같은 이미지·설정·명단을 유지합니다.

~~~powershell
.\scripts\docker.ps1 -Action LossStudy -LossConfig loss_components.json -LossManifest loss_components_14clips.json -RunName lossstudy_YYYYMMDD_HHMMSS -Resume -FailFast
# 선택한 목적·클립수·step 수로 별도 제한 실행
.\scripts\docker.ps1 -Action LossStudy -Objectives tracking_mse -Limit 1 -Steps 1 -FailFast
~~~

제한 실행은 전체 실험의 완료나 실측시간으로 보고하지 않습니다. `-Objectives`, `-Limit`, `-Steps`는 지정한 실행의 범위를 바꾸므로 전체 연구를 재개할 때 원래 범위를 유지해야 합니다.

## 현재 본 실험: PO 4 + DR 4, 각 클립 전체 128프레임

사용자의 전체 프레임 요청에 맞춰 `configs/loss_components_8clips_allframes.json`과 같은 이름의 manifest를 사용합니다. 각 클립의 원본 0~127번을 모두 공격·평가합니다. 다섯 목적은 같은 8클립·GT·초기 noise·epsilon=4/255·PGD-20을 공유합니다. 원본 temporal field 길이를 NPY header에서 검사하고, 128개가 전부 로드되지 않으면 실행을 중단합니다. 64프레임 사전 측정은 별도 기록으로 유지합니다.

고정 RoPE/위치/GT cache, CPU shape, 프레임별 VJP와 묶음 검증을 사용하며 모델 weights와 FP32 정책은 같습니다. RTX 5080의 동일 프로세스 원본/개선 교차 측정에서 128프레임 joint loss gradient는 중앙값 44.62초에서 38.08초로 줄었습니다(14.65%). 예측값과 로스는 같고 RGB gradient 차이는 측정된 CUDA 반복 오차 한도 안입니다. 이 수치는 PGD 한 단계에 해당하는 계산이며 저장·지표 계산 시간은 별도입니다. 약 2시간/목적의 예상은 실제 완료 시간으로 대체해야 합니다.

실제 `lossstudy_allframes_20261004_183750`의 첫 완료 측정은 아래와 같습니다. 모두 클립당 원본 128프레임·PGD-20입니다. MSE는 8클립 전체가 완료됐고 native tracking은 첫 PO·DR 2클립의 부분 측정입니다. 이 시점에는 clean 8개를 포함해 18/48조건이 완료됐으며, 나머지 목적함수와 전체 실험은 미완료입니다.

| 측정 범위·근거 | attack 엔진 시간 | wall 시간·범위 |
|---|---:|---:|
| `tracking_mse` 8클립 ([완료 event](../runs/docker/lossstudy_allframes_20261004_183750/events.jsonl), [8클립 결과](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_mse/)) | 합계 5,557.65초 (92분38초) | objective 전체 6,418.17초 (1시간46분58초) |
| `tracking_3d` 첫 PO `cab_e_3rd_13` ([result](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_3d/po_mini/cab_e_3rd_13/result.json)) | 890.94초 (14분51초) | condition 922.90초 (15분23초) |
| `tracking_3d` 첫 DR `9c43b3-3_obj_source_left_3` ([result](../runs/docker/lossstudy_allframes_20261004_183750/conditions/tracking_3d/ds_mini/9c43b3-3_obj_source_left_3/result.json)) | 909.95초 (15분10초) | condition 943.76초 (15분44초) |

Attack 시간은 동기화된 엔진 계산(내부 clean forward 포함)이며 저장·평가·replay를 제외합니다. Objective wall은 초기 shared clean·로딩·저장·재검증을 포함합니다. Native의 condition wall은 해당 공격과 이후 저장·평가·재검증 구간이며 앞선 로딩·shared clean은 제외합니다.

이 측정은 실험 외 GPU 프로세스도 함께 실행되는 비전용 Windows GPU 환경에서 진행했습니다. `nvidia-smi`의 전체 utilization/memory는 실험과 다른 앱을 모두 합친 수치입니다. [실행환경 관측 기록](../runs/docker/lossstudy_allframes_20261004_183750/runtime_observations.json)은 한 번의 순간 관측이고 앱별 GPU memory가 N/A였으므로, 다른 앱의 실제 GPU 점유나 실험 소요시간에 미친 영향은 정량화할 수 없습니다.

~~~powershell
docker compose -f compose.yaml -f compose.loss.yaml build experiment
# study → 독립 artifact 감사 → 한국어 분석 보고서/그림 순서
docker compose -f compose.yaml -f compose.loss.yaml run --no-deps experiment python -u scripts/run_loss_campaign.py --config configs/loss_components_8clips_allframes.json --manifest docker/manifests/loss_components_8clips_allframes.json --run-dir runs/docker/lossstudy_allframes_RUNNAME
# LossStudy 기본값도 전체128프레임 설정입니다.
.\scripts\docker.ps1 -Action LossStudy -FailFast
~~~

`run_loss_campaign.py`는 다섯 공격과 8개 shared clean, 총48조건을 완료한 뒤 raw RGB/delta/두 head/GT/source SHA와 전체 프레임 증거를 감사합니다. 이어서 `analysis/study_report.md`, `study_analysis.json`, PNG 그림을 생성합니다. `campaign_execution.json`의 complete는 전체128프레임 범위의 감사와 분석까지 통과해야 기록됩니다.
