# St4RTrack 전체 프레임 PGD 로스 비교 실험 재현 안내

작성일: 2026-10-04, Asia/Seoul. 기준 프로젝트: `C:\code\st4rtrack_pgd`.

이 문서는 실제 본 실험의 설정 파일, 구현 코드, 실행 당시 `run.json`을 확인하여 작성했다. 다른 AI에게 이 문서와 실험 소스를 전달해 같은 실험을 실행하거나 별도로 구현할 수 있도록 데이터 선택, 손실 수식, 입력 gradient, 실행 명령, 검증 기준을 정리한다. **현재 기준은 PO 4클립 + DR 4클립, 각 클립의 원본 128프레임 전체, 다섯 목적함수별 PGD-20이다.** 이전 README에 있는 16/64프레임 검증과 14클립 설정은 과거 실행 범위다.

문서 작성 중에는 실험을 새로 실행하거나 현재 컨테이너·소스·설정·결과를 변경하지 않았다. 아래 실행 명령은 재현 환경에서 사용할 절차다.

## 1. 다른 AI에게 전달할 요청문

아래 요청문에 이 문서를 함께 첨부한다. 가능하면 11절의 프로젝트 파일도 전달한다.

```text
첨부한 EXPERIMENT_REPRODUCTION.md를 기준으로 St4RTrack PGD 실험을 재현해줘.

공식 St4RTrack commit 0f9a3f44a7ebac76600cd31ec9eea5228ad7db91과
저자 배포 St4RTrack_Seqmode_reweightMax5.pth를 사용해.
WorldTrack의 Point Odyssey 4클립과 Dynamic Replica 4클립은 문서의
고정 명단과 SHA-256을 따라야 해. 각 클립의 원본 0~127 프레임을 전부 사용해.

tracking_mse, tracking_3d, reconstruction_3d, confidence, joint_training을
각각 최대화하는 PGD를 별도로 실행해. epsilon=4/255, step=1/255,
20회 갱신, restart=1, base seed=20261002야.
같은 클립의 다섯 공격은 동일한 초기 노이즈를 사용하고 clean은 공유해.
alpha=0.2와 동적 점의 max-static-confidence 5배 가중치를 유지해.
GT와 query/mask는 고정하고, confidence와 예측 정규화는 매 반복 재계산해.
동적 confidence maximum에만 원본 코드와 같은 stop-gradient를 적용해.

모델을 strict하게 로드하고 eval/frozen/FP32로 사용해.
공식 batch_size=1 clean과 native loss/gradient를 검증한 뒤 본 실험을 실행해.
메모리가 부족하면 문서의 정확한 recompute/VJP를 사용하고,
프레임 수, 해상도, batch, precision, loss를 임의로 바꾸지 마.

현재 실행 중인 다른 실험이나 컨테이너는 중지·삭제하지 말고,
새 결과 디렉터리를 사용해. 재현 소스가 있으면 그 소스를 우선 사용해.
새로 구현한다면 수식이 비슷하다는 이유만으로 동일 구현이라 보고하지 말고
원본 loss 값, head gradient, RGB gradient 및 저장 입력 replay를 검증해.

최종적으로 8개 clean + 40개 공격 = 48조건과 전체 128프레임의 저장 결과를
감사하고 분석 보고서를 생성해. 실패/미완료/수치 차이는 그대로 기록해.
스모크나 일부 클립의 완료를 전체 실험의 완료로 보고하지 마.
```

## 2. 실험 목적과 범위

동결된 동일 모델에서 **어떤 목적함수를 최대화해 RGB를 공격하느냐에 따라 3D tracking과 reconstruction의 성능이 어떻게 달라지는지** 비교한다. 각 목적함수마다 clean RGB에서 독립적으로 공격을 시작한다. 앞선 공격 결과를 다음 공격의 초기 입력으로 사용하지 않는다.

저자 체크포인트의 활성 학습 로스는 confidence가 포함된 tracking/reconstruction 두 branch다. 이를 가중 tracking 기하 항, 가중 reconstruction 기하 항, 두 branch의 log-confidence 항으로 분해하고 전체 합을 추가한다. 기존 평가 정렬 MSE는 별도의 대조 목적함수다. 따라서 다섯 개의 독립적인 원래 학습 로스를 발견한 실험으로 해석하지 않는다.

모델 재학습이나 loss 삭제 후 재학습은 수행하지 않는다. WorldTrack 평가 GT와 공식 평가 RGB 전처리에서 원본 활성 수식을 적용한다. 원래 학습 augmentation, mesh query 모집단, TTA/pseudo-label 학습까지 재현한 실험은 아니다.

## 3. 공식 코드, 모델, 환경 고정

| 항목 | 기준 |
| --- | --- |
| 공식 저장소 | https://github.com/HavenFeng/St4RTrack |
| 공식 commit | `0f9a3f44a7ebac76600cd31ec9eea5228ad7db91` |
| 체크포인트 | `St4RTrack_Seqmode_reweightMax5.pth` |
| 체크포인트 크기 | 4,411,404,231 bytes |
| 체크포인트 SHA-256 | `cae4712e0265f7d5ecadade19a0cb2ccf7b7e786fa0f230a80603a3437bcdfd3` |
| 저자 체크포인트 폴더 | https://drive.google.com/drive/folders/1uSfnZbzqa8pfIb6k383-BerLQ0m9-R1l |
| 체크포인트 Drive file ID | `1ElgLYxWNHmps7-xvmHz2D6w4B0kZbBd1` |
| 실제 본 실험 환경 | Docker Desktop Linux engine / WSL2, NVIDIA GeForce RTX 5080 |
| Python | 3.12.15 |
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| CUDA wheel | 12.8 |
| 이미지 태그 | `st4rtrack-pgd:loss-components-cu128` |
| 실행 중 컨테이너의 image ID | `sha256:c07e78bf94c671f41392c98e5558cb49296e4187fc7682c51f74467d76539398` |

이미지 ID는 이번 빌드의 식별자다. 동일 Dockerfile을 새로 빌드하면 image ID나 전이 의존성 버전이 달라질 수 있다. 환경까지 최대한 일치시키려면 기존 이미지를 전달하고 이미지 내부 `docker/image-environment-freeze.txt`를 보존한다. 재빌드 시에는 `Dockerfile`, `docker/runtime-requirements.txt`, `docker/torch-constraints.txt`를 사용하고 실제 패키지 목록을 다시 기록한다.

본 실험의 기록된 주요 패키지는 NumPy 2.5.2, SciPy 1.18.1, Pillow 12.3.0, OpenCV 5.0.0.93, einops 0.8.2, roma 1.6.1, huggingface-hub 1.33.0이다. 저자의 torch 2.5.1/CUDA 12.1 환경과는 다르다.

공식 checkout과 submodule을 고정하고 tracked vendor 파일은 변경하지 않는다. 체크포인트의 실제 생성자 설정을 공식 로더와 같은 방식으로 복원하되 `load_state_dict(..., strict=True)`로 로드한다. 모델은 `eval()` 및 모든 parameter의 `requires_grad=False`로 설정한다. RGB 입력에는 gradient가 필요하다. AMP/autocast와 TF32를 끄고 `cudnn.benchmark=False`로 실행한다.

## 4. 데이터셋과 고정 8클립 명단

데이터는 저자가 배포한 [WorldTrack mini-release](https://drive.google.com/drive/folders/1-JW88ru30irMYyFab_4YBQbGbd9tKpXV)의 **Point Odyssey(`po_mini`)와 Dynamic Replica(`ds_mini`)**다. 로컬에는 각각 50개, 총 100개 NPZ가 있다. 클립 하나는 시간 순서로 이어진 128장의 영상과 GT를 담은 NPZ 하나다.

본 실험은 그중 아래 8클립을 사용한다. 두 데이터셋을 번갈아 배치한 manifest 순서까지 유지한다. 이는 개발용 부분집합이며 논문 평가 명단과의 일치는 검증되지 않았다.

| 순서 | 데이터셋 | 클립 ID | 원본 프레임 |
| --- | --- | --- | --- |
| 1 | `po_mini` | `cab_e_3rd_13` | 0~127 |
| 2 | `ds_mini` | `9c43b3-3_obj_source_left_3` | 0~127 |
| 3 | `po_mini` | `dancingroom1_3rd_7` | 0~127 |
| 4 | `ds_mini` | `e09bca-3_obj_source_left_2` | 0~127 |
| 5 | `po_mini` | `dancingroom1_3rd2_4` | 0~127 |
| 6 | `ds_mini` | `fec654-3_obj_source_left_4` | 0~127 |
| 7 | `po_mini` | `seminar_g110_0315_ego2_3` | 0~127 |
| 8 | `ds_mini` | `423875-3_obj_source_left_0` | 0~127 |

원본 NPZ를 수정·재압축하지 않고 아래 SHA-256을 확인한다. 이 값은 다운로드 당시 로컬 파일 식별용 hash이며 저자가 별도로 공표한 checksum은 아니다.

```text
po_mini/cab_e_3rd_13.npz
42d544becff46958108f69c720c912f1acba16e1f5e1d5e1333b8f531eab2ef5
ds_mini/9c43b3-3_obj_source_left_3.npz
3b22401c162ad450e38acd67968f5b1cd4f6d5366bd176ba1c2f066672cdf754
po_mini/dancingroom1_3rd_7.npz
d8c6bcb1259c3b433496cfc20fa3cd67409090a23b9db0a03bf92dc9906c867e
ds_mini/e09bca-3_obj_source_left_2.npz
b7d643787c50c513e93380398627a2ba41c64af751d6dee68cdcdca2611a5a0d
po_mini/dancingroom1_3rd2_4.npz
75dbe756268730912a83a3157b1d070c46c769dd036cf30c011739976ed7fd7c
ds_mini/fec654-3_obj_source_left_4.npz
184d23d10ae8cdea65a709e99b3c5b3010d92181cc90cc4fd9011845de59dc52
po_mini/seminar_g110_0315_ego2_3.npz
677086b367ec8f237f7616262d064a04ed087e04c35ac30e59faa43c61a2b5d7
ds_mini/423875-3_obj_source_left_0.npz
f92865651ef6529a9de5d382c1a18b3ca341467f90071cb032e175633fb8ac62
```

선정 시 모든 128프레임의 GT 유한성, dynamic/static query 존재, 원본 필드 길이, dense depth와 모델 grid의 유효 GT를 검사했다. 전체 프레임 후보는 PO 48개, DR 49개였다. 기존 클립을 데이터셋별로 보존하고 `Random(20261002)`의 고정 후보 순서를 사용했으며 공격 지표로 클립을 고르지 않았다. 정확한 선정 경위는 manifest의 `selection_details`에 있다. **재현할 때 새로 추첨하지 않고 제공된 manifest를 그대로 사용한다.**

## 5. 전처리, GT, query와 두 가지 dynamic mask

### RGB와 카메라 좌표

- 공식 `load_npz_data(normalize_cam=True)` 및 `load_images(size=512, square_ok=True, crop=False, step_size=1)`를 사용한다. 종횡비를 유지하고 최종 크기를 16의 배수로 맞춘다. 현재 클립은 일반적으로 512×288 grid다.
- 공격 텐서는 전처리된 `[128,3,H,W]` float32 RGB `[0,1]`이다. 모델에 넣을 때만 `(RGB-0.5)/0.5`로 정규화한다. 원본 JPEG 공간에서 공격하거나 매 반복 JPEG로 재인코딩하지 않는다.
- `E_t`가 world-to-camera라면 `E'_t = E_t @ inverse(E_0)`로 정규화하고 `inverse(E'_t)`를 사용한다. GT와 두 head의 예측 좌표계는 첫 카메라 기준 world다.
- NPZ의 `images_jpeg_bytes`, `tracks_XYZ`, `visibility`, `depth_map`, `extrinsics_w2c` NPY header에서 원본 시간 길이를 직접 검사한다. 모두 128이어야 하며 실제 로드/저장 프레임은 순서대로 0~127이어야 한다.

### 평가 query

프레임 0에서 visible인 점을 선택하고 최종 이미지 크기로 xy를 스케일한다. 이미지 안에 있는 query만 고정한다. head 1의 `pts3d`를 **정수 truncation**한 xy로 읽는다. 뒤 프레임의 visibility로 점을 제외하지 않는다. GT/query/mask는 다섯 공격에서 동일하다.

평가의 dynamic subset은 GT의 누적 이동 거리 `sum_t ||G[t+1]-G[t]||_2 > 0.01 m`이다.

### Native tracking query와 동적 가중치 mask

Native 학습 수식을 적용하는 query는 초기 평가 query를 별도로 rasterize한다. `torch.round` 후 이미지 범위로 clamp하고, 같은 pixel에 여러 점이 오면 마지막 원본 query의 GT를 보존한다. 최종적으로 row-major pixel 순서로 읽는다. 평가 truncation query와 혼용하지 않는다.

Native dynamic 점은 고정된 GT에서 다음 점수를 계산해 판정한다.

```text
movement[q] = sum_xyz(abs(G1[127,q]-G1[0,q])) / (norm(G1[0,q])+1e-8)
training_dynamic[q] = movement[q] > mean_q(movement[q])
```

이 mask는 평가용 0.01m mask와 다르다. Native query도 처음 선택한 점을 모든 시점에서 유지한다.

### Dense reconstruction GT

`depth_map`을 최종 RGB grid에 `cv2.INTER_NEAREST`로 resize한다. `fx_fy_cx_cy`는 최종/원본 가로·세로 비율로 각각 스케일한다. 모델 grid의 pixel `(u,v)`에 대해 `(X,Y,Z)=((u-cx)*d/fx,(v-cy)*d/fy,d)`를 계산하고 정규화된 camera-to-world로 변환한다.

Regression 유효 mask는 `depth>0`이고 변환한 world point가 finite인 pixel이다. 전체 dense GT와 depth는 finite여야 한다. invalid depth pixel도 아래 native 정규화에는 원본 코드처럼 포함되므로 먼저 nonfinite GT를 거부한다. 학습용 half-pixel camera 변환을 추가하지 않는다.

## 6. 다섯 목적함수의 정확한 정의

체크포인트에 저장된 활성 criterion 설정은 다음과 같다.

```python
ConfLoss(Regr3D(L21, norm_mode='avg_dis', velo_loss=True),
         alpha=0.2, velo_weight=0, pose_weight=0, traj_weight=0.0,
         align3d_weight=0.0, depth_weight=0, cotracker=False,
         reweight_mode='max', reweight_scale=5.0)
```

계수가 0인 velocity/pose/trajectory/depth/alignment 항은 새 공격 목적에 추가하지 않는다. 두 번째 head가 내놓는 point cloud는 reconstruction 표현이며 별도의 여섯 번째 loss가 아니다.

### 예측 정규화와 confidence 가중치

`P1[t,q]`, `G1[t,q]`는 native tracking 예측/GT, `P2[t,u,v]`, `G2[t,u,v]`는 dense reconstruction 예측/GT다. head 1은 `pts3d`, head 2는 `pts3d_in_other_view`를 사용한다. `C1`, `C2`는 모델의 confidence 출력이다.

```text
s_pred[t] = max(sum_ALL_pixels(norm(P2[t])) / (3*H*W + 1e-8), 1e-8)
s_gt[t]   = max(sum_ALL_pixels(norm(G2[t])) / (3*H*W + 1e-8), 1e-8)
e1[t,q]   = norm(P1[t,q]/s_pred[t] - G1[t,q]/s_gt[t])
e2[t,u,v] = norm(P2[t,u,v]/s_pred[t] - G2[t,u,v]/s_gt[t])

static_max = max(C1[t,q] over all GT-valid static native points and all t).detach()
c1_eff[t,q] = 5*static_max  if training_dynamic[q] else C1[t,q]
```

여기서 **`3*H*W`와 invalid pixel의 정규화 참여를 유지한다.** 고정 upstream의 positional mask 전달 및 XYZ component count 동작을 그대로 재현한 것이다. 통상적인 masked mean norm으로 수정하면 실험이 달라진다.

`s_pred`는 매 PGD 반복 현재 예측에서 계산하고 gradient를 유지한다. head 1 tracking 공격에서도 `s_pred`를 통해 head 2의 입력 gradient가 필요하다. GT 정규화와 GT mask는 고정한다.

Static `C1`과 reconstruction `C2`의 gradient는 유지한다. Dynamic 가중치는 매 반복 현재 static maximum으로 다시 계산하되 **maximum의 gradient만 차단한다.** 첫 clean에서 얻은 값을 20단계 내내 고정하지 않는다. 모델 confidence의 활성화 설정은 원본 `('exp',1,inf)`다.

### Native 세 항과 전체 합

`M1`은 고정 native query/time mask, `M2`는 고정 reconstruction mask다. 모든 시점의 항을 먼저 모아 mean을 계산한다. Reconstruction은 전체 유효 pixel 수를 분모로 쓰며 프레임별 mean을 동일 비중으로 평균하지 않는다.

```text
L_track = mean_M1(c1_eff * e1)
L_recon = sum_M2(C2 * e2) / count(M2)
L_conf  = -0.2 * (mean_M1(log(max(c1_eff,1)))
                  + sum_M2(log(C2)) / count(M2))
L_joint = L_track + L_recon + L_conf
```

| 목적함수 ID | PGD가 최대화하는 값 |
| --- | --- |
| `tracking_mse` | 평가 query에 대한 global median 정렬 후 평균 제곱 3D 거리 |
| `tracking_3d` | `L_track` |
| `reconstruction_3d` | `L_recon` |
| `confidence` | `L_conf` |
| `joint_training` | `L_joint` |

`tracking_3d`와 `reconstruction_3d`는 confidence를 포함한 가중 기하 항이다. 분해할 때 `alpha=0.2`와 5배 가중치를 제거하거나 새로 튜닝하지 않았다. `joint_training`은 원본의 **활성 전체 수식**을 보존한다. 구현은 큰 dense graph를 유지하는 원본 criterion을 그대로 반복 호출하는 대신 compact sufficient statistics로 동등한 수식을 계산한다. 원본 source에서 추출한 loss class를 oracle로 사용해 값과 gradient를 비교한다.

`confidence` 최대화는 음의 log-confidence 항을 높이므로 confidence를 낮추는 방향이다. 가중 기하 항이 커졌다고 반드시 기하 오차만 커진 것은 아니다. 그래서 비가중 normalized L21과 confidence 진단도 저장한다.

### 대조 목적함수 `tracking_mse`

```text
s = median_Meval(norm(G)) / median_Meval(norm(P))
L_mse = mean_Meval(sum_xyz((s*P-G)^2))
```

Norm은 `1e-12`로 clamp한다. Median 원소 수가 짝수면 NumPy와 같이 가운데 두 값의 평균을 사용한다. `s`는 전체 시퀀스 공통 값이고 gradient를 유지한다. confidence나 native dynamic 가중치는 사용하지 않는다. Native L21 정규화와 이 평가 median 정렬을 바꾸어 사용하지 않는다.

## 7. PGD와 시퀀스 전체 입력 gradient

설정 파일은 `configs/loss_components_8clips_allframes.json`이며 현재 내용은 다음과 같다.

```json
{
  "checkpoint": "assets/checkpoints/St4RTrack_Seqmode_reweightMax5.pth",
  "num_frames": 128,
  "image_size": 512,
  "device": "cuda",
  "objectives": ["tracking_mse", "tracking_3d", "reconstruction_3d", "confidence", "joint_training"],
  "epsilon_255": 4,
  "steps": 20,
  "step_size_fraction": 0.25,
  "restarts": 1,
  "seed": 20261002,
  "alpha": 0.2,
  "reweight_scale": 5,
  "gradient_mode": "recompute",
  "save_inputs": true,
  "verify_saved_inputs": true,
  "verify_official_clean": true,
  "runtime_optimization": true,
  "campaign_expected_frames": 128,
  "all_frames": true,
  "campaign_expected_clips": 8
}
```

각 pixel과 각 프레임이 별도의 교란을 가지며, 모든 프레임을 동시에 업데이트한다. L-infinity budget은 각 pixel에 `epsilon=4/255`, step size는 `epsilon*0.25=1/255`다.

```text
x0 = project(clean + Uniform(-epsilon,+epsilon))
for k = 0,...,19:
    gradient = d L_objective(xk) / d xk  # 전체 시퀀스 loss의 gradient
    x(k+1) = clamp_to_[0,1](clamp_to_[clean-epsilon,clean+epsilon](
                 xk + (1/255)*sign(gradient)))
```

Clean, restart 초기 상태, 중간 상태, 최종 20번째 업데이트 상태 중 **해당 목적함수 값이 가장 큰 입력**을 저장한다. 따라서 저장 결과는 마지막 iterate와 다를 수 있고 `selected_state`를 기록한다. 동일값이면 strict `>` 갱신 규칙으로 기존 incumbent를 유지한다. Restart는 1이고 random start는 켠다.

클립별 seed는 Python 내장 `hash()` 대신 다음 규칙으로 만든다. 같은 클립의 모든 목적함수에서 동일 seed와 새 generator를 사용해 초기 노이즈를 공유한다.

```python
key = f"{20261002}:{dataset}:{sequence}:pgd:{4:g}".encode("utf-8")
seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
generator = torch.Generator(device=clean.device).manual_seed(seed)
noise = torch.empty_like(clean).uniform_(-4/255, 4/255, generator=generator)
```

### Recompute/VJP 절차

모델은 `(frame0,framet)` 쌍을 t=0~127에 대해 batch_size=1로 처리한다. 시퀀스 전체 graph를 GPU에 유지하는 대신:

1. 모든 쌍을 forward하여 평가/native tracks, native confidence, head 2 normalization, 유효 pixel별 reconstruction 합계 등의 compact 출력 값을 모은다.
2. Compact 값을 leaf로 두고 전체 시퀀스 loss의 출력 gradient를 계산한다. 전체 시퀀스의 static maximum, normalization 연결, global median, 유효 pixel 수를 이 단계에서 유지한다.
3. 각 쌍을 다시 forward해 출력 값이 일치하는지 검사하고 해당 출력 gradient로 vector-Jacobian product를 계산한다. 해당 쌍의 graph는 즉시 해제한다.
4. 모든 쌍에서 들어오는 frame 0의 gradient를 합친다. `(0,0)`에서는 두 입력 역할에 같은 leaf를 전달해 두 경로를 모두 더한다. 나머지는 각 `framet` gradient에 합친다.
5. 전체 gradient가 finite인지 확인한 후 128프레임을 한 번에 PGD 업데이트한다.

프레임을 독립적으로 공격하거나 쌍마다 frame 0을 따로 업데이트하면 같은 실험이 아니다. RGB backward tensor는 파일로 저장하지 않고 프레임별 norm을 기록한다.

### 적용된 runtime 최적화

`runtime_optimization=true`는 고정 grid의 RoPE 길이를 CPU 상수로 사용하고 position cache, CPU `true_shape`, 고정 GT/유효 index의 GPU cache, 프레임별 local RGB VJP, 묶음 finite/replay 검사로 불필요한 동기화·전송을 줄인다. Vendor 소스와 weights, FP32, loss 수식은 유지한다. Pair batch 증가, AMP, 프레임 생략을 사용하지 않는다.

## 8. 평가와 분석

Tracking은 고정 공식 `dust3r/track_eval_util.py` 함수를 직접 사용한다. APD3D는 0.1/0.3/0.5/1.0m threshold 성공률의 평균으로 `%`를 기록하고 EPE는 m이다. 전체 query와 GT dynamic subset을 각각 평가한다. 공식 방식에 따라 dynamic subset은 subset 내부에서 scale을 별도로 구한다.

Reconstruction은 전처리된 모델 grid의 유효 dense GT pixel에 공식 median scale/threshold 공식을 적용한다. 이는 현재 grid의 correspondence 평가이며 논문의 원본 해상도/Sim(3) reconstruction 점수와 동일하다고 표현하지 않는다.

각 공격의 clean 대비 APD 하락은 percentage point, EPE 증가는 m로 보고한다. 클립별 값을 동등 비중으로 평균하며 query/pixel 개수로 클립을 가중하지 않는다. 분석은 다섯 공격과 clean이 모두 완료된 공통 클립에 paired stratified percentile bootstrap을 적용한다. 기본값은 seed=20261004, 2,000회, 95% 구간이다. PO/DR 각각 재표집하고 전체는 두 데이터셋 50:50 비중을 유지한다. 프레임을 독립 표본으로 재표집하지 않는다.

공격의 raw loss 크기만으로 서로 다른 목적함수의 강도를 순위화하지 않는다. Tracking/reconstruction APD·EPE, 비가중 normalized L21, confidence를 함께 비교한다. Base attack seed는 하나이며 scene 독립성과 scene cluster 보정은 검증되지 않았다.

## 9. 저장 결과와 완료 판정

다섯 공격마다 8클립이며 clean 8개를 공유하므로 총 **48조건**이다. 원본 영상은 8×128=1,024프레임이며 이를 다섯 목적에서 반복 공격한다. 다른 epsilon, uniform noise, FGSM 비교는 현재 본 실험에 포함하지 않는다.

```text
runs/docker/<새 run 이름>/
  run.json
  campaign_execution.json
  events.jsonl
  failures.jsonl                 # 실패가 있는 경우
  summary.json
  sequence_metrics.csv
  aggregate_metrics.csv
  artifact_validation.json
  sequences/<dataset>/<sequence>/
    sequence.json
    sequence_manifest.json
    targets.npz
    reconstruction_gt.npy
    reconstruction_valid.npy
    official_clean_parity.json
  conditions/<clean 또는 목적함수>/<dataset>/<sequence>/
    rgb_float32.npy              # [128,3,H,W], 실제 모델 입력
    delta_float32.npy            # clean과의 차이, 같은 shape
    tracks.npz                  # 평가 예측/GT/mask/query/intrinsics
    components.npz              # native tracks/confidence/head2 통계
    reconstruction_float32.npy  # [128,H,W,3], head 2 예측
    reconstruction_confidence_float32.npy  # [128,H,W]
    history.json                # clean/반복별 loss, 항, gradient norm
    perturbation_norms.json     # 전체/프레임별 L-infinity, L2
    result.json                 # 지표, loss, seed, 시간, replay/hash
    native_loss_oracle.json     # clean 조건에서만
  analysis/
    study_analysis.json
    study_report.md
    component_changes_heatmap.png
    paired_bootstrap_ci.png
    native_l21_changes.png
```

Float32 NPY가 공격의 원본 증거다. PNG/JPEG/영상은 별도 시각화이며 이를 float 입력의 재현 자료로 대체하지 않는다. 모든 PGD iterate의 RGB, raw gradient tensor, 전체 model graph를 저장하는 구성은 아니다.

`result.json`은 저장한 float RGB를 다시 읽어 두 head를 forward하고 출력/지표가 유지되는지 검사한 후 게시한다. 파일별 SHA-256을 함께 기록한다. `run.json`은 config/manifest의 사본과 공식 commit, 체크포인트 hash, 실험 소스 hash, 실제 환경을 기록한다.

본 실험의 완전한 완료는 다음을 모두 만족해야 한다.

- `summary.json`: `complete=true`, `completed_conditions=48`, `expected_conditions=48`, 누락 없음.
- `artifact_validation.json`: `passed=true`, `full_campaign_completed=true`, `all_frames_completed=true`.
- `analysis/study_analysis.json`: `all_raw_frames_campaign_completed=true`.
- `campaign_execution.json`: `status="complete"`, study/artifact_audit/analysis 세 stage 성공.
- 각 클립의 raw temporal field, 로드 및 저장 배열은 모두 128프레임이고 `used_frame_indices=[0,...,127]`이다.

Artifact 감사는 독립 NumPy 수식, 저장 배열, GT/mask, epsilon, hash, 집계를 검사한다. 공식 모델 replay와 RGB gradient의 graph 검증은 실행 중/사전 검증에 기록된 증거를 확인하는 범위다. CPU artifact 감사 자체가 모델을 새로 실행해 RGB gradient를 독립 증명한 것으로 설명하지 않는다. Zero gradient 또는 저장된 교란이 0인 프레임을 성공한 nonzero 공격이라고 숨기지 않으며 stationarity를 따로 보고한다.

## 10. Docker 재현 실행 절차

아래 PowerShell 명령은 **소스가 복사된 새 환경의 프로젝트 루트**에서 실행한다. 경로가 다르면 첫 줄을 해당 위치로 바꾼다. 현재 PC에서 GPU 작업이 실행 중인 동안 이 문서를 확인한다는 이유로 추가 GPU 실험을 시작하지 않는다.

### 데이터와 모델 준비

공식 데이터셋·체크포인트 링크, Windows/Linux 다운로드 전용 명령, 가중치만/데이터만 받는 옵션, 저장 경로와 SHA-256 확인 방법은 [README의 데이터와 모델 가중치 다운로드](../README.md#데이터와-모델-가중치-다운로드)에 있다. Docker만 사용할 경우 이 다운로드 전용 절차를 쓰면 호스트에 PyTorch를 설치할 필요가 없다.

기존 asset을 복사하면 hash를 확인한다. 새로 받을 때는 호스트 Python 환경을 준비해 다운로드 스크립트를 사용한다.

```powershell
Set-Location C:\code\st4rtrack_pgd
.\scripts\bootstrap.ps1 -DownloadAssets
```

이미 준비된 호스트 Python 환경에서 다운로드만 할 때:

```powershell
.\.venv\Scripts\python.exe scripts\download_assets.py
```

다운로더 기본값은 정확한 저자 체크포인트와 PO/DR 각 50개다. `--limit 4`로는 본 실험의 고정 4클립씩이 보장되지 않으므로 사용하지 않는다. 필요한 모델은 `assets/checkpoints/`, 데이터는 `data/worldtrack_release/<dataset>/`에 둔다. 다운로드 출처와 file ID는 `assets/download_manifest.json`에 기록된다.

### 이미지 빌드와 스모크

Docker Desktop Linux engine와 NVIDIA GPU 연결이 준비되어 있어야 한다. Compose의 writable bind directory를 먼저 만든다.

```powershell
New-Item -ItemType Directory -Force -Path runs/docker,.cache/docker | Out-Null
docker compose -f compose.yaml -f compose.loss.yaml build experiment
docker compose -f compose.yaml -f compose.loss.yaml run --rm --no-deps experiment
docker compose -f compose.yaml -f compose.loss.yaml run --rm --no-deps experiment python -m pytest -q -p no:cacheprovider
```

기본 스모크는 PO/DR 각각 2프레임의 CUDA forward/backward, 공식 clean, 공유 frame 0, 1회 FGSM, projection, 저장/replay를 검사하며 timeout은 240초다. 다섯 native 목적의 추가 사전 검증은 별도 명령이다.

```powershell
docker compose -f compose.yaml -f compose.loss.yaml run --rm --no-deps experiment python scripts/component_preflight.py --config configs/loss_components_8clips_allframes.json --manifest docker/manifests/loss_components_8clips_allframes.json --output runs/docker/repro_preflight/report.json
```

Preflight도 PO/DR 각각 첫 2프레임만 사용한다. 공식 native loss의 값/head gradient, native-vs-official RGB gradient, 다섯 목적의 full graph-vs-recompute RGB gradient와 반복 noise를 비교한다. 스모크/preflight 통과는 128프레임 전체 실험의 완료를 뜻하지 않는다.

CUDA backward의 원소별 일치가 모든 환경에서 보장되지는 않는다. 사전 검증은 원소별 엄격 비교 결과를 보존하고, 같은 graph에서 두 번씩 계산한 반복 오차와 교차 차이를 기록한다. Native-vs-official RGB 비교의 기본 relative L2 상한은 `1e-3`, full-vs-recompute preflight 상한은 `2e-3`이며 각각 측정 반복 오차의 `1.5배` 및 최소 `1e-5` 허용폭도 함께 적용한다. 값을 바꾸어 검사를 억지로 통과시키지 않는다.

### 전체 본 실험: 공격 → 감사 → 분석

```powershell
$reproRunName = 'repro_allframes_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
docker compose -f compose.yaml -f compose.loss.yaml run --rm --no-deps experiment python -u scripts/run_loss_campaign.py --config configs/loss_components_8clips_allframes.json --manifest docker/manifests/loss_components_8clips_allframes.json --run-dir "runs/docker/$reproRunName"
```

`run_loss_campaign.py`는 study를 fail-fast로 실행하고, `audit_component_artifacts.py --expected-clips 8 --expected-frames 128 --expected-steps 20`, `analyze_component_study.py`를 순서대로 실행한다. 최종 전체 프레임 범위의 완료 flag까지 검사한다. 이 wrapper를 사용하면 감사/분석을 빠뜨리기 어렵다.

단순 `scripts/docker.ps1 -Action LossStudy`는 study 모듈을 직접 실행한다. 현재 기본 설정은 8클립/128프레임이지만 감사와 분석까지 자동 실행하는 명령은 위 wrapper다. `-Action PGD`, `-Action Full`, 예전 14클립 config를 이번 본 실험의 대체 명령으로 사용하지 않는다.

### 중단된 같은 실행 재개

환경·이미지·소스·config·manifest가 동일한 상태에서 기존 결과 이름을 넣는다. 실제 프로젝트에는 다른 run을 덮어쓰지 않는 signature 검사가 있다. 완료 조건의 hash를 검증하고 건너뛰며 미완료 조건을 다시 시작한다. PGD 중간 iterate checkpoint부터 이어가는 기능은 아니다.

```powershell
$reproRunName = 'repro_allframes_YYYYMMDD_HHMMSS'  # 실제 기존 이름으로 교체
docker compose -f compose.yaml -f compose.loss.yaml run --rm --no-deps experiment python -u scripts/run_loss_campaign.py --config configs/loss_components_8clips_allframes.json --manifest docker/manifests/loss_components_8clips_allframes.json --run-dir "runs/docker/$reproRunName" --resume
```

다른 AI/머신의 별도 재현은 새 run 이름을 사용한다. 이전 PC의 Windows native run을 Docker run으로 resume하지 않는다. 재현 범위를 바꾸려면 별도 실험으로 기록하고 같은 결과 이름에 혼합하지 않는다.

## 11. 전달할 파일과 코드 역할

기존 소스로 실행할 때는 프로젝트 구조를 보존해 다음을 전달한다.

- 이 문서, `README.md`, `pyproject.toml`, `src/`, `scripts/`, `tests/`.
- `Dockerfile`, `.dockerignore`, `compose.yaml`, `compose.loss.yaml`, `docker/runtime-requirements.txt`, `docker/torch-constraints.txt`.
- `configs/loss_components_8clips_allframes.json`, `docker/manifests/loss_components_8clips_allframes.json`.
- `docs/loss_structure.md`, `assets/PROVENANCE.md`, `assets/download_manifest.json` 및 공개 데이터 folder/listing 기록.
- 동일 checkpoint와 고정 8개 NPZ 또는 다운로드 출처. 선정 알고리즘 자체까지 재생성하려면 원래 source manifest도 전달한다.
- 비교 기준으로 이번 run의 `run.json`과 기존 검증/benchmark 보고서. 현재 실행 중인 run의 결과 폴더를 재현 실행의 출력 위치로 사용하지 않는다.

`.venv`를 다른 OS로 복사해 사용하는 대신 환경을 새로 준비한다. Docker 빌드는 고정 공식 source와 submodule을 이미지 안에 clone하므로 호스트 vendor를 이미지에 복사하지 않는다. Host의 config/manifest/checkpoint/data는 읽기 전용 bind이고 결과와 cache만 writable이다. `src`와 `scripts`는 이미지에 bake하므로 코드 수정 후에는 새 이미지가 필요하다.

| 코드 | 역할 |
| --- | --- |
| `worldtrack_adapter.py` | 공식 RGB/카메라 전처리, 평가 query와 dynamic subset |
| `st4rtrack_forward.py` | strict 모델 로딩, frozen `(0,t)` forward, 공식 clean 비교 |
| `component_forward.py` | native query raster, dense GT, head 2 통계, 공식 loss oracle |
| `tracking_loss.py` | 미분 가능한 NumPy-style global median MSE |
| `component_attack.py` | 다섯 objective, exact recompute/VJP, shared-noise PGD/incumbent |
| `runtime_optimization.py` | 고정 grid/position/shape runtime 최적화 |
| `run_component_study.py` | 고정 명단 실행, 공유 clean, float 결과, replay, 집계/resume |
| `scripts/component_preflight.py` | 실제 모델의 제한된 native/RGB gradient 검증 |
| `scripts/run_loss_campaign.py` | 전체 본 실험과 감사/분석 orchestration |
| `scripts/audit_component_artifacts.py` | CPU 독립 수식/배열/hash/전체 프레임 감사 |
| `scripts/analyze_component_study.py` | paired 비교, bootstrap, 한국어 보고서와 PNG |

문서만 전달해 새로 구현하는 경우에는 5~9절을 구현하고 10절의 검증을 동등하게 수행해야 한다. 동일 소스 및 asset을 함께 전달하는 편이 정확한 재현에 유리하다.

## 12. 실행 식별자, hash와 시간 기록

이 문서가 참조한 본 실행은 다음과 같다. 작성 시 확인한 `campaign_execution.json` 상태는 `running`이며 전체 실험이 완료된 것으로 기록하지 않는다.

```text
호스트 run: C:\code\st4rtrack_pgd\runs\docker\lossstudy_allframes_20261004_183750
컨테이너: st4rtrack-lossstudy-allframes-20261004
시작: 2026-10-04 18:37:39 KST (09:37:39 UTC)
config 파일 SHA-256:
02d775afd6e6f94cc54a067aebf75206f754380bb6ddf8fd91305e338ec3aa6c
manifest 파일 SHA-256:
3dfa82fa87cfaa715aa8a86207eabb0e8013f4780d223ed66a61d025ef126577
```

실행 시 `run.json`에 기록된 실험 패키지 hash는 아래와 같다. 문서 작성 당시 호스트의 모든 해당 파일이 실행 당시 hash와 일치함을 확인했다. Raw 파일 hash는 줄바꿈/공백 변경에도 달라진다.

```text
__init__.py          f2dd503922a2ec475edf0faa333f7849cc751a72410419cb3b16415ed5cb541c
common.py            b789911c3cd0cba5cb2782c8d87933eeb03ea8dd66ff781db89d48613239a920
component_attack.py  f8db841669434ec52c3605d0ea3657f817984157681f517f9e82e40bb89390c6
component_forward.py 767abe802d83f22b919ce0f3478bb46aaa0bd17e9e93c65595508a37a5a2a257
evaluate_attack.py   51f3c6427c9b9d14f9798253f99a82c58a2a3d58e529d2dfb1cfabc23643e4ea
make_manifest.py     4a4d6bd3b92a71fcca0920f56ac8336f298da52feb34344f379ac3d8178fd071
pgd_video.py         c4ec84c84733e22e6eedd782c3d5f7e4097f91b59daba9ddf0e58bcf07e6c8f0
run_attack.py        2254594c1e96f9be82765b1e2037884f04bbf34826e1097ee414fc94ae53a4b6
run_component_study.py cef33e0f84072fbbcbd39cc28229f2a6601505c8823fc35a5016ae81b9574e23
runtime_optimization.py 2635a9441cda8443787c15a745a17563b5a72c73e8edc944e6beac38fd376272
st4rtrack_forward.py 4e09533adba567ae082fc7ff8577eb162f0217cf142ebd9bf60863151d33b969
tracking_loss.py     dcde02b43da00be867c02532a8e72f70b8bebffcadc86586a32e2fe90d880e66
validate.py          627d4fb6871b3d6b1fe597b60fb83c69967f21cc32c5f1ae4c7dc7d122992a8a
worldtrack_adapter.py def904ee58759a039c86da867fd687f06c9923a2c9bd4e4e7cff521554255335
```

전체 실험의 정확한 소요 시간은 완료 후 기록한다. 확인된 부분 실측은 다음과 같다.

| 측정 범위 | 실측 시간 |
| --- | --- |
| `tracking_mse` 8클립 공격 엔진 합계 | 5,557.65초, 약 92분38초 |
| 첫 objective 전체 wall | 6,418.17초, 약 1시간46분58초; 초기 shared clean/로딩/저장 포함 |
| `tracking_3d` 첫 PO 공격 엔진 | 890.94초, 약 14분51초 |
| `tracking_3d` 첫 DR 공격 엔진 | 909.95초, 약 15분10초 |
| 별도 128프레임 joint gradient ABAB benchmark | 기존 중앙값 44.62초 → 최적화 38.08초, 14.65% 감소 |

공격 엔진 시간은 CUDA synchronize로 측정하며 내부 clean forward를 포함하고 저장/지표/replay는 제외한다. Objective wall과 campaign wall은 범위가 다르므로 혼합하지 않는다. 다섯 목적의 전체 시간은 일부 측정을 바탕으로 대략 10시간 규모로 예상할 수 있지만 GPU 부하/목적함수/저장에 따라 변하며 완료 실측으로 대체해야 한다.

512×288 grid에서 한 조건의 RGB/delta/head 2 XYZ/confidence float만 약 720 MiB다. 48조건이면 이 네 배열만 약 33.75 GiB이고 GT/targets/기타 결과가 추가된다. 이는 배열 shape로 계산한 예상이다. 재현 환경에서는 이미지/모델/데이터와 별도로 결과용 여유 공간을 확보한다.

## 13. 코드와 근거 자료

현재 PC에서는 다음 파일을 통해 세부 구현 및 근거를 확인할 수 있다. 다른 PC에서는 복사한 프로젝트 루트에 맞춰 경로를 바꾼다.

- [활성 loss 구조 설명](C:/code/st4rtrack_pgd/docs/loss_structure.md)
- [현재 config](C:/code/st4rtrack_pgd/configs/loss_components_8clips_allframes.json)
- [현재 manifest](C:/code/st4rtrack_pgd/docker/manifests/loss_components_8clips_allframes.json)
- [공격 및 component loss 구현](C:/code/st4rtrack_pgd/src/st4rtrack_pgd/component_attack.py)
- [GT/head 통계/oracle 구현](C:/code/st4rtrack_pgd/src/st4rtrack_pgd/component_forward.py)
- [본 실행 provenance](C:/code/st4rtrack_pgd/runs/docker/lossstudy_allframes_20261004_183750/run.json)
- [최적화 교차 측정 보고서](C:/code/st4rtrack_pgd/runs/docker/runtime_optimization_20261004/interleaved_allframes/interleaved_report.json)
- [원본 criterion](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/losses.py)
- [저자 sequence training 설정](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/scripts_run/train_seq_reweight.sh)
- [원본 normalization](https://github.com/HavenFeng/St4RTrack/blob/0f9a3f44a7ebac76600cd31ec9eea5228ad7db91/dust3r/utils/geometry.py)

다른 GPU나 CUDA kernel에서 공격 gradient와 최종 결과의 bitwise 일치까지 보장하지 않는다. 동일 명단·모델·수식·seed·설정과 검증 절차를 유지한 재현인지 먼저 확인하고, 차이는 환경 정보와 함께 수치로 비교한다.
