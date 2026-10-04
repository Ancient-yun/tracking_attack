# St4RTrack PGD 실험 완료 후 HTML 결과 보고서 제작 안내

작성일: 2026-10-04. 대상 run: `lossstudy_allframes_20261004_183750`.

이 문서는 **실험이 끝난 뒤 저장 결과만 사용해 시각화와 한국어 HTML 보고서를 만드는 작업 지시서**다. 사용할 결과 파일, 기존 코드, 필요한 코드 수정, 성능 하락 계산식, 보고서 구성과 검증 기준을 정한다. 이 문서 자체가 최종 실험 결과는 아니다. 작성 당시 나머지 세 목적함수 실험이 실행 중이며, 최종 수치와 HTML은 아직 생성하지 않았다.

현재 자동 파이프라인은 전체 실험 → artifact 감사 → JSON/Markdown 분석과 PNG 3개까지 수행한다. **HTML과 이번 결과에 맞는 정성 비교 영상은 별도 구현이 필요하다.** 아래에서 `신규 구현`으로 표시한 파일과 CLI는 제안하는 인터페이스이며 현재 존재하는 명령으로 취급하지 않는다.

관련 문서: [실험 재현 안내](EXPERIMENT_REPRODUCTION.md), [실제 손실 구조](loss_structure.md), [Docker 실행 안내](../docker/README.md).

## 1. 다른 AI에게 그대로 전달할 요청문

```text
첨부한 HTML_REPORT_GUIDE.md를 따라 현재 St4RTrack PGD 실험의 완료 결과를
시각화하고, 성능이 얼마나 떨어졌는지 설명하는 한국어 HTML 보고서를 만들어줘.

실제 프로젝트는 C:\code\st4rtrack_pgd이고,
결과는 runs\docker\lossstudy_allframes_20261004_183750에 있어.
GitHub에 공개된 reports/ 파일은 중단·재개 시점 snapshot이므로 최종 결과로 쓰지 마.

먼저 전체 48조건, PO4+DR4, 각 원본 128프레임, PGD20의 완료 증거와
artifact_validation.json 및 analysis/study_analysis.json을 검증해.
실행 중이거나 검증에 실패하면 최종 완료 보고서를 만들지 말고 이유를 기록해.
누락·실패·불리한 수치·개선된 수치를 모두 그대로 보여줘.

통계 그림은 scripts/analyze_component_study.py의 기존 PNG를 재사용해.
RGB/추적 영상은 기존 render_pgd_comparisons.py, render_tracking_comparisons.py,
tracking_projection.py를 참고하되 현재 component-study 경로와 스키마로
새 renderer를 구현해. 기존 4클립·64프레임 코드 그대로 실행하지 마.
공식 2D/3D 시각화는 고정 vendor의 tapvid3d.py::visualize_results를 재사용할 수 있어.

새 scripts/render_component_comparisons.py와
scripts/build_component_html_report.py를 구현하고 CLI와 사용법도 남겨줘.
수치 소스·설정·manifest·원본 결과를 수정하거나 공격·GPU 추론을 재실행하지 마.
CPU로 저장 배열을 읽어서 렌더링하고 ffmpeg 인코딩만 해.

clean과 각 공격을 같은 dataset/sequence로 짝지어
tracking/reconstruction APD clean·attack·감소 %p·상대 감소율과
EPE clean·attack·증가 m·배수를 계산해.
전체8·PO4·DR4는 클립 동일 가중치로 집계하고 기존 paired bootstrap CI를 써.
원본 128프레임 모두의 성능을 사용하고 시각화용 점 선택과 구분해.

전체 성능 표, 8클립×5목적의 상세 결과, 그래프, RGB 차이,
GT/clean/attack 추적, 프레임별 EPE, reconstruction 오차 지도,
PGD 진행 곡선, normalization/confidence 진단, 실제 시간,
검증 증거와 해석의 한계를 보고서에 넣어줘.

HTML은 인터넷 없이 file://로 열어도 표·그림·필터가 작동하게 만들고,
큰 영상은 같은 출력 폴더의 상대 경로 assets에 보관해.
최종 결과는 run/html_report/index.html과 assets/, data/, report_manifest.json이야.
HTML을 실제로 열어 값·레이아웃·자산·필터·영상·인쇄 상태를 검증하고
보고서 경로와 근거 파일을 알려줘.
```

## 2. 대상 경로와 실험 범위

### 2.1 경로

| 구분 | 현재 실제 위치 | 용도 |
|---|---|---|
| 실험 프로젝트 | `C:\code\st4rtrack_pgd` | 데이터·vendor·가상환경·실제 결과가 있는 폴더 |
| 코드 공개 clone | `C:\code\tracking_attack` | GitHub 코드와 문서; 대용량 결과·데이터는 없음 |
| 대상 run | `C:\code\st4rtrack_pgd\runs\docker\lossstudy_allframes_20261004_183750` | 최종 결과의 기준 |
| 원본 WorldTrack | `C:\code\st4rtrack_pgd\data\worldtrack_release` | 투영 시 카메라·visibility·원본 SHA 확인 |
| 공식 코드 | `C:\code\st4rtrack_pgd\vendor\St4RTrack` | 고정 공식 시각화 코드 |
| 최종 자동 분석 | `<RUN>\analysis` | 검증된 분석 JSON·Markdown·PNG |
| 제작할 HTML | `<RUN>\html_report\index.html` | 새 보고서의 진입점 |

이후 `<PROJECT>`와 `<RUN>`은 위 실제 경로를 뜻한다. 다른 PC에서는 경로를 인자로 바꾸되 동일한 run·데이터·hash를 사용한다. 저장 JSON 안의 `/workspace/...`는 실행 컨테이너 경로다. 호스트에서 그 문자열을 그대로 열지 말고 `--project-root`와 run 상대 경로로 해석한다.

GitHub의 `reports/lossstudy_allframes_20261004_183750/summary.json`, CSV, pause 및 resume snapshot은 당시 기록이다. **로컬 최종 run의 최신 파일을 읽어야 한다.** `runs/`, `data/`, `assets/`, `vendor/`는 Git에 포함되지 않으므로 clone만으로 영상까지 제작할 수 없다.

### 2.2 보고서에 반드시 적을 실험 조건

| 항목 | 값 |
|---|---|
| 데이터 | 저자 배포 WorldTrack mini-release의 Point Odyssey(`po_mini`)와 Dynamic Replica(`ds_mini`) |
| 선정 범위 | PO 4클립 + DR 4클립, 개발용 부분집합 |
| 시간 축 | 각 NPZ 원본 프레임 0~127 전체, 클립당 128장 |
| 서로 다른 원본 영상 프레임 | 8 × 128 = 1,024장; 목적함수마다 같은 원본을 사용 |
| 조건 수 | clean 8 + 5목적 × 8공격 = 48조건 |
| 공격 | L∞ PGD, ε=4/255, step=1/255, 20회 갱신, restart=1 |
| base seed | 20261002; 같은 클립의 목적함수들은 동일 파생 seed 사용 |
| 모델 | `St4RTrack_Seqmode_reweightMax5.pth`, eval/frozen/FP32 |
| 모델 입력 | 공식 전처리 512×288, `crop=False`, RGB [0,1] 공간 공격 |
| 원본 영상 크기 | PO 960×540, DR 1280×720; 공격은 모델 입력 해상도에서 수행 |
| 목적함수 | `tracking_mse`, `tracking_3d`, `reconstruction_3d`, `confidence`, `joint_training` |
| 공격 간 관계 | clean에서 각각 독립 시작; 앞선 공격 결과를 다음 공격에 누적하지 않음 |

`전체 프레임`은 선정한 각 클립의 시간 축 전체를 의미한다. 전체 WorldTrack 100개 NPZ를 평가했다고 쓰지 않는다. 이전 16/64프레임 실험, noise/FGSM 결과, 짧은 성능 측정 run을 이번 결과와 섞지 않는다.

고정 클립 순서는 다음과 같다. 표와 필터에서 이 순서를 유지한다.

| 순서 | 데이터셋 | 클립 |
|---:|---|---|
| 1 | `po_mini` | `cab_e_3rd_13` |
| 2 | `ds_mini` | `9c43b3-3_obj_source_left_3` |
| 3 | `po_mini` | `dancingroom1_3rd_7` |
| 4 | `ds_mini` | `e09bca-3_obj_source_left_2` |
| 5 | `po_mini` | `dancingroom1_3rd2_4` |
| 6 | `ds_mini` | `fec654-3_obj_source_left_4` |
| 7 | `po_mini` | `seminar_g110_0315_ego2_3` |
| 8 | `ds_mini` | `423875-3_obj_source_left_0` |

## 3. 최종 보고서 생성 전 완료 확인

**파일이 몇 개 생겼다는 사실이나 `summary.complete=true`만으로 완료를 선언하지 않는다.** 실행·독립 배열 감사·분석의 증거가 모두 같은 run과 범위를 가리켜야 한다.

### 3.1 실행 상태

- `<RUN>/campaign_execution.json`과 `<RUN>/resume_execution.json`의 `status`가 `complete`인지 확인한다. 현재 재개 run에는 두 파일이 모두 존재한다.
- 해당 run의 실행이 종료되어 scalar/CSV/events가 더 이상 쓰이지 않는 상태에서 읽는다. 문서 제작 때문에 컨테이너를 중지·삭제하지 않는다.
- 이전 `requested_pause.json` 등은 역사 기록이다. 재개 요청으로 이미 대체된 pause 기록을 현재 중지 지시로 해석하지 않는다.
- `run.json`의 manifest·config·signature를 읽어 목적함수5, 클립8, PO4/DR4, 128프레임, PGD20, ε4가 일치하는지 확인한다.

### 3.2 독립 artifact 감사

`<RUN>/artifact_validation.json`에서 다음을 모두 확인한다.

```text
status == "passed"
passed == true
full_campaign_completed == true
all_frames_completed == true
all_frames == true
source_files_verified == true
expected_conditions == 48
completed_conditions_checked == 48
full_campaign_expected_conditions == 48
campaign_expected_clips == 8
campaign_expected_frames == 128
requested_scope == {"clips": 8, "frames": 128, "steps": 20}
errors == []
missing == []
```

감사는 저장 배열·입력 예산·전체 프레임·고정 GT·native 수식·파일 SHA·수치 소스·공식 vendor·체크포인트를 검사한다. 저장 입력 replay는 실험 중 기록된 검증이며, CPU 감사가 모델을 새로 실행해 replay한 것처럼 설명하지 않는다.

### 3.3 자동 분석

`<RUN>/analysis/study_analysis.json`에서 다음을 확인한다.

```text
status == "complete" 또는 "complete_with_recorded_failures"
all_five_campaign_objectives_completed == true
all_raw_frames_campaign_completed == true
scope.kind == "full_campaign"
scope.clips == 8
scope.dataset_counts == {"po_mini": 4, "ds_mini": 4}
scope.num_frames == 128
scope.steps == 20
scope.epsilon_255 == 4
scope.expected_conditions == 48
scope.all_frames == true
validation.completed_conditions == 48
validation.raw_frame_verified_clips == 8
validation.errors == []
validation.missing_conditions == []
artifact_audit.validated_for_scope == true
bootstrap.common_clips == 8
bootstrap.common_dataset_counts == {"po_mini": 4, "ds_mini": 4}
```

`complete_with_recorded_failures`라면 최종 48조건의 감사 통과와 과거 실패 시도를 구분한다. 보고서에 `validation.recorded_failed_attempts` 및 관련 warnings를 표시하고 `실패 0`으로 바꾸지 않는다. `incomplete`·`failed`·누락·hash 불일치는 최종 완료 보고서를 막는다. 별도 미완료 진단 페이지를 만들 수 있으나 제목과 모든 그림에 실제 완료 수를 명시하고 다섯 목적의 최종 순위를 쓰지 않는다.

분석기는 감사의 `file_sha256`을 자신이 읽은 run/summary/CSV/result/history/sequence 정보와 결합해 오래된 감사 증거를 거부한다. HTML 생성기도 `input_scalar_file_sha256`과 해당 파일의 현재 SHA가 일치하는지 확인한다. `events.jsonl`은 시간 기록이며 배열 감사와 구분한다. 제작 과정에서 읽은 파일의 SHA, 분석기 SHA, 보고서 코드 버전을 별도 manifest에 기록한다.

현재 고정 분석기 SHA-256은 `c867a873aaffa8086d3ca8c7eeb99cacc2ccbb6f522507a55012c0c6d3c4b9c4`, run signature는 `9ddabe1be154335cef96c6b6c13a21aed622415e3514e6649f5f53d96f8c1314`다. 추후 분석기 변경본을 사용하면 별도 분석 디렉터리·코드 SHA·변경 사유를 남기고 동일 버전인 것처럼 보고하지 않는다.

## 4. 어떤 결과 파일을 읽어야 하는가

### 4.1 디렉터리 구조

```text
<RUN>/
  run.json
  summary.json
  sequence_metrics.csv
  aggregate_metrics.csv
  events.jsonl
  campaign_execution.json
  resume_execution.json
  artifact_validation.json
  analysis/
    study_analysis.json
    study_report.md
    component_changes_heatmap.png
    paired_bootstrap_ci.png
    native_l21_changes.png
  sequences/{dataset}/{sequence}/
    sequence.json
    sequence_manifest.json
    targets.npz
    reconstruction_gt.npy
    reconstruction_valid.npy
    official_clean_parity.json
  conditions/{objective}/{dataset}/{sequence}/
    result.json
    rgb_float32.npy
    delta_float32.npy
    tracks.npz
    components.npz
    reconstruction_float32.npy
    reconstruction_confidence_float32.npy
    history.json
    perturbation_norms.json
```

`objective`는 `clean` 또는 다섯 공격 ID다. **공통 sequence 메타데이터와 조건별 예측은 서로 다른 경로에 있다.** clean 조건에는 공식 native loss 대조 기록인 `native_loss_oracle.json`도 있다.

### 4.2 용도와 필드

| 파일 | 보고서에서의 사용 |
|---|---|
| `analysis/study_analysis.json` | 완료 범위, paired 평균/클립별 차분, bootstrap CI, 실제 시간, 검증 및 해석의 주 데이터 |
| `analysis/study_report.md` | 기존 해석·수식·단위의 참고; HTML을 단순 변환하는 것으로 작업을 끝내지 않음 |
| `result.json` | 실제 clean/attack 지표, objective/seed/selected state, 저장 hash, replay 확인 |
| `sequence_metrics.csv` | 48조건 scalar 교차 확인·다운로드용 |
| `aggregate_metrics.csv` | 기존 평균의 교차 확인; clean과의 paired 차분은 분석 JSON 사용 |
| `sequence.json` | 원본/전처리 크기, 전체 프레임 증거, query 규칙, 좌표계, point ID |
| `rgb_float32.npy` | 압축 전 모델 입력 RGB; clean/attack 비교의 기준 |
| `delta_float32.npy` | 실제 perturbation; RGB 차분과 일치 확인 |
| `tracks.npz` | benchmark 추적 pred/GT/mask, GT와 예측의 2D/3D 진단 시각화 |
| `reconstruction_float32.npy` | head 2 dense pointmap; reconstruction 오차 지도/3D 표시 |
| `reconstruction_confidence_float32.npy` | dense confidence 점수 지도 |
| `components.npz` | native query, tracking confidence, 프레임별 normalization 및 reconstruction 합계 진단 |
| `targets.npz` | 고정 native GT/query/mask 및 GT normalization; benchmark 모집단과 구분 |
| `history.json` | PGD loss 진행, 프레임별 gradient/perturbation, 선택 iterate |
| `perturbation_norms.json` | L∞/프레임별 L2와 attacked_frames 검증 |
| 원본 데이터 NPZ | 저장 GT와 카메라 투영·query ID·visibility 일치 검증 |

주요 배열 스키마는 다음과 같다. `T=128`, `H=288`, `W=512`; `Q`와 `Q_native`는 클립마다 다르다.

```text
rgb_float32.npy, delta_float32.npy          float32 [T,3,H,W]
tracks.npz: pred, gt                        float32 [T,Q,3]
tracks.npz: valid                           bool [T,Q]
tracks.npz: dynamic                         bool [Q]
tracks.npz: query_xy                        [Q,2]
tracks.npz: intrinsics                      원본 카메라 파라미터
reconstruction_float32.npy                  float32 [T,H,W,3]
reconstruction_confidence_float32.npy       float32 [T,H,W]
reconstruction_gt.npy                       [T,H,W,3]
reconstruction_valid.npy                    bool [T,H,W]
components.npz: track_conf                  float32 [T,Q_native]
components.npz: reconstruction_norm         float32 [T]
targets.npz: reconstruction_gt_norm         float32 [T]
targets.npz: reconstruction_valid_counts    int64 [T]
```

`components.npz`는 8개 compact 출력(`tracks`, `native_tracks`, `track_conf`, `reconstruction_norm`, `reconstruction_regression_sum`, `reconstruction_log_conf_sum`, `reconstruction_l21_sum`, `reconstruction_conf_sum`)을 담는다. full dense map의 저장소로 가정하지 않는다. Native raster query와 평가 query는 round/truncate/collision 및 mask 규칙이 달라 `Q_native=Q`라고 가정할 수 없다.

배열은 `np.load(..., allow_pickle=False)`로 읽고, 큰 NPY는 `mmap_mode="r"`로 프레임 단위 처리한다. 여러 조건의 dense map 전체를 한꺼번에 메모리에 올리지 않는다. JSON은 UTF-8/UTF-8 BOM을 지원한다. 원본 hash 및 `result.artifact_sha256`을 확인하고 입력 파일은 읽기 전용으로 취급한다.

## 5. 어느 파일의 시각화 코드를 사용할 것인가

### 5.1 이번 결과에 바로 쓸 수 있는 통계 그림

[scripts/analyze_component_study.py](../scripts/analyze_component_study.py)의 `make_plots(report, output_dir)`가 현재 component-study 구조를 지원한다. 자동 생성된 그림을 우선 재사용한다.

| 그림 | 파일 | 읽는 데이터와 해석 |
|---|---|---|
| 성능 변화 heatmap | `analysis/component_changes_heatmap.png` | 다섯 목적의 tracking/recon APD 감소, EPE 증가, 두 head confidence 변화 |
| paired 95% CI | `analysis/paired_bootstrap_ci.png` | 공통 8클립의 tracking/recon APD 감소와 EPE 증가 |
| native L21 변화 | `analysis/native_l21_changes.png` | 두 branch의 비가중 정규화 기하 오차 변화, 무차원 |

같은 파일의 `paired_row`, `aggregate_pairs`, `bootstrap_intervals`, `history_diagnostics`, `event_timing`, `make_report`를 집계·진단·서술의 기준으로 삼는다. 그림의 scope/status 캡션을 잘라내지 않는다. Heatmap은 열마다 최대 절댓값으로 따로 색을 정규화하므로 열 사이 색의 강도를 같은 물리량처럼 비교하지 않는다.

### 5.2 수정 후 재사용할 정성 시각화

다음 코드는 **이전 clean/`pgd_eps4` 실험용**이다. 아래 함수는 재사용할 수 있지만 원래 main/CLI는 이번 run에 그대로 실행할 수 없다.

| 기존 코드 | 재사용할 함수/처리 | 이번 실험에 필요한 수정 |
|---|---|---|
| [render_pgd_comparisons.py](../scripts/render_pgd_comparisons.py) | `to_rgb8`, `render_frame`, `encoder_args`, `encode_frames`; clean/attack/차분 패널 | objective 인자, 현재 경로·지표 필드, 128프레임 검증, 파일명에 objective |
| [tracking_projection.py](../scripts/tracking_projection.py) | `load_projected_tracks`의 원본 SHA/GT/query 검증, `_project`, `_camera_to_xy` | 공통 sequence와 조건 디렉터리 분리, `tracking_metrics`, replay 필드, objective 선택 |
| [render_tracking_comparisons.py](../scripts/render_tracking_comparisons.py) | `select_queries`, `farthest_points`, `overlay_panel`, `epe_series`, `chart`, `render_frame` | 새 projection 반환값과 현재 스키마, 4클립·64프레임 제한 제거 |
| [render_official_visualizations.py](../scripts/render_official_visualizations.py) | `import_official`, `checked_saved_arrays`, `probe_video`, `join_official_videos` 및 영상 생성 호출 | fixed clean/pgd_eps4·4클립 검사와 old result 필드 변경 |
| `vendor/St4RTrack/dust3r/datasets/tapvid3d.py` | 공식 `visualize_results(...)` | RGB 목록과 `tracks.npz`의 GT/scale 정렬 예측 궤적 [T,Q,3], 카메라·resize intrinsics로 호출; 추가 inference 없음 |

원래 RGB/추적 renderer의 인코딩은 Docker 안의 CPU ffmpeg를 사용한다. 공식 wrapper는 로컬 ffmpeg/ffprobe와 vendor 의존성을 사용한다. CPU renderer 구현 시 둘 중 한 방식을 명시적으로 선택하고, 렌더링을 위해 GPU 컨테이너나 모델 forward를 만들지 않는다. 실행 중인 실험 이미지나 패키지를 변경하지 않는다.

반드시 고칠 호환성 차이는 다음과 같다.

```text
old: sequences/{dataset}/{sequence}/clean 또는 pgd_eps4
new: conditions/{clean 또는 objective}/{dataset}/{sequence}
     공통 메타데이터는 sequences/{dataset}/{sequence}

old: result["method"]                 → new: result["objective"]
old: result["metrics"]                → new: result["tracking_metrics"]
old: replay["max_track_error"]        → 이번 결과에는 없음
new: saved_input_verification["passed"],
     ["capture_output_errors"], ["compact_output_errors"],
     ["max_reconstruction_error"], ["max_reconstruction_confidence_error"]
```

패키지 소스 경로는 `src/st4rtrack_pgd/`다. 저장 구조는 `run_component_study.py::save_condition`, reconstruction 평가는 같은 파일의 `reconstruction_metrics`, 추적 평가는 `evaluate_attack.py`, native 정의는 `component_forward.py`와 `component_attack.py`를 참고한다. 기존 감사기 [audit_component_artifacts.py](../scripts/audit_component_artifacts.py)를 검증 기준으로 재사용한다. 보고서 제작을 위해 서명된 공격·평가 수치 코드를 수정하지 않는다.

## 6. 성능 하락을 계산하고 설명하는 방법

### 6.1 기준 지표와 단위

| 의미 | `result.json` 필드 | 방향·단위 |
|---|---|---|
| Tracking APD3D | `tracking_metrics.apd3d_all` | 높을수록 좋음, 이미 0~100의 % 값 |
| Tracking EPE | `tracking_metrics.epe_all_m` | 낮을수록 좋음, m |
| Reconstruction APD3D | `reconstruction_metrics.apd3d` | 높을수록 좋음, 이미 % 값 |
| Reconstruction EPE | `reconstruction_metrics.epe_m` | 낮을수록 좋음, m |

APD3D는 0.1/0.3/0.5/1m 문턱 내 비율(%)의 평균이다. JSON의 APD와 `within_*`는 이미 퍼센트이므로 다시 ×100하지 않는다. 필요하면 문턱별 표와 `apd3d_dynamic`·`epe_dynamic_m`을 부록에 넣되, 주 결과는 전체 평가 query 지표로 통일한다. 동적 query가 없는 경우 `null/N/A`를 표시하고 dynamic 지표는 별도 alignment를 사용한다는 점을 적는다.

Reconstruction은 **512×288 전처리 grid에서 고정 finite positive-depth GT와 평가한 공간 대응 결과**다. Tracking과 reconstruction 모두 기록된 global median scale 정렬을 사용한다. 논문의 원본 해상도 reconstruction/Sim(3) benchmark와 같은 점수라고 설명하지 않는다.

### 6.2 clean 대비 변화

동일한 `(dataset, sequence)`의 clean과 해당 objective를 짝지어 계산한다.

```text
APD 감소량 (%p)    = clean_APD - attacked_APD
APD 상대 감소 (%) = 100 * (clean_APD - attacked_APD) / clean_APD
EPE 증가량 (m)    = attacked_EPE - clean_EPE
EPE 배수          = attacked_EPE / clean_EPE
```

`%p`와 `%`를 구분한다. 예를 들어 **설명용 가상 수치** clean APD80%, attack20%는 60%p 감소, 상대75% 감소다. clean EPE0.20m, attack1.00m는 0.80m 증가, 5배다. 이 예시는 실제 결과 표에 넣지 않는다.

감소량·증가량이 음수면 실제 개선으로 설명한다. 0으로 자르거나 `공격 성공`에 맞추어 부호를 바꾸지 않는다. 비율은 clean이 유한한 값이며 `>1e-12`일 때만 계산하고, 분모가 0에 가깝거나 값이 없으면 `null/N/A`다. `Infinity`, `NaN` 또는 0을 대신 표시하지 않는다.

### 6.3 분석 JSON 필드와 평균

`study_analysis.json.paired_conditions`는 40개 공격의 paired 상세 행, `paired_aggregates`는 `(objective, dataset)`별 평균이다. `dataset`은 `all`, `po_mini`, `ds_mini`다.

```text
tracking_apd_drop_pp
reconstruction_apd_drop_pp
tracking_epe_increase_m
reconstruction_epe_increase_m
```

각 키에는 `_clean`과 `_attacked` 값이 함께 있다. 예: `tracking_apd_drop_pp_clean`, `tracking_apd_drop_pp_attacked`. `paired_clips`는 all8, PO4, DR4여야 한다. `tracking_native_l21_change`, `reconstruction_native_l21_change`, `track_conf_mean_change`, `track_raw_conf_mean_change`, `reconstruction_conf_mean_change`는 보조 진단이다.

원칙은 **클립 동일 가중 평균**이다. 많은 query나 많은 valid pixel을 가진 클립이 전체 결과를 지배하도록 데이터를 한꺼번에 합치지 않는다. 이번 완성 범위의 all8 평균은 PO4/DR4를 각50%로 평균한 값과 같다. 클립 내부 지표는 원래 mask·모집단을 유지한다.

상대 APD 감소율과 EPE 배수는 기존 분석기가 제공하지 않으므로 HTML builder에서 추가한다. 주 요약의 비율은 `평균 clean과 평균 attack의 비율`로 정의하고 클립별 비율 평균과 혼용하지 않는다. 클립별 상세 표는 각 clip 자체 비율을 쓴다. 계산은 반올림 전에 수행하고, 화면은 APD·%p·상대% 소수2자리, EPE m 소수3자리, 배수 소수2자리 정도로 표시한다. JSON에는 원래 정밀도를 보존한다.

### 6.4 신뢰구간

기존 `bootstrap.estimates`에서 `dataset`, `objective`, `metric`으로 조회한다. 각 항목의 `estimate`, `ci95=[lower,upper]`, `unit`, `common_clips`를 사용한다.

- 모든 목적함수와 clean이 있는 공통 8클립만 사용한다.
- PO4와 DR4 안에서 클립을 복원 추출하고 all은 두 dataset 각50%다.
- 2,000회, seed20261004, paired stratified percentile 95% CI다.
- 동일한 추출 인덱스를 목적함수와 지표 전부에 공유한다.
- 기존 CI는 APD 감소 %p 및 EPE 증가 m의 네 지표에 대한 것이다. 상대%·배수의 CI처럼 붙이지 않는다.
- 128프레임을 128개의 독립 샘플로 세거나 프레임 단위 bootstrap으로 바꾸지 않는다.

CI는 선정한 개발 클립에 조건부인 변동이다. Scene 독립성·scene cluster 보정·attack seed 변동·checkpoint/GT 불확실성·다중 비교 유의성 검정을 포함하지 않는다. CI 비중첩만으로 목적함수 우열의 통계적 확정이나 전체 데이터셋 일반화를 주장하지 않는다.

### 6.5 결과 문장 생성 규칙

주 문장은 실제 표와 연결한다. 예시 템플릿:

```text
[objective] 공격 후 전체 8클립의 tracking APD는 [clean]%에서 [attack]%로
[drop]%p 감소했다(상대 [relative]%, paired 95% CI [lo,hi]%p).
Tracking EPE는 [clean]m에서 [attack]m로 [increase]m 증가했다([ratio]배).
Reconstruction APD는 […]%p, EPE는 […]m 변화했다.
PO4와 DR4에서는 각각 […]로 나타났다. 변화가 가장 큰/작은 클립은 […]다.
```

음수에는 `증가/감소`, `개선` 표현을 값에 맞춰 바꾼다. APD와 EPE가 서로 다른 결론을 보이면 둘 다 설명한다. `가장 강한 목적함수`라고 단정하지 말고 `이 8클립에서 tracking APD 감소가 가장 컸던 목적함수`처럼 지표와 범위를 붙인다. Tracking 손상과 reconstruction 손상이 다른 순위일 수 있다.

원래 활성 학습식은 confidence가 포함된 두 branch다. 다섯 공격은 이를 분해한 목적들과 MSE 대조군이다. 서로 다른 raw attack loss의 크기나 gradient norm으로 성능 손상 순위를 정하지 않는다. 본 실험은 **고정 모델의 입력 공격 민감성 비교**이며 loss 제거·재학습의 인과적 중요도, 전체 데이터셋 결과 또는 전역 최악 공격을 측정한 실험이 아니다.

## 7. 새 renderer에서 만들 정성·진단 그림

신규 파일명 제안: `scripts/render_component_comparisons.py`. 기존 renderer의 유효한 함수·투영 검사·그림 스타일을 재사용하되 이번 schema에 맞는 adapter를 둔다. 한 조건만 시험 렌더링한 뒤 전체 40쌍을 순회한다. 파일명에는 objective/dataset/sequence를 포함한다.

### 7.1 RGB·perturbation 비교

- clean, attack, `abs(attack-clean)` 증폭 차분의 3패널을 만든다.
- 기존 `to_rgb8`와 `render_frame`의 패널 구성을 재사용한다. float 입력이 성능의 기준이며 MP4/PNG는 표시용임을 적는다.
- 기존 증폭 상수는 ×64다. 재사용하면 패널에 `|δ| ×64, 표시 범위에서 clip`을 명시하고 실제 변화 크기와 혼동하지 않게 한다.
- 실제 δ와 `delta_float32.npy`의 일치, ε4/255와 저장 perturbation 검증을 유지한다.
- 모든 8클립×5목적에 local frame0/64/127 스틸을 생성한다. source index는 `sequence.json.used_frame_indices`로 표기한다.
- MP4는 **128프레임 전부** 인코딩한다. 프레임 선택 옵션은 스틸용이며 공격·평가·영상의 T를 줄이지 않는다.

### 7.2 GT/clean/attack tracking overlay

- `tracking_projection.py`의 원본 NPZ SHA, 저장 GT/query/카메라 일치 검사를 보존한다.
- 첫 카메라 world 좌표에서 `E_t = raw_E_t @ inverse(raw_E_0)`와 원본 intrinsics 및 resize를 사용한다.
- 예측은 각 조건의 `pred * np.float32(tracking_metrics.scale_all)`로 정렬한다. 새 회전·이동·프레임별 scale을 fit하지 않는다.
- GT는 정답 카메라에, 각 clean/attack 예측은 해당 RGB 패널에 투영한다. `GT 카메라를 사용한 저장 3D 궤적 진단 투영`이라고 캡션을 붙인다.
- `select_queries`/`farthest_points`를 이용해 GT 기준의 같은 point ID 최대18개를 clean 및 모든 목적에 공유한다. 선택 seed/ID/규칙을 manifest에 남긴다.
- GT visibility는 색·선 스타일에만 쓰고, 후반 occlusion·화면 밖·negative depth를 quantitative mask에서 임의 제거하지 않는다. 화면 밖 좌표를 테두리로 clamp하지 않는다.
- projection 불능·behind-camera·out-of-frame 수를 표시해 잘못 추적한 점이 단순히 사라진 것처럼 보이지 않게 한다.
- 스틸0/64/127과 전체128프레임 비교 MP4를 만든다. 기존 `overlay_panel`, `chart`, `render_frame`를 현재 데이터로 호출하도록 바꾼다.

### 7.3 프레임별 tracking EPE

신규 생성 항목이다. `tracks.npz`의 모든 평가 query와 저장 mask를 사용한다.

```python
aligned = pred * np.float32(result["tracking_metrics"]["scale_all"])
error_m = np.linalg.norm(aligned - gt, axis=-1)  # [T,Q]
per_frame_epe = [error_m[t][valid[t]].mean() for t in range(T)]
whole_clip_epe = error_m[valid].mean()
```

`whole_clip_epe`를 저장 EPE와 `rtol=1e-6, atol=1e-6`로 확인한다. empty mask는 N/A다. Overlay가18개 점만 보여도 이 곡선은 전체 Q로 계산한다. 클립당 clean+5공격, x축0~127의 곡선을 만들고 상세 탭에서 목적별 비교가 가능하게 한다. Dynamic 곡선을 추가하면 별도의 `scale_dynamic`과 mask를 사용하고 all 곡선과 구분한다.

### 7.4 Reconstruction 오차 지도와 confidence

신규 생성 항목이다. 공통 sequence의 `reconstruction_gt.npy`, `reconstruction_valid.npy`와 각 조건의 dense 예측을 읽는다.

```python
aligned = reconstruction * np.float32(result["reconstruction_metrics"]["scale"])
error_m = np.linalg.norm(aligned - reconstruction_gt, axis=-1)
# 표시·집계는 고정 reconstruction_valid를 사용한다.
```

- frame0/64/127에서 RGB·clean error·attack error·signed error difference를 만든다.
- 클립 내부 clean과 모든5공격의 오차 지도에 같은 meter color scale을 적용한다. signed difference는 0 중심 대칭 scale을 쓴다.
- percentile vmax를 사용하면 산출 모집단/백분위·상한 초과율을 명시한다. 실제 m 값을 숨기지 않고 원본 EPE는 자르지 않는다.
- invalid GT는 별도 색으로 표시한다. 공격 confidence를 이용해 mask를 바꾸거나 좋은 픽셀만 평가하지 않는다.
- 전체 클립 EPE는 valid pixel 총합의 평균이다. 프레임별 valid 수가 달라지는 경우 단순 frame mean 평균으로 바꾸지 않는다.
- Confidence는 모델 점수이며 확률이나 %가 아니다. 같은 legend/범위를 사용하고 raw/effective 값을 구분한다.

### 7.5 PGD 진행·선택 상태·normalization

`history.json`은 clean(restart=-1, step0)과 restart0의 step0~20을 담는다. 20 gradient 행은 step0~19이고 step20은 마지막 forward다. clean 포함 최고 attack objective를 갖는 iterate가 저장되므로 `selected_state`가 step20이 아닐 수 있다.

- 목적별 loss 곡선에 clean 기준선, initial random state, 마지막 상태, 실제 selected state를 표시한다.
- 다른 목적의 raw loss 축을 하나의 공격 강도 축으로 합치지 않는다. 정규화 gain을 추가하면 정의를 명시하고 분모0/음수 문제를 처리한다.
- gradient norm과 L∞/L2는 history에서 읽으며 입력 gradient가0인 프레임도 그대로 표시한다. 유한한 norm 기록만으로 모든 graph 연결을 증명했다고 쓰지 않는다.
- `components.npz.reconstruction_norm`의 clean/attack 값과 프레임별 attack/clean 비율, `targets.npz.reconstruction_gt_norm`을 비교한다.
- `result.loss_terms.diagnostics`의 비가중 `tracking_l21`, `reconstruction_l21`, raw/effective tracking confidence와 reconstruction confidence를 함께 보여준다.
- Native L21은 무차원이다. 현재 입력의 head2 normalization이 두 branch에 연결되므로 scale contraction이나 confidence 변화가 weighted loss를 크게 만들 수 있다. 이를 meter EPE 손상과 같은 크기로 해석하거나 독립 head 인과 효과로 단정하지 않는다.

### 7.6 공식 2D/3D 영상과 예시 선택

공식 tracking visualizer는 선택적 부록이다. Pinned `tapvid3d.py::visualize_results`에 RGB와 GT/scale 정렬 예측 궤적 [T,Q,3]을 전달하고 wrapper의 배열 검사·카메라 처리·ffprobe 검사를 재사용한다. PyTorch/vendor 모듈 import가 필요해도 모델·체크포인트 로딩이나 forward는 하지 않는다. 표시할 query sampling은 시각화용이며 지표를 재산출하지 않는다. Dense reconstruction은7.4절의 저장 pointmap으로 오차 지도를 만들며, 위 tracking 함수에 dense pointmap을 전달하지 않는다.

기본 대표 사례는 manifest 첫 PO(`cab_e_3rd_13`)와 첫 DR(`9c43b3-3_obj_source_left_3`)로 정해 다섯 목적을 모두 비교한다. 최악 사례를 추가하면 선택 기준(예: tracking APD drop 최대), 동률 처리와 `가장 큰 하락 사례`라는 라벨을 붙인다. 전체8클립 상세 표·그림을 유지하여 대표 사례만으로 결론을 만들지 않는다.

영상 FPS는 preview 설정이다. 128장 기준10fps는12.8초, 6fps는약21.33초, 15fps는약8.53초이며 원본 촬영 FPS를 뜻하지 않는다. 표시 frame index, 저장 source index, FPS, point 선택 수를 캡션과 manifest에 기록한다.

## 8. HTML 보고서의 구성

신규 파일명 제안: `scripts/build_component_html_report.py`. `study_analysis.json`과 verified scalar를 주 데이터로 삼고 PNG·정성 결과를 상대 경로로 묶는다. 기존 Markdown을 HTML로 바꾸는 것만으로 완성하지 않는다.

| 순서 | 섹션 | 필수 내용 |
|---:|---|---|
| 1 | 제목·핵심 결과 | run ID, 검증 상태, 작성 시각/KST, 48/48, 8클립/128프레임, 실제 핵심 수치 |
| 2 | 실험 조건 | checkpoint/commit, PO4+DR4, ε/step/PGD/seed/전처리, 다섯 objective의 의미 |
| 3 | 얼마나 떨어졌는가 | 5목적 요약 표: clean/attack APD%, drop%p, 상대%, EPE m, 증가m, 배수, 95% CI |
| 4 | 전체·PO·DR 비교 | dataset 필터와 클립 동일 가중치, 각 그룹 support 8/4/4 |
| 5 | 통계 시각화 | 기존 heatmap·paired CI·native L21 PNG와 단위/해석 캡션 |
| 6 | 클립별 상세 | 8클립×5공격의 40행, 전체 프레임 EPE 곡선, 선택 objective/clip의 RGB·추적·recon 지도 |
| 7 | PGD 및 내부 진단 | history/selected state, normalization 변화, raw/effective confidence, native 오차 |
| 8 | 실제 실행 시간 | attack/condition/objective wall, pause/resume 구분, 분석·렌더링 시간 별도 |
| 9 | 검증·실패 기록 | audit/analysis 완료 증거, replay 기록, SHA, warnings/과거 실패/정성 렌더링 실패 |
| 10 | 해석과 제한 | 고정 모델 입력 공격, 개발8클립, 단일seed, CI 한계, reconstruction protocol, 공유 normalization |
| 11 | 재현·근거 | artifact 및 코드 경로, 생성 command, 보고서 manifest, 작은 scalar/CSV 다운로드 |

요약을 먼저 보여주고 전문 진단은 뒤에 둔다. 첫 화면에 다섯 raw loss 숫자를 나열해 성능 비교처럼 보이게 하지 않는다. 자연어 결론은 실제 데이터를 이용해 생성하고 값이 없으면 문장을 생략하거나 N/A로 표시한다.

단위가 다른 지표를 한 축에 그리지 않는다. 감소/증가 부호가 양수이면 해당 APD/EPE 정의에서 손상이고 음수이면 개선이다. Confidence 변화는 별도의 score 변화로 해석한다. 극단값 때문에 log axis를 사용하면 축 이름과 0/음수 처리 방식을 표시한다.

HTML은 다음 사용성을 갖춘다.

- 한국어, `<html lang="ko">`, UTF-8, responsive 폭, 읽기 쉬운 글자/대비, 표 가로 스크롤.
- objective·dataset·clip 필터, 검색·정렬, 원래 manifest 순서로 복귀 가능.
- 관련 그림·표의 scope와 단위, alt text/figure caption, 클릭 확대 또는 원본 그림 링크.
- 영상은 `controls`, `preload="metadata"`, poster를 제공하고 여러 영상을 자동 재생하지 않는다.
- 브라우저 fetch 없이 HTML 내부의 작은 JSON으로 필터를 구동한다. file://에서도 동작한다.
- CSS/JS/폰트는 로컬 또는 inline로 제공하고 CDN·외부 Plotly script·네트워크 자산에 의존하지 않는다.
- print CSS에서 필터·대형 영상을 정리하고 표·핵심 PNG·결론을 인쇄할 수 있게 한다.
- 분석이 complete라도 일부 영상 렌더링 실패가 있으면 자산 오류를 명시하고 빈 플레이어를 성공처럼 표시하지 않는다.

## 9. 출력 폴더와 생성 manifest

```text
<RUN>/html_report/
  index.html
  assets/
    component_changes_heatmap.png
    paired_bootstrap_ci.png
    native_l21_changes.png
    comparisons/{objective}/{dataset}/{sequence}/
      rgb_frame_000.png, rgb_frame_064.png, rgb_frame_127.png
      tracking_frame_000.png, tracking_frame_064.png, tracking_frame_127.png
      reconstruction_frame_000.png, ...
      rgb_comparison.mp4
      tracking_comparison.mp4
      pgd_history.png
      normalization.png
    timelines/{dataset}/{sequence}/tracking_epe.png
  data/
    report_data.json
    sequence_metrics.csv
    aggregate_metrics.csv
  report_manifest.json
```

보고서 자체는 한국어로 작성한다. 위 출력 폴더는 제작할 산출물의 명세이며 현재 생성됐다는 뜻이 아니다. 큰 NPY/NPZ·checkpoint·원본 데이터는 복사하지 않는다. PNG는 base64로 넣은 별도 `standalone.html`을 선택적으로 만들 수 있지만 MP4까지 대량 embed하여 거대한 단일 파일을 만들 필요는 없다. 기본 납품은 `index.html`과 상대 참조하는 폴더 전체다.

`report_manifest.json`에는 최소한 다음을 기록한다.

```text
schema_version, generated_at_utc, generated_at_kst
run_id, run_signature, project_root, run_dir
analysis status/scope, 감사 상태와 완료 플래그
analysis_json_sha256, artifact_validation_sha256
scalar_input_sha256, 사용한 배열/result/source NPZ hash
analyzer_source_sha256, renderer/builder source SHA 또는 Git commit
official commit, checkpoint SHA, config/manifest hash
objective 순서, 고정8클립 명단, frame/source index, 표시 query ID
bootstrap 설정, preview FPS, δ 증폭, color scale 규칙
생성 명령, 시작/종료 시각, 출력 asset hash, 누락/실패 목록
```

동일 입력으로 재생성할 때 기존 보고서를 조용히 덮어쓰지 않고 임시 출력에서 검증한 뒤 교체하거나 새 output-dir을 쓴다. 깨진 build를 최종 `index.html`로 남기지 않는다. 경로·클립 이름·JSON 문자열을 HTML에 넣을 때 escape하고, script 내부 JSON의 `</script>` 종료 문자열도 안전하게 처리한다. 사용자 설명에 가상 수치나 placeholder가 남지 않게 한다.

## 10. 실행 절차와 명령

### 10.1 이미 구현된 분석 단계

자동 파이프라인의 `<RUN>/analysis`가 검증을 통과하면 재분석하지 않고 사용한다. 분석 결과가 없거나 의도적으로 새 분석을 만들 때만 아래를 **실험 완료 및 감사 후** 사용한다. CPU 분석이고 모델·CUDA·Docker·원본 NPZ를 실행하지 않는다.

```powershell
Set-Location C:\code\st4rtrack_pgd
$reportRun = 'C:\code\st4rtrack_pgd\runs\docker\lossstudy_allframes_20261004_183750'

# 의도적인 재분석은 기존 자동 analysis를 보존한다.
.\.venv\Scripts\python.exe scripts\analyze_component_study.py `
  --run-dir $reportRun `
  --output-dir "$reportRun\analysis_report_refresh" `
  --bootstrap-samples 2000 `
  --seed 20261004

if ($LASTEXITCODE -ne 0) { throw '분석 완료 검증에 실패함. 최종 HTML 제작을 중단한다.' }
```

분석기 종료 코드는 complete0, incomplete2, failed1이다. 재분석을 선택했다면 HTML의 `--analysis-dir`도 `analysis_report_refresh`로 맞추고 새 분석 SHA를 기록한다.

감사기 CLI는 다음과 같다. **현재 캠페인은 자동으로 감사를 수행하므로 재실행할 필요가 없다.** 별도 감사 복구 시 원래 프로젝트·수치 소스·vendor·checkpoint가 있어야 하며, published clone의 경로를 무조건 쓰지 않는다.

```text
python scripts/audit_component_artifacts.py
  --run-dir <RUN>
  --expected-clips 8 --expected-frames 128 --expected-steps 20
  --project-root <PROJECT>
  [--experiment-source-dir <서명된 원래 수치 소스 snapshot>]
```

`--experiment-source-dir`는 검사 우회를 위한 옵션이 아니라 실행 당시 14개 수치 소스 hash를 일치시키기 위한 것이다. 서명 불일치를 보고서 생성 때문에 무시하거나 원래 기록을 수정하지 않는다. Docker 원래 경로/환경이 필요한 경우 종료 후 별도 CPU 후처리 환경에서 수행한다.

### 10.2 신규 구현할 renderer 인터페이스

**다음 명령은 아직 없다. `render_component_comparisons.py`를 구현하고 검증한 후에만 실행한다.**

```powershell
# 첫 PO의 첫 objective로 저장 결과만 읽는 CPU 렌더링 QA.
.\.venv\Scripts\python.exe scripts\render_component_comparisons.py `
  --run-dir $reportRun `
  --project-root C:\code\st4rtrack_pgd `
  --analysis-dir "$reportRun\analysis" `
  --output-dir "$reportRun\html_report\assets" `
  --objective tracking_mse `
  --clip po_mini/cab_e_3rd_13 `
  --fps 10 `
  --preview-frames 0 64 127
```

`--objective`는 다섯 ID를 허용하고 clean과 자동으로 짝지어야 한다. `--clip`은 선택적 필터이며 미지정 시8클립, `--preview-frames`는 스틸 전용, 영상은128프레임이다. 옵션 예: `--frames-only`는 MP4를 생략하는 QA 모드, `--official-3d`는 공식 부록 렌더링이다. 이 옵션 정의를 코드의 `--help`와 일치시킨다.

QA 후 전체 목적을 처리한다. 아래 foreach 역시 신규 renderer 구현 후 사용한다.

```powershell
$reportObjectives = @('tracking_mse', 'tracking_3d', 'reconstruction_3d', 'confidence', 'joint_training')
foreach ($reportObjective in $reportObjectives) {
  .\.venv\Scripts\python.exe scripts\render_component_comparisons.py `
    --run-dir $reportRun `
    --project-root C:\code\st4rtrack_pgd `
    --analysis-dir "$reportRun\analysis" `
    --output-dir "$reportRun\html_report\assets" `
    --objective $reportObjective `
    --fps 10 `
    --preview-frames 0 64 127
  if ($LASTEXITCODE -ne 0) { throw "렌더링 실패: $reportObjective" }
}
```

### 10.3 신규 구현할 HTML builder 인터페이스

**다음 명령도 `build_component_html_report.py` 신규 구현 후 사용한다.**

```powershell
.\.venv\Scripts\python.exe scripts\build_component_html_report.py `
  --run-dir $reportRun `
  --analysis-dir "$reportRun\analysis" `
  --assets-dir "$reportRun\html_report\assets" `
  --output-dir "$reportRun\html_report" `
  --language ko `
  --require-complete

if ($LASTEXITCODE -ne 0) { throw 'HTML 생성 또는 검증에 실패함.' }
```

`--require-complete`는3절의 모든 gate를 적용한다. 숫자와 자산을 검증한 뒤 HTML을 저장하고 manifest를 남긴다. source NPZ가 없어 정성 그림을 만들 수 없는 경우 수치 보고서의 제한을 명확히 적고, 정성 시각화까지 완료했다고 보고하지 않는다.

### 10.4 CPU 환경과 실행 경계

NumPy/Matplotlib/Pillow와 ffmpeg/ffprobe가 필요하다. 공식 시각화는 mediapy·OpenCV·vendor 관련 의존성도 필요할 수 있다. 프로젝트의 기존 환경과 [runtime requirements](../docker/runtime-requirements.txt)를 먼저 확인한다. 실행 중인 컨테이너에 pip upgrade하거나 Docker 이미지를 재빌드하지 않는다. 필요하면 별도의 CPU 후처리 환경을 준비한다.

현재 Docker 캠페인 driver와 resume controller, 분석기, signed `src`·config·manifest를 수정하지 않는다. 보고서 코드 추가는 새 독립 스크립트와 output 폴더에서 한다. 학습·PGD·추론·CUDA benchmark를 시작하지 않고, 저장 float 배열과 기록만으로 작업한다. 렌더링 편의를 위해128프레임을64로 줄이거나 MP4를 입력으로 성능을 다시 평가하지 않는다.

## 11. 실제 실행 시간의 보고

`study_analysis.json.timing.objectives`와 `timing.scope`, `objective_sessions`를 기준으로 다음을 구분한다.

| 지표 | 범위 |
|---|---|
| `attack_seconds_sum` | 동기화된 attack 엔진 시간 합계; 내부 clean forward 포함, save/평가/replay 제외 |
| `condition_wall_seconds_sum` | attack + capture/metric/save/replay; 이전 GT/model load 및 별도 공유 clean 저장 제외 |
| `objective_finished_wall_seconds_sum` | 종료 event가 있는 objective 세션 합계; load/parity/clean 저장/resume skip 포함 |
| `study_finished_invocation_wall_seconds_sum` | 완료된 study invocation 시간; 재개 사이 중단 시간 제외 |

UTC를 KST로 변환한 시작/종료 시각과 pause/resume 이력을 함께 표시할 수 있다. 달력상의 경과 시간과 실제 계산 시간을 혼용하지 않는다. 과거 `6~7시간`이나 `5시간40분~6시간` 전망을 실측 최종 시간처럼 쓰지 않는다. Confidence 등 아직 미측정이었던 추정치를 최종 보고서의 actual 칸에 넣지 않는다.

데이터 I/O만의 시간이 별도로 기록되지 않았으므로 loading 준비 시간을 순수 dataset 읽기 시간이라고 하지 않는다. `pre_attack_preparation_wall_seconds_sum`은 GT/model/parity/clean 저장 등도 포함한다. 분석·감사·HTML 렌더링 시간은 확인 가능한 실제 기록이 있을 때 별도 표시하고 중복 합산하지 않는다. GPU 활용률의 단편 수치로 실험 throughput이나 모델 성능을 설명하지 않는다.

## 12. 납품 전 검증과 완료 기준

### 12.1 수치·범위 검증

- 48조건, clean8, 공격40, 다섯 목적, PO4+DR4, 모든 source frame0~127이 일치한다.
- 같은 clip의 clean과 공격을 비교하며 중복/누락 key가 없다.
- 요약5행 ×3dataset 및 상세40행의 숫자를 분석 JSON과 원래 result로 교차 확인한다.
- APD 단위%, 차분%p, 상대%, EPE m, 비율의 분모와 sign이 정확하다. N/A·음수·기록 실패를 유지한다.
- 집계 숫자는 반올림 전 값으로 계산하고 bootstrap estimate/CI/support를 그대로 연결한다.
- 프레임별 EPE와 reconstruction residual의 전체 집계가 저장 지표와 일치한다.
- Native L21/confidence/normalization을 benchmark metric과 구분하고 서로 다른 loss 절댓값으로 순위를 만들지 않는다.
- HTML·그림의 완료 라벨이 audit/analysis에 의해 증명되고 GitHub 과거 snapshot이 데이터로 사용되지 않는다.

### 12.2 시각화·HTML 검증

- 원본 배열은128프레임이고 encoded MP4도 ffprobe로128 decoded frames, FPS, 해상도, duration을 확인한다. packet 수만 frame 수로 쓰지 않는다.
- frame0/64/127의 스틸·query ID·GT와 같은 점의 clean/attack 대응을 검사한다.
- RGB 차분 ×64 라벨과 실제 ε, meter colorbar, 공통 scale, invalid/out-of-frame 처리가 보인다.
- 생성 index.html을 실제 브라우저로 열어 기본 화면, PO/DR 필터, 각 objective/clip, 표 정렬, 그림 확대, 영상 재생, 인쇄 화면을 확인한다.
- file://에서 작동하고 상대 자산 링크가 모두 존재하며 인터넷을 꺼도 표·그림·필터가 작동한다. 다른 폴더로 복사한 보고서도 확인한다.
- 한국어 깨짐·겹침·표 잘림·깨진 이미지·콘솔 오류·placeholder/가상 수치가 없다.
- 자동 재생과 대용량 배열 embed가 없고 HTML/asset hash 및 선택 규칙이 manifest에 기록된다.

보고서 코드의 검증은 필요한 CPU 회귀 검사로 한정한다. 이미 감사된 입력을 반복하여 GPU로 재실행하지 않는다. 특히 분모0/N/A, 음수 개선, incomplete 입력 거부, paired key 일치, 비율 평균 정의, 문자열 escape, 상대 자산 참조와 scalar hash 변경 감지를 확인한다.

### 12.3 최종 전달

완료하면 `index.html`의 실제 경로, 전체 결과 폴더, 숫자 근거 JSON/CSV, 재현 command 및 검증 상태를 전달한다. 보고서 본문은 적어도 다음 질문에 실제 수치로 답해야 한다.

1. 어떤 PGD 목적이 선정8클립에서 tracking APD/EPE에 얼마나 영향을 주었는가?
2. Reconstruction 성능도 얼마나 변했으며 PO와 DR의 양상은 어떻게 다른가?
3. 전체128프레임과 클립별 변화가 평균 결과를 어떻게 뒷받침하는가?
4. 입력 변화는 ε4/255 예산 안에 있었고 저장 결과는 검증됐는가?
5. Native loss·confidence·normalization 진단과 실제 benchmark 손상은 어떻게 다른가?
6. 이 결과로 설명할 수 있는 범위와 아직 검증하지 않은 범위는 무엇인가?

이 MD를 작성하는 작업의 완료와 실험/HTML 제작의 완료는 별개다. **현재 요청은 이 상세 지시서를 만드는 것이며, 최종 HTML 제작자는 실험의 실제 종료와3절 검증을 확인한 후 위 절차를 수행한다.**
